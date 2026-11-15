"""Frame-level geometry and graph team descriptors, ball-free (ideas FE1, FE3, FE4, FE5, FE6).

Run from the repo root:  uv run python scripts/explore/fe_geometry.py
Saves five families to out/explore/features/: goalside, shape, convergence, graph_intra, space.
Prints only aggregates (AUC screening tables and blocked-CV GBM scores).

Decisions fixed a priori (not tuned on anything):
- Coordinates: featlib.oriented_xy, H always attacks +x. H's own goal is x=0, A's own goal x=L.
  "Height" features are distances from the team's OWN goal line (H: x, A: L - x).
- Only frames inside the graded halves (HALVES) are computed; everything else is NaN.
- Goalkeeper: per half and team, the slot that is most often the deepest player of its team
  (88-98% coverage). It is excluded from the "outfield" sets (goalside, shape, convergence,
  graph_intra, hull-penetration) and kept in the Voronoi / pitch-control / Delaunay(22) space.
- Duplicates: while a team shows > 11 players, the closest same-team pair (< 1.5 m) loses one
  member (up to 3 passes). Players are then packed into K=14 columns per team.
- Validity: a frame gets features only if both teams have >= 7 players after cleaning.
- Velocities: featlib.velocities (Savitzky-Golay, 7 frames). Missing velocity -> player ignored in
  velocity features (pitch control uses v=0 instead).
- Rates: goalside, shape (except hull area), convergence and most of graph_intra are computed at
  25 Hz (vectorized). Delaunay-based features (hull area, Delaunay edge length, inter-team edge
  share, hull penetration) and the Voronoi / pitch-control grid are computed at 5 Hz (every 5th
  frame) and linearly interpolated to 25 Hz (NaN if a neighbour sample is NaN, except +-2 frames
  around a valid sample which take its value).
- Pitch control: simplified Spearman (2017)-style: per cell, team arrival time = min over players
  of 0.7 s + |p + 0.7 v - cell| / 5 m/s; P(H) = sigmoid((tA - tH) * pi / (sqrt(3) * 0.45 s)).
"""
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay

from featlib import (A, FPS, H, HALVES, gbm_cv, labels, load_families, load_tracking, oriented_xy,
                     save_family, screen, velocities)

L, W = 105.3, 68.0
K = 14            # packed columns per team
MAXP = 11
DUP_DIST = 1.5    # m
MIN_PLAYERS = 7
SUB = 5           # 5 Hz for the expensive features
GOAL = {H: np.array([0.0, W / 2], np.float32), A: np.array([L, W / 2], np.float32)}
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- preparation

def halves_mask(n):
    m = np.zeros(n, bool)
    for a, b in HALVES:
        m[a:b] = True
    return m


def goalkeeper_mask(tr, xy, valid):
    gk = np.zeros(valid.shape, bool)
    for a, b in HALVES:
        for T in (H, A):
            sl = np.flatnonzero(tr.equipo == T)
            x = xy[a:b, sl, 0]
            own = x if T == H else L - x
            own = np.where(valid[a:b, sl], own, np.inf)
            ok = np.isfinite(own.min(1))
            deep = np.argmin(own, 1)
            slot = sl[np.argmax(np.bincount(deep[ok], minlength=len(sl)))]
            gk[a:b, slot] = True
    return gk


def pack(P, V, valid, k=K):
    """Pack the valid players of a team into k columns. P, V (N,S,2), valid (N,S)."""
    order = np.argsort(~valid, axis=1, kind="stable")[:, :k]
    Pc = np.take_along_axis(P, order[..., None], 1).copy()
    Vc = np.take_along_axis(V, order[..., None], 1).copy()
    vc = np.take_along_axis(valid, order, 1).copy()
    Pc[~vc] = np.nan
    Vc[~vc] = np.nan
    return Pc, Vc, vc


def dedupe(P, valid):
    """Drop one member of the closest same-team pair while > MAXP players (in place on valid)."""
    for _ in range(3):
        rows = np.flatnonzero(valid.sum(1) > MAXP)
        if len(rows) == 0:
            break
        p = P[rows]
        d = np.linalg.norm(p[:, :, None] - p[:, None], axis=-1)
        v = valid[rows]
        d = np.where(v[:, :, None] & v[:, None], d, np.inf)
        d[:, np.arange(d.shape[1]), np.arange(d.shape[1])] = np.inf
        flat = d.reshape(len(rows), -1).argmin(1)
        dmin = d.reshape(len(rows), -1).min(1)
        j = flat % d.shape[1]
        drop = dmin < DUP_DIST
        valid[rows[drop], j[drop]] = False
        P[rows[drop], j[drop]] = np.nan


def prepare(tr):
    n = tr.n_frames
    xy = oriented_xy(tr)
    log("velocities ...")
    vel = velocities(xy).astype(np.float32)
    hm = halves_mask(n)
    valid = tr.visible & np.isfinite(xy).all(-1) & hm[:, None]
    gk = goalkeeper_mask(tr, xy, valid)
    team = {}
    for T in (H, A):
        sl = np.flatnonzero(tr.equipo == T)
        P, V, v = pack(xy[:, sl], vel[:, sl], valid[:, sl], k=len(sl))
        g = np.take_along_axis(gk[:, sl], np.argsort(~valid[:, sl], axis=1, kind="stable"), 1)
        dedupe(P, v)
        # re-pack after dedupe (full team incl. GK) and outfield
        order = np.argsort(~v, axis=1, kind="stable")
        P = np.take_along_axis(P, order[..., None], 1)
        V = np.take_along_axis(V, order[..., None], 1)
        g = np.take_along_axis(g, order, 1)
        v = np.take_along_axis(v, order, 1)
        Pf, Vf, vf = pack(P, V, v)
        Po, Vo, vo = pack(P, V, v & ~g)
        team[T] = dict(P=Pf, V=Vf, v=vf, Po=Po, Vo=Vo, vo=vo, n=vf.sum(1), no=vo.sum(1))
    ok = hm & (team[H]["n"] >= MIN_PLAYERS) & (team[A]["n"] >= MIN_PLAYERS)
    log(f"prepared: frames ok {ok.sum()} / halves {hm.sum()}")
    return team, ok


def own_height(x, T):
    return x if T == H else L - x


def interp_sub(Xs, idx, n):
    """Samples Xs (m, k) at frames idx -> (n, k) linear interpolation; see module docstring."""
    out = np.full((n, Xs.shape[1]), np.nan, np.float32)
    f = np.arange(idx[0], idx[-1] + 1)
    pos = (f - idx[0]) / SUB
    k0 = np.clip(np.floor(pos).astype(int), 0, len(idx) - 1)
    k1 = np.clip(k0 + 1, 0, len(idx) - 1)
    w = (pos - k0)[:, None].astype(np.float32)
    a, b = Xs[k0], Xs[k1]
    val = (1 - w) * a + w * b
    near = np.where(w <= 0.5, a, b)
    near_ok = np.minimum(w, 1 - w) * SUB <= 2
    val = np.where(np.isfinite(val), val, np.where(near_ok, near, np.nan))
    out[f] = val
    return out


# ----------------------------------------------------------------------------- FE1 goalside

def fam_goalside(team, ok, n):
    names = []
    cols = {}
    CH = 20000
    for nm in ["gsx", "gsgoal", "nnd", "free", "freefrac"]:
        for T in "HA":
            cols[f"{nm}_{T}"] = np.full(n, np.nan, np.float32)
    for s in range(0, n, CH):
        e = min(n, s + CH)
        PH, PA = team[H]["Po"][s:e], team[A]["Po"][s:e]
        d = np.linalg.norm(PH[:, :, None] - PA[:, None], axis=-1)
        d = np.where(np.isfinite(d), d, np.inf)
        for T, Pm, Po, dm in ((H, PH, PA, d), (A, PA, PH, d.transpose(0, 2, 1))):
            vm = np.isfinite(Pm[..., 0])
            j = dm.argmin(2)
            nnd = dm.min(2)
            opp = np.take_along_axis(Po, j[..., None], 1)
            if T == H:
                gsx = Pm[..., 0] < opp[..., 0]
            else:
                gsx = Pm[..., 0] > opp[..., 0]
            gsg = np.linalg.norm(Pm - GOAL[T], axis=-1) < np.linalg.norm(opp - GOAL[T], axis=-1)
            cnt = vm.sum(1)
            tag = "HA"[T]
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[f"gsx_{tag}"][s:e] = (gsx & vm).sum(1) / cnt
                cols[f"gsgoal_{tag}"][s:e] = (gsg & vm).sum(1) / cnt
                cols[f"nnd_{tag}"][s:e] = np.where(vm, nnd, 0).sum(1) / cnt
                free = ((nnd > 5) & vm).sum(1)
                cols[f"free_{tag}"][s:e] = free
                cols[f"freefrac_{tag}"][s:e] = free / cnt
    # Hungarian matching between outfield sets, 25 Hz
    hd = np.full(n, np.nan, np.float32)
    hgx = np.full(n, np.nan, np.float32)
    hgg_h = np.full(n, np.nan, np.float32)
    hgg_a = np.full(n, np.nan, np.float32)
    PHo, PAo, voH, voA = team[H]["Po"], team[A]["Po"], team[H]["vo"], team[A]["vo"]
    for f in np.flatnonzero(ok):
        ph, pa = PHo[f][voH[f]], PAo[f][voA[f]]
        c = np.linalg.norm(ph[:, None] - pa[None], axis=-1)
        r, q = linear_sum_assignment(c)
        hd[f] = c[r, q].mean()
        h, a = ph[r], pa[q]
        hgx[f] = np.mean(h[:, 0] < a[:, 0])
        hgg_h[f] = np.mean(np.linalg.norm(h - GOAL[H], axis=1) < np.linalg.norm(a - GOAL[H], axis=1))
        hgg_a[f] = np.mean(np.linalg.norm(a - GOAL[A], axis=1) < np.linalg.norm(h - GOAL[A], axis=1))
    out = []
    for nm in ["gsx", "gsgoal", "nnd", "free", "freefrac"]:
        out += [cols[f"{nm}_H"], cols[f"{nm}_A"], cols[f"{nm}_H"] - cols[f"{nm}_A"]]
        names += [f"{nm}_H", f"{nm}_A", f"{nm}_diff"]
    out += [hd, hgx, hgg_h, hgg_a, hgg_h - hgg_a]
    names += ["hung_dist", "hung_gsx_H", "hung_gsgoal_H", "hung_gsgoal_A", "hung_gsgoal_diff"]
    X = np.stack(out, 1)
    X[~ok] = np.nan
    return X, names


# ----------------------------------------------------------------------------- FE6 + Delaunay parts (5 Hz)

def tri_area(pts, simp):
    t = pts[simp]
    return 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                        - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))


def edges_of(simp):
    e = np.concatenate([simp[:, [0, 1]], simp[:, [1, 2]], simp[:, [0, 2]]])
    e.sort(1)
    return np.unique(e, axis=0)


def delaunay_part(team, idx):
    """Per 5 Hz sample: hull area, Delaunay mean edge per team (outfield), inter-team edge share
    of Delaunay(all players), outfield players inside the opponent's outfield hull."""
    m = len(idx)
    out = {k: np.full(m, np.nan, np.float32) for k in
           ["hull_H", "hull_A", "dedge_H", "dedge_A", "inter_share", "inhull_H", "inhull_A"]}
    for i, f in enumerate(idx):
        tri = {}
        for T in (H, A):
            po = team[T]["Po"][f][team[T]["vo"][f]]
            try:
                d = Delaunay(po)
            except Exception:
                continue
            tri[T] = (d, po)
            tag = "HA"[T]
            out[f"hull_{tag}"][i] = tri_area(po, d.simplices).sum()
            e = edges_of(d.simplices)
            out[f"dedge_{tag}"][i] = np.linalg.norm(po[e[:, 0]] - po[e[:, 1]], axis=1).mean()
        if len(tri) == 2:
            out["inhull_H"][i] = (tri[A][0].find_simplex(tri[H][1]) >= 0).sum()
            out["inhull_A"][i] = (tri[H][0].find_simplex(tri[A][1]) >= 0).sum()
        ph = team[H]["P"][f][team[H]["v"][f]]
        pa = team[A]["P"][f][team[A]["v"][f]]
        pts = np.concatenate([ph, pa])
        lab = np.r_[np.zeros(len(ph), int), np.ones(len(pa), int)]
        try:
            e = edges_of(Delaunay(pts).simplices)
            out["inter_share"][i] = np.mean(lab[e[:, 0]] != lab[e[:, 1]])
        except Exception:
            pass
    return out


def space_part(team, idx):
    """Per 5 Hz sample: Voronoi areas (1 m grid) and simplified pitch control."""
    gx, gy = np.meshgrid(np.arange(0.5, L, 1.0), np.arange(0.5, W, 1.0), indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)   # (C, 2)
    c2 = (cells ** 2).sum(1)
    half_A = cells[:, 0] > L / 2          # A's half (H attacks +x)
    half_H = ~half_A
    third_A = cells[:, 0] > 2 * L / 3     # H's final third
    third_H = cells[:, 0] < L / 3         # A's final third
    m = len(idx)
    keys = ["vor_H", "vor_A", "vor_H_opphalf", "vor_A_opphalf", "vor_H_final3", "vor_A_final3",
            "pc_H", "pc_H_opphalf", "pc_A_opphalf", "pc_H_final3", "pc_A_final3"]
    out = {k: np.full(m, np.nan, np.float32) for k in keys}
    PH = np.where(np.isfinite(team[H]["P"]), team[H]["P"], 1e4)[idx]   # (m, K, 2), far if missing
    PA = np.where(np.isfinite(team[A]["P"]), team[A]["P"], 1e4)[idx]
    VH = np.nan_to_num(team[H]["V"][idx])
    VA = np.nan_to_num(team[A]["V"][idx])
    s = np.pi / (np.sqrt(3) * 0.45)
    CH = 64
    for a in range(0, m, CH):
        b = min(m, a + CH)

        def mind(P):  # (n, K, 2) -> min over players of distance to each cell, (n, C)
            p = P.reshape(-1, 2)
            d2 = c2[:, None] - 2 * cells @ p.T + (p ** 2).sum(1)[None]
            d2 = d2.reshape(len(cells), b - a, K).min(2).T
            return np.sqrt(np.maximum(d2, 0))
        dH, dA = mind(PH[a:b]), mind(PA[a:b])
        own_H = dH < dA
        out["vor_H"][a:b] = own_H.sum(1)
        out["vor_A"][a:b] = (~own_H).sum(1)
        out["vor_H_opphalf"][a:b] = (own_H & half_A).sum(1)
        out["vor_A_opphalf"][a:b] = (~own_H & half_H).sum(1)
        out["vor_H_final3"][a:b] = (own_H & third_A).sum(1)
        out["vor_A_final3"][a:b] = (~own_H & third_H).sum(1)
        tH = 0.7 + mind(PH[a:b] + 0.7 * VH[a:b]) / 5.0
        tA = 0.7 + mind(PA[a:b] + 0.7 * VA[a:b]) / 5.0
        pc = 1 / (1 + np.exp(-s * (tA - tH)))
        out["pc_H"][a:b] = pc.mean(1)
        out["pc_H_opphalf"][a:b] = pc[:, half_A].mean(1)
        out["pc_A_opphalf"][a:b] = (1 - pc[:, half_H]).mean(1)
        out["pc_H_final3"][a:b] = pc[:, third_A].mean(1)
        out["pc_A_final3"][a:b] = (1 - pc[:, third_H]).mean(1)
    return out


def sub_features(team, ok, n):
    idx = np.flatnonzero(ok)
    idx = idx[idx % SUB == 0]
    log(f"5 Hz samples: {len(idx)}")
    d = delaunay_part(team, idx)
    log("delaunay part done")
    d.update(space_part(team, idx))
    log("voronoi / pitch control done")
    names = list(d)
    # interpolate each contiguous run of samples separately (runs split at gaps > SUB)
    Xs = np.stack([d[k] for k in names], 1)
    X = np.full((n, len(names)), np.nan, np.float32)
    br = np.flatnonzero(np.diff(idx) > SUB) + 1
    for seg in np.split(np.arange(len(idx)), br):
        if len(seg) == 0:
            continue
        Y = interp_sub(Xs[seg], idx[seg], n)
        rng = slice(idx[seg[0]], idx[seg[-1]] + 1)
        X[rng] = Y[rng]
    X[~ok] = np.nan
    return {k: X[:, j] for j, k in enumerate(names)}


# ----------------------------------------------------------------------------- FE3 shape

def smooth(x, w):
    return pd.Series(x).rolling(w, center=True, min_periods=max(1, w // 2)).mean().values


def deriv(x, w):
    """Centred difference over w frames of a lightly smoothed series, per second."""
    s = smooth(x, 5)
    h = w // 2
    d = np.full_like(s, np.nan)
    d[h:-h] = (s[2 * h:] - s[:-2 * h]) / (2 * h / FPS)
    return d.astype(np.float32)


def row_percentile(a, q):
    """Per-row linear-interpolated quantile ignoring NaN (vectorized nanpercentile)."""
    s = np.sort(a, 1)                      # NaN last
    c = np.isfinite(a).sum(1)
    pos = q * np.maximum(c - 1, 0)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, np.maximum(c - 1, 0))
    r = np.arange(len(a))
    v = s[r, lo] + (pos - lo) * (s[r, hi] - s[r, lo])
    return np.where(c > 0, v, np.nan)


def fam_shape(team, ok, n, sub):
    base = {}
    for T in (H, A):
        tag = "HA"[T]
        P, V, v = team[T]["Po"], team[T]["Vo"], team[T]["vo"]
        x, y = P[..., 0], P[..., 1]
        cx, cy = np.nanmean(x, 1), np.nanmean(y, 1)
        hx = own_height(x, T)
        base[f"cx_{tag}"] = cx
        base[f"cy_{tag}"] = cy
        base[f"adv_{tag}"] = own_height(cx, T)
        p10, p90 = row_percentile(y, 0.10), row_percentile(y, 0.90)
        base[f"width_{tag}"] = p90 - p10
        base[f"stdy_{tag}"] = np.nanstd(y, 1)
        base[f"depth_{tag}"] = np.nanmax(x, 1) - np.nanmin(x, 1)
        base[f"hull_{tag}"] = sub[f"hull_{tag}"]
        base[f"mdist_{tag}"] = np.nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), 1)
        hs = np.sort(np.where(np.isfinite(hx), hx, np.inf), 1)
        base[f"defline_{tag}"] = hs[:, :4].mean(1)
        hs2 = np.sort(np.where(np.isfinite(hx), hx, -np.inf), 1)[:, ::-1]
        base[f"front_{tag}"] = hs2[:, :2].mean(1)
        base[f"stretch_{tag}"] = base[f"front_{tag}"] - base[f"defline_{tag}"]
        vx = V[..., 0] if T == H else -V[..., 0]
        base[f"vadv_{tag}"] = smooth(np.nanmean(vx, 1), 25)  # centroid velocity along own attack
    for k in base:
        base[k] = np.where(ok, base[k], np.nan).astype(np.float32)
    names, cols = [], []
    stat = ["cx", "cy", "adv", "width", "stdy", "depth", "hull", "mdist", "defline", "front",
            "stretch", "vadv"]
    for st in stat:
        cols += [base[f"{st}_H"], base[f"{st}_A"], base[f"{st}_H"] - base[f"{st}_A"]]
        names += [f"{st}_H", f"{st}_A", f"{st}_diff"]
    for st in ["adv", "width", "depth", "hull", "mdist", "defline", "stretch"]:
        for w, wn in ((25, "1s"), (75, "3s")):
            dh, da = deriv(base[f"{st}_H"], w), deriv(base[f"{st}_A"], w)
            cols += [dh, da, dh - da]
            names += [f"d{wn}_{st}_H", f"d{wn}_{st}_A", f"d{wn}_{st}_diff"]
    X = np.stack(cols, 1)
    X[~ok] = np.nan
    return X, names


# ----------------------------------------------------------------------------- FE4 convergence

def fam_convergence(team, ok, n):
    cen = {T: np.nanmean(team[T]["Po"], 1) for T in (H, A)}
    names, cols = [], []
    res = {}
    for T in (H, A):
        tag = "HA"[T]
        P, V = team[T]["Po"], team[T]["Vo"]
        sp = np.linalg.norm(V, axis=-1)
        mov = np.isfinite(sp) & (sp > 1.0)
        u = np.where(mov[..., None], V / np.where(mov, sp, 1)[..., None], 0)
        Pz = np.where(mov[..., None], P, 0)
        nm = mov.sum(1).astype(np.float32)
        uu = u[..., :, None] * u[..., None, :]                         # (N,K,2,2)
        I = np.eye(2, dtype=np.float32)
        Mi = np.where(mov[..., None, None], I - uu, 0)
        M = Mi.sum(1)
        b = np.einsum("nkij,nkj->ni", Mi, Pz)
        ev = np.linalg.eigvalsh(M)[:, 0]
        good = ok & (nm >= 3) & (ev >= 0.5)
        Ms = np.where(good[:, None, None], M, I)
        c = np.linalg.solve(Ms, b[..., None])[..., 0]
        c[~good] = np.nan
        c[:, 0] = np.clip(c[:, 0], -30, L + 30)
        c[:, 1] = np.clip(c[:, 1], -30, W + 30)
        r = P - c[:, None]
        perp = np.abs(r[..., 0] * u[..., 1] - r[..., 1] * u[..., 0])
        resid = np.sqrt(np.where(mov, perp ** 2, 0).sum(1) / np.maximum(nm, 1))
        to_c = c[:, None] - P
        cos = (to_c * u).sum(-1) / np.maximum(np.linalg.norm(to_c, axis=-1), 1e-3)
        cosm = np.where(mov, cos, 0).sum(1) / np.maximum(nm, 1)
        opp = cen[1 - T]
        rad_vec = P - cen[T][:, None]
        rad = (V * rad_vec).sum(-1) / np.maximum(np.linalg.norm(rad_vec, axis=-1), 1e-3)
        res[T] = c
        feats = {
            "conv_cx": c[:, 0], "conv_cy": c[:, 1],
            "conv_adv": own_height(c[:, 0], T),
            "conv_resid": resid, "conv_cos": cosm,
            "conv_d_oppcen": np.linalg.norm(c - opp, axis=1),
            "conv_d_owncen": np.linalg.norm(c - cen[T], axis=1),
            "moving_frac": nm / team[T]["no"],
            "speed": np.nanmean(sp, 1),
            "radial_v": np.nanmean(rad, 1),
        }
        for k, v in feats.items():
            msk = ok if k in ("speed", "radial_v", "moving_frac") else good
            v = np.where(msk, v, np.nan)
            res[(T, k)] = v.astype(np.float32)
    for k in ["conv_cx", "conv_cy", "conv_adv", "conv_resid", "conv_cos", "conv_d_oppcen",
              "conv_d_owncen", "moving_frac", "speed", "radial_v"]:
        cols += [res[(H, k)], res[(A, k)], res[(H, k)] - res[(A, k)]]
        names += [f"{k}_H", f"{k}_A", f"{k}_diff"]
    cols.append(np.linalg.norm(res[H] - res[A], axis=1))
    names.append("conv_HA_dist")
    X = np.stack(cols, 1).astype(np.float32)
    X[~ok] = np.nan
    return X, names


# ----------------------------------------------------------------------------- FE5 graph_intra

def fam_graph(team, ok, n, sub):
    names, cols = [], []
    CH = 20000
    for T in (H, A):
        tag = "HA"[T]
        f = {k: np.full(n, np.nan, np.float32) for k in
             ["knn3_len", "knn_clust", "mst_len", "mst_mean_edge", "fiedler", "fiedler_pn"]}
        for s in range(0, n, CH):
            e = min(n, s + CH)
            P, v = team[T]["Po"][s:e], team[T]["vo"][s:e]
            m = e - s
            cnt = v.sum(1)
            d = np.linalg.norm(P[:, :, None] - P[:, None], axis=-1)
            pair = v[:, :, None] & v[:, None]
            eye = np.eye(K, dtype=bool)[None]
            dinf = np.where(pair & ~eye, d, np.inf)
            ds = np.sort(dinf, 2)[:, :, :3]
            with np.errstate(invalid="ignore"):
                f["knn3_len"][s:e] = np.where(v, np.where(np.isfinite(ds), ds, 0).mean(2), 0).sum(1) / cnt
            # kNN(3) graph, symmetrized (union); local clustering
            kth = ds[:, :, 2:3]
            Adj = (dinf <= kth) & pair & ~eye
            Adj = Adj | Adj.transpose(0, 2, 1)
            Af = Adj.astype(np.float32)
            tri = np.einsum("nij,njk,nki->ni", Af, Af, Af) / 2
            deg = Af.sum(2)
            cl = np.where(deg >= 2, 2 * tri / np.maximum(deg * (deg - 1), 1), np.nan)
            f["knn_clust"][s:e] = np.nanmean(np.where(v, cl, np.nan), 1)
            # MST length, batched Prim
            dist = np.where(pair, d, np.inf)
            dist[:, eye[0]] = np.inf
            start = np.argmax(v, 1)
            intree = np.zeros((m, K), bool)
            intree[np.arange(m), start] = True
            best = dist[np.arange(m), start].copy()
            tot = np.zeros(m, np.float32)
            for _ in range(K - 1):
                cand = np.where(intree | ~v, np.inf, best)
                j = cand.argmin(1)
                val = cand[np.arange(m), j]
                add = np.isfinite(val)
                tot += np.where(add, val, 0)
                intree[np.arange(m)[add], j[add]] = True
                best = np.where(add[:, None], np.minimum(best, dist[np.arange(m), j]), best)
            f["mst_len"][s:e] = tot
            f["mst_mean_edge"][s:e] = tot / np.maximum(cnt - 1, 1)
            # Fiedler value of exp(-d/10) weighted Laplacian; padded nodes add zero eigenvalues
            Wt = np.where(pair & ~eye, np.exp(-np.nan_to_num(d) / 10.0), 0).astype(np.float64)
            Lap = np.einsum("nij->ni", Wt)[:, :, None] * np.eye(K)[None] - Wt
            evs = np.linalg.eigvalsh(Lap)
            pad = K - cnt
            fied = evs[np.arange(m), np.clip(pad + 1, 0, K - 1)]
            f["fiedler"][s:e] = fied
            f["fiedler_pn"][s:e] = fied / np.maximum(cnt, 1)
        f["delaunay_len"] = sub[f"dedge_{tag}"]
        for k in f:
            f[k] = np.where(ok & (team[T]["no"] >= MIN_PLAYERS - 1), f[k], np.nan)
        team[T]["graph"] = f
    for k in team[H]["graph"]:
        a, b = team[H]["graph"][k], team[A]["graph"][k]
        cols += [a, b, a - b]
        names += [f"{k}_H", f"{k}_A", f"{k}_diff"]
    X = np.stack(cols, 1).astype(np.float32)
    X[~ok] = np.nan
    return X, names


def fam_space(sub, ok):
    names, cols = [], []
    for k in ["vor_H", "vor_H_opphalf", "vor_A_opphalf", "vor_H_final3", "vor_A_final3",
              "pc_H", "pc_H_opphalf", "pc_A_opphalf", "pc_H_final3", "pc_A_final3",
              "inter_share", "inhull_H", "inhull_A"]:
        cols.append(sub[k])
        names.append(k)
    cols += [sub["vor_H_opphalf"] - sub["vor_A_opphalf"], sub["pc_H_opphalf"] - sub["pc_A_opphalf"],
             sub["inhull_H"] - sub["inhull_A"]]
    names += ["vor_opphalf_diff", "pc_opphalf_diff", "inhull_diff"]
    X = np.stack(cols, 1).astype(np.float32)
    X[~ok] = np.nan
    return X, names


# ----------------------------------------------------------------------------- main

def report(fam, X, names, lab):
    print(f"\n=== family {fam}: {X.shape[1]} features ===")
    screen(X, names, lab, top=12)
    gbm_cv(X, lab)


def main():
    tr = load_tracking()
    lab = labels(tr)
    n = tr.n_frames
    team, ok = prepare(tr)
    fams = {}
    X, nm = fam_goalside(team, ok, n)
    fams["goalside"] = (X, nm)
    log("goalside done")
    sub = sub_features(team, ok, n)
    X, nm = fam_shape(team, ok, n, sub)
    fams["shape"] = (X, nm)
    log("shape done")
    X, nm = fam_convergence(team, ok, n)
    fams["convergence"] = (X, nm)
    log("convergence done")
    X, nm = fam_graph(team, ok, n, sub)
    fams["graph_intra"] = (X, nm)
    log("graph_intra done")
    fams["space"] = fam_space(sub, ok)
    for fam, (X, nm) in fams.items():
        save_family(fam, X, nm)
    log("saved")
    for fam, (X, nm) in fams.items():
        report(fam, X, nm, lab)
    Xall, nall = load_families(list(fams))
    print(f"\n=== all geometry families together: {Xall.shape[1]} features ===")
    gbm_cv(Xall, lab)
    log("done")


if __name__ == "__main__":
    main()
