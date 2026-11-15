"""Goalkeeper team-swap and cross-team duplicate cleaning (exp-054, T15).

In BAR-ATH, slots 23 (H GK) and 45 (A GK) never enter the middle third of the pitch,
but the provider's tracker places slot 45 at H's goal (oriented x < 35 m) for 13.7% of
play frames and slot 23 at A's goal (x > L - 35 m) for 1.0%. Whenever the wrong-team GK
slot has a finite velocity detection, the defending team's own GK slot has velocity=NaN
(an off-camera interpolated ghost). In addition, fb_ball.py selects each team's GK per
frame as argmin/argmax(x), so a wrong-end GK slot is treated as an outfield striker inside
the opponent's box.

`clean_goalkeepers(tr, periods)` repairs the tracking in memory and `patch_ball_features`
recomputes the 5 BALL feature families only on the affected ball-visible frames without
modifying the on-disk feature cache (`out/explore/features/*.npz`).
"""
import copy
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, QhullError

import featlib
from mtah_2027 import atd

L_PITCH, W_PITCH = float(atd.CAMPO[0]), float(atd.CAMPO[1])
H, A = 0, 1
FPS = 25

VMAX_PLAYER, VMAX_BALL = 12.0, 40.0
T_REACT, VMAX_ICPT, BALL_DECEL = 0.7, 7.0, 2.0
T_GRID = np.arange(0, 6.0 + 1e-6, 0.1, dtype=np.float32)
PC_VMAX, PC_REACT, PC_SIGMA, PC_LAMBDA = 5.0, 0.7, 0.45, 4.3
PC_DT, PC_TMAX = 0.04, 10.0
LANE_CLEAR = 2.0
LL_WIN, LL_LAGS = 4 * FPS, list(range(-FPS, FPS + 1, 5))


def find_gk_slots(tr, xy_oriented, periods):
    """Per half and team, the slot most frequently deepest towards its own goal."""
    gk_slots = {}
    for h_idx, (a, b) in enumerate(periods):
        gk_slots[h_idx] = {}
        for t in (H, A):
            sl = np.flatnonzero(tr.equipo == t)
            x = xy_oriented[a:b, sl, 0]
            own = x if t == H else L_PITCH - x
            own = np.where(np.isfinite(own), own, np.inf)
            ok = np.isfinite(own.min(1))
            deep = np.argmin(own[ok], 1)
            gk_slots[h_idx][t] = int(sl[np.argmax(np.bincount(deep, minlength=len(sl)))])
    return gk_slots


def clean_goalkeepers(tr, periods):
    """Return a cleaned copy of Tracking where wrong-end GK slots and cross-team GK duplicates are fixed."""
    tr_c = copy.copy(tr)
    tr_c.xy = tr.xy.copy()
    tr_c.velocidad = tr.velocidad.copy()

    sign = featlib.attack_sign(tr)
    xy_or = featlib.oriented(tr_c.metros(tr_c.xy), sign)
    gk_slots = find_gk_slots(tr_c, xy_or, periods)

    changed_frames = np.zeros(tr.n_frames, bool)
    stats = dict(s_a_at_h=0, s_a_copied_to_h=0, s_h_at_a=0, s_h_copied_to_a=0, out_dup_h=0, out_dup_a=0)

    for h_idx, (a, b) in enumerate(periods):
        gk_h = gk_slots[h_idx][H]
        gk_a = gk_slots[h_idx][A]

        x_or = xy_or[a:b, :, 0]
        vel = tr_c.velocidad[a:b]

        a_at_h = np.isfinite(x_or[:, gk_a]) & (x_or[:, gk_a] < 35.0)
        copy_a2h = a_at_h & np.isfinite(vel[:, gk_a]) & ~np.isfinite(vel[:, gk_h])
        stats["s_a_at_h"] += int(a_at_h.sum())
        stats["s_a_copied_to_h"] += int(copy_a2h.sum())

        h_at_a = np.isfinite(x_or[:, gk_h]) & (x_or[:, gk_h] > L_PITCH - 35.0)
        copy_h2a = h_at_a & np.isfinite(vel[:, gk_h]) & ~np.isfinite(vel[:, gk_a])
        stats["s_h_at_a"] += int(h_at_a.sum())
        stats["s_h_copied_to_a"] += int(copy_h2a.sum())

        f_copy_a2h = np.flatnonzero(copy_a2h) + a
        f_a_at_h = np.flatnonzero(a_at_h) + a
        tr_c.xy[f_copy_a2h, gk_h] = tr_c.xy[f_copy_a2h, gk_a]
        tr_c.velocidad[f_copy_a2h, gk_h] = tr_c.velocidad[f_copy_a2h, gk_a]
        tr_c.xy[f_a_at_h, gk_a] = np.nan
        tr_c.velocidad[f_a_at_h, gk_a] = np.nan

        f_copy_h2a = np.flatnonzero(copy_h2a) + a
        f_h_at_a = np.flatnonzero(h_at_a) + a
        tr_c.xy[f_copy_h2a, gk_a] = tr_c.xy[f_copy_h2a, gk_h]
        tr_c.velocidad[f_copy_h2a, gk_a] = tr_c.velocidad[f_copy_h2a, gk_h]
        tr_c.xy[f_h_at_a, gk_h] = np.nan
        tr_c.velocidad[f_h_at_a, gk_h] = np.nan

        changed_frames[f_a_at_h] = True
        changed_frames[f_h_at_a] = True

        xy_or[a:b] = featlib.oriented(tr_c.metros(tr_c.xy[a:b]), sign[a:b])

        for own_x, opp_team, gk_def, key in [
            (xy_or[a:b, :, 0], A, gk_h, "out_dup_h"),
            (L_PITCH - xy_or[a:b, :, 0], H, gk_a, "out_dup_a"),
        ]:
            opp_slots = np.array([s for s in np.flatnonzero(tr.equipo == opp_team) if s not in (gk_h, gk_a)])
            x_opp = np.where(np.isfinite(own_x[:, opp_slots]), own_x[:, opp_slots], np.inf)
            idx_s = np.argsort(x_opp, 1)
            s1 = opp_slots[idx_s[:, 0]]
            x1 = np.take_along_axis(x_opp, idx_s[:, 0:1], 1)[:, 0]
            x2 = np.take_along_axis(x_opp, idx_s[:, 1:2], 1)[:, 0]
            y1 = xy_or[np.arange(a, b), s1, 1]
            d_gk = np.hypot(own_x[:, gk_def] - x1, xy_or[a:b, gk_def, 1] - y1)
            dup = (x1 < 12.0) & ((x2 - x1) > 10.0) & (d_gk < 2.0)
            f_dup = np.flatnonzero(dup) + a
            s_dup = s1[dup]
            tr_c.xy[f_dup, s_dup] = np.nan
            tr_c.velocidad[f_dup, s_dup] = np.nan
            changed_frames[f_dup] = True
            stats[key] += int(dup.sum())

    return tr_c, changed_frames, stats


def _shift(x, L):
    y = np.full_like(x, np.nan)
    if L > 0:
        y[L:] = x[:-L]
    elif L < 0:
        y[:L] = x[-L:]
    else:
        y[:] = x
    return y


def _wcorr(x, y, win, min_n=None):
    min_n = min_n or win // 2
    ok = np.isfinite(x) & np.isfinite(y)
    xv = np.where(ok, x, 0.0)
    yv = np.where(ok, y, 0.0)
    s = lambda a: pd.Series(a).rolling(win, center=True, min_periods=1).sum().values  # noqa: E731
    n = s(ok.astype(np.float64))
    sx, sy = s(xv), s(yv)
    sxx, syy, sxy = s(xv * xv), s(yv * yv), s(xv * yv)
    cov = sxy - sx * sy / np.maximum(n, 1)
    vx = sxx - sx * sx / np.maximum(n, 1)
    vy = syy - sy * sy / np.maximum(n, 1)
    denom = np.sqrt(np.maximum(vx * vy, 0.0))
    c = np.where((n >= min_n) & (denom > 1e-9), cov / np.maximum(denom, 1e-9), np.nan)
    return np.clip(c, -1.0, 1.0).astype(np.float32)


def patch_ball_features(tr_c, X_ball_orig, changed_frames):
    """Recompute the 5 BALL families only on the ball-visible frames affected by clean_goalkeepers."""
    N, S = tr_c.n_frames, tr_c.n_slots
    team_slot = tr_c.equipo.astype(np.int8)
    P = featlib.oriented_xy(tr_c)
    B = featlib.oriented_ball(tr_c)
    vis_p = np.isfinite(P[..., 0])

    ch_vel = pd.Series(changed_frames.astype(np.int8)).rolling(7, center=True, min_periods=1).max().values > 0
    fm = tr_c.balon_visible & ch_vel & np.isfinite(B[:, 0])
    idx = np.flatnonzero(fm)
    n = len(idx)

    vb = featlib.velocities(B)
    vb[~(np.hypot(vb[:, 0], vb[:, 1]) <= VMAX_BALL)] = np.nan
    vp = featlib.velocities(P)
    bad = ~(np.hypot(vp[..., 0], vp[..., 1]) <= VMAX_PLAYER)
    vp[bad] = 0

    Pi, Bi, vpi = P[idx], B[idx], vp[idx]
    vbi = vb[idx]
    vbi0 = np.where(np.isfinite(vbi), vbi, 0)
    vis = vis_p[idx]
    isH, isA = team_slot == H, team_slot == A
    rel = Bi[:, None, :] - Pi
    d = np.hypot(rel[..., 0], rel[..., 1])
    dinf = np.where(np.isfinite(d), d, np.inf)

    fams = {}
    # FB1 ball_pressure
    cols, names = [], []
    dk = {}
    for k, m, t in ((H, isH, "H"), (A, isA, "A")):
        dd = np.sort(np.where(m[None, :], dinf, np.inf), 1)[:, :2]
        dk[k] = dd
        for r in (3, 5, 10):
            cols.append((np.where(m[None, :], dinf, np.inf) < r).sum(1))
            names.append(f"n{r}_{t}")
        cols += [np.minimum(dd[:, 0], 100), np.minimum(dd[:, 1], 100)]
        names += [f"d1_{t}", f"d2_{t}"]
    cols.append(np.minimum(dk[H][:, 0], 100) - np.minimum(dk[A][:, 0], 100))
    names.append("dmin_H_minus_A")
    cols.append(cols[names.index("n5_A")] - cols[names.index("n5_H")])
    names.append("n5_A_minus_H")
    fams["ball_pressure"] = np.column_stack(cols).astype(np.float32)

    # FB2 ball_structure
    x = Pi[..., 0]
    bx, by = Bi[:, 0], Bi[:, 1]
    out_m = {}
    for k, m in ((H, isH), (A, isA)):
        xk = np.where(m[None, :] & vis, x, np.nan)
        has = np.isfinite(xk).any(1)
        gk = np.nanargmin(np.where(has[:, None], xk, 0), 1) if k == H else np.nanargmax(np.where(has[:, None], xk, 0), 1)
        o = m[None, :] & vis
        o[np.arange(n), gk] = False
        out_m[k] = o
    cols = []
    for k, _ in ((H, "H"), (A, "A")):
        xo = np.where(out_m[k], x, np.nan)
        defl = np.nanmin(xo, 1) if k == H else np.nanmax(xo, 1)
        cen = np.nanmean(xo, 1)
        gs = (out_m[k] & ((x < bx[:, None]) if k == H else (x > bx[:, None]))).sum(1)
        cols += [bx - defl, bx - cen, gs]
    cols += [np.hypot(bx, by - W_PITCH / 2), np.hypot(L_PITCH - bx, by - W_PITCH / 2)]
    sd = np.full((n, 2), np.nan, np.float32)
    for r in range(n):
        for k in (H, A):
            pts = Pi[r, out_m[k][r]]
            if len(pts) < 3:
                continue
            try:
                eq = ConvexHull(pts).equations
            except QhullError:
                continue
            sd[r, k] = (eq[:, 0] * bx[r] + eq[:, 1] * by[r] + eq[:, 2]).max()
    cols += [sd[:, 0], sd[:, 1], (sd[:, 0] <= 0).astype(np.float32), (sd[:, 1] <= 0).astype(np.float32),
             sd[:, 0] - sd[:, 1]]
    fams["ball_structure"] = np.column_stack(cols).astype(np.float32)

    # kinematics
    close = ((vpi - vbi0[:, None, :]) * rel).sum(-1) / np.maximum(d, 1e-3)
    ttc = np.where(close > 0, d / close, np.inf)
    ttc[~np.isfinite(d) | ~np.isfinite(close)] = np.inf
    close = np.where(np.isfinite(close), close, -np.inf)
    ball_speed = np.hypot(vbi[:, 0], vbi[:, 1])

    icpt = np.full((n, S), np.inf, np.float32)
    tg = T_GRID
    for c0 in range(0, n, 2000):
        sl = slice(c0, c0 + 2000)
        s0 = np.hypot(vbi0[sl, 0], vbi0[sl, 1])
        e = np.where(s0[:, None] > 0, vbi0[sl] / np.maximum(s0, 1e-6)[:, None], 0)
        tt = np.minimum(tg[None, :], (s0 / BALL_DECEL)[:, None])
        s = s0[:, None] * tt - 0.5 * BALL_DECEL * tt ** 2
        bt = Bi[sl][:, None, :] + e[:, None, :] * s[..., None]
        q = Pi[sl] + vpi[sl] * T_REACT
        dd = bt[:, :, None, :] - q[:, None, :, :]
        reach = T_REACT + np.hypot(dd[..., 0], dd[..., 1]) / VMAX_ICPT
        margin = tg[None, :, None] - reach
        feas = margin >= 0
        hasf = feas.any(1)
        kk = feas.argmax(1)
        km1 = np.maximum(kk - 1, 0)
        m1 = np.take_along_axis(margin, kk[:, None, :], 1)[:, 0]
        m0 = np.take_along_axis(margin, km1[:, None, :], 1)[:, 0]
        frac = np.where(m1 > m0, -m0 / np.maximum(m1 - m0, 1e-6), 0)
        t_star = tg[km1] + (tg[kk] - tg[km1]) * np.clip(frac, 0, 1)
        t_star = np.where(hasf, t_star, np.maximum(tg[-1], reach[:, -1, :]))
        t_star[~vis[sl]] = np.inf
        icpt[sl] = t_star

    tmin = lambda a, m: np.where(m[None, :], a, np.inf).min(1)  # noqa: E731
    tmax = lambda a, m: np.where(m[None, :], a, -np.inf).max(1)  # noqa: E731

    # FB3 ball_space
    imin = icpt.min(1)
    arrival_team = np.where(np.isfinite(imin), team_slot[icpt.argmin(1)], np.nan)
    q = Pi + vpi * PC_REACT
    tau = PC_REACT + np.hypot(Bi[:, None, 0] - q[..., 0], Bi[:, None, 1] - q[..., 1]) / PC_VMAX
    tau = np.where(vis, tau, np.inf).astype(np.float32)
    ppcf = np.zeros((n, S), np.float32)
    tot = np.zeros(n, np.float32)
    k_sig = np.float32(np.pi / np.sqrt(3) / PC_SIGMA)
    for t in np.arange(PC_DT, PC_TMAX + 1e-9, PC_DT, dtype=np.float32):
        pint = 1.0 / (1.0 + np.exp(-k_sig * (t - tau)))
        dp = (1 - tot)[:, None] * pint * np.float32(PC_LAMBDA * PC_DT)
        ppcf += dp
        tot = ppcf.sum(1)
    pc_A = np.where(tot > 0, ppcf[:, isA].sum(1) / np.maximum(tot, 1e-6), np.nan)
    cpos_H = np.where(isH[None, :], np.clip(close, 0, None), 0).sum(1)
    cpos_A = np.where(isA[None, :], np.clip(close, 0, None), 0).sum(1)
    fams["ball_space"] = np.column_stack([arrival_team, pc_A, cpos_H, cpos_A, cpos_A - cpos_H]).astype(np.float32)

    # FB4 ball_graph
    nb = np.zeros((n, 2), np.float32)
    for r in range(n):
        v = np.flatnonzero(vis[r])
        pts = np.vstack([Pi[r, v], Bi[r]])
        try:
            tri = Delaunay(pts)
        except QhullError:
            nb[r] = np.nan
            continue
        ip, ix = tri.vertex_neighbor_vertices
        nbv = v[ix[ip[len(v)]:ip[len(v) + 1]]]
        nb[r, 0] = (team_slot[nbv] == H).sum()
        nb[r, 1] = (team_slot[nbv] == A).sum()
    lanes = np.zeros((n, 2), np.float32)
    lanes_frac = np.zeros((n, 2), np.float32)
    for k, m in ((H, isH), (A, isA)):
        ks, os_ = np.flatnonzero(m), np.flatnonzero(~m)
        dteam = np.where(m[None, :], dinf, np.inf)
        org = dteam.argmin(1)
        for c0 in range(0, n, 5000):
            sl = slice(c0, c0 + 5000)
            rr = np.arange(c0, min(c0 + 5000, n))
            O = Pi[rr, org[sl]]
            T = Pi[sl][:, ks].copy()
            T[org[sl][:, None] == ks[None, :]] = np.nan
            Q = Pi[sl][:, os_]
            seg = T - O[:, None, :]
            wv = Q - O[:, None, :]
            l2 = np.maximum((seg ** 2).sum(-1), 1e-6)
            s = np.clip(np.einsum("ckd,cmd->ckm", seg, wv) / l2[..., None], 0, 1)
            cx = seg[..., None, 0] * s - wv[:, None, :, 0]
            cy = seg[..., None, 1] * s - wv[:, None, :, 1]
            dist = np.hypot(cx, cy)
            dist = np.where(np.isfinite(dist), dist, np.inf)
            free = (dist > LANE_CLEAR).all(-1) & np.isfinite(T[..., 0])
            nt = np.isfinite(T[..., 0]).sum(1)
            lanes[sl, k] = free.sum(1)
            lanes_frac[sl, k] = np.where(nt > 0, free.sum(1) / np.maximum(nt, 1), np.nan)
    fams["ball_graph"] = np.column_stack([
        nb[:, 0], nb[:, 1], nb[:, 1] / np.maximum(nb.sum(1), 1),
        lanes[:, 0], lanes[:, 1], lanes[:, 1] - lanes[:, 0],
        lanes_frac[:, 0], lanes_frac[:, 1]
    ]).astype(np.float32)

    # FB5 ball_motion
    cols = [vbi[:, 0], ball_speed]
    names = ["ball_vx_attack", "ball_speed"]
    for k, m, t in ((H, isH, "H"), (A, isA, "A")):
        cols += [np.log1p(np.minimum(tmin(ttc, m), 30)), np.minimum(tmin(icpt, m), 15),
                 np.clip(tmax(close, m), -20, 40)]
        names += [f"ttc_min_{t}", f"icpt_min_{t}", f"close_max_{t}"]
    cols.append(cols[names.index("icpt_min_H")] - cols[names.index("icpt_min_A")])
    cols.append(cols[names.index("ttc_min_H")] - cols[names.index("ttc_min_A")])
    bvx = vb[:, 0].astype(np.float64)
    for k, m, _ in ((H, isH, "H"), (A, isA, "A")):
        cen = np.nanmean(np.where(m[None, :] & vis_p, P[..., 0], np.nan), 1)
        cvx = featlib.velocities(cen[:, None])[:, 0].astype(np.float64)
        cors = np.stack([_wcorr(bvx, _shift(cvx, -L), LL_WIN) for L in LL_LAGS], 1)
        ok = np.isfinite(cors).any(1)
        best = np.where(ok, np.nanargmax(np.where(np.isfinite(cors), cors, -9), 1), 0)
        lag_s = np.where(ok, np.array(LL_LAGS)[best] / FPS, np.nan)
        cbest = np.where(ok, cors[np.arange(N), best], np.nan)
        c0 = cors[:, LL_LAGS.index(0)]
        cols += [lag_s[idx], cbest[idx], c0[idx]]
    fams["ball_motion"] = np.column_stack(cols).astype(np.float32)

    order = sorted(["ball_pressure", "ball_structure", "ball_space", "ball_graph", "ball_motion"])
    X_patch = np.concatenate([fams[k] for k in order], 1)
    X_new = X_ball_orig.copy()
    X_new[idx] = X_patch
    return X_new
