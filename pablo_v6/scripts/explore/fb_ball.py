"""FB1-FB6: team descriptors that USE the ball, and a label-generation experiment (ideas FB*).

Families (NaN where the ball is not visible or outside the graded halves), saved with featlib:
  ball_pressure  (FB1) players per team within 3/5/10 m, nearest / 2nd nearest per team, dmin_H - dmin_A
  ball_structure (FB2) ball inside each outfield convex hull (signed distance), ball x vs each team's
                       defensive line and centroid, players goal-side of the ball, distance to goals
  ball_space     (FB3) team of the earliest interceptor (arrival-time Voronoi, exp-010 model),
                       Spearman/Shaw pitch control of A at the ball, sum of positive closing speeds
  ball_graph     (FB4) Delaunay(22 + ball) neighbours of the ball per team; free passing lanes
                       (no opponent within 2 m of the segment) from each team's nearest player
  ball_motion    (FB5) ball velocity along the attack axis, ball speed, per-team min TTC / intercept
                       time / max closing speed, lead-lag ball vs team-centroid velocity (4 s window)

FB6: GBM (fixed hyper-parameters) trained only on `train & confident` frames (clear-control runs,
outside the public window +-1 min) with the five families; out-of-fold P(A) by 5-min blocks on all
ball-visible play frames; fixed smoothing (centred 1 s rolling median, hysteresis 0.35/0.65 holding
the previous team, hold through gaps). Compared with exp-003 hold-last-control: agreement, read-only
public GT accuracy, spellF1 in the window (exp-003 speed D in both), switches/min, and the exp-007
gap-injection harness. All constants are a priori; the public window is only reported, never tuned.

Only aggregates are printed.  Run from the repo root:
    uv run python scripts/explore/fb_ball.py            (recompute features, ~min)
    uv run python scripts/explore/fb_ball.py --cached   (reuse saved families, FB6 only)
"""
import os
import sys
import time

from scipy.spatial import ConvexHull, Delaunay, QhullError
from sklearn.ensemble import HistGradientBoostingClassifier

from featlib import *  # noqa: F401,F403
import gaplib

T0 = time.time()
D, X = 2, 3
FAMILIES = ["ball_pressure", "ball_structure", "ball_space", "ball_graph", "ball_motion"]
L_PITCH, W_PITCH = float(atd.CAMPO[0]), float(atd.CAMPO[1])
# a-priori constants
VMAX_PLAYER, VMAX_BALL = 12.0, 40.0           # m/s, above = tracking artefact (exp-010)
T_REACT, VMAX_ICPT, BALL_DECEL = 0.7, 7.0, 2.0  # exp-010 interceptor
T_GRID = np.arange(0, 6.0 + 1e-6, 0.1, dtype=np.float32)
PC_VMAX, PC_REACT, PC_SIGMA, PC_LAMBDA = 5.0, 0.7, 0.45, 4.3  # Spearman 2017 / Shaw (published)
PC_DT, PC_TMAX = 0.04, 10.0
LANE_CLEAR = 2.0                               # m, opponent distance to a passing segment
LL_WIN, LL_LAGS = 4 * FPS, list(range(-FPS, FPS + 1, 5))  # lead-lag: 4 s window, lags +-1 s
FAST = 12.0                                    # m/s, "fast ball" (exp-010)
LOOSE = CTRL_DIST                              # dmin > 1.5 m
P_MED, HYST_LO, HYST_HI = FPS, 0.35, 0.65      # FB6 fixed smoothing


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


tr = load_tracking()
N, S = tr.n_frames, tr.n_slots
team_slot = tr.equipo.astype(np.int8)
lab = labels(tr)
graded = np.zeros(N, bool)
for a, b in HALVES:
    graded[a:b] = True


# ====================================================================== features
def compute_families():
    P = oriented_xy(tr)                       # (N, S, 2) metres, H attacks +x
    B = oriented_ball(tr)                     # (N, 2)
    vis_p = np.isfinite(P[..., 0])
    fm = tr.balon_visible & graded & np.isfinite(B[:, 0])
    idx = np.flatnonzero(fm)
    n = len(idx)
    log(f"ball-visible graded frames: {n} ({n / graded.sum():.1%} of graded)")

    vb = velocities(B)
    vb[~(np.hypot(vb[:, 0], vb[:, 1]) <= VMAX_BALL)] = np.nan
    vp = velocities(P)
    bad = ~(np.hypot(vp[..., 0], vp[..., 1]) <= VMAX_PLAYER)
    vp[bad] = 0                               # unknown / artefact -> static (exp-010)
    log("velocities done")

    Pi, Bi, vpi = P[idx], B[idx], vp[idx]
    vbi = vb[idx]
    vbi0 = np.where(np.isfinite(vbi), vbi, 0)  # unknown ball velocity -> static ball
    vis = vis_p[idx]
    isH, isA = team_slot == H, team_slot == A
    rel = Bi[:, None, :] - Pi
    d = np.hypot(rel[..., 0], rel[..., 1])     # (n, S), NaN invisible
    dinf = np.where(np.isfinite(d), d, np.inf)

    def full(cols):
        out = np.full((N, len(cols)), np.nan, np.float32)
        out[idx] = np.column_stack(cols)
        return out

    fams = {}
    # ---------------------------------------------------------------- FB1 ball_pressure
    cols, names = [], []
    dk = {}
    for k, m, t in ((H, isH, "H"), (A, isA, "A")):
        dd = np.sort(np.where(m[None, :], dinf, np.inf), 1)[:, :2]
        dk[k] = dd
        for r in (3, 5, 10):
            cols.append((np.where(m[None, :], dinf, np.inf) < r).sum(1)); names.append(f"n{r}_{t}")
        cols += [np.minimum(dd[:, 0], 100), np.minimum(dd[:, 1], 100)]; names += [f"d1_{t}", f"d2_{t}"]
    cols.append(np.minimum(dk[H][:, 0], 100) - np.minimum(dk[A][:, 0], 100)); names.append("dmin_H_minus_A")
    cols.append(cols[names.index("n5_A")] - cols[names.index("n5_H")]); names.append("n5_A_minus_H")
    fams["ball_pressure"] = (full(cols), names)
    log("FB1 done")

    # ---------------------------------------------------------------- FB2 ball_structure
    x = Pi[..., 0]
    bx, by = Bi[:, 0], Bi[:, 1]
    out_m = {}
    for k, m in ((H, isH), (A, isA)):
        xk = np.where(m[None, :] & vis, x, np.nan)
        has = np.isfinite(xk).any(1)
        # goalkeeper = deepest player towards the own goal (H defends x=0, A defends x=L)
        gk = np.nanargmin(np.where(has[:, None], xk, 0), 1) if k == H else np.nanargmax(np.where(has[:, None], xk, 0), 1)
        o = m[None, :] & vis
        o[np.arange(n), gk] = False
        out_m[k] = o
    cols, names = [], []
    for k, t in ((H, "H"), (A, "A")):
        xo = np.where(out_m[k], x, np.nan)
        defl = np.nanmin(xo, 1) if k == H else np.nanmax(xo, 1)
        cen = np.nanmean(xo, 1)
        gs = (out_m[k] & ((x < bx[:, None]) if k == H else (x > bx[:, None]))).sum(1)
        cols += [bx - defl, bx - cen, gs]; names += [f"bx_minus_defline_{t}", f"bx_minus_centroid_{t}", f"goalside_{t}"]
    cols += [np.hypot(bx, by - W_PITCH / 2), np.hypot(L_PITCH - bx, by - W_PITCH / 2)]
    names += ["dist_goal_H", "dist_goal_A"]
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
    names += ["hull_sd_H", "hull_sd_A", "in_hull_H", "in_hull_A", "hull_sd_H_minus_A"]
    fams["ball_structure"] = (full(cols), names)
    log("FB2 done")

    # ---------------------------------------------------------------- kinematics (exp-010)
    close = ((vpi - vbi0[:, None, :]) * rel).sum(-1) / np.maximum(d, 1e-3)   # closing speed
    ttc = np.where(close > 0, d / close, np.inf)
    ttc[~np.isfinite(d) | ~np.isfinite(close)] = np.inf
    close = np.where(np.isfinite(close), close, -np.inf)
    del rel
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
    log("interceptor done")

    def tmin(a, m):
        return np.where(m[None, :], a, np.inf).min(1)

    def tmax(a, m):
        return np.where(m[None, :], a, -np.inf).max(1)

    # ---------------------------------------------------------------- FB3 ball_space
    imin = icpt.min(1)
    arrival_team = np.where(np.isfinite(imin), team_slot[icpt.argmin(1)], np.nan)
    # Shaw's implementation of Spearman (2017) pitch control at the ball's current position
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
    log(f"pitch control done (median total {np.median(tot):.3f})")
    cpos_H = np.where(isH[None, :], np.clip(close, 0, None), 0).sum(1)
    cpos_A = np.where(isA[None, :], np.clip(close, 0, None), 0).sum(1)
    fams["ball_space"] = (full([arrival_team, pc_A, cpos_H, cpos_A, cpos_A - cpos_H]),
                          ["arrival_team", "pitch_control_A", "close_pos_sum_H", "close_pos_sum_A",
                           "close_pos_sum_A_minus_H"])
    log("FB3 done")

    # ---------------------------------------------------------------- FB4 ball_graph
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
        nbv = v[ix[ip[len(v)]:ip[len(v) + 1]]]  # neighbours of the ball vertex (last)
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
            O = Pi[rr, org[sl]]                                    # (c, 2)
            T = Pi[sl][:, ks].copy()                               # (c, K, 2)
            T[org[sl][:, None] == ks[None, :]] = np.nan            # not a lane to oneself
            Q = Pi[sl][:, os_]                                     # (c, M, 2)
            seg = T - O[:, None, :]
            wv = Q - O[:, None, :]
            l2 = np.maximum((seg ** 2).sum(-1), 1e-6)              # (c, K)
            s = np.clip(np.einsum("ckd,cmd->ckm", seg, wv) / l2[..., None], 0, 1)
            cx = seg[..., None, 0] * s - wv[:, None, :, 0]
            cy = seg[..., None, 1] * s - wv[:, None, :, 1]
            dist = np.hypot(cx, cy)
            dist = np.where(np.isfinite(dist), dist, np.inf)
            free = (dist > LANE_CLEAR).all(-1) & np.isfinite(T[..., 0])
            nt = np.isfinite(T[..., 0]).sum(1)
            lanes[sl, k] = free.sum(1)
            lanes_frac[sl, k] = np.where(nt > 0, free.sum(1) / np.maximum(nt, 1), np.nan)
    fams["ball_graph"] = (full([nb[:, 0], nb[:, 1], nb[:, 1] / np.maximum(nb.sum(1), 1),
                                lanes[:, 0], lanes[:, 1], lanes[:, 1] - lanes[:, 0],
                                lanes_frac[:, 0], lanes_frac[:, 1]]),
                          ["ball_nb_H", "ball_nb_A", "ball_nb_frac_A", "free_lanes_H", "free_lanes_A",
                           "free_lanes_A_minus_H", "free_lanes_frac_H", "free_lanes_frac_A"])
    log("FB4 done")

    # ---------------------------------------------------------------- FB5 ball_motion
    cols = [vbi[:, 0], ball_speed]
    names = ["ball_vx_attack", "ball_speed"]
    for k, m, t in ((H, isH, "H"), (A, isA, "A")):
        cols += [np.log1p(np.minimum(tmin(ttc, m), 30)), np.minimum(tmin(icpt, m), 15),
                 np.clip(tmax(close, m), -20, 40)]
        names += [f"ttc_min_{t}", f"icpt_min_{t}", f"close_max_{t}"]
    cols.append(cols[names.index("icpt_min_H")] - cols[names.index("icpt_min_A")]); names.append("icpt_H_minus_A")
    cols.append(cols[names.index("ttc_min_H")] - cols[names.index("ttc_min_A")]); names.append("ttc_H_minus_A")
    # lead-lag: windowed correlation of ball vx with each team centroid vx at lags (frames)
    bvx = vb[:, 0].astype(np.float64)
    for k, m, t in ((H, isH, "H"), (A, isA, "A")):
        cen = np.nanmean(np.where(m[None, :] & vis_p, P[..., 0], np.nan), 1)
        cvx = velocities(cen[:, None])[:, 0].astype(np.float64)
        cors = np.stack([_wcorr(bvx, _shift(cvx, -L), LL_WIN) for L in LL_LAGS], 1)  # (N, lags)
        ok = np.isfinite(cors).any(1)
        best = np.where(ok, np.nanargmax(np.where(np.isfinite(cors), cors, -9), 1), 0)
        lag_s = np.where(ok, np.array(LL_LAGS)[best] / FPS, np.nan)
        cbest = np.where(ok, cors[np.arange(N), best], np.nan)
        c0 = cors[:, LL_LAGS.index(0)]
        cols += [lag_s[idx], cbest[idx], c0[idx]]
        names += [f"leadlag_best_lag_s_{t}", f"leadlag_best_corr_{t}", f"leadlag_corr0_{t}"]
    fams["ball_motion"] = (full(cols), names)
    log("FB5 done")
    return fams


def _shift(x, L):
    """y[t] = x[t - L] (L may be negative), NaN-padded."""
    y = np.full_like(x, np.nan)
    if L > 0:
        y[L:] = x[:-L]
    elif L < 0:
        y[:L] = x[-L:]
    else:
        y[:] = x
    return y


def _wcorr(x, y, win, min_n=None):
    """Centred rolling Pearson correlation over frames where both are finite."""
    min_n = min_n or win // 2
    m = np.isfinite(x) & np.isfinite(y)
    xs, ys = np.where(m, x, 0), np.where(m, y, 0)

    def rs(a):
        c = np.concatenate(([0], np.cumsum(a)))
        h = win // 2
        i = np.arange(len(a))
        lo, hi = np.clip(i - h, 0, len(a)), np.clip(i + win - h, 0, len(a))
        return c[hi] - c[lo]

    nn = rs(m.astype(np.float64))
    sx, sy, sxx, syy, sxy = rs(xs), rs(ys), rs(xs * xs), rs(ys * ys), rs(xs * ys)
    cov = sxy - sx * sy / np.maximum(nn, 1)
    vx = sxx - sx ** 2 / np.maximum(nn, 1)
    vy = syy - sy ** 2 / np.maximum(nn, 1)
    r = cov / np.sqrt(np.maximum(vx * vy, 1e-12))
    return np.where((nn >= min_n) & (vx > 1e-9) & (vy > 1e-9), r, np.nan)


if "--cached" in sys.argv:
    log("using cached families")
else:
    fams = compute_families()
    for fam, (Xf, nm) in fams.items():
        save_family(fam, Xf, nm)
    log("families saved")
    for fam, (Xf, nm) in fams.items():
        print(f"\n=== screen {fam} ({len(nm)} features) ===")
        screen(Xf, nm, lab)
        gbm_cv(Xf, lab)
    del fams

Xall, all_names = load_families(FAMILIES)
print(f"\n=== all ball families together ({len(all_names)} features) ===")
p_cv_all = gbm_cv(Xall, lab)
log("screening done")

# ====================================================================== FB6
print("\n" + "=" * 90 + "\nFB6: labels from ball features (trained on confident frames only) vs exp-003 hold")
y = lab["team"]
play, ballv, dmin, gt = lab["play"], lab["ball"], lab["dmin"], lab["gt"]
pred_m = play & ballv
blocks = lab["blocks"]
ub = np.unique(blocks[pred_m])
fold_of_block = dict(zip(ub, np.random.default_rng(0).permutation(len(ub)) % 5))
fold = np.full(N, -1)
fold[pred_m] = [fold_of_block[b] for b in blocks[pred_m]]


def oof(train_mask):
    p = np.full(N, np.nan, np.float32)
    for k in range(5):
        trn = np.flatnonzero(train_mask & (fold != k) & (fold >= 0))
        te = np.flatnonzero(pred_m & (fold == k))
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xall[trn], y[trn])
        p[te] = clf.predict_proba(Xall[te])[:, 1]
    return p


conf_train = lab["train"] & lab["confident"]
assert not (conf_train[LO - MARGIN:HI + MARGIN]).any()
print(f"training frames: confident {conf_train.sum()} (of train {lab['train'].sum()}); "
      f"predicted frames (ball-visible play) {pred_m.sum()}")
P_CACHE = os.path.join(OUT, "fb6_oof.npz")
if "--cached" in sys.argv and os.path.exists(P_CACHE):
    z = np.load(P_CACHE)
    P_conf, P_all = z["P_conf"], z["P_all"]
else:
    P_conf = oof(conf_train)
    P_all = oof(lab["train"])
    np.savez_compressed(P_CACHE, P_conf=P_conf, P_all=P_all)
log("FB6 OOF models done")
lt = lab["train"]
for nm_, p in (("conf-trained", P_conf), ("all-trained", P_all)):
    nc = lt & ~lab["confident"]
    print(f"  {nm_}: OOF agreement with pseudo-label (p>0.5) on train frames {np.mean((p[lt] > .5) == (y[lt] == A)):.3f}; "
          f"on non-confident train frames {np.mean((p[nc] > .5) == (y[nc] == A)):.3f} (n={nc.sum()})")


def smooth_hyst(p, segs, mask):
    """Centred 1 s rolling median of P(A) over visible frames, hysteresis 0.35/0.65, hold in gaps."""
    pm_ = np.where(mask, p, np.nan)
    med = pd.Series(pm_).rolling(P_MED, center=True, min_periods=1).median().to_numpy(copy=True)
    med[~mask] = np.nan
    out = np.full(N, -1, np.int8)
    for a, b in segs:
        st = -1
        seg = med[a:b]
        o = out[a:b]
        for i in range(b - a):
            v = seg[i]
            if v >= HYST_HI:
                st = A
            elif v <= HYST_LO:
                st = H
            o[i] = st
    return out


T_base = y.copy()
T_conf = smooth_hyst(P_conf, PLAY, pred_m)
T_all = smooth_hyst(P_all, PLAY, pred_m)
T_rawconf = np.where(pred_m & np.isfinite(P_conf), (P_conf > 0.5).astype(np.int8), -1)
methods = {"exp-003 hold": T_base, "FB6 conf-trained": T_conf, "FB6 all-trained": T_all,
           "raw argmax conf (no smoothing)": T_rawconf}
bspeed = np.hypot(*velocities(tr.metros(tr.balon)).T)
bspeed[bspeed > VMAX_BALL] = np.nan
loose = pred_m & (dmin > LOOSE)
fast_loose = loose & (bspeed >= FAST)

print("\n(a) agreement with exp-003 hold, full match (frames where both known)")
cats = [("ball-visible play", pred_m), ("controlled dmin<=1.5", pred_m & (dmin <= LOOSE)),
        ("loose dmin>1.5", loose), ("loose & ball>=12 m/s", fast_loose),
        ("loose & ball<5 m/s", loose & (bspeed < 5)), ("ball-missing play (held)", play & ~ballv)]
print(f"  {'method':<32}" + "".join(f"{c:>26}" for c, _ in cats))
print(f"  {'n frames':<32}" + "".join(f"{m.sum():>26d}" for _, m in cats))
for k, T in methods.items():
    if k == "exp-003 hold":
        continue
    row = []
    for _, m in cats:
        mm = m & (T >= 0) & (T_base >= 0)
        row.append(f"{np.mean(T[mm] == T_base[mm]):>26.3f}" if mm.any() else f"{'-':>26}")
    print(f"  {k:<32}" + "".join(row))

print("\n(b) READ-ONLY public GT: team accuracy on alive frames of the window (unknown = wrong)")
w = np.zeros(N, bool)
w[LO:HI] = True
alive = w & (gt < 2)
cats_gt = [("alive ball-visible", alive & ballv), ("controlled", alive & ballv & (dmin <= LOOSE)),
           ("loose", alive & loose), ("loose & fast>=12", alive & fast_loose),
           ("alive ball-missing", alive & ~ballv)]
print(f"  {'method':<32}" + "".join(f"{c:>22}" for c, _ in cats_gt))
print(f"  {'n frames':<32}" + "".join(f"{m.sum():>22d}" for _, m in cats_gt))
for k, T in methods.items():
    print(f"  {k:<32}" + "".join(f"{np.mean(T[m] == gt[m]):>22.3f}" for _, m in cats_gt))

print("\n(c) READ-ONLY public window spellF1 / macroF1 (real ball; exp-003 speed D in both; reported once)")
F = gaplib.Features(tr)


def to_codes(T, lo=LO, hi=HI):
    out = np.full(N, X, np.int8)
    t = np.where(T[lo:hi] >= 0, T[lo:hi], X).astype(np.int8)
    t[F.slow[lo:hi] & (t != X)] = D
    out[lo:hi] = t
    return out


for k, T in methods.items():
    r = pm.evaluate(gt, to_codes(T))
    print(f"  {k:<32} spellF1 {r['spell_f1_mean']:.3f}  macroF1 {r['macro_f1']:.3f}  spells {r['n_pred_spells']}")
r = pm.evaluate(gt, gaplib.predict(F, ballv, [(LO, HI)], "(a) exp-003 baseline"))
print(f"  {'(ref) gaplib exp-003, fresh at LO':<32} spellF1 {r['spell_f1_mean']:.3f}  macroF1 {r['macro_f1']:.3f}  "
      f"spells {r['n_pred_spells']}  (n GT spells {r['n_gt_spells']})")

print("\n(d) fragmentation on the full match (play frames)")
play_min = play.sum() / FPS / 60
for k, T in methods.items():
    a_, b_ = T[:-1], T[1:]
    sw = (a_ >= 0) & (b_ >= 0) & (a_ != b_) & play[1:] & play[:-1]
    lm = loose[1:] & loose[:-1]
    print(f"  {k:<32} team switches/min {sw.sum() / play_min:6.2f}   (on loose-ball time "
          f"{(sw & lm).sum() / (loose.sum() / FPS / 60):6.2f}/min)  H share {np.mean(T[play & (T >= 0)] == H):.3f}")
log("FB6 a-d done")

# ---------------------------------------------------------------------- (e) gap-injection harness
print("\n(e) exp-007 gap-injection harness (19 donors; in gaps both methods hold; exp-003 speed D)")
BLK = HI - LO
donors = [s for s in range(LO + BLK, HALVES[0][1] - BLK + 1, BLK)]
donors += [s for s in range(KICKOFF_2H, HALVES[1][1] - BLK + 1, BLK)]
bv = tr.balon_visible
res = {"exp-003 (gaplib)": [], "FB6 conf-trained": [], "FB6 all-trained": []}
for s in [None] + donors:
    vis = bv.copy()
    if s is not None:
        vis[LO:HI] &= bv[s:s + BLK]
    r = pm.evaluate(gt, gaplib.predict(F, vis, [(LO, HI)], "(a) exp-003 baseline"))
    res["exp-003 (gaplib)"].append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
    for k, p in (("FB6 conf-trained", P_conf), ("FB6 all-trained", P_all)):
        T = smooth_hyst(p, [(LO, HI)], pred_m & vis)
        r = pm.evaluate(gt, to_codes(T))
        res[k].append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]))
ref = np.array(res["exp-003 (gaplib)"])[1:, 0]
print(f"  {'method':<22} {'noinj sF1':>9} | {'donors sF1':>14} {'dmean':>7} {'W/L/T':>9} | {'macroF1':>7} | {'#spells':>7}")
for k, lst in res.items():
    arr = np.array(lst)
    sp = arr[1:, 0]
    dl = sp - ref
    print(f"  {k:<22} {arr[0, 0]:>9.3f} | {sp.mean():>7.3f}±{sp.std():.3f} {dl.mean():>+7.3f} "
          f"{(dl > 1e-9).sum():>3d}/{(dl < -1e-9).sum():>2d}/{(np.abs(dl) <= 1e-9).sum():>2d} | "
          f"{arr[1:, 1].mean():>7.3f} | {arr[1:, 2].mean():>7.1f}")
print("  (harness caveat: P(A) comes from features computed with the real ball; lead-lag windows can see"
      " ball frames hidden by the donor mask near gap edges)")
log("done")
