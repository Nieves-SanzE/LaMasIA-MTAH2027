"""Shared contract for the team-descriptor feature screening (ideas FE*/FB*).

Every feature family script computes per-frame features, saves them with `save_family` and
screens them with `screen`. The combination step loads all families with `load_families`.
Everything here is fixed a priori; nothing is tuned on the public window.

    from featlib import *
    tr = load_tracking()
    xy = oriented_xy(tr)          # (N, slots, 2) metres, H always attacks +x
    lab = labels(tr)              # pseudo-labels and masks
    save_family("geometry", X, names); screen(X, names, lab)

Labels (all full-length arrays, frame = video frame):
- `team`: exp-003 "hold the last clear control" (nearest player < 1.5 m for >= 12 frames),
  0=H, 1=A, -1 unknown. 83.4% frame accuracy vs the public GT on alive frames.
- `confident`: frame inside a clear-control run (stricter, smaller target).
- `train`: play frames with the ball visible and a known `team`, OUTSIDE the public window
  and its +-1 min margin. This is the only mask used to fit or screen on the full match.
- `blocks`: 5-minute block id, for blocked cross-validation (no temporal leakage).
- `gt`: public GT codes (0=H 1=A 2=D 3=X); only for read-only reporting, never for choices.
"""
import os
import warnings

import numpy as np
import pandas as pd

from common import *  # noqa: F401,F403  (ROOT, OUT, LO, HI, HALVES, pm, atd, gt_codes)

H, A = 0, 1
FPS = 25
KICKOFF_2H = 113742
PLAY = [(HALVES[0][0], HALVES[0][1]), (KICKOFF_2H, HALVES[1][1])]
CTRL_DIST = 1.5   # m (exp-003)
CTRL_FRAMES = 12  # frames (exp-003)
BLOCK = 5 * 60 * FPS
MARGIN = 60 * FPS  # excluded around the public window when training/screening on the full match
FEAT_DIR = os.path.join(OUT, "features")
os.makedirs(FEAT_DIR, exist_ok=True)
warnings.simplefilter("ignore", RuntimeWarning)

_TR = None


def load_tracking():
    global _TR
    if _TR is None:
        _TR = atd.load()
    return _TR


def play_mask(n):
    m = np.zeros(n, bool)
    for a, b in PLAY:
        m[a:b] = True
    return m


def attack_sign(tr):
    """(N,) +1 where H attacks +x, -1 where H attacks -x, 0 outside the halves.

    Derived from the data, not hard-coded: per half, each team's goalkeeper is the slot with
    >50% coverage whose mean x is most extreme; H attacks away from its goalkeeper.
    """
    sign = np.zeros(tr.n_frames, np.int8)
    vis = tr.visible
    for a, b in HALVES:
        cov = vis[a:b].mean(0)
        mx = np.nanmean(tr.xy[a:b, :, 0], 0)
        ext = np.where(cov > 0.5, np.abs(mx - 0.5), -1)
        gk_h = np.argmax(np.where(tr.equipo == H, ext, -1))
        sign[a:b] = 1 if mx[gk_h] < 0.5 else -1
    return sign


def oriented(a_m, sign):
    """Rotate metric coords by 180 deg where sign < 0 so H always attacks +x. Works for (N,2) or (N,S,2)."""
    L, W = float(atd.CAMPO[0]), float(atd.CAMPO[1])
    out = a_m.astype(np.float32).copy()
    flip = sign < 0
    out[flip, ..., 0] = L - out[flip, ..., 0]
    out[flip, ..., 1] = W - out[flip, ..., 1]
    return out


def oriented_xy(tr):
    return oriented(tr.metros(tr.xy), attack_sign(tr))


def oriented_ball(tr):
    return oriented(tr.metros(tr.balon), attack_sign(tr))


def velocities(p, window=7, order=2):
    """Savitzky-Golay velocity (m/s) along axis 0 of a metric position array; NaN-safe per run."""
    from scipy.signal import savgol_filter
    flat = p.reshape(p.shape[0], -1)
    v = np.full_like(flat, np.nan)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = np.isfinite(col)
        e = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
        for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            if t - s >= window:
                v[s:t, j] = savgol_filter(col[s:t], window, order, deriv=1) * FPS
    return v.reshape(p.shape)


def labels(tr):
    """Pseudo-labels and masks; see module docstring."""
    n = tr.n_frames
    near = tr.mas_cercano_al_balon()
    near_team = np.where(near >= 0, tr.equipo[np.clip(near, 0, None)], -1)
    d = tr.distancia_al_balon()
    dmin = np.nanmin(np.where(np.isfinite(d), d, np.inf), 1)
    close = np.where(dmin < CTRL_DIST, near_team, -1)
    s = pd.Series(close)
    run = s.groupby(s.ne(s.shift()).cumsum()).cumcount().values + 1
    stable = np.where((close >= 0) & (run >= CTRL_FRAMES), close, -1)
    # whole clear-control runs (including their first 11 frames) are "confident"
    conf_run = np.zeros(n, bool)
    grp = s.ne(s.shift()).cumsum().values
    good = pd.Series((close >= 0) & (run >= CTRL_FRAMES)).groupby(grp).transform("max").values
    conf_run[(close >= 0) & good] = True
    team = np.full(n, -1, np.int8)
    for a, b in PLAY:
        last = np.maximum.accumulate(np.where(stable[a:b] >= 0, np.arange(a, b), -1))
        team[a:b] = np.where(last >= 0, stable[np.clip(last, 0, None)], -1)
    play = play_mask(n)
    ball = tr.balon_visible
    pub = np.zeros(n, bool)
    pub[LO - MARGIN:HI + MARGIN] = True
    train = play & ball & (team >= 0) & ~pub
    blocks = np.arange(n) // BLOCK
    return dict(team=team, confident=conf_run & play, train=train, play=play, ball=ball,
                blocks=blocks, dmin=dmin, near_team=near_team, gt=gt_codes())


def run_bounds(mask):
    """Start/end (inclusive) of runs of True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def timed_team_labels(dmin, near_team, vis, periods, mode, max_flight=4 * FPS):
    """Pseudo-labels with the team switch placed according to `mode` (exp-027, T10; exp-055, T16):
    base = 11 frames into the new team's stable control run (exp-003, same as labels()); ctrl = first
    frame of that run; release = frame after the old team's last close frame; mid = halfway between
    release and ctrl; clean_mid = mid when the flight is <= max_flight with >= 50% ball visibility,
    else ctrl (prevents placing switches halfway across long ball gaps or stoppages, exp-055)."""
    close = np.where((dmin < CTRL_DIST) & vis, near_team, -1).astype(np.int8)
    team = np.full(len(dmin), -1, np.int8)
    for a, b in periods:
        c = close[a:b]
        v_sub = vis[a:b]
        # stable control runs: a run of the same team close to the ball lasting >= CTRL_FRAMES
        runs = []
        for t in (0, 1):
            s, e = run_bounds(c == t)
            keep = (e - s + 1) >= CTRL_FRAMES
            runs += [(int(x), int(y), t) for x, y in zip(s[keep], e[keep])]
        runs.sort()
        lab = np.full(b - a, -1, np.int8)
        prev_team, prev_from = -1, 0
        for s, e, t in runs:
            if t != prev_team:
                if prev_team < 0:
                    sw = s + CTRL_FRAMES - 1 if mode == "base" else s
                else:
                    last_old = np.flatnonzero(c[prev_from:s] == prev_team)
                    rel = prev_from + last_old[-1] + 1 if len(last_old) else s
                    if mode == "clean_mid":
                        ok_flight = (s - rel <= max_flight) and (s == rel or v_sub[rel:s].mean() >= 0.5)
                        sw = (rel + s) // 2 if ok_flight else s
                    else:
                        sw = {"base": s + CTRL_FRAMES - 1, "ctrl": s, "release": rel, "mid": (rel + s) // 2}[mode]
                lab[sw:] = t
                prev_team = t
            prev_from = max(prev_from, s)
        team[a:b] = lab
    return team


def relabel(lab, mode):
    """labels() output with the team switch timing of timed_team_labels(mode); same masks otherwise."""
    team = timed_team_labels(lab["dmin"], lab["near_team"], lab["ball"], PLAY, mode)
    pub = np.zeros(len(team), bool)
    pub[LO - MARGIN:HI + MARGIN] = True
    return dict(lab, team=team, train=lab["play"] & lab["ball"] & (team >= 0) & ~pub)


def save_family(family, X, names):
    X = np.asarray(X, np.float32)
    assert X.ndim == 2 and X.shape[1] == len(names), (X.shape, len(names))
    np.savez_compressed(os.path.join(FEAT_DIR, f"{family}.npz"), X=X, names=np.array(names))


def load_families(families=None):
    files = sorted(f for f in os.listdir(FEAT_DIR) if f.endswith(".npz"))
    Xs, names = [], []
    for f in files:
        fam = f[:-4]
        if families is not None and fam not in families:
            continue
        z = np.load(os.path.join(FEAT_DIR, f))
        Xs.append(z["X"])
        names += [f"{fam}:{n}" for n in z["names"]]
    return np.concatenate(Xs, 1), names


def _auc(x, y):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(x)
    if ok.sum() < 100 or len(np.unique(y[ok])) < 2:
        return np.nan, ok.mean() if len(ok) else np.nan
    return roc_auc_score(y[ok], x[ok]), ok.mean()


def screen(X, names, lab, top=None):
    """Per-feature AUC for "A in possession" (oriented: >0.5 means larger value -> A).

    Columns: full-match pseudo-label (train mask), confident subset, and READ-ONLY public GT
    on alive frames with the ball visible / missing (real gaps of the window, small n).
    Prints a table sorted by |AUC - 0.5| on the full match. Returns the DataFrame.
    """
    y = lab["team"]
    tr_m, cf_m = lab["train"], lab["train"] & lab["confident"]
    g = lab["gt"]
    w = np.zeros(len(g), bool)
    w[LO:HI] = True
    alive = w & (g < 2)
    rows = []
    for j, n in enumerate(names):
        x = X[:, j]
        a_full, cov = _auc(x[tr_m], y[tr_m])
        a_conf, _ = _auc(x[cf_m], y[cf_m])
        a_gv, _ = _auc(x[alive & lab["ball"]], g[alive & lab["ball"]])
        a_gm, _ = _auc(x[alive & ~lab["ball"]], g[alive & ~lab["ball"]])
        rows.append((n, a_full, a_conf, a_gv, a_gm, cov))
    df = pd.DataFrame(rows, columns=["feature", "auc_pseudo", "auc_confident", "auc_gt_ballvis",
                                     "auc_gt_ballmiss", "coverage"])
    df["strength"] = (df["auc_pseudo"] - 0.5).abs()
    df = df.sort_values("strength", ascending=False).drop(columns="strength")
    with pd.option_context("display.width", 160, "display.max_rows", 500):
        print(df.head(top).round(3).to_string(index=False) if top else df.round(3).to_string(index=False))
    n_gm = int((alive & ~lab["ball"]).sum())
    print(f"(public GT columns are read-only; ball-missing alive frames in the window: n={n_gm})")
    return df


def gbm_cv(X, lab, mask=None, seed=0):
    """Blocked 5-fold CV (contiguous 5-min blocks) of a HistGradientBoosting classifier.

    Returns out-of-fold P(A) for the masked frames (NaN elsewhere) and prints accuracy/AUC.
    Fixed hyper-parameters (library defaults + 200 iterations), not tuned.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    m = lab["train"] if mask is None else mask
    idx = np.flatnonzero(m)
    blocks = lab["blocks"][idx]
    ub = np.unique(blocks)
    rng = np.random.default_rng(seed)
    fold_of_block = dict(zip(ub, rng.permutation(len(ub)) % 5))
    folds = np.array([fold_of_block[b] for b in blocks])
    p = np.full(len(m), np.nan, np.float32)
    y = lab["team"]
    for k in range(5):
        te, trn = idx[folds == k], idx[folds != k]
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        clf.fit(X[trn], y[trn])
        p[te] = clf.predict_proba(X[te])[:, 1]
    acc = np.mean((p[idx] > 0.5) == (y[idx] == A))
    auc = roc_auc_score(y[idx] == A, p[idx])
    print(f"GBM blocked-CV on {len(idx)} frames: acc {acc:.3f}  AUC {auc:.3f}")
    return p
