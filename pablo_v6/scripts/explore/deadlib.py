"""Ball-free live/dead features computed identically on our match and on Metrica (exp-019, D-T5).

Only player positions are used (metres, any pitch orientation): the ball cannot be used because
in Metrica it is missing almost exclusively while the ball is dead (visible 1-17% of dead frames
vs 84-91% alive), so a model trained there would learn "no ball = dead". Velocities are derived
from positions in both datasets (Savitzky-Golay, as featlib.velocities) for consistency.

Per-frame base signals (all symmetric in the two teams and in the pitch direction):
  speed: mean, median, p90, fraction < 0.5 / < 1 / > 2 / > 4 m/s, slower team mean, faster team mean
  motion coherence: Kuramoto order of the heading of players moving > 1 m/s
  shape: std of x and y, |centroid x - halfway|, |centroid y - middle|, players inside the busier box,
         players within 3 m of a touchline, players within 3 m of a goal line
Context (Lang et al. 2022): rolling means of each signal over 1, 2, 4, 8, 16, 32 s, past / centred /
future, plus future-minus-past at 2 and 8 s (onsets and restarts).
"""
import numpy as np
import pandas as pd

FPS = 25
WINDOWS_S = (1, 2, 4, 8, 16, 32)


def sg_velocity(p, window=7, order=2):
    """(N, S, 2) metres -> (N, S, 2) m/s, NaN-safe per contiguous run."""
    from scipy.signal import savgol_filter
    v = np.full_like(p, np.nan)
    for j in range(p.shape[1]):
        ok = np.isfinite(p[:, j, 0])
        e = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
        for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            if t - s >= window:
                v[s:t, j] = savgol_filter(p[s:t, j], window, order, deriv=1, axis=0) * FPS
    return v


def base_signals(xy, team, pitch):
    """xy (N, S, 2) metres, team (S,) 0/1, pitch (L, W) metres -> DataFrame of per-frame signals."""
    L, W = float(pitch[0]), float(pitch[1])
    v = sg_velocity(xy)
    sp = np.hypot(v[..., 0], v[..., 1])
    ok = np.isfinite(sp)
    n = np.maximum(ok.sum(1), 1)

    def frac(m):
        return np.where(ok.sum(1) > 0, (m & ok).sum(1) / n, np.nan)

    with np.errstate(invalid="ignore"):
        tm = np.stack([np.nanmean(np.where(team[None] == k, sp, np.nan), 1) for k in (0, 1)], 1)
        mov = ok & (sp > 1.0)
        ang = np.arctan2(v[..., 1], v[..., 0])
        kur = np.abs(np.nansum(np.where(mov, np.exp(1j * ang), 0), 1)) / np.maximum(mov.sum(1), 1)
        kur[mov.sum(1) < 4] = np.nan
        x, y = xy[..., 0], xy[..., 1]
        box = lambda x0, x1: ((x >= x0) & (x <= x1) & (np.abs(y - W / 2) <= 20.16)).sum(1)  # noqa: E731
        sig = dict(
            sp_mean=np.nanmean(sp, 1), sp_med=np.nanmedian(sp, 1), sp_p90=np.nanpercentile(sp, 90, 1),
            f_lt05=frac(sp < 0.5), f_lt1=frac(sp < 1.0), f_gt2=frac(sp > 2.0), f_gt4=frac(sp > 4.0),
            team_slow=np.nanmin(tm, 1), team_fast=np.nanmax(tm, 1), kuramoto=kur,
            std_x=np.nanstd(x, 1), std_y=np.nanstd(y, 1),
            cx_off=np.abs(np.nanmean(x, 1) - L / 2), cy_off=np.abs(np.nanmean(y, 1) - W / 2),
            in_box=np.maximum(box(0, 16.5), box(L - 16.5, L)).astype(np.float32),
            near_touch=((y < 3) | (y > W - 3)).sum(1).astype(np.float32),
            near_goalline=((x < 3) | (x > L - 3)).sum(1).astype(np.float32),
        )
    return pd.DataFrame(sig)


def context_features(sig, windows_s=WINDOWS_S):
    """Multi-scale past / centred / future rolling means + future-minus-past deltas."""
    cols = {}
    for c in sig.columns:
        s = sig[c]
        for w_s in windows_s:
            w = w_s * FPS
            past = s.rolling(w, min_periods=1).mean()
            fut = s[::-1].rolling(w, min_periods=1).mean()[::-1]
            cols[f"{c}_p{w_s}"] = past.values
            cols[f"{c}_c{w_s}"] = s.rolling(w, center=True, min_periods=1).mean().values
            cols[f"{c}_f{w_s}"] = fut.values
            if w_s in (2, 8):
                cols[f"{c}_d{w_s}"] = (fut - past).values
    return pd.DataFrame(cols).astype(np.float32)


def dead_features(xy, team, pitch, windows_s=WINDOWS_S):
    sig = base_signals(xy, team, pitch)
    if windows_s != WINDOWS_S:
        sig = sig.astype(np.float32)
    return context_features(sig, windows_s=windows_s)


def mask_no_velocity(xy, vel):
    """Mask out off-camera held/interpolated slots whose provider velocity is NaN (exp-055, D-T10)."""
    out = xy.copy()
    out[~np.isfinite(vel)] = np.nan
    return out


def dedupe(xy, team, max_per_team=11):
    """Tracker duplicates (our match): while a team has > 11 slots in a frame, drop the slot that is
    closest to a team-mate (same rule as fe_geometry). Returns a copy."""
    xy = xy.copy()
    for k in (0, 1):
        cols = np.flatnonzero(team == k)
        sub = xy[:, cols]
        cnt = np.isfinite(sub[..., 0]).sum(1)
        for f in np.flatnonzero(cnt > max_per_team):
            p = sub[f]
            while np.isfinite(p[:, 0]).sum() > max_per_team:
                ok = np.flatnonzero(np.isfinite(p[:, 0]))
                d = np.hypot(*(p[ok, None] - p[None, ok]).transpose(2, 0, 1))
                d[np.arange(len(ok)), np.arange(len(ok))] = np.inf
                p[ok[np.argmin(d.min(1))]] = np.nan
            sub[f] = p
        xy[:, cols] = sub
    return xy


def truncate_like(xy, counts, block=250, rng=None):
    """Structured dropout for training augmentation: in blocks of `block` frames keep only the `k`
    players closest in x to the block's player centroid (a camera that frames the play), with k
    drawn from `counts` (visible players per frame of our match)."""
    rng = np.random.default_rng(0) if rng is None else rng
    xy = xy.copy()
    n = len(xy)
    for s in range(0, n, block):
        k = int(rng.choice(counts))
        seg = xy[s:s + block]
        cx = np.nanmean(seg[..., 0])
        dist = np.abs(np.nanmean(seg[..., 0], 0) - cx)
        on = np.flatnonzero(np.isfinite(dist))
        if len(on) > k:
            drop = on[np.argsort(dist[on])[k:]]
            seg[:, drop] = np.nan
    return xy


def metrica_dead_model(counts, seed=0, step=5, windows_s=WINDOWS_S):
    """exp-021 model: HistGradientBoosting P(dead) trained on the 3 Metrica matches, each used twice
    (clean + camera-truncated with our visible-player `counts`). Deterministic for a given input."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from metrica_lib import load_game, PITCH
    X, y = [], []
    for gid in (1, 2, 3):
        g = load_game(gid)
        play = g.period > 0
        for xy in (g.xy, truncate_like(g.xy, counts, rng=np.random.default_rng(gid))):
            X.append(dead_features(xy, g.team, PITCH, windows_s=windows_s).values[play][::step].astype(np.float32))
            y.append(g.dead[play][::step])
    return HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(np.concatenate(X), np.concatenate(y))


def extend_only(force, dead, reach):
    """Keep the `force` frames (e.g. ball visibly out) that touch, within `reach` frames, a stoppage
    already decoded in `dead`: hard evidence may move a stoppage boundary but not create one."""
    near = pd.Series(dead.astype(np.int8)).rolling(2 * reach + 1, center=True, min_periods=1).max().values > 0
    return force & near
