"""Shared building blocks for the ball-gap experiments (Q8/Q9/Q10).

Everything takes a ball-visibility mask `vis` (full-length bool) and a list of half-open segments
`segs` [(lo, hi)] so the same code runs on the public window with a transplanted mask (harness of
exp-007) and on the full match with real or artificially hidden ball.

All parameters are a-priori constants (exp-003 / exp-008 / physics), nothing is tuned here.

    from gaplib import Features, predict, METHODS
"""
import warnings

import numpy as np
import pandas as pd

H, A, D, X = 0, 1, 2, 3
UNK = -1
FPS = 25

# exp-003 heuristic (chosen in exp-003, not re-tuned)
CTRL_DIST = 1.5        # m, nearest player to the ball
CTRL_FRAMES = 12       # consecutive frames of clear control
D_SPEED = 1.0          # m/s, 3 s centred rolling mean of the mean player speed -> global D
D_WINDOW = 3 * FPS

# Q9 / Q10 (a priori)
LOOK = 10 * FPS        # before/after team: clear control within 10 s of the gap (as exp-008)
MERGE = FPS            # gaps separated by < 1 s of visible ball are one gap (no 12-frame control fits)
Q10_SPEED = 1.5        # m/s: "walking" once restart evidence exists (looser than the global 1.0)
Q10_TAIL = 3 * FPS     # the slow run must reach the last 3 s before reappearance
Q10_MIN = 2 * FPS      # inferred D shorter than 2 s is dropped (dead balls last longer; avoids splits)
WHISTLE_DELAY = FPS // 2   # D starts 0.5 s after a strong whistle
WHISTLE_BEFORE = 2 * FPS   # a whistle up to 2 s before the gap starts counts (as exp-008)
FIELD = (105.3, 68.0)


def _runs_true(m):
    """[(start, stop)] runs of True in a boolean array (local indices)."""
    e = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def _run_len(x, forward=True):
    """Length of the constant run of x ending (forward) / starting (backward) at each frame."""
    s = pd.Series(x if forward else x[::-1])
    c = s.groupby(s.ne(s.shift()).cumsum()).cumcount().values + 1
    return c if forward else c[::-1]


def hold(conf):
    """Forward-hold the last confident team; X before the first one."""
    idx = np.where(conf >= 0, np.arange(len(conf)), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last >= 0, conf[np.clip(last, 0, None)], X).astype(np.int8)
    return out


class Features:
    """Per-frame quantities that do not depend on the visibility mask (plus raw ball arrays)."""

    def __init__(self, tr, strong_whistle_frames=()):
        warnings.simplefilter("ignore", RuntimeWarning)
        self.N = tr.n_frames
        near = tr.mas_cercano_al_balon()
        self.near_team = np.where(near >= 0, tr.equipo[np.clip(near, 0, None)], -1)
        dist = tr.distancia_al_balon()
        self.dmin = np.nanmin(np.where(np.isfinite(dist), dist, np.inf), 1)
        self.ball_vis = tr.balon_visible.copy()
        speed = np.nanmean(tr.velocidad, 1)
        roll = pd.Series(speed).rolling(D_WINDOW, center=True, min_periods=1).mean().values
        self.slow = roll < D_SPEED
        self.slow_q10 = roll < Q10_SPEED
        ok = np.isfinite(tr.velocidad)
        n_ok = ok.sum(1)
        self.stopped = np.where(n_ok > 0, ((tr.velocidad < 1.0) & ok).sum(1) / np.maximum(n_ok, 1), np.nan)
        self.ball = tr.balon.copy()           # normalised image-of-pitch coords
        self.ball_m = tr.metros(tr.balon)     # metres
        self.whistles = np.sort(np.asarray(strong_whistle_frames, int))


# ----------------------------------------------------------------------------------- gaps

def find_gaps(miss, merge=MERGE):
    """Missing-ball runs (local indices), merging those separated by < merge visible frames."""
    out = []
    for s, e in _runs_true(miss):
        if out and s - out[-1][1] < merge:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def restart_signs(F, vis, s, e, lo, hi):
    """Restart evidence at the reappearance frame e (global index); only uses visible ball.

    Signs (exp-008): ball out / on a line, corner or goal area, >= 50% players stopped in the last
    3 s, ball static for 1 s after reappearance, strong whistle in [s - 2 s, e).
    Returns (any_sign, whistle_frame or None).
    """
    k = np.searchsorted(F.whistles, s - WHISTLE_BEFORE)
    wf = int(F.whistles[k]) if k < len(F.whistles) and F.whistles[k] < e else None
    if e >= hi or not vis[e]:
        return False, wf
    bx, by = F.ball[e]
    xm, ym = F.ball_m[e]
    line = (bx < 0) | (bx > 1) | (by < 0) | (by > 1) | (xm < 2) | (xm > FIELD[0] - 2) | (ym < 2) | (ym > FIELD[1] - 2)
    corner = min(np.hypot(xm - cx, ym - cy) for cx in (0, FIELD[0]) for cy in (0, FIELD[1])) < 3
    box = ((xm < 5.5) | (xm > FIELD[0] - 5.5)) & (abs(ym - FIELD[1] / 2) < 9.16)
    stopped = np.nanmean(F.stopped[max(e - 3 * FPS, lo):e]) >= 0.5
    post = slice(e, min(e + FPS, hi))
    pm_ = F.ball_m[post][vis[post]]
    pm_ = pm_[np.isfinite(pm_[:, 0])]
    static = len(pm_) >= 10 and np.hypot(*(pm_ - pm_[0]).T).max() < 1.5
    return bool(line or corner or box or stopped or static or wf is not None), wf


def q10_dead(F, vis, s, e, lo, hi):
    """Q10: inferred dead-ball stretch [d0, d1) inside gap [s, e) (global indices) or None."""
    sign, wf = restart_signs(F, vis, s, e, lo, hi)
    if not sign:
        return None
    if wf is not None:
        d0, d1 = max(s, wf + WHISTLE_DELAY), e
    else:
        tail = np.flatnonzero(F.slow_q10[max(s, e - Q10_TAIL):e])
        if not len(tail):
            return None
        j = max(s, e - Q10_TAIL) + tail[-1]  # last slow frame before reappearance
        r = j
        while r > s and F.slow_q10[r - 1]:
            r -= 1
        d0, d1 = r, j + 1
    return (d0, d1) if d1 - d0 >= Q10_MIN else None


# ----------------------------------------------------------------------------------- methods

# name -> (q10 D, cut rule when before != after, use one known side)
#   cut rules: None = hold (exp-003), "mid" = gap midpoint, "speed" = first slow frame (3 s mean
#   < 1.0 m/s) inside the gap else midpoint, "dstart" = start of the Q10 D else midpoint,
#   "dstart_speed" = start of the Q10 D else the speed rule.
METHODS = {
    "(a) exp-003 baseline": (False, None, False),
    "Q9(i) midpoint": (False, "mid", False),
    "Q9(iii) speed drop": (False, "speed", False),
    "Q9(i)+(iv) one-sided": (False, "mid", True),
    "Q10 D only": (True, None, False),
    "Q9(ii)+Q10": (True, "dstart", False),
    "combo (ii|iii)+Q10+(iv)": (True, "dstart_speed", True),
}


def predict(F, vis, segs, method, stats=None):
    """Labels (full length, X outside segs) for one method given the ball-visibility mask."""
    use_q10, cut_rule, one_sided = METHODS[method]
    out = np.full(F.N, X, np.int8)
    for lo, hi in segs:
        v = vis[lo:hi]
        ctl = np.where(v & (F.dmin[lo:hi] < CTRL_DIST) & (F.near_team[lo:hi] >= 0), F.near_team[lo:hi], -1)
        causal = np.where((ctl >= 0) & (_run_len(ctl, True) >= CTRL_FRAMES), ctl, -1)
        anti = np.where((ctl >= 0) & (_run_len(ctl, False) >= CTRL_FRAMES), ctl, -1)
        n = hi - lo
        idx = np.arange(n)
        last_idx = np.maximum.accumulate(np.where(causal >= 0, idx, -1))
        next_idx = np.minimum.accumulate(np.where(anti >= 0, idx, n)[::-1])[::-1]
        team = hold(causal)
        q10 = np.zeros(n, bool)
        for s, e in find_gaps(~v):
            j = last_idx[s - 1] if s > 0 else -1
            tb = causal[j] if j >= 0 and j >= s - LOOK else UNK
            fa = next_idx[e] if e < n else n
            ta = anti[fa] if fa < min(e + LOOK, n) else UNK
            dq = q10_dead(F, vis, s + lo, e + lo, lo, hi) if (use_q10 or cut_rule in ("dstart", "dstart_speed")) else None
            if dq is not None:
                dq = (dq[0] - lo, dq[1] - lo)
                if use_q10:
                    q10[dq[0]:dq[1]] = True
            if stats is not None:
                stats.append((e - s, tb, ta, dq is not None))
            if ta == UNK:
                continue
            cut = None
            if tb == UNK:
                cut = s if one_sided else None
            elif tb != ta and cut_rule is not None:
                mid = (s + e) // 2
                if cut_rule in ("dstart", "dstart_speed") and dq is not None:
                    cut = dq[0]
                elif cut_rule in ("speed", "dstart_speed"):
                    sl = np.flatnonzero(F.slow[lo + s:lo + e])
                    cut = s + sl[0] if len(sl) else mid
                else:
                    cut = mid
            if cut is not None:
                # after-team from the cut until the hold itself adopts it (12th frame of control)
                team[cut:min(fa + CTRL_FRAMES - 1, n)] = ta
        p = team.copy()
        p[(F.slow[lo:hi] & (team != X)) | q10] = D  # global speed D and Q10 D, union
        out[lo:hi] = p
    return out
