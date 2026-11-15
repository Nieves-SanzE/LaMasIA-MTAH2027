"""Detect the playing periods (kick-off and end of each half) from the tracking alone.

The competition rules forbid constants tied to frames of this match, so the pipeline finds the
halves itself. Everything here is football knowledge or a physical threshold, not a frame number:

- kick-off candidate: each team stands in its own half (>= 70% of its visible players on one side,
  the two teams on opposite sides, >= 16 players visible) nearly still (1 s mean speed < 1.2 m/s)
  for >= 5 s; the kick-off is the first frame within 20 s after that where the 1 s mean speed
  reaches 1.8 m/s (play has started). The ball is not used: it is often lost around kick-offs;
- a half starts with the first kick-off candidate that is followed by >= 10 min of sustained play
  (>= 16 players on the pitch, 60 s rolling median), and the second half starts with the first such
  candidate >= 45 min after the first kick-off (kick-offs after goals happen inside a half). When
  referee whistles are given, candidates preceded by a strong whistle within 5 s are preferred
  (the kick-off whistle; teams also line up and walk off again without kicking off);
- a half ends at the first frame >= 45 min after its kick-off from which the PAST 120 s rolling
  median of players on the pitch stays < 15 for >= 5 min (players leave the pitch). Looking only
  backwards makes it err late, which is harmless: frames after the final whistle are not graded.
"""
import numpy as np
import pandas as pd

FPS = 25
OWN_HALF = 0.7        # share of a team's visible players in its own half (tracking is noisy)
MIN_PLAYERS = 16
STILL, GO = 1.2, 1.8  # m/s, 1 s mean player speed: standing for the kick-off / play has started
LINE_UP = 5 * FPS     # teams lined up in their halves and standing for >= 5 s
SUSTAINED = 10 * 60 * FPS
HALF_MIN = 45 * 60 * FPS
LEAVE_MED, LEAVE_LEN = 15, 5 * 60 * FPS


WHISTLE_BEFORE = 5 * FPS


def _rolling_median(x, seconds, center=True):
    return pd.Series(x).rolling(seconds * FPS, center=center, min_periods=1).median().values


def kickoff_candidates(xy_m, speed, team, field):
    """Frames where a kick-off is taken (see module docstring)."""
    L, W = field
    vis = np.isfinite(xy_m[..., 0])
    left = (xy_m[..., 0] < L / 2) & vis
    fr = []
    for k in (0, 1):
        n = vis[:, team == k].sum(1)
        fr.append(np.where(n > 0, left[:, team == k].sum(1) / np.maximum(n, 1), np.nan))
    split = (((fr[0] >= OWN_HALF) & (fr[1] <= 1 - OWN_HALF)) |
             ((fr[1] >= OWN_HALF) & (fr[0] <= 1 - OWN_HALF)))
    split &= vis.sum(1) >= MIN_PLAYERS
    with np.errstate(all="ignore"):
        v = pd.Series(np.nanmean(speed, 1)).rolling(FPS, min_periods=1).mean().values
    lined_up = split & (v < STILL)
    run = pd.Series(lined_up).groupby((~pd.Series(lined_up)).cumsum()).cumsum().values
    out, i, n = [], 0, len(v)
    while i < n:
        if run[i] >= LINE_UP and (i + 1 == n or not lined_up[i + 1]):
            go = np.flatnonzero(v[i + 1:i + 1 + 20 * FPS] >= GO)
            if len(go):
                out.append(i + 1 + int(go[0]))
                i += 1 + int(go[0])
        i += 1
    return np.array(out, int)


def detect_periods(xy_m, speed, team, field, whistles=()):
    """[(kickoff, end), (kickoff, end)] half-open video-frame ranges of the two halves."""
    L, W = field
    vis = np.isfinite(xy_m[..., 0])
    on = (vis & (xy_m[..., 0] > 0) & (xy_m[..., 0] < L) & (xy_m[..., 1] > 0) & (xy_m[..., 1] < W)).sum(1)
    play_on = _rolling_median(on, 60) >= 16
    leave = _rolling_median(on, 120, center=False) < LEAVE_MED
    n = len(on)
    cands = kickoff_candidates(xy_m, speed, team, field)

    def sustained(k):
        return play_on[k:k + SUSTAINED].mean() > 0.9

    def end_after(k):
        lv = leave.copy()
        lv[:k + HALF_MIN] = False
        run = pd.Series(lv).groupby((~pd.Series(lv)).cumsum()).cumsum().values
        hit = np.flatnonzero(run >= LEAVE_LEN)
        return int(hit[0] - LEAVE_LEN + 1) if len(hit) else n

    wh = np.sort(np.asarray(whistles, int))
    if len(wh):
        i = np.searchsorted(wh, cands)
        prev = np.where(i > 0, wh[np.maximum(i - 1, 0)], -10 ** 9)
        whistled = cands - prev <= WHISTLE_BEFORE
        if whistled.any():
            cands = np.concatenate([cands[whistled], cands[~whistled]])  # whistled first, in order

    def first(cond):
        return next(int(k) for k in cands if cond(k))

    ko1 = first(sustained)
    end1 = end_after(ko1)
    ko2 = first(lambda k: k >= ko1 + HALF_MIN and k >= end1 and sustained(k))
    return [(ko1, min(end1, ko2)), (ko2, end_after(ko2))]
