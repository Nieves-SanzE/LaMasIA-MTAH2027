"""RV3 (exp-042): Metrica with an official-style GT, so Metrica can judge the timing of team changes.

The event GT of exp-026 (metrica_lib.possession_codes) switches team at the new team's first owning
event (reception / recovery). The official BAR-ATH GT is generated mechanically from corrected tracking
and switches roughly at the MIDDLE of the ball's flight (exp-010: on loose balls between two
different-team controllers the official GT keeps the old team ~50% of the time). Here the D layer is
kept exactly as possession_codes builds it and only the H<->A changes are moved into the flight:

  old team's last touch  t_old = start frame of the old team's last PASS / SHOT / BALL LOST event before
                         the change (the kick of the pass that was lost);
  new team's first owning event f (where the event GT switches).
  A change is moved only if the flight is clean: t_old < f, f - t_old <= MAX_FLIGHT (4 s, longer =
  missing events, e.g. a dribble), same period, no dead frame in [t_old, f). Otherwise it stays at f.
  (i)  "mid":   switch at (t_old + f) // 2;
  (ii) "cross": switch at the start of the last run (ending at f) of frames where the ball is closer to
                the new team's nearest player than to the old team's nearest player; fallback to (i)
                when undefined (ball missing or not closer to the new team at f).

Selection rule (fixed BEFORE any model score is computed): the official-style GT is the variant whose
old-team share on loose-ball frames between two different-team clear controllers is closest to 50%
(exp-010 definition: ball visible, nobody < 1.5 m, alive, last causal clear control (>= 12 frames) and
next anti-causal clear control within 3 s with the ball visible in between, of different teams).

Re-scoring (cached exp-026/027 probabilities, no retraining): v4 system `ball+free` (logit P_ball +
logit P_free - logit prior on ball-visible frames, P_free in gaps), exp026_v5.decode (= make_submission
joint decoder, temperature 0.85, D_PARAMS), with labels base / ctrl / release / mid (exp-027). Masks:
native ball, the 4 BAR-ATH donor masks (mode a, exp-026) and 4 "aligned" masks (mode b, see
aligned_masks). Scored against the old event GT, the chosen GT and the other variant (robustness).
Independent units: match (3) and 5-min block (paired); masks are averaged inside a match / block.
Camera-like player truncation is NOT done (it needs recomputing the features and retraining).
Metrica is public; BAR-ATH enters only as a visibility pattern (aggregates).
    OMP_NUM_THREADS=4 uv run python scripts/explore/rv03_metrica_official_gt.py
"""
import os
import time

import numpy as np
import pandas as pd

import featlib
from common import pm
from exp026_v5 import FPS, GIDS, OFFSETS, barath_inputs, dead_probs, decode, logit, sig, team_lags
from metrica_lib import OWN, PITCH, load_game
from rvlib import BLOCK, block_starts, spell_f1
import make_submission as ms

XOUT = featlib.OUT
OUT = os.path.join(XOUT, "rv03")
os.makedirs(OUT, exist_ok=True)
MAX_FLIGHT = 4 * FPS
HORIZON = 3 * FPS
TOUCH = ("PASS", "SHOT", "BALL LOST")
MODES = ("base", "ctrl", "release", "mid")
GT_VARIANTS = ("event", "mid", "cross")
ALIGNED_SEEDS = (0, 1, 2, 3)
LONG_BALL_M = 30.0
LONG_GAP = 10 * FPS
P_RESTART_LONG = 0.83   # 79-87% of long BAR-ATH gaps end at a restart (eda/ball.md)
P_RESTART_SHORT = 0.5   # shorter gaps: no evidence, half restarts / half long balls
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- GT construction
def owner_array(g):
    """Team of the last owning event (possession_codes before the D layer)."""
    n = len(g.period)
    owner = np.full(n, -1, np.int8)
    ev = g.events[g.events.type.isin(OWN) & g.events.team.notna()].sort_values("start")
    for s, t in zip(ev.start.values.astype(int), ev.team.values.astype(int)):
        owner[s] = t
    return pd.Series(np.where(owner >= 0, owner, np.nan)).ffill().bfill().values.astype(np.int8)


def team_dist(g, ball=None):
    """(N, 2) distance from the ball to each team's nearest player (inf if none / ball missing)."""
    ball = g.ball if ball is None else ball
    d = np.linalg.norm(g.xy - ball[:, None, :], axis=-1)
    out = np.full((len(d), 2), np.inf, np.float32)
    for k in (0, 1):
        dk = np.where(np.isfinite(d[:, g.team == k]), d[:, g.team == k], np.inf)
        out[:, k] = dk.min(1)
    return out


def official_style_gts(g):
    """Codes 0/1/2/3 for the event GT and the two flight variants, plus per-change diagnostics."""
    owner = owner_array(g)
    ev = g.events[g.events.type.isin(TOUCH) & g.events.team.notna()]
    ts, tt = ev.start.values.astype(int), ev.team.values.astype(int)
    dist = team_dist(g)
    chg = np.flatnonzero(owner[1:] != owner[:-1]) + 1
    own_mid, own_cross = owner.copy(), owner.copy()
    rows = []
    prev_change = 0
    for f in chg:
        old, new = int(owner[f - 1]), int(owner[f])
        m = (tt == old) & (ts < f) & (ts >= prev_change)
        t_old = int(ts[m].max()) if m.any() else -1
        prev_change = f
        why = "moved"
        if g.period[f] == 0:
            why = "outside play"
        elif t_old < 0:
            why = "no old touch"
        elif f - t_old > MAX_FLIGHT:
            why = "flight > 4 s"
        elif g.period[t_old] != g.period[f]:
            why = "other period"
        elif g.dead[t_old:f].any():
            why = "dead in flight"
        sw_mid = sw_cross = f
        fallback = False
        if why == "moved":
            sw_mid = (t_old + f) // 2
            closer = dist[t_old:f + 1, new] < dist[t_old:f + 1, old]
            ok = np.isfinite(dist[t_old:f + 1]).all()
            if ok and closer[-1]:
                notc = np.flatnonzero(~closer)
                sw_cross = t_old + (int(notc[-1]) + 1 if len(notc) else 0)
            else:
                sw_cross, fallback = sw_mid, True
            own_mid[sw_mid:f] = new
            own_cross[sw_cross:f] = new
        rows.append(dict(f=int(f), t_old=t_old, why=why, sw_mid=int(sw_mid), sw_cross=int(sw_cross),
                         fallback=fallback))
    out = {}
    for name, o in (("event", owner), ("mid", own_mid), ("cross", own_cross)):
        c = np.where(g.dead, 2, o).astype(np.int8)
        c[g.period == 0] = 3
        out[name] = c
    return out, pd.DataFrame(rows)


def loose_change_frames(dmin, near_team, vis, periods, n):
    """exp-010 loose-ball frames between two different-team clear controllers: (mask, last team)."""
    seg = np.full(n, -1)
    for k, (a, b) in enumerate(periods):
        seg[a:b] = k
    ctl = np.where(vis & (dmin < featlib.CTRL_DIST) & (near_team >= 0), near_team, -1)

    def run_count(x, forward):
        s = pd.Series(x if forward else x[::-1])
        c = s.groupby(s.ne(s.shift()).cumsum()).cumcount().values + 1
        return c if forward else c[::-1]

    causal = np.where((ctl >= 0) & (run_count(ctl, True) >= featlib.CTRL_FRAMES), ctl, -1)
    anti = np.where((ctl >= 0) & (run_count(ctl, False) >= featlib.CTRL_FRAMES), ctl, -1)
    idx = np.arange(n)
    nxt = np.minimum.accumulate(np.where(anti >= 0, idx, n)[::-1])[::-1]
    lst = np.maximum.accumulate(np.where(causal >= 0, idx, -1))
    miss = np.r_[0, np.cumsum(~vis)]
    jj = np.minimum(nxt, n - 1)
    loose = (seg >= 0) & vis & (dmin > featlib.CTRL_DIST)
    ok = loose & (nxt < n) & (jj - idx <= HORIZON) & (seg[jj] == seg) & (miss[jj + 1] - miss[idx] == 0)
    target = np.where(ok, anti[jj], -1)
    last = np.where(lst >= 0, causal[np.maximum(lst, 0)], -1)
    last[(lst < 0) | (seg[np.maximum(lst, 0)] != seg)] = -1
    return (target >= 0) & (last >= 0) & (target != last), last


# ----------------------------------------------------------------------------- masks
def donor_masks(native, play, bar_bv):
    pidx = np.flatnonzero(play)
    out = {}
    for o in OFFSETS:
        m = native.copy()
        m[pidx] &= bar_bv[(o + np.arange(len(pidx))) % len(bar_bv)]
        out[f"donor {o // 1000}k"] = m
    return out


def long_ball_starts(g):
    ev = g.events[g.events.type.isin(("PASS", "BALL LOST"))]
    s, e = ev.start.values.astype(int), ev.end.values.astype(int)
    ok = (e > s) & (e < len(g.ball))
    s, e = s[ok], e[ok]
    d = np.linalg.norm(g.ball[e] - g.ball[s], axis=1)
    return np.unique(s[np.isfinite(d) & (d > LONG_BALL_M)])


def aligned_masks(g, native, periods, bar_bv):
    """Mode (b): the BAR-ATH gap lengths (shuffled, hidden share of the graded frames as target) placed
    preferentially where real camera gaps happen. Each gap: with prob P_RESTART (0.83 for gaps >= 10 s,
    0.5 otherwise) it ENDS 0-2 s after a restart (event dead-ball episode end); else it STARTS 0-0.5 s
    after the kick of a long ball (PASS / BALL LOST with ball displacement > 30 m). Anchors are tried in
    random order; a gap must lie inside one period with >= 1 s of visible ball to the next gap; if no
    anchor fits, a uniform random position. Masks = native AND not hidden."""
    s, e = featlib.run_bounds(~bar_bv)
    lens = e - s + 1
    target_share = (~bar_bv).mean()
    restarts = np.array(sorted({int(r) for _, r, *_ in g.dead_eps}))
    longb = long_ball_starts(g)
    n = len(native)
    play_len = sum(b - a for a, b in periods)
    out = {}
    for seed in ALIGNED_SEEDS:
        rng = np.random.default_rng(100 + seed)
        hidden = np.zeros(n, bool)
        used = {"restart": set(), "long": set()}
        order = rng.permutation(len(lens))
        k = 0
        while hidden.sum() < target_share * play_len and k < 3 * len(lens):
            L = int(lens[order[k % len(lens)]])
            k += 1
            p_r = P_RESTART_LONG if L >= LONG_GAP else P_RESTART_SHORT
            kind = "restart" if rng.random() < p_r else "long"
            anchors = restarts if kind == "restart" else longb
            placed = False
            for ai in rng.permutation(len(anchors)):
                if ai in used[kind]:
                    continue
                if kind == "restart":
                    end = int(anchors[ai]) + int(rng.integers(0, 2 * FPS))
                    start = end - L
                else:
                    start = int(anchors[ai]) + int(rng.integers(0, FPS // 2))
                    end = start + L
                if _fits(start, end, periods, hidden):
                    hidden[start:end] = True
                    used[kind].add(ai)
                    placed = True
                    break
            if not placed:
                for _ in range(200):
                    a, b = periods[int(rng.integers(len(periods)))]
                    if b - a <= L:
                        break
                    start = int(rng.integers(a, b - L))
                    if _fits(start, start + L, periods, hidden):
                        hidden[start:start + L] = True
                        break
        out[f"aligned s{seed}"] = native & ~hidden
    return out


def _fits(start, end, periods, hidden):
    if not any(a <= start and end <= b for a, b in periods):
        return False
    return not hidden[max(start - FPS, 0):end + FPS].any()


def mask_stats(name, vis, native, play, gt, restarts):
    """Gap anatomy of a mask over ALL invisible play frames (the native Metrica ball is itself missing on
    most dead-ball frames), next to the newly hidden share (native present, mask hides it)."""
    inv = play & ~vis
    s, e = featlib.run_bounds(inv)
    lg = (e - s + 1) >= LONG_GAP
    ends = e[lg] + 1
    near = np.array([np.any((restarts >= x - FPS) & (restarts <= x + 3 * FPS)) for x in ends]) if len(ends) else np.array([])
    return dict(mask=name, invisible_share=inv.sum() / play.sum(), new_hidden_share=(inv & native).sum() / play.sum(),
                live_visible_share=np.mean(vis[play & (gt < 2)]), d_share_in_gaps=np.mean(gt[inv] == 2),
                n_gaps=len(s), long_gaps=int(lg.sum()), long_end_at_restart=near.mean() if len(near) else np.nan)


def all_mask_stats():
    """Recompute the mask anatomy table (deterministic masks; no model involved)."""
    bar_bv, _ = barath_inputs()
    rows = []
    for gid in GIDS:
        g = load_game(gid)
        periods = [tuple(map(int, p)) for p in np.load(os.path.join(XOUT, f"exp026_game{gid}.npz"))["periods"]]
        play = np.zeros(len(g.period), bool)
        for a, b in periods:
            play[a:b] = True
        gt = official_style_gts(g)[0]["event"]
        native = np.isfinite(g.ball[:, 0])
        restarts = np.array(sorted({int(r) for _, r, *_ in g.dead_eps}))
        masks = {"native": native, **donor_masks(native, play, bar_bv), **aligned_masks(g, native, periods, bar_bv)}
        rows += [dict(game=gid, **mask_stats(k, v, native, play, gt, restarts)) for k, v in masks.items()]
    # BAR-ATH reference (visibility pattern of the graded frames only; no GT)
    s, e = featlib.run_bounds(~bar_bv)
    rows.append(dict(game=0, mask="BAR-ATH real", invisible_share=(~bar_bv).mean(), n_gaps=len(s),
                     long_gaps=int(((e - s + 1) >= LONG_GAP).sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def block_scores(gt, pred, periods):
    rows = []
    for b0 in block_starts(periods):
        gb = np.full(len(gt), 3, np.int8)
        gb[b0:b0 + BLOCK] = gt[b0:b0 + BLOCK]
        rows.append((b0 // BLOCK, spell_f1(gb, pred)))
    return rows


def main():
    bar_bv, _ = barath_inputs()
    p_dead = dead_probs(None)  # cached by exp-026
    log("inputs ready")
    res, blocks, lags, chg_rows, prop_rows, mstats = [], [], [], [], [], []
    games = {}
    # ---- stage 1: GTs and the loose-ball property (no model involved)
    for gid in GIDS:
        g = load_game(gid)
        z = np.load(os.path.join(XOUT, f"exp026_game{gid}.npz"))
        periods = [tuple(map(int, p)) for p in z["periods"]]
        n = len(g.period)
        play = np.zeros(n, bool)
        for a, b in periods:
            play[a:b] = True
        gts, ch = official_style_gts(g)
        for k in gts:
            gts[k][~play] = 3
        ch["game"] = gid
        chg_rows.append(ch)
        native = np.isfinite(g.ball[:, 0])
        lx, last = loose_change_frames(z["dmin"], z["near_team"], native, periods, n)
        lx &= gts["event"] < 2
        for k, c in gts.items():
            prop_rows.append(dict(game=gid, gt=k, n=int(lx.sum()), old_share=np.mean(c[lx] == last[lx]),
                                  new_share=np.mean(c[lx] == 1 - last[lx])))
        games[gid] = (g, z["dmin"], z["near_team"], periods, play, gts, native)
        log(f"game {gid}: GTs built")
    ch = pd.concat(chg_rows)
    prop = pd.DataFrame(prop_rows)
    ch.to_csv(os.path.join(OUT, "gt_changes.csv"), index=False)
    prop.to_csv(os.path.join(OUT, "gt_loose_property.csv"), index=False)
    print("\n=== event GT team changes: which ones move into the flight ===")
    print(ch.groupby("game").why.value_counts().unstack(fill_value=0).to_string())
    mv = ch[ch.why == "moved"]
    print(f"moved {len(mv)}/{len(ch)}; flight f - t_old (s): median {np.median(mv.f - mv.t_old) / FPS:.2f} "
          f"p90 {np.percentile(mv.f - mv.t_old, 90) / FPS:.2f}; shift f - switch (s): mid median "
          f"{np.median(mv.f - mv.sw_mid) / FPS:.2f}, cross median {np.median(mv.f - mv.sw_cross) / FPS:.2f}; "
          f"cross fallback to mid {mv.fallback.mean():.1%}")
    print("\n=== loose-ball property (exp-010 definition): share of frames labelled with the OLD team ===")
    pooled = prop.assign(k=prop.n * prop.old_share).groupby("gt")[["k", "n"]].sum()
    pooled["old_share"] = pooled.k / pooled.n
    print(prop.pivot_table(index="game", columns="gt", values="old_share").round(3).to_string())
    print("pooled: " + "  ".join(f"{k} {v:.3f}" for k, v in pooled.old_share.items()) + f"  (n={int(pooled.n.iloc[0])})")
    chosen = min(("mid", "cross"), key=lambda k: abs(pooled.old_share[k] - 0.5))
    other = "cross" if chosen == "mid" else "mid"
    print(f"CHOSEN official-style GT (closest to 50%, rule fixed beforehand): {chosen}")

    # ---- stage 2: re-scoring with the cached probabilities
    for gid in GIDS:
        g, dmin, near_team, periods, play, gts, native = games[gid]
        restarts = np.array(sorted({int(r) for _, r, *_ in g.dead_eps}))
        L, W = PITCH
        bx, by = g.ball[:, 0], g.ball[:, 1]
        with np.errstate(invalid="ignore"):
            ball_out = (bx < -ms.OUT_MARGIN) | (bx > L + ms.OUT_MARGIN) | (by < -ms.OUT_MARGIN) | (by > W + ms.OUT_MARGIN)
        masks = {"native": native, **donor_masks(native, play, bar_bv), **aligned_masks(g, native, periods, bar_bv)}
        for mk, vis in masks.items():
            mstats.append(dict(game=gid, **mask_stats(mk, vis, native, play, gts["event"], restarts)))
        runs = pm.valid_runs(gts["event"])
        minutes = play.sum() / FPS / 60
        for mode in MODES:
            team = featlib.timed_team_labels(dmin, near_team, native, periods, mode)
            train = play & native & (team >= 0)
            prior = float(np.mean(team[train] == 1))
            c = np.load(os.path.join(XOUT, f"exp026_probs_game{gid}.npz" if mode == "base"
                                     else f"exp027_probs_game{gid}_{mode}.npz"))
            p_free, p_ball = c["p_free"], c["p_ball"]
            pv = sig(logit(p_ball) + logit(p_free) - logit(prior))
            for mk, vis in masks.items():
                cond = mk.split()[0]
                pred = decode(np.where(vis, pv, p_free), p_dead[gid], prior, periods, vis, ball_out)
                rate = len(pm.possession_spells(pm.fill_abstentions(pred.copy(), runs), runs)) / minutes
                for gk, gt in gts.items():
                    r = pm.evaluate(gt, pred)
                    res.append(dict(game=gid, mask=mk, cond=cond, mode=mode, gt=gk, spellF1=r["spell_f1_mean"],
                                    macroF1=r["macro_f1"], poss_per_min=rate, d_share=np.mean(pred[play] == 2)))
                    blocks += [dict(game=gid, block=b, mask=mk, cond=cond, mode=mode, gt=gk, spellF1=s)
                               for b, s in block_scores(gt, pred, periods)]
                    lags += [dict(game=gid, mask=mk, cond=cond, mode=mode, gt=gk, lag=l, vis=v)
                             for l, v in team_lags(gt, pred, vis, runs)]
            log(f"game {gid} {mode}: " + "  ".join(
                f"{r['mask']} {r['spellF1']:.3f}" for r in res[-len(masks) * len(gts):] if r["gt"] == chosen))
    df, bk, lg, ms_ = pd.DataFrame(res), pd.DataFrame(blocks), pd.DataFrame(lags), pd.DataFrame(mstats)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    bk.to_csv(os.path.join(OUT, "blocks.csv"), index=False)
    lg.to_csv(os.path.join(OUT, "lags.csv"), index=False)
    ms_.to_csv(os.path.join(OUT, "mask_stats.csv"), index=False)
    report(df, bk, lg, ms_, chosen, other)


def report(df, bk, lg, mstats, chosen, other):
    pd.set_option("display.width", 200)
    print("\n=== masks over all invisible play frames (BAR-ATH real: 41% invisible, 56% of gap frames D in the window, 79-87% of long gaps end at a restart) ===")
    mstats = mstats.assign(cond=mstats["mask"].str.split().str[0])
    print(mstats.groupby("cond")[["invisible_share", "new_hidden_share", "live_visible_share", "d_share_in_gaps",
                                  "n_gaps", "long_gaps", "long_end_at_restart"]].mean().round(3).to_string())
    summary = []
    for gk in ("event", chosen, other):
        print(f"\n######## GT = {gk}{' (CHOSEN official-style)' if gk == chosen else ''} ########")
        print(df[(df["gt"] == gk) & (df.cond != "native")].pivot_table(index=["game", "cond"], columns="mode",
              values="spellF1").round(3).to_string())
        for cond in ("native", "donor", "aligned"):
            sub = df[(df["gt"] == gk) & (df.cond == cond)]
            per_game = sub.groupby(["mode", "game"]).spellF1.mean().unstack()
            sb = bk[(bk["gt"] == gk) & (bk.cond == cond)].groupby(["mode", "game", "block"]).spellF1.mean().unstack(0)
            agg = sub.groupby("mode")[["spellF1", "macroF1", "poss_per_min"]].mean()
            for mode in MODES:
                dg = per_game.loc[mode] - per_game.loc["base"]
                db = sb[mode] - sb["base"]
                x = lg[(lg["gt"] == gk) & (lg.cond == cond) & (lg["mode"] == mode)]
                fv, fh = x[x.vis].lag.dropna(), x[~x.vis].lag.dropna()
                row = dict(gt=gk, cond=cond, mode=mode, spellF1=agg.loc[mode, "spellF1"],
                           d_base=dg.mean(), game_WL=f"{int((dg > 1e-9).sum())}/{int((dg < -1e-9).sum())}",
                           block_median_d=db.median(),
                           block_WL=f"{int((db > 1e-9).sum())}/{int((db < -1e-9).sum())}",
                           block_win_share=(db > 1e-9).mean(), macroF1=agg.loc[mode, "macroF1"],
                           poss_per_min=agg.loc[mode, "poss_per_min"],
                           lag_vis=fv.median() if len(fv) else np.nan, lag_hid=fh.median() if len(fh) else np.nan,
                           found_vis=len(fv) / max(x.vis.sum(), 1))
                row.update({f"g{g}": per_game.loc[mode, g] for g in per_game.columns})
                summary.append(row)
            print(f"\n--- {cond} masks: mean over cells; wins vs base per match (3) and per 5-min block (n={len(sb)}) ---")
            print(pd.DataFrame([r for r in summary if r["gt"] == gk and r["cond"] == cond]).drop(columns=["gt", "cond"])
                  .round(3).to_string(index=False))
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "summary.csv"), index=False)


def report_from_csv():
    """Re-print the report from the saved tables (`--report`)."""
    rd = lambda k: pd.read_csv(os.path.join(OUT, f"{k}.csv"))  # noqa: E731
    prop = rd("gt_loose_property")
    pooled = prop.assign(k=prop.n * prop.old_share).groupby("gt")[["k", "n"]].sum()
    share = pooled.k / pooled.n
    chosen = min(("mid", "cross"), key=lambda k: abs(share[k] - 0.5))
    mst = all_mask_stats()
    mst.to_csv(os.path.join(OUT, "mask_stats.csv"), index=False)
    report(rd("results"), rd("blocks"), rd("lags"), mst, chosen, "cross" if chosen == "mid" else "mid")


if __name__ == "__main__":
    import sys
    report_from_csv() if "--report" in sys.argv else main()
