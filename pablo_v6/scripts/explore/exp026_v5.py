"""exp-026 (V5): the whole v3 system validated on the 3 public Metrica matches, v3 vs `ball+free`.

Per Metrica match (features from exp026_metrica_features.py):
  GT       event-derived H/A/D (metrica_lib.possession_codes: team of the last owning event, D from the
           event stoppages, X outside the periods). Changes of team happen at the new team's first
           owning event (reception / recovery), not mid-flight like the official GT.
  labels   exp-003 pseudo-labels (nearest player < 1.5 m for >= 12 frames, hold the last one) from the
           Metrica ball where it exists.
  P(A)     out-of-fold GBMs inside the match exactly as hmmlib.Ctx.oof (5-min blocks, 5 folds, every 5th
           frame, 200 iterations), averaged over SEEDS: P_free (ball-free families), P_both (free + ball),
           P_ball (ball families only).
  P(dead)  the exp-021 Metrica live/dead model retrained WITHOUT the evaluated match (leave-one-match-out;
           clean + camera-truncated with our visible-player counts).
  decoder  make_submission.decode: joint H/A/D Viterbi per half with the v3 parameters (team temperature
           0.85, D_PARAMS, PRIOR_D) and the ball-out extension.
  masks    native (the Metrica ball where it exists) and 4 donors: native AND the BAR-ATH ball-visibility
           pattern of the graded frames (cyclic offsets 0 / 40k / 80k / 120k frames), as the exp-007
           harness does on the public window.
Variants (ball-visible frames; P_free in the gaps for all): v3 = P_both; ball+free = logit P_ball +
logit P_free - logit prior (exp-023); free only = P_free everywhere (reference).
Metrics: official possession_metrics.evaluate, and the lag of each GT H<->A change to the nearest predicted
change to the same team (+-10 s). Aggregates only (Metrica is public, BAR-ATH only as a visibility mask).
    OMP_NUM_THREADS=8 uv run python scripts/explore/exp026_v5.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import featlib
from common import HALVES as OFFICIAL, ROOT, pm
from deadlib import dead_features, extend_only, truncate_like
from hmm3 import viterbi3
from metrica_lib import PITCH, load_game, possession_codes

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import make_submission as ms  # noqa: E402

FPS = 25
OUT = os.path.join(ROOT, "out", "explore")
SEEDS = (0, 1, 2)
STEP = 5
BLOCK = 5 * 60 * FPS
TEMP = 0.85  # v3 team temperature
OFFSETS = (0, 40000, 80000, 120000)
GIDS = (1, 2, 3)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------------- BAR-ATH aggregates
def barath_inputs():
    """Ball-visibility pattern over the graded frames and visible-player counts (cached, never printed)."""
    path = os.path.join(OUT, "exp026_barath_masks.npz")
    if not os.path.exists(path):
        tr = featlib.load_tracking()
        bv = np.concatenate([tr.balon_visible[a:b] for a, b in OFFICIAL])
        play = featlib.play_mask(tr.n_frames)
        counts = np.isfinite(tr.xy[..., 0]).sum(1)[play]
        np.savez_compressed(path, bv=bv, counts=counts)
    z = np.load(path)
    return z["bv"], z["counts"]


# ----------------------------------------------------------------------------- per-match pieces
def pseudo_labels(dmin, near_team, vis, periods):
    close = np.where((dmin < featlib.CTRL_DIST) & vis, near_team, -1)
    s = pd.Series(close)
    run = s.groupby(s.ne(s.shift()).cumsum()).cumcount().values + 1
    stable = np.where((close >= 0) & (run >= featlib.CTRL_FRAMES), close, -1)
    team = np.full(len(dmin), -1, np.int8)
    for a, b in periods:
        last = np.maximum.accumulate(np.where(stable[a:b] >= 0, np.arange(a, b), -1))
        team[a:b] = np.where(last >= 0, stable[np.clip(last, 0, None)], -1)
    return team


def oof(X, y, train, play, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    n = len(y)
    blocks = np.arange(n) // BLOCK
    fold = (np.random.default_rng(0).permutation(blocks.max() + 1) % 5)[blocks]
    sub = np.zeros(n, bool)
    sub[::STEP] = True
    p = np.full(n, np.nan, np.float32)
    for k in range(5):
        trn = np.flatnonzero(train & sub & (fold != k))
        te = np.flatnonzero(play & (fold == k))
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X[trn], y[trn])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def dead_probs(counts):
    """Leave-one-match-out P(dead) for the 3 matches (exp-021 recipe). Cached."""
    path = os.path.join(OUT, "exp026_pdead.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {g: z[f"g{g}"] for g in GIDS}
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr_X, tr_y, test = {}, {}, {}
    for gid in GIDS:
        g = load_game(gid)
        play = g.period > 0
        Xs, ys = [], []
        for k, xy in enumerate((g.xy, truncate_like(g.xy, counts, rng=np.random.default_rng(gid)))):
            F = dead_features(xy, g.team, PITCH).values.astype(np.float32)
            if k == 0:
                test[gid] = F
            Xs.append(F[play][::STEP])
            ys.append(g.dead[play][::STEP])
        tr_X[gid], tr_y[gid] = np.concatenate(Xs), np.concatenate(ys)
        log(f"dead features game {gid}")
    out = {}
    for gid in GIDS:
        others = [o for o in GIDS if o != gid]
        clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(
            np.concatenate([tr_X[o] for o in others]), np.concatenate([tr_y[o] for o in others]))
        out[gid] = clf.predict_proba(test[gid])[:, 1].astype(np.float32)
    np.savez_compressed(path, **{f"g{g}": out[g] for g in GIDS})
    return out


def decode(p_team, p_dead, prior, periods, vis, ball_out):
    """make_submission.decode with Metrica inputs (no half-time frames to fill)."""
    out = vis & ball_out
    s_ = pd.Series(out)
    out &= s_.groupby(s_.ne(s_.shift()).cumsum()).transform("size").values >= FPS // 2
    pred = np.full(len(vis), 3, np.int8)
    kw = dict(temp_t=TEMP, **ms.D_PARAMS)
    for a, b in periods:
        first = viterbi3(p_team[a:b], p_dead[a:b], prior, ms.PRIOR_D, **kw)
        force = extend_only(out[a:b], first == 2, 5 * FPS)
        pred[a:b] = viterbi3(p_team[a:b], p_dead[a:b], prior, ms.PRIOR_D, force_dead=force, **kw)
    return pred


def team_lags(gt, pred, vis, periods):
    """Signed lag (s) of each GT H<->A change to the nearest predicted change to the same team, and
    whether the ball was visible (>50% of +-1 s) at the GT change. NaN = none within 10 s."""
    gtt = [t for t in pm.transitions(gt, periods) if t[1] in (0, 1) and t[2] in (0, 1)]
    prt = pm.transitions(pm.fill_abstentions(pred.copy(), periods), periods)
    rows = []
    for f, _, b in gtt:
        cand = [p[0] - f for p in prt if p[2] == b and abs(p[0] - f) <= 10 * FPS]
        rows.append((min(cand, key=abs) / FPS if cand else np.nan, vis[max(f - FPS, 0):f + FPS].mean() > 0.5))
    return rows


# ----------------------------------------------------------------------------- main
def main():
    bar_bv, counts = barath_inputs()
    p_dead = dead_probs(counts)
    log("P(dead) ready (leave-one-match-out)")
    res, lags = [], []
    for gid in GIDS:
        g = load_game(gid)
        z = np.load(os.path.join(OUT, f"exp026_game{gid}.npz"))
        periods = [tuple(map(int, p)) for p in z["periods"]]
        n = len(g.period)
        play = np.zeros(n, bool)
        for a, b in periods:
            play[a:b] = True
        gt = possession_codes(g)
        gt[~play] = 3
        native = np.isfinite(g.ball[:, 0])
        team = pseudo_labels(z["dmin"], z["near_team"], native, periods)
        train = play & native & (team >= 0)
        y = (team == 1).astype(np.int8)
        prior = float(np.mean(y[train]))
        Xf, Xb = z["X_free"], z["X_ball"]
        cache = os.path.join(OUT, f"exp026_probs_game{gid}.npz")
        if os.path.exists(cache):
            c = np.load(cache)
            p_free, p_both, p_ball = c["p_free"], c["p_both"], c["p_ball"]
        else:
            p_free = np.mean([oof(Xf, y, train, play, s) for s in SEEDS], 0)
            p_ball = np.mean([oof(Xb, y, train, play, s) for s in SEEDS], 0)
            p_both = np.mean([oof(np.concatenate([Xf, Xb], 1), y, train, play, s) for s in SEEDS], 0)
            np.savez_compressed(cache, p_free=p_free, p_both=p_both, p_ball=p_ball)
        alive = play & (gt < 2)
        for nm, p, m in (("P_free", p_free, alive), ("P_ball", p_ball, alive & native), ("P_both", p_both, alive & native)):
            print(f"  game {gid} {nm:<7} acc vs event GT on alive frames{' (ball present)' if nm != 'P_free' else ''}: "
                  f"{np.mean((p[m] > 0.5) == (gt[m] == 1)):.3f}", flush=True)
        print(f"  game {gid} pseudo-label acc vs event GT (alive, labelled): "
              f"{np.mean(team[alive & (team >= 0)] == gt[alive & (team >= 0)]):.3f}; prior {prior:.3f}", flush=True)
        log(f"game {gid}: probabilities ready")

        L, W = PITCH
        bx, by = g.ball[:, 0], g.ball[:, 1]
        with np.errstate(invalid="ignore"):
            ball_out = (bx < -ms.OUT_MARGIN) | (bx > L + ms.OUT_MARGIN) | (by < -ms.OUT_MARGIN) | (by > W + ms.OUT_MARGIN)
        lp = logit(prior)
        variants = {"v3 (P_both)": p_both, "ball+free": sig(logit(p_ball) + logit(p_free) - lp), "free only": p_free}
        pidx = np.flatnonzero(play)
        masks = {"native": native}
        for o in OFFSETS:
            m = native.copy()
            m[pidx] &= bar_bv[(o + np.arange(len(pidx))) % len(bar_bv)]
            masks[f"donor {o // 1000}k"] = m
        gt_minutes = play.sum() / FPS / 60
        gt_spells = len(pm.possession_spells(gt, pm.valid_runs(gt)))
        print(f"  game {gid}: GT {gt_spells / gt_minutes:.2f} possessions/min, D {np.mean(gt[play] == 2):.1%}; "
              f"ball present on alive frames: native {native[alive].mean():.1%}, donors "
              + " ".join(f"{masks[k][alive].mean():.1%}" for k in masks if k != "native"), flush=True)
        for mk, vis in masks.items():
            for vk, pv in variants.items():
                pt = np.where(vis, pv, p_free)
                pred = decode(pt, p_dead[gid], prior, periods, vis, ball_out)
                r = pm.evaluate(gt, pred)
                runs = pm.valid_runs(gt)
                rate = len(pm.possession_spells(pm.fill_abstentions(pred.copy(), runs), runs)) / gt_minutes
                res.append(dict(game=gid, mask=mk, variant=vk, spellF1=r["spell_f1_mean"], macroF1=r["macro_f1"],
                                poss_per_min=rate, d_share=np.mean(pred[play] == 2)))
                lags += [dict(game=gid, mask=mk, variant=vk, lag=l, vis=v) for l, v in team_lags(gt, pred, vis, runs)]
            log(f"game {gid} mask {mk}: " + "  ".join(f"{r['variant']} {r['spellF1']:.3f}" for r in res[-len(variants):]))
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(OUT, "exp026_results.csv"), index=False)
    lg = pd.DataFrame(lags)
    lg.to_csv(os.path.join(OUT, "exp026_lags.csv"), index=False)
    report(df, lg)


def report(df, lg):
    pd.set_option("display.width", 160)
    print("\n=== spellF1 by match and mask ===")
    print(df.pivot_table(index=["game", "mask"], columns="variant", values="spellF1").round(3).to_string())
    for cond, sub in (("native ball", df[df["mask"] == "native"]), ("BAR-ATH donor masks", df[df["mask"] != "native"])):
        print(f"\n=== {cond}: mean over cells (n={len(sub) // 3}) ===")
        agg = sub.groupby("variant")[["spellF1", "macroF1", "poss_per_min", "d_share"]].mean().round(3)
        ref = sub[sub.variant == "v3 (P_both)"].set_index(["game", "mask"])["spellF1"]
        for v in agg.index:
            d = sub[sub.variant == v].set_index(["game", "mask"])["spellF1"] - ref
            agg.loc[v, "Δ v3"] = round(d.mean(), 4)
            agg.loc[v, "W/L"] = f"{int((d > 1e-9).sum())}/{int((d < -1e-9).sum())}"
        print(agg.to_string())
    print("\n=== lag of GT team changes (pred - GT, s; +-10 s window) ===")
    for (cond, sel) in (("native", lg["mask"] == "native"), ("donors", lg["mask"] != "native")):
        for v in lg.variant.unique():
            for vis_s, vm in (("ball visible", True), ("ball hidden", False)):
                x = lg[sel & (lg.variant == v) & (lg.vis == vm)].lag.values
                if len(x) == 0:
                    continue
                f = x[np.isfinite(x)]
                print(f"  {cond:<7} {v:<12} {vis_s:<13} n={len(x):4d}  found {len(f) / len(x):.0%}  median {np.median(f):+.2f}  "
                      f"|lag|<=1 s {np.mean(np.abs(f) <= 1) * len(f) / len(x):.0%}  <=2 s {np.mean(np.abs(f) <= 2) * len(f) / len(x):.0%}  "
                      f"late>1.5 s {np.mean(f > 1.5) * len(f) / len(x):.0%}")


if __name__ == "__main__":
    main()
