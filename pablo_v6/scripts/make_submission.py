"""Build the possession file for all 199,319 frames (v4: v3 + ball/shape product of experts, exp-028;
v6: v4 + goalkeeper swap cleaning + NaN-velocity mask & 8s window cap in P(dead) + 0.5*(base+clean_mid) P_ball, exp-054/055).

Pipeline (everything derived by code from the provided data; no per-frame constants of this match):
  1. periods: kick-off and end of each half detected from the tracking + referee whistles
     (src/mtah_2027/periods.py);
  2. pseudo-labels (exp-003 hold-last-clear-control; the experimental v5 moves each team switch to halfway
     through the ball's flight, exp-027; it does NOT transfer to this match, exp-029) and
     feature families (out/explore/features, produced by scripts/explore/fe_geometry.py,
     fe_temporal.py, fb_ball.py); in v6, goalkeeper team-swaps and cross-team GK duplicates are
     cleaned in memory (scripts/explore/gklib.py, exp-054);
  3. out-of-fold GBMs, averaged over 5 seeds; the public GT window +-1 min is never used for training:
     P_free from ball-free features (ball gaps), P_both from all features, P_ball from the ball
     families only (in v6: 50/50 log-odds blend of `base` and `clean_mid` P_ball models, exp-055).
     Team probability on ball-visible frames: product of experts
     logit(P_ball) + logit(P_free) - logit(prior) (exp-023/026) in v4/v5/v6; v3 = P_both;
  4. team temperature chosen without GT on the team-only decoder (v2: full-match possession rate
     closest to ~3 per minute, literature; v6 uses 0.6 validated on Metrica RV3, exp-055);
  5. P(dead): ball-free live/dead GBM trained on the 3 public Metrica matches (human event labels of
     OTHER matches, data/external/metrica; scripts/download_metrica.py), each match used clean and
     with camera-like player dropout (exp-019/021); our features after removing tracker duplicates
     (and in v6: masking off-camera interpolated slots with velocity=NaN and capping rolling windows
     to <= 8 s so short live plays are not bridged into stoppages, exp-055);
  6. joint H/A/D Viterbi per half (scripts/explore/hmm3.py) with the D parameters chosen on Metrica
     (exp-020b); the ball visibly > 1 m out of the pitch for >= 0.5 s forces D, but only next to a
     stoppage the decoder already found (it may move a boundary, never create a stoppage; exp-021c);
  7. labels: H/A/D inside each half, D between the end of the 1st half and the 2nd kick-off,
     X elsewhere (not graded).
Writes out/submission/<name>.csv. Run from the repo root:
    uv run python scripts/make_submission.py [--variant v4|v3|v5|v5b|v6] [--name NAME]
v5b = v5 (`mid` pseudo-labels) with the team temperature fixed at 0.4, chosen on Metrica (exp-046/048).
v6  = v4 + GK cleaning (exp-054) + P(dead) NaN-vel mask & <=8s windows + 0.5*(base+clean_mid) P_ball at T=0.6 (exp-055).
"""
import argparse
import copy
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "explore"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import featlib  # noqa: E402
from deadlib import dead_features, dedupe, extend_only, mask_no_velocity, metrica_dead_model  # noqa: E402
from hmm3 import viterbi3  # noqa: E402
from mtah_2027 import atd  # noqa: E402
from mtah_2027.periods import detect_periods  # noqa: E402
from whistle_detect import detect  # noqa: E402

FREE = ["goalside", "shape", "convergence", "graph_intra", "space", "leadlag", "sync", "tda", "multiscale"]
BALL = ["ball_pressure", "ball_structure", "ball_space", "ball_graph", "ball_motion"]
LABELS = np.array(list("HADX"))
SEEDS = range(5)
TEMPS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
POSS_PER_MIN = 3.0
V5B_TEMP = 0.4  # exp-046: best team temperature on Metrica with `mid` labels; exp-048 on BAR-ATH
V6_TEMP = 0.6   # exp-055: best team temperature on Metrica RV3 with 0.5*(base+clean_mid) P_ball
V6_WINDOWS_S = (1, 2, 4, 8)  # exp-055: cap P(dead) rolling windows at 8 s to prevent bridging short live plays
PRIOR_D = 0.352  # Metrica dead share (exp-019)
D_PARAMS = dict(temp_d=0.3, min_dead_s=4.0, stops_per_min=1.0, mean_dead_s=15.0, bias=-0.5)  # exp-020b
OUT_MARGIN = 1.0  # m outside the pitch


def _logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def build(variant="v4", temp=None):
    """Everything the decoder needs, derived from the data (steps 1-5). Returns a dict."""
    # 1) periods from the data
    tr = featlib.load_tracking()
    play_official = featlib.play_mask(tr.n_frames)
    ev = detect(25)
    whistles = ev[(ev.max_contrast >= 35) & (ev.max_harm >= 20) & (ev.dur_s >= 0.4)].frame.values
    periods = detect_periods(tr.metros(tr.xy), tr.velocidad, tr.equipo, tuple(atd.CAMPO), whistles)
    print("detected periods (kick-off, end):", periods)
    # the shared exploration code reads these module globals: point them at the detected periods
    featlib.PLAY = periods
    featlib.HALVES = periods

    from hmmlib import Ctx  # imported after the patch so labels() and attack_sign() use it
    ctx = Ctx()
    if variant in ("v5", "v5b"):
        ctx.lab = featlib.relabel(ctx.lab, "mid")
        ctx.prior = np.mean(ctx.lab["team"][ctx.lab["train"]] == 1)
    X_free, _ = featlib.load_families(FREE)
    X_ball, _ = featlib.load_families(BALL)
    X_all = np.concatenate([X_free, X_ball], 1)
    p_free = np.mean([ctx.oof(X_free, seed=s) for s in SEEDS], 0)
    p_both = np.mean([ctx.oof(X_all, seed=s) for s in SEEDS], 0)
    ctx.scores(p_free, "ball-free model")
    ctx.scores(p_both, "all-features model")
    if variant == "v6":
        import gklib
        tr_c, changed_gk, gk_stats = gklib.clean_goalkeepers(tr, periods)
        print("GK cleaning stats (exp-054):", gk_stats)
        X_ball_cln = gklib.patch_ball_features(tr_c, X_ball, changed_gk)
        ctx_cln = copy.copy(ctx)
        ctx_cln.tr = tr_c
        ctx_cln.lab = featlib.labels(tr_c)
        ctx_cln.prior = float(np.mean(ctx_cln.lab["team"][ctx_cln.lab["train"]] == 1))
        p_ball_base = np.mean([ctx_cln.oof(X_ball_cln, seed=s) for s in SEEDS], 0)
        ctx_cln.scores(p_ball_base, "ball-families-only model (GK-cleaned, base)")
        ctx_cmid = copy.copy(ctx_cln)
        ctx_cmid.lab = featlib.relabel(ctx_cln.lab, "clean_mid")
        p_ball_cmid = np.mean([ctx_cmid.oof(X_ball_cln, seed=s) for s in SEEDS], 0)
        ctx_cmid.scores(p_ball_cmid, "ball-families-only model (GK-cleaned, clean_mid)")
        p_ball = _sigmoid(0.5 * (_logit(p_ball_base) + _logit(p_ball_cmid)))
        ctx.prior = ctx_cln.prior
        p_vis = _sigmoid(0.5 * (_logit(p_ball_base) + _logit(p_ball_cmid)) + _logit(p_free) - _logit(ctx.prior))
        del X_ball_cln
    elif variant in ("v4", "v5", "v5b"):
        p_ball = np.mean([ctx.oof(X_ball, seed=s) for s in SEEDS], 0)
        ctx.scores(p_ball, "ball-families-only model")
        p_vis = _sigmoid(_logit(p_ball) + _logit(p_free) - _logit(ctx.prior))
    else:
        p_ball = None
        p_vis = p_both
    del X_free, X_ball, X_all

    # 4) team temperature from the full-match possession rate (no GT), team-only decoder; kept on
    #    P_both in v4 too, as validated in exp-023/026 (temperature 0.85)
    from hmmlib import pm, FPS
    minutes = sum(b - a for a, b in periods) / FPS / 60

    def rate(t):
        f = pm.fill_abstentions(ctx.decode(ctx.bv, p_both, p_free, segs=periods, temp=t), periods)
        return len(pm.possession_spells(f, periods)) / minutes

    if temp is None and variant == "v5b":
        temp = V5B_TEMP
    if temp is None and variant == "v6":
        temp = V6_TEMP
    if temp is None:
        temp = min(TEMPS, key=lambda t: abs(rate(t) - POSS_PER_MIN))
    print(f"temperature {temp} ({rate(temp):.2f} possessions/min, team-only decoder)")

    # 5) P(dead) from the Metrica model
    play = featlib.play_mask(tr.n_frames)
    team = tr.equipo.astype(np.int8)
    if variant == "v6":
        xy_d = dedupe(mask_no_velocity(tr.metros(tr.xy), tr.velocidad), team)
        counts = np.isfinite(xy_d[..., 0]).sum(1)[play_official]
        clf = metrica_dead_model(counts, windows_s=V6_WINDOWS_S)
        p_dead = clf.predict_proba(dead_features(xy_d, team, atd.CAMPO, windows_s=V6_WINDOWS_S).values)[:, 1]
    else:
        counts = np.isfinite(tr.xy[..., 0]).sum(1)[play]
        clf = metrica_dead_model(counts)
        p_dead = clf.predict_proba(dead_features(dedupe(tr.metros(tr.xy), team), team, atd.CAMPO).values)[:, 1]
    print(f"P(dead) mean over play frames {p_dead[play].mean():.3f}")

    bm = tr.metros(tr.balon)
    L, W = float(atd.CAMPO[0]), float(atd.CAMPO[1])
    with np.errstate(invalid="ignore"):
        ball_out = (bm[:, 0] < -OUT_MARGIN) | (bm[:, 0] > L + OUT_MARGIN) | (bm[:, 1] < -OUT_MARGIN) | (bm[:, 1] > W + OUT_MARGIN)
    return dict(tr=tr, ctx=ctx, periods=periods, p_both=p_both, p_free=p_free, p_ball=p_ball, p_vis=p_vis, temp=temp,
                p_dead=p_dead, ball_out=ball_out, prior=ctx.prior)


def decode(m, vis):
    """6-7) Labels for all frames given a ball-visibility mask (the real one, or a harness mask)."""
    # ball visibly out for >= 0.5 s, used only next to the stoppages the decoder already found
    out = vis & m["ball_out"]
    s_ = pd.Series(out)
    out &= s_.groupby(s_.ne(s_.shift()).cumsum()).transform("size").values >= featlib.FPS // 2
    p_team = np.where(vis, m["p_vis"], m["p_free"])
    pred = np.full(len(vis), 3, np.int8)
    kw = dict(temp_t=m["temp"], **D_PARAMS)
    prior = m["prior"] if "prior" in m else m["ctx"].prior  # older exploration scripts pass a ctx
    for a, b in m["periods"]:
        first = viterbi3(p_team[a:b], m["p_dead"][a:b], prior, PRIOR_D, **kw)
        force = extend_only(out[a:b], first == 2, 5 * featlib.FPS)
        pred[a:b] = viterbi3(p_team[a:b], m["p_dead"][a:b], prior, PRIOR_D, force_dead=force, **kw)
    (_, end1), (ko2, _) = m["periods"]
    pred[end1:ko2] = 2  # half-time and the 2nd-half line-up are dead ball
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("v4", "v3", "v5", "v5b", "v6"), default="v4")
    ap.add_argument("--name", default=None, help="default: submission_<variant>")
    args = ap.parse_args()
    args.name = args.name or f"submission_{args.variant}"
    m = build(args.variant)
    pred = decode(m, m["ctx"].bv)

    out_dir = os.path.join(ROOT, "out", "submission")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{args.name}.csv")
    # newline="\n": byte-identical output (and sha256) on every OS
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{LABELS[c]}\n" for i, c in enumerate(pred))
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    n = len(pred)
    share = {l: np.mean(LABELS[pred] == l) for l in "HADX"}
    print(f"wrote {path}: {n} rows, sha256 {sha[:16]}..., label shares "
          + " ".join(f"{k} {v:.3f}" for k, v in share.items()))


if __name__ == "__main__":
    main()
