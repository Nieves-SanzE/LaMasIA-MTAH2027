"""RV suite shared pieces (ideas/revalidation-test-plan.md): cached build() outputs and fast scoring.

RV0: everything make_submission.decode needs is cached in out/cache/build_<variant>.npz, so a decoder
experiment costs seconds instead of the ~25 min of retraining the 15 out-of-fold GBMs.

    from rvlib import load_build, decode, spell_f1
    m = load_build("v4")                 # builds once (heavy, ~5 GB) if the cache is missing
    pred = decode(m, m["bv"])            # == out/submission/submission_v4.csv
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "explore"))
import make_submission as ms  # noqa: E402
import featlib  # noqa: E402
from common import pm  # noqa: E402

FPS = 25
BLOCK = 5 * 60 * FPS
CACHE_DIR = os.path.join(ROOT, "out", "cache")
ARRAYS = ("p_free", "p_both", "p_ball", "p_vis", "p_dead", "ball_out", "temp", "periods", "prior")
LABEL_KEYS = ("team", "train", "play", "ball", "dmin", "near_team")


def cache_path(variant):
    return os.path.join(CACHE_DIR, f"build_{variant}.npz")


def save_build(m, variant):
    """Cache the decoder inputs of a build() result plus the pseudo-labels of its context."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    arrays = {k: np.asarray(m[k]) for k in ARRAYS if m.get(k) is not None}
    lab = m["ctx"].lab
    arrays.update({f"lab_{k}": np.asarray(lab[k]) for k in LABEL_KEYS})
    np.savez_compressed(cache_path(variant), **arrays)


def load_build(variant="v4"):
    """Cached build() outputs as a dict usable by make_submission.decode (no `tr` / `ctx`)."""
    path = cache_path(variant)
    if not os.path.exists(path):
        save_build(ms.build(variant), variant)
    z = np.load(path)
    m = {k: z[k] for k in z.files if not k.startswith("lab_")}
    m["temp"] = float(m["temp"])
    m["prior"] = float(m["prior"])
    m["periods"] = [tuple(int(v) for v in p) for p in m["periods"]]
    m["lab"] = {k[4:]: z[k] for k in z.files if k.startswith("lab_")}
    m["bv"] = m["lab"]["ball"]
    m["play"] = m["lab"]["play"]
    return m


def decode(m, vis, **overrides):
    """make_submission.decode with optional overrides of the cached inputs (e.g. temp, p_vis)."""
    return ms.decode(dict(m, **overrides), vis)


def read_submission(name):
    return pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "out", "submission", f"{name}.csv")))


def spell_f1(gt, pred):
    """Official spellF1 (mean over IoU thresholds) without the bootstrap; X in `gt` = not graded."""
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pred.copy(), runs), runs)["mean_f1"]


def block_starts(periods):
    """Start frame of every full 5-min block inside the graded periods (the RV independent unit)."""
    return [s for a, b in periods for s in range(a, b - BLOCK + 1, BLOCK)]


def poss_per_min(pred, periods):
    f = pm.fill_abstentions(pred.copy(), periods)
    minutes = sum(b - a for a, b in periods) / FPS / 60
    return len(pm.possession_spells(f, periods)) / minutes
