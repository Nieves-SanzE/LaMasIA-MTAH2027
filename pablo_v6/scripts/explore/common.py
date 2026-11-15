"""Shared setup for the exploration scripts: paths, the official metric module and the public GT.

Run any script from the repo root:  uv run python scripts/explore/<script>.py
Intermediate arrays (*.npy, *.pkl) are written to out/explore/, which is gitignored.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFF = os.path.join(ROOT, "docs", "more-than-a-hack-2027")
OUT = os.path.join(ROOT, "out", "explore")
os.makedirs(OUT, exist_ok=True)
os.chdir(OUT)  # scripts save/load their intermediate arrays by bare name

sys.path.insert(0, os.path.join(OFF, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import possession_metrics as pm  # noqa: E402
from mtah_2027 import atd  # noqa: E402

GT_CSV = os.path.join(OFF, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
LO, HI = 11185, 18685  # public GT window, half-open
HALVES = [(11185, 90652), (110808, 190198)]  # graded video frames, half-open


def gt_codes():
    return pm.label_codes(pm.read_label_csv(GT_CSV))


def runs(codes, lo=LO, hi=HI):
    """[(start, stop, label)] constant-label runs inside [lo, hi)."""
    out = []
    start = lo
    for i in range(lo + 1, hi + 1):
        if i == hi or codes[i] != codes[start]:
            out.append((start, i, "HADX"[codes[start]]))
            start = i
    return out
