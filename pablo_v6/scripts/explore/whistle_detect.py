"""Turn the per-hop whistle features into whistle events (out/audio/whistles.csv).

A hop is a candidate when the narrow-band contrast exceeds THR dB and the peak sits in the
whistle band found at the kickoffs (3.70-4.10 kHz). Candidates closer than MERGE_S are merged,
and an event must last at least MIN_S. Run whistle_features.py first.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIO = os.path.join(ROOT, "out", "audio")
THR = float(sys.argv[1]) if __name__ == "__main__" and len(sys.argv) > 1 else 25.0  # argv only when run as a script
HZ = (3700, 4100)
MERGE_S, MIN_S = 0.10, 0.10


def detect(thr=THR):
    f = np.load(os.path.join(AUDIO, "whistle_feats.npz"))
    hop = float(f["hop_s"])
    c, hz, harm = f["contrast"], f["peak_hz"], f["harm"]
    on = (c > thr) & (hz > HZ[0]) & (hz < HZ[1])
    # close small gaps
    gap = int(round(MERGE_S / hop))
    idx = np.flatnonzero(on)
    events = []
    if len(idx):
        start = prev = idx[0]
        for i in idx[1:]:
            if i - prev > gap + 1:
                events.append((start, prev + 1)); start = i
            prev = i
        events.append((start, prev + 1))
    rows = []
    for s, e in events:
        if (e - s) * hop < MIN_S:
            continue
        seg = slice(s, e)
        rows.append({"t_s": s * hop, "frame": int(round(s * hop * 25)), "dur_s": (e - s) * hop,
                     "max_contrast": float(c[seg].max()), "median_hz": float(np.median(hz[seg])),
                     "hz_std": float(np.std(hz[seg][c[seg] > thr])), "max_harm": float(harm[seg].max())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ev = detect()
    ev.to_csv(os.path.join(AUDIO, "whistles.csv"), index=False)
    halves = [(11185, 90652), (110808, 190198)]
    g = ev.frame.apply(lambda fr: any(a <= fr < b for a, b in halves))
    ht = (ev.frame >= 90652) & (ev.frame < 110808)
    print("threshold %.0f dB: %d events total, %d in graded halves (%.2f/min), %d in halftime (%.2f/min)"
          % (THR, len(ev), g.sum(), g.sum() / (158857 / 1500), ht.sum(), ht.sum() / (20156 / 1500)))
    print("duration pct (s):", np.percentile(ev.dur_s, [25, 50, 75, 95]).round(2))
