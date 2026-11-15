"""Shared evaluation pieces from exp-013 (fe_combine.py): out-of-fold GBM, 2-state HMM decoder,
public-window gap-injection harness and full-match self-consistency test.

All constants are the ones fixed in exp-013 (temperature 0.2 was chosen there without GT).

    from hmmlib import Ctx
    ctx = Ctx()                       # loads tracking, labels, whistles
    p = ctx.oof(X)                    # out-of-fold P(A) on every play frame
    ctx.harness({"name": lambda vis: ctx.decode(vis, p_vis, p_free)})
"""
import time

from featlib import *  # noqa: F403
sys.path.insert(0, os.path.join(ROOT, "scripts", "explore"))
from gaplib import Features, find_gaps, predict, q10_dead  # noqa: E402
from whistle_detect import detect  # noqa: E402

D, X = 2, 3
STEP = 5
CLIP = 0.02
POSS_PER_MIN = 3.0
TEMP = 0.2  # exp-013, chosen by matching ~3 possessions/min on the full match (no GT)
T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


class Ctx:
    def __init__(self):
        self.tr = load_tracking()
        self.lab = labels(self.tr)
        self.N = self.tr.n_frames
        self.play = self.lab["play"]
        self.gt = self.lab["gt"]
        self.bv = self.lab["ball"]
        nb = self.N // BLOCK + 1
        self.fold = (np.random.default_rng(0).permutation(nb) % 5)[self.lab["blocks"]]
        self.prior = np.mean(self.lab["team"][self.lab["train"]] == A)
        self.p_sw = POSS_PER_MIN / 60 / FPS
        ev = detect(25)
        ev = ev[(ev.max_contrast >= 35) & (ev.max_harm >= 20) & (ev.dur_s >= 0.4)]
        self.F = Features(self.tr, ev.frame.values)

    # ------------------------------------------------------------------ models
    def oof(self, Xm, fit_mask=None, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        fit_mask = self.lab["train"] if fit_mask is None else fit_mask
        p = np.full(self.N, np.nan, np.float32)
        y = self.lab["team"]
        sub = np.zeros(self.N, bool)
        sub[::STEP] = True
        for k in range(5):
            trn = np.flatnonzero(fit_mask & sub & (self.fold != k))
            te = np.flatnonzero(self.play & (self.fold == k))
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
            clf.fit(Xm[trn], y[trn])
            p[te] = clf.predict_proba(Xm[te])[:, 1]
        return p

    def scores(self, p, name):
        from sklearn.metrics import roc_auc_score
        m = self.lab["train"]
        y = self.lab["team"][m] == A
        w = np.zeros(self.N, bool)
        w[LO:HI] = True
        al = w & (self.gt < 2)
        gv, gm = al & self.bv, al & ~self.bv
        print(f"  {name:<40} pseudo acc {np.mean((p[m] > 0.5) == y):.3f} AUC {roc_auc_score(y, p[m]):.3f} | "
              f"read-only GT acc: ball vis {np.mean((p[gv] > 0.5) == (self.gt[gv] == A)):.3f}  "
              f"ball missing {np.mean((p[gm] > 0.5) == (self.gt[gm] == A)):.3f}", flush=True)

    # ------------------------------------------------------------------ decoder
    def viterbi(self, p, temp=TEMP):
        prior = self.prior
        p = np.clip(np.where(np.isfinite(p), p, prior), CLIP, 1 - CLIP)
        la = temp * np.log(p / prior)
        lh = temp * np.log((1 - p) / (1 - prior))
        ls, lc = np.log(self.p_sw), np.log(1 - self.p_sw)
        n = len(p)
        back = np.zeros((n, 2), bool)
        vh, va = lh[0], la[0]
        for i in range(1, n):
            sh, oh = vh + lc, va + ls
            sa, oa = va + lc, vh + ls
            back[i, 0] = oh > sh
            back[i, 1] = oa > sa
            vh = max(sh, oh) + lh[i]
            va = max(sa, oa) + la[i]
        out = np.zeros(n, np.int8)
        s = int(va > vh)
        for i in range(n - 1, -1, -1):
            out[i] = s
            if back[i, s]:
                s = 1 - s
        return out

    def decode(self, vis, p_vis, p_miss, segs=((LO, HI),), q10=False, temp=TEMP):
        e = np.where(vis, p_vis, np.nan if p_miss is None else p_miss)
        out = np.full(self.N, X, np.int8)
        for lo, hi in segs:
            pred = self.viterbi(e[lo:hi], temp)
            dead = self.F.slow[lo:hi].copy()
            if q10:
                for s, t in find_gaps(~vis[lo:hi]):
                    dq = q10_dead(self.F, vis, s + lo, t + lo, lo, hi)
                    if dq is not None:
                        dead[dq[0] - lo:dq[1] - lo] = True
            pred[dead] = D
            out[lo:hi] = pred
        return out

    # ------------------------------------------------------------------ evaluations
    def masks(self):
        donors = [s for s in range(LO + BLOCK, HALVES[0][1] - BLOCK + 1, BLOCK)]
        donors += [s for s in range(KICKOFF_2H, HALVES[1][1] - BLOCK + 1, BLOCK)]
        out = []
        for s in [None] + donors:
            vis = self.bv.copy()
            if s is not None:
                vis[LO:HI] &= self.bv[s:s + BLOCK]
            out.append(vis)
        return out

    def harness(self, variants, ref=None):
        """variants: name -> fn(vis) -> full-length prediction. Paired Δ vs `ref` (default: first)."""
        masks = self.masks()
        res = {k: np.array([[pm.evaluate(self.gt, fn(v))[m] for m in ("spell_f1_mean", "macro_f1", "n_pred_spells")]
                            for v in masks]) for k, fn in variants.items()}
        ref = ref or next(iter(variants))
        r0 = res[ref][1:, 0]
        print(f"\n{'method':<44} {'no gaps':>8} {'donors sF1':>13} {'Δ ref':>7} {'W/L':>6} {'macroF1':>8} {'spells':>7}")
        for k, v in res.items():
            d = v[1:, 0]
            print(f"{k:<44} {v[0, 0]:8.3f} {d.mean():7.3f}±{d.std():.3f} {np.mean(d - r0):+7.3f} "
                  f"{np.sum(d > r0):>2}/{np.sum(d < r0):<3} {v[1:, 1].mean():8.3f} {v[1:, 2].mean():7.1f}", flush=True)
        return res

    def self_consistency(self, variants):
        """variants: name -> (p_vis, p_miss). Hide the ball of every 5-min block with the pattern of
        the block 3 positions later; agreement on newly hidden frames with each variant's real-ball output."""
        bv, play = self.bv, self.play
        starts = [s for a, b in PLAY for s in range(a, b - BLOCK + 1, BLOCK)]
        vis_h = bv.copy()
        for i, s in enumerate(starts):
            d = starts[(i + 3) % len(starts)]
            vis_h[s:s + BLOCK] &= bv[d:d + BLOCK]
        hidden = play & bv & ~vis_h
        pseudo = self.lab["team"]
        for name, (pv, pmiss) in variants.items():
            ref = self.decode(bv, pv, pmiss, PLAY)
            pr = self.decode(vis_h, pv, pmiss, PLAY)
            m = hidden & (pr < 2) & (ref < 2)
            m2 = hidden & (pr < 2) & (pseudo >= 0)
            print(f"  {name:<44} vs own real-ball output {np.mean(pr[m] == ref[m]):.3f}   "
                  f"vs exp-003 pseudo-label {np.mean(pr[m2] == pseudo[m2]):.3f}", flush=True)
