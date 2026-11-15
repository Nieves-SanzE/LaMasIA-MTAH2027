"""Joint H / A / D Viterbi decoder (D-T1): the dead ball is a state, not a mask painted afterwards.

States: H, A and a chain D_1 .. D_m (minimum stoppage duration m frames; D_m loops). Transitions
per frame: H<->A with p_sw, H/A -> D_1 with p_stop, D_m -> H or A with p_end/2 each (so the team may
change for free through a stoppage, which is where most changes of possession are). Emissions:
  team   temp_t * log(P(A)/prior)           on A, temp_t * log((1-P(A))/(1-prior)) on H
  dead   temp_d * (log(P(D)/prior_d) + bias) on D, temp_d * log((1-P(D))/(1-prior_d)) on H and A
The team classifier says nothing about the ball being dead, so D also gets the better of the two
team emissions: alive vs dead is decided by P(D) alone, whatever the team temperature (exp-020b).
NaN probabilities are uninformative (= prior). Returns codes 0=H, 1=A, 2=D.
"""
import numpy as np

H, A, D = 0, 1, 2
FPS = 25
CLIP = 0.02


def _loglik(p, prior, temp, bias=0.0):
    p = np.clip(np.where(np.isfinite(p), p, prior), CLIP, 1 - CLIP)
    return temp * (np.log(p / prior) + bias), temp * np.log((1 - p) / (1 - prior))


def viterbi3(p_team, p_dead, prior_t, prior_d, *, temp_t, temp_d, poss_per_min=3.0,
             stops_per_min=1.0, min_dead_s=3.0, mean_dead_s=15.0, bias=0.0, force_dead=None):
    n = len(p_dead)
    la, lh = _loglik(p_team, prior_t, temp_t)
    ld, ll = _loglik(p_dead, prior_d, temp_d, bias)
    if force_dead is not None:  # e.g. ball visibly out of the pitch
        ld = np.where(force_dead, 0.0, ld)
        ll = np.where(force_dead, -50.0, ll)
    eh, ea = lh + ll, la + ll
    ld = ld + np.maximum(la, lh)
    m = max(int(min_dead_s * FPS), 1)
    p_sw = poss_per_min / 60 / FPS
    p_stop = stops_per_min / 60 / FPS
    p_end = 1.0 / max((mean_dead_s - min_dead_s) * FPS, 1.0)
    l_stay = np.log(1 - p_sw - p_stop)
    l_sw, l_stop = np.log(p_sw), np.log(p_stop)
    l_end, l_dstay = np.log(p_end / 2), np.log(1 - p_end)

    vh, va = eh[0], ea[0]
    vd = np.full(m, -np.inf)
    vd[0] = ld[0] + l_stop
    # back pointers: H from {H, A, Dm}, A from {A, H, Dm}, D1 from {H, A}, Dm from {Dm-1, Dm}
    bh = np.zeros(n, np.int8)
    ba = np.zeros(n, np.int8)
    b1 = np.zeros(n, np.int8)
    bm = np.zeros(n, np.int8)
    for i in range(1, n):
        dm = vd[-1] + l_end
        ch = (vh + l_stay, va + l_sw, dm)
        ca = (va + l_stay, vh + l_sw, dm)
        kh = int(np.argmax(ch))
        ka = int(np.argmax(ca))
        c1 = (vh + l_stop, va + l_stop)
        k1 = int(c1[1] > c1[0])
        new_d = np.empty(m)
        new_d[0] = c1[k1]
        if m > 1:
            new_d[1:] = vd[:-1]
            stay = vd[-1] + l_dstay
            km = int(stay > vd[-2])
            new_d[-1] = stay if km else vd[-2]
        else:
            stay = vd[0] + l_dstay
            km = int(stay > c1[k1])
            new_d[0] = max(stay, c1[k1])
        new_d += ld[i]
        vh, va, vd = ch[kh] + eh[i], ca[ka] + ea[i], new_d
        bh[i], ba[i], b1[i], bm[i] = kh, ka, k1, km
    out = np.zeros(n, np.int8)
    # state encoding during backtrack: ("H",), ("A",), ("D", k) with k in 0..m-1
    s = int(np.argmax([vh, va, vd[-1]]))
    st = ("H",) if s == 0 else ("A",) if s == 1 else ("D", m - 1)
    for i in range(n - 1, -1, -1):
        if st[0] == "H":
            out[i] = H
            k = bh[i]
            st = ("H",) if k == 0 else ("A",) if k == 1 else ("D", m - 1)
        elif st[0] == "A":
            out[i] = A
            k = ba[i]
            st = ("A",) if k == 0 else ("H",) if k == 1 else ("D", m - 1)
        else:
            out[i] = D
            j = st[1]
            if j == m - 1 and bm[i]:
                st = ("D", m - 1)          # D_m looping
            elif j == 0:
                st = ("H",) if b1[i] == 0 else ("A",)
            else:
                st = ("D", j - 1)
    return out


def dead_only(p_dead, prior_d, **kw):
    """Alive/dead decoding with no team information (P(A) = prior everywhere)."""
    return viterbi3(np.full(len(p_dead), np.nan), p_dead, 0.5, prior_d, temp_t=0.0, **kw) == D


def episodes(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def episode_scores(true_mask, pred_mask, valid=None, iou_thr=0.5):
    """Episode-level precision / recall at temporal IoU >= thr (greedy one-to-one), plus median
    absolute start / end error (s) of the matched pairs."""
    if valid is not None:
        true_mask, pred_mask = true_mask & valid, pred_mask & valid
    te, pe = episodes(true_mask), episodes(pred_mask)
    pairs = []
    for i, (a, b) in enumerate(te):
        for j, (c, d) in enumerate(pe):
            inter = min(b, d) - max(a, c)
            if inter > 0:
                pairs.append((inter / (max(b, d) - min(a, c)), i, j))
    pairs.sort(reverse=True)
    ut, up, err_s, err_e = set(), set(), [], []
    for iou, i, j in pairs:
        if iou < iou_thr or i in ut or j in up:
            continue
        ut.add(i)
        up.add(j)
        err_s.append(abs(pe[j][0] - te[i][0]) / FPS)
        err_e.append(abs(pe[j][1] - te[i][1]) / FPS)
    prec = len(up) / max(len(pe), 1)
    rec = len(ut) / max(len(te), 1)
    return dict(n_true=len(te), n_pred=len(pe), prec=prec, rec=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-9),
                start_err=np.median(err_s) if err_s else np.nan, end_err=np.median(err_e) if err_e else np.nan)
