"""Contexto temporal en P_free de E1 (exp-087), para la generación de e1_v5 (make_e1.py).

P_free se entrena y se aplica con sus 166 features más f(t + 2 s) - f(t) y f(t) - f(t - 2 s), dentro de la misma
parte (fuera, NaN). Mismas filas, etiqueta (control 1 s) y GBM que make_e1.team_training / fit_team.
No contiene nada de los jueces; los jueces (dfl_ctx_judge.py y siguientes) importan de aquí.
"""
import numpy as np

import dfllib
import make_e1 as e1
from dfl_teacher import SEEDS, STEP, fit_external

K = 2 * 25  # 2 s a 25 fps


def seg_ids(n, periods):
    """Índice de parte de cada frame (-1 fuera de las partes)."""
    s = np.full(n, -1, np.int32)
    for i, (a, b) in enumerate(periods):
        s[a:b] = i
    return s


def context(X, seg, rows=None):
    """[X, X(t+K)-X(t), X(t)-X(t-K)] en las filas pedidas (todas si rows es None)."""
    n = len(X)
    r = np.arange(n) if rows is None else np.asarray(rows)
    out = [X[r]]
    for sh in (K, -K):
        j = np.clip(r + sh, 0, n - 1)
        ok = (r + sh >= 0) & (r + sh < n) & (seg[j] == seg[r]) & (seg[r] >= 0)
        diff = (X[j] - X[r]) if sh > 0 else (X[r] - X[j])
        diff[~ok] = np.nan
        out.append(diff)
    return np.concatenate(out, 1).astype(np.float32)


def dfl_periods(mid):
    """Partes de un partido de la DFL tal como las concatena dfllib (con un frame separador)."""
    d = dfllib.load(mid)
    n1, n2 = len(d["status_1"]), len(d["status_2"])
    return [(0, n1), (n1 + 1, n1 + 1 + n2)]


def train_rows(mids):
    Xs, ys = [], []
    for mid in mids:
        d = dfllib.load(mid)
        ball, pl = dfllib.stacked(d, "ball")[:, :2], dfllib.stacked(d, "players").astype(np.float64)
        gt = dfllib.official_style(dfllib.gt_labels(d), ball, pl, d["team"], 1.0, True)
        Xf, _, _, _, vis = e1.load_train(mid)
        assert len(Xf) == dfl_periods(mid)[1][1] == len(gt), "longitudes distintas"
        idx = np.flatnonzero(((gt == "H") | (gt == "A")) & vis)[::STEP]  # mismas filas que make_e1.team_training
        Xs.append(context(Xf, seg_ids(len(Xf), dfl_periods(mid)), idx))
        ys.append((gt[idx] == "A").astype(np.int8))
    return Xs, ys


def fit_pf(mids, ws=None):
    """ws (opcional, e1_v7): pesos por fila de cada partido, de make_e1.spell_weights (mismas filas)."""
    Xs, ys = train_rows(mids)
    if ws is not None:
        assert [len(w) for w in ws] == [len(y) for y in ys], "pesos y filas distintos"
    return [fit_external(Xs, ys, s, ws) for s in SEEDS]


def predict(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], 0)
