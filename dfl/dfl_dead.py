"""Balón parado (P(muerto)) entrenado con la DFL: prueba cruzada Metrica <-> DFL. Solo agregados.

    cd pablo_v6
    set OMP_NUM_THREADS=4
    .venv\\Scripts\\python.exe ..\\dfl\\dfl_dead.py

Mismo modelo que la v6 (deadlib: features sin balón, ventanas <= 8 s, GBM de 300 iteraciones,
cada partido de entrenamiento limpio + recortado 'tipo cámara' con los jugadores visibles de
BAR-ATH). Solo cambian los datos de entrenamiento:
  M   = Metrica (3 partidos; verdad de D derivada de eventos)          <- la v6 actual
  D   = DFL (BallStatus del tracking)
  M+D = los dos
Siempre se deja fuera el partido evaluado. Evaluación en los dos lados, para no premiar al modelo
que aprende la convención del juez: test DFL (degradada como en el juez, semilla 0) y test Metrica.
Métricas: AUC por frame y parones con IoU >= 0,5 (decodificador de solo vivo/muerto de hmm3 con los
D_PARAMS de la v6 y el prior de D de cada conjunto de entrenamiento). Criterio fijado antes: un
modelo es mejor si mejora el F1 de parones en los DOS lados y en la mayoría de partidos.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PABLO = os.path.join(os.path.dirname(HERE), "pablo_v6")
for p in (os.path.join(PABLO, "scripts"), os.path.join(PABLO, "scripts", "explore"),
          os.path.join(PABLO, "src"), HERE):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import dfllib  # noqa: E402
import featlib  # noqa: E402,F401  (common.py: rutas y os.chdir de la copia de Pablo)
import make_submission as ms  # noqa: E402
from deadlib import dead_features, truncate_like  # noqa: E402
from hmm3 import dead_only, episode_scores  # noqa: E402
from metrica_lib import PITCH as M_PITCH, load_game  # noqa: E402
from dfl_v6 import PITCH as D_PITCH, to_top_left  # noqa: E402

COUNTS = dfllib.COUNTS_BLOCKS  # bloques oficiales (antes barath_counts_v6.npy, con 113.742; exp-092)
W = ms.V6_WINDOWS_S
STEP = 5
T0 = time.time()
DECODE = {k: v for k, v in ms.D_PARAMS.items()}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def feats(xy, team, pitch):
    return dead_features(xy, team, pitch, windows_s=W).values.astype(np.float32)


def train_set(xy, team, pitch, dead, play, counts, seed):
    """Como deadlib.metrica_dead_model: partido limpio + recortado tipo cámara, 1 de cada 5 frames."""
    X, y = [], []
    for v in (xy, truncate_like(xy, counts, rng=np.random.default_rng(seed))):
        X.append(feats(v, team, pitch)[play][::STEP])
        y.append(dead[play][::STEP])
    return np.concatenate(X), np.concatenate(y)


def fit(parts):
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X, y), float(y.mean())


def evaluate(models, X, dead, play):
    out = {}
    for name, (clf, prior) in models.items():
        p = clf.predict_proba(X)[:, 1]
        pred = np.zeros(len(p), bool)
        pred[play] = dead_only(p[play], prior, **DECODE)
        sc = episode_scores(dead, pred, valid=play)
        out[name] = dict(auc=roc_auc_score(dead[play], p[play]), f1=sc["f1"], prec=sc["prec"],
                         rec=sc["rec"], start=sc["start_err"], end=sc["end_err"],
                         share=pred[play].mean(), true_share=dead[play].mean())
    return out


def main():
    counts = np.load(COUNTS)
    met, dfl = {}, {}
    for gid in (1, 2, 3):
        g = load_game(gid)
        play = g.period > 0
        met[gid] = dict(tr=train_set(g.xy, g.team, M_PITCH, g.dead, play, counts, gid), play=play)
    log("Metrica: conjuntos de entrenamiento listos")
    mids = dfllib.match_ids()
    for i, mid in enumerate(mids):
        d = dfllib.load(mid)
        xy = to_top_left(dfllib.stacked(d, "players")).astype(np.float32)
        dead = dfllib.stacked(d, "status", 1) == 0
        play = np.ones(len(dead), bool)
        play[len(d["status_1"])] = False  # frame separador entre partes
        dfl[mid] = dict(tr=train_set(xy, d["team"], D_PITCH, dead, play, counts, 100 + i), play=play)
    log("DFL: conjuntos de entrenamiento listos")

    rows = {"DFL": [], "Metrica": []}
    m_all = fit([met[g]["tr"] for g in met])
    d_all = fit([dfl[m]["tr"] for m in dfl])
    for i, mid in enumerate(mids):
        d = dfllib.load(mid)
        _, players = dfllib.degrade(d, 2, seed=i)  # misma degradación que el juez (semilla 0)
        X = feats(to_top_left(players).astype(np.float32), d["team"], D_PITCH)
        dead = dfllib.stacked(d, "status", 1) == 0
        others = [dfl[o]["tr"] for o in mids if o != mid]
        models = {"M (v6 actual)": m_all, "D": fit(others), "M+D": fit([met[g]["tr"] for g in met] + others)}
        rows["DFL"].append((mid, evaluate(models, X, dead, dfl[mid]["play"])))
        log(f"test DFL {mid} hecho")
    for gid in (1, 2, 3):
        g = load_game(gid)
        X = feats(g.xy, g.team, M_PITCH)
        others = [met[o]["tr"] for o in met if o != gid]
        models = {"M (v6 actual)": fit(others), "D": d_all, "M+D": fit(others + [dfl[m]["tr"] for m in dfl])}
        rows["Metrica"].append((gid, evaluate(models, X, g.dead, met[gid]["play"])))
        log(f"test Metrica {gid} hecho")

    for side, rs in rows.items():
        print(f"\n=== test {side} ({len(rs)} partidos; siempre sin el partido evaluado) ===")
        print("%-14s %6s %6s %6s %6s %8s %8s %8s %8s" % ("modelo", "AUC", "F1", "P", "R", "inicio s",
                                                        "fin s", "D pred", "D real"))
        for name in rs[0][1]:
            v = {k: np.nanmean([r[1][name][k] for r in rs]) for k in rs[0][1][name]}
            print("%-14s %6.3f %6.3f %6.3f %6.3f %8.2f %8.2f %7.0f%% %7.0f%%" % (
                name, v["auc"], v["f1"], v["prec"], v["rec"], v["start"], v["end"],
                100 * v["share"], 100 * v["true_share"]))
        for name in ("D", "M+D"):
            df = np.array([r[1][name]["f1"] - r[1]["M (v6 actual)"]["f1"] for r in rs])
            print("  %-4s vs M: F1 %+.3f, partidos ganados/perdidos %d/%d" % (
                name, df.mean(), (df > 1e-9).sum(), (df < -1e-9).sum()))


if __name__ == "__main__":
    main()
