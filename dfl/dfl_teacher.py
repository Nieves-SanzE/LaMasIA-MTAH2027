"""¿Cuánto vale cambiar las pseudo-etiquetas por etiquetas reales? Juez DFL, nivel 2, semilla 0.

    cd pablo_v6
    set OMP_NUM_THREADS=4
    .venv\\Scripts\\python.exe -u ..\\dfl\\dfl_teacher.py

Referencia B0 = v6 + D Metrica+DFL (exp-060). Solo cambia P(equipo); mismo P(muerto) M+D (sin el partido
evaluado), mismo decodificador y temperatura 0,6. Todo se entrena sobre datos degradados.
  T1  profesor perfecto: como la v6, dentro del partido y fuera de pliegue, con etiquetas reales.
      Mide el valor de la etiqueta. Solo existe en la DFL: en BAR-ATH no hay etiquetas.
  E1  externo: entrenado con los OTROS 6 partidos y etiquetas reales. Sí se puede aplicar a BAR-ATH.
  E2  mezcla: evidencia media en log-odds de la v6 (pseudo-etiquetas) y de E1.
Verdades: control 1 s y control 2 s + mitad del vuelo; cada variante se entrena con la verdad con la que
se puntúa. Los modelos solo aprenden de frames vivos con el balón visible, como la v6.

Criterios fijados antes de ver resultados:
  - T1: la etiqueta vale la pena si sube spellF1 >= +0,05 frente a B0 con las dos verdades.
  - E1 / E2: candidata si gana a B0 con las dos verdades y en al menos 5 de 7 partidos en cada una.

Salidas (locales, datos abiertos): features en cache/idsse/feat, P(muerto) M+D en cache/idsse/pred,
resultados por partido en dfl/resultados/teacher_s0.csv. Solo se imprimen agregados.
"""
import csv
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
import featlib  # noqa: E402
import make_submission as ms  # noqa: E402
import exp026_metrica_features as mf  # noqa: E402
import dfl_dead  # noqa: E402
from exp026_v5 import oof  # noqa: E402
from dfl_v6 import PITCH, PRED_DIR, decode, degraded_game, logit, sig, to_top_left  # noqa: E402
from metrica_lib import PITCH as M_PITCH, load_game  # noqa: E402

FEAT_DIR = os.path.join(dfllib.CACHE, "feat")
OUT_CSV = os.path.join(HERE, "resultados", "teacher_s0.csv")
SEEDS = (0, 1, 2)
STEP = 5
STYLES = (("control 1 s", 1.0), ("control 2 s", 2.0))
LABELS = np.array(list("HADX"))
FPS = 25
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def feat_path(mid, seed=0):
    return os.path.join(FEAT_DIR, f"{mid}_s{seed}.npz")


def build_features(mid, seed=0):
    """5 familias sin balón + 5 con balón sobre el partido degradado, como dfl_v6.run_match."""
    if os.path.exists(feat_path(mid, seed)):
        return
    g, _ = degraded_game(mid, seed)
    periods = mf.periods_of(g)
    tr = mf.as_tracking(g)
    mf.point_globals_at(tr, periods)
    Xf, nf = mf.free_families(tr)
    Xb, nb = mf.ball_families()
    os.makedirs(FEAT_DIR, exist_ok=True)
    np.savez(feat_path(mid, seed), Xf=Xf.astype(np.float32), Xb=Xb.astype(np.float32),
             nf=np.array(nf), nb=np.array(nb))
    log(f"{mid}: features guardadas {Xf.shape[1]}+{Xb.shape[1]}")


def load_features(mid, seed=0):
    z = np.load(feat_path(mid, seed))
    return z["Xf"], z["Xb"], [str(x) for x in z["nf"]], [str(x) for x in z["nb"]]


def dead_md_all(mids):
    """P(muerto) M+D de cada partido, entrenado sin ese partido (como dfl_redecode_dead). Con caché."""
    paths = {m: os.path.join(PRED_DIR, f"{m}_s0_pdeadMD.npz") for m in mids}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    counts = np.load(dfl_dead.COUNTS)
    met = [dfl_dead.train_set(g.xy, g.team, M_PITCH, g.dead, g.period > 0, counts, gid)
           for gid, g in ((gid, load_game(gid)) for gid in (1, 2, 3))]
    tr = {}
    all_ids = dfllib.match_ids()
    for i, mid in enumerate(all_ids):
        d = dfllib.load(mid)
        xy = to_top_left(dfllib.stacked(d, "players")).astype(np.float32)
        dead = dfllib.stacked(d, "status", 1) == 0
        play = np.ones(len(dead), bool)
        play[len(d["status_1"])] = False
        tr[mid] = dfl_dead.train_set(xy, d["team"], PITCH, dead, play, counts, 100 + i)
    for mid in mids:
        d = dfllib.load(mid)
        _, players = dfllib.degrade(d, 2, seed=all_ids.index(mid))
        X = dfl_dead.feats(to_top_left(players).astype(np.float32), d["team"], PITCH)
        clf, prior = dfl_dead.fit(met + [tr[o] for o in all_ids if o != mid])
        np.savez_compressed(paths[mid], p=clf.predict_proba(X)[:, 1].astype(np.float32), prior=prior)
        log(f"{mid}: P(muerto) M+D listo")
    return paths


def fit_external(Xs, ys, seed, ws=None):
    """GBM con las filas de varios partidos. ws (opcional, e1_v7 / exp-110): pesos por fila, una lista por partido como
    Xs, normalizados a media 1 en el conjunto. Sin ws, exactamente el ajuste de siempre."""
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    if ws is None:
        return clf.fit(np.concatenate(Xs), np.concatenate(ys))
    w = np.concatenate(ws).astype(np.float64)
    return clf.fit(np.concatenate(Xs), np.concatenate(ys), sample_weight=w / w.mean())


def product(p_ball, p_free, prior):
    return sig(logit(p_ball) + logit(p_free) - logit(prior))


def mix(p_a, prior_a, p_b, prior_b):
    """Probabilidad con el prior de A y la evidencia media en log-odds de A y B."""
    return sig(logit(prior_a) + 0.5 * ((logit(p_a) - logit(prior_a)) + (logit(p_b) - logit(prior_b))))


def recalls(gt, pred):
    """Recall a IoU 0,5 con el mismo equipo de las posesiones < 5 s y >= 5 s."""
    runs = dfllib.pm.valid_runs(dfllib.pm.label_codes(gt))
    gs = dfllib.pm.possession_spells(dfllib.pm.label_codes(gt), runs)
    ps = dfllib.pm.possession_spells(dfllib.pm.label_codes(pred), runs)
    short, long_, j = [], [], 0
    for a, b, k in gs:
        while j < len(ps) and ps[j][1] <= a:
            j += 1
        best, m = 0.0, j
        while m < len(ps) and ps[m][0] < b:
            if ps[m][2] == k:
                inter = min(b, ps[m][1]) - max(a, ps[m][0])
                best = max(best, inter / (max(b, ps[m][1]) - min(a, ps[m][0])))
            m += 1
        (short if (b - a) < 5 * FPS else long_).append(best >= 0.5)
    return short, long_


def main():
    mids = dfllib.match_ids()
    assert len(mids) == 7
    for mid in mids:
        build_features(mid)
    log("features listas")
    pd_paths = dead_md_all(mids)
    log("P(muerto) M+D listo")

    # Datos de entrenamiento externos (1 de cada 5 frames vivos con balón visible), por verdad y partido
    gts, tr = {}, {}
    for mid in mids:
        d = dfllib.load(mid)
        z = np.load(os.path.join(PRED_DIR, f"{mid}_s0_probs.npz"))
        vis = z["vis"]
        ball, pl = dfllib.stacked(d, "ball")[:, :2], dfllib.stacked(d, "players").astype(np.float64)
        gt0 = dfllib.gt_labels(d)
        Xf, Xb, _, _ = load_features(mid)
        for sname, t in STYLES:
            gt = dfllib.official_style(gt0, ball, pl, d["team"], t, True)
            gts[(mid, sname)] = gt
            idx = np.flatnonzero(((gt == "H") | (gt == "A")) & vis)[::STEP]
            tr[(mid, sname)] = (Xf[idx], Xb[idx], (gt[idx] == "A").astype(np.int8))
        del Xf, Xb
    log("datos de entrenamiento externos listos")

    rows, res = [], {}
    for mid in mids:
        z = np.load(os.path.join(PRED_DIR, f"{mid}_s0_probs.npz"))
        periods = [tuple(map(int, p)) for p in z["periods"]]
        vis, ball_out, prior = z["vis"], z["ball_out"], float(z["prior"])
        zd = np.load(pd_paths[mid])
        p_dead, prior_d = zd["p"], float(zd["prior"])
        Xf, Xb, _, _ = load_features(mid)
        play = np.zeros(len(vis), bool)
        for a, b in periods:
            play[a:b] = True
        p_vis6 = sig(0.5 * (logit(z["p_ball_b"]) + logit(z["p_ball_c"])) + logit(z["p_free"]) - logit(prior))
        p_free6 = z["p_free"]

        def dec(p_v, p_f, pr):
            old = ms.PRIOR_D
            ms.PRIOR_D = prior_d
            try:
                return LABELS[decode(np.where(vis, p_v, p_f), p_dead, pr, periods, vis, ball_out, ms.V6_TEMP)]
            finally:
                ms.PRIOR_D = old

        for sname, _ in STYLES:
            gt = gts[(mid, sname)]
            alive = (gt == "H") | (gt == "A")
            y = (gt == "A").astype(np.int8)
            train = alive & vis
            pr_t1 = float(y[train].mean())
            pb_t1 = np.mean([oof(Xb, y, train, play, s) for s in SEEDS], 0)
            pf_t1 = np.mean([oof(Xf, y, train, play, s) for s in SEEDS], 0)
            others = [tr[(o, sname)] for o in mids if o != mid]
            yo = [o[2] for o in others]
            pr_e = float(np.concatenate(yo).mean())
            pb_e = np.mean([fit_external([o[1] for o in others], yo, s).predict_proba(Xb)[:, 1] for s in SEEDS], 0)
            pf_e = np.mean([fit_external([o[0] for o in others], yo, s).predict_proba(Xf)[:, 1] for s in SEEDS], 0)
            pv_e = product(pb_e, pf_e, pr_e)
            variants = {
                "B0 v6 + D M+D": (dec(p_vis6, p_free6, prior), p_vis6, p_free6),
                "T1 profesor perfecto": (dec(product(pb_t1, pf_t1, pr_t1), pf_t1, pr_t1), product(pb_t1, pf_t1, pr_t1), pf_t1),
                "E1 externo": (dec(pv_e, pf_e, pr_e), pv_e, pf_e),
                "E2 mezcla": (dec(mix(p_vis6, prior, pv_e, pr_e), mix(p_free6, prior, pf_e, pr_e), prior),
                              mix(p_vis6, prior, pv_e, pr_e), mix(p_free6, prior, pf_e, pr_e)),
            }
            for k, (pred, pv, pf) in variants.items():
                r = dfllib.evaluate(gt, pred)
                s, l = recalls(gt, pred)
                av, ag = alive & vis, alive & ~vis
                auc_v = roc_auc_score(y[av], np.nan_to_num(pv[av], nan=.5))
                auc_g = roc_auc_score(y[ag], np.nan_to_num(pf[ag], nan=.5))
                row = dict(partido=mid, verdad=sname, variante=k, spellF1=r["spell_f1_mean"], macroF1=r["macro_f1"],
                           poses=r["n_pred_spells"] / r["n_gt_spells"], rec_cortas=np.mean(s), rec_largas=np.mean(l),
                           auc_visible=auc_v, auc_hueco=auc_g, n_cortas=len(s), n_largas=len(l))
                rows.append(row)
                res.setdefault((sname, k), []).append(row)
            log(f"{mid} {sname}: " + "  ".join(f"{k.split()[0]} {res[(sname, k)][-1]['spellF1']:.3f}" for k in variants))
        del Xf, Xb

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    names = ["B0 v6 + D M+D", "T1 profesor perfecto", "E1 externo", "E2 mezcla"]
    print(f"\nJuez DFL nivel 2, semilla 0, {len(mids)} partidos")
    print("%-12s %-22s %8s %8s %6s %10s %10s %8s %8s %9s" % (
        "verdad", "variante", "spellF1", "macroF1", "poses", "rec <5 s", "rec >=5 s", "AUC vis", "AUC hueco", "vs B0"))
    for sname, _ in STYLES:
        base = np.array([x["spellF1"] for x in res[(sname, names[0])]])
        for k in names:
            rs = res[(sname, k)]
            v = {c: np.mean([x[c] for x in rs]) for c in rs[0] if c not in ("partido", "verdad", "variante")}
            d = np.array([x["spellF1"] for x in rs]) - base
            wl = "" if k == names[0] else "%+.3f %d/%d" % (d.mean(), (d > 1e-9).sum(), (d < -1e-9).sum())
            print("%-12s %-22s %8.3f %8.3f %6.2f %10.3f %10.3f %8.3f %8.3f  %s" % (
                sname, k, v["spellF1"], v["macroF1"], v["poses"], v["rec_cortas"], v["rec_largas"],
                v["auc_visible"], v["auc_hueco"], wl))


if __name__ == "__main__":
    main()
