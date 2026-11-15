"""CSV de BAR-ATH con el modelo de equipo entrenado con etiquetas reales de la DFL (E1 y E2 de dfl_teacher.py).

    cd pablo_v6
    set OMP_NUM_THREADS=4
    .venv\\Scripts\\python.exe -u ..\\dfl\\make_v6_external.py

Parte de v6_deadMD (make_v6_deaddfl.py: v6 + D Metrica+DFL) y solo cambia P(equipo):
  E1 = P_ball y P_free entrenados con los 7 partidos de la DFL degradados y etiquetas reales;
  E2 = evidencia media en log-odds de la v6 y de E1.
Una versión por verdad de entrenamiento (control 1 s y 2 s). Features de BAR-ATH: las mismas 5 + 5 familias
ya calculadas por el pipeline de la v6 (con la limpieza de porteros), alineadas por nombre. La verdad pública
no se usa aquí para nada. Escribe en evaluation/results/ del clon oficial con '\\n'. Solo agregados.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PABLO = os.path.join(os.path.dirname(HERE), "pablo_v6")
for p in (os.path.join(PABLO, "scripts"), os.path.join(PABLO, "scripts", "explore"),
          os.path.join(PABLO, "src"), HERE):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402

import dfllib  # noqa: E402
import featlib  # noqa: E402
import gklib  # noqa: E402
import make_submission as ms  # noqa: E402
import make_v6_deaddfl as base  # noqa: E402
from deadlib import dead_features, dedupe, mask_no_velocity  # noqa: E402
from mtah_2027 import atd  # noqa: E402
from dfl_v6 import PRED_DIR  # noqa: E402
from dfl_teacher import SEEDS, STEP, STYLES, fit_external, load_features, mix, product  # noqa: E402

FREE5 = ["goalside", "shape", "convergence", "graph_intra", "space"]
PD_CACHE = os.path.join(dfllib.DATA, "cache", "v6_pdeadMD_blocks.npz")  # recuentos de los bloques oficiales (exp-092)


def aligned(X, names, wanted):
    """Reordena las columnas de X según `wanted` (nombres de la DFL); exige coincidencia exacta."""
    pos = {n: i for i, n in enumerate(names)}
    assert set(pos) == set(wanted), (sorted(set(pos) ^ set(wanted))[:5])
    return X[:, [pos[n] for n in wanted]]


def dead_md(tr):
    """P(muerto) M+D de BAR-ATH y su prior (con caché)."""
    if os.path.exists(PD_CACHE):
        z = np.load(PD_CACHE)
        return z["p"], float(z["prior"])
    team = tr.equipo.astype(np.int8)
    xy_d = dedupe(mask_no_velocity(tr.metros(tr.xy), tr.velocidad), team)
    X = dead_features(xy_d, team, atd.CAMPO, windows_s=ms.V6_WINDOWS_S).values
    clf, prior_d = base.dead_model_md()
    p_dead = clf.predict_proba(X)[:, 1]
    np.savez_compressed(PD_CACHE, p=p_dead.astype(np.float32), prior=prior_d)
    return p_dead, prior_d


def barath_features(tr, periods):
    """Features de equipo de BAR-ATH (con la limpieza de porteros), alineadas por nombre con las de la DFL."""
    X_free, nf_bar = featlib.load_families(FREE5)
    nf_bar = [n.split(":", 1)[1] for n in nf_bar]
    X_ball, nb_bar = featlib.load_families(ms.BALL)
    tr_c, changed, stats = gklib.clean_goalkeepers(tr, periods)
    X_ball = gklib.patch_ball_features(tr_c, X_ball, changed)
    _, _, nf, nb = load_features(dfllib.match_ids()[0])
    return aligned(X_free, nf_bar, nf), aligned(X_ball, nb_bar, nb)


def external_probs(X_free, X_ball, t):
    """P_ball, P_free y prior del modelo entrenado con los 7 partidos de la DFL (verdad: control t s)."""
    Xfs, Xbs, ys = [], [], []
    for mid in dfllib.match_ids():
        d = dfllib.load(mid)
        z = np.load(os.path.join(PRED_DIR, f"{mid}_s0_probs.npz"))
        ball, pl = dfllib.stacked(d, "ball")[:, :2], dfllib.stacked(d, "players").astype(np.float64)
        gt = dfllib.official_style(dfllib.gt_labels(d), ball, pl, d["team"], t, True)
        idx = np.flatnonzero(((gt == "H") | (gt == "A")) & z["vis"])[::STEP]
        Xf, Xb, _, _ = load_features(mid)
        Xfs.append(Xf[idx])
        Xbs.append(Xb[idx])
        ys.append((gt[idx] == "A").astype(np.int8))
        del Xf, Xb
    pr_e = float(np.concatenate(ys).mean())
    pb_e = np.mean([fit_external(Xbs, ys, s).predict_proba(X_ball)[:, 1] for s in SEEDS], 0)
    pf_e = np.mean([fit_external(Xfs, ys, s).predict_proba(X_free)[:, 1] for s in SEEDS], 0)
    return pb_e, pf_e, pr_e


def external_variants(m, pb_e, pf_e, pr_e, tag):
    """Entradas al decodificador de E1 y E2 (lo que cambia respecto a la v6)."""
    pv_e = product(pb_e, pf_e, pr_e)
    return {
        f"ext_E1_{tag}": dict(p_vis=pv_e, p_free=pf_e, prior=pr_e),
        f"ext_E2_{tag}": dict(p_vis=mix(m["p_vis"], m["prior"], pv_e, pr_e),
                              p_free=mix(m["p_free"], m["prior"], pf_e, pr_e), prior=m["prior"]),
    }


def main():
    m = base.build_cached()
    print(f"v6 build: temperatura {m['temp']}, prior {m['prior']:.3f}", flush=True)
    periods = m["periods"]
    featlib.PLAY, featlib.HALVES = periods, periods
    tr = featlib.load_tracking()

    p_dead, prior_d = dead_md(tr)
    print(f"P(muerto) M+D listo (prior {prior_d:.3f})", flush=True)

    X_free, X_ball = barath_features(tr, periods)
    print(f"features de BAR-ATH alineadas: {X_free.shape[1]} + {X_ball.shape[1]}", flush=True)

    for sname, t in STYLES:
        tag = "c1" if t == 1.0 else "c2"
        variants = external_variants(m, *external_probs(X_free, X_ball, t), tag)
        reference = base.decode_with(m, p_dead, prior_d)
        play = featlib.play_mask(len(reference))
        for name, ov in variants.items():
            pred = base.decode_with(dict(m, **ov), p_dead, prior_d)
            base.write(name, pred)
            print(f"  {name}: frames jugados distintos de v6_deadMD {np.mean((pred != reference)[play]):.1%}; "
                  f"D {np.mean(pred[play] == 2):.1%}", flush=True)


if __name__ == "__main__":
    main()
