"""Entrega candidata: v6 de Pablo con el P(muerto) entrenado con Metrica + DFL (dfl_dead.py, modelo M+D).

    cd pablo_v6
    set OMP_NUM_THREADS=4
    .venv\\Scripts\\python.exe ..\\dfl\\make_v6_deaddfl.py

Un solo cambio respecto a la v6: los datos de entrenamiento del modelo de balón parado (3 partidos
de Metrica + 7 de la DFL, cada uno limpio y recortado tipo cámara con los jugadores visibles de
BAR-ATH). Mismas features (deadlib, ventanas <= 8 s, slots sin velocidad fuera, dedupe), mismo GBM
y mismo decodificador; el prior de D es la proporción de D del conjunto de entrenamiento, como
PRIOR_D lo es para Metrica.

1. make_submission.build("v6") de Pablo (sin modificar). Se guardan sus entradas al decodificador en
   MTAH_DATA_DIR\\cache\\v6_build.npz para no repetirlo (~30 min).
2. P(muerto) M+D sobre las mismas features de BAR-ATH que usa la v6.
3. Decodificación igual que make_submission.decode y escritura con '\\n' (mismos bytes en cualquier SO).
Salidas: evaluation/results/v6_deadMD.csv del clon oficial (y v6_rebuild.csv como control: debe ser
la v6 de siempre). Solo se imprimen agregados.
"""
import hashlib
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
import make_submission as ms  # noqa: E402
from deadlib import dead_features, dedupe, mask_no_velocity  # noqa: E402
from mtah_2027 import atd  # noqa: E402
import dfl_dead  # noqa: E402

BUILD = os.path.join(dfllib.DATA, "cache", "v6_build.npz")
RESULTS = os.path.join(os.path.dirname(HERE), "more-than-a-hack-2027", "evaluation", "results")
LABELS = np.array(list("HADX"))


def build_cached():
    keys = ("p_vis", "p_free", "p_dead", "ball_out", "bv")
    if os.path.exists(BUILD):
        z = np.load(BUILD)
        m = {k: z[k] for k in keys}
        m.update(prior=float(z["prior"]), temp=float(z["temp"]),
                 periods=[tuple(map(int, p)) for p in z["periods"]])
        return m
    b = ms.build("v6")
    m = {k: b[k] for k in ("p_vis", "p_free", "p_dead", "ball_out")}
    m.update(bv=b["ctx"].bv, prior=float(b["prior"]), temp=float(b["temp"]), periods=b["periods"])
    np.savez_compressed(BUILD, **{k: m[k] for k in keys}, prior=m["prior"], temp=m["temp"],
                        periods=np.array(m["periods"]))
    return m


def dead_model_md():
    """Modelo M+D de dfl_dead.py entrenado con TODOS los partidos (Metrica 3 + DFL 7)."""
    counts = np.load(dfl_dead.COUNTS)
    from metrica_lib import PITCH as M_PITCH, load_game
    from dfl_v6 import PITCH as D_PITCH, to_top_left
    parts = []
    for gid in (1, 2, 3):
        g = load_game(gid)
        parts.append(dfl_dead.train_set(g.xy, g.team, M_PITCH, g.dead, g.period > 0, counts, gid))
    for i, mid in enumerate(dfllib.match_ids()):
        d = dfllib.load(mid)
        xy = to_top_left(dfllib.stacked(d, "players")).astype(np.float32)
        dead = dfllib.stacked(d, "status", 1) == 0
        play = np.ones(len(dead), bool)
        play[len(d["status_1"])] = False
        parts.append(dfl_dead.train_set(xy, d["team"], D_PITCH, dead, play, counts, 100 + i))
    return dfl_dead.fit(parts)


def write(name, pred):
    path = os.path.join(RESULTS, name + ".csv")
    with open(path, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{LABELS[c]}\n" for i, c in enumerate(pred))
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    share = " ".join(f"{l} {np.mean(LABELS[pred] == l):.3f}" for l in "HADX")
    print(f"{name}: sha256 {sha[:16]}..., {share}")


def decode_with(m, p_dead, prior_d):
    old = ms.PRIOR_D
    ms.PRIOR_D = prior_d  # make_submission.decode lee PRIOR_D del módulo
    try:
        return ms.decode(dict(m, p_dead=p_dead), m["bv"])
    finally:
        ms.PRIOR_D = old


def main():
    m = build_cached()
    print(f"v6 build: temperatura {m['temp']}, prior equipo {m['prior']:.3f}")
    write("v6_rebuild", ms.decode(m, m["bv"]))  # control: la v6 de siempre

    tr = featlib.load_tracking()
    play_official = featlib.play_mask(tr.n_frames)
    team = tr.equipo.astype(np.int8)
    xy_d = dedupe(mask_no_velocity(tr.metros(tr.xy), tr.velocidad), team)
    X = dead_features(xy_d, team, atd.CAMPO, windows_s=ms.V6_WINDOWS_S).values
    clf, prior_d = dead_model_md()
    p_dead = clf.predict_proba(X)[:, 1]
    print(f"P(muerto) M+D: prior de entrenamiento {prior_d:.3f} (Metrica sola {ms.PRIOR_D}); "
          f"media en lo jugado {p_dead[play_official].mean():.3f} (v6: {m['p_dead'][play_official].mean():.3f})")
    pred = decode_with(m, p_dead, prior_d)
    write("v6_deadMD", pred)
    play = featlib.play_mask(len(pred))
    base = ms.decode(m, m["bv"])
    print(f"frames jugados distintos frente a la v6: {np.mean((pred != base)[play]):.2%}; "
          f"D en lo jugado: v6 {np.mean(base[play] == 2):.1%}, nueva {np.mean(pred[play] == 2):.1%}")


if __name__ == "__main__":
    main()
