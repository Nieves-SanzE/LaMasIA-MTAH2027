"""Features de BAR-ATH que usa make_e1.py (5 familias sin balón + 5 con balón), SIN entrenar nada con el partido.

Las features son las de la rama kura-eda (commit 524becb): `fe_geometry.py` (goalside, shape, convergence, graph_intra,
space) y `fb_ball.py` (ball_pressure, ball_structure, ball_space, ball_graph, ball_motion). Sus scripts, además de
calcular las features, hacen pseudo-etiquetas, cargan la verdad pública para informes y entrenan GBM de cribado con
BAR-ATH. Nada de eso cambia las features que guardan, pero no debe estar en el camino de la entrega. Este script
calcula solo las features, con las mismas funciones y sin modificar el código de Pablo:
  - fe_geometry: se llaman sus funciones de familia, en el mismo orden que su main(), sin labels() ni report().
  - fb_ball: es un script sin main(); se ejecuta su código solo hasta antes del cribado (la línea
    `if "--cached" in sys.argv:`), con featlib.labels sustituida por una función que no carga nada y falla si alguien
    usa su resultado. Después se llama a su compute_families() y se guardan las familias con save_family.
Las partes salen del código de Pablo (common.HALVES); se comprueba que coinciden con games.half_video_blocks.
Escribe en out/explore/features/ de la copia de Pablo. Solo imprime tamaños y tiempos.

    cd <copia de pablo_v6>
    $env:OMP_NUM_THREADS="4"; .venv\\Scripts\\python.exe -u ..\\dfl\\barath_features.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_v6_external as ext  # noqa: E402,F401  (añade las rutas de la copia de Pablo)

import dfllib  # noqa: E402
import featlib  # noqa: E402

T0 = time.time()
FREE = ["goalside", "shape", "convergence", "graph_intra", "space"]
BALL = ["ball_pressure", "ball_structure", "ball_space", "ball_graph", "ball_motion"]
CUT = 'if "--cached" in sys.argv:'


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


class _NoLabels:
    """Resultado de labels() sustituida: cualquier uso falla (las features no deben depender de etiquetas)."""

    def __getitem__(self, k):
        raise RuntimeError(f"las features han pedido etiquetas ({k!r})")

    __getattr__ = __getitem__


def _no_labels(tr):
    return _NoLabels()


def free_families():
    import fe_geometry as fg
    tr = featlib.load_tracking()
    n = tr.n_frames
    team, ok = fg.prepare(tr)
    fams = {}
    fams["goalside"] = fg.fam_goalside(team, ok, n)
    sub = fg.sub_features(team, ok, n)
    fams["shape"] = fg.fam_shape(team, ok, n, sub)
    fams["convergence"] = fg.fam_convergence(team, ok, n)
    fams["graph_intra"] = fg.fam_graph(team, ok, n, sub)
    fams["space"] = fg.fam_space(sub, ok)
    assert list(fams) == FREE
    for fam, (X, nm) in fams.items():
        featlib.save_family(fam, X, nm)
        log(f"{fam}: {X.shape[1]} features")


def ball_families():
    path = os.path.join(os.path.dirname(featlib.__file__), "fb_ball.py")
    src = open(path, encoding="utf-8").read()
    assert src.count(CUT) == 1, "fb_ball.py ha cambiado: revisar el punto de corte"
    old = featlib.labels
    featlib.labels = _no_labels
    try:
        ns = {"__name__": "fb_ball_features", "__file__": path}
        exec(compile(src[:src.index(CUT)], path, "exec"), ns)
    finally:
        featlib.labels = old
    assert ns["FAMILIES"] == BALL
    fams = ns["compute_families"]()
    assert list(fams) == BALL
    for fam, (X, nm) in fams.items():
        featlib.save_family(fam, X, nm)
        log(f"{fam}: {X.shape[1]} features")


def main():
    dfllib.forbid_public_truth()  # la verdad pública no se carga: si algo lo intenta, falla
    halves = dfllib.official_halves()
    assert [tuple(h) for h in featlib.HALVES] == halves, "las partes de la copia de Pablo no son los bloques oficiales"
    free_families()
    ball_families()
    log(f"hecho -> {featlib.FEAT_DIR}")


if __name__ == "__main__":
    main()
