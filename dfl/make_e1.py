"""E1 según las normas del correo del 03/11: todo lo que se entrena o se ajusta sale de datos externos.

    cd pablo_v6
    set OMP_NUM_THREADS=4
    .venv\\Scripts\\python.exe -u ..\\dfl\\make_e1.py build <partido>     # features de entrenamiento de la DFL
    .venv\\Scripts\\python.exe -u ..\\dfl\\make_e1.py                     # CSV de BAR-ATH

    .venv\\Scripts\\python.exe -u ..\\dfl\\make_e1.py v10                 # otra versión (v2 ... v13)

Diferencias con make_v6_external.py (ext_E1_c1):
  - No ejecuta el build de la v6, que entrena con pseudo-etiquetas de BAR-ATH. Solo toma de BAR-ATH sus entradas:
    tracking, visibilidad del balón y las features ya calculadas (y, hasta e1_v8, el balón fuera del campo).
  - Partes = bloques puntuados oficiales (games.half_video_blocks) en vez de la detección con el pitido del vídeo.
  - Lo demás (modelo de huecos, ventanas de D, limpieza de porteros, temperatura...) sale de CONFIG, decidido en datos
    externos por dfl_e1_judge.py y los jueces siguientes.
Entrega: CONFIG = e1_v14s (15/11): pesos por duración de la posesión (e1_v7), huecos del balón medidos en SkillCorner
(e1_v8), sin la regla de balón fuera (e1_v9), D hasta el saque inicial de la Smart Tagging (e1_v10), jugadores visibles
del entrenamiento de D medidos en SkillCorner (e1_v12), modelo de D entrenado solo con la DFL, sin Metrica (e1_v13), red
temporal mezclada con los árboles (e1_v14) y árboles entrenados también con cada partido con los equipos intercambiados
(e1_v14s). La red se lee de prod.npz, que `tcn_train.py prod_pesos` calcula con los pesos guardados (pesos_red/).
`make_e1.py v13` regenera la entrega anterior (sin red). `make_e1.py v10` necesita Metrica y barath_counts.py, que ya
no están en el camino.
Escribe evaluation/results/<nombre>.csv del clon oficial con '\\n'. La verdad está bloqueada (forbid_public_truth).
Solo agregados.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_v6_external as ext  # noqa: E402  (añade las rutas de la v6)

import numpy as np  # noqa: E402

import dfllib  # noqa: E402
import featlib  # noqa: E402
import gklib  # noqa: E402
import make_submission as ms  # noqa: E402
import make_v6_deaddfl as base  # noqa: E402
import exp026_metrica_features as mf  # noqa: E402
import dfl_dead  # noqa: E402
from deadlib import dead_features, dedupe, mask_no_velocity  # noqa: E402
from dfl_teacher import SEEDS, STEP, fit_external, mix, product  # noqa: E402
from dfl_v6 import PITCH, degraded_game, to_top_left  # noqa: E402
from metrica_lib import PITCH as M_PITCH, load_game  # noqa: E402
from metrica_lib import Game  # noqa: E402  (e1_v13s / e1_v14s: swap_game)
from mtah_2027 import atd  # noqa: E402

FEAT_DIR = os.path.join(dfllib.CACHE, "feat_v2")  # features de entrenamiento de la DFL (lo cambia use_feat_dir)
# Elegido por dfl_e1_judge.py (03/11) solo con datos externos: ventanas 8 s, porteros limpios, T 0,4.
CONFIG_V2 = dict(name="e1_v2", t_ctrl=1.0, windows=ms.V6_WINDOWS_S, gk_clean=True, temp=0.4, metrica=False,
                 counts=os.path.join(dfllib.DATA, "cache", "barath_counts_v6.npy"), feat="feat_v2")
# e1_v3 (04/11, exp-085): igual que e1_v2, pero los recuentos de jugadores visibles de BAR-ATH (abandono tipo cámara
# del entrenamiento de D) salen de los bloques oficiales y no de la constante 113.742 (barath_counts.py).
CONFIG_V3 = dict(CONFIG_V2, name="e1_v3", counts=dfllib.COUNTS_BLOCKS, d_params=None)
# e1_v4 (04/11, exp-086): e1_v3 con los parámetros del D en el decodificador reajustados en la DFL l3s0 y confirmados en
# Metrica (dfl_dparams_judge.py). Solo cambia la decodificación final (el modelo de huecos sigue con los D_PARAMS v6).
CONFIG_V4 = dict(CONFIG_V3, name="e1_v4",
                 d_params=dict(temp_d=0.2, min_dead_s=4.0, stops_per_min=0.5, mean_dead_s=15.0, bias=0.0), ctx=False)
# e1_v5 (04/11, exp-087): e1_v4 con contexto temporal en P_free: sus features más f(t+2 s)-f(t) y f(t)-f(t-2 s)
# dentro de cada parte (dfl_ctx_judge.py). P_ball no cambia.
CONFIG_V5 = dict(CONFIG_V4, name="e1_v5", ctx=True)
# e1_v6 (04/11, exp-092): e1_v5 sin ningún resto de la constante 113.742. El modelo de huecos de BAR-ATH (con el que se
# degrada la DFL de entrenamiento) toma el estado D de un modelo entrenado con los recuentos de los bloques oficiales,
# y las features de entrenamiento de la DFL se rehacen con él (feat_v3). Lo demás, igual que e1_v5.
CONFIG_V6 = dict(CONFIG_V5, name="e1_v6", feat="feat_v3")
# e1_v7 (09/11, exp-110 a 114): e1_v6 con cada fila de entrenamiento de P_ball y P_free pesada por 1 / max(duración de
# su posesión en la verdad de entrenamiento, 2 s) (pesos a media 1, prior ponderado) y la temperatura de la regla de
# igual ritmo de exp-111: el mismo total de posesiones en la DFL l3s0 que e1_v6 con T 0,4 (bisección en [0,15; 0,40];
# valor tal cual salió, sin redondear). Todo decidido en la DFL y Metrica; nada de BAR-ATH.
CONFIG_V7 = dict(CONFIG_V6, name="e1_v7", w_spell=2.0, temp=0.23471679687500002)
# e1_v8 (10/11, exp-119): e1_v7 con la DFL de entrenamiento degradada con el modelo de huecos del balón medido en
# SkillCorner open data (skillcorner_gap_model.py) en vez de en BAR-ATH; features en feat_sc. T de la misma regla de
# igual ritmo (objetivo: el total de posesiones de e1_v7 en la DFL l3s0), tal cual salió de la bisección.
CONFIG_V8 = dict(CONFIG_V7, name="e1_v8", feat="feat_sc", temp=0.23544921875000002,
                 gap_model=os.path.join(dfllib.DATA, "cache", "skillcorner_gap_model.json"))
# e1_v9 (11/11, exp-121, candidata): e1_v8 sin la regla de «balón fuera» del decodificador (auditoría H2: su forma se
# decidió en parte con el arnés de la ventana pública). Solo cambia la decodificación: los modelos son los de e1_v8.
CONFIG_V9 = dict(CONFIG_V8, name="e1_v9", ball_out=False)
# e1_v10 (11/11, exp-122, candidata): e1_v9 con D desde el inicio de cada bloque puntuado hasta el inicio de la fase
# «Kick Off (Start)» de esa parte en la Smart Tagging (entrada permitida). El FAQ oficial avisa de que la 2ª parte se
# puntúa desde antes del saque y que ese tramo es todo D (el balón no está en juego). Ningún frame escrito a mano y
# ningún parámetro ajustado: se usa el inicio de la fase tal cual (empieza unos segundos antes del golpeo).
CONFIG_V10 = dict(CONFIG_V9, name="e1_v10", kickoff_tag=True)
# e1_v11 (12/11, exp-126, candidata): e1_v10, pero la 2ª parte se decodifica desde el inicio del bloque con D obligado
# (force_dead de hmm3.viterbi3) hasta el inicio de la fase «Kick Off (Start)», en vez de cortar el bloque ahí. El
# decodificador llega a la fase ya dentro del parón y sale cuando P(muerto) baja (e1_v10 arrancaba en frío a ~3 s del
# golpeo, donde no cabe el parón mínimo de 4 s, y abría una posesión). Mismas entradas y modelos que e1_v10.
CONFIG_V11 = dict(CONFIG_V10, name="e1_v11", kickoff_force=True)
# e1_v12 (12/11, exp-129, candidata): e1_v10 con los jugadores visibles del entrenamiento de D (deadlib.truncate_like)
# medidos en SkillCorner open data, solo en los frames con >= 1 jugador detectado (cámara que encuadra el juego; equivale
# a los frames con proyección sobre el campo), en vez de en BAR-ATH (G2). skillcorner_counts.py + dfl_g2b_confirm.py.
CONFIG_V12 = dict(CONFIG_V10, name="e1_v12", counts=os.path.join(dfllib.DATA, "cache", "skillcorner_counts_det.npy"))
# e1_v13 (12/11, exp-135, candidata): e1_v12 con el modelo de balón parado entrenado solo con la DFL (sin Metrica sample
# data, que no declara licencia; la organización no dio permiso). Elegido por la regla de M1 frente a DFL + SkillCorner.
CONFIG_V13 = dict(CONFIG_V12, name="e1_v13", dead_metrica=False)
# e1_v14 (14/11, exp-139, candidata): e1_v13 con una red temporal (TCN, tcnlib.py) sobre las mismas features, entrenada
# solo con la DFL (tcn_export.py + tcn_train.py prod, en dfl/.venv-tcn) y mezclada con los árboles en log-odds alrededor
# del prior (dfl_teacher.mix) en p_vis y p_free; T de la regla de igual ritmo en los jueces DFL suave + duro (bisección
# de dfl_tcn_judge.py, ejecución 2 determinista, valor tal cual). Lee los logits de BAR-ATH ya calculados por la red
# (fichero `tcn`, de tcn_train.py prod, que predice con la red cargada de sus pesos guardados).
CONFIG_V14 = dict(CONFIG_V13, name="e1_v14", temp=0.33291015625,
                  tcn=os.path.join(dfllib.DATA, "cache", "tcn", "out", "prod.npz"))
# B2 (14/11, exp-138, en juicio): e1_v14 con las features de entrenamiento de la DFL rehechas ocultando jugadores con las
# oclusiones medidas en SkillCorner open data (skillcorner_camera.py: frac_occluded y occlusion_mean_s, tal cual) en vez
# de 0,07 / 4 s; features en feat_sc_occ. La red y la T de esta versión saldrían de su propio entrenamiento y juez
# (tcn: prod_b2.npz, que aún no existe; temp: la de E5 hasta que el juez dé la suya).
OCC_SKILLCORNER = (0.0274603332279767, 1.1453182674199625)
CONFIG_B2 = dict(CONFIG_V14, name="e1_b2", feat="feat_sc_occ", occ=OCC_SKILLCORNER,
                 tcn=os.path.join(dfllib.DATA, "cache", "tcn", "out", "prod_b2.npz"))
# exp-142 (mejora 4): segunda copia degradada de la DFL de entrenamiento (semilla de degradación 1: 1000 + índice del
# partido, como dfl_v6.degraded_game); solo para las secuencias de la red, no es una versión que se entregue.
CONFIG_B2C1 = dict(CONFIG_B2, name="e1_b2c1", feat="feat_sc_occ_c1", deg_seed=1)
# La entrega (CONFIG) se fija más abajo, después de CONFIG_V14S.
VERSIONS = {"v2": CONFIG_V2, "v3": CONFIG_V3, "v4": CONFIG_V4, "v5": CONFIG_V5, "v6": CONFIG_V6, "v7": CONFIG_V7,
            "v8": CONFIG_V8, "v9": CONFIG_V9, "v10": CONFIG_V10, "v11": CONFIG_V11, "v12": CONFIG_V12, "v13": CONFIG_V13,
            "v14": CONFIG_V14, "b2": CONFIG_B2, "b2c1": CONFIG_B2C1}
# B2b (14/11, exp-144, en juicio): e1_v14 con la pérdida de jugadores de la DFL de entrenamiento medida solo en SkillCorner
# open data: ventana de cámara (skillcorner_camera.json, misma estructura que el juez duro) + oclusiones de SkillCorner,
# en vez de 0,07 / 4 s; features en feat_sc_cam. Red y T: las de su propio entrenamiento y juez (prod_b2b.npz aún no
# existe; temp: la de E5 hasta que el juez dé la suya).
CONFIG_B2B = dict(CONFIG_V14, name="e1_b2b", feat="feat_sc_cam", occ=OCC_SKILLCORNER, camera=True,
                  tcn=os.path.join(dfllib.DATA, "cache", "tcn", "out", "prod_b2b.npz"))
VERSIONS["b2b"] = CONFIG_B2B
# exp-146 (mejora 4 sobre e1_v14): segunda copia degradada de la DFL de entrenamiento con la degradación de e1_v14 (semilla
# de degradación 1: 1000 + índice del partido); features en feat_sc_c1. Solo para las secuencias de la red; no se entrega.
CONFIG_V14C1 = dict(CONFIG_V14, name="e1_v14c1", feat="feat_sc_c1", deg_seed=1)
VERSIONS["v14c1"] = CONFIG_V14C1
# exp-149 (P4): copia de la DFL de entrenamiento con los equipos intercambiados (local <-> visitante) antes de calcular las
# features (la orientación del campo sale de los datos, así que las features se recalculan de forma exacta); misma
# degradación que e1_v14 (semilla 0). Features en feat_sc_swap (clave swap_teams; «swap» es la de e1_v13s de Pablo). Solo para las secuencias de la red; no se entrega.
CONFIG_V14SW = dict(CONFIG_V14, name="e1_v14sw", feat="feat_sc_swap", swap_teams=True)
VERSIONS["v14sw"] = CONFIG_V14SW
# e1_v13s / e1_v14s (14/11, exp-148 / exp-150): candidatas de Pablo, no adoptadas. Intercambio de equipos en el
# entrenamiento de los árboles (P_ball y P_free): cada partido de la DFL entra dos veces, tal cual y con local y visitante
# intercambiados (etiqueta invertida, mismos pesos por posesión). Las features de la copia las calcula build() sobre el
# mismo partido degradado con `team` invertido (degradar primero e intercambiar después; la degradación no lee `team`):
# `make_e1.py build <partido> swap`, en SWAP_FEAT. Las copias van detrás de los 7 partidos y en su orden. Todo decidido
# con datos externos (juez DFL suave / duro dejando un partido fuera); nada de BAR-ATH.
CONFIG_V13S = dict(CONFIG_V13, name="e1_v13s", swap=True)
# e1_v14s: e1_v14 (la red y su mezcla, sin cambios) con los árboles de e1_v13s. T de la regla de igual ritmo frente a
# e1_v14 en los jueces DFL suave + duro (exp-150, xswap_v14_judge.py, valor tal cual; redes por pliegue entrenadas en
# la máquina de Pablo).
CONFIG_V14S = dict(CONFIG_V14, name="e1_v14s", swap=True, temp=0.3216796875)
VERSIONS["v13s"] = CONFIG_V13S
VERSIONS["v14s"] = CONFIG_V14S
# La entrega (15/11, Nieves): e1_v14s, elegida por la regla fijada antes de medir (los pesos reproducen en Linux y pasa la
# RV2 estricta). Antes CONFIG_V13 (14/11), CONFIG_V10 (11/11) y CONFIG_V7 (hasta el 11/11).
CONFIG = CONFIG_V14S
SWAP_FEAT = "feat_swap_sc"  # e1_v13s / e1_v14s: features de las copias intercambiadas (degradación de feat_sc)
KICKOFF_MAX_S = 600  # salvaguarda a priori: si la fase de saque cae a más de 10 min del inicio del bloque, no se usa


def kickoff_tags(halves):
    """e1_v10: [(inicio del bloque, inicio de la fase Kick Off (Start) de esa parte)] para forzar D entre medias.

    Lee la Smart Tagging con el cargador oficial (starter/load_smart_tagging.load, ruta por games.require). La parte de
    cada fase sale de su etiqueta Half ("1st Half" / "2nd Half"); si hay varias, la primera. Solo se fuerza si la fase
    empieza dentro del bloque y a menos de KICKOFF_MAX_S de su inicio; si no, ese bloque se decodifica entero, como
    en e1_v9. No se escribe ningún frame del partido.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "load_smart_tagging_oficial", os.path.join(dfllib.OFFICIAL, "starter", "load_smart_tagging.py"))
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    events = [e for e in st.load()
              if (e["code"] or "").strip().lower() == st.KICKOFF_CODE
              and (e["tags"].get("Type") or "").strip().lower() == st.KICKOFF_TYPE]
    out = []
    for i, (a, b) in enumerate(halves):
        name = ("1st Half", "2nd Half")[i]
        starts = sorted(e["start_frame"] for e in events if e["tags"].get("Half") == name)
        assert len(starts) == 1, f"{name}: {len(starts)} fases Kick Off (Start) (se espera exactamente una)"
        k = starts[0] if starts else a
        if a < k < b and k - a <= KICKOFF_MAX_S * featlib.FPS:
            out.append((a, int(k)))
        print(f"{name}: {len(starts)} fase(s) Kick Off (Start); D forzado al inicio del bloque "
              f"{(k - a) / featlib.FPS if out and out[-1][0] == a else 0:.1f} s", flush=True)
    return out
GAP_MODEL_BARATH = dfllib.GAP_MODEL


def use_feat_dir(cfg):
    """Carpeta de las features de entrenamiento de la DFL de la configuración (feat_v2 hasta e1_v5, feat_v3 en e1_v6) y
    modelo de huecos con el que build() degrada la DFL (el de BAR-ATH salvo que la configuración diga otro, e1_v8)."""
    global FEAT_DIR, DEG_SEED, SWAP
    SWAP = bool(cfg.get("swap_teams", False))  # exp-149 (P4): equipos intercambiados en build()
    FEAT_DIR = os.path.join(dfllib.CACHE, cfg.get("feat", "feat_v2"))
    dfllib.GAP_MODEL = cfg.get("gap_model", GAP_MODEL_BARATH)
    # exp-138 (B2): oclusiones de jugadores de la degradación (None = 0,07 / 4 s) y semilla de degradación de build()
    dfllib.OCCLUSION = cfg.get("occ")
    DEG_SEED = cfg.get("deg_seed", 0)
    # exp-144 (B2b): ventana de cámara de SkillCorner en la degradación (None = sin ventana)
    dfllib.CAMERA = None
    if cfg.get("camera"):
        with open(SKILLCORNER_CAMERA, encoding="utf-8") as f:
            dfllib.CAMERA = json.load(f)
    if dfllib.OCCLUSION is not None and os.path.exists(SKILLCORNER_CAMERA):
        with open(SKILLCORNER_CAMERA, encoding="utf-8") as f:
            cam = json.load(f)
        assert (cam["frac_occluded"], cam["occlusion_mean_s"]) == tuple(dfllib.OCCLUSION), \
            "las oclusiones de la configuración no son las medidas en SkillCorner"


SKILLCORNER_CAMERA = os.path.join(dfllib.DATA, "cache", "skillcorner_camera.json")
DEG_SEED = 0
SWAP = False


def use_swap_dir():
    """e1_v13s / e1_v14s: FEAT_DIR pasa a SWAP_FEAT. Se llama después de use_feat_dir de esas versiones: las copias se
    degradan como feat_sc (modelo de huecos de SkillCorner, semilla 0, oclusiones 0,07 / 4 s, sin ventana de cámara)."""
    global FEAT_DIR
    assert os.path.basename(FEAT_DIR) == "feat_sc" and dfllib.GAP_MODEL != GAP_MODEL_BARATH and DEG_SEED == 0 \
        and dfllib.OCCLUSION is None and dfllib.CAMERA is None, "SWAP_FEAT solo vale con la degradación de feat_sc"
    FEAT_DIR = os.path.join(dfllib.CACHE, SWAP_FEAT)


def feat_path(mid):
    return os.path.join(FEAT_DIR, mid + ".npz")


def swap_game(g):
    """e1_v13s: el mismo partido con los equipos intercambiados (team 0 <-> 1); posiciones, balón y partes intactos."""
    assert set(np.unique(g.team)) <= {0, 1}, "team solo puede tener 0 y 1"
    return Game(g.gid, g.xy, (1 - g.team).astype(np.int8), g.ball, g.period, g.events)


class _NoLabels:
    """Resultado de featlib.labels() sustituida en build(): cualquier uso falla (las features no usan etiquetas)."""

    def __getitem__(self, k):
        raise RuntimeError(f"las features han pedido etiquetas ({k!r})")

    __getattr__ = __getitem__


def _no_labels(tr):
    return _NoLabels()


def build(mid, swap=False):
    """Features de un partido de la DFL degradado (nivel 2, semilla 0, modelo de huecos nuevo) y su máscara de balón.

    Auditoría H1 (11/11): mf.ball_families() ejecuta fb_ball.py hasta el corte, y su línea `lab = labels(tr)` llamaba a
    featlib.labels, que lee la verdad pública (common.gt_codes) aunque las features no la usan. Ahora la verdad queda
    bloqueada (forbid_public_truth: cualquier apertura de un fichero de verdad hace fallar el script) y labels se
    sustituye, como en barath_features.py, por una función que no lee nada y cuyo resultado falla si se usa.
    swap=True (e1_v13s, exp-148): el mismo partido degradado con los equipos intercambiados después de degradar
    (swap_game), en SWAP_FEAT. Mismo código de features.
    """
    if swap:
        use_swap_dir()
    if os.path.exists(feat_path(mid)):
        return
    dfllib.forbid_public_truth()
    old_labels = featlib.labels
    featlib.labels = _no_labels
    try:
        g, _ = degraded_game(mid, DEG_SEED)
        if SWAP:  # exp-149 (P4)
            assert not swap, "swap_teams y swap a la vez"
            g.team = (1 - g.team).astype(np.int8)
        if swap:
            g = swap_game(g)
        periods = mf.periods_of(g)
        tr = mf.as_tracking(g)
        mf.point_globals_at(tr, periods)
        Xf, nf = mf.free_families(tr)
        Xb, nb = mf.ball_families()
    finally:
        featlib.labels = old_labels
    os.makedirs(FEAT_DIR, exist_ok=True)
    np.savez(feat_path(mid), Xf=Xf.astype(np.float32), Xb=Xb.astype(np.float32), nf=np.array(nf), nb=np.array(nb),
             vis=np.isfinite(g.ball[:, 0]))
    print(f"{mid}: features {Xf.shape[1]}+{Xb.shape[1]}, balón visible {np.isfinite(g.ball[:, 0]).mean():.1%}", flush=True)


def load_train(mid):
    z = np.load(feat_path(mid))
    return z["Xf"], z["Xb"], [str(x) for x in z["nf"]], [str(x) for x in z["nb"]], z["vis"]


def dead_train_sets(windows, counts_path=dfl_dead.COUNTS, metrica=True):
    """Conjuntos de entrenamiento de D (Metrica 3 + DFL 7) con las ventanas dadas, como dfl_dead.train_set.
    metrica=False (e1_v13, exp-135): no se carga Metrica sample data (sin licencia declarada); solo la DFL."""
    old = dfl_dead.W
    dfl_dead.W = windows
    try:
        counts = np.load(counts_path)
        met = {gid: dfl_dead.train_set(g.xy, g.team, M_PITCH, g.dead, g.period > 0, counts, gid)
               for gid, g in ((gid, load_game(gid)) for gid in (1, 2, 3))} if metrica else {}
        dfl = {}
        for i, mid in enumerate(dfllib.match_ids()):
            d = dfllib.load(mid)
            xy = to_top_left(dfllib.stacked(d, "players")).astype(np.float32)
            dead = dfllib.stacked(d, "status", 1) == 0
            play = np.ones(len(dead), bool)
            play[len(d["status_1"])] = False
            dfl[mid] = dfl_dead.train_set(xy, d["team"], PITCH, dead, play, counts, 100 + i)
    finally:
        dfl_dead.W = old
    return met, dfl


def team_training(mids, t_ctrl, extra=()):
    """Frames de entrenamiento de E1: 1 de cada 5 frames vivos con balón visible de cada partido de la DFL."""
    Xfs, Xbs, ys = [], [], []
    for mid in mids:
        d = dfllib.load(mid)
        ball, pl = dfllib.stacked(d, "ball")[:, :2], dfllib.stacked(d, "players").astype(np.float64)
        gt = dfllib.official_style(dfllib.gt_labels(d), ball, pl, d["team"], t_ctrl, True)
        Xf, Xb, _, _, vis = load_train(mid)
        idx = np.flatnonzero(((gt == "H") | (gt == "A")) & vis)[::STEP]
        Xfs.append(Xf[idx])
        Xbs.append(Xb[idx])
        ys.append((gt[idx] == "A").astype(np.int8))
    for Xf, Xb, y in extra:
        Xfs.append(Xf)
        Xbs.append(Xb)
        ys.append(y)
    return Xfs, Xbs, ys


def spell_weights(mids, t_ctrl, w_min):
    """e1_v7 (exp-110): peso de cada fila de team_training = 1 / max(duración en s de la posesión de la verdad de
    entrenamiento que la contiene, w_min). Mismas filas y verdad que team_training; una lista por partido."""
    pm = dfllib.pm
    ws = []
    for mid in mids:
        d = dfllib.load(mid)
        ball, pl = dfllib.stacked(d, "ball")[:, :2], dfllib.stacked(d, "players").astype(np.float64)
        gt = dfllib.official_style(dfllib.gt_labels(d), ball, pl, d["team"], t_ctrl, True)
        vis = load_train(mid)[4]
        idx = np.flatnonzero(((gt == "H") | (gt == "A")) & vis)[::STEP]
        codes = pm.label_codes(gt)
        dur = np.zeros(len(gt))
        for a, b, _ in pm.possession_spells(codes, pm.valid_runs(codes)):
            dur[a:b] = (b - a) / featlib.FPS
        assert (dur[idx] > 0).all(), "fila de entrenamiento fuera de una posesión"
        ws.append(1.0 / np.maximum(dur[idx], w_min))
    return ws


def swap_training(mids, t_ctrl):
    """e1_v13s (exp-148): filas de entrenamiento de las copias con los equipos intercambiados (SWAP_FEAT).

    Las mismas filas que team_training / ctxlib.train_rows del partido original (la máscara de balón visible de la copia
    es la misma: se comprueba) con la etiqueta invertida (1 - y). Los pesos son los de spell_weights del original (la
    duración de las posesiones no cambia con el intercambio). El contexto de P_free (Xc) se calcula sobre las features
    intercambiadas. Una lista por partido, en el orden de mids.
    """
    import ctxlib as ctx
    global FEAT_DIR
    vis = [load_train(mid)[4] for mid in mids]
    old = FEAT_DIR
    use_swap_dir()
    try:
        assert all(np.array_equal(v, load_train(mid)[4]) for v, mid in zip(vis, mids)), "otra máscara de balón visible"
        Xfs, Xbs, ys = team_training(mids, t_ctrl)
        Xcs, ycs = ctx.train_rows(mids)
    finally:
        FEAT_DIR = old
    assert all(np.array_equal(a, b) for a, b in zip(ys, ycs)), "filas distintas entre P_ball y P_free"
    return dict(Xf=Xfs, Xb=Xbs, Xc=Xcs, y=[(1 - y).astype(np.int8) for y in ys])


def fit_pf_swap(mids, ws, sw):
    """e1_v13s: ctxlib.fit_pf con las filas de las copias intercambiadas (swap_training) detrás de las de los partidos;
    ws ya trae los pesos de las dos mitades."""
    import ctxlib as ctx
    Xs, ys = ctx.train_rows(mids)
    Xs, ys = Xs + sw["Xc"], ys + sw["y"]
    assert [len(w) for w in ws] == [len(y) for y in ys], "pesos y filas distintos"
    return [fit_external(Xs, ys, s, ws) for s in SEEDS]


def fit_team(Xfs, Xbs, ys, ws=None):
    if ws is None:
        prior = float(np.concatenate(ys).mean())
    else:  # prior ponderado con los mismos pesos (a media 1)
        assert [len(w) for w in ws] == [len(y) for y in ys], "pesos y filas distintos"
        w, y = np.concatenate(ws).astype(np.float64), np.concatenate(ys)
        w = w / w.mean()
        prior = float(np.sum(w * y) / np.sum(w))
    return dict(pb=[fit_external(Xbs, ys, s, ws) for s in SEEDS], pf=[fit_external(Xfs, ys, s, ws) for s in SEEDS],
                prior=prior)


def team_probs(md, Xf, Xb):
    pb = np.mean([c.predict_proba(Xb)[:, 1] for c in md["pb"]], 0)
    pf = np.mean([c.predict_proba(Xf)[:, 1] for c in md["pf"]], 0)
    return product(pb, pf, md["prior"]), pf, md["prior"]


def main(cfg=CONFIG):
    dfllib.forbid_public_truth()  # la verdad pública no se carga: si algo lo intenta, falla
    m = build_inputs(cfg)
    pred = decode_cfg(m, m["bv"])
    base.write(cfg["name"], pred)
    play = featlib.play_mask(len(pred))
    print(f"{cfg}: D en lo puntuado {np.mean(pred[play] == 2):.1%}")
    return pred


def decode_cfg(m, vis):
    """Decodifica las entradas de build_inputs con la máscara de balón visible `vis` (la real o una de la guarda)."""
    if m.get("kickoff_force"):
        return decode_force(m, vis)
    old, old_dp = ms.PRIOR_D, ms.D_PARAMS
    ms.PRIOR_D = m["prior_d"]
    if m.get("d_params"):
        ms.D_PARAMS = dict(m["d_params"])
    forced = m.get("kickoff_d") or []
    if forced:  # e1_v10: cada bloque se decodifica desde el saque de la Smart Tagging y antes es D
        start = dict(forced)
        m = dict(m, periods=[(start.get(a, a), b) for a, b in m["periods"]])
    try:
        pred = ms.decode(m, vis)
    finally:
        ms.PRIOR_D, ms.D_PARAMS = old, old_dp
    for a, k in forced:
        pred[a:k] = 2
    return pred


def decode_force(m, vis):
    """e1_v11 (exp-126): cada bloque oficial entero, con D obligado (force_dead) desde su inicio hasta el inicio de la
    fase de saque de kickoff_d. Como make_submission.decode sin la regla de balón fuera (apagada desde e1_v9): X fuera
    de los bloques y D entre el final del primero y el inicio del segundo."""
    import hmm3
    assert not m["ball_out"].any(), "kickoff_force supone la regla de balón fuera apagada"
    dp = dict(m["d_params"]) if m.get("d_params") else dict(ms.D_PARAMS)
    p_team = np.where(vis, m["p_vis"], m["p_free"])
    pred = np.full(len(vis), 3, np.int8)
    ends = dict(m.get("kickoff_d") or [])
    for a, b in m["periods"]:
        force = np.zeros(b - a, bool)
        if a in ends:
            force[:ends[a] - a] = True
        pred[a:b] = hmm3.viterbi3(p_team[a:b], m["p_dead"][a:b], m["prior"], m["prior_d"],
                                  force_dead=force, temp_t=m["temp"], **dp)
    (_, end1), (ko2, _) = m["periods"]
    pred[end1:ko2] = 2
    return pred


def build_inputs(cfg):
    """Todo lo que necesita el decodificador para la configuración cfg (BAR-ATH)."""
    use_feat_dir(cfg)
    halves = dfllib.official_halves()  # bloques puntuados oficiales (games.half_video_blocks)
    featlib.PLAY = halves
    tr = featlib.load_tracking()
    bv = tr.balon_visible
    bm = tr.metros(tr.balon)
    L, W = float(atd.CAMPO[0]), float(atd.CAMPO[1])
    with np.errstate(invalid="ignore"):
        ball_out = (bm[:, 0] < -ms.OUT_MARGIN) | (bm[:, 0] > L + ms.OUT_MARGIN) | \
                   (bm[:, 1] < -ms.OUT_MARGIN) | (bm[:, 1] > W + ms.OUT_MARGIN)
    if cfg.get("ball_out", True) is False:  # e1_v9 (exp-121): sin la regla de balón fuera
        ball_out = np.zeros_like(ball_out)

    team = tr.equipo.astype(np.int8)
    xy_d = dedupe(mask_no_velocity(tr.metros(tr.xy), tr.velocidad), team)
    Xd = dead_features(xy_d, team, atd.CAMPO, windows_s=cfg["windows"]).values
    del xy_d
    met, dfl = dead_train_sets(cfg["windows"], cfg["counts"], cfg.get("dead_metrica", True))
    clf, prior_d = dfl_dead.fit(list(met.values()) + list(dfl.values()))
    p_dead = clf.predict_proba(Xd)[:, 1]
    del Xd

    X_free, X_ball = barath_features(tr, halves) if cfg["gk_clean"] else raw_features()
    mids = dfllib.match_ids()
    extra = metrica_training(cfg["t_ctrl"]) if cfg["metrica"] else ()
    ws = None
    if cfg.get("w_spell"):  # e1_v7: pesos por duración de la posesión (exp-110)
        assert not cfg["metrica"] and cfg.get("ctx")
        ws = spell_weights(mids, cfg["t_ctrl"], cfg["w_spell"])
    sw = None
    if cfg.get("swap"):  # e1_v13s (exp-148): las copias intercambiadas van detrás de los 7 partidos, en el mismo orden
        assert not cfg["metrica"] and cfg.get("ctx") and ws is not None
        sw = swap_training(mids, cfg["t_ctrl"])
        extra, ws = list(zip(sw["Xf"], sw["Xb"], sw["y"])), ws + ws
    md = fit_team(*team_training(mids, cfg["t_ctrl"], extra), ws)
    if sw is not None:
        print(f"{cfg['name']}: filas de equipo {2 * sum(map(len, sw['y']))} (la mitad, copias intercambiadas con "
              f"etiqueta media {np.concatenate(sw['y']).mean():.4f}); prior ponderado {md['prior']!r}", flush=True)
    pv, pf, prior = team_probs(md, X_free, X_ball)
    if cfg.get("ctx"):
        import ctxlib as ctx  # P_free con contexto temporal (exp-087), mismas filas y etiqueta
        assert cfg["t_ctrl"] == 1.0 and not cfg["metrica"]
        pb = np.mean([c.predict_proba(X_ball)[:, 1] for c in md["pb"]], 0)
        pf = ctx.predict(ctx.fit_pf(mids, ws) if sw is None else fit_pf_swap(mids, ws, sw),
                         ctx.context(X_free, ctx.seg_ids(len(X_free), halves)))
        pv = product(pb, pf, prior)
    del X_free, X_ball
    if cfg.get("tcn"):  # e1_v14 (exp-139): mezcla con la red temporal, igual que dfl_tcn_judge.py
        lg, prior_net = tcn_logits25(cfg["tcn"], halves, len(pv), "barath")
        pv, pf = tcn_mix(pv, pf, prior, lg, prior_net)

    kickoff_d = kickoff_tags(halves) if cfg.get("kickoff_tag") else []
    return dict(p_vis=pv, p_free=pf, prior=prior, p_dead=p_dead, ball_out=ball_out, temp=cfg["temp"], periods=halves,
                prior_d=prior_d, d_params=cfg.get("d_params"), bv=bv, kickoff_d=kickoff_d,
                kickoff_force=cfg.get("kickoff_force", False))


def tcn_logits25(path, periods, n, key):
    """e1_v14: logits de la red (5 Hz, calibrados y redondeados a 3 decimales) de cada semilla `<key>_s<s>_<parte>`,
    media en log-odds e interpolación lineal a 25 Hz (centro de cada bloque de 5 frames); fuera de las partes, el logit
    del prior (sin evidencia). Como dfl_tcn_judge.net25."""
    z = np.load(path)
    seeds = sorted(int(k.split("_s")[1]) for k in z.files if k.startswith("prior_s"))
    lgs = []
    for s in seeds:
        lg = np.full(n, np.nan)
        for i, (a, b) in enumerate(periods):
            c = np.minimum(np.arange(a, b, 5) + 2, b - 1)
            lg[a:b] = np.interp(np.arange(a, b), c, z[f"{key}_s{s}_{i}"])
        lgs.append(lg)
    prior = float(np.mean([float(z[f"prior_s{s}"]) for s in seeds]))
    lg = np.mean(lgs, 0)
    lg[np.isnan(lg)] = np.log(prior / (1 - prior))
    return lg, prior


def tcn_mix(pv, pf, prior, lg, prior_net):
    """dfl_teacher.mix (0,5 / 0,5 en log-odds alrededor del prior) de los árboles con la red, en p_vis y en p_free."""
    p = np.clip(1 / (1 + np.exp(-lg)), 1e-6, 1 - 1e-6)
    return mix(pv, prior, p, prior_net), mix(pf, prior, p, prior_net)


def barath_features(tr, periods):
    """Features de BAR-ATH con la limpieza de porteros, alineadas por nombre con las de entrenamiento de la DFL.

    Igual que make_v6_external.barath_features, pero toma los nombres de las features de load_train (FEAT_DIR de la
    configuración) en vez de la caché antigua de dfl_teacher (feat/), que la copia limpia no tiene. Los nombres y el
    orden son los mismos en las dos (comprobado el 08/11), así que el resultado no cambia.
    """
    X_free, nf_bar = featlib.load_families(ext.FREE5)
    nf_bar = [n.split(":", 1)[1] for n in nf_bar]
    X_ball, nb_bar = featlib.load_families(ms.BALL)
    tr_c, changed, _ = gklib.clean_goalkeepers(tr, periods)
    X_ball = gklib.patch_ball_features(tr_c, X_ball, changed)
    _, _, nf, nb, _ = load_train(dfllib.match_ids()[0])
    return ext.aligned(X_free, nf_bar, nf), ext.aligned(X_ball, nb_bar, nb)


def raw_features():
    """Features de BAR-ATH sin la limpieza de porteros, alineadas con las de la DFL."""
    X_free, nf_bar = featlib.load_families(ext.FREE5)
    nf_bar = [n.split(":", 1)[1] for n in nf_bar]
    X_ball, nb_bar = featlib.load_families(ms.BALL)
    _, _, nf, nb, _ = load_train(dfllib.match_ids()[0])
    return ext.aligned(X_free, nf_bar, nf), ext.aligned(X_ball, nb_bar, nb)


def metrica_training(t_ctrl, gids=(1, 2, 3)):
    """Frames de entrenamiento de los partidos de Metrica (verdad de eventos `mid`), alineados con la DFL."""
    import rv03_metrica_official_gt as rv03
    _, _, nf, nb, _ = load_train(dfllib.match_ids()[0])
    out = []
    for gid in gids:
        g = load_game(gid)
        z = np.load(os.path.join(featlib.OUT, f"exp026_game{gid}.npz"))
        gts, _ = rv03.official_style_gts(g)
        gt = gts["mid"]
        play = np.zeros(len(gt), bool)
        for a, b in z["periods"]:
            play[a:b] = True
        idx = np.flatnonzero(play & ((gt == 0) | (gt == 1)) & np.isfinite(g.ball[:, 0]))[::STEP]
        Xf = ext.aligned(z["X_free"], [str(n) for n in z["names_free"]], nf)
        Xb = ext.aligned(z["X_ball"], [str(n) for n in z["names_ball"]], nb)
        out.append((Xf[idx], Xb[idx], (gt[idx] == 1).astype(np.int8)))
    return out


if __name__ == "__main__":
    # exp-101: ctxlib hace "import make_e1"; ejecutado como script habría dos copias del módulo y ctxlib leería las
    # features de la carpeta por defecto (feat_v2). Se registra el script como make_e1 para que usen la misma.
    assert sys.modules.get("make_e1") in (None, sys.modules[__name__]), "make_e1 ya importado por otra vía"
    sys.modules["make_e1"] = sys.modules[__name__]
    if len(sys.argv) > 3 and sys.argv[1] == "build" and sys.argv[3] == "swap":  # e1_v13s: build <partido> swap
        use_feat_dir(CONFIG_V13S)  # degradación de feat_sc (modelo de huecos de SkillCorner)
        build(sys.argv[2], swap=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "build":  # build <partido> [versión]
        use_feat_dir(VERSIONS[sys.argv[3]] if len(sys.argv) > 3 else CONFIG)  # features con su modelo de huecos
        build(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] in VERSIONS:
        main(VERSIONS[sys.argv[1]])  # otras versiones (v8: modelo de huecos de SkillCorner)
    else:
        main()
