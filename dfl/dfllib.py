"""Utilidades comunes para usar la DFL (IDSSE) como juez externo.

- `load(match_id)`: arrays de la caché (idsse_cache.py).
- `gt_labels(d)`: verdad por frame como la oficial: D si BallStatus = 0; si no, H/A por
  BallPossession. Las dos partes se concatenan con un frame 'X' en medio, para que el evaluador
  oficial (valid_runs) las trate por separado.
- `degrade(d, level, seed)`: estropea el balón y los jugadores como en BAR-ATH.
- `nearest_codes`, `smooth`: el baseline oficial (jugador más cercano + voto por mayoría).
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.path.join(os.path.dirname(HERE), "more-than-a-hack-2027")
sys.path.insert(0, os.path.join(OFFICIAL, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Datos del reto y cachés: siempre bajo MTAH_DATA_DIR (nunca rutas escritas a mano ni dentro del repo).
DATA = os.environ["MTAH_DATA_DIR"]
CACHE = os.path.join(DATA, "cache", "idsse")
GAP_MODEL = os.path.join(DATA, "cache", "barath_gap_model.json")  # derivado de BAR-ATH: fuera del repo
# exp-138 (B2): (fracción, duración media en s) de los jugadores perdidos en degrade(level 2). None = 0,07 / 4 s, como
# hasta e1_v14 (origen no documentado). make_e1.use_feat_dir lo fija con el «occ» de la configuración.
OCCLUSION = None
# exp-144 (B2b): parámetros de la cámara medidos en SkillCorner (skillcorner_camera.json) o None (sin ventana, como hasta
# e1_v14). Si se fija, degrade(level 2) oculta además a los jugadores fuera de una ventana en x que sigue al balón, con
# la misma estructura que el juez duro (dfl_judge_sc.window) y su propio generador. make_e1.use_feat_dir lo fija.
CAMERA = None
COUNTS_BLOCKS = os.path.join(DATA, "cache", "barath_counts_blocks.npy")  # barath_counts.py


def official_halves():
    """Bloques puntuados oficiales de BAR-ATH como intervalos semiabiertos [(a, b), ...].

    Salen de games.half_video_blocks del clon oficial (norma: nada de frames escritos a mano). games.py se carga
    por su ruta para no depender de sys.path.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("games_oficial", os.path.join(OFFICIAL, "games.py"))
    games = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(games)
    return [(int(a), int(b) + 1) for _, a, b in games.half_video_blocks(games.GAME)]
def is_truth_path(path):
    """True si `path` es un fichero de verdad del reto: *possession_gt* o cualquier cosa bajo evaluation/gt/."""
    if isinstance(path, int):  # descriptor de fichero ya abierto
        return False
    p = os.path.normcase(os.path.abspath(os.fsdecode(path)))
    parts = p.split(os.sep)
    return "possession_gt" in os.path.basename(p) or any(
        a == "evaluation" and b == "gt" for a, b in zip(parts, parts[1:]))


_TRUTH_HOOK = []


def _truth_audit(event, args):
    if event == "open" and args and args[0] is not None and is_truth_path(args[0]):
        raise RuntimeError(f"el camino de la entrega ha intentado abrir un fichero de verdad: {os.fsdecode(args[0])}")


def forbid_public_truth():
    """Bloquea la verdad en el camino de la entrega: cualquier intento de leerla hace fallar el script.

    Dos capas:
      1. Un audit hook de Python (sys.addaudithook, evento "open") que hace fallar cualquier apertura de un fichero de
         verdad (is_truth_path), venga de donde venga (open, pandas, numpy...). No se puede quitar una vez puesto.
      2. common.gt_codes / featlib.gt_codes y possession_metrics.read_label_csv se sustituyen por funciones que fallan
         (las del evaluador siguen leyendo predicciones).
    Requiere las rutas de la copia de Pablo (common, featlib).
    """
    if not _TRUTH_HOOK:
        sys.addaudithook(_truth_audit)
        _TRUTH_HOOK.append(True)
    import common
    import featlib

    def _no(*args, **kwargs):
        raise RuntimeError("el camino de la entrega ha intentado leer la verdad pública")

    common.gt_codes = featlib.gt_codes = _no
    read = pm.read_label_csv

    def _read(path, *args, **kwargs):
        name = os.path.basename(str(path)).lower()
        if "gt" in name.split("_") or "possession_gt" in name:
            _no()
        return read(path, *args, **kwargs)

    pm.read_label_csv = _read
    if getattr(common, "pm", None) is not None and common.pm is not pm:
        common.pm.read_label_csv = _read


HOME, AWAY, NOBODY = 0, 1, -1
LENGTH, WIDTH = 105.0, 68.0


def match_ids():
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(CACHE, "*.npz")))


def load(match_id):
    with np.load(os.path.join(CACHE, match_id + ".npz")) as z:
        return {k: z[k] for k in z.files}


def concat_halves(a1, a2, fill):
    sep = np.array([fill], dtype=a1.dtype) if a1.ndim == 1 else np.full((1,) + a1.shape[1:], fill, a1.dtype)
    return np.concatenate([a1, sep, a2])


def gt_labels(d):
    halves = []
    for h in (1, 2):
        lab = np.where(d["poss_%d" % h] == 1, "H", "A").astype("<U1")
        lab[d["status_%d" % h] == 0] = "D"
        halves.append(lab)
    return concat_halves(halves[0], halves[1], "X")


def stacked(d, name, fill=np.nan):
    return concat_halves(d["%s_1" % name], d["%s_2" % name], fill)


# -- verdad "estilo oficial" ---------------------------------------------------------------------

def _runs(lab):
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.r_[0, b]
    stops = np.r_[b, len(lab)]
    return [(lab[s], s, e) for s, e in zip(starts, stops)]


def official_style(gt, ball, players, team, t_ctrl_s, mid=True, max_flight_s=4.0, touch_m=1.0):
    """Convierte la posesión por 'último toque' de la DFL a la definición del reto.

    1. Control: un tramo del rival de menos de t_ctrl_s que viene justo después de una posesión
       viva del otro equipo es un toque sin control y se devuelve a ese equipo. Tras un D no se
       toca (un reinicio siempre tiene un poseedor claro). Se repite hasta que no cambia nada.
    2. Momento (mid=True): en cada cambio directo A->B, el cambio pasa a la mitad entre el último
       frame con un jugador de A a < touch_m del balón y el primer toque de B, si ese vuelo dura
       <= max_flight_s. El tramo entre medias se marca como B.
    """
    lab = gt.copy()
    min_len = int(round(t_ctrl_s * 25))
    changed = True
    while changed and min_len > 0:
        changed = False
        r = _runs(lab)
        for i in range(1, len(r)):
            l, s, e = r[i]
            pl = lab[s - 1]  # etiqueta ya corregida del frame anterior
            if l in "HA" and pl in "HA" and pl != l and e - s < min_len:
                lab[s:e] = pl
                changed = True
    if mid:
        dist = {}
        for k in (0, 1):
            sel = players[:, team == k, :]
            dd = np.linalg.norm(sel - ball[:, None, :], axis=2)
            dist[k] = np.nanmin(np.where(np.isnan(dd), np.inf, dd), axis=1)
        code = {"H": 0, "A": 1}
        max_f = int(max_flight_s * 25)
        r = _runs(lab)
        for i in range(1, len(r)):
            a, b = r[i - 1][0], r[i][0]
            if a not in code or b not in code or a == b:
                continue
            t, s_prev = r[i][1], r[i - 1][1]
            lo = max(s_prev, t - max_f)
            near = np.flatnonzero(dist[code[a]][lo:t] < touch_m)
            if not len(near):
                continue
            release = lo + near[-1]
            m = (release + t + 1) // 2
            if m < t:
                lab[m:t] = b
    return lab


# -- degradación tipo BAR-ATH ------------------------------------------------------------------

def load_gap_model():
    with open(GAP_MODEL) as f:
        return json.load(f)


def markov_mask(state_dead, p_start, p_end, rng, p0):
    """True = observación perdida. Cadena de 2 estados cuyas tasas dependen del estado del frame."""
    n = len(state_dead)
    u = rng.random(n)
    miss = np.empty(n, dtype=bool)
    cur = rng.random() < p0
    for t in range(n):
        s = 1 if state_dead[t] else 0
        cur = (u[t] >= p_end[s]) if cur else (u[t] < p_start[s])
        miss[t] = cur
    return miss


def alternating_mask(n, frac, mean_len, rng):
    """Episodios de pérdida de duración geométrica (media mean_len frames), fracción ~frac."""
    mean_gap = mean_len * (1 - frac) / frac
    k = int(2 * n / (mean_len + mean_gap)) + 10
    lens = np.empty(2 * k, dtype=np.int64)
    lens[0::2] = rng.geometric(1.0 / mean_gap, k)
    lens[1::2] = rng.geometric(1.0 / mean_len, k)
    edges = np.cumsum(lens)
    flag = np.zeros(n + 1, dtype=np.int8)
    starts, stops = edges[0::2], edges[1::2]
    flag[np.minimum(starts, n)] += 1
    flag[np.minimum(stops, n)] -= 1
    return np.cumsum(flag)[:n] > 0


def camera_window(ball_x, rng, p):
    """Centro y ancho de la ventana por frame: copia literal de dfl_judge_sc.window (control C5 de exp-144)."""
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d

    import dfl_camera as cam
    fps = 25
    n = len(ball_x)
    b = pd.Series(ball_x).ffill().bfill().values
    smooth = gaussian_filter1d(b, 2 * fps)
    lag = int(round(p["lag_s"] * fps))
    center = (np.r_[np.full(lag, smooth[0]), smooth[:-lag]] if lag else smooth) + \
        cam.ou(n, p["center_sigma_m"], p["center_tau_s"], rng)
    width = p["width_m"] * np.exp(cam.ou(n, p["width_log_sigma"], p["width_tau_s"], rng))
    center = np.clip(center, -cam.HALF_L - 8 + width / 2, cam.HALF_L + 8 - width / 2)
    return np.clip(center, b - width / 2 + 1, b + width / 2 - 1), width


def camera_seen(ball_x, players, p, rng):
    """Jugadores dentro de la ventana (con el cierre de 13 frames del juez duro) y presentes en la DFL."""
    from scipy.ndimage import binary_closing
    center, width = camera_window(ball_x, rng, p)
    seen = np.isfinite(players[..., 0]) & (np.abs(players[..., 0] - center[:, None]) <= width[:, None] / 2)
    for j in range(players.shape[1]):
        seen[:, j] = binary_closing(seen[:, j], np.ones(13, bool))
    return seen & np.isfinite(players[..., 0])


def degrade(d, level, seed):
    """Devuelve (ball_xy, players) degradados, ya concatenados.

    level 0: nativo (balón y 22 jugadores siempre).
    level 1: huecos del balón con el modelo de BAR-ATH (P(falta) 0,09 vivo / 0,84 parado).
    level 2: level 1 + jugadores perdidos (~7%, episodios de ~4 s) + ruido de 0,3 m en el balón.
    """
    rng = np.random.default_rng(seed)
    ball = stacked(d, "ball")[:, :2].copy()
    players = stacked(d, "players").astype(np.float64)
    if level == 0:
        return ball, players
    ball_x_true = ball[:, 0].copy()
    gm = load_gap_model()
    dead = stacked(d, "status", 0) == 0
    p_start = (gm["vivo"]["p_empieza"], gm["D"]["p_empieza"])
    p_end = (gm["vivo"]["p_acaba"], gm["D"]["p_acaba"])
    ball[markov_mask(dead, p_start, p_end, rng, 0.5)] = np.nan
    if level >= 2:
        ball += rng.normal(0, 0.3, ball.shape)
        n, p = players.shape[:2]
        if CAMERA is not None:  # exp-144: después del ruido del balón, antes de las oclusiones (balón idéntico)
            players[~camera_seen(ball_x_true, players, CAMERA, rng)] = np.nan
        frac, mean_s = OCCLUSION or (0.07, 4.0)
        for j in range(p):
            players[alternating_mask(n, frac, mean_s * 25, rng), j] = np.nan
    return ball, players


# -- baseline oficial ---------------------------------------------------------------------------

def nearest_codes(ball, players, team):
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    j = np.argmin(dist, axis=1)
    codes = np.where(team[j] == 0, HOME, AWAY).astype(np.int8)
    codes[~np.isfinite(dist[np.arange(len(j)), j])] = NOBODY
    codes[np.isnan(ball[:, 0])] = NOBODY
    return codes


def smooth(codes, window):
    """Igual que starter/baseline.py: voto por mayoría centrado de +/- window frames."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D")).astype("<U1")
    hc = np.concatenate(([0], np.cumsum(codes == HOME)))
    ac = np.concatenate(([0], np.cumsum(codes == AWAY)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    lab = np.full(len(codes), "D", dtype="<U1")
    seen = (h + a) > 0
    lab[seen] = np.where(h[seen] >= a[seen], "H", "A")
    return lab


def evaluate(gt, pred):
    """El dict completo de possession_metrics.evaluate (todas las métricas del evaluador oficial)."""
    pred = pred.copy()
    pred[gt == "X"] = "X"
    return pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))


def score(gt, pred):
    r = evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]
