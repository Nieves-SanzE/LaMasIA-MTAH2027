"""v4 y v6 de Pablo sobre la DFL degradada como BAR-ATH (juez externo). Solo agregados.

    cd pablo_v6
    set OMP_NUM_THREADS=4
    .venv\\Scripts\\python.exe ..\\dfl\\dfl_v6.py [--seed 0] [--matches J03WMX ...]

Usa el código de la rama kura-eda (commit 524becb, exportado en pablo_v6) sin
modificarlo, igual que su validación en Metrica (exp-026 / exp-055):
  features   5 familias sin balón (fe_geometry) + 5 con balón (fb_ball), sobre el partido degradado
             (el balón solo existe donde la máscara lo deja ver, como en BAR-ATH). Las familias
             temporales de v3 (leadlag, sync, tda, multiscale) no están portadas, igual que en Metrica.
  etiquetas  pseudo-etiquetas `base` (control >= 12 frames a < 1,5 m) y, para v6, `clean_mid`.
  P(equipo)  GBM fuera de pliegue dentro del partido (bloques de 5 min), media de 3 semillas:
             v4 = logit P_ball + logit P_free - logit prior;
             v6 = 0,5 (logit P_ball base + logit P_ball clean_mid) + logit P_free - logit prior.
  P(muerto)  modelo de Metrica (deadlib.metrica_dead_model) con la aumentación de cámara de
             BAR-ATH (barath_counts.py); v6 con ventanas <= 8 s.
  decodif.   Viterbi conjunto H/A/D de make_submission (D_PARAMS, PRIOR_D, balón fuera que solo
             extiende), temperatura 0,85 (v4) o 0,6 (v6).
  Sin limpieza de porteros: en la DFL no hay cambios de equipo de los porteros.
Guarda las predicciones en MTAH_DATA_DIR\\cache\\idsse\\pred\\<id>_s<seed>.npz (datos abiertos).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PABLO = os.path.join(os.path.dirname(HERE), "pablo_v6")
sys.path.insert(0, os.path.join(PABLO, "scripts"))
sys.path.insert(0, os.path.join(PABLO, "scripts", "explore"))
sys.path.insert(0, os.path.join(PABLO, "src"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import dfllib  # noqa: E402
import featlib  # noqa: E402  (common.py hace os.chdir a out/explore de la copia de Pablo)
import make_submission as ms  # noqa: E402
import exp026_metrica_features as mf  # noqa: E402
from exp026_v5 import oof, pseudo_labels  # noqa: E402
from deadlib import dead_features, extend_only, metrica_dead_model  # noqa: E402
from hmm3 import viterbi3  # noqa: E402
from metrica_lib import Game  # noqa: E402

FPS = 25
SEEDS = (0, 1, 2)
PITCH = np.array([105.0, 68.0], np.float32)
PRED_DIR = os.path.join(dfllib.CACHE, "pred")
COUNTS = dfllib.COUNTS_BLOCKS  # bloques oficiales (antes barath_counts_v6.npy, con 113.742; exp-092)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def to_top_left(a):
    """DFL: metros con origen en el centro e y hacia arriba -> origen arriba a la izquierda (Metrica)."""
    out = a.copy()
    out[..., 0] = a[..., 0] + PITCH[0] / 2
    out[..., 1] = PITCH[1] / 2 - a[..., 1]
    return out


def degraded_game(mid, seed):
    d = dfllib.load(mid)
    mids = dfllib.match_ids()
    ball, players = dfllib.degrade(d, 2, seed=1000 * seed + mids.index(mid))  # mismas semillas que el juez
    n, n1 = len(ball), len(d["status_1"])
    period = np.zeros(n, np.int8)
    period[:n1], period[n1 + 1:] = 1, 2
    g = Game(mid, to_top_left(players).astype(np.float32), d["team"].astype(np.int8),
             to_top_left(ball).astype(np.float32), period, pd.DataFrame())
    return g, d


def decode(p_team, p_dead, prior, periods, vis, ball_out, temp):
    """make_submission.decode sin el relleno del descanso (no hay frames entre partes)."""
    out = vis & ball_out
    s_ = pd.Series(out)
    out &= s_.groupby(s_.ne(s_.shift()).cumsum()).transform("size").values >= FPS // 2
    pred = np.full(len(vis), 3, np.int8)
    kw = dict(temp_t=temp, **ms.D_PARAMS)
    for a, b in periods:
        first = viterbi3(p_team[a:b], p_dead[a:b], prior, ms.PRIOR_D, **kw)
        force = extend_only(out[a:b], first == 2, 5 * FPS)
        pred[a:b] = viterbi3(p_team[a:b], p_dead[a:b], prior, ms.PRIOR_D, force_dead=force, **kw)
    return pred


def run_match(mid, seed, dead_models):
    g, d = degraded_game(mid, seed)
    periods = mf.periods_of(g)
    tr = mf.as_tracking(g)
    mf.point_globals_at(tr, periods)
    vis = np.isfinite(g.ball[:, 0])
    play = g.period > 0
    log(f"{mid}: {tr.n_frames} frames, balón visible {vis[play].mean():.1%} de lo jugado")
    Xf, _ = mf.free_families(tr)
    Xb, _ = mf.ball_families()
    log(f"{mid}: features {Xf.shape[1]} sin balón, {Xb.shape[1]} con balón")
    dist = tr.distancia_al_balon()
    dmin = np.nanmin(np.where(np.isfinite(dist), dist, np.inf), 1)
    near = tr.mas_cercano_al_balon()
    near_team = np.where(near >= 0, tr.equipo[np.clip(near, 0, None)], -1).astype(np.int8)

    base = pseudo_labels(dmin, near_team, vis, periods)
    cmid = featlib.timed_team_labels(dmin, near_team, vis, periods, "clean_mid")
    tr_b, tr_c = play & vis & (base >= 0), play & vis & (cmid >= 0)
    y_b, y_c = (base == 1).astype(np.int8), (cmid == 1).astype(np.int8)
    prior = float(np.mean(y_b[tr_b]))
    p_free = np.mean([oof(Xf, y_b, tr_b, play, s) for s in SEEDS], 0)
    p_ball_b = np.mean([oof(Xb, y_b, tr_b, play, s) for s in SEEDS], 0)
    p_ball_c = np.mean([oof(Xb, y_c, tr_c, play, s) for s in SEEDS], 0)
    log(f"{mid}: P(equipo) listas (prior {prior:.3f})")

    xy = g.xy  # sin duplicados ni velocidades del proveedor: dedupe y mask_no_velocity no aplican
    p_dead = {k: m.predict_proba(dead_features(xy, g.team, PITCH, windows_s=w).values)[:, 1]
              for k, (m, w) in dead_models.items()}
    with np.errstate(invalid="ignore"):
        bx, by = g.ball[:, 0], g.ball[:, 1]
        ball_out = (bx < -ms.OUT_MARGIN) | (bx > PITCH[0] + ms.OUT_MARGIN) | \
                   (by < -ms.OUT_MARGIN) | (by > PITCH[1] + ms.OUT_MARGIN)
    lp = logit(prior)
    pv4 = sig(logit(p_ball_b) + logit(p_free) - lp)
    pv6 = sig(0.5 * (logit(p_ball_b) + logit(p_ball_c)) + logit(p_free) - lp)
    preds = {
        "v4": decode(np.where(vis, pv4, p_free), p_dead["w32"], prior, periods, vis, ball_out, 0.85),
        "v6": decode(np.where(vis, pv6, p_free), p_dead["w8"], prior, periods, vis, ball_out, ms.V6_TEMP),
        # piezas de v6 por separado, para saber qué aporta cada una
        "v4+D(w8)": decode(np.where(vis, pv4, p_free), p_dead["w8"], prior, periods, vis, ball_out, 0.85),
        "v4+mezcla+T0.6": decode(np.where(vis, pv6, p_free), p_dead["w32"], prior, periods, vis, ball_out, ms.V6_TEMP),
    }
    os.makedirs(PRED_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(PRED_DIR, f"{mid}_s{seed}.npz"),
                        **{k: np.array(list("HADX"))[v] for k, v in preds.items()})
    # todo lo que hace falta para volver a decodificar con otro P(muerto) sin recalcular features
    np.savez_compressed(os.path.join(PRED_DIR, f"{mid}_s{seed}_probs.npz"), p_free=p_free,
                        p_ball_b=p_ball_b, p_ball_c=p_ball_c, prior=prior, vis=vis, ball_out=ball_out,
                        periods=np.array(periods), p_dead_w32=p_dead["w32"], p_dead_w8=p_dead["w8"])
    gt = dfllib.gt_labels(d)
    for k, v in preds.items():
        sp, mfm, npred, ngt = dfllib.score(gt, np.array(list("HADX"))[v])
        log(f"{mid}: {k:<15} spellF1 {sp:.3f} macroF1 {mfm:.3f} posesiones {npred}/{ngt} (verdad tal cual)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matches", nargs="*", default=None)
    args = ap.parse_args()
    counts = np.load(COUNTS)
    log("entrenando P(muerto) en Metrica (v4: ventanas hasta 32 s; v6: hasta 8 s) ...")
    dead_models = {"w32": (metrica_dead_model(counts), deadlib_windows()),
                   "w8": (metrica_dead_model(counts, windows_s=ms.V6_WINDOWS_S), ms.V6_WINDOWS_S)}
    log("modelos de P(muerto) listos")
    for mid in args.matches or dfllib.match_ids():
        if os.path.exists(os.path.join(PRED_DIR, f"{mid}_s{args.seed}_probs.npz")):
            log(f"{mid}: ya hecho")
            continue
        run_match(mid, args.seed, dead_models)


def deadlib_windows():
    import deadlib
    return deadlib.WINDOWS_S


if __name__ == "__main__":
    main()
