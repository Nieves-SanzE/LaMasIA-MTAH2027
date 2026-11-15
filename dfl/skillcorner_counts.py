"""exp-127, paso 1: jugadores visibles por frame medidos en SkillCorner open data (datos externos, licencia MIT), para la
aumentación «tipo cámara» del entrenamiento de D (deadlib.truncate_like), en lugar del vector de BAR-ATH
(barath_counts.py, G2 de la auditoría). Plantilla: skillcorner_gap_model.py. Nada de BAR-ATH.

Estructura comprobada antes (12/11) en un *_tracking_extrapolated.jsonl: cada línea trae frame, timestamp, period,
ball_data, possession, image_corners_projection y player_data; cada jugador de player_data trae solo x, y, player_id e
is_detected. El fichero NO da el equipo de cada jugador (solo possession.group del poseedor), y skillcorner_download.py
no descarga el match.json. Por eso se usa el tope de 22 en total, no 11 por equipo.

Definiciones, fijadas antes de mirar ninguna cifra:
  - frames que cuentan: period 1 o 2 y timestamp no nulo (los mismos que skillcorner_gap_model.py), vivos y parados,
    igual que barath_counts.py cuenta todos los frames de los bloques puntuados;
  - jugador visible: is_detected es True (las posiciones extrapoladas no cuentan, como en BAR-ATH los slots
    interpolados sin velocidad);
  - recuento por frame = min(nº de visibles, 22);
  - se juntan los frames de todos los partidos, en el orden de los ficheros (sorted por id) y de las líneas. Solo
    importa la distribución: truncate_like sortea k de este vector bloque a bloque. A 10 fps (SkillCorner) en vez de
    25; no se remuestrea porque la distribución por frame no depende de la frecuencia.
Escribe MTAH_DATA_DIR/cache/skillcorner_counts.npy (int16). Es un derivado de datos externos. Solo imprime agregados.

    cd pablo_v6
    .venv\\Scripts\\python.exe -I -X utf8 -u ..\\dfl\\skillcorner_counts.py
"""
import glob
import json
import os

import numpy as np

SRC = os.path.join(os.environ["MTAH_DATA_DIR"], "external", "skillcorner")
OUT = os.path.join(os.environ["MTAH_DATA_DIR"], "cache", "skillcorner_counts.npy")
# e1_v12 en adelante (exp-129): solo los frames con >= 1 jugador detectado (cámara que encuadra el juego). Es el vector
# que usa make_e1 (CONFIG_V12 / V13); hasta el 12/11 lo escribía el juez dfl_g2b_confirm.py, con el mismo contenido.
OUT_DET = os.path.join(os.environ["MTAH_DATA_DIR"], "cache", "skillcorner_counts_det.npy")
MAX_PLAYERS = 22


def match_counts(folder):
    mid = os.path.basename(folder)
    out, n_entries = [], []
    with open(os.path.join(folder, f"{mid}_tracking_extrapolated.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("period") not in (1, 2) or r.get("timestamp") is None:
                continue
            pl = r.get("player_data") or []
            n_entries.append(len(pl))
            out.append(min(sum(1 for p in pl if p.get("is_detected") is True), MAX_PLAYERS))
    return np.array(out, np.int16), np.array(n_entries)


def main():
    folders = sorted(glob.glob(os.path.join(SRC, "*")))
    assert folders, f"no hay partidos en {SRC} (skillcorner_download.py)"
    parts, per_match, entries = [], [], []
    for folder in folders:
        c, e = match_counts(folder)
        parts.append(c)
        per_match.append(c.mean())
        entries.append(e)
    counts = np.concatenate(parts)
    e = np.concatenate(entries)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.save(OUT, counts.astype(np.int16))
    np.save(OUT_DET, counts[counts > 0].astype(np.int16))
    print(f"{len(folders)} partidos de SkillCorner; {len(counts)} frames ({len(counts) / 10 / 60:.0f} min a 10 fps)")
    print(f"jugadores visibles por frame: media {counts.mean():.1f}, p10 {np.percentile(counts, 10):.0f}, "
          f"p50 {np.percentile(counts, 50):.0f}, p90 {np.percentile(counts, 90):.0f}; "
          f"0 visibles en el {np.mean(counts == 0):.1%} de los frames")
    print(f"media por partido: {min(per_match):.1f}-{max(per_match):.1f}; entradas de player_data por frame: "
          f"{e.min()}-{e.max()} (con > 22: {np.mean(e > 22):.2%})")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
