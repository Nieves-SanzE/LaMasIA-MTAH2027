"""exp-119, paso 1: modelo de huecos del balón medido en SkillCorner (datos externos), sin nada de BAR-ATH.

Mismo modelo que barath_gap_model.py (cadena de 2 estados: visible / ausente, con tasas según el frame esté vivo o
parado), con las mismas cuentas (gap_stats), pero medido en el tracking de retransmisión de SkillCorner open data
(skillcorner_download.py). Definiciones, fijadas antes de mirar ninguna cifra:
  - frames que cuentan: period 1 o 2 y timestamp no nulo; transiciones solo entre frames consecutivos de la misma parte;
  - balón ausente: ball_data.is_detected distinto de True (False o nulo) o x nulo. Las posiciones extrapoladas cuentan
    como ausentes, igual que en BAR-ATH, donde el balón no detectado es NaN;
  - estado vivo: el frame está dentro de una fase de juego (phases_of_play.csv, frame_start <= f < frame_end, misma
    parte); si no, parado (D). SkillCorner solo define fases con el balón en juego;
  - recuentos sumados en todos los partidos (no medias de probabilidades);
  - de 10 a 25 fps conservando la tasa por segundo: p25 = 1 - (1 - p10) ** (10 / 25).
Escribe MTAH_DATA_DIR/cache/skillcorner_gap_model.json con las mismas claves que barath_gap_model.json. Es un
derivado de datos externos (licencia MIT): no contiene nada del partido del reto. Solo imprime agregados.

    python -I skillcorner_gap_model.py
"""
import glob
import json
import os

import numpy as np

SRC = os.path.join(os.environ["MTAH_DATA_DIR"], "external", "skillcorner")
OUT = os.path.join(os.environ["MTAH_DATA_DIR"], "cache", "skillcorner_gap_model.json")
FPS_SRC, FPS_DST = 10, 25


def load_match(folder):
    mid = os.path.basename(folder)
    frame, period, seen = [], [], []
    with open(os.path.join(folder, f"{mid}_tracking_extrapolated.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("period") not in (1, 2) or r.get("timestamp") is None:
                continue
            b = r.get("ball_data") or {}
            frame.append(int(r["frame"]))
            period.append(int(r["period"]))
            seen.append(b.get("is_detected") is True and b.get("x") is not None)
    frame, period, seen = np.array(frame), np.array(period), np.array(seen)
    live = np.zeros(len(frame), bool)
    import csv
    with open(os.path.join(folder, f"{mid}_phases_of_play.csv"), encoding="utf-8") as f:
        for ph in csv.DictReader(f):
            a, b, p = int(ph["frame_start"]), int(ph["frame_end"]), int(ph["period"])
            live |= (period == p) & (frame >= a) & (frame < b)
    return frame, period, seen, live


def counts(frame, period, seen, live):
    """Recuentos de gap_stats (barath_gap_model.py) por estado del frame actual."""
    ok = (np.diff(frame) == 1) & (period[1:] == period[:-1])  # transición t-1 -> t válida
    prev, cur = seen[:-1], seen[1:]
    out = {}
    for name, st in (("vivo", live), ("D", ~live)):
        mm = ok & st[1:]
        out[name] = dict(n=int(st.sum()), falta=int((st & ~seen).sum()),
                         start=int((mm & prev & ~cur).sum()), start_den=int((mm & prev).sum()),
                         end=int((mm & ~prev & cur).sum()), end_den=int((mm & ~prev).sum()))
    return out


def main():
    folders = sorted(glob.glob(os.path.join(SRC, "*")))
    assert folders, f"no hay partidos en {SRC} (skillcorner_download.py)"
    tot = {}
    per_match = []
    for folder in folders:
        c = counts(*load_match(folder))
        per_match.append((os.path.basename(folder), c))
        for s, v in c.items():
            for k, x in v.items():
                tot.setdefault(s, {}).setdefault(k, 0)
                tot[s][k] += x
    conv = lambda p: 1.0 - (1.0 - p) ** (FPS_SRC / FPS_DST)  # noqa: E731
    model = {}
    print(f"{len(folders)} partidos de SkillCorner")
    for s, v in tot.items():
        p_falta = v["falta"] / v["n"]
        ps, pe = v["start"] / v["start_den"], v["end"] / v["end_den"]
        model[s] = {"p_falta": p_falta, "p_empieza": conv(ps), "p_acaba": conv(pe)}
        model.setdefault("10hz", {})[s] = {"p_empieza": ps, "p_acaba": pe}
        print(f"  {s:<5} frames {v['n'] / FPS_SRC / 60:7.0f} min | P(falta) {p_falta:.3f} | hueco medio "
              f"{1 / pe / FPS_SRC:5.1f} s | tramo visible medio {1 / ps / FPS_SRC:5.1f} s | a 25 fps: P(empieza) "
              f"{model[s]['p_empieza']:.4f} P(acaba) {model[s]['p_acaba']:.4f}")
    pf = np.array([[c[s]["falta"] / c[s]["n"] for s in ("vivo", "D")] for _, c in per_match])
    print(f"  dispersión entre partidos de P(falta): vivo {pf[:, 0].min():.3f}-{pf[:, 0].max():.3f}, "
          f"D {pf[:, 1].min():.3f}-{pf[:, 1].max():.3f}; parado {tot['D']['n'] / (tot['D']['n'] + tot['vivo']['n']):.1%}")
    model["fuente"] = f"SkillCorner open data, {len(folders)} partidos (skillcorner_sha256.txt)"
    with open(OUT, "w") as f:
        json.dump(model, f, indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
