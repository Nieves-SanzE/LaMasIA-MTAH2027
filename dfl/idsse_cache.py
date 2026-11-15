"""Lee los XML de posiciones de la DFL (IDSSE, CC-BY 4.0) y guarda arrays por partido y parte.

    .venv\Scripts\python.exe idsse_cache.py            # todos los partidos descargados

Entrada: MTAH_DATA_DIR\\external\\idsse\\ (download_idsse.sh). Salida: MTAH_DATA_DIR\\cache\\idsse\\<id>.npz
con, por parte h in {1, 2}: ball_h (n, 3) x/y/z en m con origen en el centro, poss_h (n,) 1 local /
2 visitante / 0 sin dato, status_h (n,) 1 vivo / 0 muerto, players_h (n, P, 2) con NaN fuera del
campo, y team (P,) 0 local / 1 visitante. Árbitros fuera. 25 fps.
"""
import glob
import os
import re

import numpy as np
from lxml import etree

SRC = os.path.join(os.environ["MTAH_DATA_DIR"], "external", "idsse")
OUT = os.path.join(os.environ["MTAH_DATA_DIR"], "cache", "idsse")
HALVES = {"firstHalf": 1, "secondHalf": 2}


def home_team_id(match_id):
    path = glob.glob(os.path.join(SRC, "DFL_02_01_matchinformation_*%s.xml" % match_id))[0]
    with open(path, encoding="utf-8") as f:
        return re.search(r'HomeTeamId="([^"]+)"', f.read()).group(1)


def parse(match_id):
    path = glob.glob(os.path.join(SRC, "DFL_04_03_positions_raw_observed_*%s.xml" % match_id))[0]
    home = home_team_id(match_id)
    sets = {}  # (half, team, person) -> (N, X, Y, Z, poss, status)
    key, rows = None, None
    for event, el in etree.iterparse(path, events=("start", "end"), tag=("FrameSet", "Frame")):
        if el.tag == "FrameSet" and event == "start":
            team = el.get("TeamId")
            key = (HALVES[el.get("GameSection")], team, el.get("PersonId")) if team != "referee" else None
            rows = []
        elif el.tag == "Frame" and event == "end":
            if key is not None:
                g = el.get
                rows.append((int(g("N")), float(g("X")), float(g("Y")), float(g("Z") or 0.0),
                             int(g("BallPossession") or 0), int(g("BallStatus") or 0)))
            el.clear()
        elif el.tag == "FrameSet" and event == "end":
            if key is not None:
                sets[key] = np.array(rows, dtype=np.float64)
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]

    out = {}
    persons = sorted({(t, p) for (_, t, p) in sets if t != "BALL"})
    out["team"] = np.array([0 if t == home else 1 for t, _ in persons], dtype=np.int8)
    for half in (1, 2):
        ball = sets[next(k for k in sets if k[0] == half and k[1] == "BALL")]
        n0, n = int(ball[:, 0].min()), int(ball[:, 0].max() - ball[:, 0].min()) + 1
        b = np.full((n, 3), np.nan)
        poss = np.zeros(n, dtype=np.int8)
        status = np.zeros(n, dtype=np.int8)
        i = ball[:, 0].astype(int) - n0
        b[i] = ball[:, 1:4]
        poss[i], status[i] = ball[:, 4], ball[:, 5]
        players = np.full((n, len(persons), 2), np.nan, dtype=np.float32)
        for j, (t, p) in enumerate(persons):
            rows = sets.get((half, t, p))
            if rows is None:
                continue
            k = rows[:, 0].astype(int) - n0
            ok = (k >= 0) & (k < n)
            players[k[ok], j] = rows[ok, 1:3]
        out["ball_%d" % half], out["poss_%d" % half] = b, poss
        out["status_%d" % half], out["players_%d" % half] = status, players
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "DFL_04_03_positions_raw_observed_*.xml")))
    for path in files:
        match_id = re.search(r"DFL-MAT-(\w+)\.xml", path).group(1)
        dst = os.path.join(OUT, match_id + ".npz")
        if os.path.exists(dst):
            print("ya existe", match_id)
            continue
        data = parse(match_id)
        np.savez_compressed(dst, **data)
        n = sum(len(data["status_%d" % h]) for h in (1, 2))
        print("%s: %d frames (%.1f min), %d jugadores" % (match_id, n, n / 1500.0, len(data["team"])))


if __name__ == "__main__":
    main()
