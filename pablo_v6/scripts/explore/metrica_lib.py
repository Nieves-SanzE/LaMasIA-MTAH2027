"""Metrica Sports sample data (3 public matches, other teams) as external labelled data for the
live/dead-ball model (ideas V4 / L1 / D-T5). The rules allow human labels of OTHER matches.

Source: https://github.com/metrica-sports/sample-data (downloaded to data/external/metrica/,
gitignored). Tracking at 25 fps, normalised coords with (0,0) top-left, pitch 105 x 68 m.

    from metrica_lib import load_game
    g = load_game(1)
    g.xy (N, S, 2) metres, g.team (S,) 0/1, g.ball (N, 2) metres, g.period (N,) 0 = none,
    g.dead (N,) bool  dead ball derived from the events (see dead_intervals)
"""
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MDIR = os.path.join(ROOT, "data", "external", "metrica", "data")
CACHE = os.path.join(ROOT, "out", "explore", "metrica")
PITCH = np.array([105.0, 68.0], np.float32)
FPS = 25


@dataclass
class Game:
    gid: int
    xy: np.ndarray       # (N, S, 2) metres, NaN = not on the pitch
    team: np.ndarray     # (S,) 0 = home / team A, 1 = away / team B
    ball: np.ndarray     # (N, 2) metres, NaN = unknown
    period: np.ndarray   # (N,) int8, 1/2 in play periods, 0 elsewhere
    events: pd.DataFrame  # columns: type, subtype, team, start, end (0-based frames), period
    dead: np.ndarray = None
    dead_eps: list = None


# ----------------------------------------------------------------------------------- loading

def _csv_team(path):
    raw = pd.read_csv(path, skiprows=2, low_memory=False)
    cols = list(raw.columns)
    players = [c for c in cols if c.startswith("Player")]
    xy = np.stack([np.stack([raw[c].values, raw[cols[cols.index(c) + 1]].values], -1) for c in players], 1)
    ball = np.stack([raw["Ball"].values, raw[cols[cols.index("Ball") + 1]].values], -1)
    return raw["Period"].values, xy.astype(np.float32), ball.astype(np.float32)


def _load_csv_game(gid):
    d = os.path.join(MDIR, f"Sample_Game_{gid}")
    per, xh, ball = _csv_team(os.path.join(d, f"Sample_Game_{gid}_RawTrackingData_Home_Team.csv"))
    _, xa, _ = _csv_team(os.path.join(d, f"Sample_Game_{gid}_RawTrackingData_Away_Team.csv"))
    xy = np.concatenate([xh, xa], 1)
    team = np.array([0] * xh.shape[1] + [1] * xa.shape[1], np.int8)
    ev = pd.read_csv(os.path.join(d, f"Sample_Game_{gid}_RawEventsData.csv"))
    ev = pd.DataFrame(dict(type=ev["Type"], subtype=ev["Subtype"].fillna(""),
                           team=ev["Team"].map({"Home": 0, "Away": 1}),
                           start=ev["Start Frame"] - 1, end=ev["End Frame"] - 1, period=ev["Period"]))
    return xy, team, ball, per.astype(np.int8), ev


def _load_epts_game3():
    d = os.path.join(MDIR, "Sample_Game_3")
    root = ET.parse(os.path.join(d, "Sample_Game_3_metadata.xml")).getroot()
    team_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    teams = sorted(set(team_of.values()))
    chan = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    pids = sorted(team_of)
    slot = {p: i for i, p in enumerate(pids)}
    specs = []
    for s in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in s.iter("PlayerChannelRef")]
        specs.append((int(s.get("startFrame")), int(s.get("endFrame")), [chan[r] for r in refs[::2]]))
    lines = open(os.path.join(d, "Sample_Game_3_tracking.txt")).read().split("\n")
    lines = [ln for ln in lines if ln]
    n = max(int(ln.split(":", 1)[0]) for ln in lines)
    xy = np.full((n, len(pids), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    for ln in lines:
        f, pl, b = ln.split(":")
        f = int(f)
        spec = next(sp for sp in specs if sp[0] <= f <= sp[1])[2]
        vals = np.array(pl.replace(";", ",").split(","), np.float32).reshape(-1, 2)
        for p, v in zip(spec, vals):
            xy[f - 1, slot[p]] = v
        ball[f - 1] = np.array(b.split(","), np.float32)
    team = np.array([teams.index(team_of[p]) for p in pids], np.int8)
    raw = json.load(open(os.path.join(d, "Sample_Game_3_events.json")))["data"]

    def sub(e):
        s = e.get("subtypes")
        if not s:
            return ""
        return s["name"] if isinstance(s, dict) else "-".join(x["name"] for x in s)
    ev = pd.DataFrame(dict(type=[e["type"]["name"] for e in raw], subtype=[sub(e) for e in raw],
                           team=[teams.index(e["team"]["id"]) if e.get("team") else np.nan for e in raw],
                           start=[e["start"]["frame"] - 1 for e in raw],
                           end=[(e["end"]["frame"] or e["start"]["frame"]) - 1 for e in raw],
                           period=[e["period"] for e in raw]))
    # periods from the events (first / last event frame of each period)
    per = np.zeros(n, np.int8)
    for p, g in ev.groupby("period"):
        per[int(g.start.min()):int(max(g.start.max(), g.end.max())) + 1] = p
    return xy, team, ball, per, ev


def load_game(gid):
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"game{gid}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        g = Game(gid, z["xy"], z["team"], z["ball"], z["period"], pd.DataFrame(z["events"].item()))
    else:
        if gid == 3:
            xy, team, ball, per, ev = _load_epts_game3()
        else:
            xy, team, ball, per, ev = _load_csv_game(gid)
        xy = xy * PITCH
        ball = ball * PITCH
        g = Game(gid, xy, team, ball, per, ev)
        np.savez_compressed(cache, xy=xy, team=team, ball=ball, period=per, events=np.array(ev.to_dict("list")))
    g.dead, g.dead_eps = dead_intervals(g)
    return g


# ----------------------------------------------------------------------------------- dead ball

def _is_stop(t, s):
    s = s.upper()
    if t == "BALL OUT":
        return "end"
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return "end"
    if t in ("FAULT RECEIVED",) or (t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s):
        return "start"
    if t == "BALL LOST" and ("OFFSIDE" in s or "HAND BALL" in s or "END HALF" in s):
        return "end"
    return None


def _is_restart(t, s):
    return t == "SET PIECE" or ("GOAL KICK" in s.upper() and "INTERCEPTION" not in s.upper())


LIVE = ("PASS", "CARRY", "RECOVERY", "BALL LOST", "CHALLENGE", "SHOT", "BALL OUT")


def dead_intervals(g):
    """Dead ball from the events: [stop, restart) where restart = a SET PIECE or a goal kick and stop =
    the last stopping event before it (ball out end frame, shot out/goal end frame, foul start frame,
    offside / hand ball) with no live event in between; if there is none, the end of the last
    live event before the restart. The time before each period's first event and after its last
    is outside play (period 0). Returns (dead mask, [(start, stop, kind)])."""
    ev = g.events.sort_values(["start", "end"]).reset_index(drop=True)
    n = len(g.period)
    dead = np.zeros(n, bool)
    eps = []
    for i, r in ev.iterrows():
        if not _is_restart(r.type, r.subtype) or "KICK OFF" in r.subtype.upper():
            continue
        stop, kind = None, "fallback"
        for j in range(i - 1, -1, -1):
            e = ev.loc[j]
            if e.period != r.period:
                break
            if e.start >= r.start:  # e.g. the keeper's RECOVERY logged on the goal-kick frame
                continue
            if _is_restart(e.type, e.subtype):
                break
            st = _is_stop(e.type, e.subtype)
            if st is not None:
                stop, kind = int(e.end if st == "end" else e.start), e.type
                break
            if e.type in LIVE:
                stop = int(max(e.start, e.end))
                break
        if stop is None or stop >= r.start:
            continue
        eps.append((stop, int(r.start), kind, r.subtype))
        dead[stop:int(r.start)] = True
    # kick-offs after goals: dead from the goal to the kick-off (covered by the shot stop above)
    for i, r in ev.iterrows():
        if r.type == "SET PIECE" and "KICK OFF" in r.subtype.upper():
            for j in range(i - 1, max(i - 6, -1), -1):
                e = ev.loc[j]
                if e.period == r.period and e.type == "SHOT" and "GOAL" in e.subtype.upper():
                    dead[int(e.end):int(r.start)] = True
                    eps.append((int(e.end), int(r.start), "GOAL", "KICK OFF"))
                    break
    return dead, eps


# ----------------------------------------------------------------------------------- possession

OWN = ("PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT")


def possession_codes(g):
    """Event-derived H/A/D/X codes (0 = team 0, 1 = team 1, 2 = dead, 3 = outside play): the team of
    the last owning event (pass, carry, recovery, set piece, shot) since its start frame. Only a
    proxy GT to measure how a D layer interacts with spells (exp-019); not used for training."""
    n = len(g.period)
    owner = np.full(n, -1, np.int8)
    ev = g.events[g.events.type.isin(OWN) & g.events.team.notna()].sort_values("start")
    for s, t in zip(ev.start.values.astype(int), ev.team.values.astype(int)):
        owner[s] = t
    owner = pd.Series(np.where(owner >= 0, owner, np.nan)).ffill().bfill().values.astype(np.int8)
    codes = np.where(g.dead, 2, owner).astype(np.int8)
    codes[g.period == 0] = 3
    return codes
