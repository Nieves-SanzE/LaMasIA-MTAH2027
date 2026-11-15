"""exp-026 (V5), step 1: our team features on the 3 public Metrica matches.

Each Metrica match is wrapped in an atd.Tracking (normalised by atd.CAMPO, so `metros` gives the
Metrica metres) and pushed through the SAME feature code as our match by pointing the shared module
globals (featlib.HALVES / PLAY / _TR) at it:
  FREE (ball-free): the five fe_geometry families (goalside, shape, convergence, graph_intra, space).
        The fe_temporal families of v3 (leadlag, sync, tda, multiscale) are NOT ported (module-level
        script, costly); exp-012 found the families redundant (the core is dispersion and advance).
  BALL: the five fb_ball families (ball_pressure, ball_structure, ball_space, ball_graph,
        ball_motion), computed on every frame where the Metrica ball exists; gaps are simulated
        later by masking (step 2).
Saves out/explore/exp026_game{gid}.npz (features + what the labels need). Aggregates only.
    OMP_NUM_THREADS=8 uv run python scripts/explore/exp026_metrica_features.py [gid ...]
"""
import os
import sys
import time

import numpy as np

import featlib
from metrica_lib import load_game
from mtah_2027 import atd

T0 = time.time()
OUT = os.path.join(featlib.ROOT, "out", "explore")
FB_SRC = os.path.join(featlib.ROOT, "scripts", "explore", "fb_ball.py")


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def periods_of(g):
    out = []
    for p in (1, 2):
        idx = np.flatnonzero(g.period == p)
        out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def as_tracking(g, min_slots=14):
    """fe_geometry packs each team into 14 columns: pad teams with fewer slots with empty (NaN) ones."""
    n = len(g.period)
    xy, team = g.xy, g.team.astype(np.int8)
    for k in (0, 1):
        pad = min_slots - int(np.sum(team == k))
        if pad > 0:
            xy = np.concatenate([xy, np.full((n, pad, 2), np.nan, np.float32)], 1)
            team = np.concatenate([team, np.full(pad, k, np.int8)])
    return atd.Tracking(xy=(xy / atd.CAMPO).astype(np.float32), velocidad=np.full(xy.shape[:2], np.nan, np.float32),
                        balon=(g.ball / atd.CAMPO).astype(np.float32), balon_velocidad=np.full(n, np.nan, np.float32),
                        equipo=team, nombres=("home", "away"))


def point_globals_at(tr, periods):
    featlib._TR = tr
    featlib.HALVES = periods
    featlib.PLAY = periods
    featlib.LO = featlib.HI = 10 ** 9  # no public-window exclusion on Metrica


def free_families(tr):
    import fe_geometry as fg
    fg.HALVES = featlib.HALVES
    n = tr.n_frames
    team, ok = fg.prepare(tr)
    fams = [fg.fam_goalside(team, ok, n)]
    sub = fg.sub_features(team, ok, n)
    fams += [fg.fam_shape(team, ok, n, sub), fg.fam_convergence(team, ok, n), fg.fam_graph(team, ok, n, sub),
             fg.fam_space(sub, ok)]
    X = np.concatenate([f[0] for f in fams], 1).astype(np.float32)
    names = sum([list(f[1]) for f in fams], [])
    return X, names


def ball_families():
    """Run fb_ball.compute_families on featlib._TR (only the feature part of that script)."""
    src = open(FB_SRC).read()
    src = src[:src.index('if "--cached" in sys.argv')]
    ns = {"__name__": "fb_ball_metrica", "__file__": FB_SRC}
    exec(compile(src, FB_SRC, "exec"), ns)
    fams = ns["compute_families"]()
    names = sum([[f"{k}:{n}" for n in fams[k][1]] for k in fams], [])
    return np.concatenate([fams[k][0] for k in fams], 1).astype(np.float32), names


def main(gids):
    for gid in gids:
        g = load_game(gid)
        periods = periods_of(g)
        tr = as_tracking(g)
        point_globals_at(tr, periods)
        log(f"game {gid}: {tr.n_frames} frames, {tr.n_slots} slots, periods {periods}, "
            f"ball present {np.mean(tr.balon_visible[g.period > 0]):.1%} of play")
        Xf, nf = free_families(tr)
        log(f"game {gid}: free features {Xf.shape}")
        Xb, nb = ball_families()
        log(f"game {gid}: ball features {Xb.shape}")
        d = tr.distancia_al_balon()
        dmin = np.nanmin(np.where(np.isfinite(d), d, np.inf), 1)
        near = tr.mas_cercano_al_balon()
        near_team = np.where(near >= 0, tr.equipo[np.clip(near, 0, None)], -1).astype(np.int8)
        sign = featlib.attack_sign(tr)
        np.savez_compressed(os.path.join(OUT, f"exp026_game{gid}.npz"), X_free=Xf, X_ball=Xb,
                            names_free=np.array(nf), names_ball=np.array(nb), dmin=dmin.astype(np.float32),
                            near_team=near_team, periods=np.array(periods), attack_sign=sign)
        log(f"game {gid}: saved (team 0 attacks +x in {np.mean(sign[g.period > 0] > 0):.0%} of play)")
        del Xf, Xb, d


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1, 2, 3])
