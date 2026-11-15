"""E5 / exp-139, paso 2: exporta las secuencias a 5 Hz para la red temporal (tcnlib.py). Solo imprime formas y agregados.

    cd pablo_v6
    $env:OMP_NUM_THREADS="4"; .venv\\Scripts\\python.exe -u ..\\dfl\\tcn_export.py

Escribe en MTAH_DATA_DIR/cache/tcn/:
  - dfl_train_<partido>.npz: features de entrenamiento de e1_v13 (feat_sc: DFL degradada nivel 2, semilla 0, huecos de
    SkillCorner), etiqueta control 1 s + cambio a mitad del vuelo (dfllib.official_style, como make_e1.team_training),
    pesos 1 / max(duración de la posesión, 2 s) (como make_e1.spell_weights, pero en todos los frames vivos) y la
    auxiliar de cambio;
  - judge_<suave|duro>_<partido>.npz: features de los jueces DFL de exp-133 (stress/l3sc_*), columnas por nombre;
  - barath.npz: features de BAR-ATH de e1_v13 (make_e1.barath_features: limpieza de porteros), bloques oficiales.
    Sin etiquetas ni verdad: la verdad pública queda bloqueada (forbid_public_truth).

Definiciones (fijadas antes de mirar nada, 14/11):
  - cada parte se trocea en bloques de 5 frames desde su inicio (el último puede ser más corto); paso = bloque;
  - canales: media (nanmean) de las 166 features sin balón y de las 54 con balón; fracción de frames con balón visible;
    fracción de NaN de las 166 sin balón (sus nombres no traen la familia) y fracción de NaN de cada familia con balón
    (prefijo «familia:» del nombre). Las 220 primeras columnas se normalizan en tcnlib; las demás no;
  - etiqueta del paso = la del frame central (inicio + 2, o el último si el bloque es más corto); el paso es vivo solo
    si sus 5 frames son H o A; y = 1 si A, 0 si H, -1 si no es vivo; peso = el del frame central;
  - cambio (auxiliar): frame H/A cuya etiqueta difiere de la última H/A anterior de la misma parte (también tras un
    parón); el paso vale 1 si hay un cambio a <= 0,5 s (12 frames) de su frame central.
"""
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import make_e1 as e1  # noqa: E402  (añade las rutas de la v6)

import numpy as np  # noqa: E402

import ctxlib  # noqa: E402
import dfllib  # noqa: E402
import featlib  # noqa: E402

OUT = os.path.join(dfllib.DATA, "cache", "tcn")
STRESS = os.path.join(dfllib.CACHE, "stress")
SCEN = ("suave", "duro")
B = 5
CHG_F = 12  # 0,5 s a 25 fps


def families(nb):
    fams = sorted({n.split(":", 1)[0] for n in nb})
    assert all(":" in n for n in nb), "una feature con balón sin familia en el nombre"
    return fams


def by_name(X, names, want):
    idx = {str(n): i for i, n in enumerate(names)}
    return X[:, [idx[w] for w in want]]


def to5(X, a, b):
    """Media de cada bloque de 5 frames de X[a:b] (NaN si el bloque entero es NaN)."""
    n = b - a
    k = -(-n // B)
    Z = np.full((k * B,) + X.shape[1:], np.nan, np.float32)
    Z[:n] = X[a:b]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(Z.reshape((k, B) + X.shape[1:]), 1)


def nanfrac5(X, a, b):
    n = b - a
    k = -(-n // B)
    Z = np.ones((k * B,), np.float32)
    Z[:n] = np.isnan(X[a:b]).mean(1)
    return Z.reshape(k, B).mean(1)


def channels(Xf, Xb, vis, nb, fams, a, b):
    cols = [to5(Xf, a, b), to5(Xb, a, b), to5(vis.astype(np.float32)[:, None], a, b), nanfrac5(Xf, a, b)[:, None]]
    for f in fams:
        j = [i for i, n in enumerate(nb) if n.split(":", 1)[0] == f]
        cols.append(nanfrac5(Xb[:, j], a, b)[:, None])
    return np.concatenate(cols, 1).astype(np.float32)


def centers(a, b):
    s = np.arange(a, b, B)
    return np.minimum(s + 2, b - 1), s


def labels25(mid):
    """Etiqueta control 1 s (como make_e1.team_training), pesos de la posesión en todos los frames vivos y cambios."""
    d = dfllib.load(mid)
    ball, pl = dfllib.stacked(d, "ball")[:, :2], dfllib.stacked(d, "players").astype(np.float64)
    gt = dfllib.official_style(dfllib.gt_labels(d), ball, pl, d["team"], 1.0, True)
    pm = dfllib.pm
    codes = pm.label_codes(gt)
    dur = np.zeros(len(gt))
    for a, b, _ in pm.possession_spells(codes, pm.valid_runs(codes)):
        dur[a:b] = (b - a) / featlib.FPS
    live = (gt == "H") | (gt == "A")
    assert (dur[live] > 0).all(), "frame vivo fuera de una posesión"
    w = np.where(live, 1.0 / np.maximum(dur, 2.0), 0.0)
    chg = np.zeros(len(gt), bool)
    for a, b in ctxlib.dfl_periods(mid):
        last = ""
        for t in np.flatnonzero(live[a:b]) + a:
            if last and gt[t] != last:
                chg[t] = True
            last = gt[t]
    return gt, live, w, chg


def export_train(mid, nf, nb, fams, prefix="", swap=False):
    gt, live, w, chg = labels25(mid)
    if swap:  # exp-149 (P4): equipos intercambiados (las features vienen de la copia con los equipos intercambiados)
        gt = np.where(gt == "H", "A", np.where(gt == "A", "H", gt))
    Xf, Xb, nf_m, nb_m, vis = e1.load_train(mid)
    assert nf_m == nf and nb_m == nb, "columnas distintas entre partidos"
    out, st = {}, []
    cum = np.r_[0, np.cumsum(chg)]
    for i, (a, b) in enumerate(ctxlib.dfl_periods(mid)):
        c, s = centers(a, b)
        e = np.minimum(s + B, b)
        cl = np.r_[0, np.cumsum(live)]
        all_live = (cl[e] - cl[s]) == (e - s)
        y = np.where(all_live, (gt[c] == "A").astype(np.int8), -1).astype(np.int8)
        near = (cum[np.minimum(c + CHG_F + 1, len(gt))] - cum[np.maximum(c - CHG_F, 0)]) > 0
        out.update({f"X{i}": channels(Xf, Xb, vis, nb, fams, a, b), f"y{i}": y,
                    f"w{i}": np.where(all_live, w[c], 0).astype(np.float32), f"chg{i}": near.astype(np.int8),
                    f"ab{i}": np.array([a, b])})
        st.append((len(y), np.mean(y >= 0), np.mean(y[y >= 0] == 1), np.mean(near[y >= 0])))
    np.savez(os.path.join(OUT, f"dfl_train_{prefix}{mid}.npz"), **out)
    n = sum(x[0] for x in st)
    print(f"{mid}: {n} pasos ({n / 5 / 60:.0f} min), vivos {np.mean([x[1] for x in st]):.1%}, "
          f"A entre los vivos {np.mean([x[2] for x in st]):.1%}, pasos con cambio cerca {np.mean([x[3] for x in st]):.1%}, "
          f"canales {out['X0'].shape[1]}", flush=True)


def export_judge(scen, mid, nf, nb, fams):
    z = np.load(os.path.join(STRESS, f"l3sc_{scen}", mid + ".npz"))
    Xf, Xb = by_name(z["Xf"], z["nf"], nf), by_name(z["Xb"], z["nb"], nb)
    out = {}
    for i, (a, b) in enumerate(map(tuple, z["periods"].astype(int))):
        out[f"X{i}"] = channels(Xf, Xb, z["vis"], nb, fams, a, b)
        out[f"ab{i}"] = np.array([a, b])
    np.savez(os.path.join(OUT, f"judge_{scen}_{mid}.npz"), **out)


def barath_inputs():
    e1.use_feat_dir(e1.CONFIG_V13)
    halves = dfllib.official_halves()
    featlib.PLAY = halves
    tr = featlib.load_tracking()
    X_free, X_ball = e1.barath_features(tr, halves)
    return X_free, X_ball, np.asarray(tr.balon_visible), halves


OFFSETS = (1, 2, 3, 4)


def export_offsets():
    """exp-140 (mejora 1): los mismos canales con la rejilla de 5 Hz desplazada o = 1..4 frames dentro de cada parte
    (bloques [a + o + 5k, a + o + 5k + 5)); los o primeros frames de la parte no tienen bloque propio (la interpolación
    les da el valor del primero). Jueces suave / duro y BAR-ATH; claves X<parte>_o<o> y ab<parte>_o<o>."""
    dfllib.forbid_public_truth()
    e1.use_feat_dir(e1.CONFIG_V13)
    mids = dfllib.match_ids()
    _, _, nf, nb, _ = e1.load_train(mids[0])
    fams = families(nb)
    for mid in mids:
        for scen in SCEN:
            z = np.load(os.path.join(STRESS, f"l3sc_{scen}", mid + ".npz"))
            Xf, Xb = by_name(z["Xf"], z["nf"], nf), by_name(z["Xb"], z["nb"], nb)
            out = {}
            for i, (a, b) in enumerate(map(tuple, z["periods"].astype(int))):
                for o in OFFSETS:
                    out[f"X{i}_o{o}"] = channels(Xf, Xb, z["vis"], nb, fams, a + o, b)
                    out[f"ab{i}_o{o}"] = np.array([a + o, b])
            np.savez(os.path.join(OUT, f"judge_{scen}_{mid}_offsets.npz"), **out)
        print(f"{mid}: desfases 1-4 de suave y duro exportados", flush=True)
    X_free, X_ball, vis, halves = barath_inputs()
    out = {}
    for i, (a, b) in enumerate(halves):
        for o in OFFSETS:
            out[f"X{i}_o{o}"] = channels(X_free, X_ball, vis, nb, fams, a + o, b)
            out[f"ab{i}_o{o}"] = np.array([a + o, b])
    np.savez(os.path.join(OUT, "barath_offsets.npz"), **out)
    print("BAR-ATH: desfases 1-4 exportados", flush=True)


def export_barath(nf, nb, fams):
    X_free, X_ball, vis, halves = barath_inputs()
    out = {}
    for i, (a, b) in enumerate(halves):
        out[f"X{i}"] = channels(X_free, X_ball, vis, nb, fams, a, b)
        out[f"ab{i}"] = np.array([a, b])
    np.savez(os.path.join(OUT, "barath.npz"), **out)
    X = np.concatenate([out["X0"], out["X1"]])
    print(f"BAR-ATH: {len(X)} pasos en 2 partes, canales {X.shape[1]}; NaN en las 220 features "
          f"{np.isnan(X[:, :220]).mean():.1%}; balón visible medio {np.nanmean(X[:, 220]):.1%}", flush=True)


def write_meta():
    """meta.npz (nombres de los canales, columnas normalizadas, partidos) a partir de las features de e1_v13."""
    dfllib.forbid_public_truth()
    os.makedirs(OUT, exist_ok=True)
    e1.use_feat_dir(e1.CONFIG_V13)
    mids = dfllib.match_ids()
    _, _, nf, nb, _ = e1.load_train(mids[0])
    fams = families(nb)
    names = nf + nb + ["vis_frac", "nan_free"] + [f"nan_{f}" for f in fams]
    np.savez(os.path.join(OUT, "meta.npz"), names=np.array(names), n_norm=len(nf) + len(nb), mids=np.array(mids))
    print(f"canales: {len(nf)} sin balón + {len(nb)} con balón + 1 visible + 1 NaN sin balón + {len(fams)} NaN por "
          f"familia ({', '.join(fams)}) = {len(names)}", flush=True)
    return mids, nf, nb, fams


def export_entrega():
    """Solo lo que necesita la entrega (regenerar_e1.sh): meta.npz y las secuencias de BAR-ATH (barath.npz), que la red
    de producción predice con sus pesos guardados (tcn_train.py prod_pesos). No exporta el entrenamiento ni los jueces."""
    _, nf, nb, fams = write_meta()
    export_barath(nf, nb, fams)


def main():
    mids, nf, nb, fams = write_meta()
    for mid in mids:
        export_train(mid, nf, nb, fams)
        for s in SCEN:
            export_judge(s, mid, nf, nb, fams)
    print("jueces suave y duro exportados (7 partidos)", flush=True)
    export_barath(nf, nb, fams)


def export_train_version(version, data):
    """exp-138 / exp-142: solo las secuencias de entrenamiento de la DFL, con las features de otra versión de make_e1
    (b2: oclusiones de SkillCorner; b2c1: su segunda copia degradada) -> dfl_train_<data>_<partido>.npz. Mismas
    etiquetas, pesos y canales (comprobado contra meta.npz); jueces y BAR-ATH no cambian."""
    dfllib.forbid_public_truth()
    e1.use_feat_dir(e1.VERSIONS[version])
    meta = np.load(os.path.join(OUT, "meta.npz"))
    mids = [str(m) for m in meta["mids"]]
    _, _, nf, nb, _ = e1.load_train(mids[0])
    fams = families(nb)
    names = nf + nb + ["vis_frac", "nan_free"] + [f"nan_{f}" for f in fams]
    assert names == [str(x) for x in meta["names"]], "canales distintos de meta.npz"
    print(f"versión {version} ({e1.FEAT_DIR}; oclusión {dfllib.OCCLUSION}) -> dfl_train_{data}_*.npz", flush=True)
    for mid in mids:
        export_train(mid, nf, nb, fams, prefix=f"{data}_", swap=bool(e1.VERSIONS[version].get("swap_teams", False)))


if __name__ == "__main__":
    if sys.argv[1:2] == ["train"]:  # train <versión de make_e1> <nombre de los datos>
        export_train_version(sys.argv[2], sys.argv[3])
    elif sys.argv[1:] == ["entrega"]:  # solo meta.npz y barath.npz (regenerar_e1.sh)
        export_entrega()
    else:
        export_offsets() if sys.argv[1:] == ["offsets"] else main()
