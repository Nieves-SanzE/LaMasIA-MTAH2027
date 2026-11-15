"""E5 / exp-139: entrenamiento e inferencia de la red temporal (tcnlib) sobre las secuencias de tcn_export.py.
Se ejecuta en dfl/.venv-tcn. Solo imprime pérdidas y agregados; no lee ninguna verdad de los jueces.

    cd dfl
    .venv-tcn\\Scripts\\python.exe -u tcn_train.py pilot              # paso 3: pliegue del 1er partido, semilla 0
    .venv-tcn\\Scripts\\python.exe -u tcn_train.py fold <partido> [--placebo]   # paso 4: un pliegue, 3 semillas
    .venv-tcn\\Scripts\\python.exe -u tcn_train.py prod               # producción: 7 partidos, 3 semillas, BAR-ATH
    .venv-tcn\\Scripts\\python.exe -u tcn_train.py prod_pesos pesos_red   # la entrega: BAR-ATH con los pesos guardados

Pliegue del partido J (el del juez): entrenamiento = los otros 6, en el orden de match_ids. Validación interna rotada
por semilla (fijado con Nieves el 14/11): la semilla s valida en el partido s de esa lista y entrena con los otros 5;
la parada temprana y la temperatura salen de ese partido. El partido J no se carga nunca para entrenar (assert).
Producción: igual con los 7 partidos (la semilla s valida en el partido s).
Placebo: la etiqueta, el peso y la auxiliar de cada parte de cada partido (entrenamiento y validación interna) se
desplazan circularmente media parte (>= 10 min); las features no se tocan.
Salidas en MTAH_DATA_DIR/cache/tcn/out/: logits a 5 Hz calibrados y redondeados a 3 decimales por semilla y parte, y
el prior ponderado de cada semilla; en producción, además, los pesos de cada semilla (fichero intermedio de la entrega).
"""
import os
import sys
import time

import numpy as np

import tcnlib as T

# exp-147 (P1): «--dir <carpeta>» lee y escribe todo (meta, secuencias, jueces, BAR-ATH y salidas) en cache/<carpeta>
# en vez de cache/tcn (otros canales de entrada sin tocar los de E5). Por defecto, tcn: sin cambios.
_DIRNAME = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "tcn"
DIR = os.path.join(os.environ["MTAH_DATA_DIR"], "cache", _DIRNAME)
OUT = os.path.join(DIR, "out")
SEEDS = (0, 1, 2)
SCEN = ("suave", "duro")


def meta():
    z = np.load(os.path.join(DIR, "meta.npz"))
    return [str(m) for m in z["mids"]], int(z["n_norm"]), len(z["names"])


def train_seqs(mid, placebo=False, data=None):
    data = DATA if data is None else data
    z = np.load(os.path.join(DIR, f"dfl_train_{data + '_' if data else ''}{mid}.npz"))
    out = []
    for i in (0, 1):
        s = dict(X=z[f"X{i}"], y=z[f"y{i}"], w=z[f"w{i}"], chg=z[f"chg{i}"])
        if placebo:
            k = len(s["y"]) // 2
            assert k >= 10 * 60 * 5, "parte demasiado corta para el placebo"
            s.update(y=np.roll(s["y"], k), w=np.roll(s["w"], k), chg=np.roll(s["chg"], k))
        out.append(s)
    return out


def x_parts(path):
    z = np.load(path)
    return [z["X0"], z["X1"]]


def offset_parts(path):
    """exp-140: {(o, parte): X} de un fichero *_offsets.npz de tcn_export.py (vacío si no existe)."""
    if not os.path.exists(path):
        return {}
    z = np.load(path)
    return {(int(k.split("_o")[1]), int(k[1])): z[k] for k in z.files if k.startswith("X")}


def opt(name, default):
    """--margen N / --semillas-desde N / --etiqueta X (exp-141)."""
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


MARGIN = opt("--margen", 0)        # exp-141: 126 = medio campo receptivo; 0 = E5
SEED0 = opt("--semillas-desde", 0)  # exp-141 placebo: semillas 3-5 con las mismas validaciones internas
TAG = opt("--etiqueta", "")
DATA = opt("--datos", "")           # exp-138 (B2): "occ" = dfl_train_occ_<p>.npz (oclusiones de SkillCorner)
COPIES = opt("--copias", 1)         # exp-142: 2 = el entrenamiento añade la segunda copia (dfl_train_<datos>1_<p>.npz)
SUF = "_" + TAG if TAG else ""   # producción de otra versión (p. ej. b2): prod_b2.npz y prod_b2_weights_s*.npz
assert COPIES in (1, 2) and (COPIES == 1 or DATA), "--copias 2 necesita --datos"
# exp-150 (B2c): «--val-ambas 1» valida también con la copia 1 del partido de validación interna (cuando las dos copias
# son dos degradaciones distintas, la parada temprana no favorece a una). 0 = como siempre (solo la copia 0).
VAL_BOTH = opt("--val-ambas", 0)
assert VAL_BOTH in (0, 1) and (VAL_BOTH == 0 or COPIES == 2), "--val-ambas 1 necesita --copias 2"
# exp-151 (TTA): «--tta» predice además las copias perturbadas de los jueces (judge_<juez>_<p>_tta.npz). Sin cambios en el
# entrenamiento: solo se añaden salidas.
TTA = "--tta" in sys.argv
# exp-152 (pseudo-etiquetas de SkillCorner): «--pred-skc» predice además las entradas de los partidos de SkillCorner
# (skc_x_<id>.npz, claves X0 / X1) con la red de cada semilla: es el profesor del pliegue (no vio el partido del juez).
PRED_SKC = "--pred-skc" in sys.argv
# exp-154: «--suavizado 0.1» = suavizado de etiquetas en la pérdida de entrenamiento de la red (tcnlib.LABEL_SMOOTH).
# La validación interna, la parada y la calibración siguen con las etiquetas duras. 0 = E5 sin cambios.
T.LABEL_SMOOTH = opt("--suavizado", 0.0)
# exp-148 (P2): «--extra sc» añade al ENTRENAMIENTO (nunca a la validación interna ni al juez) todas las secuencias
# dfl_train_sc_<partido>.npz de DIR (partidos de SkillCorner; «scp» = su placebo). Vacío = E5 sin cambios.
EXTRA = opt("--extra", "")


def extra_seqs():
    if not EXTRA:
        return []
    import glob
    files = sorted(glob.glob(os.path.join(DIR, f"dfl_train_{EXTRA}_*.npz")))
    assert files, f"no hay secuencias {EXTRA}"
    out = []
    for f in files:
        z = np.load(f)
        out += [dict(X=z[f"X{i}"], y=z[f"y{i}"], w=z[f"w{i}"], chg=z[f"chg{i}"]) for i in (0, 1)]
    return out


def fit_seed(train_m, val_m, seed, n_norm, placebo, judge=None):
    assert judge not in train_m and judge != val_m, "el partido del juez está en el entrenamiento"
    tr = [s for m in train_m for s in train_seqs(m, placebo)]
    if COPIES == 2:  # exp-142: segunda copia degradada de cada partido de entrenamiento; la validación, solo la copia 0
        tr += [s for m in train_m for s in train_seqs(m, placebo, DATA + "1")]
    tr += extra_seqs()
    va = train_seqs(val_m, placebo)
    if VAL_BOTH:  # exp-150 (B2c)
        va += train_seqs(val_m, placebo, DATA + "1")
    t0 = time.time()
    fit = T.train(tr, va, np.arange(n_norm), seed + SEED0, log=lambda s: print(s, flush=True), margin=MARGIN)
    ep = len(fit["hist"])
    print(f"semilla {seed}: validación interna {val_m}; {ep} épocas en {time.time() - t0:.0f} s; mejor validación "
          f"{fit['best_val']:.4f}; tau {fit['tau']:.4f}; prior {fit['prior']:.4f}", flush=True)
    return fit


def run_fold(J, placebo=False, seeds=SEEDS, extra_val=False):
    mids, n_norm, _ = meta()
    others = [m for m in mids if m != J]
    out = {}
    for s in seeds:
        val_m = others[s]
        train_m = [m for m in others if m != val_m]
        fit = fit_seed(train_m, val_m, s, n_norm, placebo, judge=J)
        out[f"prior_s{s}"] = np.array(fit["prior"])
        out[f"tau_s{s}"] = np.array(fit["tau"])
        out[f"epochs_s{s}"] = np.array(len(fit["hist"]))
        out[f"val_mid_s{s}"] = np.array(val_m)
        out[f"train_mids_s{s}"] = np.array(train_m)
        out["judge_mid"] = np.array(J)
        out["placebo"] = np.array(placebo)
        for sc in SCEN:
            for i, X in enumerate(x_parts(os.path.join(DIR, f"judge_{sc}_{J}.npz"))):
                out[f"{sc}_s{s}_{i}"] = T.predict(fit, X)
            for (o, i), X in offset_parts(os.path.join(DIR, f"judge_{sc}_{J}_offsets.npz")).items():
                out[f"{sc}_o{o}_s{s}_{i}"] = T.predict(fit, X)
            if TTA:  # exp-151: la misma red aplicada a las 4 copias perturbadas del juez (claves X<parte>_t<k>)
                Z = np.load(os.path.join(DIR, f"judge_{sc}_{J}_tta.npz"))
                for key in Z.files:
                    if key.startswith("X"):
                        i, k = key[1:].split("_t")
                        out[f"{sc}_t{k}_s{s}_{i}"] = T.predict(fit, Z[key])
        if PRED_SKC:  # exp-152
            import glob
            files = sorted(glob.glob(os.path.join(DIR, "skc_x_*.npz")))
            assert files, "no hay entradas de SkillCorner (skc_x_*.npz)"
            for f in files:
                sid = os.path.basename(f)[len("skc_x_"):-4]
                for i, X in enumerate(x_parts(f)):
                    out[f"skc_{sid}_s{s}_{i}"] = T.predict(fit, X)
        if extra_val:  # piloto: logits en el partido de validación interno, para la puerta del paso 3
            for i, X in enumerate(x_parts(os.path.join(DIR, f"dfl_train_{val_m}.npz"))):
                out[f"val_s{s}_{i}"] = T.predict(fit, X)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1]
    mids, n_norm, _ = meta()
    if mode == "pilot":
        out = run_fold(mids[0], seeds=(0,), extra_val=True)
        np.savez(os.path.join(OUT, "pilot.npz"), **out)
    elif mode == "fold":
        J, placebo = sys.argv[2], "--placebo" in sys.argv
        out = run_fold(J, placebo)
        out["margin"], out["seed0"] = np.array(MARGIN), np.array(SEED0)
        out["data"], out["copies"] = np.array(DATA), np.array(COPIES)
        out["extra"] = np.array(EXTRA)
        out["val_both"] = np.array(VAL_BOTH)
        out["tta"] = np.array(TTA)
        out["pred_skc"] = np.array(PRED_SKC)
        out["label_smooth"] = np.array(T.LABEL_SMOOTH)
        np.savez(os.path.join(OUT, f"fold_{J}{'_placebo' if placebo else ''}{'_' + TAG if TAG else ''}.npz"), **out)
    elif mode == "prod":
        # La predicción se hace SIEMPRE con la red cargada de sus pesos guardados (también justo después de entrenar):
        # el modelo en memoria tras entrenar no da bit a bit lo mismo que el cargado (10 de 95.000 logits cambiaban
        # 0,001 en la frontera del redondeo, 14/11), y así reentrenar y cargar los pesos dan el mismo CSV.
        _, _, c_in = meta()
        out = {}
        for s in SEEDS:
            fit = fit_seed([m for m in mids if m != mids[s]], mids[s], s, n_norm, False)
            path = os.path.join(OUT, f"prod{SUF}_weights_s{s}.npz")
            np.savez(path, **T.state_bytes(fit))
            fit = T.load_state(np.load(path), c_in)
            out[f"prior_s{s}"] = np.array(fit["prior"])
            for i, X in enumerate(x_parts(os.path.join(DIR, "barath.npz"))):
                out[f"barath_s{s}_{i}"] = T.predict(fit, X)
            for (o, i), X in offset_parts(os.path.join(DIR, "barath_offsets.npz")).items():
                out[f"barath_o{o}_s{s}_{i}"] = T.predict(fit, X)
        np.savez(os.path.join(OUT, f"prod{SUF}.npz"), **out)
    elif mode == "prod_pesos":  # prod_pesos <carpeta> [salida.npz]: la entrega, sin reentrenar (regenerar_e1.sh)
        # Lo mismo que «prod» después de entrenar (la red cargada de prod_weights_s<s>.npz predice BAR-ATH), pero con
        # los pesos de la entrega (entrenados solo con la DFL; reentrenar en otra máquina no da bit a bit los mismos).
        # Solo escribe lo que lee make_e1 (prior_s<s>, barath_s<s>_<parte>); por defecto en OUT/prod.npz.
        src = sys.argv[2]
        dst = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else os.path.join(OUT, "prod.npz")
        _, _, c_in = meta()
        out = {}
        for s in SEEDS:
            fit = T.load_state(np.load(os.path.join(src, f"prod_weights_s{s}.npz")), c_in)
            out[f"prior_s{s}"] = np.array(fit["prior"])
            for i, X in enumerate(x_parts(os.path.join(DIR, "barath.npz"))):
                out[f"barath_s{s}_{i}"] = T.predict(fit, X)
        np.savez(dst, **out)
        print(f"red de la entrega: {len(SEEDS)} semillas cargadas de {src} -> {dst}", flush=True)
    elif mode == "infer":  # infer <entrada.npz> <salida.npz>: red de producción cargada de sus pesos guardados
        src, dst = sys.argv[2], sys.argv[3]
        _, _, c_in = meta()
        fits = [T.load_state(np.load(os.path.join(OUT, f"prod{SUF}_weights_s{s}.npz")), c_in) for s in SEEDS]
        Z = np.load(src)
        out = {f"prior_s{s}": np.array(f["prior"]) for s, f in enumerate(fits)}
        for k in Z.files:  # claves <nombre>_<parte>
            name, i = k.rsplit("_", 1)
            for s, f in enumerate(fits):
                out[f"{name}_s{s}_{i}"] = T.predict(f, Z[k])
        np.savez(dst, **out)
        print(f"{len(Z.files)} secuencias inferidas con los pesos guardados -> {os.path.basename(dst)}")
    elif mode == "check_prod":  # los pesos guardados reproducen exactamente los logits de BAR-ATH de prod.npz
        _, _, c_in = meta()
        P = np.load(os.path.join(OUT, f"prod{SUF}.npz"))
        ok = True
        for s in SEEDS:
            fit = T.load_state(np.load(os.path.join(OUT, f"prod{SUF}_weights_s{s}.npz")), c_in)
            assert abs(fit["prior"] - float(P[f"prior_s{s}"])) < 1e-12
            for i, X in enumerate(x_parts(os.path.join(DIR, "barath.npz"))):
                same = np.array_equal(T.predict(fit, X), P[f"barath_s{s}_{i}"])
                ok &= same
                print(f"semilla {s} parte {i}: logits con los pesos guardados idénticos: {'sí' if same else 'NO'}")
        print("PESOS GUARDADOS: " + ("reproducen prod.npz" if ok else "NO reproducen prod.npz"))
        sys.exit(0 if ok else 1)
    else:
        raise SystemExit(f"modo desconocido: {mode}")


if __name__ == "__main__":
    main()
