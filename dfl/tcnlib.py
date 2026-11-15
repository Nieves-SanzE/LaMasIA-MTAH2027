"""E5 / exp-139: red temporal (TCN) del modelo de equipo, para mezclarla con los árboles de e1_v13.

Se ejecuta en dfl/.venv-tcn (torch 2.14.1+cpu, requirements-tcn.txt). Solo numpy y torch: no lee ningún dato; recibe
secuencias ya exportadas a 5 Hz (tcn_export.py) y devuelve logits. Diseño fijado el 14/11/2026 antes de ver ningún
resultado (registro del experimento E5):

  - entrada: una secuencia por parte, (pasos, canales) a 5 Hz; normalización robusta (mediana / IQR) de las columnas de
    features calculada solo con los partidos de entrenamiento; NaN -> 0 después de normalizar;
  - red: conv 1x1 -> 64 canales; 6 bloques residuales x + dropout(relu(conv dilatada)), núcleo 5, dilataciones
    1-2-4-8-16-32, no causal (campo receptivo 253 pasos, ~ +-25 s); cabezas 1x1: equipo (logit de A) y cambio;
  - pérdida: BCE del equipo en los pasos vivos con pesos 1 / max(duración de su posesión, 2 s) (media 1 en el
    entrenamiento) + 0,3 x BCE de la auxiliar «cambio de posesión a <= 0,5 s» en los pasos vivos;
  - AdamW (lr 1e-3, weight decay 1e-3), recortes aleatorios de 120 s (600 pasos), lotes de 16, épocas de
    ceil(pasos de entrenamiento / 600) recortes; parada temprana con la BCE ponderada del equipo en el partido de
    validación interno (máximo 40 épocas, paciencia 5; se queda la mejor época);
  - calibración: escalado de temperatura (logit / tau) con la BCE ponderada en el partido de validación interno, tau en
    una rejilla logarítmica fija [0,25; 4] de 401 puntos;
  - salida: logits calibrados redondeados a 3 decimales (medida de reproducibilidad entre máquinas, 14/11).
Determinismo: torch.set_num_threads(4) fijo (no depende de OMP_NUM_THREADS), use_deterministic_algorithms(True),
semillas en torch y numpy, sin workers, orden de lotes con semilla. Además (14/11, tras ver que con los datos reales,
227 canales, dos entrenamientos iguales daban pesos distintos): oneDNN (mkldnn) desactivado, porque sus convoluciones
no son deterministas con varios hilos y eligen núcleos según la CPU; MKL en modo reproducible AVX2 (MKL_CBWR) y los
núcleos de torch fijados a AVX2 (ATEN_CPU_CAPABILITY), para dar lo mismo en cualquier CPU x86 con AVX2 y 4 hilos.
"""
import hashlib
import math
import os

os.environ["MKL_CBWR"] = "AVX2"  # antes de importar torch
os.environ["ATEN_CPU_CAPABILITY"] = "avx2"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

THREADS = 4
CH = 64
KERNEL = 5
DILATIONS = (1, 2, 4, 8, 16, 32)
DROPOUT = 0.2
AUX_W = 0.3
LR = 1e-3
WD = 1e-3
CROP = 600  # 120 s a 5 Hz
BATCH = 16
MAX_EPOCHS = 40
PATIENCE = 5
TAU_GRID = np.exp(np.linspace(math.log(0.25), math.log(4.0), 401))
ROUND = 3

_SETUP = []


def setup():
    """Hilos fijos y algoritmos deterministas (una vez por proceso)."""
    if _SETUP:
        return
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    _SETUP.append(True)


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Conv1d(CH, CH, KERNEL, dilation=d, padding=d * (KERNEL - 1) // 2)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return x + self.drop(F.relu(self.conv(x)))


class TCN(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.inp = nn.Conv1d(c_in, CH, 1)
        self.blocks = nn.Sequential(*[Block(d) for d in DILATIONS])
        self.team = nn.Conv1d(CH, 1, 1)
        self.chg = nn.Conv1d(CH, 1, 1)

    def forward(self, x):  # x: (lote, canales, pasos)
        h = self.blocks(self.inp(x))
        return self.team(h)[:, 0], self.chg(h)[:, 0]


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- normalización

def fit_norm(seqs, norm_cols):
    """Mediana e IQR de las columnas norm_cols con todos los pasos de las secuencias de entrenamiento (sin NaN)."""
    X = np.concatenate([s["X"][:, norm_cols] for s in seqs]).astype(np.float64)
    med = np.nanmedian(X, 0)
    q75, q25 = np.nanpercentile(X, 75, 0), np.nanpercentile(X, 25, 0)
    iqr = q75 - q25
    med = np.where(np.isfinite(med), med, 0.0)
    iqr = np.where(np.isfinite(iqr) & (iqr > 0), iqr, 1.0)
    return dict(cols=np.asarray(norm_cols), med=med.astype(np.float32), iqr=iqr.astype(np.float32))


def apply_norm(X, norm):
    Z = X.astype(np.float32).copy()
    c = norm["cols"]
    Z[:, c] = (Z[:, c] - norm["med"]) / norm["iqr"]
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------- entrenamiento

def _tensors(s, norm):
    return dict(X=torch.from_numpy(apply_norm(s["X"], norm).T.copy()),  # (canales, pasos)
                y=torch.from_numpy(s["y"].astype(np.float32)),            # 1 = A, 0 = H, -1 = no vivo
                w=torch.from_numpy(s["w"].astype(np.float32)),
                chg=torch.from_numpy(s["chg"].astype(np.float32)))


LABEL_SMOOTH = 0.0  # exp-154: eps del suavizado de etiquetas, solo en la pérdida de entrenamiento (0 = E5 sin cambios)


def smooth_labels(y):
    """y (1 = A, 0 = H, -1 = no vivo) -> y (1 - eps) + eps / 2 en los pasos vivos; con eps = 0, el mismo tensor."""
    if not LABEL_SMOOTH:
        return y
    return torch.where(y >= 0, y * (1 - LABEL_SMOOTH) + LABEL_SMOOTH / 2, y)


def weighted_bce(logit, y, w):
    """BCE ponderada media en los pasos vivos (y >= 0)."""
    m = y >= 0
    if not bool(m.any()):
        return logit.sum() * 0.0
    l_ = F.binary_cross_entropy_with_logits(logit[m], y[m], reduction="none")
    return (l_ * w[m]).sum() / m.sum()


def _batch(seqs, starts, length=CROP):
    xs, ys, ws, cs = [], [], [], []
    for i, a in starts:
        s = seqs[i]
        xs.append(s["X"][:, a:a + length])
        ys.append(s["y"][a:a + length])
        ws.append(s["w"][a:a + length])
        cs.append(s["chg"][a:a + length])
    return torch.stack(xs), torch.stack(ys), torch.stack(ws), torch.stack(cs)


def _crop_plan(seqs, rng, length=CROP):
    """Recortes de una época: ceil(pasos / CROP) inicios al azar, cada uno en una secuencia elegida en proporción a su
    longitud (todas las secuencias son de una sola parte: ningún recorte cruza el descanso). length = CROP + 2 x margen
    (exp-141); con margen 0, exactamente lo de E5."""
    lens = np.array([s["X"].shape[1] for s in seqs])
    assert (lens >= length).all(), "secuencia más corta que un recorte"
    n = int(math.ceil(lens.sum() / CROP))
    which = rng.choice(len(seqs), size=n, p=lens / lens.sum())
    return [(int(i), int(rng.integers(0, lens[i] - length + 1))) for i in which]


HALF_RF = sum((KERNEL - 1) // 2 * d for d in DILATIONS)  # 126 pasos: medio campo receptivo (253 = 2 x 126 + 1)


@torch.no_grad()
def raw_logits(model, s_t):
    model.eval()
    return model(s_t["X"][None])[0][0]


def val_loss(model, vals_t):
    num, den = 0.0, 0
    for s in vals_t:
        lg = raw_logits(model, s)
        m = s["y"] >= 0
        num += float(weighted_bce(lg, s["y"], s["w"])) * int(m.sum())
        den += int(m.sum())
    return num / max(den, 1)


def train(train_seqs, val_seqs, norm_cols, seed, log=print, max_epochs=MAX_EPOCHS, margin=0):
    """Entrena una semilla. train_seqs / val_seqs: listas de dict(X (pasos, canales), y, w, chg), una por parte.
    Los pesos w de entrenamiento se normalizan a media 1 en sus pasos vivos. Devuelve el modelo (mejor época), la
    normalización, tau, el prior ponderado y el historial.
    margin (exp-141, mejora 2): cada recorte se alarga `margin` pasos por cada lado y la pérdida solo cuenta los CROP
    pasos centrales, que con margin = HALF_RF tienen todo su campo receptivo dentro del recorte (como al predecir una
    parte entera). margin = 0 es exactamente E5."""
    setup()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    norm = fit_norm(train_seqs, norm_cols)
    live = np.concatenate([s["y"] >= 0 for s in train_seqs])
    wall = np.concatenate([s["w"] for s in train_seqs])[live].astype(np.float64)
    yall = np.concatenate([s["y"] for s in train_seqs])[live].astype(np.float64)
    wmean = wall.mean()
    prior = float(np.sum(wall * yall) / np.sum(wall))
    tr = []
    for s in train_seqs:
        t = _tensors(s, norm)
        t["w"] = t["w"] / wmean
        tr.append(t)
    va = []
    for s in val_seqs:  # misma escala de pesos que el entrenamiento (no cambia la época ni la tau elegidas)
        t = _tensors(s, norm)
        t["w"] = t["w"] / wmean
        va.append(t)
    model = TCN(tr[0]["X"].shape[0])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(max_epochs):
        model.train()
        plan = _crop_plan(tr, rng, CROP + 2 * margin)
        tot = 0.0
        for b in range(0, len(plan), BATCH):
            X, y, w, c = _batch(tr, plan[b:b + BATCH], CROP + 2 * margin)
            lt, lc = model(X)
            if margin:  # solo los pasos centrales, con el contexto completo
                sl = slice(margin, margin + CROP)
                lt, lc, y, w, c = lt[:, sl], lc[:, sl], y[:, sl], w[:, sl], c[:, sl]
            loss = weighted_bce(lt, smooth_labels(y), w)
            live_b = y >= 0
            if bool(live_b.any()):
                loss = loss + AUX_W * F.binary_cross_entropy_with_logits(lc[live_b], c[live_b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(plan[b:b + BATCH])
        vl = val_loss(model, va)
        hist.append((ep + 1, tot / len(plan), vl))
        log(f"  semilla {seed} época {ep + 1}: pérdida de entrenamiento {tot / len(plan):.4f}, validación {vl:.4f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    tau = calibrate(model, va)
    return dict(model=model, norm=norm, tau=tau, prior=prior, hist=hist, best_val=best)


def calibrate(model, vals_t):
    """tau de la rejilla fija que minimiza la BCE ponderada de logit / tau en la validación interna."""
    lg = torch.cat([raw_logits(model, s) for s in vals_t]).double()
    y = torch.cat([s["y"] for s in vals_t]).double()
    w = torch.cat([s["w"] for s in vals_t]).double()
    m = y >= 0
    lg, y, w = lg[m], y[m], w[m]
    losses = [float((F.binary_cross_entropy_with_logits(lg / t, y, reduction="none") * w).sum()) for t in TAU_GRID]
    return float(TAU_GRID[int(np.argmin(losses))])


def predict(fit, X):
    """Logits calibrados y redondeados a 3 decimales de una secuencia (pasos, canales) a 5 Hz."""
    setup()
    s_t = dict(X=torch.from_numpy(apply_norm(X, fit["norm"]).T.copy()))
    lg = raw_logits(fit["model"], s_t).double().numpy() / fit["tau"]
    return np.round(lg, ROUND)


def state_bytes(fit):
    """Pesos, normalización, tau y prior en un dict de arrays (para guardar como fichero intermedio)."""
    out = {f"w:{k}": v.detach().numpy() for k, v in fit["model"].state_dict().items()}
    out.update({"norm_cols": fit["norm"]["cols"], "norm_med": fit["norm"]["med"], "norm_iqr": fit["norm"]["iqr"],
                "tau": np.array(fit["tau"]), "prior": np.array(fit["prior"])})
    return out


def load_state(z, c_in):
    """Inverso de state_bytes (z: npz cargado)."""
    setup()
    model = TCN(c_in)
    model.load_state_dict({k[2:]: torch.from_numpy(np.asarray(z[k])) for k in z.files if k.startswith("w:")})
    return dict(model=model, norm=dict(cols=z["norm_cols"], med=z["norm_med"], iqr=z["norm_iqr"]),
                tau=float(z["tau"]), prior=float(z["prior"]))


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
