"""Lectura del Automatic Tracking Data a arrays de numpy.

La forma natural del dato NO es una tabla: es un tensor (frame, slot, eje). Todo lo que
vamos a calcular -- distancias al balon, centroides por equipo, dispersion, jugador mas
cercano -- es una reduccion sobre el eje de slots, y eso en un DataFrame ancho de 141
columnas se escribe fatal y se ejecuta peor.

    from mtah_2027.atd import load
    tr = load()                      # ~1.5 s la primera vez, ~0.05 s despues
    tr.xy[frame, slot]               # (x, y) normalizado 0-1
    tr.xy[:, tr.es_local]            # solo los slots del equipo local
    tr.ball[frame]                   # (x, y) del balon, NaN si no se detecto

Un slot NO es un jugador: son 46 huecos de tracklet para 22 personas, aparecen y
desaparecen y se intercambian. Lo unico estable es a que equipo pertenecen.
"""

from __future__ import annotations

import io
import os
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

# Campo real en metros, medido sobre el tracking corregido (el XML dice 105x68).
# Son DOS constantes distintas y mezclarlas deforma cualquier distancia un 55%:
# una unidad normalizada vale 105.3 m en x y 68.0 m en y.
CAMPO = np.array([105.3, 68.0], dtype=np.float32)

LOCAL, VISITANTE = 0, 1
_SEPARADORES = bytes.maketrans(b":;", b",,")


def _raiz(inicio=__file__):
    """Raiz del repo: primer directorio hacia arriba con pyproject.toml."""
    directorio = os.path.dirname(os.path.abspath(inicio))
    while True:
        if os.path.isfile(os.path.join(directorio, "pyproject.toml")):
            return directorio
        padre = os.path.dirname(directorio)
        if padre == directorio:
            raise RuntimeError("no encuentro la raiz del repo desde " + inicio)
        directorio = padre


RAIZ = _raiz()


def source_dir(source):
    """Folder holding one recording's files (`tactical` or `broadcast`).

    The bundle has been unpacked in more than one layout: data/<source>/<source>/ (zip with a
    wrapper folder), data/<source>/ (the official layout) and the old raw/<source>/. MTAH_DATA_DIR
    overrides the data/ root, as in the official games.py.
    """
    data_root = os.environ.get("MTAH_DATA_DIR") or os.path.join(RAIZ, "data")
    candidates = (os.path.join(data_root, source, source),
                  os.path.join(data_root, source),
                  os.path.join(RAIZ, "raw", source))
    stem = "BAR-ATH_%s_atd.xml" % source
    for folder in candidates:
        if os.path.isfile(os.path.join(folder, stem)):
            return folder
    raise FileNotFoundError("%s not found in any of %s" % (stem, candidates))


def video_path(source="tactical"):
    """Path to one recording's mp4.

    Official layout: <data_root>/<source>/BAR-ATH_<source>.mp4, next to the tracking files.
    Our local unpack keeps it at data/BAR-ATH_<source>-001.mp4, so that is the fallback.
    """
    data_root = os.environ.get("MTAH_DATA_DIR") or os.path.join(RAIZ, "data")
    candidates = (os.path.join(data_root, source, "BAR-ATH_%s.mp4" % source),
                  os.path.join(data_root, source, source, "BAR-ATH_%s.mp4" % source),
                  os.path.join(data_root, "BAR-ATH_%s-001.mp4" % source))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("BAR-ATH_%s video not found in any of %s" % (source, candidates))


@dataclass
class Tracking:
    """Un partido visto por una camara.

    xy, velocidad y balon comparten el eje 0: el frame de video, contado desde 0.
    El fichero cuenta los frames desde 1; la conversion ya esta hecha aqui.
    """

    xy: np.ndarray          # (frames, slots, 2) float32, normalizado 0-1. NaN = no visto
    velocidad: np.ndarray   # (frames, slots) float32, m/s (el XML dice kph y miente)
    balon: np.ndarray       # (frames, 2) float32, normalizado. NaN = no detectado
    balon_velocidad: np.ndarray  # (frames,) float32. Presente solo en ~10% de frames
    equipo: np.ndarray      # (slots,) int8: 0 local, 1 visitante
    nombres: tuple          # (nombre_local, nombre_visitante)

    @property
    def n_frames(self):
        return self.xy.shape[0]

    @property
    def n_slots(self):
        return self.xy.shape[1]

    @property
    def es_local(self):
        """Mascara (slots,) de los slots del equipo local."""
        return self.equipo == LOCAL

    @property
    def es_visitante(self):
        return self.equipo == VISITANTE

    @property
    def visible(self):
        """(frames, slots) bool: si ese slot tiene posicion en ese frame."""
        return np.isfinite(self.xy[..., 0])

    @property
    def balon_visible(self):
        """(frames,) bool."""
        return np.isfinite(self.balon[..., 0])

    def metros(self, a):
        """Pasa coordenadas normalizadas a metros. Vale para xy y para balon."""
        return a * CAMPO

    def centroide(self, mascara_slots=None):
        """(frames, 2) centro de masas de los slots visibles, en normalizado.

        nanmean sobre el eje de slots: los slots que faltan en un frame simplemente no
        entran, que es el comportamiento correcto -- el tracker pierde gente todo el rato.
        """
        xy = self.xy if mascara_slots is None else self.xy[:, mascara_slots]
        with warnings.catch_warnings():
            # Los frames sin ningun slot visible (descanso) dan NaN, que es la respuesta
            # correcta; nanmean avisa igualmente y el aviso solo es ruido.
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(xy, axis=1)

    def distancia_al_balon(self):
        """(frames, slots) distancia en METROS de cada slot al balon. NaN donde no hay dato."""
        d = (self.xy - self.balon[:, None, :]) * CAMPO
        return np.hypot(d[..., 0], d[..., 1])

    def mas_cercano_al_balon(self):
        """(frames,) indice del slot mas cercano al balon, o -1 si no hay respuesta."""
        d = self.distancia_al_balon()
        hay = np.isfinite(d).any(axis=1)
        fuera = np.where(np.isfinite(d), d, np.inf)
        idx = np.argmin(fuera, axis=1)
        return np.where(hay, idx, -1)


def _equipos(ruta_xml):
    """((slots,) int8 con el equipo de cada slot, (nombre_local, nombre_visitante)).

    El primer Team del XML es el local: convencion FIFA/EPTS, la que sigue el baseline
    oficial. Para este partido son 24 slots del Barcelona y 22 del Athletic.
    """
    raiz = ET.parse(ruta_xml).getroot()
    equipos = list(raiz.iter("Team"))
    if len(equipos) != 2:
        raise ValueError("%s: esperaba 2 equipos, hay %d" % (ruta_xml, len(equipos)))
    id_local = equipos[0].get("id")
    slots = [LOCAL if p.get("teamId") == id_local else VISITANTE
             for p in raiz.iter("Player")]
    if not slots:
        raise ValueError("%s: sin entradas Player" % ruta_xml)
    return (np.array(slots, dtype=np.int8),
            tuple(e.findtext("Name") for e in equipos))


def _parsear(ruta_txt, n_slots):
    """El .txt entero a (xy, velocidad, balon, balon_velocidad).

    Truco: cada linea es "<frame>:<slot>;<slot>;...:<balon>" con el mismo numero de campos
    siempre, asi que traduciendo ':' y ';' a ',' queda un CSV rectangular y lo lee el
    parser en C de pandas de una vez. Linea a linea en Python son ~40 s; asi son ~1.5.
    """
    import pandas as pd

    bruto = open(ruta_txt, "rb").read().translate(_SEPARADORES)
    # BytesIO y no decode(): evita materializar 195 MB de str y baja el pico de memoria
    # de ~1.2 GB a ~420 MB, ademas de ser mas rapido.
    tabla = pd.read_csv(io.BytesIO(bruto), header=None, dtype=np.float32).values

    esperadas = 1 + n_slots * 3 + 3
    if tabla.shape[1] != esperadas:
        raise ValueError("%s: %d columnas, esperaba %d (%d slots)"
                         % (ruta_txt, tabla.shape[1], esperadas, n_slots))
    n = tabla.shape[0]
    if not np.array_equal(tabla[:, 0], np.arange(1, n + 1)):
        raise ValueError("%s: los frame_id no son 1..%d contiguos" % (ruta_txt, n))

    cuerpo = tabla[:, 1:1 + n_slots * 3].reshape(n, n_slots, 3)
    cola = tabla[:, 1 + n_slots * 3:]
    return (np.ascontiguousarray(cuerpo[:, :, :2]),
            np.ascontiguousarray(cuerpo[:, :, 2]),
            np.ascontiguousarray(cola[:, :2]),
            np.ascontiguousarray(cola[:, 2]))


def load(fuente="tactical", cache=True):
    """El ATD de una camara. `fuente` es "tactical" o "broadcast".

    Parsear cuesta ~1.5 s; con cache las siguientes cargas son ~0.05 s. La cache vive en
    out/cache/*.npz y se invalida sola si el .txt es mas reciente. `cache=False` fuerza
    releer el fichero original.
    """
    carpeta = source_dir(fuente)
    sello = "BAR-ATH_%s" % fuente
    ruta_xml = os.path.join(carpeta, sello + "_atd.xml")
    ruta_txt = os.path.join(carpeta, sello + "_atd.txt")
    for ruta in (ruta_xml, ruta_txt):
        if not os.path.isfile(ruta):
            raise FileNotFoundError(ruta)

    equipo, nombres = _equipos(ruta_xml)
    ruta_cache = os.path.join(RAIZ, "out", "cache", sello + "_atd.npz")

    if cache and os.path.isfile(ruta_cache) \
            and os.path.getmtime(ruta_cache) > os.path.getmtime(ruta_txt):
        with np.load(ruta_cache) as d:
            xy, velocidad, balon, balon_velocidad = (
                d["xy"], d["velocidad"], d["balon"], d["balon_velocidad"])
    else:
        xy, velocidad, balon, balon_velocidad = _parsear(ruta_txt, len(equipo))
        if cache:
            os.makedirs(os.path.dirname(ruta_cache), exist_ok=True)
            # Escritura atomica: si el proceso muere a medio guardar, un .npz truncado
            # se leeria en la siguiente ejecucion como si fuera valido.
            temporal = ruta_cache + ".tmp.npz"
            np.savez(temporal, xy=xy, velocidad=velocidad, balon=balon,
                     balon_velocidad=balon_velocidad)
            os.replace(temporal, ruta_cache)

    return Tracking(xy, velocidad, balon, balon_velocidad, equipo, nombres)


if __name__ == "__main__":
    import time

    for fuente in ("tactical", "broadcast"):
        t = time.time()
        tr = load(fuente)
        print("%s  %d frames x %d slots   (%.2f s)"
              % (fuente, tr.n_frames, tr.n_slots, time.time() - t))
        print("   equipos: H = %s (%d slots), A = %s (%d slots)"
              % (tr.nombres[0], tr.es_local.sum(), tr.nombres[1], tr.es_visitante.sum()))
        print("   memoria: %.0f MB" % ((tr.xy.nbytes + tr.velocidad.nbytes
                                        + tr.balon.nbytes) / 1e6))
        print("   balon visible: %d/%d = %.1f%%"
              % (tr.balon_visible.sum(), tr.n_frames, 100 * tr.balon_visible.mean()))
        print("   slots visibles por frame: media %.2f" % tr.visible.sum(1).mean())
