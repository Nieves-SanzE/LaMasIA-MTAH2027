"""Comprueba (o, con --write, anota) el sha256 de los datos externos descargados: DFL / IDSSE.

    python verify_external.py <carpeta de pablo_v6>            # comprueba external_sha256.txt
    python verify_external.py <carpeta de pablo_v6> --write    # lo vuelve a escribir (solo al fijar la versión)

- DFL: MTAH_DATA_DIR/external/idsse/DFL_*.xml (download_idsse.sh).
Desde e1_v13 (14/11, D9) ya no se comprueba Metrica sample data: no declara licencia y la entrega no la usa (exp-135).
Las versiones anteriores (e1_v2 ... e1_v12) sí la necesitan; sus 9 ficheros quedaron anotados hasta el 13/11 en el
historial de external_sha256.txt. El argumento <carpeta de pablo_v6> se mantiene para no cambiar la llamada.
Son datos abiertos de otros partidos; no se toca nada de BAR-ATH. Solo imprime nombres de fichero y sí/no.
"""
import glob
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "external_sha256.txt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files(pablo):
    idsse = os.path.join(os.environ["MTAH_DATA_DIR"], "external", "idsse")
    out = {}
    for p in sorted(glob.glob(os.path.join(idsse, "DFL_*.xml"))):
        out["idsse/" + os.path.basename(p)] = p
    return out


def main():
    pablo = sys.argv[1]
    fs = files(pablo)
    if "--write" in sys.argv:
        with open(MANIFEST, "w", newline="\n") as f:
            for k, p in fs.items():
                f.write(f"{sha256(p)}  {k}\n")
        print(f"{len(fs)} ficheros anotados en {MANIFEST}")
        return
    want = {}
    for line in open(MANIFEST):
        h, k = line.split()
        want[k] = h
    bad = [k for k in want if k not in fs or sha256(fs[k]) != want[k]]
    extra = sorted(set(fs) - set(want))
    print(f"datos externos: {len(want) - len(bad)}/{len(want)} con el sha256 esperado; sobran {len(extra)}")
    for k in bad:
        print("  distinto o falta:", k)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
