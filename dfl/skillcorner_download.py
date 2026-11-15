"""Descarga SkillCorner open data (tracking de retransmisión de la A-League 2024/25, licencia MIT) para exp-119.

Solo el tracking (`*_tracking_extrapolated.jsonl`, Git LFS) y las fases de juego (`*_phases_of_play.csv`) de todos los
partidos del repo, fijados a un commit concreto. Cada fichero LFS se comprueba con el sha256 de su puntero. Guarda los
ficheros en MTAH_DATA_DIR/external/skillcorner/<partido>/ y escribe skillcorner_sha256.txt junto a este script (commit,
fichero, sha256), para que la descarga se pueda repetir y verificar. No toca nada de BAR-ATH.

    python -I skillcorner_download.py            # todos los partidos del commit actual de master (lo anota)
    python -I skillcorner_download.py <commit>   # repetir la descarga de un commit ya anotado
"""
import hashlib
import json
import os
import sys
import urllib.request

REPO = "SkillCorner/opendata"
UA = {"User-Agent": "mtah-research"}
HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(os.environ["MTAH_DATA_DIR"], "external", "skillcorner")
SHA_FILE = os.path.join(HERE, "skillcorner_sha256.txt")


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, expect=None):
    if os.path.exists(path) and (expect is None or sha256(path) == expect):
        return
    tmp = path + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if expect is not None:
        got = sha256(tmp)
        assert got == expect, f"{path}: sha256 {got} distinto del puntero LFS {expect}"
    os.replace(tmp, path)


def main():
    commit = sys.argv[1] if len(sys.argv) > 1 else json.loads(get(f"https://api.github.com/repos/{REPO}/commits/master"))["sha"]
    tree = json.loads(get(f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1"))
    assert not tree.get("truncated"), "el árbol del repo viene truncado"
    paths = sorted(t["path"] for t in tree["tree"] if t["type"] == "blob" and t["path"].startswith("data/matches/")
                   and t["path"].endswith(("_tracking_extrapolated.jsonl", "_phases_of_play.csv")))
    lines = [f"# SkillCorner open data ({REPO}, licencia MIT), commit {commit}"]
    expected = {}  # si ya hay hashes anotados para este commit, cada fichero tiene que dar el mismo sha256
    if os.path.exists(SHA_FILE):
        with open(SHA_FILE, encoding="utf-8") as f:
            old = f.read().splitlines()
        if old and old[0] == lines[0]:
            expected = {ln.split("  ", 1)[1]: ln.split("  ", 1)[0] for ln in old[1:] if ln.strip()}
    for p in paths:
        mid = p.split("/")[2]
        out = os.path.join(DEST, mid, os.path.basename(p))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        raw = f"https://raw.githubusercontent.com/{REPO}/{commit}/{p}"
        if p.endswith(".jsonl"):
            ptr = get(raw).decode()
            oid = [ln.split(":", 1)[1] for ln in ptr.splitlines() if ln.startswith("oid sha256:")][0]
            download(f"https://media.githubusercontent.com/media/{REPO}/{commit}/{p}", out, oid)
        else:
            download(raw, out)
        rel, got = f"{mid}/{os.path.basename(p)}", sha256(out)
        assert rel not in expected or expected[rel] == got, f"{rel}: sha256 {got} distinto del anotado {expected[rel]}"
        lines.append(f"{got}  {rel}")
        print(f"{mid}/{os.path.basename(p)}: {os.path.getsize(out) / 2**20:.0f} MB", flush=True)
    assert not expected or len(expected) == len(paths), "el commit tiene otros ficheros que los anotados"
    with open(SHA_FILE, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(paths)} ficheros -> {DEST}; hashes en {SHA_FILE}"
          + (" (iguales a los anotados)" if expected else ""))


if __name__ == "__main__":
    main()
