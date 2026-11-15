# LaMasIA: ball possession for More than a Hack 2027 (phase 1)

Team **LaMasIA**. This repository regenerates our submission for FC Barcelona vs Athletic Club: one label per frame
(`H`, `A`, `D` or `X`) for the 199,319 frames of the match.

The models are trained **only on external open data**: 7 matches of the DFL open dataset. Nothing is trained, tuned or
selected on the provided match or on its public ground truth. The provided match is used only as model input:
tracking, ball visibility and, for the kick-off of each half, the Smart Tagging. The scripts block any attempt to open a
ground-truth file (`dfl/dfllib.py`, `forbid_public_truth`).

**Expected output:** `LaMasIA.csv`, sha256
`b8d75e1aea161b6ea2883417454016a5e4833babb42aae7d9a0d11b86eafb184`.

## Layout

```
README.md          this file
dfl/               the pipeline: data download and checks, training on the DFL, network weights, CSV
  regenerar_e1.sh  the command that regenerates the submission (11 steps)
  make_e1.py       training of the tree models and the dead-ball model, mixing, decoding, CSV
  tcnlib.py, tcn_*.py   temporal network (weights in pesos_red/, with their sha256)
  README.md        method and steps in detail (in Spanish)
pablo_v6/          feature, dead-ball and decoder code used by dfl/ (see its README)
```

## Requirements

- Git, [uv](https://docs.astral.sh/uv/) and a POSIX shell: Git Bash on Windows, or `sh` on Linux and macOS.
- About 10 GB of free disk space. The DFL download is about 2.6 GB and SkillCorner about 1.7 GB.
- A CPU is enough. It takes about 1 h 50 min on a laptop with 4 threads.

## Setup (once)

From Git Bash (Windows) or a terminal (Linux, macOS):

```sh
git clone https://github.com/Nieves-SanzE/LaMasIA-MTAH2027
cd LaMasIA-MTAH2027

# official repository (scorer, loaders, paths), pinned to the commit we used; a second copy is read by pablo_v6
git clone https://github.com/metrica-sports/more-than-a-hack-2027
git -C more-than-a-hack-2027 checkout f8f2629ff1e0985f81d7e59d8413678514ee1530
git clone https://github.com/metrica-sports/more-than-a-hack-2027 pablo_v6/docs/more-than-a-hack-2027
git -C pablo_v6/docs/more-than-a-hack-2027 checkout f8f2629ff1e0985f81d7e59d8413678514ee1530

# environments, all pinned (uv.lock and requirements files with a hash for every package)
(cd pablo_v6 && uv sync --frozen)
cd dfl
uv venv --python 3.11 .venv-idsse
uv pip install --python .venv-idsse -r requirements-idsse.txt
uv venv --python 3.12.0 .venv-tcn
uv pip install --python .venv-tcn --index-strategy unsafe-best-match --require-hashes -r requirements-tcn.txt
cd ..
```

Data: put the match bundle in `$MTAH_DATA_DIR/tactical/`, keeping the names it arrives with. Then download the DFL
open data into `$MTAH_DATA_DIR/external/idsse/`:

```sh
export MTAH_DATA_DIR=/path/to/data      # on Windows, e.g. export MTAH_DATA_DIR='C:\mtah-data'
sh dfl/download_idsse.sh
```

## Regenerate the submission

From the root of this repository:

```sh
export MTAH_DATA_DIR=/path/to/data
sh dfl/regenerar_e1.sh
```

The script checks the sha256 of every external file, downloads SkillCorner open data at a fixed commit, computes the
features, trains the models on the DFL, runs the network from its stored weights and writes the CSV. At the end it
compares the CSV with the expected sha256 and copies it to `LaMasIA.csv` at the root. It stops at the first error.

Derived files go to `$MTAH_DATA_DIR/cache/` and `pablo_v6/out/`, never into a tracked folder. The steps that read the
provided match print only aggregate figures. Threads are fixed at 4, and the same file comes out with 8.

The network is not retrained by default, because retraining on another machine does not give bit-identical weights.
Its stored weights (`dfl/pesos_red/`, trained only on the DFL) give the same CSV on Windows and Linux. `dfl/README.md`
explains how to retrain them.

Checked in a clean copy (fresh clones, new environments, no cache) on 15/11/2026: same sha256.

## Data

- **Match data (FC Barcelona vs Athletic Club):** not included and not redistributable. It is licensed for the
  competition only, and nothing in this repository is derived from it.
- **DFL open data (IDSSE):** Bassek, Rein, Weber and Memmert (2025), *Scientific Data*,
  doi:10.1038/s41597-025-04505-y. Data on figshare, doi:10.6084/m9.figshare.28196177, licence CC BY 4.0, downloaded
  from the mirror `huggingface.co/datasets/pysport/idsse-data`. Seven matches (2 Bundesliga, 5 2. Bundesliga, 2022/23):
  tracking at 25 Hz with per-frame ball possession and ball status. It is the only data the models are trained on.
  The weights in `dfl/pesos_red/` are derived from it.
- **SkillCorner open data:** <https://github.com/SkillCorner/opendata>, MIT licence, commit
  `4340d274572876239c154c90bc507a9b3250a656`. It is used only to measure how often the ball and the players are missing
  in broadcast tracking. No model is trained on it.
- **Official challenge repository:** <https://github.com/metrica-sports/more-than-a-hack-2027>, MIT licence. It is
  cloned during setup and not copied here.
