#!/bin/sh
# Descarga el dataset abierto IDSSE (DFL, CC-BY 4.0) desde el mirror de kloppy en Hugging Face.
set -e
OUT="${MTAH_DATA_DIR:?define MTAH_DATA_DIR}/external/idsse"
mkdir -p "$OUT"
BASE=https://huggingface.co/datasets/pysport/idsse-data/resolve/main
cd "$OUT"
for f in $(curl -s https://huggingface.co/api/datasets/pysport/idsse-data/tree/main | grep -o '"path":"DFL[^"]*"' | cut -d'"' -f4); do
  [ -s "$f" ] || curl -sSL --retry 3 -o "$f" "$BASE/$f"
  echo "ok $f"
done
