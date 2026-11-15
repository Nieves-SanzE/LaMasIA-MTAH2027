#!/bin/sh
# Regenera la entrega e1_v14s desde cero (Git Bash en Windows o sh en Linux/macOS). Ver README.md.
#
#   export MTAH_DATA_DIR=...        # datos del reto (tactical/) y de la DFL (external/idsse/)
#   sh dfl/regenerar_e1.sh          # desde la raíz del repo (contiene more-than-a-hack-2027/, pablo_v6/ y dfl/)
#
# Supone hecha la preparación del README de la raíz: clon oficial, pablo_v6 (uv sync + docs/more-than-a-hack-2027),
# .venv-idsse y .venv-tcn en dfl/ y los XML de la DFL. SkillCorner open data se descarga aquí (paso 3, ~1,7 GB), fijado a un commit y
# comprobado con skillcorner_sha256.txt. Ninguna caché previa: todo lo derivado se calcula aquí, en MTAH_DATA_DIR/cache/
# y en pablo_v6/out/. Los pasos que leen BAR-ATH solo imprimen cifras agregadas.
# La red temporal no se reentrena: se usan sus pesos guardados (dfl/pesos_red/, entrenados solo con la DFL y
# comprobados con su sha256), porque reentrenarla en otra máquina no da bit a bit los mismos pesos. Con estos pesos el
# CSV sale idéntico en Windows y en Linux. `tcn_export.py` + `tcn_train.py prod` (en .venv-tcn) los reentrenan.
# Historial: hasta el 11/11 regeneraba e1_v7 (sha 2751cc5f...), hasta el 14/11 e1_v10 (sha 0672ecaa...) y hasta el
# 15/11 e1_v13 (sha 9607b7eb..., sin red ni intercambio de equipos).
set -e
: "${MTAH_DATA_DIR:?define MTAH_DATA_DIR}"
RAIZ=$(pwd)
PY="$RAIZ/pablo_v6/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="$RAIZ/pablo_v6/.venv/bin/python"
PYX="$RAIZ/dfl/.venv-idsse/Scripts/python.exe"
[ -x "$PYX" ] || PYX="$RAIZ/dfl/.venv-idsse/bin/python"
PYT="$RAIZ/dfl/.venv-tcn/Scripts/python.exe"
[ -x "$PYT" ] || PYT="$RAIZ/dfl/.venv-tcn/bin/python"
export OMP_NUM_THREADS=4
MATCHES="J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9"
SKILLCORNER_COMMIT=4340d274572876239c154c90bc507a9b3250a656
CSV=more-than-a-hack-2027/evaluation/results/e1_v14s.csv
ESPERADO=b8d75e1aea161b6ea2883417454016a5e4833babb42aae7d9a0d11b86eafb184
paso() { echo; echo "=== $1 ($(date +%H:%M:%S))"; }

paso "1/11 datos externos de la DFL con el sha256 esperado"
"$PY" -u dfl/verify_external.py "$RAIZ/pablo_v6"

paso "2/11 XML de la DFL -> npz"
"$PYX" -u dfl/idsse_cache.py

paso "3/11 SkillCorner open data (commit fijo; cada fichero con el sha256 anotado)"
"$PY" -I -X utf8 -u dfl/skillcorner_download.py "$SKILLCORNER_COMMIT"

paso "4/11 modelo de huecos del balón medido en SkillCorner"
"$PY" -I -X utf8 -u dfl/skillcorner_gap_model.py

paso "5/11 jugadores visibles por frame medidos en SkillCorner (entrenamiento del modelo de balón parado)"
"$PY" -I -X utf8 -u dfl/skillcorner_counts.py

cd "$RAIZ/pablo_v6"
paso "6/11 features de BAR-ATH (sin etiquetas ni entrenamiento)"
"$PY" -u ../dfl/barath_features.py

paso "7/11 features de entrenamiento de la DFL degradada con los huecos de SkillCorner (7 partidos, 4 a la vez)"
export OMP_NUM_THREADS=2
printf "%s\n" $MATCHES | xargs -P 4 -I{} "$PY" -u ../dfl/make_e1.py build {}

paso "8/11 las mismas features con los equipos intercambiados (local <-> visitante; 7 partidos, 4 a la vez)"
printf "%s\n" $MATCHES | xargs -P 4 -I{} "$PY" -u ../dfl/make_e1.py build {} swap
export OMP_NUM_THREADS=4

paso "9/11 entradas de la red temporal para BAR-ATH (sin etiquetas ni entrenamiento)"
"$PY" -u ../dfl/tcn_export.py entrega
cd "$RAIZ"

paso "10/11 red temporal con sus pesos guardados (sha256 comprobado; sin reentrenar)"
(cd dfl/pesos_red && sha256sum -c sha256.txt)
"$PYT" -u dfl/tcn_train.py prod_pesos dfl/pesos_red

cd "$RAIZ/pablo_v6"
paso "11/11 modelos y CSV de BAR-ATH (e1_v14s)"
"$PY" -u ../dfl/make_e1.py
cd "$RAIZ"
"$PY" more-than-a-hack-2027/evaluation/validate_submission.py "$CSV"
SHA=$(sha256sum "$CSV" | cut -d' ' -f1)
echo "sha256:   $SHA"
echo "esperado: $ESPERADO"
[ "$SHA" = "$ESPERADO" ] || { echo "ERROR: el CSV no coincide"; exit 1; }
cp "$CSV" LaMasIA.csv
echo "OK: e1_v14s regenerado idéntico; copiado a $RAIZ/LaMasIA.csv"
