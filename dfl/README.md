# E1: posesión con modelos entrenados solo con datos externos

`make_e1.py` genera el CSV de BAR-ATH sin entrenar ni ajustar nada con el partido dado (correo de la organización del
03/11). Los modelos de equipo y de balón parado se entrenan solo con 7 partidos de la DFL. Todos los parámetros se
eligieron solo con datos externos, con criterios fijados antes de ver los resultados. No se usan la verdad pública, el
pitido ni el vídeo, ni constantes ligadas a frames del partido: las partes son los bloques oficiales
(`games.half_video_blocks`) y el saque inicial sale de la Smart Tagging. La verdad, además, queda bloqueada en tiempo de
ejecución (`dfllib.forbid_public_truth`): cualquier intento de abrir un fichero de verdad hace fallar el script.

Versión de la entrega: **`e1_v14s`** (15/11/2026). Las configuraciones anteriores (`v2` … `v14`) se conservan en el
código como historial; algunas necesitan datos que la entrega ya no usa (Metrica sample data, `barath_counts.py`).

## Qué hace

1. **DFL degradada como una cámara.** Se le quitan trozos del balón con tasas de huecos medidas en **SkillCorner open
   data** (con el juego vivo y parado), se pierden jugadores (~7 %, episodios de ~4 s) y se añade un ruido de 0,3 m al
   balón. Sobre ella se calculan las mismas features que en BAR-ATH: 166 sin balón (forma de los equipos) y 54 con balón.
2. **Etiqueta de entrenamiento.** La posesión de la DFL se comporta como de último toque. Se convierte a la definición
   del reto: un tramo rival de menos de 1 s es un toque sin control, y el cambio de equipo se pone a mitad del vuelo del
   pase.
3. **Árboles del modelo de equipo** (HistGradientBoosting, 3 semillas). P_ball usa las features con balón. P_free usa
   las de sin balón más su cambio en ±2 s (`ctxlib.py`). Con el balón visible se combinan en un producto de expertos;
   sin él se usa solo P_free. Cada fila de entrenamiento pesa 1 / max(duración de su posesión, 2 s), con los pesos
   normalizados a media 1 y el prior ponderado igual (`make_e1.spell_weights`): así cada posesión cuenta casi lo mismo,
   sea corta o larga. **Intercambio de equipos:** cada partido de la DFL entra dos veces, la segunda con local y
   visitante intercambiados y la etiqueta invertida (`make_e1.swap_game`; features en `feat_swap_sc`).
4. **Red temporal (TCN, `tcnlib.py`).** Las mismas features a 5 Hz (227 canales), 6 bloques residuales con
   dilataciones 1-32 (ve ±25 s), 3 semillas; entrenada solo con la DFL (`tcn_train.py prod`). Se mezcla con los árboles
   0,5 / 0,5 en log-odds alrededor del prior. **La entrega no la reentrena:** usa sus pesos guardados en `pesos_red/`
   (ver «Reproducibilidad»).
5. **Balón parado.** Un GBM sin balón entrenado solo con la DFL (`dfl_dead.py`), con jugadores ocultos como los oculta
   la cámara: el número de jugadores visibles por frame se sortea de la distribución medida en SkillCorner open data
   (`skillcorner_counts.py`, solo frames con algún jugador detectado).
6. **Decodificación.** HMM conjunto H/A/D por parte (`hmm3.viterbi3` de `scripts/explore/`), con una duración mínima
   de los parones. Temperatura 0,3216796875 para el equipo: la que da en la DFL el mismo total de posesiones que la
   versión anterior (bisección, sin mirar ninguna verdad); para el balón parado, `temp_d` 0,2 y `bias` 0.
7. **Saque inicial.** En cada parte, `D` desde el inicio del bloque puntuado hasta el inicio de la fase `Kick Off
   (Start)` de la Smart Tagging (cargador oficial `starter/load_smart_tagging.py`). El FAQ oficial avisa de que la 2ª
   parte se puntúa desde antes del saque y de que ese tramo es todo `D`. Se usa el inicio de la fase tal cual, que cae
   unos segundos antes del golpeo; en la 1ª parte no cambia nada (se puntúa desde el saque).

## Datos externos

- **DFL / IDSSE:** Bassek, Rein, Weber y Memmert (2025), *Scientific Data*, doi:10.1038/s41597-025-04505-y, licencia
  CC-BY 4.0 (datos en figshare, doi:10.6084/m9.figshare.28196177; se descargan del mirror
  `huggingface.co/datasets/pysport/idsse-data`). Datos oficiales de partido de la DFL: 7 partidos (2 de Bundesliga y 5
  de 2. Bundesliga, 2022/23), posiciones TRACAB a 25 Hz con la posesión y el balón activo / inactivo por frame. **Es el
  único dato con el que se entrena** (árboles, red y balón parado). No se usan sus eventos.
- **SkillCorner open data:** <https://github.com/SkillCorner/opendata>, licencia MIT, commit
  `4340d274572876239c154c90bc507a9b3250a656`: tracking de retransmisión (10 fps) y fases de juego de 20 partidos de la
  A-League 2024/25. Solo se usa para medir cuánto y cuánto tiempo falta el balón con el juego vivo y parado, y cuántos
  jugadores ve una cámara; no se entrena con él.
- **Metrica Sports sample data: ya no se usa.** Entrenaba el modelo de balón parado hasta e1_v12; se retiró en e1_v13
  porque su repositorio no declara licencia.
- `external_sha256.txt` guarda el sha256 de los 21 ficheros de la DFL (`verify_external.py` lo comprueba antes de
  empezar) y `skillcorner_sha256.txt` el de los 40 de SkillCorner (`skillcorner_download.py` lo comprueba al
  descargar).

## Preparación (una vez)

Los comandos están en el [README de la raíz](../README.md). Estructura (`<raíz>` = la raíz de este repo):

```
<raíz>\
  more-than-a-hack-2027\           git clone del repo oficial (commit f8f2629)
  pablo_v6\                        código de features y decodificador + uv sync --frozen
    docs\more-than-a-hack-2027\    otro clon del repo oficial (lo usa scripts/explore/common.py)
  dfl\                             estos scripts (con pesos_red\)
    .venv-idsse\                   uv venv --python 3.11 .venv-idsse
                                   uv pip install --python .venv-idsse -r requirements-idsse.txt
    .venv-tcn\                     uv venv --python 3.12.0 .venv-tcn
                                   uv pip install --python .venv-tcn --index-strategy unsafe-best-match
                                     --require-hashes -r requirements-tcn.txt
MTAH_DATA_DIR\                     datos del reto (tactical\) y de la DFL (external\idsse\: sh dfl/download_idsse.sh)
```

`pablo_v6/` es el commit `524becb` ("Write submission CSV with LF line endings on every OS", 29/10) del repo de
trabajo del equipo, con el código sin modificar (solo se han quitado un cuaderno y cinco scripts que no se usan): features (`fe_geometry`, `fb_ball`), limpieza de porteros (`gklib`), modelo de D (`deadlib`) y
decodificador (`hmm3`, `make_submission.decode`). Versiones que fija su `uv.lock`: Python 3.12.0, numpy 2.5.4,
scikit-learn 1.9.1, pandas 3.1.0, scipy 1.18.1. `requirements-tcn.txt` (generado desde `requirements-tcn.in`) fija la
versión y el sha256 de cada paquete de la red: torch 2.14.1+cpu, numpy 2.5.4, scikit-learn 1.9.1.

## Regenerar `e1_v14s.csv`

Con un solo comando, desde `<raíz>` en Git Bash:

```
export MTAH_DATA_DIR=<ruta a los datos>
sh dfl/regenerar_e1.sh
```

| paso | script | qué hace | lee BAR-ATH |
|---|---|---|---|
| 1 | `verify_external.py` | sha256 de los datos de la DFL | no |
| 2 | `idsse_cache.py` | XML de la DFL → `.npz` | no |
| 3 | `skillcorner_download.py <commit>` | descarga SkillCorner (~1,7 GB) y comprueba cada sha256 | no |
| 4 | `skillcorner_gap_model.py` | 4 tasas de huecos del balón (vivo / parado) medidas en SkillCorner | no |
| 5 | `skillcorner_counts.py` | jugadores visibles por frame medidos en SkillCorner (para ocultar jugadores al entrenar D) | no |
| 6 | `barath_features.py` | features de BAR-ATH; ni etiquetas ni entrenamiento | sí |
| 7 | `make_e1.py build <partido>` ×7 | features de entrenamiento de la DFL degradada | no |
| 8 | `make_e1.py build <partido> swap` ×7 | las mismas, con los equipos intercambiados | no |
| 9 | `tcn_export.py entrega` | entradas de la red para BAR-ATH (5 Hz); ni etiquetas ni entrenamiento | sí |
| 10 | `tcn_train.py prod_pesos pesos_red` | comprueba el sha256 de los pesos y predice BAR-ATH con la red cargada de ellos | sí |
| 11 | `make_e1.py` | entrena los árboles y el balón parado con la DFL, mezcla con la red, decodifica y escribe el CSV | sí (y la Smart Tagging) |

El CSV sale en `more-than-a-hack-2027/evaluation/results/e1_v14s.csv`, con saltos de línea `\n`, y se copia a
`<raíz>/LaMasIA.csv` si el sha256 coincide con el esperado. Lo derivado se guarda en `MTAH_DATA_DIR/cache/` y en
`pablo_v6/out/`; nada se escribe dentro de un repo. Los pasos que leen BAR-ATH solo imprimen cifras agregadas.

**Resultado esperado:** sha256 `b8d75e1aea161b6ea2883417454016a5e4833babb42aae7d9a0d11b86eafb184` (e1_v14s). Generado
el 15/11/2026; mismo sha con 4 y con 8 hilos. Comprobado en una copia limpia
(clones nuevos, entornos nuevos, sin cachés, SkillCorner descargado de nuevo) el 15/11/2026: mismo sha, 1 h 48 min en
un portátil con Windows 11.

## Reproducibilidad de la red

Reentrenar la red en otra máquina no da bit a bit los mismos pesos (las librerías de cálculo de la CPU cambian el orden
de las sumas), y así el CSV cambiaba en unos pocos frames. Por eso la entrega incluye los pesos de producción:
`pesos_red/prod_weights_s{0,1,2}.npz` (~550 KB cada uno; pesos de la red, normalización de las entradas, temperatura y
prior, todo calculado solo con la DFL), con su sha256 en `pesos_red/sha256.txt`. Predecir con ellos es determinista:
con estos pesos el CSV sale idéntico en Windows y en Linux. Para reentrenarlos: `tcn_export.py` (exporta también el
entrenamiento de la DFL) y `tcn_train.py prod`, en `.venv-tcn`.

## Qué se usa de BAR-ATH y por qué (para el documento técnico)

- **Solo como entrada del modelo:** tracking, visibilidad del balón y las features calculadas a partir de ellos; la
  Smart Tagging, solo para el saque inicial de cada parte (entrada permitida de forma explícita).
- **Nada de BAR-ATH se usa para entrenar, ajustar, degradar ni elegir.** Desde e1_v12 ninguna estadística de BAR-ATH
  entra en el entrenamiento (antes: los jugadores visibles por frame, `barath_counts.py`); los huecos del balón y los
  jugadores visibles salen de SkillCorner. Los modelos se entrenan con la DFL, y las decisiones (ventanas de D, limpieza
  de porteros, temperatura, parámetros de D, contexto temporal, pesos por duración de la posesión, red temporal,
  intercambio de equipos) se tomaron con jueces externos degradados sin ninguna cifra de BAR-ATH.

## Historial de versiones

| versión | cambio | sha256 |
|---|---|---|
| e1_v7 (09/11) | pesos por duración de la posesión y T del mismo ritmo | `2751cc5f…` |
| e1_v8 (10/11) | huecos del balón de la DFL de entrenamiento medidos en SkillCorner en vez de en BAR-ATH | `57bb8818…` |
| e1_v9 (11/11) | sin la regla de «balón fuera» del decodificador (su forma venía de un ajuste antiguo con la ventana pública) | `be845dd0…` |
| e1_v10 (11/11) | `D` hasta el saque inicial de la Smart Tagging | `0672ecaa…` |
| e1_v11 (12/11) | descartada: saque de la 2ª parte con `D` obligado | — |
| e1_v12 (12/11) | jugadores visibles del entrenamiento de D medidos en SkillCorner en vez de en BAR-ATH | `72d1c17c…` |
| e1_v13 (12/11) | modelo de balón parado entrenado solo con la DFL (sin Metrica sample data) | `9607b7eb…` |
| e1_v14 (14/11) | red temporal (TCN) mezclada con los árboles | `5c40cb8d…` |
| **e1_v14s** (15/11) | árboles entrenados también con los equipos intercambiados | `b8d75e1a…` |

e1_v8 a e1_v13 son cambios de cumplimiento; en los jueces externos su coste conjunto frente a e1_v7 es de unas
milésimas de spellF1. e1_v14 sube +0,012 de media en las 4 verdades de los jueces DFL y e1_v14s, +0,016 más (sobre
todo en el juez duro).

**Bloqueo de la verdad (11/11):** `dfllib.forbid_public_truth` instala un audit hook de Python que hace fallar cualquier
apertura de un fichero de verdad (`*possession_gt*` o lo que esté bajo `evaluation/gt/`), venga de `open`, pandas o
numpy. `make_e1.py build` lo activa y sustituye `featlib.labels` de `scripts/explore/` (que leía la verdad pública al
construir las features de la DFL, sin usarla) por una función que no lee nada. Se comprobó que las features y el CSV
salen idénticos byte a byte con el bloqueo.

La validación (jueces DFL «suave» y «duro» degradados con cifras de SkillCorner y de la documentación oficial, guarda
RV2 y regla de ruido entre semillas) está en el repo privado de notas del equipo.
