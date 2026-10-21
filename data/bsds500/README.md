# BSDS500

Las imágenes van en `images/{train,test,val}/` y **no se versionan** (~70 MB). Lo único versionado
de esta carpeta es la lista de evaluación y este archivo.

## El split, que es fácil de equivocar

BSDS500 reorganizó BSDS300: sus 200 imágenes de `train` siguen siendo `train`, sus 100 de `test`
pasaron a ser **`val`**, y se añadieron 200 nuevas como `test`.

**CBSD68** —el conjunto sobre el que el paper reporta 31.53 dB a σ=25— son 68 imágenes del test de
BSDS300, o sea del directorio **`val`** de BSDS500. Por eso las 400 imágenes de entrenamiento son
`train + test`, y `val` se reserva entero. Entrenar sobre `val` sería fuga de datos y volvería
incomparable el número publicado.

Que los 68 nombres de `cbsd68.txt` estén todos contenidos en `images/val/` es la verificación
directa de ese mapeo.

## `cbsd68.txt`

Los 68 nombres del conjunto de evaluación, uno por línea, sin extensión.
Fuente: https://github.com/clausmichele/CBSD68-dataset (`CBSD68/original`).

Es la lista que consume `--file-list` en `evaluate.py` y `EVAL_LIST` en `train.py`. Para otro
dataset, la lista equivalente va junto a *sus* imágenes, no acá.
