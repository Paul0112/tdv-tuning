# Checkpoints TDV

Este directorio contiene cuatro variantes de TDV entrenadas en el fork
[Training-TDV](https://github.com/Paul0112/Training-TDV). Se usan para comparar ruido
gaussiano y Poisson, fidelidades L2 y Poisson, y entrenamiento con peak fijo o variable.
Las evaluaciones y ejemplos estan en
[`checkpoints_inference.ipynb`](../notebooks/checkpoints_inference.ipynb).

## Resumen

| Nombre en los experimentos | Carpeta | Ruido de entrenamiento | Fidelidad interna | `lambda_init` | Recorte de gradiente |
|:--|:--|:--|:--|--:|:--|
| G-5k / G-L2 | `test_5k_gaussian/` | Gaussiano, sigma=25 | L2 (`denoise`) | 0.1 | Desactivado (`clip=Infinity`) |
| P-L2 | `test_5k_poisson/` | Poisson, peak=50 | L2 (`denoise`) | 0.1 | Desactivado (`clip=Infinity`) |
| P-P | `test_5k_poisson_fidelity/` | Poisson, peak=50 | Poisson, sin factor peak explicito | 0.1 | Desactivado (`clip=Infinity`) |
| P-P mixed / P-P v2 | `test_5k_poisson_fidelity_v2/` | Poisson, peaks entre 5 y 100 | Poisson, con factor peak explicito | 0.002 | Norma maxima 2.0 |



## Configuracion compartida

- 5000 iteraciones solicitadas, semilla 0 y modo `referencia`.
- Imagenes de tres canales y regularizador `tdv` de 32 features, tres escalas y tres
  macrobloques; `multiplier=1`.
- Red desenrollada con `S=10` pasos y fidelidad proximal: `use_prox=True`.
- `T` y `lambda` aprendidos, restringidos al intervalo [0, 1000].
- `T_init=0.1011928851253881`.
- Dataset BSDS500: entrenamiento en `train,test` y validacion en `val`, restringida
  mediante `datasets/bsds500/cbsd68.txt`.

- `sigma_min=0` y `sigma_max=0`: el modo gaussiano de sigma variable esta desactivado.
  En las corridas Poisson, el ruido lo determina el peak, aunque `config.json`
  conserve el argumento `sigma=25`.

test_5k_poisson y test_5k_poisson_fidelity y pl2_50_unclipped_20 y pp50_unclipped_s0 corresponden al mismo entrenamiento respectivamente, la unica variación es el ruido poisson recortado en los primeros.

## G-5k: ruido gaussiano y fidelidad L2

[`test_5k_gaussian/`](test_5k_gaussian/) es la referencia gaussiana: aprende a restaurar
imagenes degradadas con sigma=25, equivalente a una desviacion 25/255 sobre imagenes
normalizadas. 


## P-L2: ruido Poisson y fidelidad L2

[`test_5k_poisson/`](test_5k_poisson/) cambia la degradacion de entrenamiento a Poisson
con peak=50, pero mantiene la fidelidad L2 y `lambda_init=0.1`.

Permite estudiar cuanto puede aprender el regularizador del ruido Poisson sin
introducir una fidelidad Poisson. Su inferencia directa usa los pesos cargados, sin
el escalado de fidelidad por peak aplicado a las variantes P-P.

## P-P: ruido y fidelidad Poisson, peak fijo

[`test_5k_poisson_fidelity/`](test_5k_poisson_fidelity/) usa ruido Poisson con peak=50,
fidelidad Poisson y `lambda_init=0.1`. En esta version el factor peak no se introdujo
explicitamente en cada paso del proximal:

```python
tau_R = self.T / self.S
tau_D = self.lmbda / self.S
```

Como el peak de entrenamiento es fijo y `lambda` se aprende, este parametro puede
absorber la escala de la fidelidad.

En el barrido experimental **P-P scaled**, se adapta la fidelidad a otro peak mediante:

```python
lambda_inferencia = lambda_checkpoint * (peak / 50.0)
```

Esto es una adaptacion de inferencia del checkpoint fijo, no un nuevo entrenamiento.

## P-P mixed: peaks variables y fidelidad escalada

[`test_5k_poisson_fidelity_v2/`](test_5k_poisson_fidelity_v2/) incorpora estos cambios
respecto a P-P:

1. Peaks variables entre 5 y 100 durante el entrenamiento.
2. Factor peak explicito dentro de la fidelidad proximal.
3. `LAMBDA_INIT=0.002` y recorte de norma de gradiente `--clip 2.0`.





## Archivos y registro de entrenamiento

Cada carpeta incluye:

- `best.pth`: checkpoint con mejor PSNR en las evaluaciones realizadas durante la corrida.
- `last.pth`: checkpoint al finalizar.
- `config.json`: argumentos de entrenamiento. La arquitectura y la inicializacion
  de `lambda` tambien deben consultarse en `checkpoint['config']`.
- `estado.json`: resumen de la corrida, rechazos y rebobinados.
- `history.json`: historial de validacion.
- `traza.npy`: diagnosticos de entrenamiento por iteracion.

En las cuatro corridas guardadas, tanto `best.pth` como `last.pth` reportan la
iteracion 5000. Los registros muestran:

| Modelo | Lotes rechazados / 5000 | Rebobinados |
|:--|--:|--:|
| G-5k | 0 | 0 |
| P-L2 | 37 | 1 |
| P-P | 116 | 3 |
| P-P mixed | 528 | 6 |


