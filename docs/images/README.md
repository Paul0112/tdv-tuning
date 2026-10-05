# Procedencia de las figuras

Exportadas el 30 de septiembre de 2026. Las celdas se cuentan desde 1, incluyendo Markdown; la figura se cuenta desde 1 dentro de las salidas PNG de esa celda. El contador de ejecucion identifica el estado guardado, no una ejecucion nueva ni una garantia de reproducibilidad.

Las figuras siguientes se extrajeron directamente del contenido `image/png` de los notebooks, sin editar sus pixeles ni recalcular resultados.

| Archivo | Notebook | Celda | Figura | Contador de ejecucion |
|---|---|---:|---:|---:|
| [tdv-iterations.png](tdv-iterations.png) | [hyperparameter_sensibility](../../notebooks/hyperparameter_sensibility.ipynb) | 23 | 1 | 17 |
| [tdv-time.png](tdv-time.png) | [hyperparameter_sensibility](../../notebooks/hyperparameter_sensibility.ipynb) | 25 | 1 | 18 |
| [tdv-lambda.png](tdv-lambda.png) | [hyperparameter_sensibility](../../notebooks/hyperparameter_sensibility.ipynb) | 27 | 1 | 19 |
| [poisson-noise.png](poisson-noise.png) | [noise_types](../../notebooks/noise_types.ipynb) | 15 | 1 | 11 |
| [quantization.png](quantization.png) | [bits_depth](../../notebooks/bits_depth.ipynb) | 16 | 1 | 11 |
| [bits8-denoising.png](bits8-denoising.png) | [bits_depth](../../notebooks/bits_depth.ipynb) | 18 | 2 | 13 |
| [float32-denoising.png](float32-denoising.png) | [bits_depth](../../notebooks/bits_depth.ipynb) | 18 | 4 | 13 |
| [photometry-ratio.png](photometry-ratio.png) | [photometry_anscombe](../../notebooks/photometry_anscombe.ipynb) | 25 | 2 | 12 |
| [salt-pepper-noise.png](salt-pepper-noise.png) | [noise_types](../../notebooks/noise_types.ipynb) | 21 | 1 | 15 |
| [gaussian-noise.png](gaussian-noise.png) | [hyperparameter_sensibility](../../notebooks/hyperparameter_sensibility.ipynb) | 14 | 1 | 11 |
| [bits4-denoising.png](bits4-denoising.png) | [bits_depth](../../notebooks/bits_depth.ipynb) | 18 | 1 | 13 |

## Grafico de stretches

[domain-stretch.png](domain-stretch.png) se dibujo con Matplotlib a partir de las salidas de texto de las celdas 47, 51 y 55 de [domain_shift](../../notebooks/domain_shift.ipynb). No es una nueva inferencia ni una extraccion de pixeles de otra figura.

| Rama | PSNR utilizado (dB) |
|---|---:|
| Lineal | 64.45924 |
| Asinh | 58.566277 |
| Raiz cuadrada | 57.678726 |
| Log | 57.376595 |

Para actualizar las figuras, ejecutar el experimento correspondiente y exportar su salida PNG. Revisar tambien cifras y conclusiones del README; las imagenes no se actualizan automaticamente.

## Actualización de la comparación Poisson (1 de octubre de 2026)

Estas tres figuras corresponden a las pruebas con peak=100. La figura de fidelidad Poisson reemplaza la exportación anterior. El caso raw se toma de la sección Poisson Noise (celda 18), que tiene sus mediciones de flujo y residuo inmediatamente después, no de la otra ejecución de Raw poisson (celda 28).

| Archivo | Notebook | Celda | Figura | Contador de ejecución |
|---|---|---:|---:|---:|
| [poisson-raw-peak100.png](poisson-raw-peak100.png) | [photometry_anscombe](../../notebooks/photometry_anscombe.ipynb) | 18 | 1 | 33 |
| [anscombe-peak100.png](anscombe-peak100.png) | [photometry_anscombe](../../notebooks/photometry_anscombe.ipynb) | 38 | 1 | 47 |
| [poisson-fidelity.png](poisson-fidelity.png) | [photometry_anscombe](../../notebooks/photometry_anscombe.ipynb) | 61 | 1 | 65 |

Flujos, cocientes y residuos: celdas 19–21 (raw), 39–41 (Anscombe) y 62–64 (fidelidad Poisson). PSNR y SSIM transcritos de las etiquetas de las figuras. Los números de celda corresponden a esta versión del notebook.


### Escalado en otros ruidos (2026-10-05)

Exportaciones directas de las salidas PNG guardadas de `notebooks/noise_types.ipynb`, sin volver a ejecutar los experimentos. Índices de celda basados en cero:

- `poisson-scaled-sigma5.png`: celda 19, primera figura (PSNR).
- `poisson-scaled-sigma100.png`: celda 31, primera figura (PSNR).
- `salt-pepper-scaled-sigma5.png`: celda 39, primera figura (PSNR).
- `salt-pepper-scaled-sigma100.png`: celda 47, primera figura (PSNR).
