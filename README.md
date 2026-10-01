# TDV: denoising e imágenes astronómicas

Este repositorio explora hasta dónde se puede utilizar el algoritmo de Total Deep Variation (TDV), con sus correpondientes pesos entrenados para ruido gaussiano en imágenes naturales en tareas externas; cambio de ruido, de representación de la imagen o el dominio de aplicación (imágenes astronómicas). 

Los experimentos toman de base los checkpoints gris y color de [VLOGroup/tdv](https://github.com/VLOGroup/tdv), asociados al trabajo de [Kobler et al., *Total Deep Variation for Linear Inverse Problems* (CVPR 2020)](https://arxiv.org/abs/2001.05005). Los notebooks exploran esos modelos mediante inferencia y cambios de configuración (no hay un reentrenamiento de la red como tal).

## Overview

- **Más iteraciones no garantizan mejor reconstrucción.** El modelo depende del equilibrio entre regularizador, fidelidad y tamaño de paso, no se evidenció una solo parámetro que mejorará el modelo a per se.
- **El preprocesamiento cambia el problema que se debe resolver.** Aplicar clipping y stretching modifican la distribución del ruido y reducen la calidad de la reconstrucción.
- **La profundida de bits no incide en la calidad de denoising.** Cuantizar la misma observación a 8 bits apenas cambia las métricas respecto de float32, sin embargo, llegados a 4 bits la decadencia en el rendimiento es evidente.
- **Una mejora visual (PSNR, SSIM) no garantiza conservación del flujo.** En las pruebas fotométricas (relativas) se evidencian pérdidas en las aperturas estelares, incluso cuando la imagen parece más limpia (adempas, el residuo que se mantiene parcialmente nulo).
- **Modelar ruido Poisson presenta mejoras en métricas básicas.** Anscombe y la fidelidad Poisson, estrategias empleadas para modelar el input de ruido Poisson, requieren un mayor número de pruebas, en primera instancia presentan mejoras al modelo base de TDV.


## 1. Variación de hiperparámetros

En [hyperparameter_sensibility.ipynb](notebooks/hyperparameter_sensibility.ipynb) se varían el ruido, el escalado de entrada, el número de pasos `S`, `T` y el peso de fidelidad `lambda`. Con la fidelidad L2 y el proximal utilizados por el checkpoint, la actualización implementada es:

$$
x_{k+1}=\frac{x_k-(T/S)\nabla R(x_k)+(\lambda/S)z}{1+\lambda/S}.
$$

En el barrido sobre `water-castle`, pasar de `S=1` a `S=10` eleva el PSNR de aproximadamente 14.0 a 32.4 dB. Continuar hasta `S=30` no lo mejora. Al variar `S` con `T` y `lambda` fijos, también cambian los tamaños de paso; no se está aumentando simplemente el tiempo total de denoising. Los resultados obtenidos se condicen con lo presentado en el paper de Kobler et. al:

![PSNR frente al número de pasos de TDV](docs/images/tdv-iterations.png)

![PSNR frente al tiempo T](docs/images/tdv-time.png)



Con `lambda=0` desaparece la fidelidad, pero sigue actuando el regularizador desde la inicialización `x₀=z`. El ensayo conserva un PSNR de **32.43 dB**, cercano a los **32.49 dB** de `lambda=0.1` (esto lleva a la idea de que el regularizador está pudiendo trabajar por sí solo, en dado caso concreto).

![PSNR frente al peso de fidelidad lambda](docs/images/tdv-lambda.png)

Adicionalmente, se explora la incidencia de disminuir los macroblocks en la inicialización del modelo, dobde se obtuvieron resultados claramente degradados (una especie de aproximación a $TDV_1$, $TDV_2$).

## 2. Generalización a otro tipo de ruidos 

[noise_types.ipynb](notebooks/noise_types.ipynb) prueba ruido gaussiano, Poisson, sal y pimienta, y una combinación gaussiana–Poisson. En los barridos no gaussianos se utiliza el checkpoint original, con fidelidad L2 y `scaled=False` (ya que como no hay ruido gaussiano, no se puede transferir directamente).


![PSNR frente al nivel de ruido gaussiano](docs/images/gaussian-noise.png)

Para Poisson se genera `z = Poisson(peak · y) / peak`. donde `peak` representa el número de fotones capturados para un pixel, por tanto a mayor peak se obtiene una imagen menos degradada (No clip en esta parte). En (`peak=1000`), la curva guardada muestra que la salida puede ser peor que la entrada, un denoising "obligado" puede degradar innecesariamente.
![PSNR de entrada y salida bajo ruido Poisson](docs/images/poisson-noise.png)


El en caso de añadir ruido de sal y pimienta el resultado es notablemente inferior a los demás casos probados. La función log-t-student, según Kobler et. al, es usada por su suavidad en el problema, no está diseñada para un ruido abrupto y aleatorio como el presentado.

![PSNR de entrada y salida bajo ruido de sal y pimienta](docs/images/salt-pepper-noise.png)

## 3. Representación de la imagen 

En [bits_depth.ipynb](notebooks/bits_depth.ipynb) se genera una única imagen ruidosa y luego la representa mediante cuantización de 4, 8 y 16 bits, conservando float32 como referencia sin cuantización adicional para que TDV pueda leerlo. Además se analiza la incidencia del clipping antes del input a la red.


![Efecto visual de cuantizar la imagen limpia a distintas profundidades](docs/images/quantization.png)


No se observó una clara ventaja en la representación de los bits a distintas profundidades, a excepción de la cuantización a 4 bits, que empeorece el resultado de las métricas.

**Cuantización a 4 bits: resultado del denoising.**

![Denoising de la observación cuantizada a 4 bits](docs/images/bits4-denoising.png)

**Cuantización a 8 bits: resultado del denoising.**
![Denoising de la observación cuantizada a 8 bits](docs/images/bits8-denoising.png)


**Cuantización a 8 bits: resultado del denoising.**

![Denoising de la observación en float32](docs/images/float32-denoising.png)



### Clipping


| Tratamiento en `water-castle` | PSNR de salida | SSIM |
|---|---:|---:|
| Sin recorte de entrada ni salida | 32.449 dB | 0.9055 |
| Recorte solo de salida | 32.464 dB | 0.9056 |
| Recorte de entrada, sin recorte de salida | 31.559 dB | 0.9023 |

Recortar la entrada a `[0,1]` mejora su PSNR antes de denoising, pero empeora la reconstrucción posterior en este ejemplo, lo esperado era que, al tener menor ruido, debido a outliers en los pixeles, TDV rindiera mejor. La idea de estos resultados es que el recorte elimina información y deja de preservar el ruido gaussiano original, lo que dificulta la tarea de denoising ala lgoritmo.

## 4. Dominio astronómico

[domain_shift.ipynb](notebooks/domain_shift.ipynb) combina simulaciones, imágenes planetarias, nebulosas y observaciones con ruido real. Se analiza además, la implicancia del preprocesamiento: se aplica TDV antes y después de tres stretches.

![PSNR en dominio lineal para los tres stretches y la rama lineal](docs/images/domain-stretch.png)

En general, TDV tiende a difuminar/borrar/extrapolar fuentes puntuales de estrellas, que se postula, a priori, confunde con ruido gaussiano.




## 5. Fotometría relativa 

[photometry_anscombe.ipynb](notebooks/photometry_anscombe.ipynb) incorpora fotometría relativa por aperturas, con posiciones detectadas en una referencia común (la imagen limpia original) y estimación de fondo mediante anillos (en base a y_clean). El objetivo es comprobar conservación de flujo dentro del mismo procesamiento.

Luego de testear en diferentes archivos, teniendo en cuanta que Un cociente ideal de conservación sería 1, se aprecia que al aplicar, ya sea ruido Poisson o Gaussiano, se subestima la conservación de flujo.
| Ensayo | Mediana del cociente de flujos por fuente |
|---|---:|
| M20 real: denoised / noisy | 0.946 |
| Estrellas, ruido gaussiano: noisy / referencia | 0.990 |
| Estrellas, ruido gaussiano: denoised / referencia | 0.901 |
| Estrellas, ruido Poisson: noisy / referencia | 1.006 |
| Estrellas, ruido Poisson: denoised / referencia | 0.897 |


![Mediana del flujo relativo de las fuentes tras denoising](docs/images/photometry-ratio.png)



Además, se analizó el residuo del proceso de denoising; este puede tener media casi nula y aun así contener estructura correlacionada con la referencia. Lo anterior, se puede usar de explicación a la subestimación en el flujo de la imagen denoised.

## 6. Anscombe y fidelidad Poisson

Se exploran dos formas de aproximar TDV al ruido de Poisson:

- Transformada de Anscombe: Aproximar ruido poisson a gaussiano.
$$f(z) = 2 \sqrt{z + \frac{3}{8}}$$

- Fidelidad Poisson: Modelar el término de fidelidad como ruido poisson, 

$$p(z \mid x) = \prod_{i} \frac{x_i^{z_i} e^{-x_i}}{z_i!}$$

### Comparación, peak = 100


**Poisson raw + TDV (fidelidad L2).**

![Denoising de Poisson raw en el campo estelar con peak 100](docs/images/poisson-raw-peak100.png)

**Anscombe + TDV + transformación inversa.**

![Denoising con Anscombe en el campo estelar con peak 100](docs/images/anscombe-peak100.png)

**TDV con fidelidad Poisson.**

![Prueba de fidelidad Poisson en el campo estelar con peak 100](docs/images/poisson-fidelity.png)


Los resultados presentan poca variación en PSNR, sin embargo SSIM se ve favorecido por las transformaciones. En mayores escenarios de degradación (peak = 50, 10), la diferencia se hace más evidente. A priori, la transformación ayuda a TDV a manejar mejor el ruido Poisson.
| Método | PSNR de salida (dB) | SSIM de salida |
|---|---:|---:|
| Poisson raw + L2 | 34.66 | 0.7862 |
| Anscombe + L2 | 36.01 | 0.8510 |
| Fidelidad Poisson | 35.30 | 0.8214 |



**Flujo en aperturas**:

| Método | Flujo de referencia y | Flujo ruidoso z | Flujo denoised x | Mediana Fz/Fy | Mediana Fx/Fy |
|---|---:|---:|---:|---:|---:|
| Poisson raw + L2 | 788.244742 | 785.838125 | 717.384693 | 0.997506 | 0.875619 |
| Anscombe + L2 | 788.244767 | 791.963183 | 765.865012 | 1.002195 | 0.969031 |
| Fidelidad Poisson | 788.244767 | 792.152113 | 731.045381 | 1.002450 | 0.896806 |

**Residuo del denoising:**

| Método | Media de r | Correlación corr(r, y) |
|---|---:|---:|
| Poisson raw + L2 | −0.00000004 | 0.2033 |
| Anscombe + L2 | 0.00313895 | 0.1897 |
| Fidelidad Poisson | 0.00190325 | 0.1913 |



