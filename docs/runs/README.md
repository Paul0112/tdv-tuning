# Realizaciones de noise_types

Carpetas: `{imagen}_{ruido}_{scaled}_{sigma_tdv}`. Cada PNG contiene z (ruidosa), x (TDV) e y (referencia). El nombre indica el parametro del ruido; para ruido mixto incluye ambos parametros.

Los 18 barridos de Poisson (water-castle y stars_bg) y sal y pimienta (water-castle), con sigma TDV 5, 15, 25, 35, 50 y 100, se exportaron directamente desde los results del kernel activo, sin regenerar ruido ni ejecutar TDV. Sigma 25 usa scaled=False; los otros valores usan scaled=True. Son 96 figuras.

Las figuras adicionales de camera-man y ruido mixto son exportaciones de los PNG previamente guardados en el notebook. Camera-man contiene solo prob=0.001, la unica realizacion que tenia figura guardada; su results no estaba presente en el kernel.
