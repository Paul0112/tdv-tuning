# TDV: Descenso por gradiente y operador proximal

## Gauss

Para el ruido gaussino, tenemos que $n \sim \mathcal{N}(0, \sigma^2 \mathrm{Id})$, y la imagen ruidosa definida como $z = y + n$, lo que repercute en $z \mid y \sim \mathcal{N}(y, \, \sigma^2 \mathrm{Id})$. Luego, el término de fidelidad se desprende de:

$$p(z_i \mid x_i) \propto \exp\left(-\frac{1}{2\sigma^2} \Vert{}Ax - z\Vert{}_2^2\right),  \qquad  \mathcal{D}(x,z) = -\log p(z \mid x), \qquad \mathcal{D}(x,z) = \frac{1}{2}\Vert{}x - z\Vert{}_2^2$$

Ahora, la derivación del gradiente considerando fidelidad gaussiana, es directa, ya que $\nabla_x \mathcal{D}(x,z) = x - z$.

Luego, Kobler et. al (2020) definen el esquema semi-implícito para discretizar la ecuación de estado tal que:

$$x_{s+1} = x_s - \frac{T}{S} A^\top (A x_{s+1} - z) - \frac{T}{S} \nabla_1 \mathcal{R}(x_s, \theta)$$

$$x_{s+1} = \left(\mathrm{Id} + \frac{T}{S} A^\top A\right)^{-1} \left( x_s - \frac{T}{S} \nabla_1 \mathcal{R}(x_s, \theta) + \frac{T}{S} A^\top z \right)$$

Lo que en código (`tdv/ddr/model.py`) definen como un paso del operador proximal. La definición del operador nos dice que:

$$\mathrm{prox}_{\tau f}(v) := \arg\min_x \left( f(x) + \frac{1}{2\tau} \Vert{}x - v\Vert{}_2^2 \right)$$

Por lo que, considerando $v = x_s - \tau \nabla_1 \mathcal{R}(x_s, \theta)$ y $f(x) = \mathcal{D}(x,z)$ (paso explícito), tenemos el operador proximal definido sobre la fidelidad de la energía:


$$x_{s+1} = \mathrm{prox}_{\tau \mathcal{D}}\left(x_s - \tau \nabla_1 \mathcal{R}(x_s, \theta)\right)$$

Luego:

$$\mathrm{prox}_{\tau \mathcal{D}}(v) = \arg\min_x \left( \frac{1}{2}\Vert{}x - z\Vert{}_2^2 + \frac{1}{2\tau} \Vert{}x - v\Vert{}_2^2 \right)$$

Ahora, sea $$g_1(x) = \left( \frac{1}{2}\Vert{}x - z\Vert{}_2^2 + \frac{1}{2\tau} \Vert{}x - v\Vert{}_2^2 \right), \qquad \nabla_x g_1(x) = 0, \qquad (x - z) + \frac{1}{\tau}(x - v) = 0, \qquad x = \frac{v + \tau z}{1 + \tau}$$

conde se obtiene finalmente $x_{s+1}$, generando el paso implícito.



## Poisson

Para el ruido poisson, tenemos que $z_i \mid y_i \sim \mathrm{Poisson}(y_i)$. Luego, el término de fidelidad se desprende de:

$$p(z_i \mid x_i) \propto x_i^{z_i} e^{-x_i}$$

$$p(z_i \mid x_i) \propto x_i^{z_i} e^{-x_i},  \qquad  \mathcal{D}(x, z) = -\log p(z \mid x), \qquad \mathcal{D}(x,z) = \sum_i \big( x_i - z_i \log x_i \big)$$

Luego, la derivación del gradiente considerando fidelidad Poisson está dado por $\nabla \mathcal{D}(x, z) = 1 - \frac{z}{x}$ .


Por otro lado, el operador proximal está definido por:

$$\mathrm{prox}_{\tau \mathcal{D}}(v) = \arg\min_x \left( \sum_i \big( x_i - z_i \log x_i \big) + \frac{1}{2\tau} \Vert{}x - v\Vert{}_2^2 \right)$$

Ahora, sea

$$
g_2(x) = \left( \sum_i (x_i - z_i \log x_i) + \frac{1}{2\tau}\|x - v\|_2^2 \right),
$$

Considerando cada píxel respectivo:
$$
\nabla_x g_2(x_i) = 0,
$$

$$
1 - \frac{z_i}{x_i} + \frac{1}{\tau}(x_i - v_i) = 0,
$$

$$
x_i^2 + (\tau - v_i)x_i - \tau z_i = 0,
$$

$$
x_i = \frac{v_i - \tau \pm \sqrt{(v_i - \tau)^2 + 4\tau z_i}}{2}.
$$


Como $|v_i - \tau| < |\sqrt{(v_i - \tau)^2 + 4\tau z_i}|$, tomar la raíz positiva dará $x_i > 0$ y la negativa resultará en $x_i < 0$. En `tdv/ddr/model.py`, esto se encuentra modelado en la clase `PoissonDenoiseDataterm`, considerando la raíz positiva.


### Modelación ruido poisson

Sea $x$ la imagen limpia normalizada y $p$ el parámetro $peak$. El ruido Poisson se modela como:

$$
n_i \sim \mathrm{Poisson}(p x_i),
$$

donde $n_i$ representa el conteo observado (modalción de los fotones) en el píxel $i$. Luego, la imagen ruidosa se obtiene normalizando estos conteos:

$$
z_i = \frac{n_i}{p}.
$$

Por tanto, el modelo completo es:

$$
z_i = \frac{1}{p}\mathrm{Poisson}(p x_i)
$$

En consecuencia, la esperanza y la varianza de la observación son:

$$
\mathbb{E}[z_i] = x_i,
\qquad
{Var}(z_i) = \frac{x_i}{p}.
$$


En `notebooks/utils/utils.py`, esto se encuentra modelado en `add_poisson_noise`.

También, como se puede apreciar, se pierde la propiedad Poisosn de media y varianza.

Luego, como el modelo de fidelidad presupone $z_i \mid y_i \sim \mathrm{Poisson}(y_i)$ y como la modelación del ruido es $z_i \mid y_i \sim\frac{1}{p}\mathrm{Poisson}(p y_i)$ se debe implementar algún factor de escalado (En `notebooks/utils/poisson_methods.py` por ejemplo, especificamente en la función `load_poisson_model`, se multiplica el $\lambda$ por $peak$, ya que en el forward $\tau = \frac{\lambda}{s}$, lo que escala internamente la fidelidad) o modelar la fidelidad considerando desde el principio dicha distribución de Poisson escalada.


