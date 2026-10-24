# TDV: Descenso por gradiente y operador proximal

Primeramente, la derivación del gradiente considerando fidelidad gaussiana (esto es, $\mathcal{D}(x,z) = \frac{1}{2}\Vert{}x - z\Vert{}_2^2$), es directa, ya que $\nabla_x \mathcal{D}(x,z) = x - z$.

Luego, Kobler et. al (2020) definen el esquema semi-implícito para discretizar la ecuación de estado tal que:

$$x_{s+1} = x_s - \frac{T}{S} A^\top (A x_{s+1} - z) - \frac{T}{S} \nabla_1 \mathcal{R}(x_s, \theta)$$

$$x_{s+1} = \left(\mathrm{Id} + \frac{T}{S} A^\top A\right)^{-1} \left( x_s - \frac{T}{S} \nabla_1 \mathcal{R}(x_s, \theta) + \frac{T}{S} A^\top z \right)$$

Lo que en código (tdv/ddr/model.py) definen como un paso del operador proximal. La definición del operador nos dice que:

$$\mathrm{prox}_{\tau f}(v) := \arg\min_x \left( f(x) + \frac{1}{2\tau} \Vert{}x - v\Vert{}_2^2 \right)$$

Por lo que, considerando $v = x_s - \tau \nabla_1 \mathcal{R}(x_s, \theta)$ y $f(x) = \mathcal{D}(x,z)$ (paso explícito), tenemos el operador proximal definido sobre la fidelidad de la energía:


$$x_{s+1} = \mathrm{prox}_{\tau \mathcal{D}}\left(x_s - \tau \nabla_1 \mathcal{R}(x_s, \theta)\right)$$

Luego:

$$\mathrm{prox}_{\tau \mathcal{D}}(v) = \arg\min_x \left( \frac{1}{2}\Vert{}x - z\Vert{}_2^2 + \frac{1}{2\tau} \Vert{}x - v\Vert{}_2^2 \right)$$

Ahora, sea $$g_1(x) = \left( \frac{1}{2}\Vert{}x - z\Vert{}_2^2 + \frac{1}{2\tau} \Vert{}x - v\Vert{}_2^2 \right), \qquad \nabla_x g_1(x) = 0, \qquad (x - z) + \frac{1}{\tau}(x - v) = 0, \qquad x = \frac{v + \tau z}{1 + \tau}$$

conde se obtiene finalmente $x_{s+1}$, generando el paso implícito.




