import numpy as np
from tdv import model
from tdv.denoise_utils import check_color, denoise_z, load_model, DEVICE
import torch

"""
Biased asncombe transform (fails at peaks << 50)

def anscombe(x):
    return 2.0 * np.sqrt(np.maximum(x, 0.0) + 3.0 / 8.0)

def anscombe_inv(d):
    return (d / 2.0)**2 - 3.0 / 8.0
"""

def anscombe(x):
  return 2.0 * np.sqrt(np.maximum(x, 0.0) + 3.0 / 8.0)

def anscombe_inv(d):
    return (d / 2.0)**2 - 3.0 / 8.0



def anscombe_inv_unbiased(d):
    """Inverse Anscombe (Mäkitalo & Foi)."""
    threshold = 2.0 * np.sqrt(3.0 / 8.0)
    d_safe = np.maximum(d, threshold)

    x = (
        0.25 * (d_safe**2)
        + 0.25 * np.sqrt(1.5) * (d_safe ** (-1))
        - (11.0 / 8.0) * (d_safe ** (-2))
        + (5.0 / 8.0) * np.sqrt(1.5) * (d_safe ** (-3))
        - 0.125
    )
    return np.where(d > threshold, np.maximum(x, 0.0), 0.0)




def denoise_anscombe(z, peak, vn, unbiased= True):
    z, color = check_color(np.asarray(z, dtype=np.float32))
    if vn is None: vn, _ = load_model(color)
    z_counts = z * peak
    z_ansc = anscombe(z_counts)
    
    a_min = 2.0 * np.sqrt(3.0 / 8.0) # minimax norm
    delta = 2.0 * np.sqrt(peak + 3.0 / 8.0) - a_min
    z_input = ((z_ansc - a_min) / delta).astype(np.float32)
    
    x = denoise_z(vn=vn, z=z_input, sigma=255.0 / delta, scaled=True)
    d = x * delta + a_min

    if unbiased: x_counts = anscombe_inv_unbiased(d)
    else: x_counts = anscombe_inv(d)

    return np.clip(x_counts / peak, 0.0, 1.0)



def load_poisson_model(color='gray', peak = 1):
    _, checkpoint_poisson = load_model(color)
    config_poisson = checkpoint_poisson['config']
    config_poisson['D']['type'] = 'poisson'
    config_poisson['D']['config']['use_prox'] = False

    vn_poisson = model.VNet(config_poisson, efficient=False)
    vn_poisson.load_state_dict(checkpoint_poisson['model'], strict=True)

    with torch.no_grad():
        vn_poisson.lmbda.mul_(peak) # scaling (peak)

    return vn_poisson.to(DEVICE).eval()

def denoise_poisson_fidelity(z, vn):
    z, color = check_color(np.asarray(z, dtype=np.float32))
    if vn is None: vn = load_poisson_model(color)
    return denoise_z(vn=vn, z=z, scaled=False)
