import numpy as np
from tdv import model
from tdv.denoise_utils import check_color, denoise_z, load_model, DEVICE
import torch


def anscombe(x):
    return 2.0 * np.sqrt(np.maximum(x, 0.0) + 3.0 / 8.0)

def anscombe_inv(d):
    return (d / 2.0)**2 - 3.0 / 8.0


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


def denoise_anscombe(z, peak, vn):
    z, color = check_color(np.asarray(z, dtype=np.float32))
    if vn is None: vn, _ = load_model(color)
    a_min = 2.0 * np.sqrt(3.0 / 8.0)
    delta = 2.0 * np.sqrt(peak + 3.0 / 8.0) - a_min
    z_input = ((anscombe(z * peak) - a_min) / delta).astype(np.float32)
    x = denoise_z(vn=vn, z=z_input, sigma=255.0 / delta, scaled=True)
    return np.clip(anscombe_inv(x * delta + a_min) / peak, 0.0, 1.0)


def denoise_poisson_fidelity(z, vn):
    z, color = check_color(np.asarray(z, dtype=np.float32))
    if vn is None: vn = load_poisson_model(color)
    return denoise_z(vn=vn, z=z, scaled=False)
