import torch
import numpy as np

"""
Implementation of optoth.pad2d (pad2d y pad2d_transpose), mode='symmetric' on Pytorch
to running Total Deep Variation framework on CPU
"""

def reflect(x, minx, maxx):
    """ Reflects an array around two points making a triangular waveform that ramps up
    and down,  allowing for pad lengths greater than the input length """
    rng = maxx - minx
    double_rng = 2*rng
    mod = np.fmod(x - minx, double_rng)
    normed_mod = np.where(mod < 0, mod+double_rng, mod)
    out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx

    return np.array(out, dtype=x.dtype)


def pad2d(x, pad, mode="symmetric"):
     h, w = x.shape[-2:]
     left, right, top, bottom = pad
 
     x_idx = np.arange(-left, w+right)
     y_idx = np.arange(-top, h+bottom)

     x_pad = reflect(x_idx, -0.5, w-0.5)
     y_pad = reflect(y_idx, -0.5, h-0.5)
     xx, yy = np.meshgrid(x_pad, y_pad)
     return x[..., yy, xx]


def pad2d_transpose(x, pad, mode='symmetric'):
    left, right, top, bottom = pad
    N, C, Hp, Wp = x.shape
    H = Hp - top - bottom
    W = Wp - left - right

    y_idx = np.arange(-top, H + bottom)
    x_idx = np.arange(-left, W + right)
    y_pad = reflect(y_idx, -0.5, H - 0.5)   
    x_pad = reflect(x_idx, -0.5, W - 0.5)  

    h_idx = torch.as_tensor(y_pad, dtype=torch.long, device=x.device)
    w_idx = torch.as_tensor(x_pad, dtype=torch.long, device=x.device)

    grad_h = torch.zeros(N, C, H, Wp, dtype=x.dtype, device=x.device)
    grad_h.index_add_(2, h_idx, x)
    grad_out = torch.zeros(N, C, H, W, dtype=x.dtype, device=x.device)
    grad_out.index_add_(3, w_idx, grad_h)

    return grad_out
