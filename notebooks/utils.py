# requeriments
import sys
import os
root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path: sys.path.append(root_path)
import numpy as np
import torch
import matplotlib.pyplot as plt
from tdv.denoise_utils import apply_vn, check_color, psnr, load_model, denoise_z
from skimage.metrics import structural_similarity as ssim
from astropy.visualization import simple_norm
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float
from astropy.io import fits
import imageio.v2 as imageio



def plot_results(z, x_S, y=None, sigma=None, peak=None, prob=None, figsize=(18, 12)):
    """
    z: noisy image (normalizada 0 a 1)
    x_S: denoised image (normalizada 0 a 1)
    y: ground truth (normalizada 0 a 1)
    """
    z, x_S = np.squeeze(z), np.squeeze(x_S)
    y = np.squeeze(y) if y is not None else None

    c_axis = -1 if z.ndim == 3 else None
    cmap = 'gray' if z.ndim == 2 else None

    if (sigma is not None) and (peak is not None):
        title_z = rf'z ($\sigma$={sigma}, $peak$={peak})'
    elif sigma is not None:
        title_z = rf'z ($\sigma$={sigma})'
    elif peak is not None:
        title_z = rf'z ($peak$={peak})'
    elif prob is not None:
        title_z = rf'z ($salt\ pepper$={prob})'
    else:
        title_z = 'z'

    if y is not None:
        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)
        ax[0].imshow(np.clip(z, 0, 1), cmap=cmap, origin='lower')
        ax[0].set_title(title_z)
        ax[0].set_xlabel(f'PSNR={psnr(z, y):.2f}dB, SSIM={ssim(z, y, data_range=1, channel_axis=c_axis):.4f}')

        ax[1].imshow(np.clip(x_S, 0, 1), cmap=cmap, origin='lower')
        ax[1].set_title('x_S')
        ax[1].set_xlabel(f'PSNR={psnr(x_S, y):.2f}dB, SSIM={ssim(x_S, y, data_range=1, channel_axis=c_axis):.4f}')

        ax[2].imshow(np.clip(y, 0, 1), cmap=cmap, origin='lower')
        ax[2].set_title('y')

    else:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
        ax[0].imshow(np.clip(z, 0, 1), cmap=cmap, origin='lower')
        ax[0].set_title(title_z)
        ax[1].imshow(np.clip(x_S, 0, 1), cmap=cmap, origin='lower')
        ax[1].set_title('Denoised')

    plt.tight_layout()
    plt.show()


def fix_hyperparams(vn, param, param_value):
    p = param.lower()
    if p == 's': vn.set_end(param_value)
    elif p == 'lambda': vn.lmbda.data.fill_(param_value)
    else: vn.T.data.fill_(param_value)


def test_hyperparams(param, param_values, vn, img, param_ref, sigma = 25):
    y, _ = check_color(img)
    # add noise
    z = y + sigma / 255 * np.random.randn(*y.shape).astype(np.float32)
    z_th = torch.from_numpy(np.transpose(z, (2, 0, 1))[None])

    x_all_by_param = {} # each intermediate step (img)
    x_psnr_param = []
    z_psnr = psnr(z, y)

    try:
        with torch.no_grad():
            for p in param_values:
                # select the correct paramt o set
                fix_hyperparams(vn=vn, param=param, param_value=p)

                x_all = apply_vn(z_th, z_th, vn, sigma, sigma_ref=25)
                # CPU arrays with shape (S + 1, H, W, C)
                x_all_by_param[p] = np.stack([
                    np.transpose(x[0].cpu().numpy(), (1, 2, 0))
                    for x in x_all
                ])
                x_final = x_all_by_param[p][-1] # denoising result of steps
                current_psnr = psnr(x_final, y)
                x_psnr_param.append(current_psnr)
                print(f'{param}={p} | PSNR={x_psnr_param[-1]:.2f} dB')
    finally:
        # return to optimum value
        fix_hyperparams(vn=vn, param=param, param_value=param_ref)

    return x_psnr_param, x_all_by_param, z_psnr, z, y


def plot_hyperparams(param, param_values, param_ref, x_psnr_param, z_psnr=None):

    plt.figure(figsize=(12, 5))
    plt.plot(param_values, x_psnr_param, 'o-', label=r'$x_S$ PSNR')
    if z_psnr is not None: plt.axhline(z_psnr, linestyle=':', color='gray', label='z PSNR')
    plt.axvline(param_ref, linestyle='--', color='red', label=f'{param}={param_ref} (checkpoint)')
    plt.title(rf'PSNR v/s {param} ($\sigma=25$)')
    plt.xlabel(f'{param}')
    plt.ylabel('PSNR (dB)')
    plt.xticks(param_values)
    plt.grid(True)
    plt.legend()
    plt.show()


def minimax_norm(img_raw, img_scale_reference= None):
    # apply minimax norm
    if img_scale_reference is None: img_scale_reference = img_raw
    max_v = img_scale_reference.max()
    min_v = img_scale_reference.min()
    scale = max_v - min_v

    norm_img = (img_raw - min_v) / scale
    return norm_img

def minimax_unnorm(norm_img, img_scale_reference= None):
    if img_scale_reference is None:
        raise ValueError('Se requiere img_scale_reference para invertir min-max.')
    max_v = img_scale_reference.max()
    min_v = img_scale_reference.min()
    scale = max_v - min_v

    return (norm_img * scale) + min_v

### NOISES

def add_gaussian_noise(img, sigma=25, max_value=255):
    y= img
    z = y + sigma / max_value * np.random.randn(*y.shape).astype(np.float32)
    #z = np.clip(z, 0, 1)
    return z

def add_poisson_noise(img, peak= 100):
    # add poisson noise by setting a max intensity (peak)
    img = np.clip(img, 0.0, 1.0)
    poisson_noise = np.random.poisson(img * peak).astype(np.float32)
    noisy_img = poisson_noise/peak 
    noisy_img  = noisy_img 
    #noisy_img = np.clip(noisy_img, 0, 1)

    return noisy_img, poisson_noise

def add_gp_noise(img, sigma= 25, peak= 100, max_value=255):
    # add poisson noise and apply gaussian noise after
    y_poisson = add_poisson_noise(img, peak= peak)
    z = add_gaussian_noise(y_poisson, sigma, max_value)
    return z


def test_astro(file_raw, type='fits', norm=True, scaled= False, add_noise= None, noise_param=None):
    img_path = os.path.join('..', 'data', file_raw)
    
    if type=='fits': img_raw = img_as_float(fits.getdata(img_path, ext=0)).astype(np.float32)
    else: img_raw = img_as_float(imageio.imread(img_path)).astype(np.float32) #already norm (/255)

    if norm: img_raw = minimax_norm(img_raw)
    y = img_raw
    z_noised = y
    if add_noise is not None:
        if add_noise == 'Gaussian': 
            if noise_param is None: noise_param = 25
            z_noised = add_gaussian_noise(y, sigma= noise_param)
        if add_noise == 'Poisson': 
            if noise_param is None: noise_param = 100
            z_noised, _ = add_poisson_noise(y, peak= noise_param)

    z, color = check_color(z_noised)
    vn, _ = load_model(color) 
    x_denoised = denoise_z(vn= vn, z=z, scaled= scaled)

    return z, x_denoised, y