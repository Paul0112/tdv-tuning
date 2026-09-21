# requeriments
import sys
import os
root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path: sys.path.append(root_path)
import numpy as np
import torch
import matplotlib.pyplot as plt
from tdv.denoise_utils import apply_vn, check_color, psnr
from skimage.metrics import structural_similarity as ssim
from astropy.visualization import simple_norm
from skimage.metrics import peak_signal_noise_ratio



def plot_results(z, x_S, y=None, sigma=None, peak=None, prob= None, figsize= (18, 12)):

    """
    z: noisy image
    x_S: denoised image
    y: ground truth
    """

    z, x_S = np.squeeze(z), np.squeeze(x_S)
    y = np.squeeze(y) if y is not None else None

    c_axis = -1 if z.ndim == 3 else None
    cmap = 'gray' if z.ndim == 2 else None

    if y is not None:

        fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=figsize)

        ax[0].imshow(np.clip(z, 0, 1), cmap=cmap)
        if sigma is not None: ax[0].set_title(rf'z ($\sigma$={sigma})')
        if peak is not None: ax[0].set_title(rf'z ($peak$={peak})')
        if prob is not None: ax[0].set_title(rf'z ($salt pepper$={prob})')
        if peak and sigma is not None: ax[0].set_title(rf'z ($\sigma$={sigma}, $peak$={peak})')
        if sigma is None and peak is None and prob is None: ax[0].set_title(rf'z')
        ax[0].set_xlabel(f'PSNR={psnr(z,y):.2f}dB, 'f'SSIM={ssim(z,y,data_range=1,channel_axis=c_axis):.4f}')
        ax[1].imshow(np.clip(x_S, 0, 1), cmap=cmap)
        ax[1].set_title('x_S')
        ax[1].set_xlabel(f'PSNR={psnr(x_S,y):.2f}dB, 'f'SSIM={ssim(x_S,y,data_range=1,channel_axis=c_axis):.4f}')
        ax[2].imshow(np.clip(y, 0, 1), cmap=cmap)
        ax[2].set_title('y')

    else:

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)

        ax[0].imshow(np.clip(z, 0, 1), cmap=cmap)
        ax[0].set_title('Noisy')
        ax[1].imshow(np.clip(x_S, 0, 1), cmap=cmap)
        ax[1].set_title('Denoised')

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

### NOISES

def add_gaussian_noise(img, sigma=25, max_value=255):
    y= img
    z = y + sigma / max_value * np.random.randn(*y.shape).astype(np.float32)
    #z = np.clip(z, 0, 1)
    return z

def add_poisson_noise(img, peak= 100):
    # add poisson noise by setting a max intensity (peak)
    poisson_noise = np.random.poisson(img * peak) 
    noisy_img = poisson_noise/peak 
    noisy_img  = noisy_img 
    #noisy_img = np.clip(noisy_img, 0, 1)

    return noisy_img.astype(np.float32)

def add_gp_noise(img, sigma= 25, peak= 100, max_value=255):
    # add poisson noise and apply gaussian noise after
    y_poisson = add_poisson_noise(img, peak= peak)
    z = add_gaussian_noise(y_poisson, sigma, max_value)
    return z


def compute_metrics(img, ref, data_range):
    img, ref = img.astype(np.float64), ref.astype(np.float64)
    
    # PSNR
    psnr_val = np.inf if np.array_equal(img, ref) else peak_signal_noise_ratio(ref, img, data_range=data_range)
    
    # SSIM 
    win_size = min(7, min(ref.shape))
    if win_size % 2 == 0:
        win_size -= 1
        
    if win_size >= 3:
        ssim_val = ssim(ref, img, data_range=data_range, win_size=win_size)
        return f"PSNR={psnr_val:.2f} dB | SSIM={ssim_val:.4f}"
    return f"PSNR={psnr_val:.2f} dB | SSIM: image too small"


def plot_results_fits(z, x_S, y=None, sigma=None, figsize=(18, 6), stretch="asinh", percent=99.5, data_range=1.0, vmin=None, vmax=None):
    
    images = [np.squeeze(np.asarray(img)) for img in ([z, x_S] if y is None else [z, x_S, y])]
    titles = [f"Noisy (sigma={sigma:g})" if sigma is not None else "Noisy", "TDV"]
    if y is not None:titles.append("Reference")
        
    labels = [""] * len(images)
    if y is not None and data_range is not None:
        ref = images[2]
        labels[0] = compute_metrics(images[0], ref, data_range)
        labels[1] = compute_metrics(images[1], ref, data_range)

    norm = simple_norm(images[0], stretch=stretch, percent=percent, vmin=vmin, vmax=vmax, clip=True)
    fig, axes = plt.subplots(1, len(images), figsize=figsize, sharex=True, sharey=True, layout="constrained")
    
    for ax, img, title, label in zip(axes, images, titles, labels):
        im = ax.imshow(img, origin="lower", cmap="gray", norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel(label)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Intensity (input units)")
    plt.show()
    return fig, axes