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