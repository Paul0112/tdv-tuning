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
from skimage import img_as_float
from astropy.io import fits
import imageio.v2 as imageio
from photutils.detection import DAOStarFinder
from astropy.stats import mad_std, sigma_clipped_stats
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry



def plot_results(z, x_S, y=None, sigma=None, peak=None, prob=None, figsize=(18, 12), origin='lower', save_path=None, show=True):
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
        ax[0].imshow(np.clip(z, 0, 1), cmap=cmap, origin=origin)
        ax[0].set_title(title_z)
        ax[0].set_xlabel(f'PSNR={psnr(z, y):.2f}dB, SSIM={ssim(z, y, data_range=1, channel_axis=c_axis):.4f}')

        ax[1].imshow(np.clip(x_S, 0, 1), cmap=cmap, origin=origin)
        ax[1].set_title('x_S')
        ax[1].set_xlabel(f'PSNR={psnr(x_S, y):.2f}dB, SSIM={ssim(x_S, y, data_range=1, channel_axis=c_axis):.4f}')

        ax[2].imshow(np.clip(y, 0, 1), cmap=cmap, origin=origin)
        ax[2].set_title('y')

    else:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=figsize)
        ax[0].imshow(np.clip(z, 0, 1), cmap=cmap, origin=origin)
        ax[0].set_title(title_z)
        ax[1].imshow(np.clip(x_S, 0, 1), cmap=cmap, origin=origin)
        ax[1].set_title('Denoised')

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def fix_hyperparams(vn, param, param_value):
    p = param.lower()
    if p == 's': vn.set_end(param_value)
    elif p == 'lambda': vn.lmbda.data.fill_(param_value)
    else: vn.T.data.fill_(param_value)


def test_hyperparams(param, param_values, vn, img, param_ref, sigma = 25):
    y, _ = check_color(img)
    # add noise
    z = y + sigma / 255 * np.random.randn(*y.shape).astype(np.float32)
    #z_th = torch.from_numpy(np.transpose(z, (2, 0, 1))[None])

    parameter = next(vn.parameters())
    z_th = torch.from_numpy(
        np.ascontiguousarray(z.transpose(2, 0, 1)[None])
    ).to(device=parameter.device, dtype=parameter.dtype)

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
    img = np.asarray(img, dtype=np.float32)
    poisson_noise = np.random.poisson(img * peak).astype(np.float32)
    noisy_img = (poisson_noise/peak).astype(np.float32)

    return noisy_img, poisson_noise

def add_salt_pepper_noise(img, prob = 0.5):
    output = img.copy()    
    black = np.float32(0.0)
    white = np.float32(1.0)
  
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
  
    return output

def add_gp_noise(img, sigma=25, peak=100, max_value=255):
    y_poisson, _ = add_poisson_noise(img, peak=peak)
    return add_gaussian_noise(y_poisson, sigma, max_value)

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



### PLOT, FLUX, METRICS
def plot_stars(image, fwhm=2.5, threshold=3.5, radius=6.0, figsize=(10, 10), cmap='gray'):

    image = np.asarray(image, dtype=float)
    _, median, std = sigma_clipped_stats(image, sigma=3)
    sources = None
    if std > 0: sources = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)(image - median)
    

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, origin='lower', cmap=cmap,
              norm=simple_norm(image, 'sqrt', percent=99))
    if sources is not None and len(sources):
        positions = np.column_stack((sources['x_centroid'], sources['y_centroid']))
        CircularAperture(positions, r=radius).plot(ax=ax, color='red', lw=1)
    ax.set_title(f'Stars found: {0 if sources is None else len(sources)}')
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def photometry_val(x, img_coords_ref=None, fwhm=2.5, threshold=3.5, radius=6.0, r_in=8, r_out=10):
    if len(x.shape) == 3: x=x[:, :, 0]
    if img_coords_ref is None:img_coords_ref = x

    _, median, std = sigma_clipped_stats(img_coords_ref, sigma=3.0)
    data_bkg_subtracted = img_coords_ref - median

    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold * std)
    sources = daofind(data_bkg_subtracted)
    positions = np.transpose((sources['x_centroid'], sources['y_centroid']))
    apertures = CircularAperture(positions, r=radius)       
    rings = CircularAnnulus(positions, r_in=r_in, r_out=r_out) 

    phot_df = aperture_photometry(x, [apertures, rings])
    raw_flux = np.array(phot_df['aperture_sum_0'])
    ring_flux = np.array(phot_df['aperture_sum_1'])
    
    ring_flux_per_pix = ring_flux / rings.area
    bias_flux = ring_flux_per_pix * apertures.area
    true_flux = raw_flux - bias_flux

    return true_flux

def residual_val(y_clean, z_noised, x_denoised):
    y_clean, z_noised, x_denoised = map(np.squeeze, (y_clean, z_noised, x_denoised))
    residual = z_noised - x_denoised
    
    corr_matrix = np.corrcoef(residual.ravel(), y_clean.ravel())
    leakage_corr = float(corr_matrix[0, 1])
    res_mean = float(np.mean(residual))
    
    return leakage_corr, res_mean

def plot_flux(fz, fx, show_quadratic=True):
    fz = np.asarray(fz, dtype=float).ravel()
    fx = np.asarray(fx, dtype=float).ravel()

    finite = np.isfinite(fz) & np.isfinite(fx)
    fz, fx = fz[finite], fx[finite]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    datasets = [(fz, fx, "Flux z (noisy)", "Flux x (denoised)","Flux per source")]

    positive = (fz > 0) & (fx > 0)
    if positive.sum() >= 2:
        mag_z = -2.5 * np.log10(fz[positive])
        mag_x = -2.5 * np.log10(fx[positive])
        datasets.append((mag_z, mag_x, "z magnitude", "x magnitude",f"Magnitude per source (N={positive.sum()})"))

    for ax, (x, y, xlabel, ylabel, title) in zip(axes, datasets):
        ax.scatter(x, y, s=20, alpha=0.6, label="Data")
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", label="Ideal: y = x")

        if x.size >= 2 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * x + intercept
            ss_tot = np.sum((y - y.mean())**2)
            r2 = (1 - np.sum((y - pred)**2) / ss_tot if ss_tot > 0 else np.nan)

            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, slope * xs + intercept, "r-.",label=f"Linear: m={slope:.3f}, R²={r2:.4f}")

            if show_quadratic and x.size >= 3:
                coeffs = np.polyfit(x, y, 2)
                pred2 = np.polyval(coeffs, x)
                r2_quad = (1 - np.sum((y - pred2)**2) / ss_tot if ss_tot > 0 else np.nan)
                ax.plot(xs, np.polyval(coeffs, xs), "b-.", label=f"Quadratic: R²={r2_quad:.4f}, "f"ΔR²={r2_quad-r2:.4f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    if len(datasets) == 2:
        axes[1].invert_xaxis()
        axes[1].invert_yaxis()
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.show()



def plot_residuals(fy, fx, z_image=None, x_image=None):
    fy = np.asarray(fy, dtype=float).ravel()
    fx = np.asarray(fx, dtype=float).ravel()

    slope, intercept = np.polyfit(fy, fx, 1)
    residuals = fx - (slope * fy + intercept)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    ax = axes[0, 0]
    ax.scatter(fy, residuals, s=20, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")

    if fy.size >= 10:
        bins = np.linspace(fy.min(), fy.max(), 11)
        centers, medians = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (fy >= lo) & (fy < hi)
            if mask.sum() > 0:
                centers.append(np.median(fy[mask]))
                medians.append(np.median(residuals[mask]))

        ax.plot(centers, medians, "ko-", ms=4, label="Binned median")
        ax.legend()

    ax.set_xlabel("Flux y (clean)")
    ax.set_ylabel("Residual: fy - predicted fx")
    ax.set_title("Residuals vs flux")
    ax.grid(alpha=0.2)

    # z - x
    r = np.squeeze(z_image - x_image)
    vmax = np.percentile(np.abs(r), 99.5)
    im = axes[0, 1].imshow(r, cmap="gray", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=axes[0, 1], label="Residual: z - x")
    axes[0, 1].set_title("Spatial residual map (z - x)")

    plt.tight_layout()
    plt.show()