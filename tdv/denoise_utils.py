# requeriments
import os
import numpy as np
import torch
import numpy as np
import torch
from tdv import model

# define the evaluation metric
def psnr(x, y): 
    return 20*np.log10(1.0/np.sqrt(np.mean((x-y) ** 2)))

# define the application of the VN
def apply_vn(x_0, z, vn, sigma, sigma_ref=25, scaled= True):
    # tranform to reference noise level
    scale = 1
    if scaled: scale = sigma_ref/sigma
    x = vn(x_0 * scale, z * scale)
    # convert back to original scale
    x = [j/scale for j in x]
    return x

def check_color(raw_image):
    # check color of the input image and slice the img if necesary (+3 channels)
    y = None
    color = None
    if raw_image.ndim == 2 or (raw_image.ndim == 3 and raw_image.shape[-1] == 1):
        color = "gray"
        #y = np.mean(raw_image, 2, keepdims=True)
        y = raw_image[:, :, np.newaxis] if raw_image.ndim == 2 else raw_image
    elif raw_image.ndim == 3 and raw_image.shape[-1] in (3, 4):
        color = "color"
        y = raw_image[:, :, :3] # chop down 3 channels
    else:
        raise ValueError('Expected an HxW, HxWx1, HxWx3 or HxWx4 image')

    return y, color

def load_model(color):
    # load the model state dict
    checkpoint = torch.load(os.path.join('..', 'tdv/checkpoints', f'tdv3-3-25-f32-{color}.pth'), map_location=torch.device('cpu'))

    # get the variational network with the TDV regularizer
    vn = model.VNet(checkpoint['config'], efficient=False)
    vn.load_state_dict(checkpoint['model'])
    #vn.eval()

    return vn, checkpoint

def denoise_z(vn, z, sigma=25, sigma_ref=25, scaled=True):
    z_th = torch.from_numpy(np.transpose(z, (2,0,1))[None])
    
    with torch.no_grad():
        x_th = apply_vn(z_th, z_th, vn, sigma, sigma_ref, scaled)
    x_S = np.transpose(x_th[-1][0].cpu().numpy(), (1,2,0))

    return x_S

def test_img(vn, img, sigma, add_noise=True, noisy_img_path=None, n_bits = 8, scaled= True):
    max_value = 2**n_bits - 1
    sigma_ref = 25
    img = img.astype(np.float32)

    if add_noise:
        y, color = check_color(img)
        z = y + sigma / max_value * np.random.randn(*y.shape).astype(np.float32)
    else:
        z, color = check_color(img)
        y = None # no ground truth img

    # push the images to torch
    #y_th = torch.from_numpy(np.transpose(y, (2,0,1))[None])
    x_S = denoise_z(vn, z, sigma, sigma_ref, scaled)

    return z, x_S, y
