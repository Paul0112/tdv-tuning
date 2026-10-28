"""Autofunciones del regularizador TDV; guarda PNG y JSON en data/eigenfunctions_unclipped/.

P-L2 y P-P seleccionan los nuevos checkpoints de peak fijo 50. No se modifica lambda:
el flujo analiza R.grad, sin utilizar la fidelidad ni generar observaciones Poisson.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tdv.model import VNet

MODEL_DIRS = {"G-5k": "test_5k_gaussian", "P-L2": "pl2_50_unclipped_s0",
              "P-P": "pp50_unclipped_s0"}


@torch.no_grad()
def calcular_autofuncion(model, x_init, iters=500, lr=1e-2):
    """Flujo proyectado sobre la esfera; devuelve (imagen, autovalor final).
    No garantiza convergencia: comprobar el residuo final.
    """
    if iters < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError("iters debe ser >= 1 y lr positivo y finito.")
    x = x_init.clone().detach()
    target = x.norm()
    if not torch.isfinite(x).all() or not torch.isfinite(target) or target <= 0:
        raise ValueError("La entrada debe ser finita y tener norma no nula.")
    for it in range(iters):
        g = model.R.grad(x)
        lam = (g * x).sum() / x.square().sum()
        residual = g - lam * x
        candidate = x - lr * residual
        norm = candidate.norm()
        if not torch.isfinite(candidate).all() or not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Estado no finito o norma nula en iteracion {it + 1}; revisar lr.")
        x = candidate * (target / norm)
        if (it + 1) % 100 == 0:
            print(f"Iter {it+1}/{iters} | lambda previa={lam.item():+.4e} | "
                  f"residuo previo/||x||={(residual.norm()/target).item():.4e}", flush=True)
    g = model.R.grad(x)
    lam = (g * x).sum() / x.square().sum()
    if not torch.isfinite(g).all() or not torch.isfinite(lam):
        raise RuntimeError("Gradiente o autovalor final no finito.")
    return x.detach(), lam.item()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODEL_DIRS, default="P-L2")
    p.add_argument("--checkpoint", type=Path, help="Opcional: otro checkpoint")
    p.add_argument("--image", type=Path, default=ROOT / "data/water-castle.png")
    p.add_argument("--crop", type=int, help="Recorte central cuadrado para pruebas en CPU")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=ROOT / "data/eigenfunctions_unclipped")
    args = p.parse_args()
    if args.iters < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        p.error("--iters debe ser >= 1 y --lr positivo y finito.")
    if args.crop is not None and args.crop < 1:
        p.error("--crop debe ser positivo.")
    checkpoint = args.checkpoint or ROOT / "checkpoints" / MODEL_DIRS[args.model] / "best.pth"
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ck["config"]
    channels = config["R"]["config"]["in_channels"]
    if channels not in (1, 3):
        p.error("Solo se admiten checkpoints 2D de 1 o 3 canales.")
    model = VNet(config, efficient=False)
    model.load_state_dict(ck["model"], strict=True)
    model.to(args.device).eval().requires_grad_(False)
    with Image.open(args.image) as image:
        y = np.asarray(image.convert("L" if channels == 1 else "RGB"), dtype=np.float32) / 255.0
    if channels == 1:
        y = y[..., None]
    if args.crop is not None:
        h, w = y.shape[:2]
        if args.crop > min(h, w):
            p.error("--crop supera las dimensiones de la imagen.")
        top, left = (h-args.crop)//2, (w-args.crop)//2
        y = y[top:top+args.crop, left:left+args.crop]
    x_init = torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1)[None])).to(args.device)
    print(f"{args.model} | {args.device} | {tuple(x_init.shape)} | iteraciones={args.iters}", flush=True)
    started = time.perf_counter()
    x_bar, lam = calcular_autofuncion(model, x_init, args.iters, args.lr)
    with torch.no_grad():
        residual = (model.R.grad(x_bar) - lam * x_bar).norm().item()
    elapsed = time.perf_counter() - started
    eigen = x_bar[0].cpu().numpy().transpose(1, 2, 0)
    crop_label = f"crop{args.crop}" if args.crop else "full"
    directory = args.output / f"{args.model}_eigenfunctions"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{args.image.stem}_{crop_label}.png"
    report = {"model": args.model, "checkpoint": str(checkpoint.resolve()),
              "checkpoint_config": config,
              "iteration": ck.get("iteration"), "image": str(args.image.resolve()),
              "crop": args.crop, "shape_hwc": list(y.shape), "iters": args.iters,
              "lr": args.lr, "device": args.device, "elapsed_seconds": elapsed,
              "eigenvalue": lam, "residual_norm": residual,
              "residual_over_x_norm": residual / x_bar.norm().item(),
              "initial_norm": x_init.norm().item(), "final_norm": x_bar.norm().item()}

    
    # Aproximación: imagen sola, sin títulos ni ejes.
    pixels = np.rint(np.clip(eigen, 0, 1) * 255).astype(np.uint8)

    if channels == 1:
        pixels = pixels[..., 0]

    Image.fromarray(pixels).save(output)

    # Parámetros y resultados.
    config_path = output.with_suffix(".json")
    config_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Imagen: {output}")
    print(f"Configuración: {config_path}")

if __name__ == "__main__":
    main()
