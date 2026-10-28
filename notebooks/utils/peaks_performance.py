"""PSNR y SSIM medios en CBSD68 frente al peak Poisson para los cinco metodos.

Desde tdv-tuning: python notebooks/utils/peaks_performance.py
Guarda solo datos: CSV con promedios y JSON con resultados por imagen.
Protocolo: best.pth, seed=1234+i, observacion Poisson sin recorte.
Los CSV historicos con entrada recortada no se regeneran con este protocolo.
Los pesos del regularizador permanecen fijos; Anscombe adapta G-5k.
P-L2 y P-P usan los nuevos checkpoints de peak fijo 50. Solo P-P adapta su fidelidad
con lambda_checkpoint * (peak / 50), sin acumular factores entre niveles.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from notebooks.utils.poisson_methods import denoise_anscombe
from notebooks.utils.sigma_tuning import cbsd68_files, psnr
from tdv.model import VNet

PEAKS = (1000, 500, 100, 50, 20, 10, 5)
POISSON_PEAK_TRAIN = 50.0
MODEL_DIRS = {
    "G-5k": "test_5k_gaussian",
    "P-L2": "pl2_50_unclipped_s0",
    "P-P": "pp50_unclipped_s0",
}
METHODS = ("G-5k", "G-5k + Anscombe biased", "G-5k + Anscombe unbiased", "P-L2", "P-P")


def tensor_ssim(x, y):
    x = x[0].detach().float().cpu().permute(1, 2, 0).numpy()
    y = y[0].detach().float().cpu().permute(1, 2, 0).numpy()
    return float(structural_similarity(x, y, data_range=1.0, channel_axis=-1))


def load_models(checkpoint_root, device):
    models, metadata = {}, {}
    for name, directory in MODEL_DIRS.items():
        path = checkpoint_root / directory / "best.pth"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        config = ck["config"]
        expected = "poisson" if name == "P-P" else "denoise"
        if config["D"]["type"] != expected or config["R"]["config"]["in_channels"] != 3:
            raise ValueError(f"Checkpoint incompatible para {name}: {path}")
        vn = VNet(config, efficient=False)
        vn.load_state_dict(ck["model"], strict=True)
        models[name] = vn.to(device).eval()
        metadata[name] = {"path": str(path.resolve()), "iteration": ck.get("iteration"),
                          "fidelity": expected}
    return models, metadata


@torch.no_grad()
def evaluate_peak(models, files, peak, seed=1234):
    """Una realizacion por imagen y peak, compartida entre los cinco metodos."""
    if not files or not np.isfinite(peak) or peak <= 0:
        raise ValueError("Se requieren imagenes y un peak positivo y finito.")
    parameter = next(models["G-5k"].parameters())
    rows = []
    for i, path in enumerate(files):
        with Image.open(path) as image:
            y = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        clean_cpu = torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1)[None]))
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        noisy_cpu = torch.poisson(clean_cpu * peak, generator=gen) / peak
        clean = clean_cpu.to(device=parameter.device, dtype=parameter.dtype)
        noisy = noisy_cpu.to(device=parameter.device, dtype=parameter.dtype)
        row = {"name": Path(path).stem, "psnr_noisy": psnr(noisy, clean),
               "ssim_noisy": tensor_ssim(noisy, clean), "psnr": {}, "ssim": {}}

        for name in METHODS:
            if name in MODEL_DIRS:
                restored = models[name](noisy, noisy)[-1]
            else:
                z = noisy_cpu[0].permute(1, 2, 0).numpy()
                x = denoise_anscombe(z, peak=peak, vn=models["G-5k"],
                                     unbiased=(name == "G-5k + Anscombe unbiased"))
                restored = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
                restored = restored.to(device=parameter.device, dtype=parameter.dtype)
            if not torch.isfinite(restored).all():
                raise RuntimeError(f"Salida no finita: {name}, peak={peak:g}, imagen={path}")
            row["psnr"][name] = psnr(restored, clean)
            row["ssim"][name] = tensor_ssim(restored, clean)
        rows.append(row)
        print(f"peak={peak:g} [{i + 1}/{len(files)}] {Path(path).name}", flush=True)

    return {"peak": float(peak), "psnr_noisy": float(np.mean([r["psnr_noisy"] for r in rows])),
            "psnr": {name: float(np.mean([r["psnr"][name] for r in rows])) for name in METHODS},
            "ssim_noisy": float(np.mean([r["ssim_noisy"] for r in rows])),
            "ssim": {name: float(np.mean([r["ssim"][name] for r in rows])) for name in METHODS},
            "per_image": rows}


def save_outputs(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["peak", "PSNR entrada (dB)", *METHODS,
                         "SSIM entrada", *[f"SSIM | {m}" for m in METHODS]])
        for row in report["results"]:
            writer.writerow([row["peak"], row["psnr_noisy"], *[row["psnr"][m] for m in METHODS],
                             row["ssim_noisy"], *[row["ssim"][m] for m in METHODS]])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/bsds500")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--peaks", type=float, nargs="+", default=PEAKS)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "data/results/peaks_performance_pp_scaled_unclipped")
    args = parser.parse_args()
    if not all(np.isfinite(p) and p > 0 for p in args.peaks):
        parser.error("Todos los peaks deben ser positivos y finitos.")
    peaks = sorted(set(args.peaks), reverse=True)
    files = cbsd68_files(args.data)
    models, checkpoints = load_models(args.checkpoints, args.device)
    lambda_pp_original = models["P-P"].lmbda.detach().clone()
    report = {"dataset": "CBSD68", "data": str(args.data.resolve()), "num_images": len(files),
              "seed": args.seed, "device": args.device, "clip_noisy": False,
              "poisson_prox": "stable_exact_v1",
              "clip_output": {m: "Anscombe" in m for m in METHODS},
              "sigma_ref_anscombe": 25, "adapt_poisson_weights": True,
              "poisson_peak_train": POISSON_PEAK_TRAIN,
              "poisson_peak_range_train": None,
              "poisson_lambda_scaling": "peak / 50.0",
              "checkpoints": checkpoints, "peaks": peaks, "results": []}
    for peak in peaks:
        with torch.no_grad():
            models["P-P"].lmbda.copy_(lambda_pp_original * (peak / POISSON_PEAK_TRAIN))
        try:
            row = evaluate_peak(models, files, peak, args.seed)
        finally:
            with torch.no_grad():
                models["P-P"].lmbda.copy_(lambda_pp_original)
        report["results"].append(row)
        # Guardar cada peak para conservar resultados si se interrumpe la corrida.
        save_outputs(report, args.output)
        print(f"\npeak={peak:g} | entrada={row['psnr_noisy']:.2f} dB | SSIM={row['ssim_noisy']:.4f}", flush=True)
        for name in METHODS:
            print(f"  {name}: {row['psnr'][name]:.2f} dB | SSIM={row['ssim'][name]:.4f}", flush=True)
    print(f"\nResultados: {args.output}.[json,csv]")


if __name__ == "__main__":
    main()
