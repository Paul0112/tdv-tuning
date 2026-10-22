"""Busca sigma_tdv por PSNR medio en CBSD68 con ruido Poisson fijo.
Desde la raiz: python notebooks/utils/sigma_tuning.py --model P-L2
              python notebooks/utils/sigma_tuning.py --model P-P
Protocolo: Training-TDV/evaluate.py y ddr/data.py (seed=1234+i,
imagenes completas /255, Poisson recortado a [0,1], PSNR sin clip de salida).
El sigma ganador es un ajuste sobre CBSD68, no una evaluacion independiente.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdv.model import PoissonDenoiseDataterm, VNet

MODEL_DIRS = {"P-L2": "test_5k_poisson", "P-P": "test_5k_poisson_fidelity"}
SIGMAS = (5, 15, 25, 50, 75, 100)


class TrainingPoissonDataterm(PoissonDenoiseDataterm):
    """Conserva el piso del discriminante usado en Training-TDV/ddr/model.py."""

    def prox(self, x, z, tau):
        b = x - tau
        return 0.5 * (b + torch.sqrt((b * b + 4 * tau * z).clamp_min(1e-8)))


def cbsd68_files(data):
    data = Path(data)
    names = {Path(line.strip()).stem for line in
             (data / "cbsd68.txt").read_text().splitlines() if line.strip()}
    if len(names) != 68:
        raise ValueError("cbsd68.txt debe contener 68 nombres distintos.")
    files = sorted(p for p in (data / "images" / "val").glob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stem in names)
    missing = names - {p.stem for p in files}
    if missing:
        raise FileNotFoundError(
            f"Faltan {len(missing)} imagenes en {data / 'images' / 'val'}. "
            f"Ejemplos: {', '.join(sorted(missing)[:5])}")
    if len(files) != 68:
        raise ValueError("Hay nombres duplicados con distintas extensiones en images/val.")
    return files


def psnr(x, y):
    # Mismo promedio float32 y rango 1 que el evaluador de entrenamiento.
    mse = float(np.mean((x.detach().cpu().numpy() - y.detach().cpu().numpy()) ** 2))
    return float("inf") if mse == 0 else float(20 * np.log10(1 / np.sqrt(mse)))


@torch.no_grad()
def evaluate_sigmas(vn, files, sigmas=SIGMAS, peak=50, sigma_ref=25, seed=1234):
    sigmas = list(dict.fromkeys(float(s) for s in sigmas))
    if not sigmas or not all(np.isfinite(s) and s > 0 for s in [*sigmas, peak, sigma_ref]):
        raise ValueError("sigmas, peak y sigma_ref deben ser positivos y finitos.")
    if not files:
        raise ValueError("No hay imagenes para evaluar.")
    parameter = next(vn.parameters())
    channels = vn.R.in_channels
    mode = {1: "L", 3: "RGB"}[channels]
    vn.eval()
    rows = []
    for i, path in enumerate(files):
        with Image.open(path) as image:
            y = np.asarray(image.convert(mode), dtype=np.float32) / 255.0
        if channels == 1:
            y = y[..., None]
        clean_cpu = torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1)[None]))
        generator = torch.Generator(device="cpu").manual_seed(seed + i)
        noisy_cpu = (torch.poisson(clean_cpu.clamp_min(0) * peak, generator=generator) / peak).clamp(0, 1)
        clean = clean_cpu.to(device=parameter.device, dtype=parameter.dtype)
        noisy = noisy_cpu.to(device=parameter.device, dtype=parameter.dtype)
        row = {"name": Path(path).stem, "psnr_noisy": psnr(noisy, clean), "psnr": {}}
        for sigma in sigmas:
            scale = sigma_ref / sigma
            scaled = noisy * scale
            denoised = vn(scaled, scaled)[-1] / scale
            if not torch.isfinite(denoised).all():
                raise RuntimeError(f"Salida no finita: {path}, sigma_tdv={sigma:g}")
            row["psnr"][sigma] = psnr(denoised, clean)
        rows.append(row)
        print(f"[{i + 1}/{len(files)}] {Path(path).name}", flush=True)
    summary = [{"sigma_tdv": s, "scale": sigma_ref / s,
                "psnr_denoised": float(np.mean([r["psnr"][s] for r in rows]))}
               for s in sigmas]
    best = max(summary, key=lambda r: r["psnr_denoised"])
    return {"peak": peak, "sigma_ref": sigma_ref, "seed": seed, "scaled": True,
            "clip_noisy": True, "num_images": len(rows),
            "psnr_noisy": float(np.mean([r["psnr_noisy"] for r in rows])),
            "best_sigma": best["sigma_tdv"], "summary": summary, "per_image": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_DIRS, default="P-L2")
    parser.add_argument("--checkpoint", type=Path, help="Opcional: otro .pth, por ejemplo last.pth")
    parser.add_argument("--data", type=Path, default=ROOT / "data/bsds500")
    parser.add_argument("--sigmas", nargs="+", type=float, default=SIGMAS)
    parser.add_argument("--peak", type=float, default=50)
    parser.add_argument("--sigma-ref", type=float, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Guardar tabla y resultados por imagen en JSON")
    args = parser.parse_args()
    files = cbsd68_files(args.data)
    checkpoint = args.checkpoint or ROOT / "checkpoints" / MODEL_DIRS[args.model] / "best.pth"
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ck["config"]
    expected = "poisson" if args.model == "P-P" else "denoise"
    if config["D"]["type"] != expected:
        raise ValueError(f"{args.model} requiere fidelidad {expected}, revisa --checkpoint.")
    vn = VNet(config, efficient=False)
    if expected == "poisson":
        vn.D = TrainingPoissonDataterm(config["D"]["config"])
    vn.load_state_dict(ck["model"], strict=True)
    vn.to(args.device).eval()
    print(f"{args.model} | {checkpoint} | iteracion={ck.get('iteration')} | {args.device}")
    print(f"peak={args.peak:g} | scaled=True | sigma_ref={args.sigma_ref:g}")
    report = evaluate_sigmas(vn, files, args.sigmas, args.peak, args.sigma_ref, args.seed)
    report.update(model=args.model, checkpoint=str(checkpoint.resolve()),
                  iteration=ck.get("iteration"), device=args.device)
    print(f"\nPSNR entrada medio: {report['psnr_noisy']:.4f} dB")
    print(f"{'sigma_tdv':>10} {'scale':>10} {'PSNR medio (dB)':>17}")
    for row in report["summary"]:
        print(f"{row['sigma_tdv']:10g} {row['scale']:10.4f} {row['psnr_denoised']:17.4f}")
    print(f"Mejor sigma_tdv entre los probados: {report['best_sigma']:g}")
    print("sigma_tdv=25 equivale a scaled=False cuando sigma_ref=25.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Reporte: {args.output}")


if __name__ == "__main__":
    main()
