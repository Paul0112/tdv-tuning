"""Compara S=10 y S=20 sobre evento_paso.pth, con forward y backward.

Desde tdv-tuning: python utils/comparar_pasos.py
Requiere el repositorio hermano Training-TDV con la instrumentación DEBUG activa.
Usa el modelo de entrenamiento y sus hooks para medir los tensores internos reales.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def medir(model, muestra, steps, acumulacion, device):
    """La loss se suma por imagen, igual que en train.py."""
    model.S = steps
    model.zero_grad(set_to_none=True)
    etapas = []

    def observar(nombre, tensor, backward=True):
        if nombre == "x_in":
            etapas.append({"etapa": len(etapas) + 1})
        if nombre not in ("v", "x_out"):
            return
        fila = etapas[-1]

        def guardar(grad):
            grad = grad.detach().float()
            fila["grad_" + nombre] = {
                "norma": grad.norm().item(),
                "max_abs": grad.abs().amax().item(),
            }

        if tensor.requires_grad:
            tensor.register_hook(guardar)

    clean = muestra["clean"].to(device)
    noisy = muestra["noisy"].to(device)
    model.D.observe = observar
    try:
        ultimo = model(noisy, noisy)[-1]
        loss = (ultimo.float() - clean.float()).square().flatten(1).sum(1).mean()
        (loss / acumulacion).backward()
    finally:
        del model.D.observe
    if len(etapas) != steps or any("grad_v" not in e or "grad_x_out" not in e for e in etapas):
        raise RuntimeError("Faltan hooks DEBUG en Training-TDV/ddr/model.py.")
    gx = [e["grad_x_out"]["norma"] for e in etapas]
    if not math.isfinite(loss.item()) or not all(
        math.isfinite(value) for e in etapas
        for name in ("grad_v", "grad_x_out") for value in e[name].values()
    ):
        raise RuntimeError(f"Salida o gradientes no finitos con S={steps}.")
    return dict(S=steps, T=model.T.item(), lmbda=model.lmbda.item(),
                h=model.T.item()/steps, tau_D=model.lmbda.item()/steps,
                loss=loss.item(), amplificacion=gx[0]/gx[-1] if gx[-1] > 0 else None,
                etapas=etapas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capturas", nargs="*", type=Path, default=[
        ROOT / "checkpoints/paso_pl2_peak50_s0_1k/evento_paso.pth",
        ROOT / "checkpoints/paso_pp_peak50_s0_1k/evento_paso.pth",
    ])
    parser.add_argument("--training-repo", type=Path, default=ROOT.parent / "Training-TDV")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sin-compilar", action="store_true",
                        help="usa R.grad eager; por defecto se compila en CUDA como en entrenamiento")
    parser.add_argument("--output", type=Path, default=ROOT / "data/results/sensibilidad_s.json")
    args = parser.parse_args()
    if not (args.training_repo / "ddr/model.py").is_file():
        parser.error("--training-repo debe apuntar al repositorio Training-TDV.")
    if not all(p.is_file() for p in args.capturas):
        parser.error("Falta alguno de los archivos evento_paso.pth.")
    sys.path.insert(0, str(args.training_repo.resolve()))
    from ddr.model import VNet2D

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    compilado = args.device == "cuda" and not args.sin_compilar
    reporte = dict(device=args.device, torch=torch.__version__, compilado=compilado,
                   training_repo=str(args.training_repo.resolve()),
                   medicion="norma dL/dx1 dividida por norma dL/dxS; dirección de la loss",
                   control="datos previamente aceptados, evaluados con pesos del evento",
                   resultados=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in args.capturas:
        captura = torch.load(path, map_location="cpu", weights_only=True)
        if captura["config"]["S"] != 10:
            raise ValueError(f"{path}: esta comparación requiere una captura original con S=10.")
        model = VNet2D(captura["config"], efficient=False).to(args.device)
        model.load_state_dict(captura["model"], strict=True)
        model.train()
        if compilado:
            model.R.grad = torch.compile(model.R.grad, dynamic=False)
        for tipo in ("evento", "control"):
            muestra = captura[tipo]
            if muestra is None:
                continue
            for steps in (10, 20):
                fila = medir(model, muestra, steps, captura["acumulacion"], args.device)
                fila.update(captura=str(path.resolve()), tipo=tipo,
                            it=muestra["it"], micro=muestra["micro"])
                if tipo == "evento" and steps == 10:
                    gx = [e["grad_x_out"]["norma"] for e in fila["etapas"]]
                    fila["reproduce_original"] = (
                        math.isclose(fila["loss"], muestra["loss"], rel_tol=1e-5, abs_tol=1e-7)
                        and len(gx) == len(muestra["grad_x"])
                        and all(math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-7)
                                for a, b in zip(gx, muestra["grad_x"]))
                    )
                    print(f"{path.parent.name}: reproduce_original={fila['reproduce_original']}", flush=True)
                reporte["resultados"].append(fila)
                args.output.write_text(json.dumps(reporte, indent=2, allow_nan=False), encoding="utf-8")
                print(f"{path.parent.name} {tipo} S={steps}: loss={fila['loss']:.6g} "
                      f"amplificación={fila['amplificacion']}", flush=True)
        del model
    print(f"Guardado en {args.output.resolve()}")


if __name__ == "__main__":
    main()
