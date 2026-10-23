"""Graficos a partir del CSV de peaks_performance.py, sin ejecutar inferencia."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _plot_peaks(csv_path, modalities, metric, figsize, save_path, show):
    if metric not in ("psnr", "ssim"):
        raise ValueError("metric debe ser 'psnr' o 'ssim'.")
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        rows = list(reader)

    # Las columnas SSIM del nuevo CSV no son curvas PSNR adicionales.
    available = [c for c in columns if c not in ("peak", "PSNR entrada (dB)")
                 and c != "SSIM entrada" and not c.startswith("SSIM | ")]
    selected = available if modalities is None else modalities
    if not rows or not selected or any(m not in available for m in selected):
        raise ValueError(f"CSV vacio o modalidad desconocida. Disponibles: {available}")
    noisy_column = "PSNR entrada (dB)" if metric == "psnr" else "SSIM entrada"
    method_columns = {m: m if metric == "psnr" else f"SSIM | {m}" for m in selected}
    required = {"peak", noisy_column, *method_columns.values()}
    missing = required - set(columns)
    if missing:
        raise ValueError(f"Faltan columnas para {metric}: {sorted(missing)}. Regenera el CSV.")
    rows.sort(key=lambda r: float(r["peak"]), reverse=True)
    peaks = np.array([float(r["peak"]) for r in rows])
    values = {m: np.array([float(r[column]) for r in rows])
              for m, column in method_columns.items()}
    noisy_values = np.array([float(r[noisy_column]) for r in rows])

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for name in selected:
        color = colors[available.index(name) % len(colors)]
        ax.plot(peaks, values[name], marker="o", color=color, label=name, alpha=0.65)
    ax.plot(peaks, noisy_values, "k--", marker="o", label="Entrada ruidosa", alpha=0.35)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(peaks, labels=[f"{p:g}" for p in peaks])
    ax.set_xlabel("Peak Poisson (menor peak = mayor ruido)")
    ax.set_ylabel("PSNR medio en CBSD68 (dB)" if metric == "psnr" else "SSIM medio en CBSD68")
    title = selected[0] if len(selected) == 1 else "Comparacion de modalidades"
    ax.set_title(f"CBSD68 | Ruido Poisson | {metric.upper()} | {title}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_peak_modality(csv_path, modality, *, metric="psnr", figsize=(10, 6), save_path=None, show=True):
    """Una modalidad frente a la entrada ruidosa. Devuelve (fig, ax).

    metric='psnr' por defecto; metric='ssim' usa las columnas SSIM.
    Ejemplo: plot_peak_modality('data/results/peaks_performance.csv', 'P-P', metric='ssim').
    Las rutas relativas se resuelven desde el directorio actual del notebook.
    Con show=False, cerrar la figura con plt.close(fig) despues de usarla.
    """
    return _plot_peaks(csv_path, [modality], metric, figsize, save_path, show)


def plot_peaks_comparison(csv_path, *, metric="psnr", figsize=(11, 6), save_path=None, show=True):
    """Todas las modalidades y la entrada ruidosa; metric='psnr' o 'ssim'. Devuelve (fig, ax)."""
    return _plot_peaks(csv_path, None, metric, figsize, save_path, show)
