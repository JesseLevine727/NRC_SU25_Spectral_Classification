#!/usr/bin/env python3
"""Replace the schematic input spectra with preprocessed model-input spectra."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sers_siamese_substrate_agnostic as sers  # noqa: E402


DATA = ROOT / "Workspace" / "data" / "processed" / "consolidated_SERS.csv"
FIGURE = ROOT / "docs" / "cscce_2026_poster" / "figures" / "siamese_metric_learning_4np_pico_positive_average.png"

PANELS = [
    ("4np", "Ag", "#0B3D82", (35, 287, 280, 170)),
    ("4np", "PICO", "#0B8E8E", (35, 564, 280, 170)),
    ("benzenethiol", "PICO", "#D71920", (35, 847, 280, 170)),
]
CARD_BORDERS = [
    ((26, 211, 320, 460), "#0B3D82"),
    ((26, 489, 320, 738), "#0B8E8E"),
    ((26, 768, 320, 1017), "#D71920"),
]


def maxabs(y: np.ndarray) -> np.ndarray:
    scale = float(np.nanmax(np.abs(y)))
    if scale == 0:
        return np.zeros_like(y)
    return y / scale


def panel_image(x: np.ndarray, y: np.ndarray, color: str, width_px: int, height_px: int) -> Image.Image:
    dpi = 100
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0.20, 0.28, 0.74, 0.64])
    ax.plot(x, y, color=color, linewidth=1.8)
    ax.axhline(0, color="#888888", linewidth=0.55, alpha=0.8)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([float(x.min()), float(x.max())])
    ax.set_xticklabels([f"{int(round(x.min()))}", f"{int(round(x.max()))}"])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("cm$^{-1}$", fontsize=7, labelpad=1)
    ax.set_ylabel("1st deriv.", fontsize=7, labelpad=1)
    ax.tick_params(axis="both", labelsize=7, length=2.5, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    fig.canvas.draw()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGBA")


def main() -> int:
    df, cols = sers.load_dataset(
        DATA,
        crop_min=330.0,
        crop_max=1800.0,
        min_substrates=1,
        canonicalize_labels=True,
        group_metal_substrates=True,
    )
    x = np.array([float(col) for col in cols], dtype=float)
    X = sers.prepare_features(df[cols].to_numpy(dtype=np.float64), type("Args", (), {"feature": "derivative_1"})())

    base = Image.open(FIGURE).convert("RGBA")
    for label, substrate, color, (left, top, width, height) in PANELS:
        mask = (df["Label"].to_numpy() == label) & (df["Substrate"].to_numpy() == substrate)
        if not np.any(mask):
            raise ValueError(f"No spectra found for {label} on {substrate}")
        y = maxabs(X[mask].mean(axis=0))
        plot = panel_image(x, y, color, width, height)
        base.paste(plot, (left, top), plot)

    draw = ImageDraw.Draw(base)
    for box, color in CARD_BORDERS:
        draw.rounded_rectangle(box, radius=18, outline=color, width=3)

    base.save(FIGURE)
    print(f"Wrote {FIGURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
