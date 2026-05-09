from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFUSION_DIR = (
    ROOT
    / "Workspace"
    / "substrate_agnostic"
    / "grouped_metal_substrates"
    / "current"
    / "best_siamese_triplet"
    / "confusions"
)
COMPACT_OUTPUT = (
    ROOT
    / "docs"
    / "cscce_2026_poster"
    / "figures"
    / "heldout_confusion_matrices_compact.png"
)
GRID_OUTPUT = (
    ROOT
    / "docs"
    / "cscce_2026_poster"
    / "figures"
    / "heldout_confusion_matrices_2x2.png"
)
FOLDS = ["Ag", "Au", "PICO", "pSERS"]
LABELS = ["4np", "benzenethiol", "pyridine"]


def load_matrix(path: Path) -> np.ndarray:
    rows: list[list[int]] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            rows.append([int(value) for value in row[1:]])
    return np.asarray(rows, dtype=int)


def draw_matrix(ax: plt.Axes, fold: str, index: int, *, layout: str) -> None:
    matrix = load_matrix(CONFUSION_DIR / f"{fold}.csv")
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=25)

    if layout == "compact":
        title_size = 8
        tick_size = 5.8
        label_size = 6.2
        value_size = 6.2
        title_pad = 2
        rotation = 34
        show_ylabel = index == 0
    else:
        title_size = 9.5
        tick_size = 7.2
        label_size = 7.8
        value_size = 8.2
        title_pad = 3
        rotation = 30
        show_ylabel = index % 2 == 0

    ax.set_title(f"Held-out {fold}", fontsize=title_size, pad=title_pad)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=rotation, ha="right", fontsize=tick_size)
    if show_ylabel:
        ax.set_yticklabels(LABELS, fontsize=tick_size)
        ax.set_ylabel("True", fontsize=label_size)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Predicted", fontsize=label_size, labelpad=1)
    ax.tick_params(length=0, pad=1)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value >= 13 else "black"
            ax.text(col, row, str(value), ha="center", va="center", fontsize=value_size, color=color)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def main() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(8.8, 2.15), dpi=300)

    for index, (ax, fold) in enumerate(zip(axes.flat, FOLDS)):
        draw_matrix(ax, fold, index, layout="compact")

    fig.subplots_adjust(left=0.055, right=0.995, top=0.84, bottom=0.33, wspace=0.12)
    COMPACT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPACT_OUTPUT)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(5.25, 4.25), dpi=300)
    for index, (ax, fold) in enumerate(zip(axes.flat, FOLDS)):
        draw_matrix(ax, fold, index, layout="grid")

    fig.subplots_adjust(left=0.12, right=0.985, top=0.94, bottom=0.16, wspace=0.32, hspace=0.55)
    fig.savefig(GRID_OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
