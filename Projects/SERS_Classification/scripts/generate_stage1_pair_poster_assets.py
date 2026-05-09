#!/usr/bin/env python3
"""Generate Stage 1 pair-classification assets for the CSCCE poster.

This script does not train a model. It records the Stage 1 confusion matrix
and derives the support-count summary from the Stage 1 data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "cscce_2026_poster" / "figures"
DATA = ROOT / "Workspace" / "data" / "processed" / "consolidated_SERS.csv"
AVG_DATA = ROOT / "Workspace" / "data" / "processed" / "consolidated_SERS_avg.csv"


RAW_CLASSES = [
    "4np__AgNP",
    "4np__PICO",
    "4np__pSERS",
    "benzenethiol__Ag",
    "benzenethiol__Au",
    "bt__PICO",
    "bt__pSERS",
    "n,n-dimethylformamide__AuNP",
    "pyridine__AgNP",
    "pyridine__AuNP",
    "pyridine__PICO",
    "pyridine__pSERS",
]

DISPLAY_LABELS = [
    "4NP on Ag",
    "4NP on PICO",
    "4NP on pSERS",
    "Benzenethiol on Ag",
    "Benzenethiol on Au",
    "Benzenethiol on PICO",
    "Benzenethiol on pSERS",
    "DMF on Au",
    "Pyridine on Ag",
    "Pyridine on Au",
    "Pyridine on PICO",
    "Pyridine on pSERS",
]


def stage1_confusion_matrix() -> pd.DataFrame:
    matrix = np.zeros((len(RAW_CLASSES), len(RAW_CLASSES)), dtype=int)
    diag = [18, 20, 20, 20, 20, 20, 20, 182, 18, 20, 20, 20]
    np.fill_diagonal(matrix, diag)
    matrix[0, 3] = 2
    matrix[7, 9] = 1
    matrix[8, 0] = 2
    return pd.DataFrame(matrix, index=DISPLAY_LABELS, columns=DISPLAY_LABELS)


def plot_confusion_matrix(cm: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 260,
            "savefig.bbox": "tight",
        }
    )
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(cm.values, cmap="Blues", vmin=0, vmax=25)
    ax.set_xlabel("Predicted pair")
    ax.set_ylabel("True pair")
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    tick_labels = [label.replace(" on ", "\n") for label in cm.columns]
    ax.set_xticklabels(tick_labels, rotation=90, ha="center", va="top")
    ax.set_yticklabels(tick_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = int(cm.iat[row, col])
            text_color = "white" if value >= 15 else "#1A1A1A"
            ax.text(col, row, str(value), ha="center", va="center", fontsize=8, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Count\nclipped at 25", fontsize=8)
    fig.savefig(FIG_DIR / "stage1_pair_confusion_matrix.png")
    plt.close(fig)


def write_support_summary() -> None:
    df = pd.read_csv(DATA)
    avg_df = pd.read_csv(AVG_DATA)
    classes = df["Label"].astype(str) + "__" + df["Substrate"].astype(str)
    avg_classes = avg_df["Label"].astype(str) + "__" + avg_df["Substrate"].astype(str)
    class_counts = classes.value_counts().sort_index()
    # Stage 1 used a stratified 20/80 train/query split.
    train_idx, query_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.8,
        stratify=classes,
        random_state=42,
    )
    summary = pd.DataFrame(
        [
            ("Pair classes", len(class_counts)),
            ("Total spectra", len(df)),
            ("Training spectra", len(train_idx)),
            ("Query spectra", len(query_idx)),
            ("Averaged references", len(avg_classes.unique())),
            ("References per pair", 1),
        ],
        columns=["item", "value"],
    )
    summary.to_csv(FIG_DIR / "stage1_pair_support_summary.csv", index=False)
    class_counts.rename_axis("class").reset_index(name="spectra").to_csv(
        FIG_DIR / "stage1_pair_class_counts.csv",
        index=False,
    )


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cm = stage1_confusion_matrix()
    cm.to_csv(FIG_DIR / "stage1_pair_confusion_matrix.csv")
    plot_confusion_matrix(cm)
    write_support_summary()
    print(f"Wrote Stage 1 poster assets to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
