#!/usr/bin/env python3
"""Generate verified figure assets for the CSCCE 2026 SERS poster.

The only training performed here is a representative held-out Ag grouped-substrate
Siamese/triplet run used to produce an actual epoch-loss trace for the poster.
CUDA is required.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sers_siamese_substrate_agnostic as sers  # noqa: E402


OUT_DIR = ROOT / "docs" / "cscce_2026_poster"
FIG_DIR = OUT_DIR / "figures"
PAIR_NOTEBOOK = ROOT / "Workspace" / "notebooks" / "Siamese_Network_OneShot.ipynb"


COLORS = {
    "blue": "#174A7C",
    "teal": "#2A9D8F",
    "gold": "#E9A23B",
    "red": "#C8553D",
    "green": "#3A7D44",
    "gray": "#5D6470",
    "light": "#F6F1E7",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
        }
    )


def extract_notebook_losses(path: Path) -> pd.DataFrame:
    text = path.read_text()
    pairs = re.findall(r"Epoch\s+(\d+),\s+Loss:\s+([0-9.]+)", text)
    if not pairs:
        raise RuntimeError(f"No epoch losses found in {path}")
    rows = [{"model": "pair contrastive notebook", "epoch": int(epoch), "loss": float(loss)} for epoch, loss in pairs]
    return pd.DataFrame(rows).drop_duplicates(subset=["epoch"], keep="first")


def current_training_args() -> SimpleNamespace:
    return SimpleNamespace(
        data=ROOT / "Workspace" / "data" / "processed" / "consolidated_SERS.csv",
        crop_min=330.0,
        crop_max=1800.0,
        min_substrates=2,
        feature="derivative_1",
        baseline_lam=1e4,
        baseline_p=0.01,
        baseline_niter=10,
        noise_std=0.01,
        shift_max=2,
        batch_size=32,
        embed_dim=64,
        lr=1e-3,
        margin=0.2,
        loss="triplet",
        epochs=100,
        seed=42,
    )


def train_representative_substrate_agnostic_loss() -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for poster asset training; CPU training is intentionally disabled.")

    args = current_training_args()
    sers.set_seed(args.seed)
    device = torch.device("cuda")

    df, cols = sers.load_dataset(
        args.data,
        args.crop_min,
        args.crop_max,
        args.min_substrates,
        canonicalize_labels=True,
        group_metal_substrates=True,
    )
    X_raw = df[cols].to_numpy(dtype=np.float64)
    X = sers.prepare_features(X_raw, args)
    labels = df["Label"].astype(str).to_numpy()
    substrates = df["Substrate"].astype(str).to_numpy()

    held_out = "Ag"
    train_mask = substrates != held_out
    dataset = sers.RamanTripletDataset(
        X[train_mask],
        labels[train_mask],
        substrates[train_mask],
        args.noise_std,
        args.shift_max,
    )
    loader = sers.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = sers.SiameseNet(X.shape[1], embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    rows: list[dict[str, float | int | str]] = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            loss = sers.triplet_loss(model(anchor), model(positive), model(negative), args.margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * anchor.size(0)
        rows.append(
            {
                "model": "substrate-agnostic triplet, held-out Ag",
                "epoch": epoch,
                "loss": total / len(dataset),
            }
        )
    return pd.DataFrame(rows)


def plot_training_losses(losses: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for model, group in losses.groupby("model"):
        color = COLORS["blue"] if "substrate" in model else COLORS["gold"]
        ax.plot(group["epoch"], group["loss"], linewidth=2.5, label=model, color=color)
    ax.set_title("Training loss traces used in poster narrative")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric-learning loss")
    ax.set_xlim(1, 100)
    ax.grid(axis="y", color="#D8D0C2", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIG_DIR / "training_loss_traces.png")
    plt.close(fig)


def plot_performance_milestones() -> None:
    rows = [
        ("Pair one-shot\nnotebook", 0.9876, 0.98),
        ("Six-label\nSiamese", 0.854, 0.686),
        ("Grouped\nSiamese", 0.975, 0.977),
        ("K-shot\nSiamese K=5", 0.873, 0.850),
        ("Classical\ncentroid", 0.987, 0.987),
        ("Raw spectra\ncontrol", 0.440, 0.399),
    ]
    df = pd.DataFrame(rows, columns=["method", "accuracy", "macro_f1"])
    df.to_csv(FIG_DIR / "performance_milestones.csv", index=False)

    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x - width / 2, df["accuracy"], width, label="Accuracy", color=COLORS["blue"])
    ax.bar(x + width / 2, df["macro_f1"], width, label="Macro F1", color=COLORS["teal"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("From pair classification to substrate-family transfer")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=18, ha="right")
    ax.grid(axis="y", color="#D8D0C2", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    fig.savefig(FIG_DIR / "performance_milestones.png")
    plt.close(fig)


def plot_dataset_matrix() -> None:
    matrix = pd.DataFrame(
        {
            "Ag": [25, 25, 25, 0],
            "Au": [0, 25, 25, 228],
            "PICO": [25, 25, 25, 0],
            "pSERS": [25, 25, 25, 0],
        },
        index=["4NP", "Benzenethiol", "Pyridine", "DMF"],
    )
    matrix.to_csv(FIG_DIR / "dataset_matrix_counts.csv")

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    im = ax.imshow(matrix.values, cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    ax.set_title("Current measured spectra per chemical-substrate family")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix.iloc[row, col])
            color = "white" if value >= 80 else "#1D2433"
            ax.text(col, row, str(value), ha="center", va="center", fontsize=12, weight="bold", color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spectra")
    fig.savefig(FIG_DIR / "dataset_matrix_heatmap.png")
    plt.close(fig)


def plot_kshot_summary() -> None:
    summary = pd.read_csv(ROOT / "Workspace" / "substrate_agnostic" / "grouped_metal_substrates" / "kshot_siamese" / "summary_by_k.csv")
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.errorbar(
        summary["k"],
        summary["mean_true_label_macro_f1"],
        yerr=summary["std_true_label_macro_f1"],
        marker="o",
        markersize=7,
        linewidth=2.5,
        capsize=5,
        label="True-label macro F1",
        color=COLORS["blue"],
    )
    ax.plot(summary["k"], summary["mean_accuracy"], marker="s", linewidth=2.5, label="Accuracy", color=COLORS["teal"])
    ax.set_xscale("log")
    ax.set_xticks(summary["k"])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Support spectra per held-in chemical-substrate cell (K)")
    ax.set_ylabel("Mean score across folds/seeds")
    ax.set_title("Formal K-shot stress test")
    ax.grid(axis="y", color="#D8D0C2", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG_DIR / "kshot_summary.png")
    plt.close(fig)


def write_manifest(losses: pd.DataFrame) -> None:
    manifest = {
        "generated_by": "scripts/generate_cscce_poster_assets.py",
        "requires_cuda": True,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "representative_loss": {
            "fold": "held-out Ag",
            "feature": "derivative_1",
            "loss": "triplet",
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "margin": 0.2,
            "final_loss": float(losses[losses["model"].str.contains("substrate")]["loss"].iloc[-1]),
        },
        "assets": sorted(path.name for path in FIG_DIR.iterdir() if path.is_file()),
    }
    (OUT_DIR / "poster_asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    pair_losses = extract_notebook_losses(PAIR_NOTEBOOK)
    current_losses = train_representative_substrate_agnostic_loss()
    losses = pd.concat([pair_losses, current_losses], ignore_index=True)
    losses.to_csv(FIG_DIR / "training_loss_traces.csv", index=False)
    plot_training_losses(losses)
    plot_performance_milestones()
    plot_dataset_matrix()
    plot_kshot_summary()
    write_manifest(losses)
    print(f"Wrote poster assets to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
