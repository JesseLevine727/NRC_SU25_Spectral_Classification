#!/usr/bin/env python3
"""Deep-dive diagnostics for AgNP/pSERS substrate-agnostic SERS failures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from umap import UMAP

import sers_siamese_substrate_agnostic as sers


INTEREST = [
    ("4np", "AgNP"),
    ("4np", "PICO"),
    ("4np", "pSERS"),
    ("benzenethiol", "Ag"),
    ("benzenethiol", "Au"),
    ("benzenethiol", "PICO"),
    ("benzenethiol", "pSERS"),
    ("pyridine", "AgNP"),
]


def train_triplet_model(X, y, groups, held_out, args, device):
    train_mask = groups != held_out
    dataset = sers.RamanTripletDataset(
        X[train_mask],
        y[train_mask],
        groups[train_mask],
        args.noise_std,
        args.shift_max,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = sers.SiameseNet(X.shape[1], embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.epochs):
        model.train()
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            loss = sers.triplet_loss(model(anchor), model(positive), model(negative), args.margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, train_mask


def substrate_balanced_prototypes(embeddings, labels, substrates):
    prototype_labels = sorted(np.unique(labels))
    prototypes = []
    for label in prototype_labels:
        label_mask = labels == label
        centroids = []
        for substrate in sorted(np.unique(substrates[label_mask])):
            mask = label_mask & (substrates == substrate)
            centroids.append(embeddings[mask].mean(axis=0))
        prototypes.append(np.vstack(centroids).mean(axis=0))
    return prototype_labels, np.vstack(prototypes)


def save_average_spectra(df, cols, out_dir):
    rows = []
    x = np.array([float(c) for c in cols])
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for label, substrate in INTEREST:
        mask = (df["Label"] == label) & (df["Substrate"] == substrate)
        if not mask.any():
            continue
        spectra = df.loc[mask, cols].to_numpy(dtype=float)
        mean = spectra.mean(axis=0)
        std = spectra.std(axis=0)
        rows.append(
            {
                "label": label,
                "substrate": substrate,
                "n": int(mask.sum()),
                "mean_intensity": float(mean.mean()),
                "std_intensity": float(mean.std()),
                "max_wavenumber": float(x[int(np.argmax(mean))]),
                "max_intensity": float(mean.max()),
            }
        )
        axes[0].plot(x, mean, label=f"{label}_{substrate} (n={mask.sum()})")
        norm = mean - np.median(mean)
        denom = np.linalg.norm(norm)
        axes[1].plot(x, norm / denom if denom else norm, label=f"{label}_{substrate}")
        pd.DataFrame({"wavenumber": x, "mean": mean, "std": std}).to_csv(
            out_dir / f"avg_{label}_{substrate}.csv", index=False
        )
    axes[0].set_title("Average cropped raw SERS spectra")
    axes[0].set_ylabel("Intensity")
    axes[1].set_title("Median-centered L2-normalized averages")
    axes[1].set_xlabel("Wavenumber (cm^-1)")
    axes[1].set_ylabel("Normalized intensity")
    for ax in axes:
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "average_spectra_interest.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / "average_spectra_summary.csv", index=False)


def plot_projection(coords, labels, substrates, title, out_path, x_label, y_label):
    plot_df = pd.DataFrame(
        {x_label: coords[:, 0], y_label: coords[:, 1], "label": labels, "substrate": substrates}
    )
    plot_df.to_csv(out_path.with_suffix(".csv"), index=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    for (label, substrate), group in plot_df.groupby(["label", "substrate"]):
        ax.scatter(group[x_label], group[y_label], s=28, alpha=0.75, label=f"{label}_{substrate}")
    ax.set_title(title)
    ax.set_xlabel(x_label.upper())
    ax.set_ylabel(y_label.upper())
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def pca_plot(features, labels, substrates, title, out_path):
    coords = PCA(n_components=2, random_state=42).fit_transform(features)
    plot_projection(coords, labels, substrates, title, out_path, "pc1", "pc2")


def tsne_plot(features, labels, substrates, title, out_path, seed):
    coords = TSNE(
        n_components=2,
        perplexity=20,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)
    plot_projection(coords, labels, substrates, title, out_path, "tsne1", "tsne2")


def umap_plot(features, labels, substrates, title, out_path, seed):
    coords = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    ).fit_transform(features)
    plot_projection(coords, labels, substrates, title, out_path, "umap1", "umap2")


def nonlinear_diagnostics(features, labels, substrates, prefix, title_prefix, out_dir, seed):
    tsne_plot(
        features,
        labels,
        substrates,
        f"t-SNE of {title_prefix}",
        out_dir / f"{prefix}_tsne.png",
        seed,
    )
    umap_plot(
        features,
        labels,
        substrates,
        f"UMAP of {title_prefix}",
        out_dir / f"{prefix}_umap.png",
        seed,
    )


def fold_diagnostics(X, y, groups, held_out, args, out_dir, device):
    model, train_mask = train_triplet_model(X, y, groups, held_out, args, device)
    embeddings = sers.embed(model, X, device, args.batch_size)
    test_mask = groups == held_out
    prototype_labels, prototypes = substrate_balanced_prototypes(
        embeddings[train_mask],
        y[train_mask],
        groups[train_mask],
    )
    distances = np.linalg.norm(embeddings[test_mask, None, :] - prototypes[None, :, :], axis=2)
    pred = np.array([prototype_labels[i] for i in np.argmin(distances, axis=1)])
    test_labels = y[test_mask]
    rows = []
    for i, original_idx in enumerate(np.where(test_mask)[0]):
        rows.append(
            {
                "row_index": int(original_idx),
                "true_label": test_labels[i],
                "pred_label": pred[i],
                **{f"distance_to_{label}": distances[i, j] for j, label in enumerate(prototype_labels)},
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / f"{held_out}_prototype_distances.csv", index=False)
    pd.DataFrame(
        confusion_matrix(test_labels, pred, labels=prototype_labels),
        index=[f"true:{x}" for x in prototype_labels],
        columns=[f"pred:{x}" for x in prototype_labels],
    ).to_csv(out_dir / f"{held_out}_confusion.csv")
    pca_plot(
        embeddings,
        y,
        groups,
        f"Siamese embedding PCA, held out {held_out}",
        out_dir / f"{held_out}_embedding_pca.png",
    )
    nonlinear_diagnostics(
        embeddings,
        y,
        groups,
        f"{held_out}_embedding",
        f"Siamese embedding, held out {held_out}",
        out_dir,
        args.seed,
    )


def raw_file_audit(out_dir):
    paths = sorted(Path("Workspace/data/raw_curated/SERs/4-NP - 632nm/AgNP").glob("*.txt"))
    rows = []
    for path in paths:
        data = np.loadtxt(path)
        rows.append(
            {
                "path": str(path),
                "rows": int(data.shape[0]),
                "cols": int(data.shape[1]) if data.ndim > 1 else 1,
                "wn_min": float(data[:, 0].min()),
                "wn_max": float(data[:, 0].max()),
                "intensity_mean": float(data[:, 1].mean()),
                "intensity_max": float(data[:, 1].max()),
                "max_intensity_wn": float(data[int(np.argmax(data[:, 1])), 0]),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "raw_4np_agnp_file_audit.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Workspace/substrate_agnostic/diagnostics/agnp_failure"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--shift-max", type=int, default=2)
    parser.add_argument("--crop-min", type=float, default=330.0)
    parser.add_argument("--crop-max", type=float, default=1800.0)
    parser.add_argument("--min-substrates", type=int, default=2)
    parser.add_argument("--feature", default="derivative_1")
    parser.add_argument("--baseline-lam", type=float, default=1e4)
    parser.add_argument("--baseline-p", type=float, default=0.01)
    parser.add_argument("--baseline-niter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sers.set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    df, cols = sers.load_dataset(
        Path("Workspace/data/processed/consolidated_SERS.csv"),
        args.crop_min,
        args.crop_max,
        args.min_substrates,
        canonicalize_labels=True,
    )
    df.to_csv(args.out_dir / "canonical_dataset_rows.csv", index=False)
    pd.crosstab(df["Label"], df["Substrate"]).to_csv(args.out_dir / "coverage.csv")
    save_average_spectra(df, cols, args.out_dir)
    X_raw = df[cols].to_numpy(dtype=float)
    X = sers.prepare_features(X_raw, args)
    y = df["Label"].astype(str).to_numpy()
    groups = df["Substrate"].astype(str).to_numpy()
    pca_plot(X, y, groups, "PCA of derivative_1 input features", args.out_dir / "input_derivative1_pca.png")
    nonlinear_diagnostics(
        X,
        y,
        groups,
        "input_derivative1",
        "derivative_1 input features",
        args.out_dir,
        args.seed,
    )
    for held_out in ["AgNP", "pSERS"]:
        fold_diagnostics(X, y, groups, held_out, args, args.out_dir, device)
    raw_file_audit(args.out_dir)
    print(f"Saved AgNP/pSERS diagnostics to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
