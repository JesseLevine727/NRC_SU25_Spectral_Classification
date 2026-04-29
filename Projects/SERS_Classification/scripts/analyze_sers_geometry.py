#!/usr/bin/env python3
"""Quantify substrate-agnostic geometry before and after Siamese embedding."""

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
from sklearn.metrics import silhouette_samples, silhouette_score
from torch.utils.data import DataLoader
from umap import UMAP

import sers_siamese_substrate_agnostic as sers


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


def substrate_balanced_prototypes(features, labels, substrates):
    prototype_labels = sorted(np.unique(labels))
    prototypes = []
    for label in prototype_labels:
        label_mask = labels == label
        centroids = []
        for substrate in sorted(np.unique(substrates[label_mask])):
            mask = label_mask & (substrates == substrate)
            centroids.append(features[mask].mean(axis=0))
        prototypes.append(np.vstack(centroids).mean(axis=0))
    return prototype_labels, np.vstack(prototypes)


def separation_scores(features, labels, substrates, space, held_out):
    rows = []
    label_score = np.nan
    substrate_score = np.nan
    if len(np.unique(labels)) > 1:
        label_score = float(silhouette_score(features, labels, metric="euclidean"))
    if len(np.unique(substrates)) > 1:
        substrate_score = float(silhouette_score(features, substrates, metric="euclidean"))
    rows.append(
        {
            "space": space,
            "held_out": held_out,
            "silhouette_label": label_score,
            "silhouette_substrate": substrate_score,
            "label_minus_substrate_silhouette": label_score - substrate_score,
        }
    )
    return rows


def sample_silhouettes(features, labels, substrates, space, held_out):
    label_values = silhouette_samples(features, labels, metric="euclidean")
    substrate_values = silhouette_samples(features, substrates, metric="euclidean")
    rows = []
    for idx, (label_value, substrate_value) in enumerate(zip(label_values, substrate_values)):
        rows.append(
            {
                "space": space,
                "held_out": held_out,
                "row_index": idx,
                "label": labels[idx],
                "substrate": substrates[idx],
                "silhouette_label": float(label_value),
                "silhouette_substrate": float(substrate_value),
                "label_minus_substrate_silhouette": float(label_value - substrate_value),
            }
        )
    return rows


def prototype_geometry(features, labels, substrates, held_out, space):
    train_mask = substrates != held_out
    test_mask = substrates == held_out
    prototype_labels, prototypes = substrate_balanced_prototypes(
        features[train_mask],
        labels[train_mask],
        substrates[train_mask],
    )
    test_features = features[test_mask]
    test_labels = labels[test_mask]
    distances = np.linalg.norm(test_features[:, None, :] - prototypes[None, :, :], axis=2)
    pred_indices = np.argmin(distances, axis=1)
    pred_labels = np.array([prototype_labels[idx] for idx in pred_indices])

    sample_rows = []
    for local_idx, row_idx in enumerate(np.where(test_mask)[0]):
        row = {
            "space": space,
            "held_out": held_out,
            "row_index": int(row_idx),
            "true_label": test_labels[local_idx],
            "pred_label": pred_labels[local_idx],
            "correct": bool(pred_labels[local_idx] == test_labels[local_idx]),
        }
        for prototype_idx, prototype_label in enumerate(prototype_labels):
            row[f"distance_to_{prototype_label}"] = float(distances[local_idx, prototype_idx])
        sample_rows.append(row)

    class_rows = []
    prototype_index = {label: idx for idx, label in enumerate(prototype_labels)}
    for label in sorted(np.unique(test_labels)):
        mask = test_labels == label
        if label not in prototype_index:
            continue
        own_idx = prototype_index[label]
        own_dist = distances[mask, own_idx]
        wrong_indices = [idx for idx, prototype_label in enumerate(prototype_labels) if prototype_label != label]
        wrong_dists = distances[mask][:, wrong_indices]
        nearest_wrong_indices = np.argmin(wrong_dists, axis=1)
        nearest_wrong_dist = wrong_dists[np.arange(wrong_dists.shape[0]), nearest_wrong_indices]
        nearest_wrong_labels = np.array([prototype_labels[wrong_indices[idx]] for idx in nearest_wrong_indices])
        class_rows.append(
            {
                "space": space,
                "held_out": held_out,
                "true_label": label,
                "n": int(mask.sum()),
                "accuracy": float((pred_labels[mask] == label).mean()),
                "mean_own_distance": float(own_dist.mean()),
                "mean_nearest_wrong_distance": float(nearest_wrong_dist.mean()),
                "mean_margin_wrong_minus_own": float((nearest_wrong_dist - own_dist).mean()),
                "dominant_wrong_neighbor": pd.Series(nearest_wrong_labels).mode().iat[0],
                "dominant_prediction": pd.Series(pred_labels[mask]).mode().iat[0],
            }
        )
    return sample_rows, class_rows


def projection_summary(df, projection, space, held_out):
    coord_cols = [col for col in df.columns if col not in {"label", "substrate"}]
    rows = []
    centroids = df.groupby(["label", "substrate"])[coord_cols].mean()
    label_centroids = df.groupby("label")[coord_cols].mean()
    for (label, substrate), centroid in centroids.iterrows():
        own = np.linalg.norm(centroid.to_numpy() - label_centroids.loc[label].to_numpy())
        wrong = {
            other_label: np.linalg.norm(centroid.to_numpy() - other_centroid.to_numpy())
            for other_label, other_centroid in label_centroids.iterrows()
            if other_label != label
        }
        nearest_wrong_label = min(wrong, key=wrong.get)
        rows.append(
            {
                "projection": projection,
                "space": space,
                "held_out": held_out,
                "label": label,
                "substrate": substrate,
                "distance_to_own_label_centroid": float(own),
                "nearest_wrong_label": nearest_wrong_label,
                "distance_to_nearest_wrong_label_centroid": float(wrong[nearest_wrong_label]),
                "projection_margin_wrong_minus_own": float(wrong[nearest_wrong_label] - own),
            }
        )
    return rows


def projection_frames(features, labels, substrates, seed):
    frames = {}
    frames["pca"] = pd.DataFrame(
        PCA(n_components=2, random_state=seed).fit_transform(features),
        columns=["x1", "x2"],
    )
    frames["tsne"] = pd.DataFrame(
        TSNE(
            n_components=2,
            perplexity=20,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(features),
        columns=["x1", "x2"],
    )
    frames["umap"] = pd.DataFrame(
        UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        ).fit_transform(features),
        columns=["x1", "x2"],
    )
    for frame in frames.values():
        frame["label"] = labels
        frame["substrate"] = substrates
    return frames


def save_projection_analysis(features, labels, substrates, space, held_out, out_dir, seed):
    projection_dir = out_dir / "projections"
    projection_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for projection, frame in projection_frames(features, labels, substrates, seed).items():
        stem = f"{space}__heldout_{held_out}__{projection}"
        frame.to_csv(projection_dir / f"{stem}.csv", index=False)
        rows.extend(projection_summary(frame, projection, space, held_out))
    return rows


def plot_silhouette_bars(score_df, out_dir):
    spaces = ["derivative_input", "siamese_embedding"]
    held_outs = sorted(score_df["held_out"].unique())
    x = np.arange(len(held_outs))
    width = 0.2
    series = [
        ("derivative_input", "silhouette_label", "Derivative label", "#4C78A8"),
        ("derivative_input", "silhouette_substrate", "Derivative substrate", "#9ECAE9"),
        ("siamese_embedding", "silhouette_label", "Embedding label", "#F58518"),
        ("siamese_embedding", "silhouette_substrate", "Embedding substrate", "#FFBF79"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    for offset_idx, (space, column, label, color) in enumerate(series):
        values = []
        for held_out in held_outs:
            row = score_df[(score_df["space"] == space) & (score_df["held_out"] == held_out)].iloc[0]
            values.append(row[column])
        ax.bar(x + (offset_idx - 1.5) * width, values, width, label=label, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(held_outs)
    ax.set_ylabel("Silhouette score")
    ax.set_title("Chemical-label vs substrate silhouette by held-out fold")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "silhouette_scores_by_fold.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset_idx, space in enumerate(spaces):
        values = [
            score_df[(score_df["space"] == space) & (score_df["held_out"] == held_out)][
                "label_minus_substrate_silhouette"
            ].iloc[0]
            for held_out in held_outs
        ]
        ax.bar(x + (offset_idx - 0.5) * 0.35, values, 0.35, label=space)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(held_outs)
    ax.set_ylabel("Label silhouette - substrate silhouette")
    ax.set_title("Representation preference for chemistry over substrate")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "silhouette_label_minus_substrate_by_fold.png", dpi=180)
    plt.close(fig)


def plot_sample_silhouette_distributions(sample_df, out_dir):
    long_rows = []
    for _, row in sample_df.iterrows():
        long_rows.append(
            {
                "space": row["space"],
                "held_out": row["held_out"],
                "label": row["label"],
                "substrate": row["substrate"],
                "metric": "chemical label",
                "silhouette": row["silhouette_label"],
            }
        )
        long_rows.append(
            {
                "space": row["space"],
                "held_out": row["held_out"],
                "label": row["label"],
                "substrate": row["substrate"],
                "metric": "substrate",
                "silhouette": row["silhouette_substrate"],
            }
        )
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out_dir / "silhouette_samples_long.csv", index=False)

    groups = [
        ("derivative_input", "chemical label"),
        ("siamese_embedding", "chemical label"),
        ("derivative_input", "substrate"),
        ("siamese_embedding", "substrate"),
    ]
    data = [
        long_df[(long_df["space"] == space) & (long_df["metric"] == metric)]["silhouette"].to_numpy()
        for space, metric in groups
    ]
    labels = ["Derivative\nlabel", "Embedding\nlabel", "Derivative\nsubstrate", "Embedding\nsubstrate"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Per-spectrum silhouette score")
    ax.set_title("Per-spectrum silhouette distributions")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "silhouette_sample_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, column, title in [
        (axes[0], "silhouette_label", "Chemical-label silhouette by class"),
        (axes[1], "silhouette_substrate", "Substrate silhouette by substrate"),
    ]:
        group_col = "label" if column == "silhouette_label" else "substrate"
        groups_sorted = sorted(sample_df[group_col].unique())
        positions = np.arange(len(groups_sorted))
        for offset, space in [(-0.18, "derivative_input"), (0.18, "siamese_embedding")]:
            data = [
                sample_df[(sample_df["space"] == space) & (sample_df[group_col] == group)][column].to_numpy()
                for group in groups_sorted
            ]
            box = ax.boxplot(
                data,
                positions=positions + offset,
                widths=0.28,
                patch_artist=True,
                showfliers=False,
            )
            color = "#4C78A8" if space == "derivative_input" else "#F58518"
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.45)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(groups_sorted, rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Per-spectrum silhouette score")
    axes[1].legend(
        handles=[
            plt.Line2D([0], [0], color="#4C78A8", linewidth=8, alpha=0.45, label="derivative_input"),
            plt.Line2D([0], [0], color="#F58518", linewidth=8, alpha=0.45, label="siamese_embedding"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "silhouette_by_class_and_substrate.png", dpi=180)
    plt.close(fig)


def write_markdown(out_dir, class_df, score_df, projection_df):
    lines = [
        "# SERS Geometry Analysis",
        "",
        "This analysis compares derivative-input geometry against Siamese embedding geometry.",
        "Positive margins mean the held-out class is closer to its own chemical prototype than to the nearest wrong chemical prototype.",
        "Negative margins mean collapse toward another chemical.",
        "",
        "## Separation Scores",
        "",
        score_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Silhouette visuals:",
        "",
        "- `silhouette_scores_by_fold.png` shows chemical-label and substrate silhouette scores side by side.",
        "- `silhouette_label_minus_substrate_by_fold.png` shows how strongly each space favors chemical identity over substrate identity.",
        "- `silhouette_sample_distributions.png` shows per-spectrum silhouette distributions.",
        "- `silhouette_by_class_and_substrate.png` breaks per-spectrum silhouettes down by chemical class and substrate.",
        "",
        "## Held-Out Class Prototype Margins",
        "",
        class_df[
            [
                "space",
                "held_out",
                "true_label",
                "n",
                "accuracy",
                "mean_own_distance",
                "mean_nearest_wrong_distance",
                "mean_margin_wrong_minus_own",
                "dominant_wrong_neighbor",
                "dominant_prediction",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Projection Centroid Margins",
        "",
        "PCA, UMAP, and t-SNE are qualitative projections. These margins are supporting evidence only because nonlinear projections can distort global distances.",
        "",
        projection_df.sort_values(
            ["projection_margin_wrong_minus_own", "space", "held_out", "label", "substrate"]
        ).to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    (out_dir / "geometry_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Workspace/data/processed/consolidated_SERS.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("Workspace/substrate_agnostic/diagnostics/geometry_analysis"))
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
        args.data,
        args.crop_min,
        args.crop_max,
        args.min_substrates,
        canonicalize_labels=True,
    )
    X_raw = df[cols].to_numpy(dtype=float)
    X = sers.prepare_features(X_raw, args)
    labels = df["Label"].astype(str).to_numpy()
    substrates = df["Substrate"].astype(str).to_numpy()

    sample_rows = []
    class_rows = []
    score_rows = []
    silhouette_rows = []
    projection_rows = []

    for held_out in sorted(np.unique(substrates)):
        input_sample, input_class = prototype_geometry(X, labels, substrates, held_out, "derivative_input")
        sample_rows.extend(input_sample)
        class_rows.extend(input_class)
        score_rows.extend(separation_scores(X, labels, substrates, "derivative_input", held_out))
        silhouette_rows.extend(sample_silhouettes(X, labels, substrates, "derivative_input", held_out))
        projection_rows.extend(
            save_projection_analysis(
                X,
                labels,
                substrates,
                "derivative_input",
                held_out,
                args.out_dir,
                args.seed,
            )
        )

        model, _ = train_triplet_model(X, labels, substrates, held_out, args, device)
        embeddings = sers.embed(model, X, device, args.batch_size)
        embedding_sample, embedding_class = prototype_geometry(
            embeddings,
            labels,
            substrates,
            held_out,
            "siamese_embedding",
        )
        sample_rows.extend(embedding_sample)
        class_rows.extend(embedding_class)
        score_rows.extend(separation_scores(embeddings, labels, substrates, "siamese_embedding", held_out))
        silhouette_rows.extend(sample_silhouettes(embeddings, labels, substrates, "siamese_embedding", held_out))
        projection_rows.extend(
            save_projection_analysis(
                embeddings,
                labels,
                substrates,
                "siamese_embedding",
                held_out,
                args.out_dir,
                args.seed,
            )
        )

    sample_df = pd.DataFrame(sample_rows)
    class_df = pd.DataFrame(class_rows)
    score_df = pd.DataFrame(score_rows)
    silhouette_df = pd.DataFrame(silhouette_rows)
    sample_df.to_csv(args.out_dir / "sample_prototype_distances.csv", index=False)
    class_df.to_csv(args.out_dir / "class_prototype_margins.csv", index=False)
    score_df.to_csv(args.out_dir / "space_separation_scores.csv", index=False)
    silhouette_df.to_csv(args.out_dir / "silhouette_samples.csv", index=False)
    plot_silhouette_bars(score_df, args.out_dir)
    plot_sample_silhouette_distributions(silhouette_df, args.out_dir)

    projection_df = pd.DataFrame(projection_rows)
    projection_df.to_csv(args.out_dir / "projection_centroid_margins.csv", index=False)

    write_markdown(args.out_dir, class_df, score_df, projection_df)
    print(f"Saved geometry analysis to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
