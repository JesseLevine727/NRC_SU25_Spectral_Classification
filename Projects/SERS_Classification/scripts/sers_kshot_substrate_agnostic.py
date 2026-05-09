#!/usr/bin/env python3
"""K-shot Siamese SERS detection with leave-one-substrate-family-out testing.

This script formalizes the "few-shot" claim for the substrate-agnostic work.
For each held-out substrate family, it samples only K support spectra from the
held-in training data, trains the same Siamese metric-learning encoder, and
tests chemical-label prediction on all known chemicals in the held-out family.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

import analyze_sers_geometry as geometry
import sers_siamese_substrate_agnostic as sers


DEFAULT_OUT_DIR = Path("Workspace/substrate_agnostic/grouped_metal_substrates/kshot_siamese")


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("K-shot values and seeds must be positive integers.")
    return values


def support_indices_for_fold(
    labels: np.ndarray,
    substrates: np.ndarray,
    held_out: str,
    k: int,
    seed: int,
    shot_unit: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    train_mask = substrates != held_out
    support: list[int] = []

    if shot_unit == "chemical_substrate":
        for label in sorted(np.unique(labels[train_mask])):
            label_mask = train_mask & (labels == label)
            for substrate in sorted(np.unique(substrates[label_mask])):
                candidates = np.where(label_mask & (substrates == substrate))[0]
                if len(candidates) == 0:
                    continue
                n_take = min(k, len(candidates))
                support.extend(rng.choice(candidates, size=n_take, replace=False).tolist())
    elif shot_unit == "chemical":
        for label in sorted(np.unique(labels[train_mask])):
            candidates = np.where(train_mask & (labels == label))[0]
            if len(candidates) == 0:
                continue
            n_take = min(k, len(candidates))
            support.extend(rng.choice(candidates, size=n_take, replace=False).tolist())
    else:
        raise ValueError(f"Unsupported shot unit: {shot_unit}")

    if not support:
        raise ValueError(f"No support rows sampled for held-out substrate {held_out}.")
    return np.array(sorted(support), dtype=np.int64)


def query_indices_for_fold(labels: np.ndarray, substrates: np.ndarray, held_out: str, support_idx: np.ndarray) -> np.ndarray:
    support_labels = np.unique(labels[support_idx])
    test_mask = (substrates == held_out) & np.isin(labels, support_labels)
    return np.where(test_mask)[0]


def support_count_string(labels: np.ndarray, substrates: np.ndarray, support_idx: np.ndarray) -> str:
    frame = pd.DataFrame({"label": labels[support_idx], "substrate": substrates[support_idx]})
    counts = frame.groupby(["label", "substrate"]).size()
    return ";".join(f"{label}/{substrate}:{count}" for (label, substrate), count in counts.items())


def train_model_on_support(
    X: np.ndarray,
    labels: np.ndarray,
    substrates: np.ndarray,
    support_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[sers.SiameseNet, float]:
    support_labels = labels[support_idx]
    support_substrates = substrates[support_idx]
    X_support = X[support_idx]

    if len(np.unique(support_labels)) < 2:
        raise ValueError("Siamese training requires at least two chemical labels in the support set.")

    if args.loss == "batch_hard_triplet":
        label_to_id = {label: idx for idx, label in enumerate(sorted(np.unique(support_labels)))}
        substrate_to_id = {group: idx for idx, group in enumerate(sorted(np.unique(support_substrates)))}
        label_ids = np.array([label_to_id[label] for label in support_labels], dtype=np.int64)
        substrate_ids = np.array([substrate_to_id[group] for group in support_substrates], dtype=np.int64)
        dataset = sers.RamanEmbeddingDataset(
            X_support,
            label_ids,
            substrate_ids,
            args.noise_std,
            args.shift_max,
        )
        batches_per_epoch = args.batches_per_epoch or max(1, int(np.ceil(len(dataset) / args.batch_size)))
        sampler = sers.BalancedLabelBatchSampler(
            label_ids,
            substrate_ids,
            args.labels_per_batch,
            args.samples_per_label,
            batches_per_epoch,
        )
        loader = DataLoader(dataset, batch_sampler=sampler)
    elif args.loss == "triplet":
        dataset = sers.RamanTripletDataset(X_support, support_labels, support_substrates, args.noise_std, args.shift_max)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        dataset = sers.RamanPairDataset(X_support, support_labels, args.noise_std, args.shift_max)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = sers.SiameseNet(X.shape[1], embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    final_loss = 0.0
    for _ in range(args.epochs):
        total = 0.0
        model.train()
        for batch in loader:
            if args.loss == "batch_hard_triplet":
                spectra, label_ids, substrate_ids = (item.to(device) for item in batch)
                loss = sers.batch_hard_triplet_loss(model(spectra), label_ids, substrate_ids, args.margin)
                batch_size = spectra.size(0)
            elif args.loss == "triplet":
                anchor, positive, negative = (item.to(device) for item in batch)
                loss = sers.triplet_loss(model(anchor), model(positive), model(negative), args.margin)
                batch_size = anchor.size(0)
            else:
                x1, x2, lbl = batch
                x1 = x1.to(device)
                x2 = x2.to(device)
                lbl = lbl.to(device)
                loss = sers.contrastive_loss(model(x1), model(x2), lbl, args.margin)
                batch_size = x1.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch_size
        final_loss = total / max(1, len(dataset))
    return model, final_loss


def build_prototypes(
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    support_substrates: np.ndarray,
    prototype_mode: str,
) -> tuple[list[str], np.ndarray]:
    prototype_labels = sorted(np.unique(support_labels))
    prototypes = []
    for label in prototype_labels:
        label_mask = support_labels == label
        if prototype_mode == "substrate_balanced":
            centroids = []
            for substrate in sorted(np.unique(support_substrates[label_mask])):
                group_mask = label_mask & (support_substrates == substrate)
                centroids.append(support_embeddings[group_mask].mean(axis=0))
            prototypes.append(np.vstack(centroids).mean(axis=0))
        elif prototype_mode == "row_mean":
            prototypes.append(support_embeddings[label_mask].mean(axis=0))
        else:
            raise ValueError(f"Unsupported prototype mode: {prototype_mode}")
    return prototype_labels, np.vstack(prototypes)


def predict_with_support_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    substrates: np.ndarray,
    support_idx: np.ndarray,
    query_idx: np.ndarray,
    prototype_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prototype_labels, prototypes = build_prototypes(
        features[support_idx],
        labels[support_idx],
        substrates[support_idx],
        prototype_mode,
    )
    distances = np.linalg.norm(features[query_idx, None, :] - prototypes[None, :, :], axis=2)
    pred = np.array([prototype_labels[idx] for idx in np.argmin(distances, axis=1)])
    return pred, distances, prototype_labels


@dataclass(frozen=True)
class KShotFoldResult:
    k: int
    shot_unit: str
    seed: int
    held_out_substrate: str
    prototype_mode: str
    n_train_available: int
    n_support: int
    n_test: int
    support_labels: str
    support_substrates: str
    support_counts: str
    test_labels: str
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    true_label_macro_f1: float
    final_loss: float


def evaluate_kshot_fold(
    X: np.ndarray,
    labels: np.ndarray,
    substrates: np.ndarray,
    held_out: str,
    k: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[KShotFoldResult | None, pd.DataFrame]:
    sers.set_seed(seed)
    support_idx = support_indices_for_fold(labels, substrates, held_out, k, seed, args.shot_unit)
    query_idx = query_indices_for_fold(labels, substrates, held_out, support_idx)
    if len(query_idx) == 0:
        return None, pd.DataFrame()

    model, final_loss = train_model_on_support(X, labels, substrates, support_idx, args, device)
    support_embeddings = sers.embed(model, X[support_idx], device, args.batch_size)
    query_embeddings = sers.embed(model, X[query_idx], device, args.batch_size)
    prototype_labels, prototypes = build_prototypes(
        support_embeddings,
        labels[support_idx],
        substrates[support_idx],
        args.prototype_mode,
    )
    distances = np.linalg.norm(query_embeddings[:, None, :] - prototypes[None, :, :], axis=2)
    pred = np.array([prototype_labels[idx] for idx in np.argmin(distances, axis=1)])
    truth = labels[query_idx]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        balanced_accuracy = balanced_accuracy_score(truth, pred)

    true_label_list = sorted(np.unique(truth))
    result = KShotFoldResult(
        k=k,
        shot_unit=args.shot_unit,
        seed=seed,
        held_out_substrate=held_out,
        prototype_mode=args.prototype_mode,
        n_train_available=int((substrates != held_out).sum()),
        n_support=int(len(support_idx)),
        n_test=int(len(query_idx)),
        support_labels=",".join(sorted(np.unique(labels[support_idx]))),
        support_substrates=",".join(sorted(np.unique(substrates[support_idx]))),
        support_counts=support_count_string(labels, substrates, support_idx),
        test_labels=",".join(true_label_list),
        accuracy=accuracy_score(truth, pred),
        balanced_accuracy=balanced_accuracy,
        macro_f1=f1_score(truth, pred, average="macro", zero_division=0),
        true_label_macro_f1=f1_score(truth, pred, labels=true_label_list, average="macro", zero_division=0),
        final_loss=final_loss,
    )

    matrix_labels = sorted(np.unique(np.concatenate([truth, pred])))
    matrix = confusion_matrix(truth, pred, labels=matrix_labels)
    confusion = pd.DataFrame(
        matrix,
        index=[f"true:{label}" for label in matrix_labels],
        columns=[f"pred:{label}" for label in matrix_labels],
    )
    return result, confusion


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby("k", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        std_balanced_accuracy=("balanced_accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_true_label_macro_f1=("true_label_macro_f1", "mean"),
        std_true_label_macro_f1=("true_label_macro_f1", "std"),
        min_fold_accuracy=("accuracy", "min"),
        min_fold_true_label_macro_f1=("true_label_macro_f1", "min"),
        mean_support_rows=("n_support", "mean"),
        n_fold_runs=("accuracy", "count"),
    )
    return grouped.sort_values("k")


def support_prototype_geometry(
    features: np.ndarray,
    labels: np.ndarray,
    substrates: np.ndarray,
    held_out: str,
    support_idx: np.ndarray,
    k: int,
    seed: int,
    space: str,
    prototype_mode: str,
) -> tuple[list[dict], list[dict]]:
    query_idx = query_indices_for_fold(labels, substrates, held_out, support_idx)
    if len(query_idx) == 0:
        return [], []

    pred, distances, prototype_labels = predict_with_support_prototypes(
        features,
        labels,
        substrates,
        support_idx,
        query_idx,
        prototype_mode,
    )
    truth = labels[query_idx]

    sample_rows = []
    for local_idx, row_idx in enumerate(query_idx):
        row = {
            "k": k,
            "seed": seed,
            "space": space,
            "held_out": held_out,
            "prototype_mode": prototype_mode,
            "row_index": int(row_idx),
            "true_label": truth[local_idx],
            "pred_label": pred[local_idx],
            "correct": bool(pred[local_idx] == truth[local_idx]),
        }
        for prototype_idx, prototype_label in enumerate(prototype_labels):
            row[f"distance_to_{prototype_label}"] = float(distances[local_idx, prototype_idx])
        sample_rows.append(row)

    class_rows = []
    prototype_index = {label: idx for idx, label in enumerate(prototype_labels)}
    for label in sorted(np.unique(truth)):
        mask = truth == label
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
                "k": k,
                "seed": seed,
                "space": space,
                "held_out": held_out,
                "prototype_mode": prototype_mode,
                "true_label": label,
                "n": int(mask.sum()),
                "accuracy": float((pred[mask] == label).mean()),
                "mean_own_distance": float(own_dist.mean()),
                "mean_nearest_wrong_distance": float(nearest_wrong_dist.mean()),
                "mean_margin_wrong_minus_own": float((nearest_wrong_dist - own_dist).mean()),
                "dominant_wrong_neighbor": pd.Series(nearest_wrong_labels).mode().iat[0],
                "dominant_prediction": pd.Series(pred[mask]).mode().iat[0],
            }
        )
    return sample_rows, class_rows


def write_performance_markdown(out_dir: Path, results: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    best_row = summary.sort_values("mean_true_label_macro_f1", ascending=False).iloc[0]
    lines = [
        "# K-Shot Substrate-Agnostic Siamese Evaluation",
        "",
        "This run tests whether the current substrate-agnostic result is still a formal few-shot result.",
        f"`K` is sampled per `{args.shot_unit}` unit from the held-in substrate families only.",
        "Each fold then tests all known chemical labels in the held-out substrate family.",
        "",
        "## Protocol",
        "",
        f"- Feature: `{args.feature}`",
        f"- Loss: `{args.loss}`",
        f"- Prototype mode: `{args.prototype_mode}`",
        f"- Epochs: `{args.epochs}`",
        f"- Device requirement: CUDA unless `--allow-cpu` is explicitly passed",
        f"- Grouped metal substrates: `{args.group_metal_substrates}`",
        f"- K values: `{','.join(str(k) for k in args.ks)}`",
        f"- Seeds: `{','.join(str(seed) for seed in args.seeds)}`",
        "",
        "## Summary By K",
        "",
        summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "This is a formal few-shot test, but it is not yet a robust few-shot substrate-agnostic result.",
        "The model can perform well with very few support spectra, but the variance is large across seeds and held-out substrate families.",
        f"`K={int(best_row['k'])}` has the strongest mean true-label macro F1 in this run "
        f"({best_row['mean_true_label_macro_f1']:.3f}), while the weakest fold accuracy is "
        f"{best_row['min_fold_accuracy']:.3f}.",
        "",
        "For poster wording: the project began as few-shot chemical-substrate pair learning and has now moved to substrate-agnostic transfer.",
        "The current substrate-agnostic result should be described as Siamese metric learning on a small dataset unless the K-shot protocol is explicitly reported with its variance.",
        "",
        "## Fold Means By K And Held-Out Family",
        "",
        results.groupby(["k", "held_out_substrate"], as_index=False)[
            ["accuracy", "balanced_accuracy", "macro_f1", "true_label_macro_f1", "n_support"]
        ]
        .mean()
        .to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Files",
        "",
        "- `detailed_results.csv`",
        "- `summary_by_k.csv`",
        "- `confusions/k*/seed*/*.csv`",
        "- `diagnostics/geometry_k*/geometry_analysis.md`",
        "- `diagnostics/geometry_k*/projections/*.png`",
        "- `diagnostics/geometry_k*/silhouette_*.png`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_geometry_markdown(out_dir: Path, class_df: pd.DataFrame, score_df: pd.DataFrame, projection_df: pd.DataFrame) -> None:
    lines = [
        "# K-Shot SERS Geometry Analysis",
        "",
        "This analysis mirrors the full-data geometry diagnostics, but the Siamese encoder is trained only on the sampled K-shot support set for each held-out fold.",
        "The derivative-input space is unchanged; the Siamese-embedding space reflects the K-shot-trained encoder.",
        "",
        "## Separation Scores",
        "",
        score_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Held-Out Class Prototype Margins",
        "",
        class_df[
            [
                "k",
                "seed",
                "space",
                "held_out",
                "prototype_mode",
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


def run_geometry_analysis(
    X: np.ndarray,
    labels: np.ndarray,
    substrates: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if args.geometry_k is None:
        return
    k = args.geometry_k
    seed = args.geometry_seed
    out_dir = args.geometry_out_dir or (args.out_dir / "diagnostics" / f"geometry_k{k}_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict] = []
    class_rows: list[dict] = []
    score_rows: list[dict] = []
    silhouette_rows: list[dict] = []
    projection_rows: list[dict] = []

    for held_out in sorted(np.unique(substrates)):
        sers.set_seed(seed)
        support_idx = support_indices_for_fold(labels, substrates, held_out, k, seed, args.shot_unit)

        input_sample, input_class = support_prototype_geometry(
            X,
            labels,
            substrates,
            held_out,
            support_idx,
            k,
            seed,
            "derivative_input",
            args.prototype_mode,
        )
        sample_rows.extend(input_sample)
        class_rows.extend(input_class)
        score_rows.extend(geometry.separation_scores(X, labels, substrates, "derivative_input", held_out))
        silhouette_rows.extend(geometry.sample_silhouettes(X, labels, substrates, "derivative_input", held_out))
        projection_rows.extend(
            geometry.save_projection_analysis(X, labels, substrates, "derivative_input", held_out, out_dir, seed)
        )

        model, _ = train_model_on_support(X, labels, substrates, support_idx, args, device)
        embeddings = sers.embed(model, X, device, args.batch_size)
        embedding_sample, embedding_class = support_prototype_geometry(
            embeddings,
            labels,
            substrates,
            held_out,
            support_idx,
            k,
            seed,
            "siamese_embedding",
            args.prototype_mode,
        )
        sample_rows.extend(embedding_sample)
        class_rows.extend(embedding_class)
        score_rows.extend(geometry.separation_scores(embeddings, labels, substrates, "siamese_embedding", held_out))
        silhouette_rows.extend(geometry.sample_silhouettes(embeddings, labels, substrates, "siamese_embedding", held_out))
        projection_rows.extend(
            geometry.save_projection_analysis(embeddings, labels, substrates, "siamese_embedding", held_out, out_dir, seed)
        )

    sample_df = pd.DataFrame(sample_rows)
    class_df = pd.DataFrame(class_rows)
    score_df = pd.DataFrame(score_rows)
    silhouette_df = pd.DataFrame(silhouette_rows)
    projection_df = pd.DataFrame(projection_rows)

    sample_df.to_csv(out_dir / "sample_prototype_distances.csv", index=False)
    class_df.to_csv(out_dir / "class_prototype_margins.csv", index=False)
    score_df.to_csv(out_dir / "space_separation_scores.csv", index=False)
    silhouette_df.to_csv(out_dir / "silhouette_samples.csv", index=False)
    projection_df.to_csv(out_dir / "projection_centroid_margins.csv", index=False)
    geometry.plot_silhouette_bars(score_df, out_dir)
    geometry.plot_sample_silhouette_distributions(silhouette_df, out_dir)
    write_geometry_markdown(out_dir, class_df, score_df, projection_df)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=sers.DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--crop-min", type=float, default=330.0)
    parser.add_argument("--crop-max", type=float, default=1800.0)
    parser.add_argument("--min-substrates", type=int, default=2)
    parser.add_argument("--ks", type=parse_int_list, default=parse_int_list("1,3,5,10,25"))
    parser.add_argument("--seeds", type=parse_int_list, default=parse_int_list("42,43,44,45,46"))
    parser.add_argument(
        "--shot-unit",
        choices=["chemical_substrate", "chemical"],
        default="chemical_substrate",
        help="Sampling unit for K. The default matches the original chemical-substrate pair few-shot framing.",
    )
    parser.add_argument(
        "--group-metal-substrates",
        action="store_true",
        help="Group AgNP with Ag and AuNP with Au before leave-substrate-out evaluation.",
    )
    parser.add_argument(
        "--no-canonicalize-labels",
        action="store_true",
        help="Disable chemical label canonicalization such as bt -> benzenethiol.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument(
        "--loss",
        choices=["batch_hard_triplet", "triplet", "contrastive"],
        default="triplet",
    )
    parser.add_argument(
        "--prototype-mode",
        choices=["row_mean", "substrate_balanced"],
        default="row_mean",
    )
    parser.add_argument("--labels-per-batch", type=int, default=4)
    parser.add_argument("--samples-per-label", type=int, default=8)
    parser.add_argument("--batches-per-epoch", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--shift-max", type=int, default=2)
    parser.add_argument("--baseline-lam", type=float, default=1e4)
    parser.add_argument("--baseline-p", type=float, default=0.01)
    parser.add_argument("--baseline-niter", type=int, default=10)
    parser.add_argument(
        "--feature",
        choices=["raw", "als", "snv_l2", "derivative_1", "derivative_2", "peak_emphasis"],
        default="derivative_1",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global seed before all fold sampling/training.")
    parser.add_argument(
        "--device",
        choices=["cuda", "auto", "cpu"],
        default="cuda",
        help="Training device. Defaults to CUDA and fails if no GPU is available.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit CPU training. Without this flag, CPU execution is rejected.",
    )
    parser.add_argument(
        "--geometry-k",
        type=int,
        default=5,
        help="Generate matching clustering/geometry diagnostics for this K. Use 0 to skip.",
    )
    parser.add_argument("--geometry-seed", type=int, default=42)
    parser.add_argument("--geometry-out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.geometry_k == 0:
        args.geometry_k = None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sers.set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "CPU training is disabled. Use --device cuda on a CUDA machine, "
            "or pass --allow-cpu only for an intentional CPU debug run."
        )

    df, cols = sers.load_dataset(
        args.data,
        args.crop_min,
        args.crop_max,
        args.min_substrates,
        canonicalize_labels=not args.no_canonicalize_labels,
        group_metal_substrates=args.group_metal_substrates,
    )
    X_raw = df[cols].to_numpy(dtype=float)
    X = sers.prepare_features(X_raw, args)
    labels = df["Label"].astype(str).to_numpy()
    substrates = df["Substrate"].astype(str).to_numpy()

    rows: list[KShotFoldResult] = []
    confusions_dir = args.out_dir / "confusions"
    for k in args.ks:
        for seed in args.seeds:
            for held_out in sorted(np.unique(substrates)):
                result, confusion = evaluate_kshot_fold(X, labels, substrates, held_out, k, seed, args, device)
                if result is None:
                    continue
                rows.append(result)
                fold_dir = confusions_dir / f"k{k}" / f"seed{seed}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                confusion.to_csv(fold_dir / f"{held_out}.csv")
                print(
                    f"k={k:>2} seed={seed} held_out={held_out}: "
                    f"acc={result.accuracy:.3f}, true_macro_f1={result.true_label_macro_f1:.3f}, "
                    f"support={result.n_support}"
                )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No K-shot folds were evaluated.")
    summary = summarize_results(results)
    results.to_csv(args.out_dir / "detailed_results.csv", index=False)
    summary.to_csv(args.out_dir / "summary_by_k.csv", index=False)
    with (args.out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, indent=2)
    write_performance_markdown(args.out_dir, results, summary, args)

    run_geometry_analysis(X, labels, substrates, args, device)

    print("\nDataset:", args.data)
    print("Device:", device)
    print("Feature:", args.feature)
    print("Loss:", args.loss)
    print("Prototype mode:", args.prototype_mode)
    print("Shot unit:", args.shot_unit)
    print("Grouped metal substrates:", args.group_metal_substrates)
    print("Labels:", ", ".join(sorted(np.unique(labels))))
    print("Substrates:", ", ".join(sorted(np.unique(substrates))))
    print("\nSummary by K:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved K-shot results to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
