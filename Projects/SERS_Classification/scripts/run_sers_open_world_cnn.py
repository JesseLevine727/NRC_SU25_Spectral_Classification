#!/usr/bin/env python3
"""Run the preregistered compact residual CNN SERS open-world study."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.special import logsumexp

import sers_open_world_cnn_common as cnn
import sers_open_world_common as common
import sers_open_world_neural_common as neural


SCORES = (
    "one_minus_calibrated_max_probability",
    "predictive_entropy",
    "energy",
    "minimum_class_mahalanobis_distance",
    "mc_dropout_mutual_information",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("surrogate", "select", "calibration", "final", "all"),
        default="all",
    )
    parser.add_argument("--held-unknown", default=None)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/sers_open_world_cnn_v1.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/sers_open_world_shift_v1.json"),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("Workspace/nato_sers_field_trial/preprocessing_v2"),
    )
    parser.add_argument(
        "--partition-dir",
        type=Path,
        default=Path("Workspace/sers_open_world_shift/open_world_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Workspace/sers_open_world_shift/open_world_v1/cnn"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def initialize(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, source in (
        ("predeclared_cnn_registry.json", args.registry),
        ("predeclared_protocol.json", args.protocol),
    ):
        target = args.output_dir / filename
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Different CNN immutable snapshot: {filename}")
        if not target.exists():
            shutil.copyfile(source, target)
    validation = json.loads(
        (args.partition_dir / "validation_report.json").read_text()
    )
    if validation["status"] != "pass":
        raise ValueError("Partitions are not validated")
    common.write_json(
        args.output_dir / "input_hashes.json",
        {
            "cnn_registry": {
                "path": str(args.registry.resolve()),
                "sha256": common.sha256_file(args.registry),
            },
            "protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": common.sha256_file(args.protocol),
            },
            "partition_validation": {
                "path": str(
                    (
                        args.partition_dir / "validation_report.json"
                    ).resolve()
                ),
                "sha256": common.sha256_file(
                    args.partition_dir / "validation_report.json"
                ),
            },
            "preprocessing_bundle_hash_catalog": {
                "path": str((args.bundle / "artifact_hashes.json").resolve()),
                "sha256": common.sha256_file(
                    args.bundle / "artifact_hashes.json"
                ),
            },
        },
    )
    common.write_json(
        args.output_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            "device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    )


def uid_lookup(dataset: common.baseline.SpectralDataset) -> dict[str, int]:
    return {
        uid: index
        for index, uid in enumerate(
            dataset.manifest["observation_uid"].astype(str)
        )
    }


def partitions(
    frame: pd.DataFrame, lookup: dict[str, int]
) -> dict[str, np.ndarray]:
    return {
        str(name): np.asarray(
            [lookup[str(uid)] for uid in group["observation_uid"]], dtype=int
        )
        for name, group in frame.groupby("partition", sort=False)
    }


def output_scores(
    trained: cnn.Trained,
    density: neural.EmbeddingDensity,
    values: np.ndarray,
    device: torch.device,
    temperature: float,
    mc_passes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    embedding, logits = cnn.deterministic_outputs(
        trained.model, values, device
    )
    probabilities = common.probabilities_from_scores(logits, temperature)
    _, mutual_information = cnn.mc_dropout_outputs(
        trained.model, values, device, mc_passes
    )
    scores = {
        "one_minus_calibrated_max_probability": (
            1.0 - probabilities.max(axis=1)
        ),
        "predictive_entropy": common.predictive_entropy(probabilities),
        "energy": -logsumexp(logits, axis=1),
        "minimum_class_mahalanobis_distance": density.mahalanobis(embedding),
        "mc_dropout_mutual_information": mutual_information,
    }
    return embedding, logits, probabilities, scores


def same_master_prediction_agreement(
    manifest: pd.DataFrame,
    predictions: np.ndarray,
    domain_column: str,
) -> dict[str, Any]:
    frame = manifest.reset_index(drop=True)
    prediction = np.asarray(predictions, dtype=str)
    agreements: list[bool] = []
    masters = 0
    for _, indices in frame.groupby("master_sample_id", sort=True).groups.items():
        index = list(indices)
        pairs = [
            (left, right)
            for left, right in combinations(index, 2)
            if str(frame.iloc[left][domain_column])
            != str(frame.iloc[right][domain_column])
        ]
        if not pairs:
            continue
        masters += 1
        agreements.extend(
            prediction[left] == prediction[right] for left, right in pairs
        )
    return {
        "masters_with_cross_domain_pairs": masters,
        "cross_domain_pairs": len(agreements),
        "prediction_agreement": (
            float(np.mean(agreements)) if agreements else np.nan
        ),
    }


def fit_density(
    trained: cnn.Trained,
    values: np.ndarray,
    manifest: pd.DataFrame,
    registry: dict[str, Any],
    seed: int,
    device: torch.device,
) -> neural.EmbeddingDensity:
    embedding, _ = cnn.deterministic_outputs(trained.model, values, device)
    # Reuse the preregistered stable density implementation; the CNN registry
    # declares the same 16-D PCA and Ledoit-Wolf policy by reference.
    density_registry = {
        "embedding_density": {
            "pca_components": 16,
            "pca_whiten": False,
        }
    }
    return neural.fit_embedding_density(
        embedding,
        manifest["target_analyte"].astype(str).to_numpy(),
        trained.classes,
        seed,
        density_registry,
    )


def surrogate_one(
    dataset: common.baseline.SpectralDataset,
    registry: dict[str, Any],
    variant: cnn.Variant,
    context: dict[str, Any],
    split: dict[str, np.ndarray],
    device: torch.device,
) -> list[dict[str, Any]]:
    train = split["train_known"]
    known = split["validation_known"]
    unknown = split["validation_surrogate_unknown"]
    values = dataset.representations[variant.representation]
    manifest = dataset.manifest
    seed = cnn.stable_seed(
        cnn.REGISTRY_VERSION,
        "surrogate",
        context["held_unknown"],
        context["outer_fold"],
        context["inner_fold"],
        context["surrogate_unknown"],
        variant.name,
        registry["training"]["selection_seed"],
    )
    trained = cnn.train(
        values[train],
        manifest.iloc[train].reset_index(drop=True),
        values[known],
        manifest.iloc[known].reset_index(drop=True),
        variant,
        registry,
        seed,
        device,
    )
    density = fit_density(
        trained,
        values[train],
        manifest.iloc[train].reset_index(drop=True),
        registry,
        seed,
        device,
    )
    evaluation = np.concatenate([known, unknown])
    _, _, probabilities, scores = output_scores(
        trained,
        density,
        values[evaluation],
        device,
        1.0,
        int(registry["mc_dropout_passes"]),
    )
    predictions = trained.classes[np.argmax(probabilities, axis=1)]
    unknown_mask = np.concatenate(
        [np.zeros(len(known), bool), np.ones(len(unknown), bool)]
    )
    rows = []
    for score_order, score_name in enumerate(SCORES):
        metrics, _ = common.open_set_metrics(
            manifest.iloc[evaluation].reset_index(drop=True),
            manifest.iloc[evaluation]["target_analyte"].astype(str).to_numpy(),
            predictions,
            probabilities,
            trained.classes,
            scores[score_name],
            unknown_mask,
        )
        rows.append(
            {
                **context,
                "variant_order": variant.order,
                "variant": variant.name,
                "representation": variant.representation,
                "score_order": score_order,
                "score_name": score_name,
                "seed": seed,
                "best_epoch": trained.best_epoch,
                "epochs_run": len(trained.history),
                "parameter_count": trained.parameter_count,
                "state_sha256": trained.state_sha256,
                **metrics,
            }
        )
    return rows


def select_surrogate(metrics: pd.DataFrame, output: Path) -> None:
    summary = (
        metrics.groupby(
            [
                "held_unknown",
                "outer_fold",
                "variant_order",
                "variant",
                "representation",
                "score_order",
                "score_name",
            ],
            as_index=False,
        )
        .agg(
            mean_master_oscr_auc=("master_oscr_auc", "mean"),
            mean_master_known_balanced_accuracy=(
                "master_known_balanced_accuracy",
                "mean",
            ),
            mean_master_unknown_auroc=("master_unknown_auroc", "mean"),
            scenarios_completed=("master_oscr_auc", "count"),
            selected_epoch=("best_epoch", "median"),
        )
    )
    summary.to_csv(output / "surrogate_candidate_summary.csv", index=False)
    winners = []
    for keys, frame in summary.groupby(
        ["held_unknown", "outer_fold"], sort=True
    ):
        eligible = frame[frame.scenarios_completed == 20]
        ranked = eligible.sort_values(
            [
                "mean_master_oscr_auc",
                "mean_master_known_balanced_accuracy",
                "mean_master_unknown_auroc",
                "variant_order",
                "score_order",
            ],
            ascending=[False, False, False, True, True],
            kind="stable",
        )
        winner = ranked.iloc[0].to_dict()
        winner["selected_epoch"] = int(
            np.floor(float(winner["selected_epoch"]) + 0.5)
        )
        winners.append(winner)
    if len(winners) != 30:
        raise ValueError(f"Expected 30 CNN winners, got {len(winners)}")
    pd.DataFrame(winners).to_csv(
        output / "surrogate_selection.csv", index=False
    )


def run_surrogate(
    args: argparse.Namespace,
    dataset: common.baseline.SpectralDataset,
    registry: dict[str, Any],
    device: torch.device,
) -> None:
    frame = pd.read_csv(
        args.partition_dir / "surrogate_unknown_inner_partitions.csv"
    )
    if args.held_unknown is not None:
        frame = frame[frame["held_unknown"] == args.held_unknown].copy()
        if frame.empty:
            raise ValueError(f"No surrogate rows for {args.held_unknown}")
    lookup = uid_lookup(dataset)
    columns = [
        "held_unknown",
        "outer_fold",
        "inner_fold",
        "surrogate_unknown",
    ]
    variants = cnn.variants(registry)
    total = frame.groupby(columns).ngroups * len(variants)
    shard = args.held_unknown or "all"
    cache_dir = args.output_dir / "cache" / "surrogate"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / f"surrogate_metrics__{shard}.csv"
    rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[Any, ...]] = set()
    if checkpoint.exists():
        prior = pd.read_csv(checkpoint)
        required = set(columns + ["variant", "score_name"])
        if not required.issubset(prior.columns):
            raise ValueError(
                f"Invalid surrogate checkpoint columns: {checkpoint}"
            )
        rows = prior.to_dict("records")
        counts = prior.groupby(columns + ["variant"])["score_name"].nunique()
        if not (counts == len(SCORES)).all():
            raise ValueError(
                f"Incomplete fitted-model rows in checkpoint: {checkpoint}"
            )
        completed_keys = {
            tuple(key if isinstance(key, tuple) else (key,))
            for key in counts.index
        }
        if len(completed_keys) > total:
            raise ValueError(
                f"Surrogate checkpoint exceeds expected fits: {checkpoint}"
            )
        if completed_keys:
            print(
                f"RESUME SURROGATE {len(completed_keys)}/{total} "
                f"from {checkpoint}",
                flush=True,
            )

    def save_checkpoint() -> None:
        temporary = checkpoint.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(checkpoint)

    completed = len(completed_keys)
    for keys, scenario in frame.groupby(columns, sort=True):
        context = dict(zip(columns, keys))
        split = partitions(scenario, lookup)
        for variant in variants:
            fit_key = tuple(keys) + (variant.name,)
            if fit_key in completed_keys:
                continue
            rows.extend(
                surrogate_one(
                    dataset, registry, variant, context, split, device
                )
            )
            completed += 1
            completed_keys.add(fit_key)
            if completed % 5 == 0 or completed == total:
                save_checkpoint()
            if completed % 25 == 0:
                print(f"SURROGATE {completed}/{total}", flush=True)
    metrics = pd.DataFrame(rows)
    if len(completed_keys) != total or len(metrics) != total * len(SCORES):
        raise ValueError(
            f"Incomplete surrogate result: {len(completed_keys)}/{total} "
            f"fits and {len(metrics)}/{total * len(SCORES)} score rows"
        )
    if args.held_unknown is None:
        metrics.to_csv(args.output_dir / "surrogate_metrics.csv", index=False)
        select_surrogate(metrics, args.output_dir)
    else:
        metrics.to_csv(
            args.output_dir
            / f"surrogate_metrics__{args.held_unknown}.csv",
            index=False,
        )
    shutil.copyfile(
        (
            args.output_dir / "surrogate_metrics.csv"
            if args.held_unknown is None
            else args.output_dir
            / f"surrogate_metrics__{args.held_unknown}.csv"
        ),
        checkpoint,
    )


def merge_surrogate_shards(args: argparse.Namespace) -> None:
    paths = sorted(args.output_dir.glob("surrogate_metrics__*.csv"))
    if len(paths) != 6:
        raise ValueError(f"Expected six CNN surrogate shards, got {len(paths)}")
    metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = 600 * 7 * 5
    if len(metrics) != expected:
        raise ValueError(
            f"Expected {expected} surrogate score rows, got {len(metrics)}"
        )
    metrics.to_csv(args.output_dir / "surrogate_metrics.csv", index=False)
    cache_dir = args.output_dir / "cache" / "surrogate"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copyfile(path, cache_dir / path.name)
    select_surrogate(metrics, args.output_dir)


def train_selected(
    dataset: common.baseline.SpectralDataset,
    registry: dict[str, Any],
    selected: pd.Series,
    train: np.ndarray,
    monitor: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[cnn.Trained, neural.EmbeddingDensity]:
    variant = next(
        item
        for item in cnn.variants(registry)
        if item.name == str(selected["variant"])
    )
    values = dataset.representations[variant.representation]
    manifest = dataset.manifest
    trained = cnn.train(
        values[train],
        manifest.iloc[train].reset_index(drop=True),
        None,
        None,
        variant,
        registry,
        seed,
        device,
        fixed_epochs=int(selected["selected_epoch"]),
    )
    density = fit_density(
        trained,
        values[train],
        manifest.iloc[train].reset_index(drop=True),
        registry,
        seed,
        device,
    )
    return trained, density


def run_calibration(
    args: argparse.Namespace,
    dataset: common.baseline.SpectralDataset,
    registry: dict[str, Any],
    device: torch.device,
) -> None:
    selection = pd.read_csv(args.output_dir / "surrogate_selection.csv")
    frame = pd.read_csv(
        args.partition_dir / "known_threshold_crossfit_partitions.csv"
    )
    lookup = uid_lookup(dataset)
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    metric_rows, threshold_rows, score_rows, run_rows = [], [], [], []
    for _, selected in selection.iterrows():
        held = str(selected["held_unknown"])
        outer = int(selected["outer_fold"])
        target_classes = np.asarray(sorted(set(labels) - {held}), dtype=str)
        logits_parts, labels_parts, indices_parts = [], [], []
        score_parts = {name: [] for name in SCORES}
        for inner in sorted(set(range(5)) - {outer}):
            scenario = frame[
                (frame.held_unknown == held)
                & (frame.outer_fold == outer)
                & (frame.calibration_fold == inner)
            ]
            split = partitions(scenario, lookup)
            train, calibration = split["train_known"], split["calibration_known"]
            seed = cnn.stable_seed(
                cnn.REGISTRY_VERSION, "calibration", held, outer, inner, 1729
            )
            trained, density = train_selected(
                dataset, registry, selected, train, calibration, seed, device
            )
            values = dataset.representations[str(selected["representation"])]
            _, logits, _, scores = output_scores(
                trained,
                density,
                values[calibration],
                device,
                1.0,
                int(registry["mc_dropout_passes"]),
            )
            logits_parts.append(logits)
            labels_parts.append(labels[calibration])
            indices_parts.append(calibration)
            for name in SCORES:
                score_parts[name].append(scores[name])
            run_rows.append(
                {
                    "held_unknown": held,
                    "outer_fold": outer,
                    "calibration_fold": inner,
                    "variant": selected["variant"],
                    "representation": selected["representation"],
                    "epoch": int(selected["selected_epoch"]),
                    "seed": seed,
                    "state_sha256": trained.state_sha256,
                    "parameter_count": trained.parameter_count,
                }
            )
        indices = np.concatenate(indices_parts)
        order = np.argsort(indices, kind="stable")
        logits = np.vstack(logits_parts)[order]
        truth = np.concatenate(labels_parts)[order]
        indices = indices[order]
        temperature = common.fit_temperature(
            logits,
            truth,
            target_classes,
            tuple(registry["calibration"]["temperature_bounds"]),
        )
        probabilities = common.probabilities_from_scores(logits, temperature)
        calibrated = {
            "one_minus_calibrated_max_probability": (
                1.0 - probabilities.max(axis=1)
            ),
            "predictive_entropy": common.predictive_entropy(probabilities),
            **{
                name: np.concatenate(score_parts[name])[order]
                for name in SCORES[2:]
            },
        }
        prediction = target_classes[np.argmax(probabilities, axis=1)]
        metrics = common.classification_metrics(
            target_classes,
            truth,
            prediction,
            probabilities,
            target_classes,
        )
        metric_rows.append(
            {
                "held_unknown": held,
                "outer_fold": outer,
                "variant": selected["variant"],
                "representation": selected["representation"],
                "selected_score": selected["score_name"],
                "selected_epoch": int(selected["selected_epoch"]),
                "temperature": temperature,
                **{k: v for k, v in metrics.items() if k != "supported_mask"},
            }
        )
        for name, score in calibrated.items():
            for coverage in registry["calibration"]["known_retained_coverages"]:
                threshold_rows.append(
                    {
                        "held_unknown": held,
                        "outer_fold": outer,
                        "score_name": name,
                        "score_selected": name == selected["score_name"],
                        "requested_known_coverage": coverage,
                        "threshold": common.threshold_from_known(
                            score, float(coverage)
                        ),
                    }
                )
        for offset, index in enumerate(indices):
            score_rows.append(
                {
                    "held_unknown": held,
                    "outer_fold": outer,
                    "observation_uid": manifest.iloc[index]["observation_uid"],
                    "master_sample_id": manifest.iloc[index]["master_sample_id"],
                    **{
                        name: float(score[offset])
                        for name, score in calibrated.items()
                    },
                }
            )
        print(f"CALIBRATION CNN H={held} O={outer}", flush=True)
    pd.DataFrame(metric_rows).to_csv(
        args.output_dir / "calibration_metrics.csv", index=False
    )
    pd.DataFrame(threshold_rows).to_csv(
        args.output_dir / "calibration_thresholds.csv", index=False
    )
    pd.DataFrame(score_rows).to_csv(
        args.output_dir / "calibration_scores.csv", index=False
    )
    pd.DataFrame(run_rows).to_csv(
        args.output_dir / "calibration_run_registry.csv", index=False
    )


def run_final(
    args: argparse.Namespace,
    dataset: common.baseline.SpectralDataset,
    registry: dict[str, Any],
    device: torch.device,
) -> None:
    selection = pd.read_csv(args.output_dir / "surrogate_selection.csv")
    calibration = pd.read_csv(args.output_dir / "calibration_metrics.csv")
    thresholds = pd.read_csv(args.output_dir / "calibration_thresholds.csv")
    outer_frame = pd.read_csv(
        args.partition_dir / "open_set_outer_partitions.csv"
    )
    lookup = uid_lookup(dataset)
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    outputs = {name: [] for name in (
        "predictions", "classification", "open_metrics",
        "threshold_metrics", "run_registry", "histories",
        "target_adjusted_domain_probes", "same_master_agreement",
        "mc_pass_sensitivity",
    )}
    for _, selected in selection.iterrows():
        held, outer = str(selected.held_unknown), int(selected.outer_fold)
        split = partitions(
            outer_frame[
                (outer_frame.held_unknown == held)
                & (outer_frame.outer_fold == outer)
            ],
            lookup,
        )
        development = split["known_development_quality"]
        cal = calibration[
            (calibration.held_unknown == held)
            & (calibration.outer_fold == outer)
        ].iloc[0]
        temperature = float(cal.temperature)
        threshold_map = {
            name: {
                float(row.requested_known_coverage): float(row.threshold)
                for _, row in group.iterrows()
            }
            for name, group in thresholds[
                (thresholds.held_unknown == held)
                & (thresholds.outer_fold == outer)
            ].groupby("score_name")
        }
        for declared_seed in registry["training"]["final_seeds"]:
            seed = cnn.stable_seed(
                cnn.REGISTRY_VERSION, "final", held, outer, declared_seed
            )
            trained, density = train_selected(
                dataset, registry, selected, development, development, seed, device
            )
            base = {
                "held_unknown": held,
                "outer_fold": outer,
                "declared_seed": declared_seed,
                "seed": seed,
                "variant": selected.variant,
                "representation": selected.representation,
                "selected_score": selected.score_name,
                "selected_epoch": int(selected.selected_epoch),
                "temperature": temperature,
            }
            outputs["run_registry"].append(
                {
                    **base,
                    "state_sha256": trained.state_sha256,
                    "parameter_count": trained.parameter_count,
                    "n_development": len(development),
                    "n_development_masters": manifest.iloc[
                        development
                    ].master_sample_id.nunique(),
                }
            )
            outputs["histories"].extend(
                [{**base, **row} for row in trained.history.to_dict("records")]
            )
            evaluated = {}
            values = dataset.representations[str(selected.representation)]
            development_embedding, _ = cnn.deterministic_outputs(
                trained.model, values[development], device
            )
            mc_sensitivity: dict[
                str, dict[int, tuple[np.ndarray, np.ndarray]]
            ] = {}
            for partition_name in (
                "known_quality_test", "unknown_quality",
                "known_stress", "unknown_stress"
            ):
                index = split[partition_name]
                mc_seed = cnn.stable_seed(
                    seed, "mc_dropout", partition_name
                )
                common.baseline.configure_determinism(mc_seed)
                embedding, _, probabilities, score_values = output_scores(
                    trained,
                    density,
                    values[index],
                    device,
                    temperature,
                    int(registry["mc_dropout_passes"]),
                )
                common.baseline.configure_determinism(mc_seed)
                mc_sensitivity[partition_name] = (
                    cnn.mc_dropout_prefix_outputs(
                        trained.model,
                        values[index],
                        device,
                        (10, int(registry["mc_dropout_passes"]), 50),
                    )
                )
                registered_mi = mc_sensitivity[partition_name][
                    int(registry["mc_dropout_passes"])
                ][1]
                if not np.allclose(
                    registered_mi,
                    score_values["mc_dropout_mutual_information"],
                    rtol=0.0,
                    atol=1.0e-7,
                ):
                    raise ValueError(
                        "MC-dropout registered score does not match its "
                        "pass-count sensitivity prefix"
                    )
                prediction = trained.classes[np.argmax(probabilities, axis=1)]
                context = {**base, "evaluation_partition": partition_name}
                metrics = common.classification_metrics(
                    trained.classes,
                    labels[index],
                    prediction,
                    probabilities,
                    trained.classes,
                )
                outputs["classification"].append(
                    {**context, **{k: v for k, v in metrics.items() if k != "supported_mask"}}
                )
                for offset, row_index in enumerate(index):
                    row = manifest.iloc[row_index]
                    outputs["predictions"].append(
                        {
                            **context,
                            "observation_uid": row.observation_uid,
                            "master_sample_id": row.master_sample_id,
                            "target_analyte": labels[row_index],
                            "predicted_analyte": prediction[offset],
                            "correct": prediction[offset] == labels[row_index],
                            "instrument": row.instrument,
                            "sensor_family": row.sensor_family,
                            "probability_classes_json": json.dumps(
                                trained.classes.tolist()
                            ),
                            "probabilities_json": json.dumps(
                                probabilities[offset].tolist()
                            ),
                            **{
                                name: float(score[offset])
                                for name, score in score_values.items()
                            },
                        }
                    )
                evaluated[partition_name] = (
                    index,
                    probabilities,
                    prediction,
                    score_values,
                    embedding,
                )
                for name, score in score_values.items():
                    for coverage, threshold in threshold_map[name].items():
                        accepted = score <= threshold
                        supported = np.isin(labels[index], trained.classes)
                        retained = accepted & supported
                        outputs["threshold_metrics"].append(
                            {
                                **context,
                                "score_name": name,
                                "score_selected": name == selected.score_name,
                                "requested_known_coverage": coverage,
                                "threshold": threshold,
                                "acceptance_rate": float(accepted.mean()),
                                "supported_accuracy_accepted": (
                                    float(np.mean(labels[index][retained] == prediction[retained]))
                                    if retained.any() else np.nan
                                ),
                            }
                        )
            for evaluation, (known_name, unknown_name) in {
                "quality_open": ("known_quality_test", "unknown_quality"),
                "stress_open": ("known_stress", "unknown_stress"),
                "field_quality": ("known_quality_test", "known_stress"),
            }.items():
                known, unknown = evaluated[known_name], evaluated[unknown_name]
                index = np.concatenate([known[0], unknown[0]])
                probability = np.vstack([known[1], unknown[1]])
                prediction = np.concatenate([known[2], unknown[2]])
                unknown_mask = np.concatenate(
                    [np.zeros(len(known[0]), bool), np.ones(len(unknown[0]), bool)]
                )
                for name in SCORES:
                    score = np.concatenate([known[3][name], unknown[3][name]])
                    metrics, _ = common.open_set_metrics(
                        manifest.iloc[index].reset_index(drop=True),
                        labels[index],
                        prediction,
                        probability,
                        trained.classes,
                        score,
                        unknown_mask,
                    )
                    outputs["open_metrics"].append(
                        {
                            **base,
                            "evaluation": evaluation,
                            "score_name": name,
                            "score_selected": name == selected.score_name,
                            "direction_control_auroc": 1.0 - metrics["master_unknown_auroc"],
                            **metrics,
                        }
                    )
                for pass_count in (10, int(registry["mc_dropout_passes"]), 50):
                    score = np.concatenate(
                        [
                            mc_sensitivity[known_name][pass_count][1],
                            mc_sensitivity[unknown_name][pass_count][1],
                        ]
                    )
                    metrics, _ = common.open_set_metrics(
                        manifest.iloc[index].reset_index(drop=True),
                        labels[index],
                        prediction,
                        probability,
                        trained.classes,
                        score,
                        unknown_mask,
                    )
                    outputs["mc_pass_sensitivity"].append(
                        {
                            **base,
                            "evaluation": evaluation,
                            "pass_count": pass_count,
                            **metrics,
                        }
                    )
            quality_index, _, quality_prediction, _, quality_embedding = (
                evaluated["known_quality_test"]
            )
            quality_manifest = manifest.iloc[quality_index].reset_index(
                drop=True
            )
            development_manifest = manifest.iloc[development].reset_index(
                drop=True
            )
            for domain_column in ("instrument", "sensor_family"):
                probe = common.baseline.target_adjusted_domain_probe(
                    development_embedding,
                    quality_embedding,
                    labels[development],
                    labels[quality_index],
                    development_manifest[domain_column].astype(str).to_numpy(),
                    quality_manifest[domain_column].astype(str).to_numpy(),
                    cnn.stable_seed(seed, "domain_probe", domain_column),
                )
                outputs["target_adjusted_domain_probes"].append(
                    {
                        **base,
                        "evaluation_partition": "known_quality_test",
                        "domain_type": domain_column,
                        "n_train": len(development),
                        "n_test": len(quality_index),
                        **probe,
                    }
                )
            for partition_name in ("known_quality_test", "known_stress"):
                index, _, prediction, _, _ = evaluated[partition_name]
                partition_manifest = manifest.iloc[index].reset_index(
                    drop=True
                )
                for domain_column in ("instrument", "sensor_family"):
                    agreement = same_master_prediction_agreement(
                        partition_manifest, prediction, domain_column
                    )
                    outputs["same_master_agreement"].append(
                        {
                            **base,
                            "evaluation_partition": partition_name,
                            "domain_type": domain_column,
                            **agreement,
                        }
                    )
            print(
                f"FINAL CNN H={held} O={outer} seed={declared_seed}", flush=True
            )
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(
            args.output_dir / f"final_{name}.csv", index=False
        )


def main() -> None:
    args = parse_args()
    initialize(args)
    registry = cnn.load_registry(args.registry)
    common.load_protocol(args.protocol)
    dataset = common.baseline.load_nato_dataset(args.bundle)
    device = torch.device(args.device)
    stages = (
        ("surrogate", "calibration", "final")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        print(f"START CNN {stage}", flush=True)
        if stage == "surrogate":
            run_surrogate(args, dataset, registry, device)
        elif stage == "select":
            merge_surrogate_shards(args)
        elif stage == "calibration":
            run_calibration(args, dataset, registry, device)
        else:
            run_final(args, dataset, registry, device)
        print(f"DONE CNN {stage}", flush=True)


if __name__ == "__main__":
    main()
