#!/usr/bin/env python3
"""Run leakage-safe classical open-world baselines on frozen NATO partitions."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

import sers_open_world_common as common


SCORES = (
    "one_minus_calibrated_max_probability",
    "predictive_entropy",
    "minimum_class_mahalanobis_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/sers_open_world_shift_v1.json"),
    )
    parser.add_argument(
        "--baseline-registry",
        type=Path,
        default=Path("configs/sers_open_world_baselines_v1.json"),
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
        default=Path(
            "Workspace/sers_open_world_shift/open_world_v1/baselines"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("surrogate", "outer", "negative", "all"),
        default="all",
    )
    parser.add_argument("--jobs", type=int, default=16)
    return parser.parse_args()


def initialize(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    registry: dict[str, Any],
    candidates: list[common.Candidate],
) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "predeclared_protocol.json": args.protocol,
        "predeclared_baseline_registry.json": args.baseline_registry,
    }
    for filename, source in snapshots.items():
        destination = output / filename
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Different immutable snapshot: {filename}")
        if not destination.exists():
            shutil.copyfile(source, destination)
    partition_validation = json.loads(
        (args.partition_dir / "validation_report.json").read_text()
    )
    if partition_validation.get("status") != "pass":
        raise ValueError("Frozen partitions have not passed validation")
    dataset = common.baseline.load_nato_dataset(args.bundle)
    if len(dataset.manifest) != int(
        protocol["immutable_input"]["strict_core_rows"]
    ):
        raise ValueError("Dataset row count changed")
    pd.DataFrame([candidate.row() for candidate in candidates]).to_csv(
        output / "candidate_registry.csv", index=False
    )
    common.write_json(
        output / "input_hashes.json",
        {
            "protocol": {
                "path": str(args.protocol.resolve()),
                "sha256": common.sha256_file(args.protocol),
            },
            "baseline_registry": {
                "path": str(args.baseline_registry.resolve()),
                "sha256": common.sha256_file(args.baseline_registry),
            },
            "partition_artifacts": {
                filename: common.sha256_file(args.partition_dir / filename)
                for filename in (
                    "artifact_hashes.json",
                    "validation_report.json",
                    "open_set_outer_partitions.csv",
                    "known_threshold_crossfit_partitions.csv",
                    "surrogate_unknown_inner_partitions.csv",
                )
            },
            "dataset_hash_catalog": common.sha256_file(
                args.bundle / "artifact_hashes.json"
            ),
            "scripts": {
                str(Path(__file__).resolve()): common.sha256_file(
                    Path(__file__).resolve()
                ),
                str(Path(common.__file__).resolve()): common.sha256_file(
                    Path(common.__file__).resolve()
                ),
            },
        },
    )
    common.write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    )


def indices_by_partition(
    scenario: pd.DataFrame, uid_to_index: dict[str, int]
) -> dict[str, np.ndarray]:
    return {
        str(partition): np.asarray(
            [
                uid_to_index[uid]
                for uid in group["observation_uid"].astype(str)
            ],
            dtype=int,
        )
        for partition, group in scenario.groupby("partition", sort=False)
    }


def surrogate_task(
    context: dict[str, Any],
    partitions: dict[str, np.ndarray],
    candidate: common.Candidate,
    dataset: common.baseline.SpectralDataset,
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    values = dataset.representations[candidate.representation]
    train = partitions["train_known"]
    known = partitions["validation_known"]
    unknown = partitions["validation_surrogate_unknown"]
    seed = common.stable_seed(
        common.PROTOCOL_VERSION,
        common.BASELINE_REGISTRY_VERSION,
        "surrogate",
        *context.values(),
        candidate.candidate_id,
        registry["selection_seed"],
    )
    base_context = {
        **context,
        **candidate.row(),
        "seed": seed,
        "n_train": len(train),
        "n_train_masters": len(np.unique(groups[train])),
        "n_validation_known": len(known),
        "n_validation_unknown": len(unknown),
    }
    try:
        fitted = common.fit_baseline(
            candidate,
            values[train],
            labels[train],
            groups[train],
            seed,
            registry,
        )
        # The validation-known fold participates in candidate/score selection,
        # so it must not also fit its own calibration parameter. Final outer
        # evaluation instead uses development-only cross-fitted temperature.
        temperature = float(
            registry["surrogate_selection"]["fixed_temperature"]
        )
        evaluation = np.concatenate([known, unknown])
        _, probabilities, scores = common.anomaly_scores(
            fitted, values[evaluation], temperature
        )
        predictions = fitted.classes[np.argmax(probabilities, axis=1)]
        unknown_mask = np.concatenate(
            [
                np.zeros(len(known), dtype=bool),
                np.ones(len(unknown), dtype=bool),
            ]
        )
        evaluation_manifest = manifest.iloc[evaluation].reset_index(drop=True)
        rows: list[dict[str, Any]] = []
        for score_order, score_name in enumerate(SCORES):
            metrics, _ = common.open_set_metrics(
                evaluation_manifest,
                labels[evaluation],
                predictions,
                probabilities,
                fitted.classes,
                scores[score_name],
                unknown_mask,
            )
            rows.append(
                {
                    **base_context,
                    "score_order": score_order,
                    "score_name": score_name,
                    "temperature": temperature,
                    "temperature_source": (
                        "predeclared_identity_not_validation_fitted"
                    ),
                    "status": "pass",
                    "error": "",
                    **metrics,
                }
            )
        return rows
    except Exception as exc:
        return [
            {
                **base_context,
                "score_order": score_order,
                "score_name": score_name,
                "temperature": np.nan,
                "temperature_source": (
                    "predeclared_identity_not_validation_fitted"
                ),
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            }
            for score_order, score_name in enumerate(SCORES)
        ]


def select_surrogate_winners(
    output_dir: Path, metrics: pd.DataFrame
) -> None:
    passed = metrics[metrics["status"] == "pass"].copy()
    summary = (
        passed.groupby(
            [
                "held_unknown",
                "outer_fold",
                "candidate_order",
                "candidate_id",
                "model_family",
                "representation",
                "parameters_json",
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
            sd_master_oscr_auc=("master_oscr_auc", "std"),
            scenarios_completed=("master_oscr_auc", "count"),
        )
    )
    summary.to_csv(output_dir / "surrogate_candidate_summary.csv", index=False)
    eligible = summary[summary["scenarios_completed"] == 20].copy()
    winners: list[dict[str, Any]] = []
    for (held_unknown, outer_fold), frame in eligible.groupby(
        ["held_unknown", "outer_fold"], sort=True
    ):
        ranked = frame.sort_values(
            [
                "mean_master_oscr_auc",
                "mean_master_known_balanced_accuracy",
                "mean_master_unknown_auroc",
                "candidate_order",
                "score_order",
            ],
            ascending=[False, False, False, True, True],
            kind="stable",
        )
        if ranked.empty:
            raise RuntimeError(
                f"No surrogate winner for {held_unknown}, outer {outer_fold}"
            )
        winner = ranked.iloc[0].to_dict()
        winners.append(
            {
                **winner,
                "selection_source": (
                    "20 development-only surrogate scenarios; true held "
                    "chemical, stress, outer known test, and domains unused"
                ),
            }
        )
    if len(winners) != 30:
        raise RuntimeError(f"Expected 30 winners, got {len(winners)}")
    pd.DataFrame(winners).sort_values(
        ["held_unknown", "outer_fold"]
    ).to_csv(output_dir / "surrogate_selection.csv", index=False)


def run_surrogate(
    args: argparse.Namespace,
    dataset: common.baseline.SpectralDataset,
    registry: dict[str, Any],
    candidates: list[common.Candidate],
) -> None:
    partitions = pd.read_csv(
        args.partition_dir / "surrogate_unknown_inner_partitions.csv"
    )
    uid_to_index = {
        uid: index
        for index, uid in enumerate(
            dataset.manifest["observation_uid"].astype(str).to_numpy()
        )
    }
    scenarios: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    columns = [
        "held_unknown",
        "outer_fold",
        "inner_fold",
        "surrogate_unknown",
    ]
    for keys, scenario in partitions.groupby(columns, sort=True):
        context = dict(zip(columns, keys))
        scenarios.append(
            (context, indices_by_partition(scenario, uid_to_index))
        )
    tasks = (
        joblib.delayed(surrogate_task)(
            context, partition_indices, candidate, dataset, registry
        )
        for context, partition_indices in scenarios
        for candidate in candidates
    )
    nested = joblib.Parallel(n_jobs=args.jobs, verbose=5)(tasks)
    metrics = pd.DataFrame([row for rows in nested for row in rows])
    metrics.sort_values(
        [
            "held_unknown",
            "outer_fold",
            "inner_fold",
            "surrogate_unknown",
            "candidate_order",
            "score_order",
        ],
        kind="stable",
    ).to_csv(args.output_dir / "surrogate_inner_metrics.csv", index=False)
    select_surrogate_winners(args.output_dir, metrics)


def crossfit_known_calibration(
    held_unknown: str,
    outer_fold: int,
    candidate: common.Candidate,
    final_seed: int,
    threshold_scenarios: list[dict[str, np.ndarray]],
    dataset: common.baseline.SpectralDataset,
    registry: dict[str, Any],
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    values = dataset.representations[candidate.representation]
    target_classes = np.asarray(
        sorted(set(labels) - {held_unknown}), dtype=str
    )
    logits_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    mahalanobis_parts: list[np.ndarray] = []
    for calibration_fold, partitions in enumerate(threshold_scenarios):
        train = partitions["train_known"]
        calibration = partitions["calibration_known"]
        seed = common.stable_seed(
            common.PROTOCOL_VERSION,
            "outer_calibration",
            held_unknown,
            outer_fold,
            calibration_fold,
            candidate.candidate_id,
            final_seed,
        )
        fitted = common.fit_baseline(
            candidate,
            values[train],
            labels[train],
            groups[train],
            seed,
            registry,
        )
        logits_parts.append(
            common.align_scores(
                fitted.logits(values[calibration]),
                fitted.classes,
                target_classes,
            )
        )
        mahalanobis_parts.append(fitted.mahalanobis(values[calibration]))
        label_parts.append(labels[calibration])
        index_parts.append(calibration)
    logits = np.vstack(logits_parts)
    calibration_labels = np.concatenate(label_parts)
    indices = np.concatenate(index_parts)
    order = np.argsort(indices, kind="stable")
    logits = logits[order]
    calibration_labels = calibration_labels[order]
    indices = indices[order]
    mahalanobis = np.concatenate(mahalanobis_parts)[order]
    bounds = tuple(float(x) for x in registry["calibration"]["temperature_bounds"])
    temperature = common.fit_temperature(
        logits,
        calibration_labels,
        target_classes,
        bounds,  # type: ignore[arg-type]
    )
    probabilities = common.probabilities_from_scores(logits, temperature)
    score_values = {
        "one_minus_calibrated_max_probability": 1.0
        - probabilities.max(axis=1),
        "predictive_entropy": common.predictive_entropy(probabilities),
        "minimum_class_mahalanobis_distance": mahalanobis,
    }
    return (
        temperature,
        indices,
        calibration_labels,
        score_values,
        target_classes,
        logits,
    )


def prediction_records(
    context: dict[str, Any],
    manifest: pd.DataFrame,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    score_values: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    manifest = manifest.reset_index(drop=True)
    for index, row in manifest.iterrows():
        records.append(
            {
                **context,
                "observation_uid": str(row["observation_uid"]),
                "master_sample_id": str(row["master_sample_id"]),
                "target_analyte": str(labels[index]),
                "predicted_analyte": str(predictions[index]),
                "correct": bool(predictions[index] == labels[index]),
                "instrument": str(row["instrument"]),
                "sensor_family": str(row["sensor_family"]),
                "probability_classes_json": json.dumps(classes.tolist()),
                "probabilities_json": json.dumps(
                    probabilities[index].tolist()
                ),
                **{
                    score_name: float(values[index])
                    for score_name, values in score_values.items()
                },
            }
        )
    return records


def classification_row(
    context: dict[str, Any],
    classes: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    metrics = common.classification_metrics(
        classes, labels, predictions, probabilities, classes
    )
    return {
        **context,
        **{
            key: value
            for key, value in metrics.items()
            if key != "supported_mask"
        },
    }


def outer_task(
    selection: dict[str, Any],
    final_seed: int,
    outer_partitions: dict[str, np.ndarray],
    threshold_scenarios: list[dict[str, np.ndarray]],
    candidate: common.Candidate,
    dataset: common.baseline.SpectralDataset,
    protocol: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    held_unknown = str(selection["held_unknown"])
    outer_fold = int(selection["outer_fold"])
    selected_score = str(selection["score_name"])
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    values = dataset.representations[candidate.representation]
    (
        temperature,
        calibration_indices,
        calibration_labels,
        calibration_scores,
        classes,
        calibration_logits,
    ) = crossfit_known_calibration(
            held_unknown,
            outer_fold,
            candidate,
            final_seed,
            threshold_scenarios,
            dataset,
            registry,
        )
    coverages = [
        float(value)
        for value in protocol["threshold_policy"]["known_retained_coverages"]
    ]
    thresholds = {
        score_name: {
            coverage: common.threshold_from_known(score_values, coverage)
            for coverage in coverages
        }
        for score_name, score_values in calibration_scores.items()
    }
    seed = common.stable_seed(
        common.PROTOCOL_VERSION,
        "outer_final",
        held_unknown,
        outer_fold,
        candidate.candidate_id,
        final_seed,
    )
    development = outer_partitions["known_development_quality"]
    fitted = common.fit_baseline(
        candidate,
        values[development],
        labels[development],
        groups[development],
        seed,
        registry,
    )
    base_context = {
        "held_unknown": held_unknown,
        "outer_fold": outer_fold,
        "candidate_id": candidate.candidate_id,
        "candidate_order": candidate.order,
        "model_family": candidate.model_family,
        "representation": candidate.representation,
        "selected_score": selected_score,
        "final_seed": final_seed,
        "seed": seed,
        "temperature": temperature,
        "n_development": len(development),
        "n_development_masters": len(np.unique(groups[development])),
        "n_calibration": len(calibration_indices),
    }
    outputs: dict[str, list[dict[str, Any]]] = {
        "predictions": [],
        "classification": [],
        "open_metrics": [],
        "thresholds": [],
        "threshold_metrics": [],
        "calibration": [],
    }
    evaluated: dict[str, dict[str, Any]] = {}
    for partition_name in (
        "known_quality_test",
        "unknown_quality",
        "known_stress",
        "unknown_stress",
    ):
        indices = outer_partitions[partition_name]
        logits = common.align_scores(
            fitted.logits(values[indices]), fitted.classes, classes
        )
        probabilities = common.probabilities_from_scores(logits, temperature)
        predictions = classes[np.argmax(probabilities, axis=1)]
        score_values = {
            "one_minus_calibrated_max_probability": 1.0
            - probabilities.max(axis=1),
            "predictive_entropy": common.predictive_entropy(probabilities),
            "minimum_class_mahalanobis_distance": fitted.mahalanobis(
                values[indices]
            ),
        }
        partition_context = {
            **base_context,
            "evaluation_partition": partition_name,
        }
        outputs["predictions"].extend(
            prediction_records(
                partition_context,
                manifest.iloc[indices],
                labels[indices],
                predictions,
                probabilities,
                classes,
                score_values,
            )
        )
        outputs["classification"].append(
            classification_row(
                partition_context,
                classes,
                labels[indices],
                predictions,
                probabilities,
            )
        )
        evaluated[partition_name] = {
            "indices": indices,
            "probabilities": probabilities,
            "predictions": predictions,
            "scores": score_values,
        }
        for score_name in SCORES:
            selected = score_name == selected_score
            for coverage, threshold in thresholds[score_name].items():
                scores = score_values[score_name]
                accepted = scores <= threshold
                supported = np.isin(labels[indices], classes)
                retained = accepted & supported
                outputs["threshold_metrics"].append(
                    {
                        **partition_context,
                        "score_name": score_name,
                        "score_selected": selected,
                        "requested_known_coverage": coverage,
                        "threshold": threshold,
                        "n_rows": len(indices),
                        "n_accepted": int(accepted.sum()),
                        "acceptance_rate": float(accepted.mean()),
                        "supported_accuracy_accepted": (
                            float(
                                np.mean(
                                    labels[indices][retained]
                                    == predictions[retained]
                                )
                            )
                            if retained.any()
                            else np.nan
                        ),
                    }
                )
    for score_name in SCORES:
        for coverage, threshold in thresholds[score_name].items():
            calibration_acceptance = float(
                np.mean(calibration_scores[score_name] <= threshold)
            )
            outputs["thresholds"].append(
                {
                    **base_context,
                    "score_name": score_name,
                    "score_selected": score_name == selected_score,
                    "requested_known_coverage": coverage,
                    "threshold": threshold,
                    "calibration_known_acceptance": calibration_acceptance,
                }
            )
    for calibration_name, applied_temperature in (
        ("uncalibrated", 1.0),
        ("temperature_scaled", temperature),
    ):
        calibration_probabilities = common.probabilities_from_scores(
            calibration_logits, applied_temperature
        )
        calibration_predictions = classes[
            np.argmax(calibration_probabilities, axis=1)
        ]
        calibration_metrics = common.classification_metrics(
            classes,
            calibration_labels,
            calibration_predictions,
            calibration_probabilities,
            classes,
        )
        outputs["calibration"].append(
            {
                **base_context,
                "calibration": calibration_name,
                "applied_temperature": applied_temperature,
                **{
                    key: value
                    for key, value in calibration_metrics.items()
                    if key != "supported_mask"
                },
            }
        )

    pair_definitions = {
        "quality_open": ("known_quality_test", "unknown_quality"),
        "stress_open": ("known_stress", "unknown_stress"),
        "field_quality": ("known_quality_test", "known_stress"),
    }
    for evaluation_name, (known_name, unknown_name) in pair_definitions.items():
        known_result = evaluated[known_name]
        unknown_result = evaluated[unknown_name]
        pair_indices = np.concatenate(
            [known_result["indices"], unknown_result["indices"]]
        )
        pair_manifest = manifest.iloc[pair_indices].reset_index(drop=True)
        pair_probabilities = np.vstack(
            [
                known_result["probabilities"],
                unknown_result["probabilities"],
            ]
        )
        pair_predictions = np.concatenate(
            [known_result["predictions"], unknown_result["predictions"]]
        )
        unknown_mask = np.concatenate(
            [
                np.zeros(len(known_result["indices"]), dtype=bool),
                np.ones(len(unknown_result["indices"]), dtype=bool),
            ]
        )
        for score_name in SCORES:
            pair_scores = np.concatenate(
                [
                    known_result["scores"][score_name],
                    unknown_result["scores"][score_name],
                ]
            )
            metrics, _ = common.open_set_metrics(
                pair_manifest,
                labels[pair_indices],
                pair_predictions,
                pair_probabilities,
                classes,
                pair_scores,
                unknown_mask,
            )
            outputs["open_metrics"].append(
                {
                    **base_context,
                    "evaluation": evaluation_name,
                    "known_partition": known_name,
                    "unknown_partition": unknown_name,
                    "score_name": score_name,
                    "score_selected": score_name == selected_score,
                    "direction_control_auroc": (
                        1.0 - metrics["master_unknown_auroc"]
                        if np.isfinite(metrics["master_unknown_auroc"])
                        else np.nan
                    ),
                    **metrics,
                }
            )
    return outputs


def run_outer(
    args: argparse.Namespace,
    dataset: common.baseline.SpectralDataset,
    protocol: dict[str, Any],
    registry: dict[str, Any],
    candidates: list[common.Candidate],
) -> None:
    selection = pd.read_csv(args.output_dir / "surrogate_selection.csv")
    lookup = {candidate.candidate_id: candidate for candidate in candidates}
    outer_frame = pd.read_csv(
        args.partition_dir / "open_set_outer_partitions.csv"
    )
    threshold_frame = pd.read_csv(
        args.partition_dir / "known_threshold_crossfit_partitions.csv"
    )
    uid_to_index = {
        uid: index
        for index, uid in enumerate(
            dataset.manifest["observation_uid"].astype(str).to_numpy()
        )
    }
    tasks = []
    for _, selected in selection.iterrows():
        held_unknown = str(selected["held_unknown"])
        outer_fold = int(selected["outer_fold"])
        candidate = lookup[str(selected["candidate_id"])]
        outer_scenario = outer_frame[
            (outer_frame["held_unknown"] == held_unknown)
            & (outer_frame["outer_fold"] == outer_fold)
        ]
        outer_partitions = indices_by_partition(
            outer_scenario, uid_to_index
        )
        threshold_scenarios: list[dict[str, np.ndarray]] = []
        for calibration_fold in sorted(
            set(range(5)) - {outer_fold}
        ):
            scenario = threshold_frame[
                (threshold_frame["held_unknown"] == held_unknown)
                & (threshold_frame["outer_fold"] == outer_fold)
                & (
                    threshold_frame["calibration_fold"]
                    == calibration_fold
                )
            ]
            threshold_scenarios.append(
                indices_by_partition(scenario, uid_to_index)
            )
        seeds = (
            registry["final_seeds"]
            if candidate.model_family == "random_forest"
            else [registry["selection_seed"]]
        )
        selection_record = selected.to_dict()
        for final_seed in seeds:
            tasks.append(
                joblib.delayed(outer_task)(
                    selection_record,
                    int(final_seed),
                    outer_partitions,
                    threshold_scenarios,
                    candidate,
                    dataset,
                    protocol,
                    registry,
                )
            )
    results = joblib.Parallel(n_jobs=args.jobs, verbose=5)(tasks)
    for output_name in (
        "predictions",
        "classification",
        "open_metrics",
        "thresholds",
        "threshold_metrics",
        "calibration",
    ):
        rows = [
            row for result in results for row in result[output_name]
        ]
        pd.DataFrame(rows).to_csv(
            args.output_dir / f"outer_{output_name}.csv", index=False
        )


def negative_task(
    selection: dict[str, Any],
    candidate: common.Candidate,
    outer_partitions: dict[str, np.ndarray],
    dataset: common.baseline.SpectralDataset,
    registry: dict[str, Any],
) -> dict[str, Any]:
    held_unknown = str(selection["held_unknown"])
    outer_fold = int(selection["outer_fold"])
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    values = dataset.representations[candidate.representation]
    train = outer_partitions["known_development_quality"]
    test = outer_partitions["known_quality_test"]
    master_labels = (
        manifest.iloc[train][["master_sample_id", "target_analyte"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    seed = common.stable_seed(
        common.PROTOCOL_VERSION,
        "master_label_permutation",
        held_unknown,
        outer_fold,
        candidate.candidate_id,
    )
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(
        master_labels["target_analyte"].astype(str).to_numpy()
    )
    mapping = dict(
        zip(master_labels["master_sample_id"].astype(str), permuted)
    )
    train_labels = np.asarray(
        [mapping[str(group)] for group in groups[train]], dtype=str
    )
    fitted = common.fit_baseline(
        candidate,
        values[train],
        train_labels,
        groups[train],
        seed,
        registry,
    )
    logits, probabilities, _ = common.anomaly_scores(fitted, values[test])
    del logits
    predictions = fitted.classes[np.argmax(probabilities, axis=1)]
    metrics = common.classification_metrics(
        fitted.classes,
        labels[test],
        predictions,
        probabilities,
        fitted.classes,
    )
    return {
        "held_unknown": held_unknown,
        "outer_fold": outer_fold,
        "candidate_id": candidate.candidate_id,
        "model_family": candidate.model_family,
        "representation": candidate.representation,
        "seed": seed,
        **{
            key: value
            for key, value in metrics.items()
            if key != "supported_mask"
        },
    }


def run_negative(
    args: argparse.Namespace,
    dataset: common.baseline.SpectralDataset,
    registry: dict[str, Any],
    candidates: list[common.Candidate],
) -> None:
    selection = pd.read_csv(args.output_dir / "surrogate_selection.csv")
    lookup = {candidate.candidate_id: candidate for candidate in candidates}
    outer_frame = pd.read_csv(
        args.partition_dir / "open_set_outer_partitions.csv"
    )
    uid_to_index = {
        uid: index
        for index, uid in enumerate(
            dataset.manifest["observation_uid"].astype(str).to_numpy()
        )
    }
    tasks = []
    for _, selected in selection.iterrows():
        held_unknown = str(selected["held_unknown"])
        outer_fold = int(selected["outer_fold"])
        scenario = outer_frame[
            (outer_frame["held_unknown"] == held_unknown)
            & (outer_frame["outer_fold"] == outer_fold)
        ]
        tasks.append(
            joblib.delayed(negative_task)(
                selected.to_dict(),
                lookup[str(selected["candidate_id"])],
                indices_by_partition(scenario, uid_to_index),
                dataset,
                registry,
            )
        )
    rows = joblib.Parallel(n_jobs=args.jobs, verbose=5)(tasks)
    pd.DataFrame(rows).to_csv(
        args.output_dir / "negative_control_metrics.csv", index=False
    )


def main() -> None:
    args = parse_args()
    protocol = common.load_protocol(args.protocol)
    registry = common.load_baseline_registry(args.baseline_registry)
    candidates = common.candidates_from_registry(registry)
    initialize(args, protocol, registry, candidates)
    dataset = common.baseline.load_nato_dataset(args.bundle)
    stages = (
        ("surrogate", "outer", "negative")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        print(f"START {stage}", flush=True)
        if stage == "surrogate":
            run_surrogate(args, dataset, registry, candidates)
        elif stage == "outer":
            run_outer(args, dataset, protocol, registry, candidates)
        elif stage == "negative":
            run_negative(args, dataset, registry, candidates)
        print(f"DONE {stage}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
