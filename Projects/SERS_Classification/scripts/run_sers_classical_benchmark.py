#!/usr/bin/env python3
"""Execute the predeclared NATO SERS classical benchmark v2."""

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

import sers_classical_benchmark_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/sers_classical_benchmark_v2.json"),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("Workspace/nato_sers_field_trial/preprocessing_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Workspace/sers_classical_benchmark/classical_benchmark_v2"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("search", "outer", "domain", "learning", "negative", "all"),
        default="all",
    )
    parser.add_argument("--jobs", type=int, default=4)
    return parser.parse_args()


def initialize(
    output_dir: Path,
    protocol_path: Path,
    bundle: Path,
    protocol: dict[str, Any],
    candidates: list[common.Candidate],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = output_dir / "predeclared_protocol.json"
    if copied.exists() and copied.read_bytes() != protocol_path.read_bytes():
        raise ValueError("Output directory has a different protocol snapshot")
    if not copied.exists():
        shutil.copyfile(protocol_path, copied)
    common.baseline.verify_hash_catalog(bundle)
    expected = protocol["immutable_input"]
    dataset = common.baseline.load_nato_dataset(bundle)
    if len(dataset.manifest) != int(expected["strict_core_rows"]):
        raise ValueError("Strict-core row count differs from protocol")
    if int(dataset.manifest["include_sers_qc_pass"].sum()) != int(
        expected["quality_pass_rows"]
    ):
        raise ValueError("Quality row count differs from protocol")
    if int(dataset.manifest["field_quality_stress"].sum()) != int(
        expected["field_quality_stress_rows"]
    ):
        raise ValueError("Stress row count differs from protocol")
    protocol_representations = list(expected["representations"])
    if set(protocol_representations) != set(dataset.representations):
        raise ValueError("Authorized representations differ from frozen input")
    scripts = [
        Path(__file__).resolve(),
        Path(common.__file__).resolve(),
        Path(common.baseline.__file__).resolve(),
    ]
    common.write_json(
        output_dir / "input_hashes.json",
        {
            "protocol": {
                "path": str(protocol_path.resolve()),
                "sha256": common.sha256_file(protocol_path),
            },
            "bundle_hash_catalog": {
                "path": str((bundle / "artifact_hashes.json").resolve()),
                "sha256": common.sha256_file(bundle / "artifact_hashes.json"),
            },
            "bundle_dataset_version": {
                "path": str((bundle / "dataset_version.json").resolve()),
                "sha256": common.sha256_file(bundle / "dataset_version.json"),
            },
            "scripts": {
                str(path): common.sha256_file(path) for path in scripts
            },
        },
    )
    common.write_json(
        output_dir / "environment.json",
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
    pd.DataFrame([candidate.row() for candidate in candidates]).to_csv(
        output_dir / "candidate_registry.csv", index=False
    )


def evaluate_candidate(
    candidate: common.Candidate,
    protocol: dict[str, Any],
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    subset: str,
    outer_fold: int,
    inner_fold: int,
) -> dict[str, Any]:
    seed = common.stable_seed(
        common.PROTOCOL_VERSION,
        "inner",
        subset,
        outer_fold,
        inner_fold,
        candidate.candidate_id,
    )
    context = {
        **candidate.row(),
        "subset": subset,
        "outer_fold": outer_fold,
        "inner_validation_fold": inner_fold,
        "seed": seed,
        "n_train": int(train_mask.sum()),
        "n_validation": int(validation_mask.sum()),
        "n_train_groups": int(pd.Series(groups[train_mask]).nunique()),
        "n_validation_groups": int(
            pd.Series(groups[validation_mask]).nunique()
        ),
    }
    try:
        estimator = common.fit_estimator(
            candidate,
            protocol,
            values[train_mask],
            labels[train_mask],
            seed,
        )
        classes = common.estimator_classes(estimator)
        scores = common.estimator_scores(estimator, values[validation_mask])
        probabilities = common.probabilities_from_scores(scores)
        predictions = classes[np.argmax(probabilities, axis=1)]
        metrics = common.classification_metrics(
            classes,
            labels[validation_mask],
            predictions,
            probabilities,
            classes,
        )
        return {
            **context,
            "status": "pass",
            "error": "",
            **{
                key: value
                for key, value in metrics.items()
                if key != "supported_mask"
            },
        }
    except Exception as exc:  # retained as an auditable failed candidate
        return {
            **context,
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "accuracy": np.nan,
            "negative_log_likelihood": np.nan,
            "brier_multiclass": np.nan,
            "expected_calibration_error_10": np.nan,
        }


def run_search(
    output_dir: Path,
    protocol: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    candidates: list[common.Candidate],
    jobs: int,
) -> None:
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    shard_dir = output_dir / "search_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for subset in ("strict_core", "quality_pass"):
        eligible = common.subset_mask(manifest, subset)
        for outer_fold in range(5):
            for inner_fold in sorted(set(folds[eligible]) - {outer_fold}):
                shard = (
                    shard_dir
                    / f"{subset}__outer_{outer_fold}__inner_{inner_fold}.csv"
                )
                if shard.exists():
                    prior = pd.read_csv(shard)
                    if len(prior) == len(candidates):
                        continue
                train_mask = (
                    eligible
                    & (folds != outer_fold)
                    & (folds != inner_fold)
                )
                validation_mask = eligible & (folds == inner_fold)
                tasks = (
                    joblib.delayed(evaluate_candidate)(
                        candidate,
                        protocol,
                        dataset.representations[candidate.representation],
                        labels,
                        groups,
                        train_mask,
                        validation_mask,
                        subset,
                        outer_fold,
                        int(inner_fold),
                    )
                    for candidate in candidates
                )
                rows = joblib.Parallel(n_jobs=jobs, verbose=5)(tasks)
                pd.DataFrame(rows).sort_values("candidate_order").to_csv(
                    shard, index=False
                )
    shards = [pd.read_csv(path) for path in sorted(shard_dir.glob("*.csv"))]
    metrics = pd.concat(shards, ignore_index=True)
    metrics.sort_values(
        ["subset", "outer_fold", "inner_validation_fold", "candidate_order"]
    ).to_csv(output_dir / "inner_fold_metrics.csv", index=False)
    select_candidates(output_dir, metrics, candidates)


def ranked_summary(
    frame: pd.DataFrame,
    required_folds: int,
) -> pd.DataFrame:
    summary = (
        frame[frame["status"] == "pass"]
        .groupby(
            [
                "candidate_order",
                "candidate_id",
                "representation",
                "model_family",
                "parameters_json",
            ],
            as_index=False,
        )
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            sd_balanced_accuracy=("balanced_accuracy", "std"),
            folds_evaluated=("balanced_accuracy", "count"),
        )
    )
    summary = summary[summary["folds_evaluated"] == required_folds].copy()
    return summary.sort_values(
        [
            "mean_balanced_accuracy",
            "mean_macro_f1",
            "candidate_order",
        ],
        ascending=[False, False, True],
        kind="stable",
    )


def select_candidates(
    output_dir: Path,
    metrics: pd.DataFrame,
    candidates: list[common.Candidate],
) -> None:
    outer_rows: list[dict[str, Any]] = []
    global_champions: dict[str, Any] = {}
    for subset in ("strict_core", "quality_pass"):
        subset_frame = metrics[metrics["subset"] == subset]
        for outer_fold in range(5):
            ranked = ranked_summary(
                subset_frame[subset_frame["outer_fold"] == outer_fold], 4
            )
            if ranked.empty:
                raise RuntimeError(
                    f"No eligible candidate for {subset}, outer {outer_fold}"
                )
            winner = ranked.iloc[0].to_dict()
            outer_rows.append(
                {
                    "subset": subset,
                    "outer_fold": outer_fold,
                    **winner,
                }
            )
        global_ranked = ranked_summary(subset_frame, 20)
        if global_ranked.empty:
            raise RuntimeError(f"No global champion for {subset}")
        winner = global_ranked.iloc[0].to_dict()
        global_champions[subset] = {
            **winner,
            "selection_source": "20 nested inner folds; no outer/stress/domain outcomes",
            "top_10": global_ranked.head(10).to_dict(orient="records"),
        }
    pd.DataFrame(outer_rows).sort_values(
        ["subset", "outer_fold"]
    ).to_csv(output_dir / "outer_selection.csv", index=False)
    common.write_json(output_dir / "global_champions.json", global_champions)


def crossfit_temperature(
    candidate: common.Candidate,
    protocol: dict[str, Any],
    values: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    development_mask: np.ndarray,
    identity: str,
) -> tuple[float, dict[str, Any]]:
    classes = np.unique(labels[development_mask])
    score_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for validation_fold in sorted(np.unique(folds[development_mask])):
        train_mask = development_mask & (folds != validation_fold)
        validation_mask = development_mask & (folds == validation_fold)
        if not np.any(validation_mask):
            continue
        if len(np.unique(labels[train_mask])) != len(classes):
            continue
        seed = common.stable_seed(
            common.PROTOCOL_VERSION,
            "calibration",
            identity,
            validation_fold,
            candidate.candidate_id,
        )
        estimator = common.fit_estimator(
            candidate,
            protocol,
            values[train_mask],
            labels[train_mask],
            seed,
        )
        scores = common.align_scores(
            common.estimator_scores(estimator, values[validation_mask]),
            common.estimator_classes(estimator),
            classes,
        )
        score_parts.append(scores)
        label_parts.append(labels[validation_mask])
    if not score_parts:
        return 1.0, {"n_calibration": 0, "calibration_folds": 0}
    scores = np.vstack(score_parts)
    calibration_labels = np.concatenate(label_parts)
    bounds = tuple(float(value) for value in protocol["calibration"]["temperature_bounds"])
    temperature = common.fit_temperature(
        scores, calibration_labels, classes, bounds  # type: ignore[arg-type]
    )
    raw = calibration_metric_summary(scores, calibration_labels, classes, 1.0)
    calibrated = calibration_metric_summary(
        scores, calibration_labels, classes, temperature
    )
    return temperature, {
        "n_calibration": len(calibration_labels),
        "calibration_folds": len(score_parts),
        "crossfit_raw_nll": raw["negative_log_likelihood"],
        "crossfit_calibrated_nll": calibrated["negative_log_likelihood"],
        "crossfit_raw_ece": raw["expected_calibration_error_10"],
        "crossfit_calibrated_ece": calibrated[
            "expected_calibration_error_10"
        ],
    }


def calibration_metric_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    temperature: float,
) -> dict[str, Any]:
    probabilities = common.probabilities_from_scores(scores, temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    return common.classification_metrics(
        classes, labels, predictions, probabilities, classes
    )


def evaluate_fitted(
    estimator: Any,
    values: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    classes = common.estimator_classes(estimator)
    scores = common.estimator_scores(estimator, values)
    probabilities = common.probabilities_from_scores(scores, temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metrics = common.classification_metrics(
        classes, labels, predictions, probabilities, classes
    )
    return predictions, probabilities, classes, metrics


def prediction_rows(
    context: dict[str, Any],
    manifest: pd.DataFrame,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    supported_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(len(manifest)):
        rows.append(
            {
                **context,
                "observation_uid": str(manifest.iloc[index]["observation_uid"]),
                "master_sample_id": str(
                    manifest.iloc[index]["master_sample_id"]
                ),
                "target_analyte": str(labels[index]),
                "predicted_analyte": str(predictions[index]),
                "correct": bool(predictions[index] == labels[index]),
                "supported": bool(supported_mask[index]),
                "confidence": float(np.max(probabilities[index])),
                "instrument": str(manifest.iloc[index]["instrument"]),
                "sensor_family": str(manifest.iloc[index]["sensor_family"]),
                "probability_classes_json": json.dumps(classes.tolist()),
                "probabilities_json": json.dumps(
                    probabilities[index].tolist()
                ),
            }
        )
    return rows


def metric_row(context: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        **context,
        **{key: value for key, value in metrics.items() if key != "supported_mask"},
    }


def calibration_rows(
    context: dict[str, Any],
    estimator: Any,
    values: np.ndarray,
    labels: np.ndarray,
    temperature: float,
    calibration_context: dict[str, Any],
) -> list[dict[str, Any]]:
    classes = common.estimator_classes(estimator)
    scores = common.estimator_scores(estimator, values)
    rows: list[dict[str, Any]] = []
    for name, applied_temperature in (
        ("uncalibrated", 1.0),
        ("temperature_scaled", temperature),
    ):
        metrics = calibration_metric_summary(
            scores, labels, classes, applied_temperature
        )
        rows.append(
            {
                **context,
                **calibration_context,
                "calibration": name,
                "temperature": float(applied_temperature),
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "supported_mask"
                },
            }
        )
    return rows


def run_outer(
    output_dir: Path,
    protocol: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    candidates: list[common.Candidate],
) -> None:
    lookup = common.candidate_lookup(candidates)
    selection = pd.read_csv(output_dir / "outer_selection.csv")
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    metric_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    selective: list[dict[str, Any]] = []
    for subset in ("strict_core", "quality_pass"):
        eligible = common.subset_mask(manifest, subset)
        for outer_fold in range(5):
            selected = selection[
                (selection["subset"] == subset)
                & (selection["outer_fold"] == outer_fold)
            ].iloc[0]
            candidate = lookup[str(selected["candidate_id"])]
            values = dataset.representations[candidate.representation]
            development_mask = eligible & (folds != outer_fold)
            test_masks = [(subset, eligible & (folds == outer_fold))]
            if subset == "quality_pass":
                test_masks.append(
                    (
                        "field_quality_stress",
                        common.subset_mask(manifest, "field_quality_stress")
                        & (folds == outer_fold),
                    )
                )
            temperature, calibration_context = crossfit_temperature(
                candidate,
                protocol,
                values,
                labels,
                folds,
                development_mask,
                f"{subset}__outer_{outer_fold}",
            )
            seed = common.stable_seed(
                common.PROTOCOL_VERSION,
                "outer_final",
                subset,
                outer_fold,
                candidate.candidate_id,
            )
            estimator = common.fit_estimator(
                candidate,
                protocol,
                values[development_mask],
                labels[development_mask],
                seed,
            )
            for evaluation_subset, test_mask in test_masks:
                test_manifest = manifest.loc[test_mask].reset_index(drop=True)
                test_labels = labels[test_mask]
                predicted, probabilities, classes, metrics = evaluate_fitted(
                    estimator,
                    values[test_mask],
                    test_labels,
                    temperature,
                )
                context = {
                    "stage": "outer",
                    "training_subset": subset,
                    "evaluation_subset": evaluation_subset,
                    "outer_fold": outer_fold,
                    "candidate_id": candidate.candidate_id,
                    "model_family": candidate.model_family,
                    "representation": candidate.representation,
                    "temperature": temperature,
                    "seed": seed,
                }
                metric_rows.append(metric_row(context, metrics))
                predictions.extend(
                    prediction_rows(
                        context,
                        test_manifest,
                        test_labels,
                        predicted,
                        probabilities,
                        classes,
                        metrics["supported_mask"],
                    )
                )
                per_class.extend(
                    common.per_class_rows(
                        context,
                        test_labels,
                        predicted,
                        metrics["supported_mask"],
                    )
                )
                selective.extend(
                    common.selective_rows(
                        context,
                        test_labels,
                        predicted,
                        np.max(probabilities, axis=1),
                        metrics["supported_mask"],
                        protocol["evaluation"]["selective_coverages"],
                    )
                )
                calibration.extend(
                    calibration_rows(
                        context,
                        estimator,
                        values[test_mask],
                        test_labels,
                        temperature,
                        calibration_context,
                    )
                )
    metrics_frame = pd.DataFrame(metric_rows)
    prediction_frame = pd.DataFrame(predictions)
    metrics_frame.to_csv(output_dir / "outer_metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(per_class).to_csv(
        output_dir / "per_class_metrics.csv", index=False
    )
    pd.DataFrame(calibration).to_csv(
        output_dir / "calibration_metrics.csv", index=False
    )
    pd.DataFrame(selective).to_csv(
        output_dir / "selective_metrics.csv", index=False
    )
    write_field_stress_ood(output_dir, prediction_frame)
    write_outer_domain_slices(output_dir, prediction_frame)


def write_field_stress_ood(
    output_dir: Path, predictions: pd.DataFrame
) -> None:
    rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        quality = predictions[
            (predictions["training_subset"] == "quality_pass")
            & (predictions["evaluation_subset"] == "quality_pass")
            & (predictions["outer_fold"] == outer_fold)
        ]
        stress = predictions[
            (predictions["training_subset"] == "quality_pass")
            & (predictions["evaluation_subset"] == "field_quality_stress")
            & (predictions["outer_fold"] == outer_fold)
        ]
        metrics = common.ood_metrics(
            1.0 - quality["confidence"].to_numpy(float),
            1.0 - stress["confidence"].to_numpy(float),
        )
        rows.append(
            {
                "outer_fold": outer_fold,
                "score": "one_minus_calibrated_max_probability",
                "n_quality": len(quality),
                "n_stress": len(stress),
                **metrics,
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "field_stress_ood_metrics.csv", index=False
    )


def write_outer_domain_slices(
    output_dir: Path, predictions: pd.DataFrame
) -> None:
    rows: list[dict[str, Any]] = []
    for keys, frame in predictions[predictions["supported"]].groupby(
        [
            "training_subset",
            "evaluation_subset",
            "instrument",
            "sensor_family",
        ],
        dropna=False,
    ):
        training_subset, evaluation_subset, instrument, sensor = keys
        rows.append(
            {
                "training_subset": training_subset,
                "evaluation_subset": evaluation_subset,
                "instrument": instrument,
                "sensor_family": sensor,
                "n": len(frame),
                "accuracy": float(frame["correct"].mean()),
                "mean_confidence": float(frame["confidence"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "outer_domain_slices.csv", index=False
    )


def load_global_champions(
    output_dir: Path,
    candidates: list[common.Candidate],
) -> dict[str, common.Candidate]:
    champions = json.loads((output_dir / "global_champions.json").read_text())
    lookup = common.candidate_lookup(candidates)
    return {
        subset: lookup[str(record["candidate_id"])]
        for subset, record in champions.items()
    }


def run_domain(
    output_dir: Path,
    bundle: Path,
    protocol: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    candidates: list[common.Candidate],
) -> None:
    champions = load_global_champions(output_dir, candidates)
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    uid_to_index = {
        uid: index
        for index, uid in enumerate(
            manifest["observation_uid"].astype(str).to_numpy()
        )
    }
    metric_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    selective: list[dict[str, Any]] = []
    for subset in ("strict_core", "quality_pass"):
        partition_path = bundle / (
            "domain_evaluation_partitions_core.csv"
            if subset == "strict_core"
            else "domain_evaluation_partitions_quality.csv"
        )
        partitions = pd.read_csv(partition_path)
        candidate = champions[subset]
        values = dataset.representations[candidate.representation]
        for (domain_protocol, domain_type, heldout), scenario in partitions.groupby(
            ["protocol", "domain_type", "heldout_domain"], sort=True
        ):
            train_uids = scenario.loc[
                scenario["partition"] == "train", "observation_uid"
            ].astype(str)
            test_uids = scenario.loc[
                scenario["partition"] == "test", "observation_uid"
            ].astype(str)
            train_indices = np.asarray(
                [uid_to_index[uid] for uid in train_uids], dtype=int
            )
            test_indices = np.asarray(
                [uid_to_index[uid] for uid in test_uids], dtype=int
            )
            development_mask = np.zeros(len(manifest), dtype=bool)
            development_mask[train_indices] = True
            temperature, calibration_context = crossfit_temperature(
                candidate,
                protocol,
                values,
                labels,
                folds,
                development_mask,
                f"{subset}__{domain_protocol}__{domain_type}__{heldout}",
            )
            seed = common.stable_seed(
                common.PROTOCOL_VERSION,
                "domain",
                subset,
                domain_protocol,
                domain_type,
                heldout,
                candidate.candidate_id,
            )
            estimator = common.fit_estimator(
                candidate,
                protocol,
                values[train_indices],
                labels[train_indices],
                seed,
            )
            predicted, probabilities, classes, metrics = evaluate_fitted(
                estimator,
                values[test_indices],
                labels[test_indices],
                temperature,
            )
            context = {
                "stage": "domain",
                "subset": subset,
                "protocol": domain_protocol,
                "domain_type": domain_type,
                "heldout_domain": heldout,
                "candidate_id": candidate.candidate_id,
                "model_family": candidate.model_family,
                "representation": candidate.representation,
                "temperature": temperature,
                "seed": seed,
                **calibration_context,
            }
            metric_rows.append(metric_row(context, metrics))
            test_manifest = manifest.iloc[test_indices].reset_index(drop=True)
            prediction_records.extend(
                prediction_rows(
                    context,
                    test_manifest,
                    labels[test_indices],
                    predicted,
                    probabilities,
                    classes,
                    metrics["supported_mask"],
                )
            )
            per_class.extend(
                common.per_class_rows(
                    context,
                    labels[test_indices],
                    predicted,
                    metrics["supported_mask"],
                )
            )
            selective.extend(
                common.selective_rows(
                    context,
                    labels[test_indices],
                    predicted,
                    np.max(probabilities, axis=1),
                    metrics["supported_mask"],
                    protocol["evaluation"]["selective_coverages"],
                )
            )
    pd.DataFrame(metric_rows).to_csv(
        output_dir / "domain_metrics.csv", index=False
    )
    pd.DataFrame(prediction_records).to_csv(
        output_dir / "domain_predictions.csv", index=False
    )
    pd.DataFrame(per_class).to_csv(
        output_dir / "domain_per_class_metrics.csv", index=False
    )
    pd.DataFrame(selective).to_csv(
        output_dir / "domain_selective_metrics.csv", index=False
    )


def stratified_group_sample(
    manifest: pd.DataFrame,
    eligible_indices: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    frame = manifest.iloc[eligible_indices][
        ["master_sample_id", "target_analyte"]
    ].drop_duplicates()
    if frame.groupby("master_sample_id")["target_analyte"].nunique().max() != 1:
        raise ValueError("A master sample maps to multiple analytes")
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for _, class_groups in frame.groupby("target_analyte", sort=True):
        group_ids = class_groups["master_sample_id"].astype(str).to_numpy()
        count = min(
            len(group_ids),
            max(1, int(np.ceil(float(fraction) * len(group_ids)))),
        )
        selected.extend(rng.choice(group_ids, size=count, replace=False))
    return np.asarray(selected, dtype=str)


def run_learning(
    output_dir: Path,
    protocol: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    candidates: list[common.Candidate],
) -> None:
    lookup = common.candidate_lookup(candidates)
    selection = pd.read_csv(output_dir / "outer_selection.csv")
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for subset in ("strict_core", "quality_pass"):
        eligible = common.subset_mask(manifest, subset)
        for outer_fold in range(5):
            selected = selection[
                (selection["subset"] == subset)
                & (selection["outer_fold"] == outer_fold)
            ].iloc[0]
            candidate = lookup[str(selected["candidate_id"])]
            values = dataset.representations[candidate.representation]
            development_indices = np.flatnonzero(
                eligible & (folds != outer_fold)
            )
            test_mask = eligible & (folds == outer_fold)
            for fraction in protocol["learning_curves"][
                "training_group_fractions"
            ]:
                for declared_seed in protocol["learning_curves"]["repeats"]:
                    seed = common.stable_seed(
                        common.PROTOCOL_VERSION,
                        "learning",
                        subset,
                        outer_fold,
                        fraction,
                        declared_seed,
                    )
                    selected_groups = stratified_group_sample(
                        manifest, development_indices, float(fraction), seed
                    )
                    train_mask = (
                        eligible
                        & (folds != outer_fold)
                        & np.isin(groups, selected_groups)
                    )
                    estimator = common.fit_estimator(
                        candidate,
                        protocol,
                        values[train_mask],
                        labels[train_mask],
                        seed,
                    )
                    _, _, _, metrics = evaluate_fitted(
                        estimator, values[test_mask], labels[test_mask], 1.0
                    )
                    rows.append(
                        {
                            "subset": subset,
                            "outer_fold": outer_fold,
                            "training_group_fraction": float(fraction),
                            "declared_seed": declared_seed,
                            "run_seed": seed,
                            "candidate_id": candidate.candidate_id,
                            "model_family": candidate.model_family,
                            "representation": candidate.representation,
                            "n_train": int(train_mask.sum()),
                            "n_train_groups": int(
                                pd.Series(groups[train_mask]).nunique()
                            ),
                            **{
                                key: value
                                for key, value in metrics.items()
                                if key != "supported_mask"
                            },
                        }
                    )
    pd.DataFrame(rows).to_csv(
        output_dir / "learning_curve_metrics.csv", index=False
    )


def run_negative(
    output_dir: Path,
    protocol: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    candidates: list[common.Candidate],
) -> None:
    candidate = load_global_champions(output_dir, candidates)["strict_core"]
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    values = dataset.representations[candidate.representation]
    rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        train_mask = folds != outer_fold
        test_mask = folds == outer_fold
        group_frame = (
            manifest.loc[train_mask, ["master_sample_id", "target_analyte"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        seed = common.stable_seed(
            common.PROTOCOL_VERSION,
            "negative",
            protocol["negative_control"]["seed"],
            outer_fold,
        )
        rng = np.random.default_rng(seed)
        permuted = rng.permutation(
            group_frame["target_analyte"].astype(str).to_numpy()
        )
        group_to_label = dict(
            zip(group_frame["master_sample_id"].astype(str), permuted)
        )
        permuted_train_labels = np.asarray(
            [group_to_label[group] for group in groups[train_mask]], dtype=str
        )
        estimator = common.fit_estimator(
            candidate,
            protocol,
            values[train_mask],
            permuted_train_labels,
            seed,
        )
        _, _, _, metrics = evaluate_fitted(
            estimator, values[test_mask], labels[test_mask], 1.0
        )
        rows.append(
            {
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
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "negative_control_metrics.csv", index=False
    )


def main() -> None:
    args = parse_args()
    protocol = common.load_protocol(args.protocol)
    candidates = common.candidate_grid(protocol)
    initialize(
        args.output_dir,
        args.protocol,
        args.bundle,
        protocol,
        candidates,
    )
    dataset = common.baseline.load_nato_dataset(args.bundle)
    stages = (
        ("search", "outer", "domain", "learning", "negative")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        print(f"START {stage}", flush=True)
        if stage == "search":
            run_search(
                args.output_dir,
                protocol,
                dataset,
                candidates,
                args.jobs,
            )
        elif stage == "outer":
            run_outer(args.output_dir, protocol, dataset, candidates)
        elif stage == "domain":
            run_domain(
                args.output_dir,
                args.bundle,
                protocol,
                dataset,
                candidates,
            )
        elif stage == "learning":
            run_learning(args.output_dir, protocol, dataset, candidates)
        elif stage == "negative":
            run_negative(args.output_dir, protocol, dataset, candidates)
        print(f"DONE {stage}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

