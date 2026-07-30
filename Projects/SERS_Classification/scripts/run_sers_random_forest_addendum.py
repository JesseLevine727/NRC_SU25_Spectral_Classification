#!/usr/bin/env python3
"""Run the predeclared leakage-safe NATO SERS random-forest addendum."""

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
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

import sers_random_forest_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/sers_random_forest_addendum_v1.json"),
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
            "Workspace/sers_random_forest_addendum/random_forest_v1"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("search", "outer", "domain", "negative", "all"),
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
    snapshot = output_dir / "predeclared_protocol.json"
    if snapshot.exists() and snapshot.read_bytes() != protocol_path.read_bytes():
        raise ValueError("Output directory contains another protocol snapshot")
    if not snapshot.exists():
        shutil.copyfile(protocol_path, snapshot)
    common.baseline.verify_hash_catalog(bundle)
    dataset = common.baseline.load_nato_dataset(bundle)
    expected = protocol["immutable_input"]
    manifest = dataset.manifest
    checks = {
        "strict_core_rows": len(manifest),
        "quality_pass_rows": int(manifest["include_sers_qc_pass"].sum()),
        "field_quality_stress_rows": int(manifest["field_quality_stress"].sum()),
    }
    for key, actual in checks.items():
        if actual != int(expected[key]):
            raise ValueError(f"{key}: expected {expected[key]}, got {actual}")
    if set(expected["representations"]) != set(dataset.representations):
        raise ValueError("Frozen representations do not match protocol")
    pd.DataFrame([candidate.row() for candidate in candidates]).to_csv(
        output_dir / "candidate_registry.csv", index=False
    )
    scripts = [Path(__file__).resolve(), Path(common.__file__).resolve()]
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


def metric_values(
    estimator: Any,
    x: np.ndarray,
    labels: np.ndarray,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    classes = np.asarray(estimator.classes_, dtype=str)
    scores = common.estimator_scores(estimator, x)
    probabilities = common.probabilities_from_scores(scores, temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metrics = common.classification_metrics(
        classes, labels, predictions, probabilities, classes
    )
    return predictions, probabilities, classes, metrics


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
            groups[train_mask],
            seed,
        )
        _, _, _, metrics = metric_values(
            estimator, values[validation_mask], labels[validation_mask]
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
    except Exception as exc:
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


def ranked_summary(frame: pd.DataFrame, required_folds: int) -> pd.DataFrame:
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
    return summary[summary["folds_evaluated"] == required_folds].sort_values(
        ["mean_balanced_accuracy", "mean_macro_f1", "candidate_order"],
        ascending=[False, False, True],
        kind="stable",
    )


def select_candidates(
    output_dir: Path,
    metrics: pd.DataFrame,
) -> None:
    outer_rows: list[dict[str, Any]] = []
    champions: dict[str, Any] = {}
    for subset in ("strict_core", "quality_pass"):
        subset_frame = metrics[metrics["subset"] == subset]
        for outer_fold in range(5):
            ranked = ranked_summary(
                subset_frame[subset_frame["outer_fold"] == outer_fold], 4
            )
            if ranked.empty:
                raise RuntimeError(f"No eligible {subset} outer-fold winner")
            outer_rows.append(
                {
                    "subset": subset,
                    "outer_fold": outer_fold,
                    **ranked.iloc[0].to_dict(),
                }
            )
        ranked = ranked_summary(subset_frame, 20)
        if ranked.empty:
            raise RuntimeError(f"No eligible global {subset} winner")
        champions[subset] = {
            **ranked.iloc[0].to_dict(),
            "selection_source": (
                "20 nested inner results; no outer, stress, or domain outcome"
            ),
            "top_10": ranked.head(10).to_dict(orient="records"),
        }
    pd.DataFrame(outer_rows).sort_values(
        ["subset", "outer_fold"]
    ).to_csv(output_dir / "outer_selection.csv", index=False)
    common.write_json(output_dir / "global_champions.json", champions)


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
                if shard.exists() and len(pd.read_csv(shard)) == len(candidates):
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
                rows = joblib.Parallel(n_jobs=jobs, verbose=3)(tasks)
                pd.DataFrame(rows).sort_values("candidate_order").to_csv(
                    shard, index=False
                )
    metrics = pd.concat(
        [pd.read_csv(path) for path in sorted(shard_dir.glob("*.csv"))],
        ignore_index=True,
    ).sort_values(
        ["subset", "outer_fold", "inner_validation_fold", "candidate_order"]
    )
    metrics.to_csv(output_dir / "inner_fold_metrics.csv", index=False)
    select_candidates(output_dir, metrics)


def crossfit_temperature(
    candidate: common.Candidate,
    protocol: dict[str, Any],
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    development_mask: np.ndarray,
    identity: str,
    final_seed: int,
) -> tuple[float, dict[str, Any]]:
    classes = np.unique(labels[development_mask])
    scores: list[np.ndarray] = []
    calibration_labels: list[np.ndarray] = []
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
            final_seed,
            validation_fold,
        )
        estimator = common.fit_estimator(
            candidate,
            protocol,
            values[train_mask],
            labels[train_mask],
            groups[train_mask],
            seed,
        )
        scores.append(
            common.align_scores(
                common.estimator_scores(estimator, values[validation_mask]),
                estimator.classes_,
                classes,
            )
        )
        calibration_labels.append(labels[validation_mask])
    if not scores:
        return 1.0, {"n_calibration": 0, "calibration_folds": 0}
    all_scores = np.vstack(scores)
    all_labels = np.concatenate(calibration_labels)
    bounds = tuple(
        float(value) for value in protocol["calibration"]["temperature_bounds"]
    )
    temperature = common.fit_temperature(
        all_scores, all_labels, classes, bounds  # type: ignore[arg-type]
    )
    raw_probabilities = common.probabilities_from_scores(all_scores)
    scaled_probabilities = common.probabilities_from_scores(
        all_scores, temperature
    )
    raw_predictions = classes[np.argmax(raw_probabilities, axis=1)]
    scaled_predictions = classes[np.argmax(scaled_probabilities, axis=1)]
    raw = common.classification_metrics(
        classes, all_labels, raw_predictions, raw_probabilities, classes
    )
    scaled = common.classification_metrics(
        classes,
        all_labels,
        scaled_predictions,
        scaled_probabilities,
        classes,
    )
    return temperature, {
        "n_calibration": len(all_labels),
        "calibration_folds": len(scores),
        "crossfit_raw_nll": raw["negative_log_likelihood"],
        "crossfit_calibrated_nll": scaled["negative_log_likelihood"],
        "crossfit_raw_ece": raw["expected_calibration_error_10"],
        "crossfit_calibrated_ece": scaled[
            "expected_calibration_error_10"
        ],
    }


def prediction_rows(
    context: dict[str, Any],
    manifest: pd.DataFrame,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    supported: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in manifest.reset_index(drop=True).iterrows():
        rows.append(
            {
                **context,
                "observation_uid": str(record["observation_uid"]),
                "master_sample_id": str(record["master_sample_id"]),
                "target_analyte": str(labels[index]),
                "predicted_analyte": str(predictions[index]),
                "correct": bool(predictions[index] == labels[index]),
                "supported": bool(supported[index]),
                "confidence": float(probabilities[index].max()),
                "instrument": str(record["instrument"]),
                "sensor_family": str(record["sensor_family"]),
                "probability_classes_json": json.dumps(classes.tolist()),
                "probabilities_json": json.dumps(
                    probabilities[index].tolist()
                ),
            }
        )
    return rows


def evaluate_and_record(
    context: dict[str, Any],
    estimator: Any,
    values: np.ndarray,
    labels: np.ndarray,
    manifest: pd.DataFrame,
    temperature: float,
    protocol: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    predicted, probabilities, classes, metrics = metric_values(
        estimator, values, labels, temperature
    )
    metric = {
        **context,
        **{
            key: value
            for key, value in metrics.items()
            if key != "supported_mask"
        },
    }
    predictions = prediction_rows(
        context,
        manifest,
        labels,
        predicted,
        probabilities,
        classes,
        metrics["supported_mask"],
    )
    per_class = common.per_class_rows(
        context, labels, predicted, metrics["supported_mask"]
    )
    selective = common.selective_rows(
        context,
        labels,
        predicted,
        probabilities.max(axis=1),
        metrics["supported_mask"],
        protocol["locked_evaluation"]["selective_coverages"],
    )
    calibration: list[dict[str, Any]] = []
    scores = common.estimator_scores(estimator, values)
    for name, applied_temperature in (
        ("uncalibrated", 1.0),
        ("temperature_scaled", temperature),
    ):
        probs = common.probabilities_from_scores(scores, applied_temperature)
        preds = classes[np.argmax(probs, axis=1)]
        values_out = common.classification_metrics(
            classes, labels, preds, probs, classes
        )
        calibration.append(
            {
                **context,
                "calibration": name,
                "applied_temperature": applied_temperature,
                **{
                    key: value
                    for key, value in values_out.items()
                    if key != "supported_mask"
                },
            }
        )
    return metric, predictions, per_class, selective, calibration


def band_permutation_rows(
    estimator: Any,
    values: np.ndarray,
    labels: np.ndarray,
    axis: np.ndarray,
    context: dict[str, Any],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    width = int(protocol["interpretability"]["band_width_cm1"])
    repeats = int(protocol["interpretability"]["permutation_repeats"])
    base_predictions = estimator.predict(values)
    baseline = balanced_accuracy_score(labels, base_predictions)
    seed = common.stable_seed(
        protocol["interpretability"]["permutation_seed"],
        context["outer_fold"],
        context["candidate_id"],
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    lower = int(axis.min())
    upper_limit = int(axis.max()) + 1
    for start in range(lower, upper_limit, width):
        stop = min(start + width, upper_limit)
        columns = np.flatnonzero((axis >= start) & (axis < stop))
        changes: list[float] = []
        for _ in range(repeats):
            permuted = values.copy()
            order = rng.permutation(len(values))
            permuted[:, columns] = permuted[order][:, columns]
            score = balanced_accuracy_score(labels, estimator.predict(permuted))
            changes.append(float(baseline - score))
        rows.append(
            {
                **context,
                "band_start_cm1": start,
                "band_stop_cm1_exclusive": stop,
                "n_variables": len(columns),
                "baseline_balanced_accuracy": baseline,
                "importance_mean_ba_drop": float(np.mean(changes)),
                "importance_sd_ba_drop": float(np.std(changes, ddof=1)),
                "permutation_repeats": repeats,
            }
        )
    return rows


def outer_seed_task(
    subset: str,
    outer_fold: int,
    final_seed: int,
    candidate: common.Candidate,
    protocol: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    values = dataset.representations[candidate.representation]
    eligible = common.subset_mask(manifest, subset)
    development = eligible & (folds != outer_fold)
    test_sets = [(subset, eligible & (folds == outer_fold))]
    if subset == "quality_pass":
        test_sets.append(
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
        groups,
        folds,
        development,
        f"{subset}__outer_{outer_fold}",
        int(final_seed),
    )
    seed = common.stable_seed(
        common.PROTOCOL_VERSION,
        "outer_final",
        subset,
        outer_fold,
        candidate.candidate_id,
        final_seed,
    )
    estimator = common.fit_estimator(
        candidate,
        protocol,
        values[development],
        labels[development],
        groups[development],
        seed,
    )
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    selective: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    impurity: list[dict[str, Any]] = []
    band_permutation: list[dict[str, Any]] = []
    if subset == "strict_core":
        for wavenumber, importance in zip(
            dataset.axis_cm1, estimator.feature_importances_
        ):
            impurity.append(
                {
                    "outer_fold": outer_fold,
                    "final_seed": final_seed,
                    "candidate_id": candidate.candidate_id,
                    "representation": candidate.representation,
                    "wavenumber_cm1": float(wavenumber),
                    "importance": float(importance),
                }
            )
    for evaluation_subset, test_mask in test_sets:
        context = {
            "stage": "outer",
            "training_subset": subset,
            "evaluation_subset": evaluation_subset,
            "outer_fold": outer_fold,
            "final_seed": final_seed,
            "seed": seed,
            "candidate_id": candidate.candidate_id,
            "model_family": "random_forest",
            "representation": candidate.representation,
            "temperature": temperature,
            **calibration_context,
        }
        result = evaluate_and_record(
            context,
            estimator,
            values[test_mask],
            labels[test_mask],
            manifest.loc[test_mask],
            temperature,
            protocol,
        )
        metrics.append(result[0])
        predictions.extend(result[1])
        per_class.extend(result[2])
        selective.extend(result[3])
        calibration.extend(result[4])
        if (
            subset == "strict_core"
            and evaluation_subset == "strict_core"
            and int(final_seed)
            == int(protocol["locked_evaluation"]["final_seeds"][0])
        ):
            band_permutation.extend(
                band_permutation_rows(
                    estimator,
                    values[test_mask],
                    labels[test_mask],
                    dataset.axis_cm1,
                    context,
                    protocol,
                )
            )
    return (
        metrics,
        predictions,
        per_class,
        selective,
        calibration,
        impurity,
        band_permutation,
    )


def run_outer(
    output_dir: Path,
    protocol: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    candidates: list[common.Candidate],
    jobs: int,
) -> None:
    lookup = common.candidate_lookup(candidates)
    selection = pd.read_csv(output_dir / "outer_selection.csv")
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    selective: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    impurity: list[dict[str, Any]] = []
    band_permutation: list[dict[str, Any]] = []
    tasks = []
    for subset in ("strict_core", "quality_pass"):
        for outer_fold in range(5):
            selected = selection[
                (selection["subset"] == subset)
                & (selection["outer_fold"] == outer_fold)
            ].iloc[0]
            candidate = lookup[str(selected["candidate_id"])]
            for final_seed in protocol["locked_evaluation"]["final_seeds"]:
                tasks.append(
                    joblib.delayed(outer_seed_task)(
                        subset,
                        outer_fold,
                        int(final_seed),
                        candidate,
                        protocol,
                        dataset,
                    )
                )
    results = joblib.Parallel(n_jobs=jobs, verbose=5)(tasks)
    for result in results:
        metrics.extend(result[0])
        predictions.extend(result[1])
        per_class.extend(result[2])
        selective.extend(result[3])
        calibration.extend(result[4])
        impurity.extend(result[5])
        band_permutation.extend(result[6])
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(output_dir / "outer_metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(per_class).to_csv(
        output_dir / "per_class_metrics.csv", index=False
    )
    pd.DataFrame(selective).to_csv(
        output_dir / "selective_metrics.csv", index=False
    )
    pd.DataFrame(calibration).to_csv(
        output_dir / "calibration_metrics.csv", index=False
    )
    pd.DataFrame(impurity).to_csv(
        output_dir / "impurity_importance.csv", index=False
    )
    pd.DataFrame(band_permutation).to_csv(
        output_dir / "band_permutation_importance.csv", index=False
    )
    write_stress_ood(output_dir, prediction_frame)


def write_stress_ood(output_dir: Path, predictions: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    for (outer_fold, final_seed), frame in predictions[
        predictions["training_subset"] == "quality_pass"
    ].groupby(["outer_fold", "final_seed"]):
        quality = frame[frame["evaluation_subset"] == "quality_pass"]
        stress = frame[
            frame["evaluation_subset"] == "field_quality_stress"
        ]
        result = common.ood_metrics(
            1.0 - quality["confidence"].to_numpy(float),
            1.0 - stress["confidence"].to_numpy(float),
        )
        rows.append(
            {
                "outer_fold": outer_fold,
                "final_seed": final_seed,
                "score": "one_minus_calibrated_max_probability",
                "n_quality": len(quality),
                "n_stress": len(stress),
                **result,
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "field_stress_ood_metrics.csv", index=False
    )


def global_champions(
    output_dir: Path, candidates: list[common.Candidate]
) -> dict[str, common.Candidate]:
    records = json.loads((output_dir / "global_champions.json").read_text())
    lookup = common.candidate_lookup(candidates)
    return {
        subset: lookup[str(record["candidate_id"])]
        for subset, record in records.items()
    }


def domain_seed_task(
    subset: str,
    domain_protocol: str,
    domain_type: str,
    heldout: str,
    final_seed: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    candidate: common.Candidate,
    protocol: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    values = dataset.representations[candidate.representation]
    development = np.zeros(len(manifest), dtype=bool)
    development[train_indices] = True
    temperature, calibration_context = crossfit_temperature(
        candidate,
        protocol,
        values,
        labels,
        groups,
        folds,
        development,
        f"{subset}__{domain_protocol}__{domain_type}__{heldout}",
        int(final_seed),
    )
    seed = common.stable_seed(
        common.PROTOCOL_VERSION,
        "domain",
        subset,
        domain_protocol,
        domain_type,
        heldout,
        final_seed,
    )
    estimator = common.fit_estimator(
        candidate,
        protocol,
        values[train_indices],
        labels[train_indices],
        groups[train_indices],
        seed,
    )
    context = {
        "stage": "domain",
        "subset": subset,
        "protocol": domain_protocol,
        "domain_type": domain_type,
        "heldout_domain": heldout,
        "final_seed": final_seed,
        "seed": seed,
        "candidate_id": candidate.candidate_id,
        "model_family": "random_forest",
        "representation": candidate.representation,
        "temperature": temperature,
        **calibration_context,
    }
    result = evaluate_and_record(
        context,
        estimator,
        values[test_indices],
        labels[test_indices],
        manifest.iloc[test_indices],
        temperature,
        protocol,
    )
    return result[0], result[1]


def run_domain(
    output_dir: Path,
    bundle: Path,
    protocol: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    candidates: list[common.Candidate],
    jobs: int,
) -> None:
    champions = global_champions(output_dir, candidates)
    manifest = dataset.manifest
    uid_to_index = {
        uid: index
        for index, uid in enumerate(
            manifest["observation_uid"].astype(str).to_numpy()
        )
    }
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    tasks = []
    for subset in ("strict_core", "quality_pass"):
        partition_file = bundle / (
            "domain_evaluation_partitions_core.csv"
            if subset == "strict_core"
            else "domain_evaluation_partitions_quality.csv"
        )
        partitions = pd.read_csv(partition_file)
        candidate = champions[subset]
        scenario_groups = partitions.groupby(
            ["protocol", "domain_type", "heldout_domain"], sort=True
        )
        for (domain_protocol, domain_type, heldout), scenario in scenario_groups:
            train_indices = np.asarray(
                [
                    uid_to_index[uid]
                    for uid in scenario.loc[
                        scenario["partition"] == "train", "observation_uid"
                    ].astype(str)
                ],
                dtype=int,
            )
            test_indices = np.asarray(
                [
                    uid_to_index[uid]
                    for uid in scenario.loc[
                        scenario["partition"] == "test", "observation_uid"
                    ].astype(str)
                ],
                dtype=int,
            )
            for final_seed in protocol["locked_evaluation"]["final_seeds"]:
                tasks.append(
                    joblib.delayed(domain_seed_task)(
                        subset,
                        str(domain_protocol),
                        str(domain_type),
                        str(heldout),
                        int(final_seed),
                        train_indices,
                        test_indices,
                        candidate,
                        protocol,
                        dataset,
                    )
                )
    results = joblib.Parallel(n_jobs=jobs, verbose=5)(tasks)
    for metric, prediction_rows_out in results:
        metrics.append(metric)
        predictions.extend(prediction_rows_out)
    pd.DataFrame(metrics).to_csv(
        output_dir / "domain_metrics.csv", index=False
    )
    pd.DataFrame(predictions).to_csv(
        output_dir / "domain_predictions.csv", index=False
    )


def run_negative(
    output_dir: Path,
    protocol: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    candidates: list[common.Candidate],
) -> None:
    candidate = global_champions(output_dir, candidates)["strict_core"]
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    groups = manifest["master_sample_id"].astype(str).to_numpy()
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    values = dataset.representations[candidate.representation]
    rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        train_mask = folds != outer_fold
        test_mask = folds == outer_fold
        masters = manifest.loc[
            train_mask, ["master_sample_id", "target_analyte"]
        ].drop_duplicates()
        for final_seed in protocol["negative_control"]["seeds"]:
            seed = common.stable_seed(
                common.PROTOCOL_VERSION,
                "negative",
                outer_fold,
                final_seed,
            )
            rng = np.random.default_rng(seed)
            shuffled = rng.permutation(
                masters["target_analyte"].astype(str).to_numpy()
            )
            mapping = dict(
                zip(masters["master_sample_id"].astype(str), shuffled)
            )
            permuted_labels = np.asarray(
                [mapping[group] for group in groups[train_mask]], dtype=str
            )
            estimator = common.fit_estimator(
                candidate,
                protocol,
                values[train_mask],
                permuted_labels,
                groups[train_mask],
                seed,
            )
            _, _, _, metrics = metric_values(
                estimator, values[test_mask], labels[test_mask]
            )
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "final_seed": final_seed,
                    "seed": seed,
                    "candidate_id": candidate.candidate_id,
                    "representation": candidate.representation,
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
        args.output_dir, args.protocol, args.bundle, protocol, candidates
    )
    dataset = common.baseline.load_nato_dataset(args.bundle)
    stages = (
        ("search", "outer", "domain", "negative")
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
            run_outer(
                args.output_dir, protocol, dataset, candidates, args.jobs
            )
        elif stage == "domain":
            run_domain(
                args.output_dir,
                args.bundle,
                protocol,
                dataset,
                candidates,
                args.jobs,
            )
        elif stage == "negative":
            run_negative(output_dir=args.output_dir, protocol=protocol,
                         dataset=dataset, candidates=candidates)
        print(f"DONE {stage}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
