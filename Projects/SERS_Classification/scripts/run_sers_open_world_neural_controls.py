#!/usr/bin/env python3
"""Run frozen neural controls under the SERS open-world partition protocol."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

import sers_open_world_common as common
import sers_open_world_neural_common as neural


METHODS = (
    "siamese_triplet",
    "supervised_contrastive",
    "standard_vae",
    "structured_vae",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--stage", choices=("calibration", "final", "all"), default="all"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/sers_open_world_neural_controls_v1.json"),
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
        default=Path(
            "Workspace/sers_open_world_shift/open_world_v1/neural_controls"
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def method_dir(args: argparse.Namespace) -> Path:
    return args.output_dir / args.method


def initialize(
    args: argparse.Namespace,
    registry: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    output = method_dir(args)
    output.mkdir(parents=True, exist_ok=True)
    for filename, source in (
        ("predeclared_neural_registry.json", args.registry),
        ("predeclared_protocol.json", args.protocol),
    ):
        destination = output / filename
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f"Different immutable snapshot: {filename}")
        if not destination.exists():
            shutil.copyfile(source, destination)
    validation = json.loads(
        (args.partition_dir / "validation_report.json").read_text()
    )
    if validation.get("status") != "pass":
        raise ValueError("Open-world partitions have not passed validation")
    dataset = common.baseline.load_nato_dataset(args.bundle)
    if len(dataset.manifest) != protocol["immutable_input"]["strict_core_rows"]:
        raise ValueError("Neural-control input row count changed")
    common.write_json(
        output / "input_hashes.json",
        {
            "registry": common.sha256_file(args.registry),
            "protocol": common.sha256_file(args.protocol),
            "partition_validation": common.sha256_file(
                args.partition_dir / "validation_report.json"
            ),
            "outer_partitions": common.sha256_file(
                args.partition_dir / "open_set_outer_partitions.csv"
            ),
            "threshold_partitions": common.sha256_file(
                args.partition_dir
                / "known_threshold_crossfit_partitions.csv"
            ),
            "dataset_catalog": common.sha256_file(
                args.bundle / "artifact_hashes.json"
            ),
            "runner": common.sha256_file(Path(__file__)),
            "neural_common": common.sha256_file(Path(neural.__file__)),
            "open_common": common.sha256_file(Path(common.__file__)),
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
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
            "gpu": (
                torch.cuda.get_device_name()
                if torch.cuda.is_available()
                else None
            ),
            "method": args.method,
            "control": neural.control_record(registry, args.method),
        },
    )


def uid_index(
    dataset: common.baseline.SpectralDataset,
) -> dict[str, int]:
    return {
        uid: index
        for index, uid in enumerate(
            dataset.manifest["observation_uid"].astype(str).to_numpy()
        )
    }


def scenario_indices(
    frame: pd.DataFrame,
    uid_to_index: dict[str, int],
) -> dict[str, np.ndarray]:
    return {
        str(partition): np.asarray(
            [uid_to_index[str(uid)] for uid in group["observation_uid"]],
            dtype=int,
        )
        for partition, group in frame.groupby("partition", sort=False)
    }


def calibration_for_task(
    args: argparse.Namespace,
    registry: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    threshold_frame: pd.DataFrame,
    uid_to_index: dict[str, int],
    held_unknown: str,
    outer_fold: int,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    spec = neural.control_record(registry, args.method)
    values = dataset.representations[str(spec["representation"])]
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    target_classes = np.asarray(
        sorted(set(labels) - {held_unknown}), dtype=str
    )
    calibration_seed = int(registry["calibration"]["calibration_seed"])
    per_head: dict[str, dict[str, list[Any]]] = {}
    run_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    development_folds = sorted(set(range(5)) - {outer_fold})
    for calibration_fold in development_folds:
        frame = threshold_frame[
            (threshold_frame["held_unknown"] == held_unknown)
            & (threshold_frame["outer_fold"] == outer_fold)
            & (
                threshold_frame["calibration_fold"]
                == calibration_fold
            )
        ]
        partitions = scenario_indices(frame, uid_to_index)
        train = partitions["train_known"]
        calibration = partitions["calibration_known"]
        seed = common.stable_seed(
            neural.REGISTRY_VERSION,
            args.method,
            "calibration",
            held_unknown,
            outer_fold,
            calibration_fold,
            calibration_seed,
        )
        run_identifier = (
            f"{args.method}__h{held_unknown}__o{outer_fold}__"
            f"i{calibration_fold}__s{calibration_seed}"
        )
        fit, history, run = neural.load_or_train(
            args.method,
            registry,
            values[train],
            manifest.iloc[train].reset_index(drop=True),
            values[calibration],
            manifest.iloc[calibration].reset_index(drop=True),
            seed,
            run_identifier,
            method_dir(args) / "cache" / "calibration",
            device,
        )
        run_rows.append(
            {
                "held_unknown": held_unknown,
                "outer_fold": outer_fold,
                "calibration_fold": calibration_fold,
                "stage": "calibration",
                "run_seed": seed,
                "n_train": len(train),
                "n_train_masters": int(
                    manifest.iloc[train]["master_sample_id"].nunique()
                ),
                "n_calibration": len(calibration),
                **run,
            }
        )
        history_context = {
            "method": args.method,
            "held_unknown": held_unknown,
            "outer_fold": outer_fold,
            "calibration_fold": calibration_fold,
            "stage": "calibration",
            "run_seed": seed,
        }
        history_rows.extend(
            [
                {**history_context, **record}
                for record in history.to_dict("records")
            ]
        )
        for head in fit.heads:
            output = fit.output(
                values[calibration],
                manifest.iloc[calibration].reset_index(drop=True),
                head,
            )
            logits = common.align_scores(
                output.logits, output.classes, target_classes
            )
            _, _, score_values = fit.scores(output, head, 1.0)
            container = per_head.setdefault(
                head,
                {
                    "indices": [],
                    "labels": [],
                    "logits": [],
                    **{
                        f"score::{score_name}": []
                        for score_name in fit.applicable_scores
                    },
                },
            )
            container["indices"].append(calibration)
            container["labels"].append(labels[calibration])
            container["logits"].append(logits)
            for score_name, score in score_values.items():
                container[f"score::{score_name}"].append(score)
        print(
            f"CALIBRATION {args.method} H={held_unknown} "
            f"O={outer_fold} I={calibration_fold}",
            flush=True,
        )

    summary_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    bounds = tuple(
        float(value)
        for value in registry["calibration"]["temperature_bounds"]
    )
    coverages = [
        float(value)
        for value in registry["calibration"]["known_retained_coverages"]
    ]
    for head, container in per_head.items():
        indices = np.concatenate(container["indices"])
        order = np.argsort(indices, kind="stable")
        indices = indices[order]
        calibration_labels = np.concatenate(container["labels"])[order]
        logits = np.vstack(container["logits"])[order]
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Neural calibration duplicates observations")
        temperature = common.fit_temperature(
            logits,
            calibration_labels,
            target_classes,
            bounds,  # type: ignore[arg-type]
        )
        uncalibrated_probabilities = common.probabilities_from_scores(
            logits, 1.0
        )
        probabilities = common.probabilities_from_scores(
            logits, temperature
        )
        calibrated_scores: dict[str, np.ndarray] = {}
        for score_name in spec["applicable_scores"]:
            if score_name == "one_minus_calibrated_max_probability":
                calibrated_scores[score_name] = (
                    1.0 - probabilities.max(axis=1)
                )
            elif score_name == "predictive_entropy":
                calibrated_scores[score_name] = common.predictive_entropy(
                    probabilities
                )
            else:
                calibrated_scores[score_name] = np.concatenate(
                    container[f"score::{score_name}"]
                )[order]
        base = {
            "method": args.method,
            "head": head,
            "head_primary": head
            == str(spec.get("primary_head", head)),
            "held_unknown": held_unknown,
            "outer_fold": outer_fold,
            "temperature": temperature,
            "n_calibration": len(indices),
            "n_calibration_masters": int(
                manifest.iloc[indices]["master_sample_id"].nunique()
            ),
            "classes_json": json.dumps(target_classes.tolist()),
        }
        for name, probability in (
            ("uncalibrated", uncalibrated_probabilities),
            ("temperature_scaled", probabilities),
        ):
            prediction = target_classes[np.argmax(probability, axis=1)]
            metrics = common.classification_metrics(
                target_classes,
                calibration_labels,
                prediction,
                probability,
                target_classes,
            )
            summary_rows.append(
                {
                    **base,
                    "calibration": name,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key != "supported_mask"
                    },
                }
            )
        for score_name, score_values in calibrated_scores.items():
            for coverage in coverages:
                threshold_rows.append(
                    {
                        **base,
                        "score_name": score_name,
                        "score_primary": score_name == spec["primary_score"],
                        "requested_known_coverage": coverage,
                        "threshold": common.threshold_from_known(
                            score_values, coverage
                        ),
                    }
                )
        for offset, index in enumerate(indices):
            score_rows.append(
                {
                    **base,
                    "observation_uid": str(
                        manifest.iloc[index]["observation_uid"]
                    ),
                    "master_sample_id": str(
                        manifest.iloc[index]["master_sample_id"]
                    ),
                    "target_analyte": str(labels[index]),
                    "probabilities_json": json.dumps(
                        probabilities[offset].tolist()
                    ),
                    **{
                        score_name: float(score_values[offset])
                        for score_name, score_values in calibrated_scores.items()
                    },
                }
            )
    return summary_rows, threshold_rows, score_rows, run_rows + history_rows


def run_calibration(
    args: argparse.Namespace,
    registry: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    device: torch.device,
) -> None:
    threshold_frame = pd.read_csv(
        args.partition_dir / "known_threshold_crossfit_partitions.csv"
    )
    uid_to_index = uid_index(dataset)
    held_chemicals = sorted(threshold_frame["held_unknown"].unique())
    summaries: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    combined: list[dict[str, Any]] = []
    for held_unknown in held_chemicals:
        for outer_fold in range(5):
            summary, threshold, score, combined_rows = calibration_for_task(
                args,
                registry,
                dataset,
                threshold_frame,
                uid_to_index,
                str(held_unknown),
                outer_fold,
                device,
            )
            summaries.extend(summary)
            thresholds.extend(threshold)
            scores.extend(score)
            combined.extend(combined_rows)
    output = method_dir(args)
    pd.DataFrame(summaries).to_csv(
        output / "calibration_metrics.csv", index=False
    )
    pd.DataFrame(thresholds).to_csv(
        output / "calibration_thresholds.csv", index=False
    )
    pd.DataFrame(scores).to_csv(
        output / "calibration_scores.csv", index=False
    )
    # Run records have checkpoint fields; histories have epoch fields.
    combined_frame = pd.DataFrame(combined)
    combined_frame[combined_frame["checkpoint"].notna()].to_csv(
        output / "calibration_run_registry.csv", index=False
    )
    combined_frame[combined_frame["epoch"].notna()].to_csv(
        output / "calibration_training_histories.csv", index=False
    )


def final_task(
    args: argparse.Namespace,
    registry: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    outer_partitions: dict[str, np.ndarray],
    calibration_metrics: pd.DataFrame,
    calibration_thresholds: pd.DataFrame,
    held_unknown: str,
    outer_fold: int,
    declared_seed: int,
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    spec = neural.control_record(registry, args.method)
    values = dataset.representations[str(spec["representation"])]
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    development = outer_partitions["known_development_quality"]
    development_folds = sorted(set(range(5)) - {outer_fold})
    monitor_fold = development_folds[0]
    monitor = development[
        manifest.iloc[development]["grouped_sample_fold_5"].to_numpy(int)
        == monitor_fold
    ]
    seed = common.stable_seed(
        neural.REGISTRY_VERSION,
        args.method,
        "final",
        held_unknown,
        outer_fold,
        declared_seed,
    )
    run_identifier = (
        f"{args.method}__h{held_unknown}__o{outer_fold}__"
        f"final__s{declared_seed}"
    )
    fit, history, run = neural.load_or_train(
        args.method,
        registry,
        values[development],
        manifest.iloc[development].reset_index(drop=True),
        values[monitor],
        manifest.iloc[monitor].reset_index(drop=True),
        seed,
        run_identifier,
        method_dir(args) / "cache" / "final",
        device,
    )
    outputs: dict[str, list[dict[str, Any]]] = {
        "predictions": [],
        "classification": [],
        "open_metrics": [],
        "threshold_metrics": [],
        "run_registry": [],
        "histories": [],
    }
    outputs["run_registry"].append(
        {
            "method": args.method,
            "held_unknown": held_unknown,
            "outer_fold": outer_fold,
            "declared_seed": declared_seed,
            "run_seed": seed,
            "stage": "final",
            "n_development": len(development),
            "n_development_masters": int(
                manifest.iloc[development]["master_sample_id"].nunique()
            ),
            **run,
        }
    )
    outputs["histories"].extend(
        [
            {
                "method": args.method,
                "held_unknown": held_unknown,
                "outer_fold": outer_fold,
                "declared_seed": declared_seed,
                "run_seed": seed,
                "stage": "final",
                **record,
            }
            for record in history.to_dict("records")
        ]
    )
    for head in fit.heads:
        calibration = calibration_metrics[
            (calibration_metrics["held_unknown"] == held_unknown)
            & (calibration_metrics["outer_fold"] == outer_fold)
            & (calibration_metrics["head"] == head)
            & (
                calibration_metrics["calibration"]
                == "temperature_scaled"
            )
        ]
        if len(calibration) != 1:
            raise ValueError("Missing unique neural calibration record")
        temperature = float(calibration.iloc[0]["temperature"])
        threshold_frame = calibration_thresholds[
            (calibration_thresholds["held_unknown"] == held_unknown)
            & (calibration_thresholds["outer_fold"] == outer_fold)
            & (calibration_thresholds["head"] == head)
        ]
        thresholds = {
            str(score_name): {
                float(row["requested_known_coverage"]): float(row["threshold"])
                for _, row in score_frame.iterrows()
            }
            for score_name, score_frame in threshold_frame.groupby(
                "score_name", sort=False
            )
        }
        base = {
            "method": args.method,
            "head": head,
            "head_primary": head
            == str(spec.get("primary_head", head)),
            "representation": spec["representation"],
            "held_unknown": held_unknown,
            "outer_fold": outer_fold,
            "declared_seed": declared_seed,
            "run_seed": seed,
            "temperature": temperature,
            "primary_score": spec["primary_score"],
            "n_development": len(development),
            "n_development_masters": int(
                manifest.iloc[development]["master_sample_id"].nunique()
            ),
        }
        evaluated: dict[str, dict[str, Any]] = {}
        for partition_name in (
            "known_quality_test",
            "unknown_quality",
            "known_stress",
            "unknown_stress",
        ):
            indices = outer_partitions[partition_name]
            partition_manifest = manifest.iloc[indices].reset_index(drop=True)
            output = fit.output(values[indices], partition_manifest, head)
            _, probabilities, score_values = fit.scores(
                output, head, temperature
            )
            predictions = fit.classes[np.argmax(probabilities, axis=1)]
            context = {
                **base,
                "evaluation_partition": partition_name,
            }
            outputs["predictions"].extend(
                [
                    {
                        **context,
                        "observation_uid": str(row["observation_uid"]),
                        "master_sample_id": str(row["master_sample_id"]),
                        "target_analyte": str(labels[indices[offset]]),
                        "predicted_analyte": str(predictions[offset]),
                        "correct": bool(
                            predictions[offset] == labels[indices[offset]]
                        ),
                        "instrument": str(row["instrument"]),
                        "sensor_family": str(row["sensor_family"]),
                        "probability_classes_json": json.dumps(
                            fit.classes.tolist()
                        ),
                        "probabilities_json": json.dumps(
                            probabilities[offset].tolist()
                        ),
                        **{
                            score_name: float(score[offset])
                            for score_name, score in score_values.items()
                        },
                    }
                    for offset, (_, row) in enumerate(
                        partition_manifest.iterrows()
                    )
                ]
            )
            classification = common.classification_metrics(
                fit.classes,
                labels[indices],
                predictions,
                probabilities,
                fit.classes,
            )
            outputs["classification"].append(
                {
                    **context,
                    **{
                        key: value
                        for key, value in classification.items()
                        if key != "supported_mask"
                    },
                }
            )
            evaluated[partition_name] = {
                "indices": indices,
                "probabilities": probabilities,
                "predictions": predictions,
                "scores": score_values,
            }
            for score_name, score in score_values.items():
                for coverage, threshold in thresholds[score_name].items():
                    accepted = score <= threshold
                    supported = np.isin(labels[indices], fit.classes)
                    retained = accepted & supported
                    outputs["threshold_metrics"].append(
                        {
                            **context,
                            "score_name": score_name,
                            "score_primary": (
                                score_name == spec["primary_score"]
                            ),
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
        for evaluation, (known_name, unknown_name) in {
            "quality_open": ("known_quality_test", "unknown_quality"),
            "stress_open": ("known_stress", "unknown_stress"),
            "field_quality": ("known_quality_test", "known_stress"),
        }.items():
            known = evaluated[known_name]
            unknown = evaluated[unknown_name]
            indices = np.concatenate([known["indices"], unknown["indices"]])
            probabilities = np.vstack(
                [known["probabilities"], unknown["probabilities"]]
            )
            predictions = np.concatenate(
                [known["predictions"], unknown["predictions"]]
            )
            unknown_mask = np.concatenate(
                [
                    np.zeros(len(known["indices"]), dtype=bool),
                    np.ones(len(unknown["indices"]), dtype=bool),
                ]
            )
            for score_name in fit.applicable_scores:
                score = np.concatenate(
                    [
                        known["scores"][score_name],
                        unknown["scores"][score_name],
                    ]
                )
                metrics, _ = common.open_set_metrics(
                    manifest.iloc[indices].reset_index(drop=True),
                    labels[indices],
                    predictions,
                    probabilities,
                    fit.classes,
                    score,
                    unknown_mask,
                )
                outputs["open_metrics"].append(
                    {
                        **base,
                        "evaluation": evaluation,
                        "known_partition": known_name,
                        "unknown_partition": unknown_name,
                        "score_name": score_name,
                        "score_primary": (
                            score_name == spec["primary_score"]
                        ),
                        "direction_control_auroc": (
                            1.0 - metrics["master_unknown_auroc"]
                            if np.isfinite(metrics["master_unknown_auroc"])
                            else np.nan
                        ),
                        **metrics,
                    }
                )
    print(
        f"FINAL {args.method} H={held_unknown} O={outer_fold} "
        f"seed={declared_seed}",
        flush=True,
    )
    return outputs


def run_final(
    args: argparse.Namespace,
    registry: dict[str, Any],
    dataset: common.baseline.SpectralDataset,
    device: torch.device,
) -> None:
    output = method_dir(args)
    calibration_metrics = pd.read_csv(output / "calibration_metrics.csv")
    calibration_thresholds = pd.read_csv(
        output / "calibration_thresholds.csv"
    )
    outer_frame = pd.read_csv(
        args.partition_dir / "open_set_outer_partitions.csv"
    )
    uid_to_index = uid_index(dataset)
    results = {
        name: []
        for name in (
            "predictions",
            "classification",
            "open_metrics",
            "threshold_metrics",
            "run_registry",
            "histories",
        )
    }
    for held_unknown in sorted(outer_frame["held_unknown"].unique()):
        for outer_fold in range(5):
            frame = outer_frame[
                (outer_frame["held_unknown"] == held_unknown)
                & (outer_frame["outer_fold"] == outer_fold)
            ]
            partitions = scenario_indices(frame, uid_to_index)
            for declared_seed in registry["final_seeds"]:
                task = final_task(
                    args,
                    registry,
                    dataset,
                    partitions,
                    calibration_metrics,
                    calibration_thresholds,
                    str(held_unknown),
                    outer_fold,
                    int(declared_seed),
                    device,
                )
                for name in results:
                    results[name].extend(task[name])
    for name, rows in results.items():
        pd.DataFrame(rows).to_csv(output / f"final_{name}.csv", index=False)


def main() -> None:
    args = parse_args()
    registry = neural.load_registry(args.registry)
    protocol = common.load_protocol(args.protocol)
    initialize(args, registry, protocol)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dataset = common.baseline.load_nato_dataset(args.bundle)
    stages = (
        ("calibration", "final")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        print(f"START {args.method} {stage}", flush=True)
        if stage == "calibration":
            run_calibration(args, registry, dataset, device)
        else:
            run_final(args, registry, dataset, device)
        print(f"DONE {args.method} {stage}", flush=True)


if __name__ == "__main__":
    main()
