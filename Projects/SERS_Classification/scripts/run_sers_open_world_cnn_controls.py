#!/usr/bin/env python3
"""Run preregistered CNN label, preprocessing, and augmentation controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

import sers_open_world_cnn_common as cnn
import sers_open_world_common as common
from run_sers_open_world_cnn import (
    SCORES,
    fit_density,
    output_scores,
    partitions,
    uid_lookup,
)


CONTROL_REGISTRY_VERSION = "sers-open-world-cnn-controls-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("permutation", "sensitivity", "all"),
        default="all",
    )
    parser.add_argument(
        "--cnn-registry",
        type=Path,
        default=Path("configs/sers_open_world_cnn_v1.json"),
    )
    parser.add_argument(
        "--control-registry",
        type=Path,
        default=Path("configs/sers_open_world_cnn_controls_v1.json"),
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
        "--cnn-dir",
        type=Path,
        default=Path("Workspace/sers_open_world_shift/open_world_v1/cnn"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_control_registry(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("registry_version") != CONTROL_REGISTRY_VERSION:
        raise ValueError("Unexpected CNN control registry")
    expected = (
        "predeclared_after_surrogate_execution_started_before_surrogate_metrics_"
        "or_any_cnn_locked_outcome_were_inspected"
    )
    if value.get("status_before_execution") != expected:
        raise ValueError("CNN control registry was not frozen at the declared stage")
    return value


def initialize(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    cnn_registry = cnn.load_registry(args.cnn_registry)
    control_registry = load_control_registry(args.control_registry)
    for name, source in (
        ("predeclared_cnn_registry.json", args.cnn_registry),
        ("predeclared_cnn_controls_registry.json", args.control_registry),
    ):
        target = args.cnn_dir / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Different immutable snapshot already exists: {name}")
        if not target.exists():
            shutil.copyfile(source, target)
    fingerprint = (
        sha256_file(args.cnn_registry)
        + ":"
        + sha256_file(args.control_registry)
    )
    return cnn_registry, control_registry, fingerprint


def slug(*parts: object) -> str:
    text = "__".join(str(part) for part in parts)
    return "".join(character if character.isalnum() else "_" for character in text)


def cache_file(cache_dir: Path, *parts: object) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{slug(*parts)}.json"


def task_fingerprint(
    registry_fingerprint: str,
    context: dict[str, Any],
    train_uids: np.ndarray,
    test_uids: np.ndarray,
) -> str:
    digest = hashlib.sha256(registry_fingerprint.encode())
    digest.update(json.dumps(context, sort_keys=True).encode())
    for values in (train_uids, test_uids):
        for value in values.astype(str):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def read_cache(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("fingerprint") != fingerprint:
        raise ValueError(f"Stale control cache: {path}")
    return value


def write_cache(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(
        json.dumps(common.json_clean(value), indent=2, sort_keys=True)
    )
    temporary.replace(path)


def selected_variant(
    selected: pd.Series, cnn_registry: dict[str, Any]
) -> cnn.Variant:
    return next(
        variant
        for variant in cnn.variants(cnn_registry)
        if variant.name == str(selected.variant)
    )


def permute_master_labels(
    manifest: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    masters = (
        manifest[["master_sample_id", "target_analyte"]]
        .drop_duplicates()
        .sort_values("master_sample_id", kind="stable")
        .reset_index(drop=True)
    )
    if masters.groupby("master_sample_id")["target_analyte"].nunique().max() != 1:
        raise ValueError("A development master maps to multiple labels")
    rng = np.random.default_rng(seed)
    original = masters["target_analyte"].astype(str).to_numpy()
    permuted = original[rng.permutation(len(original))]
    mapping = dict(
        zip(masters["master_sample_id"].astype(str).to_numpy(), permuted)
    )
    output = manifest.copy()
    output["target_analyte"] = [
        mapping[str(value)] for value in output["master_sample_id"]
    ]
    return output, {
        "development_masters": len(masters),
        "master_label_unchanged_fraction": float(np.mean(original == permuted)),
        "row_label_unchanged_fraction": float(
            np.mean(
                manifest["target_analyte"].astype(str).to_numpy()
                == output["target_analyte"].astype(str).to_numpy()
            )
        ),
    }


def run_permutation(
    args: argparse.Namespace,
    dataset: common.baseline.SpectralDataset,
    cnn_registry: dict[str, Any],
    control_registry: dict[str, Any],
    registry_fingerprint: str,
    device: torch.device,
) -> None:
    selection = pd.read_csv(args.cnn_dir / "surrogate_selection.csv")
    frame = pd.read_csv(args.partition_dir / "open_set_outer_partitions.csv")
    lookup = uid_lookup(dataset)
    manifest = dataset.manifest
    true_labels = manifest["target_analyte"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    total = len(selection) * len(control_registry["final_seeds"])
    completed = 0
    cache_dir = args.cnn_dir / "cache" / "controls_permutation"
    for selected in selection.itertuples(index=False):
        held, outer = str(selected.held_unknown), int(selected.outer_fold)
        split = partitions(
            frame[
                (frame["held_unknown"] == held)
                & (frame["outer_fold"] == outer)
            ],
            lookup,
        )
        development = split["known_development_quality"]
        test = split["known_quality_test"]
        variant = selected_variant(pd.Series(selected._asdict()), cnn_registry)
        values = dataset.representations[variant.representation]
        for declared_seed in control_registry["final_seeds"]:
            seed = cnn.stable_seed(
                CONTROL_REGISTRY_VERSION,
                "master_label_permutation",
                held,
                outer,
                declared_seed,
            )
            context = {
                "held_unknown": held,
                "outer_fold": outer,
                "declared_seed": int(declared_seed),
                "run_seed": seed,
                "variant": variant.name,
                "representation": variant.representation,
                "selected_epoch": int(selected.selected_epoch),
            }
            fingerprint = task_fingerprint(
                registry_fingerprint,
                context,
                manifest.iloc[development]["observation_uid"].astype(str).to_numpy(),
                manifest.iloc[test]["observation_uid"].astype(str).to_numpy(),
            )
            path = cache_file(
                cache_dir, held, outer, declared_seed, variant.name
            )
            payload = read_cache(path, fingerprint)
            if payload is None:
                permuted, audit = permute_master_labels(
                    manifest.iloc[development].reset_index(drop=True),
                    cnn.stable_seed(seed, "labels"),
                )
                trained = cnn.train(
                    values[development],
                    permuted,
                    None,
                    None,
                    variant,
                    cnn_registry,
                    seed,
                    device,
                    fixed_epochs=int(selected.selected_epoch),
                )
                _, logits = cnn.deterministic_outputs(
                    trained.model, values[test], device
                )
                probabilities = common.probabilities_from_scores(logits, 1.0)
                prediction = trained.classes[np.argmax(probabilities, axis=1)]
                metrics = common.classification_metrics(
                    trained.classes,
                    true_labels[test],
                    prediction,
                    probabilities,
                    trained.classes,
                )
                payload = {
                    "fingerprint": fingerprint,
                    "metric": {
                        **context,
                        **audit,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key != "supported_mask"
                        },
                    },
                    "run": {
                        **context,
                        **audit,
                        "state_sha256": trained.state_sha256,
                        "parameter_count": trained.parameter_count,
                        "epochs_run": len(trained.history),
                        "n_development": len(development),
                        "n_test": len(test),
                    },
                }
                write_cache(path, payload)
            rows.append(payload["metric"])
            run_rows.append(payload["run"])
            completed += 1
            if completed % 15 == 0 or completed == total:
                print(f"PERMUTATION CNN {completed}/{total}", flush=True)
    pd.DataFrame(rows).to_csv(
        args.cnn_dir / "negative_master_label_permutation_metrics.csv",
        index=False,
    )
    pd.DataFrame(run_rows).to_csv(
        args.cnn_dir / "negative_master_label_permutation_run_registry.csv",
        index=False,
    )


def epoch_lookup(args: argparse.Namespace) -> dict[str, int]:
    frame = pd.read_csv(args.cnn_dir / "domain_epoch_registry.csv")
    return {
        str(row.variant): int(row.selected_epoch)
        for row in frame.itertuples(index=False)
    }


def control_variant(record: dict[str, Any]) -> cnn.Variant:
    return cnn.Variant(
        order=-1,
        name="erm_derivative",
        representation=str(record["representation"]),
    )


def run_sensitivity(
    args: argparse.Namespace,
    dataset: common.baseline.SpectralDataset,
    cnn_registry: dict[str, Any],
    control_registry: dict[str, Any],
    registry_fingerprint: str,
    device: torch.device,
) -> None:
    epochs = epoch_lookup(args)
    outer_frame = pd.read_csv(
        args.partition_dir / "open_set_outer_partitions.csv"
    )
    scenarios = (
        outer_frame[["held_unknown", "outer_fold"]]
        .drop_duplicates()
        .sort_values(["held_unknown", "outer_fold"])
    )
    lookup = uid_lookup(dataset)
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    output: dict[str, list[dict[str, Any]]] = {
        "classification": [],
        "open_metrics": [],
        "predictions": [],
        "run_registry": [],
    }
    controls = control_registry["fixed_erm_sensitivities"]
    total = len(scenarios) * len(controls) * len(
        control_registry["final_seeds"]
    )
    completed = 0
    cache_dir = args.cnn_dir / "cache" / "controls_sensitivity"
    for scenario in scenarios.itertuples(index=False):
        held, outer = str(scenario.held_unknown), int(scenario.outer_fold)
        split = partitions(
            outer_frame[
                (outer_frame["held_unknown"] == held)
                & (outer_frame["outer_fold"] == outer)
            ],
            lookup,
        )
        development = split["known_development_quality"]
        for record in controls:
            control = str(record["control"])
            variant = control_variant(record)
            representation = str(record["representation"])
            epoch = epochs[str(record["epoch_source_variant"])]
            values = dataset.representations[representation]
            for declared_seed in control_registry["final_seeds"]:
                seed = cnn.stable_seed(
                    CONTROL_REGISTRY_VERSION,
                    "fixed_erm_sensitivity",
                    held,
                    outer,
                    declared_seed,
                )
                context = {
                    "held_unknown": held,
                    "outer_fold": outer,
                    "control": control,
                    "representation": representation,
                    "augmentation": bool(record["augmentation"]),
                    "epoch_source_variant": str(record["epoch_source_variant"]),
                    "selected_epoch": epoch,
                    "declared_seed": int(declared_seed),
                    "run_seed": seed,
                    "temperature": 1.0,
                }
                all_test = np.concatenate(
                    [
                        split[name]
                        for name in control_registry["locked_partitions"]
                    ]
                )
                fingerprint = task_fingerprint(
                    registry_fingerprint,
                    context,
                    manifest.iloc[development]["observation_uid"]
                    .astype(str)
                    .to_numpy(),
                    manifest.iloc[all_test]["observation_uid"]
                    .astype(str)
                    .to_numpy(),
                )
                path = cache_file(
                    cache_dir, held, outer, control, declared_seed
                )
                payload = read_cache(path, fingerprint)
                if payload is None:
                    trained = cnn.train(
                        values[development],
                        manifest.iloc[development].reset_index(drop=True),
                        None,
                        None,
                        variant,
                        cnn_registry,
                        seed,
                        device,
                        fixed_epochs=epoch,
                        augmentation_enabled=bool(record["augmentation"]),
                    )
                    density = fit_density(
                        trained,
                        values[development],
                        manifest.iloc[development].reset_index(drop=True),
                        cnn_registry,
                        seed,
                        device,
                    )
                    evaluated: dict[
                        str,
                        tuple[
                            np.ndarray,
                            np.ndarray,
                            np.ndarray,
                            dict[str, np.ndarray],
                        ],
                    ] = {}
                    classification_rows: list[dict[str, Any]] = []
                    prediction_rows: list[dict[str, Any]] = []
                    for partition_name in control_registry["locked_partitions"]:
                        index = split[partition_name]
                        common.baseline.configure_determinism(
                            cnn.stable_seed(
                                seed, "sensitivity_mc", partition_name
                            )
                        )
                        _, _, probabilities, scores = output_scores(
                            trained,
                            density,
                            values[index],
                            device,
                            1.0,
                            int(cnn_registry["mc_dropout_passes"]),
                        )
                        prediction = trained.classes[
                            np.argmax(probabilities, axis=1)
                        ]
                        metrics = common.classification_metrics(
                            trained.classes,
                            labels[index],
                            prediction,
                            probabilities,
                            trained.classes,
                        )
                        classification_rows.append(
                            {
                                **context,
                                "evaluation_partition": partition_name,
                                **{
                                    key: value
                                    for key, value in metrics.items()
                                    if key != "supported_mask"
                                },
                            }
                        )
                        for offset, row_index in enumerate(index):
                            row = manifest.iloc[row_index]
                            prediction_rows.append(
                                {
                                    **context,
                                    "evaluation_partition": partition_name,
                                    "observation_uid": str(row.observation_uid),
                                    "master_sample_id": str(row.master_sample_id),
                                    "target_analyte": labels[row_index],
                                    "predicted_analyte": str(
                                        prediction[offset]
                                    ),
                                    "instrument": str(row.instrument),
                                    "sensor_family": str(row.sensor_family),
                                    "probability_classes_json": json.dumps(
                                        trained.classes.tolist()
                                    ),
                                    "probabilities_json": json.dumps(
                                        probabilities[offset].tolist()
                                    ),
                                    **{
                                        name: float(score[offset])
                                        for name, score in scores.items()
                                    },
                                }
                            )
                        evaluated[partition_name] = (
                            index,
                            probabilities,
                            prediction,
                            scores,
                        )
                    open_rows: list[dict[str, Any]] = []
                    for evaluation, (known_name, unknown_name) in {
                        "quality_open": (
                            "known_quality_test",
                            "unknown_quality",
                        ),
                        "stress_open": ("known_stress", "unknown_stress"),
                        "field_quality": (
                            "known_quality_test",
                            "known_stress",
                        ),
                    }.items():
                        known, unknown = (
                            evaluated[known_name],
                            evaluated[unknown_name],
                        )
                        index = np.concatenate([known[0], unknown[0]])
                        probabilities = np.vstack([known[1], unknown[1]])
                        prediction = np.concatenate([known[2], unknown[2]])
                        unknown_mask = np.concatenate(
                            [
                                np.zeros(len(known[0]), bool),
                                np.ones(len(unknown[0]), bool),
                            ]
                        )
                        for score_name in SCORES:
                            score = np.concatenate(
                                [
                                    known[3][score_name],
                                    unknown[3][score_name],
                                ]
                            )
                            metrics, _ = common.open_set_metrics(
                                manifest.iloc[index].reset_index(drop=True),
                                labels[index],
                                prediction,
                                probabilities,
                                trained.classes,
                                score,
                                unknown_mask,
                            )
                            open_rows.append(
                                {
                                    **context,
                                    "evaluation": evaluation,
                                    "score_name": score_name,
                                    **metrics,
                                }
                            )
                    payload = {
                        "fingerprint": fingerprint,
                        "classification": classification_rows,
                        "open_metrics": open_rows,
                        "predictions": prediction_rows,
                        "run_registry": [
                            {
                                **context,
                                "state_sha256": trained.state_sha256,
                                "parameter_count": trained.parameter_count,
                                "epochs_run": len(trained.history),
                                "n_development": len(development),
                                "n_development_masters": manifest.iloc[
                                    development
                                ]["master_sample_id"].nunique(),
                            }
                        ],
                    }
                    write_cache(path, payload)
                for key in output:
                    output[key].extend(payload[key])
                completed += 1
                if completed % 20 == 0 or completed == total:
                    print(f"SENSITIVITY CNN {completed}/{total}", flush=True)
    for name, rows in output.items():
        pd.DataFrame(rows).to_csv(
            args.cnn_dir / f"sensitivity_{name}.csv", index=False
        )


def main() -> None:
    args = parse_args()
    cnn_registry, control_registry, fingerprint = initialize(args)
    dataset = common.baseline.load_nato_dataset(args.bundle)
    device = torch.device(args.device)
    stages = (
        ("permutation", "sensitivity")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        print(f"START CNN CONTROL {stage}", flush=True)
        if stage == "permutation":
            run_permutation(
                args,
                dataset,
                cnn_registry,
                control_registry,
                fingerprint,
                device,
            )
        else:
            run_sensitivity(
                args,
                dataset,
                cnn_registry,
                control_registry,
                fingerprint,
                device,
            )
        print(f"DONE CNN CONTROL {stage}", flush=True)


if __name__ == "__main__":
    main()
