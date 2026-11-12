#!/usr/bin/env python3
"""Run the frozen source-only held-domain CNN factorial evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from itertools import combinations
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


DOMAIN_REGISTRY_VERSION = "sers-open-world-cnn-domain-v1"
MODEL_KEY = [
    "held_unknown",
    "outer_fold",
    "inner_fold",
    "surrogate_unknown",
    "variant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("epochs", "domain", "merge", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--cnn-registry",
        type=Path,
        default=Path("configs/sers_open_world_cnn_v1.json"),
    )
    parser.add_argument(
        "--domain-registry",
        type=Path,
        default=Path("configs/sers_open_world_cnn_domain_v1.json"),
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
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, shard-count)")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_domain_registry(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("registry_version") != DOMAIN_REGISTRY_VERSION:
        raise ValueError("Unexpected CNN domain registry")
    expected = (
        "predeclared_after_surrogate_execution_started_before_surrogate_metrics_"
        "or_any_cnn_locked_outcome_were_inspected"
    )
    if value.get("status_before_execution") != expected:
        raise ValueError("CNN domain registry was not frozen at the declared stage")
    return value


def initialize(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    args.cnn_dir.mkdir(parents=True, exist_ok=True)
    cnn_registry = cnn.load_registry(args.cnn_registry)
    domain_registry = load_domain_registry(args.domain_registry)
    for name, source in (
        ("predeclared_cnn_registry.json", args.cnn_registry),
        ("predeclared_cnn_domain_registry.json", args.domain_registry),
    ):
        target = args.cnn_dir / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Different immutable snapshot already exists: {name}")
        if not target.exists():
            shutil.copyfile(source, target)
    validation = json.loads(
        (args.partition_dir / "validation_report.json").read_text()
    )
    if validation["status"] != "pass":
        raise ValueError("Open-world partitions are not validated")
    provenance = {
        "cnn_registry_sha256": sha256_file(args.cnn_registry),
        "domain_registry_sha256": sha256_file(args.domain_registry),
        "domain_core_sha256": sha256_file(
            args.partition_dir / "domain_evaluation_partitions_core.csv"
        ),
        "domain_quality_sha256": sha256_file(
            args.partition_dir / "domain_evaluation_partitions_quality.csv"
        ),
    }
    common.write_json(args.cnn_dir / "domain_input_hashes.json", provenance)
    return cnn_registry, domain_registry


def freeze_epochs(
    args: argparse.Namespace, cnn_registry: dict[str, Any]
) -> pd.DataFrame:
    source = args.cnn_dir / "surrogate_metrics.csv"
    metrics = pd.read_csv(source)
    missing = set(MODEL_KEY + ["best_epoch", "state_sha256"]) - set(metrics)
    if missing:
        raise ValueError(f"Surrogate metrics lack columns: {sorted(missing)}")
    consistency = metrics.groupby(MODEL_KEY, dropna=False).agg(
        best_epochs=("best_epoch", "nunique"),
        states=("state_sha256", "nunique"),
        score_rows=("score_name", "nunique"),
    )
    if (
        (consistency["best_epochs"] != 1).any()
        or (consistency["states"] != 1).any()
        or (consistency["score_rows"] != 5).any()
    ):
        raise ValueError("Surrogate fitted-model rows are internally inconsistent")
    models = metrics.drop_duplicates(MODEL_KEY)
    expected = 600 * len(cnn.variants(cnn_registry))
    if len(models) != expected:
        raise ValueError(
            f"Expected {expected} unique surrogate fitted models, got {len(models)}"
        )
    rows: list[dict[str, Any]] = []
    for variant in cnn.variants(cnn_registry):
        frame = models[models["variant"] == variant.name]
        if len(frame) != 600:
            raise ValueError(
                f"Expected 600 fitted models for {variant.name}, got {len(frame)}"
            )
        median = float(frame["best_epoch"].median())
        selected = int(np.floor(median + 0.5))
        rows.append(
            {
                "variant_order": variant.order,
                "variant": variant.name,
                "representation": variant.representation,
                "surrogate_fitted_models": len(frame),
                "median_best_epoch": median,
                "selected_epoch": selected,
                "minimum_best_epoch": int(frame["best_epoch"].min()),
                "maximum_best_epoch": int(frame["best_epoch"].max()),
                "source_file": str(source),
                "source_sha256": sha256_file(source),
            }
        )
    epochs = pd.DataFrame(rows)
    epochs.to_csv(args.cnn_dir / "domain_epoch_registry.csv", index=False)
    return epochs


def uid_lookup(
    dataset: common.baseline.SpectralDataset,
) -> dict[str, int]:
    return {
        str(uid): index
        for index, uid in enumerate(
            dataset.manifest["observation_uid"].astype(str).to_numpy()
        )
    }


def scenario_indices(
    scenario: pd.DataFrame, lookup: dict[str, int], partition: str
) -> np.ndarray:
    return np.asarray(
        [
            lookup[str(uid)]
            for uid in scenario.loc[
                scenario["partition"] == partition, "observation_uid"
            ].astype(str)
        ],
        dtype=int,
    )


def slug(*parts: object) -> str:
    text = "__".join(str(part) for part in parts)
    return "".join(character if character.isalnum() else "_" for character in text)


def cache_paths(cache_dir: Path, context: dict[str, Any]) -> tuple[Path, Path]:
    stem = slug(
        context["subset"],
        context["protocol"],
        context["domain_type"],
        context["heldout_domain"],
        context["variant"],
        f"s{context['declared_seed']}",
    )
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def task_fingerprint(
    context: dict[str, Any],
    train_uids: np.ndarray,
    test_uids: np.ndarray,
    registry_hash: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
    )
    for values in (train_uids, test_uids):
        for value in values.astype(str):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    digest.update(registry_hash.encode())
    return digest.hexdigest()


def save_cache(
    array_path: Path,
    meta_path: Path,
    fingerprint: str,
    trained: cnn.Trained,
    test_indices: np.ndarray,
    embeddings: np.ndarray,
    logits: np.ndarray,
) -> None:
    temporary = array_path.with_suffix(".npz.partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            test_indices=np.asarray(test_indices, dtype=np.int64),
            embeddings=np.asarray(embeddings, dtype=np.float32),
            logits=np.asarray(logits, dtype=np.float32),
            classes=np.asarray(trained.classes, dtype=str),
        )
    temporary.replace(array_path)
    payload = {
        "fingerprint": fingerprint,
        "state_sha256": trained.state_sha256,
        "parameter_count": trained.parameter_count,
        "best_epoch": trained.best_epoch,
        "epochs_run": len(trained.history),
        "history": trained.history.to_dict("records"),
    }
    temporary_meta = meta_path.with_suffix(".json.partial")
    temporary_meta.write_text(json.dumps(common.json_clean(payload), indent=2))
    temporary_meta.replace(meta_path)


def load_or_fit(
    cache_dir: Path,
    context: dict[str, Any],
    fingerprint: str,
    values: np.ndarray,
    manifest: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    variant: cnn.Variant,
    cnn_registry: dict[str, Any],
    seed: int,
    epoch: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    array_path, meta_path = cache_paths(cache_dir, context)
    if array_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("fingerprint") != fingerprint:
            raise ValueError(f"Stale domain cache fingerprint: {meta_path}")
        cached = np.load(array_path, allow_pickle=False)
        cached_indices = cached["test_indices"].astype(int)
        if not np.array_equal(cached_indices, test_indices):
            raise ValueError(f"Stale domain cache indices: {array_path}")
        return (
            meta,
            cached_indices,
            cached["embeddings"].astype(np.float32, copy=False),
            cached["logits"].astype(np.float32, copy=False),
            cached["classes"].astype(str),
        )
    trained = cnn.train(
        values[train_indices],
        manifest.iloc[train_indices].reset_index(drop=True),
        None,
        None,
        variant,
        cnn_registry,
        seed,
        device,
        fixed_epochs=epoch,
    )
    embeddings, logits = cnn.deterministic_outputs(
        trained.model, values[test_indices], device
    )
    save_cache(
        array_path,
        meta_path,
        fingerprint,
        trained,
        test_indices,
        embeddings,
        logits,
    )
    return (
        json.loads(meta_path.read_text()),
        test_indices,
        np.asarray(embeddings, dtype=np.float32),
        np.asarray(logits, dtype=np.float32),
        np.asarray(trained.classes, dtype=str),
    )


def support_rows(
    context: dict[str, Any],
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
) -> list[dict[str, Any]]:
    train_counts = train_manifest["target_analyte"].astype(str).value_counts()
    test_counts = test_manifest["target_analyte"].astype(str).value_counts()
    labels = sorted(set(train_counts.index) | set(test_counts.index))
    return [
        {
            **context,
            "target_analyte": label,
            "train_rows": int(train_counts.get(label, 0)),
            "test_rows": int(test_counts.get(label, 0)),
            "test_class_supported": label in train_counts.index,
        }
        for label in labels
    ]


def run_domain(
    args: argparse.Namespace,
    dataset: common.baseline.SpectralDataset,
    cnn_registry: dict[str, Any],
    domain_registry: dict[str, Any],
    device: torch.device,
) -> None:
    epochs = pd.read_csv(args.cnn_dir / "domain_epoch_registry.csv")
    epoch_lookup = {
        str(row.variant): int(row.selected_epoch)
        for row in epochs.itertuples(index=False)
    }
    variants = cnn.variants(cnn_registry)
    lookup = uid_lookup(dataset)
    manifest = dataset.manifest
    labels = manifest["target_analyte"].astype(str).to_numpy()
    cache_dir = args.cnn_dir / "cache" / "domain"
    cache_dir.mkdir(parents=True, exist_ok=True)
    registry_hash = (
        sha256_file(args.cnn_registry) + ":" + sha256_file(args.domain_registry)
    )
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    selective_rows: list[dict[str, Any]] = []
    support_records: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    total = 2 * 28 * len(variants) * len(
        domain_registry["evaluation"]["final_seeds"]
    )
    local_total = len(
        range(args.shard_index, total, args.shard_count)
    )
    completed = 0
    task_index = 0
    for subset, filename in (
        ("strict_core", "domain_evaluation_partitions_core.csv"),
        ("quality_pass", "domain_evaluation_partitions_quality.csv"),
    ):
        frame = pd.read_csv(args.partition_dir / filename)
        for (protocol, domain_type, heldout), scenario in frame.groupby(
            ["protocol", "domain_type", "heldout_domain"], sort=True
        ):
            train_indices = scenario_indices(scenario, lookup, "train")
            test_indices = scenario_indices(scenario, lookup, "test")
            if len(train_indices) == 0 or len(test_indices) == 0:
                raise ValueError(
                    f"Empty domain partition: {subset}/{protocol}/{domain_type}/{heldout}"
                )
            train_manifest = manifest.iloc[train_indices].reset_index(drop=True)
            test_manifest = manifest.iloc[test_indices].reset_index(drop=True)
            support_context = {
                "subset": subset,
                "protocol": str(protocol),
                "domain_type": str(domain_type),
                "heldout_domain": str(heldout),
            }
            support_records.extend(
                support_rows(support_context, train_manifest, test_manifest)
            )
            for variant in variants:
                values = dataset.representations[variant.representation]
                epoch = epoch_lookup[variant.name]
                for declared_seed in domain_registry["evaluation"]["final_seeds"]:
                    assigned = (
                        task_index % args.shard_count
                        == args.shard_index
                    )
                    task_index += 1
                    if not assigned:
                        continue
                    seed = cnn.stable_seed(
                        DOMAIN_REGISTRY_VERSION,
                        subset,
                        protocol,
                        domain_type,
                        heldout,
                        variant.name,
                        declared_seed,
                    )
                    context = {
                        **support_context,
                        "variant_order": variant.order,
                        "variant": variant.name,
                        "representation": variant.representation,
                        "declared_seed": int(declared_seed),
                        "run_seed": seed,
                        "selected_epoch": epoch,
                        "temperature": 1.0,
                        "calibration_status": domain_registry["evaluation"][
                            "calibration_status"
                        ],
                    }
                    fingerprint = task_fingerprint(
                        context,
                        manifest.iloc[train_indices][
                            "observation_uid"
                        ].astype(str).to_numpy(),
                        manifest.iloc[test_indices][
                            "observation_uid"
                        ].astype(str).to_numpy(),
                        registry_hash,
                    )
                    meta, index, embedding, logits, classes = load_or_fit(
                        cache_dir,
                        context,
                        fingerprint,
                        values,
                        manifest,
                        train_indices,
                        test_indices,
                        variant,
                        cnn_registry,
                        seed,
                        epoch,
                        device,
                    )
                    probabilities = common.probabilities_from_scores(logits, 1.0)
                    predictions = classes[np.argmax(probabilities, axis=1)]
                    metrics = common.classification_metrics(
                        classes,
                        labels[index],
                        predictions,
                        probabilities,
                        classes,
                    )
                    metric_context = {
                        **context,
                        "n_train": len(train_indices),
                        "n_train_masters": train_manifest[
                            "master_sample_id"
                        ].nunique(),
                        "n_test_masters": test_manifest[
                            "master_sample_id"
                        ].nunique(),
                    }
                    metrics_rows.append(
                        {
                            **metric_context,
                            **{
                                key: value
                                for key, value in metrics.items()
                                if key != "supported_mask"
                            },
                        }
                    )
                    per_class_rows.extend(
                        common.classical.per_class_rows(
                            context,
                            labels[index],
                            predictions,
                            metrics["supported_mask"],
                        )
                    )
                    selective_rows.extend(
                        common.classical.selective_rows(
                            context,
                            labels[index],
                            predictions,
                            probabilities.max(axis=1),
                            metrics["supported_mask"],
                            domain_registry["evaluation"]["selective_coverages"],
                        )
                    )
                    for offset, row_index in enumerate(index):
                        row = manifest.iloc[row_index]
                        prediction_rows.append(
                            {
                                **context,
                                "observation_uid": str(row.observation_uid),
                                "master_sample_id": str(row.master_sample_id),
                                "target_analyte": labels[row_index],
                                "predicted_analyte": str(predictions[offset]),
                                "supported": bool(metrics["supported_mask"][offset]),
                                "correct": bool(
                                    predictions[offset] == labels[row_index]
                                ),
                                "instrument": str(row.instrument),
                                "sensor_family": str(row.sensor_family),
                                "confidence": float(
                                    probabilities[offset].max()
                                ),
                                "embedding_norm": float(
                                    np.linalg.norm(embedding[offset])
                                ),
                                "probability_classes_json": json.dumps(
                                    classes.tolist()
                                ),
                                "probabilities_json": json.dumps(
                                    probabilities[offset].tolist()
                                ),
                            }
                        )
                    run_rows.append(
                        {
                            **metric_context,
                            "state_sha256": meta["state_sha256"],
                            "parameter_count": int(meta["parameter_count"]),
                            "epochs_run": int(meta["epochs_run"]),
                            "fingerprint": fingerprint,
                        }
                    )
                    completed += 1
                    if completed % 20 == 0 or completed == local_total:
                        print(
                            f"DOMAIN CNN shard={args.shard_index}/"
                            f"{args.shard_count} {completed}/{local_total}",
                            flush=True,
                        )
    suffix = (
        ""
        if args.shard_count == 1
        else (
            f"__shard_{args.shard_index:02d}_of_"
            f"{args.shard_count:02d}"
        )
    )
    outputs = {
        "domain_metrics": pd.DataFrame(metrics_rows),
        "domain_predictions": pd.DataFrame(prediction_rows),
        "domain_per_class_metrics": pd.DataFrame(per_class_rows),
        "domain_selective_metrics": pd.DataFrame(selective_rows),
        "domain_support_audit": (
            pd.DataFrame(support_records).drop_duplicates()
        ),
        "domain_run_registry": pd.DataFrame(run_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(args.cnn_dir / f"{name}{suffix}.csv", index=False)


def merge_domain_shards(args: argparse.Namespace) -> None:
    if args.shard_count <= 1:
        raise ValueError("Domain merge requires --shard-count greater than 1")
    names = (
        "domain_metrics",
        "domain_predictions",
        "domain_per_class_metrics",
        "domain_selective_metrics",
        "domain_support_audit",
        "domain_run_registry",
    )
    merged: dict[str, pd.DataFrame] = {}
    for name in names:
        paths = [
            args.cnn_dir
            / (
                f"{name}__shard_{index:02d}_of_"
                f"{args.shard_count:02d}.csv"
            )
            for index in range(args.shard_count)
        ]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise ValueError(
                f"Missing {name} domain shards: {missing}"
            )
        frame = pd.concat(
            [pd.read_csv(path) for path in paths], ignore_index=True
        )
        if name == "domain_support_audit":
            frame = frame.drop_duplicates()
        merged[name] = frame
        frame.to_csv(args.cnn_dir / f"{name}.csv", index=False)
    scenario_keys = [
        "subset",
        "protocol",
        "domain_type",
        "heldout_domain",
    ]
    if (
        len(merged["domain_metrics"]) != 1176
        or len(merged["domain_run_registry"]) != 1176
        or merged["domain_support_audit"].groupby(
            scenario_keys
        ).ngroups
        != 56
    ):
        raise ValueError("Merged domain shard dimensions are incomplete")


def agreement_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    instrument = predictions[
        (predictions["domain_type"] == "instrument")
        & predictions["supported"].astype(bool)
    ].copy()
    keys = ["subset", "protocol", "variant", "declared_seed"]
    for values, frame in instrument.groupby(keys, sort=True):
        agreements: list[bool] = []
        master_count = 0
        for _, group in frame.groupby("master_sample_id", sort=True):
            pairs = [
                (left, right)
                for left, right in combinations(range(len(group)), 2)
                if str(group.iloc[left]["instrument"])
                != str(group.iloc[right]["instrument"])
            ]
            if not pairs:
                continue
            master_count += 1
            prediction = group["predicted_analyte"].astype(str).to_numpy()
            agreements.extend(prediction[left] == prediction[right] for left, right in pairs)
        rows.append(
            {
                **dict(zip(keys, values)),
                "masters_with_cross_instrument_pairs": master_count,
                "cross_instrument_pairs": len(agreements),
                "prediction_agreement": (
                    float(np.mean(agreements)) if agreements else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def finalize(
    args: argparse.Namespace, domain_registry: dict[str, Any]
) -> None:
    metrics = pd.read_csv(args.cnn_dir / "domain_metrics.csv")
    predictions = pd.read_csv(args.cnn_dir / "domain_predictions.csv")
    seed_means = (
        metrics.groupby(
            [
                "subset",
                "protocol",
                "domain_type",
                "heldout_domain",
                "variant_order",
                "variant",
                "representation",
            ],
            as_index=False,
        )
        .agg(
            technical_seeds=("declared_seed", "nunique"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            accuracy=("accuracy", "mean"),
            negative_log_likelihood=("negative_log_likelihood", "mean"),
            brier_multiclass=("brier_multiclass", "mean"),
            expected_calibration_error_10=(
                "expected_calibration_error_10",
                "mean",
            ),
            n_supported=("n_supported", "first"),
            n_unsupported=("n_unsupported", "first"),
        )
    )
    seed_means.to_csv(
        args.cnn_dir / "domain_seed_averaged_metrics.csv", index=False
    )
    summaries = (
        seed_means.groupby(
            [
                "subset",
                "protocol",
                "domain_type",
                "variant_order",
                "variant",
                "representation",
            ],
            as_index=False,
        )
        .agg(
            held_domains_total=("heldout_domain", "nunique"),
            held_domains_with_supported_metric=(
                "balanced_accuracy",
                "count",
            ),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            worst_balanced_accuracy=("balanced_accuracy", "min"),
            standard_deviation_across_domains=("balanced_accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_accuracy=("accuracy", "mean"),
        )
    )
    summaries.to_csv(args.cnn_dir / "domain_summary.csv", index=False)
    agreement_summary(predictions).to_csv(
        args.cnn_dir / "domain_same_master_agreement.csv", index=False
    )
    contrast_rows: list[dict[str, Any]] = []
    for comparison in domain_registry["registered_contrasts"]:
        candidate = str(comparison["candidate"])
        reference = str(comparison["reference"])
        left = seed_means[seed_means["variant"] == candidate]
        right = seed_means[seed_means["variant"] == reference]
        join = [
            "subset",
            "protocol",
            "domain_type",
            "heldout_domain",
        ]
        paired = left.merge(
            right,
            on=join,
            suffixes=("_candidate", "_reference"),
            validate="one_to_one",
        ).dropna(
            subset=[
                "balanced_accuracy_candidate",
                "balanced_accuracy_reference",
            ]
        )
        for keys, frame in paired.groupby(
            ["subset", "protocol", "domain_type"], sort=True
        ):
            differences = (
                frame["balanced_accuracy_candidate"]
                - frame["balanced_accuracy_reference"]
            )
            contrast_rows.append(
                {
                    **dict(zip(["subset", "protocol", "domain_type"], keys)),
                    "candidate": candidate,
                    "reference": reference,
                    "held_domains": len(frame),
                    "mean_paired_balanced_accuracy_difference": float(
                        differences.mean()
                    ),
                    "worst_paired_balanced_accuracy_difference": float(
                        differences.min()
                    ),
                    "domains_improved": int((differences > 0).sum()),
                    "domains_tied": int((differences == 0).sum()),
                    "domains_worsened": int((differences < 0).sum()),
                }
            )
    pd.DataFrame(contrast_rows).to_csv(
        args.cnn_dir / "domain_registered_contrasts.csv", index=False
    )


def main() -> None:
    args = parse_args()
    cnn_registry, domain_registry = initialize(args)
    stages = (
        ("epochs", "domain", "finalize")
        if args.stage == "all"
        else (args.stage,)
    )
    dataset = None
    for stage in stages:
        print(f"START CNN DOMAIN {stage}", flush=True)
        if stage == "epochs":
            freeze_epochs(args, cnn_registry)
        elif stage == "domain":
            if dataset is None:
                dataset = common.baseline.load_nato_dataset(args.bundle)
            run_domain(
                args,
                dataset,
                cnn_registry,
                domain_registry,
                torch.device(args.device),
            )
        elif stage == "merge":
            merge_domain_shards(args)
        else:
            finalize(args, domain_registry)
        print(f"DONE CNN DOMAIN {stage}", flush=True)


if __name__ == "__main__":
    main()
