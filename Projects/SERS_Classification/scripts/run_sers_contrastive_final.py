#!/usr/bin/env python3
"""Run locked outer and domain evaluation for supervised-contrastive SERS."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

import sers_baseline_common as baseline
import sers_classical_benchmark_common as classical
import sers_contrastive_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/sers_supervised_contrastive_v1.json"),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("Workspace/nato_sers_field_trial/preprocessing_v2"),
    )
    parser.add_argument(
        "--classical-bundle",
        type=Path,
        default=Path(
            "Workspace/sers_classical_benchmark/classical_benchmark_v2"
        ),
    )
    parser.add_argument(
        "--prior-bundle",
        type=Path,
        default=Path(
            "Workspace/sers_representation_baselines/baselines_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Workspace/sers_supervised_contrastive/contrastive_v1"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("outer", "domain", "negative", "all"),
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            classical.json_clean(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def initialize(
    args: argparse.Namespace, protocol: dict[str, Any]
) -> baseline.SpectralDataset:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = args.output_dir / "predeclared_protocol.json"
    if copied.exists() and copied.read_bytes() != args.protocol.read_bytes():
        raise ValueError("Output directory contains another protocol snapshot")
    if not copied.exists():
        shutil.copyfile(args.protocol, copied)
    dataset = baseline.load_nato_dataset(args.bundle)
    for required in (
        "stage1_selection.csv",
        "stage2_selection.csv",
        "stage1_inner_metrics.csv",
        "stage2_inner_metrics.csv",
    ):
        if not (args.output_dir / required).exists():
            raise ValueError(f"Contrastive selection is incomplete: {required}")
    if json.loads(
        (args.classical_bundle / "validation_report.json").read_text()
    ).get("status") != "pass":
        raise ValueError("Classical bundle is not validated")
    if json.loads(
        (args.classical_bundle / "clean_rebuild_comparison.json").read_text()
    ).get("status") != "exact_match":
        raise ValueError("Classical exact rebuild is not closed")
    if json.loads(
        (args.prior_bundle / "validation_report.json").read_text()
    ).get("status") != "pass":
        raise ValueError("Prior Siamese bundle is not validated")
    script_paths = [
        Path(__file__).resolve(),
        Path(common.__file__).resolve(),
        Path(classical.__file__).resolve(),
        Path("scripts/finalize_sers_contrastive.py").resolve(),
        Path("scripts/validate_sers_contrastive.py").resolve(),
    ]
    write_json(
        args.output_dir / "final_input_hashes.json",
        {
            "protocol_sha256": baseline.sha256_file(args.protocol),
            "preprocessing_hash_catalog_sha256": baseline.sha256_file(
                args.bundle / "artifact_hashes.json"
            ),
            "classical_hash_catalog_sha256": baseline.sha256_file(
                args.classical_bundle / "artifact_hashes.json"
            ),
            "prior_siamese_hash_catalog_sha256": baseline.sha256_file(
                args.prior_bundle / "artifact_hashes.json"
            ),
            "scripts": {
                str(path): baseline.sha256_file(path)
                for path in script_paths
            },
        },
    )
    write_json(
        args.output_dir / "final_environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    )
    return dataset


def selected_spec(
    output_dir: Path, outer_fold: int | None
) -> tuple[common.ModelSpec, int]:
    selection = pd.read_csv(output_dir / "stage2_selection.csv")
    if outer_fold is None:
        row = selection[selection["scope"] == "global"].iloc[0]
    else:
        row = selection[
            (selection["scope"] == "outer")
            & (selection["outer_fold"] == outer_fold)
        ].iloc[0]
    spec = common.ModelSpec(
        name="full_domain_aware",
        representation=str(row["representation"]),
        architecture=str(row["architecture"]),
        embedding_dimension=int(row["embedding_dimension"]),
        classification_weight=float(row["classification_weight"]),
        supervised_contrastive_weight=float(
            row["supervised_contrastive_weight"]
        ),
        pair_margin_weight=float(row["pair_margin_weight"]),
        contrastive_temperature=float(row["contrastive_temperature"]),
        pair_margin=float(row["pair_margin"]),
        domain_aware_positives=bool(row["domain_aware_positives"]),
        hard_negative_mining=bool(row["hard_negative_mining"]),
        domain_balanced_batches=bool(row["domain_balanced_batches"]),
    )
    epochs = int(np.clip(round(float(row["mean_best_epoch"])), 50, 200))
    return spec, epochs


def variant_specs(full: common.ModelSpec, representations: list[str]) -> list[common.ModelSpec]:
    shared = {
        "representation": full.representation,
        "architecture": full.architecture,
        "embedding_dimension": full.embedding_dimension,
        "contrastive_temperature": full.contrastive_temperature,
        "pair_margin": full.pair_margin,
    }
    variants = [
        full,
        common.ModelSpec(
            name="cross_entropy_only",
            **shared,
            classification_weight=1.0,
        ),
        common.ModelSpec(
            name="supervised_contrastive_only",
            **shared,
            classification_weight=0.0,
            supervised_contrastive_weight=1.0,
        ),
        common.ModelSpec(
            name="ce_plus_supervised_contrastive",
            **shared,
            classification_weight=1.0,
            supervised_contrastive_weight=0.5,
        ),
        common.ModelSpec(
            name="random_positive_ablation",
            **shared,
            classification_weight=full.classification_weight,
            supervised_contrastive_weight=full.supervised_contrastive_weight,
            pair_margin_weight=full.pair_margin_weight,
            domain_aware_positives=False,
            hard_negative_mining=True,
            domain_balanced_batches=True,
        ),
        common.ModelSpec(
            name="ordinary_negative_ablation",
            **shared,
            classification_weight=full.classification_weight,
            supervised_contrastive_weight=full.supervised_contrastive_weight,
            pair_margin_weight=full.pair_margin_weight,
            domain_aware_positives=True,
            hard_negative_mining=False,
            domain_balanced_batches=True,
        ),
        common.ModelSpec(
            name="no_domain_balanced_batches",
            **shared,
            classification_weight=full.classification_weight,
            supervised_contrastive_weight=full.supervised_contrastive_weight,
            pair_margin_weight=full.pair_margin_weight,
            domain_aware_positives=True,
            hard_negative_mining=True,
            domain_balanced_batches=False,
        ),
    ]
    for representation in representations:
        if representation == full.representation:
            continue
        sensitivity_parameters = {
            key: value
            for key, value in full.__dict__.items()
            if key not in {"name", "representation"}
        }
        variants.append(
            common.ModelSpec(
                name=f"preprocessing_sensitivity_{representation}",
                representation=representation,
                **sensitivity_parameters,
            )
        )
    return variants


def cache_stem(
    scenario: str, spec: common.ModelSpec, seed: int, epochs: int
) -> str:
    suffix = hashlib_sha(
        json.dumps(
            {
                "scenario": scenario,
                "candidate_id": spec.candidate_id,
                "seed": seed,
                "epochs": epochs,
            },
            sort_keys=True,
        )
    )
    return f"{scenario}__{spec.name}__{suffix[:16]}"


def hashlib_sha(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def train_cached(
    output_dir: Path,
    scenario: str,
    values: np.ndarray,
    manifest: pd.DataFrame,
    spec: common.ModelSpec,
    protocol: dict[str, Any],
    seed: int,
    epochs: int,
    device: torch.device,
) -> common.TrainedModel:
    stem = cache_stem(scenario, spec, seed, epochs)
    cache_dir = output_dir / "training_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / f"{stem}.pt"
    metadata_path = cache_dir / f"{stem}.json"
    history_path = cache_dir / f"{stem}__history.csv"
    class_names = np.asarray(
        sorted(manifest["target_analyte"].astype(str).unique()), dtype=str
    )
    if checkpoint.exists() and metadata_path.exists() and history_path.exists():
        metadata = json.loads(metadata_path.read_text())
        model = common.ContrastiveClassifier(
            values.shape[1],
            len(class_names),
            spec.architecture,
            spec.embedding_dimension,
        ).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        return common.TrainedModel(
            model=model,
            spec=spec,
            history=pd.read_csv(history_path),
            best_epoch=int(metadata["best_epoch"]),
            run_seed=seed,
            state_sha256=str(metadata["state_sha256"]),
            parameters={
                key: int(value)
                for key, value in metadata["parameters"].items()
            },
            classes=class_names,
        )
    trained = common.train(
        values,
        manifest,
        None,
        None,
        spec,
        protocol,
        seed,
        device,
        fixed_epochs=epochs,
    )
    state = {
        key: value.detach().cpu().clone()
        for key, value in trained.model.state_dict().items()
    }
    torch.save(state, checkpoint)
    trained.history.to_csv(history_path, index=False)
    write_json(
        metadata_path,
        {
            "scenario": scenario,
            "spec": spec.__dict__,
            "candidate_id": spec.candidate_id,
            "seed": seed,
            "best_epoch": trained.best_epoch,
            "state_sha256": trained.state_sha256,
            "parameters": trained.parameters,
        },
    )
    return trained


def calibration_temperature(
    output_dir: Path,
    dataset: baseline.SpectralDataset,
    protocol: dict[str, Any],
    full: common.ModelSpec,
    epochs: int,
    subset: str,
    outer_fold: int,
    device: torch.device,
) -> tuple[float, str, dict[str, Any], list[dict[str, Any]]]:
    manifest = dataset.manifest
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    eligible = (
        np.ones(len(manifest), dtype=bool)
        if subset == "strict_core"
        else manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    )
    development = eligible & (folds != outer_fold)
    classes = np.asarray(
        sorted(manifest.loc[development, "target_analyte"].astype(str).unique()),
        dtype=str,
    )
    score_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    energy_confidence_parts: list[np.ndarray] = []
    mahalanobis_confidence_parts: list[np.ndarray] = []
    for inner_fold in sorted(set(range(5)) - {outer_fold}):
        train_mask = development & (folds != inner_fold)
        validation_mask = development & (folds == inner_fold)
        seed = common.stable_seed(
            common.PROTOCOL_VERSION,
            "calibration",
            subset,
            outer_fold,
            inner_fold,
            full.candidate_id,
            protocol["fixed_training"]["selection_seed"],
        )
        scenario = f"calibration_{subset}_o{outer_fold}_i{inner_fold}"
        trained = train_cached(
            output_dir,
            scenario,
            dataset.representations[full.representation][train_mask],
            manifest.loc[train_mask].reset_index(drop=True),
            full,
            protocol,
            seed,
            epochs,
            device,
        )
        scores, run_classes, train_embedding, validation_embedding = (
            common.prediction_scores(
            trained.model,
            full,
            dataset.representations[full.representation][train_mask],
            manifest.loc[train_mask, "target_analyte"].astype(str).to_numpy(),
            dataset.representations[full.representation][validation_mask],
            device,
            )
        )
        aligned_scores = classical.align_scores(scores, run_classes, classes)
        score_parts.append(aligned_scores)
        label_parts.append(
            manifest.loc[validation_mask, "target_analyte"]
            .astype(str)
            .to_numpy()
        )
        energy_confidence_parts.append(-common.energy_score(aligned_scores))
        mahalanobis_model = common.fit_class_mahalanobis(
            train_embedding,
            manifest.loc[train_mask, "target_analyte"].astype(str).to_numpy(),
        )
        mahalanobis_confidence_parts.append(
            -common.mahalanobis_scores(
                mahalanobis_model, validation_embedding
            )
        )
    scores = np.vstack(score_parts)
    labels = np.concatenate(label_parts)
    temperature = classical.fit_temperature(
        scores, labels, classes, (0.05, 20.0)
    )
    raw = calibration_summary(scores, labels, classes, 1.0)
    scaled = calibration_summary(scores, labels, classes, temperature)
    probabilities = classical.probabilities_from_scores(scores, temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    confidence_sets = {
        "calibrated_max_probability": np.max(probabilities, axis=1),
        "energy": np.concatenate(energy_confidence_parts),
        "class_conditional_embedding_mahalanobis": np.concatenate(
            mahalanobis_confidence_parts
        ),
    }
    selection_rows: list[dict[str, Any]] = []
    utilities: dict[str, float] = {}
    coverages = protocol["calibration_and_ood"]["selective_coverages"]
    supported = np.isin(labels, classes)
    for score_name in protocol["calibration_and_ood"][
        "rejection_selection"
    ]["tie_break_order"]:
        rows = classical.selective_rows(
            {
                "subset": subset,
                "outer_fold": outer_fold,
                "rejection_score": score_name,
            },
            labels,
            predictions,
            confidence_sets[score_name],
            supported,
            coverages,
        )
        utility_values = [
            float(row["accuracy"])
            for row in rows
            if float(row["requested_coverage"]) < 1.0
        ]
        utilities[score_name] = float(np.mean(utility_values))
        selection_rows.extend(rows)
    tie_order = protocol["calibration_and_ood"]["rejection_selection"][
        "tie_break_order"
    ]
    selected_rejection = max(
        tie_order,
        key=lambda name: (utilities[name], -tie_order.index(name)),
    )
    for row in selection_rows:
        row["selection_utility"] = utilities[row["rejection_score"]]
        row["selected"] = row["rejection_score"] == selected_rejection
    return temperature, selected_rejection, {
        "n_calibration": len(labels),
        "raw_nll": raw["negative_log_likelihood"],
        "calibrated_nll": scaled["negative_log_likelihood"],
        "raw_ece": raw["expected_calibration_error_10"],
        "calibrated_ece": scaled["expected_calibration_error_10"],
        "selected_rejection_score": selected_rejection,
        "selected_rejection_utility": utilities[selected_rejection],
    }, selection_rows


def calibration_summary(
    scores: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    temperature: float,
) -> dict[str, Any]:
    probabilities = classical.probabilities_from_scores(scores, temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    return classical.classification_metrics(
        classes, labels, predictions, probabilities, classes
    )


def evaluate_model(
    trained: common.TrainedModel,
    spec: common.ModelSpec,
    train_values: np.ndarray,
    train_manifest: pd.DataFrame,
    test_values: np.ndarray,
    test_manifest: pd.DataFrame,
    temperature: float,
    device: torch.device,
) -> dict[str, Any]:
    train_labels = train_manifest["target_analyte"].astype(str).to_numpy()
    test_labels = test_manifest["target_analyte"].astype(str).to_numpy()
    scores, classes, train_embedding, test_embedding = common.prediction_scores(
        trained.model,
        spec,
        train_values,
        train_labels,
        test_values,
        device,
    )
    probabilities = classical.probabilities_from_scores(scores, temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    metrics = classical.classification_metrics(
        classes, test_labels, predictions, probabilities, classes
    )
    geometry = common.pair_geometry(test_embedding, test_manifest)
    probes = {
        "analyte_probe_balanced_accuracy": common.domain_probe(
            train_embedding,
            train_manifest["target_analyte"].astype(str),
            test_embedding,
            test_manifest["target_analyte"].astype(str),
            common.stable_seed(trained.run_seed, "analyte_probe"),
        ),
        "instrument_probe_balanced_accuracy": common.domain_probe(
            train_embedding,
            train_manifest["instrument"].astype(str),
            test_embedding,
            test_manifest["instrument"].astype(str),
            common.stable_seed(trained.run_seed, "instrument_probe"),
        ),
        "sensor_probe_balanced_accuracy": common.domain_probe(
            train_embedding,
            train_manifest["sensor_family"].astype(str),
            test_embedding,
            test_manifest["sensor_family"].astype(str),
            common.stable_seed(trained.run_seed, "sensor_probe"),
        ),
    }
    if spec.name == "full_domain_aware":
        test_groups = test_manifest["master_sample_id"].astype(str)
        for probe_name, target_column in (
            ("analyte", "target_analyte"),
            ("instrument", "instrument"),
            ("sensor", "sensor_family"),
        ):
            grouped_probe = common.leave_one_group_out_probe(
                test_embedding,
                test_manifest[target_column].astype(str),
                test_groups,
                common.stable_seed(
                    trained.run_seed, probe_name, "heldout_master_probe"
                ),
            )
            probes.update(
                {
                    f"{probe_name}_heldout_master_probe_balanced_accuracy": (
                        grouped_probe["balanced_accuracy"]
                    ),
                    f"{probe_name}_heldout_master_probe_n_supported": (
                        grouped_probe["n_supported"]
                    ),
                    f"{probe_name}_heldout_master_probe_supported_fraction": (
                        grouped_probe["supported_fraction"]
                    ),
                }
            )
    return {
        "scores": scores,
        "classes": classes,
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": metrics,
        "geometry": geometry,
        "probes": probes,
        "train_embedding": train_embedding,
        "test_embedding": test_embedding,
    }


def prediction_rows(
    context: dict[str, Any],
    manifest: pd.DataFrame,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = manifest["target_analyte"].astype(str).to_numpy()
    probabilities = result["probabilities"]
    predictions = result["predictions"]
    supported = result["metrics"]["supported_mask"]
    classes = result["classes"]
    rows: list[dict[str, Any]] = []
    for index in range(len(manifest)):
        rows.append(
            {
                **context,
                "observation_uid": str(manifest.iloc[index]["observation_uid"]),
                "master_sample_id": str(
                    manifest.iloc[index]["master_sample_id"]
                ),
                "target_analyte": labels[index],
                "predicted_analyte": predictions[index],
                "correct": bool(labels[index] == predictions[index]),
                "supported": bool(supported[index]),
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


def metrics_row(
    context: dict[str, Any],
    result: dict[str, Any],
    trained: common.TrainedModel,
) -> dict[str, Any]:
    return {
        **context,
        **{
            key: value
            for key, value in result["metrics"].items()
            if key != "supported_mask"
        },
        **result["geometry"],
        **result["probes"],
        **trained.parameters,
        "state_sha256": trained.state_sha256,
    }


def run_outer(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    dataset: baseline.SpectralDataset,
    device: torch.device,
) -> None:
    manifest = dataset.manifest
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    representations = list(protocol["inputs"]["representations"])
    final_seeds = [int(value) for value in protocol["fixed_training"]["final_seeds"]]
    metric_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    selective: list[dict[str, Any]] = []
    temperatures: list[dict[str, Any]] = []
    rejection_selection_rows: list[dict[str, Any]] = []
    rejection_decisions: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    attribution_metrics: list[dict[str, Any]] = []
    attribution_peaks: list[dict[str, Any]] = []
    for outer_fold in range(5):
        full, epochs = selected_spec(args.output_dir, outer_fold)
        variants = variant_specs(full, representations)
        for subset in ("strict_core", "quality_pass"):
            eligible = (
                np.ones(len(manifest), dtype=bool)
                if subset == "strict_core"
                else manifest["include_sers_qc_pass"].astype(bool).to_numpy()
            )
            train_mask = eligible & (folds != outer_fold)
            evaluation_masks = [(subset, eligible & (folds == outer_fold))]
            if subset == "quality_pass":
                evaluation_masks.append(
                    (
                        "field_quality_stress",
                        manifest["field_quality_stress"].astype(bool).to_numpy()
                        & (folds == outer_fold),
                    )
                )
            (
                temperature,
                selected_rejection,
                calibration,
                calibration_selective,
            ) = calibration_temperature(
                args.output_dir,
                dataset,
                protocol,
                full,
                epochs,
                subset,
                outer_fold,
                device,
            )
            rejection_selection_rows.extend(calibration_selective)
            temperatures.append(
                {
                    "subset": subset,
                    "outer_fold": outer_fold,
                    "candidate_id": full.candidate_id,
                    "temperature": temperature,
                    "epochs": epochs,
                    **calibration,
                }
            )
            for spec in variants:
                for declared_seed in final_seeds:
                    seed = common.stable_seed(
                        common.PROTOCOL_VERSION,
                        "outer_final",
                        subset,
                        outer_fold,
                        spec.candidate_id,
                        declared_seed,
                    )
                    scenario = (
                        f"outer_{subset}_o{outer_fold}_seed{declared_seed}"
                    )
                    train_values = dataset.representations[
                        spec.representation
                    ][train_mask]
                    train_manifest = manifest.loc[train_mask].reset_index(
                        drop=True
                    )
                    trained = train_cached(
                        args.output_dir,
                        scenario,
                        train_values,
                        train_manifest,
                        spec,
                        protocol,
                        seed,
                        epochs,
                        device,
                    )
                    registry.append(
                        {
                            "stage": "outer",
                            "training_subset": subset,
                            "outer_fold": outer_fold,
                            "variant": spec.name,
                            "candidate_id": spec.candidate_id,
                            "declared_seed": declared_seed,
                            "run_seed": seed,
                            "epochs": epochs,
                            "temperature": (
                                temperature
                                if spec.name == "full_domain_aware"
                                else 1.0
                            ),
                            **spec.__dict__,
                            **trained.parameters,
                            "state_sha256": trained.state_sha256,
                        }
                    )
                    evaluation_results: dict[str, dict[str, Any]] = {}
                    for evaluation_subset, test_mask in evaluation_masks:
                        test_manifest = manifest.loc[test_mask].reset_index(
                            drop=True
                        )
                        applied_temperature = (
                            temperature
                            if spec.name == "full_domain_aware"
                            else 1.0
                        )
                        result = evaluate_model(
                            trained,
                            spec,
                            train_values,
                            train_manifest,
                            dataset.representations[spec.representation][
                                test_mask
                            ],
                            test_manifest,
                            applied_temperature,
                            device,
                        )
                        evaluation_results[evaluation_subset] = result
                        context = {
                            "stage": "outer",
                            "training_subset": subset,
                            "evaluation_subset": evaluation_subset,
                            "outer_fold": outer_fold,
                            "variant": spec.name,
                            "candidate_id": spec.candidate_id,
                            "representation": spec.representation,
                            "architecture": spec.architecture,
                            "embedding_dimension": spec.embedding_dimension,
                            "declared_seed": declared_seed,
                            "run_seed": seed,
                            "epochs": epochs,
                            "temperature": applied_temperature,
                        }
                        metric_rows.append(
                            metrics_row(context, result, trained)
                        )
                        predictions.extend(
                            prediction_rows(context, test_manifest, result)
                        )
                        per_class.extend(
                            classical.per_class_rows(
                                context,
                                test_manifest["target_analyte"]
                                .astype(str)
                                .to_numpy(),
                                result["predictions"],
                                result["metrics"]["supported_mask"],
                            )
                        )
                        confidence_sets = {
                            "calibrated_max_probability": np.max(
                                result["probabilities"], axis=1
                            )
                        }
                        if spec.name == "full_domain_aware":
                            mahalanobis_model = common.fit_class_mahalanobis(
                                result["train_embedding"],
                                train_manifest["target_analyte"]
                                .astype(str)
                                .to_numpy(),
                            )
                            confidence_sets.update(
                                {
                                    "energy": -common.energy_score(
                                        result["scores"]
                                    ),
                                    "class_conditional_embedding_mahalanobis": (
                                        -common.mahalanobis_scores(
                                            mahalanobis_model,
                                            result["test_embedding"],
                                        )
                                    ),
                                }
                            )
                        for rejection_score, confidence in confidence_sets.items():
                            rejection_context = {
                                **context,
                                "rejection_score": rejection_score,
                                "rejection_selected": (
                                    spec.name == "full_domain_aware"
                                    and rejection_score
                                    == selected_rejection
                                ),
                                "rejection_selection_source": (
                                    "cross_fitted_development"
                                    if spec.name == "full_domain_aware"
                                    else "control_not_selected"
                                ),
                            }
                            selective.extend(
                                classical.selective_rows(
                                    rejection_context,
                                    test_manifest["target_analyte"]
                                    .astype(str)
                                    .to_numpy(),
                                    result["predictions"],
                                    confidence,
                                    result["metrics"]["supported_mask"],
                                    protocol["calibration_and_ood"][
                                        "selective_coverages"
                                    ],
                                )
                            )
                            if (
                                spec.name == "full_domain_aware"
                                and rejection_score
                                == selected_rejection
                            ):
                                supported_mask = np.asarray(
                                    result["metrics"]["supported_mask"],
                                    dtype=bool,
                                )
                                supported_indices = np.flatnonzero(
                                    supported_mask
                                )
                                ordered_supported = supported_indices[
                                    np.argsort(
                                        -np.asarray(confidence)[
                                            supported_indices
                                        ],
                                        kind="stable",
                                    )
                                ]
                                retained = max(
                                    1,
                                    int(
                                        math.ceil(
                                            0.8 * len(ordered_supported)
                                        )
                                    ),
                                )
                                accepted = np.zeros(
                                    len(test_manifest), dtype=bool
                                )
                                accepted[
                                    ordered_supported[:retained]
                                ] = True
                                for row_index in range(
                                    len(test_manifest)
                                ):
                                    rejection_decisions.append(
                                        {
                                            **context,
                                            "observation_uid": str(
                                                test_manifest.iloc[
                                                    row_index
                                                ]["observation_uid"]
                                            ),
                                            "master_sample_id": str(
                                                test_manifest.iloc[
                                                    row_index
                                                ]["master_sample_id"]
                                            ),
                                            "target_analyte": str(
                                                test_manifest.iloc[
                                                    row_index
                                                ]["target_analyte"]
                                            ),
                                            "predicted_analyte": str(
                                                result["predictions"][
                                                    row_index
                                                ]
                                            ),
                                            "correct": bool(
                                                result["predictions"][
                                                    row_index
                                                ]
                                                == str(
                                                    test_manifest.iloc[
                                                        row_index
                                                    ]["target_analyte"]
                                                )
                                            ),
                                            "supported": bool(
                                                supported_mask[row_index]
                                            ),
                                            "rejection_score": (
                                                rejection_score
                                            ),
                                            "selection_source": (
                                                "cross_fitted_development"
                                            ),
                                            "requested_coverage": 0.8,
                                            "acceptance_confidence": float(
                                                confidence[row_index]
                                            ),
                                            "accepted": bool(
                                                accepted[row_index]
                                            ),
                                            "decision": (
                                                "accept"
                                                if accepted[row_index]
                                                else (
                                                    "reject_unsupported_class"
                                                    if not supported_mask[
                                                        row_index
                                                    ]
                                                    else "reject_low_confidence"
                                                )
                                            ),
                                            "instrument": str(
                                                test_manifest.iloc[
                                                    row_index
                                                ]["instrument"]
                                            ),
                                            "sensor_family": str(
                                                test_manifest.iloc[
                                                    row_index
                                                ]["sensor_family"]
                                            ),
                                        }
                                    )
                        if (
                            spec.name == "full_domain_aware"
                            and declared_seed
                            == int(protocol["fixed_training"]["final_seeds"][0])
                        ):
                            test_labels = (
                                test_manifest["target_analyte"]
                                .astype(str)
                                .to_numpy()
                            )
                            class_index = {
                                label: index
                                for index, label in enumerate(result["classes"])
                            }
                            fallback = np.argmax(result["scores"], axis=1)
                            target_indices = np.asarray(
                                [
                                    class_index.get(label, int(fallback[index]))
                                    for index, label in enumerate(test_labels)
                                ],
                                dtype=int,
                            )
                            test_values = dataset.representations[
                                spec.representation
                            ][test_mask]
                            attributions = common.integrated_gradients(
                                trained.model,
                                test_values,
                                target_indices,
                                device,
                                steps=16,
                            )
                            attribution_metrics.append(
                                {
                                    **context,
                                    **common.attribution_peak_stability(
                                        attributions, test_manifest
                                    ),
                                }
                            )
                            axis = dataset.axis_cm1
                            for spectrum_index, attribution in enumerate(
                                np.abs(attributions)
                            ):
                                top = np.argsort(attribution)[-30:][::-1]
                                for rank_index, axis_index in enumerate(top):
                                    attribution_peaks.append(
                                        {
                                            **context,
                                            "observation_uid": str(
                                                test_manifest.iloc[
                                                    spectrum_index
                                                ]["observation_uid"]
                                            ),
                                            "target_analyte": test_labels[
                                                spectrum_index
                                            ],
                                            "rank": rank_index + 1,
                                            "wavenumber_cm1": float(
                                                axis[axis_index]
                                            ),
                                            "absolute_attribution": float(
                                                attribution[axis_index]
                                            ),
                                        }
                                    )
                    if (
                        subset == "quality_pass"
                        and spec.name == "full_domain_aware"
                    ):
                        quality = evaluation_results["quality_pass"]
                        stress = evaluation_results["field_quality_stress"]
                        mahalanobis = common.fit_class_mahalanobis(
                            quality["train_embedding"],
                            train_manifest["target_analyte"]
                            .astype(str)
                            .to_numpy(),
                        )
                        score_sets = {
                            "one_minus_max_probability": (
                                1.0
                                - np.max(quality["probabilities"], axis=1),
                                1.0
                                - np.max(stress["probabilities"], axis=1),
                            ),
                            "energy": (
                                common.energy_score(quality["scores"]),
                                common.energy_score(stress["scores"]),
                            ),
                            "class_mahalanobis": (
                                common.mahalanobis_scores(
                                    mahalanobis, quality["test_embedding"]
                                ),
                                common.mahalanobis_scores(
                                    mahalanobis, stress["test_embedding"]
                                ),
                            ),
                        }
                        for score_name, (quality_score, stress_score) in score_sets.items():
                            ood_rows.append(
                                {
                                    "outer_fold": outer_fold,
                                    "declared_seed": declared_seed,
                                    "run_seed": seed,
                                    "score": score_name,
                                    "selected_rejection_score": (
                                        selected_rejection
                                    ),
                                    "selected": (
                                        score_name
                                        == {
                                            "calibrated_max_probability": "one_minus_max_probability",
                                            "energy": "energy",
                                            "class_conditional_embedding_mahalanobis": "class_mahalanobis",
                                        }[selected_rejection]
                                    ),
                                    "n_quality": len(quality_score),
                                    "n_stress": len(stress_score),
                                    **classical.ood_metrics(
                                        quality_score, stress_score
                                    ),
                                }
                            )
    pd.DataFrame(metric_rows).to_csv(
        args.output_dir / "outer_metrics.csv", index=False
    )
    pd.DataFrame(predictions).to_csv(
        args.output_dir / "outer_predictions.csv", index=False
    )
    pd.DataFrame(per_class).to_csv(
        args.output_dir / "outer_per_class_metrics.csv", index=False
    )
    pd.DataFrame(selective).to_csv(
        args.output_dir / "outer_selective_metrics.csv", index=False
    )
    pd.DataFrame(temperatures).to_csv(
        args.output_dir / "outer_temperatures.csv", index=False
    )
    pd.DataFrame(rejection_selection_rows).to_csv(
        args.output_dir / "rejection_selection_metrics.csv", index=False
    )
    pd.DataFrame(rejection_decisions).to_csv(
        args.output_dir / "rejection_decisions_at_80.csv", index=False
    )
    pd.DataFrame(ood_rows).to_csv(
        args.output_dir / "outer_ood_metrics.csv", index=False
    )
    pd.DataFrame(registry).to_csv(
        args.output_dir / "outer_run_registry.csv", index=False
    )
    pd.DataFrame(attribution_metrics).to_csv(
        args.output_dir / "attribution_metrics.csv", index=False
    )
    pd.DataFrame(attribution_peaks).to_csv(
        args.output_dir / "attribution_peaks.csv", index=False
    )
    copy_siamese_control(args, dataset, protocol)


def correctness_calibration_metrics(
    probability: np.ndarray, correct: np.ndarray
) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1.0e-8, 1 - 1.0e-8)
    correct = np.asarray(correct, dtype=float)
    nll = -np.mean(
        correct * np.log(probability)
        + (1.0 - correct) * np.log(1.0 - probability)
    )
    brier = np.mean((probability - correct) ** 2)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index in range(10):
        in_bin = (
            (probability >= edges[index])
            & (
                (probability <= edges[index + 1])
                if index == 9
                else (probability < edges[index + 1])
            )
        )
        if np.any(in_bin):
            ece += float(in_bin.mean()) * abs(
                float(probability[in_bin].mean())
                - float(correct[in_bin].mean())
            )
    return {
        "correctness_negative_log_likelihood": float(nll),
        "correctness_brier": float(brier),
        "correctness_ece_10": float(ece),
    }


def copy_siamese_control(
    args: argparse.Namespace,
    dataset: baseline.SpectralDataset,
    protocol: dict[str, Any],
) -> None:
    metrics = pd.read_csv(args.prior_bundle / "outer_fold_metrics.csv")
    siamese = metrics[metrics["model_family"] == "siamese"].copy()
    siamese.to_csv(
        args.output_dir / "siamese_control_metrics.csv", index=False
    )
    prior_domain = pd.read_csv(
        args.prior_bundle / "domain_transfer_metrics.csv"
    )
    prior_domain[
        prior_domain["model_family"] == "siamese"
    ].to_csv(
        args.output_dir / "siamese_control_domain_metrics.csv",
        index=False,
    )
    prior_predictions = pd.read_csv(
        args.prior_bundle / "outer_fold_predictions.csv"
    )
    predictions = prior_predictions[
        prior_predictions["model_family"] == "siamese"
    ].copy()
    subset_tokens = {
        "test_strict_core": "strict_core",
        "test_quality_pass": "quality_pass",
        "test_field_quality_stress": "field_quality_stress",
    }
    predictions["evaluation_subset"] = predictions["scenario"].map(
        lambda value: next(
            subset
            for token, subset in subset_tokens.items()
            if token in value
        )
    )
    predictions["calibrated_correctness_probability"] = np.nan
    for (_, seed), seed_frame in predictions.groupby(
        ["evaluation_subset", "seed"]
    ):
        for outer_fold in range(5):
            train = seed_frame["outer_fold"].astype(int) != outer_fold
            test = ~train
            train_correct = (
                seed_frame.loc[train, "correct"].astype(bool).astype(int)
            )
            if train_correct.nunique() < 2:
                fitted = np.full(
                    int(test.sum()), float(train_correct.mean()), dtype=float
                )
            else:
                calibrator = LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    random_state=common.stable_seed(
                        "siamese_correctness_calibration",
                        seed,
                        outer_fold,
                    ),
                )
                calibrator.fit(
                    seed_frame.loc[
                        train, "prediction_confidence"
                    ].to_numpy(dtype=float)[:, None],
                    train_correct,
                )
                fitted = calibrator.predict_proba(
                    seed_frame.loc[
                        test, "prediction_confidence"
                    ].to_numpy(dtype=float)[:, None]
                )[:, 1]
            predictions.loc[
                seed_frame.index[test],
                "calibrated_correctness_probability",
            ] = fitted

    manifest_by_uid = dataset.manifest.set_index(
        dataset.manifest["observation_uid"].astype(str), drop=False
    )
    diagnostic_rows: list[dict[str, Any]] = []
    selective_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    embeddings_dir = args.prior_bundle / "embeddings"
    for (scenario, outer_fold, seed), frame in predictions.groupby(
        ["scenario", "outer_fold", "seed"], sort=True
    ):
        embedding_path = (
            embeddings_dir
            / (
                f"outer__{scenario}__siamese__derivative_1"
                f"__s{int(seed)}.npz"
            )
        )
        archive = np.load(embedding_path, allow_pickle=False)
        uids = archive["observation_uid"].astype(str)
        embedding = np.asarray(archive["latent"], dtype=float)
        ordered = frame.set_index(
            frame["observation_uid"].astype(str), drop=False
        ).loc[uids]
        test_manifest = manifest_by_uid.loc[uids].reset_index(drop=True)
        geometry = common.pair_geometry(embedding, test_manifest)
        probes: dict[str, Any] = {}
        for probe_name, target_column in (
            ("analyte", "target_analyte"),
            ("instrument", "instrument"),
            ("sensor", "sensor_family"),
        ):
            probe = common.leave_one_group_out_probe(
                embedding,
                test_manifest[target_column].astype(str),
                test_manifest["master_sample_id"].astype(str),
                common.stable_seed(
                    "siamese",
                    scenario,
                    seed,
                    probe_name,
                    "heldout_master_probe",
                ),
            )
            probes.update(
                {
                    f"{probe_name}_heldout_master_probe_balanced_accuracy": (
                        probe["balanced_accuracy"]
                    ),
                    f"{probe_name}_heldout_master_probe_n_supported": (
                        probe["n_supported"]
                    ),
                    f"{probe_name}_heldout_master_probe_supported_fraction": (
                        probe["supported_fraction"]
                    ),
                }
            )
        calibrated = ordered[
            "calibrated_correctness_probability"
        ].to_numpy(dtype=float)
        correct = ordered["correct"].astype(bool).to_numpy()
        supported = ordered["test_class_supported"].astype(bool).to_numpy()
        evaluation_subset = str(ordered["evaluation_subset"].iloc[0])
        context = {
            "scenario": scenario,
            "outer_fold": int(outer_fold),
            "declared_seed": int(seed),
            "evaluation_subset": evaluation_subset,
            "representation": "derivative_1",
            "confidence_kind": "cross_fitted_correctness_platt_from_negative_prototype_distance",
        }
        diagnostic_rows.append(
            {
                **context,
                "n_test": len(ordered),
                "n_supported": int(supported.sum()),
                "accuracy": float(correct[supported].mean()),
                "predicted_class_count": int(
                    ordered.loc[supported, "predicted_label"].nunique()
                ),
                "encoder_parameters": int(
                    protocol["legacy_encoder"][
                        "expected_encoder_parameters"
                    ]
                ),
                **geometry,
                **probes,
                **correctness_calibration_metrics(
                    calibrated[supported], correct[supported]
                ),
            }
        )
        selective_rows.extend(
            classical.selective_rows(
                context,
                ordered["true_label"].astype(str).to_numpy(),
                ordered["predicted_label"].astype(str).to_numpy(),
                calibrated,
                supported,
                protocol["calibration_and_ood"]["selective_coverages"],
            )
        )
        incorrect = ordered[supported & ~correct].copy()
        for _, row in incorrect.iterrows():
            failure_rows.append(
                {
                    **context,
                    "observation_uid": str(row["observation_uid"]),
                    "true_label": str(row["true_label"]),
                    "predicted_label": str(row["predicted_label"]),
                    "raw_prototype_confidence": float(
                        row["prediction_confidence"]
                    ),
                    "calibrated_correctness_probability": float(
                        row["calibrated_correctness_probability"]
                    ),
                    "instrument": str(row["instrument"]),
                    "sensor_family": str(row["sensor_family"]),
                }
            )
    predictions.to_csv(
        args.output_dir / "siamese_control_predictions.csv", index=False
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        args.output_dir / "siamese_control_diagnostics.csv", index=False
    )
    pd.DataFrame(selective_rows).to_csv(
        args.output_dir / "siamese_control_selective_metrics.csv",
        index=False,
    )
    pd.DataFrame(failure_rows).to_csv(
        args.output_dir / "siamese_control_failures.csv", index=False
    )
    source_hash = baseline.sha256_file(
        args.prior_bundle / "outer_fold_metrics.csv"
    )
    write_json(
        args.output_dir / "siamese_control_provenance.json",
        {
            "source": str(
                (args.prior_bundle / "outer_fold_metrics.csv").resolve()
            ),
            "source_sha256": source_hash,
            "source_validation_status": "pass",
            "embedding_source_hash_catalog_sha256": baseline.sha256_file(
                args.prior_bundle / "artifact_hashes.json"
            ),
            "interpretation": "Previously rebuilt deterministic Siamese control imported without reselection. Geometry and heldout-master probes were recomputed from its immutable outer-test embeddings. Its single nearest-prototype distance cannot support multiclass temperature scaling, so correctness confidence was Platt-calibrated using other outer folds and is reported explicitly as a different calibration target.",
        },
    )


def run_domain(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    dataset: baseline.SpectralDataset,
    device: torch.device,
) -> None:
    full, epochs = selected_spec(args.output_dir, None)
    temperatures = pd.read_csv(args.output_dir / "outer_temperatures.csv")
    manifest = dataset.manifest
    uid_to_index = {
        uid: index
        for index, uid in enumerate(
            manifest["observation_uid"].astype(str).to_numpy()
        )
    }
    final_seeds = [int(value) for value in protocol["fixed_training"]["final_seeds"]]
    metric_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for subset, filename in (
        ("strict_core", "domain_evaluation_partitions_core.csv"),
        ("quality_pass", "domain_evaluation_partitions_quality.csv"),
    ):
        temperature = float(
            temperatures.loc[
                temperatures["subset"] == subset, "temperature"
            ].median()
        )
        partitions = pd.read_csv(args.bundle / filename)
        for (domain_protocol, domain_type, heldout), scenario_frame in partitions.groupby(
            ["protocol", "domain_type", "heldout_domain"], sort=True
        ):
            train_uids = scenario_frame.loc[
                scenario_frame["partition"] == "train", "observation_uid"
            ].astype(str)
            test_uids = scenario_frame.loc[
                scenario_frame["partition"] == "test", "observation_uid"
            ].astype(str)
            train_indices = np.asarray(
                [uid_to_index[uid] for uid in train_uids], dtype=int
            )
            test_indices = np.asarray(
                [uid_to_index[uid] for uid in test_uids], dtype=int
            )
            train_manifest = manifest.iloc[train_indices].reset_index(drop=True)
            test_manifest = manifest.iloc[test_indices].reset_index(drop=True)
            for declared_seed in final_seeds:
                seed = common.stable_seed(
                    common.PROTOCOL_VERSION,
                    "domain_final",
                    subset,
                    domain_protocol,
                    domain_type,
                    heldout,
                    full.candidate_id,
                    declared_seed,
                )
                scenario = (
                    f"domain_{subset}_{domain_protocol}_{domain_type}_{heldout}"
                    f"_seed{declared_seed}"
                )
                train_values = dataset.representations[full.representation][
                    train_indices
                ]
                trained = train_cached(
                    args.output_dir,
                    scenario,
                    train_values,
                    train_manifest,
                    full,
                    protocol,
                    seed,
                    epochs,
                    device,
                )
                result = evaluate_model(
                    trained,
                    full,
                    train_values,
                    train_manifest,
                    dataset.representations[full.representation][test_indices],
                    test_manifest,
                    temperature,
                    device,
                )
                context = {
                    "stage": "domain",
                    "subset": subset,
                    "protocol": domain_protocol,
                    "domain_type": domain_type,
                    "heldout_domain": heldout,
                    "variant": "full_domain_aware",
                    "candidate_id": full.candidate_id,
                    "representation": full.representation,
                    "architecture": full.architecture,
                    "embedding_dimension": full.embedding_dimension,
                    "declared_seed": declared_seed,
                    "run_seed": seed,
                    "epochs": epochs,
                    "temperature": temperature,
                }
                metric_rows.append(metrics_row(context, result, trained))
                prediction_records.extend(
                    prediction_rows(context, test_manifest, result)
                )
                registry.append(
                    {
                        **context,
                        **full.__dict__,
                        **trained.parameters,
                        "state_sha256": trained.state_sha256,
                    }
                )
    pd.DataFrame(metric_rows).to_csv(
        args.output_dir / "domain_metrics.csv", index=False
    )
    pd.DataFrame(prediction_records).to_csv(
        args.output_dir / "domain_predictions.csv", index=False
    )
    pd.DataFrame(registry).to_csv(
        args.output_dir / "domain_run_registry.csv", index=False
    )


def run_negative(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    dataset: baseline.SpectralDataset,
    device: torch.device,
) -> None:
    manifest = dataset.manifest
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    declared_seed = int(protocol["fixed_training"]["selection_seed"])
    for outer_fold in range(5):
        full, epochs = selected_spec(args.output_dir, outer_fold)
        train_mask = folds != outer_fold
        test_mask = folds == outer_fold
        original_train_manifest = manifest.loc[train_mask].reset_index(
            drop=True
        )
        test_manifest = manifest.loc[test_mask].reset_index(drop=True)
        for control in (
            "master_group_analyte_label_permutation",
            "randomized_domain_relationships",
        ):
            train_manifest = original_train_manifest.copy()
            seed = common.stable_seed(
                common.PROTOCOL_VERSION,
                "negative_control",
                control,
                outer_fold,
                declared_seed,
            )
            rng = np.random.default_rng(seed)
            if control == "master_group_analyte_label_permutation":
                group_frame = train_manifest[
                    ["master_sample_id", "target_analyte"]
                ].drop_duplicates()
                shuffled = rng.permutation(
                    group_frame["target_analyte"].astype(str).to_numpy()
                )
                mapping = dict(
                    zip(group_frame["master_sample_id"].astype(str), shuffled)
                )
                train_manifest["target_analyte"] = (
                    train_manifest["master_sample_id"]
                    .astype(str)
                    .map(mapping)
                )
            else:
                train_manifest["instrument"] = rng.permutation(
                    train_manifest["instrument"].astype(str).to_numpy()
                )
                train_manifest["sensor_family"] = rng.permutation(
                    train_manifest["sensor_family"].astype(str).to_numpy()
                )
            scenario = f"negative_{control}_o{outer_fold}"
            train_values = dataset.representations[full.representation][
                train_mask
            ]
            trained = train_cached(
                args.output_dir,
                scenario,
                train_values,
                train_manifest,
                full,
                protocol,
                seed,
                epochs,
                device,
            )
            result = evaluate_model(
                trained,
                full,
                train_values,
                train_manifest,
                dataset.representations[full.representation][test_mask],
                test_manifest,
                1.0,
                device,
            )
            rows.append(
                metrics_row(
                    {
                        "stage": "negative_control",
                        "control": control,
                        "outer_fold": outer_fold,
                        "candidate_id": full.candidate_id,
                        "representation": full.representation,
                        "architecture": full.architecture,
                        "embedding_dimension": full.embedding_dimension,
                        "declared_seed": declared_seed,
                        "run_seed": seed,
                        "epochs": epochs,
                    },
                    result,
                    trained,
                )
            )
    pd.DataFrame(rows).to_csv(
        args.output_dir / "negative_control_metrics.csv", index=False
    )


def main() -> None:
    args = parse_args()
    protocol = common.load_protocol(args.protocol)
    dataset = initialize(args, protocol)
    device = baseline.resolve_device(args.device)
    stages = (
        ("outer", "domain", "negative")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        print(f"START {stage}", flush=True)
        if stage == "outer":
            run_outer(args, protocol, dataset, device)
        elif stage == "domain":
            run_domain(args, protocol, dataset, device)
        else:
            run_negative(args, protocol, dataset, device)
        print(f"DONE {stage}", flush=True)


if __name__ == "__main__":
    main()
