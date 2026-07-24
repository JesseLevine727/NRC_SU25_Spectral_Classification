#!/usr/bin/env python3
"""Run sealed final evaluations for the selected SERS standard VAE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import run_sers_baseline_final as baseline_final
import run_sers_representation_baselines as baseline_selection
import run_sers_standard_vae_selection as vae_selection
import sers_baseline_common as baseline
import sers_vae_common as vae


FINAL_VERSION = "sers-standard-vae-final-v1"


def write_json(path: Path, value: Any) -> None:
    baseline_selection.write_json(path, value)


def array_sha256(values: Sequence[str] | np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def config_from_selected(record: dict[str, Any]) -> vae.VAETrainingConfig:
    return vae.VAETrainingConfig(
        channels=tuple(int(value) for value in record["channels"]),
        latent_dimension=int(record["latent_dimension"]),
        reconstruction_loss=str(record["reconstruction_loss"]),
        kl_schedule=str(record["kl_schedule"]),
        beta=float(record["beta"]),
        kl_normalization_divisor=int(record["kl_normalization_divisor"]),
        learning_rate=float(record["learning_rate"]),
        weight_decay=float(record["weight_decay"]),
        batch_size=int(record["batch_size"]),
        maximum_epochs=int(record["maximum_epochs"]),
        minimum_epochs=int(record["minimum_epochs"]),
        early_stopping_patience=int(record["early_stopping_patience"]),
        early_stopping_minimum_delta=float(
            record["early_stopping_minimum_delta"]
        ),
        gradient_clip_norm=float(record["gradient_clip_norm"]),
    )


def selected_epoch_count(
    output_dir: Path,
    subset: str,
    outer_fold: int | None,
    representation: str,
    configuration: str,
) -> int:
    frame = pd.read_csv(output_dir / "selection_fold_metrics.csv")
    selected = frame[
        (frame["subset"] == subset)
        & (frame["representation"] == representation)
        & (frame["configuration"] == configuration)
    ]
    if outer_fold is not None:
        selected = selected[selected["outer_fold"] == outer_fold]
    if selected.empty and subset != "strict_core":
        selected = frame[
            (frame["subset"] == "strict_core")
            & (frame["representation"] == representation)
            & (frame["configuration"] == configuration)
        ]
        if outer_fold is not None:
            selected = selected[selected["outer_fold"] == outer_fold]
    if selected.empty:
        raise ValueError(
            f"No inner epoch evidence for {subset}/{outer_fold}/"
            f"{representation}/{configuration}"
        )
    return max(
        1,
        int(math.floor(float(selected["best_epoch"].median()) + 0.5)),
    )


def train_vae_fixed(
    train_values: np.ndarray,
    train_uids: Sequence[str],
    config: vae.VAETrainingConfig,
    epochs: int,
    run_seed: int,
    device: torch.device,
) -> tuple[vae.ConvVariationalAutoencoder, pd.DataFrame, str]:
    baseline.configure_determinism(run_seed)
    model = vae.ConvVariationalAutoencoder(
        train_values.shape[1],
        config.channels,
        config.latent_dimension,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    dataset = baseline.ReconstructionDataset(
        train_values,
        train_uids,
        "clean",
        run_seed,
        config.maximum_epochs,
    )
    generator = torch.Generator()
    generator.manual_seed(baseline.stable_seed(run_seed, "final_loader"))
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        beta = vae.beta_for_epoch(
            config.kl_schedule, epoch, config.maximum_epochs
        )
        model.train()
        totals = np.zeros(3, dtype=float)
        count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            reconstruction, mu, log_variance, _ = model(inputs, sample=True)
            total, reconstruction_term, kl_unnormalized = vae.loss_components(
                reconstruction,
                targets,
                mu,
                log_variance,
                config,
                beta,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            batch_count = len(inputs)
            totals += (
                np.asarray(
                    [
                        float(total.detach().cpu()),
                        float(reconstruction_term.detach().cpu()),
                        float(kl_unnormalized.detach().cpu()),
                    ]
                )
                * batch_count
            )
            count += batch_count
        means = totals / max(count, 1)
        rows.append(
            {
                "epoch": epoch + 1,
                "beta": beta,
                "train_loss": means[0],
                "train_reconstruction_loss": means[1],
                "train_kl_unnormalized": means[2],
                "train_kl_normalized": (
                    means[2] / config.kl_normalization_divisor
                ),
            }
        )
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    return model, pd.DataFrame(rows), baseline.state_dict_sha256(state)


def checkpoint_paths(
    output_dir: Path, stage: str, run_identifier: str
) -> tuple[Path, Path]:
    return (
        output_dir / "checkpoints" / stage / f"{run_identifier}.pt",
        output_dir / "run_cache" / stage / f"{run_identifier}.csv",
    )


def load_or_train_vae(
    output_dir: Path,
    stage: str,
    run_identifier: str,
    train_values: np.ndarray,
    train_uids: Sequence[str],
    config: vae.VAETrainingConfig,
    epochs: int,
    run_seed: int,
    device: torch.device,
) -> tuple[
    vae.ConvVariationalAutoencoder,
    pd.DataFrame,
    dict[str, Any],
]:
    checkpoint_path, history_path = checkpoint_paths(
        output_dir, stage, run_identifier
    )
    expected = {
        "evaluation_version": FINAL_VERSION,
        "run_identifier": run_identifier,
        "configuration": vae.config_record(config),
        "epochs": epochs,
        "run_seed": run_seed,
        "train_uids_sha256": array_sha256(
            np.asarray(train_uids, dtype=str)
        ),
        "train_values_sha256": array_sha256(
            np.asarray(train_values, dtype=np.float32)
        ),
    }
    if checkpoint_path.exists() and history_path.exists():
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if payload["metadata"] != expected:
            raise ValueError(f"Stale VAE checkpoint: {run_identifier}")
        actual = baseline.state_dict_sha256(payload["state_dict"])
        if actual != payload["state_sha256"]:
            raise ValueError(f"Corrupt VAE checkpoint: {run_identifier}")
        model = vae.ConvVariationalAutoencoder(
            train_values.shape[1],
            config.channels,
            config.latent_dimension,
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return model, pd.read_csv(history_path), {
            **expected,
            "state_sha256": actual,
            "parameter_count": baseline.model_parameter_count(model),
            "cache_status": "loaded",
            "checkpoint": str(checkpoint_path.relative_to(output_dir)),
        }
    model, history, state_hash = train_vae_fixed(
        train_values,
        train_uids,
        config,
        epochs,
        run_seed,
        device,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.save_checkpoint(checkpoint_path, model, expected)
    history.to_csv(history_path, index=False)
    return model, history, {
        **expected,
        "state_sha256": state_hash,
        "parameter_count": baseline.model_parameter_count(model),
        "cache_status": "trained",
        "checkpoint": str(checkpoint_path.relative_to(output_dir)),
    }


def evaluate_vae(
    model: vae.ConvVariationalAutoencoder,
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    context: dict[str, Any],
    config: vae.VAETrainingConfig,
    evaluation_device: torch.device,
    output_dir: Path,
    artifact_stem: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame,
    list[dict[str, Any]],
]:
    model.to(evaluation_device)
    train_targets = train_manifest["target_analyte"].astype(str).to_numpy()
    test_targets = test_manifest["target_analyte"].astype(str).to_numpy()
    _, train_mu, _ = vae.vae_outputs(
        model, train_values, evaluation_device
    )
    test_reconstruction, test_mu, test_log_variance = vae.vae_outputs(
        model, test_values, evaluation_device
    )
    probe = baseline.fit_latent_probe_model(
        train_mu,
        train_targets,
        baseline.stable_seed(artifact_stem, "final_probe"),
    )
    predictions = probe.predict(test_mu).astype(str)
    confidence = np.max(probe.predict_proba(test_mu), axis=1)
    classification = baseline_final.supported_classification(
        train_targets, test_targets, predictions
    )
    repeatable = baseline_final.repeatable_test_positions(
        train_values,
        train_manifest,
        test_values,
        test_manifest,
    )
    per_reconstruction = baseline.reconstruction_metrics(
        test_values,
        test_reconstruction,
        test_manifest["observation_uid"].astype(str),
        repeatable,
    )
    per_reconstruction = pd.concat(
        [
            pd.DataFrame([context] * len(per_reconstruction)),
            per_reconstruction,
        ],
        axis=1,
    )
    reconstruction = baseline.aggregate_reconstruction_metrics(
        per_reconstruction
    )
    variability = vae.posterior_sample_reconstruction_variability(
        model,
        test_values,
        evaluation_device,
        baseline.stable_seed(artifact_stem, "posterior_samples"),
    )
    variational = vae.variational_metrics(
        test_mu,
        test_log_variance,
        normalization_divisor=config.kl_normalization_divisor,
        sample_reconstruction_variability=variability,
    )
    baseline_final.save_array_artifact(
        output_dir / "embeddings" / f"{artifact_stem}.npz",
        test_manifest["observation_uid"].astype(str),
        posterior_mean=test_mu.astype(np.float32),
        posterior_log_variance=test_log_variance.astype(np.float32),
    )
    baseline_final.save_array_artifact(
        output_dir / "reconstructions" / f"{artifact_stem}.npz",
        test_manifest["observation_uid"].astype(str),
        clean=test_values.astype(np.float32),
        reconstructed=test_reconstruction.astype(np.float32),
    )
    corruption_rows: list[dict[str, Any]] = []
    for corruption in baseline.CORRUPTION_NAMES:
        for severity in (0.5, 1.0, 1.5):
            corrupted = baseline_selection.deterministic_corrupted_matrix(
                test_values,
                test_manifest["observation_uid"].astype(str),
                corruption,
                severity,
            )
            recovered, corrupted_mu, corrupted_log_variance = vae.vae_outputs(
                model, corrupted, evaluation_device
            )
            corrupted_prediction = probe.predict(corrupted_mu).astype(str)
            corrupted_classification = (
                baseline_final.supported_classification(
                    train_targets, test_targets, corrupted_prediction
                )
            )
            recovery = baseline.aggregate_reconstruction_metrics(
                baseline.reconstruction_metrics(
                    test_values,
                    recovered,
                    test_manifest["observation_uid"].astype(str),
                    repeatable,
                )
            )
            corrupted_variational = vae.variational_metrics(
                corrupted_mu,
                corrupted_log_variance,
                normalization_divisor=config.kl_normalization_divisor,
                sample_reconstruction_variability=np.nan,
            )
            corruption_rows.append(
                {
                    **context,
                    "corruption": corruption,
                    "severity": severity,
                    "balanced_accuracy_supported": (
                        corrupted_classification["balanced_accuracy"]
                    ),
                    "macro_f1_supported": corrupted_classification[
                        "macro_f1_supported"
                    ],
                    "prediction_agreement": float(
                        np.mean(corrupted_prediction == predictions)
                    ),
                    "latent_cosine_drift": float(
                        np.mean(
                            baseline_selection.cosine_drift(
                                test_mu, corrupted_mu
                            )
                        )
                    ),
                    **recovery,
                    **{
                        f"corrupted_{key}": value
                        for key, value in corrupted_variational.items()
                    },
                }
            )
    metric = {
        **baseline_final.classification_metric_row(context, classification),
        **reconstruction,
        **variational,
    }
    prediction_records = baseline_final.prediction_rows(
        context,
        test_manifest,
        test_targets,
        predictions,
        confidence,
        classification["supported_mask"],
    )
    return (
        metric,
        prediction_records,
        per_reconstruction,
        corruption_rows,
    )


def empty_outputs() -> dict[str, list[Any]]:
    return {
        "metrics": [],
        "predictions": [],
        "reconstruction": [],
        "corruption": [],
        "histories": [],
        "registry": [],
    }


def merge_outputs(
    target: dict[str, list[Any]], source: dict[str, list[Any]]
) -> None:
    for key in target:
        target[key].extend(source[key])


def evaluate_partition(
    output_dir: Path,
    stage: str,
    scenario: str,
    training_scenario: str,
    train_dataset: baseline.SpectralDataset,
    test_dataset: baseline.SpectralDataset,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    config: vae.VAETrainingConfig,
    representations: Sequence[str],
    epoch_subset: str,
    outer_fold: int | None,
    training_device: torch.device,
    evaluation_device: torch.device,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, list[Any]]:
    extra_context = extra_context or {}
    outputs = empty_outputs()
    train_manifest = train_dataset.manifest.loc[train_mask].reset_index(drop=True)
    test_manifest = test_dataset.manifest.loc[test_mask].reset_index(drop=True)
    for representation in representations:
        epochs = selected_epoch_count(
            output_dir,
            epoch_subset,
            outer_fold,
            representation,
            config.identifier,
        )
        train_values = train_dataset.representations[representation][train_mask]
        test_values = test_dataset.representations[representation][test_mask]
        for declared_seed in (1729, 2718, 3141):
            run_seed = baseline.stable_seed(
                vae.PROTOCOL_VERSION,
                train_dataset.name,
                stage,
                training_scenario,
                representation,
                config.identifier,
                declared_seed,
            )
            run_identifier = (
                f"{training_scenario}__vae__{representation}__"
                f"{config.identifier}__s{declared_seed}"
            )
            model, history, registry = load_or_train_vae(
                output_dir,
                stage,
                run_identifier,
                train_values,
                train_manifest["observation_uid"].astype(str),
                config,
                epochs,
                run_seed,
                training_device,
            )
            context = {
                **extra_context,
                "stage": stage,
                "scenario": scenario,
                "outer_fold": outer_fold,
                "model_family": "vae",
                "model": config.identifier,
                "representation": representation,
                "curriculum": "clean",
                "kl_schedule": config.kl_schedule,
                "seed": declared_seed,
            }
            metric, predictions, reconstruction, corruption = evaluate_vae(
                model,
                train_values,
                test_values,
                train_manifest,
                test_manifest,
                context,
                config,
                evaluation_device,
                output_dir,
                (
                    f"{stage}__{scenario}__vae__{representation}__"
                    f"{config.identifier}__s{declared_seed}"
                ),
            )
            outputs["metrics"].append(metric)
            outputs["predictions"].extend(predictions)
            outputs["reconstruction"].append(reconstruction)
            outputs["corruption"].extend(corruption)
            history = history.copy()
            for key, value in context.items():
                history[key] = value
            history["fixed_epoch_count"] = epochs
            outputs["histories"].append(history)
            outputs["registry"].append(registry)
    return outputs


def save_outputs(
    output_dir: Path, prefix: str, outputs: dict[str, list[Any]]
) -> None:
    pd.DataFrame(outputs["metrics"]).to_csv(
        output_dir / f"{prefix}_metrics.csv", index=False
    )
    pd.DataFrame(outputs["predictions"]).to_csv(
        output_dir / f"{prefix}_predictions.csv", index=False
    )
    reconstructions = (
        pd.concat(outputs["reconstruction"], ignore_index=True)
        if outputs["reconstruction"]
        else pd.DataFrame()
    )
    reconstructions.to_csv(
        output_dir / f"{prefix}_reconstruction_metrics.csv", index=False
    )
    pd.DataFrame(outputs["corruption"]).to_csv(
        output_dir / f"{prefix}_corruption_metrics.csv", index=False
    )
    histories = (
        pd.concat(outputs["histories"], ignore_index=True)
        if outputs["histories"]
        else pd.DataFrame()
    )
    histories.to_csv(
        output_dir / f"{prefix}_training_histories.csv", index=False
    )
    write_json(
        output_dir / f"{prefix}_run_registry.json", outputs["registry"]
    )


def run_outer(
    output_dir: Path,
    nato_bundle: Path,
    config: vae.VAETrainingConfig,
    training_device: torch.device,
    evaluation_device: torch.device,
) -> None:
    core = baseline.load_nato_dataset(nato_bundle)
    quality = baseline_final.load_nato_subset(nato_bundle, "quality_pass")
    stress = baseline_final.load_nato_subset(
        nato_bundle, "field_quality_stress"
    )
    outputs = empty_outputs()
    for outer_fold in range(5):
        for train_dataset, test_dataset, train_subset, test_subset in (
            (core, core, "strict_core", "strict_core"),
            (quality, quality, "quality_pass", "quality_pass"),
            (quality, stress, "quality_pass", "field_quality_stress"),
        ):
            train_folds = train_dataset.manifest[
                "grouped_sample_fold_5"
            ].to_numpy(dtype=int)
            test_folds = test_dataset.manifest[
                "grouped_sample_fold_5"
            ].to_numpy(dtype=int)
            scenario = (
                f"nato_outer_o{outer_fold}__train_{train_subset}__"
                f"test_{test_subset}"
            )
            result = evaluate_partition(
                output_dir,
                "outer",
                scenario,
                f"nato_outer_o{outer_fold}__train_{train_subset}",
                train_dataset,
                test_dataset,
                train_folds != outer_fold,
                test_folds == outer_fold,
                config,
                ("arpls_minmax", "minimal_minmax"),
                train_subset,
                outer_fold,
                training_device,
                evaluation_device,
            )
            merge_outputs(outputs, result)
            print(
                json.dumps(
                    {
                        "stage": "outer",
                        "outer_fold": outer_fold,
                        "test_subset": test_subset,
                        "metrics": len(result["metrics"]),
                    }
                ),
                flush=True,
            )
    save_outputs(output_dir, "outer_fold", outputs)


def run_poster(
    output_dir: Path,
    poster_csv: Path,
    config: vae.VAETrainingConfig,
    training_device: torch.device,
    evaluation_device: torch.device,
) -> None:
    poster = baseline.load_poster_dataset(poster_csv)
    outputs = empty_outputs()
    families = poster.manifest["substrate_family"].astype(str).to_numpy()
    for heldout in sorted(np.unique(families)):
        scenario = f"poster_holdout_{heldout}"
        result = evaluate_partition(
            output_dir,
            "poster",
            scenario,
            scenario,
            poster,
            poster,
            families != heldout,
            families == heldout,
            config,
            ("arpls_minmax", "minimal_minmax"),
            "strict_core",
            None,
            training_device,
            evaluation_device,
            {
                "heldout_substrate_family": heldout,
                "poster_axis": "400-1800_common",
            },
        )
        merge_outputs(outputs, result)
        print(
            json.dumps(
                {
                    "stage": "poster",
                    "heldout_substrate_family": heldout,
                    "metrics": len(result["metrics"]),
                }
            ),
            flush=True,
        )
    save_outputs(output_dir, "poster", outputs)


def run_domain(
    output_dir: Path,
    nato_bundle: Path,
    config: vae.VAETrainingConfig,
    training_device: torch.device,
    evaluation_device: torch.device,
) -> None:
    datasets = {
        "strict_core": baseline.load_nato_dataset(nato_bundle),
        "quality_pass": baseline_final.load_nato_subset(
            nato_bundle, "quality_pass"
        ),
    }
    partition_files = {
        "strict_core": nato_bundle
        / "domain_evaluation_partitions_core.csv",
        "quality_pass": nato_bundle
        / "domain_evaluation_partitions_quality.csv",
    }
    outputs = empty_outputs()
    for subset, dataset in datasets.items():
        partitions = pd.read_csv(partition_files[subset])
        for keys, frame in partitions.groupby(
            ["protocol", "domain_type", "heldout_domain"],
            sort=True,
        ):
            protocol_name, domain_type, heldout_domain = [
                str(value) for value in keys
            ]
            train_uids = set(
                frame.loc[
                    frame["partition"] == "train", "observation_uid"
                ].astype(str)
            )
            test_uids = set(
                frame.loc[
                    frame["partition"] == "test", "observation_uid"
                ].astype(str)
            )
            uids = dataset.manifest["observation_uid"].astype(str)
            train_mask = uids.isin(train_uids).to_numpy()
            test_mask = uids.isin(test_uids).to_numpy()
            scenario = (
                f"domain__{subset}__{protocol_name}__{domain_type}__"
                f"{heldout_domain}"
            )
            result = evaluate_partition(
                output_dir,
                "domain",
                scenario,
                scenario,
                dataset,
                dataset,
                train_mask,
                test_mask,
                config,
                ("arpls_minmax",),
                subset,
                None,
                training_device,
                evaluation_device,
                {
                    "evaluation_subset": subset,
                    "domain_protocol": protocol_name,
                    "domain_type": domain_type,
                    "heldout_domain": heldout_domain,
                    "n_train": int(train_mask.sum()),
                    "n_test_partition": int(test_mask.sum()),
                },
            )
            merge_outputs(outputs, result)
            print(
                json.dumps(
                    {
                        "stage": "domain",
                        "subset": subset,
                        "protocol": protocol_name,
                        "domain_type": domain_type,
                        "heldout_domain": heldout_domain,
                        "metrics": len(result["metrics"]),
                    }
                ),
                flush=True,
            )
    save_outputs(output_dir, "domain_transfer", outputs)


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repository / "configs" / "sers_standard_vae_v1.json",
    )
    parser.add_argument(
        "--nato-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v2",
    )
    parser.add_argument(
        "--baseline-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_representation_baselines"
        / "baselines_v1",
    )
    parser.add_argument(
        "--poster-csv",
        type=Path,
        default=repository
        / "Workspace"
        / "data"
        / "processed"
        / "consolidated_SERS.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_standard_vae"
        / "standard_vae_v1",
    )
    parser.add_argument(
        "--stage",
        choices=["outer", "poster", "domain", "all"],
        default="all",
    )
    parser.add_argument(
        "--training-device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
    )
    parser.add_argument(
        "--evaluation-device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    protocol_path = args.protocol.resolve()
    nato_bundle = args.nato_bundle.resolve()
    baseline_bundle = args.baseline_bundle.resolve()
    poster_csv = args.poster_csv.resolve()
    output_dir = args.output_dir.resolve()
    protocol = vae_selection.load_protocol(protocol_path)
    vae_selection.verify_frozen_inputs(
        protocol, nato_bundle, baseline_bundle, poster_csv
    )
    copied_protocol = output_dir / "predeclared_protocol.json"
    if copied_protocol.read_bytes() != protocol_path.read_bytes():
        raise ValueError("Selection and final protocol snapshots differ")
    selected = json.loads(
        (output_dir / "selected_configuration.json").read_text()
    )
    if not selected.get("selection_closed", False):
        raise ValueError("VAE selection is not closed")
    config = config_from_selected(selected)
    training_device = baseline.resolve_device(args.training_device)
    evaluation_device = baseline.resolve_device(args.evaluation_device)
    if args.stage in {"outer", "all"}:
        run_outer(
            output_dir,
            nato_bundle,
            config,
            training_device,
            evaluation_device,
        )
    if args.stage in {"poster", "all"}:
        run_poster(
            output_dir,
            poster_csv,
            config,
            training_device,
            evaluation_device,
        )
    if args.stage in {"domain", "all"}:
        run_domain(
            output_dir,
            nato_bundle,
            config,
            training_device,
            evaluation_device,
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "stage": args.stage,
                "configuration": selected["identifier"],
                "training_device": str(training_device),
                "evaluation_device": str(evaluation_device),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
