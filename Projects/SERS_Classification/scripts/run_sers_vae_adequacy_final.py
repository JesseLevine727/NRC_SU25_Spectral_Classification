#!/usr/bin/env python3
"""Run locked confirmatory evaluations for selected SERS VAE adequacy v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import run_sers_baseline_final as baseline_final
import run_sers_standard_vae_final as standard_final
import sers_baseline_common as baseline
import sers_vae_adequacy_common as adequacy


FINAL_VERSION = "sers-vae-adequacy-final-v1"
DECLARED_SEEDS = (1729, 2718, 3141)


def write_json(path: Path, value: Any) -> None:
    baseline_final.write_json(path, value)


def array_sha256(values: Sequence[str] | np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def config_from_selected(record: dict[str, Any]) -> adequacy.AdequacyConfig:
    return adequacy.AdequacyConfig(
        architecture=str(record["architecture"]),
        latent_dimension=int(record["latent_dimension"]),
        reconstruction_loss=str(record["reconstruction_loss"]),
        beta_target=float(record["beta_target"]),
        optimizer_policy=str(record["optimizer_policy"]),
        maximum_epoch=int(record["maximum_epoch"]),
        learning_rate=float(record["learning_rate"]),
        weight_decay=float(record["weight_decay"]),
        batch_size=int(record["batch_size"]),
        gradient_clip_norm=float(record["gradient_clip_norm"]),
        kl_normalization_divisor=int(record["kl_normalization_divisor"]),
    )


def train_fixed(
    train_values: np.ndarray,
    train_uids: Sequence[str],
    config: adequacy.AdequacyConfig,
    run_seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, torch.Tensor]]:
    baseline.configure_determinism(run_seed)
    model = adequacy.build_model(
        train_values.shape[1], config.architecture, config.latent_dimension
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
        config.maximum_epoch,
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
    records: list[dict[str, Any]] = []
    for epoch_zero in range(config.maximum_epoch):
        epoch = epoch_zero + 1
        beta = adequacy.beta_for_epoch(epoch, config.beta_target)
        dataset.set_epoch(epoch_zero)
        model.train()
        totals = np.zeros(3, dtype=float)
        count = 0
        gradient_norm_sum = 0.0
        gradient_steps = 0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            reconstruction, mu, log_variance, _ = model(inputs, sample=True)
            total, reconstruction_term, kl_unnormalized = (
                adequacy.loss_components(
                    reconstruction,
                    targets,
                    mu,
                    log_variance,
                    config,
                    beta,
                )
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            optimizer.step()
            batch_count = len(inputs)
            totals += np.asarray(
                [
                    float(total.detach().cpu()),
                    float(reconstruction_term.detach().cpu()),
                    float(kl_unnormalized.detach().cpu()),
                ]
            ) * batch_count
            count += batch_count
            gradient_norm_sum += float(gradient_norm.detach().cpu())
            gradient_steps += 1
        means = totals / max(count, 1)
        records.append(
            {
                "epoch": epoch,
                "beta": beta,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": means[0],
                "train_reconstruction_loss": means[1],
                "train_kl_unnormalized": means[2],
                "train_kl_normalized": (
                    means[2] / config.kl_normalization_divisor
                ),
                "mean_unclipped_gradient_norm": (
                    gradient_norm_sum / max(gradient_steps, 1)
                ),
            }
        )
    return model, pd.DataFrame(records), adequacy.cpu_state_dict(model)


def checkpoint_paths(
    output_dir: Path, stage: str, run_identifier: str
) -> tuple[Path, Path]:
    return (
        output_dir / "checkpoints" / stage / f"{run_identifier}.pt",
        output_dir / "run_cache" / stage / f"{run_identifier}.csv",
    )


def load_or_train(
    output_dir: Path,
    stage: str,
    run_identifier: str,
    train_values: np.ndarray,
    train_uids: Sequence[str],
    config: adequacy.AdequacyConfig,
    run_seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, pd.DataFrame, dict[str, Any]]:
    checkpoint_path, history_path = checkpoint_paths(
        output_dir, stage, run_identifier
    )
    expected = {
        "evaluation_version": FINAL_VERSION,
        "run_identifier": run_identifier,
        "configuration": config.record(),
        "epochs": config.maximum_epoch,
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
            raise ValueError(f"Stale adequacy checkpoint: {run_identifier}")
        state_hash = baseline.state_dict_sha256(payload["state_dict"])
        if state_hash != payload["state_sha256"]:
            raise ValueError(f"Corrupt adequacy checkpoint: {run_identifier}")
        model = adequacy.build_model(
            train_values.shape[1],
            config.architecture,
            config.latent_dimension,
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        return model, pd.read_csv(history_path), {
            **expected,
            "state_sha256": state_hash,
            "parameter_count": baseline.model_parameter_count(model),
            "cache_status": "loaded",
            "checkpoint": str(checkpoint_path.relative_to(output_dir)),
        }
    model, history, state = train_fixed(
        train_values, train_uids, config, run_seed, device
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.save_checkpoint(checkpoint_path, model, expected)
    history.to_csv(history_path, index=False)
    return model, history, {
        **expected,
        "state_sha256": baseline.state_dict_sha256(state),
        "parameter_count": baseline.model_parameter_count(model),
        "cache_status": "trained",
        "checkpoint": str(checkpoint_path.relative_to(output_dir)),
    }


def evaluate_partition(
    output_dir: Path,
    stage: str,
    scenario: str,
    training_scenario: str,
    train_dataset: baseline.SpectralDataset,
    test_dataset: baseline.SpectralDataset,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    config: adequacy.AdequacyConfig,
    representations: Sequence[str],
    outer_fold: int | None,
    training_device: torch.device,
    evaluation_device: torch.device,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, list[Any]]:
    extra_context = extra_context or {}
    outputs = standard_final.empty_outputs()
    train_manifest = train_dataset.manifest.loc[train_mask].reset_index(drop=True)
    test_manifest = test_dataset.manifest.loc[test_mask].reset_index(drop=True)
    for representation in representations:
        train_values = train_dataset.representations[representation][train_mask]
        test_values = test_dataset.representations[representation][test_mask]
        for declared_seed in DECLARED_SEEDS:
            run_seed = baseline.stable_seed(
                adequacy.PROTOCOL_VERSION,
                train_dataset.name,
                stage,
                training_scenario,
                representation,
                config.identifier,
                declared_seed,
            )
            run_identifier = (
                f"{training_scenario}__vae_adequacy__{representation}__"
                f"{config.identifier}__s{declared_seed}"
            )
            model, history, registry = load_or_train(
                output_dir,
                stage,
                run_identifier,
                train_values,
                train_manifest["observation_uid"].astype(str),
                config,
                run_seed,
                training_device,
            )
            context = {
                **extra_context,
                "stage": stage,
                "scenario": scenario,
                "outer_fold": outer_fold,
                "model_family": "vae_adequacy",
                "model": config.identifier,
                "representation": representation,
                "curriculum": "clean",
                "kl_schedule": "four25_then_fixed",
                "beta_target": config.beta_target,
                "seed": declared_seed,
            }
            metric, predictions, reconstruction, corruption = (
                standard_final.evaluate_vae(
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
                        f"{stage}__{scenario}__vae_adequacy__"
                        f"{representation}__{config.identifier}__"
                        f"s{declared_seed}"
                    ),
                )
            )
            outputs["metrics"].append(metric)
            outputs["predictions"].extend(predictions)
            outputs["reconstruction"].append(reconstruction)
            outputs["corruption"].extend(corruption)
            history = history.copy()
            for key, value in context.items():
                history[key] = value
            history["fixed_epoch_count"] = config.maximum_epoch
            outputs["histories"].append(history)
            outputs["registry"].append(registry)
    return outputs


def run_outer(
    output_dir: Path,
    nato_bundle: Path,
    config: adequacy.AdequacyConfig,
    training_device: torch.device,
    evaluation_device: torch.device,
) -> None:
    core = baseline.load_nato_dataset(nato_bundle)
    quality = baseline_final.load_nato_subset(nato_bundle, "quality_pass")
    stress = baseline_final.load_nato_subset(
        nato_bundle, "field_quality_stress"
    )
    outputs = standard_final.empty_outputs()
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
                outer_fold,
                training_device,
                evaluation_device,
            )
            standard_final.merge_outputs(outputs, result)
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
    standard_final.save_outputs(output_dir, "outer_fold", outputs)


def run_poster(
    output_dir: Path,
    poster_csv: Path,
    config: adequacy.AdequacyConfig,
    training_device: torch.device,
    evaluation_device: torch.device,
) -> None:
    poster = baseline.load_poster_dataset(poster_csv)
    outputs = standard_final.empty_outputs()
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
            None,
            training_device,
            evaluation_device,
            {
                "heldout_substrate_family": heldout,
                "poster_axis": "400-1800_common",
            },
        )
        standard_final.merge_outputs(outputs, result)
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
    standard_final.save_outputs(output_dir, "poster", outputs)


def run_domain(
    output_dir: Path,
    nato_bundle: Path,
    config: adequacy.AdequacyConfig,
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
        "strict_core": nato_bundle / "domain_evaluation_partitions_core.csv",
        "quality_pass": (
            nato_bundle / "domain_evaluation_partitions_quality.csv"
        ),
    }
    outputs = standard_final.empty_outputs()
    for subset, dataset in datasets.items():
        partitions = pd.read_csv(partition_files[subset])
        for keys, frame in partitions.groupby(
            ["protocol", "domain_type", "heldout_domain"], sort=True
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
            standard_final.merge_outputs(outputs, result)
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
    standard_final.save_outputs(output_dir, "domain_transfer", outputs)


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_vae_adequacy"
        / "adequacy_v1",
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
        "--poster-csv",
        type=Path,
        default=repository
        / "Workspace"
        / "data"
        / "processed"
        / "consolidated_SERS.csv",
    )
    parser.add_argument(
        "--stage", choices=["outer", "poster", "domain", "all"], default="all"
    )
    parser.add_argument(
        "--training-device",
        choices=["cuda", "cpu", "auto"],
        default="cuda",
    )
    parser.add_argument(
        "--evaluation-device", choices=["cpu", "cuda"], default="cpu"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    output_dir = args.output_dir.resolve()
    selected = json.loads(
        (output_dir / "selected_configuration.json").read_text()
    )
    if not selected.get("selection_closed", False):
        raise ValueError("Adequacy selection is not closed")
    if any(
        bool(selected.get(field, False))
        for field in (
            "outer_used",
            "field_quality_stress_used",
            "domain_used",
            "poster_used",
        )
    ):
        raise ValueError("Locked outcomes were used during selection")
    config = config_from_selected(selected)
    if config.identifier != selected["identifier"]:
        raise ValueError("Selected adequacy configuration is inconsistent")
    training_device = baseline.resolve_device(args.training_device)
    evaluation_device = baseline.resolve_device(args.evaluation_device)
    if args.stage in {"outer", "all"}:
        run_outer(
            output_dir,
            args.nato_bundle.resolve(),
            config,
            training_device,
            evaluation_device,
        )
    if args.stage in {"poster", "all"}:
        run_poster(
            output_dir,
            args.poster_csv.resolve(),
            config,
            training_device,
            evaluation_device,
        )
    if args.stage in {"domain", "all"}:
        run_domain(
            output_dir,
            args.nato_bundle.resolve(),
            config,
            training_device,
            evaluation_device,
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "stage": args.stage,
                "configuration": config.identifier,
                "training_device": str(training_device),
                "evaluation_device": str(evaluation_device),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
