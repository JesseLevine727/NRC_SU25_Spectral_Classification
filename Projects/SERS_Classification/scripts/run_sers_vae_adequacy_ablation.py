#!/usr/bin/env python3
"""Run preregistered sequential inner-only SERS VAE adequacy ablations."""

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

import pandas as pd
import torch

import run_sers_representation_baselines as baseline_selection
import run_sers_vae_adequacy_selection as selection
import sers_baseline_common as baseline
import sers_vae_adequacy_common as adequacy


def write_json(path: Path, value: Any) -> None:
    baseline_selection.write_json(path, value)


def fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode())
        digest.update(baseline.sha256_file(path).encode())
    digest.update("sers-vae-adequacy-ablation-cache-v1".encode())
    return digest.hexdigest()


def copy_amendment(output_dir: Path, source: Path) -> None:
    target = output_dir / "protocol_amendment_2.json"
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError("Output contains a different stage-2 amendment")
    if not target.exists():
        target.write_bytes(source.read_bytes())


def matching_records(
    records: pd.DataFrame, config: adequacy.AdequacyConfig, subset: str,
    representation: str
) -> pd.DataFrame:
    return records[
        (records["identifier"] == config.identifier)
        & (records["checkpoint_epoch"] == config.maximum_epoch)
        & (records["subset"] == subset)
        & (records["representation"] == representation)
    ].copy()


def matching_histories(
    histories: pd.DataFrame, config: adequacy.AdequacyConfig, subset: str,
    representation: str
) -> pd.DataFrame:
    return histories[
        (histories["configuration"] == config.identifier)
        & (histories["subset"] == subset)
        & (histories["representation"] == representation)
    ].copy()


def append_unique(
    current: pd.DataFrame, additional: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    return (
        pd.concat([current, additional], ignore_index=True)
        .drop_duplicates(keys, keep="last")
        .reset_index(drop=True)
    )


def rank_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["converged"] = ranked["converged"].fillna(False)
    return ranked.sort_values(
        [
            "converged",
            "passes_all_gates",
            "gate_count",
            "selection_utility",
            "parameter_count",
            "identifier",
        ],
        ascending=[False, False, False, False, True, True],
    )


def stage(
    name: str,
    configs: Sequence[adequacy.AdequacyConfig],
    dataset: baseline.SpectralDataset,
    protocol: dict[str, Any],
    baseline_bundle: Path,
    standard_bundle: Path,
    output_dir: Path,
    cache_fingerprint: str,
    device: torch.device,
    all_records: pd.DataFrame,
    all_histories: pd.DataFrame,
) -> tuple[
    adequacy.AdequacyConfig, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    stage_records: list[pd.DataFrame] = []
    stage_histories: list[pd.DataFrame] = []
    for config in configs:
        reused_records = matching_records(
            all_records, config, "strict_core", "arpls_minmax"
        )
        reused_histories = matching_histories(
            all_histories, config, "strict_core", "arpls_minmax"
        )
        if len(reused_records) == 20 and not reused_histories.empty:
            stage_records.append(reused_records)
            stage_histories.append(reused_histories)
            continue
        new_records, new_histories = selection.run_grid(
            dataset,
            "arpls_minmax",
            "strict_core",
            [config],
            [config.maximum_epoch],
            output_dir,
            name,
            cache_fingerprint,
            device,
            torch.device("cpu"),
            selection.original_seed_lookup(standard_bundle),
        )
        stage_records.append(new_records)
        stage_histories.append(new_histories)
        all_records = append_unique(
            all_records,
            new_records,
            [
                "identifier",
                "checkpoint_epoch",
                "representation",
                "subset",
                "outer_fold",
                "inner_fold",
            ],
        )
        all_histories = append_unique(
            all_histories,
            new_histories,
            [
                "configuration",
                "representation",
                "subset",
                "outer_fold",
                "inner_fold",
                "epoch",
            ],
        )
        all_records.to_csv(output_dir / "checkpoint_metrics.csv", index=False)
        all_histories.to_csv(output_dir / "training_histories.csv", index=False)
    records = pd.concat(stage_records, ignore_index=True)
    histories = pd.concat(stage_histories, ignore_index=True)
    ae_reference = json.loads(
        (baseline_bundle / "selected_configurations.json").read_text()
    )["autoencoders"]["arpls_minmax"]
    summary = selection.apply_gates(
        selection.aggregate_checkpoint_metrics(records),
        protocol,
        ae_reference,
    )
    convergence = selection.convergence_table(histories, summary)
    summary = summary.merge(
        convergence,
        on=["identifier", "checkpoint_epoch"],
        how="left",
    )
    summary["stage"] = name
    ranked = rank_summary(summary)
    winner_row = ranked.iloc[0]
    lookup = {config.identifier: config for config in configs}
    winner = lookup[str(winner_row["identifier"])]
    records.to_csv(output_dir / f"{name}_fold_metrics.csv", index=False)
    summary.to_csv(output_dir / f"{name}_summary.csv", index=False)
    write_json(
        output_dir / f"{name}_decision.json",
        {
            "stage": name,
            "selected_identifier": winner.identifier,
            "selected_metrics": baseline_selection.json_clean(
                winner_row.to_dict()
            ),
            "candidate_count": len(configs),
            "all_candidates": [
                baseline_selection.json_clean(row)
                for row in ranked.to_dict(orient="records")
            ],
            "outer_used": False,
            "field_quality_stress_used": False,
            "domain_used": False,
            "poster_used": False,
        },
    )
    return winner, all_records, all_histories, ranked


def quality_confirmation(
    configs: Sequence[adequacy.AdequacyConfig],
    strict_ranked: pd.DataFrame,
    dataset: baseline.SpectralDataset,
    protocol: dict[str, Any],
    baseline_bundle: Path,
    standard_bundle: Path,
    output_dir: Path,
    cache_fingerprint: str,
    device: torch.device,
    all_records: pd.DataFrame,
    all_histories: pd.DataFrame,
) -> tuple[
    adequacy.AdequacyConfig, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    carried_ids = strict_ranked.head(2)["identifier"].astype(str).tolist()
    lookup = {config.identifier: config for config in configs}
    carried = [lookup[value] for value in carried_ids]
    records, histories = selection.run_grid(
        dataset,
        "arpls_minmax",
        "quality_pass",
        carried,
        [500],
        output_dir,
        "stage_2_quality_confirmation",
        cache_fingerprint,
        device,
        torch.device("cpu"),
        None,
    )
    all_records = append_unique(
        all_records,
        records,
        [
            "identifier",
            "checkpoint_epoch",
            "representation",
            "subset",
            "outer_fold",
            "inner_fold",
        ],
    )
    all_histories = append_unique(
        all_histories,
        histories,
        [
            "configuration",
            "representation",
            "subset",
            "outer_fold",
            "inner_fold",
            "epoch",
        ],
    )
    quality = selection.aggregate_checkpoint_metrics(records)
    strict = strict_ranked[
        strict_ranked["identifier"].isin(carried_ids)
    ][
        [
            "identifier",
            "latent_target_balanced_accuracy",
            "passes_all_gates",
            "gate_count",
            "selection_utility",
            "parameter_count",
        ]
    ].rename(
        columns={
            "latent_target_balanced_accuracy": (
                "strict_latent_target_balanced_accuracy"
            )
        }
    )
    combined = strict.merge(
        quality[
            ["identifier", "latent_target_balanced_accuracy"]
        ].rename(
            columns={
                "latent_target_balanced_accuracy": (
                    "quality_latent_target_balanced_accuracy"
                )
            }
        ),
        on="identifier",
        how="left",
    )
    combined["quality_target_delta"] = (
        combined["quality_latent_target_balanced_accuracy"]
        - combined["strict_latent_target_balanced_accuracy"]
    )
    combined["gate_quality_sensitivity"] = (
        combined["quality_target_delta"] >= -0.05
    )
    combined = combined.sort_values(
        [
            "gate_quality_sensitivity",
            "passes_all_gates",
            "gate_count",
            "selection_utility",
            "parameter_count",
            "identifier",
        ],
        ascending=[False, False, False, False, True, True],
    )
    winner = lookup[str(combined.iloc[0]["identifier"])]
    records.to_csv(
        output_dir / "stage_2_quality_confirmation_fold_metrics.csv",
        index=False,
    )
    combined.to_csv(
        output_dir / "stage_2_quality_confirmation_summary.csv", index=False
    )
    all_records.to_csv(output_dir / "checkpoint_metrics.csv", index=False)
    all_histories.to_csv(output_dir / "training_histories.csv", index=False)
    return winner, all_records, all_histories, combined


def mandatory_minimal(
    config: adequacy.AdequacyConfig,
    dataset: baseline.SpectralDataset,
    standard_bundle: Path,
    output_dir: Path,
    cache_fingerprint: str,
    device: torch.device,
    all_records: pd.DataFrame,
    all_histories: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    for subset in ("strict_core", "quality_pass"):
        records, history = selection.run_grid(
            dataset,
            "minimal_minmax",
            subset,
            [config],
            [500],
            output_dir,
            f"stage_2_minimal_{subset}",
            cache_fingerprint,
            device,
            torch.device("cpu"),
            (
                selection.original_seed_lookup(standard_bundle)
                if subset == "strict_core"
                else None
            ),
        )
        frames.append(records)
        histories.append(history)
        all_records = append_unique(
            all_records,
            records,
            [
                "identifier",
                "checkpoint_epoch",
                "representation",
                "subset",
                "outer_fold",
                "inner_fold",
            ],
        )
        all_histories = append_unique(
            all_histories,
            history,
            [
                "configuration",
                "representation",
                "subset",
                "outer_fold",
                "inner_fold",
                "epoch",
            ],
        )
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(output_dir / "minimal_sensitivity_metrics.csv", index=False)
    all_records.to_csv(output_dir / "checkpoint_metrics.csv", index=False)
    all_histories.to_csv(output_dir / "training_histories.csv", index=False)
    return result, all_records, all_histories


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repository / "configs" / "sers_vae_adequacy_v1.json",
    )
    parser.add_argument(
        "--amendment-1",
        type=Path,
        default=repository
        / "configs"
        / "sers_vae_adequacy_v1_amendment_1.json",
    )
    parser.add_argument(
        "--amendment-2",
        type=Path,
        default=repository
        / "configs"
        / "sers_vae_adequacy_v1_amendment_2.json",
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
        "--standard-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_standard_vae"
        / "standard_vae_v1",
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
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_vae_adequacy"
        / "adequacy_v1",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    protocol = selection.load_protocol(args.protocol)
    amendment = json.loads(args.amendment_2.read_text())
    if (
        amendment["amendment_version"]
        != "sers-vae-adequacy-v1-amendment-2"
        or amendment["status_before_stage_2_model_execution"]
        != "predeclared"
    ):
        raise ValueError("Stage-2 amendment is not predeclared")
    copy_amendment(args.output_dir, args.amendment_2)
    cache_fingerprint = fingerprint(
        [
            args.protocol,
            args.amendment_1,
            args.amendment_2,
            Path(adequacy.__file__),
            Path(selection.__file__),
            Path(__file__),
        ]
    )
    dataset = baseline.load_nato_dataset(args.nato_bundle)
    all_records = pd.read_csv(args.output_dir / "checkpoint_metrics.csv")
    all_histories = pd.read_csv(args.output_dir / "training_histories.csv")
    base = adequacy.AdequacyConfig(maximum_epoch=500)
    architecture_configs = [
        base,
        adequacy.AdequacyConfig(
            architecture="residual_multiscale", maximum_epoch=500
        ),
        adequacy.AdequacyConfig(
            architecture="single_pool_peak", maximum_epoch=500
        ),
    ]
    architecture, all_records, all_histories, _ = stage(
        "stage_2_architecture",
        architecture_configs,
        dataset,
        protocol,
        args.baseline_bundle,
        args.standard_bundle,
        args.output_dir,
        cache_fingerprint,
        torch.device(args.device),
        all_records,
        all_histories,
    )
    loss_configs = [
        adequacy.AdequacyConfig(
            architecture=architecture.architecture,
            reconstruction_loss=loss,
            maximum_epoch=500,
        )
        for loss in ("spectral_composite", "peak_multiscale")
    ]
    loss, all_records, all_histories, _ = stage(
        "stage_2_loss",
        loss_configs,
        dataset,
        protocol,
        args.baseline_bundle,
        args.standard_bundle,
        args.output_dir,
        cache_fingerprint,
        torch.device(args.device),
        all_records,
        all_histories,
    )
    latent_configs = [
        adequacy.AdequacyConfig(
            architecture=loss.architecture,
            reconstruction_loss=loss.reconstruction_loss,
            latent_dimension=dimension,
            maximum_epoch=500,
        )
        for dimension in (32, 64, 128)
    ]
    latent, all_records, all_histories, _ = stage(
        "stage_2_latent",
        latent_configs,
        dataset,
        protocol,
        args.baseline_bundle,
        args.standard_bundle,
        args.output_dir,
        cache_fingerprint,
        torch.device(args.device),
        all_records,
        all_histories,
    )
    beta_configs = [
        adequacy.AdequacyConfig(
            architecture=latent.architecture,
            reconstruction_loss=latent.reconstruction_loss,
            latent_dimension=latent.latent_dimension,
            beta_target=beta,
            maximum_epoch=500,
        )
        for beta in (0.25, 1.0, 4.0)
    ]
    _, all_records, all_histories, beta_ranked = stage(
        "stage_2_beta",
        beta_configs,
        dataset,
        protocol,
        args.baseline_bundle,
        args.standard_bundle,
        args.output_dir,
        cache_fingerprint,
        torch.device(args.device),
        all_records,
        all_histories,
    )
    selected, all_records, all_histories, quality = quality_confirmation(
        beta_configs,
        beta_ranked,
        dataset,
        protocol,
        args.baseline_bundle,
        args.standard_bundle,
        args.output_dir,
        cache_fingerprint,
        torch.device(args.device),
        all_records,
        all_histories,
    )
    minimal, all_records, all_histories = mandatory_minimal(
        selected,
        dataset,
        args.standard_bundle,
        args.output_dir,
        cache_fingerprint,
        torch.device(args.device),
        all_records,
        all_histories,
    )
    write_json(
        args.output_dir / "selected_configuration.json",
        {
            **selected.record(),
            "selection_closed": True,
            "selection_data": (
                "NATO master-sample-grouped nested inner validation only"
            ),
            "selection_representation": "arpls_minmax",
            "mandatory_sensitivity_representation": "minimal_minmax",
            "quality_confirmation": baseline_selection.json_clean(
                quality.to_dict(orient="records")
            ),
            "minimal_sensitivity_summary": baseline_selection.json_clean(
                selection.aggregate_checkpoint_metrics(minimal).to_dict(
                    orient="records"
                )
            ),
            "outer_used": False,
            "field_quality_stress_used": False,
            "domain_used": False,
            "poster_used": False,
        },
    )


if __name__ == "__main__":
    main()
