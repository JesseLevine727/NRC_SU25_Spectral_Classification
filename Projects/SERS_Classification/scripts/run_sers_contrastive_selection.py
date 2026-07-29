#!/usr/bin/env python3
"""Run nested selection for the predeclared supervised-contrastive SERS model."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import sers_baseline_common as baseline
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
        "--output-dir",
        type=Path,
        default=Path(
            "Workspace/sers_supervised_contrastive/contrastive_v1"
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "all"),
        default="all",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
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
    if len(dataset.manifest) != int(protocol["inputs"]["strict_core_rows"]):
        raise ValueError("Unexpected strict-core count")
    classical_validation = json.loads(
        (args.classical_bundle / "validation_report.json").read_text()
    )
    if classical_validation.get("status") != "pass":
        raise ValueError("Classical bundle has not passed validation")
    classical_rebuild = args.classical_bundle / "clean_rebuild_comparison.json"
    if not classical_rebuild.exists():
        raise ValueError("Classical exact rebuild has not closed")
    if json.loads(classical_rebuild.read_text()).get("status") != "exact_match":
        raise ValueError("Classical clean rebuild is not exact")
    scripts = [Path(__file__).resolve(), Path(common.__file__).resolve()]
    write_json(
        args.output_dir / "input_hashes.json",
        {
            "protocol_sha256": baseline.sha256_file(args.protocol),
            "preprocessing_hash_catalog_sha256": baseline.sha256_file(
                args.bundle / "artifact_hashes.json"
            ),
            "classical_hash_catalog_sha256": baseline.sha256_file(
                args.classical_bundle / "artifact_hashes.json"
            ),
            "scripts": {
                str(path): baseline.sha256_file(path) for path in scripts
            },
        },
    )
    write_json(
        args.output_dir / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    )
    return dataset


def stage1_specs(protocol: dict[str, Any]) -> list[common.ModelSpec]:
    objective = protocol["objectives"]["domain_aware_successor"]
    specs: list[common.ModelSpec] = []
    for representation in protocol["inputs"]["representations"]:
        for contrastive_weight in objective[
            "supervised_contrastive_weights"
        ]:
            for margin_weight in objective["pair_margin_weights"]:
                specs.append(
                    common.ModelSpec(
                        name="domain_aware_stage1",
                        representation=representation,
                        architecture="legacy",
                        embedding_dimension=64,
                        classification_weight=float(
                            objective["classification_weight"]
                        ),
                        supervised_contrastive_weight=float(
                            contrastive_weight
                        ),
                        pair_margin_weight=float(margin_weight),
                        contrastive_temperature=float(
                            objective["temperature"]
                        ),
                        pair_margin=float(objective["margin"]),
                        domain_aware_positives=True,
                        hard_negative_mining=True,
                        domain_balanced_batches=True,
                    )
                )
    return specs


def stage2_specs(base: common.ModelSpec) -> list[common.ModelSpec]:
    specs: list[common.ModelSpec] = []
    for architecture in ("legacy", "compact"):
        for embedding_dimension in (32, 64):
            specs.append(
                common.ModelSpec(
                    **{
                        **base.__dict__,
                        "name": f"domain_aware_stage2_{architecture}_{embedding_dimension}",
                        "architecture": architecture,
                        "embedding_dimension": embedding_dimension,
                    }
                )
            )
    return specs


def spec_row(spec: common.ModelSpec) -> dict[str, Any]:
    return {
        "candidate_id": spec.candidate_id,
        **spec.__dict__,
    }


def evaluate_run(
    output_dir: Path,
    stage: str,
    dataset: baseline.SpectralDataset,
    protocol: dict[str, Any],
    spec: common.ModelSpec,
    outer_fold: int,
    inner_fold: int,
    device: torch.device,
    scope: str = "outer",
) -> dict[str, Any]:
    shard_dir = output_dir / "selection_shards" / stage
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{scope}__o{outer_fold}__i{inner_fold}__{spec.candidate_id}"
    )
    result_path = shard_dir / f"{stem}.json"
    history_path = shard_dir / f"{stem}__history.csv"
    if result_path.exists() and history_path.exists():
        return json.loads(result_path.read_text())
    manifest = dataset.manifest
    folds = manifest["grouped_sample_fold_5"].astype(int).to_numpy()
    quality_pass = (
        manifest["include_sers_qc_pass"].astype(bool).to_numpy()
    )
    train_mask = (
        quality_pass
        & (folds != outer_fold)
        & (folds != inner_fold)
    )
    validation_mask = quality_pass & (folds == inner_fold)
    train_groups = set(
        manifest.loc[train_mask, "master_sample_id"].astype(str)
    )
    validation_groups = set(
        manifest.loc[validation_mask, "master_sample_id"].astype(str)
    )
    if train_groups & validation_groups:
        raise ValueError("Master leakage in contrastive selection")
    seed = common.stable_seed(
        common.PROTOCOL_VERSION,
        stage,
        scope,
        outer_fold,
        inner_fold,
        spec.candidate_id,
        protocol["fixed_training"]["selection_seed"],
    )
    trained = common.train(
        dataset.representations[spec.representation][train_mask],
        manifest.loc[train_mask].reset_index(drop=True),
        dataset.representations[spec.representation][validation_mask],
        manifest.loc[validation_mask].reset_index(drop=True),
        spec,
        protocol,
        seed,
        device,
    )
    metrics = common.evaluate_validation(
        trained.model,
        spec,
        dataset.representations[spec.representation][train_mask],
        manifest.loc[train_mask].reset_index(drop=True),
        dataset.representations[spec.representation][validation_mask],
        manifest.loc[validation_mask].reset_index(drop=True),
        device,
    )
    gates = protocol["nested_selection"]["collapse_gates"]
    gate_margin = metrics["different_minus_same_margin"] > 0
    gate_rank = metrics["embedding_effective_rank"] >= float(
        gates["embedding_effective_rank_minimum"]
    )
    gate_classes = metrics["predicted_class_count"] >= 2
    result = {
        "stage": stage,
        "scope": scope,
        "outer_fold": outer_fold,
        "inner_validation_fold": inner_fold,
        **spec_row(spec),
        "seed": seed,
        "best_epoch": trained.best_epoch,
        "epochs_run": len(trained.history),
        "state_sha256": trained.state_sha256,
        **trained.parameters,
        "n_train": int(train_mask.sum()),
        "n_validation": int(validation_mask.sum()),
        "n_train_groups": len(train_groups),
        "n_validation_groups": len(validation_groups),
        **metrics,
        "gate_positive_margin": gate_margin,
        "gate_effective_rank": gate_rank,
        "gate_predicted_classes": gate_classes,
        "eligible": gate_margin and gate_rank and gate_classes,
    }
    write_json(result_path, result)
    trained.history.assign(
        stage=stage,
        scope=scope,
        outer_fold=outer_fold,
        inner_validation_fold=inner_fold,
        candidate_id=spec.candidate_id,
        seed=seed,
    ).to_csv(history_path, index=False)
    return result


def rank(frame: pd.DataFrame, required_folds: int) -> pd.DataFrame:
    eligible = frame[frame["eligible"].astype(bool)]
    summary = (
        eligible.groupby(
            [
                "candidate_id",
                "name",
                "representation",
                "architecture",
                "embedding_dimension",
                "classification_weight",
                "supervised_contrastive_weight",
                "pair_margin_weight",
                "contrastive_temperature",
                "pair_margin",
                "domain_aware_positives",
                "hard_negative_mining",
                "domain_balanced_batches",
                "total_parameters",
            ],
            as_index=False,
        )
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_distance_margin=("different_minus_same_margin", "mean"),
            mean_effective_rank=("embedding_effective_rank", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
            folds_evaluated=("balanced_accuracy", "count"),
        )
    )
    summary = summary[summary["folds_evaluated"] == required_folds]
    return summary.sort_values(
        [
            "mean_balanced_accuracy",
            "mean_macro_f1",
            "mean_distance_margin",
            "total_parameters",
            "candidate_id",
        ],
        ascending=[False, False, False, True, True],
        kind="stable",
    )


def run_stage1(
    output_dir: Path,
    dataset: baseline.SpectralDataset,
    protocol: dict[str, Any],
    device: torch.device,
) -> None:
    specs = stage1_specs(protocol)
    pd.DataFrame([spec_row(spec) for spec in specs]).to_csv(
        output_dir / "stage1_candidate_registry.csv", index=False
    )
    rows: list[dict[str, Any]] = []
    for outer_fold in range(5):
        for inner_fold in sorted(set(range(5)) - {outer_fold}):
            for spec in specs:
                rows.append(
                    evaluate_run(
                        output_dir,
                        "stage1",
                        dataset,
                        protocol,
                        spec,
                        outer_fold,
                        inner_fold,
                        device,
                    )
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "stage1_inner_metrics.csv", index=False)
    choices: list[dict[str, Any]] = []
    for outer_fold in range(5):
        ranked = rank(frame[frame["outer_fold"] == outer_fold], 4)
        if ranked.empty:
            raise RuntimeError(f"No eligible stage-1 model for outer {outer_fold}")
        choices.append({"scope": "outer", "outer_fold": outer_fold, **ranked.iloc[0].to_dict()})
    global_ranked = rank(frame, 20)
    if global_ranked.empty:
        raise RuntimeError("No global eligible stage-1 model")
    choices.append({"scope": "global", "outer_fold": -1, **global_ranked.iloc[0].to_dict()})
    pd.DataFrame(choices).to_csv(
        output_dir / "stage1_selection.csv", index=False
    )


def spec_from_selection(row: pd.Series) -> common.ModelSpec:
    return common.ModelSpec(
        name=str(row["name"]),
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


def run_stage2(
    output_dir: Path,
    dataset: baseline.SpectralDataset,
    protocol: dict[str, Any],
    device: torch.device,
) -> None:
    stage1 = pd.read_csv(output_dir / "stage1_selection.csv")
    rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for _, selected_row in stage1.iterrows():
        scope = str(selected_row["scope"])
        outer_fold = int(selected_row["outer_fold"])
        base = spec_from_selection(selected_row)
        specs = stage2_specs(base)
        registry.extend(
            [
                {
                    "scope": scope,
                    "selection_outer_fold": outer_fold,
                    **spec_row(spec),
                }
                for spec in specs
            ]
        )
        outer_folds = [outer_fold] if scope == "outer" else list(range(5))
        for evaluation_outer in outer_folds:
            for inner_fold in sorted(set(range(5)) - {evaluation_outer}):
                for spec in specs:
                    rows.append(
                        evaluate_run(
                            output_dir,
                            "stage2",
                            dataset,
                            protocol,
                            spec,
                            evaluation_outer,
                            inner_fold,
                            device,
                            scope=f"{scope}_{outer_fold}",
                        )
                    )
        relevant = pd.DataFrame(
            [
                row
                for row in rows
                if row["scope"] == f"{scope}_{outer_fold}"
            ]
        )
        ranked = rank(relevant, 4 if scope == "outer" else 20)
        if ranked.empty:
            raise RuntimeError(
                f"No eligible stage-2 model for {scope} {outer_fold}"
            )
        selections.append(
            {
                "scope": scope,
                "outer_fold": outer_fold,
                **ranked.iloc[0].to_dict(),
            }
        )
    pd.DataFrame(registry).drop_duplicates().to_csv(
        output_dir / "stage2_candidate_registry.csv", index=False
    )
    pd.DataFrame(rows).to_csv(
        output_dir / "stage2_inner_metrics.csv", index=False
    )
    pd.DataFrame(selections).to_csv(
        output_dir / "stage2_selection.csv", index=False
    )


def main() -> None:
    args = parse_args()
    protocol = common.load_protocol(args.protocol)
    dataset = initialize(args, protocol)
    device = baseline.resolve_device(args.device)
    stages = ("stage1", "stage2") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"START {stage}", flush=True)
        if stage == "stage1":
            run_stage1(args.output_dir, dataset, protocol, device)
        else:
            run_stage2(args.output_dir, dataset, protocol, device)
        print(f"DONE {stage}", flush=True)


if __name__ == "__main__":
    main()
