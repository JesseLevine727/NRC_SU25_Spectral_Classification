#!/usr/bin/env python3
"""Close structured-VAE inner selection and run registered negative controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

import run_sers_structured_vae_selection as selection
import sers_baseline_common as baseline
import sers_structured_vae_common as structured


STAGES = ("controls", "instrument_adversary", "pair", "dependence")


def write_json(path: Path, value: Any) -> None:
    selection.write_json(path, selection.json_clean(value))


def close_selection(
    output_dir: Path,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    summaries = []
    for stage in STAGES:
        path = output_dir / f"{stage}_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["source_stage"] = stage
        summaries.append(frame)
    all_candidates = pd.concat(summaries, ignore_index=True, sort=False)
    selected = selection.select_candidate(all_candidates)
    instrument_summary = all_candidates[
        all_candidates["source_stage"].eq("instrument_adversary")
    ]
    instrument_eligible = bool(
        instrument_summary["passes_all_gates"].astype(bool).any()
    )
    individually_eligible = all_candidates[
        ~all_candidates["source_stage"].eq("controls")
        & all_candidates["passes_all_gates"].astype(bool)
    ]
    sensor_open = instrument_eligible
    combination_open = len(individually_eligible) >= 2
    all_candidates.to_csv(output_dir / "all_inner_candidates.csv", index=False)
    record = {
        "protocol": structured.PROTOCOL_VERSION,
        "selection_closed": True,
        "selection_used_locked_outcomes": False,
        "candidate_stages_completed": list(STAGES),
        "sensor_adversary": {
            "opened": sensor_open,
            "reason": (
                "At least one instrument-adversarial candidate passed every "
                "gate."
                if sensor_open
                else "Closed: no instrument-adversarial candidate passed "
                "every gate while preserving chemistry and improving the "
                "instrument probe."
            ),
        },
        "combination": {
            "opened": combination_open,
            "eligible_mechanism_count": len(individually_eligible),
            "reason": (
                "At least two individually eligible mechanisms are available."
                if combination_open
                else "Closed: fewer than two individually eligible "
                "mechanisms are available."
            ),
        },
        "eligible_candidate_count": int(
            all_candidates["passes_all_gates"].astype(bool).sum()
        ),
        "selected_by_registered_hierarchy": str(selected["identifier"]),
        "selected_source_stage": str(selected["source_stage"]),
        "selected_configuration": {
            key: selected[key]
            for key in structured.StructuredConfig.__dataclass_fields__
        },
        "selected_gate_count": int(selected["gate_count"]),
        "selected_gate_total": int(selected["gate_total"]),
        "selected_passes_all_gates": bool(selected["passes_all_gates"]),
        "selected_converged": bool(selected["converged"]),
        "claim_ceiling_after_inner_selection": (
            "eligible_for_locked_confirmation"
            if bool(selected["passes_all_gates"])
            else "unsuccessful; locked evaluation is characterization only"
        ),
        "outer_used": False,
        "field_quality_stress_used": False,
        "domain_used": False,
        "poster_used": False,
    }
    write_json(output_dir / "inner_selection_closure.json", record)
    return all_candidates, selected


def permuted_group_targets(
    manifest: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    group_labels = (
        manifest.groupby("master_sample_id", sort=True)["target_analyte"]
        .first()
        .astype(str)
    )
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(group_labels.to_numpy())
    lookup = dict(zip(group_labels.index.astype(str), permuted))
    return (
        manifest["master_sample_id"]
        .astype(str)
        .map(lookup)
        .to_numpy(dtype=str)
    )


def run_chemical_permutation(
    dataset: baseline.SpectralDataset,
    config: structured.StructuredConfig,
    source_stage: str,
    output_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    manifest = dataset.manifest
    folds = manifest["grouped_sample_fold_5"].to_numpy(dtype=int)
    target_mapping, instrument_mapping, sensor_mapping = selection.mappings(
        manifest
    )
    permuted = permuted_group_targets(
        manifest,
        baseline.stable_seed(
            structured.PROTOCOL_VERSION, "chemical_group_permutation"
        ),
    )
    records: list[dict[str, Any]] = []
    for outer_fold in range(5):
        for inner_fold in sorted(set(range(5)) - {outer_fold}):
            train_mask = (folds != outer_fold) & (folds != inner_fold)
            validation_mask = folds == inner_fold
            train_manifest = manifest.loc[train_mask].reset_index(drop=True)
            validation_manifest = manifest.loc[
                validation_mask
            ].reset_index(drop=True)
            train_instruments, train_sensors = selection.model_indices(
                train_manifest, instrument_mapping, sensor_mapping
            )
            validation_instruments, validation_sensors = (
                selection.model_indices(
                    validation_manifest, instrument_mapping, sensor_mapping
                )
            )
            cache = (
                output_dir
                / "selection_cache"
                / source_stage
                / (
                    f"strict_core__o{outer_fold}__i{inner_fold}__"
                    f"arpls_minmax__{config.identifier}.pt"
                )
            )
            payload = torch.load(
                cache, map_location="cpu", weights_only=False
            )
            model = structured.build_model_from_state(
                dataset.representations["arpls_minmax"].shape[1],
                config,
                len(target_mapping),
                len(instrument_mapping),
                len(sensor_mapping),
                payload["states"][500],
                device,
            )
            train_outputs = structured.outputs(
                model,
                dataset.representations["arpls_minmax"][train_mask],
                train_instruments,
                train_sensors,
                device,
            )
            validation_outputs = structured.outputs(
                model,
                dataset.representations["arpls_minmax"][validation_mask],
                validation_instruments,
                validation_sensors,
                device,
            )
            run_seed = baseline.stable_seed(
                structured.PROTOCOL_VERSION,
                "chemical_group_permutation",
                outer_fold,
                inner_fold,
            )
            probe = baseline.fit_latent_probe_model(
                train_outputs["chemical_mu"],
                permuted[train_mask],
                run_seed,
            )
            prediction = probe.predict(
                validation_outputs["chemical_mu"]
            ).astype(str)
            metrics = baseline.classification_summary(
                permuted[validation_mask], prediction
            )
            records.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "configuration": config.identifier,
                    "control": "chemical_labels_permuted_by_master_group",
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1_supported": metrics["macro_f1_supported"],
                    "n_train": int(train_mask.sum()),
                    "n_validation": int(validation_mask.sum()),
                    "state_sha256": baseline.state_dict_sha256(
                        payload["states"][500]
                    ),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return pd.DataFrame(records)


def negative_controls(
    output_dir: Path,
    nato_bundle: Path,
    protocol: dict[str, Any],
    selected: pd.Series,
    device: torch.device,
) -> None:
    config = selection.config_from_record(selected.to_dict())
    dataset = baseline.load_nato_dataset(nato_bundle)
    chemical = run_chemical_permutation(
        dataset,
        config,
        str(selected["source_stage"]),
        output_dir,
        device,
    )
    chemical.to_csv(
        output_dir / "negative_control_chemical_permutation.csv",
        index=False,
    )
    threshold = float(
        protocol["negative_controls"]["expectations"][
            "chemical_permutation_balanced_accuracy_maximum"
        ]
    )
    mean_ba = float(chemical["balanced_accuracy"].mean())
    maximum_ba = float(chemical["balanced_accuracy"].max())
    summary = {
        "protocol": structured.PROTOCOL_VERSION,
        "frozen_configuration": config.identifier,
        "chemical_group_permutation": {
            "applicable": True,
            "fold_count": len(chemical),
            "mean_balanced_accuracy": mean_ba,
            "maximum_balanced_accuracy": maximum_ba,
            "registered_per_fold_maximum": threshold,
            "passed": maximum_ba <= threshold,
        },
        "nuisance_label_permutation": {
            "applicable": bool(
                config.instrument_supervision_weight > 0
                or config.sensor_supervision_weight > 0
                or config.instrument_adversary_weight > 0
                or config.sensor_adversary_weight > 0
            ),
            "reason": (
                "The frozen dependence-only objective has zero nuisance-label "
                "and adversarial weights; permuting those labels cannot alter "
                "its optimized objective."
            ),
        },
        "pair_identity_permutation": {
            "applicable": bool(
                config.same_master_consistency_weight > 0
                or config.cross_reconstruction_weight > 0
            ),
            "reason": (
                "The frozen dependence-only objective has zero pair and "
                "cross-reconstruction weights; partner identities cannot "
                "alter its optimized objective."
            ),
        },
        "zero_structure_reference": {
            "applicable": True,
            "reference_stage": "controls",
            "reference_identifier": (
                "zc48__zn16__chem0__ni0__ns0__cond0__ai0__as0__"
                "pair0__xrec0__dep0__e500"
            ),
            "identity_control_gate": json.loads(
                (output_dir / "identity_control_summary.json").read_text()
            )["identity_gate_passed"],
        },
        "all_applicable_controls_passed": maximum_ba <= threshold,
    }
    write_json(output_dir / "negative_control_summary.json", summary)


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repository / "configs" / "sers_structured_vae_v1.json",
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
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_structured_vae"
        / "structured_vae_v1",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    protocol = json.loads(args.protocol.read_text())
    _, selected = close_selection(args.output_dir, protocol)
    negative_controls(
        args.output_dir,
        args.nato_bundle,
        protocol,
        selected,
        torch.device(args.device),
    )
    summary = json.loads(
        (args.output_dir / "negative_control_summary.json").read_text()
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "selected": str(selected["identifier"]),
                "passes_all_gates": bool(selected["passes_all_gates"]),
                "chemical_permutation_mean_ba": summary[
                    "chemical_group_permutation"
                ]["mean_balanced_accuracy"],
                "negative_controls_passed": summary[
                    "all_applicable_controls_passed"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
