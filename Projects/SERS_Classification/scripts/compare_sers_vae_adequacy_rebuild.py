#!/usr/bin/env python3
"""Compare two SERS VAE adequacy bundles with registered semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


HISTORY_TOLERANCE = 1.0e-12


def compare_csv(
    left: Path, right: Path, tolerance: float | None = None
) -> tuple[bool, str]:
    if left.read_bytes() == right.read_bytes():
        return True, "byte-identical"
    first = pd.read_csv(left, low_memory=False)
    second = pd.read_csv(right, low_memory=False)
    try:
        pd.testing.assert_frame_equal(
            first,
            second,
            check_dtype=True,
            check_exact=tolerance is None,
            rtol=0.0,
            atol=0.0 if tolerance is None else tolerance,
            check_like=False,
        )
        return (
            True,
            "parsed-frame-exact"
            if tolerance is None
            else f"parsed-frame-within-{tolerance:g}",
        )
    except AssertionError as error:
        return False, str(error)


def compare_json(left: Path, right: Path) -> tuple[bool, str]:
    first = json.loads(left.read_text())
    second = json.loads(right.read_text())
    return (
        first == second,
        "parsed-json-exact" if first == second else "JSON differs",
    )


def compare_text(left: Path, right: Path) -> tuple[bool, str]:
    same = left.read_bytes() == right.read_bytes()
    return same, "byte-identical" if same else "bytes differ"


def compare_npz(left: Path, right: Path) -> tuple[bool, str]:
    first = np.load(left)
    second = np.load(right)
    if list(first.files) != list(second.files):
        return False, "archive keys differ"
    for key in first.files:
        a, b = first[key], second[key]
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            return False, f"array differs: {key}"
    return True, "arrays-exact"


def compare_state_dict(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> tuple[bool, str]:
    if set(first) != set(second):
        return False, "state keys differ"
    for key in first:
        a, b = first[key], second[key]
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            return False, f"tensor differs: {key}"
    return True, "tensors-exact"


def compare_checkpoint(left: Path, right: Path) -> tuple[bool, str]:
    first = torch.load(left, map_location="cpu", weights_only=False)
    second = torch.load(right, map_location="cpu", weights_only=False)
    if first["metadata"] != second["metadata"]:
        return False, "metadata differs"
    if first["state_sha256"] != second["state_sha256"]:
        return False, "state hash differs"
    return compare_state_dict(first["state_dict"], second["state_dict"])


def compare_optimizer_state(first: Any, second: Any) -> bool:
    if torch.is_tensor(first) and torch.is_tensor(second):
        return (
            first.dtype == second.dtype
            and first.shape == second.shape
            and torch.equal(first, second)
        )
    if isinstance(first, dict) and isinstance(second, dict):
        return set(first) == set(second) and all(
            compare_optimizer_state(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            compare_optimizer_state(a, b) for a, b in zip(first, second)
        )
    return first == second


def compare_selection_cache(left: Path, right: Path) -> tuple[bool, str]:
    first = torch.load(left, map_location="cpu", weights_only=False)
    second = torch.load(right, map_location="cpu", weights_only=False)
    for key in ("execution_fingerprint", "config", "run_seed"):
        if first[key] != second[key]:
            return False, f"{key} differs"
    if list(first["states"]) != list(second["states"]):
        return False, "checkpoint epoch keys differ"
    for epoch in first["states"]:
        ok, detail = compare_state_dict(
            first["states"][epoch], second["states"][epoch]
        )
        if not ok:
            return False, f"epoch {epoch}: {detail}"
    if not compare_optimizer_state(
        first["optimizer_states"], second["optimizer_states"]
    ):
        return False, "optimizer state differs"
    try:
        pd.testing.assert_frame_equal(
            first["history"],
            second["history"],
            check_dtype=True,
            check_exact=False,
            rtol=0.0,
            atol=HISTORY_TOLERANCE,
        )
    except AssertionError as error:
        return False, f"history differs: {error}"
    return True, "states-exact-history-within-tolerance"


def compare_file_group(
    reference: Path,
    rebuild: Path,
    paths: list[str],
    comparator: Any,
) -> dict[str, Any]:
    failures = []
    for relative in paths:
        left, right = reference / relative, rebuild / relative
        if not left.is_file() or not right.is_file():
            failures.append({"path": relative, "detail": "missing"})
            continue
        ok, detail = comparator(left, right)
        if not ok:
            failures.append({"path": relative, "detail": detail})
    return {
        "status": "exact" if not failures else "mismatch",
        "files": len(paths),
        "failures": failures,
    }


def compare_tree(
    reference: Path,
    rebuild: Path,
    directory: str,
    suffix: str,
    comparator: Any,
) -> dict[str, Any]:
    left_paths = sorted(
        str(path.relative_to(reference))
        for path in (reference / directory).rglob(f"*{suffix}")
    )
    right_paths = sorted(
        str(path.relative_to(rebuild))
        for path in (rebuild / directory).rglob(f"*{suffix}")
    )
    failures = []
    if left_paths != right_paths:
        failures.append(
            {
                "path": directory,
                "detail": (
                    f"relative path sets differ: {len(left_paths)} vs "
                    f"{len(right_paths)}"
                ),
            }
        )
    else:
        for relative in left_paths:
            ok, detail = comparator(
                reference / relative, rebuild / relative
            )
            if not ok:
                failures.append({"path": relative, "detail": detail})
    return {
        "status": "exact" if not failures else "mismatch",
        "files": len(left_paths),
        "failures": failures,
    }


def compare(reference: Path, rebuild: Path) -> dict[str, Any]:
    json_files = [
        "predeclared_protocol.json",
        "protocol_amendment_1.json",
        "protocol_amendment_2.json",
        "input_hashes.json",
        "environment.json",
        "existing_run_audit.json",
        "first_100_reproduction.json",
        "stage_1_decision.json",
        "stage_2_architecture_decision.json",
        "stage_2_loss_decision.json",
        "stage_2_latent_decision.json",
        "stage_2_beta_decision.json",
        "selected_configuration.json",
        "failure_attribution.json",
        "compute_accounting.json",
        "final_decisions.json",
    ]
    scientific_csv = [
        "checkpoint_metrics.csv",
        "stage_summaries.csv",
        "stage_2_architecture_fold_metrics.csv",
        "stage_2_architecture_summary.csv",
        "stage_2_loss_fold_metrics.csv",
        "stage_2_loss_summary.csv",
        "stage_2_latent_fold_metrics.csv",
        "stage_2_latent_summary.csv",
        "stage_2_beta_fold_metrics.csv",
        "stage_2_beta_summary.csv",
        "stage_2_quality_confirmation_fold_metrics.csv",
        "stage_2_quality_confirmation_summary.csv",
        "minimal_sensitivity_metrics.csv",
        "outer_fold_metrics.csv",
        "outer_fold_predictions.csv",
        "outer_fold_reconstruction_metrics.csv",
        "outer_fold_corruption_metrics.csv",
        "poster_metrics.csv",
        "poster_predictions.csv",
        "poster_reconstruction_metrics.csv",
        "poster_corruption_metrics.csv",
        "domain_transfer_metrics.csv",
        "domain_transfer_predictions.csv",
        "domain_transfer_reconstruction_metrics.csv",
        "domain_transfer_corruption_metrics.csv",
        "per_spectrum_predictions.csv",
        "reconstruction_metrics.csv",
        "corruption_metrics.csv",
        "variational_metrics.csv",
        "failure_cases.csv",
        "per_class_metrics.csv",
        "per_instrument_failures.csv",
        "outer_fold_same_master_geometry.csv",
        "same_master_geometry_summary.csv",
        "instrument_reconstruction_summary.csv",
        "parameter_and_compute_accounting.csv",
        "comparator_summary.csv",
        "uncertainty_summary.csv",
    ]
    history_files = [
        "training_histories.csv",
        "outer_fold_training_histories.csv",
        "poster_training_histories.csv",
        "domain_transfer_training_histories.csv",
        "confirmatory_training_histories.csv",
    ]
    text_files = [
        "README.md",
        "DECISION_REGISTRY.md",
        "RESULTS_REPORT.md",
        "reproduction_commands.sh",
    ]
    comparisons = {
        "protocol_selection_and_decisions": compare_file_group(
            reference, rebuild, json_files, compare_json
        ),
        "canonical_scientific_tables": compare_file_group(
            reference,
            rebuild,
            scientific_csv,
            lambda left, right: compare_csv(left, right, None),
        ),
        "floating_training_histories": compare_file_group(
            reference,
            rebuild,
            history_files,
            lambda left, right: compare_csv(
                left, right, HISTORY_TOLERANCE
            ),
        ),
        "documentation": compare_file_group(
            reference, rebuild, text_files, compare_text
        ),
        "confirmatory_checkpoint_tensors": compare_tree(
            reference, rebuild, "checkpoints", ".pt", compare_checkpoint
        ),
        "selection_checkpoint_tensors": compare_tree(
            reference,
            rebuild,
            "selection_cache",
            ".pt",
            compare_selection_cache,
        ),
        "embeddings": compare_tree(
            reference, rebuild, "embeddings", ".npz", compare_npz
        ),
        "reconstructions": compare_tree(
            reference, rebuild, "reconstructions", ".npz", compare_npz
        ),
    }
    exact = all(value["status"] == "exact" for value in comparisons.values())
    return {
        "comparison_version": "sers-vae-adequacy-clean-rebuild-v1",
        "reference": str(reference),
        "rebuild": str(rebuild),
        "comparison_semantics": {
            "canonical_CSV": (
                "byte-identical or parsed frames exactly equal"
            ),
            "history_CSV": (
                f"exact schema/order and floating values within "
                f"{HISTORY_TOLERANCE:g}"
            ),
            "JSON": "parsed values exactly equal",
            "NPZ": (
                "keys, dtypes, shapes, and every array value exactly equal"
            ),
            "confirmatory_checkpoints": (
                "metadata, state hashes, and every tensor exactly equal"
            ),
            "selection_checkpoints": (
                "configuration/seed, every model and optimizer tensor exactly "
                "equal, histories within registered tolerance"
            ),
        },
        "comparisons": comparisons,
        "all_required_comparisons_exact": exact,
        "status": "exact_match" if exact else "mismatch",
    }


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_vae_adequacy"
        / "adequacy_v1",
    )
    parser.add_argument(
        "--rebuild",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_vae_adequacy"
        / "adequacy_v1_rebuild",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    reference = args.reference.resolve()
    rebuild = args.rebuild.resolve()
    result = compare(reference, rebuild)
    for directory in (reference, rebuild):
        (directory / "clean_rebuild_comparison.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(result, indent=2))
    return 0 if result["all_required_comparisons_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
