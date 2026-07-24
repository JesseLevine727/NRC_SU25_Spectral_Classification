#!/usr/bin/env python3
"""Compare two standard-VAE bundles using exact scientific semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def compare_csv(left: Path, right: Path) -> tuple[bool, str]:
    if left.read_bytes() == right.read_bytes():
        return True, "byte-identical"
    first = pd.read_csv(left, low_memory=False)
    second = pd.read_csv(right, low_memory=False)
    try:
        pd.testing.assert_frame_equal(
            first,
            second,
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
        return True, "parsed-frame-exact"
    except AssertionError as error:
        return False, str(error)


def compare_json(left: Path, right: Path) -> tuple[bool, str]:
    first = json.loads(left.read_text())
    second = json.loads(right.read_text())
    return (first == second, "parsed-json-exact" if first == second else "JSON differs")


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


def compare_checkpoint(left: Path, right: Path) -> tuple[bool, str]:
    first = torch.load(left, map_location="cpu", weights_only=False)
    second = torch.load(right, map_location="cpu", weights_only=False)
    if first["metadata"] != second["metadata"]:
        return False, "metadata differs"
    if first["state_sha256"] != second["state_sha256"]:
        return False, "state hash differs"
    if set(first["state_dict"]) != set(second["state_dict"]):
        return False, "state keys differ"
    for key in first["state_dict"]:
        a = first["state_dict"][key]
        b = second["state_dict"][key]
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            return False, f"tensor differs: {key}"
    return True, "tensors-exact"


def compare_file_group(
    reference: Path,
    rebuild: Path,
    paths: list[str],
    kind: str,
) -> dict[str, Any]:
    comparator = {"csv": compare_csv, "json": compare_json}[kind]
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
        failures.append({"path": directory, "detail": "relative path sets differ"})
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
    selections = compare_file_group(
        reference,
        rebuild,
        [
            "predeclared_protocol.json",
            "selected_configuration.json",
            "final_decisions.json",
        ],
        "json",
    )
    # CSV members in the mixed group are compared separately.
    selections_csv = compare_file_group(
        reference,
        rebuild,
        [
            "selection_core_summary.csv",
            "selection_summary.csv",
            "minimal_sensitivity_summary.csv",
        ],
        "csv",
    )
    selections["failures"] = selections["failures"] + selections_csv["failures"]
    selections["status"] = (
        "exact" if not selections["failures"] else "mismatch"
    )
    selections["files"] = 6

    metric_files = [
        "selection_fold_metrics.csv",
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
        "uncertainty_summary.csv",
        "split_seed_assignments.csv",
        "per_class_metrics.csv",
        "poster_localized_4np_failures.csv",
    ]
    comparisons = {
        "selected_configuration_and_protocol": selections,
        "fold_metrics_and_predictions": compare_file_group(
            reference, rebuild, metric_files, "csv"
        ),
        "training_histories": compare_file_group(
            reference,
            rebuild,
            [
                "selection_training_histories.csv",
                "outer_fold_training_histories.csv",
                "poster_training_histories.csv",
                "domain_transfer_training_histories.csv",
                "training_histories.csv",
            ],
            "csv",
        ),
        "checkpoint_tensors": compare_tree(
            reference, rebuild, "checkpoints", ".pt", compare_checkpoint
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
        "comparison_version": "sers-standard-vae-clean-rebuild-v1",
        "reference": str(reference),
        "rebuild": str(rebuild),
        "comparison_semantics": {
            "CSV": "byte-identical or parsed frames exactly equal",
            "JSON": "parsed values exactly equal",
            "NPZ": "keys, dtypes, shapes, and every array value exactly equal",
            "checkpoints": "metadata, state hashes, and every tensor exactly equal",
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
        / "sers_standard_vae"
        / "standard_vae_v1",
    )
    parser.add_argument(
        "--rebuild",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_standard_vae"
        / "standard_vae_v1_rebuild",
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
