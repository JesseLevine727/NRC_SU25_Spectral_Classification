#!/usr/bin/env python3
"""Compare a clean SERS baseline rebuild with the frozen reference exactly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import sers_baseline_common as common


EXACT_JSON = [
    "predeclared_protocol.json",
    "selected_configurations.json",
    "ae_selected_configurations.json",
    "dae_selected_configurations.json",
    "pre_outer_advancing_view_decision.json",
]

EXACT_CSV = [
    "ae_search_fold_metrics.csv",
    "dae_search_fold_metrics.csv",
    "ae_search_core_summary.csv",
    "dae_search_core_summary.csv",
    "ae_selection_metrics.csv",
    "dae_selection_metrics.csv",
    "control_metrics.csv",
    "control_predictions.csv",
    "outer_fold_metrics.csv",
    "outer_fold_predictions.csv",
    "outer_fold_corruption_metrics.csv",
    "outer_fold_reconstruction_metrics.csv",
    "poster_metrics.csv",
    "poster_predictions.csv",
    "poster_corruption_metrics.csv",
    "poster_reconstruction_metrics.csv",
    "domain_transfer_metrics.csv",
    "domain_transfer_predictions.csv",
    "domain_transfer_corruption_metrics.csv",
    "domain_transfer_reconstruction_metrics.csv",
]

HISTORY_CSV = [
    "ae_search_training_histories.csv",
    "dae_search_training_histories.csv",
    "outer_fold_training_histories.csv",
    "poster_training_histories.csv",
    "domain_transfer_training_histories.csv",
    "training_histories.csv",
]

# The original AE core screen predates a non-model source snapshot change.
# Its execution fingerprint must remain different as honest provenance even
# though all tensors, histories, folds, and scientific metric values are exact.
ALLOWED_PROVENANCE_DIFFERENCES = {
    "ae_search_fold_metrics.csv": ["execution_fingerprint"],
}


def compare_json(reference: Path, rebuild: Path) -> dict[str, Any]:
    failures: list[str] = []
    for name in EXACT_JSON:
        left = json.loads((reference / name).read_text())
        right = json.loads((rebuild / name).read_text())
        if left != right:
            failures.append(name)
    return {
        "status": "exact" if not failures else "different",
        "files": len(EXACT_JSON),
        "failures": failures,
    }


def compare_csv_files(
    reference: Path,
    rebuild: Path,
    files: list[str],
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    for name in files:
        left_path = reference / name
        right_path = rebuild / name
        if common.sha256_file(left_path) == common.sha256_file(right_path):
            continue
        try:
            left = pd.read_csv(left_path, low_memory=False)
            right = pd.read_csv(right_path, low_memory=False)
            for column in ALLOWED_PROVENANCE_DIFFERENCES.get(name, []):
                if column not in left or column not in right:
                    raise ValueError(
                        f"allowed provenance column missing: {column}"
                    )
                left = left.drop(columns=column)
                right = right.drop(columns=column)
            pd.testing.assert_frame_equal(
                left,
                right,
                check_dtype=True,
                check_exact=True,
                check_like=False,
            )
        except Exception as exc:
            failures.append(
                {
                    "file": name,
                    "difference": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "status": "exact" if not failures else "different",
        "files": len(files),
        "failures": failures,
    }


def compare_npz_directory(
    reference: Path,
    rebuild: Path,
    directory: str,
) -> dict[str, Any]:
    left_files = {
        str(path.relative_to(reference / directory)): path
        for path in (reference / directory).rglob("*.npz")
    }
    right_files = {
        str(path.relative_to(rebuild / directory)): path
        for path in (rebuild / directory).rglob("*.npz")
    }
    failures: list[str] = []
    if set(left_files) != set(right_files):
        missing = sorted(set(left_files) - set(right_files))
        extra = sorted(set(right_files) - set(left_files))
        failures.extend(f"missing: {name}" for name in missing)
        failures.extend(f"extra: {name}" for name in extra)
    for relative in sorted(set(left_files) & set(right_files)):
        try:
            with np.load(left_files[relative], allow_pickle=False) as left:
                with np.load(right_files[relative], allow_pickle=False) as right:
                    if left.files != right.files:
                        failures.append(f"{relative}: archive keys/order differ")
                        continue
                    for key in left.files:
                        left_value = left[key]
                        right_value = right[key]
                        if np.issubdtype(left_value.dtype, np.number):
                            values_equal = np.array_equal(
                                left_value,
                                right_value,
                                equal_nan=True,
                            )
                        else:
                            values_equal = np.array_equal(
                                left_value,
                                right_value,
                            )
                        if (
                            left_value.dtype != right_value.dtype
                            or left_value.shape != right_value.shape
                            or not values_equal
                        ):
                            failures.append(f"{relative}: {key} differs")
                            break
        except Exception as exc:
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")
    return {
        "status": "exact" if not failures else "different",
        "files": len(left_files),
        "failures": failures,
    }


def metadata_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            metadata_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            metadata_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right, equal_nan=True)
        )
    return left == right


def compare_checkpoints(reference: Path, rebuild: Path) -> dict[str, Any]:
    directory = "checkpoints"
    left_files = {
        str(path.relative_to(reference / directory)): path
        for path in (reference / directory).rglob("*.pt")
    }
    right_files = {
        str(path.relative_to(rebuild / directory)): path
        for path in (rebuild / directory).rglob("*.pt")
    }
    failures: list[str] = []
    if set(left_files) != set(right_files):
        failures.extend(
            f"missing: {name}"
            for name in sorted(set(left_files) - set(right_files))
        )
        failures.extend(
            f"extra: {name}"
            for name in sorted(set(right_files) - set(left_files))
        )
    for relative in sorted(set(left_files) & set(right_files)):
        try:
            left = torch.load(
                left_files[relative],
                map_location="cpu",
                weights_only=False,
            )
            right = torch.load(
                right_files[relative],
                map_location="cpu",
                weights_only=False,
            )
            left_hash = common.state_dict_sha256(left["state_dict"])
            right_hash = common.state_dict_sha256(right["state_dict"])
            if left_hash != right_hash:
                failures.append(f"{relative}: state tensors differ")
                continue
            if left.get("state_sha256") != left_hash:
                failures.append(f"{relative}: reference state hash invalid")
            if right.get("state_sha256") != right_hash:
                failures.append(f"{relative}: rebuild state hash invalid")
            if not metadata_equal(left.get("metadata"), right.get("metadata")):
                failures.append(f"{relative}: checkpoint metadata differ")
        except Exception as exc:
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")
    return {
        "status": "exact" if not failures else "different",
        "files": len(left_files),
        "failures": failures,
    }


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    root = repository / "Workspace" / "sers_representation_baselines"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=root / "baselines_v1",
    )
    parser.add_argument(
        "--rebuild",
        type=Path,
        default=root / "baselines_v1_rebuild",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Defaults to <reference>/clean_rebuild_comparison.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    reference = args.reference.resolve()
    rebuild = args.rebuild.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else reference / "clean_rebuild_comparison.json"
    )
    comparisons = {
        "selected_configurations_and_protocol": compare_json(
            reference,
            rebuild,
        ),
        "fold_metrics_and_predictions": compare_csv_files(
            reference,
            rebuild,
            EXACT_CSV,
        ),
        "training_histories": compare_csv_files(
            reference,
            rebuild,
            HISTORY_CSV,
        ),
        "embeddings": compare_npz_directory(
            reference,
            rebuild,
            "embeddings",
        ),
        "reconstructions": compare_npz_directory(
            reference,
            rebuild,
            "reconstructions",
        ),
        "checkpoint_tensors": compare_checkpoints(reference, rebuild),
    }
    all_exact = all(
        item["status"] == "exact" for item in comparisons.values()
    )
    report = {
        "comparison_version": "sers-baseline-clean-rebuild-comparison-v1",
        "reference": str(reference),
        "rebuild": str(rebuild),
        "status": "exact_match" if all_exact else "mismatch",
        "all_required_comparisons_exact": all_exact,
        "comparison_semantics": {
            "JSON": "parsed values exactly equal",
        "CSV": (
                "byte-identical or parsed frames equal with exact values, "
                "dtypes, columns, and row order; the original AE-core "
                "execution fingerprint is retained as an allowed provenance "
                "difference and is not a scientific metric"
            ),
            "NPZ": "archive keys, dtypes, shapes, and every array value equal",
            "checkpoints": (
                "relative path set, every state tensor hash, and metadata equal"
            ),
        },
        "comparisons": comparisons,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "comparisons": len(comparisons),
            }
        )
    )
    return 0 if all_exact else 1


if __name__ == "__main__":
    sys.exit(main())
