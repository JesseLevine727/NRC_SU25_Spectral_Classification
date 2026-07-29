#!/usr/bin/env python3
"""Compare a clean classical-benchmark rebuild with the canonical bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sers_classical_benchmark_common as common


CANONICAL_COMPUTATIONAL_ARTIFACTS = (
    "candidate_registry.csv",
    "inner_fold_metrics.csv",
    "outer_selection.csv",
    "global_champions.json",
    "outer_metrics.csv",
    "predictions.csv",
    "per_class_metrics.csv",
    "calibration_metrics.csv",
    "selective_metrics.csv",
    "field_stress_ood_metrics.csv",
    "outer_domain_slices.csv",
    "domain_metrics.csv",
    "domain_predictions.csv",
    "domain_per_class_metrics.csv",
    "domain_selective_metrics.csv",
    "learning_curve_metrics.csv",
    "negative_control_metrics.csv",
    "uncertainty_summary.csv",
    "domain_summary.csv",
    "learning_curve_summary.csv",
    "confusion_matrices.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path(
            "Workspace/sers_classical_benchmark/classical_benchmark_v2"
        ),
    )
    parser.add_argument(
        "--rebuild",
        type=Path,
        default=Path(
            "Workspace/sers_classical_benchmark/classical_benchmark_v2_rebuild"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for relative in CANONICAL_COMPUTATIONAL_ARTIFACTS:
        canonical = args.canonical / relative
        rebuild = args.rebuild / relative
        canonical_hash = common.sha256_file(canonical) if canonical.exists() else None
        rebuild_hash = common.sha256_file(rebuild) if rebuild.exists() else None
        rows.append(
            {
                "artifact": relative,
                "canonical_exists": canonical.exists(),
                "rebuild_exists": rebuild.exists(),
                "canonical_sha256": canonical_hash,
                "rebuild_sha256": rebuild_hash,
                "exact": canonical_hash is not None
                and canonical_hash == rebuild_hash,
            }
        )
    exact = all(row["exact"] for row in rows)
    report = {
        "status": "exact_match" if exact else "mismatch",
        "artifacts_compared": len(rows),
        "artifacts_exact": sum(row["exact"] for row in rows),
        "details": rows,
    }
    common.write_json(
        args.canonical / "clean_rebuild_comparison.json", report
    )
    print(json.dumps(report, indent=2))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
