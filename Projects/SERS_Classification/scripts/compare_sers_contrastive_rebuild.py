#!/usr/bin/env python3
"""Compare a clean supervised-contrastive rebuild with the canonical bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sers_baseline_common as baseline
import sers_classical_benchmark_common as classical


ARTIFACTS = (
    "final_input_hashes.json",
    "stage1_candidate_registry.csv",
    "stage1_inner_metrics.csv",
    "stage1_selection.csv",
    "stage2_candidate_registry.csv",
    "stage2_inner_metrics.csv",
    "stage2_selection.csv",
    "outer_metrics.csv",
    "outer_predictions.csv",
    "outer_per_class_metrics.csv",
    "outer_selective_metrics.csv",
    "outer_temperatures.csv",
    "rejection_selection_metrics.csv",
    "rejection_decisions_at_80.csv",
    "outer_ood_metrics.csv",
    "outer_run_registry.csv",
    "attribution_metrics.csv",
    "attribution_peaks.csv",
    "siamese_control_metrics.csv",
    "siamese_control_domain_metrics.csv",
    "siamese_control_predictions.csv",
    "siamese_control_diagnostics.csv",
    "siamese_control_selective_metrics.csv",
    "siamese_control_failures.csv",
    "siamese_control_provenance.json",
    "domain_metrics.csv",
    "domain_predictions.csv",
    "domain_run_registry.csv",
    "negative_control_metrics.csv",
    "selection_training_histories.csv",
    "final_training_histories.csv",
    "outer_uncertainty_summary.csv",
    "locked_model_comparison.csv",
    "preprocessing_sensitivity_summary.csv",
    "terminal_decision.json",
    "successor_confusion_matrices.json",
    "successor_failure_cases.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path(
            "Workspace/sers_supervised_contrastive/contrastive_v1"
        ),
    )
    parser.add_argument(
        "--rebuild",
        type=Path,
        default=Path(
            "Workspace/sers_supervised_contrastive/contrastive_v1_rebuild"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for relative in ARTIFACTS:
        canonical = args.canonical / relative
        rebuild = args.rebuild / relative
        canonical_hash = (
            baseline.sha256_file(canonical) if canonical.exists() else None
        )
        rebuild_hash = (
            baseline.sha256_file(rebuild) if rebuild.exists() else None
        )
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
    classical.write_json(
        args.canonical / "clean_rebuild_comparison.json", report
    )
    print(json.dumps(report, indent=2))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
