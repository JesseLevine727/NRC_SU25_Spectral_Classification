#!/usr/bin/env python3
"""Validate the closed SERS standard-VAE v1 result bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: Any = None) -> None:
        self.checks.append(
            {
                "name": name,
                "status": "pass" if condition else "fail",
                "detail": detail,
            }
        )


def validate(output_dir: Path, require_clean_rebuild: bool) -> dict[str, Any]:
    audit = Audit()
    required = [
        "README.md",
        "DECISION_REGISTRY.md",
        "predeclared_protocol.json",
        "environment.json",
        "input_hashes.json",
        "split_seed_assignments.csv",
        "selection_fold_metrics.csv",
        "selection_summary.csv",
        "selected_configuration.json",
        "outer_fold_metrics.csv",
        "domain_transfer_metrics.csv",
        "poster_metrics.csv",
        "corruption_metrics.csv",
        "reconstruction_metrics.csv",
        "variational_metrics.csv",
        "per_class_metrics.csv",
        "poster_localized_4np_failures.csv",
        "per_spectrum_predictions.csv",
        "failure_cases.csv",
        "uncertainty_summary.csv",
        "final_decisions.json",
        "artifact_hashes.json",
        "reproduction_commands.sh",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    audit.check("required_artifacts", not missing, missing)

    protocol = json.loads((output_dir / "predeclared_protocol.json").read_text())
    selected = json.loads((output_dir / "selected_configuration.json").read_text())
    decisions = json.loads((output_dir / "final_decisions.json").read_text())
    audit.check(
        "protocol_version",
        protocol.get("protocol_version") == "sers-standard-vae-v1",
    )
    audit.check(
        "selection_closed_without_sealed_data",
        selected.get("selection_closed") is True
        and not any(
            selected.get(key, True)
            for key in (
                "outer_used",
                "field_quality_stress_used",
                "domain_used",
                "poster_used",
            )
        ),
    )
    audit.check("standard_beta_fixed", float(selected["beta"]) == 1.0)
    audit.check(
        "selected_schedule_registered",
        selected["kl_schedule"]
        in protocol["vae_specific_search"]["candidates"],
    )
    audit.check(
        "diagnostic_ineligibility_recorded",
        selected["eligible_to_advance"] is False
        and decisions["eligible_to_advance"] is False,
    )

    selection = pd.read_csv(output_dir / "selection_fold_metrics.csv")
    outer = pd.read_csv(output_dir / "outer_fold_metrics.csv")
    poster = pd.read_csv(output_dir / "poster_metrics.csv")
    domain = pd.read_csv(output_dir / "domain_transfer_metrics.csv")
    corruption = pd.read_csv(output_dir / "corruption_metrics.csv")
    predictions = pd.read_csv(
        output_dir / "per_spectrum_predictions.csv", low_memory=False
    )
    audit.check("selection_row_count", len(selection) == 140, len(selection))
    audit.check("outer_metric_row_count", len(outer) == 90, len(outer))
    audit.check("poster_metric_row_count", len(poster) == 24, len(poster))
    audit.check("domain_metric_row_count", len(domain) == 168, len(domain))
    audit.check(
        "corruption_row_count",
        len(corruption) == 1890 + 504 + 3528,
        len(corruption),
    )
    audit.check(
        "only_vae_model_family",
        set(outer["model_family"]) == {"vae"}
        and set(poster["model_family"]) == {"vae"}
        and set(domain["model_family"]) == {"vae"},
    )
    audit.check(
        "outer_cohorts_complete",
        set(outer["scenario"].str.extract(r"test_(.+)$")[0])
        == {"strict_core", "quality_pass", "field_quality_stress"},
    )
    audit.check(
        "outer_views_complete",
        set(outer["representation"])
        == {"arpls_minmax", "minimal_minmax"},
    )
    audit.check(
        "domain_protocols_complete",
        set(domain["domain_protocol"]) == {"domain_only", "domain_and_sample"},
    )
    audit.check(
        "domain_types_complete",
        set(domain["domain_type"]) == {"instrument", "sensor_family"},
    )
    supported = predictions["test_class_supported"].astype(bool)
    audit.check(
        "support_flags_complete",
        predictions["test_class_supported"].notna().all(),
    )
    audit.check(
        "unsupported_predictions_retained",
        int((~supported).sum()) > 0,
        int((~supported).sum()),
    )

    numeric_checks = {
        "outer_balanced_accuracy_range": outer["balanced_accuracy_supported"],
        "poster_balanced_accuracy_range": poster[
            "balanced_accuracy_supported"
        ],
    }
    for name, values in numeric_checks.items():
        finite = values.dropna().to_numpy(dtype=float)
        audit.check(
            name,
            bool(len(finite)) and bool(((finite >= 0) & (finite <= 1)).all()),
        )
    audit.check(
        "finite_variational_health",
        np.isfinite(
            outer[
                [
                    "vae_kl_unnormalized_per_observation",
                    "vae_active_units_var_mu_gt_0_01",
                ]
            ].to_numpy(dtype=float)
        ).all(),
    )
    audit.check(
        "noncollapsed_outer",
        float(outer["vae_kl_unnormalized_per_observation"].min()) > 0.01
        and float(outer["vae_active_units_var_mu_gt_0_01"].mean()) >= 4,
    )

    checkpoint_paths = sorted((output_dir / "checkpoints").rglob("*.pt"))
    embedding_paths = sorted((output_dir / "embeddings").glob("*.npz"))
    reconstruction_paths = sorted(
        (output_dir / "reconstructions").glob("*.npz")
    )
    audit.check("checkpoint_count", len(checkpoint_paths) == 252, len(checkpoint_paths))
    audit.check("embedding_count", len(embedding_paths) == 282, len(embedding_paths))
    audit.check(
        "reconstruction_array_count",
        len(reconstruction_paths) == 282,
        len(reconstruction_paths),
    )
    checkpoint_ok = True
    for path in checkpoint_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" not in payload or "state_sha256" not in payload:
            checkpoint_ok = False
            break
        digest = hashlib.sha256()
        for key in sorted(payload["state_dict"]):
            tensor = payload["state_dict"][key].detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        if digest.hexdigest() != payload["state_sha256"]:
            checkpoint_ok = False
            break
    audit.check("checkpoint_tensor_hashes", checkpoint_ok)

    catalog = json.loads((output_dir / "artifact_hashes.json").read_text())
    hash_failures = []
    for relative, expected in catalog["files"].items():
        path = output_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            hash_failures.append(relative)
    audit.check("artifact_hash_catalog", not hash_failures, hash_failures)

    figures = [
        "outer_comparison",
        "corruption_comparison",
        "poster_comparison",
    ]
    missing_figures = [
        f"{name}.{suffix}"
        for name in figures
        for suffix in ("png", "pdf")
        if not (output_dir / "figures" / f"{name}.{suffix}").is_file()
    ]
    audit.check("figures_complete", not missing_figures, missing_figures)
    if require_clean_rebuild:
        comparison_path = output_dir / "clean_rebuild_comparison.json"
        comparison = (
            json.loads(comparison_path.read_text())
            if comparison_path.is_file()
            else {}
        )
        audit.check(
            "clean_rebuild_exact",
            comparison.get("all_required_comparisons_exact") is True,
            comparison.get("status", "missing"),
        )

    failed = sum(item["status"] == "fail" for item in audit.checks)
    return {
        "validator": "validate_sers_standard_vae.py",
        "protocol_version": "sers-standard-vae-v1",
        "require_clean_rebuild": require_clean_rebuild,
        "status": "pass" if failed == 0 else "fail",
        "summary": {
            "checks": len(audit.checks),
            "passed": len(audit.checks) - failed,
            "failed": failed,
        },
        "checks": audit.checks,
    }


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_standard_vae"
        / "standard_vae_v1",
    )
    parser.add_argument("--require-clean-rebuild", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    output_dir = args.output_dir.resolve()
    report = validate(output_dir, args.require_clean_rebuild)
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
