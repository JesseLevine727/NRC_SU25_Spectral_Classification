#!/usr/bin/env python3
"""Validate the closed SERS VAE adequacy v1 bundle."""

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


def checkpoint_hash_valid(path: Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in payload or "state_sha256" not in payload:
        return False
    digest = hashlib.sha256()
    for key in sorted(payload["state_dict"]):
        tensor = payload["state_dict"][key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest() == payload["state_sha256"]


def selection_cache_valid(path: Path) -> bool:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not {"config", "history", "states"} <= set(payload):
        return False
    if not payload["states"]:
        return False
    for state in payload["states"].values():
        if not state:
            return False
        for tensor in state.values():
            if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
                return False
    return True


def validate(output_dir: Path, require_clean_rebuild: bool) -> dict[str, Any]:
    audit = Audit()
    required = [
        "README.md",
        "DECISION_REGISTRY.md",
        "RESULTS_REPORT.md",
        "predeclared_protocol.json",
        "protocol_amendment_1.json",
        "protocol_amendment_2.json",
        "input_hashes.json",
        "environment.json",
        "existing_run_audit.json",
        "first_100_reproduction.json",
        "checkpoint_metrics.csv",
        "training_histories.csv",
        "stage_summaries.csv",
        "selected_configuration.json",
        "minimal_sensitivity_metrics.csv",
        "outer_fold_metrics.csv",
        "outer_fold_predictions.csv",
        "outer_fold_reconstruction_metrics.csv",
        "outer_fold_corruption_metrics.csv",
        "domain_transfer_metrics.csv",
        "poster_metrics.csv",
        "per_spectrum_predictions.csv",
        "per_class_metrics.csv",
        "per_instrument_failures.csv",
        "failure_cases.csv",
        "failure_attribution.json",
        "uncertainty_summary.csv",
        "comparator_summary.csv",
        "outer_fold_same_master_geometry.csv",
        "same_master_geometry_summary.csv",
        "instrument_reconstruction_summary.csv",
        "parameter_and_compute_accounting.csv",
        "compute_accounting.json",
        "final_decisions.json",
        "artifact_hashes.json",
        "reproduction_commands.sh",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    audit.check("required_artifacts", not missing, missing)
    report_text = (
        (output_dir / "RESULTS_REPORT.md").read_text()
        if (output_dir / "RESULTS_REPORT.md").is_file()
        else ""
    )
    audit.check(
        "results_report_complete",
        all(
            phrase in report_text
            for phrase in (
                "Executive answer",
                "What was actually trained",
                "Metric glossary",
                "Locked grouped-outer confirmation",
                "Failure attribution",
                "Reproducibility status",
            )
        ),
    )

    protocol = json.loads((output_dir / "predeclared_protocol.json").read_text())
    amendment1 = json.loads(
        (output_dir / "protocol_amendment_1.json").read_text()
    )
    amendment2 = json.loads(
        (output_dir / "protocol_amendment_2.json").read_text()
    )
    selected = json.loads(
        (output_dir / "selected_configuration.json").read_text()
    )
    decisions = json.loads(
        (output_dir / "final_decisions.json").read_text()
    )
    attribution = json.loads(
        (output_dir / "failure_attribution.json").read_text()
    )
    compute = json.loads((output_dir / "compute_accounting.json").read_text())
    audit.check(
        "protocol_version",
        protocol.get("protocol_version") == "sers-vae-adequacy-v1",
    )
    audit.check(
        "amendments_registered",
        amendment1.get("amendment_version")
        == "sers-vae-adequacy-v1-amendment-1"
        and amendment2.get("amendment_version")
        == "sers-vae-adequacy-v1-amendment-2",
    )
    audit.check(
        "selection_closed_without_locked_data",
        selected.get("selection_closed") is True
        and not any(
            bool(selected.get(key, True))
            for key in (
                "outer_used",
                "field_quality_stress_used",
                "domain_used",
                "poster_used",
            )
        ),
    )
    audit.check(
        "selected_configuration_frozen",
        selected.get("architecture") == "base_maxpool"
        and int(selected.get("latent_dimension", -1)) == 64
        and selected.get("reconstruction_loss") == "spectral_composite"
        and float(selected.get("beta_target", -1)) == 0.25
        and selected.get("optimizer_policy") == "constant_lr"
        and int(selected.get("maximum_epoch", -1)) == 500,
    )
    audit.check(
        "final_decision_matches_selection",
        decisions.get("selected_configuration") == selected.get("identifier")
        and decisions.get("converged") is True
        and decisions.get("inner_gate_count") == 7
        and decisions.get("inner_gate_total") == 9
        and set(decisions.get("failed_inner_gates", []))
        == {"gate_instrument_probe", "gate_same_master_distance"},
    )
    audit.check(
        "attribution_categories_complete",
        set(attribution.get("categories", {}))
        == {
            "convergence",
            "architecture_downsampling",
            "reconstruction_objective",
            "latent_capacity_or_kl_pressure",
            "data_coverage_domain_shift",
            "unresolved_interaction",
        },
    )
    audit.check(
        "compute_accounting_complete",
        compute.get("distinct_inner_training_runs") == 260
        and compute.get("distinct_confirmatory_training_runs") == 252
        and compute.get("total_distinct_training_runs") == 512
        and compute.get("total_training_epochs") == 256000
        and compute.get("confirmatory_checkpoint_count") == 252
        and compute.get("selection_checkpoint_count") == 260,
        {
            key: compute.get(key)
            for key in (
                "distinct_inner_training_runs",
                "distinct_confirmatory_training_runs",
                "total_distinct_training_runs",
                "total_training_epochs",
            )
        },
    )

    reproduction = json.loads(
        (output_dir / "first_100_reproduction.json").read_text()
    )
    audit.check(
        "first_100_within_registered_tolerance",
        reproduction.get(
            "all_shared_histories_within_predeclared_tolerance"
        )
        is True
        and float(reproduction.get("maximum_absolute_difference", np.inf))
        <= float(reproduction.get("predeclared_history_tolerance", -1)),
        reproduction.get("maximum_absolute_difference"),
    )

    checkpoint = pd.read_csv(output_dir / "checkpoint_metrics.csv")
    histories = pd.read_csv(
        output_dir / "training_histories.csv", low_memory=False
    )
    stages = pd.read_csv(output_dir / "stage_summaries.csv")
    architecture = pd.read_csv(
        output_dir / "stage_2_architecture_summary.csv"
    )
    loss = pd.read_csv(output_dir / "stage_2_loss_summary.csv")
    latent = pd.read_csv(output_dir / "stage_2_latent_summary.csv")
    beta = pd.read_csv(output_dir / "stage_2_beta_summary.csv")
    minimal = pd.read_csv(output_dir / "minimal_sensitivity_metrics.csv")
    outer = pd.read_csv(output_dir / "outer_fold_metrics.csv")
    poster = pd.read_csv(output_dir / "poster_metrics.csv")
    domain = pd.read_csv(output_dir / "domain_transfer_metrics.csv")
    corruption = pd.read_csv(output_dir / "corruption_metrics.csv")
    predictions = pd.read_csv(
        output_dir / "per_spectrum_predictions.csv", low_memory=False
    )
    geometry = pd.read_csv(
        output_dir / "outer_fold_same_master_geometry.csv"
    )
    audit.check("checkpoint_metric_row_count", len(checkpoint) == 580, len(checkpoint))
    audit.check(
        "selection_history_row_count", len(histories) == 130000, len(histories)
    )
    audit.check("architecture_candidates", len(architecture) == 3, len(architecture))
    audit.check("loss_candidates", len(loss) == 2, len(loss))
    audit.check("latent_candidates", len(latent) == 3, len(latent))
    audit.check("beta_candidates", len(beta) == 3, len(beta))
    audit.check("minimal_sensitivity_folds", len(minimal) == 40, len(minimal))
    audit.check("outer_metric_row_count", len(outer) == 90, len(outer))
    audit.check("poster_metric_row_count", len(poster) == 24, len(poster))
    audit.check("domain_metric_row_count", len(domain) == 168, len(domain))
    audit.check(
        "corruption_row_count",
        len(corruption) == 1890 + 504 + 3528,
        len(corruption),
    )
    audit.check(
        "outer_cohorts_complete",
        set(outer["scenario"].str.extract(r"__test_(.+)$")[0])
        == {"strict_core", "quality_pass", "field_quality_stress"},
    )
    audit.check(
        "outer_representations_complete",
        set(outer["representation"])
        == {"arpls_minmax", "minimal_minmax"},
    )
    audit.check(
        "domain_protocols_complete",
        set(domain["domain_protocol"]) == {"domain_only", "domain_and_sample"}
        and set(domain["domain_type"]) == {"instrument", "sensor_family"},
    )
    audit.check(
        "geometry_comparators_complete",
        set(geometry["display_model"])
        == {"AE", "DAE", "Siamese", "VAE-100 β=1", "VAE-500 β=0.25"}
        and {"strict_core", "quality_pass", "field_quality_stress"}
        <= set(geometry["subset"]),
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

    for name, values in {
        "outer_balanced_accuracy_range": outer[
            "balanced_accuracy_supported"
        ],
        "poster_balanced_accuracy_range": poster[
            "balanced_accuracy_supported"
        ],
        "domain_balanced_accuracy_range": domain[
            "balanced_accuracy_supported"
        ],
    }.items():
        finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
        audit.check(
            name,
            bool(len(finite)) and bool(((finite >= 0) & (finite <= 1)).all()),
        )
    audit.check(
        "outer_variational_metrics_finite",
        np.isfinite(
            outer[
                [
                    "vae_kl_unnormalized_per_observation",
                    "vae_active_units_var_mu_gt_0_01",
                    "reconstruction_median_row_correlation",
                ]
            ].to_numpy(dtype=float)
        ).all(),
    )
    audit.check(
        "selected_stage1_converged",
        bool(
            stages[
                stages["optimizer_policy"].eq("constant_lr")
                & stages["checkpoint_epoch"].eq(500)
            ]["converged"].iloc[0]
        ),
    )

    checkpoint_paths = sorted((output_dir / "checkpoints").rglob("*.pt"))
    selection_paths = sorted((output_dir / "selection_cache").rglob("*.pt"))
    embedding_paths = sorted((output_dir / "embeddings").glob("*.npz"))
    reconstruction_paths = sorted(
        (output_dir / "reconstructions").glob("*.npz")
    )
    audit.check("confirmatory_checkpoint_count", len(checkpoint_paths) == 252, len(checkpoint_paths))
    audit.check("selection_checkpoint_count", len(selection_paths) == 260, len(selection_paths))
    audit.check("embedding_count", len(embedding_paths) == 282, len(embedding_paths))
    audit.check(
        "reconstruction_array_count",
        len(reconstruction_paths) == 282,
        len(reconstruction_paths),
    )
    bad_checkpoints = [
        str(path.relative_to(output_dir))
        for path in checkpoint_paths
        if not checkpoint_hash_valid(path)
    ]
    bad_selection_cache = [
        str(path.relative_to(output_dir))
        for path in selection_paths
        if not selection_cache_valid(path)
    ]
    audit.check(
        "checkpoint_tensor_hashes",
        not bad_checkpoints,
        bad_checkpoints[:10],
    )
    audit.check(
        "selection_cache_tensor_integrity",
        not bad_selection_cache,
        bad_selection_cache[:10],
    )

    catalog = json.loads((output_dir / "artifact_hashes.json").read_text())
    hash_failures = []
    for relative, expected in catalog["files"].items():
        path = output_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            hash_failures.append(relative)
    audit.check("artifact_hash_catalog", not hash_failures, hash_failures)

    figures = [
        "convergence_diagnostics",
        "bounded_ablation",
        "outer_comparison",
        "preservation_comparison",
        "corruption_robustness",
        "domain_transfer",
        "poster_transfer",
        "instrument_spectra",
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
        "validator": "validate_sers_vae_adequacy.py",
        "protocol_version": "sers-vae-adequacy-v1",
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
        / "sers_vae_adequacy"
        / "adequacy_v1",
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
    if report["status"] != "pass":
        failures = [
            item for item in report["checks"] if item["status"] == "fail"
        ]
        print(json.dumps(failures, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
