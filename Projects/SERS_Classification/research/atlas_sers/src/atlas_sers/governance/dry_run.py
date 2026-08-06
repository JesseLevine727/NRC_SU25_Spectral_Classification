"""Deterministic P00 work enumeration that cannot authorize or fit a model."""

from __future__ import annotations

import csv
import io
from typing import Any

from atlas_sers.governance.canonical import canonical_json_bytes, sha256_bytes, sha256_value
from atlas_sers.governance.registries import GovernanceBundle
from atlas_sers.governance.runs import RunIdentity, deterministic_run_id

STAGE_PHASES = {
    "data_and_split_validation": "P01|P02",
    "classical_nested": "P03",
    "compact_D0_development": "P04",
    "D1_to_D5_source_development": "P05",
    "definitive_two_deep_models_initial": "P06",
    "full_repeat_confirmation_additional": "P06",
    "adaptation_robustness_open_set": "P07|P08|P09",
}


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return stream.getvalue().encode()


def _provisional_state(phase: str, task: str, technical_seeds: str) -> dict[str, str]:
    unresolved = ["registered_configuration_expansion"]
    if phase == "P01":
        unresolved.append("representation_expansion")
        repeat = fold = "not_applicable"
    else:
        unresolved.extend(["outer_repeat", "outer_fold"])
        repeat = fold = "P02_UNRESOLVED"
    held_domain = "P02_UNRESOLVED" if any(key in task for key in ("T3", "T4")) else "not_applicable"
    if held_domain != "not_applicable":
        unresolved.append("held_domain")
    seed = "not_applicable" if technical_seeds == "none" else "P02_UNRESOLVED"
    if seed != "not_applicable":
        unresolved.append("seed")
    return {
        "outer_repeat": repeat,
        "outer_fold": fold,
        "held_domain": held_domain,
        "seed": seed,
        "unresolved_fields": "|".join(unresolved),
    }


def build_dry_run_outputs(
    bundle: GovernanceBundle,
    *,
    protocol_version: str,
    code_sha256: str,
    config_sha256: str,
    input_sha256: str,
    resource_snapshot: dict[str, Any],
) -> dict[str, bytes]:
    experiments = sorted(
        bundle.rows("experiment_registry.csv"), key=lambda row: row["experiment_id"]
    )
    models = {row["model_id"]: row for row in bundle.rows("model_registry.csv")}
    hyperparameters = bundle.contracts["hyperparameter_registry.json"]
    phases = {row["phase_id"]: row for row in bundle.rows("phase_registry.csv")}
    run_rows: list[dict[str, Any]] = []
    for experiment in experiments:
        provisional = _provisional_state(
            experiment["phase"], experiment["task_id"], experiment["technical_seeds"]
        )
        hyperparameter_hash = sha256_value(
            {"model": models[experiment["model_id"]], "registry": hyperparameters}
        )
        identity = RunIdentity(
            protocol_version=protocol_version,
            experiment_id=experiment["experiment_id"],
            task_id=experiment["task_id"],
            information_regime=experiment["information_regime"],
            outer_repeat=provisional["outer_repeat"],
            outer_fold=provisional["outer_fold"],
            held_domain=provisional["held_domain"],
            population_id=experiment["population"],
            representation_id=experiment["representation"],
            model_id=experiment["model_id"],
            hyperparameter_sha256=hyperparameter_hash,
            seed=provisional["seed"],
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            input_sha256=input_sha256,
        )
        run_rows.append(
            {
                "provisional_run_id": deterministic_run_id(identity),
                "finalized": "false",
                "resolution_status": f"awaiting_{experiment['phase']}",
                "unresolved_fields": provisional["unresolved_fields"],
                "experiment_id": experiment["experiment_id"],
                "scope": experiment["scope"],
                "phase": experiment["phase"],
                "task_id": experiment["task_id"],
                "information_regime": experiment["information_regime"],
                "population_id": experiment["population"],
                "representation_id": experiment["representation"],
                "model_id": experiment["model_id"],
                "outer_repeat": provisional["outer_repeat"],
                "outer_fold": provisional["outer_fold"],
                "held_domain": provisional["held_domain"],
                "seed": provisional["seed"],
                "hyperparameter_sha256": hyperparameter_hash,
                "code_sha256": code_sha256,
                "config_sha256": config_sha256,
                "input_sha256": input_sha256,
                "artifact_ids": experiment["artifact_ids"],
                "fit_authorized": "false",
            }
        )

    run_fields = [
        "provisional_run_id",
        "finalized",
        "resolution_status",
        "unresolved_fields",
        "experiment_id",
        "scope",
        "phase",
        "task_id",
        "information_regime",
        "population_id",
        "representation_id",
        "model_id",
        "outer_repeat",
        "outer_fold",
        "held_domain",
        "seed",
        "hyperparameter_sha256",
        "code_sha256",
        "config_sha256",
        "input_sha256",
        "artifact_ids",
        "fit_authorized",
    ]

    compute_budget = bundle.contracts["compute_budget.json"]
    fit_rows: list[dict[str, Any]] = []
    gpu_rows: list[dict[str, Any]] = []
    cpu_rows: list[dict[str, Any]] = []
    for stage in compute_budget["stages"]:
        base = {
            "stage": stage["stage"],
            "phases": STAGE_PHASES[stage["stage"]],
            "model_id": "multiple_or_unresolved",
            "task_id": "multiple_or_unresolved",
            "fit_estimate_low": stage["fit_estimate_low"],
            "fit_estimate_high": (
                "unbounded_pending_registry"
                if stage["fit_estimate_high"] is None
                else stage["fit_estimate_high"]
            ),
            "exact_count_status": "awaiting_P02_run_registry",
            "fit_authorized": "false",
        }
        fit_rows.append(base)
        estimate = {
            "stage": stage["stage"],
            "fit_estimate_low": stage["fit_estimate_low"],
            "fit_estimate_high": base["fit_estimate_high"],
            "hours_low": "not_estimated",
            "hours_high": "not_estimated",
            "status": "runtime_benchmark_and_P02_registry_required",
        }
        if any(phase in base["phases"] for phase in ("P04", "P05", "P06", "P07")):
            gpu_rows.append(estimate)
        else:
            cpu_rows.append(estimate)

    shard_rows = [
        {
            "shard_id": f"SHARD-{sha256_value(row['provisional_run_id'])[:16]}",
            "provisional_run_id": row["provisional_run_id"],
            "experiment_id": row["experiment_id"],
            "phase": row["phase"],
            "depends_on": phases[row["phase"]]["depends_on"],
            "status": "provisional_awaiting_P02",
            "fit_authorized": "false",
        }
        for row in run_rows
    ]
    disk = {
        "schema_version": "p00-disk-estimate-v1",
        "artifact_filesystem_total_bytes": resource_snapshot.get("artifact_filesystem_total_bytes"),
        "artifact_filesystem_free_bytes_at_capture": resource_snapshot.get(
            "artifact_filesystem_free_bytes_at_capture"
        ),
        "estimated_output_bytes": None,
        "estimate_status": "runtime_benchmarks_and_P02_run_registry_required",
        "registered_artifact_classes": len(bundle.rows("artifact_registry.csv")),
        "fit_authorized": False,
    }
    outputs = {
        "expected_run_registry.csv": _csv_bytes(run_rows, run_fields),
        "fit_count_by_phase_model_task.csv": _csv_bytes(
            fit_rows,
            [
                "stage",
                "phases",
                "model_id",
                "task_id",
                "fit_estimate_low",
                "fit_estimate_high",
                "exact_count_status",
                "fit_authorized",
            ],
        ),
        "estimated_gpu_hours.csv": _csv_bytes(
            gpu_rows,
            [
                "stage",
                "fit_estimate_low",
                "fit_estimate_high",
                "hours_low",
                "hours_high",
                "status",
            ],
        ),
        "estimated_cpu_hours.csv": _csv_bytes(
            cpu_rows,
            [
                "stage",
                "fit_estimate_low",
                "fit_estimate_high",
                "hours_low",
                "hours_high",
                "status",
            ],
        ),
        "estimated_disk_bytes.json": canonical_json_bytes(disk, pretty=True),
        "shard_manifest.csv": _csv_bytes(
            shard_rows,
            [
                "shard_id",
                "provisional_run_id",
                "experiment_id",
                "phase",
                "depends_on",
                "status",
                "fit_authorized",
            ],
        ),
    }
    return outputs


def dry_run_bundle_sha256(outputs: dict[str, bytes]) -> str:
    return sha256_value(
        {
            name: {"sha256": sha256_bytes(content), "size_bytes": len(content)}
            for name, content in outputs.items()
        }
    )
