"""Reconcile P13 shards and materialize locked aggregate evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p13_results import build_p13_aggregate_tables
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file, sha256_value
from atlas_sers.governance.p03_store import P03ShardStore
from atlas_sers.governance.p13_execution import execution_context


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        na_rep="",
    ).encode()


def _prediction_store(context_root: Path) -> P03ShardStore:
    return P03ShardStore(run_root=context_root / "prediction")


def _read_shards(
    *, context_root: Path, expected_shards: int, protected_state_sha256: str
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    store = _prediction_store(context_root)
    expected = {index: protected_state_sha256 for index in range(expected_shards)}
    validation = store.validation_table(expected)
    if not all(bool(row["valid"]) for row in validation):
        missing = [int(row["shard_id"]) for row in validation if not row["valid"]]
        raise RuntimeError(f"P13 prediction shards are incomplete: {missing[:20]}")
    frames: dict[str, list[pd.DataFrame]] = {
        "fit_status": [],
        "calibration_status": [],
        "fold_endpoint_status": [],
        "predictions": [],
    }
    for index in range(expected_shards):
        root = store.shards / f"shard-{index:06d}"
        frames["fit_status"].append(pd.read_csv(root / "fit_status.csv", low_memory=False))
        frames["calibration_status"].append(
            pd.read_csv(root / "calibration_status.csv", low_memory=False)
        )
        frames["fold_endpoint_status"].append(
            pd.read_csv(root / "fold_endpoint_status.csv", low_memory=False)
        )
        predictions = pd.read_parquet(root / "predictions.parquet")
        if not predictions.empty:
            frames["predictions"].append(predictions)
    combined = {
        name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for name, parts in frames.items()
    }
    return combined, validation


def aggregate_p13(
    *, artifact_root: Path, project_root: Path
) -> tuple[dict[str, Any], Path, str]:
    context = execution_context(artifact_root=artifact_root, project_root=project_root)
    plan_shards = pd.read_csv(context.plan_run / "shard_manifest.csv")
    combined, shard_validation = _read_shards(
        context_root=context.execution_root,
        expected_shards=len(plan_shards),
        protected_state_sha256=context.protected_state_sha256,
    )
    fit_manifest = pd.read_csv(context.plan_run / "fit_manifest.csv", low_memory=False)
    procedure_registry = pd.read_csv(
        context.plan_run / "procedure_registry.csv", low_memory=False
    )
    fit_status = combined["fit_status"]
    fold_endpoints = combined["fold_endpoint_status"]
    if len(fit_status) != len(fit_manifest) or not fit_status.fit_id.astype(str).is_unique:
        raise RuntimeError("P13 fit terminal accounting differs from its no-fit manifest.")
    if set(fit_status.fit_id.astype(str)) != set(fit_manifest.fit_id.astype(str)):
        raise RuntimeError("P13 fit IDs differ from their no-fit manifest.")
    if len(fold_endpoints) != len(procedure_registry):
        raise RuntimeError("P13 fold endpoint accounting is incomplete.")
    domain_support = pd.read_csv(
        project_root / "plan/registries/p13_domain_support_registry.csv"
    )
    crossover_support = pd.read_csv(
        project_root / "plan/registries/p13_crossover_support_registry.csv"
    )
    manifest = pd.read_csv(context.p01_run / "primary_manifest.csv", low_memory=False)
    with np.load(
        context.p01_run / "representations/R_MIN_400_1800.npz",
        allow_pickle=False,
    ) as payload:
        min_intensity = payload["intensity"].copy()
    tables = build_p13_aggregate_tables(
        predictions=combined["predictions"],
        fit_status=fit_status,
        fold_endpoints=fold_endpoints,
        domain_support=domain_support,
        crossover_support=crossover_support,
        manifest=manifest,
        min_intensity=min_intensity,
    )
    shard_states = {
        f"shard-{index:06d}": sha256_file(
            context.execution_root
            / "prediction"
            / "shards"
            / f"shard-{index:06d}"
            / "_STATE.json"
        )
        for index in range(len(plan_shards))
    }
    aggregation_state = {
        "schema_version": "nato-sers-p13-aggregation-state-v1",
        "execution_protected_state_sha256": context.protected_state_sha256,
        "results_code_sha256": sha256_file(
            project_root / "src/atlas_sers/evaluation/p13_results.py"
        ),
        "aggregation_boundary_sha256": sha256_file(
            project_root / "src/atlas_sers/governance/p13_results.py"
        ),
        "shard_state_bundle_sha256": sha256_value(shard_states),
    }
    protected_hash = sha256_value(aggregation_state)
    store = P03ShardStore(run_root=context.execution_root / "aggregation")
    lease = store.begin(shard_id=0, protected_state_sha256=protected_hash)
    if lease.action == "verified_skip":
        report = json.loads(
            (lease.final_dir / "P13_EXECUTION_VALIDATION_REPORT.json").read_text()
        )
        return report, lease.final_dir, lease.action
    if lease.temporary_dir is None:
        raise RuntimeError("P13 aggregation lease has no temporary directory.")
    private_tables = {
        "master_view_predictions.parquet": tables.master_view_predictions,
        "fit_status.csv": fit_status,
        "calibration_status.csv": combined["calibration_status"],
        "fold_endpoint_status.csv": fold_endpoints,
    }
    public_tables = {
        "domain_metrics.csv": tables.domain_metrics,
        "interval_table.csv": tables.interval_table,
        "domain_claims.csv": tables.domain_claims,
        "class_cell_claims.csv": tables.class_cell_claims,
        "substrate_claims.csv": tables.substrate_claims,
        "preprocessing_sensitivity.csv": tables.preprocessing_sensitivity,
        "procedure_comparison.csv": tables.procedure_comparison,
        "crossover_effects.csv": tables.crossover_effects,
        "field_log_results.csv": tables.field_log_results,
        "failure_table.csv": tables.failure_table,
    }
    for name, frame in private_tables.items():
        path = lease.temporary_dir / name
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False, compression="zstd")
        else:
            path.write_bytes(_csv_bytes(frame))
    for name, frame in public_tables.items():
        (lease.temporary_dir / name).write_bytes(_csv_bytes(frame))
    primary_claims = tables.domain_claims[
        tables.domain_claims.support_tier.astype(str).eq("confirmatory")
    ]
    view_identity = [
        "domain_id",
        "procedure_id",
        "prediction_role",
        "master_sample_id",
        "instrument",
        "true_label",
    ]
    endpoint_status = tables.domain_metrics.pivot(
        index=["domain_id", "procedure_id"],
        columns="policy_id",
        values="endpoint_status",
    )
    common_policy_endpoints = set(
        endpoint_status[
            endpoint_status.eq("complete").all(axis=1)
        ].index.to_list()
    )
    complete_endpoint_keys = set(
        map(
            tuple,
            tables.domain_metrics.loc[
                tables.domain_metrics.endpoint_status.astype(str).eq("complete"),
                ["domain_id", "policy_id", "procedure_id"],
            ]
            .astype(str)
            .itertuples(index=False, name=None),
        )
    )
    complete_views = tables.master_view_predictions[
        tables.master_view_predictions.apply(
            lambda row: (
                str(row.domain_id),
                str(row.policy_id),
                str(row.procedure_id),
            )
            in complete_endpoint_keys,
            axis=1,
        )
    ]
    common_views = tables.master_view_predictions[
        tables.master_view_predictions.apply(
            lambda row: (str(row.domain_id), str(row.procedure_id))
            in common_policy_endpoints,
            axis=1,
        )
    ]
    policy_view_sets = {
        policy: set(
            map(
                tuple,
                group[view_identity].astype(str).itertuples(index=False, name=None),
            )
        )
        for policy, group in common_views.groupby("policy_id")
    }
    checks = {
        "all_240_shards_valid": len(shard_validation) == 240
        and all(bool(row["valid"]) for row in shard_validation),
        "all_42360_fit_rows_terminal": len(fit_status) == 42_360,
        "fit_ids_unique_and_exact": fit_status.fit_id.astype(str).is_unique
        and set(fit_status.fit_id.astype(str)) == set(fit_manifest.fit_id.astype(str)),
        "all_6720_fold_endpoints_retained": len(fold_endpoints) == 6_720,
        "complete_out_of_fold_views_have_five_repeat_predictions": (
            complete_views.outer_repeat_predictions_min.eq(5).all()
        ),
        "common_successful_preprocessing_endpoints_have_identical_views": len(
            policy_view_sets
        )
        == 3
        and len(set(map(frozenset, policy_view_sets.values()))) == 1,
        "all_336_eligible_domain_procedure_policy_rows_retained": len(
            tables.domain_metrics
        )
        == 16 * 3 * 7,
        "all_34_domains_have_one_primary_state": len(tables.domain_claims) == 34
        and tables.domain_claims.domain_id.astype(str).is_unique,
        "all_102_primary_class_cells_retained": len(tables.class_cell_claims)
        == 34 * 3
        and not tables.class_cell_claims.duplicated(
            ["domain_id", "target_analyte"]
        ).any(),
        "all_five_completion_states_valid": set(
            tables.domain_claims.completion_state.astype(str)
        )
        <= {
            "supports_portability",
            "inferior_portability",
            "inconclusive",
            "unsupported_by_design",
            "unavailable_terminal_failure",
        },
        "thirteen_confirmatory_primary_claims_retained": len(primary_claims) == 13,
        "all_34_crossover_blocks_retained_per_procedure": (
            tables.crossover_effects.groupby("procedure_id").crossover_block_id.nunique()
            == 34
        ).all(),
        "bootstrap_resamples_are_10000": tables.interval_table.bootstrap_resamples.eq(
            10_000
        ).all(),
        "private_predictions_excluded_from_public_tables": all(
            "observation_uid" not in frame and "master_sample_id" not in frame
            for frame in public_tables.values()
        ),
    }
    report = {
        "schema_version": "nato-sers-p13-execution-validation-v1",
        "protocol_version": "nato-sers-p13-v1-locked",
        "phase": "P13-C01-C04",
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": context.run_id,
        "execution_protected_state_sha256": context.protected_state_sha256,
        "aggregation_protected_state_sha256": protected_hash,
        "checks": checks,
        "counts": {
            "fit_rows": len(fit_status),
            "complete_fits": int(fit_status.status.astype(str).eq("complete").sum()),
            "failed_or_excluded_fits": int(
                fit_status.status.astype(str).ne("complete").sum()
            ),
            "fold_endpoints": len(fold_endpoints),
            "observation_prediction_rows": len(combined["predictions"]),
            "master_view_prediction_rows": len(tables.master_view_predictions),
            "domain_metric_rows": len(tables.domain_metrics),
            "crossover_rows": len(tables.crossover_effects),
            "field_log_rows": len(tables.field_log_results),
        },
    }
    (lease.temporary_dir / "aggregation_state.json").write_bytes(
        canonical_json_bytes(aggregation_state, pretty=True)
    )
    (lease.temporary_dir / "P13_EXECUTION_VALIDATION_REPORT.json").write_bytes(
        canonical_json_bytes(report, pretty=True)
    )
    if report["status"] != "pass":
        store.abort(lease, reason="p13_aggregation_validation_failed")
        raise RuntimeError("P13 aggregate evidence failed validation.")
    final_dir = store.commit(lease)
    latest = {
        "schema_version": "nato-sers-p13-latest-v1",
        "run_id": context.run_id,
        "status": report["status"],
        "execution_protected_state_sha256": context.protected_state_sha256,
        "aggregation_protected_state_sha256": protected_hash,
        "report_sha256": sha256_file(
            final_dir / "P13_EXECUTION_VALIDATION_REPORT.json"
        ),
    }
    latest_path = artifact_root / "p13" / "LATEST.json"
    temporary = artifact_root / "p13" / ".LATEST.json.tmp"
    temporary.write_bytes(canonical_json_bytes(latest, pretty=True))
    os.replace(temporary, latest_path)
    return report, final_dir, lease.action
