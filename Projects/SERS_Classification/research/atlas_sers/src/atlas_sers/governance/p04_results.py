"""Collect immutable P04 context shards and derive reconciled private evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_sers.evaluation.p04_results import (
    endpoint_coverage,
    endpoint_metrics,
    ensemble_seed_predictions,
    learning_curve_summary,
    normalize_p04_predictions,
    summarize_endpoints,
    validate_prediction_schema,
)
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file, sha256_value
from atlas_sers.governance.p03_store import P03ShardStore
from atlas_sers.governance.p04_execution import (
    _tables,
    _validate_development_freeze,
    execution_context,
)


def _read_optional(path: Path, kind: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_parquet(path) if kind == "parquet" else pd.read_csv(path, low_memory=False)


def _concatenate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    available = [frame for frame in frames if not frame.empty]
    return pd.concat(available, ignore_index=True) if available else pd.DataFrame()


def aggregate_p04(*, artifact_root: Path, project_root: Path) -> dict[str, Any]:
    context = execution_context(artifact_root=artifact_root, project_root=project_root)
    _validate_development_freeze(context)
    tables = _tables(context)
    expected_contexts = tables["context_registry"].sort_values("shard_index")
    context_store = P03ShardStore(run_root=context.execution_root / "contexts")
    validation = context_store.validation_table(
        {
            int(row.shard_index): context.protected_state_sha256
            for row in expected_contexts.itertuples(index=False)
        }
    )
    if not all(record["valid"] for record in validation):
        missing = [record["shard_id"] for record in validation if not record["valid"]]
        raise RuntimeError(f"P04 contexts are incomplete or invalid: {missing}")
    status_frames = []
    trace_frames = []
    calibration_frames = []
    history_frames = []
    source_prediction_frames = []
    test_prediction_frames = []
    for row in expected_contexts.itertuples(index=False):
        shard = context.execution_root / "contexts/shards" / f"shard-{int(row.shard_index):06d}"
        status_frames.append(_read_optional(shard / "fit_status.csv", "csv"))
        trace_frames.append(_read_optional(shard / "selection_trace.csv", "csv"))
        calibration_frames.append(_read_optional(shard / "calibration_status.csv", "csv"))
        history_frames.append(_read_optional(shard / "training_histories.parquet", "parquet"))
        source_prediction_frames.append(
            _read_optional(shard / "source_validation_predictions.parquet", "parquet")
        )
        test_prediction_frames.append(_read_optional(shard / "test_predictions.parquet", "parquet"))
    fit_status = _concatenate(status_frames)
    selections = _concatenate(trace_frames)
    calibrations = _concatenate(calibration_frames)
    histories = _concatenate(history_frames)
    source_predictions = _concatenate(source_prediction_frames)
    raw_seed_predictions = _concatenate(test_prediction_frames)
    raw_ensemble = ensemble_seed_predictions(raw_seed_predictions)
    manifest = pd.read_csv(context.p01_run / "primary_manifest.csv", low_memory=False)
    execution_state = json.loads((context.execution_root / "protected_state.json").read_text())
    plan_state = json.loads((context.plan_run / "protected_state.json").read_text())
    identity = {
        "run_id": context.run_id,
        "code_sha256": sha256_value(
            {
                key: execution_state[key]
                for key in (
                    "runtime_sha256",
                    "execution_boundary_sha256",
                    "model_sha256",
                )
            }
        ),
        "config_sha256": execution_state["contract_sha256"],
        "input_sha256": plan_state["input_hashes_sha256"],
        "preprocessing_policy_sha256": sha256_file(
            project_root / "plan/contracts/preprocessing_policy_contract.json"
        ),
    }
    seed_predictions = normalize_p04_predictions(
        raw_seed_predictions,
        contexts=tables["context_registry"],
        manifest=manifest,
        fit_status=fit_status,
        identity=identity,
        ensemble=False,
    )
    ensemble = normalize_p04_predictions(
        raw_ensemble,
        contexts=tables["context_registry"],
        manifest=manifest,
        fit_status=fit_status,
        identity=identity,
        ensemble=True,
    )
    result_schema = json.loads((project_root / "plan/contracts/result_schema.json").read_text())
    seed_schema = validate_prediction_schema(seed_predictions, result_schema)
    ensemble_schema = validate_prediction_schema(ensemble, result_schema)
    spectrum, master = endpoint_metrics(ensemble)
    metrics = pd.concat([spectrum, master], ignore_index=True)
    coverage = endpoint_coverage(tables["expected_endpoint_registry"], fit_status, ensemble)
    performance = summarize_endpoints(metrics, coverage)
    curves, epochs = learning_curve_summary(histories, selections)
    result_code_hash = sha256_value(
        {
            "evaluation": sha256_file(project_root / "src/atlas_sers/evaluation/p04_results.py"),
            "governance": sha256_file(project_root / "src/atlas_sers/governance/p04_results.py"),
            "platform_family_mapping": sha256_file(project_root / "src/atlas_sers/splits/p02.py"),
        }
    )
    aggregation_state = sha256_value(
        {
            "execution_state": context.protected_state_sha256,
            "result_code_sha256": result_code_hash,
            "context_shard_state_sha256": sha256_value(
                [
                    sha256_file(
                        context.execution_root
                        / "contexts/shards"
                        / f"shard-{int(row.shard_index):06d}/_STATE.json"
                    )
                    for row in expected_contexts.itertuples(index=False)
                ]
            ),
        }
    )
    store = P03ShardStore(run_root=context.execution_root / "final_aggregation")
    lease = store.begin(shard_id=0, protected_state_sha256=aggregation_state)
    if lease.action == "verified_skip":
        return json.loads((lease.final_dir / "P04_AGGREGATION_REPORT.json").read_text())
    if lease.temporary_dir is None:
        raise RuntimeError("P04 aggregation lease has no temporary directory.")
    fit_status.to_csv(lease.temporary_dir / "fit_status.csv", index=False)
    selections.to_csv(lease.temporary_dir / "selection_trace.csv", index=False)
    calibrations.to_csv(lease.temporary_dir / "calibration_status.csv", index=False)
    histories.to_parquet(
        lease.temporary_dir / "training_histories.parquet",
        index=False,
        compression="zstd",
    )
    source_predictions.to_parquet(
        lease.temporary_dir / "source_validation_predictions.parquet",
        index=False,
        compression="zstd",
    )
    seed_predictions.to_parquet(
        lease.temporary_dir / "seed_test_predictions.parquet",
        index=False,
        compression="zstd",
    )
    ensemble.to_parquet(
        lease.temporary_dir / "ensemble_test_predictions.parquet",
        index=False,
        compression="zstd",
    )
    metrics.to_csv(lease.temporary_dir / "endpoint_metrics.csv", index=False)
    coverage.to_csv(lease.temporary_dir / "endpoint_coverage.csv", index=False)
    performance.to_csv(lease.temporary_dir / "performance_summary.csv", index=False)
    curves.to_csv(lease.temporary_dir / "learning_curve_summary.csv", index=False)
    epochs.to_csv(lease.temporary_dir / "selected_epoch_summary.csv", index=False)
    checks = {
        "all_320_context_shards_valid": len(validation) == 320
        and all(record["valid"] for record in validation),
        "all_16458_planned_fits_recorded": len(fit_status) == len(tables["fit_manifest"]),
        "fit_id_sets_match_exactly_once": not fit_status.fit_id.duplicated().any()
        and set(fit_status.fit_id) == set(tables["fit_manifest"].fit_id),
        "test_seeds_match_registered_seeds": set(raw_seed_predictions.seed.astype(int))
        == set(context.contract["optimization"]["training_seeds"]),
        "all_fit_statuses_terminal": bool(
            fit_status.status.isin(
                {
                    "complete",
                    "numerical_failure",
                    "resource_failure",
                    "data_failure",
                    "fit_failure",
                    "excluded_by_protocol",
                }
            ).all()
        ),
        "all_320_endpoints_reconciled": len(coverage) == 320,
        "all_260_t3_endpoints_reconciled": int(coverage.experiment_id.eq("EXP-N00-T3").sum())
        == 260,
        "ensemble_only_after_three_seeds": bool(ensemble.seed_count.eq(3).all()),
        "prediction_uid_hashes_match": bool(
            coverage[coverage.status.eq("complete")]
            .expected_test_uid_sha256.eq(
                coverage[coverage.status.eq("complete")].observed_test_uid_sha256
            )
            .all()
        ),
        "one_selected_candidate_per_context": bool(
            selections.groupby("context_id").selected.sum().eq(1).all()
        ),
        "seed_prediction_schema_passes": seed_schema["status"] == "pass",
        "ensemble_prediction_schema_passes": ensemble_schema["status"] == "pass",
    }
    context_files = [
        path for path in (context.execution_root / "contexts/shards").rglob("*") if path.is_file()
    ]
    checkpoints = [path for path in context_files if path.suffix == ".pt"]
    report: dict[str, Any] = {
        "schema_version": "nato-sers-p04-aggregation-report-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": context.run_id,
        "protected_state_sha256": context.protected_state_sha256,
        "aggregation_state_sha256": aggregation_state,
        "result_code_sha256": result_code_hash,
        "checks": checks,
        "counts": {
            "fit_status_rows": len(fit_status),
            "complete_fits": int(fit_status.status.eq("complete").sum()),
            "failed_fits": int((~fit_status.status.eq("complete")).sum()),
            "history_rows": len(histories),
            "source_validation_prediction_rows": len(source_predictions),
            "seed_test_prediction_rows": len(seed_predictions),
            "ensemble_test_prediction_rows": len(ensemble),
            "complete_endpoints": int(coverage.status.eq("complete").sum()),
            "terminal_failure_endpoints": int(coverage.status.eq("terminal_failure").sum()),
        },
        "diagnostic_counts": fit_status.diagnostic.fillna("failure").value_counts().to_dict(),
        "result_schema_validation": {
            "seed_predictions": seed_schema,
            "ensemble_predictions": ensemble_schema,
        },
        "selection_counts": (
            selections[selections.selected].candidate_id.value_counts().sort_index().to_dict()
        ),
        "resource_accounting": {
            "optimizer_epochs": int(fit_status.epochs_completed.fillna(0).sum()),
            "optimizer_fit_seconds": float(fit_status.elapsed_seconds.sum()),
            "optimizer_fit_hours": float(fit_status.elapsed_seconds.sum() / 3600),
            "peak_cuda_bytes": int(fit_status.peak_cuda_bytes.fillna(0).max()),
            "private_context_file_count": len(context_files),
            "private_context_bytes": sum(path.stat().st_size for path in context_files),
            "final_checkpoint_count": len(checkpoints),
            "final_checkpoint_bytes": sum(path.stat().st_size for path in checkpoints),
        },
        "claim_boundary": (
            "private D0 aggregation; classical comparison and public scientific claims require "
            "the separate frozen comparison/release step"
        ),
    }
    (lease.temporary_dir / "P04_AGGREGATION_REPORT.json").write_bytes(
        canonical_json_bytes(report, pretty=True)
    )
    if report["status"] != "pass":
        store.abort(lease, reason="p04_aggregation_validation_failed")
        raise RuntimeError("P04 final aggregation failed reconciliation.")
    store.commit(lease)
    latest = {
        "schema_version": "nato-sers-p04-latest-v1",
        "run_id": context.run_id,
        "status": "pass",
        "protected_state_sha256": context.protected_state_sha256,
        "aggregation_state_sha256": aggregation_state,
    }
    (artifact_root / "p04/LATEST.json").write_bytes(canonical_json_bytes(latest, pretty=True))
    return report
