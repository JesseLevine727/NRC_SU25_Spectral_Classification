"""Comprehensive P05 reporting: public aggregate tables and figures.

This stage authenticates the completed comprehensive-comparison stage, rebuilds
the seven public aggregate metric tables, the five source-diagnostic tables and
the two reliability tables purely from evidence already authenticated by the
comparison gate, persists them as CSV beneath the private run root and renders
the paired and diagnostic figures.  It builds no model, runs no optimizer, fits
no temperature, loads no feature array and performs no additional direct legacy
prediction-shard reads beyond those already pinned inside the comparison
authority: every artefact is a deterministic function of authenticated evidence.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_sers.evaluation import p05_comprehensive_comparison as prior_stage
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_recovery_source as recovery_source
from atlas_sers.governance.canonical import sha256_value

__all__ = ["P05ComprehensiveReportingError", "run_reporting"]

SCHEMA = "nato-sers-p05-comprehensive-reporting-v1"
PROTOCOL = development.PROTOCOL_VERSION
COMMAND = "run_comprehensive_reporting"
STAGE_NAME = "reporting"
RECEIPT_NAME = "reporting_receipt.json"
MANIFEST_NAME = evaluation.MANIFEST_NAME
SUMMARY_NAME = "summary.json"
PROVENANCE_BEFORE_NAME = "provenance_before.json"
PROVENANCE_AFTER_NAME = "provenance_after.json"
BINDINGS_NAME = "reporting_bindings.json"
COMPARISON_BINDING_NAME = "comparison_binding.json"
COMPREHENSIVE_DIR = evaluation.COMPREHENSIVE_DIR
RUNS_DIR = evaluation.RUNS_DIR
MAXIMUM_STORAGE_BYTES = evaluation.MAXIMUM_STORAGE_BYTES
MINIMUM_BUDGET_HEADROOM_BYTES = evaluation.MINIMUM_BUDGET_HEADROOM_BYTES
PRIOR_FIELD = "scientific_seconds_cumulative_bound"
COMPARISON_RECEIPT_NAME = prior_stage.RECEIPT_NAME
COMPARISON_MANIFEST_NAME = prior_stage.MANIFEST_NAME
COMPARISON_STAGE_NAME = prior_stage.STAGE_NAME
PUBLIC_ROOT_NAME = "public"
TABLES_DIR_NAME = "tables"
FIGURES_DIR_NAME = "figures"
PAIRED_UNIT_NAME = "paired_bundle"
DIAGNOSTIC_UNIT_NAME = "diagnostic_bundle"
RENDER_DIR_NAME = "figures"
COSTS_NAME = "costs.json"
FIGURE_FILES_PER_FIGURE = 4
PUBLIC_TABLE_COUNT = 7
SOURCE_DIAGNOSTIC_TABLE_COUNT = 5
RELIABILITY_TABLE_COUNT = 2
RELIABILITY_TABLE_NAMES = ("reliability_bins", "reliability_summary")
TOTAL_TABLE_COUNT = PUBLIC_TABLE_COUNT + SOURCE_DIAGNOSTIC_TABLE_COUNT + RELIABILITY_TABLE_COUNT
COUNTER_KEYS = (
    "public_tables",
    "public_rows",
    "source_diagnostic_tables",
    "source_diagnostic_rows",
    "reliability_tables",
    "reliability_rows",
    "paired_figures",
    "diagnostic_figures",
    "fits",
    "calibrations",
    "outer_predictions",
    "updates",
)
ZERO_COUNTERS = ("fits", "calibrations", "outer_predictions", "updates")
PRIVATE_COLUMN_NAMES = frozenset(
    {
        "observation_uid",
        "master_sample_id",
        "context_id",
        "slot_id",
        "refit_id",
        "source_path",
        "class_labels",
        "truth",
        "logits",
    }
)
PRIVATE_COLUMN_PREFIXES = ("probability_", "logits_")


class P05ComprehensiveReportingError(core.P05CoreError):
    """Stable, path-free comprehensive-reporting failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05ComprehensiveReportingError(code)


def _finite_seconds(value: Any, code: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), code)
    number = float(value)
    _require(math.isfinite(number), code)
    return number


def _integer(value: Any, code: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code)
    return int(value)


def _import_runtime() -> dict[str, Any]:
    core._configure_environment()
    import torch

    from atlas_sers.evaluation import p05_comparison_authority as authority
    from atlas_sers.evaluation import p05_pilot as pilot
    from atlas_sers.evaluation import p05_public_metrics as public_metrics
    from atlas_sers.evaluation import p05_reliability as reliability_metrics
    from atlas_sers.evaluation import p05_reporting_inputs as reporting_inputs
    from atlas_sers.evaluation import p05_source_diagnostics as source_diagnostics
    from atlas_sers.visualization import p05_benchmark_figures as benchmark_figures
    from atlas_sers.visualization import p05_diagnostic_figures as diagnostic_figures

    return {
        "torch": torch,
        "authority": authority,
        "pilot": pilot,
        "public_metrics": public_metrics,
        "source_diagnostics": source_diagnostics,
        "reliability_metrics": reliability_metrics,
        "reporting_inputs": reporting_inputs,
        "benchmark_figures": benchmark_figures,
        "diagnostic_figures": diagnostic_figures,
    }


def _frame_from(source: Mapping[str, Any], name: str, group: str) -> pd.DataFrame:
    frame = source.get(name)
    _require(isinstance(frame, pd.DataFrame), f"{group}_table_{name}_malformed")
    return frame


def _collect_frames(
    public: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    reliability: Mapping[str, Any],
    public_metrics: Any,
    source_diagnostics: Any,
) -> dict[str, pd.DataFrame]:
    public_names = tuple(public_metrics.P05_PUBLIC_TABLE_NAMES)
    diagnostic_names = tuple(source_diagnostics.P05_SOURCE_DIAGNOSTIC_TABLE_NAMES)
    _require(
        len(public_names) == PUBLIC_TABLE_COUNT and len(set(public_names)) == PUBLIC_TABLE_COUNT,
        "public_table_names_mismatch",
    )
    _require(
        len(diagnostic_names) == SOURCE_DIAGNOSTIC_TABLE_COUNT
        and len(set(diagnostic_names)) == SOURCE_DIAGNOSTIC_TABLE_COUNT,
        "source_diagnostic_table_names_mismatch",
    )
    _require(
        len(RELIABILITY_TABLE_NAMES) == RELIABILITY_TABLE_COUNT,
        "reliability_table_names_mismatch",
    )

    for mapping, names, code in (
        (public, public_names, "public"),
        (diagnostics, diagnostic_names, "source_diagnostic"),
        (reliability, RELIABILITY_TABLE_NAMES, "reliability"),
    ):
        _require(isinstance(mapping, Mapping), f"{code}_tables_malformed")
        _require(set(mapping) == set(names), f"{code}_table_keys_mismatch")

    frames: dict[str, pd.DataFrame] = {}
    for name in public_names:
        frames[name] = _frame_from(public, name, "public")
    for name in diagnostic_names:
        _require(name not in frames, "public_table_name_collision")
        frames[name] = _frame_from(diagnostics, name, "source_diagnostic")
    for name in RELIABILITY_TABLE_NAMES:
        _require(name not in frames, "public_table_name_collision")
        frames[name] = _frame_from(reliability, name, "reliability")
    _require(len(frames) == TOTAL_TABLE_COUNT, "public_table_count_mismatch")
    return frames


def _reject_private_columns(frames: Mapping[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        for column in frame.columns:
            text = str(column)
            _require(text not in PRIVATE_COLUMN_NAMES, f"private_column_rejected_{name}")
            _require(
                not text.startswith(PRIVATE_COLUMN_PREFIXES),
                f"private_column_rejected_{name}",
            )


def _check_public_costs(
    sources: Mapping[str, Any], reporting_inputs: Any
) -> dict[str, int | float]:
    costs = sources.get("public_costs")
    _require(isinstance(costs, Mapping), "public_costs_malformed")
    allowed = set(reporting_inputs.PUBLIC_COST_KEYS)
    required = set(reporting_inputs.REQUIRED_PUBLIC_COST_KEYS)
    _require(allowed, "public_cost_keys_empty")
    _require(required <= allowed, "public_cost_keys_malformed")
    validated: dict[str, int | float] = {}
    for key, value in costs.items():
        _require(isinstance(key, str) and key != "", "public_cost_key_malformed")
        _require(key in allowed, "public_cost_key_not_allowed")
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            "public_cost_value_malformed",
        )
        number = float(value)
        _require(math.isfinite(number), "public_cost_value_malformed")
        _require(number >= 0.0, "public_cost_value_negative")
        validated[key] = value
    _require(required <= set(validated), "public_cost_key_missing")
    from atlas_sers.evaluation import p05_reporting_inputs as actual_reporting_inputs

    recovery_keys = frozenset(actual_reporting_inputs.RECOVERY_PUBLIC_COST_KEYS)
    if recovery_keys & set(validated):
        actual_reporting_inputs._check_public_costs(validated)
    return validated


def _frame_counters(
    frames: Mapping[str, pd.DataFrame], public_metrics: Any, source_diagnostics: Any
) -> dict[str, int]:
    public_names = tuple(public_metrics.P05_PUBLIC_TABLE_NAMES)
    diagnostic_names = tuple(source_diagnostics.P05_SOURCE_DIAGNOSTIC_TABLE_NAMES)
    return {
        "public_tables": len(public_names),
        "public_rows": sum(len(frames[name]) for name in public_names),
        "source_diagnostic_tables": len(diagnostic_names),
        "source_diagnostic_rows": sum(len(frames[name]) for name in diagnostic_names),
        "reliability_tables": len(RELIABILITY_TABLE_NAMES),
        "reliability_rows": sum(len(frames[name]) for name in RELIABILITY_TABLE_NAMES),
        "paired_figures": 0,
        "diagnostic_figures": 0,
        "fits": 0,
        "calibrations": 0,
        "outer_predictions": 0,
        "updates": 0,
    }


def _write_public_csv(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(index=False, float_format="%.17g")
    core._atomic_write(path, payload.encode("utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    core._atomic_write(path, core._canon().canonical_json_bytes(dict(payload)))


def _verify_costs_json(path: Path, expected: Mapping[str, int | float]) -> None:
    core._reject_symlink_chain(path)
    _require(path.is_file(), "costs_json_missing")
    _require(not path.is_symlink(), "costs_json_symlink")
    stored = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(stored, Mapping), "costs_json_malformed")
    _require(
        core._canon().canonical_json_bytes(stored)
        == core._canon().canonical_json_bytes(dict(expected)),
        "costs_json_changed",
    )


def _comparable(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        series = out[column]
        if pd.api.types.is_bool_dtype(series):
            mapped = series.astype("object").map({True: "True", False: "False"})
            out[column] = mapped.where(series.notna(), "")
        elif pd.api.types.is_numeric_dtype(series):
            out[column] = pd.to_numeric(series, errors="coerce")
        else:
            out[column] = series.astype("object").where(series.notna(), "")
    return out


def _roundtrip_verify(path: Path, original: pd.DataFrame) -> None:
    core._reject_symlink_chain(path)
    numeric_columns = [
        column
        for column in original.columns
        if pd.api.types.is_numeric_dtype(original[column])
        and not pd.api.types.is_bool_dtype(original[column])
    ]
    numeric_set = set(numeric_columns)
    string_columns = [column for column in original.columns if column not in numeric_set]
    read = pd.read_csv(
        path,
        float_precision="round_trip",
        keep_default_na=False,
        dtype={column: str for column in string_columns},
    )
    _require(list(read.columns) == list(original.columns), "roundtrip_columns_changed")
    _require(len(read) == len(original), "roundtrip_row_count_changed")
    for column in numeric_columns:
        try:
            read[column] = pd.to_numeric(read[column].replace("", pd.NA), errors="raise")
        except (TypeError, ValueError) as error:
            raise P05ComprehensiveReportingError("roundtrip_numeric_malformed") from error
    pd.testing.assert_frame_equal(
        _comparable(original),
        _comparable(read),
        check_dtype=False,
        atol=1e-14,
        rtol=0.0,
    )


def _within_root(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def _is_single_filename(value: Any) -> bool:
    if not isinstance(value, str) or value == "":
        return False
    if value in {".", ".."}:
        return False
    if value != Path(value).name:
        return False
    if Path(value).is_absolute():
        return False
    return "/" not in value and "\\" not in value and os.sep not in value


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _actual_inventory(root: Path) -> set[str]:
    names: set[str] = set()
    for child in root.iterdir():
        _require(not child.is_symlink(), "figure_symlink_rejected")
        _require(child.is_file(), "figure_inventory_malformed")
        names.add(child.name)
    return names


def _verify_figure_files(
    output_root: Path,
    manifest: Mapping[str, Any],
    code: str,
    manifest_name: str,
) -> tuple[int, dict[str, str]]:
    _require(_is_single_filename(manifest_name), f"{code}_manifest_name_malformed")
    _require(isinstance(manifest, Mapping), f"{code}_figure_manifest_malformed")
    root = Path(os.path.abspath(os.fspath(output_root)))
    core._reject_symlink_chain(root)
    manifest_path = root / manifest_name
    _require(_within_root(root, manifest_path), f"{code}_manifest_escapes_root")
    _require(not manifest_path.is_symlink(), f"{code}_manifest_symlink")
    _require(manifest_path.is_file(), f"{code}_manifest_missing")
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = json.loads(core._canon().canonical_json_bytes(dict(manifest)))
    _require(stored == expected, f"{code}_manifest_changed")

    files = manifest.get("files")
    _require(isinstance(files, (list, tuple)) and len(files) > 0, f"{code}_files_missing")
    listed: dict[str, str] = {}
    for entry in files:
        _require(isinstance(entry, Mapping), f"{code}_file_entry_malformed")
        _require(set(entry) == {"path", "sha256"}, f"{code}_file_entry_keys_malformed")
        relative = entry["path"]
        digest = entry["sha256"]
        _require(_is_single_filename(relative), f"{code}_file_path_malformed")
        _require(_is_sha256_hex(digest), f"{code}_file_hash_malformed")
        _require(relative not in listed, f"{code}_file_duplicate")
        listed[relative] = digest

    _require(
        _actual_inventory(root) == set(listed) | {manifest_name},
        f"{code}_file_inventory_mismatch",
    )
    for relative, digest in listed.items():
        candidate = root / relative
        _require(_within_root(root, candidate), f"{code}_file_escapes_root")
        _require(not candidate.is_symlink(), f"{code}_file_symlink")
        _require(candidate.is_file(), f"{code}_file_missing")
        _require(core._canon().sha256_file(candidate) == digest, f"{code}_file_hash_mismatch")
    return len(listed), listed


def _figure_records(manifest: Mapping[str, Any], code: str) -> list[Any]:
    figures = manifest.get("figures")
    _require(isinstance(figures, (list, tuple)) and len(figures) > 0, f"{code}_figures_missing")
    return list(figures)


def _figure_identifier(record: Mapping[str, Any], code: str) -> str:
    value = record.get("figure_id")
    _require(isinstance(value, str) and bool(value), f"{code}_figure_id_missing")
    return value


def _verify_figure_records(
    manifest: Mapping[str, Any], listed: Mapping[str, str], code: str
) -> int:
    figures = _figure_records(manifest, code)
    semantic_path = manifest.get("semantic_path")
    data_sha = manifest.get("data_sha256")
    _require(semantic_path == "semantic_data.csv", f"{code}_semantic_path_malformed")
    _require(
        _is_sha256_hex(data_sha) and listed.get(semantic_path) == data_sha,
        f"{code}_semantic_sha_mismatch",
    )
    identifiers: set[str] = set()
    expected_files = {semantic_path}
    for record in figures:
        _require(isinstance(record, Mapping), f"{code}_figure_record_malformed")
        identifier = _figure_identifier(record, code)
        _require(identifier not in identifiers, f"{code}_figure_id_duplicate")
        identifiers.add(identifier)
        _require(record.get("semantic_sha256") == data_sha, f"{code}_semantic_sha_mismatch")
        if code == "paired":
            reference = record.get("reference_model_id")
            aggregation = record.get("aggregation_id")
            _require(isinstance(reference, str) and bool(reference), "paired_reference_missing")
            _require(aggregation in ("M01", "M06"), "paired_aggregation_invalid")
            safe = "".join(character if character.isalnum() else "_" for character in reference)
            stem = f"P05B_{safe}_{aggregation}"
        else:
            stem = record.get("stem")
        _require(_is_single_filename(stem), f"{code}_figure_stem_malformed")
        for suffix in ("tex", "html", "pdf", "png"):
            filename = f"{stem}.{suffix}"
            digest = record.get(f"{suffix}_sha256")
            _require(
                _is_sha256_hex(digest) and listed.get(filename) == digest,
                f"{code}_figure_file_hash_mismatch",
            )
            _require(filename not in expected_files, f"{code}_figure_file_duplicate")
            expected_files.add(filename)
    _require(set(listed) == expected_files, f"{code}_figure_files_mismatch")
    return len(figures)


def _verify_public_inventory(
    public_root: Path,
    frames: Mapping[str, pd.DataFrame],
    figures: Mapping[str, tuple[Path, str, Mapping[str, str]]],
) -> None:
    root = Path(os.path.abspath(os.fspath(public_root)))
    core._reject_symlink_chain(root)
    _require(root.is_dir() and not root.is_symlink(), "public_root_malformed")
    expected_files = {f"{TABLES_DIR_NAME}/{name}.csv" for name in frames}
    expected_files.add(f"{TABLES_DIR_NAME}/{COSTS_NAME}")
    expected_dirs = {TABLES_DIR_NAME}
    for code, (render_root, manifest_name, listed) in figures.items():
        render = Path(os.path.abspath(os.fspath(render_root)))
        _require(_within_root(root, render), f"{code}_render_escapes_public")
        relative = render.relative_to(root)
        expected_dirs.add(relative.as_posix())
        for parent in relative.parents:
            if parent.as_posix() != ".":
                expected_dirs.add(parent.as_posix())
        expected_files.add((relative / manifest_name).as_posix())
        for name in listed:
            expected_files.add((relative / name).as_posix())
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for candidate in root.rglob("*"):
        _require(not candidate.is_symlink(), "public_symlink_rejected")
        if candidate.is_dir():
            actual_dirs.add(candidate.relative_to(root).as_posix())
        elif candidate.is_file():
            actual_files.add(candidate.relative_to(root).as_posix())
        else:
            _require(False, "public_entry_not_regular")
    _require(actual_files == expected_files, "public_file_inventory_mismatch")
    _require(actual_dirs == expected_dirs, "public_directory_inventory_mismatch")


def _bound_payload(
    identity: Mapping[str, Any],
    counters: Mapping[str, Any],
    prior: float,
    elapsed: float,
) -> dict[str, Any]:
    _require(
        math.isfinite(prior) and math.isfinite(elapsed) and elapsed >= 0.0,
        "reporting_cumulative_time_invalid",
    )
    _require(
        development.PRELAUNCH_AUDIT_RESERVE_SECONDS <= prior,
        "reporting_cumulative_time_invalid",
    )
    _require(
        prior + elapsed <= development.MAXIMUM_TOTAL_SECONDS,
        "reporting_cumulative_time_invalid",
    )
    return {
        **identity,
        "status": "complete",
        "reporting_complete": True,
        "counters": {**counters, "elapsed_seconds": elapsed},
        "scientific_seconds_this_stage": elapsed,
        "prior_scientific_seconds_cumulative_bound": prior,
        "scientific_seconds_cumulative_bound": prior + elapsed,
        "prelaunch_audit_reserve_seconds": development.PRELAUNCH_AUDIT_RESERVE_SECONDS,
        "maximum_total_seconds": development.MAXIMUM_TOTAL_SECONDS,
    }


def _record_failure(
    *,
    stage: Path,
    identity: Mapping[str, Any],
    counters: Mapping[str, Any],
    prior: float,
    wall_start: float,
    error: BaseException,
) -> None:
    elapsed = time.perf_counter() - wall_start
    payload = {
        **identity,
        "status": "fail",
        "reporting_complete": False,
        "reason_code": getattr(error, "reason_code", type(error).__name__),
        "counters": {**counters, "elapsed_seconds": elapsed},
        "scientific_seconds_this_stage": elapsed,
        "prior_scientific_seconds_cumulative_bound": prior,
        "scientific_seconds_cumulative_bound": prior + elapsed,
    }
    try:
        core._atomic_write(stage / SUMMARY_NAME, core._canon().canonical_json_bytes(payload))
        core._write_manifest(stage)
    except BaseException:
        pass


def run_reporting(
    *,
    project_root: Any,
    artifact_root: Any,
    contract_path: Any,
    permit_path: Any,
) -> dict[str, Any]:
    """Rebuild and persist the public P05 reporting aggregates and figures."""

    wall_start = time.perf_counter()
    core._configure_environment()
    bundle = inputs.prepare(
        project_root, artifact_root, contract_path, permit_path, require_unstarted=False
    )
    _require(isinstance(bundle, Mapping), "bundle_malformed")

    permit_id = str(bundle["permit_sha256"])
    root = Path(bundle["artifact_root"])
    run_root = root / COMPREHENSIVE_DIR / RUNS_DIR / permit_id
    comparison_receipt_path = run_root / COMPARISON_RECEIPT_NAME
    comparison_receipt = freeze._read_mapping(comparison_receipt_path, "comparison_receipt")
    prior = _finite_seconds(comparison_receipt.get(PRIOR_FIELD), "comparison_prior_invalid")
    _require(development.PRELAUNCH_AUDIT_RESERVE_SECONDS <= prior, "comparison_prior_out_of_range")
    _require(prior <= development.MAXIMUM_TOTAL_SECONDS, "comparison_prior_out_of_range")
    deadline = wall_start + development.MAXIMUM_TOTAL_SECONDS - prior
    freeze._check_deadline(deadline)

    runtime = _import_runtime()
    runtime["torch"].set_num_threads(1)
    authority = runtime["authority"]
    pilot = runtime["pilot"]
    public_metrics = runtime["public_metrics"]
    source_diagnostics = runtime["source_diagnostics"]
    reliability_metrics = runtime["reliability_metrics"]
    reporting_inputs = runtime["reporting_inputs"]
    benchmark_figures = runtime["benchmark_figures"]
    diagnostic_figures = runtime["diagnostic_figures"]

    auth = authority.authenticate_comparison(bundle, deadline=deadline)
    _require(isinstance(auth, Mapping), "authority_malformed")
    _require(
        _finite_seconds(auth.get("prior_seconds"), "authority_prior_malformed") == prior,
        "authority_prior_mismatch",
    )
    _require(
        dict(auth["comparison_receipt"]) == dict(comparison_receipt),
        "comparison_receipt_changed",
    )
    plan = auth.get("plan")
    _require(isinstance(plan, Mapping), "authority_plan_malformed")
    plan_id = plan.get("plan_id")
    _require(plan_id is not None and str(plan_id) != "", "authority_plan_id_malformed")
    _require(
        isinstance(auth.get("comparison_tables"), Mapping), "authority_comparison_tables_malformed"
    )
    _require(
        isinstance(auth.get("aggregation_tables"), Mapping),
        "authority_aggregation_tables_malformed",
    )
    source_steps = _integer(auth.get("source_optimizer_steps"), "authority_source_steps_malformed")
    refit_steps = _integer(auth.get("refit_optimizer_steps"), "authority_refit_steps_malformed")
    try:
        source_accounting = recovery_source.from_authenticated(auth)
    except recovery_source.RecoverySourceError as error:
        raise P05ComprehensiveReportingError("source_accounting_malformed") from error
    _require(isinstance(source_accounting, Mapping), "source_accounting_malformed")
    recovery_source.validate_accounting(source_accounting, source_optimizer_steps=source_steps)
    accounting_mode = source_accounting.get("mode")
    _require(accounting_mode in ("clean", "recovered"), "source_accounting_mode_malformed")
    comparison_receipt_sha256 = core._canon().sha256_file(comparison_receipt_path)
    comparison_stage = run_root / COMPARISON_STAGE_NAME
    comparison_manifest_sha256 = core._canon().sha256_file(
        comparison_stage / COMPARISON_MANIFEST_NAME
    )
    _require(
        comparison_manifest_sha256 == comparison_receipt["stage_manifest_sha256"],
        "comparison_manifest_changed",
    )

    provenance_before = core._capture_provenance(
        bundle["repository_root"], bundle["project_root"], bundle["artifact_root"]
    )

    identity = {
        "schema_version": SCHEMA,
        "protocol_version": PROTOCOL,
        "command": COMMAND,
        "stage": STAGE_NAME,
        "permit_sha256": permit_id,
        "core_contract_sha256": bundle["contract_sha256"],
        "core_plan_id": bundle["core_plan_id"],
        "ledger_id": bundle["ledger"]["ledger_id"],
        "selection_plan_id": str(plan_id),
        "comparison_receipt_sha256": comparison_receipt_sha256,
        "comparison_manifest_sha256": comparison_manifest_sha256,
        "source_optimizer_steps": source_steps,
        "refit_optimizer_steps": refit_steps,
        "reporting_complete": False,
    }
    if accounting_mode == "recovered":
        identity["source_execution_accounting"] = dict(source_accounting)
    counters: dict[str, Any] = {key: 0 for key in COUNTER_KEYS}

    stage = run_root / STAGE_NAME
    consumed = False
    try:
        freeze._reject_preexisting(stage, "reporting_stage_exists")
        freeze._reject_preexisting(run_root / RECEIPT_NAME, "reporting_receipt_exists")
        freeze._check_deadline(deadline)
        core._mkdir_exclusive(stage, "reporting_stage_exists")
        consumed = True
        budget = freeze.StorageBudget(root, run_root, ceiling=MAXIMUM_STORAGE_BYTES)
        freeze._budgeted_write(stage / PROVENANCE_BEFORE_NAME, provenance_before, budget)

        sources = reporting_inputs.load_reporting_sources(
            bundle, authenticated=auth, deadline=deadline
        )
        _require(isinstance(sources, Mapping), "reporting_sources_malformed")
        bindings = sources.get("bindings")
        _require(isinstance(bindings, Mapping), "reporting_bindings_malformed")
        selector_records = sources.get("selector_records")
        _require(selector_records is not None, "selector_records_missing")
        public_costs = _check_public_costs(sources, reporting_inputs)
        recovery_keys = frozenset(reporting_inputs.RECOVERY_PUBLIC_COST_KEYS)
        present_recovery = frozenset(public_costs) & recovery_keys
        if accounting_mode == "recovered":
            _require(present_recovery == recovery_keys, "recovery_cost_keys_incomplete")
        else:
            _require(not present_recovery, "recovery_cost_keys_unexpected")
        binding_digest = sha256_value(dict(bindings))
        identity = {**identity, "reporting_binding_digest": binding_digest}
        freeze._budgeted_write(stage / BINDINGS_NAME, dict(bindings), budget)
        freeze._budgeted_write(
            stage / COMPARISON_BINDING_NAME,
            {
                "comparison_receipt_sha256": comparison_receipt_sha256,
                "comparison_manifest_sha256": comparison_manifest_sha256,
            },
            budget,
        )

        public = public_metrics.build_public_metrics(
            aggregation_tables=auth["aggregation_tables"],
            comparison_tables=auth["comparison_tables"],
        )
        _require(isinstance(public, Mapping), "public_metrics_malformed")
        strategy_contexts = public.get("strategy_contexts")
        _require(strategy_contexts is not None, "strategy_contexts_missing")
        paired_domains = public.get("paired_domains")
        _require(paired_domains is not None, "paired_domains_missing")
        diagnostics = source_diagnostics.build_source_diagnostics(
            plan=plan,
            selector_records=selector_records,
            slots=bundle["slots"],
            strategy_contexts=strategy_contexts,
        )
        _require(isinstance(diagnostics, Mapping), "source_diagnostics_malformed")
        reliability = reliability_metrics.build_reliability(
            ensemble_predictions=auth["aggregation_tables"]["ensemble_predictions"],
            strategy_contexts=strategy_contexts,
        )
        _require(isinstance(reliability, Mapping), "reliability_malformed")

        frames = _collect_frames(
            public, diagnostics, reliability, public_metrics, source_diagnostics
        )
        _reject_private_columns(frames)
        counters.update(_frame_counters(frames, public_metrics, source_diagnostics))

        public_root = stage / PUBLIC_ROOT_NAME
        tables_dir = public_root / TABLES_DIR_NAME
        figures_root = public_root / FIGURES_DIR_NAME
        paired_unit = figures_root / PAIRED_UNIT_NAME
        diagnostic_unit = figures_root / DIAGNOSTIC_UNIT_NAME

        budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)
        freeze._check_deadline(deadline)
        budget.activate_unit(tables_dir)
        for name, frame in frames.items():
            _write_public_csv(tables_dir / f"{name}.csv", frame)
        _write_json(tables_dir / COSTS_NAME, public_costs)
        for name, frame in frames.items():
            _roundtrip_verify(tables_dir / f"{name}.csv", frame)
        _verify_costs_json(tables_dir / COSTS_NAME, public_costs)
        budget.close_unit()
        budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)

        budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)
        freeze._check_deadline(deadline)
        budget.activate_unit(paired_unit)
        paired_output = paired_unit / RENDER_DIR_NAME
        paired_manifest = benchmark_figures.generate_pair_figures(
            paired_domains, output_root=paired_output, deadline=deadline
        )
        _require(isinstance(paired_manifest, Mapping), "paired_figure_manifest_malformed")
        budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)
        freeze._check_deadline(deadline)
        budget.close_unit()

        budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)
        freeze._check_deadline(deadline)
        budget.activate_unit(diagnostic_unit)
        diagnostic_output = diagnostic_unit / RENDER_DIR_NAME
        diagnostic_manifest = diagnostic_figures.generate_diagnostic_figures(
            public_metrics=public,
            source_diagnostics=diagnostics,
            reliability=reliability,
            output_root=diagnostic_output,
            deadline=deadline,
        )
        _require(isinstance(diagnostic_manifest, Mapping), "diagnostic_figure_manifest_malformed")
        budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)
        freeze._check_deadline(deadline)
        budget.close_unit()

        paired_files, paired_listed = _verify_figure_files(
            paired_output, paired_manifest, "paired", benchmark_figures.MANIFEST_NAME
        )
        diagnostic_files, diagnostic_listed = _verify_figure_files(
            diagnostic_output,
            diagnostic_manifest,
            "diagnostic",
            diagnostic_figures.MANIFEST_NAME,
        )
        counters["paired_figures"] = _verify_figure_records(
            paired_manifest, paired_listed, "paired"
        )
        counters["diagnostic_figures"] = _verify_figure_records(
            diagnostic_manifest, diagnostic_listed, "diagnostic"
        )
        _require(
            paired_files == FIGURE_FILES_PER_FIGURE * counters["paired_figures"] + 1,
            "paired_figure_count_mismatch",
        )
        _require(
            diagnostic_files == FIGURE_FILES_PER_FIGURE * counters["diagnostic_figures"] + 1,
            "diagnostic_figure_count_mismatch",
        )
        for name in ZERO_COUNTERS:
            _require(
                _integer(counters[name], f"counter_{name}_invalid") == 0, f"counter_{name}_nonzero"
            )

        freeze._check_deadline(deadline)
        provenance_after = pilot._post_run_reauth(
            bundle["artifact_root"],
            bundle["contract"],
            bundle["support"],
            provenance_before,
            bundle["repository_root"],
            bundle["project_root"],
        )
        freeze._budgeted_write(stage / PROVENANCE_AFTER_NAME, provenance_after, budget)
        inputs.prepare(
            bundle["project_root"],
            bundle["artifact_root"],
            contract_path,
            permit_path,
            require_unstarted=False,
        )
        freeze._check_deadline(deadline)

        reporting_inputs.verify_reporting_sources(bundle, bindings=bindings, deadline=deadline)
        _require(
            dict(freeze._read_mapping(stage / BINDINGS_NAME, "reporting_bindings"))
            == dict(bindings),
            "persisted_reporting_bindings_changed",
        )
        expected_comparison_binding = {
            "comparison_receipt_sha256": comparison_receipt_sha256,
            "comparison_manifest_sha256": comparison_manifest_sha256,
        }
        _require(
            dict(freeze._read_mapping(stage / COMPARISON_BINDING_NAME, "comparison_binding"))
            == expected_comparison_binding,
            "persisted_comparison_binding_changed",
        )
        _require(
            core._canon().sha256_file(comparison_receipt_path) == comparison_receipt_sha256,
            "comparison_receipt_changed",
        )
        _require(
            dict(freeze._read_mapping(comparison_receipt_path, "comparison_receipt"))
            == dict(comparison_receipt),
            "comparison_receipt_changed",
        )
        _require(
            core._canon().sha256_file(comparison_stage / COMPARISON_MANIFEST_NAME)
            == comparison_manifest_sha256,
            "comparison_manifest_changed",
        )
        pilot._verify_manifest(comparison_stage)
        for name, frame in frames.items():
            _roundtrip_verify(tables_dir / f"{name}.csv", frame)
        _verify_costs_json(tables_dir / COSTS_NAME, public_costs)
        _verify_figure_files(
            paired_output, paired_manifest, "paired", benchmark_figures.MANIFEST_NAME
        )
        _verify_figure_files(
            diagnostic_output,
            diagnostic_manifest,
            "diagnostic",
            diagnostic_figures.MANIFEST_NAME,
        )
        _verify_public_inventory(
            public_root,
            frames,
            {
                PAIRED_UNIT_NAME: (
                    paired_output,
                    benchmark_figures.MANIFEST_NAME,
                    paired_listed,
                ),
                DIAGNOSTIC_UNIT_NAME: (
                    diagnostic_output,
                    diagnostic_figures.MANIFEST_NAME,
                    diagnostic_listed,
                ),
            },
        )

        elapsed = time.perf_counter() - wall_start
        freeze._budgeted_write(
            stage / SUMMARY_NAME, _bound_payload(identity, counters, prior, elapsed), budget
        )
        budget.check(headroom_bytes=MINIMUM_BUDGET_HEADROOM_BYTES)
        core._write_manifest(stage)
        budget.account_new_file(stage / MANIFEST_NAME)
        pilot._verify_manifest(stage)
        budget.reconcile()
        freeze._check_deadline(deadline)
        final_elapsed = time.perf_counter() - wall_start
        receipt = {
            **_bound_payload(identity, counters, prior, final_elapsed),
            "stage_manifest_sha256": core._canon().sha256_file(stage / MANIFEST_NAME),
        }
        freeze._reject_preexisting(run_root / RECEIPT_NAME, "reporting_receipt_exists")
        freeze._budgeted_write(run_root / RECEIPT_NAME, receipt, budget)
        budget.check()
        freeze._check_deadline(deadline)
        return receipt
    except BaseException as error:
        if consumed:
            _record_failure(
                stage=stage,
                identity=identity,
                counters=counters,
                prior=prior,
                wall_start=wall_start,
                error=error,
            )
        raise
