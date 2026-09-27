"""Read-only publication gate for the sealed P05 comprehensive reporting output.

Authenticates a caller-reviewed reporting-receipt digest against the fixed
comprehensive run root and re-verifies, in place, the sealed public aggregate
tables, cost summary and figure manifests.  It never writes, copies, renders,
trains, scores or re-derives science; publication is a later, separate review.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_comprehensive_reporting as reporting
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_evaluation_authority as evaluation_authority
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_public_metrics as public_metrics
from atlas_sers.evaluation import p05_recovery_source as recovery_source
from atlas_sers.evaluation import p05_reporting_inputs as reporting_inputs
from atlas_sers.evaluation import p05_source_diagnostics as source_diagnostics
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.visualization import p05_benchmark_figures as benchmark_figures
from atlas_sers.visualization import p05_diagnostic_figures as diagnostic_figures

__all__ = [
    "P05PublicationGateError",
    "PublicBundle",
    "load_public_bundle",
    "verify_public_bundle",
]

_ELAPSED_FIELDS = frozenset(
    {"scientific_seconds_this_stage", "scientific_seconds_cumulative_bound"}
)
_FIXED_RESERVE_SECONDS = 3600
_FIXED_MAXIMUM_SECONDS = 172800


class P05PublicationGateError(core.P05CoreError):
    """Stable, path-free publication-gate failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class PublicBundle(NamedTuple):
    public_root: Path
    files: dict[str, str]
    tables: dict[str, pd.DataFrame]
    costs: dict[str, Any]
    receipt_sha256: str
    cumulative_seconds: float
    figure_manifests: dict[str, Any]


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05PublicationGateError(code)


def _finite_seconds(value: Any, code: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), code)
    number = float(value)
    _require(math.isfinite(number), code)
    return number


def _integer(value: Any, code: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code)
    return int(value)


def _hex64(value: Any, code: str) -> str:
    _require(isinstance(value, str) and core._is_hex64(value), code)
    return value


def _stable_sha256(path: Path, code: str) -> str:
    core._reject_symlink_chain(path)
    _require(path.is_file() and not path.is_symlink(), code)
    return core._canon().sha256_file(path)


def _run_root(artifact_root: Any) -> Path:
    return (
        Path(artifact_root)
        / reporting.COMPREHENSIVE_DIR
        / reporting.RUNS_DIR
        / inputs.COMPREHENSIVE_PERMIT_SHA256
    )


def _load_receipt(run_root: Path, expected_sha256: Any) -> tuple[dict[str, Any], str]:
    _hex64(expected_sha256, "expected_receipt_sha_malformed")
    core._reject_symlink_chain(run_root)
    receipt_path = run_root / reporting.RECEIPT_NAME
    receipt_sha256 = _stable_sha256(receipt_path, "reporting_receipt_missing")
    _require(receipt_sha256 == expected_sha256, "reporting_receipt_digest_mismatch")
    receipt = core._read_json(receipt_path, "reporting_receipt")
    _require(isinstance(receipt, Mapping), "reporting_receipt_malformed")
    return dict(receipt), receipt_sha256


def _check_receipt_identity(receipt: Mapping[str, Any]) -> None:
    _require(str(receipt.get("schema_version")) == reporting.SCHEMA, "receipt_schema_mismatch")
    _require(
        str(receipt.get("protocol_version")) == reporting.PROTOCOL, "receipt_protocol_mismatch"
    )
    _require(str(receipt.get("stage")) == reporting.STAGE_NAME, "receipt_stage_mismatch")
    _require(str(receipt.get("command")) == reporting.COMMAND, "receipt_command_mismatch")
    _require(str(receipt.get("status")) == "complete", "receipt_status_incomplete")
    _require(receipt.get("reporting_complete") is True, "receipt_reporting_incomplete")
    for key, expected in (
        ("permit_sha256", inputs.COMPREHENSIVE_PERMIT_SHA256),
        ("core_contract_sha256", inputs.CORE_CONTRACT_SHA256),
        ("core_plan_id", inputs.CORE_PLAN_ID),
        ("ledger_id", inputs.LEDGER_ID),
    ):
        _require(str(receipt.get(key)) == expected, f"receipt_{key}_mismatch")
    _hex64(receipt.get("selection_plan_id"), "receipt_selection_plan_id_malformed")
    for key in (
        "permit_sha256",
        "core_contract_sha256",
        "comparison_receipt_sha256",
        "comparison_manifest_sha256",
        "reporting_binding_digest",
        "stage_manifest_sha256",
    ):
        _hex64(receipt.get(key), f"receipt_{key}_malformed")
    source_steps = _integer(
        receipt.get("source_optimizer_steps"), "receipt_source_optimizer_steps_malformed"
    )
    refit_steps = _integer(
        receipt.get("refit_optimizer_steps"), "receipt_refit_optimizer_steps_malformed"
    )
    _require(
        0
        <= refit_steps
        <= evaluation_authority.MAXIMUM_REFITS * evaluation_authority.MAXIMUM_UPDATES_PER_REFIT,
        "receipt_refit_optimizer_steps_out_of_range",
    )
    if "source_execution_accounting" not in receipt:
        _require(
            0 <= source_steps <= evaluation_authority.SOURCE_MAXIMUM_UPDATES,
            "receipt_source_optimizer_steps_out_of_range",
        )
    try:
        accounting = recovery_source.from_authenticated(receipt)
    except recovery_source.RecoverySourceError as error:
        raise P05PublicationGateError("receipt_source_accounting_malformed") from error
    _require(isinstance(accounting, Mapping), "receipt_source_accounting_malformed")
    recovery_source.validate_accounting(accounting, source_optimizer_steps=source_steps)
    accounting_mode = str(accounting.get("mode"))
    _require(
        accounting_mode in ("clean", "recovered"),
        "receipt_source_accounting_mode_malformed",
    )
    if accounting_mode == "recovered":
        charged_source = _integer(
            accounting.get("source_optimizer_steps_charged_upper_bound"),
            "receipt_source_charged_malformed",
        )
        maximum_source = _integer(
            accounting.get("maximum_source_optimizer_steps"),
            "receipt_source_maximum_malformed",
        )
        maximum_new = _integer(
            accounting.get("maximum_new_optimizer_steps"),
            "receipt_maximum_new_optimizer_steps_malformed",
        )
        _require(charged_source <= maximum_source, "receipt_source_charged_exceeded")
        _require(
            charged_source + refit_steps <= maximum_new,
            "receipt_combined_updates_exceeded",
        )
    else:
        _require(
            0 <= source_steps <= evaluation_authority.SOURCE_MAXIMUM_UPDATES,
            "receipt_source_optimizer_steps_out_of_range",
        )
        _require(
            source_steps + refit_steps <= evaluation_authority.MAXIMUM_COMBINED_UPDATES,
            "receipt_combined_updates_exceeded",
        )


def _check_stage_manifest(run_root: Path, receipt: Mapping[str, Any]) -> tuple[Path, str]:
    stage = run_root / reporting.STAGE_NAME
    core._reject_symlink_chain(stage)
    _require(stage.is_dir() and not stage.is_symlink(), "reporting_stage_missing")
    manifest_path = stage / reporting.MANIFEST_NAME
    manifest_sha256 = _stable_sha256(manifest_path, "reporting_manifest_missing")
    _require(
        manifest_sha256 == str(receipt.get("stage_manifest_sha256")),
        "reporting_manifest_digest_mismatch",
    )
    pilot._verify_manifest(stage)
    _require(
        _stable_sha256(manifest_path, "reporting_manifest_missing") == manifest_sha256,
        "reporting_manifest_changed",
    )
    return stage, manifest_sha256


def _counter_mapping(value: Any, code: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), code)
    _require(
        set(value) == set(reporting.COUNTER_KEYS) | {"elapsed_seconds"},
        "counters_keys_mismatch",
    )
    for name in reporting.COUNTER_KEYS:
        _require(
            isinstance(value.get(name), int) and not isinstance(value.get(name), bool),
            f"counter_{name}_invalid",
        )
    _finite_seconds(value.get("elapsed_seconds"), "counter_elapsed_seconds_malformed")
    return value


def _check_summary(receipt: Mapping[str, Any], summary: Mapping[str, Any]) -> float:
    reserve = development.PRELAUNCH_AUDIT_RESERVE_SECONDS
    limit = development.MAXIMUM_TOTAL_SECONDS
    _require(reserve == _FIXED_RESERVE_SECONDS, "reserve_constant_changed")
    _require(limit == _FIXED_MAXIMUM_SECONDS, "maximum_total_constant_changed")
    receipt_identity = {
        key: value
        for key, value in receipt.items()
        if key not in _ELAPSED_FIELDS and key not in ("counters", "stage_manifest_sha256")
    }
    summary_identity = {
        key: value
        for key, value in summary.items()
        if key not in _ELAPSED_FIELDS and key != "counters"
    }
    _require(
        core._canon().canonical_json_bytes(summary_identity)
        == core._canon().canonical_json_bytes(receipt_identity),
        "summary_identity_mismatch",
    )
    receipt_counters = _counter_mapping(receipt.get("counters"), "counters_malformed")
    summary_counters = _counter_mapping(summary.get("counters"), "counters_malformed")
    for name in sorted(set(receipt_counters) | set(summary_counters)):
        if name == "elapsed_seconds":
            continue
        _require(
            receipt_counters.get(name) == summary_counters.get(name),
            f"summary_counter_{name}_mismatch",
        )
    summary_elapsed = _finite_seconds(
        summary.get("scientific_seconds_this_stage"), "summary_elapsed_malformed"
    )
    receipt_elapsed = _finite_seconds(
        receipt.get("scientific_seconds_this_stage"), "receipt_elapsed_malformed"
    )
    _require(
        _finite_seconds(
            summary_counters.get("elapsed_seconds"), "summary_counter_elapsed_malformed"
        )
        == summary_elapsed,
        "summary_elapsed_mismatch",
    )
    _require(
        _finite_seconds(
            receipt_counters.get("elapsed_seconds"), "receipt_counter_elapsed_malformed"
        )
        == receipt_elapsed,
        "receipt_elapsed_mismatch",
    )
    _require(0.0 <= summary_elapsed <= receipt_elapsed, "summary_elapsed_out_of_range")
    prior = _finite_seconds(
        receipt.get("prior_scientific_seconds_cumulative_bound"), "receipt_prior_malformed"
    )
    _require(
        _finite_seconds(
            summary.get("prior_scientific_seconds_cumulative_bound"), "summary_prior_malformed"
        )
        == prior,
        "summary_prior_mismatch",
    )
    _require(reserve <= prior, "receipt_prior_out_of_range")
    _require(prior + receipt_elapsed <= limit, "receipt_cumulative_out_of_range")
    _require(
        _finite_seconds(
            summary.get("scientific_seconds_cumulative_bound"), "summary_cumulative_malformed"
        )
        == prior + summary_elapsed,
        "summary_cumulative_mismatch",
    )
    _require(
        _finite_seconds(
            receipt.get("scientific_seconds_cumulative_bound"), "receipt_cumulative_malformed"
        )
        == prior + receipt_elapsed,
        "receipt_cumulative_mismatch",
    )
    _require(
        _finite_seconds(receipt.get("prelaunch_audit_reserve_seconds"), "receipt_reserve_malformed")
        == reserve,
        "receipt_reserve_mismatch",
    )
    _require(
        _finite_seconds(receipt.get("maximum_total_seconds"), "receipt_limit_malformed") == limit,
        "receipt_limit_mismatch",
    )
    return prior + receipt_elapsed


def _check_counters(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    counters = _counter_mapping(receipt.get("counters"), "counters_malformed")
    for name, expected in (
        ("public_tables", reporting.PUBLIC_TABLE_COUNT),
        ("source_diagnostic_tables", reporting.SOURCE_DIAGNOSTIC_TABLE_COUNT),
        ("reliability_tables", reporting.RELIABILITY_TABLE_COUNT),
    ):
        _require(
            _integer(counters.get(name), f"counter_{name}_invalid") == expected,
            f"counter_{name}_mismatch",
        )
    for name in reporting.ZERO_COUNTERS:
        _require(
            _integer(counters.get(name), f"counter_{name}_invalid") == 0, f"counter_{name}_nonzero"
        )
    for name in ("paired_figures", "diagnostic_figures"):
        _require(
            _integer(counters.get(name), f"counter_{name}_invalid") > 0, f"counter_{name}_missing"
        )
    return counters


def _check_bindings(stage: Path, receipt: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    bindings_path = stage / reporting.BINDINGS_NAME
    bindings_sha256 = _stable_sha256(bindings_path, "reporting_bindings_missing")
    bindings = core._read_json(bindings_path, "reporting_bindings")
    _require(isinstance(bindings, Mapping), "reporting_bindings_malformed")
    bindings = dict(bindings)
    _require(
        sha256_value(bindings) == str(receipt.get("reporting_binding_digest")),
        "reporting_binding_digest_mismatch",
    )
    return bindings, bindings_sha256


def _check_comparison_binding(
    stage: Path, receipt: Mapping[str, Any], bindings: Mapping[str, Any]
) -> None:
    path = stage / reporting.COMPARISON_BINDING_NAME
    core._reject_symlink_chain(path)
    _require(path.is_file() and not path.is_symlink(), "comparison_binding_missing")
    stored = core._read_json(path, "comparison_binding_missing")
    _require(isinstance(stored, Mapping), "comparison_binding_malformed")
    expected = {
        "comparison_receipt_sha256": str(receipt.get("comparison_receipt_sha256")),
        "comparison_manifest_sha256": str(receipt.get("comparison_manifest_sha256")),
    }
    _require(dict(stored) == expected, "comparison_binding_mismatch")
    _require(
        str(bindings.get("comparison_receipt_sha256")) == expected["comparison_receipt_sha256"],
        "comparison_binding_receipt_mismatch",
    )
    _require(
        str(bindings.get("comparison_manifest_sha256")) == expected["comparison_manifest_sha256"],
        "comparison_binding_manifest_mismatch",
    )


def _verify_sources(artifact_root: Any, bindings: Mapping[str, Any], deadline: float) -> None:
    bundle = {
        "artifact_root": Path(artifact_root),
        "permit_sha256": inputs.COMPREHENSIVE_PERMIT_SHA256,
    }
    reporting_inputs.verify_reporting_sources(bundle, bindings=bindings, deadline=deadline)


def _table_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    public_names = tuple(public_metrics.P05_PUBLIC_TABLE_NAMES)
    diagnostic_names = tuple(source_diagnostics.P05_SOURCE_DIAGNOSTIC_TABLE_NAMES)
    _require(
        len(public_names) == reporting.PUBLIC_TABLE_COUNT
        and len(set(public_names)) == reporting.PUBLIC_TABLE_COUNT,
        "public_table_names_mismatch",
    )
    _require(
        len(diagnostic_names) == reporting.SOURCE_DIAGNOSTIC_TABLE_COUNT
        and len(set(diagnostic_names)) == reporting.SOURCE_DIAGNOSTIC_TABLE_COUNT,
        "source_diagnostic_table_names_mismatch",
    )
    _require(
        len(reporting.RELIABILITY_TABLE_NAMES) == reporting.RELIABILITY_TABLE_COUNT,
        "reliability_table_names_mismatch",
    )
    return public_names, diagnostic_names


def _read_tables(
    public_root: Path, names: tuple[str, ...], deadline: float
) -> dict[str, pd.DataFrame]:
    """Return lossless lexical CSV frames; every column is read as ``str``.

    Metadata that merely looks numeric and empty values are preserved exactly;
    numeric callers must convert explicitly.
    """

    tables_dir = public_root / reporting.TABLES_DIR_NAME
    core._reject_symlink_chain(tables_dir)
    _require(tables_dir.is_dir() and not tables_dir.is_symlink(), "public_tables_missing")
    frames: dict[str, pd.DataFrame] = {}
    for name in names:
        freeze._check_deadline(deadline)
        _require(name not in frames, "public_table_name_collision")
        path = tables_dir / f"{name}.csv"
        core._reject_symlink_chain(path)
        _require(path.is_file() and not path.is_symlink(), "public_table_missing")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        _require(isinstance(frame, pd.DataFrame), "public_table_malformed")
        frames[name] = frame
    _require(len(frames) == reporting.TOTAL_TABLE_COUNT, "public_table_count_mismatch")
    reporting._reject_private_columns(frames)
    return frames


_INT_COST_KEYS = (
    "new_source_fits",
    "reused_pilot_fits",
    "source_evidence_fits",
    "unique_refitted_models",
    "unique_scalar_calibrations",
    "new_neural_fits_total",
    "strategy_alias_count",
    "new_source_optimizer_updates",
    "refit_optimizer_updates",
    "combined_new_optimizer_updates",
)
_TIME_COST_KEYS = (
    "source_scientific_seconds",
    "refit_scientific_seconds",
    "scientific_seconds_cumulative_bound_through_comparison",
)


def _check_cost_agreement(costs: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    reporting_inputs._check_public_costs(dict(costs))
    try:
        accounting = recovery_source.from_authenticated(receipt)
    except recovery_source.RecoverySourceError as error:
        raise P05PublicationGateError("public_cost_source_accounting_malformed") from error
    _require(isinstance(accounting, Mapping), "public_cost_source_accounting_malformed")
    accounting_mode = str(accounting.get("mode"))
    recovery_keys = frozenset(reporting_inputs.RECOVERY_PUBLIC_COST_KEYS)
    present_recovery = frozenset(costs) & recovery_keys
    if accounting_mode == "recovered":
        _require(
            present_recovery == recovery_keys,
            "public_cost_recovery_keys_incomplete",
        )
    else:
        _require(not present_recovery, "public_cost_recovery_keys_unexpected")
    for name in _INT_COST_KEYS:
        _integer(costs.get(name), f"public_cost_{name}_malformed")
    for name in _TIME_COST_KEYS:
        _require(
            _finite_seconds(costs.get(name), f"public_cost_{name}_malformed") >= 0.0,
            f"public_cost_{name}_negative",
        )
    peak = costs.get("refit_peak_allocated_gpu_bytes")
    if peak is not None:
        _require(
            _integer(peak, "public_cost_refit_peak_allocated_gpu_bytes_malformed")
            <= evaluation_authority.MAXIMUM_CUDA_ALLOCATED_BYTES,
            "public_cost_refit_peak_allocated_gpu_bytes_out_of_range",
        )
    _require(
        costs["new_source_fits"] == reporting_inputs.NEW_SOURCE_FITS,
        "public_cost_new_source_fits_mismatch",
    )
    _require(
        costs["reused_pilot_fits"] == reporting_inputs.REUSED_PILOT_FITS,
        "public_cost_reused_pilot_fits_mismatch",
    )
    _require(
        costs["source_evidence_fits"] == reporting_inputs.SOURCE_EVIDENCE_FITS,
        "public_cost_source_evidence_fits_mismatch",
    )
    _require(
        costs["strategy_alias_count"] == reporting_inputs.STRATEGY_ALIASES,
        "public_cost_strategy_alias_count_mismatch",
    )
    unique = costs["unique_refitted_models"]
    _require(
        0 < unique <= evaluation_authority.MAXIMUM_REFITS,
        "public_cost_unique_refitted_models_out_of_range",
    )
    _require(
        costs["unique_scalar_calibrations"] == unique,
        "public_cost_unique_scalar_calibrations_mismatch",
    )
    _require(
        costs["new_neural_fits_total"] == reporting_inputs.NEW_SOURCE_FITS + unique,
        "public_cost_new_neural_fits_total_mismatch",
    )
    source_steps = _integer(
        receipt.get("source_optimizer_steps"), "receipt_source_optimizer_steps_malformed"
    )
    refit_steps = _integer(
        receipt.get("refit_optimizer_steps"), "receipt_refit_optimizer_steps_malformed"
    )
    _require(
        costs["new_source_optimizer_updates"] == source_steps,
        "public_cost_source_updates_mismatch",
    )
    _require(
        costs["refit_optimizer_updates"] == refit_steps,
        "public_cost_refit_updates_mismatch",
    )
    _require(
        costs["combined_new_optimizer_updates"] == source_steps + refit_steps,
        "public_cost_combined_updates_mismatch",
    )
    if accounting_mode == "recovered":
        _require(
            costs["new_source_attempts"] == accounting["source_attempts"],
            "public_cost_new_source_attempts_mismatch",
        )
        _require(
            costs["new_neural_attempts_total"]
            <= accounting["maximum_new_neural_executions"],
            "public_cost_new_neural_attempts_exceeded",
        )
        _require(
            costs["source_optimizer_updates_observed_lower_bound"]
            == accounting["source_optimizer_steps_observed_lower_bound"],
            "public_cost_source_updates_observed_mismatch",
        )
        _require(
            costs["source_optimizer_updates_charged_upper_bound"]
            == accounting["source_optimizer_steps_charged_upper_bound"],
            "public_cost_source_updates_charged_mismatch",
        )
        _require(
            costs["combined_optimizer_updates_observed_lower_bound"]
            == accounting["source_optimizer_steps_observed_lower_bound"] + refit_steps,
            "public_cost_combined_updates_observed_mismatch",
        )
        _require(
            costs["combined_optimizer_updates_charged_upper_bound"]
            == accounting["source_optimizer_steps_charged_upper_bound"] + refit_steps,
            "public_cost_combined_updates_charged_mismatch",
        )
    _require(
        refit_steps <= unique * evaluation_authority.MAXIMUM_UPDATES_PER_REFIT,
        "public_cost_refit_updates_exceeded",
    )
    prior = _finite_seconds(
        receipt.get("prior_scientific_seconds_cumulative_bound"), "receipt_prior_malformed"
    )
    _require(
        _finite_seconds(
            costs["scientific_seconds_cumulative_bound_through_comparison"],
            "public_cost_cumulative_malformed",
        )
        == prior,
        "public_cost_cumulative_mismatch",
    )


def _read_costs(public_root: Path, receipt: Mapping[str, Any], deadline: float) -> dict[str, Any]:
    freeze._check_deadline(deadline)
    path = public_root / reporting.TABLES_DIR_NAME / reporting.COSTS_NAME
    core._reject_symlink_chain(path)
    _require(path.is_file() and not path.is_symlink(), "costs_json_missing")
    raw = path.read_bytes()
    stored = json.loads(raw.decode("utf-8"))
    _require(isinstance(stored, Mapping), "costs_json_malformed")
    _require(core._canon().canonical_json_bytes(dict(stored)) == raw, "costs_json_not_canonical")
    costs = dict(reporting._check_public_costs({"public_costs": stored}, reporting_inputs))
    _check_cost_agreement(costs, receipt)
    return costs


def _read_figures(
    public_root: Path, deadline: float
) -> tuple[
    dict[str, Any],
    dict[str, tuple[Path, str, Mapping[str, str]]],
    dict[str, int],
]:
    figures_root = public_root / reporting.FIGURES_DIR_NAME
    core._reject_symlink_chain(figures_root)
    specs = (
        (reporting.PAIRED_UNIT_NAME, benchmark_figures.MANIFEST_NAME, "paired"),
        (reporting.DIAGNOSTIC_UNIT_NAME, diagnostic_figures.MANIFEST_NAME, "diagnostic"),
    )
    manifests: dict[str, Any] = {}
    inventory: dict[str, tuple[Path, str, Mapping[str, str]]] = {}
    counts: dict[str, int] = {}
    for unit, manifest_name, code in specs:
        freeze._check_deadline(deadline)
        render_root = figures_root / unit / reporting.RENDER_DIR_NAME
        core._reject_symlink_chain(render_root)
        _require(render_root.is_dir() and not render_root.is_symlink(), f"{code}_render_missing")
        manifest_path = render_root / manifest_name
        core._reject_symlink_chain(manifest_path)
        _require(
            manifest_path.is_file() and not manifest_path.is_symlink(),
            f"{code}_manifest_missing",
        )
        manifest = core._read_json(manifest_path, f"{code}_manifest_missing")
        _require(isinstance(manifest, Mapping), f"{code}_manifest_malformed")
        manifest = dict(manifest)
        _files, listed = reporting._verify_figure_files(render_root, manifest, code, manifest_name)
        counts[code] = reporting._verify_figure_records(manifest, listed, code)
        manifests[unit] = manifest
        inventory[unit] = (render_root, manifest_name, listed)
    return manifests, inventory, counts


def _collect_files(
    public_root: Path,
    frames: Mapping[str, pd.DataFrame],
    inventory: Mapping[str, tuple[Path, str, Mapping[str, str]]],
    deadline: float,
) -> dict[str, str]:
    files: dict[str, str] = {}
    for name in frames:
        freeze._check_deadline(deadline)
        relative = f"{reporting.TABLES_DIR_NAME}/{name}.csv"
        files[relative] = _stable_sha256(public_root / relative, "public_table_missing")
    costs_relative = f"{reporting.TABLES_DIR_NAME}/{reporting.COSTS_NAME}"
    files[costs_relative] = _stable_sha256(public_root / costs_relative, "costs_json_missing")
    for render_root, manifest_name, listed in inventory.values():
        freeze._check_deadline(deadline)
        directory = render_root.relative_to(public_root).as_posix()
        manifest_relative = f"{directory}/{manifest_name}"
        files[manifest_relative] = _stable_sha256(
            public_root / manifest_relative, "figure_manifest_missing"
        )
        for relative_name, digest in listed.items():
            freeze._check_deadline(deadline)
            key = f"{directory}/{relative_name}"
            _require(
                _stable_sha256(public_root / key, "figure_file_missing") == digest,
                "figure_file_changed",
            )
            files[key] = digest
    return files


def _check_row_counters(
    counters: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    public_names: tuple[str, ...],
    diagnostic_names: tuple[str, ...],
) -> None:
    for name, names in (
        ("public_rows", public_names),
        ("source_diagnostic_rows", diagnostic_names),
        ("reliability_rows", reporting.RELIABILITY_TABLE_NAMES),
    ):
        expected = sum(len(frames[table]) for table in names)
        _require(
            _integer(counters.get(name), f"counter_{name}_invalid") == expected,
            f"counter_{name}_mismatch",
        )


def _final_rehash(
    run_root: Path,
    stage: Path,
    receipt_sha256: str,
    manifest_sha256: str,
    bindings_sha256: str,
    deadline: float,
) -> None:
    anchors = (
        (
            run_root / reporting.RECEIPT_NAME,
            "reporting_receipt_missing",
            receipt_sha256,
            "reporting_receipt_changed",
        ),
        (
            stage / reporting.MANIFEST_NAME,
            "reporting_manifest_missing",
            manifest_sha256,
            "reporting_manifest_changed",
        ),
        (
            stage / reporting.BINDINGS_NAME,
            "reporting_bindings_missing",
            bindings_sha256,
            "reporting_bindings_changed",
        ),
    )
    for path, missing, digest, changed in anchors:
        freeze._check_deadline(deadline)
        _require(_stable_sha256(path, missing) == digest, changed)
    pilot._verify_manifest(stage)
    for path, missing, digest, changed in anchors:
        freeze._check_deadline(deadline)
        _require(_stable_sha256(path, missing) == digest, changed)


def load_public_bundle(
    artifact_root: Any,
    *,
    expected_reporting_receipt_sha256: Any,
    deadline: Any,
) -> PublicBundle:
    """Authenticate a reviewed receipt and return its unchanged public files.

    Tables contain lexical CSV strings; numerical consumers convert explicitly.
    The returned cumulative bound is the sealed reporting-stage bound, not a
    fresh scientific allowance. The caller supplies the remaining audit deadline.
    """

    deadline = _finite_seconds(deadline, "deadline_malformed")
    freeze._check_deadline(deadline)
    run_root = _run_root(artifact_root)
    receipt, receipt_sha256 = _load_receipt(run_root, expected_reporting_receipt_sha256)
    _check_receipt_identity(receipt)
    stage, manifest_sha256 = _check_stage_manifest(run_root, receipt)
    freeze._check_deadline(deadline)
    summary = core._read_json(stage / reporting.SUMMARY_NAME, "reporting_summary_missing")
    _require(isinstance(summary, Mapping), "reporting_summary_malformed")
    cumulative_seconds = _check_summary(receipt, dict(summary))
    counters = _check_counters(receipt)
    bindings, bindings_sha256 = _check_bindings(stage, receipt)
    _check_comparison_binding(stage, receipt, bindings)
    _verify_sources(artifact_root, bindings, deadline)

    public_names, diagnostic_names = _table_names()
    public_root = stage / reporting.PUBLIC_ROOT_NAME
    core._reject_symlink_chain(public_root)
    _require(public_root.is_dir() and not public_root.is_symlink(), "public_root_malformed")
    frames = _read_tables(
        public_root,
        (*public_names, *diagnostic_names, *reporting.RELIABILITY_TABLE_NAMES),
        deadline,
    )
    _check_row_counters(counters, frames, public_names, diagnostic_names)
    costs = _read_costs(public_root, receipt, deadline)
    figure_manifests, inventory, counts = _read_figures(public_root, deadline)
    _require(
        counts["paired"]
        == _integer(counters.get("paired_figures"), "counter_paired_figures_invalid"),
        "paired_figure_counter_mismatch",
    )
    _require(
        counts["diagnostic"]
        == _integer(counters.get("diagnostic_figures"), "counter_diagnostic_figures_invalid"),
        "diagnostic_figure_counter_mismatch",
    )
    reporting._verify_public_inventory(public_root, frames, inventory)
    files = _collect_files(public_root, frames, inventory, deadline)
    _verify_sources(artifact_root, bindings, deadline)
    _final_rehash(run_root, stage, receipt_sha256, manifest_sha256, bindings_sha256, deadline)
    freeze._check_deadline(deadline)
    return PublicBundle(
        public_root,
        files,
        frames,
        costs,
        receipt_sha256,
        cumulative_seconds,
        figure_manifests,
    )


def verify_public_bundle(
    artifact_root: Any,
    *,
    expected_reporting_receipt_sha256: Any,
    deadline: Any,
) -> PublicBundle:
    """Re-run the read-only publication checks and return the sealed bundle."""

    return load_public_bundle(
        artifact_root,
        expected_reporting_receipt_sha256=expected_reporting_receipt_sha256,
        deadline=deadline,
    )
