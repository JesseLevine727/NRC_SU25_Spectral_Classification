"""P05 legacy-reference loading: read-only frozen P03/P04 evidence.

Opens exactly the two pinned legacy aggregation shards for the completed
classical P03 execution and the completed private P04 execution.  Every legacy
file is authenticated against its shard ``_STATE.json`` inventory and content
hash before the parquet read; prediction and state hashes are checked again
after both reads. Nothing is written, no feature array,
model, optimizer or prediction is touched, and no legacy governing execution
context is constructed.  The caller must already have completed the full
frozen-prediction gate; this module independently re-checks the new-run
evaluation receipt and stage manifest before opening any legacy evidence.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_core_run as core

__all__ = ["P05LegacyReferenceError", "load_references"]

P03_RUN_ID = "P03-513a0f9686c37cbc0d682645"
P04_RUN_ID = "P04-e845290bb15d37882f29da6f"
P03_EXECUTION_ID = "513a0f9686c37cbc0d682645"
P04_PROTECTED_STATE_SHA256 = "e845290bb15d37882f29da6f843620846ae61c75bfca2231c6e6c7686d748d19"
P04_AGGREGATION_STATE_SHA256 = "85280d95620696993af0fc93afb61f86d66ce1f7bfb6e1205139dc2b80bf9793"
SHARD_STATE_SCHEMA = "p03-shard-state-v1"
P03_DESCRIPTOR_SCHEMA = "p03-final-aggregation-v1"
P04_REPORT_SCHEMA = "nato-sers-p04-aggregation-report-v1"
STATE_NAME = "_STATE.json"
P03_DESCRIPTOR_NAME = "final_aggregation_descriptor.json"
P03_VALIDATION_NAME = "prediction_schema_validation.json"
P03_PREDICTIONS_NAME = "final_predictions.parquet"
P04_REPORT_NAME = "P04_AGGREGATION_REPORT.json"
P04_PREDICTIONS_NAME = "ensemble_test_predictions.parquet"
FINAL_AGGREGATION_DIR = "final_aggregation"
SHARDS_DIR = "shards"
SHARD_NAME = "shard-000000"
SHARD_ID = 0
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class P05LegacyReferenceError(core.P05CoreError):
    """Stable, path-free legacy-reference loading failure."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05LegacyReferenceError(code)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _finite(value: Any, code: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), code)
    number = float(value)
    _require(math.isfinite(number), code)
    return number


def _mapping(path: Path, name: str) -> Mapping[str, Any]:
    core._reject_symlink_chain(path)
    _require(path.is_file() and not path.is_symlink(), f"{name}_missing")
    value = core._read_json(path, name)
    _require(isinstance(value, Mapping), f"{name}_malformed")
    return value


def _safe_relative(name: Any) -> bool:
    if not isinstance(name, str) or not name or "\\" in name:
        return False
    path = Path(name)
    if path.is_absolute():
        return False
    return all(part not in ("", ".", "..") for part in name.split("/"))


def _legacy_shard(artifact_root: Any, family: str, run_id: str) -> Path:
    return (
        Path(artifact_root)
        / family
        / "runs"
        / run_id
        / FINAL_AGGREGATION_DIR
        / SHARDS_DIR
        / SHARD_NAME
    )


def _authenticate_shard(
    shard_dir: Path, *, deadline: float
) -> tuple[Mapping[str, Any], Mapping[str, str], str]:
    core._reject_symlink_chain(shard_dir)
    _require(shard_dir.is_dir() and not shard_dir.is_symlink(), "legacy_shard_missing")
    state_path = shard_dir / STATE_NAME
    core._reject_symlink_chain(state_path)
    _require(state_path.is_file() and not state_path.is_symlink(), "legacy_state_missing")
    state = core._read_json(state_path, "legacy_state")
    _require(isinstance(state, Mapping), "legacy_state_malformed")
    _require(state.get("schema_version") == SHARD_STATE_SCHEMA, "legacy_state_schema_mismatch")
    _require(state.get("execution_status") == "complete", "legacy_state_incomplete")
    _require(
        type(state.get("shard_id")) is int and state["shard_id"] == SHARD_ID,
        "legacy_state_shard_mismatch",
    )
    _require(_is_hex64(state.get("protected_state_sha256")), "legacy_state_protected_malformed")
    files = state.get("files")
    _require(isinstance(files, Mapping) and bool(files), "legacy_state_files_malformed")
    for name, digest in files.items():
        _require(_safe_relative(name), "legacy_state_path_unsafe")
        _require(_is_hex64(digest), "legacy_state_digest_malformed")
        target = shard_dir / name
        core._reject_symlink_chain(target)
        _require(target.is_file() and not target.is_symlink(), "legacy_state_file_missing")
    actual: set[str] = set()
    for entry in shard_dir.rglob("*"):
        _require(not entry.is_symlink(), "legacy_shard_symlink_rejected")
        if entry.is_file() and entry != state_path:
            actual.add(entry.relative_to(shard_dir).as_posix())
    _require(actual == set(files), "legacy_state_inventory_mismatch")
    for name, digest in files.items():
        freeze._check_deadline(deadline)
        _require(
            core._canon().sha256_file(shard_dir / name) == digest, "legacy_state_hash_mismatch"
        )
    freeze._check_deadline(deadline)
    return state, files, core._canon().sha256_file(state_path)


def _validate_p03_evidence(shard_dir: Path, protected: str) -> None:
    descriptor = _mapping(shard_dir / P03_DESCRIPTOR_NAME, "p03_descriptor")
    _require(
        descriptor.get("schema_version") == P03_DESCRIPTOR_SCHEMA, "p03_descriptor_schema_mismatch"
    )
    _require(descriptor.get("execution_run_id") == P03_RUN_ID, "p03_descriptor_run_mismatch")
    _require(
        descriptor.get("protected_state_sha256") == protected, "p03_descriptor_protected_mismatch"
    )
    validation = _mapping(shard_dir / P03_VALIDATION_NAME, "p03_prediction_validation")
    _require(validation.get("status") == "pass", "p03_prediction_validation_failed")


def _validate_p04_evidence(shard_dir: Path) -> None:
    report = _mapping(shard_dir / P04_REPORT_NAME, "p04_report")
    _require(report.get("schema_version") == P04_REPORT_SCHEMA, "p04_report_schema_mismatch")
    _require(report.get("status") == "pass", "p04_report_status_failed")
    _require(report.get("run_id") == P04_RUN_ID, "p04_report_run_mismatch")
    _require(
        report.get("protected_state_sha256") == P04_PROTECTED_STATE_SHA256,
        "p04_report_protected_mismatch",
    )
    _require(
        report.get("aggregation_state_sha256") == P04_AGGREGATION_STATE_SHA256,
        "p04_report_aggregation_mismatch",
    )


def _read_verified(
    shard_dir: Path,
    files: Mapping[str, str],
    name: str,
    *,
    state_sha256: str,
    deadline: float,
) -> tuple[pd.DataFrame, str]:
    _require(name in files, "legacy_predictions_unregistered")
    target = shard_dir / name
    core._reject_symlink_chain(target)
    freeze._check_deadline(deadline)
    try:
        frame = pd.read_parquet(target)
    except Exception as error:  # noqa: BLE001
        raise P05LegacyReferenceError("legacy_predictions_read_failed") from error
    freeze._check_deadline(deadline)
    _require(isinstance(frame, pd.DataFrame), "legacy_predictions_malformed")
    _require(core._canon().sha256_file(target) == files[name], "legacy_predictions_hash_changed")
    _require(
        core._canon().sha256_file(shard_dir / STATE_NAME) == state_sha256,
        "legacy_state_changed",
    )
    freeze._check_deadline(deadline)
    return frame, files[name]


def load_references(
    bundle: Any,
    *,
    authenticated: Any,
    deadline: Any,
) -> dict[str, Any]:
    """Load the two pinned legacy prediction frames read-only."""

    _require(isinstance(bundle, Mapping), "bundle_malformed")
    _require(isinstance(authenticated, Mapping), "authenticated_malformed")
    deadline = _finite(deadline, "deadline_malformed")
    freeze._check_deadline(deadline)

    receipt = authenticated.get("evaluation_receipt")
    _require(isinstance(receipt, Mapping), "evaluation_receipt_malformed")
    _require(receipt.get("status") == "complete", "evaluation_receipt_incomplete")
    for name in ("predictions_frozen", "predictions_complete", "all_complete"):
        _require(receipt.get(name) is True, f"evaluation_receipt_{name}_not_true")

    permit_sha256 = bundle.get("permit_sha256")
    _require(_is_hex64(permit_sha256), "permit_sha256_malformed")
    _require(receipt.get("permit_sha256") == permit_sha256, "evaluation_permit_mismatch")
    artifact_root = bundle.get("artifact_root")
    _require(artifact_root is not None, "artifact_root_missing")

    run_root = (
        Path(artifact_root) / evaluation.COMPREHENSIVE_DIR / evaluation.RUNS_DIR / permit_sha256
    )
    receipt_path = run_root / evaluation.RECEIPT_NAME
    on_disk = _mapping(receipt_path, "evaluation_receipt")
    _require(dict(on_disk) == dict(receipt), "evaluation_receipt_changed")
    manifest_sha256 = receipt.get("stage_manifest_sha256")
    _require(_is_hex64(manifest_sha256), "stage_manifest_sha256_malformed")
    manifest_path = run_root / evaluation.STAGE_NAME / evaluation.MANIFEST_NAME
    core._reject_symlink_chain(manifest_path)
    _require(
        manifest_path.is_file() and not manifest_path.is_symlink(), "evaluation_manifest_missing"
    )
    _require(
        core._canon().sha256_file(manifest_path) == manifest_sha256, "evaluation_manifest_changed"
    )
    receipt_sha256 = core._canon().sha256_file(receipt_path)
    freeze._check_deadline(deadline)

    p03_dir = _legacy_shard(artifact_root, "p03", P03_RUN_ID)
    p03_state, p03_files, p03_state_sha256 = _authenticate_shard(p03_dir, deadline=deadline)
    p03_protected = str(p03_state["protected_state_sha256"])
    _require(p03_protected.startswith(P03_EXECUTION_ID), "p03_protected_mismatch")
    _validate_p03_evidence(p03_dir, p03_protected)
    freeze._check_deadline(deadline)

    p04_dir = _legacy_shard(artifact_root, "p04", P04_RUN_ID)
    p04_state, p04_files, p04_state_sha256 = _authenticate_shard(p04_dir, deadline=deadline)
    p04_protected = str(p04_state["protected_state_sha256"])
    _require(p04_protected == P04_AGGREGATION_STATE_SHA256, "p04_protected_mismatch")
    _validate_p04_evidence(p04_dir)
    freeze._check_deadline(deadline)

    p03_frame, p03_predictions_sha256 = _read_verified(
        p03_dir, p03_files, P03_PREDICTIONS_NAME, state_sha256=p03_state_sha256, deadline=deadline
    )
    p04_frame, p04_predictions_sha256 = _read_verified(
        p04_dir, p04_files, P04_PREDICTIONS_NAME, state_sha256=p04_state_sha256, deadline=deadline
    )
    # A later read must not mask a change to an earlier input.
    for directory, state_hash, name, prediction_hash in (
        (p03_dir, p03_state_sha256, P03_PREDICTIONS_NAME, p03_predictions_sha256),
        (p04_dir, p04_state_sha256, P04_PREDICTIONS_NAME, p04_predictions_sha256),
    ):
        freeze._check_deadline(deadline)
        core._reject_symlink_chain(directory / name)
        core._reject_symlink_chain(directory / STATE_NAME)
        _require(
            core._canon().sha256_file(directory / name) == prediction_hash,
            "legacy_predictions_hash_changed",
        )
        _require(
            core._canon().sha256_file(directory / STATE_NAME) == state_hash,
            "legacy_state_changed",
        )
    _require(
        core._canon().sha256_file(receipt_path) == receipt_sha256,
        "evaluation_receipt_changed",
    )
    _require(
        core._canon().sha256_file(manifest_path) == manifest_sha256,
        "evaluation_manifest_changed",
    )
    freeze._check_deadline(deadline)

    return {
        "p03_predictions": p03_frame,
        "p04_ensemble": p04_frame,
        "bindings": {
            "p03_run_id": P03_RUN_ID,
            "p04_run_id": P04_RUN_ID,
            "p03_state_sha256": p03_state_sha256,
            "p04_state_sha256": p04_state_sha256,
            "p03_predictions_sha256": p03_predictions_sha256,
            "p04_predictions_sha256": p04_predictions_sha256,
            "p03_protected_state_sha256": p03_protected,
            "p04_shard_protected_state_sha256": p04_protected,
            "p04_execution_protected_state_sha256": P04_PROTECTED_STATE_SHA256,
            "evaluation_receipt_sha256": receipt_sha256,
        },
    }
