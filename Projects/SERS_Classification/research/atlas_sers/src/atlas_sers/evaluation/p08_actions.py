"""Validate already-loaded frozen P08 action arrays against P01 metadata.

The caller loads and authenticates artifacts separately (for example with
``allow_pickle=False`` plus artifact hash checks).  This module only inspects
in-memory NumPy arrays and frozen registry / row-QC metadata: it never
transforms spectra, fits models, selects representations, or touches files.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_ACTIONS = ("R_MIN_400_1800", "R_SG_400_1800", "R_ARPLS_400_1800")
FEATURES = 1401

_ACTION_KEYS = frozenset({"axis_cm1", "intensity", "observation_uid"})
_AXIS_START = 400
_AXIS_END = 1800
_DTYPE_NAME = "float32"
_NORMALIZATION_ATOL = 1e-6
_HEX_DIGITS = frozenset("0123456789abcdef")
_REGISTRY_COLUMNS = (
    "representation_id",
    "rows",
    "features",
    "dtype",
    "axis_start_cm1",
    "axis_end_cm1",
    "axis_sha256",
    "array_sha256",
    "row_order_sha256",
    "invalid_rows",
    "invariant_status",
)
_ROW_QC_COLUMNS = (
    "observation_uid",
    "representation_id",
    "valid",
    "reason_code",
    "representation_invariant_status",
)


class ActionAuditError(ValueError):
    """Frozen-action validation failure with a fixed, data-free reason code."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


def _fail(reason_code: str) -> None:
    raise ActionAuditError(reason_code)


def _is_bool(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _as_integral(value: Any, reason_code: str) -> int:
    if _is_bool(value):
        _fail(reason_code)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
    _fail(reason_code)


def _as_real(value: Any, reason_code: str) -> float:
    if _is_bool(value) or isinstance(value, (complex, np.complexfloating, str, bytes)):
        _fail(reason_code)
    if isinstance(value, (int, np.integer, float, np.floating)):
        try:
            number = float(value)
        except OverflowError:
            _fail(reason_code)
        if math.isfinite(number):
            return number
    _fail(reason_code)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _normalize_manifest(manifest_uids: Any) -> list[str]:
    if isinstance(manifest_uids, np.ndarray):
        if manifest_uids.ndim != 1 or manifest_uids.dtype.kind != "U":
            _fail("manifest_invalid")
        values = [str(item) for item in manifest_uids.tolist()]
    elif isinstance(manifest_uids, (list, tuple)):
        values = []
        for item in manifest_uids:
            if not isinstance(item, str):
                _fail("manifest_invalid")
            values.append(item)
    else:
        _fail("manifest_invalid")
    if not values or any(not value.strip() for value in values):
        _fail("manifest_invalid")
    if len(set(values)) != len(values):
        _fail("manifest_invalid")
    return values


def _read_registry_row(row: pd.Series, n_rows: int) -> dict[str, Any]:
    rows = _as_integral(row["rows"], "registry_rows_invalid")
    if rows != n_rows:
        _fail("registry_rows_invalid")
    features = _as_integral(row["features"], "registry_features_invalid")
    if features != FEATURES:
        _fail("registry_features_invalid")
    invalid_rows = _as_integral(row["invalid_rows"], "registry_invalid_rows_invalid")
    if invalid_rows != 0:
        _fail("registry_invalid_rows_invalid")
    if not isinstance(row["dtype"], str) or row["dtype"] != _DTYPE_NAME:
        _fail("registry_dtype_invalid")
    start = _as_real(row["axis_start_cm1"], "registry_bounds_invalid")
    end = _as_real(row["axis_end_cm1"], "registry_bounds_invalid")
    if start != float(_AXIS_START) or end != float(_AXIS_END):
        _fail("registry_bounds_invalid")
    if not isinstance(row["invariant_status"], str) or row["invariant_status"] != "pass":
        _fail("registry_invariant_status_invalid")
    for column in ("axis_sha256", "array_sha256", "row_order_sha256"):
        if not _is_sha256(row[column]):
            _fail("registry_sha_invalid")
    return {
        "rows": rows,
        "features": features,
        "dtype": str(row["dtype"]),
        "invalid_rows": invalid_rows,
        "axis_sha256": str(row["axis_sha256"]),
        "array_sha256": str(row["array_sha256"]),
        "row_order_sha256": str(row["row_order_sha256"]),
    }


def _validate_action(
    action: Any,
    manifest: list[str],
    row_meta: dict[str, Any],
    row_order_sha256: str,
) -> None:
    if not isinstance(action, Mapping):
        _fail("action_invalid")
    if set(action.keys()) != _ACTION_KEYS:
        _fail("action_keys")
    n_rows = len(manifest)
    axis = action["axis_cm1"]
    intensity = action["intensity"]
    uids = action["observation_uid"]
    if not isinstance(axis, np.ndarray) or axis.dtype != np.float32:
        _fail("axis_invalid")
    if axis.ndim != 1 or axis.shape[0] != FEATURES:
        _fail("axis_invalid")
    if not np.array_equal(axis, np.arange(_AXIS_START, _AXIS_END + 1, dtype=np.float32)):
        _fail("axis_invalid")
    if not isinstance(intensity, np.ndarray) or intensity.dtype != np.float32:
        _fail("intensity_invalid")
    if intensity.ndim != 2 or intensity.shape != (n_rows, FEATURES):
        _fail("intensity_invalid")
    if not bool(np.all(np.isfinite(intensity))):
        _fail("intensity_nonfinite")
    if not bool(np.allclose(intensity.min(axis=1), 0.0, rtol=0.0, atol=_NORMALIZATION_ATOL)):
        _fail("normalization_failed")
    if not bool(np.allclose(intensity.max(axis=1), 1.0, rtol=0.0, atol=_NORMALIZATION_ATOL)):
        _fail("normalization_failed")
    if not isinstance(uids, np.ndarray) or uids.dtype.kind != "U":
        _fail("uid_invalid")
    if uids.ndim != 1 or uids.shape[0] != n_rows:
        _fail("uid_invalid")
    if [str(item) for item in uids.tolist()] != manifest:
        _fail("uid_mismatch")
    if hashlib.sha256(axis.tobytes(order="C")).hexdigest() != row_meta["axis_sha256"]:
        _fail("axis_sha_mismatch")
    if hashlib.sha256(intensity.tobytes(order="C")).hexdigest() != row_meta["array_sha256"]:
        _fail("array_sha_mismatch")
    if row_order_sha256 != row_meta["row_order_sha256"]:
        _fail("row_order_sha_mismatch")


def _validate_row_qc(row_qc: pd.DataFrame, manifest: list[str]) -> None:
    if row_qc.columns.duplicated().any():
        _fail("row_qc_columns")
    if any(column not in row_qc.columns for column in _ROW_QC_COLUMNS):
        _fail("row_qc_columns")
    representation_ids = row_qc["representation_id"].tolist()
    if any(
        not isinstance(representation_id, str) or not representation_id.strip()
        for representation_id in representation_ids
    ):
        _fail("row_qc_representation_id_invalid")
    expected = set(manifest)
    n_rows = len(manifest)
    for representation_id in REQUIRED_ACTIONS:
        subset = row_qc.loc[row_qc["representation_id"] == representation_id]
        if len(subset) != n_rows:
            _fail("row_qc_coverage")
        uids = subset["observation_uid"].tolist()
        if any(not isinstance(uid, str) or not uid.strip() for uid in uids):
            _fail("row_qc_coverage")
        if len(set(uids)) != n_rows or set(uids) != expected:
            _fail("row_qc_coverage")
        for valid in subset["valid"].tolist():
            if not _is_bool(valid) or not bool(valid):
                _fail("row_qc_valid")
        for reason in subset["reason_code"].tolist():
            if not isinstance(reason, str) or reason != "included":
                _fail("row_qc_reason_invalid")
        for status in subset["representation_invariant_status"].tolist():
            if not isinstance(status, str) or status != "pass":
                _fail("row_qc_status_invalid")


def audit_frozen_actions(
    actions: Any,
    registry: Any,
    manifest_uids: Any,
    row_qc: Any,
) -> dict[str, Any]:
    """Validate loaded frozen action arrays against frozen P01 metadata."""
    if not isinstance(registry, pd.DataFrame):
        _fail("registry_type")
    if not isinstance(row_qc, pd.DataFrame):
        _fail("row_qc_type")
    if not isinstance(actions, Mapping):
        _fail("actions_type")
    if set(actions.keys()) != set(REQUIRED_ACTIONS):
        _fail("actions_keys")
    if registry.columns.duplicated().any():
        _fail("registry_columns")
    if any(column not in registry.columns for column in _REGISTRY_COLUMNS):
        _fail("registry_columns")
    representation_ids = registry["representation_id"].tolist()
    if any(
        not isinstance(representation_id, str) or not representation_id.strip()
        for representation_id in representation_ids
    ):
        _fail("registry_representation_id_invalid")
    unique_ids = set(representation_ids)
    if len(unique_ids) != len(representation_ids):
        _fail("registry_duplicate")
    for representation_id in REQUIRED_ACTIONS:
        if representation_id not in unique_ids:
            _fail("registry_missing")
    manifest = _normalize_manifest(manifest_uids)
    row_order_sha256 = hashlib.sha256("\n".join(manifest).encode("utf-8")).hexdigest()
    _validate_row_qc(row_qc, manifest)
    records: list[dict[str, Any]] = []
    for representation_id in REQUIRED_ACTIONS:
        row = registry.loc[registry["representation_id"] == representation_id].iloc[0]
        row_meta = _read_registry_row(row, len(manifest))
        _validate_action(actions[representation_id], manifest, row_meta, row_order_sha256)
        records.append(
            {
                "representation_id": representation_id,
                "rows": row_meta["rows"],
                "features": row_meta["features"],
                "dtype": row_meta["dtype"],
                "axis_sha256": row_meta["axis_sha256"],
                "array_sha256": row_meta["array_sha256"],
                "row_order_sha256": row_meta["row_order_sha256"],
                "invalid_rows": 0,
                "normalization_passed": True,
            }
        )
    records.sort(key=lambda record: record["representation_id"])
    return {
        "status": "pass",
        "scope": "loaded_frozen_action_integrity_only",
        "action_count": len(REQUIRED_ACTIONS),
        "rows": len(manifest),
        "features": FEATURES,
        "actions": records,
    }
