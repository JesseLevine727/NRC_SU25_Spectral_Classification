"""Row-level QC routing for the P08 supervised evaluation slice.

Frozen single-row routing only: no quantile fitting, batch statistics, target
adaptation, dataset reads, model fitting, prediction or execution
authorization. Source-only threshold provenance, valid action data and
test-leakage avoidance remain the caller's responsibility; digests alone do not
prove roles. Row QC-derived values are private operational content, so do not
publish actual per-row records produced by this module.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from atlas_sers.splits.p02 import build_qc_gate_library

from .p08_qc_blocks import canonical_sha256
from .p08_qc_thresholds import (
    FEATURES,
    QUANTILES,
    derive_qc_features,
    validate_threshold_state,
)

ACTION_IDS = ("R_MIN_400_1800", "R_SG_400_1800", "R_ARPLS_400_1800")
MINIMAL_ACTION = ACTION_IDS[0]

_REASON_CODES = frozenset(
    {
        "invalid_qc_routing_input",
        "invalid_qc_row",
        "invalid_threshold_state",
        "threshold_state_mismatch",
        "invalid_gate_candidate",
        "invalid_action_validity",
        "invalid_minimal_action",
        "scientific_execution_not_authorized",
    }
)

_P02_GATE_CONTRACT = {
    "qc_gate_enumeration": {
        "source_quantiles": [0.5, 0.75, 0.9],
        "priority_orders": ["baseline_then_noise", "noise_then_baseline"],
    }
}

_GATE_MAP: dict[str, dict[str, Any]] | None = None


class QCRoutingError(ValueError):
    """Static, value-free routing failure carried as ``reason_code``."""

    __slots__ = ("reason_code",)

    def __init__(self, reason_code: Any) -> None:
        code = (
            reason_code
            if type(reason_code) is str and reason_code in _REASON_CODES
            else "invalid_qc_routing_input"
        )
        self.reason_code = code
        super().__init__(code)


def require_scientific_execution(*args: Any, **kwargs: Any) -> None:
    raise QCRoutingError("scientific_execution_not_authorized")


def _is_lower64(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _gate_map() -> dict[str, dict[str, Any]]:
    global _GATE_MAP
    if _GATE_MAP is None:
        library = build_qc_gate_library(_P02_GATE_CONTRACT)
        _GATE_MAP = {r["gate_candidate_id"]: r for r in library.to_dict("records")}
    return _GATE_MAP


def _index_of(values: Any, needle: Any) -> int | None:
    for index, value in enumerate(values):
        if value == needle:
            return index
    return None


def _triggered(record: Any, kind: str, features: Any, cutpoints: Any) -> bool:
    feature_name = record.get(f"{kind}_feature")
    quantile = record.get(f"{kind}_quantile")
    if feature_name == "none" or quantile == "none" or features is None:
        return False
    feature_index = _index_of(FEATURES, feature_name)
    quantile_index = _index_of(QUANTILES, quantile)
    if feature_index is None or quantile_index is None:
        raise QCRoutingError("invalid_gate_candidate")
    try:
        cutpoint = cutpoints[feature_index][quantile_index]
    except (TypeError, IndexError, KeyError, ValueError):
        raise QCRoutingError("invalid_threshold_state") from None
    return bool(features[feature_index] > cutpoint)


def _route_qc_row_impl(
    qc_row: Any,
    threshold_state: Any,
    gate_candidate_id: Any,
    action_validity: Any,
    *,
    expected_state_sha256: Any,
) -> dict[str, Any]:
    """Route one row from frozen, pre-authenticated inputs only."""

    if not _is_lower64(expected_state_sha256):
        raise QCRoutingError("invalid_qc_routing_input")

    try:
        state = validate_threshold_state(threshold_state)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise QCRoutingError("invalid_threshold_state") from None
    sealed = state["state_sha256"]
    if sealed != expected_state_sha256:
        raise QCRoutingError("threshold_state_mismatch")

    if type(qc_row) is not np.ndarray or qc_row.shape != (6,):
        raise QCRoutingError("invalid_qc_row")
    if qc_row.dtype.kind not in ("i", "u", "f"):
        raise QCRoutingError("invalid_qc_row")

    try:
        with np.errstate(all="ignore"):
            normalized = np.array(qc_row, dtype="<f8", order="C", copy=True)
            matrix, available = derive_qc_features(normalized.reshape(1, 6))
        qc_available = bool(available[0])
        if qc_available:
            features = [float(value) for value in matrix[0]]
            if len(features) != 5 or not np.isfinite(features).all():
                raise ValueError("non-finite derived features")
        else:
            features = None
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        raise QCRoutingError("invalid_qc_row") from None
    qc_row_sha256 = hashlib.sha256(normalized.tobytes(order="C")).hexdigest()

    if type(action_validity) is not dict:
        raise QCRoutingError("invalid_action_validity")
    if set(action_validity) != set(ACTION_IDS):
        raise QCRoutingError("invalid_action_validity")
    for action_id in ACTION_IDS:
        if type(action_validity[action_id]) is not bool:
            raise QCRoutingError("invalid_action_validity")
    if action_validity[MINIMAL_ACTION] is not True:
        raise QCRoutingError("invalid_minimal_action")

    if type(gate_candidate_id) is not str:
        raise QCRoutingError("invalid_gate_candidate")
    try:
        record = _gate_map().get(gate_candidate_id)
        if record is None:
            raise QCRoutingError("invalid_gate_candidate")
        gate_definition_sha256 = canonical_sha256(record)
    except (KeyboardInterrupt, SystemExit):
        raise
    except QCRoutingError:
        raise
    except Exception:
        raise QCRoutingError("invalid_gate_candidate") from None

    requested = selected = MINIMAL_ACTION
    noise_triggered = baseline_triggered = False
    fallback_reason = "none"
    status = state["status"]
    if status == "no_finite_source_qc":
        fallback_reason = "no_finite_source_qc"
    elif not qc_available:
        fallback_reason = "missing_or_nonfinite_qc"
    elif record.get("gate_kind") != "baseline":
        cutpoints = state["cutpoints"]
        noise_triggered = _triggered(record, "noise", features, cutpoints)
        baseline_triggered = _triggered(record, "baseline", features, cutpoints)
        if noise_triggered and baseline_triggered:
            noise_first = record.get("priority_order") == "noise_then_baseline"
            requested = record["noise_action"] if noise_first else record["baseline_action"]
        elif noise_triggered:
            requested = record["noise_action"]
        elif baseline_triggered:
            requested = record["baseline_action"]
        else:
            requested = record["default_action"]
        if requested not in ACTION_IDS:
            raise QCRoutingError("invalid_gate_candidate")
        if action_validity[requested] is not True:
            fallback_reason = "selected_action_invalid_for_row"
        else:
            selected = requested

    payload = {
        "schema_version": "nato-sers-p08-qc-route-v1",
        "execution_authorized": False,
        "gate_candidate_id": gate_candidate_id,
        "gate_definition_sha256": gate_definition_sha256,
        "threshold_state_sha256": sealed,
        "qc_row_sha256": qc_row_sha256,
        "features": None if features is None else list(features),
        "qc_available": qc_available,
        "noise_triggered": noise_triggered,
        "baseline_triggered": baseline_triggered,
        "requested_action": requested,
        "selected_action": selected,
        "action_validity": {a: bool(action_validity[a]) for a in ACTION_IDS},
        "fallback_reason": fallback_reason,
    }
    payload["route_sha256"] = canonical_sha256(payload)
    return payload


def route_qc_row(
    qc_row: Any,
    threshold_state: Any,
    gate_candidate_id: Any,
    action_validity: Any,
    *,
    expected_state_sha256: Any,
) -> dict[str, Any]:
    """Route one row from frozen, pre-authenticated inputs only."""

    try:
        return _route_qc_row_impl(
            qc_row,
            threshold_state,
            gate_candidate_id,
            action_validity,
            expected_state_sha256=expected_state_sha256,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except QCRoutingError:
        raise
    except Exception:
        raise QCRoutingError("invalid_qc_routing_input") from None
