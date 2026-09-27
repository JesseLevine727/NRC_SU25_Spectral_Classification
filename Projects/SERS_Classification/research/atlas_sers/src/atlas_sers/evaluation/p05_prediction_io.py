"""P05 prediction I/O: endpoint indexing and durable prediction persistence.

This module exposes exactly two helpers.  :func:`index_endpoints` validates a
prediction plan and returns the context-indexed endpoint and canonical spec
layout.  :func:`persist_prediction` validates one calibrated prediction frame
and its audit record, then durably writes ``predictions.csv`` and ``audit.json``
into an already-existing, exclusive unit directory.

The caller owns budget, deadline and unit-manifest concerns.  This module never
fits, augments, calibrates, selects refits, runs inference or loads spectra.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_refit_io as refit_io

EXPECTED_CONTEXTS = 320
EXPECTED_ALIASES = 2880
EXPECTED_CLASS_COUNT = refit_io.EXPECTED_CLASS_COUNT
MAX_PEAK_CUDA_BYTES = 4294967296
PROBABILITY_SUM_TOLERANCE = 1e-12
DIGEST_LENGTH = 64

UID_COLUMN = "observation_uid"
PREDICTIONS_FILENAME = "predictions.csv"
AUDIT_FILENAME = "audit.json"


class P05PredictionIOError(core.P05CoreError):
    """Stable, path-free prediction I/O failure."""


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05PredictionIOError(code)


def _columns() -> list[str]:
    columns = [UID_COLUMN]
    for index in range(EXPECTED_CLASS_COUNT):
        columns.extend((f"logit_{index}", f"probability_{index}"))
    return columns


def _hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == DIGEST_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_spec(spec: Any) -> Mapping[str, Any]:
    canonical = refit_io._check_spec(spec)
    _require(isinstance(canonical, Mapping), "spec_malformed")
    for name in ("context_id", "recipe_id", "refit_id", "seed", "classes"):
        _require(name in canonical, "spec_field_missing")
    return canonical


def _spec_order(spec: Mapping[str, Any]) -> tuple[int, str, str]:
    return (int(spec["seed"]), str(spec["recipe_id"]), str(spec["refit_id"]))


def index_endpoints(plan: Any) -> dict[str, dict[str, Any]]:
    """Validate *plan* and return ``{context_id: {endpoint, specs}}``."""

    _require(isinstance(plan, Mapping), "plan_malformed")
    endpoints = plan.get("endpoints")
    specs = plan.get("unique_refits")
    aliases = plan.get("strategy_aliases")
    _require(
        isinstance(endpoints, Sequence) and not isinstance(endpoints, (str, bytes)),
        "plan_endpoints_malformed",
    )
    _require(
        isinstance(specs, Mapping) and 0 < len(specs) <= EXPECTED_ALIASES, "plan_specs_malformed"
    )
    _require(
        isinstance(aliases, Sequence) and not isinstance(aliases, (str, bytes)),
        "plan_aliases_malformed",
    )
    _require(len(endpoints) == EXPECTED_CONTEXTS, "plan_context_count_mismatch")

    _require(
        all(isinstance(endpoint, Mapping) for endpoint in endpoints), "plan_endpoint_malformed"
    )
    _require(
        all(
            isinstance(endpoint.get("context_id"), str) and endpoint["context_id"]
            for endpoint in endpoints
        ),
        "plan_endpoint_context_mismatch",
    )
    endpoint_map = {endpoint["context_id"]: dict(endpoint) for endpoint in endpoints}
    _require(len(endpoint_map) == len(endpoints), "plan_context_duplicate")
    canonical_specs = []
    for refit_id, raw in specs.items():
        spec = _canonical_spec(raw)
        _require(refit_id == spec["refit_id"], "plan_spec_key_mismatch")
        canonical_specs.append(spec)
    _require(
        {str(spec["context_id"]) for spec in canonical_specs} == set(endpoint_map),
        "plan_context_set_mismatch",
    )

    alias_values = list(aliases)
    _require(len(alias_values) == EXPECTED_ALIASES, "plan_alias_count_mismatch")
    registered = {str(spec["refit_id"]) for spec in canonical_specs}
    referenced: set[str] = set()
    for value in alias_values:
        _require(isinstance(value, Mapping), "plan_alias_malformed")
        refit_id = value.get("refit_id")
        _require(str(refit_id) in registered, "plan_alias_unregistered_refit")
        referenced.add(str(refit_id))
    _require(referenced == registered, "plan_alias_reference_mismatch")

    indexed: dict[str, dict[str, Any]] = {}
    for context_id in sorted(endpoint_map):
        endpoint = endpoint_map[context_id]
        _require(isinstance(endpoint, Mapping), "plan_endpoint_malformed")
        _require(
            endpoint.get("context_id") == context_id,
            "plan_endpoint_context_mismatch",
        )
        context_specs = sorted(
            (spec for spec in canonical_specs if str(spec["context_id"]) == context_id),
            key=_spec_order,
        )
        _require(bool(context_specs), "plan_context_without_specs")
        indexed[context_id] = {"endpoint": dict(endpoint), "specs": context_specs}
    return indexed


def _validate_endpoint_uids(endpoint: Mapping[str, Any]) -> list[str]:
    test_uids = endpoint.get("test_uids")
    _require(
        isinstance(test_uids, Sequence) and not isinstance(test_uids, (str, bytes)),
        "endpoint_test_uids_malformed",
    )
    _require(all(isinstance(uid, str) for uid in test_uids), "endpoint_test_uids_malformed")
    uids = list(test_uids)
    _require(bool(uids), "endpoint_test_uids_empty")
    _require(all(uid and uid == uid.strip() for uid in uids), "endpoint_test_uids_malformed")
    _require(len(set(uids)) == len(uids), "endpoint_test_uids_duplicate")
    _require(uids == sorted(uids), "endpoint_test_uids_unsorted")
    return uids


def _validate_frame(frame: Any, test_uids: list[str]) -> None:
    _require(isinstance(frame, pd.DataFrame), "frame_malformed")
    _require(list(frame.columns) == _columns(), "frame_columns_mismatch")
    _require(len(frame) == len(test_uids), "frame_row_count_mismatch")
    _require(frame[UID_COLUMN].tolist() == test_uids, "frame_uids_mismatch")
    probabilities = np.empty((len(frame), EXPECTED_CLASS_COUNT), dtype=np.float64)
    for index in range(EXPECTED_CLASS_COUNT):
        logits = frame[f"logit_{index}"].to_numpy(dtype=np.float64)
        _require(bool(np.isfinite(logits).all()), "frame_logits_nonfinite")
        values = frame[f"probability_{index}"].to_numpy(dtype=np.float64)
        _require(bool(np.isfinite(values).all()), "frame_probabilities_nonfinite")
        _require(bool(((values >= 0.0) & (values <= 1.0)).all()), "frame_probabilities_range")
        probabilities[:, index] = values
    _require(
        bool(np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=PROBABILITY_SUM_TOLERANCE)),
        "frame_probabilities_sum",
    )


def _validate_audit(
    audit: Any, spec: Mapping[str, Any], test_uids: list[str], rows: int
) -> Mapping[str, Any]:
    _require(isinstance(audit, Mapping), "audit_malformed")
    _require(audit.get("refit_id") == spec["refit_id"], "audit_refit_id_mismatch")
    _require(list(audit.get("classes", ())) == list(spec["classes"]), "audit_classes_mismatch")
    _require(
        isinstance(audit.get("rows"), int)
        and not isinstance(audit["rows"], bool)
        and audit["rows"] == rows,
        "audit_rows_mismatch",
    )
    _require(
        audit.get("test_uid_set_sha256") == core._canon().sha256_value(sorted(test_uids)),
        "audit_uid_hash_mismatch",
    )
    _require(
        type(audit.get("optimizer_steps")) is int and audit["optimizer_steps"] == 0,
        "audit_optimizer_steps_invalid",
    )
    elapsed = audit.get("elapsed_seconds")
    _require(
        isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed))
        and float(elapsed) >= 0.0,
        "audit_elapsed_invalid",
    )
    peak = audit.get("peak_cuda_bytes")
    _require(
        isinstance(peak, int) and not isinstance(peak, bool) and 0 <= peak <= MAX_PEAK_CUDA_BYTES,
        "audit_peak_cuda_invalid",
    )
    for name in ("model_state_sha256", "calibration_state_sha256"):
        _require(_hex_digest(audit.get(name)), "audit_digest_invalid")
    return audit


def _write_predictions(path: Path, frame: pd.DataFrame) -> int:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    core._atomic_write(path, payload)
    return len(payload)


def _reload_predictions(path: Path, test_uids: list[str], frame: pd.DataFrame) -> None:
    reloaded = pd.read_csv(
        path,
        dtype={UID_COLUMN: str},
        keep_default_na=False,
        float_precision="round_trip",
    )
    _require(list(reloaded.columns) == _columns(), "predictions_columns_mismatch")
    _require(len(reloaded) == len(frame), "predictions_rows_mismatch")
    _require(
        [str(uid) for uid in reloaded[UID_COLUMN].tolist()] == test_uids,
        "predictions_uids_mismatch",
    )
    for index in range(EXPECTED_CLASS_COUNT):
        for name in (f"logit_{index}", f"probability_{index}"):
            original = frame[name].to_numpy(dtype=np.float64)
            stored = reloaded[name].to_numpy(dtype=np.float64)
            _require(bool(np.array_equal(original, stored)), "predictions_values_mismatch")


def _write_audit(path: Path, audit: Mapping[str, Any]) -> int:
    payload = core._canon().canonical_json_bytes(audit)
    core._atomic_write(path, payload)
    return len(payload)


def _reload_audit(path: Path, payload: bytes) -> None:
    reloaded = core._read_json(path, "prediction_audit")
    _require(isinstance(reloaded, Mapping), "prediction_audit_reload_mismatch")
    _require(
        core._canon().canonical_json_bytes(reloaded) == payload,
        "prediction_audit_reload_mismatch",
    )


def persist_prediction(
    unit_dir: Any,
    spec: Any,
    endpoint: Any,
    frame: Any,
    audit: Any,
) -> dict[str, Any]:
    """Validate and durably write one prediction frame and its audit."""

    canonical = _canonical_spec(spec)
    _require(isinstance(endpoint, Mapping), "endpoint_malformed")
    _require(endpoint.get("context_id") == canonical["context_id"], "endpoint_context_mismatch")
    test_uids = _validate_endpoint_uids(endpoint)
    _require(not set(test_uids).intersection(canonical["fitting_uids"]), "source_test_uid_overlap")
    _validate_frame(frame, test_uids)
    _validate_audit(audit, canonical, test_uids, len(frame))

    unit = Path(unit_dir)
    core._reject_symlink_chain(unit)
    _require(not unit.is_symlink() and unit.is_dir(), "unit_directory_missing")
    predictions_path = unit / PREDICTIONS_FILENAME
    audit_path = unit / AUDIT_FILENAME
    for path in (predictions_path, audit_path):
        core._reject_symlink_chain(path)
        _require(not path.exists() and not path.is_symlink(), "prediction_output_exists")

    audit_payload = core._canon().canonical_json_bytes(audit)
    predictions_bytes = _write_predictions(predictions_path, frame)
    audit_bytes = _write_audit(audit_path, audit)
    _reload_predictions(predictions_path, test_uids, frame)
    _reload_audit(audit_path, audit_payload)

    return {
        "row_count": int(len(frame)),
        "peak_cuda_bytes": int(audit["peak_cuda_bytes"]),
        "elapsed_seconds": float(audit["elapsed_seconds"]),
        "predictions_bytes": predictions_bytes,
        "audit_bytes": audit_bytes,
    }


__all__ = ["P05PredictionIOError", "index_endpoints", "persist_prediction"]
