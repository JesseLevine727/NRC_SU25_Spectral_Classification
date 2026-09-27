"""P05 outer input preparation.

Authenticates the pinned context registry, verifies one frozen outer-test
endpoint against the authoritative support contexts, roles and manifest, and
exposes only the outer-test representation rows requested by that endpoint.
This module prepares inputs only: it never trains, predicts, calibrates,
augments, perturbs or writes any artifact, and none of its helpers carry
execution authority.
"""

from __future__ import annotations

import csv
import importlib
import io
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_refit_io as refit_io
from atlas_sers.evaluation.p05_core_run import P05CoreError

OUTER_FIT_ROLE = refit_io.OUTER_FIT_ROLE
OUTER_TEST_ROLE = refit_io.OUTER_TEST_ROLE
HELD_INSTRUMENT_SENTINELS = refit_io.HELD_INSTRUMENT_SENTINELS
EXPECTED_CLASS_COUNT = refit_io.EXPECTED_CLASS_COUNT
CONTEXT_REGISTRY_DIR = Path("p04plan") / "runs"
CONTEXT_REGISTRY_NAME = "context_registry.csv"
CONTEXT_REGISTRY_PIN = "contexts_sha256"
REPRESENTATION_REL = Path("representations") / "R_MIN_400_1800.npz"
REQUIRED_CONTEXT_FIELDS = (
    "context_id",
    "experiment_id",
    "phase_gate",
    "outer_repeat",
    "outer_fold",
    "station",
    "domain",
    "held_instrument",
    "partition_id",
)
PHASE_GATES = frozenset({"development", "held_evaluation"})
CANONICAL_INT = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")


class P05OuterInputsError(P05CoreError):
    """Stable outer-input failure carrying a path-free reason code."""


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05OuterInputsError(code)


def _module(name: str) -> Any:
    return importlib.import_module(name)


def _canonical_int(value: Any, code: str) -> int:
    _require(
        isinstance(value, str)
        and value == value.strip()
        and CANONICAL_INT.match(value) is not None,
        code,
    )
    number = int(value)
    _require(str(number) == value, code)
    return number


def _registry_path(bundle: Mapping[str, Any]) -> Path:
    contract = bundle.get("contract")
    artifact_root = bundle.get("artifact_root")
    _require(isinstance(contract, Mapping) and artifact_root is not None, "bundle_incomplete")
    run_id = contract["input_pins"]["p04plan_run_id"]
    _require(isinstance(run_id, str) and bool(run_id), "contract_pin_malformed")
    return Path(artifact_root) / CONTEXT_REGISTRY_DIR / run_id / CONTEXT_REGISTRY_NAME


def _registry_rows(bundle: Mapping[str, Any]) -> list[dict[str, str]]:
    contract = bundle["contract"]
    raw = core._read_bytes(_registry_path(bundle), "context_registry")
    _require(
        core._canon().sha256_bytes(raw) == contract["input_pins"][CONTEXT_REGISTRY_PIN],
        "context_registry_digest_mismatch",
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise P05OuterInputsError("context_registry_not_utf8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = reader.fieldnames
    _require(
        header is not None and bool(header) and len(header) == len(set(header)),
        "context_registry_header_invalid",
    )
    rows: list[dict[str, str]] = []
    for record in reader:
        _require(
            None not in record and all(isinstance(value, str) for value in record.values()),
            "context_registry_row_malformed",
        )
        rows.append(dict(record))
    _require(bool(rows), "context_registry_empty")
    return rows


def load_context_rows(bundle: Any) -> list[dict[str, str]]:
    """Authenticate and return the pinned context registry rows."""

    _require(isinstance(bundle, Mapping), "bundle_malformed")
    support = bundle.get("support")
    _require(support is not None, "bundle_incomplete")
    rows = _registry_rows(bundle)
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        for field in REQUIRED_CONTEXT_FIELDS:
            _require(field in row, "context_registry_field_missing")
        context_id = row["context_id"]
        _require(bool(context_id) and context_id == context_id.strip(), "context_id_malformed")
        _require(context_id not in by_id, "context_registry_context_duplicate")
        by_id[context_id] = row
    support_contexts = list(getattr(support, "contexts", None) or [])
    _require(len(support_contexts) == len(rows), "context_registry_context_mismatch")
    seen: set[str] = set()
    for context in support_contexts:
        context_id = str(context.get("context_id", ""))
        _require(
            context_id in by_id and context_id not in seen, "context_registry_context_mismatch"
        )
        seen.add(context_id)
        source = by_id[context_id]
        for field, actual in context.items():
            _require(
                field in source and str(actual) == source[field],
                "context_registry_field_mismatch",
            )
        _require(source["phase_gate"] in PHASE_GATES, "context_phase_gate_unknown")
        _canonical_int(source["outer_repeat"], "context_outer_repeat_malformed")
        _canonical_int(source["outer_fold"], "context_outer_fold_malformed")
    return rows


def _endpoint_context(support: Any, endpoint: Mapping[str, Any]) -> Any:
    context_id = endpoint.get("context_id")
    _require(isinstance(context_id, str) and bool(context_id), "endpoint_context_malformed")
    return refit_io._context_row(support, context_id)


def _registered_roles(
    support: Any, context_id: str, role_name: str, manifest: Mapping[str, Mapping[str, str]]
) -> list[Any]:
    role_id = refit_io._role_id_for(support, context_id, role_name)
    rows = refit_io._role_rows(support, context_id, role_id=role_id)
    _require({str(row.get("role")) for row in rows} == {role_name}, "outer_role_mismatch")
    refit_io._check_role_manifest(rows, manifest)
    return rows


def prepare_outer_inputs(bundle: Any, endpoint: Any) -> dict[str, Any]:
    """Expose only the outer-test representation rows for one frozen endpoint."""

    _require(isinstance(bundle, Mapping), "bundle_malformed")
    support = bundle.get("support")
    contract = bundle.get("contract")
    p01_run = bundle.get("p01_path")
    _require(
        support is not None and isinstance(contract, Mapping) and p01_run is not None,
        "bundle_incomplete",
    )
    _require(isinstance(endpoint, Mapping), "endpoint_malformed")
    manifest = _module("atlas_sers.evaluation.p05_pilot")._manifest_rows(support)
    context = _endpoint_context(support, endpoint)
    context_id = str(context["context_id"])
    for field, value in context.items():
        if field in endpoint:
            _require(str(endpoint[field]) == str(value), "endpoint_context_mismatch")
    _require(
        endpoint.get("outer_test_role_id")
        == refit_io._role_id_for(support, context_id, OUTER_TEST_ROLE),
        "outer_test_role_mismatch",
    )
    test_rows = _registered_roles(support, context_id, OUTER_TEST_ROLE, manifest)
    test_uids = list(endpoint.get("test_uids", ()))
    _require(test_uids == refit_io._sorted_uids(test_rows), "outer_test_uids_mismatch")
    test_masters = sorted(
        {refit_io._text(row.get("master_sample_id"), "role_master_malformed") for row in test_rows}
    )
    _require(list(endpoint.get("test_masters", ())) == test_masters, "outer_test_masters_mismatch")
    test_classes = sorted(
        {refit_io._text(row.get("target_analyte"), "role_target_malformed") for row in test_rows}
    )
    _require(1 <= len(test_classes) <= EXPECTED_CLASS_COUNT, "outer_test_class_count_mismatch")
    _require(list(endpoint.get("test_classes", ())) == test_classes, "outer_test_classes_mismatch")
    fit_rows = _registered_roles(support, context_id, OUTER_FIT_ROLE, manifest)
    fit_uids = refit_io._sorted_uids(fit_rows)
    fit_masters = {
        refit_io._text(row.get("master_sample_id"), "role_master_malformed") for row in fit_rows
    }
    source_classes = sorted(
        {refit_io._text(row.get("target_analyte"), "role_target_malformed") for row in fit_rows}
    )
    _require(len(source_classes) == EXPECTED_CLASS_COUNT, "source_class_count_mismatch")
    _require(set(test_classes) <= set(source_classes), "outer_test_class_unknown")
    _require(not (set(test_uids) & set(fit_uids)), "outer_test_uid_overlap")
    _require(not (set(test_masters) & fit_masters), "outer_test_master_overlap")
    _require(
        {manifest[uid]["station"] for uid in [*fit_uids, *test_uids]} == {str(context["station"])},
        "outer_station_mismatch",
    )
    held_instrument = str(context.get("held_instrument", ""))
    if held_instrument not in HELD_INSTRUMENT_SENTINELS:
        test_instruments = {manifest[uid]["instrument"] for uid in test_uids}
        fit_instruments = {manifest[uid]["instrument"] for uid in fit_uids}
        _require(test_instruments == {held_instrument}, "held_instrument_test_mismatch")
        _require(held_instrument not in fit_instruments, "held_instrument_in_source")
    expected_rows = int(contract["population"]["rows"])
    expected_features = int(contract["population"]["features"])
    intensity, labels = core._load_representation(
        Path(p01_run) / REPRESENTATION_REL,
        contract["input_pins"]["representation_sha256"],
        core._manifest_uids(support, expected_rows),
        expected_rows,
    )
    uid_index = {str(uid): index for index, uid in enumerate(labels)}
    _require(len(uid_index) == len(labels), "representation_uid_duplicate")
    try:
        indices = [uid_index[uid] for uid in test_uids]
    except KeyError as error:
        raise P05OuterInputsError("outer_uid_missing_representation") from error
    values = intensity[indices].astype("float32")
    _require(values.ndim == 2 and values.shape[0] == len(test_uids), "outer_row_count_mismatch")
    _require(values.shape[1] == expected_features, "outer_feature_count_mismatch")
    return {
        "values": values,
        "observation_uids": tuple(test_uids),
        "classes": tuple(source_classes),
    }


__all__ = ["P05OuterInputsError", "load_context_rows", "prepare_outer_inputs"]
