"""T310 read-only final universal evidence loader (private analysis draft).

This module aggregates already-authenticated P08-U1 governance inputs into an
in-memory evidence bundle for a later reviewed runner: the frozen graph
archive, the sealed MIN operation bridge, the package-pinned review frames and
a completed/closed SG/arPLS ledger run.  Every read is at most a single-link
regular file inside an explicit existing allowed root and every refusal is a
path-free fixed reason code.

It never writes, fits, scores, calibrates, resamples, averages, clips,
renormalizes, draws or deserializes estimator/checkpoint pickles.  Endpoint
probability files are retained only as untouched frames.  The pure
``assemble_panel`` consistency adapter is owned by a later reviewed runner and
is deliberately not implemented or copied here.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import sqlite3
import stat
import urllib.parse
from pathlib import Path

import pandas as pd

from atlas_sers.evaluation.p06p11_inputs import PINS
from atlas_sers.evaluation.p08_u1_artifacts import ArtifactError, ArtifactStore
from atlas_sers.evaluation.p08_u1_standing_accounting import (
    KNOWN_STAGES,
    MODEL_FIT_STAGES,
    OPERATION_STAGES,
    PREDICTION_STAGES,
    SCALAR_STAGE,
    SELECTOR_STAGES,
    STANDING_ACCOUNTING_SCHEMA,
    classify_stage_kind,
)
from atlas_sers.evaluation.p08_u1_store import (
    MAX_SCALAR_ATTEMPTS,
    MAX_UNIQUE_FIT_JOBS,
    R2_ACCOUNTING_SCHEMA,
    R2_ADDITIONAL_FIT_COUNT,
    R2_EXPECTED_REUSE_PAIRS,
    R2_HISTORICAL_OVERHEAD_ATTEMPTS,
    R2_MAX_FIT_TOTAL,
    R2_REPLAY_FIT_COUNT,
    R2_SELECTOR_COUNT,
    _canonical,
    _meta_get,
    _sha256_json,
    _verify_events,
    _verify_registered_jobs,
    execution_accounting,
    job_sha256,
)
from atlas_sers.splits.p02 import instrument_family

__all__ = [
    "ENDPOINT_STATUS",
    "GRAPH_PLAN_SHA256",
    "GRAPH_SHA256",
    "MIN_BRIDGE_SCHEMA",
    "MIN_BRIDGE_SHA256",
    "SCHEMA_VERSION",
    "UniversalEvidenceError",
    "apply_selector",
    "audit_operation_coverage",
    "load_evidence",
    "normalize_operation_binding",
    "read_endpoint_frame",
    "validate_pointer",
    "validate_selector",
]

SCHEMA_VERSION = "nato-sers-p08-universal-evidence-v1"

# Frozen archive pins.  Monkeypatchable ONLY for small synthetic tests.
GRAPH_SHA256 = "c636358211edf350e2d36e4de40531a27e6402c937d191a4ebdeba83f1a086e8"
GRAPH_PLAN_SHA256 = "179b95e8011a5f6cc02c65c7fab1acf0f6a6241ba1ef02378aac207b9e19cb03"
MIN_BRIDGE_SHA256 = "7954181851586f24e648ddda98d2ba3380bdabfb50ec767ebb21a9ae07c5e74e"

MIN_BRIDGE_SCHEMA = "nato-sers-p08-minimum-operation-evidence-v1"
ENDPOINT_STATUS = "complete_saved_pipeline_endpoint"

# Trusted identity of the single reviewed permit.  Monkeypatchable only for
# small synthetic tests; production comparisons always use this pinned digest.
REVIEWED_PERMIT_SHA256 = (
    "9540bfcaa9a4ca904483e4fc05cb77812979bf981aa1ea4e7cc0f5caad4fa894"
)

# Frozen structural counts.
EXPECTED_GRAPH_JOBS = 607221
EXPECTED_MIN_JOBS = 202407
EXPECTED_SG_ARPLS_JOBS = 404814
EXPECTED_ALIASES = 1560
EXPECTED_MIN_OPERATION_BINDINGS = 202407
EXPECTED_VERIFIED_FILE_HASHES = 20934
EXPECTED_MIN_ENDPOINTS = 1079
EXPECTED_NEW_ENDPOINTS = 2158
EXPECTED_DISTINCT_ENDPOINTS = 3237
EXPECTED_REPORT_CELLS = 3900

# Approved R2 accounting profile counters.
R2_NEW_FITS = 186652
R2_NEW_CAL = 3354
R2_NEW_OPERATIONS = 387713
R2_REUSED_FITS = 8550
R2_REUSED_PREDICTIONS = 8550
R2_REUSED_EPOCH = 1

_NEURAL_RECIPES = ("D0-M", "D1", "D2", "D3")
_D0_RECIPE = "D0-M"
_SELECTED_RECIPE = "P05-SELECTED"
_HEX = frozenset("0123456789abcdef")
_RUN_FILES = ("permit.json", "binding.json", "completion.json", "close.json")
_CHUNK = 1 << 20


class UniversalEvidenceError(RuntimeError):
    """Fail-closed, path-free evidence-loading refusal."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _is_hex(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def _reject_constant(_value):
    raise ValueError("nonfinite_number")


def _reject_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("nonfinite_number")
    return value


def _reject_duplicate_keys(pairs):
    seen = set()
    mapping = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError("duplicate_key")
        seen.add(key)
        mapping[key] = value
    return mapping


def _strict_loads(text):
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_reject_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (TypeError, ValueError) as exc:
        raise UniversalEvidenceError("json_invalid") from exc


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _reject_symlink_ancestors(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise UniversalEvidenceError("allowed_root_symlink")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _prepare_roots(allowed_evidence_roots):
    if not isinstance(allowed_evidence_roots, (list, tuple)) or isinstance(
        allowed_evidence_roots, (str, bytes)
    ):
        raise UniversalEvidenceError("allowed_roots_invalid")
    roots = []
    for raw in allowed_evidence_roots:
        try:
            candidate = Path(os.path.abspath(os.fspath(raw)))
        except TypeError as exc:
            raise UniversalEvidenceError("allowed_root_invalid") from exc
        _reject_symlink_ancestors(candidate)
        if not candidate.is_dir():
            raise UniversalEvidenceError("allowed_root_invalid")
        real = candidate.resolve(strict=True)
        if real not in roots:
            roots.append(real)
    if not roots:
        raise UniversalEvidenceError("allowed_roots_empty")
    return tuple(roots)


def _matching_root(path: Path, roots):
    for root in roots:
        if _is_under(path, root):
            return root
    return None


def _reject_symlinks_below(path: Path, root: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise UniversalEvidenceError("symlink_rejected")
        if current == root:
            return
        parent = current.parent
        if parent == current:
            return
        current = parent


def _secure_path(path, roots, *, directory: bool) -> Path:
    try:
        candidate = Path(os.path.abspath(os.fspath(path)))
    except TypeError as exc:
        raise UniversalEvidenceError("path_invalid") from exc
    root = _matching_root(candidate, roots)
    if root is None:
        raise UniversalEvidenceError("path_escape")
    _reject_symlinks_below(candidate, root)
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise UniversalEvidenceError("path_missing") from exc
    if stat.S_ISLNK(info.st_mode):
        raise UniversalEvidenceError("symlink_rejected")
    if directory:
        if not stat.S_ISDIR(info.st_mode):
            raise UniversalEvidenceError("not_a_directory")
    else:
        if not stat.S_ISREG(info.st_mode):
            raise UniversalEvidenceError("not_regular_file")
        if info.st_nlink != 1:
            raise UniversalEvidenceError("hardlink_rejected")
    real = candidate.resolve(strict=True)
    if not _is_under(real, root):
        raise UniversalEvidenceError("path_escape")
    return real


def _secure_file(path, roots) -> Path:
    return _secure_path(path, roots, directory=False)


def _secure_dir(path, roots) -> Path:
    return _secure_path(path, roots, directory=True)


def _secure_relative(root: Path, rel: str, roots) -> Path:
    if not isinstance(rel, str) or rel.startswith("/") or "\\" in rel:
        raise UniversalEvidenceError("unsafe_relative_path")
    parts = rel.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise UniversalEvidenceError("unsafe_relative_path")
    return _secure_file(root.joinpath(*parts), roots)


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise UniversalEvidenceError("file_open_failed") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise UniversalEvidenceError("not_regular_file")
        if info.st_nlink != 1:
            raise UniversalEvidenceError("hardlink_rejected")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
    finally:
        if fd != -1:
            os.close(fd)
    return digest.hexdigest()


def _authenticated_file(path, roots, expected_sha256) -> Path:
    real = _secure_file(path, roots)
    if _stream_sha256(real) != expected_sha256:
        raise UniversalEvidenceError("hash_mismatch")
    return real


def _load_gzip_json(path: Path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            text = handle.read()
    except (OSError, EOFError) as exc:
        raise UniversalEvidenceError("gzip_read_failed") from exc
    return _strict_loads(text)


def _read_json_file(path, roots):
    real = _secure_file(path, roots)
    try:
        with open(real, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise UniversalEvidenceError("file_read_failed") from exc
    return _strict_loads(text)


def _graph_job_id(raw) -> str:
    job_id = raw.get("job_id")
    if (
        not isinstance(job_id, str)
        or len(job_id) != 71
        or not job_id.startswith("P08JOB-")
    ):
        raise UniversalEvidenceError("graph_job_id_invalid")
    if job_id != "P08JOB-" + job_sha256(raw):
        raise UniversalEvidenceError("graph_job_hash_mismatch")
    return job_id


def _authenticate_graph(graph):
    if not isinstance(graph, dict):
        raise UniversalEvidenceError("graph_invalid")
    if graph.get("plan_sha256") != GRAPH_PLAN_SHA256:
        raise UniversalEvidenceError("graph_plan_mismatch")
    jobs = graph.get("jobs")
    aliases = graph.get("aliases")
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_GRAPH_JOBS:
        raise UniversalEvidenceError("graph_jobs_count_mismatch")
    if not isinstance(aliases, (list, dict)) or len(aliases) != EXPECTED_ALIASES:
        raise UniversalEvidenceError("graph_aliases_count_mismatch")
    by_id = {}
    min_ids = set()
    new_ids = set()
    for raw in jobs:
        if not isinstance(raw, dict):
            raise UniversalEvidenceError("graph_job_invalid")
        job_id = _graph_job_id(raw)
        if job_id in by_id:
            raise UniversalEvidenceError("graph_job_duplicate")
        policy = raw.get("policy_id")
        if policy == "PP-U-MIN":
            min_ids.add(job_id)
        elif policy in ("PP-U-SG", "PP-U-ARPLS"):
            new_ids.add(job_id)
        else:
            raise UniversalEvidenceError("graph_policy_invalid")
        by_id[job_id] = raw
    if set(by_id) != min_ids | new_ids:
        raise UniversalEvidenceError("graph_partition_mismatch")
    for raw in by_id.values():
        dependencies = raw.get("dependencies")
        if not isinstance(dependencies, list):
            raise UniversalEvidenceError("graph_job_dependencies_invalid")
        policy = raw.get("policy_id")
        for dep in dependencies:
            dep_record = by_id.get(dep)
            if dep_record is None:
                raise UniversalEvidenceError("graph_job_dependency_missing")
            if dep_record.get("policy_id") != policy:
                raise UniversalEvidenceError("graph_job_dependency_policy_mismatch")
    return by_id, aliases, jobs, min_ids, new_ids


def validate_pointer(pointer):
    if not isinstance(pointer, dict) or set(pointer) != {"file", "sha256", "selector"}:
        raise UniversalEvidenceError("pointer_invalid")
    if not isinstance(pointer["file"], str) or not pointer["file"]:
        raise UniversalEvidenceError("pointer_invalid")
    if not _is_hex(pointer["sha256"]):
        raise UniversalEvidenceError("pointer_invalid")
    selector = pointer["selector"]
    if not isinstance(selector, dict) or not selector:
        raise UniversalEvidenceError("selector_invalid")
    return pointer


def validate_selector(selector, *, model_id, context_id):
    """Return ``(kind, value)`` for an exact MIN/run endpoint selector.

    Classical selectors return ``("classical", outer_run_id)``.  Neural
    selectors return ``("neural", (context_id, strategy))`` where ``strategy``
    is ``D0-M`` for the D0 recipe and ``P05-SELECTED`` for every other recipe.
    """
    if not isinstance(selector, dict) or not selector:
        raise UniversalEvidenceError("selector_invalid")
    if set(selector) == {"outer_run_id"}:
        if model_id in _NEURAL_RECIPES:
            raise UniversalEvidenceError("selector_recipe_mismatch")
        value = selector["outer_run_id"]
        if not isinstance(value, str) or not value:
            raise UniversalEvidenceError("selector_invalid")
        return "classical", value
    if set(selector) == {"context_id", "model_id"}:
        if model_id not in _NEURAL_RECIPES:
            raise UniversalEvidenceError("selector_recipe_mismatch")
        strategy = selector["model_id"]
        if not isinstance(strategy, str):
            raise UniversalEvidenceError("selector_invalid")
        expected_strategy = _D0_RECIPE if model_id == _D0_RECIPE else _SELECTED_RECIPE
        if strategy != expected_strategy:
            raise UniversalEvidenceError("selector_recipe_mismatch")
        selected_context = selector["context_id"]
        if not isinstance(selected_context, str) or not selected_context:
            raise UniversalEvidenceError("selector_invalid")
        if context_id is not None and selected_context != context_id:
            raise UniversalEvidenceError("selector_context_mismatch")
        return "neural", (selected_context, strategy)
    raise UniversalEvidenceError("selector_unexpected")


def apply_selector(frame, *, kind, value):
    if kind == "classical":
        if "outer_run_id" not in getattr(frame, "columns", ()):
            raise UniversalEvidenceError("selector_column_missing")
        subset = frame[frame["outer_run_id"] == value]
    else:
        selected_context, strategy = value
        for column in ("context_id", "model_id"):
            if column not in getattr(frame, "columns", ()):
                raise UniversalEvidenceError("selector_column_missing")
        subset = frame[
            (frame["context_id"] == selected_context) & (frame["model_id"] == strategy)
        ]
    if subset.empty:
        raise UniversalEvidenceError("selector_no_match")
    return subset.reset_index(drop=True)


def normalize_operation_binding(binding, record, verified):
    """Validate one MIN operation binding against graph+bridge identity."""
    if not isinstance(binding, dict) or not isinstance(record, dict):
        raise UniversalEvidenceError("min_binding_invalid")
    digest = binding.get("binding_sha256")
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if not _is_hex(digest) or _sha256_json(body) != digest:
        raise UniversalEvidenceError("min_binding_sha_mismatch")
    if binding.get("job_id") != record.get("job_id"):
        raise UniversalEvidenceError("min_binding_job_mismatch")
    for field in ("context_id", "model_id", "stage", "seed"):
        if binding.get(field) != record.get(field):
            raise UniversalEvidenceError("min_binding_" + field + "_mismatch")
    evidence = binding.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise UniversalEvidenceError("min_evidence_empty")
    for item in evidence:
        if not isinstance(item, dict):
            raise UniversalEvidenceError("min_evidence_invalid")
        file_path = item.get("file")
        file_sha = item.get("sha256")
        if not isinstance(file_path, str) or not file_path:
            raise UniversalEvidenceError("min_evidence_invalid")
        if not _is_hex(file_sha):
            raise UniversalEvidenceError("min_evidence_invalid")
        if verified.get(file_path) != file_sha:
            raise UniversalEvidenceError("min_evidence_unlisted")
    return binding


def audit_operation_coverage(expected, attempts, reuse) -> None:
    expected = set(expected)
    attempts = set(attempts)
    reuse = set(reuse)
    if attempts & reuse:
        raise UniversalEvidenceError("operation_overlap")
    if attempts | reuse != expected:
        raise UniversalEvidenceError("operation_coverage")


def authenticate_operation_binding(binding, record, verified):
    if not isinstance(binding, dict):
        raise UniversalEvidenceError("min_binding_invalid")
    return normalize_operation_binding(binding, record, verified)


def _authenticate_min_bridge(bridge, graph_by_id):
    if not isinstance(bridge, dict):
        raise UniversalEvidenceError("min_bridge_invalid")
    if bridge.get("schema_version") != MIN_BRIDGE_SCHEMA:
        raise UniversalEvidenceError("min_bridge_schema_mismatch")
    if bridge.get("parent_plan_sha256") != GRAPH_PLAN_SHA256:
        raise UniversalEvidenceError("min_bridge_plan_mismatch")
    bindings = bridge.get("operation_bindings")
    verified = bridge.get("verified_file_hashes")
    if (
        not isinstance(bindings, list)
        or len(bindings) != EXPECTED_MIN_OPERATION_BINDINGS
    ):
        raise UniversalEvidenceError("min_bindings_count_mismatch")
    if not isinstance(verified, dict) or len(verified) != EXPECTED_VERIFIED_FILE_HASHES:
        raise UniversalEvidenceError("min_verified_count_mismatch")
    min_ids = set()
    endpoints = []
    for binding in bindings:
        if not isinstance(binding, dict):
            raise UniversalEvidenceError("min_binding_invalid")
        job_id = binding.get("job_id")
        record = graph_by_id.get(job_id)
        if record is None:
            raise UniversalEvidenceError("min_binding_job_missing")
        if record.get("policy_id") != "PP-U-MIN":
            raise UniversalEvidenceError("min_binding_policy_mismatch")
        normalize_operation_binding(binding, record, verified)
        if job_id in min_ids:
            raise UniversalEvidenceError("min_binding_duplicate")
        min_ids.add(job_id)
        if (
            binding.get("stage") == "seed_ensemble_prediction"
            and binding.get("evidence_status") == ENDPOINT_STATUS
        ):
            evidence = binding.get("evidence")
            if not isinstance(evidence, list) or len(evidence) != 1:
                raise UniversalEvidenceError("min_endpoint_pointer_count")
            endpoints.append((job_id, validate_pointer(evidence[0]), record))
    return min_ids, verified, endpoints


def _verify_listed_files(verified, roots):
    if len(verified) != EXPECTED_VERIFIED_FILE_HASHES:
        raise UniversalEvidenceError("min_verified_count_mismatch")
    for raw_path, expected in verified.items():
        if not isinstance(raw_path, str) or not _is_hex(expected):
            raise UniversalEvidenceError("min_verified_entry_invalid")
        real = _secure_file(raw_path, roots)
        if _stream_sha256(real) != expected:
            raise UniversalEvidenceError("hash_mismatch")


def _require_completed_files(run_root: Path) -> None:
    for name in _RUN_FILES:
        candidate = run_root / name
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise UniversalEvidenceError("run_incomplete") from exc
        if not stat.S_ISREG(info.st_mode):
            raise UniversalEvidenceError("run_incomplete")


def _authenticate_permit(permit, binding, run_root) -> None:
    if not isinstance(permit, dict) or not isinstance(binding, dict):
        raise UniversalEvidenceError("permit_invalid")
    digest = permit.get("permit_sha256")
    body = {key: value for key, value in permit.items() if key != "permit_sha256"}
    if not _is_hex(digest) or _sha256_json(body) != digest:
        raise UniversalEvidenceError("permit_sha256_mismatch")
    if digest != REVIEWED_PERMIT_SHA256:
        raise UniversalEvidenceError("permit_not_reviewed")
    if permit.get("reviewed") is not True:
        raise UniversalEvidenceError("permit_not_reviewed")
    plan = permit.get("graph_plan_sha256", permit.get("plan_sha256"))
    archive = permit.get("graph_archive_sha256", permit.get("graph_sha256"))
    if plan != GRAPH_PLAN_SHA256:
        raise UniversalEvidenceError("permit_plan_mismatch")
    if archive != GRAPH_SHA256:
        raise UniversalEvidenceError("permit_archive_mismatch")
    if binding.get("permit_sha256") != digest:
        raise UniversalEvidenceError("binding_permit_mismatch")
    if binding.get("graph_plan_sha256") != GRAPH_PLAN_SHA256:
        raise UniversalEvidenceError("binding_graph_plan_mismatch")
    destination = str(run_root)
    if (
        permit.get("destination") != destination
        or binding.get("destination") != destination
    ):
        raise UniversalEvidenceError("destination_mismatch")
    if binding.get("recovery_accounting") != permit.get("recovery_accounting"):
        raise UniversalEvidenceError("binding_recovery_accounting_mismatch")


def _binding_is_standing(binding):
    if not isinstance(binding, dict):
        return False
    recovery = binding.get("recovery_accounting")
    return (
        isinstance(recovery, dict)
        and recovery.get("schema_version") == STANDING_ACCOUNTING_SCHEMA
    )


def _is_standing_accounting(accounting):
    return (
        isinstance(accounting, dict)
        and accounting.get("schema_version") == STANDING_ACCOUNTING_SCHEMA
    )


def _derive_standing_reuse_sets(accounting):
    completed = accounting["completed_job_ids_by_stage"]
    fit_ids = tuple(sorted(j for stage in MODEL_FIT_STAGES for j in completed[stage]))
    prediction_ids = tuple(
        sorted(j for stage in PREDICTION_STAGES for j in completed[stage])
    )
    selector_ids = tuple(
        sorted(j for stage in SELECTOR_STAGES for j in completed[stage])
    )
    scalar_ids = tuple(sorted(completed[SCALAR_STAGE]))
    operation_ids = tuple(
        sorted(j for stage in OPERATION_STAGES for j in completed[stage])
    )
    return fit_ids, prediction_ids, selector_ids, scalar_ids, operation_ids


def _check_accounting_profile(accounting):
    if not isinstance(accounting, dict):
        raise UniversalEvidenceError("accounting_profile_invalid")
    schema = accounting.get("schema_version")
    if schema == R2_ACCOUNTING_SCHEMA:
        if accounting.get("max_fit_total") != R2_MAX_FIT_TOTAL:
            raise UniversalEvidenceError("accounting_max_fit_total")
        if (
            accounting.get("historical_overhead_attempts")
            != R2_HISTORICAL_OVERHEAD_ATTEMPTS
        ):
            raise UniversalEvidenceError("accounting_overhead")
        if accounting.get("expected_reuse_fits") != R2_EXPECTED_REUSE_PAIRS:
            raise UniversalEvidenceError("accounting_reuse_fits")
        if accounting.get("expected_reuse_predictions") != R2_EXPECTED_REUSE_PAIRS:
            raise UniversalEvidenceError("accounting_reuse_predictions")
        return accounting
    if schema == STANDING_ACCOUNTING_SCHEMA:
        return _check_standing_accounting_profile(accounting)
    raise UniversalEvidenceError("accounting_profile_unsupported")


def _check_standing_accounting_profile(accounting):
    if accounting.get("standing") is not True:
        raise UniversalEvidenceError("accounting_profile_unsupported")
    completed = accounting.get("completed_job_ids_by_stage")
    history = accounting.get("replay_counts_by_stage")
    if (
        not isinstance(completed, dict)
        or set(completed) != KNOWN_STAGES
        or not isinstance(history, dict)
        or set(history) != KNOWN_STAGES
    ):
        raise UniversalEvidenceError("accounting_profile_unsupported")
    fit_ids, prediction_ids, selector_ids, scalar_ids, operation_ids = (
        _derive_standing_reuse_sets(accounting)
    )
    if accounting.get("expected_reuse_fits") != len(fit_ids):
        raise UniversalEvidenceError("accounting_reuse_fits")
    if accounting.get("expected_reuse_predictions") != len(prediction_ids):
        raise UniversalEvidenceError("accounting_reuse_predictions")
    if tuple(accounting.get("additional_reuse_fit_job_ids", ())) != fit_ids:
        raise UniversalEvidenceError("accounting_reuse_fits")
    if tuple(accounting.get("selector_job_ids", ())) != selector_ids:
        raise UniversalEvidenceError("accounting_selector")
    if tuple(accounting.get("scalar_job_ids", ())) != scalar_ids:
        raise UniversalEvidenceError("accounting_scalar")
    if tuple(accounting.get("operation_job_ids", ())) != operation_ids:
        raise UniversalEvidenceError("accounting_operation")
    if accounting.get("max_unique_fit_jobs") != MAX_UNIQUE_FIT_JOBS:
        raise UniversalEvidenceError("accounting_unique_fit_total")
    if accounting.get("max_scalar_attempts") != MAX_SCALAR_ATTEMPTS:
        raise UniversalEvidenceError("accounting_scalar_total")
    max_fit_total = accounting.get("max_fit_total")
    historical = accounting.get("historical_overhead_attempts")
    if (
        isinstance(max_fit_total, bool)
        or not isinstance(max_fit_total, int)
        or isinstance(historical, bool)
        or not isinstance(historical, int)
        or historical < 0
    ):
        raise UniversalEvidenceError("accounting_overhead")
    if MAX_UNIQUE_FIT_JOBS + historical > max_fit_total:
        raise UniversalEvidenceError("accounting_max_fit_total")
    return accounting


_COUNTER_FIELDS = {
    "completed_new_fits": R2_NEW_FITS,
    "completed_new_calibrations": R2_NEW_CAL,
    "completed_new_operations": R2_NEW_OPERATIONS,
    "reused_fits": R2_REUSED_FITS,
    "reused_predictions": R2_REUSED_PREDICTIONS,
    "reused_epoch_selections": R2_REUSED_EPOCH,
}

# Ledger public-summary reuse keys mapped to the derived reuse kind each one
# counts.  ``reuse_operations`` is the generic alias/ensemble operation-kind
# count; the completion counter ``reused_operations`` is a different quantity
# (the total across all five kinds) and must never be mapped onto it.
_STANDING_LEDGER_REUSE_KEYS = {
    "reuse_fits": "fit",
    "reuse_predictions": "prediction",
    "reuse_epoch_selections": "selector",
    "reuse_scalar_calibrations": "calibration",
    "reuse_operations": "operation",
}


def _nonnegative_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _standing_completion_counters(accounting):
    fit_ids, prediction_ids, selector_ids, scalar_ids, operation_ids = (
        _derive_standing_reuse_sets(accounting)
    )
    reused_fits = len(fit_ids)
    reused_predictions = len(prediction_ids)
    reused_selectors = len(selector_ids)
    reused_scalar = len(scalar_ids)
    # ``reused_operations`` is the total number of reused jobs across all five
    # reuse kinds, not the generic alias/ensemble operation-kind count.
    reused_operations = (
        reused_fits
        + reused_predictions
        + reused_selectors
        + reused_scalar
        + len(operation_ids)
    )
    return {
        "completed_new_fits": MAX_UNIQUE_FIT_JOBS - reused_fits,
        "completed_new_calibrations": MAX_SCALAR_ATTEMPTS - reused_scalar,
        "completed_new_operations": EXPECTED_SG_ARPLS_JOBS - reused_operations,
        "reused_fits": reused_fits,
        "reused_predictions": reused_predictions,
        "reused_selectors": reused_selectors,
        "reused_scalar_calibrations": reused_scalar,
        "reused_operations": reused_operations,
    }


def _check_completion(completion, accounting=None) -> None:
    if _is_standing_accounting(accounting):
        _check_standing_completion(completion, accounting)
        return
    if not isinstance(completion, dict):
        raise UniversalEvidenceError("completion_invalid")
    if set(completion) != {
        "summary",
        "counters",
        "ledger_summary",
        "event_verification",
    }:
        raise UniversalEvidenceError("completion_keys_invalid")
    counters = completion["counters"]
    if not isinstance(counters, dict) or set(counters) != set(_COUNTER_FIELDS):
        raise UniversalEvidenceError("completion_counters_invalid")
    for name, expected in _COUNTER_FIELDS.items():
        value = counters[name]
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise UniversalEvidenceError("completion_counter_" + name + "_mismatch")
    if not isinstance(completion["ledger_summary"], dict):
        raise UniversalEvidenceError("completion_ledger_summary_invalid")
    if not isinstance(completion["event_verification"], dict):
        raise UniversalEvidenceError("completion_event_verification_invalid")


def _check_standing_ledger_summary(ledger_summary, accounting, counters) -> None:
    if not isinstance(ledger_summary, dict):
        raise UniversalEvidenceError("completion_ledger_summary_invalid")
    fit_ids, prediction_ids, selector_ids, scalar_ids, operation_ids = (
        _derive_standing_reuse_sets(accounting)
    )
    derived = {
        "fit": len(fit_ids),
        "prediction": len(prediction_ids),
        "selector": len(selector_ids),
        "calibration": len(scalar_ids),
        "operation": len(operation_ids),
    }
    for key, kind in _STANDING_LEDGER_REUSE_KEYS.items():
        if key not in ledger_summary:
            raise UniversalEvidenceError("completion_ledger_" + key + "_missing")
        value = ledger_summary[key]
        if not _nonnegative_int(value) or value != derived[kind]:
            raise UniversalEvidenceError("completion_ledger_" + key + "_mismatch")
    reuse_total = sum(derived.values())
    # ``counters['reused_operations']`` is the total across all five reuse
    # kinds, while the ledger's ``reuse_operations`` key checked above is the
    # generic alias/ensemble operation-kind count.  They are not equal.
    if counters["reused_operations"] != reuse_total:
        raise UniversalEvidenceError("completion_ledger_reuse_total_mismatch")

    jobs_total = ledger_summary.get("jobs_total")
    if not _nonnegative_int(jobs_total) or jobs_total != EXPECTED_SG_ARPLS_JOBS:
        raise UniversalEvidenceError("completion_ledger_jobs_total")
    expected_attempts = EXPECTED_SG_ARPLS_JOBS - reuse_total
    attempts_total = ledger_summary.get("attempts_total")
    if not _nonnegative_int(attempts_total) or attempts_total != expected_attempts:
        raise UniversalEvidenceError("completion_ledger_attempts_total")
    running_attempts = ledger_summary.get("running_attempts")
    if not _nonnegative_int(running_attempts) or running_attempts != 0:
        raise UniversalEvidenceError("completion_ledger_running_attempts")
    status_counts = ledger_summary.get("attempt_status_counts")
    if not isinstance(status_counts, dict):
        raise UniversalEvidenceError("completion_ledger_attempt_status_counts")
    for status, count in status_counts.items():
        if not isinstance(status, str) or not _nonnegative_int(count):
            raise UniversalEvidenceError("completion_ledger_attempt_status_counts")
    if status_counts.get("complete") != expected_attempts:
        raise UniversalEvidenceError("completion_ledger_attempt_status_counts")
    if sum(status_counts.values()) != expected_attempts:
        raise UniversalEvidenceError("completion_ledger_attempt_status_counts")

    historical = accounting.get("historical_overhead_attempts")
    max_fit_total = accounting.get("max_fit_total")
    if not _nonnegative_int(historical) or not _nonnegative_int(max_fit_total):
        raise UniversalEvidenceError("accounting_overhead")
    expected_accounted_fit = MAX_UNIQUE_FIT_JOBS + historical
    if expected_accounted_fit > max_fit_total:
        raise UniversalEvidenceError("completion_ledger_accounted_fit_total")
    accounted_fit = ledger_summary.get("accounted_model_fit_attempts")
    if not _nonnegative_int(accounted_fit) or accounted_fit != expected_accounted_fit:
        raise UniversalEvidenceError("completion_ledger_accounted_fit_total")

    expected_new_scalar = MAX_SCALAR_ATTEMPTS - derived["calibration"]
    new_scalar = ledger_summary.get("new_scalar_attempts")
    if not _nonnegative_int(new_scalar) or new_scalar != expected_new_scalar:
        raise UniversalEvidenceError("completion_ledger_new_scalar_attempts")
    scalar_overhead = accounting.get("scalar_overhead_attempts")
    if not _nonnegative_int(scalar_overhead):
        raise UniversalEvidenceError("completion_ledger_scalar_overhead_attempts")
    reported_overhead = ledger_summary.get("scalar_overhead_attempts")
    if (
        not _nonnegative_int(reported_overhead)
        or reported_overhead != scalar_overhead
    ):
        raise UniversalEvidenceError("completion_ledger_scalar_overhead_attempts")
    expected_accounted_scalar = (
        expected_new_scalar + derived["calibration"] + scalar_overhead
    )
    accounted_scalar = ledger_summary.get("accounted_scalar_attempts")
    if (
        not _nonnegative_int(accounted_scalar)
        or accounted_scalar != expected_accounted_scalar
    ):
        raise UniversalEvidenceError("completion_ledger_accounted_scalar_attempts")

    flags = ledger_summary.get("flags")
    if not isinstance(flags, dict):
        raise UniversalEvidenceError("completion_ledger_flags")
    if flags.get("failure") is not False:
        raise UniversalEvidenceError("completion_ledger_failure")
    if flags.get("sealed") is not True:
        raise UniversalEvidenceError("completion_ledger_sealed")
    if flags.get("clean") is not False:
        raise UniversalEvidenceError("completion_ledger_clean")
    if ledger_summary.get("state") != "open":
        raise UniversalEvidenceError("completion_ledger_open")


def _check_event_verification(event_verification, ledger_summary) -> None:
    if not isinstance(event_verification, dict):
        raise UniversalEvidenceError("completion_event_verification_invalid")
    if event_verification.get("ok") is not True:
        raise UniversalEvidenceError("completion_event_verification_invalid")
    events = event_verification.get("events")
    head = event_verification.get("event_head")
    if not _nonnegative_int(events) or not _is_hex(head):
        raise UniversalEvidenceError("completion_event_verification_invalid")
    # The completion is captured before the closing event, so the count/head
    # must match the pre-close ledger summary, never a later closed head.
    if ledger_summary.get("event_count") != events:
        raise UniversalEvidenceError("completion_event_count_mismatch")
    if ledger_summary.get("event_head") != head:
        raise UniversalEvidenceError("completion_event_head_mismatch")


def _check_standing_completion(completion, accounting) -> None:
    if not isinstance(completion, dict):
        raise UniversalEvidenceError("completion_invalid")
    if set(completion) != {
        "summary",
        "counters",
        "ledger_summary",
        "event_verification",
    }:
        raise UniversalEvidenceError("completion_keys_invalid")
    counters = completion["counters"]
    expected = _standing_completion_counters(accounting)
    if not isinstance(counters, dict) or set(counters) != set(expected):
        raise UniversalEvidenceError("completion_counters_invalid")
    for name, value in expected.items():
        raw = counters[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw != value:
            raise UniversalEvidenceError("completion_counter_" + name + "_mismatch")
    summary = completion["summary"]
    if not isinstance(summary, dict):
        raise UniversalEvidenceError("completion_summary_invalid")
    if summary.get("state") != "complete":
        raise UniversalEvidenceError("completion_summary_state")
    # The accepted controller summary has no ``total`` key; ``complete`` and
    # ``completed`` carry the whole-graph job count.
    for field in ("complete", "completed"):
        value = summary.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != EXPECTED_SG_ARPLS_JOBS
        ):
            raise UniversalEvidenceError("completion_summary_" + field)
    for field in ("remaining", "running"):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise UniversalEvidenceError("completion_summary_" + field)
    ledger_summary = completion["ledger_summary"]
    if not isinstance(ledger_summary, dict):
        raise UniversalEvidenceError("completion_ledger_summary_invalid")
    _check_standing_ledger_summary(ledger_summary, accounting, counters)
    _check_event_verification(completion["event_verification"], ledger_summary)


def _check_close(close) -> None:
    if not isinstance(close, dict):
        raise UniversalEvidenceError("close_invalid")
    if set(close) != {"store", "worker_shutdown_errors"}:
        raise UniversalEvidenceError("close_keys_invalid")
    store = close["store"]
    if not isinstance(store, dict) or set(store) != {"already_closed", "clean"}:
        raise UniversalEvidenceError("close_store_invalid")
    if store["already_closed"] is not False:
        raise UniversalEvidenceError("close_already_closed")
    if store["clean"] is not True:
        raise UniversalEvidenceError("close_not_clean")
    errors = close["worker_shutdown_errors"]
    if not isinstance(errors, list) or errors:
        raise UniversalEvidenceError("worker_shutdown_errors")


def _open_readonly_ledger(path: Path) -> sqlite3.Connection:
    uri = "file:" + urllib.parse.quote(str(path)) + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    except sqlite3.Error as exc:
        raise UniversalEvidenceError("ledger_open_failed") from exc
    conn.row_factory = sqlite3.Row
    return conn


def _meta_required_count(conn, key):
    # A missing counter is an error; it is never replaced by an expected or
    # reconstructed count.
    raw = _meta_get(conn, key, None)
    if raw is None:
        raise UniversalEvidenceError("ledger_meta_missing")
    if not isinstance(raw, str) or not raw.isdigit() or raw != str(int(raw)):
        raise UniversalEvidenceError("ledger_meta_invalid")
    return int(raw)


def _reuse_counts_from_ledger(conn):
    counts = {
        "fit": 0,
        "prediction": 0,
        "selector": 0,
        "calibration": 0,
        "operation": 0,
    }
    rows = conn.execute(
        "SELECT r.job_id AS job_id, j.job_json AS job_json "
        "FROM reuse r JOIN jobs j ON j.job_id = r.job_id"
    ).fetchall()
    for row in rows:
        record = json.loads(row["job_json"])
        kind = classify_stage_kind(record["stage"])
        if kind not in counts:
            raise UniversalEvidenceError("ledger_reuse_kind_unknown")
        counts[kind] += 1
    return counts


def _check_counters_standing(conn, accounting) -> None:
    fit_ids, prediction_ids, selector_ids, scalar_ids, operation_ids = (
        _derive_standing_reuse_sets(accounting)
    )
    expected_reuse = {
        "fit": len(fit_ids),
        "prediction": len(prediction_ids),
        "selector": len(selector_ids),
        "calibration": len(scalar_ids),
        "operation": len(operation_ids),
    }
    expected_new_fits = MAX_UNIQUE_FIT_JOBS - expected_reuse["fit"]
    expected_new_scalar = MAX_SCALAR_ATTEMPTS - expected_reuse["calibration"]
    table_counts = _reuse_counts_from_ledger(conn)
    fit = _meta_required_count(conn, "fit_attempt_count")
    scalar = _meta_required_count(conn, "scalar_attempt_count")
    ledger_reuse = {
        "fit": _meta_required_count(conn, "reuse_fit_count"),
        "prediction": _meta_required_count(conn, "reuse_prediction_count"),
        "selector": _meta_required_count(conn, "reuse_selector_count"),
        "calibration": _meta_required_count(conn, "reuse_calibration_count"),
        "operation": _meta_required_count(conn, "reuse_operation_count"),
    }
    if ledger_reuse != table_counts:
        raise UniversalEvidenceError("reuse_count_table_mismatch")
    if fit != expected_new_fits:
        raise UniversalEvidenceError("fit_attempt_count_mismatch")
    if scalar != expected_new_scalar:
        raise UniversalEvidenceError("scalar_attempt_count_mismatch")
    if ledger_reuse != expected_reuse:
        raise UniversalEvidenceError("reuse_count_mismatch")
    if fit + ledger_reuse["fit"] != MAX_UNIQUE_FIT_JOBS:
        raise UniversalEvidenceError("unique_fit_total_mismatch")
    if scalar + ledger_reuse["calibration"] != MAX_SCALAR_ATTEMPTS:
        raise UniversalEvidenceError("unique_scalar_total_mismatch")
    historical = accounting.get("historical_overhead_attempts")
    max_fit_total = accounting.get("max_fit_total")
    if (
        isinstance(historical, bool)
        or not isinstance(historical, int)
        or historical < 0
        or isinstance(max_fit_total, bool)
        or not isinstance(max_fit_total, int)
    ):
        raise UniversalEvidenceError("accounting_overhead")
    if fit + ledger_reuse["fit"] + historical > max_fit_total:
        raise UniversalEvidenceError("accounted_fit_total_mismatch")


def _check_counters(conn, accounting=None) -> None:
    if _is_standing_accounting(accounting):
        _check_counters_standing(conn, accounting)
        return
    fit = _meta_required_count(conn, "fit_attempt_count")
    scalar = _meta_required_count(conn, "scalar_attempt_count")
    reuse_fit = _meta_required_count(conn, "reuse_fit_count")
    reuse_pred = _meta_required_count(conn, "reuse_prediction_count")
    reuse_selector = _meta_required_count(conn, "reuse_selector_count")
    if fit != R2_NEW_FITS:
        raise UniversalEvidenceError("fit_attempt_count_mismatch")
    if scalar != R2_NEW_CAL or scalar != MAX_SCALAR_ATTEMPTS:
        raise UniversalEvidenceError("scalar_attempt_count_mismatch")
    if reuse_fit != R2_REUSED_FITS:
        raise UniversalEvidenceError("reuse_fit_count_mismatch")
    if reuse_pred != R2_REUSED_PREDICTIONS:
        raise UniversalEvidenceError("reuse_prediction_count_mismatch")
    if reuse_selector != R2_REUSED_EPOCH:
        raise UniversalEvidenceError("reuse_selector_count_mismatch")
    if fit + reuse_fit != MAX_UNIQUE_FIT_JOBS:
        raise UniversalEvidenceError("unique_fit_total_mismatch")
    if fit + reuse_fit + R2_HISTORICAL_OVERHEAD_ATTEMPTS != R2_MAX_FIT_TOTAL:
        raise UniversalEvidenceError("accounted_fit_total_mismatch")


def _artifact_store(run_root: Path, binding_sha256: str) -> ArtifactStore:
    artifacts = run_root / "artifacts"
    jobs_dir = artifacts / "jobs"
    if jobs_dir.is_symlink() or not jobs_dir.is_dir():
        raise UniversalEvidenceError("artifacts_missing")
    try:
        return ArtifactStore(artifacts, binding_sha256=binding_sha256)
    except ArtifactError as exc:
        raise UniversalEvidenceError("artifact_store_invalid") from exc


def _predictions_frame(blob):
    try:
        return pd.read_csv(io.BytesIO(blob), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise UniversalEvidenceError("predictions_read_failed") from exc


def _audit_new_dependencies(new_ids, attempts, reuse, graph_by_id):
    completed = set(attempts) | set(reuse)
    for job_id in new_ids:
        record = graph_by_id[job_id]
        for dependency in record.get("dependencies", ()):
            if dependency not in new_ids:
                raise UniversalEvidenceError("dependency_outside_new_set")
            if dependency not in completed:
                raise UniversalEvidenceError("dependency_incomplete")


def _read_complete_receipt(store, record, recorded, prefix):
    if not isinstance(recorded, dict):
        raise UniversalEvidenceError(prefix + "receipt_invalid")
    digest = recorded.get("sha256")
    if not _is_hex(digest):
        raise UniversalEvidenceError(prefix + "receipt_sha256_invalid")
    body = {key: value for key, value in recorded.items() if key != "sha256"}
    if _sha256_json(body) != digest:
        raise UniversalEvidenceError(prefix + "receipt_hash_mismatch")
    try:
        receipt, files = store.verify(record, expected_receipt_sha256=digest)
    except ArtifactError as exc:
        raise UniversalEvidenceError(prefix + "artifact_verify_failed") from exc
    if receipt != recorded:
        raise UniversalEvidenceError(prefix + "receipt_mismatch")
    if receipt.get("status") != "complete":
        raise UniversalEvidenceError(prefix + "receipt_not_complete")
    return receipt, files


def _audit_standing_reuse(accounting, reuse, graph_by_id) -> None:
    fit_ids, prediction_ids, selector_ids, scalar_ids, operation_ids = (
        _derive_standing_reuse_sets(accounting)
    )
    expected = {}
    for job_id in fit_ids:
        expected[job_id] = "fit"
    for job_id in prediction_ids:
        expected[job_id] = "prediction"
    for job_id in selector_ids:
        expected[job_id] = "selector"
    for job_id in scalar_ids:
        expected[job_id] = "calibration"
    for job_id in operation_ids:
        expected[job_id] = "operation"
    if set(reuse) != set(expected):
        raise UniversalEvidenceError("standing_reuse_set_mismatch")
    for job_id, kind in expected.items():
        row = reuse.get(job_id)
        if row is None:
            raise UniversalEvidenceError("standing_reuse_job_missing")
        recorded = row["kind"]
        if not isinstance(recorded, str) or recorded != kind:
            raise UniversalEvidenceError("standing_reuse_kind_mismatch")
        record = graph_by_id.get(job_id)
        if record is None:
            raise UniversalEvidenceError("standing_reuse_job_missing")
        if classify_stage_kind(record["stage"]) != kind:
            raise UniversalEvidenceError("standing_reuse_kind_mismatch")


_TRAINING_RECIPES = frozenset(("D0-M", "D1", "D2", "D3"))
_TRAINING_POLICIES = frozenset(("PP-U-SG", "PP-U-ARPLS"))
_TRAINING_STAGES = frozenset(("source_fit", "final_refit"))
_TRAINING_SUMMARY_NAME = "summary.json"


def _collect_training_summary(record, receipt, files, sink):
    """Retain one authenticated neural fit summary for later reporting.

    Called only after ``_read_complete_receipt`` proved an authoritative
    complete receipt and verified every payload byte.  Classical, MIN and
    non-fit slots are ignored.  No checkpoint is loaded and no new IO occurs.
    """
    if sink is None:
        return
    if (
        record.get("model_id") not in _TRAINING_RECIPES
        or record.get("policy_id") not in _TRAINING_POLICIES
        or record.get("stage") not in _TRAINING_STAGES
    ):
        return
    job_id = record.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise UniversalEvidenceError("training_job_id_invalid")
    if job_id in sink:
        raise UniversalEvidenceError("training_summary_duplicate")
    raw = files.get(_TRAINING_SUMMARY_NAME)
    if raw is None:
        raise UniversalEvidenceError("training_summary_missing")
    if not isinstance(raw, (bytes, bytearray)):
        raise UniversalEvidenceError("training_summary_invalid")
    raw = bytes(raw)
    try:
        summary = _strict_loads(raw)
    except Exception as exc:
        raise UniversalEvidenceError("training_summary_malformed") from exc
    if not isinstance(summary, dict):
        raise UniversalEvidenceError("training_summary_malformed")
    digest = receipt.get("sha256") if isinstance(receipt, dict) else None
    if not _is_hex(digest):
        raise UniversalEvidenceError("training_receipt_sha256_invalid")
    sink[job_id] = {
        "job": copy.deepcopy(record),
        "summary": summary,
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_sha256": digest,
    }


def _audit_ledger(
    run_root, roots, binding, graph_by_id, new_ids, accounting=None, *, training_sink=None
):
    ledger_path = _secure_file(run_root / "ledger" / "ledger.sqlite3", roots)
    binding_json = _canonical(binding)
    binding_sha = hashlib.sha256(binding_json.encode("utf-8")).hexdigest()
    standing = _is_standing_accounting(accounting) or _binding_is_standing(binding)
    if standing and not _is_standing_accounting(accounting):
        accounting = _check_accounting_profile(execution_accounting(binding))
    conn = _open_readonly_ledger(ledger_path)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise UniversalEvidenceError("ledger_integrity_check_failed")
        _verify_events(conn, binding_json)
        _verify_registered_jobs(conn)
        if _meta_get(conn, "binding_sha256", None) != binding_sha:
            raise UniversalEvidenceError("binding_sha256_mismatch")
        for key, value in (
            ("state", "closed"),
            ("clean", "1"),
            ("failure", "0"),
            ("sealed", "1"),
        ):
            if _meta_get(conn, key, None) != value:
                raise UniversalEvidenceError("ledger_" + key + "_mismatch")
        registered = {
            row["job_id"]: row["job_json"]
            for row in conn.execute("SELECT job_id,job_json FROM jobs")
        }
        if set(registered) != set(new_ids):
            raise UniversalEvidenceError("ledger_job_set_mismatch")
        for job_id in new_ids:
            if registered[job_id] != _canonical(graph_by_id[job_id]):
                raise UniversalEvidenceError("ledger_job_json_mismatch")
        attempts = {
            row["job_id"]: row
            for row in conn.execute(
                "SELECT job_id,status,receipt_json,receipt_sha FROM attempts"
            )
        }
        reuse = {
            row["job_id"]: row
            for row in conn.execute("SELECT job_id,kind,evidence_json FROM reuse")
        }
        audit_operation_coverage(new_ids, attempts, reuse)
        _audit_new_dependencies(new_ids, attempts, reuse, graph_by_id)
        _check_counters(conn, accounting if standing else None)
        if standing:
            _audit_standing_reuse(accounting, reuse, graph_by_id)
        store = _artifact_store(run_root, binding_sha)
        endpoint_frames = {}
        for job_id, row in attempts.items():
            if row["status"] != "complete":
                raise UniversalEvidenceError("attempt_not_complete")
            record = graph_by_id[job_id]
            recorded = json.loads(row["receipt_json"])
            if _sha256_json(recorded) != row["receipt_sha"]:
                raise UniversalEvidenceError("receipt_ledger_hash_mismatch")
            _receipt, files = _read_complete_receipt(store, record, recorded, "attempt_")
            if "error.json" in files:
                raise UniversalEvidenceError("error_artifact_present")
            _collect_training_summary(record, _receipt, files, training_sink)
            if record["stage"] == "seed_ensemble_prediction":
                blob = files.pop("predictions.csv", None)
                if blob is None:
                    raise UniversalEvidenceError("endpoint_artifact_missing")
                endpoint_frames[job_id] = _predictions_frame(blob)
            files = None
        for job_id, row in reuse.items():
            record = graph_by_id[job_id]
            evidence = json.loads(row["evidence_json"])
            recorded = evidence.get("current_receipt")
            if not isinstance(recorded, dict):
                raise UniversalEvidenceError("reuse_receipt_missing")
            if recorded.get("job_id") != job_id:
                raise UniversalEvidenceError("reuse_receipt_job_mismatch")
            _receipt, files = _read_complete_receipt(store, record, recorded, "reuse_")
            if "error.json" in files:
                raise UniversalEvidenceError("reuse_error_artifact_present")
            _collect_training_summary(record, _receipt, files, training_sink)
            if record["stage"] == "seed_ensemble_prediction":
                blob = files.pop("predictions.csv", None)
                if blob is None:
                    raise UniversalEvidenceError("endpoint_artifact_missing")
                endpoint_frames[job_id] = _predictions_frame(blob)
            files = None
        return endpoint_frames
    finally:
        conn.close()


def _require_parquet_engine() -> None:
    for module in ("pyarrow", "fastparquet"):
        if importlib.util.find_spec(module) is not None:
            return
    raise UniversalEvidenceError("parquet_engine_missing")


def read_endpoint_frame(path, sha256, roots) -> pd.DataFrame:
    real = _secure_file(path, roots)
    if _stream_sha256(real) != sha256:
        raise UniversalEvidenceError("hash_mismatch")
    suffix = real.suffix.lower()
    if suffix == ".parquet":
        _require_parquet_engine()
        try:
            return pd.read_parquet(real)
        except UniversalEvidenceError:
            raise
        except Exception as exc:
            raise UniversalEvidenceError("parquet_read_failed") from exc
    if suffix == ".csv":
        try:
            return pd.read_csv(real, dtype=str, keep_default_na=False)
        except Exception as exc:
            raise UniversalEvidenceError("csv_read_failed") from exc
    raise UniversalEvidenceError("endpoint_kind_unsupported")


def _endpoint_frame(path, sha256, roots, cache) -> pd.DataFrame:
    key = (str(path), sha256)
    frame = cache.get(key)
    if frame is None:
        frame = read_endpoint_frame(path, sha256, roots)
        cache[key] = frame
    return frame


def _pinned_frame(root: Path, key: str, roots) -> pd.DataFrame:
    rel, expected = PINS[key]
    path = _secure_relative(root, rel, roots)
    if _stream_sha256(path) != expected:
        raise UniversalEvidenceError("pin_hash_mismatch")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise UniversalEvidenceError("csv_read_failed") from exc
    if key == "manifest":
        # P01 does not store this derived P02 field. Preserve the frozen mapping
        # for the preplanned leave-one-platform-family descriptive sensitivity.
        if "instrument" not in frame:
            raise UniversalEvidenceError("manifest_instrument_missing")
        try:
            derived = frame["instrument"].map(instrument_family)
        except (TypeError, ValueError) as exc:
            raise UniversalEvidenceError("manifest_platform_family_invalid") from exc
        if "instrument_family" in frame and not frame["instrument_family"].eq(derived).all():
            raise UniversalEvidenceError("manifest_platform_family_conflict")
        frame["instrument_family"] = derived
    return frame


def load_evidence(
    *,
    private_root,
    graph_path,
    bridge_path,
    completed_run_root,
    allowed_evidence_roots,
):
    """Read-only aggregation of the frozen graph, MIN bridge, pins and run.

    This loader authenticates the frozen inputs it can read.  Runtime,
    launcher and dependency reverification that requires executing or
    re-deriving the run remains the outer analysis-runner's responsibility and
    is deliberately not claimed complete here.
    """
    roots = _prepare_roots(allowed_evidence_roots)
    private = _secure_dir(private_root, roots)
    run_root = _secure_dir(completed_run_root, roots)
    _require_completed_files(run_root)

    graph = _load_gzip_json(_authenticated_file(graph_path, roots, GRAPH_SHA256))
    graph_by_id, aliases, graph_jobs, graph_min_ids, new_ids = _authenticate_graph(
        graph
    )

    bridge = _load_gzip_json(_authenticated_file(bridge_path, roots, MIN_BRIDGE_SHA256))
    min_ids, verified, min_endpoints = _authenticate_min_bridge(bridge, graph_by_id)
    _verify_listed_files(verified, roots)

    if min_ids != graph_min_ids:
        raise UniversalEvidenceError("graph_min_set_mismatch")
    if len(min_ids) != EXPECTED_MIN_JOBS or len(new_ids) != EXPECTED_SG_ARPLS_JOBS:
        raise UniversalEvidenceError("graph_partition_mismatch")

    permit = _read_json_file(run_root / "permit.json", roots)
    binding = _read_json_file(run_root / "binding.json", roots)
    completion = _read_json_file(run_root / "completion.json", roots)
    close = _read_json_file(run_root / "close.json", roots)
    _authenticate_permit(permit, binding, run_root)
    accounting = _check_accounting_profile(execution_accounting(binding))
    _check_completion(completion, accounting)
    _check_close(close)

    training_sink = {}
    new_endpoint_frames = _audit_ledger(
        run_root,
        roots,
        binding,
        graph_by_id,
        new_ids,
        accounting,
        training_sink=training_sink,
    )

    manifest = _pinned_frame(private, "manifest", roots)
    contexts = _pinned_frame(private, "contexts", roots)
    roles = _pinned_frame(private, "roles", roots)

    if len(min_endpoints) != EXPECTED_MIN_ENDPOINTS:
        raise UniversalEvidenceError("min_endpoint_count_mismatch")
    if len(new_endpoint_frames) != EXPECTED_NEW_ENDPOINTS:
        raise UniversalEvidenceError("new_endpoint_count_mismatch")

    endpoint_frames: dict[str, pd.DataFrame] = {}
    frame_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for job_id, pointer, record in min_endpoints:
        if job_id in endpoint_frames:
            raise UniversalEvidenceError("duplicate_endpoint")
        kind, value = validate_selector(
            pointer["selector"],
            model_id=record["model_id"],
            context_id=record.get("context_id"),
        )
        frame = _endpoint_frame(pointer["file"], pointer["sha256"], roots, frame_cache)
        endpoint_frames[job_id] = apply_selector(frame, kind=kind, value=value)
    for job_id, frame in new_endpoint_frames.items():
        if job_id in endpoint_frames:
            raise UniversalEvidenceError("duplicate_endpoint")
        endpoint_frames[job_id] = frame
    if len(endpoint_frames) != EXPECTED_DISTINCT_ENDPOINTS:
        raise UniversalEvidenceError("endpoint_count_mismatch")

    if _is_standing_accounting(accounting):
        (
            standing_fit_ids,
            standing_prediction_ids,
            standing_selector_ids,
            standing_scalar_ids,
            standing_operation_ids,
        ) = _derive_standing_reuse_sets(accounting)
        replay_fit_jobs = len(accounting.get("replay_fit_job_ids", ()))
        current_replay_jobs = len(accounting.get("current_replay_ids", ()))
        additional_reuse_fit_jobs = len(standing_fit_ids)
        reused_selector_jobs = len(standing_selector_ids)
        reused_scalar_jobs = len(standing_scalar_ids)
        reused_operation_jobs = len(standing_operation_ids)
        reused_operations_total = (
            len(standing_fit_ids)
            + len(standing_prediction_ids)
            + len(standing_selector_ids)
            + len(standing_scalar_ids)
            + len(standing_operation_ids)
        )
        expected_reuse_pairs = len(standing_prediction_ids)
        historical_overhead = int(accounting.get("historical_overhead_attempts", 0))
    else:
        replay_fit_jobs = R2_REPLAY_FIT_COUNT
        current_replay_jobs = 0
        additional_reuse_fit_jobs = R2_ADDITIONAL_FIT_COUNT
        reused_selector_jobs = R2_SELECTOR_COUNT
        reused_scalar_jobs = 0
        reused_operation_jobs = 0
        reused_operations_total = (
            R2_REUSED_FITS + R2_REUSED_PREDICTIONS + R2_REUSED_EPOCH
        )
        expected_reuse_pairs = R2_EXPECTED_REUSE_PAIRS
        historical_overhead = R2_HISTORICAL_OVERHEAD_ATTEMPTS
    diagnostics = {
        "schema_version": SCHEMA_VERSION,
        "graph_sha256": GRAPH_SHA256,
        "graph_plan_sha256": GRAPH_PLAN_SHA256,
        "min_bridge_sha256": MIN_BRIDGE_SHA256,
        "graph_jobs": len(graph_by_id),
        "graph_min_jobs": len(min_ids),
        "graph_sg_arpls_jobs": len(new_ids),
        "aliases": len(aliases),
        "min_operation_bindings": len(bridge.get("operation_bindings", ())),
        "verified_file_hashes": len(verified),
        "min_endpoints": len(min_endpoints),
        "new_endpoints": len(new_endpoint_frames),
        "distinct_endpoints": len(endpoint_frames),
        "report_cells": EXPECTED_REPORT_CELLS,
        "accounting_schema": accounting.get("schema_version"),
        "standing": _is_standing_accounting(accounting),
        "replay_fit_jobs": replay_fit_jobs,
        "current_replay_jobs": current_replay_jobs,
        "additional_reuse_fit_jobs": additional_reuse_fit_jobs,
        "reused_selector_jobs": reused_selector_jobs,
        "reused_scalar_jobs": reused_scalar_jobs,
        "reused_operation_jobs": reused_operation_jobs,
        "reused_operations_total": reused_operations_total,
        "expected_reuse_pairs": expected_reuse_pairs,
        "historical_overhead_attempts": historical_overhead,
        "outer_reverification_required": True,
    }
    return {
        "manifest": manifest,
        "contexts": contexts,
        "roles": roles,
        "jobs": graph_jobs,
        "aliases": aliases,
        "endpoint_frames": endpoint_frames,
        "diagnostics": diagnostics,
        "training_fit_summaries": training_sink,
    }
