"""P08-U1 durable full-run ledger primitive (T275).

This module owns durable governance state for a future U1 full run.  It is a
ledger, not a launcher and not an artifact store:

* SQLite (WAL, ``synchronous=FULL``, ``foreign_keys=ON``) holds an immutable
  run binding, registered job records, the reuse import, attempts, resource
  snapshots and an append-only hash-linked event journal.
* An exclusive lifetime advisory lock (``flock``) gives one controller process
  ownership.  Worker processes never open the store.
* Nothing here fits a model, reads spectra, writes scientific artifacts,
  authenticates frozen graph bytes, validates receipts or measures hardware.
  Registered jobs and reuse evidence are *inputs* supplied by a future
  controller that already performed authentication.  This primitive records
  what it was given and fails closed on internal inconsistency.
* There is no restore/reset/delete API and no no-reuse execution mode: full U1
  requires the exact sealed reuse import before any new attempt starts.

Recorded limits are inherited workflow bounds, not new scientific choices.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sqlite3
from datetime import datetime, timezone

__all__ = [
    "P08U1Error",
    "ValidationError",
    "ReviewRequiredError",
    "LockError",
    "BudgetError",
    "P08U1Store",
    "SCHEMA_VERSION",
    "MODEL_FIT_STAGES",
    "SCALAR_STAGE",
    "REUSE_STAGES",
    "KNOWN_STAGES",
    "KNOWN_MODELS",
    "ALLOWED_POLICIES",
    "ENUM_WORKER_KINDS",
    "RESOURCE_FIELDS",
    "job_sha256",
    "execution_accounting",
]

SCHEMA_VERSION = "nato-sers-p08-u1-ledger-v1"

ALLOWED_POLICIES = ("PP-U-SG", "PP-U-ARPLS")
CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
NEURAL_RECIPES = ("D0-M", "D1", "D2", "D3")
KNOWN_MODELS = CLASSICAL_MODELS + NEURAL_RECIPES

MODEL_FIT_STAGES = frozenset(("source_fit", "calibration_model_fit", "final_refit"))
SCALAR_STAGE = "scalar_calibration"
REUSE_STAGES = frozenset(("source_fit", "source_validation_prediction"))

KNOWN_STAGES = frozenset(
    (
        "source_fit",
        "source_validation_prediction",
        "select_hyperparameters",
        "calibration_model_fit",
        "calibration_validation_prediction",
        "calibration_prediction_alias",
        "scalar_calibration",
        "final_refit",
        "held_prediction",
        "seed_ensemble_prediction",
        "select_refit_epochs",
    )
)

ENUM_WORKER_KINDS = ("CPU", "GPU")
RESOURCE_FIELDS = (
    "active_seconds",
    "artifact_bytes",
    "rss_bytes",
    "gpu_bytes",
    "free_disk_bytes",
    "active_workers",
)

_INT64_MAX = 2**63 - 1

# Model-bearing stages run on a GPU for neural recipes and on a CPU for
# classical models; every other stage is a CPU lane operation.
PREDICTION_STAGES = frozenset(
    (
        "source_validation_prediction",
        "calibration_validation_prediction",
        "held_prediction",
    )
)
_GPU_MODEL_STAGES = MODEL_FIT_STAGES | PREDICTION_STAGES

# Inherited accounting baseline and shared approved bounds.  These are hard
# caps for this primitive, not a caller-selectable budget.
BASELINE_ACTIVE_SECONDS = 1357.694676884
BASELINE_ARTIFACT_BYTES = 64846625
HISTORICAL_OVERHEAD_ATTEMPTS = 5
MAX_FIT_TOTAL = 195207
MAX_UNIQUE_FIT_JOBS = 195202
MAX_SCALAR_ATTEMPTS = 3354
MAX_WALL_SECONDS = 172800
MAX_ARTIFACT_BYTES = 80 * 2**30
MAX_RAM_BYTES = 24 * 2**30
MAX_GPU_BYTES = 8 * 2**30
FREE_SPACE_FLOOR_BYTES = 30 * 2**30
MAX_CPU_WORKERS = 4
MAX_GPU_WORKERS = 1

# Expected/sealed reuse import for full U1.  Tests may monkeypatch these to
# exercise small synthetic graphs; the production API exposes no override.
EXPECTED_REUSE_FITS = 78
EXPECTED_REUSE_PREDICTIONS = 78
MAX_REUSE_FITS = 78
MAX_REUSE_PREDICTIONS = 78

# Optional immutable R1 recovery accounting profile.  When the caller binds a
# ``recovery_accounting`` dict into the binding these derived values replace the
# inherited defaults below; every scientific/resource cap stays untouched.
RECOVERY_ACCOUNTING_SCHEMA = "nato-sers-p08-u1-r1-accounting-v1"
RECOVERY_REPLAY_FIT_COUNT = 5
RECOVERY_ADDITIONAL_FIT_COUNT = 6
RECOVERY_EXPECTED_REUSE_PAIRS = 84
RECOVERY_HISTORICAL_OVERHEAD_ATTEMPTS = 10
RECOVERY_MAX_FIT_TOTAL = 195212
RECOVERY_MIN_ACTIVE_SECONDS = 1516
RECOVERY_MIN_ARTIFACT_BYTES = 1427760770
_RECOVERY_ACCOUNTING_FIELDS = frozenset(
    (
        "schema_version",
        "parent_binding_sha256",
        "parent_inventory_sha256",
        "replay_fit_job_ids",
        "additional_reuse_fit_job_ids",
        "baseline_active_seconds",
        "baseline_artifact_bytes",
    )
)

_HEX = frozenset("0123456789abcdef")
_TERMINAL_STATUSES = frozenset(("complete", "failed", "interrupted"))


class P08U1Error(RuntimeError):
    """Base class for ledger refusals."""


class ValidationError(P08U1Error):
    """Malformed caller input or an internally inconsistent ledger."""


class ReviewRequiredError(P08U1Error):
    """A prior failure or unclean session requires human review."""


class LockError(P08U1Error):
    """The exclusive lifetime lock is held by another owner."""


class BudgetError(P08U1Error):
    """A registered run limit was reached; new starts are blocked."""


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError("value_not_canonical_json") from exc


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value):
    return _sha256_text(_canonical(value))


def job_sha256(record):
    """Canonical sha256 of a registered job record excluding ``job_id``."""
    if not isinstance(record, dict):
        raise ValidationError("job_must_be_mapping")
    return _sha256_json({key: value for key, value in record.items() if key != "job_id"})


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _is_hex_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def _is_job_id(value):
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("P08JOB-")
        and all(character in _HEX for character in value[7:])
    )


def _normalize_job_id_list(value, expected_count, field):
    if not isinstance(value, list):
        raise ValidationError(field + "_invalid")
    items = list(value)
    if len(items) != expected_count:
        raise ValidationError(field + "_invalid")
    if any(not _is_job_id(item) for item in items):
        raise ValidationError(field + "_invalid")
    if len(set(items)) != len(items):
        raise ValidationError(field + "_duplicate")
    if items != sorted(items):
        raise ValidationError(field + "_not_sorted")
    return tuple(items)


def _validate_recovery_accounting(profile):
    if not isinstance(profile, dict):
        raise ValidationError("recovery_accounting_must_be_mapping")
    if set(profile.keys()) != _RECOVERY_ACCOUNTING_FIELDS:
        raise ValidationError("recovery_accounting_keys_invalid")
    if profile["schema_version"] != RECOVERY_ACCOUNTING_SCHEMA:
        raise ValidationError("recovery_accounting_schema_unsupported")
    for field in ("parent_binding_sha256", "parent_inventory_sha256"):
        if not _is_hex_digest(profile[field]):
            raise ValidationError(field + "_invalid")
    replay = _normalize_job_id_list(
        profile["replay_fit_job_ids"], RECOVERY_REPLAY_FIT_COUNT, "replay_fit_job_ids"
    )
    additional = _normalize_job_id_list(
        profile["additional_reuse_fit_job_ids"],
        RECOVERY_ADDITIONAL_FIT_COUNT,
        "additional_reuse_fit_job_ids",
    )
    if set(replay) & set(additional):
        raise ValidationError("recovery_accounting_lists_overlap")
    active = profile["baseline_active_seconds"]
    if (
        isinstance(active, bool)
        or not isinstance(active, (int, float))
        or not math.isfinite(active)
        or active < RECOVERY_MIN_ACTIVE_SECONDS
        or active >= MAX_WALL_SECONDS
    ):
        raise ValidationError("baseline_active_seconds_invalid")
    artifact = profile["baseline_artifact_bytes"]
    if (
        isinstance(artifact, bool)
        or not isinstance(artifact, int)
        or artifact < RECOVERY_MIN_ARTIFACT_BYTES
        or artifact >= MAX_ARTIFACT_BYTES
    ):
        raise ValidationError("baseline_artifact_bytes_invalid")
    return {
        "parent_binding_sha256": profile["parent_binding_sha256"],
        "parent_inventory_sha256": profile["parent_inventory_sha256"],
        "replay_fit_job_ids": replay,
        "additional_reuse_fit_job_ids": additional,
        "baseline_active_seconds": float(active),
        "baseline_artifact_bytes": int(artifact),
    }


def _default_accounting():
    return {
        "schema_version": "nato-sers-p08-u1-ledger-accounting-v1",
        "expected_reuse_fits": EXPECTED_REUSE_FITS,
        "expected_reuse_predictions": EXPECTED_REUSE_PREDICTIONS,
        "max_reuse_fits": MAX_REUSE_FITS,
        "max_reuse_predictions": MAX_REUSE_PREDICTIONS,
        "historical_overhead_attempts": HISTORICAL_OVERHEAD_ATTEMPTS,
        "max_fit_total": MAX_FIT_TOTAL,
        "baseline_active_seconds": float(BASELINE_ACTIVE_SECONDS),
        "baseline_artifact_bytes": int(BASELINE_ARTIFACT_BYTES),
        "replay_fit_job_ids": (),
        "additional_reuse_fit_job_ids": (),
        "parent_binding_sha256": None,
        "parent_inventory_sha256": None,
    }


def execution_accounting(binding):
    """Return the normalized accounting profile bound into ``binding``.

    Absent ``recovery_accounting`` yields the inherited defaults (live module
    constants so synthetic tests may still patch them).  A present profile is
    strictly validated and mapped to the fixed R1 derived accounting values.
    """
    if not isinstance(binding, dict):
        raise ValidationError("binding_invalid")
    if "recovery_accounting" not in binding:
        return _default_accounting()
    profile = _validate_recovery_accounting(binding["recovery_accounting"])
    return {
        "schema_version": RECOVERY_ACCOUNTING_SCHEMA,
        "expected_reuse_fits": RECOVERY_EXPECTED_REUSE_PAIRS,
        "expected_reuse_predictions": RECOVERY_EXPECTED_REUSE_PAIRS,
        "max_reuse_fits": RECOVERY_EXPECTED_REUSE_PAIRS,
        "max_reuse_predictions": RECOVERY_EXPECTED_REUSE_PAIRS,
        "historical_overhead_attempts": RECOVERY_HISTORICAL_OVERHEAD_ATTEMPTS,
        "max_fit_total": RECOVERY_MAX_FIT_TOTAL,
        "baseline_active_seconds": profile["baseline_active_seconds"],
        "baseline_artifact_bytes": profile["baseline_artifact_bytes"],
        "replay_fit_job_ids": profile["replay_fit_job_ids"],
        "additional_reuse_fit_job_ids": profile["additional_reuse_fit_job_ids"],
        "parent_binding_sha256": profile["parent_binding_sha256"],
        "parent_inventory_sha256": profile["parent_inventory_sha256"],
    }


class _Lock:
    def __init__(self, path):
        self._path = path
        self._fd = None

    def acquire(self):
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise LockError("lock_held_by_another_owner") from exc
        self._fd = fd

    def release(self):
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_json TEXT NOT NULL,
    job_sha TEXT NOT NULL,
    stage TEXT NOT NULL,
    model_id TEXT NOT NULL,
    policy_id TEXT NOT NULL,
    dependencies_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reuse (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id),
    kind TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    evidence_sha TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(job_id),
    worker_kind TEXT NOT NULL,
    status TEXT NOT NULL,
    receipt_json TEXT,
    receipt_sha TEXT,
    start_seq INTEGER,
    finish_seq INTEGER
);
CREATE TABLE IF NOT EXISTS resources (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    active_seconds REAL NOT NULL,
    artifact_bytes INTEGER NOT NULL,
    rss_bytes INTEGER NOT NULL,
    gpu_bytes INTEGER NOT NULL,
    free_disk_bytes INTEGER NOT NULL,
    active_workers INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    payload_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_status ON attempts(status);
"""


def _open_conn(path):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _meta_get(conn, key, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return default if row is None else row["value"]


def _meta_set(conn, key, value):
    conn.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def _append_event(conn, payload):
    payload_json = _canonical(payload)
    prev = _meta_get(conn, "event_head", "")
    event_hash = _sha256_text(prev + payload_json)
    cursor = conn.execute(
        "INSERT INTO events(payload_json,prev_hash,event_hash) VALUES(?,?,?)",
        (payload_json, prev, event_hash),
    )
    _meta_set(conn, "event_head", event_hash)
    return event_hash, cursor.lastrowid


def _validate_binding(binding):
    if not isinstance(binding, dict) or not binding:
        raise ValidationError("binding_invalid")
    if "recovery_accounting" in binding:
        _validate_recovery_accounting(binding["recovery_accounting"])
    return _canonical(binding)


def _validate_jobs(jobs):
    if not isinstance(jobs, (list, tuple)) or isinstance(jobs, (str, bytes)):
        raise ValidationError("jobs_must_be_sequence")
    seen = set()
    parsed = []
    for raw in jobs:
        if not isinstance(raw, dict):
            raise ValidationError("job_must_be_mapping")
        job_id = raw.get("job_id")
        if not isinstance(job_id, str) or not job_id.startswith("P08JOB-"):
            raise ValidationError("job_id_invalid")
        if raw.get("stage") not in KNOWN_STAGES:
            raise ValidationError("job_stage_invalid")
        if raw.get("model_id") not in KNOWN_MODELS:
            raise ValidationError("job_model_invalid")
        if raw.get("policy_id") not in ALLOWED_POLICIES:
            raise ValidationError("job_policy_not_permitted")
        dependencies = raw.get("dependencies")
        if not isinstance(dependencies, (list, tuple)) or isinstance(dependencies, (str, bytes)):
            raise ValidationError("job_dependencies_invalid")
        dependency_list = list(dependencies)
        if any(not isinstance(dep, str) for dep in dependency_list):
            raise ValidationError("job_dependency_invalid")
        if len(set(dependency_list)) != len(dependency_list):
            raise ValidationError("job_dependency_duplicate")
        if job_id != "P08JOB-" + job_sha256(raw):
            raise ValidationError("job_id_hash_mismatch")
        if job_id in seen:
            raise ValidationError("duplicate_job_id")
        seen.add(job_id)
        parsed.append(raw)
    for raw in parsed:
        for dep in raw["dependencies"]:
            if dep == raw["job_id"]:
                raise ValidationError("self_dependency")
            if dep not in seen:
                raise ValidationError("missing_dependency")
    parsed.sort(key=lambda record: record["job_id"])
    return parsed


def _validate_recovery_jobs(binding, parsed_jobs):
    if not isinstance(binding, dict) or "recovery_accounting" not in binding:
        return
    accounting = execution_accounting(binding)
    by_id = {record["job_id"]: record for record in parsed_jobs}
    listed = tuple(accounting["replay_fit_job_ids"]) + tuple(
        accounting["additional_reuse_fit_job_ids"]
    )
    for job_id in listed:
        record = by_id.get(job_id)
        if record is None:
            raise ValidationError("recovery_accounting_job_missing")
        if record["stage"] != "source_fit":
            raise ValidationError("recovery_accounting_job_stage_invalid")


def _validate_resources(snapshot):
    if not isinstance(snapshot, dict):
        raise ValidationError("resources_must_be_mapping")
    if set(snapshot.keys()) != set(RESOURCE_FIELDS):
        raise ValidationError("resources_keys_invalid")
    active = snapshot["active_seconds"]
    if (
        isinstance(active, bool)
        or not isinstance(active, (int, float))
        or not math.isfinite(active)
        or active < 0
    ):
        raise ValidationError("active_seconds_invalid")
    normalized = {"active_seconds": float(active)}
    for field in RESOURCE_FIELDS[1:]:
        value = snapshot[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValidationError(field + "_invalid")
        if value > _INT64_MAX:
            raise ValidationError(field + "_overflow")
        normalized[field] = int(value)
    return normalized


def _normalize_worker_kind(worker_kind):
    if not isinstance(worker_kind, str):
        raise ValidationError("worker_kind_invalid")
    kind = worker_kind.strip().upper()
    if kind not in ENUM_WORKER_KINDS:
        raise ValidationError("worker_kind_invalid")
    return kind


def _validate_receipt(receipt):
    if not isinstance(receipt, dict):
        raise ValidationError("receipt_must_be_mapping")
    if not _is_hex_digest(receipt.get("sha256")):
        raise ValidationError("receipt_sha256_invalid")
    return _canonical(receipt), _sha256_json(receipt)


def _cumulative_resources(conn, accounting):
    # Rows are persisted monotonic, so the newest row by primary-key lookup
    # already carries the running maximum.  No aggregate table scan is needed.
    row = conn.execute(
        "SELECT active_seconds,artifact_bytes FROM resources ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    baseline_active = float(accounting["baseline_active_seconds"])
    baseline_artifact = int(accounting["baseline_artifact_bytes"])
    if row is None:
        return baseline_active, baseline_artifact
    active = max(baseline_active, float(row["active_seconds"]))
    artifact = max(baseline_artifact, int(row["artifact_bytes"]))
    return active, artifact


def _check_resource_record_limits(conn, snapshot, accounting):
    active, artifact = _cumulative_resources(conn, accounting)
    if snapshot["active_seconds"] < active:
        raise ValidationError("active_seconds_regressed")
    if snapshot["artifact_bytes"] < artifact:
        raise ValidationError("artifact_bytes_regressed")
    if snapshot["active_seconds"] > MAX_WALL_SECONDS:
        raise BudgetError("wall_seconds_ceiling")
    if snapshot["artifact_bytes"] > MAX_ARTIFACT_BYTES:
        raise BudgetError("artifact_bytes_ceiling")
    if snapshot["rss_bytes"] > MAX_RAM_BYTES:
        raise BudgetError("ram_bytes_ceiling")
    if snapshot["gpu_bytes"] > MAX_GPU_BYTES:
        raise BudgetError("gpu_bytes_ceiling")
    remaining = MAX_ARTIFACT_BYTES - snapshot["artifact_bytes"]
    if snapshot["free_disk_bytes"] < FREE_SPACE_FLOOR_BYTES + remaining:
        raise BudgetError("free_space_ceiling")


def _check_resource_admission(conn, snapshot, accounting):
    # A snapshot exactly at an exact wall/artifact ceiling is still recordable
    # as final boundary usage, but no new effectful work may start once the
    # ledger is already exhausted.  RAM/GPU equal-cap is allowed.
    _check_resource_record_limits(conn, snapshot, accounting)
    if snapshot["active_seconds"] >= MAX_WALL_SECONDS:
        raise BudgetError("wall_seconds_ceiling")
    if snapshot["artifact_bytes"] >= MAX_ARTIFACT_BYTES:
        raise BudgetError("artifact_bytes_ceiling")


def _insert_resources(conn, snapshot):
    cursor = conn.execute(
        "INSERT INTO resources(active_seconds,artifact_bytes,rss_bytes,gpu_bytes,"
        "free_disk_bytes,active_workers) VALUES(?,?,?,?,?,?)",
        (
            snapshot["active_seconds"],
            snapshot["artifact_bytes"],
            snapshot["rss_bytes"],
            snapshot["gpu_bytes"],
            snapshot["free_disk_bytes"],
            snapshot["active_workers"],
        ),
    )
    return cursor.lastrowid


def _require_not_blocked(conn):
    if _meta_get(conn, "failure", "0") == "1":
        raise ReviewRequiredError("blocked_by_previous_failure")


def _count_reuse(conn, kind):
    key = "reuse_fit_count" if kind == "fit" else "reuse_prediction_count"
    return int(_meta_get(conn, key, "0"))


def _fit_attempt_count(conn):
    return int(_meta_get(conn, "fit_attempt_count", "0"))


def _scalar_attempt_count(conn):
    return int(_meta_get(conn, "scalar_attempt_count", "0"))


def _expected_worker_kind(stage, model_id):
    if stage in _GPU_MODEL_STAGES:
        return "GPU" if model_id in NEURAL_RECIPES else "CPU"
    return "CPU"


def _check_attempt_budgets(conn, stage, accounting):
    reuse_fits = _count_reuse(conn, "fit")
    new_fits = _fit_attempt_count(conn)
    new_scalar = _scalar_attempt_count(conn)
    prospective_fits = new_fits + (1 if stage in MODEL_FIT_STAGES else 0)
    overhead = accounting["historical_overhead_attempts"]
    if reuse_fits + prospective_fits + overhead > accounting["max_fit_total"]:
        raise BudgetError("fit_attempt_ceiling")
    if reuse_fits + prospective_fits > MAX_UNIQUE_FIT_JOBS:
        raise BudgetError("unique_fit_job_ceiling")
    prospective_scalar = new_scalar + (1 if stage == SCALAR_STAGE else 0)
    if prospective_scalar > MAX_SCALAR_ATTEMPTS:
        raise BudgetError("scalar_attempt_ceiling")


def _dependency_complete(conn, job_id):
    if conn.execute("SELECT 1 FROM reuse WHERE job_id=?", (job_id,)).fetchone():
        return True
    row = conn.execute("SELECT status FROM attempts WHERE job_id=?", (job_id,)).fetchone()
    return row is not None and row["status"] == "complete"


def _verify_registered_jobs(conn):
    for row in conn.execute(
        "SELECT job_id,job_json,job_sha,stage,model_id,policy_id,dependencies_json FROM jobs"
    ):
        record = json.loads(row["job_json"])
        if record.get("job_id") != row["job_id"]:
            raise ValidationError("registered_job_id_mismatch")
        if job_sha256(record) != row["job_sha"]:
            raise ValidationError("registered_job_hash_mismatch")
        if row["job_id"] != "P08JOB-" + job_sha256(record):
            raise ValidationError("registered_job_id_hash_mismatch")
        if _canonical(list(record.get("dependencies", []))) != row["dependencies_json"]:
            raise ValidationError("registered_job_dependencies_mismatch")
        if (
            record.get("stage") != row["stage"]
            or record.get("model_id") != row["model_id"]
            or record.get("policy_id") != row["policy_id"]
        ):
            raise ValidationError("registered_job_index_mismatch")


def _verify_events(conn, binding_json=None):
    prev = ""
    count = 0
    reuse_jobs = set()
    reuse_evidence = {}
    attempt_state = {}
    resources_by_seq = {}
    sealed = False
    failure = False
    state = "open"
    clean = False
    create_payload = None
    fit_counter = 0
    scalar_counter = 0
    reuse_fit = 0
    reuse_pred = 0
    for row in conn.execute(
        "SELECT seq,payload_json,prev_hash,event_hash FROM events ORDER BY seq"
    ):
        if row["prev_hash"] != prev:
            raise ValidationError("event_chain_prev_mismatch")
        computed = _sha256_text(prev + row["payload_json"])
        if computed != row["event_hash"]:
            raise ValidationError("event_chain_hash_mismatch")
        prev = computed
        count += 1
        payload = json.loads(row["payload_json"])
        event_type = payload.get("type")
        if event_type == "create":
            create_payload = payload
        elif event_type == "reuse":
            job_id = payload["job_id"]
            reuse_jobs.add(job_id)
            reuse_evidence[job_id] = (payload["kind"], payload["evidence_sha256"])
            if payload["kind"] == "fit":
                reuse_fit += 1
            else:
                reuse_pred += 1
        elif event_type == "seal_reuse":
            sealed = True
        elif event_type == "start":
            job_id = payload["job_id"]
            stage = payload["stage"]
            attempt_state[job_id] = {
                "worker_kind": payload["worker_kind"],
                "status": "running",
                "start_seq": row["seq"],
                "finish_seq": None,
                "receipt_sha": None,
            }
            if stage in MODEL_FIT_STAGES:
                fit_counter += 1
            elif stage == SCALAR_STAGE:
                scalar_counter += 1
            if payload.get("resource_seq") is not None:
                resources_by_seq[payload["resource_seq"]] = payload["resources"]
        elif event_type == "finish":
            job_id = payload["job_id"]
            entry = attempt_state.get(job_id)
            if entry is None:
                raise ValidationError("finish_without_start")
            entry["status"] = payload["status"]
            entry["finish_seq"] = row["seq"]
            entry["receipt_sha"] = payload["receipt_sha256"]
        elif event_type == "resources":
            resources_by_seq[payload["resource_seq"]] = payload["resources"]
        elif event_type == "block":
            failure = True
            if payload.get("resource_seq") is not None:
                resources_by_seq[payload["resource_seq"]] = payload["resources"]
        elif event_type == "close":
            clean = bool(payload.get("clean"))
            state = "closed" if clean else "open"
        elif event_type == "open":
            state = "open"
            clean = False
    if _meta_get(conn, "event_head", "") != prev:
        raise ValidationError("event_head_mismatch")
    if create_payload is None:
        raise ValidationError("create_event_missing")
    stored_binding_json = _meta_get(conn, "binding_json")
    stored_binding_sha = _meta_get(conn, "binding_sha256")
    if stored_binding_json is None or _sha256_text(stored_binding_json) != stored_binding_sha:
        raise ValidationError("binding_ledger_mismatch")
    if create_payload.get("binding_sha256") != stored_binding_sha:
        raise ValidationError("create_binding_mismatch")
    if binding_json is not None and binding_json != stored_binding_json:
        raise ValidationError("binding_mismatch")
    job_ids = sorted(r["job_id"] for r in conn.execute("SELECT job_id FROM jobs"))
    if int(create_payload.get("job_count", -1)) != len(job_ids):
        raise ValidationError("create_job_count_mismatch")
    if create_payload.get("jobs_sha256") != _sha256_json(job_ids):
        raise ValidationError("create_jobs_digest_mismatch")
    table_reuse = {
        r["job_id"]: r
        for r in conn.execute("SELECT job_id,kind,evidence_json,evidence_sha FROM reuse")
    }
    if set(table_reuse) != reuse_jobs:
        raise ValidationError("reuse_ledger_mismatch")
    for job_id, (kind, evidence_sha) in reuse_evidence.items():
        r = table_reuse[job_id]
        if r["kind"] != kind or r["evidence_sha"] != evidence_sha:
            raise ValidationError("reuse_ledger_mismatch")
        if _sha256_json(json.loads(r["evidence_json"])) != r["evidence_sha"]:
            raise ValidationError("reuse_ledger_mismatch")
    table_attempts = {
        r["job_id"]: r
        for r in conn.execute(
            "SELECT job_id,worker_kind,status,receipt_json,receipt_sha,start_seq,"
            "finish_seq FROM attempts"
        )
    }
    if set(table_attempts) != set(attempt_state):
        raise ValidationError("attempt_ledger_mismatch")
    for job_id, expected in attempt_state.items():
        r = table_attempts[job_id]
        if (
            r["worker_kind"] != expected["worker_kind"]
            or r["status"] != expected["status"]
            or r["start_seq"] != expected["start_seq"]
            or r["finish_seq"] != expected["finish_seq"]
        ):
            raise ValidationError("attempt_ledger_mismatch")
        if expected["receipt_sha"] is None:
            if r["receipt_sha"] is not None or r["receipt_json"] is not None:
                raise ValidationError("receipt_ledger_mismatch")
        else:
            if r["receipt_sha"] != expected["receipt_sha"] or r["receipt_json"] is None:
                raise ValidationError("receipt_ledger_mismatch")
            if _sha256_json(json.loads(r["receipt_json"])) != r["receipt_sha"]:
                raise ValidationError("receipt_ledger_mismatch")
    actual_resources = {r["seq"]: r for r in conn.execute("SELECT * FROM resources")}
    if set(actual_resources) != set(resources_by_seq):
        raise ValidationError("resource_ledger_mismatch")
    for seq, snapshot in resources_by_seq.items():
        r = actual_resources[seq]
        for field in RESOURCE_FIELDS:
            if field == "active_seconds":
                if float(r[field]) != float(snapshot[field]):
                    raise ValidationError("resource_ledger_mismatch")
            elif int(r[field]) != int(snapshot[field]):
                raise ValidationError("resource_ledger_mismatch")
    if int(_meta_get(conn, "fit_attempt_count", "0")) != fit_counter:
        raise ValidationError("fit_counter_mismatch")
    if int(_meta_get(conn, "scalar_attempt_count", "0")) != scalar_counter:
        raise ValidationError("scalar_counter_mismatch")
    if int(_meta_get(conn, "reuse_fit_count", "0")) != reuse_fit:
        raise ValidationError("reuse_fit_counter_mismatch")
    if int(_meta_get(conn, "reuse_prediction_count", "0")) != reuse_pred:
        raise ValidationError("reuse_prediction_counter_mismatch")
    if (_meta_get(conn, "sealed", "0") == "1") != sealed:
        raise ValidationError("seal_ledger_mismatch")
    if (_meta_get(conn, "failure", "0") == "1") != failure:
        raise ValidationError("failure_ledger_mismatch")
    if _meta_get(conn, "state", "open") != state:
        raise ValidationError("state_ledger_mismatch")
    if (_meta_get(conn, "clean", "0") == "1") != clean:
        raise ValidationError("clean_ledger_mismatch")
    return {"ok": True, "events": count, "event_head": prev}


class P08U1Store:
    """Durable single-controller governance ledger for a future U1 run."""

    def __init__(self, directory, conn, lock, binding):
        self._directory = directory
        self._conn = conn
        self._lock = lock
        self._accounting = execution_accounting(binding)
        self._closed = False
        self._persistence_failed = False

    @staticmethod
    def _db_path(directory):
        return os.path.join(directory, "ledger.sqlite3")

    @staticmethod
    def _lock_path(directory):
        return os.path.join(directory, "ledger.lock")

    @classmethod
    def create(cls, directory, binding, jobs):
        directory = os.path.abspath(directory)
        binding_json = _validate_binding(binding)
        binding_sha = _sha256_text(binding_json)
        parsed_jobs = _validate_jobs(jobs)
        _validate_recovery_jobs(binding, parsed_jobs)
        if os.path.exists(directory):
            raise ValidationError("directory_already_exists")
        os.makedirs(directory, mode=0o700)
        lock = _Lock(cls._lock_path(directory))
        conn = None
        try:
            lock.acquire()
            conn = _open_conn(cls._db_path(directory))
            conn.executescript(_SCHEMA)
            conn.execute("BEGIN IMMEDIATE")
            _meta_set(conn, "schema", SCHEMA_VERSION)
            _meta_set(conn, "binding_json", binding_json)
            _meta_set(conn, "binding_sha256", binding_sha)
            _meta_set(conn, "state", "open")
            _meta_set(conn, "clean", "0")
            _meta_set(conn, "failure", "0")
            _meta_set(conn, "sealed", "0")
            _meta_set(conn, "block_reason", "")
            _meta_set(conn, "event_head", "")
            _meta_set(conn, "created_utc", _utc_now())
            _meta_set(conn, "fit_attempt_count", "0")
            _meta_set(conn, "scalar_attempt_count", "0")
            _meta_set(conn, "reuse_fit_count", "0")
            _meta_set(conn, "reuse_prediction_count", "0")
            for record in parsed_jobs:
                conn.execute(
                    "INSERT INTO jobs(job_id,job_json,job_sha,stage,model_id,policy_id,"
                    "dependencies_json) VALUES(?,?,?,?,?,?,?)",
                    (
                        record["job_id"],
                        _canonical(record),
                        job_sha256(record),
                        record["stage"],
                        record["model_id"],
                        record["policy_id"],
                        _canonical(list(record["dependencies"])),
                    ),
                )
            _append_event(
                conn,
                {
                    "type": "create",
                    "binding_sha256": binding_sha,
                    "job_count": len(parsed_jobs),
                    "jobs_sha256": _sha256_json(sorted(record["job_id"] for record in parsed_jobs)),
                },
            )
            conn.execute("COMMIT")
            return cls(directory, conn, lock, binding)
        except BaseException:
            if conn is not None:
                try:
                    conn.close()
                except BaseException:
                    pass
            try:
                lock.release()
            except BaseException:
                pass
            raise

    @classmethod
    def reopen(cls, directory, binding):
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            raise ValidationError("directory_missing")
        binding_json = _validate_binding(binding)
        binding_sha = _sha256_text(binding_json)
        lock = _Lock(cls._lock_path(directory))
        conn = None
        try:
            lock.acquire()
            conn = _open_conn(cls._db_path(directory))
            if _meta_get(conn, "schema") != SCHEMA_VERSION:
                raise ValidationError("schema_mismatch")
            store = cls(directory, conn, lock, binding)
            _verify_events(conn, binding_json)
            _verify_registered_jobs(conn)
            if _meta_get(conn, "binding_sha256") != binding_sha:
                raise ValidationError("binding_mismatch")
            if _meta_get(conn, "state") != "closed" or _meta_get(conn, "clean", "0") != "1":
                raise ReviewRequiredError("previous_session_not_cleanly_closed")
            running = conn.execute(
                "SELECT COUNT(*) AS c FROM attempts WHERE status='running'"
            ).fetchone()["c"]
            if running:
                raise ReviewRequiredError("running_attempts_present")
            conn.execute("BEGIN IMMEDIATE")
            _meta_set(conn, "state", "open")
            _meta_set(conn, "clean", "0")
            _append_event(conn, {"type": "open"})
            conn.execute("COMMIT")
            return store
        except BaseException:
            if conn is not None:
                try:
                    conn.close()
                except BaseException:
                    pass
            try:
                lock.release()
            except BaseException:
                pass
            raise

    def _require_open(self):
        if self._closed:
            raise ValidationError("store_not_open")
        if self._persistence_failed:
            raise ReviewRequiredError("persistence_failure_latched")

    def _latch_persistence_failure(self):
        self._persistence_failed = True

    def _abort_transaction(self, exc):
        try:
            self._conn.execute("ROLLBACK")
        except BaseException:
            pass
        if isinstance(exc, sqlite3.Error):
            self._latch_persistence_failure()

    def _persist_block(self, reason, snapshot=None):
        conn = self._conn
        try:
            conn.execute("BEGIN IMMEDIATE")
            _meta_set(conn, "failure", "1")
            _meta_set(conn, "block_reason", reason)
            payload = {"type": "block", "reason": reason}
            if snapshot is not None:
                resource_seq = _insert_resources(conn, snapshot)
                payload["resource_seq"] = resource_seq
                payload["resources"] = snapshot
            _append_event(conn, payload)
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except BaseException:
                pass
            self._latch_persistence_failure()
            raise

    def record_reuse(self, job_id, evidence):
        self._require_open()
        conn = self._conn
        accounting = self._accounting
        try:
            conn.execute("BEGIN IMMEDIATE")
            _require_not_blocked(conn)
            if _meta_get(conn, "sealed", "0") == "1":
                raise ValidationError("reuse_import_sealed")
            if job_id in accounting["replay_fit_job_ids"]:
                raise ValidationError("replay_fit_cannot_be_reused")
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise ValidationError("unknown_job_id")
            if row["stage"] not in REUSE_STAGES:
                raise ValidationError("stage_not_reusable")
            if conn.execute("SELECT COUNT(*) AS c FROM attempts").fetchone()["c"]:
                raise ValidationError("reuse_after_new_start")
            if conn.execute("SELECT 1 FROM reuse WHERE job_id=?", (job_id,)).fetchone():
                raise ValidationError("duplicate_reuse")
            kind = "fit" if row["stage"] == "source_fit" else "prediction"
            if kind == "fit" and _count_reuse(conn, "fit") >= accounting["max_reuse_fits"]:
                raise ValidationError("reuse_fit_ceiling")
            if (
                kind == "prediction"
                and _count_reuse(conn, "prediction") >= accounting["max_reuse_predictions"]
            ):
                raise ValidationError("reuse_prediction_ceiling")
            if kind == "prediction":
                deps = json.loads(row["dependencies_json"])
                if len(deps) != 1:
                    raise ValidationError("reuse_prediction_dependency_invalid")
                parent = deps[0]
                if not conn.execute("SELECT 1 FROM reuse WHERE job_id=?", (parent,)).fetchone():
                    raise ValidationError("reuse_parent_missing")
                parent_row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (parent,)).fetchone()
                if parent_row is None or parent_row["stage"] != "source_fit":
                    raise ValidationError("reuse_prediction_dependency_invalid")
                if (
                    parent_row["policy_id"] != row["policy_id"]
                    or parent_row["model_id"] != row["model_id"]
                ):
                    raise ValidationError("reuse_prediction_context_mismatch")
                parent_record = json.loads(parent_row["job_json"])
                record = json.loads(row["job_json"])
                for field in ("seed", "candidate_id", "context_id", "dataset_id", "fold"):
                    if field in parent_record or field in record:
                        if parent_record.get(field) != record.get(field):
                            raise ValidationError("reuse_prediction_context_mismatch")
            if not isinstance(evidence, dict):
                raise ValidationError("reuse_evidence_must_be_mapping")
            if evidence.get("job_id") != job_id:
                raise ValidationError("reuse_evidence_job_mismatch")
            if evidence.get("job_sha256") != row["job_sha"]:
                raise ValidationError("reuse_evidence_job_sha_mismatch")
            evidence_json = _canonical(evidence)
            evidence_sha = _sha256_json(evidence)
            conn.execute(
                "INSERT INTO reuse(job_id,kind,evidence_json,evidence_sha) VALUES(?,?,?,?)",
                (job_id, kind, evidence_json, evidence_sha),
            )
            key = "reuse_fit_count" if kind == "fit" else "reuse_prediction_count"
            _meta_set(conn, key, str(_count_reuse(conn, kind) + 1))
            _append_event(
                conn,
                {"type": "reuse", "job_id": job_id, "kind": kind, "evidence_sha256": evidence_sha},
            )
            conn.execute("COMMIT")
            return {"job_id": job_id, "kind": kind, "evidence_sha256": evidence_sha}
        except BaseException as exc:
            self._abort_transaction(exc)
            raise

    def seal_reuse(self):
        self._require_open()
        conn = self._conn
        accounting = self._accounting
        try:
            conn.execute("BEGIN IMMEDIATE")
            fits = _count_reuse(conn, "fit")
            predictions = _count_reuse(conn, "prediction")
            if (
                fits != accounting["expected_reuse_fits"]
                or predictions != accounting["expected_reuse_predictions"]
            ):
                raise ValidationError("reuse_import_incomplete")
            additional = set(accounting["additional_reuse_fit_job_ids"])
            if additional:
                imported_fits = {
                    row["job_id"]
                    for row in conn.execute("SELECT job_id FROM reuse WHERE kind='fit'")
                }
                if not additional <= imported_fits:
                    raise ValidationError("recovery_additional_fits_missing")
                if imported_fits & set(accounting["replay_fit_job_ids"]):
                    raise ValidationError("recovery_replay_fit_imported")
            if _meta_get(conn, "sealed", "0") != "1":
                _meta_set(conn, "sealed", "1")
                _append_event(
                    conn, {"type": "seal_reuse", "fits": fits, "predictions": predictions}
                )
            conn.execute("COMMIT")
            return {"sealed": True, "fits": fits, "predictions": predictions}
        except BaseException as exc:
            self._abort_transaction(exc)
            raise

    def record_resources(self, resources):
        self._require_open()
        snapshot = _validate_resources(resources)
        conn = self._conn
        try:
            conn.execute("BEGIN IMMEDIATE")
            _check_resource_record_limits(conn, snapshot, self._accounting)
            resource_seq = _insert_resources(conn, snapshot)
            _append_event(
                conn,
                {
                    "type": "resources",
                    "resource_seq": resource_seq,
                    "resources": snapshot,
                },
            )
            conn.execute("COMMIT")
        except BudgetError as exc:
            self._abort_transaction(exc)
            self._persist_block(str(exc), snapshot)
            raise
        except BaseException as exc:
            self._abort_transaction(exc)
            raise
        return {
            "active_seconds": snapshot["active_seconds"],
            "artifact_bytes": snapshot["artifact_bytes"],
        }

    def start(self, job_id, worker_kind, resources):
        self._require_open()
        kind = _normalize_worker_kind(worker_kind)
        snapshot = _validate_resources(resources)
        conn = self._conn
        try:
            conn.execute("BEGIN IMMEDIATE")
            _require_not_blocked(conn)
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise ValidationError("unknown_job_id")
            if conn.execute("SELECT 1 FROM reuse WHERE job_id=?", (job_id,)).fetchone():
                raise ValidationError("job_already_reused")
            if _meta_get(conn, "sealed", "0") != "1":
                raise ValidationError("reuse_import_not_sealed")
            if conn.execute("SELECT 1 FROM attempts WHERE job_id=?", (job_id,)).fetchone():
                raise ValidationError("job_already_attempted")
            if kind != _expected_worker_kind(row["stage"], row["model_id"]):
                raise ValidationError("worker_kind_stage_mismatch")
            for dep in json.loads(row["dependencies_json"]):
                if not _dependency_complete(conn, dep):
                    raise ValidationError("dependency_incomplete")
            running = conn.execute(
                "SELECT COUNT(*) AS c FROM attempts WHERE status='running' AND worker_kind=?",
                (kind,),
            ).fetchone()["c"]
            cap = MAX_CPU_WORKERS if kind == "CPU" else MAX_GPU_WORKERS
            if int(running) >= cap:
                raise ValidationError("worker_slot_unavailable")
            _check_resource_admission(conn, snapshot, self._accounting)
            _check_attempt_budgets(conn, row["stage"], self._accounting)
            cursor = conn.execute(
                "INSERT INTO attempts(job_id,worker_kind,status) VALUES(?,?, 'running')",
                (job_id, kind),
            )
            attempt_id = cursor.lastrowid
            resource_seq = _insert_resources(conn, snapshot)
            if row["stage"] in MODEL_FIT_STAGES:
                _meta_set(conn, "fit_attempt_count", str(_fit_attempt_count(conn) + 1))
            elif row["stage"] == SCALAR_STAGE:
                _meta_set(conn, "scalar_attempt_count", str(_scalar_attempt_count(conn) + 1))
            _, seq = _append_event(
                conn,
                {
                    "type": "start",
                    "job_id": job_id,
                    "worker_kind": kind,
                    "attempt_id": attempt_id,
                    "stage": row["stage"],
                    "resource_seq": resource_seq,
                    "resources": snapshot,
                },
            )
            conn.execute("UPDATE attempts SET start_seq=? WHERE attempt_id=?", (seq, attempt_id))
            conn.execute("COMMIT")
            return {
                "job_id": job_id,
                "attempt_id": attempt_id,
                "status": "running",
                "worker_kind": kind,
            }
        except BudgetError as exc:
            self._abort_transaction(exc)
            self._persist_block(str(exc), snapshot)
            raise
        except BaseException as exc:
            self._abort_transaction(exc)
            raise

    def finish(self, job_id, status, receipt):
        self._require_open()
        if status not in _TERMINAL_STATUSES:
            raise ValidationError("invalid_finish_status")
        receipt_json, receipt_sha = _validate_receipt(receipt)
        conn = self._conn
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status FROM attempts WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise ValidationError("unknown_attempt")
            if row["status"] != "running":
                raise ValidationError("attempt_not_running")
            _, seq = _append_event(
                conn,
                {
                    "type": "finish",
                    "job_id": job_id,
                    "status": status,
                    "receipt_sha256": receipt_sha,
                },
            )
            conn.execute(
                "UPDATE attempts SET status=?,receipt_json=?,receipt_sha=?,finish_seq=? "
                "WHERE job_id=?",
                (status, receipt_json, receipt_sha, seq, job_id),
            )
            if status != "complete":
                _meta_set(conn, "failure", "1")
                _meta_set(conn, "block_reason", "operation_" + status)
                _append_event(conn, {"type": "block", "reason": "operation_" + status})
            conn.execute("COMMIT")
            return {"job_id": job_id, "status": status, "receipt_sha256": receipt_sha}
        except BaseException as exc:
            self._abort_transaction(exc)
            raise

    def verify_events(self):
        self._require_open()
        return _verify_events(self._conn)

    def status(self):
        self._require_open()
        conn = self._conn
        jobs = conn.execute(
            "SELECT job_id,stage,model_id,policy_id FROM jobs ORDER BY job_id"
        ).fetchall()
        jobs_by_stage = {}
        for row in jobs:
            jobs_by_stage[row["stage"]] = jobs_by_stage.get(row["stage"], 0) + 1
        attempts = {}
        for row in conn.execute(
            "SELECT job_id,worker_kind,status,receipt_sha FROM attempts ORDER BY job_id"
        ):
            attempts[row["job_id"]] = {
                "worker_kind": row["worker_kind"],
                "status": row["status"],
                "receipt_sha256": row["receipt_sha"],
            }
        reuse = {"fits": 0, "predictions": 0, "job_ids": []}
        for row in conn.execute("SELECT job_id,kind FROM reuse ORDER BY job_id"):
            if row["kind"] == "fit":
                reuse["fits"] += 1
            else:
                reuse["predictions"] += 1
            reuse["job_ids"].append(row["job_id"])
        active, artifact = _cumulative_resources(conn, self._accounting)
        accounted_fit_attempts = (
            reuse["fits"]
            + _fit_attempt_count(conn)
            + self._accounting["historical_overhead_attempts"]
        )
        return {
            "state": _meta_get(conn, "state"),
            "clean": _meta_get(conn, "clean", "0") == "1",
            "failure": _meta_get(conn, "failure", "0") == "1",
            "sealed": _meta_get(conn, "sealed", "0") == "1",
            "block_reason": _meta_get(conn, "block_reason", ""),
            "binding_sha256": _meta_get(conn, "binding_sha256"),
            "event_head": _meta_get(conn, "event_head", ""),
            "event_count": conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"],
            "jobs_total": len(jobs),
            "jobs_by_stage": dict(sorted(jobs_by_stage.items())),
            "attempts": attempts,
            "attempts_total": len(attempts),
            "running_attempts": sum(
                1 for attempt in attempts.values() if attempt["status"] == "running"
            ),
            "reuse": reuse,
            "cumulative_active_seconds": active,
            "cumulative_artifact_bytes": artifact,
            "accounted_model_fit_attempts": accounted_fit_attempts,
        }

    def public_summary(self):
        self._require_open()
        conn = self._conn
        jobs_by_stage = {
            row["stage"]: row["c"]
            for row in conn.execute("SELECT stage,COUNT(*) AS c FROM jobs GROUP BY stage")
        }
        status_counts = {
            row["status"]: row["c"]
            for row in conn.execute("SELECT status,COUNT(*) AS c FROM attempts GROUP BY status")
        }
        active, artifact = _cumulative_resources(conn, self._accounting)
        reuse_fits = _count_reuse(conn, "fit")
        accounted_fit_attempts = (
            reuse_fits + _fit_attempt_count(conn) + self._accounting["historical_overhead_attempts"]
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "binding_sha256": _meta_get(conn, "binding_sha256"),
            "state": _meta_get(conn, "state"),
            "flags": {
                "clean": _meta_get(conn, "clean", "0") == "1",
                "failure": _meta_get(conn, "failure", "0") == "1",
                "sealed": _meta_get(conn, "sealed", "0") == "1",
            },
            "event_head": _meta_get(conn, "event_head", ""),
            "event_count": conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"],
            "jobs_total": sum(jobs_by_stage.values()),
            "jobs_by_stage": dict(sorted(jobs_by_stage.items())),
            "attempts_total": conn.execute("SELECT COUNT(*) AS c FROM attempts").fetchone()["c"],
            "attempt_status_counts": dict(sorted(status_counts.items())),
            "running_attempts": conn.execute(
                "SELECT COUNT(*) AS c FROM attempts WHERE status='running'"
            ).fetchone()["c"],
            "reuse_fits": reuse_fits,
            "reuse_predictions": _count_reuse(conn, "prediction"),
            "accounted_model_fit_attempts": accounted_fit_attempts,
            "cumulative_active_seconds": active,
            "cumulative_artifact_bytes": artifact,
            "budgets": {
                "max_fit_total": self._accounting["max_fit_total"],
                "max_unique_fit_jobs": MAX_UNIQUE_FIT_JOBS,
                "max_scalar_attempts": MAX_SCALAR_ATTEMPTS,
                "expected_reuse_fits": self._accounting["expected_reuse_fits"],
                "expected_reuse_predictions": self._accounting["expected_reuse_predictions"],
                "historical_overhead_attempts": self._accounting["historical_overhead_attempts"],
                "max_wall_seconds": MAX_WALL_SECONDS,
                "max_artifact_bytes": MAX_ARTIFACT_BYTES,
                "max_ram_bytes": MAX_RAM_BYTES,
                "max_gpu_bytes": MAX_GPU_BYTES,
                "free_space_floor_bytes": FREE_SPACE_FLOOR_BYTES,
                "max_cpu_workers": MAX_CPU_WORKERS,
                "max_gpu_workers": MAX_GPU_WORKERS,
            },
        }

    def close(self):
        if self._closed:
            return {"clean": None, "already_closed": True}
        conn = self._conn
        clean = None
        primary = None
        try:
            running = conn.execute(
                "SELECT COUNT(*) AS c FROM attempts WHERE status='running'"
            ).fetchone()["c"]
            clean = int(running) == 0
            conn.execute("BEGIN IMMEDIATE")
            if clean:
                _meta_set(conn, "state", "closed")
                _meta_set(conn, "clean", "1")
                _append_event(conn, {"type": "close", "clean": True})
            else:
                _meta_set(conn, "clean", "0")
                _append_event(conn, {"type": "close", "clean": False, "running": int(running)})
            conn.execute("COMMIT")
        except BaseException as exc:
            primary = exc
            try:
                conn.execute("ROLLBACK")
            except BaseException:
                pass
        finally:
            self._closed = True
            close_error = None
            try:
                conn.close()
            except BaseException as exc:
                close_error = exc
            lock_error = None
            try:
                self._lock.release()
            except BaseException as exc:
                lock_error = exc
            if primary is not None:
                raise primary
            if close_error is not None:
                raise close_error
            if lock_error is not None:
                raise lock_error
        return {"clean": clean, "already_closed": False}
