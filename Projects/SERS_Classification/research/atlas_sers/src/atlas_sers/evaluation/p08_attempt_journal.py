"""P08-T037 pure attempt-journal replay kernel.

Reconstructs cumulative attempted work from invented, hash-linked session
events.  This is a pure logical/structural validator: it never authorizes
execution and never authenticates durable storage, leases, live owners or the
official frozen-U0 manifest.  The returned restart_allowed flag is not a
recovery lease.
"""

from __future__ import annotations

from .p08_qc_blocks import canonical_sha256

__all__ = [
    "JournalError",
    "replay_attempt_journal",
    "require_scientific_execution",
]

_MANIFEST_SCHEMA = "nato-sers-p08-attempt-manifest-v1"
_SUMMARY_SCHEMA = "nato-sers-p08-attempt-summary-v1"

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "proposal_sha256",
        "jobs",
        "manifest_sha256",
    }
)
_JOB_KEYS = frozenset({"job_id", "stage", "worker", "dependencies"})
_EVENT_KEYS = frozenset(
    {
        "seq",
        "previous_sha256",
        "session_id",
        "event_type",
        "elapsed_ns",
        "artifact_bytes",
        "job_id",
        "status",
        "receipt_sha256",
        "event_sha256",
    }
)
_STAGES = frozenset({"source_fit", "source_validation_prediction"})
_WORKERS = frozenset({"cpu", "gpu"})
_EVENT_TYPES = frozenset(
    {"session_open", "progress", "attempt_start", "attempt_finish", "session_close"}
)
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "interrupted"})
_REASON_CODES = frozenset(
    {
        "invalid_journal_input",
        "invalid_manifest",
        "manifest_hash_mismatch",
        "invalid_event",
        "event_hash_mismatch",
        "journal_head_mismatch",
        "invalid_transition",
        "scientific_execution_not_authorized",
    }
)
_HEX_DIGITS = frozenset("0123456789abcdef")


class JournalError(ValueError):
    """Static, data-free journal validation failure."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "invalid_journal_input"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _fail(reason_code):
    raise JournalError(reason_code) from None


def _is_lower64(value):
    if type(value) is not str or len(value) != 64:
        return False
    for char in value:
        if char not in _HEX_DIGITS:
            return False
    return True


def _is_clean_id(value):
    return type(value) is str and value != "" and value == value.strip()


def _has_exact_keys(obj, expected):
    if type(obj) is not dict:
        return False
    count = 0
    for key in obj:
        if type(key) is not str:
            return False
        count += 1
    if count != len(expected):
        return False
    return set(obj) == expected


def _canonical_digest(value, reason_code="invalid_journal_input"):
    try:
        digest = canonical_sha256(value)
    except JournalError:
        raise
    except Exception:
        raise JournalError(reason_code) from None
    if not _is_lower64(digest):
        raise JournalError(reason_code) from None
    return digest


def _validate_job_shallow(job):
    if not _has_exact_keys(job, _JOB_KEYS):
        _fail("invalid_manifest")
    if not _is_clean_id(job["job_id"]):
        _fail("invalid_manifest")
    stage = job["stage"]
    if type(stage) is not str or stage not in _STAGES:
        _fail("invalid_manifest")
    worker = job["worker"]
    if type(worker) is not str or worker not in _WORKERS:
        _fail("invalid_manifest")
    deps = job["dependencies"]
    if type(deps) is not list:
        _fail("invalid_manifest")
    for dep in deps:
        if not _is_clean_id(dep):
            _fail("invalid_manifest")


def _validate_jobs_semantic(jobs):
    previous_id = None
    fits = {}
    predictions = {}
    jobs_by_id = {}
    for job in jobs:
        job_id = job["job_id"]
        if previous_id is not None and not job_id > previous_id:
            _fail("invalid_manifest")
        previous_id = job_id
        jobs_by_id[job_id] = job
        if job["stage"] == "source_fit":
            fits[job_id] = job
        else:
            predictions[job_id] = job

    if not 1 <= len(fits) <= 78:
        _fail("invalid_manifest")
    if len(predictions) != len(fits):
        _fail("invalid_manifest")

    for job in fits.values():
        if job["dependencies"]:
            _fail("invalid_manifest")

    paired = {}
    for job in predictions.values():
        deps = job["dependencies"]
        if len(deps) != 1:
            _fail("invalid_manifest")
        dependency = deps[0]
        fit = fits.get(dependency)
        if fit is None:
            _fail("invalid_manifest")
        if fit["worker"] != job["worker"]:
            _fail("invalid_manifest")
        paired[dependency] = paired.get(dependency, 0) + 1

    for fit_id in fits:
        if paired.get(fit_id, 0) != 1:
            _fail("invalid_manifest")
    return jobs_by_id


def _validate_manifest(manifest, expected_manifest_sha256):
    if not _has_exact_keys(manifest, _MANIFEST_KEYS):
        _fail("invalid_manifest")
    schema_version = manifest["schema_version"]
    if type(schema_version) is not str or schema_version != _MANIFEST_SCHEMA:
        _fail("invalid_manifest")
    if manifest["execution_authorized"] is not False:
        _fail("invalid_manifest")
    if not _is_lower64(manifest["proposal_sha256"]):
        _fail("invalid_manifest")
    if not _is_lower64(manifest["manifest_sha256"]):
        _fail("invalid_manifest")
    jobs = manifest["jobs"]
    if type(jobs) is not list:
        _fail("invalid_manifest")
    for job in jobs:
        _validate_job_shallow(job)

    payload = {
        "schema_version": manifest["schema_version"],
        "execution_authorized": manifest["execution_authorized"],
        "proposal_sha256": manifest["proposal_sha256"],
        "jobs": jobs,
    }
    digest = _canonical_digest(payload, "invalid_manifest")
    if manifest["manifest_sha256"] != digest:
        _fail("manifest_hash_mismatch")
    if digest != expected_manifest_sha256:
        _fail("manifest_hash_mismatch")
    return _validate_jobs_semantic(jobs), digest


def _validate_event_shape(event, index):
    if not _has_exact_keys(event, _EVENT_KEYS):
        _fail("invalid_event")
    seq = event["seq"]
    if type(seq) is not int or seq != index + 1:
        _fail("invalid_event")
    if not _is_lower64(event["previous_sha256"]):
        _fail("invalid_event")
    if not _is_lower64(event["event_sha256"]):
        _fail("invalid_event")
    session_id = event["session_id"]
    if type(session_id) is not int or session_id <= 0:
        _fail("invalid_event")
    event_type = event["event_type"]
    if type(event_type) is not str or event_type not in _EVENT_TYPES:
        _fail("invalid_event")
    elapsed_ns = event["elapsed_ns"]
    if type(elapsed_ns) is not int or elapsed_ns < 0:
        _fail("invalid_event")
    artifact_bytes = event["artifact_bytes"]
    if type(artifact_bytes) is not int or artifact_bytes < 0:
        _fail("invalid_event")
    job_id = event["job_id"]
    if job_id is not None and not _is_clean_id(job_id):
        _fail("invalid_event")
    status = event["status"]
    if status is not None and type(status) is not str:
        _fail("invalid_event")
    receipt = event["receipt_sha256"]
    if receipt is not None and not _is_lower64(receipt):
        _fail("invalid_event")


def _event_payload(event):
    return {
        "seq": event["seq"],
        "previous_sha256": event["previous_sha256"],
        "session_id": event["session_id"],
        "event_type": event["event_type"],
        "elapsed_ns": event["elapsed_ns"],
        "artifact_bytes": event["artifact_bytes"],
        "job_id": event["job_id"],
        "status": event["status"],
        "receipt_sha256": event["receipt_sha256"],
    }


def _replay_attempt_journal(manifest, events, *, expected_manifest_sha256, expected_head_sha256):
    """Replay invented session events into a cumulative attempt summary."""
    if not _is_lower64(expected_manifest_sha256):
        _fail("invalid_journal_input")
    if not _is_lower64(expected_head_sha256):
        _fail("invalid_journal_input")
    jobs_by_id, manifest_digest = _validate_manifest(manifest, expected_manifest_sha256)
    if type(events) is not list:
        _fail("invalid_journal_input")

    session_count = 0
    active_session_id = None
    last_elapsed = 0
    closed_wall = 0
    artifact_highwater = 0
    attempts = {}
    running = set()
    succeeded = set()

    previous = manifest_digest
    for index, event in enumerate(events):
        _validate_event_shape(event, index)
        if event["previous_sha256"] != previous:
            _fail("event_hash_mismatch")
        if _canonical_digest(_event_payload(event), "invalid_event") != event["event_sha256"]:
            _fail("event_hash_mismatch")
        artifact_bytes = event["artifact_bytes"]
        if artifact_bytes < artifact_highwater:
            _fail("invalid_transition")
        artifact_highwater = artifact_bytes

        event_type = event["event_type"]
        if event_type == "session_open":
            if active_session_id is not None:
                _fail("invalid_transition")
            if event["session_id"] != session_count + 1:
                _fail("invalid_transition")
            if event["elapsed_ns"] != 0:
                _fail("invalid_transition")
            if (
                event["job_id"] is not None
                or event["status"] is not None
                or event["receipt_sha256"] is not None
            ):
                _fail("invalid_transition")
            active_session_id = event["session_id"]
            session_count += 1
            last_elapsed = 0
        else:
            if active_session_id is None or event["session_id"] != active_session_id:
                _fail("invalid_transition")
            elapsed = event["elapsed_ns"]
            if elapsed < last_elapsed:
                _fail("invalid_transition")

            if event_type == "progress":
                if (
                    event["job_id"] is not None
                    or event["status"] is not None
                    or event["receipt_sha256"] is not None
                ):
                    _fail("invalid_transition")
            elif event_type == "attempt_start":
                job_id = event["job_id"]
                if job_id is None or job_id not in jobs_by_id or job_id in attempts:
                    _fail("invalid_transition")
                if event["status"] is not None or event["receipt_sha256"] is not None:
                    _fail("invalid_transition")
                job = jobs_by_id[job_id]
                for dependency in job["dependencies"]:
                    if dependency not in succeeded:
                        _fail("invalid_transition")
                attempts[job_id] = {
                    "stage": job["stage"],
                    "worker": job["worker"],
                    "session_id": active_session_id,
                    "status": "running",
                    "receipt_sha256": None,
                }
                running.add(job_id)
            elif event_type == "attempt_finish":
                job_id = event["job_id"]
                if job_id is None or job_id not in attempts:
                    _fail("invalid_transition")
                record = attempts[job_id]
                if record["status"] != "running" or record["session_id"] != active_session_id:
                    _fail("invalid_transition")
                status = event["status"]
                if type(status) is not str or status not in _TERMINAL_STATUSES:
                    _fail("invalid_transition")
                receipt = event["receipt_sha256"]
                if not _is_lower64(receipt):
                    _fail("invalid_transition")
                record["status"] = status
                record["receipt_sha256"] = receipt
                running.discard(job_id)
                if status == "succeeded":
                    succeeded.add(job_id)
            else:  # session_close
                if running:
                    _fail("invalid_transition")
                if (
                    event["job_id"] is not None
                    or event["status"] is not None
                    or event["receipt_sha256"] is not None
                ):
                    _fail("invalid_transition")
                closed_wall += elapsed
                active_session_id = None
                last_elapsed = 0

            last_elapsed = elapsed
        previous = event["event_sha256"]

    head = previous
    if head != expected_head_sha256:
        _fail("journal_head_mismatch")

    if not events:
        journal_state = "not_started"
    elif active_session_id is None:
        journal_state = "closed"
    else:
        journal_state = "open"
    active_elapsed = last_elapsed if active_session_id is not None else 0

    attempts_out = {}
    in_flight = []
    succeeded_ids = []
    failed_ids = []
    interrupted_ids = []
    fit_attempts = 0
    prediction_attempts = 0
    active_cpu = 0
    active_gpu = 0
    for job_id in sorted(attempts):
        record = attempts[job_id]
        status = record["status"]
        attempts_out[job_id] = {
            "stage": record["stage"],
            "worker": record["worker"],
            "session_id": record["session_id"],
            "status": status,
            "receipt_sha256": record["receipt_sha256"],
        }
        if record["stage"] == "source_fit":
            fit_attempts += 1
        else:
            prediction_attempts += 1
        if status == "running":
            in_flight.append(job_id)
            if record["worker"] == "cpu":
                active_cpu += 1
            else:
                active_gpu += 1
        elif status == "succeeded":
            succeeded_ids.append(job_id)
        elif status == "failed":
            failed_ids.append(job_id)
        elif status == "interrupted":
            interrupted_ids.append(job_id)

    summary = {
        "schema_version": _SUMMARY_SCHEMA,
        "execution_authorized": False,
        "manifest_sha256": manifest_digest,
        "head_sha256": head,
        "session_count": session_count,
        "active_session_id": active_session_id,
        "closed_session_wall_ns": closed_wall,
        "active_session_elapsed_ns": active_elapsed,
        "active_wall_ns": closed_wall + active_elapsed,
        "new_artifact_bytes": artifact_highwater,
        "model_fit_attempts": fit_attempts,
        "source_prediction_attempts": prediction_attempts,
        "active_cpu_workers": active_cpu,
        "active_gpu_workers": active_gpu,
        "attempts": attempts_out,
        "in_flight_job_ids": in_flight,
        "succeeded_job_ids": succeeded_ids,
        "failed_job_ids": failed_ids,
        "interrupted_job_ids": interrupted_ids,
        "restart_allowed": active_session_id is None,
        "journal_state": journal_state,
    }
    summary["summary_sha256"] = _canonical_digest(summary)
    return summary


def replay_attempt_journal(manifest, events, *, expected_manifest_sha256, expected_head_sha256):
    """Replay invented session events into a cumulative attempt summary."""
    try:
        return _replay_attempt_journal(
            manifest,
            events,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_head_sha256=expected_head_sha256,
        )
    except JournalError:
        raise
    except Exception:
        raise JournalError("invalid_journal_input") from None


def require_scientific_execution(*args, **kwargs):
    """Always deny: this kernel never authorizes scientific execution."""
    raise JournalError("scientific_execution_not_authorized") from None
