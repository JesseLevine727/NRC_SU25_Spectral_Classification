"""P08-T063 read-only terminal receipt/artifact byte verifier.

This leaf closes one narrow gap: the P08 attempt journal can validate
digest-shaped receipt references but cannot prove that a referenced receipt
document or its artifact bytes exist.  Given an invented manifest, an invented
current journal, and a proposed terminal ``attempt_finish`` event, this module
replays the journal, binds the proposed event to the exactly one prior
``attempt_start`` for the same job and to one exact manifest job entry, then
reads one bounded receipt document together with its bounded artifact bytes
under held directory descriptors.

What this module establishes and what it explicitly does NOT establish:

* It establishes *only* receipt/artifact byte integrity (existence, bounded
  size and SHA-256 equality for the exact bytes read).
* It does NOT establish checkpoint loadability, prediction schema or values,
  scientific acceptance, a live owner, a fresh resource reading, a wall-clock
  timestamp, durable-storage authenticity or execution permission.
* It performs no journal append, store creation, lease acquisition, write,
  delete, rename, truncation or repair.
* It performs no unpickling, dataset access or model/tensor import.

POSIX/Linux is the explicit supported platform.  The verifier is strictly
read-only and never authorizes scientific execution.
"""

from __future__ import annotations

import hashlib
import re

from . import p08_u0_admission as admission
from . import p08_u0_store as store
from .p08_attempt_journal import JournalError, replay_attempt_journal
from .p08_qc_blocks import canonical_sha256

__all__ = [
    "ReceiptError",
    "verify_terminal_receipt",
    "require_scientific_execution",
]

_RECEIPT_SCHEMA = "nato-sers-p08-terminal-receipt-v1"
_REPORT_SCHEMA = "nato-sers-p08-terminal-receipt-check-v1"

_MAX_RECEIPT_BYTES = 65536
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_ARTIFACTS = 8

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "interrupted"})

_REASON_CODES = frozenset(
    {
        "invalid_input",
        "invalid_journal",
        "smoke_binding_mismatch",
        "invalid_receipt",
        "artifact_mismatch",
        "receipt_io_error",
        "scientific_execution_not_authorized",
    }
)

_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "proposal_sha256",
        "manifest_sha256",
        "job_id",
        "stage",
        "worker",
        "session_id",
        "start_event_sha256",
        "status",
        "artifacts",
        "receipt_sha256",
    }
)

_ARTIFACT_KEYS = frozenset({"name", "size_bytes", "sha256"})

_HEX_DIGITS = frozenset("0123456789abcdef")
_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class ReceiptError(ValueError):
    """Static, data-free receipt verification failure."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "invalid_input"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _fail(reason_code):
    raise ReceiptError(reason_code) from None


def _is_lower64(value):
    if type(value) is not str or len(value) != 64:
        return False
    for char in value:
        if char not in _HEX_DIGITS:
            return False
    return True


def _is_clean_id(value):
    return type(value) is str and value != "" and value == value.strip()


def _is_safe_name(value):
    if type(value) is not str:
        return False
    return _NAME_PATTERN.fullmatch(value) is not None


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


def _replay_current(manifest, events, expected_head_sha256):
    try:
        summary = replay_attempt_journal(
            manifest,
            events,
            expected_manifest_sha256=admission.U0_MANIFEST_SHA256,
            expected_head_sha256=expected_head_sha256,
        )
    except JournalError:
        _fail("invalid_journal")
    try:
        bound = admission._u0_binding_ok(manifest)
    except Exception:
        _fail("smoke_binding_mismatch")
    if bound is not True:
        _fail("smoke_binding_mismatch")
    return summary


def _validate_terminal_event(manifest, events, terminal_event):
    if type(terminal_event) is not dict:
        _fail("invalid_journal")
    if terminal_event.get("event_type") != "attempt_finish":
        _fail("invalid_journal")
    declared = terminal_event.get("event_sha256")
    if not _is_lower64(declared):
        _fail("invalid_journal")
    try:
        replay_attempt_journal(
            manifest,
            events + [terminal_event],
            expected_manifest_sha256=admission.U0_MANIFEST_SHA256,
            expected_head_sha256=declared,
        )
    except JournalError:
        _fail("invalid_journal")
    job_id = terminal_event["job_id"]
    starts = [
        event
        for event in events
        if type(event) is dict
        and event.get("event_type") == "attempt_start"
        and event.get("job_id") == job_id
    ]
    if len(starts) != 1:
        _fail("invalid_journal")
    return starts[0]


def _find_job(manifest, job_id):
    for job in manifest["jobs"]:
        if job["job_id"] == job_id:
            return job
    _fail("invalid_journal")


def _validate_expected_artifacts(receipt_name, expected_artifact_names):
    if not _is_safe_name(receipt_name):
        _fail("invalid_input")
    if type(expected_artifact_names) is not list:
        _fail("invalid_input")
    count = len(expected_artifact_names)
    if count < 1 or count > _MAX_ARTIFACTS:
        _fail("invalid_input")
    for name in expected_artifact_names:
        if not _is_safe_name(name):
            _fail("invalid_input")
    if len(set(expected_artifact_names)) != count:
        _fail("invalid_input")
    if list(expected_artifact_names) != sorted(expected_artifact_names):
        _fail("invalid_input")
    if receipt_name in expected_artifact_names:
        _fail("invalid_input")


def _read_checked(dir_fd, name, max_bytes):
    try:
        return store._read_file(dir_fd, name, max_bytes)
    except store.StoreError:
        _fail("receipt_io_error")
    except OSError:
        _fail("receipt_io_error")


def _load_receipt(root_fd, receipt_name):
    raw = _read_checked(root_fd, receipt_name, _MAX_RECEIPT_BYTES)
    try:
        parsed = store._parse_json(raw)
    except store.StoreError:
        _fail("invalid_receipt")
    if type(parsed) is not dict:
        _fail("invalid_receipt")
    return raw, parsed


def _validate_receipt_structure(receipt):
    if type(receipt) is not dict:
        _fail("invalid_receipt")
    if not _has_exact_keys(receipt, _RECEIPT_KEYS):
        _fail("invalid_receipt")
    schema_version = receipt["schema_version"]
    if type(schema_version) is not str or schema_version != _RECEIPT_SCHEMA:
        _fail("invalid_receipt")
    if receipt["execution_authorized"] is not False:
        _fail("invalid_receipt")
    for field in (
        "proposal_sha256",
        "manifest_sha256",
        "start_event_sha256",
        "receipt_sha256",
    ):
        if not _is_lower64(receipt[field]):
            _fail("invalid_receipt")
    if not _is_clean_id(receipt["job_id"]):
        _fail("invalid_receipt")
    if type(receipt["stage"]) is not str:
        _fail("invalid_receipt")
    if type(receipt["worker"]) is not str:
        _fail("invalid_receipt")
    session_id = receipt["session_id"]
    if type(session_id) is not int or session_id <= 0:
        _fail("invalid_receipt")
    status = receipt["status"]
    if type(status) is not str or status not in _TERMINAL_STATUSES:
        _fail("invalid_receipt")
    artifacts = receipt["artifacts"]
    if type(artifacts) is not list:
        _fail("invalid_receipt")
    if len(artifacts) < 1 or len(artifacts) > _MAX_ARTIFACTS:
        _fail("invalid_receipt")
    for entry in artifacts:
        if type(entry) is not dict or not _has_exact_keys(entry, _ARTIFACT_KEYS):
            _fail("invalid_receipt")
        if not _is_safe_name(entry["name"]):
            _fail("invalid_receipt")
        size = entry["size_bytes"]
        if type(size) is not int or size < 0 or size > _MAX_ARTIFACT_BYTES:
            _fail("invalid_receipt")
        if not _is_lower64(entry["sha256"]):
            _fail("invalid_receipt")


def _validate_receipt_binding(
    receipt, manifest_job, terminal_event, prior_start, expected_artifact_names
):
    if receipt["proposal_sha256"] != admission.U0_PROPOSAL_SHA256:
        _fail("invalid_receipt")
    if receipt["manifest_sha256"] != admission.U0_MANIFEST_SHA256:
        _fail("invalid_receipt")
    if receipt["job_id"] != terminal_event["job_id"]:
        _fail("invalid_receipt")
    if receipt["stage"] != manifest_job["stage"]:
        _fail("invalid_receipt")
    if receipt["worker"] != manifest_job["worker"]:
        _fail("invalid_receipt")
    if receipt["session_id"] != terminal_event["session_id"]:
        _fail("invalid_receipt")
    if receipt["start_event_sha256"] != prior_start["event_sha256"]:
        _fail("invalid_receipt")
    if receipt["status"] != terminal_event["status"]:
        _fail("invalid_receipt")
    if receipt["receipt_sha256"] != terminal_event["receipt_sha256"]:
        _fail("invalid_receipt")
    names = [entry["name"] for entry in receipt["artifacts"]]
    if names != sorted(names) or len(set(names)) != len(names):
        _fail("invalid_receipt")
    if names != list(expected_artifact_names):
        _fail("invalid_receipt")
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    try:
        digest = canonical_sha256(payload)
    except Exception:
        _fail("invalid_receipt")
    if receipt["receipt_sha256"] != digest:
        _fail("invalid_receipt")


def _verify_artifacts(root_fd, artifacts):
    total = 0
    for entry in artifacts:
        raw = _read_checked(root_fd, entry["name"], _MAX_ARTIFACT_BYTES)
        if len(raw) != entry["size_bytes"]:
            _fail("artifact_mismatch")
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            _fail("artifact_mismatch")
        total += len(raw)
    return total


def _verify_terminal_receipt(
    root,
    receipt_name,
    manifest,
    events,
    terminal_event,
    *,
    expected_head_sha256,
    expected_artifact_names,
):
    summary = _replay_current(manifest, events, expected_head_sha256)
    prior_start = _validate_terminal_event(manifest, events, terminal_event)
    manifest_job = _find_job(manifest, terminal_event["job_id"])
    _validate_expected_artifacts(receipt_name, expected_artifact_names)

    parent_fd = -1
    root_fd = -1
    try:
        try:
            parent_fd, root_name = store._resolve_parent(root)
            root_fd = store._open_dir(parent_fd, root_name)
        except store.StoreError:
            _fail("receipt_io_error")
        except OSError:
            _fail("receipt_io_error")

        raw_receipt, receipt = _load_receipt(root_fd, receipt_name)
        _validate_receipt_structure(receipt)
        _validate_receipt_binding(
            receipt,
            manifest_job,
            terminal_event,
            prior_start,
            expected_artifact_names,
        )
        artifact_total = _verify_artifacts(root_fd, receipt["artifacts"])

        report = {
            "schema_version": _REPORT_SCHEMA,
            "execution_authorized": False,
            "byte_integrity_verified": True,
            "scientific_semantics_verified": False,
            "manifest_sha256": summary["manifest_sha256"],
            "previous_head_sha256": expected_head_sha256,
            "proposed_terminal_event_sha256": terminal_event["event_sha256"],
            "receipt_sha256": receipt["receipt_sha256"],
            "receipt_file_sha256": hashlib.sha256(raw_receipt).hexdigest(),
            "artifact_count": len(receipt["artifacts"]),
            "artifact_bytes": artifact_total,
            "status": terminal_event["status"],
        }
        report["check_sha256"] = canonical_sha256(report)
    except BaseException:
        # Verification already failed: release every owned descriptor, but
        # preserve the original failure object even if cleanup raises.
        try:
            store._close_fds(root_fd, parent_fd)
        except BaseException:
            pass
        raise
    else:
        # Successful verification: cleanup failures are never swallowed.
        store._close_fds(root_fd, parent_fd)
        return report


def verify_terminal_receipt(
    root,
    receipt_name,
    manifest,
    events,
    terminal_event,
    *,
    expected_head_sha256,
    expected_artifact_names,
):
    """Read-only terminal receipt/artifact byte-integrity check.

    Never appends to any journal, never mutates caller objects and never
    authorizes scientific execution.  The returned flag set is a static
    structural opinion, not a permit or proof of a live owner.
    """
    try:
        return _verify_terminal_receipt(
            root,
            receipt_name,
            manifest,
            events,
            terminal_event,
            expected_head_sha256=expected_head_sha256,
            expected_artifact_names=expected_artifact_names,
        )
    except ReceiptError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except OSError:
        raise ReceiptError("receipt_io_error") from None
    except Exception:
        raise ReceiptError("invalid_input") from None


def require_scientific_execution(*args, **kwargs):
    """Always deny: this leaf grants no scientific execution authority."""
    raise ReceiptError("scientific_execution_not_authorized") from None
