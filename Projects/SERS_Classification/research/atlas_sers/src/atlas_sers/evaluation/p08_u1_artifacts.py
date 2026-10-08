"""T295 compact durable per-job artifact IO for the P08-U1 protocol.

Durable, exclusive, no-fail-open storage of opaque artifact bytes for one
already-authenticated P08 job.  This module never fits models, never authorizes
scientific execution, never reads a graph, never reuses prior output and never
loads pickles.  It writes only beneath an already-existing real root and never
follows symlinks.  It never deletes, cleans, truncates, renames or auto-resumes
anything: a partially written job directory is left for caller review.

``verify`` is filesystem integrity checking only.  It is not scientific result
validation and does not authenticate upstream roles, units or provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from atlas_sers.evaluation.p08_plan import JOB_FIELDS
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_value

__all__ = ["ArtifactError", "ArtifactStore"]

SCHEMA_VERSION = "nato-sers-p08-u1-artifacts-v1"
STATUSES = frozenset(("complete", "failed", "interrupted"))

_RECEIPT = "receipt.json"
_NAME_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*\Z")
_JOB_RE = re.compile(r"P08JOB-[0-9a-f]{64}\Z")
_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")
_JOB_KEYS = frozenset(JOB_FIELDS) | {"job_id"}
_RECEIPT_KEYS = frozenset(
    (
        "schema_version",
        "binding_sha256",
        "job_id",
        "job_sha256",
        "job",
        "status",
        "artifacts",
        "extra",
        "sha256",
    )
)


class ArtifactError(ValueError):
    """Raised for malformed inputs, unsafe paths or failed verification."""


def _fail(code):
    raise ArtifactError(code)


def _hash_value(value):
    try:
        return sha256_value(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError("value_not_json") from exc


def _deep_json(value, code):
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(code) from exc


def _job_fields(job):
    return {name: job[name] for name in JOB_FIELDS}


def _validate_job(job):
    if not isinstance(job, dict):
        _fail("job_must_be_mapping")
    if set(job) != _JOB_KEYS:
        _fail("job_keys_invalid")
    job_id = job["job_id"]
    if not isinstance(job_id, str) or _JOB_RE.fullmatch(job_id) is None:
        _fail("job_id_invalid")
    if "P08JOB-" + _hash_value(_job_fields(job)) != job_id:
        _fail("job_id_hash_mismatch")
    return job_id


def _validate_name(name):
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        _fail("artifact_name_invalid")
    if name in (".", "..", _RECEIPT):
        _fail("artifact_name_invalid")


def _resolve_root(root):
    try:
        path = Path(os.fspath(root))
    except TypeError:
        _fail("root_invalid")
    if path.is_symlink():
        _fail("root_symlink")
    if not path.exists():
        _fail("root_missing")
    if not path.is_dir():
        _fail("root_not_directory")
    return path.resolve(strict=True)


def _ensure_dir(path):
    if path.is_symlink():
        _fail("directory_symlink")
    if path.exists():
        if not path.is_dir():
            _fail("not_a_directory")
        return
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        if path.is_symlink() or not path.is_dir():
            _fail("not_a_directory")
    except OSError as exc:
        raise ArtifactError("directory_create_failed") from exc


def _exclusive_mkdir(path):
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        _fail("job_dir_exists")
    except OSError as exc:
        raise ArtifactError("job_dir_create_failed") from exc


def _write_exclusive(directory, name, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        fd = os.open(directory / name, flags, 0o600)
    except OSError as exc:
        raise ArtifactError("artifact_create_failed") from exc
    try:
        handle = os.fdopen(fd, "wb")
    except OSError as exc:
        os.close(fd)
        raise ArtifactError("artifact_open_failed") from exc
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ArtifactError("artifact_write_failed") from exc


def _read_file(path, code):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ArtifactError(code) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            _fail("not_regular_file")
        if info.st_nlink != 1:
            _fail("hardlink_detected")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            return handle.read()
    finally:
        if fd != -1:
            os.close(fd)


def _require_dir(path):
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ArtifactError("path_missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        _fail("path_not_directory")


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ArtifactStore:
    """Durable per-job artifact store rooted at an existing real directory."""

    def __init__(self, root, *, binding_sha256):
        if not isinstance(binding_sha256, str) or _HEX_RE.fullmatch(binding_sha256) is None:
            _fail("binding_sha256_invalid")
        self._root = _resolve_root(root)
        self._binding_sha256 = binding_sha256
        self._jobs = self._root / "jobs"
        _ensure_dir(self._jobs)

    def _job_dir(self, job_id):
        return self._jobs / job_id[7:9] / job_id

    def write(self, job, artifacts, *, status, extra=None):
        job_id = _validate_job(job)
        if status not in STATUSES:
            _fail("status_invalid")
        if not isinstance(artifacts, dict) or not artifacts:
            _fail("artifacts_invalid")
        blobs = {}
        for name, data in artifacts.items():
            _validate_name(name)
            if not isinstance(data, bytes):
                _fail("artifact_must_be_bytes")
            blobs[name] = data
        if extra is None:
            extra_value = {}
        elif isinstance(extra, dict):
            extra_value = _deep_json(extra, "extra_not_json")
        else:
            _fail("extra_invalid")

        _require_dir(self._root)
        _require_dir(self._jobs)
        job_dir = self._job_dir(job_id)
        _ensure_dir(job_dir.parent)
        _exclusive_mkdir(job_dir)

        records = []
        for name in sorted(blobs):
            data = blobs[name]
            _write_exclusive(job_dir, name, data)
            records.append(
                {
                    "name": name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )

        job_copy = _deep_json(job, "job_not_json")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "binding_sha256": self._binding_sha256,
            "job_id": job_id,
            "job_sha256": _hash_value(_job_fields(job_copy)),
            "job": job_copy,
            "status": status,
            "artifacts": records,
            "extra": extra_value,
        }
        receipt = dict(payload)
        receipt["sha256"] = _hash_value(payload)
        _write_exclusive(job_dir, _RECEIPT, canonical_json_bytes(receipt))
        for directory in (job_dir, job_dir.parent, self._jobs, self._root):
            _fsync_dir(directory)
        return receipt

    def verify(self, job, *, expected_receipt_sha256=None):
        job_id = _validate_job(job)
        job_dir = self._job_dir(job_id)
        _require_dir(self._root)
        _require_dir(self._jobs)
        _require_dir(job_dir.parent)
        _require_dir(job_dir)
        raw = _read_file(job_dir / _RECEIPT, "receipt_read_failed")
        try:
            receipt = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ArtifactError("receipt_invalid") from exc
        if not isinstance(receipt, dict):
            _fail("receipt_invalid")
        if canonical_json_bytes(receipt) != raw:
            _fail("receipt_not_canonical")
        self._validate_receipt(receipt, job_id)
        if receipt["job"] != _deep_json(job, "job_not_json"):
            _fail("receipt_job_mismatch")
        if expected_receipt_sha256 is not None and receipt["sha256"] != expected_receipt_sha256:
            _fail("receipt_hash_mismatch")
        files = {}
        for entry in receipt["artifacts"]:
            data = _read_file(job_dir / entry["name"], "artifact_read_failed")
            if (
                len(data) != entry["size_bytes"]
                or hashlib.sha256(data).hexdigest() != entry["sha256"]
            ):
                _fail("artifact_mismatch")
            files[entry["name"]] = data
        listed = {entry["name"] for entry in receipt["artifacts"]} | {_RECEIPT}
        if set(os.listdir(job_dir)) != listed:
            _fail("unlisted_files")
        return receipt, files

    def _validate_receipt(self, receipt, job_id):
        if set(receipt) != set(_RECEIPT_KEYS):
            _fail("receipt_keys_invalid")
        if receipt["sha256"] != _hash_value(
            {key: value for key, value in receipt.items() if key != "sha256"}
        ):
            _fail("receipt_hash_mismatch")
        if receipt["schema_version"] != SCHEMA_VERSION:
            _fail("receipt_schema_invalid")
        if receipt["binding_sha256"] != self._binding_sha256:
            _fail("receipt_binding_mismatch")
        if receipt["job_id"] != job_id:
            _fail("receipt_job_id_mismatch")
        if receipt["status"] not in STATUSES:
            _fail("receipt_status_invalid")
        stored = receipt["job"]
        if not isinstance(stored, dict) or set(stored) != _JOB_KEYS:
            _fail("receipt_job_invalid")
        if receipt["job_sha256"] != _hash_value(_job_fields(stored)):
            _fail("receipt_job_hash_mismatch")
        entries = receipt["artifacts"]
        if not isinstance(entries, list) or not entries or not isinstance(receipt["extra"], dict):
            _fail("receipt_artifacts_invalid")
        seen = set()
        names = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"name", "sha256", "size_bytes"}:
                _fail("receipt_artifacts_invalid")
            name = entry["name"]
            _validate_name(name)
            if name in seen:
                _fail("receipt_artifacts_invalid")
            seen.add(name)
            names.append(name)
            if not isinstance(entry["sha256"], str) or _HEX_RE.fullmatch(entry["sha256"]) is None:
                _fail("receipt_artifacts_invalid")
            size = entry["size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                _fail("receipt_artifacts_invalid")
        if names != sorted(names):
            _fail("receipt_artifacts_unsorted")
