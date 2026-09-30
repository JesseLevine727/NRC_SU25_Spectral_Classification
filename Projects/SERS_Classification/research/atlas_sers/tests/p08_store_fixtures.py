"""Reusable, no-execution fixtures for the invented P08-T052 U0 store tests.

Everything produced here is invented metadata.  These helpers never fit
models, read datasets, touch devices, authenticate storage or authorize
scientific execution.  The only filesystem touch is delegated to a caller's
``owner.append_event``; ``readtree`` reads only.
"""

from __future__ import annotations

import pathlib

from atlas_sers.evaluation import p08_u0_admission as admission
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256

GIB = 1024**3
NANOS = 10**9

_MANIFEST_SCHEMA = "nato-sers-p08-attempt-manifest-v1"
_CPU_FITS = 42
_TOTAL_FITS = 78
_RECEIPT = "a" * 64

__all__ = [
    "GIB",
    "NANOS",
    "make_manifest",
    "bind_manifest",
    "resources",
    "make_event",
    "append",
    "readtree",
]


def make_manifest():
    """Return an invented, canonical 78-fit / 78-prediction U0 manifest."""
    jobs = []
    for index in range(1, _TOTAL_FITS + 1):
        jobs.append(
            {
                "job_id": f"fit{index:03d}",
                "stage": "source_fit",
                "worker": "cpu" if index <= _CPU_FITS else "gpu",
                "dependencies": [],
            }
        )
    for index in range(1, _TOTAL_FITS + 1):
        jobs.append(
            {
                "job_id": f"pred{index:03d}",
                "stage": "source_validation_prediction",
                "worker": "cpu" if index <= _CPU_FITS else "gpu",
                "dependencies": [f"fit{index:03d}"],
            }
        )
    payload = {
        "schema_version": _MANIFEST_SCHEMA,
        "execution_authorized": False,
        "proposal_sha256": admission.U0_PROPOSAL_SHA256,
        "jobs": jobs,
    }
    manifest = dict(payload)
    manifest["manifest_sha256"] = canonical_sha256(payload)
    return manifest


def bind_manifest(monkeypatch):
    """Return an invented manifest admitted only via a monkeypatched digest."""
    manifest = make_manifest()
    monkeypatch.setattr(admission, "U0_MANIFEST_SHA256", manifest["manifest_sha256"])
    return manifest


def resources(cpu=0, gpu=0, **overrides):
    """Return an exact resource snapshot with invented, overridable values."""
    record = {
        "filesystem_free_bytes": 38 * GIB,
        "process_tree_rss_bytes": 0,
        "cuda_allocated_bytes": 0,
        "cuda_reserved_bytes": 0,
        "cuda_device_used_bytes": 0,
        "active_cpu_workers": cpu,
        "active_gpu_workers": gpu,
        "model_threads": 1,
        "blas_threads": 1,
        "torch_threads": 1,
    }
    record.update(overrides)
    return record


def make_event(
    state,
    event_type,
    *,
    elapsed_ns=None,
    artifact_bytes=None,
    job_id=None,
    status=None,
    session_id=None,
):
    """Seal one invented 10-field event against a ``store.snapshot()`` state."""
    summary = state["summary"]
    events = state["events"]
    if session_id is None:
        if summary["active_session_id"] is not None:
            session_id = summary["active_session_id"]
        else:
            session_id = summary["session_count"] + 1
    if elapsed_ns is None:
        if event_type == "session_open":
            elapsed_ns = 0
        else:
            elapsed_ns = summary["active_session_elapsed_ns"]
    if artifact_bytes is None:
        artifact_bytes = summary["new_artifact_bytes"]
    event = {
        "seq": len(events) + 1,
        "previous_sha256": summary["head_sha256"],
        "session_id": session_id,
        "event_type": event_type,
        "elapsed_ns": elapsed_ns,
        "artifact_bytes": artifact_bytes,
        "job_id": job_id,
        "status": status,
        "receipt_sha256": _RECEIPT if status is not None else None,
    }
    event["event_sha256"] = canonical_sha256(event)
    return event


def append(owner, event_type, *, resources=None, **fields):
    """Snapshot, seal and append one event; return the resulting summary."""
    state = owner.snapshot()
    event = make_event(state, event_type, **fields)
    return owner.append_event(
        event,
        expected_head_sha256=state["summary"]["head_sha256"],
        resources=resources,
    )


def readtree(root):
    """Map path-relative regular-file bytes below ``root`` (reads only)."""
    base = pathlib.Path(root)
    result = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_file():
            result[str(path.relative_to(base))] = path.read_bytes()
    return result
