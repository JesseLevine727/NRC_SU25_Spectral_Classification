"""P08-T046 exact-U0 prospective candidate guard (pure, no execution).

This leaf performs bookkeeping only.  It replays an invented attempt journal
with the exact public replay kernel, requires the published U0 proposal and
manifest planning digests, maps the replayed usage onto the proposed U0
resource ceilings and reports whether a named candidate *would* be admissible.

Nothing here reserves a slot, mutates an input, authenticates durable storage,
leases, live owners, fresh resource readings, timestamps or checkpoint
identity, and nothing here authorizes scientific execution.  The returned
``proposed_candidate_admissible`` flag is a proposed structural opinion, never
a permit, reservation or proof of a live owner.
"""

from __future__ import annotations

from .p08_attempt_journal import JournalError, replay_attempt_journal
from .p08_qc_blocks import canonical_sha256
from .p08_resources import ResourceGuardError, evaluate_resource_snapshot

__all__ = [
    "U0_PROPOSAL_SHA256",
    "U0_MANIFEST_SHA256",
    "AdmissionError",
    "evaluate_u0_candidate",
    "require_scientific_execution",
]

U0_PROPOSAL_SHA256 = "6639a32c1dd930612ead6ae59adf9aff5831904e883081f16c3490721af5089a"
U0_MANIFEST_SHA256 = "aac8523a1614610cf99cf1ab548d8970b4f8054fe8821b5250c347376e30b28a"

_SCHEMA_VERSION = "nato-sers-p08-u0-candidate-check-v1"

_REASON_CODES = frozenset(
    {
        "invalid_admission_input",
        "invalid_journal",
        "smoke_binding_mismatch",
        "invalid_candidate",
        "unregistered_job",
        "invalid_resources",
        "worker_count_mismatch",
        "scientific_execution_not_authorized",
    }
)

_U0_JOB_COUNT = 156
_U0_FIT_COUNT = 78
_U0_PREDICTION_COUNT = 78
_U0_CPU_FIT_COUNT = 42
_U0_GPU_FIT_COUNT = 36


class AdmissionError(ValueError):
    """Static, data-free admission failure exposing ``reason_code``."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "invalid_admission_input"
        self.reason_code = reason_code
        super().__init__(reason_code)


def _fail(reason_code):
    raise AdmissionError(reason_code) from None


def _validate_candidate_id(job_id):
    if type(job_id) is not str or job_id == "" or job_id != job_id.strip():
        _fail("invalid_candidate")
    try:
        job_id.encode("utf-8")
    except UnicodeEncodeError:
        _fail("invalid_candidate")


def _u0_binding_ok(manifest):
    if manifest["proposal_sha256"] != U0_PROPOSAL_SHA256:
        return False
    jobs = manifest["jobs"]
    if len(jobs) != _U0_JOB_COUNT:
        return False
    fits = 0
    predictions = 0
    cpu_fits = 0
    gpu_fits = 0
    for job in jobs:
        if job["stage"] == "source_fit":
            fits += 1
            if job["worker"] == "cpu":
                cpu_fits += 1
            else:
                gpu_fits += 1
        else:
            predictions += 1
    return (
        fits == _U0_FIT_COUNT
        and predictions == _U0_PREDICTION_COUNT
        and cpu_fits == _U0_CPU_FIT_COUNT
        and gpu_fits == _U0_GPU_FIT_COUNT
    )


def _collect_reasons(job, job_id, summary, snapshot, prospective):
    reasons = []
    if summary["journal_state"] != "open":
        reasons.append("session_not_open")
    if summary["failed_job_ids"] or summary["interrupted_job_ids"]:
        reasons.append("prior_failed_or_interrupted_attempt_requires_review")
    if job_id in summary["attempts"]:
        reasons.append("job_already_attempted")
    succeeded = summary["succeeded_job_ids"]
    for dependency in job["dependencies"]:
        if dependency not in succeeded:
            reasons.append("fit_dependency_not_succeeded")
            break
    limits = snapshot["limits"]
    if prospective["model_fit_attempts"] > limits["model_fit_attempts"]:
        reasons.append("model_fit_capacity_exhausted")
    if prospective["cpu_workers"] > limits["max_cpu_workers"]:
        reasons.append("cpu_worker_capacity_exhausted")
    if prospective["gpu_workers"] > limits["max_gpu_workers"]:
        reasons.append("gpu_worker_capacity_exhausted")
    if not snapshot["within_proposed_limits"]:
        reasons.append("resource_limits_breached")
    return reasons


def _evaluate_u0_candidate(manifest, events, resources, job_id, expected_head_sha256):
    try:
        summary = replay_attempt_journal(
            manifest,
            events,
            expected_manifest_sha256=U0_MANIFEST_SHA256,
            expected_head_sha256=expected_head_sha256,
        )
    except JournalError:
        raise AdmissionError("invalid_journal") from None

    if not _u0_binding_ok(manifest):
        _fail("smoke_binding_mismatch")

    _validate_candidate_id(job_id)
    jobs_by_id = {job["job_id"]: job for job in manifest["jobs"]}
    job = jobs_by_id.get(job_id)
    if job is None:
        _fail("unregistered_job")

    usage = {
        "model_fit_attempts": summary["model_fit_attempts"],
        "scalar_calibration_attempts": 0,
        "active_wall_ns": summary["active_wall_ns"],
        "new_artifact_bytes": summary["new_artifact_bytes"],
    }
    try:
        snapshot = evaluate_resource_snapshot("U0", usage, resources)
    except ResourceGuardError:
        raise AdmissionError("invalid_resources") from None

    if (
        resources["active_cpu_workers"] != summary["active_cpu_workers"]
        or resources["active_gpu_workers"] != summary["active_gpu_workers"]
    ):
        _fail("worker_count_mismatch")

    stage = job["stage"]
    worker = job["worker"]
    prospective = {
        "model_fit_attempts": summary["model_fit_attempts"] + (1 if stage == "source_fit" else 0),
        "cpu_workers": summary["active_cpu_workers"] + (1 if worker == "cpu" else 0),
        "gpu_workers": summary["active_gpu_workers"] + (1 if worker == "gpu" else 0),
    }
    reasons = _collect_reasons(job, job_id, summary, snapshot, prospective)

    candidate = {
        "schema_version": _SCHEMA_VERSION,
        "execution_authorized": False,
        "proposal_sha256": U0_PROPOSAL_SHA256,
        "manifest_sha256": summary["manifest_sha256"],
        "journal_head_sha256": summary["head_sha256"],
        "journal_summary_sha256": summary["summary_sha256"],
        "resource_snapshot_sha256": snapshot["snapshot_sha256"],
        "candidate_sha256": canonical_sha256(job),
        "candidate_stage": stage,
        "candidate_worker": worker,
        "prospective_model_fit_attempts": prospective["model_fit_attempts"],
        "prospective_cpu_workers": prospective["cpu_workers"],
        "prospective_gpu_workers": prospective["gpu_workers"],
        "proposed_candidate_admissible": len(reasons) == 0,
        "reasons": reasons,
        "resource_breaches": list(snapshot["breaches"]),
    }
    candidate["check_sha256"] = canonical_sha256(candidate)
    return candidate


def evaluate_u0_candidate(manifest, events, resources, *, job_id, expected_head_sha256):
    """Pure prospective U0 candidate check; never a scientific permit."""
    try:
        return _evaluate_u0_candidate(manifest, events, resources, job_id, expected_head_sha256)
    except AdmissionError:
        raise
    except Exception:
        raise AdmissionError("invalid_admission_input") from None


def require_scientific_execution(*args, **kwargs):
    """Always deny: this leaf grants no scientific execution authority."""
    raise AdmissionError("scientific_execution_not_authorized") from None
