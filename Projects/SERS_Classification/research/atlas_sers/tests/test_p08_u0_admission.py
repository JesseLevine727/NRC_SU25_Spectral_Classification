"""Synthetic, no-execution tests for the T046 exact-U0 candidate adapter.

Every manifest, event and resource reading here is invented.  These tests
never run jobs, read data, fit models or authorize execution.
"""

from __future__ import annotations

import copy
import json

import pytest

from atlas_sers.evaluation import p08_u0_admission as admission
from atlas_sers.evaluation.p08_attempt_journal import replay_attempt_journal
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256

_PROPOSAL = "6639a32c1dd930612ead6ae59adf9aff5831904e883081f16c3490721af5089a"
_REAL_MANIFEST_SHA256 = "aac8523a1614610cf99cf1ab548d8970b4f8054fe8821b5250c347376e30b28a"
_MANIFEST_SCHEMA = "nato-sers-p08-attempt-manifest-v1"
_RECEIPT = "a" * 64
_GIB = 1024**3
_NANOS = 10**9

_CANDIDATE_KEYS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "proposal_sha256",
        "manifest_sha256",
        "journal_head_sha256",
        "journal_summary_sha256",
        "resource_snapshot_sha256",
        "candidate_sha256",
        "candidate_stage",
        "candidate_worker",
        "prospective_model_fit_attempts",
        "prospective_cpu_workers",
        "prospective_gpu_workers",
        "proposed_candidate_admissible",
        "reasons",
        "resource_breaches",
        "check_sha256",
    }
)

_UNSET = object()


class _StrSubclass(str):
    pass


class _HostileReason:
    def __hash__(self):
        raise AssertionError("hash must not be consulted")

    def __eq__(self, other):
        raise AssertionError("eq must not be consulted")

    def __str__(self):
        raise AssertionError("str must not be consulted")


class _ReasonSubclass(str):
    pass


def _build_manifest(*, fit_count=78, cpu_fits=42, proposal_sha256=_PROPOSAL):
    jobs = []
    for index in range(1, fit_count + 1):
        worker = "cpu" if index <= cpu_fits else "gpu"
        jobs.append(
            {
                "job_id": f"fit{index:03d}",
                "stage": "source_fit",
                "worker": worker,
                "dependencies": [],
            }
        )
    for index in range(1, fit_count + 1):
        worker = "cpu" if index <= cpu_fits else "gpu"
        jobs.append(
            {
                "job_id": f"pred{index:03d}",
                "stage": "source_validation_prediction",
                "worker": worker,
                "dependencies": [f"fit{index:03d}"],
            }
        )
    payload = {
        "schema_version": _MANIFEST_SCHEMA,
        "execution_authorized": False,
        "proposal_sha256": proposal_sha256,
        "jobs": jobs,
    }
    manifest = dict(payload)
    manifest["manifest_sha256"] = canonical_sha256(payload)
    return manifest


@pytest.fixture
def u0_manifest():
    return _build_manifest()


@pytest.fixture
def patched(monkeypatch, u0_manifest):
    monkeypatch.setattr(admission, "U0_MANIFEST_SHA256", u0_manifest["manifest_sha256"])
    return u0_manifest


def _event(
    seq,
    previous_sha256,
    session_id,
    event_type,
    *,
    elapsed_ns=0,
    artifact_bytes=0,
    job_id=None,
    status=None,
    receipt_sha256=None,
):
    payload = {
        "seq": seq,
        "previous_sha256": previous_sha256,
        "session_id": session_id,
        "event_type": event_type,
        "elapsed_ns": elapsed_ns,
        "artifact_bytes": artifact_bytes,
        "job_id": job_id,
        "status": status,
        "receipt_sha256": receipt_sha256,
    }
    payload["event_sha256"] = canonical_sha256(payload)
    return payload


def _chain(manifest, specs):
    events = []
    previous = manifest["manifest_sha256"]
    for index, spec in enumerate(specs, start=1):
        event = _event(index, previous, **spec)
        events.append(event)
        previous = event["event_sha256"]
    return events


def _open_spec(session_id=1):
    return {"session_id": session_id, "event_type": "session_open"}


def _progress_spec(*, elapsed_ns=0, artifact_bytes=0, session_id=1):
    return {
        "session_id": session_id,
        "event_type": "progress",
        "elapsed_ns": elapsed_ns,
        "artifact_bytes": artifact_bytes,
    }


def _start_spec(job_id, *, elapsed_ns=0, artifact_bytes=0, session_id=1):
    return {
        "session_id": session_id,
        "event_type": "attempt_start",
        "job_id": job_id,
        "elapsed_ns": elapsed_ns,
        "artifact_bytes": artifact_bytes,
    }


def _finish_spec(job_id, status, *, elapsed_ns=0, artifact_bytes=0, session_id=1):
    return {
        "session_id": session_id,
        "event_type": "attempt_finish",
        "job_id": job_id,
        "status": status,
        "receipt_sha256": _RECEIPT,
        "elapsed_ns": elapsed_ns,
        "artifact_bytes": artifact_bytes,
    }


def _close_spec(*, elapsed_ns=0, artifact_bytes=0, session_id=1):
    return {
        "session_id": session_id,
        "event_type": "session_close",
        "elapsed_ns": elapsed_ns,
        "artifact_bytes": artifact_bytes,
    }


def _resources(**overrides):
    base = {
        "filesystem_free_bytes": 38 * _GIB,
        "process_tree_rss_bytes": 0,
        "cuda_allocated_bytes": 0,
        "cuda_reserved_bytes": 0,
        "cuda_device_used_bytes": 0,
        "active_cpu_workers": 0,
        "active_gpu_workers": 0,
        "model_threads": 1,
        "blas_threads": 1,
        "torch_threads": 1,
    }
    base.update(overrides)
    return base


def _run_candidate(manifest, events, *, job_id, head=_UNSET, resources=_UNSET):
    if head is _UNSET:
        head = events[-1]["event_sha256"] if events else manifest["manifest_sha256"]
    if resources is _UNSET:
        resources = _resources()
    return admission.evaluate_u0_candidate(
        manifest,
        events,
        resources,
        job_id=job_id,
        expected_head_sha256=head,
    )


def test_public_constants_and_signature():
    assert admission.U0_PROPOSAL_SHA256 == _PROPOSAL
    assert admission.U0_MANIFEST_SHA256 == _REAL_MANIFEST_SHA256
    with pytest.raises(TypeError):
        admission.evaluate_u0_candidate(object(), [], object(), "fit001", object())
    with pytest.raises(TypeError):
        admission.evaluate_u0_candidate(
            object(),
            [],
            object(),
            job_id="fit001",
            expected_head_sha256="0" * 64,
            expected_manifest_sha256="0" * 64,
        )
    with pytest.raises(TypeError):
        admission.evaluate_u0_candidate(
            object(),
            [],
            object(),
            job_id="fit001",
            expected_head_sha256="0" * 64,
            proposal_sha256=_PROPOSAL,
        )


def test_unpatched_invented_manifest_is_rejected(u0_manifest):
    events = _chain(u0_manifest, [_open_spec()])
    with pytest.raises(admission.AdmissionError) as exc:
        _run_candidate(u0_manifest, events, job_id="fit001")
    assert exc.value.reason_code == "invalid_journal"


def test_open_session_candidates_are_prospectively_admissible(patched):
    manifest = patched
    events = _chain(manifest, [_open_spec()])
    head = events[-1]["event_sha256"]

    fit = _run_candidate(manifest, events, job_id="fit001", head=head)
    assert isinstance(fit, dict)
    assert "summary" not in fit
    assert set(fit) == set(_CANDIDATE_KEYS)
    assert fit["execution_authorized"] is False
    assert fit["proposed_candidate_admissible"] is True
    assert fit["reasons"] == []
    assert fit["candidate_stage"] == "source_fit"
    assert fit["candidate_worker"] == "cpu"
    assert fit["prospective_model_fit_attempts"] == 1
    assert fit["prospective_cpu_workers"] == 1
    assert fit["prospective_gpu_workers"] == 0
    body = {key: value for key, value in fit.items() if key != "check_sha256"}
    assert canonical_sha256(body) == fit["check_sha256"]
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    assert fit["candidate_sha256"] == canonical_sha256(jobs["fit001"])
    assert "fit001" not in json.dumps(fit)

    gpu = _run_candidate(manifest, events, job_id="fit043", head=head)
    assert gpu["execution_authorized"] is False
    assert gpu["proposed_candidate_admissible"] is True
    assert gpu["candidate_worker"] == "gpu"
    assert gpu["prospective_model_fit_attempts"] == 1
    assert gpu["prospective_cpu_workers"] == 0
    assert gpu["prospective_gpu_workers"] == 1


def test_empty_and_closed_journals_block_session_not_open(patched):
    manifest = patched

    empty = _run_candidate(manifest, [], job_id="fit001", head=manifest["manifest_sha256"])
    assert empty["proposed_candidate_admissible"] is False
    assert empty["reasons"] == ["session_not_open"]

    summary = replay_attempt_journal(
        manifest,
        [],
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_head_sha256=manifest["manifest_sha256"],
    )
    assert summary["restart_allowed"] is True
    assert empty["proposed_candidate_admissible"] is False

    closed_events = _chain(manifest, [_open_spec(), _close_spec(elapsed_ns=1)])
    closed = _run_candidate(manifest, closed_events, job_id="fit001")
    assert closed["proposed_candidate_admissible"] is False
    assert closed["reasons"] == ["session_not_open"]


def test_prediction_requires_succeeded_fit(patched):
    manifest = patched

    open_events = _chain(manifest, [_open_spec()])
    blocked = _run_candidate(manifest, open_events, job_id="pred001")
    assert blocked["reasons"] == ["fit_dependency_not_succeeded"]
    assert blocked["proposed_candidate_admissible"] is False

    done_events = _chain(
        manifest,
        [
            _open_spec(),
            _start_spec("fit001", elapsed_ns=1),
            _finish_spec("fit001", "succeeded", elapsed_ns=2),
        ],
    )
    allowed = _run_candidate(manifest, done_events, job_id="pred001")
    assert allowed["reasons"] == []
    assert allowed["proposed_candidate_admissible"] is True


def test_job_already_attempted(patched):
    manifest = patched

    running = _chain(manifest, [_open_spec(), _start_spec("fit001")])
    running_result = _run_candidate(
        manifest,
        running,
        job_id="fit001",
        resources=_resources(active_cpu_workers=1),
    )
    assert running_result["reasons"] == ["job_already_attempted"]

    finished = _chain(
        manifest,
        [
            _open_spec(),
            _start_spec("fit001", elapsed_ns=1),
            _finish_spec("fit001", "succeeded", elapsed_ns=2),
        ],
    )
    finished_result = _run_candidate(manifest, finished, job_id="fit001")
    assert finished_result["reasons"] == ["job_already_attempted"]

    next_session = _chain(
        manifest,
        [
            _open_spec(),
            _start_spec("fit001", elapsed_ns=1),
            _finish_spec("fit001", "succeeded", elapsed_ns=2),
            _close_spec(elapsed_ns=3),
            _open_spec(session_id=2),
            _progress_spec(elapsed_ns=1, session_id=2),
        ],
    )
    replay_result = _run_candidate(manifest, next_session, job_id="fit001")
    assert replay_result["reasons"] == ["job_already_attempted"]


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_failed_attempt_blocks_other_fit_and_dependency(patched, status):
    manifest = patched
    events = _chain(
        manifest,
        [
            _open_spec(),
            _start_spec("fit001", elapsed_ns=1),
            _finish_spec("fit001", status, elapsed_ns=2),
        ],
    )

    other = _run_candidate(manifest, events, job_id="fit002")
    assert other["reasons"] == ["prior_failed_or_interrupted_attempt_requires_review"]

    prediction = _run_candidate(manifest, events, job_id="pred001")
    assert prediction["reasons"] == [
        "prior_failed_or_interrupted_attempt_requires_review",
        "fit_dependency_not_succeeded",
    ]
    assert prediction["proposed_candidate_admissible"] is False


def test_concurrency_limits_single_session(patched):
    manifest = patched

    both_cpu = _chain(
        manifest,
        [
            _open_spec(),
            _start_spec("fit001"),
            _start_spec("fit002"),
        ],
    )
    cpu_resources = _resources(active_cpu_workers=2)
    third_cpu = _run_candidate(manifest, both_cpu, job_id="fit003", resources=cpu_resources)
    assert "cpu_worker_capacity_exhausted" in third_cpu["reasons"]
    assert third_cpu["proposed_candidate_admissible"] is False
    gpu_free = _run_candidate(manifest, both_cpu, job_id="fit043", resources=cpu_resources)
    assert gpu_free["proposed_candidate_admissible"] is True

    mixed = _chain(
        manifest,
        [
            _open_spec(),
            _start_spec("fit001"),
            _start_spec("fit043"),
        ],
    )
    mixed_resources = _resources(active_cpu_workers=1, active_gpu_workers=1)
    second_gpu = _run_candidate(manifest, mixed, job_id="fit044", resources=mixed_resources)
    assert "gpu_worker_capacity_exhausted" in second_gpu["reasons"]
    second_cpu = _run_candidate(manifest, mixed, job_id="fit002", resources=mixed_resources)
    assert second_cpu["proposed_candidate_admissible"] is True


def test_worker_count_mismatch(patched):
    manifest = patched

    cpu_events = _chain(manifest, [_open_spec(), _start_spec("fit001")])
    with pytest.raises(admission.AdmissionError) as cpu_exc:
        _run_candidate(
            manifest,
            cpu_events,
            job_id="fit002",
            resources=_resources(active_cpu_workers=0),
        )
    assert cpu_exc.value.reason_code == "worker_count_mismatch"

    gpu_events = _chain(manifest, [_open_spec(), _start_spec("fit043")])
    with pytest.raises(admission.AdmissionError) as gpu_exc:
        _run_candidate(
            manifest,
            gpu_events,
            job_id="fit044",
            resources=_resources(active_gpu_workers=0),
        )
    assert gpu_exc.value.reason_code == "worker_count_mismatch"


def test_full_fit_budget_and_retry(patched):
    manifest = patched
    specs = [_open_spec()]
    elapsed = 0
    for index in range(1, 79):
        elapsed += 1
        specs.append(_start_spec(f"fit{index:03d}", elapsed_ns=elapsed))
        elapsed += 1
        specs.append(_finish_spec(f"fit{index:03d}", "succeeded", elapsed_ns=elapsed))
    events = _chain(manifest, specs)
    head = events[-1]["event_sha256"]

    summary = replay_attempt_journal(
        manifest,
        events,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_head_sha256=head,
    )
    assert len(summary["attempts"]) == 78
    assert summary["succeeded_job_ids"] == [f"fit{index:03d}" for index in range(1, 79)]
    assert summary["active_cpu_workers"] == 0
    assert summary["active_gpu_workers"] == 0

    pending = _run_candidate(manifest, events, job_id="pred001", head=head)
    assert pending["prospective_model_fit_attempts"] == 78
    assert pending["proposed_candidate_admissible"] is True

    retry = _run_candidate(manifest, events, job_id="fit001", head=head)
    assert retry["reasons"] == ["job_already_attempted", "model_fit_capacity_exhausted"]


@pytest.mark.parametrize(
    "elapsed_ns,artifact_bytes,overrides,expected_breach",
    [
        (5400 * _NANOS, 0, {}, "active_wall_budget_exhausted"),
        (5400 * _NANOS - 1, 0, {}, None),
        (0, 8 * _GIB, {}, "artifact_budget_exhausted"),
        (0, 0, {"process_tree_rss_bytes": 16 * _GIB}, None),
        (0, 0, {"process_tree_rss_bytes": 16 * _GIB + 1}, "process_tree_memory_exceeded"),
        (0, 0, {"cuda_allocated_bytes": 8 * _GIB}, None),
        (0, 0, {"cuda_allocated_bytes": 8 * _GIB + 1}, "cuda_allocated_memory_exceeded"),
        (0, 0, {"filesystem_free_bytes": 38 * _GIB}, None),
        (0, 0, {"filesystem_free_bytes": 38 * _GIB - 1}, "filesystem_reserve_insufficient"),
        (0, 0, {"filesystem_free_bytes": 37 * _GIB}, "filesystem_reserve_insufficient"),
        (0, 1 * _GIB, {"filesystem_free_bytes": 37 * _GIB}, None),
        (0, 0, {"cuda_reserved_bytes": 10**15, "cuda_device_used_bytes": 10**15}, None),
        (0, 0, {"model_threads": 0}, "worker_thread_limit_violated"),
        (0, 0, {"blas_threads": 2}, "worker_thread_limit_violated"),
        (0, 0, {"torch_threads": 0}, "worker_thread_limit_violated"),
    ],
)
def test_resource_bounds(patched, elapsed_ns, artifact_bytes, overrides, expected_breach):
    manifest = patched
    specs = [_open_spec()]
    if elapsed_ns or artifact_bytes:
        specs.append(_progress_spec(elapsed_ns=elapsed_ns, artifact_bytes=artifact_bytes))
    events = _chain(manifest, specs)
    result = _run_candidate(manifest, events, job_id="fit001", resources=_resources(**overrides))
    if expected_breach is None:
        assert result["resource_breaches"] == []
        assert result["proposed_candidate_admissible"] is True
    else:
        assert expected_breach in result["resource_breaches"]
        assert "resource_limits_breached" in result["reasons"]
        assert result["proposed_candidate_admissible"] is False


@pytest.mark.parametrize(
    "manifest",
    [
        _build_manifest(proposal_sha256="0" * 64),
        _build_manifest(fit_count=77, cpu_fits=41),
        _build_manifest(fit_count=78, cpu_fits=41),
    ],
    ids=["wrong-proposal", "fewer-fits", "wrong-worker-distribution"],
)
def test_smoke_binding_mismatch(monkeypatch, manifest):
    monkeypatch.setattr(admission, "U0_MANIFEST_SHA256", manifest["manifest_sha256"])
    events = _chain(manifest, [_open_spec()])
    head = events[-1]["event_sha256"]
    summary = replay_attempt_journal(
        manifest,
        events,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_head_sha256=head,
    )
    assert summary["journal_state"] == "open"
    with pytest.raises(admission.AdmissionError) as exc:
        _run_candidate(manifest, events, job_id="fit001", head=head)
    assert exc.value.reason_code == "smoke_binding_mismatch"


@pytest.mark.parametrize(
    "bad_id",
    [None, True, False, "", "   ", "\t", _StrSubclass("fit001"), "\ud800"],
    ids=["none", "true", "false", "empty", "spaces", "tab", "subclass", "surrogate"],
)
def test_invalid_candidate_ids(patched, bad_id):
    manifest = patched
    events = _chain(manifest, [_open_spec()])
    with pytest.raises(admission.AdmissionError) as exc:
        _run_candidate(manifest, events, job_id=bad_id)
    assert exc.value.reason_code == "invalid_candidate"
    assert str(exc.value) == "invalid_candidate"


@pytest.mark.parametrize("job_id", ["fit999", "pred999", "unknown-job"])
def test_unregistered_job(patched, job_id):
    manifest = patched
    events = _chain(manifest, [_open_spec()])
    with pytest.raises(admission.AdmissionError) as exc:
        _run_candidate(manifest, events, job_id=job_id)
    assert exc.value.reason_code == "unregistered_job"


def test_wrong_head_is_invalid_journal(patched):
    manifest = patched
    events = _chain(manifest, [_open_spec()])
    with pytest.raises(admission.AdmissionError) as exc:
        _run_candidate(manifest, events, job_id="fit001", head="0" * 64)
    assert exc.value.reason_code == "invalid_journal"


def test_modified_event_without_reseal_is_invalid_journal(patched):
    manifest = patched
    events = _chain(manifest, [_open_spec(), _progress_spec(elapsed_ns=5)])
    events[1] = dict(events[1])
    events[1]["elapsed_ns"] = 6
    with pytest.raises(admission.AdmissionError) as exc:
        _run_candidate(
            manifest,
            events,
            job_id="fit001",
            head=events[1]["event_sha256"],
        )
    assert exc.value.reason_code == "invalid_journal"


def test_modified_manifest_without_reseal_is_invalid_journal(patched):
    manifest = patched
    events = _chain(manifest, [_open_spec()])
    tampered = dict(manifest)
    tampered["proposal_sha256"] = "0" * 64
    with pytest.raises(admission.AdmissionError) as exc:
        _run_candidate(
            tampered,
            events,
            job_id="fit001",
            head=events[-1]["event_sha256"],
        )
    assert exc.value.reason_code == "invalid_journal"


def test_invalid_resources_none(patched):
    manifest = patched
    events = _chain(manifest, [_open_spec()])
    with pytest.raises(admission.AdmissionError) as exc:
        _run_candidate(manifest, events, job_id="fit001", resources=None)
    assert exc.value.reason_code == "invalid_resources"


def test_invalid_resources_extra_key(patched):
    manifest = patched
    events = _chain(manifest, [_open_spec()])
    resources = _resources()
    resources["unexpected"] = 3
    with pytest.raises(admission.AdmissionError) as exc:
        _run_candidate(manifest, events, job_id="fit001", resources=resources)
    assert exc.value.reason_code == "invalid_resources"


def test_invalid_resources_bool_count(patched):
    manifest = patched
    events = _chain(manifest, [_open_spec()])
    resources = _resources(active_cpu_workers=True)
    with pytest.raises(admission.AdmissionError) as exc:
        _run_candidate(manifest, events, job_id="fit001", resources=resources)
    assert exc.value.reason_code == "invalid_resources"


def test_deterministic_non_mutating_and_isolated(patched):
    manifest = patched
    events = _chain(manifest, [_open_spec(), _start_spec("fit001")])
    resources = _resources(active_cpu_workers=1)
    manifest_copy = copy.deepcopy(manifest)
    events_copy = copy.deepcopy(events)
    resources_copy = copy.deepcopy(resources)

    first = _run_candidate(manifest, events, job_id="fit002", resources=resources)
    second = _run_candidate(manifest, events, job_id="fit002", resources=resources)
    assert first == second

    first["reasons"].append("tampered")
    first["resource_breaches"].append("tampered")

    third = _run_candidate(manifest, events, job_id="fit002", resources=resources)
    assert third == second
    assert manifest == manifest_copy
    assert events == events_copy
    assert resources == resources_copy


def test_journal_and_snapshot_bindings(patched):
    manifest = patched
    events = _chain(manifest, [_open_spec(), _start_spec("fit001")])
    head = events[-1]["event_sha256"]
    summary = replay_attempt_journal(
        manifest,
        events,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_head_sha256=head,
    )
    result = _run_candidate(
        manifest,
        events,
        job_id="fit002",
        resources=_resources(active_cpu_workers=1),
    )
    assert result["journal_head_sha256"] == summary["head_sha256"] == head
    assert result["journal_summary_sha256"] == summary["summary_sha256"]
    assert summary["summary_sha256"] == canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    assert result["manifest_sha256"] == manifest["manifest_sha256"]
    body = {key: value for key, value in result.items() if key != "check_sha256"}
    assert canonical_sha256(body) == result["check_sha256"]


def test_admission_error_reason_normalization():
    assert admission.AdmissionError("invalid_journal").reason_code == "invalid_journal"
    for bad in (_HostileReason(), ["invalid_journal"], _ReasonSubclass("invalid_journal"), 7, None):
        error = admission.AdmissionError(bad)
        assert error.reason_code == "invalid_admission_input"


def test_unexpected_internal_error_is_suppressed(monkeypatch):
    def _boom(*args, **kwargs):
        raise ValueError("private sentinel")

    monkeypatch.setattr(admission, "_evaluate_u0_candidate", _boom)
    with pytest.raises(admission.AdmissionError) as exc:
        admission.evaluate_u0_candidate(
            object(),
            [],
            object(),
            job_id="x",
            expected_head_sha256="0" * 64,
        )
    assert exc.value.reason_code == "invalid_admission_input"
    assert "private sentinel" not in str(exc.value)
    assert exc.value.__suppress_context__ is True


def test_keyboard_interrupt_propagates(monkeypatch):
    def _boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(admission, "_evaluate_u0_candidate", _boom)
    with pytest.raises(KeyboardInterrupt):
        admission.evaluate_u0_candidate(
            object(),
            [],
            object(),
            job_id="x",
            expected_head_sha256="0" * 64,
        )


def test_require_scientific_execution_always_denies(patched):
    manifest = patched
    events = _chain(manifest, [_open_spec()])
    admissible = _run_candidate(manifest, events, job_id="fit001")
    assert admissible["proposed_candidate_admissible"] is True

    with pytest.raises(admission.AdmissionError) as exc:
        admission.require_scientific_execution(
            execution_authorized=True, job_id="fit001", manifest=manifest
        )
    assert exc.value.reason_code == "scientific_execution_not_authorized"


def test_cumulative_usage_resource_hash_survives_clean_pause(patched):
    manifest = patched
    second_open = _open_spec(session_id=2)
    second_open["artifact_bytes"] = _GIB
    specs = [
        _open_spec(session_id=1),
        _start_spec("fit001", elapsed_ns=1),
        _finish_spec("fit001", "succeeded", elapsed_ns=2, artifact_bytes=_GIB),
        _close_spec(session_id=1, elapsed_ns=5399 * _NANOS, artifact_bytes=_GIB),
        second_open,
    ]
    resources = _resources(filesystem_free_bytes=37 * _GIB)

    admissible = _run_candidate(
        manifest, _chain(manifest, specs), job_id="fit002", resources=resources
    )
    assert admissible["proposed_candidate_admissible"] is True
    assert admissible["prospective_model_fit_attempts"] == 2

    progress = _progress_spec(session_id=2, elapsed_ns=2 * _NANOS, artifact_bytes=_GIB)
    blocked = _run_candidate(
        manifest,
        _chain(manifest, specs + [progress]),
        job_id="fit002",
        resources=resources,
    )
    assert blocked["proposed_candidate_admissible"] is False
    assert blocked["reasons"] == ["resource_limits_breached"]
    assert blocked["resource_breaches"] == ["active_wall_budget_exhausted"]
    assert blocked["prospective_model_fit_attempts"] == 2

    usage = {
        "model_fit_attempts": 1,
        "scalar_calibration_attempts": 0,
        "active_wall_ns": 5401 * _NANOS,
        "new_artifact_bytes": _GIB,
    }
    from atlas_sers.evaluation.p08_resources import evaluate_resource_snapshot

    snapshot = evaluate_resource_snapshot("U0", usage, resources)
    assert blocked["resource_snapshot_sha256"] == snapshot["snapshot_sha256"]


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_combined_stop_reasons_survive_clean_pause(patched, status):
    manifest = patched
    events = _chain(
        manifest,
        [
            _open_spec(session_id=1),
            _start_spec("fit001", elapsed_ns=1),
            _finish_spec("fit001", status, elapsed_ns=2),
            _close_spec(session_id=1, elapsed_ns=3),
            _open_spec(session_id=2),
            _close_spec(session_id=2, elapsed_ns=5400 * _NANOS),
        ],
    )
    result = _run_candidate(manifest, events, job_id="pred001", resources=_resources())
    assert result["proposed_candidate_admissible"] is False
    assert result["reasons"] == [
        "session_not_open",
        "prior_failed_or_interrupted_attempt_requires_review",
        "fit_dependency_not_succeeded",
        "resource_limits_breached",
    ]
    assert result["resource_breaches"] == ["active_wall_budget_exhausted"]
    assert result["execution_authorized"] is False
    assert result["prospective_model_fit_attempts"] == 1
