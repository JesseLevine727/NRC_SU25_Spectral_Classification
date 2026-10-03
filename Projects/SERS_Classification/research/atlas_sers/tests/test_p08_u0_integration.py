"""P08-T077 test-only synthetic integration composition coverage.

This module exercises COMPOSITION of already-accepted no-fit U0 safety leaves:
cooperative store ownership plus durable journal progress, serial
resource/candidate evidence, prospective attempt admission and read-only
receipt/artifact byte verification.  Every manifest, event, resource reading
and artifact here is invented.  The coverage is synthetic integration testing
only: it is not live runtime acceptance, not fresh durable metering, not
semantic checkpoint/prediction validity, not scientific semantic acceptance
and grants no launch permit.  Scientific execution entry always denies; a
byte-valid receipt still reports ``scientific_semantics_verified`` False and
``execution_authorized`` False.  No production code or existing test is
modified.

The synthetic route injects the fake CUDA stand-in for CPU jobs as well;
this is a declared test artifact, not evidence of GPU utilization.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import types

import pytest
import threadpoolctl

from atlas_sers.evaluation import p08_serial_resources as serial
from atlas_sers.evaluation import p08_terminal_receipt as receipt
from atlas_sers.evaluation import p08_u0_admission as admission
from atlas_sers.evaluation import p08_u0_store as store
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256
from tests.p08_store_fixtures import (
    GIB,
    NANOS,
    append,
    bind_manifest,
    make_event,
    readtree,
)
from tests.p08_store_fixtures import resources as resource_fixture

_RECEIPT_SCHEMA = "nato-sers-p08-terminal-receipt-v1"
_RSS_BYTES = 256 * 1024 * 1024
_FREE_BYTES = 64 * GIB
_PROGRESS_ELAPSED = 1 * NANOS
_PROGRESS_BYTES = 4096
_CLOSE_ELAPSED = 2 * NANOS
_SECOND_ELAPSED = 5 * NANOS
_SECOND_BYTES = 5000
_SECOND_CLOSE_ELAPSED = 6 * NANOS


class _Clock:
    """Deterministic increasing monotonic clock, scoped to ``serial.time``."""

    def __init__(self):
        self.value = 0

    def monotonic_ns(self):
        self.value += 1_000_000
        return self.value


class _FakeCuda:
    def is_initialized(self):
        return True

    def current_device(self):
        return 0

    def memory_allocated(self, device):
        return 0

    def memory_reserved(self, device):
        return 0

    def max_memory_allocated(self, device):
        return 0

    def mem_get_info(self, device):
        return (_FREE_BYTES, _FREE_BYTES)


class _FakeTorch:
    """Tiny injected torch stand-in: one thread each, fake CUDA observations."""

    def __init__(self):
        self.cuda = _FakeCuda()

    def get_num_threads(self):
        return 1

    def get_num_interop_threads(self):
        return 1


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(serial, "_assert_no_children", lambda: None)
    monkeypatch.setattr(serial, "_read_vm_rss_bytes", lambda: _RSS_BYTES)
    monkeypatch.setattr(serial, "_measure_filesystem", lambda output_directory: _FREE_BYTES)
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 1}])
    monkeypatch.setattr(serial, "time", _Clock())
    return types.SimpleNamespace(torch=_FakeTorch())


def _job(manifest, job_id):
    for job in manifest["jobs"]:
        if job["job_id"] == job_id:
            return job
    raise AssertionError(job_id)


def _serial_report(manifest, state, job_id, tmp_path, env):
    return serial.check_serial_u0_candidate(
        manifest,
        state["events"],
        job_id=job_id,
        expected_head_sha256=state["summary"]["head_sha256"],
        output_directory=str(tmp_path),
        torch_module=env.torch,
        model_threads=1,
    )


def _start(owner, manifest, job_id, tmp_path, env):
    state = owner.snapshot()
    summary = state["summary"]
    job = _job(manifest, job_id)
    report = _serial_report(manifest, state, job_id, tmp_path, env)
    assert report["execution_authorized"] is False
    assert report["proposed_serial_candidate_admissible"] is True, report["reasons"]
    resources = report["resources"]
    candidate_check = report["candidate_check"]
    assert resources["active_cpu_workers"] == 0
    assert resources["active_gpu_workers"] == 0
    expected_active = {"cpu": (1, 0), "gpu": (0, 1)}[job["worker"]]
    assert candidate_check["prospective_cpu_workers"] == expected_active[0]
    assert candidate_check["prospective_gpu_workers"] == expected_active[1]
    assert candidate_check["candidate_stage"] == job["stage"]
    assert candidate_check["candidate_worker"] == job["worker"]
    event = make_event(
        state,
        "attempt_start",
        elapsed_ns=summary["active_session_elapsed_ns"],
        artifact_bytes=summary["new_artifact_bytes"],
        job_id=job_id,
    )
    owner.append_event(
        event,
        expected_head_sha256=summary["head_sha256"],
        resources=resources,
    )
    active = owner.snapshot()["summary"]
    assert active["active_cpu_workers"] == expected_active[0]
    assert active["active_gpu_workers"] == expected_active[1]
    return report


def _make_bundle(manifest, state, job_id, status, declared_bytes):
    summary = state["summary"]
    job = _job(manifest, job_id)
    starts = [
        event
        for event in state["events"]
        if event.get("event_type") == "attempt_start" and event.get("job_id") == job_id
    ]
    assert len(starts) == 1
    artifact_name = f"{job_id}-artifact.bin"
    receipt_name = f"{job_id}-receipt.json"
    entry = {
        "name": artifact_name,
        "size_bytes": len(declared_bytes),
        "sha256": hashlib.sha256(declared_bytes).hexdigest(),
    }
    payload = {
        "schema_version": _RECEIPT_SCHEMA,
        "execution_authorized": False,
        "proposal_sha256": admission.U0_PROPOSAL_SHA256,
        "manifest_sha256": admission.U0_MANIFEST_SHA256,
        "job_id": job_id,
        "stage": job["stage"],
        "worker": job["worker"],
        "session_id": summary["active_session_id"],
        "start_event_sha256": starts[0]["event_sha256"],
        "status": status,
        "artifacts": [entry],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    terminal = {
        "seq": len(state["events"]) + 1,
        "previous_sha256": summary["head_sha256"],
        "session_id": summary["active_session_id"],
        "event_type": "attempt_finish",
        "elapsed_ns": summary["active_session_elapsed_ns"],
        "artifact_bytes": summary["new_artifact_bytes"],
        "job_id": job_id,
        "status": status,
        "receipt_sha256": payload["receipt_sha256"],
    }
    terminal["event_sha256"] = canonical_sha256(terminal)
    return terminal, payload, artifact_name, receipt_name


def _write_bundle(root, receipt_name, receipt_payload, artifact_name, disk_bytes):
    base = pathlib.Path(root)
    base.mkdir(parents=True, exist_ok=True)
    (base / artifact_name).write_bytes(disk_bytes)
    text = (
        json.dumps(
            receipt_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    (base / receipt_name).write_bytes(text.encode())


def _finish(owner, manifest, job_id, root, status="succeeded"):
    state = owner.snapshot()
    declared = f"INVENTED-NON-MODEL-ARTIFACT:{job_id}\n".encode()
    terminal, payload, artifact_name, receipt_name = _make_bundle(
        manifest, state, job_id, status, declared
    )
    _write_bundle(root, receipt_name, payload, artifact_name, declared)
    report = receipt.verify_terminal_receipt(
        str(root),
        receipt_name,
        manifest,
        state["events"],
        terminal,
        expected_head_sha256=state["summary"]["head_sha256"],
        expected_artifact_names=[artifact_name],
    )
    assert report["execution_authorized"] is False
    assert report["byte_integrity_verified"] is True
    assert report["scientific_semantics_verified"] is False
    owner.append_event(
        terminal,
        expected_head_sha256=state["summary"]["head_sha256"],
        resources=None,
    )
    return report


# ``_FakeCuda`` is injected for both the CPU and GPU routes by ``env``; the
# synthetic route declares that fake CUDA observation for CPU jobs too.
@pytest.mark.parametrize(("index", "worker"), [(1, "cpu"), (43, "gpu")])
def test_full_success_sequence_survives_reopen(index, worker, tmp_path, monkeypatch, env):
    manifest = bind_manifest(monkeypatch)
    fit_id = f"fit{index:03d}"
    pred_id = f"pred{index:03d}"
    assert _job(manifest, fit_id)["stage"] == "source_fit"
    assert _job(manifest, fit_id)["worker"] == worker
    assert _job(manifest, pred_id)["stage"] == "source_validation_prediction"
    assert _job(manifest, pred_id)["worker"] == worker
    root = tmp_path / "journal"
    artifacts = tmp_path / "receipts"

    owner = store.create_store(root, manifest)
    try:
        assert owner.snapshot()["summary"]["head_sha256"] == manifest["manifest_sha256"]
        append(owner, "session_open")
        append(owner, "progress", elapsed_ns=_PROGRESS_ELAPSED, artifact_bytes=_PROGRESS_BYTES)
        durable = owner.snapshot()
        assert durable["summary"]["active_session_elapsed_ns"] == _PROGRESS_ELAPSED
        assert durable["summary"]["new_artifact_bytes"] == _PROGRESS_BYTES

        _start(owner, manifest, fit_id, tmp_path, env)
        active = owner.snapshot()
        with pytest.raises(serial.SerialResourceError) as excinfo:
            _serial_report(manifest, active, pred_id, tmp_path, env)
        assert excinfo.value.reason_code == "serial_active_jobs_present"

        _finish(owner, manifest, fit_id, artifacts)
        _start(owner, manifest, pred_id, tmp_path, env)
        _finish(owner, manifest, pred_id, artifacts)
        append(owner, "session_close", elapsed_ns=_CLOSE_ELAPSED)
        first = owner.snapshot()["summary"]
        head = first["head_sha256"]
        assert first["model_fit_attempts"] == 1
        assert first["source_prediction_attempts"] == 1
        assert first["closed_session_wall_ns"] == _CLOSE_ELAPSED
    finally:
        owner.close()

    owner = store.open_store(root, expected_head_sha256=head)
    try:
        assert owner.snapshot()["summary"]["succeeded_job_ids"] == sorted([fit_id, pred_id])
        reopened_before = owner.snapshot()["summary"]
        assert reopened_before["model_fit_attempts"] == 1
        assert reopened_before["source_prediction_attempts"] == 1
        assert reopened_before["new_artifact_bytes"] == _PROGRESS_BYTES
        assert reopened_before["active_session_elapsed_ns"] == 0
        assert reopened_before["active_wall_ns"] == _CLOSE_ELAPSED
        append(owner, "session_open")
        reopened = owner.snapshot()["summary"]
        assert reopened["model_fit_attempts"] == 1
        assert reopened["source_prediction_attempts"] == 1
        assert reopened["new_artifact_bytes"] == _PROGRESS_BYTES
        assert reopened["active_session_elapsed_ns"] == 0
        assert reopened["active_wall_ns"] == _CLOSE_ELAPSED
        append(owner, "progress", elapsed_ns=_SECOND_ELAPSED, artifact_bytes=_SECOND_BYTES)
        append(owner, "session_close", elapsed_ns=_SECOND_CLOSE_ELAPSED)
        final = owner.snapshot()["summary"]
        assert final["model_fit_attempts"] == 1
        assert final["source_prediction_attempts"] == 1
        assert final["in_flight_job_ids"] == []
        assert final["succeeded_job_ids"] == sorted([fit_id, pred_id])
        assert final["closed_session_wall_ns"] == _CLOSE_ELAPSED + _SECOND_CLOSE_ELAPSED
        assert final["active_wall_ns"] == _CLOSE_ELAPSED + _SECOND_CLOSE_ELAPSED
        assert final["new_artifact_bytes"] == _SECOND_BYTES
        assert final["session_count"] == 2
    finally:
        owner.close()


def test_dependent_prediction_blocked_before_fit(tmp_path, monkeypatch, env):
    manifest = bind_manifest(monkeypatch)
    root = tmp_path / "journal"
    owner = store.create_store(root, manifest)
    try:
        append(owner, "session_open")
        append(owner, "progress", elapsed_ns=_PROGRESS_ELAPSED, artifact_bytes=_PROGRESS_BYTES)
        state = owner.snapshot()
        report = _serial_report(manifest, state, "pred001", tmp_path, env)
        assert report["proposed_serial_candidate_admissible"] is False
        assert "fit_dependency_not_succeeded" in report["reasons"]

        before_tree = readtree(root)
        event = make_event(
            state,
            "attempt_start",
            elapsed_ns=state["summary"]["active_session_elapsed_ns"],
            artifact_bytes=state["summary"]["new_artifact_bytes"],
            job_id="pred001",
        )
        with pytest.raises(store.StoreError) as excinfo:
            owner.append_event(
                event,
                expected_head_sha256=state["summary"]["head_sha256"],
                resources=report["resources"],
            )
        assert excinfo.value.reason_code == "invalid_journal"
        after = owner.snapshot()["summary"]
        assert after["attempts"] == {}
        assert after["head_sha256"] == state["summary"]["head_sha256"]
        assert readtree(root) == before_tree
    finally:
        owner.close()


def test_fresh_progress_required_then_accepted(tmp_path, monkeypatch, env):
    manifest = bind_manifest(monkeypatch)
    root = tmp_path / "journal"
    owner = store.create_store(root, manifest)
    try:
        append(owner, "session_open")
        state = owner.snapshot()
        before_tree = readtree(root)
        premature = make_event(
            state,
            "attempt_start",
            elapsed_ns=_PROGRESS_ELAPSED,
            artifact_bytes=_PROGRESS_BYTES,
            job_id="fit001",
        )
        with pytest.raises(store.StoreError) as excinfo:
            owner.append_event(
                premature,
                expected_head_sha256=state["summary"]["head_sha256"],
                resources=resource_fixture(filesystem_free_bytes=_FREE_BYTES),
            )
        assert excinfo.value.reason_code == "fresh_progress_required"
        assert readtree(root) == before_tree

        append(owner, "progress", elapsed_ns=_PROGRESS_ELAPSED, artifact_bytes=_PROGRESS_BYTES)
        _start(owner, manifest, "fit001", tmp_path, env)
        summary = owner.snapshot()["summary"]
        assert summary["model_fit_attempts"] == 1
        assert summary["in_flight_job_ids"] == ["fit001"]
    finally:
        owner.close()


def test_corrupted_artifact_blocks_and_keeps_job_in_flight(tmp_path, monkeypatch, env):
    manifest = bind_manifest(monkeypatch)
    root = tmp_path / "journal"
    artifacts = tmp_path / "receipts"
    owner = store.create_store(root, manifest)
    try:
        append(owner, "session_open")
        append(owner, "progress", elapsed_ns=_PROGRESS_ELAPSED, artifact_bytes=_PROGRESS_BYTES)
        _start(owner, manifest, "fit001", tmp_path, env)
        state = owner.snapshot()
        declared = b"INVENTED-NON-MODEL-ARTIFACT:fit001\n"
        corrupted = bytes([declared[0] ^ 0x01]) + declared[1:]
        assert len(corrupted) == len(declared)
        terminal, payload, artifact_name, receipt_name = _make_bundle(
            manifest, state, "fit001", "succeeded", declared
        )
        _write_bundle(artifacts, receipt_name, payload, artifact_name, corrupted)
        before_tree = readtree(root)
        with pytest.raises(receipt.ReceiptError) as excinfo:
            receipt.verify_terminal_receipt(
                str(artifacts),
                receipt_name,
                manifest,
                state["events"],
                terminal,
                expected_head_sha256=state["summary"]["head_sha256"],
                expected_artifact_names=[artifact_name],
            )
        assert excinfo.value.reason_code == "artifact_mismatch"
        after = owner.snapshot()["summary"]
        assert after["in_flight_job_ids"] == ["fit001"]
        assert after["succeeded_job_ids"] == []
        assert readtree(root) == before_tree
    finally:
        owner.close()


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_terminal_failure_consumes_attempt_and_persists(status, tmp_path, monkeypatch, env):
    manifest = bind_manifest(monkeypatch)
    root = tmp_path / "journal"
    artifacts = tmp_path / "receipts"
    owner = store.create_store(root, manifest)
    try:
        append(owner, "session_open")
        append(owner, "progress", elapsed_ns=_PROGRESS_ELAPSED, artifact_bytes=_PROGRESS_BYTES)
        _start(owner, manifest, "fit001", tmp_path, env)
        _finish(owner, manifest, "fit001", artifacts, status=status)
        summary = owner.snapshot()["summary"]
        assert summary["model_fit_attempts"] == 1
        assert summary["failed_job_ids"] == (["fit001"] if status == "failed" else [])
        assert summary["interrupted_job_ids"] == (["fit001"] if status == "interrupted" else [])

        with pytest.raises(serial.SerialResourceError) as excinfo:
            _serial_report(manifest, owner.snapshot(), "fit002", tmp_path, env)
        assert excinfo.value.reason_code == "prior_failed_or_interrupted_attempt_requires_review"

        state = owner.snapshot()
        before_tree = readtree(root)
        blocked = make_event(
            state,
            "attempt_start",
            elapsed_ns=state["summary"]["active_session_elapsed_ns"],
            artifact_bytes=state["summary"]["new_artifact_bytes"],
            job_id="fit002",
        )
        with pytest.raises(store.StoreError) as excinfo:
            owner.append_event(
                blocked,
                expected_head_sha256=state["summary"]["head_sha256"],
                resources=resource_fixture(filesystem_free_bytes=_FREE_BYTES),
            )
        assert excinfo.value.reason_code == "candidate_blocked"
        assert owner.snapshot() == state
        assert readtree(root) == before_tree

        append(owner, "session_close", elapsed_ns=_CLOSE_ELAPSED)
        head = owner.snapshot()["summary"]["head_sha256"]
    finally:
        owner.close()

    owner = store.open_store(root, expected_head_sha256=head)
    try:
        append(owner, "session_open")
        summary = owner.snapshot()["summary"]
        assert summary["model_fit_attempts"] == 1
        assert summary["source_prediction_attempts"] == 0
        assert summary["new_artifact_bytes"] == _PROGRESS_BYTES
        assert summary["active_wall_ns"] == _CLOSE_ELAPSED
        assert len(summary["attempts"]) == 1
        assert summary["attempts"]["fit001"]["status"] == status
        with pytest.raises(serial.SerialResourceError) as excinfo:
            _serial_report(manifest, owner.snapshot(), "fit002", tmp_path, env)
        assert excinfo.value.reason_code == "prior_failed_or_interrupted_attempt_requires_review"
        state = owner.snapshot()
        before_tree = readtree(root)
        blocked = make_event(state, "attempt_start", job_id="fit002")
        with pytest.raises(store.StoreError) as excinfo:
            owner.append_event(
                blocked,
                expected_head_sha256=state["summary"]["head_sha256"],
                resources=resource_fixture(filesystem_free_bytes=_FREE_BYTES),
            )
        assert excinfo.value.reason_code == "candidate_blocked"
        assert owner.snapshot() == state
        assert readtree(root) == before_tree
    finally:
        owner.close()


@pytest.mark.parametrize("with_attempt", [False, True])
def test_release_with_open_session_requires_review(with_attempt, tmp_path, monkeypatch, env):
    manifest = bind_manifest(monkeypatch)
    root = tmp_path / "journal"
    owner = store.create_store(root, manifest)
    try:
        append(owner, "session_open")
        if with_attempt:
            append(owner, "progress", elapsed_ns=_PROGRESS_ELAPSED, artifact_bytes=_PROGRESS_BYTES)
            _start(owner, manifest, "fit001", tmp_path, env)
        head = owner.snapshot()["summary"]["head_sha256"]
    finally:
        owner.close()

    before_tree = readtree(root)
    inspected = store.inspect_store(root, expected_head_sha256=head)
    assert inspected["summary"]["journal_state"] == "open"
    if with_attempt:
        assert inspected["summary"]["in_flight_job_ids"] == ["fit001"]
    else:
        assert inspected["summary"]["in_flight_job_ids"] == []
    with pytest.raises(store.StoreError) as excinfo:
        store.open_store(root, expected_head_sha256=head)
    assert excinfo.value.reason_code == "incomplete_session_requires_review"
    assert readtree(root) == before_tree


def test_stale_head_rejected_before_append(tmp_path, monkeypatch):
    manifest = bind_manifest(monkeypatch)
    root = tmp_path / "journal"
    owner = store.create_store(root, manifest)
    try:
        append(owner, "session_open")
        stale = owner.snapshot()
        append(owner, "progress", elapsed_ns=_PROGRESS_ELAPSED, artifact_bytes=_PROGRESS_BYTES)
        current = owner.snapshot()
        assert current["summary"]["head_sha256"] != stale["summary"]["head_sha256"]
        event = make_event(stale, "attempt_start", job_id="fit001")
        before_tree = readtree(root)
        with pytest.raises(store.StoreError) as excinfo:
            owner.append_event(
                event,
                expected_head_sha256=stale["summary"]["head_sha256"],
                resources=None,
            )
        assert excinfo.value.reason_code == "head_mismatch"
        after = owner.snapshot()["summary"]
        assert after["head_sha256"] == current["summary"]["head_sha256"]
        assert after["in_flight_job_ids"] == []
        assert readtree(root) == before_tree
    finally:
        owner.close()


def test_scientific_execution_still_denied_after_accepted_flow(tmp_path, monkeypatch, env):
    manifest = bind_manifest(monkeypatch)
    root = tmp_path / "journal"
    artifacts = tmp_path / "receipts"
    owner = store.create_store(root, manifest)
    try:
        append(owner, "session_open")
        append(owner, "progress", elapsed_ns=_PROGRESS_ELAPSED, artifact_bytes=_PROGRESS_BYTES)
        _start(owner, manifest, "fit001", tmp_path, env)
        _finish(owner, manifest, "fit001", artifacts)
        checks = [
            (admission.AdmissionError, admission.require_scientific_execution),
            (store.StoreError, store.require_scientific_execution),
            (serial.SerialResourceError, serial.require_scientific_execution),
            (receipt.ReceiptError, receipt.require_scientific_execution),
        ]
        for exc_type, call in checks:
            with pytest.raises(exc_type) as excinfo:
                call()
            assert excinfo.value.reason_code == "scientific_execution_not_authorized"
    finally:
        owner.close()
