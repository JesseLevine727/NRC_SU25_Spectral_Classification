"""P08-T052 functional tests for the invented, bounded U0 journal store.

All manifests, events and summaries are invented.  Every filesystem path is a
``tmp_path`` child.  Nothing here fits models, reads real datasets, touches
devices or authorizes scientific execution; metadata success grants nothing.
"""

from __future__ import annotations

import copy
import json
import stat

import pytest

from atlas_sers.evaluation import p08_u0_admission as admission
from atlas_sers.evaluation import p08_u0_store as storage
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256
from tests.p08_store_fixtures import (
    GIB,
    NANOS,
    append,
    bind_manifest,
    make_event,
    make_manifest,
    readtree,
    resources,
)

_HEAD_SCHEMA = "nato-sers-p08-u0-head-v1"


@pytest.fixture
def manifest(monkeypatch):
    return bind_manifest(monkeypatch)


# ---------------------------------------------------------------------------
# 1. Creation, layout and isolation
# ---------------------------------------------------------------------------


def test_create_store_layout_and_head(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        assert {entry.name for entry in root.iterdir()} == {
            ".lock",
            "events",
            "manifest.json",
            "head.json",
        }
        assert (root / "events").is_dir()
        state = store.snapshot()
        assert set(state) == {"manifest", "events", "summary"}
        assert state["events"] == []
        assert state["summary"]["execution_authorized"] is False
        assert state["summary"]["journal_state"] == "not_started"
        assert state["summary"]["head_sha256"] == admission.U0_MANIFEST_SHA256
        assert state["summary"]["session_count"] == 0
        assert state["summary"]["new_artifact_bytes"] == 0
        assert json.loads((root / "head.json").read_text()) == {
            "schema_version": _HEAD_SCHEMA,
            "manifest_sha256": admission.U0_MANIFEST_SHA256,
            "head_sha256": admission.U0_MANIFEST_SHA256,
            "event_count": 0,
        }
        assert json.loads((root / "manifest.json").read_text()) == manifest


def test_create_store_private_permissions(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest):
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert stat.S_IMODE((root / "events").stat().st_mode) == 0o700
        for name in (".lock", "manifest.json", "head.json"):
            assert stat.S_IMODE((root / name).stat().st_mode) == 0o600


def test_returned_state_and_manifest_are_isolated(manifest, tmp_path):
    root = tmp_path / "store"
    manifest_copy = copy.deepcopy(manifest)
    with storage.create_store(root, manifest) as store:
        original = copy.deepcopy(store.snapshot())
        mutated = store.snapshot()
        mutated["manifest"]["jobs"].clear()
        mutated["manifest"]["execution_authorized"] = True
        mutated["summary"]["execution_authorized"] = True
        mutated["events"].append({"bogus": True})
        manifest_copy["jobs"].clear()
        manifest_copy["execution_authorized"] = True
        fresh = store.snapshot()
        assert fresh == original
        assert fresh["summary"]["execution_authorized"] is False


# ---------------------------------------------------------------------------
# 2. Manifest rejection before any root side effect
# ---------------------------------------------------------------------------


def test_non_dict_manifest_rejected_before_root(tmp_path):
    root = tmp_path / "store"
    with pytest.raises(storage.StoreError) as info:
        storage.create_store(root, ["not", "a", "manifest"])
    assert info.value.reason_code == "invalid_manifest"
    assert not root.exists()


def test_unpatched_invented_manifest_rejected(tmp_path):
    root = tmp_path / "store"
    with pytest.raises(storage.StoreError) as info:
        storage.create_store(root, make_manifest())
    assert info.value.reason_code == "invalid_manifest"
    assert not root.exists()


def test_bad_proposal_rejected_before_root(tmp_path, monkeypatch):
    manifest = make_manifest()
    manifest["proposal_sha256"] = "b" * 64
    payload = {
        "schema_version": manifest["schema_version"],
        "execution_authorized": manifest["execution_authorized"],
        "proposal_sha256": manifest["proposal_sha256"],
        "jobs": manifest["jobs"],
    }
    manifest["manifest_sha256"] = canonical_sha256(payload)
    monkeypatch.setattr(admission, "U0_MANIFEST_SHA256", manifest["manifest_sha256"])
    root = tmp_path / "store"
    with pytest.raises(storage.StoreError) as info:
        storage.create_store(root, manifest)
    assert info.value.reason_code == "invalid_manifest"
    assert not root.exists()


def test_existing_root_preserves_sentinel(manifest, tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    sentinel = root / "sentinel.bin"
    sentinel.write_bytes(b"keep-me")
    before = readtree(root)
    with pytest.raises(storage.StoreError) as info:
        storage.create_store(root, manifest)
    assert info.value.reason_code == "store_exists"
    assert readtree(root) == before
    assert sentinel.read_bytes() == b"keep-me"


# ---------------------------------------------------------------------------
# 3. Valid round trip and reopen
# ---------------------------------------------------------------------------


def test_valid_roundtrip_chain(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=10, artifact_bytes=GIB)
        append(
            store,
            "attempt_start",
            job_id="fit001",
            resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
        )
        append(
            store,
            "attempt_finish",
            job_id="fit001",
            status="succeeded",
            elapsed_ns=12,
            artifact_bytes=GIB,
        )
        append(
            store,
            "attempt_start",
            job_id="pred001",
            resources=resources(),
        )
        append(
            store,
            "attempt_finish",
            job_id="pred001",
            status="succeeded",
            elapsed_ns=15,
            artifact_bytes=GIB,
        )
        summary = append(store, "session_close", elapsed_ns=20, artifact_bytes=GIB)
        assert summary["model_fit_attempts"] == 1
        assert summary["source_prediction_attempts"] == 1
        assert summary["new_artifact_bytes"] == GIB
        assert summary["active_wall_ns"] == 20
        assert summary["journal_state"] == "closed"
        head = summary["head_sha256"]

    with storage.open_store(root, expected_head_sha256=head) as store:
        append(store, "session_open")
        summary = append(store, "progress", elapsed_ns=5)
        assert summary["new_artifact_bytes"] == GIB
        assert summary["active_wall_ns"] == 25
        assert summary["model_fit_attempts"] == 1
        assert summary["source_prediction_attempts"] == 1
        assert summary["session_count"] == 2


# ---------------------------------------------------------------------------
# 4. Exclusive lease
# ---------------------------------------------------------------------------


def test_lease_unavailable_while_owner_held(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        head = store.snapshot()["summary"]["head_sha256"]
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            storage.open_store(root, expected_head_sha256=head)
        assert info.value.reason_code == "lease_unavailable"
        with pytest.raises(storage.StoreError) as info:
            storage.inspect_store(root, expected_head_sha256=head)
        assert info.value.reason_code == "lease_unavailable"
        assert readtree(root) == before
        assert store.snapshot()["summary"]["head_sha256"] == head
    with storage.open_store(root, expected_head_sha256=head) as reopened:
        assert reopened.snapshot()["summary"]["head_sha256"] == head


# ---------------------------------------------------------------------------
# 5. Incomplete session requires review
# ---------------------------------------------------------------------------


def test_release_without_session_close_requires_review(manifest, tmp_path):
    root = tmp_path / "store"
    store = storage.create_store(root, manifest)
    try:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=1, artifact_bytes=GIB)
        append(
            store,
            "attempt_start",
            job_id="fit001",
            resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
        )
        head = store.snapshot()["summary"]["head_sha256"]
    finally:
        store.close()

    state = storage.inspect_store(root, expected_head_sha256=head)
    assert state["summary"]["journal_state"] == "open"
    assert state["summary"]["in_flight_job_ids"] == ["fit001"]
    assert state["summary"]["model_fit_attempts"] == 1
    assert state["summary"]["attempts"]["fit001"]["status"] == "running"

    with pytest.raises(storage.StoreError) as info:
        storage.open_store(root, expected_head_sha256=head)
    assert info.value.reason_code == "incomplete_session_requires_review"

    again = storage.inspect_store(root, expected_head_sha256=head)
    assert again["summary"]["journal_state"] == "open"
    assert again["summary"]["attempts"]["fit001"]["status"] == "running"


# ---------------------------------------------------------------------------
# 6. Stale heads and closed stores
# ---------------------------------------------------------------------------


def test_stale_head_rejected_and_owner_usable(manifest, tmp_path):
    root = tmp_path / "store"
    store = storage.create_store(root, manifest)
    try:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=1, artifact_bytes=GIB)
        state = store.snapshot()
        event = make_event(state, "progress", elapsed_ns=2)
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            store.append_event(event, expected_head_sha256="0" * 64)
        assert info.value.reason_code == "head_mismatch"
        assert readtree(root) == before
        summary = append(store, "progress", elapsed_ns=2)
        assert summary["active_session_elapsed_ns"] == 2
    finally:
        store.close()


def test_closed_store_is_unusable(manifest, tmp_path):
    root = tmp_path / "store"
    store = storage.create_store(root, manifest)
    try:
        append(store, "session_open")
        state = store.snapshot()
        head = state["summary"]["head_sha256"]
        event = make_event(state, "session_close")
        before = readtree(root)
        store.close()
        with pytest.raises(storage.StoreError) as info:
            store.snapshot()
        assert info.value.reason_code == "closed_store"
        with pytest.raises(storage.StoreError) as info:
            store.append_event(event, expected_head_sha256=head)
        assert info.value.reason_code == "closed_store"
        with pytest.raises(storage.StoreError) as info:
            store.__enter__()
        assert info.value.reason_code == "closed_store"
        assert readtree(root) == before
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 7. Prospective candidate guards and pre-write atomicity
# ---------------------------------------------------------------------------


def test_third_cpu_worker_blocked(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=1, artifact_bytes=GIB)
        append(
            store,
            "attempt_start",
            job_id="fit001",
            resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
        )
        append(
            store,
            "attempt_start",
            job_id="fit002",
            resources=resources(cpu=1, filesystem_free_bytes=37 * GIB),
        )
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            append(
                store,
                "attempt_start",
                job_id="fit003",
                resources=resources(cpu=2, filesystem_free_bytes=37 * GIB),
            )
        assert info.value.reason_code == "candidate_blocked"
        assert readtree(root) == before


def test_second_gpu_worker_blocked(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=1, artifact_bytes=GIB)
        append(
            store,
            "attempt_start",
            job_id="fit043",
            resources=resources(gpu=0, filesystem_free_bytes=37 * GIB),
        )
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            append(
                store,
                "attempt_start",
                job_id="fit044",
                resources=resources(gpu=1, filesystem_free_bytes=37 * GIB),
            )
        assert info.value.reason_code == "candidate_blocked"
        assert readtree(root) == before


def test_bad_declared_workers_blocked(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=1, artifact_bytes=GIB)
        append(
            store,
            "attempt_start",
            job_id="fit001",
            resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
        )
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            append(
                store,
                "attempt_start",
                job_id="fit002",
                resources=resources(cpu=1, model_threads=2, filesystem_free_bytes=37 * GIB),
            )
        assert info.value.reason_code == "candidate_blocked"
        assert readtree(root) == before


def test_attempt_start_without_resources_invalid(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=1, artifact_bytes=GIB)
        state = store.snapshot()
        event = make_event(state, "attempt_start", job_id="fit001")
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            store.append_event(event, expected_head_sha256=state["summary"]["head_sha256"])
        assert info.value.reason_code == "invalid_resources"
        assert readtree(root) == before


def test_prediction_before_fit_rejected(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=1, artifact_bytes=GIB)
        state = store.snapshot()
        event = make_event(state, "attempt_start", job_id="pred001")
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            store.append_event(
                event,
                expected_head_sha256=state["summary"]["head_sha256"],
                resources=resources(filesystem_free_bytes=37 * GIB),
            )
        assert info.value.reason_code == "invalid_journal"
        assert readtree(root) == before


def test_repeated_attempt_rejected(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=1, artifact_bytes=GIB)
        append(
            store,
            "attempt_start",
            job_id="fit001",
            resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
        )
        state = store.snapshot()
        event = make_event(state, "attempt_start", job_id="fit001")
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            store.append_event(
                event,
                expected_head_sha256=state["summary"]["head_sha256"],
                resources=resources(cpu=1, filesystem_free_bytes=37 * GIB),
            )
        assert info.value.reason_code == "invalid_journal"
        assert readtree(root) == before


# ---------------------------------------------------------------------------
# 8. Fresh-progress requirement
# ---------------------------------------------------------------------------


def test_stale_progress_required(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=10, artifact_bytes=GIB)
        state = store.snapshot()
        stale_elapsed = make_event(state, "attempt_start", job_id="fit001", elapsed_ns=11)
        stale_artifact = make_event(state, "attempt_start", job_id="fit001", artifact_bytes=2 * GIB)
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            store.append_event(
                stale_elapsed,
                expected_head_sha256=state["summary"]["head_sha256"],
                resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
            )
        assert info.value.reason_code == "fresh_progress_required"
        with pytest.raises(storage.StoreError) as info:
            store.append_event(
                stale_artifact,
                expected_head_sha256=state["summary"]["head_sha256"],
                resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
            )
        assert info.value.reason_code == "fresh_progress_required"
        assert readtree(root) == before

        summary = append(store, "progress", elapsed_ns=11, artifact_bytes=2 * GIB)
        assert summary["active_session_elapsed_ns"] == 11
        assert summary["new_artifact_bytes"] == 2 * GIB
        summary = append(
            store,
            "attempt_start",
            job_id="fit001",
            resources=resources(cpu=0, filesystem_free_bytes=36 * GIB),
        )
        assert summary["attempts"]["fit001"]["status"] == "running"


# ---------------------------------------------------------------------------
# 9. Terminal statuses are retained and block new starts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terminal", ["failed", "interrupted"])
def test_terminal_first_attempt_blocks_new_start(manifest, tmp_path, terminal):
    root = tmp_path / "store"
    store = storage.create_store(root, manifest)
    try:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=1, artifact_bytes=GIB)
        append(
            store,
            "attempt_start",
            job_id="fit001",
            resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
        )
        summary = append(
            store,
            "attempt_finish",
            job_id="fit001",
            status=terminal,
            elapsed_ns=2,
            artifact_bytes=GIB,
        )
        assert summary["attempts"]["fit001"]["status"] == terminal
        bucket = (
            summary["failed_job_ids"] if terminal == "failed" else summary["interrupted_job_ids"]
        )
        assert bucket == ["fit001"]
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            append(
                store,
                "attempt_start",
                job_id="fit002",
                resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
            )
        assert info.value.reason_code == "candidate_blocked"
        assert readtree(root) == before
        summary = append(store, "session_close", elapsed_ns=3, artifact_bytes=GIB)
        head = summary["head_sha256"]
    finally:
        store.close()

    with storage.open_store(root, expected_head_sha256=head) as store:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=1, artifact_bytes=GIB)
        state = store.snapshot()
        assert state["summary"]["attempts"]["fit001"]["status"] == terminal
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            append(
                store,
                "attempt_start",
                job_id="fit002",
                resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
            )
        assert info.value.reason_code == "candidate_blocked"
        assert readtree(root) == before
        with pytest.raises(storage.StoreError) as info:
            append(
                store,
                "attempt_start",
                job_id="fit001",
                resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
            )
        assert info.value.reason_code == "invalid_journal"


# ---------------------------------------------------------------------------
# 10. Wall cap blocks new work but never suppresses terminal evidence
# ---------------------------------------------------------------------------


def test_wall_cap_blocks_new_start_but_records_terminal(manifest, tmp_path):
    cap = 5400 * NANOS
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        append(store, "session_open")
        append(store, "progress", elapsed_ns=1, artifact_bytes=GIB)
        append(
            store,
            "attempt_start",
            job_id="fit001",
            resources=resources(cpu=0, filesystem_free_bytes=37 * GIB),
        )
        summary = append(store, "progress", elapsed_ns=cap, artifact_bytes=GIB)
        assert summary["active_session_elapsed_ns"] == cap
        assert summary["active_wall_ns"] == cap
        before = readtree(root)
        with pytest.raises(storage.StoreError) as info:
            append(
                store,
                "attempt_start",
                job_id="fit002",
                resources=resources(cpu=1, filesystem_free_bytes=37 * GIB),
            )
        assert info.value.reason_code == "candidate_blocked"
        assert readtree(root) == before
        summary = append(
            store,
            "attempt_finish",
            job_id="fit001",
            status="failed",
            elapsed_ns=cap,
            artifact_bytes=GIB,
        )
        assert summary["attempts"]["fit001"]["status"] == "failed"
        summary = append(store, "session_close", elapsed_ns=cap, artifact_bytes=GIB)
        assert summary["journal_state"] == "closed"
        assert summary["closed_session_wall_ns"] == cap


# ---------------------------------------------------------------------------
# 11. Errors, static reasons and scientific denial
# ---------------------------------------------------------------------------


def test_scientific_execution_always_denied(manifest, tmp_path):
    root = tmp_path / "store"
    with storage.create_store(root, manifest) as store:
        state = store.snapshot()
        assert state["summary"]["execution_authorized"] is False
        for args, kwargs in [
            ((), {}),
            (("run",), {"authorized": True}),
            ((store, state), {"execution_authorized": True, "force": True}),
        ]:
            with pytest.raises(storage.StoreError) as info:
                storage.require_scientific_execution(*args, **kwargs)
            assert info.value.reason_code == "scientific_execution_not_authorized"


def test_store_error_reason_codes_are_static():
    assert storage.StoreError("closed_store").reason_code == "closed_store"
    for bad in (None, 123, 3.5, b"x", [], {}, object(), "not_a_reason", True):
        error = storage.StoreError(bad)
        assert error.reason_code == "invalid_store_input"
        assert str(error) == "invalid_store_input"


class _Reason(str):
    pass


def test_str_subclass_reason_code_rejected():
    error = storage.StoreError(_Reason("closed_store"))
    assert error.reason_code == "invalid_store_input"


def test_internal_valueerror_suppressed(monkeypatch, manifest, tmp_path):
    sentinel = "private-sentinel-abc"

    def boom(*args, **kwargs):
        raise ValueError(sentinel)

    monkeypatch.setattr(storage, "_validate_new_manifest", boom)
    with pytest.raises(storage.StoreError) as info:
        storage.create_store(tmp_path / "store", manifest)
    assert info.value.reason_code == "invalid_store_input"
    assert sentinel not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True


def test_keyboardinterrupt_propagates(monkeypatch, manifest, tmp_path):
    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(storage, "_validate_new_manifest", boom)
    with pytest.raises(KeyboardInterrupt):
        storage.create_store(tmp_path / "store", manifest)


# ---------------------------------------------------------------------------
# 12. Path rejection and containment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_root",
    ["", "relative/store", "/", "/tmp/../etc", "bad\x00name", b"/tmp/x"],
)
def test_invalid_roots_rejected(manifest, tmp_path, bad_root):
    with pytest.raises(storage.StoreError) as info:
        storage.create_store(bad_root, manifest)
    assert info.value.reason_code == "invalid_path"


def test_dotdot_root_creates_nothing(manifest, tmp_path):
    bad = str(tmp_path / "victim" / ".." / "store")
    with pytest.raises(storage.StoreError) as info:
        storage.create_store(bad, manifest)
    assert info.value.reason_code == "invalid_path"
    assert not (tmp_path / "victim").exists()


def test_nested_tmp_path_root_ok(manifest, tmp_path):
    parent = tmp_path / "nest"
    parent.mkdir()
    root = parent / "store"
    with storage.create_store(root, manifest) as store:
        assert store.snapshot()["summary"]["head_sha256"] == admission.U0_MANIFEST_SHA256
