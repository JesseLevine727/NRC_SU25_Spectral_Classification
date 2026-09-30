"""Fault-injection tests for the invented P08-T052 U0 store writer.

Everything here is invented metadata under ``tmp_path``.  These tests never
fit models, read datasets, touch devices or authorize science; they only
mutate synthetic fixture bytes with ``read_bytes``/``write_bytes``.
"""

from __future__ import annotations

import errno
import os

import pytest

from atlas_sers.evaluation import p08_u0_store as storemod
from tests.p08_store_fixtures import bind_manifest, make_event, readtree

_HEAD_JSON = "head.json"
_PENDING_JSON = "head.pending"
_EVENT_ONE = os.path.join("events", "00000001.json")
_EXTERNAL_BYTES = b"EXTERNAL-TARGET"


def _create_owner(tmp_path, monkeypatch):
    manifest = bind_manifest(monkeypatch)
    root = tmp_path / "store"
    owner = storemod.create_store(root, manifest)
    return root, manifest, owner


def _seal_session_open(owner):
    state = owner.snapshot()
    event = make_event(state, "session_open")
    owner.append_event(event, expected_head_sha256=state["summary"]["head_sha256"])
    return state, event


def _prepare_append(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    state = owner.snapshot()
    event = make_event(state, "session_open")
    return {
        "root": root,
        "manifest": manifest,
        "owner": owner,
        "state": state,
        "event": event,
        "expected_head": state["summary"]["head_sha256"],
        "encoded_event": storemod._encode_json(event),
        "encoded_head": storemod._encode_json(storemod._make_head(event["event_sha256"], 1)),
        "old_head": (root / _HEAD_JSON).read_bytes(),
    }


def _assert_poisoned(owner, root, event, expected_head):
    with pytest.raises(storemod.StoreError) as exc:
        owner.snapshot()
    assert exc.value.reason_code == "poisoned_store"
    frozen = readtree(root)
    with pytest.raises(storemod.StoreError) as exc2:
        owner.append_event(event, expected_head_sha256=expected_head)
    assert exc2.value.reason_code == "poisoned_store"
    assert readtree(root) == frozen


# ---------------------------------------------------------------------------
# Corrupt / oversize head.json
# ---------------------------------------------------------------------------


_CORRUPT_HEAD_JSON = [
    pytest.param(b'{"a":1,"a":2}\n', "invalid_json", id="duplicate-keys"),
    pytest.param(b"NaN\n", "invalid_json", id="nan-constant"),
    pytest.param(b"1e999\n", "invalid_json", id="infinite-float"),
    pytest.param(b"\xff\xfe\xfd\n", "invalid_json", id="invalid-utf8"),
]


@pytest.mark.parametrize("payload,reason", _CORRUPT_HEAD_JSON)
def test_inspect_rejects_corrupt_head_json(tmp_path, monkeypatch, payload, reason):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    owner.close()
    head = root / _HEAD_JSON
    head.write_bytes(payload)
    retained = head.read_bytes()
    with pytest.raises(storemod.StoreError) as exc:
        storemod.inspect_store(root, expected_head_sha256=manifest["manifest_sha256"])
    assert exc.value.reason_code == reason
    assert head.read_bytes() == retained


def test_inspect_rejects_oversize_head(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    owner.close()
    head = root / _HEAD_JSON
    head.write_bytes(b"{" + b" " * (storemod.MAX_HEAD_BYTES + 64) + b"}\n")
    with pytest.raises(storemod.StoreError) as exc:
        storemod.inspect_store(root, expected_head_sha256=manifest["manifest_sha256"])
    assert exc.value.reason_code == "size_limit"


# ---------------------------------------------------------------------------
# Pending/orphan/missing/truncated/stale layout evidence
# ---------------------------------------------------------------------------


def test_pending_head_refused_and_bytes_retained(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    owner.close()
    pending = root / _PENDING_JSON
    pending.write_bytes(b'{"schema_version":"nato-sers-p08-u0-head-v1"}\n')
    retained = pending.read_bytes()
    with pytest.raises(storemod.StoreError) as exc:
        storemod.inspect_store(root, expected_head_sha256=manifest["manifest_sha256"])
    assert exc.value.reason_code == "incomplete_write"
    assert pending.read_bytes() == retained


def test_orphan_event_refused_and_bytes_retained(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    owner.close()
    orphan = root / _EVENT_ONE
    orphan.write_bytes(b'{"seq":1}\n')
    retained = orphan.read_bytes()
    with pytest.raises(storemod.StoreError) as exc:
        storemod.inspect_store(root, expected_head_sha256=manifest["manifest_sha256"])
    assert exc.value.reason_code == "incomplete_write"
    assert orphan.read_bytes() == retained


def test_missing_event_refused(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    state, event = _seal_session_open(owner)
    owner.close()
    (root / _EVENT_ONE).unlink()
    with pytest.raises(storemod.StoreError) as exc:
        storemod.inspect_store(root, expected_head_sha256=event["event_sha256"])
    assert exc.value.reason_code == "incomplete_write"


def test_truncated_event_refused_and_bytes_retained(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    state, event = _seal_session_open(owner)
    owner.close()
    path = root / _EVENT_ONE
    path.write_bytes(path.read_bytes()[:5])
    retained = path.read_bytes()
    with pytest.raises(storemod.StoreError) as exc:
        storemod.inspect_store(root, expected_head_sha256=event["event_sha256"])
    assert exc.value.reason_code == "invalid_json"
    assert path.read_bytes() == retained


def test_stale_external_head_refused(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    state, event = _seal_session_open(owner)
    owner.close()
    with pytest.raises(storemod.StoreError) as exc:
        storemod.inspect_store(root, expected_head_sha256=manifest["manifest_sha256"])
    assert exc.value.reason_code == "head_mismatch"


# ---------------------------------------------------------------------------
# Symlink / hardlink metadata rejection
# ---------------------------------------------------------------------------


def test_manifest_symlink_rejected_without_writing_target(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    owner.close()
    target = tmp_path / "target.bin"
    target.write_bytes(_EXTERNAL_BYTES)
    path = root / "manifest.json"
    path.unlink()
    os.symlink(target, path)
    with pytest.raises(storemod.StoreError) as exc:
        storemod.inspect_store(root, expected_head_sha256=manifest["manifest_sha256"])
    assert exc.value.reason_code == "symlink_or_nonregular"
    assert target.read_bytes() == _EXTERNAL_BYTES


def test_manifest_hardlink_rejected_without_writing_target(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    owner.close()
    target = tmp_path / "target.bin"
    target.write_bytes(_EXTERNAL_BYTES)
    path = root / "manifest.json"
    path.unlink()
    os.link(target, path)
    with pytest.raises(storemod.StoreError) as exc:
        storemod.inspect_store(root, expected_head_sha256=manifest["manifest_sha256"])
    assert exc.value.reason_code == "symlink_or_nonregular"
    assert target.read_bytes() == _EXTERNAL_BYTES


def test_event_symlink_rejected_without_writing_target(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    state, event = _seal_session_open(owner)
    owner.close()
    target = tmp_path / "target.bin"
    target.write_bytes(_EXTERNAL_BYTES)
    path = root / _EVENT_ONE
    path.unlink()
    os.symlink(target, path)
    with pytest.raises(storemod.StoreError) as exc:
        storemod.inspect_store(root, expected_head_sha256=event["event_sha256"])
    assert exc.value.reason_code == "symlink_or_nonregular"
    assert target.read_bytes() == _EXTERNAL_BYTES


def test_event_hardlink_rejected_without_writing_target(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    state, event = _seal_session_open(owner)
    owner.close()
    target = tmp_path / "target.bin"
    target.write_bytes(_EXTERNAL_BYTES)
    path = root / _EVENT_ONE
    path.unlink()
    os.link(target, path)
    with pytest.raises(storemod.StoreError) as exc:
        storemod.inspect_store(root, expected_head_sha256=event["event_sha256"])
    assert exc.value.reason_code == "symlink_or_nonregular"
    assert target.read_bytes() == _EXTERNAL_BYTES


# ---------------------------------------------------------------------------
# Permanent poisoning
# ---------------------------------------------------------------------------


def test_owner_poisoned_by_corrupt_head_stays_poisoned(tmp_path, monkeypatch):
    root, manifest, owner = _create_owner(tmp_path, monkeypatch)
    head = root / _HEAD_JSON
    original = head.read_bytes()
    assert owner.snapshot()["summary"]["head_sha256"] == manifest["manifest_sha256"]
    head.write_bytes(b'{"a":1,"a":2}\n')
    with pytest.raises(storemod.StoreError) as exc:
        owner.snapshot()
    assert exc.value.reason_code == "invalid_json"
    head.write_bytes(original)
    with pytest.raises(storemod.StoreError) as exc2:
        owner.snapshot()
    assert exc2.value.reason_code == "poisoned_store"
    owner.close()


# ---------------------------------------------------------------------------
# Append durability fault injection
# ---------------------------------------------------------------------------


def test_append_fault_partial_event_write(tmp_path, monkeypatch):
    data = _prepare_append(tmp_path, monkeypatch)
    root, owner, event = data["root"], data["owner"], data["event"]
    written = {}

    def partial_write_exclusive(dir_fd, name, payload):
        fd = os.open(
            name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=dir_fd,
        )
        try:
            chunk = payload[: max(1, len(payload) // 2)]
            os.write(fd, chunk)
            written["bytes"] = chunk
        finally:
            os.close(fd)
        raise OSError(errno.EIO, "injected partial event write")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(storemod, "_write_exclusive", partial_write_exclusive)
            with pytest.raises(storemod.StoreError) as exc:
                owner.append_event(event, expected_head_sha256=data["expected_head"])
        assert exc.value.reason_code == "storage_io_error"
        assert (root / _EVENT_ONE).read_bytes() == written["bytes"]
        assert not (root / _PENDING_JSON).exists()
        assert (root / _HEAD_JSON).read_bytes() == data["old_head"]
        _assert_poisoned(owner, root, event, data["expected_head"])
    finally:
        owner.close()


def test_append_fault_event_file_fsync(tmp_path, monkeypatch):
    data = _prepare_append(tmp_path, monkeypatch)
    root, owner, event = data["root"], data["owner"], data["event"]
    real_fsync = os.fsync
    calls = {"n": 0}

    def failing_fsync(fd):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.EIO, "injected event-file fsync")
        return real_fsync(fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", failing_fsync)
            with pytest.raises(storemod.StoreError) as exc:
                owner.append_event(event, expected_head_sha256=data["expected_head"])
        assert exc.value.reason_code == "storage_io_error"
        assert (root / _EVENT_ONE).read_bytes() == data["encoded_event"]
        assert not (root / _PENDING_JSON).exists()
        assert (root / _HEAD_JSON).read_bytes() == data["old_head"]
        _assert_poisoned(owner, root, event, data["expected_head"])
    finally:
        owner.close()


def test_append_fault_events_directory_fsync(tmp_path, monkeypatch):
    data = _prepare_append(tmp_path, monkeypatch)
    root, owner, event = data["root"], data["owner"], data["event"]
    real_fsync = os.fsync
    calls = {"n": 0}

    def failing_fsync(fd):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(errno.EIO, "injected events-directory fsync")
        return real_fsync(fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", failing_fsync)
            with pytest.raises(storemod.StoreError) as exc:
                owner.append_event(event, expected_head_sha256=data["expected_head"])
        assert exc.value.reason_code == "storage_io_error"
        assert (root / _EVENT_ONE).read_bytes() == data["encoded_event"]
        assert not (root / _PENDING_JSON).exists()
        assert (root / _HEAD_JSON).read_bytes() == data["old_head"]
        _assert_poisoned(owner, root, event, data["expected_head"])
    finally:
        owner.close()


def test_append_fault_pending_head_fsync(tmp_path, monkeypatch):
    data = _prepare_append(tmp_path, monkeypatch)
    root, owner, event = data["root"], data["owner"], data["event"]
    real_fsync = os.fsync
    calls = {"n": 0}

    def failing_fsync(fd):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError(errno.EIO, "injected pending-head fsync")
        return real_fsync(fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", failing_fsync)
            with pytest.raises(storemod.StoreError) as exc:
                owner.append_event(event, expected_head_sha256=data["expected_head"])
        assert exc.value.reason_code == "storage_io_error"
        assert (root / _EVENT_ONE).read_bytes() == data["encoded_event"]
        assert (root / _PENDING_JSON).read_bytes() == data["encoded_head"]
        assert (root / _HEAD_JSON).read_bytes() == data["old_head"]
        _assert_poisoned(owner, root, event, data["expected_head"])
    finally:
        owner.close()


def test_append_fault_pending_head_write(tmp_path, monkeypatch):
    data = _prepare_append(tmp_path, monkeypatch)
    root, owner, event = data["root"], data["owner"], data["event"]
    real_write_exclusive = storemod._write_exclusive
    calls = {"n": 0}
    written = {}

    def failing_write_exclusive(dir_fd, name, payload):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write_exclusive(dir_fd, name, payload)
        fd = os.open(
            name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=dir_fd,
        )
        try:
            chunk = payload[: max(1, len(payload) // 2)]
            os.write(fd, chunk)
            written["bytes"] = chunk
        finally:
            os.close(fd)
        raise OSError(errno.EIO, "injected pending-head write")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(storemod, "_write_exclusive", failing_write_exclusive)
            with pytest.raises(storemod.StoreError) as exc:
                owner.append_event(event, expected_head_sha256=data["expected_head"])
        assert exc.value.reason_code == "storage_io_error"
        assert (root / _EVENT_ONE).read_bytes() == data["encoded_event"]
        assert (root / _PENDING_JSON).read_bytes() == written["bytes"]
        assert (root / _HEAD_JSON).read_bytes() == data["old_head"]
        _assert_poisoned(owner, root, event, data["expected_head"])
    finally:
        owner.close()


def test_append_fault_replace(tmp_path, monkeypatch):
    data = _prepare_append(tmp_path, monkeypatch)
    root, owner, event = data["root"], data["owner"], data["event"]

    def failing_replace(*args, **kwargs):
        raise OSError(errno.EIO, "injected replace")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", failing_replace)
            with pytest.raises(storemod.StoreError) as exc:
                owner.append_event(event, expected_head_sha256=data["expected_head"])
        assert exc.value.reason_code == "storage_io_error"
        assert (root / _EVENT_ONE).read_bytes() == data["encoded_event"]
        assert (root / _PENDING_JSON).read_bytes() == data["encoded_head"]
        assert (root / _HEAD_JSON).read_bytes() == data["old_head"]
        _assert_poisoned(owner, root, event, data["expected_head"])
    finally:
        owner.close()


def test_append_fault_final_root_fsync(tmp_path, monkeypatch):
    data = _prepare_append(tmp_path, monkeypatch)
    root, owner, event = data["root"], data["owner"], data["event"]
    real_fsync = os.fsync
    calls = {"n": 0}

    def failing_fsync(fd):
        calls["n"] += 1
        if calls["n"] == 4:
            raise OSError(errno.EIO, "injected root-directory fsync")
        return real_fsync(fd)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", failing_fsync)
            with pytest.raises(storemod.StoreError) as exc:
                owner.append_event(event, expected_head_sha256=data["expected_head"])
        assert exc.value.reason_code == "storage_io_error"
        assert (root / _EVENT_ONE).read_bytes() == data["encoded_event"]
        assert not (root / _PENDING_JSON).exists()
        # A root-fsync error can still leave the already-replaced head on disk.
        assert (root / _HEAD_JSON).read_bytes() == data["encoded_head"]
        _assert_poisoned(owner, root, event, data["expected_head"])
    finally:
        owner.close()


def test_append_keyboardinterrupt_poisoned(tmp_path, monkeypatch):
    data = _prepare_append(tmp_path, monkeypatch)
    root, owner, event = data["root"], data["owner"], data["event"]
    real_write_exclusive = storemod._write_exclusive
    calls = {"n": 0}

    def interrupting_write_exclusive(dir_fd, name, payload):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return real_write_exclusive(dir_fd, name, payload)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(storemod, "_write_exclusive", interrupting_write_exclusive)
            with pytest.raises(KeyboardInterrupt):
                owner.append_event(event, expected_head_sha256=data["expected_head"])
        assert calls["n"] == 1
        assert not (root / _EVENT_ONE).exists()
        assert not (root / _PENDING_JSON).exists()
        assert (root / _HEAD_JSON).read_bytes() == data["old_head"]
        _assert_poisoned(owner, root, event, data["expected_head"])
        assert calls["n"] == 1
    finally:
        owner.close()


# ---------------------------------------------------------------------------
# Durability ordering trace and input immutability
# ---------------------------------------------------------------------------


def test_append_durability_order_trace(tmp_path, monkeypatch):
    data = _prepare_append(tmp_path, monkeypatch)
    root, owner, event = data["root"], data["owner"], data["event"]
    real_fsync = os.fsync
    real_replace = os.replace
    real_write_exclusive = storemod._write_exclusive
    trace = []
    role = {"current": None}

    def traced_write_exclusive(dir_fd, name, payload):
        role["current"] = "pending_file" if name == _PENDING_JSON else "event_file"
        try:
            return real_write_exclusive(dir_fd, name, payload)
        finally:
            role["current"] = None

    def traced_fsync(fd):
        if role["current"] is not None:
            trace.append(role["current"])
        elif fd == owner._events_fd:
            trace.append("events_directory")
        elif fd == owner._root_fd:
            trace.append("root_directory")
        else:
            raise AssertionError(f"unexpected fsync descriptor: {fd!r}")
        return real_fsync(fd)

    def traced_replace(*args, **kwargs):
        trace.append("replace")
        return real_replace(*args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(storemod, "_write_exclusive", traced_write_exclusive)
            patch.setattr(os, "fsync", traced_fsync)
            patch.setattr(os, "replace", traced_replace)
            summary = owner.append_event(event, expected_head_sha256=data["expected_head"])
        assert trace == [
            "event_file",
            "events_directory",
            "pending_file",
            "replace",
            "root_directory",
        ]
        assert summary["head_sha256"] == event["event_sha256"]
        assert (root / _EVENT_ONE).read_bytes() == data["encoded_event"]
        assert (root / _HEAD_JSON).read_bytes() == data["encoded_head"]
        assert not (root / _PENDING_JSON).exists()
        assert owner.snapshot()["summary"]["head_sha256"] == event["event_sha256"]
    finally:
        owner.close()


def test_mutating_input_manifest_leaves_store_unchanged(tmp_path, monkeypatch):
    manifest = bind_manifest(monkeypatch)
    original_sha = manifest["manifest_sha256"]
    root = tmp_path / "store"
    owner = storemod.create_store(root, manifest)
    try:
        before = readtree(root)
        manifest["execution_authorized"] = True
        manifest["manifest_sha256"] = "f" * 64
        manifest["jobs"].append(
            {
                "job_id": "extra",
                "stage": "source_fit",
                "worker": "cpu",
                "dependencies": [],
            }
        )
        assert readtree(root) == before
        snapshot = owner.snapshot()
        assert snapshot["manifest"]["manifest_sha256"] == original_sha
        assert snapshot["manifest"]["execution_authorized"] is False
    finally:
        owner.close()
