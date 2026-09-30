"""Process-level lease/journal tests for the invented P08 U0 store.

These tests only use invented metadata.  They never fit a model, read a
dataset, touch a device or authorize scientific execution.
"""

from __future__ import annotations

import json
import os
import pathlib
import select
import stat
import subprocess
import sys

import pytest

from atlas_sers.evaluation import p08_u0_store as storage
from tests.p08_store_fixtures import bind_manifest, readtree

ROOT = pathlib.Path(__file__).resolve().parents[1]

_CHILD_SCRIPT = r"""
import json
import sys

from atlas_sers.evaluation import p08_u0_store as storage, p08_u0_admission as admission
from tests.p08_store_fixtures import make_manifest, append, resources

mode = sys.argv[1]
root = sys.argv[2]

manifest = make_manifest()
admission.U0_MANIFEST_SHA256 = manifest["manifest_sha256"]

if mode == "fifo":
    try:
        storage.inspect_store(root, expected_head_sha256=manifest["manifest_sha256"])
    except storage.StoreError as exc:
        sys.stdout.write(exc.reason_code + "\n")
    else:
        sys.stdout.write("no_error\n")
    sys.stdout.flush()
    sys.stdin.readline()
    raise SystemExit(0)

owner = storage.create_store(root, manifest)
if mode == "open":
    append(owner, "session_open")
    append(owner, "attempt_start", job_id="fit001", resources=resources())

summary = owner.snapshot()["summary"]
sys.stdout.write(json.dumps(summary) + "\n")
sys.stdout.flush()
sys.stdin.readline()
"""


def _child_env():
    env = os.environ.copy()
    parts = [str(ROOT), str(ROOT / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _spawn(root, mode):
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD_SCRIPT, mode, str(root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_child_env(),
        close_fds=True,
    )


def _wait_ready(child, timeout=10):
    ready, _, _ = select.select([child.stdout], [], [], timeout)
    assert ready, "child store was not ready in time"
    line = child.stdout.readline()
    if not line:
        child.kill()
        _, err = child.communicate(timeout=10)
        raise AssertionError("child wrote no readiness line; stderr=" + repr(err))
    return json.loads(line)


def _reap(child):
    if child.poll() is None:
        child.kill()
    child.communicate(timeout=10)


def test_live_child_blocks_parent_lease(tmp_path, monkeypatch):
    bind_manifest(monkeypatch)
    root = tmp_path / "store"
    child = _spawn(root, "open")
    try:
        head = _wait_ready(child)["head_sha256"]
        with pytest.raises(storage.StoreError) as opened:
            storage.open_store(root, expected_head_sha256=head)
        assert opened.value.reason_code == "lease_unavailable"
        with pytest.raises(storage.StoreError) as inspected:
            storage.inspect_store(root, expected_head_sha256=head)
        assert inspected.value.reason_code == "lease_unavailable"
    finally:
        _reap(child)


def test_killed_empty_store_reopens_without_events(tmp_path, monkeypatch):
    bind_manifest(monkeypatch)
    root = tmp_path / "store"
    child = _spawn(root, "empty")
    try:
        head = _wait_ready(child)["head_sha256"]
    finally:
        _reap(child)

    owner = storage.open_store(root, expected_head_sha256=head)
    try:
        state = owner.snapshot()
        assert state["events"] == []
        assert state["summary"]["journal_state"] == "not_started"
    finally:
        owner.close()


def test_killed_open_fit_store_needs_review(tmp_path, monkeypatch):
    bind_manifest(monkeypatch)
    root = tmp_path / "store"
    child = _spawn(root, "open")
    try:
        head = _wait_ready(child)["head_sha256"]
    finally:
        _reap(child)

    state = storage.inspect_store(root, expected_head_sha256=head)
    summary = state["summary"]
    assert summary["model_fit_attempts"] == 1
    assert summary["in_flight_job_ids"] == ["fit001"]
    assert summary["attempts"]["fit001"]["status"] == "running"

    with pytest.raises(storage.StoreError) as refused:
        storage.open_store(root, expected_head_sha256=head)
    assert refused.value.reason_code == "incomplete_session_requires_review"


@pytest.mark.parametrize("entry", ["manifest.json", "head.json", ".lock"])
def test_fifo_entry_is_rejected_as_nonregular(tmp_path, monkeypatch, entry):
    manifest = bind_manifest(monkeypatch)
    root = tmp_path / "store"
    owner = storage.create_store(root, manifest)
    owner.close()

    target = root / entry
    if target.exists() or target.is_symlink():
        target.unlink()
    os.mkfifo(target)

    before = readtree(root)

    child = _spawn(root, "fifo")
    try:
        out, err = child.communicate(input="x\n", timeout=5)
    finally:
        _reap(child)

    assert child.returncode == 0
    assert out.strip() == "symlink_or_nonregular"
    assert err == ""
    after = readtree(root)
    assert after == before
    assert stat.S_ISFIFO(target.stat().st_mode)
    assert entry not in after
