"""P08-T146 regressions for integrated outer setup and storage costs.

Everything here is invented metadata and temporary directories.  No model is
fitted, no dataset is read, no device is touched and no scientific execution
is authorized.  No private scientific storage or data is used; the invented
launch record and temporary storage are checked only as implementation
evidence.  Resource observations are faked, there are zero fits and zero GPU
work, and each assertion is implementation evidence only, never a science
result.
"""

from __future__ import annotations

import hashlib
import os
import time
from types import SimpleNamespace

import pytest

from atlas_sers.evaluation import p08_u0_session_io as sio
from atlas_sers.evaluation import p08_u0_store as store
from tests.p08_store_fixtures import bind_manifest, readtree

_ARTIFACT = "artifacts"
_JOURNAL = "journal"
_CONTROL = "control"
_LAUNCH = "launch.json"
_LAUNCH_RECORD = b'{"invented":"launch-record"}'
_RECEIVED = b"invented-record-bytes"


@pytest.fixture(autouse=True)
def fake_capacity(monkeypatch):
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda fd: SimpleNamespace(f_bavail=64 * 1024**3, f_frsize=1),
    )


@pytest.fixture
def io_env(monkeypatch, tmp_path, request):
    owner = store.create_store(
        str(tmp_path / _JOURNAL), bind_manifest(monkeypatch)
    )
    request.addfinalizer(owner.close)
    return SimpleNamespace(owner=owner, tmp_path=tmp_path)


def _total_bytes(root):
    return sum(len(payload) for payload in readtree(root).values())


def _write_control(base, payload, name=_CONTROL):
    control = base / name
    control.mkdir()
    (control / _LAUNCH).write_bytes(payload)
    return control, hashlib.sha256(payload).hexdigest()


def _make_control_io(env, request, payload=_LAUNCH_RECORD):
    control, digest = _write_control(env.tmp_path, payload)
    io = sio._SessionIO(
        env.owner,
        str(env.tmp_path / _ARTIFACT),
        started_monotonic_ns=time.monotonic_ns(),
        launch_control_root=str(control),
        launch_record_sha256=digest,
    )
    request.addfinalizer(io.close)
    return io, control


def _expect_rejected(env, reason_code, **kwargs):
    artifact_path = env.tmp_path / _ARTIFACT
    with pytest.raises(sio.SessionIOError) as captured:
        sio._SessionIO(
            env.owner,
            str(artifact_path),
            started_monotonic_ns=time.monotonic_ns(),
            **kwargs,
        )
    assert captured.value.reason_code == reason_code
    assert not artifact_path.exists()
    summary = env.owner.snapshot()["summary"]
    assert summary["model_fit_attempts"] == 0
    assert summary["journal_state"] == "not_started"


def _session_modules():
    pytest.importorskip("torch")
    from atlas_sers.evaluation import p08_u0_session as core
    from tests import test_p08_u0_runtime_inputs as runtime

    return core, runtime


@pytest.fixture
def source_env(monkeypatch, tmp_path, request):
    core, runtime = _session_modules()
    ctx, inputs = runtime._prepare(monkeypatch, "master_cv")
    owner = store.create_store(str(tmp_path / _JOURNAL), ctx["case"]["attempt"])
    request.addfinalizer(owner.close)
    return SimpleNamespace(
        core=core, inputs=inputs, ctx=ctx, owner=owner, tmp_path=tmp_path
    )


# ----------------------------------------------------------------------
# Outer setup elapsed: earlier monotonic start is charged once
# ----------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["bool", "negative", "float", "future"])
def test_invalid_start_rejected_before_directory(source_env, kind):
    env = source_env
    if kind == "bool":
        started = True
    elif kind == "negative":
        started = -1
    elif kind == "float":
        started = 1.5
    else:
        started = time.monotonic_ns() + 10**12
    with pytest.raises(env.core.SessionError) as captured:
        env.core._SourceSession(
            env.owner,
            env.inputs,
            artifact_root=str(env.tmp_path / _ARTIFACT),
            torch_module=object(),
            started_monotonic_ns=started,
        )
    assert captured.value.reason_code == "invalid_session_input"
    assert not (env.tmp_path / _ARTIFACT).exists()
    assert env.owner.snapshot()["summary"]["journal_state"] == "not_started"


def test_earlier_start_charges_setup_once(source_env, request, monkeypatch):
    env = source_env
    delta = 10**8
    clock_now = time.monotonic_ns() + 10**12
    started = clock_now - delta
    fake_time = SimpleNamespace(monotonic_ns=lambda: clock_now)
    monkeypatch.setattr(env.core, "time", fake_time)
    monkeypatch.setattr(sio, "time", fake_time)

    control, digest = _write_control(env.tmp_path, _LAUNCH_RECORD)
    session = env.core._SourceSession(
        env.owner,
        env.inputs,
        artifact_root=str(env.tmp_path / _ARTIFACT),
        torch_module=object(),
        started_monotonic_ns=started,
        launch_control_root=str(control),
        launch_record_sha256=digest,
    )
    request.addfinalizer(session._io.close)

    events = env.owner.snapshot()["events"]
    opens = [e for e in events if e["event_type"] == "session_open"]
    assert len(opens) == 1
    assert opens[0]["elapsed_ns"] == 0

    session._io.progress()
    session.close()
    report = session.report()

    events = env.owner.snapshot()["events"]
    progress = [e for e in events if e["event_type"] == "progress"]
    close_events = [e for e in events if e["event_type"] == "session_close"]
    assert progress and progress[-1]["elapsed_ns"] == delta
    assert close_events and close_events[-1]["elapsed_ns"] == delta

    assert report["measured_elapsed_ns"] == delta
    assert report["finalization_tail_ns"] == 0
    assert report["fit_attempt_count"] == 0
    assert report["prediction_attempt_count"] == 0
    assert report["completed_pair_count"] == 0
    assert report["incomplete"] is True

    assert report["observed_logical_bytes"] == _total_bytes(env.tmp_path)
    assert (control / _LAUNCH).read_bytes() == _LAUNCH_RECORD
    assert readtree(env.tmp_path / _ARTIFACT) == {}
    assert env.owner.snapshot()["summary"]["model_fit_attempts"] == 0

    assert report["execution_authorized"] is False
    assert report["live_runtime_accepted"] is False
    assert report["loaded_runtime_code_verified"] is False


# ----------------------------------------------------------------------
# Three-root byte accounting (journal + artifacts + control)
# ----------------------------------------------------------------------


def test_default_two_root_observation(io_env, request):
    io = sio._SessionIO(
        io_env.owner,
        str(io_env.tmp_path / _ARTIFACT),
        started_monotonic_ns=time.monotonic_ns(),
    )
    request.addfinalizer(io.close)
    io.open_session()
    assert io.observed_bytes() == (
        _total_bytes(io_env.tmp_path / _JOURNAL)
        + _total_bytes(io_env.tmp_path / _ARTIFACT)
    )


def test_three_root_observation_matches_readtree(io_env, request):
    io, control = _make_control_io(io_env, request)
    io.open_session()
    observed_open = io.observed_bytes()
    assert observed_open == _total_bytes(io_env.tmp_path)

    io.write("record.bin", _RECEIVED)
    io.progress()
    observed_progress = io.observed_bytes()
    assert observed_progress == _total_bytes(io_env.tmp_path)
    assert observed_progress > observed_open

    io.close_session()
    observed_close = io.observed_bytes()
    assert observed_close == _total_bytes(io_env.tmp_path)
    assert observed_close > observed_progress
    assert observed_close == (
        _total_bytes(io_env.tmp_path / _JOURNAL)
        + _total_bytes(io_env.tmp_path / _ARTIFACT)
        + _total_bytes(control)
    )


# ----------------------------------------------------------------------
# Launch-control argument and layout faults
# ----------------------------------------------------------------------


def test_control_argument_pair_mismatch(io_env):
    control, digest = _write_control(io_env.tmp_path, _LAUNCH_RECORD)
    _expect_rejected(
        io_env,
        "invalid_input",
        launch_control_root=str(control),
        launch_record_sha256=None,
    )
    _expect_rejected(
        io_env,
        "invalid_input",
        launch_control_root=None,
        launch_record_sha256=digest,
    )


@pytest.mark.parametrize(
    "bad_sha", ["", "abc", "0" * 63, "0" * 65, "g" * 64, 1234, b"0" * 64]
)
def test_control_sha_shape_rejected(io_env, bad_sha):
    control, _ = _write_control(io_env.tmp_path, _LAUNCH_RECORD)
    _expect_rejected(
        io_env,
        "invalid_input",
        launch_control_root=str(control),
        launch_record_sha256=bad_sha,
    )


def test_missing_control_record_rejected(io_env):
    control = io_env.tmp_path / _CONTROL
    control.mkdir()
    _expect_rejected(
        io_env,
        "control_mismatch",
        launch_control_root=str(control),
        launch_record_sha256="a" * 64,
    )


def test_extra_control_record_rejected(io_env):
    control, digest = _write_control(io_env.tmp_path, _LAUNCH_RECORD)
    (control / "extra.bin").write_bytes(b"x")
    _expect_rejected(
        io_env,
        "control_mismatch",
        launch_control_root=str(control),
        launch_record_sha256=digest,
    )


@pytest.mark.parametrize("payload", [b"", b"x" * (65536 + 1)])
def test_control_record_size_rejected(io_env, payload):
    control, digest = _write_control(io_env.tmp_path, payload)
    _expect_rejected(
        io_env,
        "control_mismatch",
        launch_control_root=str(control),
        launch_record_sha256=digest,
    )


def test_control_content_hash_mismatch_rejected(io_env):
    control, _ = _write_control(io_env.tmp_path, _LAUNCH_RECORD)
    _expect_rejected(
        io_env,
        "control_mismatch",
        launch_control_root=str(control),
        launch_record_sha256="0" * 64,
    )


def test_control_parent_mismatch_rejected(io_env):
    other = io_env.tmp_path / "other"
    other.mkdir()
    control, digest = _write_control(other, _LAUNCH_RECORD)
    _expect_rejected(
        io_env,
        "parent_mismatch",
        launch_control_root=str(control),
        launch_record_sha256=digest,
    )


def test_control_root_alias_rejected(io_env):
    _, digest = _write_control(io_env.tmp_path, _LAUNCH_RECORD)
    _expect_rejected(
        io_env,
        "artifact_exists",
        launch_control_root=str(io_env.tmp_path / _ARTIFACT),
        launch_record_sha256=digest,
    )
    _expect_rejected(
        io_env,
        "artifact_exists",
        launch_control_root=str(io_env.tmp_path / _JOURNAL),
        launch_record_sha256=digest,
    )


def test_control_symlink_rejected(io_env):
    target = io_env.tmp_path / "target.json"
    target.write_bytes(_LAUNCH_RECORD)
    control = io_env.tmp_path / _CONTROL
    control.mkdir()
    (control / _LAUNCH).symlink_to(target)
    _expect_rejected(
        io_env,
        "symlink_or_nonregular",
        launch_control_root=str(control),
        launch_record_sha256=hashlib.sha256(_LAUNCH_RECORD).hexdigest(),
    )


def test_control_hardlink_rejected(io_env):
    control, digest = _write_control(io_env.tmp_path, _LAUNCH_RECORD)
    os.link(control / _LAUNCH, io_env.tmp_path / "hardlink.json")
    _expect_rejected(
        io_env,
        "symlink_or_nonregular",
        launch_control_root=str(control),
        launch_record_sha256=digest,
    )


def test_control_nonregular_rejected(io_env):
    control = io_env.tmp_path / _CONTROL
    control.mkdir()
    os.mkfifo(control / _LAUNCH)
    _expect_rejected(
        io_env,
        "symlink_or_nonregular",
        launch_control_root=str(control),
        launch_record_sha256="a" * 64,
    )


def test_control_directory_replacement_poisons(io_env, request):
    io, control = _make_control_io(io_env, request)
    io.open_session()
    assert io.observed_bytes() == _total_bytes(io_env.tmp_path)
    artifacts_before = readtree(io_env.tmp_path / _ARTIFACT)

    control.rename(io_env.tmp_path / "control-old")
    replacement = io_env.tmp_path / _CONTROL
    replacement.mkdir()
    (replacement / _LAUNCH).write_bytes(_LAUNCH_RECORD)

    with pytest.raises(sio.SessionIOError) as captured:
        io.observed_bytes()
    assert captured.value.reason_code == "identity_mismatch"
    with pytest.raises(sio.SessionIOError) as captured:
        io.observed_bytes()
    assert captured.value.reason_code == "poisoned"
    assert readtree(io_env.tmp_path / _ARTIFACT) == artifacts_before
    assert io_env.owner.snapshot()["summary"]["model_fit_attempts"] == 0


def test_control_identical_content_inode_replacement_rejected(io_env, request):
    io, control = _make_control_io(io_env, request)
    io.open_session()
    io.observed_bytes()
    staged = io_env.tmp_path / "replacement.json"
    staged.write_bytes(_LAUNCH_RECORD)
    os.replace(staged, control / _LAUNCH)
    with pytest.raises(sio.SessionIOError) as captured:
        io.observed_bytes()
    assert captured.value.reason_code == "control_mismatch"


def test_control_same_size_overwrite_rejected(io_env, request):
    io, control = _make_control_io(io_env, request)
    io.open_session()
    io.observed_bytes()
    mutated = b"X" * len(_LAUNCH_RECORD)
    assert mutated != _LAUNCH_RECORD
    (control / _LAUNCH).write_bytes(mutated)
    with pytest.raises(sio.SessionIOError) as captured:
        io.observed_bytes()
    assert captured.value.reason_code == "control_mismatch"


# ----------------------------------------------------------------------
# Storage ceiling accounting with retained control bytes
# ----------------------------------------------------------------------


def test_control_bytes_counted_once_at_ceiling(io_env, request):
    io, _ = _make_control_io(io_env, request)
    io.open_session()
    observed = io.observed_bytes()
    payload = b"abcd"
    ceiling = observed + len(payload) + 1
    io._limits = lambda: (0, ceiling)
    io.write("ok.bin", payload)
    assert (io_env.tmp_path / _ARTIFACT / "ok.bin").read_bytes() == payload
    assert io.observed_bytes() == observed + len(payload)


def test_control_bytes_exhaust_allowance(io_env, request):
    io, _ = _make_control_io(io_env, request)
    io.open_session()
    observed = io.observed_bytes()
    previous = io_env.owner.snapshot()["summary"]["new_artifact_bytes"]
    control_bytes = len(_LAUNCH_RECORD)
    assert control_bytes > 0
    assert max(previous, observed - control_bytes + 1) < observed
    before = readtree(io_env.tmp_path)
    io._limits = lambda: (0, observed)
    with pytest.raises(sio.SessionIOError) as captured:
        io.write("no.bin", b"x")
    assert captured.value.reason_code == "storage_limited"
    assert readtree(io_env.tmp_path) == before
    assert not (io_env.tmp_path / _ARTIFACT / "no.bin").exists()
    assert io_env.owner.snapshot()["summary"]["model_fit_attempts"] == 0


def test_precharged_growth_does_not_reduce_free_requirement(
    io_env, request, monkeypatch
):
    io, _ = _make_control_io(io_env, request)
    io.open_session()
    io.progress(next_job_id="fit001")
    state = io_env.owner.snapshot()
    observed = io.observed_bytes()
    previous = state["summary"]["new_artifact_bytes"]
    assert previous > observed
    ceiling = previous + 64
    io._limits = lambda: (0, ceiling)
    gap = ceiling - observed
    free = gap - 1
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda fd: SimpleNamespace(f_bavail=free, f_frsize=1),
    )
    before = readtree(io_env.tmp_path)
    with pytest.raises(sio.SessionIOError) as captured:
        io.write("value.bin", b"x")
    assert captured.value.reason_code == "storage_limited"
    assert readtree(io_env.tmp_path) == before


# ----------------------------------------------------------------------
# Concurrent mutation injected inside store._read_file
# ----------------------------------------------------------------------


@pytest.mark.parametrize("mutation", ["overwrite", "replace", "extra"])
def test_control_mutation_during_read_refused_same_observation(
    io_env, request, monkeypatch, mutation
):
    io, control = _make_control_io(io_env, request)
    io.open_session()
    io.observed_bytes()

    real_read = sio.store._read_file

    def injected(fd, name, limit):
        data = real_read(fd, name, limit)
        if fd == io._control_fd and name == _LAUNCH:
            if mutation == "overwrite":
                (control / _LAUNCH).write_bytes(b"Z" * len(data))
            elif mutation == "replace":
                staged = control / "staged.json"
                staged.write_bytes(data)
                os.replace(staged, control / _LAUNCH)
            else:
                (control / "extra.bin").write_bytes(b"x")
        return data

    monkeypatch.setattr(sio.store, "_read_file", injected)
    with pytest.raises(sio.SessionIOError) as captured:
        io.observed_bytes()
    assert captured.value.reason_code == "control_mismatch"
    with pytest.raises(sio.SessionIOError) as captured:
        io.observed_bytes()
    assert captured.value.reason_code == "poisoned"


# ----------------------------------------------------------------------
# Interrupted close / constructor teardown
# ----------------------------------------------------------------------


def test_close_interrupt_closes_all_owned_descriptors(io_env, monkeypatch):
    control, digest = _write_control(io_env.tmp_path, _LAUNCH_RECORD)
    io = sio._SessionIO(
        io_env.owner,
        str(io_env.tmp_path / _ARTIFACT),
        started_monotonic_ns=time.monotonic_ns(),
        launch_control_root=str(control),
        launch_record_sha256=digest,
    )
    owned = (io._root_fd, io._parent_fd, io._control_fd)
    assert all(fd >= 0 for fd in owned)
    signal = KeyboardInterrupt("invented-close-fault")
    real_close = os.close
    seen = []

    def closing(fd):
        real_close(fd)
        seen.append(fd)
        raise signal

    monkeypatch.setattr(os, "close", closing)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            io.close()
        assert captured.value is signal
        assert io._root_fd == -1
        assert io._parent_fd == -1
        assert io._control_fd == -1
        assert set(owned) <= set(seen)
        for fd in owned:
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        monkeypatch.setattr(os, "close", real_close)

    assert os.fstat(io_env.owner._root_fd)
    assert os.fstat(io_env.owner._parent_fd)


def test_constructor_failure_closes_acquired_descriptors(io_env, monkeypatch):
    control, _ = _write_control(io_env.tmp_path, _LAUNCH_RECORD)
    closed = []
    real_close_fds = store._close_fds

    def spy(*fds):
        closed.extend(fds)
        return real_close_fds(*fds)

    monkeypatch.setattr(store, "_close_fds", spy)
    with pytest.raises(sio.SessionIOError) as captured:
        sio._SessionIO(
            io_env.owner,
            str(io_env.tmp_path / _ARTIFACT),
            started_monotonic_ns=time.monotonic_ns(),
            launch_control_root=str(control),
            launch_record_sha256="0" * 64,
        )
    assert captured.value.reason_code == "control_mismatch"
    assert not (io_env.tmp_path / _ARTIFACT).exists()
    valid = [fd for fd in closed if fd >= 0]
    assert valid
    for fd in valid:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_constructor_interrupt_preserved_and_descriptors_closed(
    io_env, monkeypatch
):
    control, digest = _write_control(io_env.tmp_path, _LAUNCH_RECORD)
    closed = []
    real_close_fds = store._close_fds

    def spy(*fds):
        closed.append(fds)
        return real_close_fds(*fds)

    monkeypatch.setattr(store, "_close_fds", spy)
    signal = KeyboardInterrupt("invented-init-fault")
    real_read = store._read_file
    hit = []
    read_fd = []

    def exploding(fd, name, limit):
        if name == _LAUNCH:
            hit.append(name)
            read_fd.append(fd)
            raise signal
        return real_read(fd, name, limit)

    monkeypatch.setattr(store, "_read_file", exploding)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            sio._SessionIO(
                io_env.owner,
                str(io_env.tmp_path / _ARTIFACT),
                started_monotonic_ns=time.monotonic_ns(),
                launch_control_root=str(control),
                launch_record_sha256=digest,
            )
        assert captured.value is signal
    finally:
        monkeypatch.setattr(store, "_read_file", real_read)
    assert hit
    assert not (io_env.tmp_path / _ARTIFACT).exists()
    valid = [fd for call in closed for fd in call if fd >= 0]
    assert valid
    for fd in valid:
        with pytest.raises(OSError):
            os.fstat(fd)
    final = closed[-1]
    assert len(final) == 3
    artifact_fd, *persistent = final
    assert artifact_fd == -1
    assert all(fd >= 0 for fd in persistent)
    assert len(set(persistent)) == 2
    control_fd = read_fd[-1]
    assert control_fd in persistent
    parent_fd = persistent[1] if persistent[0] == control_fd else persistent[0]
    for fd in (control_fd, parent_fd):
        with pytest.raises(OSError):
            os.fstat(fd)
    assert os.fstat(io_env.owner._root_fd)
    assert os.fstat(io_env.owner._parent_fd)


def test_primary_interrupt_preserved_over_secondary_close_fault(
    io_env, monkeypatch
):
    """The control-read interruption survives a distinct close-time fault."""
    control, digest = _write_control(io_env.tmp_path, _LAUNCH_RECORD)
    primary = KeyboardInterrupt("invented-primary-read-fault")
    secondary = KeyboardInterrupt("invented-secondary-close-fault")

    real_resolve = store._resolve_parent
    control_parent_fds = []

    def resolving(path):
        fd, name = real_resolve(path)
        if path == str(control):
            control_parent_fds.append(fd)
        return fd, name

    monkeypatch.setattr(store, "_resolve_parent", resolving)

    real_read = store._read_file
    hits = []

    def exploding(fd, name, limit):
        if name == _LAUNCH:
            hits.append(name)
            raise primary
        return real_read(fd, name, limit)

    monkeypatch.setattr(store, "_read_file", exploding)

    closed_calls = []
    real_close_fds = store._close_fds

    def spying_close_fds(*fds):
        closed_calls.append(fds)
        return real_close_fds(*fds)

    monkeypatch.setattr(store, "_close_fds", spying_close_fds)

    real_close = os.close
    closed = []

    def closing(fd):
        real_close(fd)
        closed.append(fd)
        if control_parent_fds and fd == control_parent_fds[-1]:
            raise secondary

    monkeypatch.setattr(os, "close", closing)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            sio._SessionIO(
                io_env.owner,
                str(io_env.tmp_path / _ARTIFACT),
                started_monotonic_ns=time.monotonic_ns(),
                launch_control_root=str(control),
                launch_record_sha256=digest,
            )
        assert captured.value is primary
    finally:
        monkeypatch.setattr(os, "close", real_close)

    assert hits
    assert control_parent_fds
    assert control_parent_fds[-1] in closed
    assert not (io_env.tmp_path / _ARTIFACT).exists()
    persistent = [fd for call in closed_calls for fd in call if fd >= 0]
    assert persistent
    for fd in persistent:
        with pytest.raises(OSError):
            os.fstat(fd)
    with pytest.raises(OSError):
        os.fstat(control_parent_fds[-1])
    assert os.fstat(io_env.owner._root_fd)
    assert os.fstat(io_env.owner._parent_fd)


def test_control_close_interrupt_propagates_when_read_succeeds(
    io_env, monkeypatch
):
    """With a clean read the close interruption is the failure and surfaces."""
    control, digest = _write_control(io_env.tmp_path, _LAUNCH_RECORD)
    signal = KeyboardInterrupt("invented-success-close-fault")

    real_resolve = store._resolve_parent
    control_parent_fds = []

    def resolving(path):
        fd, name = real_resolve(path)
        if path == str(control):
            control_parent_fds.append(fd)
        return fd, name

    monkeypatch.setattr(store, "_resolve_parent", resolving)

    closed_calls = []
    real_close_fds = store._close_fds

    def spying_close_fds(*fds):
        closed_calls.append(fds)
        return real_close_fds(*fds)

    monkeypatch.setattr(store, "_close_fds", spying_close_fds)

    real_close = os.close
    closed = []

    def closing(fd):
        real_close(fd)
        closed.append(fd)
        if control_parent_fds and fd == control_parent_fds[-1]:
            raise signal

    monkeypatch.setattr(os, "close", closing)
    try:
        with pytest.raises(KeyboardInterrupt) as captured:
            sio._SessionIO(
                io_env.owner,
                str(io_env.tmp_path / _ARTIFACT),
                started_monotonic_ns=time.monotonic_ns(),
                launch_control_root=str(control),
                launch_record_sha256=digest,
            )
        assert captured.value is signal
    finally:
        monkeypatch.setattr(os, "close", real_close)

    assert control_parent_fds
    assert control_parent_fds[-1] in closed
    assert not (io_env.tmp_path / _ARTIFACT).exists()
    persistent = [fd for call in closed_calls for fd in call if fd >= 0]
    assert persistent
    for fd in persistent:
        with pytest.raises(OSError):
            os.fstat(fd)
    with pytest.raises(OSError):
        os.fstat(control_parent_fds[-1])
    assert os.fstat(io_env.owner._root_fd)
    assert os.fstat(io_env.owner._parent_fd)
