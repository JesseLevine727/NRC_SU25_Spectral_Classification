"""Synthetic unit-regression tests for U0 store descriptor cleanup.

Everything here is invented metadata and bounded temporary descriptors under
``tmp_path``.  No scientific data, instruments, authentication or scientific
execution is touched.  The global ``os`` monkeypatches are scoped with
``monkeypatch.context`` so pytest internals and fixtures are never affected.
"""

from __future__ import annotations

import errno
import os

import pytest

from atlas_sers.evaluation import p08_u0_store
from tests.p08_store_fixtures import bind_manifest


def _reap(fd):
    """Close one descriptor that the test still owns, if it is still open.

    The caller must only pass descriptors it captured itself and that are not
    recorded as already closed.  This helper only checks the current open
    state; it does not verify identity or ownership, so a re-used descriptor
    number may be closed inadvertently.
    """
    if type(fd) is not int or fd < 0:
        return
    try:
        os.fstat(fd)
    except OSError:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _open_root_fd(path):
    return os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)


def test_create_lock_fstat_interrupt_closes_tracked_fd(tmp_path, monkeypatch):
    lock_dir = tmp_path / "create_lock_fstat"
    lock_dir.mkdir()
    root_fd = _open_root_fd(lock_dir)
    opened = []
    closed = []
    try:
        with monkeypatch.context() as mp:
            real_open = os.open
            real_close = os.close

            def tracking_open(*args, **kwargs):
                fd = real_open(*args, **kwargs)
                opened.append(fd)
                return fd

            def tracking_close(fd):
                real_close(fd)
                closed.append(fd)

            def interrupt_fstat(fd):
                raise KeyboardInterrupt

            mp.setattr(os, "open", tracking_open)
            mp.setattr(os, "close", tracking_close)
            mp.setattr(os, "fstat", interrupt_fstat)

            with pytest.raises(KeyboardInterrupt):
                p08_u0_store._create_lock(root_fd)

        assert len(opened) == 1
        assert opened[0] != root_fd
        with pytest.raises(OSError) as excinfo:
            os.fstat(opened[0])
        assert excinfo.value.errno == errno.EBADF
    finally:
        _reap(root_fd)
        for fd in opened:
            if fd not in closed:
                _reap(fd)


def test_create_lock_fsync_interrupt_closes_tracked_fd(tmp_path, monkeypatch):
    lock_dir = tmp_path / "create_lock_fsync"
    lock_dir.mkdir()
    root_fd = _open_root_fd(lock_dir)
    opened = []
    closed = []
    try:
        with monkeypatch.context() as mp:
            real_open = os.open
            real_close = os.close

            def tracking_open(*args, **kwargs):
                fd = real_open(*args, **kwargs)
                opened.append(fd)
                return fd

            def tracking_close(fd):
                real_close(fd)
                closed.append(fd)

            def interrupt_fsync(fd):
                raise KeyboardInterrupt

            mp.setattr(os, "open", tracking_open)
            mp.setattr(os, "close", tracking_close)
            mp.setattr(os, "fsync", interrupt_fsync)

            with pytest.raises(KeyboardInterrupt):
                p08_u0_store._create_lock(root_fd)

        assert len(opened) == 1
        assert opened[0] != root_fd
        with pytest.raises(OSError) as excinfo:
            os.fstat(opened[0])
        assert excinfo.value.errno == errno.EBADF
    finally:
        _reap(root_fd)
        for fd in opened:
            if fd not in closed:
                _reap(fd)


def test_open_lock_fstat_interrupt_closes_tracked_fd(tmp_path, monkeypatch):
    lock_dir = tmp_path / "open_lock_fstat"
    lock_dir.mkdir()
    (lock_dir / p08_u0_store._LOCK_NAME).write_bytes(b"")
    root_fd = _open_root_fd(lock_dir)
    opened = []
    closed = []
    try:
        with monkeypatch.context() as mp:
            real_open = os.open
            real_close = os.close

            def tracking_open(*args, **kwargs):
                fd = real_open(*args, **kwargs)
                opened.append(fd)
                return fd

            def tracking_close(fd):
                real_close(fd)
                closed.append(fd)

            def interrupt_fstat(fd):
                raise KeyboardInterrupt

            mp.setattr(os, "open", tracking_open)
            mp.setattr(os, "close", tracking_close)
            mp.setattr(os, "fstat", interrupt_fstat)

            with pytest.raises(KeyboardInterrupt):
                p08_u0_store._open_lock(root_fd)

        assert len(opened) == 1
        assert opened[0] != root_fd
        with pytest.raises(OSError) as excinfo:
            os.fstat(opened[0])
        assert excinfo.value.errno == errno.EBADF
    finally:
        _reap(root_fd)
        for fd in opened:
            if fd not in closed:
                _reap(fd)


def test_close_fds_propagates_first_exception_once(monkeypatch):
    fds = [910001, 910002, 910003, 910004]
    effects = [
        KeyboardInterrupt(),
        OSError(errno.EIO, "invented io error"),
        SystemExit(7),
        None,
    ]
    calls = []

    with monkeypatch.context() as mp:

        def fake_close(fd):
            calls.append(fd)
            effect = effects[fds.index(fd)]
            if effect is not None:
                raise effect

        mp.setattr(os, "close", fake_close)
        with pytest.raises(KeyboardInterrupt) as caught:
            p08_u0_store._close_fds(*fds)

    assert caught.value is effects[0]
    assert calls == fds
    assert len(calls) == len(fds)


def test_close_fds_ignores_invalid_descriptors(monkeypatch):
    calls = []

    with monkeypatch.context() as mp:

        def fake_close(fd):
            calls.append(fd)

        mp.setattr(os, "close", fake_close)
        p08_u0_store._close_fds("1", 1.5, None, -1, -9, True, b"2")

    assert calls == []


def test_store_close_releases_every_owned_fd(tmp_path, monkeypatch):
    manifest = bind_manifest(monkeypatch)
    store = p08_u0_store.create_store(tmp_path / "store_close", manifest)

    captured = [
        store._events_fd,
        store._lock_fd,
        store._root_fd,
        store._parent_fd,
    ]
    assert all(type(fd) is int and fd >= 0 for fd in captured)
    assert len(set(captured)) == len(captured)

    really_closed = []
    calls = []

    try:
        with monkeypatch.context() as mp:
            real_close = os.close
            first = captured[0]

            def fake_close(fd):
                calls.append(fd)
                real_close(fd)
                really_closed.append(fd)
                if fd == first:
                    raise KeyboardInterrupt

            mp.setattr(os, "close", fake_close)

            with pytest.raises(KeyboardInterrupt):
                store.close()

            assert calls == captured
            assert store._events_fd == -1
            assert store._lock_fd == -1
            assert store._root_fd == -1
            assert store._parent_fd == -1

            for fd in captured:
                with pytest.raises(OSError) as excinfo:
                    os.fstat(fd)
                assert excinfo.value.errno == errno.EBADF

            with pytest.raises(p08_u0_store.StoreError) as excinfo:
                store.snapshot()
            assert excinfo.value.reason_code == "closed_store"

            before = len(calls)
            store.close()
            assert len(calls) == before
    finally:
        for fd in captured:
            if fd not in really_closed:
                _reap(fd)


def test_resolve_parent_closes_current_fd_on_interrupt(tmp_path, monkeypatch):
    base = tmp_path / "resolve_parent"
    (base / "a" / "b").mkdir(parents=True)
    target = base / "a" / "b" / "store"

    captured = []
    close_calls = []
    closed = []
    real_open = os.open
    real_close = os.close

    def tracking_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        captured.append(fd)
        return fd

    def fake_close(fd):
        close_calls.append(fd)
        real_close(fd)
        closed.append(fd)
        if len(close_calls) == 1:
            raise KeyboardInterrupt

    try:
        with monkeypatch.context() as mp:
            mp.setattr(os, "open", tracking_open)
            mp.setattr(os, "close", fake_close)
            with pytest.raises(KeyboardInterrupt):
                p08_u0_store._resolve_parent(target)

        assert len(captured) >= 2
        assert captured[0] != captured[1]
        assert close_calls == captured[:2]
        assert all(fd in captured for fd in close_calls)
        for fd in captured[:2]:
            with pytest.raises(OSError) as excinfo:
                os.fstat(fd)
            assert excinfo.value.errno == errno.EBADF
    finally:
        for fd in captured:
            if fd not in closed:
                _reap(fd)
