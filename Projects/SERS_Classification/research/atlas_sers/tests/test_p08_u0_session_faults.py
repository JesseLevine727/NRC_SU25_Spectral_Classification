"""Regression packaging of supervisor's independently checked invented-data diagnostics.

No real model fits. Resource observations are fake; direct GPU-labelled
journal jobs do not execute GPU kernels. Toy receipt success tests byte/journal
guards only, not scientific acceptance.

Public spec source bytes from the runtime fixture are real public, rows/action
arrays invented. No private paths, code contents, or data are present in this
input.
"""

import json
import os
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from atlas_sers.evaluation import p08_serial_resources as serial  # noqa: E402
from atlas_sers.evaluation import p08_u0_session as core  # noqa: E402
from atlas_sers.evaluation import p08_u0_session_io as sio  # noqa: E402
from atlas_sers.evaluation import p08_u0_store as store  # noqa: E402
from tests import test_p08_u0_runtime_inputs as runtime  # noqa: E402
from tests.p08_store_fixtures import bind_manifest, readtree, resources  # noqa: E402


def measured():
    return resources(filesystem_free_bytes=64 * 1024**3)


@pytest.fixture(autouse=True)
def fake_capacity(monkeypatch):
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda fd: SimpleNamespace(f_bavail=64 * 1024**3, f_frsize=1),
    )

    def sample(*args):
        started = time.monotonic_ns()
        return (
            started,
            time.monotonic_ns(),
            measured(),
            dict(
                initialized=False,
                observed=False,
                allocated=0,
                reserved=0,
                device_used=0,
                peak=0,
            ),
        )

    monkeypatch.setattr(serial, "_sample", sample)


@pytest.fixture
def owned(monkeypatch, tmp_path, request):
    owner = store.create_store(
        str(tmp_path / "journal"), bind_manifest(monkeypatch)
    )
    request.addfinalizer(owner.close)
    io = sio._SessionIO(
        owner,
        str(tmp_path / "artifacts"),
        started_monotonic_ns=time.monotonic_ns(),
    )
    request.addfinalizer(io.close)
    io.open_session()
    return io, owner, tmp_path


@pytest.fixture
def controller(monkeypatch, tmp_path, request):
    ctx, inputs = runtime._prepare(monkeypatch, "master_cv")
    pair = next(
        p
        for p in inputs.pairs
        if json.loads(p.prepared_pair.fit_job_json)["model_id"] == "C-RBF-SVM"
    )
    fit = json.loads(pair.prepared_pair.fit_job_json)["job_id"]
    pred = json.loads(pair.prepared_pair.prediction_job_json)["job_id"]
    owner = store.create_store(str(tmp_path / "journal"), ctx["case"]["attempt"])
    request.addfinalizer(owner.close)
    session = core._SourceSession(
        owner,
        inputs,
        artifact_root=str(tmp_path / "artifacts"),
        torch_module=object(),
    )
    request.addfinalizer(session._io.close)

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent diagnostic must not fit")

    monkeypatch.setattr(core.stage_backend, "invoke_source_fit", forbidden)
    return session, owner, inputs, fit, pred, tmp_path


def start(io, job):
    io.progress(next_job_id=job)
    io.start(job, measured(), measurement_started_ns=time.monotonic_ns())


def test_progress_receipt_and_sorting(owned):
    io, owner, root = owned
    start(io, "fit001")
    original = owner.snapshot()["events"][-1]
    io.progress()
    io.write("z.bin", b"z")
    io.write("a.bin", b"a")
    head = owner.snapshot()["summary"]["head_sha256"]
    assert head != original["event_sha256"]
    io.finish("fit001", "succeeded", {"z.bin": b"z", "a.bin": b"a"})
    receipt = json.loads((root / "artifacts" / "fit001-receipt.json").read_bytes())
    assert receipt["start_event_sha256"] == original["event_sha256"]
    assert [x["name"] for x in receipt["artifacts"]] == ["a.bin", "z.bin"]
    assert owner.snapshot()["events"][-1]["previous_sha256"] == head


@pytest.mark.parametrize("bound", ["free", "artifact"])
def test_storage_boundary_preserves_tree(owned, monkeypatch, bound):
    io, owner, root = owned
    before = readtree(root)
    if bound == "free":
        monkeypatch.setattr(
            os,
            "fstatvfs",
            lambda fd: SimpleNamespace(f_bavail=0, f_frsize=1),
        )
    else:
        ceiling = io.observed_bytes() + 1
        monkeypatch.setattr(io, "_limits", lambda: (0, ceiling))
    with pytest.raises(sio.SessionIOError):
        io.write("no.bin", b"xx")
    assert readtree(root) == before


@pytest.mark.parametrize("when", ["before_progress", "during_append"])
def test_freshness_refuses_start(owned, monkeypatch, when):
    io, owner, root = owned
    before = time.monotonic_ns()
    io.progress(next_job_id="fit001")
    stamp = before if when == "before_progress" else time.monotonic_ns()
    if when == "during_append":
        real_gate = io._storage_gate

        def aged(*args):
            real_gate(*args)
            monkeypatch.setattr(
                sio,
                "time",
                SimpleNamespace(monotonic_ns=lambda: stamp + 2_000_000_000),
            )

        monkeypatch.setattr(io, "_storage_gate", aged)
    with pytest.raises(sio.SessionIOError):
        io.start("fit001", measured(), measurement_started_ns=stamp)
    assert owner.snapshot()["summary"]["model_fit_attempts"] == 0


def test_partial_write_poisoned_and_preserved(owned, monkeypatch):
    io, owner, root = owned
    original = sio.store._write_exclusive

    def partial(fd, name, payload):
        original(fd, name, payload)
        raise OSError("invented storage fault")

    monkeypatch.setattr(sio.store, "_write_exclusive", partial)
    with pytest.raises(sio.SessionIOError):
        io.write("partial.bin", b"partial")
    assert (root / "artifacts" / "partial.bin").read_bytes() == b"partial"
    with pytest.raises(sio.SessionIOError):
        io.progress()
    assert owner.snapshot()["summary"]["journal_state"] == "open"


def test_no_overwrite(owned):
    io, owner, root = owned
    io.write("same.bin", b"first")
    with pytest.raises(sio.SessionIOError):
        io.write("same.bin", b"second")
    assert (root / "artifacts" / "same.bin").read_bytes() == b"first"


def test_replaced_artifact_root_refused(owned):
    io, owner, root = owned
    (root / "artifacts").rename(root / "preserved-artifacts")
    (root / "artifacts").mkdir()
    with pytest.raises(sio.SessionIOError):
        io.write("no.bin", b"no")
    assert not list((root / "artifacts").iterdir())


def test_owner_lost_refused(owned):
    io, owner, root = owned
    before = readtree(root)
    owner.close()
    with pytest.raises(sio.SessionIOError):
        io.snapshot()
    assert readtree(root) == before


def test_tampered_saved_artifact_cannot_finish(owned):
    io, owner, root = owned
    start(io, "fit001")
    io.write("value.bin", b"original")
    (root / "artifacts" / "value.bin").write_bytes(b"tampered")
    with pytest.raises(sio.SessionIOError):
        io.finish("fit001", "succeeded", {"value.bin": b"original"})
    assert owner.snapshot()["summary"]["attempts"]["fit001"]["status"] == "running"


def test_precharged_bytes_do_not_reduce_physical_reserve(owned, monkeypatch):
    io, owner, root = owned
    io.progress(next_job_id="fit001")
    state = owner.snapshot()
    observed = io.observed_bytes()
    assert state["summary"]["new_artifact_bytes"] > observed
    reserve, ceiling = io._limits()
    free = reserve + ceiling - observed - 1
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda fd: SimpleNamespace(f_bavail=free, f_frsize=1),
    )
    before = readtree(root)
    with pytest.raises(sio.SessionIOError):
        io.write("value.bin", b"x")
    assert readtree(root) == before


def test_invalid_pair_projection_precedes_directory_creation(monkeypatch, tmp_path):
    ctx, inputs = runtime._prepare(monkeypatch, "master_cv")
    owner = store.create_store(str(tmp_path / "journal"), ctx["case"]["attempt"])
    try:
        bad = replace(inputs, pairs=inputs.pairs[:-1])
        with pytest.raises(core.SessionError):
            core._SourceSession(
                owner,
                bad,
                artifact_root=str(tmp_path / "artifacts"),
                torch_module=object(),
            )
        assert not (tmp_path / "artifacts").exists()
        assert owner.snapshot()["summary"]["journal_state"] == "not_started"
    finally:
        owner.close()


@pytest.mark.parametrize("model,worker", [("C-RBF-SVM", "cpu"), ("D0-M", "gpu")])
def test_active_counts_come_from_journal(controller, monkeypatch, model, worker):
    session, owner, inputs, fit, pred, root = controller
    pair = next(
        p
        for p in inputs.pairs
        if json.loads(p.prepared_pair.fit_job_json)["model_id"] == model
    )
    job = json.loads(pair.prepared_pair.fit_job_json)["job_id"]
    start(session._io, job)
    observed = session._observe()
    assert observed["resources"]["active_" + worker + "_workers"] == 1
    other = "gpu" if worker == "cpu" else "cpu"
    assert observed["resources"]["active_" + other + "_workers"] == 0


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_interrupt_identity_and_no_retry(controller, monkeypatch, kind):
    session, owner, inputs, fit, pred, root = controller
    signal = kind("primary")
    calls = []

    def primary(*args, **kwargs):
        calls.append(1)
        raise signal

    def secondary(*args, **kwargs):
        raise SystemExit("secondary")

    monkeypatch.setattr(core.stage_backend, "invoke_source_fit", primary)
    monkeypatch.setattr(session._io, "write", secondary)
    with pytest.raises(kind) as captured:
        session.run_pair(fit)
    assert captured.value is signal
    with pytest.raises(core.SessionError):
        session.run_pair(fit)
    assert calls == [1]
    assert owner.snapshot()["summary"]["attempts"][fit]["status"] == "running"
    assert session.report()["completed_pair_count"] == 0


@pytest.mark.parametrize("broken", ["sample", "bytes"])
def test_observation_error_prevents_kernel(controller, monkeypatch, broken):
    session, owner, inputs, fit, pred, root = controller
    calls = []
    monkeypatch.setattr(
        core.stage_backend,
        "invoke_source_fit",
        lambda *a, **k: calls.append(1),
    )

    def unavailable(*args, **kwargs):
        raise RuntimeError("unavailable")

    if broken == "sample":
        monkeypatch.setattr(serial, "_sample", unavailable)
    else:
        monkeypatch.setattr(session._io, "observed_bytes", unavailable)
    with pytest.raises(core.SessionError):
        session.run_pair(fit)
    assert not calls
    assert session.report()["completed_pair_count"] == 0


def test_pending_prediction_refuses_close_from_journal(controller):
    session, owner, inputs, fit, pred, root = controller
    start(session._io, fit)
    session._io.write("toy.bin", b"not a fitted model")
    session._io.finish(fit, "succeeded", {"toy.bin": b"not a fitted model"})
    assert not session._succeeded_fit_ids
    with pytest.raises(core.SessionError):
        session.close()
    state = owner.snapshot()["summary"]
    assert state["journal_state"] == "open"
    assert state["attempts"][fit]["status"] == "succeeded"
    assert pred not in state["attempts"]
    assert session.report()["observation_current"] is False


def test_previous_open_and_closed_refused(controller):
    session, owner, inputs, fit, pred, root = controller
    for close_first in (False, True):
        if close_first:
            session.close()
        before = readtree(root)
        with pytest.raises(core.SessionError):
            core._SourceSession(
                owner,
                inputs,
                artifact_root=str(root / "new"),
                torch_module=object(),
            )
        assert readtree(root) == before


@pytest.mark.parametrize(
    "module,error",
    [(core, core.SessionError), (sio, sio.SessionIOError)],
)
def test_execution_denied(module, error):
    with pytest.raises(error) as captured:
        module.require_scientific_execution(execution_authorized=True)
    assert captured.value.reason_code == "scientific_execution_not_authorized"
