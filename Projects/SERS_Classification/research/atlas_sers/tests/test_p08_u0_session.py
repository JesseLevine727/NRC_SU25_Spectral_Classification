"""One-pair positive-path integration test for the P08-U0 session controller.

A real toy RBF-SVM fit is executed; the resource observations are fake.  The
public-spec bytes checked here are real public artifacts, while the rows and
arrays that drive the controller are invented for this test.

Boundary: only the proposed one-pair happy path is covered.  Negative and
error cases are intentionally out of scope.
"""

import hashlib
import json
import os
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from atlas_sers.evaluation import p08_serial_resources as serial  # noqa: E402
from atlas_sers.evaluation import p08_u0_session as subject  # noqa: E402
from atlas_sers.evaluation import p08_u0_store as store  # noqa: E402
from tests import test_p08_u0_runtime_inputs as fixture  # noqa: E402
from tests.p08_store_fixtures import readtree, resources  # noqa: E402


def _total_bytes(tree):
    return sum(len(value) for value in tree.values())


def test_p08_u0_session_one_positive_pair(monkeypatch, tmp_path, request):
    def fake_sample(output_directory, torch_module, model_threads):
        started = time.monotonic_ns()
        cuda = {
            "initialized": False,
            "observed": False,
            "allocated": 0,
            "reserved": 0,
            "device_used": 0,
            "peak": 0,
        }
        observed = resources(filesystem_free_bytes=64 * 1024**3)
        return started, time.monotonic_ns(), observed, cuda

    monkeypatch.setattr(serial, "_sample", fake_sample)
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda *args, **kwargs: SimpleNamespace(
            f_bavail=64 * 1024**3, f_frsize=1
        ),
    )

    ctx, inputs = fixture._prepare(monkeypatch, "master_cv")
    manifest = ctx["case"]["attempt"]
    assert len(inputs.pairs) == 78

    pair = next(
        p
        for p in inputs.pairs
        if json.loads(p.prepared_pair.fit_job_json)["model_id"] == "C-RBF-SVM"
    )
    fitid = json.loads(pair.prepared_pair.fit_job_json)["job_id"]
    predid = json.loads(pair.prepared_pair.prediction_job_json)["job_id"]

    owner = store.create_store(str(tmp_path / "journal"), manifest)
    request.addfinalizer(owner.close)
    session = subject._SourceSession(
        owner,
        inputs,
        artifact_root=str(tmp_path / "artifacts"),
        torch_module=object(),
    )
    request.addfinalizer(session._io.close)

    calls = {"fit": 0, "pred": 0}
    saved_bytes = {}
    real_fit = subject.stage_backend.invoke_source_fit
    real_pred = subject.stage_backend.verify_source_prediction

    def counting_fit(*args, **kwargs):
        calls["fit"] += 1
        return real_fit(*args, **kwargs)

    def counting_pred(*args, **kwargs):
        calls["pred"] += 1
        payload = kwargs.get("saved_artifact_bytes")
        if payload:
            saved_bytes.update(payload)
        return real_pred(*args, **kwargs)

    monkeypatch.setattr(subject.stage_backend, "invoke_source_fit", counting_fit)
    monkeypatch.setattr(
        subject.stage_backend, "verify_source_prediction", counting_pred
    )

    try:
        session.run_pair(fitid)
        original_result = session._last_result
        assert original_result is not None

        summary = owner.snapshot()["summary"]
        assert summary["attempts"][fitid]["status"] == "succeeded"
        assert summary["attempts"][predid]["status"] == "succeeded"
        assert summary["model_fit_attempts"] == 1
        assert summary["source_prediction_attempts"] == 1

        report = session.report()
        assert report["completed_pair_count"] == 1
        assert report["incomplete"] is True
        for key in (
            "execution_authorized",
            "live_runtime_accepted",
            "loaded_runtime_code_verified",
        ):
            assert report[key] is False

        events = owner.snapshot()["events"]
        event_types = [event["event_type"] for event in events]
        assert any(kind == "progress" for kind in event_types)
        for index, event in enumerate(events):
            if event["event_type"] == "attempt_start":
                assert events[index - 1]["event_type"] == "progress"
                assert events[index - 1]["elapsed_ns"] == event["elapsed_ns"]
                assert events[index - 1]["artifact_bytes"] == event["artifact_bytes"]

        artifacts = readtree(tmp_path / "artifacts")
        assert set(artifacts) == {
            f"{fitid}-summary.json",
            f"{fitid}-predictions.csv",
            f"{fitid}-receipt.json",
            f"{predid}-verification.json",
            f"{predid}-receipt.json",
        }

        fit_receipt = json.loads(artifacts[f"{fitid}-receipt.json"])
        pred_receipt = json.loads(artifacts[f"{predid}-receipt.json"])
        fit_items = fit_receipt["artifacts"]
        pred_items = pred_receipt["artifacts"]
        for item in fit_items + pred_items:
            assert set(item) == {"name", "size_bytes", "sha256"}
        assert {item["name"] for item in pred_items} == {
            f"{fitid}-summary.json",
            f"{fitid}-predictions.csv",
            f"{predid}-verification.json",
        }
        for item in fit_items + pred_items:
            data = artifacts[item["name"]]
            assert len(data) == item["size_bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]

        assert calls["fit"] == 1
        assert calls["pred"] == 1
        assert set(saved_bytes) == {"summary.json", "predictions.csv"}
        assert saved_bytes["summary.json"] == artifacts[f"{fitid}-summary.json"]
        assert saved_bytes["predictions.csv"] == artifacts[f"{fitid}-predictions.csv"]

        session.close()
        assert owner.snapshot() is not None
        assert session._last_result is original_result

        summary = owner.snapshot()["summary"]
        closed = session.report()
        assert closed["closed"] is True
        assert closed["journal_state"] == "closed"
        assert closed["recorded_active_wall_ns"] == summary["active_wall_ns"]
        assert closed["measured_elapsed_ns"] >= closed["recorded_active_wall_ns"]
        tail = closed["measured_elapsed_ns"] - closed["recorded_active_wall_ns"]
        assert tail >= 0
        assert closed["finalization_tail_ns"] == tail
        assert closed["recorded_artifact_bytes"] == summary["new_artifact_bytes"]
        actual = _total_bytes(readtree(tmp_path / "journal")) + _total_bytes(artifacts)
        assert closed["recorded_artifact_bytes"] >= actual
        assert closed["observed_logical_bytes"] == actual

        captured = session.report()
        old_ns = time.monotonic_ns()
        monkeypatch.setattr(
            subject,
            "time",
            SimpleNamespace(
                monotonic_ns=lambda: old_ns + 5_000_000_000,
                perf_counter=time.perf_counter,
            ),
        )
        assert session.report() == captured
    finally:
        session._io.close()
        owner.close()
