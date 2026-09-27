"""Real-filesystem close-out tests for frozen P05 evaluation stages."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_comprehensive_storage as storage
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_evaluation_close as close
from atlas_sers.evaluation import p05_pilot as pilot

IDENTITY = {"predictions_frozen": True, "predictions_complete": True, "all_complete": True}
COUNTERS = {"started": 2, "completed": 2, "failed": 0, "optimizer_steps": 0}


@pytest.fixture
def env(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    run_root = artifact / "p05comprehensive" / "runs" / "p"
    run_root.mkdir(parents=True)
    stage = run_root / "evaluation"
    stage.mkdir()
    bundle = {
        "artifact_root": artifact,
        "contract": {"contract": 1},
        "support": {"support": 1},
        "repository_root": tmp_path / "repo",
        "project_root": tmp_path / "project",
    }
    monkeypatch.setattr(pilot, "_post_run_reauth", lambda *a, **k: {"reauth": "ok"})
    monkeypatch.setattr(inputs, "prepare", lambda *a, **k: None)
    return SimpleNamespace(
        tmp=tmp_path,
        run_root=run_root,
        stage=stage,
        bundle=bundle,
        budget=storage.StorageBudget(artifact, run_root),
        monkeypatch=monkeypatch,
    )


def _finish(env, **overrides):
    kwargs = dict(
        bundle=env.bundle,
        stage=env.stage,
        run_root=env.run_root,
        budget=env.budget,
        identity=dict(IDENTITY),
        counters=dict(COUNTERS),
        prior=4000.0,
        wall_start=time.perf_counter() - 1.0,
        deadline=time.perf_counter() + 100.0,
        provenance_before={"gen": 0},
        contract_path=env.tmp / "contract.json",
        permit_path=env.tmp / "permit.json",
    )
    kwargs.update(overrides)
    return close.finish_stage(**kwargs)


def _failure(env, error=None):
    close.record_failure(
        stage=env.stage,
        identity=dict(IDENTITY),
        counters=dict(COUNTERS),
        prior=4000.0,
        wall_start=time.perf_counter() - 1.0,
        error=error or RuntimeError("boom"),
    )


def _raise_base(*args, **kwargs):
    raise BaseException("nope")


def test_finish_stage_success_manifest_and_readback(env):
    calls = {}

    def fake_prepare(*args, **kwargs):
        calls["kwargs"] = kwargs

    env.monkeypatch.setattr(inputs, "prepare", fake_prepare)
    receipt = _finish(env)
    assert receipt["status"] == "complete"
    flags = (
        receipt["predictions_frozen"],
        receipt["predictions_complete"],
        receipt["all_complete"],
    )
    assert flags == (True, True, True)
    assert receipt["scientific_seconds_cumulative_bound"] == pytest.approx(
        4000.0 + receipt["scientific_seconds_this_stage"]
    )
    summary = json.loads((env.stage / "summary.json").read_text())
    assert summary["scientific_seconds_this_stage"] <= receipt["scientific_seconds_this_stage"]
    provenance = env.stage / "provenance_after.json"
    manifest = env.stage / "manifest.json"
    assert provenance.is_file() and manifest.is_file()
    assert json.loads(provenance.read_text()) == {"reauth": "ok"}
    assert provenance.stat().st_mtime_ns <= manifest.stat().st_mtime_ns
    assert receipt["stage_manifest_sha256"] == core._canon().sha256_file(manifest)
    pilot._verify_manifest(env.stage)
    assert json.loads((env.run_root / close.RECEIPT_NAME).read_text()) == receipt
    assert calls["kwargs"]["require_unstarted"] is False


@pytest.mark.parametrize("kind", ["regular", "symlink"])
def test_finish_stage_preexisting_receipt_not_overwritten(env, kind):
    target = env.tmp / "sentinel.json"
    target.write_bytes(b"sentinel")
    receipt_path = env.run_root / close.RECEIPT_NAME
    if kind == "regular":
        receipt_path.write_bytes(b"sentinel")
    else:
        receipt_path.symlink_to(target)
    with pytest.raises(core.P05CoreError):
        _finish(env)
    expected = target if kind == "symlink" else receipt_path
    assert expected.read_bytes() == b"sentinel"
    assert receipt_path.is_symlink() is (kind == "symlink")


def test_record_failure_fresh_clock_counters_and_scope(env):
    before = {p.relative_to(env.run_root) for p in env.run_root.rglob("*")}
    close.record_failure(
        stage=env.stage,
        identity=dict(IDENTITY),
        counters={
            "started": 2,
            "completed": 1,
            "failed": 0,
            "optimizer_steps": 5,
            "elapsed_seconds": 999.0,
        },
        prior=4000.0,
        wall_start=time.perf_counter() - 1.0,
        error=RuntimeError("boom"),
    )
    payload = json.loads((env.stage / "summary.json").read_text())
    assert payload["status"] == "fail"
    assert payload["predictions_frozen"] is False
    assert payload["predictions_complete"] is False
    assert payload["all_complete"] is False
    assert payload["counters"]["failed"] == 1
    assert payload["counters"]["optimizer_steps"] == 5
    assert payload["counters"]["elapsed_seconds"] != 999.0
    assert 0.0 <= payload["counters"]["elapsed_seconds"] < 5.0
    assert payload["prior_scientific_seconds_cumulative_bound"] == 4000.0
    assert payload["scientific_seconds_cumulative_bound"] == pytest.approx(
        4000.0 + payload["scientific_seconds_this_stage"]
    )
    after = {p.relative_to(env.run_root) for p in env.run_root.rglob("*")}
    assert after <= before | {Path("evaluation/summary.json"), Path("evaluation/manifest.json")}
    env.monkeypatch.setattr(core, "_atomic_write", _raise_base)
    _failure(env)


def test_no_success_on_expired_deadline_or_storage_failure(env):
    with pytest.raises(core.P05CoreError):
        _finish(env, deadline=time.perf_counter() - 1.0)
    assert not (env.run_root / close.RECEIPT_NAME).exists()
    env.monkeypatch.setattr(
        env.budget,
        "check",
        lambda *a, **k: (_ for _ in ()).throw(storage.P05StorageError("storage_ceiling_exceeded")),
    )
    with pytest.raises(storage.P05StorageError):
        _finish(env)
    assert not (env.run_root / close.RECEIPT_NAME).exists()
    _failure(env, error=RuntimeError("storage"))
    assert json.loads((env.stage / "summary.json").read_text())["status"] == "fail"


@pytest.mark.parametrize("failure_point", ["reauth", "reconcile", "receipt_accounting"])
def test_late_failure_cannot_authenticate_a_complete_stage(env, failure_point):
    def fail(*args, **kwargs):
        raise core.P05CoreError("synthetic_late_failure")

    if failure_point == "reauth":
        env.monkeypatch.setattr(pilot, "_post_run_reauth", fail)
    elif failure_point == "reconcile":
        env.monkeypatch.setattr(env.budget, "reconcile", fail)
    else:
        original = env.budget.account_new_file

        def account(path):
            if Path(path).name == close.RECEIPT_NAME:
                fail()
            return original(path)

        env.monkeypatch.setattr(env.budget, "account_new_file", account)
    with pytest.raises(core.P05CoreError, match="synthetic_late_failure") as caught:
        _finish(env)
    # The runner preserves evidence and changes the stage to failure. A receipt
    # already written before a late failure must not authenticate that stage.
    _failure(env, caught.value)
    summary = json.loads((env.stage / "summary.json").read_text())
    assert summary["status"] == "fail" and summary["predictions_frozen"] is False
    receipt_path = env.run_root / close.RECEIPT_NAME
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt["stage_manifest_sha256"] != core._canon().sha256_file(
            env.stage / "manifest.json"
        )


@pytest.mark.parametrize(
    "prior,elapsed",
    [(float("nan"), 1.0), (4000.0, -1.0), (development.MAXIMUM_TOTAL_SECONDS, 1.0)],
)
def test_bound_payload_rejects_malformed_cumulative(prior, elapsed):
    with pytest.raises(core.P05CoreError):
        close._bound_payload(dict(IDENTITY), dict(COUNTERS), prior, elapsed)
