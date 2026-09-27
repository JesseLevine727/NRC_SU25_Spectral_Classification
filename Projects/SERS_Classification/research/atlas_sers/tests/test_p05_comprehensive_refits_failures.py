"""Adversarial lifecycle tests for the comprehensive P05 refit runner.

These tests reuse the synthetic ``world`` fixture from
``test_p05_comprehensive_refits`` and never fit a scientific model.  They drive
real temporary-filesystem state and the real ``StorageBudget`` while forcing
deterministic failure paths through the runner.
"""

from __future__ import annotations

# Optional torch must be available before importing torch-dependent test helpers.
# ruff: noqa: E402
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")

from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_refits as runner
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_refit_authority as authority
from atlas_sers.evaluation import p05_refit_evidence as evidence
from atlas_sers.evaluation import p05_refit_io as refit_io
from atlas_sers.evaluation.p05_comprehensive_storage import P05StorageError, StorageBudget
from tests import test_p05_comprehensive_refits as fixtures
from tests.test_p05_comprehensive_refits import (
    _failure_summary,
    _stage_path,
)

world = fixtures.world


def test_recorder_close_failure_preserves_exact_updates(world, monkeypatch):
    real_open = pilot.open_history_recorder

    class _Closer:
        def __init__(self, recorder):
            self._recorder = recorder

        def __call__(self, entry):
            self._recorder(entry)

        def close(self):
            self._recorder.close()
            raise runner.P05ComprehensiveRefitError("recorder_close_failed")

    monkeypatch.setattr(
        pilot,
        "open_history_recorder",
        lambda unit_dir, refit_id: _Closer(real_open(unit_dir, refit_id)),
    )

    with pytest.raises(runner.P05ComprehensiveRefitError) as info:
        runner.run_refits(**world.run_kwargs())

    assert info.value.reason_code == "recorder_close_failed"
    counters = _failure_summary(world)["counters"]
    assert counters["optimizer_steps"] == 120
    assert counters["optimizer_steps_exact"] is True
    assert counters["neural_started"] == 1
    assert counters["neural_failed"] == 1
    assert len(world.record["train"]) == 1


def test_prepare_inputs_failure_consumes_lease_without_fits(world, monkeypatch):
    def boom(bundle, spec):
        raise runner.P05ComprehensiveRefitError("prepare_inputs_failed")

    monkeypatch.setattr(refit_io, "prepare_refit_inputs", boom)

    with pytest.raises(runner.P05ComprehensiveRefitError) as info:
        runner.run_refits(**world.run_kwargs())

    assert info.value.reason_code == "prepare_inputs_failed"
    counters = _failure_summary(world)["counters"]
    assert counters["calibration_started"] == 0
    assert counters["calibration_completed"] == 0
    assert counters["neural_started"] == 0
    assert world.record["train"] == []
    units = list((_stage_path(world) / "units").iterdir())
    assert len(units) == 1
    assert (units[0] / "lease.json").is_file()


def test_calibration_save_failure_charges_started_failed_without_neural(world, monkeypatch):
    def boom(unit_dir, calibrated, audit, spec):
        raise runner.P05ComprehensiveRefitError("calibration_save_failed")

    monkeypatch.setattr(evidence, "persist_calibration", boom)

    with pytest.raises(runner.P05ComprehensiveRefitError) as info:
        runner.run_refits(**world.run_kwargs())

    assert info.value.reason_code == "calibration_save_failed"
    counters = _failure_summary(world)["counters"]
    assert counters["calibration_started"] == 1
    assert counters["calibration_completed"] == 0
    assert counters["calibration_failed"] == 1
    assert counters["neural_started"] == 0
    assert counters["neural_failed"] == 0
    assert world.record["train"] == []


def test_unit_storage_check_failure_after_first_callback_charges_lower_bound(world, monkeypatch):
    real_check = StorageBudget.check

    def check(budget, headroom_bytes=0):
        usage = real_check(budget, headroom_bytes=headroom_bytes)
        if budget._unit is not None:
            history = list((budget._unit / "histories").glob("*.jsonl"))
            if history and history[0].stat().st_size > 0:
                raise P05StorageError("storage_ceiling_exceeded")
        return usage

    monkeypatch.setattr(StorageBudget, "check", check)

    with pytest.raises(P05StorageError) as info:
        runner.run_refits(**world.run_kwargs())

    assert info.value.reason_code == "storage_ceiling_exceeded"
    counters = _failure_summary(world)["counters"]
    assert counters["optimizer_steps"] == refit_io.UPDATES_PER_EPOCH
    assert counters["optimizer_steps_exact"] is False
    assert counters["neural_started"] == 1
    assert counters["neural_failed"] == 1
    assert len(world.record["train"]) == 1


def test_deadline_after_calibration_prevents_neural(world, monkeypatch):
    holder = {"deadline": None, "calibrated": False}
    real_auth = authority.authenticate_selection

    def auth(bundle, deadline):
        holder["deadline"] = deadline
        return real_auth(bundle, deadline)

    monkeypatch.setattr(authority, "authenticate_selection", auth)

    def monotonic():
        if holder["deadline"] is None:
            return 0.0
        return holder["deadline"] + (1.0 if holder["calibrated"] else -1.0)

    monkeypatch.setattr(time, "perf_counter", monotonic)

    real_persist = evidence.persist_calibration

    def persist(unit_dir, calibrated, audit, spec):
        real_persist(unit_dir, calibrated, audit, spec)
        holder["calibrated"] = True

    monkeypatch.setattr(evidence, "persist_calibration", persist)

    with pytest.raises(core.P05CoreError) as info:
        runner.run_refits(**world.run_kwargs())

    assert getattr(info.value, "reason_code", None) == "global_deadline_exceeded"
    counters = _failure_summary(world)["counters"]
    assert counters["calibration_completed"] == 1
    assert counters["neural_started"] == 0
    assert counters["neural_failed"] == 0
    assert world.record["train"] == []


def test_insufficient_free_cuda_prevents_stage_write(world, monkeypatch):
    monkeypatch.setattr(pilot, "_free_cuda_bytes", lambda torch_module: 0)

    with pytest.raises(runner.P05ComprehensiveRefitError) as info:
        runner.run_refits(**world.run_kwargs())

    assert info.value.reason_code == "insufficient_free_cuda_bytes"
    run_root = runner._run_root(world.artifact_root, world.bundle["permit_sha256"])
    assert not (run_root / runner.STAGE_NAME).exists()
    assert world.record["train"] == []
    assert world.record["calibrate"] == []


def test_existing_regular_receipt_not_overwritten(world):
    run_root = runner._run_root(world.artifact_root, world.bundle["permit_sha256"])
    run_root.mkdir(parents=True, exist_ok=True)
    receipt = run_root / runner.RECEIPT_NAME
    receipt.write_bytes(b"original-receipt")

    with pytest.raises(core.P05CoreError):
        runner.run_refits(**world.run_kwargs())

    assert receipt.read_bytes() == b"original-receipt"
    assert not (run_root / runner.STAGE_NAME).exists()


def test_existing_stage_summary_not_overwritten_or_manifest_added(world):
    stage = _stage_path(world)
    stage.mkdir(parents=True)
    summary = stage / "summary.json"
    summary.write_bytes(b"original-summary")

    with pytest.raises(core.P05CoreError):
        runner.run_refits(**world.run_kwargs())

    assert summary.read_bytes() == b"original-summary"
    assert not (stage / "manifest.json").exists()


@pytest.mark.parametrize(
    ("overrides", "reason_code", "charged"),
    [
        (
            {"elapsed": runner.MAXIMUM_FIT_SECONDS + 1.0},
            "fit_seconds_exceeded",
            ("sum_fit_elapsed_seconds", runner.MAXIMUM_FIT_SECONDS + 1.0),
        ),
        (
            {"peak": runner.MAXIMUM_CUDA_ALLOCATED_BYTES + 1},
            "peak_cuda_exceeded",
            ("peak_cuda_bytes", runner.MAXIMUM_CUDA_ALLOCATED_BYTES + 1),
        ),
        (
            {"optimizer_steps": 30 * refit_io.UPDATES_PER_EPOCH + refit_io.UPDATES_PER_EPOCH},
            "refit_updates_exceeded",
            ("optimizer_steps", 30 * refit_io.UPDATES_PER_EPOCH + refit_io.UPDATES_PER_EPOCH),
        ),
    ],
)
def test_overlimit_result_is_charged_persisted_then_fails(world, overrides, reason_code, charged):
    def handler(spec, on_epoch):
        result = world.build_result(spec, **overrides)
        for item in result.history:
            on_epoch(dict(item))
        return result

    world.train_handler = handler

    with pytest.raises(runner.P05ComprehensiveRefitError) as info:
        runner.run_refits(**world.run_kwargs())

    assert info.value.reason_code == reason_code
    first = world.ordered()[0]
    unit = _stage_path(world) / "units" / first["refit_id"]
    assert (unit / "summary.json").is_file()
    counters = _failure_summary(world)["counters"]
    key, value = charged
    assert counters[key] == value
    assert counters["neural_started"] == 1
    assert counters["neural_failed"] == 1


def test_late_post_reauth_exception_charges_all_six_without_receipt(world, monkeypatch):
    def boom(*args):
        raise runner.P05ComprehensiveRefitError("post_run_reauth_failed")

    monkeypatch.setattr(pilot, "_post_run_reauth", boom)

    with pytest.raises(runner.P05ComprehensiveRefitError) as info:
        runner.run_refits(**world.run_kwargs())

    assert info.value.reason_code == "post_run_reauth_failed"
    counters = _failure_summary(world)["counters"]
    assert counters["calibration_completed"] == 6
    assert counters["neural_completed"] == 6
    assert counters["optimizer_steps"] == 720
    assert len(world.record["train"]) == 6
    run_root = runner._run_root(world.artifact_root, world.bundle["permit_sha256"])
    assert not (run_root / runner.RECEIPT_NAME).exists()


def test_final_receipt_write_failure_leaves_valid_failure_manifest(world, monkeypatch):
    real_write = freeze._budgeted_write

    def write(path, payload, budget):
        if Path(path).name == runner.RECEIPT_NAME:
            raise runner.P05ComprehensiveRefitError("receipt_write_failed")
        return real_write(path, payload, budget)

    monkeypatch.setattr(freeze, "_budgeted_write", write)

    with pytest.raises(runner.P05ComprehensiveRefitError) as info:
        runner.run_refits(**world.run_kwargs())

    assert info.value.reason_code == "receipt_write_failed"
    stage = _stage_path(world)
    assert _failure_summary(world)["status"] == "fail"
    pilot._verify_manifest(stage)
    assert not (stage.parent / runner.RECEIPT_NAME).exists()
