"""Supervisor-authored adversarial checks for the recovery unit draft."""

import pytest

from atlas_sers.evaluation import p05_recovery_unit as unit


def _counters():
    return {
        "new_started": 0,
        "new_completed": 0,
        "new_failed": 0,
        "new_optimizer_steps": 0,
        "new_optimizer_steps_exact": True,
        "new_elapsed_seconds": 0.0,
        "new_peak_cuda_bytes": 0,
        "reused_completed": 0,
        "reused_optimizer_steps": 0,
        "replay_started": 0,
        "unstarted_started": 0,
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("new_started", -1),
        ("new_started", True),
        ("new_started", 0.5),
        ("new_started", 1),
        ("reused_completed", 8721),
        ("reused_optimizer_steps", -1),
        ("new_elapsed_seconds", float("nan")),
        ("new_elapsed_seconds", float("inf")),
        ("new_elapsed_seconds", 10**1000),
        ("new_optimizer_steps_exact", False),
        ("new_failed", 1),
        ("replay_started", 2),
        ("new_peak_cuda_bytes", 2**32 + 1),
    ],
)
def test_invalid_counter_state_cannot_enter_another_unit(key, value):
    counters = _counters()
    counters[key] = value
    with pytest.raises(unit.RecoveryUnitError):
        unit._validate_counters(counters)


def test_journal_symlink_is_rejected_before_writing(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    outsider = tmp_path / "outsider"
    outsider.write_text("untouched")
    events = stage / "events.jsonl"
    events.symlink_to(outsider)
    with pytest.raises(unit.RecoveryUnitError):
        unit._validate_journal_paths(stage, events, stage / "selector.jsonl")
    assert outsider.read_text() == "untouched"


def test_aggregate_never_claims_zero_fits_when_it_trained():
    result = unit._aggregate(
        unit_id="synthetic-unit",
        unit_kind="new",
        group=[{"slot_id": "synthetic-slot"}],
        completed_ids=[],
        interrupted_slot_id="interrupted-elsewhere",
        reused_updates=0,
        new_updates=120,
        unit_new_started=1,
        unit_new_completed=1,
        unit_new_failed=0,
        started=0.0,
    )
    assert result["fits_started"] == 1
    assert "files_written" not in result or result["files_written"] > 0
