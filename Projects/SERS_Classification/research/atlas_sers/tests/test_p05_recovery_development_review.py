"""Independent, model-free regressions for recovery orchestration."""

import pytest

from atlas_sers.evaluation import p05_recovery_development as outer


def _unit_case(kind):
    ids = [f"slot-{index:02}" for index in range(12)]
    reused_count = {"sealed": 12, "partial": 8, "new": 0}[kind]
    new_count = 12 - reused_count
    replay = ids[8] if kind == "partial" else None
    prior = outer._initial_counters()
    counters = {
        **prior,
        "new_started": new_count,
        "new_completed": new_count,
        "new_optimizer_steps": new_count * 120,
        "reused_completed": reused_count,
        "reused_optimizer_steps": reused_count * 120,
        "replay_started": int(replay is not None),
        "unstarted_started": new_count - int(replay is not None),
    }
    result = {
        "unit_id": "synthetic-unit",
        "unit_kind": kind,
        "completed_slots": ids,
        "reused_slots": ids[:reused_count],
        "new_slots": ids[reused_count:],
        "replay_slot_id": replay,
        "new_started": new_count,
        "new_completed": new_count,
        "new_failed": 0,
        "fits_started": new_count,
        "optimizer_updates_reused": reused_count * 120,
        "optimizer_updates_new": new_count * 120,
    }
    return result, set(ids), prior, counters


@pytest.mark.parametrize("kind", ["sealed", "partial", "new"])
def test_accepts_actual_unit_api_including_replay_as_a_new_attempt(kind):
    result, ids, prior, counters = _unit_case(kind)
    outer._validate_unit_result(result, "synthetic-unit", ids, prior, counters)


@pytest.mark.parametrize("key,value", [("fits_started", 0), ("unit_kind", "invented")])
def test_inconsistent_unit_aggregate_is_rejected(key, value):
    result, ids, prior, counters = _unit_case("new")
    result[key] = value
    with pytest.raises(outer.RecoveryDevelopmentError):
        outer._validate_unit_result(result, "synthetic-unit", ids, prior, counters)


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf"), 10**1000])
def test_deadline_is_finite_and_typed(deadline):
    with pytest.raises(outer.RecoveryDevelopmentError):
        outer._check_deadline(deadline)


def test_counter_helper_rejects_fractional_counter():
    counters = outer._initial_counters()
    counters["new_started"] = 0.5
    with pytest.raises(outer.RecoveryDevelopmentError):
        outer._check_counters(counters)
