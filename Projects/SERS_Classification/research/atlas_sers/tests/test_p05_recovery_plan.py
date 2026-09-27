"""Metadata-only tests for the P05 recovery plan builder.

These tests exercise pure planning over synthetic metadata.  They do not
authenticate real files, load numpy/torch, or run any numerical training.
All identifiers are synthetic; only the frozen public pin digests are reused.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from atlas_sers.evaluation.p05_recovery_plan import (
    RecoveryPlanError,
    build_recovery_plan,
)

PILOT_PERMIT_SHA256 = "652f5c07a1076a907778a9dd80394203ded9084298a95ce791cb5ee2814e576d"
BASE_PERMIT_SHA256 = "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8"
CORE_CONTRACT_SHA256 = "60e3a49753c59fb7038c83e50795614ad1cb4ca764dd487ac49692edcaf2ccae"
CORE_PLAN_ID = "a6334b2ed13a92fd953e4202bc2153e1aea4d12419d2a6f891f64f126136fe37"
LEDGER_ID = "P05DEV-8bf60eeca36d4b4663441eda"

RECIPES = ("D0-M", "D1", "D2", "D3")
SEEDS = (20260805, 20260817, 20260829)
TOTAL_UNITS = 1245
SLOTS_PER_UNIT = 12
PILOT_UNITS = 3
SOURCE_SLOTS = TOTAL_UNITS * SLOTS_PER_UNIT
PILOT_SLOTS = PILOT_UNITS * SLOTS_PER_UNIT
ORIGINAL_STARTED = 8721
ORIGINAL_COMPLETED = 8720
INTERRUPTED_UPDATES = 68
INTERRUPTED_CHARGED_UPPER = 800
ORIGINAL_COMPLETED_EXACT = 1669388
ORIGINAL_CHARGED_UPPER = ORIGINAL_COMPLETED_EXACT + INTERRUPTED_CHARGED_UPPER

INPUT_KEYS = (
    "ledger",
    "pilot_slots",
    "events",
    "selector_records",
    "leases",
    "interrupted_history",
)


def _hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canon_bytes(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _canon_sha(value) -> str:
    return hashlib.sha256(_canon_bytes(value)).hexdigest()


def _selector(slot, unit):
    return {
        "slot_id": slot["slot_id"],
        "context_id": unit["context_id"],
        "selection_unit_id": unit["selection_unit_id"],
        "slot_kind": slot["slot_kind"],
        "fitting_role_id": unit["fitting_role_id"],
        "validation_role_id": unit["validation_role_id"],
        "recipe_id": slot["recipe_id"],
        "seed": slot["seed"],
        "status": "complete",
        "best_epoch": 30,
        "best_validation_balanced_accuracy": 0.5,
        "best_validation_nll": 1.0,
        "best_validation_macro_f1": 0.5,
        "best_validation_predicted_class_count": 3,
    }


def _lease(slot, permit_sha256):
    return {
        "slot_id": slot["slot_id"],
        "unit_id": slot["unit_id"],
        "recipe_id": slot["recipe_id"],
        "seed": slot["seed"],
        "contract_sha256": CORE_CONTRACT_SHA256,
        "core_plan_id": CORE_PLAN_ID,
        "permit_sha256": permit_sha256,
    }


def _build_baseline():
    units = []
    slots = []
    for index in range(TOTAL_UNITS):
        unit_id = f"unit-{index:04d}"
        fitting_role = f"fit-{index:04d}"
        validation_role = f"val-{index:04d}"
        units.append(
            {
                "unit_id": unit_id,
                "station": "cwa",
                "context_id": f"ctx-{index:04d}",
                "selection_unit_id": f"sel-{index:04d}",
                "fitting_role_id": fitting_role,
                "validation_role_id": validation_role,
                "fitting_uid_set_sha256": _hex("fit" + unit_id),
                "validation_uid_set_sha256": _hex("val" + unit_id),
            }
        )
        for recipe_index, recipe in enumerate(RECIPES):
            for seed_index, seed in enumerate(SEEDS):
                slots.append(
                    {
                        "slot_id": f"u{index:04d}r{recipe_index}s{seed_index}",
                        "unit_id": unit_id,
                        "recipe_id": recipe,
                        "seed": seed,
                        "slot_kind": "inherited_selection_fit",
                        "excluded_by_protocol": False,
                        "fitting_role_id": fitting_role,
                        "validation_role_id": validation_role,
                    }
                )
    ledger = {
        "ledger_id": LEDGER_ID,
        "execution_authorized": False,
        "arrays_loaded": False,
        "fits_started": 0,
        "units": units,
        "slots": slots,
    }
    unit_by_id = {unit["unit_id"]: unit for unit in units}
    slots_by_unit: dict[str, list] = {}
    for slot in slots:
        slots_by_unit.setdefault(slot["unit_id"], []).append(slot)
    pilot_slots = [dict(slot) for slot in slots[:PILOT_SLOTS]]

    step_counts = [192] * 7507 + [188] * 1213
    assert len(step_counts) == ORIGINAL_COMPLETED
    assert sum(step_counts) == ORIGINAL_COMPLETED_EXACT
    assert all(step % 4 == 0 and 120 <= step <= 800 for step in step_counts)

    events = [
        {
            "event": "run_started",
            "permit_sha256": BASE_PERMIT_SHA256,
            "core_plan_id": CORE_PLAN_ID,
            "ledger_id": LEDGER_ID,
            "device": "cuda",
            "new_units": TOTAL_UNITS - PILOT_UNITS,
            "new_fits": SOURCE_SLOTS - PILOT_SLOTS,
        }
    ]
    completed_slots = []
    started_slots = []
    sealed_units = []
    started_count = 0
    completed_count = 0
    total_updates = 0
    stop = False
    for unit in units[PILOT_UNITS:]:
        if stop:
            break
        unit_id = unit["unit_id"]
        events.append({"event": "unit_started", "unit_id": unit_id, "slot_count": SLOTS_PER_UNIT})
        broke = False
        for slot in slots_by_unit[unit_id]:
            started_count += 1
            execution_id = (
                f"{unit['station']}-{slot['recipe_id']}-{slot['seed']}-{slot['slot_id'][:16]}"
            )
            events.append(
                {
                    "event": "started",
                    "execution_id": execution_id,
                    "slot_id": slot["slot_id"],
                    "unit_id": unit_id,
                    "recipe_id": slot["recipe_id"],
                    "seed": slot["seed"],
                    "used_fit_count": started_count,
                }
            )
            started_slots.append(slot)
            if started_count == ORIGINAL_STARTED:
                stop = True
                broke = True
                break
            updates = step_counts[completed_count]
            events.append(
                {
                    "event": "completed",
                    "execution_id": execution_id,
                    "slot_id": slot["slot_id"],
                    "status": "complete",
                    "optimizer_steps": updates,
                }
            )
            completed_count += 1
            total_updates += updates
            completed_slots.append(slot)
        if broke:
            break
        events.append(
            {
                "event": "unit_completed",
                "unit_id": unit_id,
                "completed": completed_count,
                "started": started_count,
                "optimizer_steps": total_updates,
            }
        )
        sealed_units.append(unit_id)

    assert started_count == ORIGINAL_STARTED
    assert completed_count == ORIGINAL_COMPLETED
    assert total_updates == ORIGINAL_COMPLETED_EXACT
    assert len(sealed_units) == 726

    interrupted_slot = started_slots[-1]
    interrupted_unit_id = interrupted_slot["unit_id"]
    started_ids = {slot["slot_id"] for slot in started_slots}
    pilot_ids = {slot["slot_id"] for slot in pilot_slots}
    unstarted_slots = [
        slot
        for slot in slots
        if slot["slot_id"] not in started_ids and slot["slot_id"] not in pilot_ids
    ]
    assert len(unstarted_slots) == SOURCE_SLOTS - PILOT_SLOTS - ORIGINAL_STARTED

    selector_records = []
    for slot in pilot_slots:
        selector_records.append(_selector(slot, unit_by_id[slot["unit_id"]]))
    for slot in completed_slots:
        selector_records.append(_selector(slot, unit_by_id[slot["unit_id"]]))

    leases = []
    for slot in pilot_slots:
        leases.append(_lease(slot, PILOT_PERMIT_SHA256))
    for slot in started_slots:
        leases.append(_lease(slot, BASE_PERMIT_SHA256))

    interrupted_history = [
        {
            "epoch": epoch,
            "epoch_optimizer_steps": 4,
            "total_optimizer_steps": 4 * epoch,
            "total_loss": 1.0,
        }
        for epoch in range(1, 18)
    ]

    expected = {
        "reused_pilot_slot_ids": [slot["slot_id"] for slot in pilot_slots],
        "reused_original_slot_ids": [slot["slot_id"] for slot in completed_slots],
        "interrupted_slot_id": interrupted_slot["slot_id"],
        "unstarted_slot_ids": [slot["slot_id"] for slot in unstarted_slots],
        "recovery_slot_ids": [interrupted_slot["slot_id"]]
        + [slot["slot_id"] for slot in unstarted_slots],
        "incomplete_unit_id": interrupted_unit_id,
        "sealed_unit_ids": sealed_units,
        "counts": {
            "source_slots": SOURCE_SLOTS,
            "pilot_reused": PILOT_SLOTS,
            "original_started": ORIGINAL_STARTED,
            "original_completed": ORIGINAL_COMPLETED,
            "original_interrupted": 1,
            "unstarted": len(unstarted_slots),
            "recovery_fits": len(unstarted_slots) + 1,
            "final_new_source_successes": SOURCE_SLOTS - PILOT_SLOTS,
            "final_new_source_attempts": SOURCE_SLOTS - PILOT_SLOTS + 1,
        },
        "optimizer_updates": {
            "original_completed_exact": ORIGINAL_COMPLETED_EXACT,
            "interrupted_observed_lower_bound": INTERRUPTED_UPDATES,
            "interrupted_charged_upper_bound": INTERRUPTED_CHARGED_UPPER,
            "original_charged_upper_bound": ORIGINAL_CHARGED_UPPER,
        },
    }
    return {
        "ledger": ledger,
        "pilot_slots": pilot_slots,
        "events": events,
        "selector_records": selector_records,
        "leases": leases,
        "interrupted_history": interrupted_history,
        "expected": expected,
    }


_BASELINE = None


def _baseline():
    global _BASELINE
    if _BASELINE is None:
        _BASELINE = _build_baseline()
    return _BASELINE


def _fresh():
    return copy.deepcopy(_baseline())


def _call(data):
    return build_recovery_plan(**{key: data[key] for key in INPUT_KEYS})


def _assert_error(data):
    with pytest.raises(RecoveryPlanError) as excinfo:
        _call(data)
    code = getattr(excinfo.value, "reason_code", None)
    assert isinstance(code, str) and code
    assert "/" not in code and "\\" not in code


def _mutate(mutator):
    data = _fresh()
    mutator(data)
    return data


def _assert_subset(observed, expected):
    assert isinstance(observed, dict)
    for key, value in expected.items():
        assert observed[key] == value


def _first_index(events, kind):
    for index, event in enumerate(events):
        if event.get("event") == kind:
            return index
    raise AssertionError("missing event " + kind)


def test_success_plan_is_exact():
    data = _fresh()
    plan = _call(data)
    expected = data["expected"]
    assert plan["schema_version"] == "nato-sers-p05-recovery-plan-v1"
    for key in (
        "reused_pilot_slot_ids",
        "reused_original_slot_ids",
        "unstarted_slot_ids",
        "recovery_slot_ids",
        "sealed_unit_ids",
    ):
        assert list(plan[key]) == expected[key]
    assert plan["interrupted_slot_id"] == expected["interrupted_slot_id"]
    assert plan["incomplete_unit_id"] == expected["incomplete_unit_id"]
    _assert_subset(plan["counts"], expected["counts"])
    _assert_subset(plan["optimizer_updates"], expected["optimizer_updates"])
    assert plan["execution_authorized"] is False
    assert plan["fits_started"] == 0


def test_unit_partition_and_partial_unit_shape():
    data = _fresh()
    plan = _call(data)
    assert len(list(plan["reused_pilot_slot_ids"])) == 36
    assert len(list(plan["reused_original_slot_ids"])) == 8720
    assert len(list(plan["sealed_unit_ids"])) == 726
    assert list(plan["sealed_unit_ids"]) == data["expected"]["sealed_unit_ids"]
    assert plan["recovery_slot_ids"][0] == plan["interrupted_slot_id"]
    assert plan["interrupted_slot_id"].startswith("u0729")
    recovery = list(plan["recovery_slot_ids"])
    assert sum(1 for slot_id in recovery if slot_id.startswith("u0729")) == 4
    assert len(list(plan["unstarted_slot_ids"])) == 6183
    assert len(recovery) == 6184


def test_deterministic_and_inputs_unmodified():
    first = _fresh()
    second = _fresh()
    plan_a = _call(first)
    plan_b = _call(second)
    assert plan_a == plan_b
    assert plan_a["plan_id"] == plan_b["plan_id"]
    reference = _baseline()
    for key in INPUT_KEYS:
        assert first[key] == reference[key]
        assert second[key] == reference[key]


def test_evidence_and_plan_hashes_independently_recomputed():
    data = _fresh()
    plan = _call(data)
    evidence = plan["evidence_sha256"]
    for key in INPUT_KEYS:
        assert key in evidence
        assert evidence[key] == _canon_sha(data[key])
        assert isinstance(evidence[key], str) and len(evidence[key]) == 64
    without_plan_id = {k: v for k, v in plan.items() if k != "plan_id"}
    assert plan["plan_id"] == _canon_sha(without_plan_id)
    assert len(plan["plan_id"]) == 64


def test_frozen_pins_present_somewhere():
    plan = _call(_fresh())
    found = set()

    def walk(value):
        if isinstance(value, str):
            found.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(plan)
    for pin in (
        PILOT_PERMIT_SHA256,
        BASE_PERMIT_SHA256,
        CORE_CONTRACT_SHA256,
        CORE_PLAN_ID,
        LEDGER_ID,
    ):
        assert pin in found


def test_module_is_pure_metadata_no_heavy_imports():
    module = importlib.import_module("atlas_sers.evaluation.p05_recovery_plan")
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"torch", "numpy", "io", "os", "pathlib"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    assert not (found & forbidden)


def test_recovery_error_contract():
    assert issubclass(RecoveryPlanError, ValueError)
    assert not issubclass(RecoveryPlanError, (KeyError, TypeError))


def m_ledger_dup_unit(data):
    data["ledger"]["units"].append(copy.deepcopy(data["ledger"]["units"][5]))


def m_ledger_dup_slot(data):
    data["ledger"]["slots"].append(copy.deepcopy(data["ledger"]["slots"][5]))


def m_ledger_unknown_unit(data):
    extra = copy.deepcopy(data["ledger"]["slots"][40])
    extra["slot_id"] = "u9999r0s0"
    extra["unit_id"] = "unit-9999"
    data["ledger"]["slots"].append(extra)


def m_ledger_bad_recipe(data):
    data["ledger"]["slots"][20]["recipe_id"] = "DX"


def m_ledger_bad_seed(data):
    data["ledger"]["slots"][20]["seed"] = 1


def m_ledger_role_mismatch(data):
    data["ledger"]["slots"][20]["fitting_role_id"] = "nope"


def m_ledger_excluded(data):
    data["ledger"]["slots"][20]["excluded_by_protocol"] = True


LEDGER_CASES = [
    ("duplicate_unit", m_ledger_dup_unit),
    ("duplicate_slot", m_ledger_dup_slot),
    ("unknown_unit", m_ledger_unknown_unit),
    ("broken_recipe", m_ledger_bad_recipe),
    ("broken_seed", m_ledger_bad_seed),
    ("role_mismatch", m_ledger_role_mismatch),
    ("nonfalse_excluded", m_ledger_excluded),
]


def m_pilot_partial(data):
    data["pilot_slots"] = data["pilot_slots"][:-1]


def m_pilot_duplicate(data):
    data["pilot_slots"].append(copy.deepcopy(data["pilot_slots"][0]))


def m_pilot_wrong_product(data):
    data["pilot_slots"][0]["recipe_id"] = "DX"


def m_pilot_unknown(data):
    data["pilot_slots"][0] = copy.deepcopy(data["ledger"]["slots"][100])


PILOT_CASES = [
    ("partial", m_pilot_partial),
    ("duplicate", m_pilot_duplicate),
    ("wrong_product", m_pilot_wrong_product),
    ("unknown_slot", m_pilot_unknown),
]


def m_events_swap(data):
    events = data["events"]
    index = _first_index(events, "started")
    events[index], events[index + 1] = events[index + 1], events[index]


def m_events_gap(data):
    events = data["events"]
    indices = [i for i, e in enumerate(events) if e.get("event") == "completed"]
    del events[indices[len(indices) // 2]]


def m_events_duplicate(data):
    events = data["events"]
    index = _first_index(events, "started")
    events.insert(index + 1, copy.deepcopy(events[index]))


def m_events_extra_start(data):
    data["events"].append(
        {
            "event": "started",
            "execution_id": "cwa-D3-20260829-u9999r3s2",
            "slot_id": "u9999r3s2",
            "unit_id": "unit-9999",
            "recipe_id": "D3",
            "seed": 20260829,
            "used_fit_count": ORIGINAL_STARTED + 1,
        }
    )


def m_events_extra_completed(data):
    events = data["events"]
    interrupted = data["expected"]["interrupted_slot_id"]
    original = next(
        e for e in events if e.get("event") == "started" and e.get("slot_id") == interrupted
    )
    extra = copy.deepcopy(original)
    extra["event"] = "completed"
    extra["status"] = "complete"
    extra["optimizer_steps"] = 4
    events.append(extra)


def m_events_unit_counter(data):
    events = data["events"]
    events[_first_index(events, "unit_completed")]["completed"] = 11


def m_events_wrong_totals(data):
    events = data["events"]
    index = _first_index(events, "unit_completed")
    events[index]["optimizer_steps"] = events[index]["optimizer_steps"] + 1


def m_events_bad_exec(data):
    events = data["events"]
    events[_first_index(events, "started")]["execution_id"] = "bad-id"


def m_events_bad_recipe(data):
    events = data["events"]
    events[_first_index(events, "started")]["recipe_id"] = "D9"


def m_events_bad_seed(data):
    events = data["events"]
    events[_first_index(events, "started")]["seed"] = 1


def m_events_base_pin(data):
    events = data["events"]
    events[_first_index(events, "run_started")]["permit_sha256"] = PILOT_PERMIT_SHA256


def m_events_status(data):
    events = data["events"]
    events[_first_index(events, "completed")]["status"] = "failed"


def m_events_bad_steps(data):
    events = data["events"]
    events[_first_index(events, "completed")]["optimizer_steps"] = 1000


def m_events_nondivisible_steps(data):
    events = data["events"]
    events[_first_index(events, "completed")]["optimizer_steps"] = 190


def m_events_bool_steps(data):
    events = data["events"]
    events[_first_index(events, "completed")]["optimizer_steps"] = True


EVENT_CASES = [
    ("wrong_order", m_events_swap),
    ("gap", m_events_gap),
    ("duplicate", m_events_duplicate),
    ("extra_start", m_events_extra_start),
    ("extra_completed_replay", m_events_extra_completed),
    ("unit_counter", m_events_unit_counter),
    ("wrong_totals", m_events_wrong_totals),
    ("bad_execution_id", m_events_bad_exec),
    ("bad_recipe", m_events_bad_recipe),
    ("bad_seed", m_events_bad_seed),
    ("bad_base_pin", m_events_base_pin),
    ("status_not_complete", m_events_status),
    ("bad_step_count", m_events_bad_steps),
    ("nondivisible_steps", m_events_nondivisible_steps),
    ("boolean_steps", m_events_bool_steps),
]


def m_sel_duplicate(data):
    data["selector_records"].append(copy.deepcopy(data["selector_records"][0]))


def m_sel_reorder(data):
    records = data["selector_records"]
    records[0], records[1] = records[1], records[0]


def m_sel_unknown(data):
    data["selector_records"][0]["slot_id"] = "unknown-slot"


def m_sel_missing(data):
    data["selector_records"].pop()


def m_sel_identity(data):
    data["selector_records"][0]["context_id"] = "wrong-context"


def m_sel_nonfinite(data):
    data["selector_records"][0]["best_validation_nll"] = float("nan")


def m_sel_range_balanced(data):
    data["selector_records"][0]["best_validation_balanced_accuracy"] = 1.5


def m_sel_range_nll(data):
    data["selector_records"][0]["best_validation_nll"] = -1.0


def m_sel_range_count(data):
    data["selector_records"][0]["best_validation_predicted_class_count"] = 0


SELECTOR_CASES = [
    ("duplicate", m_sel_duplicate),
    ("reordered", m_sel_reorder),
    ("unknown", m_sel_unknown),
    ("missing", m_sel_missing),
    ("wrong_identity", m_sel_identity),
    ("nonfinite_metric", m_sel_nonfinite),
    ("range_balanced", m_sel_range_balanced),
    ("range_nll", m_sel_range_nll),
    ("range_class_count", m_sel_range_count),
]


def m_lease_duplicate(data):
    data["leases"].append(copy.deepcopy(data["leases"][0]))


def m_lease_missing(data):
    data["leases"].pop()


def m_lease_unknown(data):
    data["leases"][0]["slot_id"] = "unknown-slot"


def m_lease_unstarted(data):
    slot = data["ledger"]["slots"][-1]
    data["leases"].append(
        {
            "slot_id": slot["slot_id"],
            "unit_id": slot["unit_id"],
            "recipe_id": slot["recipe_id"],
            "seed": slot["seed"],
            "contract_sha256": CORE_CONTRACT_SHA256,
            "core_plan_id": CORE_PLAN_ID,
            "permit_sha256": BASE_PERMIT_SHA256,
        }
    )


def m_lease_pilot_permit(data):
    data["leases"][0]["permit_sha256"] = BASE_PERMIT_SHA256


def m_lease_base_permit(data):
    for lease in data["leases"]:
        if lease["permit_sha256"] == BASE_PERMIT_SHA256:
            lease["permit_sha256"] = PILOT_PERMIT_SHA256
            return
    raise AssertionError("no base lease")


def m_lease_seed_bool(data):
    data["leases"][0]["seed"] = True


LEASE_CASES = [
    ("duplicate", m_lease_duplicate),
    ("missing", m_lease_missing),
    ("unknown", m_lease_unknown),
    ("unstarted", m_lease_unstarted),
    ("wrong_pilot_permit", m_lease_pilot_permit),
    ("wrong_base_permit", m_lease_base_permit),
    ("seed_boolean", m_lease_seed_bool),
]


def m_hist_duplicate(data):
    data["interrupted_history"].append(copy.deepcopy(data["interrupted_history"][-1]))


def m_hist_gap(data):
    del data["interrupted_history"][5]


def m_hist_count(data):
    data["interrupted_history"].pop()


def m_hist_totals(data):
    data["interrupted_history"][3]["total_optimizer_steps"] = 999


def m_hist_missteps(data):
    data["interrupted_history"][3]["epoch_optimizer_steps"] = 3


def m_hist_nonfinite(data):
    data["interrupted_history"][3]["total_loss"] = float("inf")


HISTORY_CASES = [
    ("duplicate_epoch", m_hist_duplicate),
    ("epoch_gap", m_hist_gap),
    ("epoch_count", m_hist_count),
    ("wrong_totals", m_hist_totals),
    ("wrong_epoch_steps", m_hist_missteps),
    ("nonfinite_loss", m_hist_nonfinite),
]


def _replace(key, value):
    def mutator(data):
        data[key] = value

    return mutator


MALFORMED_CASES = [
    ("ledger", _replace("ledger", [])),
    ("pilot_slots", _replace("pilot_slots", {})),
    ("events", _replace("events", {})),
    ("selector_records", _replace("selector_records", {})),
    ("leases", _replace("leases", {})),
    ("interrupted_history", _replace("interrupted_history", {})),
]


@pytest.mark.parametrize(
    "mutator", [case[1] for case in LEDGER_CASES], ids=[case[0] for case in LEDGER_CASES]
)
def test_rejects_bad_ledger(mutator):
    _assert_error(_mutate(mutator))


@pytest.mark.parametrize(
    "mutator", [case[1] for case in PILOT_CASES], ids=[case[0] for case in PILOT_CASES]
)
def test_rejects_bad_pilot_slots(mutator):
    _assert_error(_mutate(mutator))


@pytest.mark.parametrize(
    "mutator", [case[1] for case in EVENT_CASES], ids=[case[0] for case in EVENT_CASES]
)
def test_rejects_bad_events(mutator):
    _assert_error(_mutate(mutator))


@pytest.mark.parametrize(
    "mutator",
    [case[1] for case in SELECTOR_CASES],
    ids=[case[0] for case in SELECTOR_CASES],
)
def test_rejects_bad_selectors(mutator):
    _assert_error(_mutate(mutator))


@pytest.mark.parametrize(
    "mutator", [case[1] for case in LEASE_CASES], ids=[case[0] for case in LEASE_CASES]
)
def test_rejects_bad_leases(mutator):
    _assert_error(_mutate(mutator))


@pytest.mark.parametrize(
    "mutator",
    [case[1] for case in HISTORY_CASES],
    ids=[case[0] for case in HISTORY_CASES],
)
def test_rejects_bad_interrupted_history(mutator):
    _assert_error(_mutate(mutator))


@pytest.mark.parametrize(
    "mutator",
    [case[1] for case in MALFORMED_CASES],
    ids=[case[0] for case in MALFORMED_CASES],
)
def test_rejects_malformed_containers(mutator):
    _assert_error(_mutate(mutator))


@pytest.mark.parametrize("value", [20260805.0, "20260805", True])
def test_pilot_seed_requires_integer_type(value):
    data = _fresh()
    data["pilot_slots"][0]["seed"] = value
    _assert_error(data)


def test_pilot_must_equal_full_registered_slot():
    data = _fresh()
    data["pilot_slots"][0]["extra_unregistered_field"] = "changed"
    _assert_error(data)


def test_nonstring_nested_mapping_key_is_rejected():
    data = _fresh()
    data["interrupted_history"][0]["diagnostic"] = {1: 2}
    _assert_error(data)


def test_bad_unicode_maps_to_reason_code():
    data = _fresh()
    data["interrupted_history"][0]["diagnostic"] = "\ud800"
    _assert_error(data)


def test_recursive_input_maps_to_reason_code():
    data = _fresh()
    recursive = []
    recursive.append(recursive)
    data["interrupted_history"][0]["diagnostic"] = recursive
    _assert_error(data)


@pytest.mark.parametrize("field", ["best_validation_nll", "best_validation_balanced_accuracy"])
def test_unrepresentable_metric_maps_to_reason_code(field):
    data = _fresh()
    data["selector_records"][0][field] = 10**400
    _assert_error(data)


@pytest.mark.parametrize("field", ["context_id", "selection_unit_id"])
def test_optional_slot_context_must_match_unit(field):
    data = _fresh()
    data["ledger"]["slots"][-1][field] = "different-unit"
    _assert_error(data)


@pytest.mark.parametrize("key", ["units", "slots"])
def test_same_length_ledger_duplicates_rejected(key):
    data = _fresh()
    data["ledger"][key][-1] = copy.deepcopy(data["ledger"][key][-2])
    _assert_error(data)


@pytest.mark.parametrize("key", ["pilot_slots", "selector_records", "leases"])
def test_same_length_evidence_duplicates_rejected(key):
    data = _fresh()
    data[key][-1] = copy.deepcopy(data[key][-2])
    _assert_error(data)


def test_unicode_evidence_hash_uses_utf8_not_ascii_escapes():
    data = _fresh()
    data["interrupted_history"][0]["diagnostic"] = "\u03bc"
    plan = _call(data)
    assert plan["evidence_sha256"]["interrupted_history"] == _canon_sha(data["interrupted_history"])
