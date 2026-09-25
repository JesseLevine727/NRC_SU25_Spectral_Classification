"""Synthetic tests for atlas_sers.evaluation.p05_selection.

Standard-library-only fixtures. No model fitting, no torch, no external reads,
no execution claims. Every fixture is built from the registered slot schema.
"""

from __future__ import annotations

import copy
import json
import random

import pytest

from atlas_sers.evaluation.p05_selection import (
    FALLBACK_RECIPE_ID,
    FIXED_CONTROL_RECIPE_ID,
    RECIPE_IDS,
    SEEDS,
    SelectionError,
    inherit_refit_epochs,
    select_context,
)

INHERITED = "inherited_selection_fit"
GUARD = "guard_selection_fit"
GUARD_REASON = "guard_fold_lacks_three_classes_in_one_or_both_roles"


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #


def _slot(
    context_id,
    unit_id,
    kind,
    recipe,
    seed,
    *,
    guard_fold=None,
    excluded=False,
    exclusion_reason=None,
):
    return {
        "slot_id": f"{unit_id}::{recipe}::{seed}",
        "slot_kind": kind,
        "context_id": context_id,
        "selection_unit_id": unit_id,
        "guard_fold": guard_fold,
        "fitting_role_id": f"{unit_id}::fit",
        "validation_role_id": f"{unit_id}::val",
        "recipe_id": recipe,
        "seed": seed,
        "planned": True,
        "excluded_by_protocol": excluded,
        "exclusion_reason": exclusion_reason,
    }


def _unit(
    context_id,
    unit_id,
    kind,
    *,
    guard_fold=None,
    excluded=False,
    exclusion_reason=None,
):
    return [
        _slot(
            context_id,
            unit_id,
            kind,
            recipe,
            seed,
            guard_fold=guard_fold,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
        )
        for recipe in RECIPE_IDS
        for seed in SEEDS
    ]


def _pseudo_plan(context_id, inherited_units, guard_units):
    slots = []
    for unit_id in inherited_units:
        slots += _unit(context_id, unit_id, INHERITED)
    for index, unit_id in enumerate(guard_units):
        slots += _unit(context_id, unit_id, GUARD, guard_fold=index)
    return slots


def _result(
    slot,
    *,
    status="complete",
    ba=0.5,
    f1=0.5,
    nll=0.5,
    epoch=50,
    predicted=3,
    failure_reason=None,
    exclusion_reason=None,
):
    result = {
        "slot_id": slot["slot_id"],
        "context_id": slot["context_id"],
        "selection_unit_id": slot["selection_unit_id"],
        "slot_kind": slot["slot_kind"],
        "fitting_role_id": slot["fitting_role_id"],
        "validation_role_id": slot["validation_role_id"],
        "recipe_id": slot["recipe_id"],
        "seed": slot["seed"],
        "status": status,
    }
    if status == "complete":
        result["best_epoch"] = epoch
        result["best_validation_balanced_accuracy"] = ba
        result["best_validation_nll"] = nll
        result["best_validation_macro_f1"] = f1
        result["best_validation_predicted_class_count"] = predicted
    elif status == "excluded":
        result["exclusion_reason"] = (
            exclusion_reason if exclusion_reason is not None else slot["exclusion_reason"]
        )
    if failure_reason is not None:
        result["failure_reason"] = failure_reason
    return result


def _ba_map(per_recipe):
    def ba_fn(unit, recipe, seed):
        return per_recipe[recipe]

    return ba_fn


def _complete_results(
    slots,
    ba_fn,
    *,
    f1_fn=None,
    predicted_fn=None,
    epoch_fn=None,
    nll=0.5,
):
    results = []
    for slot in slots:
        unit = slot["selection_unit_id"]
        recipe = slot["recipe_id"]
        seed = slot["seed"]
        results.append(
            _result(
                slot,
                ba=ba_fn(unit, recipe, seed),
                f1=0.5 if f1_fn is None else f1_fn(unit, recipe, seed),
                nll=nll,
                epoch=50 if epoch_fn is None else epoch_fn(unit, recipe, seed),
                predicted=3 if predicted_fn is None else predicted_fn(unit, recipe, seed),
            )
        )
    return results


def _passing_setup(context_id="CTX"):
    slots = _pseudo_plan(context_id, ["U1"], ["G0", "G1", "G2"])
    per_recipe = {"D0-M": 0.56, "D1": 0.58, "D2": 0.57, "D3": 0.56}
    results = _complete_results(slots, _ba_map(per_recipe))
    return context_id, slots, results


def _rank(ba_by_recipe_unit, f1_by_recipe_unit=None):
    context_id = "CTX"
    slots = _pseudo_plan(context_id, ["U1", "U2"], ["G0", "G1", "G2"])

    def ba_fn(unit, recipe, seed):
        if unit.startswith("G"):
            return 0.5
        return ba_by_recipe_unit[(recipe, unit)]

    def f1_fn(unit, recipe, seed):
        if unit.startswith("G"):
            return 0.5
        if f1_by_recipe_unit is None:
            return 0.5
        return f1_by_recipe_unit[(recipe, unit)]

    results = _complete_results(slots, ba_fn, f1_fn=f1_fn)
    return select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )


# --------------------------------------------------------------------------- #
# Exact decision arithmetic
# --------------------------------------------------------------------------- #


def test_exact_threshold_gain_058_minus_056_passes():
    context_id, slots, results = _passing_setup()
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    assert out["selected_recipe_id"] == "D1"
    threshold = out["candidates"]["D1"]["thresholds"]["mean_pseudo_domain_ba_gain"]
    assert threshold["observed"] == 0.02
    assert threshold["threshold"] == 0.02
    assert threshold["passed"] is True


def test_worst_change_is_min_of_unit_means_not_minimum_difference():
    context_id = "CTX"
    slots = _pseudo_plan(context_id, ["U1", "U2"], ["G0", "G1", "G2"])

    def ba_fn(unit, recipe, seed):
        if unit.startswith("G"):
            return 0.5
        table = {
            ("D0-M", "U1"): 0.50,
            ("D0-M", "U2"): 0.80,
            ("D1", "U1"): 0.60,
            ("D1", "U2"): 0.79,
        }
        return table.get((recipe, unit), 0.5)

    results = _complete_results(slots, ba_fn)
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    metrics = out["candidates"]["D1"]["metrics"]
    assert metrics["worst_pseudo_domain_ba_change"] == 0.10
    assert metrics["worst_pseudo_domain_ba"] == 0.60
    minimum_difference = min(0.60 - 0.50, 0.79 - 0.80)
    assert minimum_difference != metrics["worst_pseudo_domain_ba_change"]


def test_fraction_strictly_improved_boundary_60_percent():
    context_id = "CTX"
    inherited = [f"U{i}" for i in range(1, 6)]
    slots = _pseudo_plan(context_id, inherited, ["G0", "G1", "G2"])

    def ba_fn(unit, recipe, seed):
        if unit.startswith("G"):
            return 0.5
        if recipe == "D0-M":
            return 0.50
        if recipe == "D1":
            return 0.53 if unit in {"U1", "U2", "U3"} else 0.49
        return 0.50

    results = _complete_results(slots, ba_fn)
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    threshold = out["candidates"]["D1"]["thresholds"][
        "fraction_pseudo_domains_strictly_improved"
    ]
    assert threshold["observed"] == 0.6
    assert threshold["passed"] is True

    def ba_fn_fail(unit, recipe, seed):
        if unit.startswith("G"):
            return 0.5
        if recipe == "D0-M":
            return 0.50
        if recipe == "D1":
            return 0.53 if unit in {"U1", "U2"} else 0.49
        return 0.50

    out_fail = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=_complete_results(slots, ba_fn_fail),
    )
    threshold_fail = out_fail["candidates"]["D1"]["thresholds"][
        "fraction_pseudo_domains_strictly_improved"
    ]
    assert threshold_fail["observed"] == 0.4
    assert threshold_fail["passed"] is False


def test_collapse_fraction_boundary_5_percent():
    context_id = "CTX"
    inherited = [f"U{i}" for i in range(1, 18)]
    slots = _pseudo_plan(context_id, inherited, ["G0", "G1", "G2"])
    per_recipe = {"D0-M": 0.56, "D1": 0.58, "D2": 0.57, "D3": 0.56}
    collapse_three = {("U1", SEEDS[0]), ("U1", SEEDS[1]), ("U2", SEEDS[0])}

    def predicted_three(unit, recipe, seed):
        if recipe == "D1" and (unit, seed) in collapse_three:
            return 1
        return 3

    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=_complete_results(
            slots, _ba_map(per_recipe), predicted_fn=predicted_three
        ),
    )
    threshold = out["candidates"]["D1"]["thresholds"]["collapse_fraction"]
    assert threshold["observed"] == 0.05
    assert threshold["passed"] is True

    collapse_four = collapse_three | {("U2", SEEDS[1])}

    def predicted_four(unit, recipe, seed):
        if recipe == "D1" and (unit, seed) in collapse_four:
            return 1
        return 3

    out_fail = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=_complete_results(
            slots, _ba_map(per_recipe), predicted_fn=predicted_four
        ),
    )
    threshold_fail = out_fail["candidates"]["D1"]["thresholds"]["collapse_fraction"]
    assert threshold_fail["observed"] > 0.05
    assert threshold_fail["passed"] is False


def test_seed_metrics_averaged_within_unit_before_units():
    context_id = "CTX"
    slots = _pseudo_plan(context_id, ["U1", "U2"], ["G0", "G1", "G2"])

    def ba_fn(unit, recipe, seed):
        if unit.startswith("G"):
            return 0.5
        if recipe == "D0-M":
            return 0.40
        if recipe == "D1":
            if unit == "U1":
                return {SEEDS[0]: 0.90, SEEDS[1]: 0.90, SEEDS[2]: 0.30}[seed]
            return 0.50
        return 0.40

    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=_complete_results(slots, ba_fn),
    )
    metrics = out["candidates"]["D1"]["metrics"]
    assert metrics["mean_pseudo_domain_ba"] == 0.60
    assert metrics["worst_pseudo_domain_ba"] == 0.50


# --------------------------------------------------------------------------- #
# Ranking and fallback
# --------------------------------------------------------------------------- #


def _base_ranking():
    return {
        ("D0-M", "U1"): 0.50,
        ("D0-M", "U2"): 0.50,
        ("D1", "U1"): 0.60,
        ("D1", "U2"): 0.60,
        ("D2", "U1"): 0.58,
        ("D2", "U2"): 0.58,
        ("D3", "U1"): 0.56,
        ("D3", "U2"): 0.56,
    }


def test_ranking_primary_key_mean_ba():
    out = _rank(_base_ranking())
    assert out["selected_recipe_id"] == "D1"
    assert out["candidate_ranking"] == ["D1", "D2", "D3"]


def test_ranking_secondary_key_worst_ba():
    ba = _base_ranking()
    ba[("D1", "U1")] = 0.60
    ba[("D1", "U2")] = 0.56
    ba[("D2", "U1")] = 0.58
    ba[("D2", "U2")] = 0.58
    out = _rank(ba)
    assert out["selected_recipe_id"] == "D2"


def test_ranking_tertiary_key_mean_macro_f1():
    ba = _base_ranking()
    ba[("D1", "U1")] = 0.58
    ba[("D1", "U2")] = 0.58
    ba[("D2", "U1")] = 0.58
    ba[("D2", "U2")] = 0.58
    f1 = {
        ("D0-M", "U1"): 0.5,
        ("D0-M", "U2"): 0.5,
        ("D1", "U1"): 0.5,
        ("D1", "U2"): 0.5,
        ("D2", "U1"): 0.7,
        ("D2", "U2"): 0.7,
        ("D3", "U1"): 0.6,
        ("D3", "U2"): 0.6,
    }
    out = _rank(ba, f1)
    assert out["selected_recipe_id"] == "D2"


def test_ranking_final_tie_falls_back_to_fixed_recipe_order():
    ba = _base_ranking()
    for recipe in ("D1", "D2", "D3"):
        ba[(recipe, "U1")] = 0.58
        ba[(recipe, "U2")] = 0.58
    out = _rank(ba)
    assert out["selected_recipe_id"] == "D1"
    assert out["candidate_ranking"] == ["D1", "D2", "D3"]


def test_no_pass_falls_back_to_d0m_with_usable_policy():
    context_id = "CTX"
    slots = _pseudo_plan(context_id, ["U1"], ["G0", "G1", "G2"])
    per_recipe = {"D0-M": 0.50, "D1": 0.51, "D2": 0.50, "D3": 0.50}
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=_complete_results(slots, _ba_map(per_recipe)),
    )
    assert out["selected_recipe_id"] == FALLBACK_RECIPE_ID
    assert out["selection_source"] == "fallback"
    assert out["candidate_ranking"] == []
    assert out["fallback_policy_usable"] is True
    assert out["ready_for_refit"] is True


def test_d3_fixed_mechanistic_control_identity_reported():
    context_id, slots, results = _passing_setup()
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    assert out["fixed_mechanistic_control_recipe_id"] == FIXED_CONTROL_RECIPE_ID
    assert out["selected_is_fixed_mechanistic_control"] is False


# --------------------------------------------------------------------------- #
# Ready-for-refit gating
# --------------------------------------------------------------------------- #


def test_incomplete_fixed_control_blocks_ready_for_refit():
    context_id, slots, results = _passing_setup()
    target = next(
        s
        for s in slots
        if s["recipe_id"] == "D3"
        and s["selection_unit_id"] == "U1"
        and s["seed"] == SEEDS[0]
    )
    trimmed = [r for r in results if r["slot_id"] != target["slot_id"]]
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=trimmed,
    )
    assert out["selected_recipe_id"] == "D1"
    assert out["fixed_control_evidence_complete"] is False
    assert out["ready_for_refit"] is False


def test_missing_unselected_candidate_blocks_ready_for_refit():
    context_id, slots, results = _passing_setup()
    target = next(
        s
        for s in slots
        if s["recipe_id"] == "D2"
        and s["selection_unit_id"] == "U1"
        and s["seed"] == SEEDS[0]
    )
    trimmed = [r for r in results if r["slot_id"] != target["slot_id"]]
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=trimmed,
    )
    assert out["selected_recipe_id"] == "D1"
    assert out["context_evidence_complete"] is False
    assert out["ready_for_refit"] is False
    assert out["counts"]["missing_fit_count"] == 1
    assert out["candidates"]["D2"]["counts"]["missing"] == 1


def test_baseline_failure_blocks_and_reports_unusable_fallback():
    context_id, slots, results = _passing_setup()
    for result in results:
        if (
            result["recipe_id"] == "D0-M"
            and result["selection_unit_id"] == "U1"
            and result["seed"] == SEEDS[0]
        ):
            result["status"] = "numerical_failure"
            result["failure_reason"] = "diverged"
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    assert out["selected_recipe_id"] == FALLBACK_RECIPE_ID
    assert out["selection_source"] == "fallback"
    assert out["fallback_policy_usable"] is False
    assert out["full_baseline_comparability"] is False
    assert out["ready_for_refit"] is False
    assert out["candidates"]["D1"]["blocking_reasons"]


@pytest.mark.parametrize(
    "status",
    ["failed", "numerical_failure", "resource_failure", "fit_failure", "data_failure"],
)
def test_known_failure_statuses_retained_and_counted(status):
    context_id, slots, results = _passing_setup()
    for result in results:
        if (
            result["recipe_id"] == "D1"
            and result["selection_unit_id"] == "U1"
            and result["seed"] == SEEDS[0]
        ):
            result["status"] = status
            result["failure_reason"] = "boom"
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    candidate = out["candidates"]["D1"]
    assert candidate["counts"]["failed"] == 1
    assert candidate["failure_details"][0]["status"] == status
    assert candidate["failure_details"][0]["reason"] == "boom"
    assert candidate["metrics"] is None
    assert out["selected_recipe_id"] == FALLBACK_RECIPE_ID


def test_excluded_guards_block_g3_but_keep_inherited_fallback_usable():
    context_id = "CTX"
    slots = _unit(context_id, "U1", INHERITED)
    for index, unit_id in enumerate(["G0", "G1", "G2"]):
        slots += _unit(
            context_id,
            unit_id,
            GUARD,
            guard_fold=index,
            excluded=True,
            exclusion_reason=GUARD_REASON,
        )
    per_recipe = {"D0-M": 0.56, "D1": 0.58, "D2": 0.57, "D3": 0.56}
    results = [
        _result(slot, ba=per_recipe[slot["recipe_id"]])
        for slot in slots
        if slot["slot_kind"] == INHERITED
    ]
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    assert out["selected_recipe_id"] == FALLBACK_RECIPE_ID
    assert out["selection_source"] == "fallback"
    assert out["unsupported_transfer_selection"] is False
    assert out["candidate_ranking"] == []
    assert out["fallback_policy_usable"] is True
    assert out["full_baseline_comparability"] is False
    assert out["ready_for_refit"] is True
    assert out["counts"]["excluded_fit_count"] == 3 * len(RECIPE_IDS) * len(SEEDS)


def _excluded_guard_setup():
    context_id = "CTX"
    slots = _unit(context_id, "U1", INHERITED)
    for index, unit_id in enumerate(["G0", "G1", "G2"]):
        slots += _unit(
            context_id,
            unit_id,
            GUARD,
            guard_fold=index,
            excluded=True,
            exclusion_reason=GUARD_REASON,
        )
    per_recipe = {"D0-M": 0.56, "D1": 0.58, "D2": 0.57, "D3": 0.56}
    results = [
        _result(slot, ba=per_recipe[slot["recipe_id"]])
        for slot in slots
        if slot["slot_kind"] == INHERITED
    ]
    return context_id, slots, results


@pytest.mark.parametrize(
    "field,value",
    [
        ("best_epoch", 50),
        ("best_validation_balanced_accuracy", 0.9),
        ("best_validation_nll", 0.1),
        ("best_validation_macro_f1", 0.9),
        ("best_validation_predicted_class_count", 3),
    ],
)
def test_excluded_result_with_scoring_evidence_rejected(field, value):
    context_id, slots, results = _excluded_guard_setup()
    excluded_slot = next(s for s in slots if s["slot_kind"] == GUARD)
    bad = _result(excluded_slot, status="excluded")
    bad[field] = value
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=slots,
            results=results + [bad],
        )


def test_excluded_result_without_scoring_evidence_is_accepted():
    context_id, slots, results = _excluded_guard_setup()
    excluded_slot = next(s for s in slots if s["slot_kind"] == GUARD)
    results = results + [_result(excluded_slot, status="excluded")]
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    assert out["selected_recipe_id"] == FALLBACK_RECIPE_ID
    assert out["fallback_policy_usable"] is True


def test_failed_result_may_retain_diagnostic_metrics():
    context_id, slots, results = _passing_setup()
    for result in results:
        if (
            result["recipe_id"] == "D1"
            and result["selection_unit_id"] == "U1"
            and result["seed"] == SEEDS[0]
        ):
            result["status"] = "numerical_failure"
            result["failure_reason"] = "diverged"
            result["best_epoch"] = 12
            result["best_validation_balanced_accuracy"] = 0.42
            result["best_validation_nll"] = 0.9
            result["best_validation_macro_f1"] = 0.4
            result["best_validation_predicted_class_count"] = 2
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    candidate = out["candidates"]["D1"]
    assert candidate["counts"]["failed"] == 1
    assert candidate["metrics"] is None
    assert candidate["failure_details"][0]["reason"] == "diverged"
    assert out["selected_recipe_id"] == FALLBACK_RECIPE_ID


def test_failed_result_reason_code_preserved_without_failure_reason():
    context_id, slots, results = _passing_setup()
    for result in results:
        if (
            result["recipe_id"] == "D1"
            and result["selection_unit_id"] == "U1"
            and result["seed"] == SEEDS[0]
        ):
            result["status"] = "fit_failure"
            result["reason_code"] = "oom_kernel"
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    candidate = out["candidates"]["D1"]
    assert candidate["counts"]["failed"] == 1
    assert candidate["failure_details"][0]["status"] == "fit_failure"
    assert candidate["failure_details"][0]["reason"] == "oom_kernel"


def test_failure_reason_preferred_over_reason_code():
    context_id, slots, results = _passing_setup()
    for result in results:
        if (
            result["recipe_id"] == "D1"
            and result["selection_unit_id"] == "U1"
            and result["seed"] == SEEDS[0]
        ):
            result["status"] = "data_failure"
            result["failure_reason"] = "explicit"
            result["reason_code"] = "kernel_code"
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    assert out["candidates"]["D1"]["failure_details"][0]["reason"] == "explicit"


# --------------------------------------------------------------------------- #
# Master-only modes
# --------------------------------------------------------------------------- #


def _master_only_setup(mode):
    context_id = "CTX"
    slots = _unit(context_id, "U1", INHERITED)
    per_recipe = {"D0-M": 0.56, "D1": 0.58, "D2": 0.57, "D3": 0.56}
    results = [_result(slot, ba=per_recipe[slot["recipe_id"]]) for slot in slots]
    out = select_context(
        context_id=context_id,
        selection_mode=mode,
        slots=slots,
        results=results,
    )
    return out


def test_master_cv_returns_unsupported_transfer_fallback():
    out = _master_only_setup("master_cv")
    assert out["selection_mode"] == "master_cv"
    assert out["selected_recipe_id"] == FALLBACK_RECIPE_ID
    assert out["unsupported_transfer_selection"] is True
    assert out["selection_source"] == "unsupported_transfer_selection"
    assert out["candidate_ranking"] == []


def test_inner_master_cv_mode_is_preserved():
    out = _master_only_setup("inner_master_cv")
    assert out["selection_mode"] == "inner_master_cv"
    assert out["selected_recipe_id"] == FALLBACK_RECIPE_ID
    assert out["unsupported_transfer_selection"] is True


@pytest.mark.parametrize("mode", ["master_cv", "inner_master_cv"])
def test_master_only_modes_reject_guard_units(mode):
    context_id, slots, results = _passing_setup()
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode=mode,
            slots=slots,
            results=results,
        )


def test_pseudo_domain_requires_inherited_unit():
    context_id = "CTX"
    slots = []
    for index, unit_id in enumerate(["G0", "G1", "G2"]):
        slots += _unit(context_id, unit_id, GUARD, guard_fold=index)
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=slots,
            results=[],
        )


def test_pseudo_domain_requires_exactly_three_guards():
    context_id = "CTX"
    slots = _pseudo_plan(context_id, ["U1"], ["G0", "G1"])
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=slots,
            results=[],
        )


# --------------------------------------------------------------------------- #
# Malformed slots and evidence
# --------------------------------------------------------------------------- #


def test_unknown_selection_mode_rejected():
    context_id, slots, results = _passing_setup()
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="test_scores",
            slots=slots,
            results=results,
        )


def test_float_seed_in_slot_rejected():
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(slots)
    bad[0]["seed"] = float(bad[0]["seed"])
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=bad,
            results=results,
        )


def test_float_seed_in_result_rejected():
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(results)
    bad[0]["seed"] = float(bad[0]["seed"])
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=slots,
            results=bad,
        )


def test_foreign_context_slot_rejected():
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(slots)
    bad[0]["context_id"] = "OTHER"
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=bad,
            results=results,
        )


def test_foreign_context_result_rejected():
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(results)
    bad[0]["context_id"] = "OTHER"
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=slots,
            results=bad,
        )


def test_unknown_result_slot_rejected():
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(results)
    bad[0]["slot_id"] = "NOPE"
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=slots,
            results=bad,
        )


def test_duplicate_result_rejected():
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(results)
    bad.append(copy.deepcopy(bad[0]))
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=slots,
            results=bad,
        )


def test_duplicate_slot_rejected():
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(slots)
    bad.append(copy.deepcopy(bad[0]))
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=bad,
            results=results,
        )


def test_unit_ladder_must_be_exact():
    context_id, slots, results = _passing_setup()
    target = next(
        s
        for s in slots
        if s["selection_unit_id"] == "U1"
        and s["recipe_id"] == "D1"
        and s["seed"] == SEEDS[0]
    )
    bad_slots = [s for s in slots if s["slot_id"] != target["slot_id"]]
    bad_results = [r for r in results if r["slot_id"] != target["slot_id"]]
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=bad_slots,
            results=bad_results,
        )


def test_wrong_role_identity_in_result_rejected():
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(results)
    bad[0]["fitting_role_id"] = "WRONG"
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=slots,
            results=bad,
        )


def test_inconsistent_role_across_unit_ladder_rejected():
    context_id, slots, results = _passing_setup()
    bad_slots = copy.deepcopy(slots)
    target = bad_slots[0]
    target["fitting_role_id"] = "DIFFERENT"
    bad_results = copy.deepcopy(results)
    for result in bad_results:
        if result["slot_id"] == target["slot_id"]:
            result["fitting_role_id"] = "DIFFERENT"
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=bad_slots,
            results=bad_results,
        )


def test_inconsistent_exclusion_across_unit_ladder_rejected():
    context_id, slots, results = _passing_setup()
    bad_slots = copy.deepcopy(slots)
    target = next(s for s in bad_slots if s["selection_unit_id"] == "U1")
    target["excluded_by_protocol"] = True
    target["exclusion_reason"] = "why"
    bad_results = [r for r in results if r["slot_id"] != target["slot_id"]]
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=bad_slots,
            results=bad_results,
        )


def test_excluded_slot_cannot_supply_complete_score():
    context_id = "CTX"
    slots = _unit(context_id, "U1", INHERITED)
    for index, unit_id in enumerate(["G0", "G1", "G2"]):
        slots += _unit(
            context_id,
            unit_id,
            GUARD,
            guard_fold=index,
            excluded=True,
            exclusion_reason=GUARD_REASON,
        )
    results = [
        _result(slot, ba=0.56) for slot in slots if slot["slot_kind"] == INHERITED
    ]
    excluded_slot = next(s for s in slots if s["slot_kind"] == GUARD)
    results.append(_result(excluded_slot, ba=0.99))
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=slots,
            results=results,
        )


def test_nonexcluded_slot_requires_null_exclusion_reason():
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(slots)
    bad[0]["exclusion_reason"] = "surprise"
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=bad,
            results=results,
        )


def test_unplanned_slot_rejected():
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(slots)
    bad[0]["planned"] = False
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=bad,
            results=results,
        )


def test_guard_fold_on_inherited_slot_rejected():
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(slots)
    bad[0]["guard_fold"] = 0
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=bad,
            results=results,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("best_epoch", 0),
        ("best_epoch", 201),
        ("best_epoch", 50.0),
        ("best_validation_balanced_accuracy", 1.2),
        ("best_validation_balanced_accuracy", -0.1),
        ("best_validation_balanced_accuracy", float("nan")),
        ("best_validation_balanced_accuracy", float("inf")),
        ("best_validation_macro_f1", 1.5),
        ("best_validation_nll", -0.1),
        ("best_validation_predicted_class_count", 0),
        ("best_validation_predicted_class_count", 4),
    ],
)
def test_out_of_range_or_nonfinite_complete_metrics_rejected(field, value):
    context_id, slots, results = _passing_setup()
    bad = copy.deepcopy(results)
    for result in bad:
        if (
            result["recipe_id"] == "D1"
            and result["selection_unit_id"] == "U1"
            and result["seed"] == SEEDS[0]
        ):
            result[field] = value
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=slots,
            results=bad,
        )


def test_foreign_excellent_metrics_cannot_influence_selection():
    context_id, slots, results = _passing_setup()
    foreign_slots = _unit("OTHER", "UF", INHERITED)
    foreign_results = [_result(slot, ba=0.999) for slot in foreign_slots]
    with pytest.raises(SelectionError):
        select_context(
            context_id=context_id,
            selection_mode="pseudo_domain",
            slots=slots + foreign_slots,
            results=results + foreign_results,
        )


# --------------------------------------------------------------------------- #
# Invariance, non-mutation, JSON safety
# --------------------------------------------------------------------------- #


def test_select_context_is_non_mutating_and_shuffle_invariant():
    context_id, slots, results = _passing_setup()
    slots_snapshot = copy.deepcopy(slots)
    results_snapshot = copy.deepcopy(results)
    first = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    assert slots == slots_snapshot
    assert results == results_snapshot
    rng = random.Random(0)
    shuffled_slots = copy.deepcopy(slots)
    rng.shuffle(shuffled_slots)
    shuffled_results = copy.deepcopy(results)
    rng.shuffle(shuffled_results)
    second = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=shuffled_slots,
        results=shuffled_results,
    )
    assert first["selected_recipe_id"] == second["selected_recipe_id"]
    assert first["candidate_ranking"] == second["candidate_ranking"]
    assert (
        first["candidates"]["D1"]["metrics"]
        == second["candidates"]["D1"]["metrics"]
    )


def test_select_context_output_is_json_safe():
    context_id, slots, results = _passing_setup()
    out = select_context(
        context_id=context_id,
        selection_mode="pseudo_domain",
        slots=slots,
        results=results,
    )
    round_tripped = json.loads(json.dumps(out))
    assert round_tripped["selected_recipe_id"] == out["selected_recipe_id"]
    assert round_tripped["counts"] == out["counts"]


# --------------------------------------------------------------------------- #
# Inherited refit epochs
# --------------------------------------------------------------------------- #


def test_inherit_refit_epochs_median_rounding_clipping_and_guard_exclusion():
    context_id = "CTX"
    slots = _pseudo_plan(context_id, ["U1", "U2"], ["G0", "G1", "G2"])
    epochs = {
        ("U1", "D1", SEEDS[0]): 40,
        ("U2", "D1", SEEDS[0]): 41,
        ("U1", "D1", SEEDS[1]): 41,
        ("U2", "D1", SEEDS[1]): 42,
        ("U1", "D1", SEEDS[2]): 10,
        ("U2", "D1", SEEDS[2]): 10,
    }
    results = []
    for slot in slots:
        if slot["slot_kind"] == GUARD:
            results.append(_result(slot, ba=0.5, epoch=5))
        else:
            key = (slot["selection_unit_id"], slot["recipe_id"], slot["seed"])
            results.append(_result(slot, ba=0.5, epoch=epochs.get(key, 100)))
    out = inherit_refit_epochs(recipe_id="D1", slots=slots, results=results)
    assert out == {SEEDS[0]: 40, SEEDS[1]: 42, SEEDS[2]: 30}


def test_inherit_refit_epochs_ignores_other_recipes_and_guards():
    context_id = "CTX"
    slots = _pseudo_plan(context_id, ["U1"], ["G0", "G1", "G2"])
    results = []
    for slot in slots:
        if slot["slot_kind"] == GUARD:
            results.append(_result(slot, ba=0.5, epoch=5))
        elif slot["recipe_id"] == "D1":
            results.append(_result(slot, ba=0.5, epoch=90))
        else:
            results.append(_result(slot, ba=0.5, epoch=1))
    out = inherit_refit_epochs(recipe_id="D1", slots=slots, results=results)
    assert out == {seed: 90 for seed in SEEDS}


def test_inherit_refit_epochs_rejects_excluded_inherited_slot():
    context_id = "CTX"
    slots = _unit(context_id, "U1", INHERITED)
    slots += _unit(
        context_id,
        "U2",
        INHERITED,
        excluded=True,
        exclusion_reason="no_usable_source_fit",
    )
    for index, unit_id in enumerate(["G0", "G1", "G2"]):
        slots += _unit(context_id, unit_id, GUARD, guard_fold=index)
    results = [
        _result(slot, ba=0.5)
        for slot in slots
        if slot["slot_kind"] == INHERITED and not slot["excluded_by_protocol"]
    ]
    with pytest.raises(SelectionError):
        inherit_refit_epochs(recipe_id="D1", slots=slots, results=results)


def test_inherit_refit_epochs_rejects_missing_selected_slot():
    context_id = "CTX"
    slots = _pseudo_plan(context_id, ["U1"], ["G0", "G1", "G2"])
    results = [
        _result(slot, ba=0.5)
        for slot in slots
        if not (
            slot["selection_unit_id"] == "U1"
            and slot["recipe_id"] == "D1"
            and slot["seed"] == SEEDS[0]
        )
    ]
    with pytest.raises(SelectionError):
        inherit_refit_epochs(recipe_id="D1", slots=slots, results=results)


def test_inherit_refit_epochs_rejects_multiple_contexts():
    slots = _unit("CTX", "U1", INHERITED) + _unit("OTHER", "U2", INHERITED)
    with pytest.raises(SelectionError):
        inherit_refit_epochs(recipe_id="D1", slots=slots, results=[])


def test_inherit_refit_epochs_rejects_unknown_recipe():
    context_id, slots, results = _passing_setup()
    with pytest.raises(SelectionError):
        inherit_refit_epochs(recipe_id="D9", slots=slots, results=results)
