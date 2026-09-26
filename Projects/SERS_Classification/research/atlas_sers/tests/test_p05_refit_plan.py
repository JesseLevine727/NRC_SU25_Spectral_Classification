"""Synthetic tests for atlas_sers.evaluation.p05_refit_plan.

Standard-library-only fixtures. No scientific data, no model fitting, no
external reads and no execution claims.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from atlas_sers.evaluation.p05_refit_plan import (
    RefitPlanError,
    build_refit_plan,
)

INHERITED = "inherited_selection_fit"
GUARD = "guard_selection_fit"
RECIPES = ("D0-M", "D1", "D2", "D3")
SEEDS = (20260805, 20260817, 20260829)
PERMIT = "ab" * 32

FLAT_BA = {}
D1_BA = {
    ("D0-M", "P1-U1"): 0.50,
    ("D1", "P1-U1"): 0.58,
    ("D2", "P1-U1"): 0.50,
    ("D3", "P1-U1"): 0.50,
}
D3_BA = {
    ("D0-M", "P1-U1"): 0.50,
    ("D1", "P1-U1"): 0.51,
    ("D2", "P1-U1"): 0.51,
    ("D3", "P1-U1"): 0.60,
}
EPOCHS = {
    ("D0-M", SEEDS[0]): 50,
    ("D0-M", SEEDS[1]): 60,
    ("D0-M", SEEDS[2]): 70,
    ("D1", SEEDS[0]): 40,
    ("D1", SEEDS[1]): 150,
    ("D1", SEEDS[2]): 10,
    ("D2", SEEDS[0]): 100,
    ("D2", SEEDS[1]): 100,
    ("D2", SEEDS[2]): 100,
    ("D3", SEEDS[0]): 80,
    ("D3", SEEDS[1]): 90,
    ("D3", SEEDS[2]): 100,
}


def _canon(value):
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _ctx(context_id, mode, held="H1"):
    return {
        "context_id": context_id,
        "station": "S1",
        "held_instrument": held,
        "selection_mode": mode,
        "phase_gate": "held_evaluation",
    }


def _role_rows(context_id, role, prefix, masters, targets, instrument):
    return [
        {
            "role_id": f"{context_id}::{role}",
            "context_id": context_id,
            "role": role,
            "observation_uid": f"{prefix}{index}",
            "master_sample_id": master,
            "instrument": instrument,
            "target_analyte": target,
        }
        for index, (master, target) in enumerate(zip(masters, targets, strict=True), start=1)
    ]


def _slots(context_id, unit_id, kind, guard_fold=None):
    return [
        {
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
            "excluded_by_protocol": False,
            "exclusion_reason": None,
        }
        for recipe in RECIPES
        for seed in SEEDS
    ]


def _ledger_unit(context_id, unit_id):
    return {
        "context_id": context_id,
        "selection_unit_id": unit_id,
        "fitting_uids": ["F1", "F2"],
        "validation_uids": ["F3"],
    }


def _results(slots, ba, epochs, guard_epoch=5):
    rows = []
    for slot in slots:
        guard = slot["slot_kind"] == GUARD
        row = {
            key: slot[key]
            for key in (
                "slot_id",
                "context_id",
                "selection_unit_id",
                "slot_kind",
                "fitting_role_id",
                "validation_role_id",
                "recipe_id",
                "seed",
            )
        }
        row.update(
            status="complete",
            best_epoch=guard_epoch if guard else epochs.get((slot["recipe_id"], slot["seed"]), 100),
            best_validation_balanced_accuracy=ba.get(
                (slot["recipe_id"], slot["selection_unit_id"]), 0.5
            ),
            best_validation_nll=0.5,
            best_validation_macro_f1=0.5,
            best_validation_predicted_class_count=3,
        )
        rows.append(row)
    return rows


def _bundle(context_id, mode, ba, epochs):
    inherited_id = f"{context_id}-U1"
    slots = _slots(context_id, inherited_id, INHERITED)
    units = [_ledger_unit(context_id, inherited_id)]
    if mode == "pseudo_domain":
        for fold in range(3):
            guard_id = f"{context_id}-G{fold}"
            slots += _slots(context_id, guard_id, GUARD, guard_fold=fold)
            units.append(_ledger_unit(context_id, guard_id))
    return {
        "contexts": [_ctx(context_id, mode)],
        "roles": _role_rows(
            context_id,
            "outer_fit",
            "F",
            ["M1", "M2", "M3"],
            ["T1", "T2", "T3"],
            "I1",
        )
        + _role_rows(
            context_id,
            "outer_test",
            "X",
            ["N1", "N2", "N3"],
            ["T1", "T2", "T3"],
            "I1",
        ),
        "ledger": {
            "ledger_id": f"{context_id}-ledger",
            "slots": slots,
            "units": units,
        },
        "results": _results(slots, ba, epochs),
    }


def _merge(*bundles):
    return {
        "contexts": [c for b in bundles for c in b["contexts"]],
        "roles": [r for b in bundles for r in b["roles"]],
        "ledger": {
            "ledger_id": "combined",
            "slots": [s for b in bundles for s in b["ledger"]["slots"]],
            "units": [u for b in bundles for u in b["ledger"]["units"]],
        },
        "results": [r for b in bundles for r in b["results"]],
    }


def _plan(bundle, permit=PERMIT):
    return build_refit_plan(
        ledger=bundle["ledger"],
        contexts=bundle["contexts"],
        roles=bundle["roles"],
        results=bundle["results"],
        permit_sha256=permit,
    )


def _alias(plan, context_id, strategy, seed):
    return next(
        a["refit_id"]
        for a in plan["strategy_aliases"]
        if a["context_id"] == context_id and a["strategy"] == strategy and a["seed"] == seed
    )


def _refit(plan, context_id, strategy, seed):
    return plan["unique_refits"][_alias(plan, context_id, strategy, seed)]


def test_strategy_aliasing_and_unique_refit_counts():
    master = _plan(_bundle("MC", "master_cv", FLAT_BA, EPOCHS))
    assert master["decisions"][0]["selected_recipe_id"] == "D0-M"
    assert master["counts"]["unique_refit_count"] == 6
    assert master["counts"]["strategy_alias_count"] == 9
    for seed in SEEDS:
        assert _alias(master, "MC", "P05-SELECTED", seed) == _alias(master, "MC", "D0-M", seed)
        assert _alias(master, "MC", "P05-SELECTED", seed) != _alias(master, "MC", "D3", seed)

    d1 = _plan(_bundle("P1", "pseudo_domain", D1_BA, EPOCHS))
    assert d1["decisions"][0]["selected_recipe_id"] == "D1"
    assert d1["counts"]["unique_refit_count"] == 9
    assert d1["counts"]["strategy_alias_count"] == 9
    assert len(set(d1["unique_refits"])) == 9

    d3 = _plan(_bundle("P1", "pseudo_domain", D3_BA, EPOCHS))
    assert d3["decisions"][0]["selected_recipe_id"] == "D3"
    assert d3["counts"]["unique_refit_count"] == 6
    for seed in SEEDS:
        assert _alias(d3, "P1", "P05-SELECTED", seed) == _alias(d3, "P1", "D3", seed)
        assert _alias(d3, "P1", "P05-SELECTED", seed) != _alias(d3, "P1", "D0-M", seed)


def test_epochs_and_calibration_are_inherited_only():
    plan = _plan(_bundle("P1", "pseudo_domain", D1_BA, EPOCHS))
    expected = {SEEDS[0]: 40, SEEDS[1]: 150, SEEDS[2]: 30}
    for seed, epoch in expected.items():
        refit = _refit(plan, "P1", "P05-SELECTED", seed)
        assert refit["epochs"] == epoch
        assert refit["calibration_slot_ids"] == [f"P1-U1::D1::{seed}"]
    assert all(r["epochs"] != 5 for r in plan["unique_refits"].values())
    observed = {sid for r in plan["unique_refits"].values() for sid in r["calibration_slot_ids"]}
    assert not any(sid.startswith("P1-G") for sid in observed)


def test_output_integrity_ids_and_canonical_hash():
    bundle = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    plan = _plan(bundle)
    assert _plan(copy.deepcopy(bundle))["plan_id"] == plan["plan_id"]
    assert _canon({k: v for k, v in plan.items() if k != "plan_id"}) == plan["plan_id"]
    refit = _refit(plan, "P1", "P05-SELECTED", SEEDS[0])
    spec = {
        k: refit[k]
        for k in (
            "context_id",
            "fitting_role_id",
            "source_uid_set_sha256",
            "recipe_id",
            "seed",
            "epochs",
            "calibration_slot_ids",
            "permit_sha256",
        )
    }
    assert _canon(spec) == refit["refit_id"]
    assert refit["source_uid_set_sha256"] == _canon(["F1", "F2", "F3"])
    assert refit["fitting_role_id"] == "P1::outer_fit"

    merged = _plan(_merge(bundle, _bundle("MC", "master_cv", FLAT_BA, EPOCHS)))
    assert {d["context_id"] for d in merged["decisions"]} == {"P1", "MC"}
    assert merged["counts"]["strategy_alias_count"] == 18
    pseudo = next(d for d in merged["decisions"] if d["context_id"] == "P1")
    gain = pseudo["candidates"]["D1"]["thresholds"]["mean_pseudo_domain_ba_gain"]
    assert gain["observed"] == 0.08 and gain["passed"] is True


@pytest.mark.parametrize("bad", ["short", "z" * 64, "a" * 63, "A" * 64, ""])
def test_malformed_permit_rejected(bad):
    with pytest.raises(RefitPlanError):
        _plan(_bundle("P1", "pseudo_domain", D1_BA, EPOCHS), permit=bad)


def test_malformed_support_and_uid_digest_rejected():
    bundle = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    bundle["contexts"][0]["outer_fit_uid_sha256"] = _canon(["F1", "F2", "F3"])
    bundle["contexts"][0]["outer_test_uid_sha256"] = _canon(["X1", "X2", "X3"])
    _plan(bundle)
    bundle["contexts"][0]["outer_fit_uid_sha256"] = "0" * 64
    with pytest.raises(RefitPlanError):
        _plan(bundle)

    broken = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    del broken["contexts"][0]["station"]
    with pytest.raises(RefitPlanError):
        _plan(broken)

    broken = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    del broken["roles"][0]["instrument"]
    with pytest.raises(RefitPlanError):
        _plan(broken)

    broken = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    broken["contexts"].append(dict(broken["contexts"][0]))
    with pytest.raises(RefitPlanError):
        _plan(broken)


def test_source_test_overlap_and_held_instrument_rejected():
    bundle = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    for row in bundle["roles"]:
        if row["role"] == "outer_test" and row["observation_uid"] == "X1":
            row["observation_uid"] = "F1"
            row["master_sample_id"] = "M1"
    with pytest.raises(RefitPlanError):
        _plan(bundle)

    bundle = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    for row in bundle["roles"]:
        if row["role"] == "outer_test" and row["observation_uid"] == "X1":
            row["master_sample_id"] = "M1"
    with pytest.raises(RefitPlanError):
        _plan(bundle)

    bundle = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    bundle["contexts"][0]["held_instrument"] = "I1"
    with pytest.raises(RefitPlanError):
        _plan(bundle)


def test_unit_uids_outside_outer_fit_rejected():
    bundle = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    bundle["ledger"]["units"][0]["fitting_uids"] = ["F1", "ZZZ"]
    with pytest.raises(RefitPlanError):
        _plan(bundle)


def test_bad_result_sets_rejected_never_ready():
    bundle = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    bundle["results"] = bundle["results"][1:]
    with pytest.raises(RefitPlanError):
        _plan(bundle)

    bundle = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    bundle["results"].append(copy.deepcopy(bundle["results"][0]))
    with pytest.raises(RefitPlanError):
        _plan(bundle)

    bundle = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    extra = copy.deepcopy(bundle["results"][0])
    extra["slot_id"] = "NOPE"
    bundle["results"].append(extra)
    with pytest.raises(RefitPlanError):
        _plan(bundle)

    bundle = _bundle("P1", "pseudo_domain", D1_BA, EPOCHS)
    bundle["results"][0]["context_id"] = "OTHER"
    with pytest.raises(RefitPlanError):
        _plan(bundle)


@pytest.mark.parametrize("test_class_count", [1, 2])
def test_outer_test_may_have_subset_of_training_classes(test_class_count):
    bundle = _bundle("MC", "master_cv", FLAT_BA, EPOCHS)
    bundle["roles"] = [
        row
        for row in bundle["roles"]
        if row["role"] != "outer_test" or row["target_analyte"] in ["T1", "T2"][:test_class_count]
    ]
    plan = _plan(bundle)
    assert plan["endpoints"][0]["test_classes"] == ["T1", "T2"][:test_class_count]
    assert all(refit["classes"] == ["T1", "T2", "T3"] for refit in plan["unique_refits"].values())
