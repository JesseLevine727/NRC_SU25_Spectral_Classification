"""Boundary and negative-path tests for atlas_sers.evaluation.p05_source_diagnostics.

Pure CPU. Reuses the genuine ``build_refit_plan`` fixture and helpers from
``test_p05_source_diagnostics``; never fits, reads, or selects.
"""

from __future__ import annotations

import copy

import pytest

from atlas_sers.evaluation.p05_selection import (
    INHERITED_SLOT_KIND,
    SEEDS,
)
from atlas_sers.evaluation.p05_source_diagnostics import (
    D3_MODEL_ID,
    P05_SOURCE_DIAGNOSTIC_TABLE_NAMES,
    SELECTED_MODEL_ID,
    SourceDiagnosticsError,
)
from tests import test_p05_source_diagnostics as source_fixtures
from tests.test_p05_source_diagnostics import (
    CONTEXT_META,
    _all_approx,
    _public_frame,
    _run,
)

artifacts = source_fixtures.artifacts


def _mutate_record(records, mode):
    if mode == "drop":
        records.pop(0)
    elif mode == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif mode == "failed":
        records[0]["status"] = "failed"
    elif mode == "wrong_seed":
        records[0]["seed"] = next(s for s in SEEDS if s != records[0]["seed"])
    elif mode == "nan":
        records[0]["best_validation_balanced_accuracy"] = float("nan")
    else:
        records[0]["best_validation_balanced_accuracy"] = True


def _mutate_slots(slots, mode):
    if mode == "missing_field":
        slots[0].pop("recipe_id")
    elif mode == "duplicate":
        slots.append(copy.deepcopy(slots[0]))
    else:
        slots[0]["context_id"] = "NOT-A-CONTEXT"


def _mutate_plan(plan, mode):
    if mode == "master_recipe":
        for decision in plan["decisions"]:
            if decision["selection_mode"] != "pseudo_domain":
                decision["selected_recipe_id"] = "D1"
                return
    elif mode == "pseudo_unsupported":
        for decision in plan["decisions"]:
            if decision["selection_mode"] == "pseudo_domain":
                decision["unsupported_transfer_selection"] = True
                return
    elif mode == "alias_recipe":
        chosen = next(
            alias
            for alias in plan["strategy_aliases"]
            if alias["strategy"] == SELECTED_MODEL_ID
        )
        for alias in plan["strategy_aliases"]:
            same = (
                alias["context_id"] == chosen["context_id"]
                and alias["seed"] == chosen["seed"]
            )
            if alias["strategy"] == D3_MODEL_ID and same:
                alias["refit_id"] = chosen["refit_id"]
                return
    else:
        refits = plan["unique_refits"]
        orphan = copy.deepcopy(next(iter(refits.values())))
        orphan["refit_id"] = "ORPHAN-REFIT"
        refits["ORPHAN-REFIT"] = orphan


def _mutate_public(frame, mode):
    if mode == "phase":
        frame.loc[frame["point_index"] == 1, "phase"] = "development"
    elif mode == "domain":
        frame.loc[frame["point_index"] == 1, "domain"] = "other_domain"
    elif mode == "instrument":
        frame.loc[frame["point_index"] == 1, "held_instrument"] = "H2"
    elif mode == "index":
        frame["point_index"] = frame["point_index"].replace(4, 5)
    else:
        frame.drop(frame.index[0], inplace=True)


@pytest.mark.parametrize(
    ("kind", "mutation", "code"),
    [
        ("record", "drop", "selector_record_coverage_mismatch"),
        ("record", "duplicate", "selector_record_duplicate_slot"),
        ("record", "failed", "selector_record_incomplete"),
        ("record", "wrong_seed", "selector_record_identity_mismatch"),
        ("record", "nan", "selector_record_balanced_accuracy_malformed"),
        ("record", "bool", "selector_record_balanced_accuracy_malformed"),
        ("slot", "missing_field", "slot_field_missing"),
        ("slot", "duplicate", "slot_duplicate"),
        ("slot", "foreign", "slot_unknown_context"),
        ("plan", "master_recipe", "plan_decision_master_cv_inconsistent"),
        ("plan", "pseudo_unsupported", "plan_decision_pseudo_domain_inconsistent"),
        ("plan", "alias_recipe", "plan_strategy_alias_recipe_mismatch"),
        ("plan", "orphan_refit", "plan_unique_refit_unreferenced"),
        ("public", "phase", "public_phase_mismatch"),
        ("public", "domain", "public_domain_mismatch"),
        ("public", "instrument", "public_held_instrument_mismatch"),
        ("public", "index", "point_index_mapping_mismatch"),
        ("public", "endpoint", "strategy_context_endpoint_coverage"),
    ],
)
def test_boundary_matrix(artifacts, kind, mutation, code):
    bundle, plan, contexts = artifacts
    if kind == "record":
        bundle = copy.deepcopy(bundle)
        _mutate_record(bundle["results"], mutation)
    elif kind == "slot":
        bundle = copy.deepcopy(bundle)
        _mutate_slots(bundle["ledger"]["slots"], mutation)
    elif kind == "plan":
        plan = copy.deepcopy(plan)
        _mutate_plan(plan, mutation)
    else:
        contexts = _public_frame()
        _mutate_public(contexts, mutation)
    with pytest.raises(SourceDiagnosticsError) as excinfo:
        _run(plan, bundle, contexts)
    assert excinfo.value.code == code


def test_source_semantics_and_refit_epochs(artifacts):
    bundle, plan, contexts = artifacts
    out = _run(plan, bundle, contexts)
    source = out["source_vs_held"]
    pseudo = source[source["selection_mode"] == "pseudo_domain"]
    master = source[source["selection_mode"] != "pseudo_domain"]
    assert pseudo["source_transfer_validation_available"].all()
    assert not master["source_transfer_validation_available"].any()
    p1 = source[(source["model_id"] == SELECTED_MODEL_ID) & (source["recipe"] == "D1")]
    assert len(p1) == 2
    assert _all_approx(p1["source_mean_seed_unit_balanced_accuracy"], 0.58)
    assert not _all_approx(p1["source_mean_seed_unit_balanced_accuracy"], 0.95)
    fit = out["fit_summary"]
    p1_d3 = fit[
        (fit["station"] == CONTEXT_META["P1"]["station"])
        & (fit["phase"] == CONTEXT_META["P1"]["phase"])
        & (fit["recipe"] == "D3")
        & (fit["slot_kind"] == INHERITED_SLOT_KIND)
    ].iloc[0]
    assert int(p1_d3["collapsed_fit_count"]) == 1
    assert int(p1_d3["best_epoch_min"]) == 1
    refits = out["refit_epochs"]
    assert int(refits["epochs"].min()) >= 30
    assert (refits["alias_contribution_count"] >= refits["unique_refit_count"]).all()


def test_private_sentinels_are_not_exported(artifacts):
    bundle, plan, _ = artifacts
    bundle, plan = copy.deepcopy(bundle), copy.deepcopy(plan)
    plan["PRIVATE_SENTINEL"] = "SENTINEL-VALUE"
    for record in [*plan["decisions"], *bundle["results"], *bundle["ledger"]["slots"]]:
        record["PRIVATE_SENTINEL"] = "SENTINEL-VALUE"
    contexts = _public_frame()
    contexts["PRIVATE_SENTINEL"] = "SENTINEL-VALUE"
    out = _run(plan, bundle, contexts)
    assert "PRIVATE_SENTINEL" not in out
    for frame in out.values():
        assert "SENTINEL-VALUE" not in frame.to_csv(index=False)


def test_diagnostics_never_calls_select_context(artifacts, monkeypatch):
    bundle, plan, contexts = artifacts

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("build_source_diagnostics must not call select_context")

    for target in (
        "atlas_sers.evaluation.p05_selection.select_context",
        "atlas_sers.evaluation.p05_refit_plan.select_context",
    ):
        monkeypatch.setattr(target, _forbidden, raising=False)
    assert set(_run(plan, bundle, contexts)) == set(P05_SOURCE_DIAGNOSTIC_TABLE_NAMES)
