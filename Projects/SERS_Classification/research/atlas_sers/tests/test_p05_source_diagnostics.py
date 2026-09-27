"""Producer/consumer tests for atlas_sers.evaluation.p05_source_diagnostics.

Pure CPU. No torch, no model fitting, no private data, no external reads. The
frozen plan is produced by the real
``atlas_sers.evaluation.p05_refit_plan.build_refit_plan`` so its decisions are
genuine ``select_context`` outputs, and the registered slot ledger is handed to
the diagnostics consumer verbatim.
"""

from __future__ import annotations

import pandas as pd
import pytest

from atlas_sers.evaluation.p05_refit_plan import build_refit_plan
from atlas_sers.evaluation.p05_selection import (
    FALLBACK_RECIPE_ID,
    GUARD_SLOT_KIND,
    INHERITED_SLOT_KIND,
    RECIPE_IDS,
    SEEDS,
)
from atlas_sers.evaluation.p05_source_diagnostics import (
    P05_SOURCE_DIAGNOSTIC_AGGREGATIONS,
    P05_SOURCE_DIAGNOSTIC_MODELS,
    P05_SOURCE_DIAGNOSTIC_PHASES,
    P05_SOURCE_DIAGNOSTIC_STATIONS,
    P05_SOURCE_DIAGNOSTIC_TABLE_NAMES,
    build_source_diagnostics,
)

PERMIT = "ab" * 32
DOMAIN = "instrument_holdout"
HELD = "H1"

CONTEXT_META = {
    "M1": {
        "mode": "master_cv",
        "station": P05_SOURCE_DIAGNOSTIC_STATIONS[2],
        "phase": P05_SOURCE_DIAGNOSTIC_PHASES[1],
    },
    "M2": {
        "mode": "inner_master_cv",
        "station": P05_SOURCE_DIAGNOSTIC_STATIONS[0],
        "phase": P05_SOURCE_DIAGNOSTIC_PHASES[0],
    },
    "P1": {
        "mode": "pseudo_domain",
        "station": P05_SOURCE_DIAGNOSTIC_STATIONS[0],
        "phase": P05_SOURCE_DIAGNOSTIC_PHASES[1],
    },
    "P2": {
        "mode": "pseudo_domain",
        "station": P05_SOURCE_DIAGNOSTIC_STATIONS[1],
        "phase": P05_SOURCE_DIAGNOSTIC_PHASES[1],
    },
}


# --------------------------------------------------------------------------- #
# Fixture builders (producer schema: real ledger slots and complete results)
# --------------------------------------------------------------------------- #


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
        for recipe in RECIPE_IDS
        for seed in SEEDS
    ]


def _ledger_unit(context_id, unit_id):
    return {
        "context_id": context_id,
        "selection_unit_id": unit_id,
        "fitting_uids": [f"{context_id}-F1", f"{context_id}-F2"],
        "validation_uids": [f"{context_id}-F3"],
    }


def _role_rows(context_id, role, prefix, masters, targets, instrument):
    return [
        {
            "role_id": f"{context_id}::{role}",
            "context_id": context_id,
            "role": role,
            "observation_uid": f"{context_id}-{prefix}{index}",
            "master_sample_id": master,
            "instrument": instrument,
            "target_analyte": target,
        }
        for index, (master, target) in enumerate(zip(masters, targets, strict=True), start=1)
    ]


def _roles(context_id):
    return _role_rows(
        context_id, "outer_fit", "F", ["M1", "M2", "M3"], ["T1", "T2", "T3"], "I1"
    ) + _role_rows(context_id, "outer_test", "X", ["N1", "N2", "N3"], ["T1", "T2", "T3"], "I1")


def _results(slots, ba_by_unit_recipe, *, epoch=1, guard_epoch=5, predicted=None):
    predicted = predicted or {}
    rows = []
    for slot in slots:
        unit = slot["selection_unit_id"]
        recipe = slot["recipe_id"]
        seed = slot["seed"]
        guard = slot["slot_kind"] == GUARD_SLOT_KIND
        rows.append(
            {
                "slot_id": slot["slot_id"],
                "context_id": slot["context_id"],
                "selection_unit_id": unit,
                "slot_kind": slot["slot_kind"],
                "fitting_role_id": slot["fitting_role_id"],
                "validation_role_id": slot["validation_role_id"],
                "recipe_id": recipe,
                "seed": seed,
                "status": "complete",
                "best_epoch": guard_epoch if guard else epoch,
                "best_validation_balanced_accuracy": ba_by_unit_recipe[(unit, recipe)],
                "best_validation_nll": 0.5,
                "best_validation_macro_f1": 0.5,
                "best_validation_predicted_class_count": predicted.get((unit, recipe, seed), 3),
            }
        )
    return rows


def _bundle(context_id, inherited_units, guard_units, ba, predicted=None):
    meta = CONTEXT_META[context_id]
    slots = []
    units = []
    for unit_id in inherited_units:
        slots += _slots(context_id, unit_id, INHERITED_SLOT_KIND)
        units.append(_ledger_unit(context_id, unit_id))
    for index, unit_id in enumerate(guard_units):
        slots += _slots(context_id, unit_id, GUARD_SLOT_KIND, guard_fold=index)
        units.append(_ledger_unit(context_id, unit_id))
    return {
        "contexts": [
            {
                "context_id": context_id,
                "station": meta["station"],
                "held_instrument": HELD,
                "selection_mode": meta["mode"],
                "phase_gate": meta["phase"],
                "domain": DOMAIN,
            }
        ],
        "roles": _roles(context_id),
        "ledger": {
            "ledger_id": f"{context_id}-ledger",
            "slots": slots,
            "units": units,
        },
        "results": _results(slots, ba, predicted=predicted),
    }


def _p1_ba():
    table = {
        ("P1-U1", "D0-M"): 0.54,
        ("P1-U1", "D1"): 0.56,
        ("P1-U1", "D2"): 0.55,
        ("P1-U1", "D3"): 0.54,
        ("P1-U2", "D0-M"): 0.58,
        ("P1-U2", "D1"): 0.60,
        ("P1-U2", "D2"): 0.59,
        ("P1-U2", "D3"): 0.58,
    }
    for unit_id in ("P1-G0", "P1-G1", "P1-G2"):
        for recipe in RECIPE_IDS:
            table[(unit_id, recipe)] = 0.95
    return table


def _p2_ba():
    table = {
        ("P2-U1", "D0-M"): 0.50,
        ("P2-U1", "D1"): 0.51,
        ("P2-U1", "D2"): 0.50,
        ("P2-U1", "D3"): 0.50,
    }
    for unit_id in ("P2-G0", "P2-G1", "P2-G2"):
        for recipe in RECIPE_IDS:
            table[(unit_id, recipe)] = 0.95
    return table


def _single_ba(unit_id, values):
    return {(unit_id, recipe): value for recipe, value in values.items()}


def _merge(*bundles):
    return {
        "contexts": [c for bundle in bundles for c in bundle["contexts"]],
        "roles": [r for bundle in bundles for r in bundle["roles"]],
        "ledger": {
            "ledger_id": "combined",
            "slots": [s for bundle in bundles for s in bundle["ledger"]["slots"]],
            "units": [u for bundle in bundles for u in bundle["ledger"]["units"]],
        },
        "results": [r for bundle in bundles for r in bundle["results"]],
    }


def _combined_bundle():
    p1 = _bundle(
        "P1",
        ["P1-U1", "P1-U2"],
        ["P1-G0", "P1-G1", "P1-G2"],
        _p1_ba(),
        predicted={("P1-U1", "D3", SEEDS[0]): 1},
    )
    p2 = _bundle("P2", ["P2-U1"], ["P2-G0", "P2-G1", "P2-G2"], _p2_ba())
    m1 = _bundle(
        "M1",
        ["M1-U1"],
        [],
        _single_ba("M1-U1", {"D0-M": 0.40, "D1": 0.41, "D2": 0.42, "D3": 0.43}),
    )
    m2 = _bundle(
        "M2",
        ["M2-U1"],
        [],
        _single_ba("M2-U1", {"D0-M": 0.30, "D1": 0.31, "D2": 0.32, "D3": 0.33}),
    )
    return _merge(p1, p2, m1, m2)


def _plan(bundle):
    return build_refit_plan(
        ledger=bundle["ledger"],
        contexts=bundle["contexts"],
        roles=bundle["roles"],
        results=bundle["results"],
        permit_sha256=PERMIT,
    )


def _public_frame(*, ba=0.70):
    rows = []
    for index, context_id in enumerate(sorted(CONTEXT_META), start=1):
        meta = CONTEXT_META[context_id]
        for model in P05_SOURCE_DIAGNOSTIC_MODELS:
            for aggregation in P05_SOURCE_DIAGNOSTIC_AGGREGATIONS:
                rows.append(
                    {
                        "point_index": index,
                        "station": meta["station"],
                        "phase": meta["phase"],
                        "domain": DOMAIN,
                        "held_instrument": HELD,
                        "model_id": model,
                        "aggregation_id": aggregation,
                        "balanced_accuracy": ba,
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "point_index",
            "station",
            "phase",
            "domain",
            "held_instrument",
            "model_id",
            "aggregation_id",
            "balanced_accuracy",
        ],
    )


def _run(plan, bundle, contexts, *, slots=None, records=None):
    return build_source_diagnostics(
        plan=plan,
        selector_records=bundle["results"] if records is None else records,
        slots=bundle["ledger"]["slots"] if slots is None else slots,
        strategy_contexts=contexts,
    )


def _alias_refit_id(plan, context_id, strategy, seed):
    return next(
        alias["refit_id"]
        for alias in plan["strategy_aliases"]
        if alias["context_id"] == context_id
        and alias["strategy"] == strategy
        and alias["seed"] == seed
    )


def _alias_recipe(plan, context_id, strategy, seed):
    return plan["unique_refits"][_alias_refit_id(plan, context_id, strategy, seed)]["recipe_id"]


def _fit_row(frame, context_id, recipe, slot_kind):
    meta = CONTEXT_META[context_id]
    rows = frame[
        (frame.station == meta["station"])
        & (frame.phase == meta["phase"])
        & (frame.selection_mode == meta["mode"])
        & (frame.recipe == recipe)
        & (frame.slot_kind == slot_kind)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def _all_approx(series, value, tol=1e-9):
    return bool(((series.astype(float) - float(value)).abs() <= tol).all())


@pytest.fixture(scope="module")
def artifacts():
    bundle = _combined_bundle()
    plan = _plan(bundle)
    return bundle, plan, _public_frame()


# --------------------------------------------------------------------------- #
# Table shape, exact aggregates and anonymous allowlists
# --------------------------------------------------------------------------- #


def test_output_tables_and_anonymous_allowlists(artifacts):
    bundle, plan, contexts = artifacts
    out = _run(plan, bundle, contexts)
    assert set(out) == set(P05_SOURCE_DIAGNOSTIC_TABLE_NAMES)
    assert list(out["selection_counts"].columns) == [
        "station",
        "phase",
        "selection_mode",
        "selected_recipe",
        "selection_source",
        "unsupported_transfer_selection",
        "context_count",
    ]
    assert list(out["fit_summary"].columns) == [
        "station",
        "phase",
        "selection_mode",
        "recipe",
        "slot_kind",
        "fit_count",
        "collapsed_fit_count",
        "collapse_fraction",
        "best_epoch_min",
        "best_epoch_median",
        "best_epoch_max",
        "mean_best_validation_balanced_accuracy",
        "mean_best_validation_macro_f1",
        "mean_best_validation_negative_log_likelihood",
    ]
    assert list(out["best_epoch_distribution"].columns) == [
        "station",
        "phase",
        "selection_mode",
        "recipe",
        "slot_kind",
        "best_epoch",
        "fit_count",
    ]
    assert list(out["refit_epochs"].columns) == [
        "station",
        "phase",
        "strategy",
        "recipe",
        "seed",
        "epochs",
        "context_count",
        "alias_contribution_count",
        "unique_refit_count",
        "count_semantics",
    ]
    assert list(out["source_vs_held"].columns) == [
        "point_index",
        "station",
        "phase",
        "domain",
        "held_instrument",
        "model_id",
        "aggregation_id",
        "selection_mode",
        "recipe",
        "source_mean_seed_unit_balanced_accuracy",
        "source_mean_seed_unit_negative_log_likelihood",
        "source_unit_count",
        "guard_unit_count",
        "held_balanced_accuracy",
        "source_evidence_policy",
        "source_transfer_validation_available",
        "source_vs_held_policy",
    ]
    forbidden = (
        "context_id",
        "slot_id",
        "selection_unit_id",
        "refit_id",
        "path",
        "uid",
        "label",
        "prediction",
    )
    for frame in out.values():
        for column in frame.columns:
            lowered = column.lower()
            assert not any(token in lowered for token in forbidden)


def test_selection_counts_exact(artifacts):
    bundle, plan, contexts = artifacts
    frame = _run(plan, bundle, contexts)["selection_counts"]
    observed = {
        (
            row.station,
            row.phase,
            row.selection_mode,
            row.selected_recipe,
            row.selection_source,
            row.unsupported_transfer_selection,
            row.context_count,
        )
        for row in frame.itertuples(index=False)
    }
    expected = {
        (
            CONTEXT_META["P1"]["station"],
            CONTEXT_META["P1"]["phase"],
            "pseudo_domain",
            "D1",
            "candidate",
            False,
            1,
        ),
        (
            CONTEXT_META["P2"]["station"],
            CONTEXT_META["P2"]["phase"],
            "pseudo_domain",
            FALLBACK_RECIPE_ID,
            "fallback",
            False,
            1,
        ),
        (
            CONTEXT_META["M1"]["station"],
            CONTEXT_META["M1"]["phase"],
            "master_cv",
            FALLBACK_RECIPE_ID,
            "unsupported_transfer_selection",
            True,
            1,
        ),
        (
            CONTEXT_META["M2"]["station"],
            CONTEXT_META["M2"]["phase"],
            "inner_master_cv",
            FALLBACK_RECIPE_ID,
            "unsupported_transfer_selection",
            True,
            1,
        ),
    }
    assert observed == expected
    assert int(frame["context_count"].sum()) == 4


def test_fit_summary_exact_counts_means_and_ranges(artifacts):
    bundle, plan, contexts = artifacts
    frame = _run(plan, bundle, contexts)["fit_summary"]

    p1_inherited_d1 = _fit_row(frame, "P1", "D1", INHERITED_SLOT_KIND)
    assert int(p1_inherited_d1.fit_count) == 6
    assert int(p1_inherited_d1.collapsed_fit_count) == 0
    assert p1_inherited_d1.collapse_fraction == 0.0
    assert int(p1_inherited_d1.best_epoch_min) == 1
    assert p1_inherited_d1.best_epoch_median == pytest.approx(1.0)
    assert int(p1_inherited_d1.best_epoch_max) == 1
    assert p1_inherited_d1.mean_best_validation_balanced_accuracy == pytest.approx(0.58)
    assert p1_inherited_d1.mean_best_validation_macro_f1 == pytest.approx(0.5)
    assert p1_inherited_d1.mean_best_validation_negative_log_likelihood == pytest.approx(0.5)

    p1_guard_d1 = _fit_row(frame, "P1", "D1", GUARD_SLOT_KIND)
    assert int(p1_guard_d1.fit_count) == 9
    assert p1_guard_d1.mean_best_validation_balanced_accuracy == pytest.approx(0.95)
    assert int(p1_guard_d1.best_epoch_min) == 5
    assert int(p1_guard_d1.best_epoch_max) == 5

    p1_inherited_d3 = _fit_row(frame, "P1", "D3", INHERITED_SLOT_KIND)
    assert int(p1_inherited_d3.fit_count) == 6
    assert int(p1_inherited_d3.collapsed_fit_count) == 1
    assert p1_inherited_d3.collapse_fraction == pytest.approx(1.0 / 6.0)
    assert p1_inherited_d3.mean_best_validation_balanced_accuracy == pytest.approx(0.56)

    assert _fit_row(frame, "P2", FALLBACK_RECIPE_ID, INHERITED_SLOT_KIND).fit_count == 3
    assert _fit_row(frame, "M1", FALLBACK_RECIPE_ID, INHERITED_SLOT_KIND).fit_count == 3
