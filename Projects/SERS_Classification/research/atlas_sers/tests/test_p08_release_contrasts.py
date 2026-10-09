"""Release-contrast adapter regressions on the actual accepted numerical API."""

import copy
import json
import math
import pickle

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p08_release_contrasts as release
from atlas_sers.evaluation import p08_universal_analysis as analysis
from atlas_sers.evaluation import p08_universal_units as units
from atlas_sers.visualization import p08_model_figure_data as figures
from tests.test_p08_universal_analysis import (
    GLOBAL_INSTRUMENTS,
    GLOBAL_MASTERS,
    _predictions_frame,
    _registered_frame,
)


@pytest.fixture(scope="module")
def computed():
    registered = _registered_frame()
    panels = units.build_units(_predictions_frame(registered), registered)
    old = analysis.DRAW_COUNT
    analysis.DRAW_COUNT = 4
    try:
        return analysis.analyze_panel(
            panels,
            global_masters=GLOBAL_MASTERS,
            global_instruments=GLOBAL_INSTRUMENTS,
            domain_families={"d1": "unknown"},
        )
    finally:
        analysis.DRAW_COUNT = old


@pytest.fixture()
def small(monkeypatch, computed):
    monkeypatch.setattr(figures, "EXPECTED_DOMAINS", 1)
    return copy.deepcopy(computed)


@pytest.fixture()
def bundle(small):
    return release.prepare_contrast_tables(small)


def _eq(left, right):
    if left is None or right is None:
        left_missing = left is None or (isinstance(left, float) and math.isnan(left))
        right_missing = right is None or (isinstance(right, float) and math.isnan(right))
        return left_missing and right_missing
    return bool(left == right)


def _available_effect():
    return next(
        entry
        for entry in analysis.contrast_registry()
        if entry["model_id"] is not None and entry["available"]
    )


def _available_interaction():
    return next(
        entry
        for entry in analysis.contrast_registry()
        if entry["deep_model_id"] is not None
        and entry["comparator_model_id"] is not None
        and entry["available"]
    )


def test_bundle_layout_summary_columns_and_manifest(bundle):
    assert set(bundle) == {"tables", "diagnostics", "manifest"}
    tables = bundle["tables"]
    assert set(tables) == {
        "contrast_summary",
        "paired_domains",
        "procedure_domains",
    }
    summary = tables["contrast_summary"]
    assert list(summary.columns) == list(analysis._SUMMARY_COLUMNS)
    assert len(summary) == 104
    for estimand in analysis.ESTIMANDS:
        assert int((summary["estimand"] == estimand).sum()) == 52

    manifest = bundle["manifest"]
    for estimand in analysis.ESTIMANDS:
        assert manifest["counts"][estimand] == {
            "registry": 52,
            "available": 44,
            "unavailable": 8,
        }
    assert manifest["external_authentication_verified"] is False
    assert manifest["reviewed"] is False
    assert manifest["published"] is False
    assert manifest["no_G4_decision"] is True
    assert manifest["no_model_or_policy_selection"] is True
    assert manifest["conditional_on_saved_fits_observed_support"] is True


def test_both_estimands_and_unavailable_qc_preserved(bundle):
    diagnostics = bundle["diagnostics"]
    expected_keys = {
        "available",
        "reason",
        "contrast_id",
        "family_id",
        "endpoint",
        "model_id",
        "policy_id",
        "deep_model_id",
        "comparator_model_id",
    }
    for estimand in analysis.ESTIMANDS:
        per_estimand = diagnostics[estimand]
        available = [cid for cid, record in per_estimand.items() if record["available"]]
        unavailable = [cid for cid, record in per_estimand.items() if not record["available"]]
        assert len(available) == 44
        assert len(unavailable) == 8
        for cid in unavailable:
            record = per_estimand[cid]
            assert set(record) == expected_keys
            assert record["reason"] == analysis.QC_REASON


def test_summary_copies_source_values_unchanged(bundle, small):
    contrasts = release._resolve_contrasts(small)
    summary = bundle["tables"]["contrast_summary"]
    pd.testing.assert_frame_equal(summary, small["contrast_summary"], check_exact=True)
    available = {e["contrast_id"] for e in analysis.contrast_registry() if e["available"]}
    checked = 0
    for estimand in analysis.ESTIMANDS:
        for cid, contrast in contrasts[estimand].items():
            if cid not in available:
                continue
            match = summary[(summary["estimand"] == estimand) & (summary["contrast_id"] == cid)]
            assert len(match) == 1
            row = match.iloc[0]
            assert _eq(row["point_effect"], contrast["point"]["overall"])
            for mode in ("crossed", "master_only", "instrument_only"):
                block = contrast["weighted"][mode]["overall"]["summary"]
                assert _eq(row[f"{mode}_lower"], block["lower"])
                assert _eq(row[f"{mode}_upper"], block["upper"])
                assert _eq(row[f"{mode}_reason"], block["reason_code"])
            combined = contrast["hierarchy"]["combined"]["summary"]
            assert _eq(row["hierarchy_planned"], combined["planned_draws"])
            assert _eq(row["hierarchy_defined"], combined["defined_draws"])
            assert _eq(row["hierarchy_undefined"], combined["undefined_draws"])
            adjustment = contrast["sensitivity_adjustment"]
            assert _eq(row["domain_raw_p"], adjustment["domain_raw_p"])
            assert _eq(row["domain_adjusted_p"], adjustment["domain_adjusted_p"])
            assert _eq(row["instrument_raw_p"], adjustment["instrument_raw_p"])
            assert _eq(row["instrument_adjusted_p"], adjustment["instrument_adjusted_p"])
            assert _eq(row["family_size"], adjustment["family_size"])
            assert _eq(row["adjustment"], adjustment["adjustment"])
            checked += 1
    assert checked == 88


def test_procedure_domains_four_terms_and_aliases(bundle, small):
    contrasts = release._resolve_contrasts(small)
    table = bundle["tables"]["procedure_domains"]
    estimand = analysis.ESTIMANDS[0]

    interaction = _available_interaction()
    cid = interaction["contrast_id"]
    contrast = contrasts[estimand][cid]
    rows = table[(table["estimand"] == estimand) & (table["contrast_id"] == cid)]
    domains = contrast["domains"]
    assert len(rows) == len(interaction["procedure_labels"]) * len(domains)
    assert sorted(set(rows["term_index"])) == [0, 1, 2, 3]
    for term_index, procedure_label in enumerate(interaction["procedure_labels"]):
        term = rows[rows["term_index"] == term_index]
        assert (term["term_policy_id"] == procedure_label[0]).all()
        assert (term["term_model_id"] == procedure_label[1]).all()
        assert (term["coefficient"] == interaction["coefficients"][term_index]).all()
        for domain in domains:
            cell = term[term["domain"] == domain].iloc[0]
            assert _eq(
                cell["procedure_overall_balanced_accuracy"],
                contrast["point"]["procedure_effects"][term_index],
            )
            assert _eq(
                cell["procedure_domain_balanced_accuracy"],
                contrast["point"]["procedure_domain_effects"][term_index][domain],
            )

    effect = _available_effect()
    effect_rows = table[
        (table["estimand"] == estimand) & (table["contrast_id"] == effect["contrast_id"])
    ]
    assert sorted(set(effect_rows["term_index"])) == [0, 1]


def test_weighted_modes_and_hierarchy_counts(bundle, small):
    contrasts = release._resolve_contrasts(small)
    diagnostics = bundle["diagnostics"]
    estimand = analysis.ESTIMANDS[0]

    effect = _available_effect()
    cid = effect["contrast_id"]
    record = diagnostics[estimand][cid]
    term_count = len(contrasts[estimand][cid]["point"]["procedure_effects"])
    weighted = record["weighted"]
    assert set(weighted) == {"crossed", "master_only", "instrument_only"}
    for entry in weighted.values():
        assert list(entry["overall"]["summary"]) == list(release._SUMMARY_FIELDS)
        assert len(entry["procedure_overall"]) == term_count
        assert len(entry["procedure_min_ba"]) == term_count
        assert "min_ba_difference" in entry
        assert "minimum_paired_domain_effect" in entry
        for block in (
            [entry["overall"]]
            + entry["procedure_overall"]
            + entry["procedure_min_ba"]
            + [entry["min_ba_difference"], entry["minimum_paired_domain_effect"]]
        ):
            assert set(block) == {"summary"}

    interaction = _available_interaction()
    interaction_weighted = diagnostics[estimand][interaction["contrast_id"]]["weighted"]
    for mode in interaction_weighted:
        assert "min_ba_difference" not in interaction_weighted[mode]
        assert "minimum_paired_domain_effect" not in interaction_weighted[mode]

    hierarchy = record["hierarchy"]
    source = contrasts[estimand][cid]["hierarchy"]
    assert hierarchy["empty_cells_total"] == int(np.asarray(source["empty_cells"]).sum())
    assert hierarchy["undefined_domain_occurrences_total"] == int(
        np.asarray(source["undefined_domain_occurrences"]).sum()
    )
    assert hierarchy["affected_domains_total"] == int(np.asarray(source["affected_domains"]).sum())
    assert len(hierarchy["term_summaries"]) == term_count
    assert set(hierarchy["combined"]) == {"summary"}


def test_all_ties_and_none_values_preserved(bundle, small):
    contrasts = release._resolve_contrasts(small)
    estimand = analysis.ESTIMANDS[0]
    effect = _available_effect()
    interaction = _available_interaction()
    for entry in (effect, interaction):
        cid = entry["contrast_id"]
        source = contrasts[estimand][cid]["descriptive"]
        record = bundle["diagnostics"][estimand][cid]["descriptive"]
        assert record["procedure_min_ties"] == source["procedure_min_ties"]
        assert record["leave_one_domain"] == source["leave_one_domain"]
        assert record["leave_one_instrument"] == source["leave_one_instrument"]
        assert record["leave_one_family"] == source["leave_one_family"]
    effect_record = bundle["diagnostics"][estimand][effect["contrast_id"]]["descriptive"]
    assert "minimum_paired_effect_ties" in effect_record
    interaction_record = bundle["diagnostics"][estimand][interaction["contrast_id"]]["descriptive"]
    assert "minimum_paired_effect_ties" not in interaction_record
    assert list(effect_record["leave_one_domain"].values()) == [None]


def test_input_is_not_mutated(small):
    pristine = copy.deepcopy(small)
    release.prepare_contrast_tables(small)
    assert pickle.dumps(small) == pickle.dumps(pristine)


def test_forbidden_keys_and_fixture_master_ids_absent(bundle, small):
    diagnostics = bundle["diagnostics"]
    text = json.dumps(diagnostics, sort_keys=True)
    forbidden = {
        "masters",
        "global_masters",
        "context_id",
        "unit_id",
        "observation_uid",
        "sample_ids",
        "master_sample_id",
        "weights",
        "draws",
        "raw_predictions",
        "predictions",
        "cell_keys",
        "seed",
        "job",
    }
    for token in forbidden:
        assert f'"{token}"' not in text

    master_ids = set()
    if isinstance(GLOBAL_MASTERS, (list, tuple)):
        master_ids.update(str(value) for value in GLOBAL_MASTERS)
    contrasts = release._resolve_contrasts(small)
    for estimand in analysis.ESTIMANDS:
        for contrast in contrasts[estimand].values():
            for value in contrast.get("masters", []) or []:
                master_ids.add(str(value))
    for master_id in master_ids:
        assert json.dumps(master_id) not in text
    for frame in bundle["tables"].values():
        assert not forbidden.intersection(frame.columns)
        for column in frame.columns:
            values = [value for value in frame[column] if isinstance(value, str)]
            assert not master_ids.intersection(values)


def test_malformed_numeric_payload_rejected(small):
    for bad in ({"value": 1}, [1.0], float("nan"), float("inf"), "text", True):
        with pytest.raises((TypeError, ValueError)):
            release._number(bad, "field")

    corrupt = copy.deepcopy(small)
    contrasts = release._resolve_contrasts(corrupt)
    estimand = analysis.ESTIMANDS[0]
    cid = _available_effect()["contrast_id"]
    contrasts[estimand][cid]["point"]["overall"] = {"not": "a number"}
    with pytest.raises((TypeError, ValueError)):
        release.prepare_contrast_tables(corrupt)


def test_summary_disagreement_or_extra_fields_refused(small):
    corrupt = copy.deepcopy(small)
    corrupt["contrast_summary"].loc[0, "point_effect"] += 0.01
    with pytest.raises(AssertionError):
        release.prepare_contrast_tables(corrupt)
    small["contrast_summary"]["master_sample_id"] = "m1"
    with pytest.raises(ValueError, match="summary schema"):
        release.prepare_contrast_tables(small)


def test_paired_domains_use_recorded_instrument_mapping(bundle, small):
    for row in bundle["tables"]["paired_domains"].itertuples(index=False):
        point = small["contrasts"][row.estimand][row.contrast_id]["point"]
        assert row.instrument == point["domain_instrument"][row.domain]
        assert row.point_effect == point["domain_effects"][row.domain]


def test_registry_and_boundary_refused(small):
    corrupt = copy.deepcopy(small)
    corrupt["registry"] = corrupt["registry"][:-1]
    with pytest.raises(ValueError):
        release.prepare_contrast_tables(corrupt)
    small["boundary"]["no_G4_decision"] = False
    with pytest.raises(ValueError):
        release.prepare_contrast_tables(small)
