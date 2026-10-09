"""Regressions for the private T360 release metric-table adapter."""

import copy

import pandas as pd
import pytest

from atlas_sers.evaluation import p08_release_metrics as release
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
def tables(small):
    return release.prepare_metric_tables(small)


def test_exact_values_both_estimands_and_models(small, tables):
    assert set(tables) == {
        "model_summary",
        "domain_metrics",
        "confusion",
        "reliability_bins",
        "equal_context_reliability",
        "class_sensitivity",
    }
    for frame in tables.values():
        assert list(frame.columns[:3]) == ["estimand", "policy_id", "endpoint"]
        assert set(frame["estimand"]) == set(analysis.ESTIMANDS)
        assert set(frame["policy_id"]) == set(analysis.POLICIES)
        assert set(frame["endpoint"]) == set(analysis.ENDPOINTS)
        assert set(frame["model_id"]).issubset(set(analysis.MODELS))
    assert set(tables["model_summary"]["model_id"]) == set(analysis.MODELS)

    source = small["metrics"]["pooled_four_fold"]["PP-U-MIN"]["M01"]["model_summary"]
    subset = tables["model_summary"]
    mask = (
        subset["estimand"].eq("pooled_four_fold")
        & subset["policy_id"].eq("PP-U-MIN")
        & subset["endpoint"].eq("M01")
    )
    pd.testing.assert_frame_equal(
        subset.loc[mask, source.columns].reset_index(drop=True),
        source.reset_index(drop=True),
    )

    before = copy.deepcopy(small)
    release.prepare_metric_tables(small)
    for estimand in analysis.ESTIMANDS:
        for policy in analysis.POLICIES:
            for endpoint in analysis.ENDPOINTS:
                for name, original in before["metrics"][estimand][policy][endpoint].items():
                    current = small["metrics"][estimand][policy][endpoint][name]
                    pd.testing.assert_frame_equal(current, original)


def test_absent_class_nans_and_counts(small, tables):
    keys = [
        "estimand",
        "policy_id",
        "endpoint",
        "model_id",
        "domain",
        "station",
        "class_label",
    ]
    pieces = []
    for estimand in analysis.ESTIMANDS:
        for policy in analysis.POLICIES:
            for endpoint in analysis.ENDPOINTS:
                piece = small["metrics"][estimand][policy][endpoint]["class_recall"].copy()
                piece["estimand"] = estimand
                piece["policy_id"] = policy
                piece["endpoint"] = endpoint
                pieces.append(piece)
    source = pd.concat(pieces, ignore_index=True)
    expected = source.groupby(keys, as_index=False).agg(
        planned_contexts=("recall", "size"),
        supported_contexts=("support", lambda series: int(series.gt(0).sum())),
        sum_correct=("correct", "sum"),
        sum_support=("support", "sum"),
        mean_supported_context_recall=("recall", "mean"),
    )
    expected["pooled_repeated_appearance_recall"] = expected["sum_correct"] / expected[
        "sum_support"
    ].where(expected["sum_support"].gt(0))
    columns = [
        "estimand",
        "policy_id",
        "endpoint",
        "model_id",
        "domain",
        "station",
        "class_label",
        "planned_contexts",
        "supported_contexts",
        "sum_correct",
        "sum_support",
        "pooled_repeated_appearance_recall",
        "mean_supported_context_recall",
    ]
    actual = tables["class_sensitivity"].sort_values(keys, kind="stable").reset_index(drop=True)
    expected = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[columns], expected[columns])

    absent = actual["sum_support"].eq(0)
    assert absent.any()
    assert actual.loc[absent, "pooled_repeated_appearance_recall"].isna().all()
    assert actual.loc[absent, "supported_contexts"].eq(0).all()
    assert actual.loc[~absent, "supported_contexts"].ge(1).all()


def test_alias_retention(tables):
    for frame in tables.values():
        ordinary = frame.loc[frame.model_id.eq("D0-M")]
        selected = frame.loc[frame.model_id.eq("P05-SELECTED")]
        assert len(ordinary) == len(selected) > 0
        assert list(frame.columns[:3]) == ["estimand", "policy_id", "endpoint"]
        assert set(frame["estimand"]) == set(analysis.ESTIMANDS)
        assert set(frame["policy_id"]) == set(analysis.POLICIES)
        assert set(frame["endpoint"]) == set(analysis.ENDPOINTS)


def test_deterministic_row_shuffle(small):
    shuffled = copy.deepcopy(small)
    seed = 0
    for estimand in analysis.ESTIMANDS:
        for policy in analysis.POLICIES:
            for endpoint in analysis.ENDPOINTS:
                panel = shuffled["metrics"][estimand][policy][endpoint]
                for name, frame in panel.items():
                    panel[name] = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
                    seed += 1
    baseline = release.prepare_metric_tables(small)
    actual = release.prepare_metric_tables(shuffled)
    for name, frame in baseline.items():
        pd.testing.assert_frame_equal(frame, actual[name])


def test_refuses_extra_private_column(small):
    panel = small["metrics"]["pooled_four_fold"]["PP-U-MIN"]["M01"]
    panel["model_summary"]["context_id"] = "ctx"
    with pytest.raises(ValueError):
        release.prepare_metric_tables(small)


def test_refuses_dict_valued_numeric_field(small):
    panel = small["metrics"]["pooled_four_fold"]["PP-U-MIN"]["M01"]
    frame = panel["model_summary"].copy()
    frame["balanced_accuracy"] = frame["balanced_accuracy"].astype(object)
    frame.iat[0, frame.columns.get_loc("balanced_accuracy")] = {"value": 1.0}
    panel["model_summary"] = frame
    with pytest.raises(TypeError):
        release.prepare_metric_tables(small)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, {"master": "m1"}])
def test_refuses_invalid_required_numeric_scalars(small, value):
    frame = small["metrics"]["equal_context"]["PP-U-MIN"]["M01"]["model_summary"]
    frame["balanced_accuracy"] = frame["balanced_accuracy"].astype(object)
    frame.iat[0, frame.columns.get_loc("balanced_accuracy")] = value
    with pytest.raises((TypeError, ValueError)):
        release.prepare_metric_tables(small)


def test_no_private_identifiers_in_exports(tables):
    forbidden = {"context_id", "master_sample_id", "observation_uid", "unit_id"}
    for frame in tables.values():
        assert not forbidden.intersection(frame.columns)
        for column in frame.columns:
            texts = [value for value in frame[column] if isinstance(value, str)]
            assert not set(GLOBAL_MASTERS).intersection(texts)
