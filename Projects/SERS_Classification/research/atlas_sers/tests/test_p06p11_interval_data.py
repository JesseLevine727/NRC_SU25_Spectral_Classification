import numpy as np
import pandas as pd
import pytest

from atlas_sers.visualization.p06p11_interval_data import build_interval_semantics

EFFECT_ID = "F_P06_effect_intervals"
WEIGHT_ID = "F_P06_weight_sensitivity"

INPUT_COLUMNS = [
    "model_id",
    "reference_model_id",
    "aggregation_id",
    "method",
    "point_estimate",
    "planned_draws",
    "defined_draws",
    "undefined_draws",
    "lower",
    "upper",
    "reason_code",
    "bca_lower",
    "bca_upper",
    "bca_reason_code",
]

SEMANTIC_COLUMNS = [
    "figure_id",
    "panel",
    "label",
    "series",
    "x",
    "y",
    "lower",
    "upper",
    "domain",
    "instrument",
    "complete_contexts",
    "remaining_domains",
    "color",
    "marker",
]

EFFECT_CONTRASTS = [
    ("P05-SELECTED", "C-SELECTED"),
    ("D3", "D0-M"),
    ("P05-SELECTED", "D0-M"),
    ("P05-SELECTED", "C-RANDOM-FOREST"),
    ("P05-SELECTED", "C-EXTRA-TREES"),
    ("D3", "C-RANDOM-FOREST"),
]

WEIGHT_METHODS = ("crossed_weight", "master_weight", "instrument_weight")
ENDPOINTS = ("M01", "M06")


def _record(model, reference, endpoint, method, point, lower, upper):
    return {
        "model_id": model,
        "reference_model_id": reference,
        "aggregation_id": endpoint,
        "method": method,
        "point_estimate": point,
        "planned_draws": 10000,
        "defined_draws": 10000,
        "undefined_draws": 0,
        "lower": lower,
        "upper": upper,
        "reason_code": "ok",
        "bca_lower": np.nan,
        "bca_upper": np.nan,
        "bca_reason_code": None,
    }


def _base_records():
    records = {}
    for endpoint in ENDPOINTS:
        for index, (model, reference) in enumerate(EFFECT_CONTRASTS):
            key = (model, reference, endpoint, "crossed_weight")
            records[key] = _record(
                model,
                reference,
                endpoint,
                "crossed_weight",
                0.01 * (index + 1),
                0.005 * (index + 1),
                0.02 * (index + 1),
            )
        for method in WEIGHT_METHODS:
            if method == "crossed_weight":
                continue
            key = ("P05-SELECTED", "C-SELECTED", endpoint, method)
            records[key] = _record(
                "P05-SELECTED", "C-SELECTED", endpoint, method, 0.01, 0.005, 0.02
            )
    return records


def _table(records=None, extra=()):
    records = _base_records() if records is None else records
    return pd.DataFrame(list(records.values()) + list(extra), columns=INPUT_COLUMNS)


def _tables(records=None, extra=()):
    return {"intervals": _table(records, extra)}


def test_returns_two_interval_specs():
    result = build_interval_semantics(_tables())
    assert set(result) == {EFFECT_ID, WEIGHT_ID}
    for spec in result.values():
        assert spec["kind"] == "interval"
        assert isinstance(spec["title"], str) and spec["title"]
        assert isinstance(spec["caption"], str) and spec["caption"]


def test_semantic_columns_exact_order():
    result = build_interval_semantics(_tables())
    for spec in result.values():
        assert list(spec["semantic"].columns) == SEMANTIC_COLUMNS


def test_row_counts():
    result = build_interval_semantics(_tables())
    assert len(result[EFFECT_ID]["semantic"]) == 12
    assert len(result[WEIGHT_ID]["semantic"]) == 6


def test_values_are_percentages():
    frame = build_interval_semantics(_tables())[EFFECT_ID]["semantic"]
    first = frame.iloc[0]
    assert first["x"] == pytest.approx(1.0)
    assert first["lower"] == pytest.approx(0.5)
    assert first["upper"] == pytest.approx(2.0)
    assert frame.iloc[1]["x"] == pytest.approx(2.0)


def test_panel_and_y_ordering():
    result = build_interval_semantics(_tables())
    effect = result[EFFECT_ID]["semantic"]
    assert list(effect["panel"]) == ["M01"] * 6 + ["M06"] * 6
    for panel in ENDPOINTS:
        assert list(effect[effect["panel"] == panel]["y"]) == [6, 5, 4, 3, 2, 1]
    weight = result[WEIGHT_ID]["semantic"]
    assert list(weight["panel"]) == ["M01"] * 3 + ["M06"] * 3
    for panel in ENDPOINTS:
        assert list(weight[weight["panel"] == panel]["y"]) == [3, 2, 1]


def test_series_colors_markers():
    result = build_interval_semantics(_tables())
    effect = result[EFFECT_ID]["semantic"]
    assert list(effect["series"]) == (
        ["primary"] + ["secondary"] * 5 + ["primary"] + ["secondary"] * 5
    )
    assert effect.iloc[0]["color"] == "#0072B2"
    assert effect.iloc[0]["marker"] == "circle"
    assert set(effect.iloc[1:6]["color"]) == {"#777777"}
    assert set(effect.iloc[1:6]["marker"]) == {"diamond"}
    weight = result[WEIGHT_ID]["semantic"]
    assert list(weight["series"]) == list(WEIGHT_METHODS) * 2


def test_domain_instrument_and_counts():
    result = build_interval_semantics(_tables())
    for spec in result.values():
        frame = spec["semantic"]
        assert set(frame["domain"]) == {""}
        assert set(frame["instrument"]) == {""}
        assert frame["complete_contexts"].isna().all()
        assert frame["remaining_domains"].isna().all()


def test_deterministic_and_input_untouched():
    tables = _tables()
    before = tables["intervals"].copy(deep=True)
    first = build_interval_semantics(tables)
    second = build_interval_semantics(tables)
    pd.testing.assert_frame_equal(first[EFFECT_ID]["semantic"], second[EFFECT_ID]["semantic"])
    pd.testing.assert_frame_equal(first[WEIGHT_ID]["semantic"], second[WEIGHT_ID]["semantic"])
    pd.testing.assert_frame_equal(tables["intervals"], before)


def test_no_input_columns_leak():
    result = build_interval_semantics(_tables())
    leaked = {
        "model_id",
        "reference_model_id",
        "method",
        "point_estimate",
        "planned_draws",
        "defined_draws",
        "undefined_draws",
        "reason_code",
    }
    for spec in result.values():
        assert not leaked & set(spec["semantic"].columns)


def test_missing_row_rejected():
    records = _base_records()
    del records[("D3", "D0-M", "M01", "crossed_weight")]
    with pytest.raises(ValueError):
        build_interval_semantics(_tables(records))


def test_duplicate_row_rejected():
    table = _table()
    table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        build_interval_semantics({"intervals": table})


@pytest.mark.parametrize(
    "changes",
    [
        {"lower": 0.9, "upper": 0.1},
        {"reason_code": "empty_group"},
        {"planned_draws": 9999},
        {"defined_draws": 9999},
        {"undefined_draws": 1},
        {"point_estimate": np.inf},
        {"point_estimate": np.nan},
        {"lower": np.nan},
        {"upper": -np.inf},
        {"point_estimate": True},
        {"lower": True},
        {"planned_draws": False},
        {"lower": "0.1"},
    ],
)
def test_invalid_requested_row_rejected(changes):
    records = _base_records()
    key = ("P05-SELECTED", "C-SELECTED", "M01", "crossed_weight")
    records[key] = {**records[key], **changes}
    with pytest.raises(ValueError):
        build_interval_semantics(_tables(records))


def test_point_outside_interval_allowed():
    records = _base_records()
    key = ("D3", "D0-M", "M01", "crossed_weight")
    records[key] = {**records[key], "point_estimate": 5.0, "lower": 0.1, "upper": 0.2}
    frame = build_interval_semantics(_tables(records))[EFFECT_ID]["semantic"]
    row = frame[(frame["panel"] == "M01") & (frame["label"] == "D3 - matched CNN")].iloc[0]
    assert row["x"] == pytest.approx(500.0)


def test_unrequested_rows_ignored():
    extra = [
        _record("OTHER", "C-SELECTED", "M01", "crossed_weight", 0.5, 0.4, 0.6),
        _record("P05-SELECTED", "C-SELECTED", "M01", "bootstrap", np.nan, np.nan, np.nan),
    ]
    result = build_interval_semantics(_tables(extra=extra))
    assert len(result[EFFECT_ID]["semantic"]) == 12
    assert len(result[WEIGHT_ID]["semantic"]) == 6
    assert "OTHER" not in set(result[EFFECT_ID]["semantic"]["label"])
