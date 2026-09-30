import pandas as pd
import pytest

from atlas_sers.visualization.p06p11_figure_data import FIGURE_ID, build_semantics

COLUMNS = [
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
COLORS = {"cwa": "#0072B2", "pills": "#D55E00", "surfaces": "#009E73"}
MARKERS = {"cwa": "circle", "pills": "square", "surfaces": "triangle-up"}


def _row(endpoint, station, model_ba, reference_ba):
    return {
        "model_id": "P05-SELECTED",
        "reference_model_id": "C-SELECTED",
        "aggregation_id": endpoint,
        "domain": f"{station}-ir",
        "station": station,
        "held_instrument": f"{station}-inst",
        "planned_contexts": 20,
        "complete_contexts": 20,
        "model_ba": model_ba,
        "reference_ba": reference_ba,
        "delta": model_ba - reference_ba,
        "private_note": "keep-out",
    }


def _rows():
    return [
        _row("M01", "cwa", 0.80, 0.70),
        _row("M01", "pills", 0.60, 0.50),
        _row("M01", "surfaces", 0.40, 0.45),
        _row("M06", "cwa", 0.85, 0.70),
        _row("M06", "pills", 0.65, 0.50),
        _row("M06", "surfaces", 0.45, 0.45),
    ]


def _frame(rows=None):
    return pd.DataFrame(_rows() if rows is None else rows)


def _semantic(rows=None):
    return build_semantics({"domain_metrics": _frame(rows)})[FIGURE_ID]["semantic"]


def _broken(**changes):
    rows = _rows()
    rows[0].update(changes)
    return _frame(rows)


def _rejects(frame):
    with pytest.raises(ValueError):
        build_semantics({"domain_metrics": frame})


def test_six_points_and_hundred_conversion():
    table = _semantic()
    assert len(table) == 6
    assert table.columns.tolist() == COLUMNS
    assert set(table["figure_id"]) == {FIGURE_ID}
    assert table["x"].tolist() == [70.0, 50.0, 45.0, 70.0, 50.0, 45.0]
    assert table["y"].tolist() == [80.0, 60.0, 40.0, 85.0, 65.0, 45.0]
    assert table["complete_contexts"].tolist() == [20] * 6
    assert table["lower"].isna().all() and table["upper"].isna().all()
    assert table["remaining_domains"].isna().all()


def test_deterministic_sorting():
    table = _semantic(_rows()[::-1])
    expected = sorted([[r["aggregation_id"], r["station"], r["domain"]] for r in _rows()])
    assert table[["panel", "series", "label"]].values.tolist() == expected


def test_input_not_mutated_and_extra_columns_excluded():
    frame = _frame()
    before = frame.copy(deep=True)
    table = _semantic(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert "private_note" not in table.columns
    assert table.columns.tolist() == COLUMNS


def test_station_styles():
    table = _semantic()
    for station, group in table.groupby("series"):
        assert set(group["color"]) == {COLORS[station]}
        assert set(group["marker"]) == {MARKERS[station]}


def test_duplicate_headers_rejected():
    frame = _frame()
    _rejects(pd.concat([frame, frame[["model_ba"]]], axis=1))


def test_missing_endpoint_rejected():
    _rejects(_frame([row for row in _rows() if row["aggregation_id"] == "M01"]))


def test_bad_ba_rejected():
    _rejects(_broken(model_ba=1.5, delta=0.8))


def test_boolean_ba_rejected():
    _rejects(_broken(model_ba=True, delta=0.3))


def test_inconsistent_delta_rejected():
    _rejects(_broken(delta=0.5))


def test_empty_metadata_rejected():
    _rejects(_broken(held_instrument="   "))


def test_bad_counts_rejected():
    _rejects(_broken(complete_contexts=21))
