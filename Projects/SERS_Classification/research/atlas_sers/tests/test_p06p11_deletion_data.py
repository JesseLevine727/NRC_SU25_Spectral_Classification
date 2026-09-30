"""Tests for the P06/P11 deletion-stability semantic table."""

import pandas as pd
import pytest

from atlas_sers.visualization.p06p11_deletion_data import build_deletion_semantics

SUPPORTED = 4
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
_DELETIONS = [
    ("M01", "domain", "D1", 0.50, 0.40, 1, 3),
    ("M01", "domain", "D2", 0.30, 0.35, 1, 3),
    ("M01", "instrument", "I1", 0.20, 0.10, 2, 2),
    ("M01", "instrument", "I2", 0.45, 0.55, 2, 2),
    ("M06", "domain", "D1", 0.60, 0.45, 1, 3),
    ("M06", "domain", "D2", 0.40, 0.50, 1, 3),
    ("M06", "instrument", "I1", 0.25, 0.15, 2, 2),
    ("M06", "instrument", "I2", 0.55, 0.65, 2, 2),
]


def _loo_rows():
    return [
        {
            "model_id": "P05-SELECTED",
            "reference_model_id": "C-SELECTED",
            "aggregation_id": endpoint,
            "exclusion_type": exclusion_type,
            "exclusion_id": exclusion_id,
            "removed_domains": removed,
            "remaining_domains": remaining,
            "delta_after_exclusion": model_mean - reference_mean,
            "model_mean_after_exclusion": model_mean,
            "reference_mean_after_exclusion": reference_mean,
            "reason_code": "ok",
        }
        for (
            endpoint,
            exclusion_type,
            exclusion_id,
            model_mean,
            reference_mean,
            removed,
            remaining,
        ) in _DELETIONS
    ]


def _summary_rows():
    return [
        {
            "model_id": "P05-SELECTED",
            "reference_model_id": "C-SELECTED",
            "aggregation_id": "M01",
            "mean_delta": 0.05,
            "supported_domains": SUPPORTED,
            "reason_code": "ok",
        },
        {
            "model_id": "P05-SELECTED",
            "reference_model_id": "C-SELECTED",
            "aggregation_id": "M06",
            "mean_delta": -0.05,
            "supported_domains": SUPPORTED,
            "reason_code": "ok",
        },
    ]


def _make(loo_rows=None, summary_rows=None):
    return {
        "leave_one_out": pd.DataFrame(loo_rows if loo_rows is not None else _loo_rows()),
        "summary": pd.DataFrame(summary_rows if summary_rows is not None else _summary_rows()),
    }


def _semantic(tables):
    return build_deletion_semantics(tables)["F_P06_deletion_stability"]["semantic"]


def test_output_shape_and_exact_columns():
    out = build_deletion_semantics(_make())
    assert set(out) == {"F_P06_deletion_stability"}
    assert out["F_P06_deletion_stability"]["kind"] == "scatter"
    semantic = out["F_P06_deletion_stability"]["semantic"]
    assert list(semantic.columns) == COLUMNS
    assert len(semantic) == 5
    assert semantic.loc[semantic["series"] != "full_reference", "panel"].eq("joint").all()
    assert semantic["lower"].isna().all() and semantic["upper"].isna().all()
    assert semantic["complete_contexts"].isna().all()


def test_known_conversion_and_remaining_counts():
    semantic = _semantic(_make())
    domain = semantic[semantic["label"] == "D1"].iloc[0]
    assert domain["x"] == pytest.approx(10.0)
    assert domain["y"] == pytest.approx(15.0)
    assert domain["remaining_domains"] == 3
    assert domain["domain"] == "D1" and domain["instrument"] == ""
    assert domain["color"] == "#0072B2" and domain["marker"] == "circle"
    instrument = semantic[semantic["label"] == "I1"].iloc[0]
    assert instrument["x"] == pytest.approx(10.0)
    assert instrument["y"] == pytest.approx(10.0)
    assert instrument["remaining_domains"] == 2
    assert instrument["domain"] == "" and instrument["instrument"] == "I1"
    assert instrument["color"] == "#E69F00" and instrument["marker"] == "diamond"


def test_full_reference_point():
    full = _semantic(_make()).iloc[-1]
    assert full["label"] == "Full paired support" and full["series"] == "full_reference"
    assert full["x"] == pytest.approx(5.0) and full["y"] == pytest.approx(-5.0)
    assert full["remaining_domains"] == SUPPORTED
    assert full["domain"] == "" and full["instrument"] == ""
    assert full["color"] == "#000000" and full["marker"] == "cross"


def test_sorted_order_and_inputs_unchanged():
    tables = _make()
    before = {key: value.copy(deep=True) for key, value in tables.items()}
    semantic = _semantic(tables)
    assert list(semantic["series"]) == [
        "removed_domain",
        "removed_domain",
        "removed_instrument",
        "removed_instrument",
        "full_reference",
    ]
    assert list(semantic["label"])[:4] == ["D1", "D2", "I1", "I2"]
    for key, value in tables.items():
        pd.testing.assert_frame_equal(value, before[key])


def test_extra_identifiers_are_not_copied():
    tables = _make()
    extra = pd.DataFrame(
        [
            {
                "model_id": "P05-SELECTED",
                "reference_model_id": "C-SELECTED",
                "aggregation_id": "M02",
                "exclusion_type": "domain",
                "exclusion_id": "D1",
                "removed_domains": 1,
                "remaining_domains": 3,
                "delta_after_exclusion": 0.1,
                "model_mean_after_exclusion": 0.2,
                "reference_mean_after_exclusion": 0.1,
                "reason_code": "ok",
            },
            {
                "model_id": "P05-SELECTED",
                "reference_model_id": "OTHER",
                "aggregation_id": "M01",
                "exclusion_type": "domain",
                "exclusion_id": "D1",
                "removed_domains": 1,
                "remaining_domains": 3,
                "delta_after_exclusion": 0.1,
                "model_mean_after_exclusion": 0.2,
                "reference_mean_after_exclusion": 0.1,
                "reason_code": "ok",
            },
            {
                "model_id": "OTHER",
                "reference_model_id": "C-SELECTED",
                "aggregation_id": "M01",
                "exclusion_type": "domain",
                "exclusion_id": "D1",
                "removed_domains": 1,
                "remaining_domains": 3,
                "delta_after_exclusion": 0.1,
                "model_mean_after_exclusion": 0.2,
                "reference_mean_after_exclusion": 0.1,
                "reason_code": "ok",
            },
        ]
    )
    tables["leave_one_out"] = pd.concat([tables["leave_one_out"], extra], ignore_index=True)
    assert len(_semantic(tables)) == 5


def test_missing_endpoint_rejected():
    rows = [
        row
        for row in _loo_rows()
        if not (row["aggregation_id"] == "M06" and row["exclusion_id"] == "D1")
    ]
    with pytest.raises(ValueError):
        build_deletion_semantics(_make(rows))


def test_unpaired_key_rejected():
    rows = _loo_rows()
    rows.append(
        {
            "model_id": "P05-SELECTED",
            "reference_model_id": "C-SELECTED",
            "aggregation_id": "M01",
            "exclusion_type": "domain",
            "exclusion_id": "D3",
            "removed_domains": 1,
            "remaining_domains": 3,
            "delta_after_exclusion": 0.0,
            "model_mean_after_exclusion": 0.5,
            "reference_mean_after_exclusion": 0.5,
            "reason_code": "ok",
        }
    )
    with pytest.raises(ValueError):
        build_deletion_semantics(_make(rows))


def test_duplicate_rejected():
    rows = _loo_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError):
        build_deletion_semantics(_make(rows))


def test_missing_deletion_types_rejected():
    rows = [row for row in _loo_rows() if row["exclusion_type"] == "domain"]
    with pytest.raises(ValueError):
        build_deletion_semantics(_make(rows))


def test_required_columns_rejected():
    tables = _make()
    tables["leave_one_out"] = tables["leave_one_out"].drop(columns=["reason_code"])
    with pytest.raises(ValueError):
        build_deletion_semantics(tables)
    tables = _make()
    tables["summary"] = tables["summary"].drop(columns=["supported_domains"])
    with pytest.raises(ValueError):
        build_deletion_semantics(tables)


def test_duplicate_columns_rejected():
    tables = _make()
    frame = tables["leave_one_out"]
    tables["leave_one_out"] = pd.concat([frame, frame[["exclusion_id"]]], axis=1)
    with pytest.raises(ValueError):
        build_deletion_semantics(tables)


def test_reason_code_must_be_exact():
    for bad in ("OK", " ok ", "okay"):
        rows = _loo_rows()
        rows[0]["reason_code"] = bad
        with pytest.raises(ValueError):
            build_deletion_semantics(_make(rows))
    summaries = _summary_rows()
    summaries[0]["reason_code"] = "OK"
    with pytest.raises(ValueError):
        build_deletion_semantics(_make(summary_rows=summaries))


def test_whitespace_identity_rejected():
    for bad in ("   ", " D1"):
        rows = _loo_rows()
        rows[0]["exclusion_id"] = bad
        with pytest.raises(ValueError):
            build_deletion_semantics(_make(rows))


def test_bad_counts_rejected():
    for column, value in [
        ("removed_domains", 0),
        ("removed_domains", True),
        ("remaining_domains", -1),
        ("remaining_domains", 2),
    ]:
        rows = _loo_rows()
        rows[0][column] = value
        with pytest.raises(ValueError):
            build_deletion_semantics(_make(rows))


def test_nonfinite_and_bool_rejected():
    for column, value in [
        ("delta_after_exclusion", float("nan")),
        ("delta_after_exclusion", True),
        ("model_mean_after_exclusion", float("inf")),
        ("reference_mean_after_exclusion", float("-inf")),
    ]:
        rows = _loo_rows()
        rows[0][column] = value
        with pytest.raises(ValueError):
            build_deletion_semantics(_make(rows))


def test_inconsistent_delta_and_bounds_rejected():
    rows = _loo_rows()
    rows[0]["delta_after_exclusion"] = 0.99
    with pytest.raises(ValueError):
        build_deletion_semantics(_make(rows))
    rows = _loo_rows()
    rows[0]["model_mean_after_exclusion"] = 2.0
    rows[0]["delta_after_exclusion"] = (
        rows[0]["model_mean_after_exclusion"] - rows[0]["reference_mean_after_exclusion"]
    )
    with pytest.raises(ValueError):
        build_deletion_semantics(_make(rows))


def test_bad_summary_rejected():
    summaries = _summary_rows()
    summaries[0]["supported_domains"] = SUPPORTED + 1
    with pytest.raises(ValueError):
        build_deletion_semantics(_make(summary_rows=summaries))
    with pytest.raises(ValueError):
        build_deletion_semantics(_make(summary_rows=_summary_rows()[:1]))
