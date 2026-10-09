"""Synthetic tests for the T299 universal contrast engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.p06p11_inference import compile_pair, score_weights
from atlas_sers.evaluation.p08_universal_inference import (
    _sign_sensitivity,
    analyze_contrast,
    holm_fixed_family,
)

IDENTITY = [
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "unit_id",
    "true_label",
]

# Every master belongs to exactly one station; m1/m3 repeat across contexts of
# the same station/domain so shared master weights are exercised.
TWO_ROWS = [
    ("c1", "d1", "s1", "TIRF", "m1", "u1", "a", 1, 1),
    ("c1", "d1", "s1", "TIRF", "m3", "u2", "a", 0, 0),
    ("c1", "d1", "s1", "TIRF", "m2", "u3", "b", 0, 0),
    ("c1", "d1", "s1", "TIRF", "m2", "u4", "b", 1, 1),
    ("c2", "d1", "s1", "TIRF", "m1", "u5", "a", 1, 0),
    ("c2", "d1", "s1", "TIRF", "m3", "u6", "a", 0, 0),
    ("c2", "d1", "s1", "TIRF", "m3", "u7", "a", 0, 0),
    ("c3", "d2", "s2", "TIRF", "m4", "u8", "a", 1, 1),
    ("c3", "d2", "s2", "TIRF", "m4", "u9", "a", 0, 0),
    ("c4", "d2", "s2", "TIRF", "m5", "u10", "b", 1, 1),
    ("c4", "d2", "s2", "TIRF", "m5", "u11", "b", 0, 0),
    ("c5", "d3", "s3", "CONF", "m6", "u12", "a", 1, 1),
    ("c5", "d3", "s3", "CONF", "m6", "u13", "a", 0, 0),
]

FOUR_ROWS = [
    ("c1", "d1", "s1", "TIRF", "m1", "u1", "a", 1, 1, 0, 0),
    ("c1", "d1", "s1", "TIRF", "m3", "u2", "a", 0, 0, 0, 0),
    ("c1", "d1", "s1", "TIRF", "m2", "u3", "b", 0, 0, 0, 0),
    ("c1", "d1", "s1", "TIRF", "m2", "u4", "b", 1, 1, 0, 0),
    ("c2", "d1", "s1", "TIRF", "m1", "u5", "a", 1, 0, 0, 0),
    ("c2", "d1", "s1", "TIRF", "m3", "u6", "a", 0, 0, 0, 0),
    ("c2", "d1", "s1", "TIRF", "m3", "u7", "a", 0, 0, 0, 0),
    ("c3", "d2", "s2", "TIRF", "m4", "u8", "a", 1, 1, 0, 0),
    ("c3", "d2", "s2", "TIRF", "m4", "u9", "a", 0, 0, 0, 0),
    ("c4", "d2", "s2", "TIRF", "m5", "u10", "b", 1, 1, 0, 0),
    ("c4", "d2", "s2", "TIRF", "m5", "u11", "b", 0, 0, 0, 0),
    ("c5", "d3", "s3", "CONF", "m6", "u12", "a", 1, 1, 0, 0),
    ("c5", "d3", "s3", "CONF", "m6", "u13", "a", 0, 0, 0, 0),
]

# Two cells each represented by one master of the same class pool: an
# occurrence that zeroes one master leaves the other cell undefined.
SPARSE_ROWS = [
    ("c1", "d1", "s1", "TIRF", "m1", "u1", "a", 1, 0),
    ("c2", "d1", "s1", "TIRF", "m2", "u2", "a", 1, 0),
]

GLOBAL_MASTERS = ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"]
GLOBAL_INSTRUMENTS = ["CONF", "OTHER", "TIRF"]


def _frame(rows, extra):
    return pd.DataFrame(list(rows), columns=IDENTITY + list(extra))


def _two_frame():
    return _frame(TWO_ROWS, ("policy", "min"))


def _four_frame():
    return _frame(FOUR_ROWS, ("dp", "dm", "cp", "cm"))


def _sparse_frame():
    return _frame(SPARSE_ROWS, ("policy", "min"))


def _weights(seed, draws, masters, instruments):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(0.5, 2.0, size=(draws, len(masters))),
        rng.uniform(0.5, 2.0, size=(draws, len(instruments))),
    )


def _contrast_design(frame, columns, coefficients):
    designs = []
    for column in columns:
        work = frame.copy()
        work["correct_model"] = work[column]
        work["correct_reference"] = 0
        designs.append(compile_pair(work))
    delta = np.zeros_like(np.asarray(designs[0].delta_correct, dtype=float))
    for coefficient, design in zip(coefficients, designs, strict=True):
        delta = delta + coefficient * np.asarray(design.delta_correct, dtype=float)
    return replace(designs[0], delta_correct=delta)


def _common(draws=4, seed=1):
    return {
        "global_masters": GLOBAL_MASTERS,
        "global_instruments": GLOBAL_INSTRUMENTS,
        "hierarchical_draws": draws,
        "hierarchical_seed": seed,
    }


def test_point_equal_context_equal_domain():
    frame = _two_frame()
    mw, iw = _weights(1, 8, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    result = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=16,
        hierarchical_seed=7,
    )
    point = result["point"]
    assert point["overall"] == pytest.approx(1.0 / 18.0, abs=1e-12)
    assert abs(point["overall"] - 1.0 / 9.0) > 1e-12
    assert point["domain_effects"]["d1"] == pytest.approx(1.0 / 6.0, abs=1e-12)
    assert point["domain_effects"]["d2"] == 0.0
    assert point["domain_effects"]["d3"] == 0.0
    assert list(point["domain_effects"]) == ["d1", "d2", "d3"]
    assert point["domain_instrument"] == {"d1": "TIRF", "d2": "TIRF", "d3": "CONF"}
    assert point["procedure_effects"][0] == pytest.approx(17.0 / 36.0, abs=1e-12)
    assert point["procedure_effects"][1] == pytest.approx(5.0 / 12.0, abs=1e-12)
    assert set(point["cell_effects"]) == {
        ("c1", "a"),
        ("c1", "b"),
        ("c2", "a"),
        ("c3", "a"),
        ("c4", "b"),
        ("c5", "a"),
    }


def test_point_uses_hierarchical_not_naive_mean():
    frame = _two_frame()
    mw, iw = _weights(2, 8, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    result = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=8,
        hierarchical_seed=3,
    )
    point = result["point"]
    values = frame["policy"].to_numpy() - frame["min"].to_numpy()
    d1_rows = frame["domain"].to_numpy() == "d1"
    # Naive row mean over d1 is 1/7; hierarchical context/domain mean is 1/6.
    assert values[d1_rows].mean() == pytest.approx(1.0 / 7.0, abs=1e-12)
    assert point["domain_effects"]["d1"] == pytest.approx(1.0 / 6.0, abs=1e-12)
    # Naive mean of the three d1 cell effects would be 1/9 too, so assert the
    # cell values themselves and the distinct hierarchical result.
    d1_cells = [
        value
        for (context, _label), value in point["cell_effects"].items()
        if context in {"c1", "c2"}
    ]
    assert np.mean(d1_cells) == pytest.approx(1.0 / 9.0, abs=1e-12)
    assert point["overall"] == pytest.approx(1.0 / 18.0, abs=1e-12)


def test_unit_parity_with_inherited_score_weights():
    frame = _two_frame()
    design = _contrast_design(frame, ["policy", "min"], [1, -1])
    mw, iw = _weights(2, 8, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    result = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=8,
        hierarchical_seed=3,
    )
    overall, domain_out = score_weights(
        design,
        np.ones((1, len(design.masters))),
        np.ones((1, len(design.instruments))),
    )
    assert overall[0] == pytest.approx(result["point"]["overall"], abs=1e-12)
    for position, name in enumerate(result["domains"]):
        assert domain_out[0, position] == pytest.approx(
            result["point"]["domain_effects"][name], abs=1e-12
        )


def test_weighted_modes_match_inherited_score_weights():
    frame = _two_frame()
    design = _contrast_design(frame, ["policy", "min"], [1, -1])
    mw, iw = _weights(4, 12, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    mw_panel = mw[:, [GLOBAL_MASTERS.index(name) for name in design.masters]]
    iw_panel = iw[:, [GLOBAL_INSTRUMENTS.index(name) for name in design.instruments]]
    result = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=8,
        hierarchical_seed=5,
    )
    crossed, _ = score_weights(design, mw_panel, iw_panel)
    master, _ = score_weights(design, mw_panel, np.ones_like(iw_panel))
    instrument, _ = score_weights(design, np.ones_like(mw_panel), iw_panel)
    np.testing.assert_allclose(
        result["weighted"]["crossed"]["overall"]["draws"], crossed
    )
    np.testing.assert_allclose(
        result["weighted"]["master_only"]["overall"]["draws"], master
    )
    np.testing.assert_allclose(
        result["weighted"]["instrument_only"]["overall"]["draws"], instrument
    )


def test_shared_weight_linearity_for_four_term():
    frame = _four_frame()
    mw, iw = _weights(6, 10, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    result = analyze_contrast(
        frame,
        ["dp", "dm", "cp", "cm"],
        [1, -1, -1, 1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=8,
        hierarchical_seed=9,
    )
    crossed = result["weighted"]["crossed"]
    combined = sum(
        coefficient * term["draws"]
        for coefficient, term in zip(
            [1, -1, -1, 1], crossed["procedure_overall"], strict=True
        )
    )
    np.testing.assert_allclose(crossed["overall"]["draws"], combined, atol=1e-10)
    assert len(crossed["procedure_min_ba"]) == 4
    assert "min_ba_difference" not in crossed


def test_global_weight_subsetting():
    frame = _two_frame()
    mw, iw = _weights(3, 10, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    full = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=8,
        hierarchical_seed=2,
    )
    masters = ["m1", "m2", "m3", "m4", "m5", "m6"]
    instruments = ["CONF", "TIRF"]
    sliced_mw = mw[:, [GLOBAL_MASTERS.index(name) for name in masters]]
    sliced_iw = iw[:, [GLOBAL_INSTRUMENTS.index(name) for name in instruments]]
    sliced = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=masters,
        global_instruments=instruments,
        master_weights=sliced_mw,
        instrument_weights=sliced_iw,
        hierarchical_draws=8,
        hierarchical_seed=2,
    )
    np.testing.assert_allclose(
        full["weighted"]["crossed"]["overall"]["draws"],
        sliced["weighted"]["crossed"]["overall"]["draws"],
    )
    np.testing.assert_array_equal(
        full["hierarchy"]["sampled_domains"], sliced["hierarchy"]["sampled_domains"]
    )


def test_row_order_invariance_and_no_mutation():
    frame = _two_frame()
    mw, iw = _weights(8, 9, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    snapshot_frame = frame.copy(deep=True)
    snapshot_mw = mw.copy()
    snapshot_iw = iw.copy()
    first = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=8,
        hierarchical_seed=11,
    )
    shuffled = analyze_contrast(
        frame.sample(frac=1.0, random_state=7).reset_index(drop=True),
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=8,
        hierarchical_seed=11,
    )
    for name in first["point"]["domain_effects"]:
        assert first["point"]["domain_effects"][name] == pytest.approx(
            shuffled["point"]["domain_effects"][name], abs=1e-12
        )
    np.testing.assert_allclose(
        first["weighted"]["crossed"]["overall"]["draws"],
        shuffled["weighted"]["crossed"]["overall"]["draws"],
    )
    np.testing.assert_allclose(
        first["hierarchy"]["combined"]["draws"],
        shuffled["hierarchy"]["combined"]["draws"],
        equal_nan=True,
    )
    pd.testing.assert_frame_equal(frame, snapshot_frame)
    np.testing.assert_array_equal(mw, snapshot_mw)
    np.testing.assert_array_equal(iw, snapshot_iw)


def test_hierarchy_full_defined_fixture():
    frame = _two_frame()
    mw, iw = _weights(12, 8, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    result = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=32,
        hierarchical_seed=12345,
    )
    summary = result["hierarchy"]["combined"]["summary"]
    assert summary["undefined_draws"] == 0
    assert summary["defined_draws"] == 32
    assert summary["lower"] is not None and summary["upper"] is not None
    assert result["hierarchy"]["empty_cells_total"] == 0
    assert result["hierarchy"]["affected_domains"].shape == (32,)
    assert result["hierarchy"]["affected_domains_total"] == 0


def test_hierarchy_sparse_undefined_draws():
    frame = _sparse_frame()
    mw, iw = _weights(13, 8, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    result = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=256,
        hierarchical_seed=12345,
    )
    summary = result["hierarchy"]["combined"]["summary"]
    assert summary["undefined_draws"] > 0
    assert summary["lower"] is None and summary["upper"] is None
    assert summary["reason_code"] == "hierarchical_fixed_support_undefined"
    assert result["hierarchy"]["empty_cells_total"] > 0


def test_all_identical_procedures_zero_and_sign_symmetry():
    frame = _two_frame()
    frame["min"] = frame["policy"]
    mw, iw = _weights(14, 6, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    result = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=16,
        hierarchical_seed=21,
    )
    assert result["point"]["overall"] == 0.0
    np.testing.assert_allclose(result["weighted"]["crossed"]["overall"]["draws"], 0.0)
    assert (
        result["weighted"]["crossed"]["overall"]["summary"]["reason_code"]
        == "degenerate_distribution"
    )
    assert (
        result["hierarchy"]["combined"]["summary"]["reason_code"]
        == "degenerate_distribution"
    )
    sign = result["sign_sensitivity"]
    assert sign["domain"]["p_descriptive"] == 1.0
    assert sign["instrument"]["p_descriptive"] == 1.0
    assert sign["domain"]["hits"] == sign["domain"]["assignments"]
    assert sign["instrument"]["hits"] == sign["instrument"]["assignments"]


def test_sign_domain_and_instrument_are_separate():
    result = _sign_sensitivity(np.array([1.0, 0.5]), np.array([0, 0]), 1)
    assert result["observed_statistic"] == pytest.approx(0.75)
    assert result["domain"]["groups"] == 2
    assert result["domain"]["assignments"] == 4
    assert result["domain"]["hits"] == 2
    assert result["domain"]["p_descriptive"] == pytest.approx(0.5)
    assert result["instrument"]["groups"] == 1
    assert result["instrument"]["assignments"] == 2
    assert result["instrument"]["hits"] == 2
    assert result["instrument"]["p_descriptive"] == pytest.approx(1.0)
    assert "total_assignments" not in result
    assert "count_ge_observed" not in result


def test_sign_sensitivity_zero_effect():
    result = _sign_sensitivity(np.zeros(3), np.array([0, 1, 1]), 2)
    assert result["domain"]["assignments"] == 8
    assert result["domain"]["p_descriptive"] == 1.0
    assert result["instrument"]["assignments"] == 4
    assert result["instrument"]["p_descriptive"] == 1.0


def test_sign_sensitivity_rejects_too_many_groups():
    with pytest.raises(ValueError):
        _sign_sensitivity(np.zeros(14), np.zeros(14, dtype=np.intp), 1)
    with pytest.raises(ValueError):
        _sign_sensitivity(np.zeros(2), np.zeros(2, dtype=np.intp), 11)


def test_leave_one_domain_instrument_family():
    frame = _two_frame()
    mw, iw = _weights(15, 6, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    result = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=8,
        hierarchical_seed=31,
        domain_families={"d1": "fA", "d2": "fA", "d3": None, "d9": "fB"},
    )
    descriptive = result["descriptive"]
    assert descriptive["leave_one_domain"]["d1"] == pytest.approx(0.0, abs=1e-12)
    assert descriptive["leave_one_domain"]["d2"] == pytest.approx(1.0 / 12.0, abs=1e-12)
    assert descriptive["leave_one_instrument"]["CONF"] == pytest.approx(
        1.0 / 12.0, abs=1e-12
    )
    assert descriptive["leave_one_instrument"]["TIRF"] == pytest.approx(0.0, abs=1e-12)
    assert list(descriptive["leave_one_family"]) == ["fA"]
    assert descriptive["leave_one_family"]["fA"] == pytest.approx(0.0, abs=1e-12)
    assert descriptive["contrast_domain"]["positive"] == 1
    assert descriptive["contrast_domain"]["tied"] == 2


def test_unknown_families_are_not_grouped():
    frame = _two_frame()
    mw, iw = _weights(16, 4, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    result = analyze_contrast(
        frame,
        ["policy", "min"],
        [1, -1],
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        master_weights=mw,
        instrument_weights=iw,
        hierarchical_draws=4,
        hierarchical_seed=41,
        domain_families={
            "d1": "fA",
            "d2": "unknown",
            "d3": "  Unknown  ",
            "d9": "fB",
        },
    )
    families = result["descriptive"]["leave_one_family"]
    assert list(families) == ["fA"]
    assert "unknown" not in families
    assert "fB" not in families


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("policy", 2),
        lambda d: d.__setitem__("policy", d["policy"].astype(str)),
        lambda d: d.__setitem__("policy", np.nan),
        lambda d: d.drop(columns=["min"], inplace=True),
    ],
)
def test_invalid_binary_and_columns(mutate):
    frame = _two_frame()
    mutate(frame)
    mw, iw = _weights(16, 4, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    with pytest.raises((TypeError, ValueError)):
        analyze_contrast(
            frame,
            ["policy", "min"],
            [1, -1],
            global_masters=GLOBAL_MASTERS,
            global_instruments=GLOBAL_INSTRUMENTS,
            master_weights=mw,
            instrument_weights=iw,
            hierarchical_draws=4,
            hierarchical_seed=1,
        )


def test_invalid_coefficients_and_duplicate_columns():
    frame = _two_frame()
    mw, iw = _weights(17, 4, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    common = {
        "global_masters": GLOBAL_MASTERS,
        "global_instruments": GLOBAL_INSTRUMENTS,
        "master_weights": mw,
        "instrument_weights": iw,
        "hierarchical_draws": 4,
        "hierarchical_seed": 1,
    }
    with pytest.raises(ValueError):
        analyze_contrast(frame, ["policy", "min"], [1, 1], **common)
    with pytest.raises(ValueError):
        analyze_contrast(frame, ["policy", "policy"], [1, -1], **common)
    with pytest.raises(ValueError):
        analyze_contrast(frame, ["policy", "min"], [1, -1, 1], **common)


def test_coefficients_require_exact_real_values():
    frame = _two_frame()
    mw, iw = _weights(18, 4, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    common = {
        "global_masters": GLOBAL_MASTERS,
        "global_instruments": GLOBAL_INSTRUMENTS,
        "master_weights": mw,
        "instrument_weights": iw,
        "hierarchical_draws": 4,
        "hierarchical_seed": 1,
    }
    with pytest.raises(TypeError):
        analyze_contrast(frame, ["policy", "min"], ["1", "-1"], **common)
    with pytest.raises(TypeError):
        analyze_contrast(frame, ["policy", "min"], [True, -1], **common)
    with pytest.raises(TypeError):
        analyze_contrast(frame, ["policy", "min"], [1 + 0j, -1], **common)
    with pytest.raises(ValueError):
        analyze_contrast(frame, ["policy", "min"], [1.0, -1.0000001], **common)


def test_unhashable_correct_columns():
    frame = _two_frame()
    mw, iw = _weights(19, 4, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    common = {
        "global_masters": GLOBAL_MASTERS,
        "global_instruments": GLOBAL_INSTRUMENTS,
        "master_weights": mw,
        "instrument_weights": iw,
        "hierarchical_draws": 4,
        "hierarchical_seed": 1,
    }
    with pytest.raises(ValueError):
        analyze_contrast(frame, ["policy", ["min"]], [1, -1], **common)
    with pytest.raises(TypeError):
        analyze_contrast(frame, "policy", [1, -1], **common)


@pytest.mark.parametrize(
    "weights",
    [
        np.ones(8),
        np.ones((0, 8)),
        np.ones((3, 5)),
        np.zeros((3, 8)),
        -np.ones((3, 8)),
        np.full((3, 8), np.nan),
        np.ones((10001, 8)),
        np.array([["1"] * 8] * 3),
    ],
)
def test_invalid_master_weights(weights):
    frame = _two_frame()
    iw = np.ones((3, len(GLOBAL_INSTRUMENTS)))
    with pytest.raises((TypeError, ValueError)):
        analyze_contrast(
            frame,
            ["policy", "min"],
            [1, -1],
            global_masters=GLOBAL_MASTERS,
            global_instruments=GLOBAL_INSTRUMENTS,
            master_weights=weights,
            instrument_weights=iw,
            hierarchical_draws=4,
            hierarchical_seed=1,
        )


def test_missing_global_ids_and_unsorted_identities():
    frame = _two_frame()
    mw, iw = _weights(20, 4, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    with pytest.raises(ValueError):
        analyze_contrast(
            frame,
            ["policy", "min"],
            [1, -1],
            global_masters=["m1", "m2", "m3"],
            global_instruments=GLOBAL_INSTRUMENTS,
            master_weights=mw[:, :3],
            instrument_weights=iw,
            hierarchical_draws=4,
            hierarchical_seed=1,
        )
    with pytest.raises(ValueError):
        analyze_contrast(
            frame,
            ["policy", "min"],
            [1, -1],
            global_masters=["m1", "m0", "m2", "m3", "m4", "m5", "m6", "m7"],
            global_instruments=GLOBAL_INSTRUMENTS,
            master_weights=mw,
            instrument_weights=iw,
            hierarchical_draws=4,
            hierarchical_seed=1,
        )


def test_master_cannot_span_stations():
    rows = [
        ("c1", "d1", "s1", "TIRF", "m1", "u1", "a", 1, 0),
        ("c2", "d2", "s2", "TIRF", "m1", "u2", "a", 1, 0),
    ]
    frame = _frame(rows, ("policy", "min"))
    mw = np.ones((4, len(GLOBAL_MASTERS)))
    iw = np.ones((4, len(GLOBAL_INSTRUMENTS)))
    with pytest.raises(ValueError):
        analyze_contrast(
            frame,
            ["policy", "min"],
            [1, -1],
            global_masters=GLOBAL_MASTERS,
            global_instruments=GLOBAL_INSTRUMENTS,
            master_weights=mw,
            instrument_weights=iw,
            hierarchical_draws=4,
            hierarchical_seed=1,
        )


@pytest.mark.parametrize("draws", [0, -1, 10001, 1.5, True, "10"])
def test_invalid_hierarchy_draws(draws):
    frame = _two_frame()
    mw, iw = _weights(21, 4, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    with pytest.raises((TypeError, ValueError)):
        analyze_contrast(
            frame,
            ["policy", "min"],
            [1, -1],
            global_masters=GLOBAL_MASTERS,
            global_instruments=GLOBAL_INSTRUMENTS,
            master_weights=mw,
            instrument_weights=iw,
            hierarchical_draws=draws,
            hierarchical_seed=1,
        )


def test_invalid_hierarchy_seed():
    frame = _two_frame()
    mw, iw = _weights(22, 4, GLOBAL_MASTERS, GLOBAL_INSTRUMENTS)
    with pytest.raises(ValueError):
        analyze_contrast(
            frame,
            ["policy", "min"],
            [1, -1],
            global_masters=GLOBAL_MASTERS,
            global_instruments=GLOBAL_INSTRUMENTS,
            master_weights=mw,
            instrument_weights=iw,
            hierarchical_draws=4,
            hierarchical_seed=-1,
        )


def test_holm_fixed_family_basic():
    adjusted = holm_fixed_family([0.01, 0.04, None], 3)
    assert adjusted[2] is None
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[1] == pytest.approx(0.08)


def test_holm_fixed_family_unavailable_slots():
    values = [None] * 8 + [0.001 * (i + 1) for i in range(12)]
    adjusted = holm_fixed_family(values, 20)
    assert all(value is None for value in adjusted[:8])
    available = adjusted[8:]
    assert all(value is not None and 0.0 <= value <= 1.0 for value in available)
    assert available == sorted(available)
    interaction = [None] * 8 + [0.002 * (i + 1) for i in range(24)]
    adjusted_interaction = holm_fixed_family(interaction, 32)
    assert all(value is None for value in adjusted_interaction[:8])
    assert all(0.0 <= value <= 1.0 for value in adjusted_interaction[8:])


@pytest.mark.parametrize(
    "values",
    [
        [],
        [True, 0.5],
        [1.2, 0.5],
        [-0.1, 0.5],
        [np.nan, 0.5],
        [np.inf, 0.5],
        ["0.5", 0.5],
        [0.5, 1 + 0j],
    ],
)
def test_holm_rejects_bad_values(values):
    with pytest.raises((TypeError, ValueError)):
        holm_fixed_family(values, len(values))


def test_holm_rejects_bad_family_size():
    with pytest.raises(ValueError):
        holm_fixed_family([0.5, 0.5], 3)
    with pytest.raises(TypeError):
        holm_fixed_family([0.5, 0.5], True)
