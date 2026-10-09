"""Unit tests for the preplanned leave-one-family sensitivity supplement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.p08_family_deletions import prepare_family_deletions

ESTIMANDS = ("equal_context", "pooled_four_fold")
CONTRASTS = ("policy", "four_cell_interaction")
OUTPUT_COLUMNS = [
    "estimand",
    "contrast_id",
    "excluded_family",
    "excluded_domain_count",
    "retained_domain_count",
    "effect",
    "reason",
]


def make_frame(domains, instruments, effects, estimands=ESTIMANDS, contrasts=CONTRASTS):
    rows = []
    for estimand in estimands:
        for contrast_id in contrasts:
            for domain, instrument, effect in zip(domains, instruments, effects, strict=True):
                rows.append(
                    {
                        "estimand": estimand,
                        "contrast_id": contrast_id,
                        "domain": domain,
                        "instrument": instrument,
                        "effect": effect,
                    }
                )
    return pd.DataFrame(rows)


def base_frame():
    return make_frame(
        ["d1", "d2", "d3"],
        ["Agilent-1", "Agilent-1", "Mira-2"],
        [1.0, 3.0, 5.0],
    )


def select(result, estimand, contrast_id, family):
    mask = (
        (result["estimand"] == estimand)
        & (result["contrast_id"] == contrast_id)
        & (result["excluded_family"] == family)
    )
    assert int(mask.sum()) == 1
    return result.loc[mask].iloc[0]


def test_shared_instrument_deletion_and_equal_domain_means():
    result = prepare_family_deletions(base_frame())
    assert list(result.columns) == OUTPUT_COLUMNS

    agilent = select(result, "equal_context", "policy", "Agilent")
    assert agilent["excluded_domain_count"] == 2
    assert agilent["retained_domain_count"] == 1
    assert agilent["effect"] == pytest.approx(5.0)
    assert agilent["reason"] == ""

    # Two retained domains with different effects: mean is per-domain (1+3)/2.
    mira = select(result, "equal_context", "policy", "Mira")
    assert mira["excluded_domain_count"] == 1
    assert mira["retained_domain_count"] == 2
    assert mira["effect"] == pytest.approx(2.0)


def test_two_estimands_and_both_contrasts():
    result = prepare_family_deletions(base_frame())
    assert set(result["estimand"]) == set(ESTIMANDS)
    assert set(result["contrast_id"]) == set(CONTRASTS)
    assert len(result) == len(ESTIMANDS) * len(CONTRASTS) * 2
    for contrast_id in CONTRASTS:
        for family in ("Agilent", "Mira"):
            left = select(result, "equal_context", contrast_id, family)
            right = select(result, "pooled_four_fold", contrast_id, family)
            assert left["effect"] == pytest.approx(right["effect"])


def test_all_one_family_is_undefined():
    frame = make_frame(
        ["d1", "d2"],
        ["Pendar-1", "Pendar-2"],
        [1.0, 4.0],
        estimands=("equal_context",),
        contrasts=("policy",),
    )
    row = select(prepare_family_deletions(frame), "equal_context", "policy", "Pendar")
    assert row["excluded_domain_count"] == 2
    assert row["retained_domain_count"] == 0
    assert row["effect"] is None
    assert row["reason"] == "no_retained_domains"


def test_unknown_family_is_omitted_but_domains_are_retained():
    frame = make_frame(
        ["d1", "d2"],
        ["Agilent-1", "unknown-9"],
        [2.0, 10.0],
        estimands=("equal_context",),
        contrasts=("policy",),
    )
    result = prepare_family_deletions(frame)
    assert "unknown" not in {family.lower() for family in result["excluded_family"]}
    row = select(result, "equal_context", "policy", "Agilent")
    assert row["excluded_domain_count"] == 1
    assert row["retained_domain_count"] == 1
    assert row["effect"] == pytest.approx(10.0)


def test_empty_output_when_no_known_families():
    frame = make_frame(
        ["d1", "d2"],
        ["unknown-1", "UNKNOWN-2"],
        [1.0, 2.0],
        estimands=("equal_context",),
        contrasts=("policy",),
    )
    result = prepare_family_deletions(frame)
    assert list(result.columns) == OUTPUT_COLUMNS
    assert result.empty


def test_duplicate_domains_raise():
    frame = make_frame(
        ["d1", "d1"],
        ["Agilent-1", "Mira-1"],
        [1.0, 2.0],
        estimands=("equal_context",),
        contrasts=("policy",),
    )
    with pytest.raises(ValueError):
        prepare_family_deletions(frame)


@pytest.mark.parametrize("bad_effect", [np.nan, np.inf, -np.inf, True, False])
def test_nonfinite_or_bool_effect_raises(bad_effect):
    frame = make_frame(
        ["d1"],
        ["Agilent-1"],
        [bad_effect],
        estimands=("equal_context",),
        contrasts=("policy",),
    )
    with pytest.raises(ValueError):
        prepare_family_deletions(frame)


@pytest.mark.parametrize("column", ["estimand", "contrast_id", "domain", "instrument"])
def test_empty_name_raises(column):
    frame = make_frame(
        ["d1"],
        ["Agilent-1"],
        [1.0],
        estimands=("equal_context",),
        contrasts=("policy",),
    )
    frame.loc[0, column] = ""
    with pytest.raises(ValueError):
        prepare_family_deletions(frame)


def test_unsupported_estimand_raises():
    frame = make_frame(
        ["d1"],
        ["Agilent-1"],
        [1.0],
        estimands=("bogus",),
        contrasts=("policy",),
    )
    with pytest.raises(ValueError):
        prepare_family_deletions(frame)


def test_invalid_acquisition_name_raises():
    frame = make_frame(
        ["d1"],
        ["NoSuffixInstrument"],
        [1.0],
        estimands=("equal_context",),
        contrasts=("policy",),
    )
    with pytest.raises(ValueError):
        prepare_family_deletions(frame)


def test_inconsistent_instrument_mapping_raises():
    frame = pd.DataFrame(
        [
            {
                "estimand": "equal_context",
                "contrast_id": "policy",
                "domain": "d1",
                "instrument": "Agilent-1",
                "effect": 1.0,
            },
            {
                "estimand": "pooled_four_fold",
                "contrast_id": "policy",
                "domain": "d1",
                "instrument": "Mira-1",
                "effect": 1.0,
            },
        ]
    )
    with pytest.raises(ValueError):
        prepare_family_deletions(frame)


def test_changed_support_raises():
    frame = pd.DataFrame(
        [
            {
                "estimand": "equal_context",
                "contrast_id": "policy",
                "domain": "d1",
                "instrument": "Agilent-1",
                "effect": 1.0,
            },
            {
                "estimand": "equal_context",
                "contrast_id": "policy",
                "domain": "d2",
                "instrument": "Mira-1",
                "effect": 2.0,
            },
            {
                "estimand": "equal_context",
                "contrast_id": "four_cell_interaction",
                "domain": "d1",
                "instrument": "Agilent-1",
                "effect": 1.0,
            },
            {
                "estimand": "equal_context",
                "contrast_id": "four_cell_interaction",
                "domain": "d3",
                "instrument": "Mira-1",
                "effect": 3.0,
            },
        ]
    )
    with pytest.raises(ValueError):
        prepare_family_deletions(frame)


def test_shuffle_parity():
    frame = base_frame()
    expected = prepare_family_deletions(frame)
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    pd.testing.assert_frame_equal(prepare_family_deletions(shuffled), expected)


def test_input_not_mutated():
    frame = base_frame()
    before = frame.copy(deep=True)
    prepare_family_deletions(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_extra_metadata_columns_allowed():
    frame = base_frame()
    frame["n_samples"] = 100
    result = prepare_family_deletions(frame)
    assert len(result) == len(ESTIMANDS) * len(CONTRASTS) * 2


def test_missing_required_columns_raise():
    frame = base_frame().drop(columns=["effect"])
    with pytest.raises(ValueError):
        prepare_family_deletions(frame)


def test_whitespace_name_rejected():
    frame = base_frame()
    frame.loc[0, "domain"] = "  "
    with pytest.raises(ValueError):
        prepare_family_deletions(frame)


def test_shuffle_is_exact_for_cancelling_fractional_effects():
    frame = make_frame(
        ["d1", "d2", "d3", "d4"],
        ["Agilent-1", "Mira-1", "Mira-2", "Mira-3"],
        [0.2, 0.8, -0.8, 1e-16],
    )
    pd.testing.assert_frame_equal(
        prepare_family_deletions(frame),
        prepare_family_deletions(frame.iloc[::-1]),
        check_exact=True,
    )
