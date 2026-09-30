"""Synthetic tests for the outcome-blind structural support audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.p06p11_support import SupportAuditError, audit_support

COLUMNS = [
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "observation_uid",
    "true_label",
]

DOMAIN_COLUMNS = [
    "domain",
    "station",
    "instrument",
    "contexts",
    "unique_spectra",
    "unique_masters",
    "classes",
    "class_cells",
    "singleton_class_cells",
]

RISK_COLUMNS = [
    "domain",
    "station",
    "instrument",
    "class_cells",
    "expected_empty_class_cells",
    "minimum_cell_empty_probability",
    "maximum_cell_empty_probability",
]


def _frame(records):
    return pd.DataFrame(records, columns=COLUMNS)


def _baseline():
    return _frame(
        [
            ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C1", "D1", "S1", "I1", "M2", "O2", "dog"),
            ("C2", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C2", "D1", "S1", "I1", "M2", "O2", "dog"),
        ]
    )


def test_returns_exactly_the_five_deterministic_tables():
    result = audit_support(_baseline())
    assert list(result) == [
        "domains",
        "class_cell_histogram",
        "master_domain_histogram",
        "instrument_domains",
        "empty_class_risk",
    ]
    for table in result.values():
        assert isinstance(table, pd.DataFrame)


def test_domains_table_baseline():
    domains = audit_support(_baseline())["domains"]
    assert list(domains.columns) == DOMAIN_COLUMNS
    assert len(domains) == 1
    row = domains.iloc[0]
    assert row["domain"] == "D1"
    assert row["station"] == "S1"
    assert row["instrument"] == "I1"
    assert row["contexts"] == 2
    assert row["unique_spectra"] == 2
    assert row["unique_masters"] == 2
    assert row["classes"] == 2
    assert row["class_cells"] == 4
    assert row["singleton_class_cells"] == 4


def test_support_histograms_baseline():
    result = audit_support(_baseline())

    histogram = result["class_cell_histogram"]
    assert list(histogram.columns) == ["masters_in_cell", "cells"]
    pd.testing.assert_frame_equal(
        histogram, pd.DataFrame({"masters_in_cell": [1], "cells": [4]})
    )

    master_domains = result["master_domain_histogram"]
    assert list(master_domains.columns) == ["domains_per_master", "masters"]
    pd.testing.assert_frame_equal(
        master_domains, pd.DataFrame({"domains_per_master": [1], "masters": [2]})
    )

    instruments = result["instrument_domains"]
    assert list(instruments.columns) == ["instrument", "domains", "stations"]
    pd.testing.assert_frame_equal(
        instruments, pd.DataFrame({"instrument": ["I1"], "domains": [1], "stations": [1]})
    )


def test_repeated_spectra_within_master_do_not_inflate_cell_size():
    rows = _frame(
        [
            ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C1", "D1", "S1", "I1", "M1", "O2", "cat"),
        ]
    )
    result = audit_support(rows)
    domain = result["domains"].iloc[0]
    assert domain["unique_spectra"] == 2
    assert domain["unique_masters"] == 1
    assert domain["class_cells"] == 1
    assert domain["singleton_class_cells"] == 1
    pd.testing.assert_frame_equal(
        result["class_cell_histogram"],
        pd.DataFrame({"masters_in_cell": [1], "cells": [1]}),
    )


def test_repeated_contexts_change_cells_but_not_population():
    rows = _frame(
        [
            ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C2", "D1", "S1", "I1", "M1", "O1", "cat"),
        ]
    )
    domain = audit_support(rows)["domains"].iloc[0]
    assert domain["contexts"] == 2
    assert domain["unique_spectra"] == 1
    assert domain["unique_masters"] == 1
    assert domain["classes"] == 1
    assert domain["class_cells"] == 2
    assert domain["singleton_class_cells"] == 2


def test_shared_master_across_instruments_raises_domain_histogram():
    rows = _frame(
        [
            ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C2", "D2", "S1", "I2", "M1", "O2", "cat"),
        ]
    )
    result = audit_support(rows)
    pd.testing.assert_frame_equal(
        result["master_domain_histogram"],
        pd.DataFrame({"domains_per_master": [2], "masters": [1]}),
    )
    pd.testing.assert_frame_equal(
        result["instrument_domains"],
        pd.DataFrame(
            {"instrument": ["I1", "I2"], "domains": [1, 1], "stations": [1, 1]}
        ),
    )
    assert list(result["domains"]["domain"]) == ["D1", "D2"]


def test_absent_classes_are_not_fabricated():
    rows = _frame([("C1", "D1", "S1", "I1", "M1", "O1", "cat")])
    result = audit_support(rows)
    domain = result["domains"].iloc[0]
    assert domain["classes"] == 1
    assert domain["class_cells"] == 1
    pd.testing.assert_frame_equal(
        result["class_cell_histogram"],
        pd.DataFrame({"masters_in_cell": [1], "cells": [1]}),
    )


def test_empty_class_probability_zero_when_single_master_pool():
    row = audit_support(_baseline())["empty_class_risk"].iloc[0]
    assert row["class_cells"] == 4
    assert row["expected_empty_class_cells"] == pytest.approx(0.0)
    assert row["minimum_cell_empty_probability"] == pytest.approx(0.0)
    assert row["maximum_cell_empty_probability"] == pytest.approx(0.0)


def test_empty_class_probability_matches_closed_form():
    rows = _frame(
        [
            ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C2", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C2", "D1", "S1", "I1", "M2", "O2", "cat"),
        ]
    )
    risk = audit_support(rows)["empty_class_risk"]
    assert list(risk.columns) == RISK_COLUMNS
    assert len(risk) == 1
    row = risk.iloc[0]
    assert row["class_cells"] == 2
    assert row["expected_empty_class_cells"] == pytest.approx(0.25)
    assert row["minimum_cell_empty_probability"] == pytest.approx(0.0)
    assert row["maximum_cell_empty_probability"] == pytest.approx(0.25)
    assert 0.0 <= row["minimum_cell_empty_probability"] <= 1.0
    assert 0.0 <= row["maximum_cell_empty_probability"] <= 1.0


def test_empty_class_probability_is_zero_when_pool_equals_cell_size():
    rows = _frame(
        [
            ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C1", "D1", "S1", "I1", "M2", "O2", "cat"),
            ("C2", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C2", "D1", "S1", "I1", "M2", "O2", "cat"),
        ]
    )
    row = audit_support(rows)["empty_class_risk"].iloc[0]
    assert row["class_cells"] == 2
    assert row["expected_empty_class_cells"] == pytest.approx(0.0)
    assert row["maximum_cell_empty_probability"] == pytest.approx(0.0)


def test_outcome_columns_are_ignored_and_private_values_never_exported():
    baseline = _baseline()
    augmented = baseline.assign(
        probability=[0.1, 0.9, 0.2, 0.8],
        score=[3.0, 1.0, 2.0, 0.0],
        predicted_label=["dog", "cat", "cat", "dog"],
    )
    expected = audit_support(baseline)
    observed = audit_support(augmented)
    assert list(observed) == list(expected)
    for name in expected:
        pd.testing.assert_frame_equal(observed[name], expected[name])

    private_columns = {"context_id", "observation_uid", "master_sample_id"}
    private_values = {"C1", "C2", "O1", "O2", "M1", "M2"}
    for table in expected.values():
        assert not (private_columns & set(table.columns))
        exported = set()
        for column in table.columns:
            exported.update(table[column].astype(str).tolist())
        assert not (private_values & exported)


def test_row_permutation_yields_identical_tables():
    baseline = _baseline()
    permuted = baseline.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    expected = audit_support(baseline)
    observed = audit_support(permuted)
    for name in expected:
        pd.testing.assert_frame_equal(observed[name], expected[name])


def test_input_is_not_mutated():
    rows = _baseline()
    snapshot = rows.copy(deep=True)
    audit_support(rows)
    pd.testing.assert_frame_equal(rows, snapshot)


def test_rejects_non_dataframe():
    with pytest.raises(SupportAuditError) as excinfo:
        audit_support(None)
    assert str(excinfo.value) == "input_not_dataframe"


def test_rejects_empty_frame():
    with pytest.raises(SupportAuditError) as excinfo:
        audit_support(_frame([]))
    assert str(excinfo.value) == "empty_frame"


def test_rejects_duplicate_columns():
    baseline = _baseline()
    duplicated = pd.concat([baseline, baseline[["true_label"]]], axis=1)
    with pytest.raises(SupportAuditError) as excinfo:
        audit_support(duplicated)
    assert str(excinfo.value) == "duplicate_columns"


def test_rejects_missing_columns():
    incomplete = _baseline().drop(columns=["true_label"])
    with pytest.raises(SupportAuditError) as excinfo:
        audit_support(incomplete)
    assert str(excinfo.value) == "missing_columns"


@pytest.mark.parametrize("column", COLUMNS)
@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, "missing_value"),
        (np.nan, "missing_value"),
        (1, "non_string_value"),
        (True, "non_string_value"),
        ("", "empty_value"),
        ("   ", "empty_value"),
        (" cat ", "untrimmed_value"),
    ],
)
def test_rejects_invalid_required_values(column, value, reason):
    rows = _baseline().astype(object)
    rows.loc[0, column] = value
    with pytest.raises(SupportAuditError) as excinfo:
        audit_support(rows)
    assert str(excinfo.value) == reason


def test_rejects_duplicate_context_observation_rows():
    baseline = _baseline()
    duplicated = pd.concat([baseline, baseline.iloc[[0]]], ignore_index=True)
    with pytest.raises(SupportAuditError) as excinfo:
        audit_support(duplicated)
    assert str(excinfo.value) == "duplicate_observation_rows"


def test_rejects_contradictory_observation_identity():
    rows = _baseline().astype(object)
    rows.loc[2, "master_sample_id"] = "M9"
    with pytest.raises(SupportAuditError) as excinfo:
        audit_support(rows)
    assert str(excinfo.value) == "contradictory_observation_identity"


def test_rejects_contradictory_master_identity():
    rows = _frame(
        [
            ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C1", "D1", "S1", "I1", "M1", "O2", "dog"),
        ]
    )
    with pytest.raises(SupportAuditError) as excinfo:
        audit_support(rows)
    assert str(excinfo.value) == "contradictory_master_identity"


def test_rejects_contradictory_context_identity():
    rows = _frame(
        [
            ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C1", "D2", "S1", "I2", "M2", "O2", "cat"),
        ]
    )
    with pytest.raises(SupportAuditError) as excinfo:
        audit_support(rows)
    assert str(excinfo.value) == "contradictory_context_identity"


def test_rejects_contradictory_domain_identity():
    rows = _frame(
        [
            ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
            ("C2", "D1", "S2", "I1", "M2", "O2", "cat"),
        ]
    )
    with pytest.raises(SupportAuditError) as excinfo:
        audit_support(rows)
    assert str(excinfo.value) == "contradictory_domain_identity"


def test_empty_class_risk_reports_expected_empty_cell_count_not_union():
    # Two singleton contexts in the SAME domain carrying the SAME class, with
    # distinct masters and distinct observations. Each context contributes one
    # class cell, so the class has two cells.
    records = [
        ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
        ("C2", "D1", "S1", "I1", "M2", "O2", "cat"),
    ]
    result = audit_support(_frame(records))
    risk = result["empty_class_risk"]

    rows = risk[
        (risk["domain"] == "D1")
        & (risk["station"] == "S1")
        & (risk["instrument"] == "I1")
    ]
    assert len(rows) == 1
    row = rows.iloc[0]

    assert int(row["class_cells"]) == 2
    # The audit reports an EXPECTED COUNT of empty cells, computed as the sum
    # of the per-cell empty probabilities: each of the two cells is empty with
    # probability (1/2)**2 = 0.25, so the expected count is 2 * 0.25 = 0.5.
    # This is deliberately NOT the independence-assumed union
    # 1 - (1 - 0.25)**2 = 0.4375: that value is a probability that at least
    # one cell is empty, not an expected number of empty cells.
    assert float(row["expected_empty_class_cells"]) == pytest.approx(0.5)
    assert float(row["minimum_cell_empty_probability"]) == pytest.approx(0.25)
    assert float(row["maximum_cell_empty_probability"]) == pytest.approx(0.25)
    # No random draws are involved; the risk is a closed-form expectation.


def test_instrument_domains_counts_distinct_domains_and_stations():
    # A single instrument I1 appears in domain D1 at station S1 and in domain
    # D2 at station S2, with distinct masters and observations.
    records = [
        ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
        ("C2", "D2", "S2", "I1", "M2", "O2", "cat"),
    ]
    result = audit_support(_frame(records))
    instrument_domains = result["instrument_domains"]

    assert len(instrument_domains) == 1
    row = instrument_domains.iloc[0]
    assert row["instrument"] == "I1"
    assert int(row["domains"]) == 2
    assert int(row["stations"]) == 2


def test_audit_support_is_permutation_invariant_for_multidomain_fixture():
    # One class per master and one station per master throughout; M1/M2/M3 are
    # deliberately shared across domains (including observation O1, which is
    # unique within a context but may repeat across contexts).
    records = [
        ("C1", "D1", "S1", "I1", "M1", "O1", "cat"),
        ("C1", "D1", "S1", "I1", "M1", "O2", "cat"),
        ("C1", "D1", "S1", "I1", "M2", "O3", "cat"),
        ("C1", "D1", "S1", "I1", "M3", "O4", "dog"),
        ("C2", "D1", "S1", "I1", "M1", "O1", "cat"),
        ("C2", "D1", "S1", "I1", "M3", "O4", "dog"),
        ("C3", "D2", "S1", "I2", "M1", "O5", "cat"),
        ("C3", "D2", "S1", "I2", "M2", "O6", "cat"),
        ("C3", "D2", "S1", "I2", "M3", "O7", "dog"),
        ("C4", "D3", "S2", "I1", "M4", "O8", "bird"),
    ]
    forward = audit_support(_frame(records))
    reversed_order = audit_support(_frame(list(reversed(records))))

    assert set(forward.keys()) == set(reversed_order.keys())
    for name in forward:
        pd.testing.assert_frame_equal(forward[name], reversed_order[name])
