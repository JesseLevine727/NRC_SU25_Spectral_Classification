"""Narrow regression tests for numerical-zero tie classification.

These tests pin the deterministic reporting fix in
``atlas_sers.evaluation.p06p11_diagnostics._summary_table``: deltas that are
numerically zero (tiny floating-point residues such as -2.22e-16) must be
reported as ties using the existing 1e-12 score-reproduction absolute
tolerance. They do not exercise, rerun or resample bootstrap data.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p06p11_diagnostics import _summary_table

TOL = 1e-12


def _record(domain: str, delta: float) -> dict:
    return {
        "model_id": "D3",
        "reference_model_id": "D0-M",
        "aggregation_id": "M01",
        "domain": domain,
        "station": "S01",
        "held_instrument": "H1",
        "planned_contexts": 1,
        "complete_contexts": 1,
        "model_ba": 0.5 + delta / 2.0,
        "reference_ba": 0.5 - delta / 2.0,
        "delta": delta,
        "reason_code": "ok",
    }


def _frame(deltas) -> pd.DataFrame:
    return pd.DataFrame([_record(f"domain_{i}", d) for i, d in enumerate(deltas)])


def _value(summary, name: str):
    if isinstance(summary, pd.DataFrame):
        return summary.iloc[0][name]
    return summary[name]


def _counts(summary) -> dict:
    return {
        "positive": int(_value(summary, "positive_domains")),
        "negative": int(_value(summary, "negative_domains")),
        "tied": int(_value(summary, "tied_domains")),
    }


def test_tiny_residues_are_ties():
    summary = _summary_table(_frame([1e-16, -2.22e-16, 0.0, 1e-4, -1e-4]))
    assert _counts(summary) == {"positive": 1, "negative": 1, "tied": 3}


def test_counts_sum_to_domain_count():
    deltas = [1e-16, -2.22e-16, 0.0, 1e-4, -1e-4]
    counts = _counts(_summary_table(_frame(deltas)))
    assert counts["positive"] + counts["negative"] + counts["tied"] == len(deltas)


def test_boundary_tolerance_and_nextafter():
    above = math.nextafter(TOL, math.inf)
    below = math.nextafter(-TOL, -math.inf)
    deltas = [TOL, -TOL, above, below, 0.0]
    counts = _counts(_summary_table(_frame(deltas)))
    assert counts["tied"] == 3
    assert counts["positive"] == 1
    assert counts["negative"] == 1


def test_true_gains_keep_strict_sign():
    counts = _counts(_summary_table(_frame([0.02, -0.02])))
    assert counts == {"positive": 1, "negative": 1, "tied": 0}


def test_source_generated_difference_is_tie():
    delta = 0.9916666666666665 - 0.9916666666666667
    assert delta != 0.0
    assert abs(delta) <= TOL
    counts = _counts(_summary_table(_frame([delta])))
    assert counts == {"positive": 0, "negative": 0, "tied": 1}


def test_input_is_not_mutated():
    frame = _frame([1e-16, -2.22e-16, 0.0, 1e-4, -1e-4])
    before = frame.copy(deep=True)
    _summary_table(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_summary_statistics_match_numpy_exactly():
    deltas = [1e-16, -2.22e-16, 0.0, 1e-4, -1e-4]
    row = _summary_table(_frame(deltas)).iloc[0]
    values = np.array(deltas, dtype=np.float64)

    # The summary is expected to reuse numpy's own algorithms, so the raw
    # statistics must be bit-for-bit identical: rtol=0, atol=0, no rounding.
    # (Python's builtin sum accumulates in a different order than np.mean.)
    assert row["mean_delta"] == np.mean(values)
    assert row["median_delta"] == np.median(values)
    assert row["q25_delta"] == np.quantile(values, 0.25)
    assert row["q75_delta"] == np.quantile(values, 0.75)
    assert row["min_delta"] == np.min(values)
    assert row["max_delta"] == np.max(values)

    for name in (
        "model_worst_domain_balanced_accuracy",
        "reference_worst_domain_balanced_accuracy",
        "worst_difference",
    ):
        assert np.isfinite(row[name])
