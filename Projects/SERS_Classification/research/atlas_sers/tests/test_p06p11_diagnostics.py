"""Tests for the P06/P11 domain diagnostics and the G4 checklist."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.p05_comparison import PAIRS
from atlas_sers.evaluation.p06p11_diagnostics import (
    ALL_ENDPOINTS,
    EXPLORATORY_ENDPOINTS,
    PRIMARY_ENDPOINT,
    P06P11ChecklistError,
    P06P11DiagnosticsError,
    domain_diagnostics,
    g4_checklist,
)

PRIMARY_PAIR = (PRIMARY_ENDPOINT[0], PRIMARY_ENDPOINT[1])
OTHER_PAIRS = [pair for pair in PAIRS if pair != PRIMARY_PAIR]
PAIR_X = OTHER_PAIRS[0]
PAIR_Y = OTHER_PAIRS[1]


def _row(
    model,
    reference,
    aggregation,
    context_id,
    domain,
    station,
    instrument,
    complete,
    model_ba,
    reference_ba,
):
    return {
        "model_id": model,
        "reference_model_id": reference,
        "aggregation_id": aggregation,
        "context_id": context_id,
        "domain": domain,
        "station": station,
        "held_instrument": instrument,
        "common_complete": bool(complete),
        "model_balanced_accuracy": model_ba,
        "reference_balanced_accuracy": reference_ba,
        "delta_balanced_accuracy": (
            None if not complete else float(model_ba) - float(reference_ba)
        ),
    }


def _spec(instrument, station, contexts):
    return {"instrument": instrument, "station": station, "contexts": contexts}


def _endpoint_rows(pair, aggregation, domains):
    model, reference = pair
    rows = []
    for domain, spec in domains.items():
        for position, (complete, model_ba, reference_ba) in enumerate(spec["contexts"]):
            context_id = f"{domain}::{position}"
            rows.append(
                _row(
                    model,
                    reference,
                    aggregation,
                    context_id,
                    domain,
                    spec["station"],
                    spec["instrument"],
                    complete,
                    model_ba,
                    reference_ba,
                )
            )
    return rows


def _constant_domains(deltas):
    domains = {}
    for position, delta in enumerate(deltas):
        if delta >= 0:
            model_ba, reference_ba = 1.0, 1.0 - delta
        else:
            model_ba, reference_ba = 1.0 + delta, 1.0
        domains[f"DOM-{position}"] = _spec(
            f"INST-{position}", "STA-1", [(True, model_ba, reference_ba)]
        )
    return domains


def _summary_row(pair, aggregation, row):
    return row[
        row.model_id.eq(pair[0])
        & row.reference_model_id.eq(pair[1])
        & row.aggregation_id.eq(aggregation)
    ].iloc[0]


def _manual_flip_p(deltas, group_index, groups):
    observed = float(np.mean(deltas))
    hits = 0
    total = 0
    for combo in itertools.product((1.0, -1.0), repeat=groups):
        null = float(np.mean([combo[group_index[i]] * deltas[i] for i in range(len(deltas))]))
        if abs(null) >= abs(observed) - 1e-12:
            hits += 1
        total += 1
    return hits / total


def test_equal_domain_not_pooled_and_worst_difference():
    domains = {
        "DOM-A": _spec("INST-1", "STA-1", [(True, 0.6, 0.4)] * 3),
        "DOM-B": _spec("INST-2", "STA-1", [(True, 0.9, 0.1)]),
    }
    result = domain_diagnostics(pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)))
    row = _summary_row(PAIR_X, "M01", result["summary"])
    assert row.supported_domains == 2
    assert row.mean_delta == pytest.approx(0.5)  # equal-domain, pooled would be 0.35
    assert row.model_worst_domain_balanced_accuracy == pytest.approx(0.6)
    assert row.reference_worst_domain_balanced_accuracy == pytest.approx(0.1)
    assert row.worst_difference == pytest.approx(0.5)  # difference of minima
    assert row.min_delta == pytest.approx(0.2)  # differs from worst_difference


def test_incomplete_contexts_are_not_zero_imputed():
    domains = {
        "DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4), (False, None, None)]),
    }
    result = domain_diagnostics(pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)))
    row = _summary_row(PAIR_X, "M01", result["summary"])
    assert row.complete_contexts == 1
    assert row.planned_contexts == 2
    assert row.model_mean_balanced_accuracy == pytest.approx(0.8)
    assert row.mean_delta == pytest.approx(0.4)


def test_incomplete_rows_may_hold_nan_metrics():
    domains = {
        "DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4), (False, 0.9, 0.5)]),
    }
    rows = _endpoint_rows(PAIR_X, "M01", domains)
    rows[1]["model_balanced_accuracy"] = np.nan
    rows[1]["reference_balanced_accuracy"] = np.nan
    rows[1]["delta_balanced_accuracy"] = np.nan
    result = domain_diagnostics(pd.DataFrame(rows))
    row = _summary_row(PAIR_X, "M01", result["summary"])
    assert row.complete_contexts == 1
    assert row.mean_delta == pytest.approx(0.4)


@pytest.mark.parametrize(
    "column,value",
    [
        ("model_balanced_accuracy", "0.8"),
        ("model_balanced_accuracy", True),
        ("model_balanced_accuracy", complex(0.8, 0.0)),
        ("reference_balanced_accuracy", "0.4"),
        ("reference_balanced_accuracy", False),
        ("reference_balanced_accuracy", complex(0.4, 0.0)),
        ("delta_balanced_accuracy", "0.4"),
        ("delta_balanced_accuracy", True),
        ("delta_balanced_accuracy", complex(0.4, 0.0)),
    ],
)
def test_non_real_metrics_are_rejected(column, value):
    domains = {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4)])}
    frame = pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)).astype(object)
    frame.at[0, column] = value
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(frame)


def test_domain_without_complete_rows_is_reported_missing():
    domains = {
        "DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4)]),
        "DOM-B": _spec("INST-2", "STA-1", [(False, None, None)]),
    }
    result = domain_diagnostics(pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)))
    table = result["domain_metrics"]
    missing = table[table.domain.eq("DOM-B")].iloc[0]
    assert missing.reason_code == "no_common_complete_contexts"
    assert np.isnan(missing.model_ba)
    row = _summary_row(PAIR_X, "M01", result["summary"])
    assert row.supported_domains == 1
    assert row.planned_domains == 2


def test_leave_one_out_domain_and_shared_instrument():
    domains = {
        "DOM-A": _spec("INST-1", "STA-1", [(True, 0.6, 0.2)]),  # delta 0.4
        "DOM-B": _spec("INST-1", "STA-1", [(True, 0.8, 0.2)]),  # delta 0.6
        "DOM-C": _spec("INST-2", "STA-1", [(True, 0.9, 0.1)]),  # delta 0.8
    }
    result = domain_diagnostics(pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)))
    table = result["leave_one_out"]
    instruments = table[table.exclusion_type.eq("instrument")].set_index("exclusion_id")
    assert instruments.loc["INST-1"].removed_domains == 2
    assert instruments.loc["INST-1"].remaining_domains == 1
    assert instruments.loc["INST-1"].delta_after_exclusion == pytest.approx(0.8)
    domains_table = table[table.exclusion_type.eq("domain")].set_index("exclusion_id")
    assert domains_table.loc["DOM-C"].remaining_domains == 2
    assert domains_table.loc["DOM-C"].delta_after_exclusion == pytest.approx(0.5)
    assert domains_table.loc["DOM-A"].delta_after_exclusion == pytest.approx(0.7)


def test_leave_one_out_without_residual_domains():
    domains = {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.7, 0.3)])}
    result = domain_diagnostics(pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)))
    table = result["leave_one_out"]
    row = table[
        (table.exclusion_type == "domain") & (table.exclusion_id == "DOM-A")
    ].iloc[0]
    assert row.remaining_domains == 0
    assert row.reason_code == "no_residual_domains"
    assert np.isnan(row.delta_after_exclusion)
    assert np.isnan(row.model_mean_after_exclusion)


def test_sign_flip_domain_and_instrument_manual_enumeration():
    domains = _constant_domains([1.0, 1.0, 1.0])
    result = domain_diagnostics(pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)))
    table = result["sign_flip"]
    cell = table[table.model_id.eq(PAIR_X[0]) & table.reference_model_id.eq(PAIR_X[1])]
    domain_row = cell[cell.kind.eq("domain")].iloc[0]
    deltas = np.array([1.0, 1.0, 1.0])
    assert domain_row.number_groups == 3
    assert domain_row.assignments == 8
    assert domain_row.p_descriptive == pytest.approx(_manual_flip_p(deltas, list(range(3)), 3))
    instrument_row = cell[cell.kind.eq("instrument")].iloc[0]
    assert instrument_row.p_descriptive == pytest.approx(
        _manual_flip_p(deltas, list(range(3)), 3)
    )
    label = "symmetry_sensitivity_shared_masters_not_independent"
    assert domain_row.assumption_label == label


def test_sign_flip_shared_instrument_uses_one_sign():
    domains = {
        "DOM-A": _spec("INST-1", "STA-1", [(True, 1.0, 0.6)]),  # delta 0.4
        "DOM-B": _spec("INST-1", "STA-1", [(True, 1.0, 0.4)]),  # delta 0.6
    }
    result = domain_diagnostics(pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)))
    table = result["sign_flip"]
    cell = table[table.model_id.eq(PAIR_X[0]) & table.reference_model_id.eq(PAIR_X[1])]
    domain_row = cell[cell.kind.eq("domain")].iloc[0]
    instrument_row = cell[cell.kind.eq("instrument")].iloc[0]
    deltas = np.array([0.4, 0.6])
    assert domain_row.number_groups == 2
    assert instrument_row.number_groups == 1
    assert instrument_row.assignments == 2
    assert domain_row.p_descriptive == pytest.approx(_manual_flip_p(deltas, [0, 1], 2))
    assert instrument_row.p_descriptive == pytest.approx(_manual_flip_p(deltas, [0, 0], 1))


def test_reverse_contrast_coherently_negates_scores():
    domains = {
        "DOM-A": _spec("INST-1", "STA-1", [(True, 0.6, 0.4)]),
        "DOM-B": _spec("INST-2", "STA-1", [(True, 0.9, 0.1)]),
    }
    base = domain_diagnostics(pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)))
    base_row = _summary_row(PAIR_X, "M01", base["summary"])

    swapped = {}
    for name, spec in domains.items():
        contexts = [
            (complete, reference_ba, model_ba)
            for complete, model_ba, reference_ba in spec["contexts"]
        ]
        swapped[name] = _spec(spec["instrument"], spec["station"], contexts)
    reversed_result = domain_diagnostics(
        pd.DataFrame(_endpoint_rows(PAIR_X, "M01", swapped))
    )
    reversed_row = _summary_row(PAIR_X, "M01", reversed_result["summary"])
    assert reversed_row.mean_delta == pytest.approx(-base_row.mean_delta)
    assert reversed_row.model_mean_balanced_accuracy == pytest.approx(
        base_row.reference_mean_balanced_accuracy
    )
    assert reversed_row.worst_difference == pytest.approx(-base_row.worst_difference)


def test_holm_family_size_and_monotonic_adjustment():
    rows = _endpoint_rows(PAIR_X, "M01", _constant_domains([1.0] * 7))
    rows += _endpoint_rows(PAIR_Y, "M01", _constant_domains([1.0] * 3))
    result = domain_diagnostics(pd.DataFrame(rows))
    table = result["sign_flip"]
    assert len(table) == 2 * len(ALL_ENDPOINTS)
    first = table[
        table.model_id.eq(PAIR_X[0])
        & table.reference_model_id.eq(PAIR_X[1])
        & table.kind.eq("domain")
    ].iloc[0]
    second = table[
        table.model_id.eq(PAIR_Y[0])
        & table.reference_model_id.eq(PAIR_Y[1])
        & table.kind.eq("domain")
    ].iloc[0]
    assert first.p_descriptive == pytest.approx(1 / 64)
    assert second.p_descriptive == pytest.approx(0.25)
    assert first.p_holm == pytest.approx(min(1.0, 33 / 64))
    assert second.p_holm == pytest.approx(1.0)
    assert first.p_holm <= second.p_holm
    assert first.family == "exploratory_holm"
    assert first.assumption_label == "symmetry_sensitivity_shared_masters_not_independent"


def test_primary_sign_flip_is_unadjusted():
    rows = _endpoint_rows(PRIMARY_PAIR, "M01", _constant_domains([1.0, 1.0]))
    result = domain_diagnostics(pd.DataFrame(rows))
    table = result["sign_flip"]
    primary = table[
        table.model_id.eq(PRIMARY_ENDPOINT[0])
        & table.reference_model_id.eq(PRIMARY_ENDPOINT[1])
        & table.aggregation_id.eq(PRIMARY_ENDPOINT[2])
    ]
    assert len(primary) == 2
    assert (primary.family == "primary_unadjusted_descriptive").all()
    assert primary.p_holm.isna().all()


def test_aggregate_outputs_expose_no_private_identifiers():
    domains = {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4)])}
    result = domain_diagnostics(pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)))
    for table in result.values():
        for column in table.columns:
            assert "context_id" not in column
            assert "master" not in column
            assert "observation" not in column


def test_input_is_not_mutated_and_row_order_is_deterministic():
    domains = {
        "DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4), (False, None, None)]),
        "DOM-B": _spec("INST-2", "STA-1", [(True, 0.9, 0.5)]),
    }
    frame = pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains))
    snapshot = frame.copy(deep=True)
    first = domain_diagnostics(frame)
    pd.testing.assert_frame_equal(frame, snapshot)
    shuffled = pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)).sample(
        frac=1.0, random_state=0
    ).reset_index(drop=True)
    second = domain_diagnostics(shuffled)
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])


def test_empty_or_truncated_input_is_rejected():
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame())
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame([{"model_id": "only"}]))


def test_duplicate_key_is_rejected():
    domains = {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4)])}
    rows = _endpoint_rows(PAIR_X, "M01", domains)
    rows.append(dict(rows[0]))
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame(rows))


def test_duplicate_headers_are_rejected():
    domains = {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4)])}
    frame = pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains))
    duplicated = pd.concat([frame, frame[["domain"]]], axis=1)
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(duplicated)

    summary = pd.DataFrame([_primary_summary()])
    with pytest.raises(P06P11ChecklistError):
        g4_checklist(
            pd.concat([summary, summary[["mean_delta"]]], axis=1),
            pd.DataFrame([_hierarchical_interval()]),
            input_preservation_verified=True,
        )

    intervals = pd.DataFrame([_hierarchical_interval()])
    with pytest.raises(P06P11ChecklistError):
        g4_checklist(
            pd.DataFrame([_primary_summary()]),
            pd.concat([intervals, intervals[["lower"]]], axis=1),
            input_preservation_verified=True,
        )


def test_unknown_pair_and_aggregation_are_rejected():
    domains = {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4)])}
    rows = _endpoint_rows(PAIR_X, "M01", domains)
    rows[0]["model_id"] = "NOT-A-MODEL"
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame(rows))
    rows = _endpoint_rows(PAIR_X, "M01", domains)
    rows[0]["aggregation_id"] = "M99"
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame(rows))


def test_malformed_flag_string_and_delta_are_rejected():
    domains = {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4)])}
    rows = _endpoint_rows(PAIR_X, "M01", domains)
    rows[0]["common_complete"] = 1
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame(rows))

    rows = _endpoint_rows(PAIR_X, "M01", domains)
    rows[0]["domain"] = " DOM-A "
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame(rows))

    rows = _endpoint_rows(PAIR_X, "M01", domains)
    rows[0]["delta_balanced_accuracy"] = 0.99
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame(rows))

    rows = _endpoint_rows(PAIR_X, "M01", domains)
    rows[0]["model_balanced_accuracy"] = 1.5
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame(rows))


def test_context_domain_and_endpoint_consistency_are_enforced():
    rows = _endpoint_rows(
        PAIR_X, "M01", {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4)])}
    )
    other = _endpoint_rows(
        PAIR_X, "M06", {"DOM-B": _spec("INST-1", "STA-1", [(True, 0.8, 0.4)])}
    )
    rows[0]["context_id"] = other[0]["context_id"]
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame(rows + other))

    rows = _endpoint_rows(
        PAIR_X, "M01", {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4)])}
    )
    other = _endpoint_rows(
        PAIR_X, "M06", {"DOM-A": _spec("INST-2", "STA-1", [(True, 0.8, 0.4)])}
    )
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame(rows + other))

    rows = _endpoint_rows(
        PAIR_X,
        "M01",
        {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4), (False, None, None)])},
    )
    other = _endpoint_rows(
        PAIR_X,
        "M06",
        {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4), (True, 0.9, 0.5)])},
    )
    with pytest.raises(P06P11DiagnosticsError):
        domain_diagnostics(pd.DataFrame(rows + other))


def test_matching_endpoints_with_equal_common_contexts_are_accepted():
    domains = {
        "DOM-A": _spec("INST-1", "STA-1", [(True, 0.8, 0.4), (True, 0.6, 0.2)]),
    }
    rows = _endpoint_rows(PAIR_X, "M01", domains)
    rows += _endpoint_rows(PAIR_X, "M06", domains)
    result = domain_diagnostics(pd.DataFrame(rows))
    summary = result["summary"]
    for aggregation in ("M01", "M06"):
        row = summary[
            summary.model_id.eq(PAIR_X[0])
            & summary.reference_model_id.eq(PAIR_X[1])
            & summary.aggregation_id.eq(aggregation)
        ].iloc[0]
        assert row.supported_domains == 1
        assert row.mean_delta == pytest.approx(0.4)


def _primary_summary(**overrides):
    row = {
        "model_id": PRIMARY_ENDPOINT[0],
        "reference_model_id": PRIMARY_ENDPOINT[1],
        "aggregation_id": PRIMARY_ENDPOINT[2],
        "supported_domains": 13,
        "planned_domains": 13,
        "mean_delta": 0.05,
        "positive_domains": 9,
        "worst_difference": 0.0,
    }
    row.update(overrides)
    return row


def _hierarchical_interval(**overrides):
    row = {
        "model_id": PRIMARY_ENDPOINT[0],
        "reference_model_id": PRIMARY_ENDPOINT[1],
        "aggregation_id": PRIMARY_ENDPOINT[2],
        "method": "hierarchical",
        "lower": 0.01,
        "upper": 0.09,
        "reason_code": "ok",
    }
    row.update(overrides)
    return row


def _run_g4(summary=None, intervals=None, **kwargs):
    kwargs.setdefault("input_preservation_verified", True)
    if summary is None:
        summary = _primary_summary()
    if intervals is None:
        intervals = [_hierarchical_interval()]
    return g4_checklist(pd.DataFrame([summary]), pd.DataFrame(intervals), **kwargs)


def _find(result, name):
    criteria = result["criteria"]
    return criteria[criteria.criterion.eq(name)].iloc[0]


def test_g4_all_criteria_supported():
    result = _run_g4(t1_difference=0.0, t1_provenance_verified=True)
    assert list(result["criteria"].status) == ["supported"] * 6
    assert result["decision"] == {
        "promote": True,
        "status": "supported",
        "supported_criteria": 6,
        "failed_criteria": 0,
        "unassessable_criteria": 0,
    }


@pytest.mark.parametrize(
    "overrides,name,expected",
    [
        ({"mean_delta": 0.03}, "mean_delta_at_least_0.03", "supported"),
        ({"mean_delta": 0.0299}, "mean_delta_at_least_0.03", "failed"),
        ({"positive_domains": 8}, "positive_domains_at_least_8", "supported"),
        ({"positive_domains": 7}, "positive_domains_at_least_8", "failed"),
        (
            {"worst_difference": -0.03},
            "worst_domain_difference_at_least_neg_0.03",
            "supported",
        ),
        (
            {"worst_difference": -0.0301},
            "worst_domain_difference_at_least_neg_0.03",
            "failed",
        ),
    ],
)
def test_g4_numeric_boundaries(overrides, name, expected):
    result = _run_g4(_primary_summary(**overrides))
    assert _find(result, name).status == expected


@pytest.mark.parametrize(
    "supported,planned,positive",
    [
        (8, 8, 8),
        (12, 12, 12),
        (14, 14, 14),
        (13.5, 13.5, 8),
        (13, 12, 8),
    ],
)
def test_g4_requires_exact_primary_domain_support(supported, planned, positive):
    summary = _primary_summary(
        supported_domains=supported,
        planned_domains=planned,
        positive_domains=positive,
    )
    result = _run_g4(summary, t1_difference=0.0, t1_provenance_verified=True)
    for name in (
        "mean_delta_at_least_0.03",
        "positive_domains_at_least_8",
        "worst_domain_difference_at_least_neg_0.03",
    ):
        assert _find(result, name).status == "unassessable"
    assert result["decision"]["promote"] is False


@pytest.mark.parametrize("positive", [-1, 8.5, 14])
def test_g4_invalid_positive_domains_cannot_promote(positive):
    summary = _primary_summary(positive_domains=positive)
    result = _run_g4(summary, t1_difference=0.0, t1_provenance_verified=True)
    criterion = _find(result, "positive_domains_at_least_8")
    assert criterion.status == "unassessable"
    assert criterion.reason_code == "positive_domains_invalid"
    assert result["decision"]["promote"] is False


def test_g4_interval_lower_is_strictly_positive():
    result = _run_g4(intervals=[_hierarchical_interval(lower=0.0, upper=0.5)])
    assert _find(result, "hierarchical_interval_lower_gt_0").status == "failed"
    result = _run_g4(intervals=[_hierarchical_interval(lower=0.001, upper=0.5)])
    assert _find(result, "hierarchical_interval_lower_gt_0").status == "supported"


def test_g4_weighted_interval_cannot_substitute():
    intervals = [dict(_hierarchical_interval(method="crossed_weight", lower=0.5, upper=0.9))]
    result = _run_g4(intervals=intervals)
    assert _find(result, "hierarchical_interval_lower_gt_0").status == "unassessable"
    assert result["decision"]["promote"] is False


def test_g4_interval_reason_and_bounds_are_checked():
    result = _run_g4(intervals=[_hierarchical_interval(reason_code="missing")])
    criterion = _find(result, "hierarchical_interval_lower_gt_0")
    assert criterion.status == "unassessable"
    assert criterion.reason_code == "hierarchical_interval_unavailable"
    result = _run_g4(intervals=[_hierarchical_interval(reason_code="degenerate_distribution")])
    assert _find(result, "hierarchical_interval_lower_gt_0").status == "supported"
    result = _run_g4(intervals=[_hierarchical_interval(lower=0.5, upper=0.1)])
    assert _find(result, "hierarchical_interval_lower_gt_0").status == "unassessable"


def test_g4_interval_bounds_range_is_enforced():
    result = _run_g4(intervals=[_hierarchical_interval(lower=-1.5, upper=0.1)])
    criterion = _find(result, "hierarchical_interval_lower_gt_0")
    assert criterion.status == "unassessable"
    assert criterion.reason_code == "hierarchical_interval_bounds_out_of_range"
    assert result["decision"]["promote"] is False
    result = _run_g4(intervals=[_hierarchical_interval(lower=0.1, upper=1.5)])
    assert _find(result, "hierarchical_interval_lower_gt_0").status == "unassessable"


def test_g4_incomplete_domain_support_is_unassessable():
    result = _run_g4(_primary_summary(supported_domains=12, planned_domains=13))
    for name in (
        "mean_delta_at_least_0.03",
        "positive_domains_at_least_8",
        "worst_domain_difference_at_least_neg_0.03",
    ):
        assert _find(result, name).status == "unassessable"


def test_g4_missing_metric_is_unassessable():
    summary = _primary_summary()
    del summary["mean_delta"]
    result = _run_g4(summary)
    assert _find(result, "mean_delta_at_least_0.03").status == "unassessable"


def test_g4_t1_requirements():
    result = _run_g4(t1_difference=0.0, t1_provenance_verified=False)
    assert _find(result, "t1_difference_at_least_neg_0.02").status == "unassessable"
    result = _run_g4(t1_difference=None, t1_provenance_verified=True)
    assert _find(result, "t1_difference_at_least_neg_0.02").status == "unassessable"
    result = _run_g4(t1_difference=-0.02, t1_provenance_verified=True)
    assert _find(result, "t1_difference_at_least_neg_0.02").status == "supported"
    result = _run_g4(t1_difference=-0.0201, t1_provenance_verified=True)
    assert _find(result, "t1_difference_at_least_neg_0.02").status == "failed"


def test_g4_input_preservation_and_unassessable_decision():
    result = _run_g4(input_preservation_verified=False)
    assert _find(result, "input_preservation_verified").status == "failed"
    assert result["decision"]["status"] == "failed"
    result = _run_g4(t1_difference=0.0, t1_provenance_verified=False)
    assert result["decision"]["status"] == "unassessable"
    assert result["decision"]["promote"] is False
    assert result["decision"]["unassessable_criteria"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_preservation_verified": 1},
        {"input_preservation_verified": "yes"},
        {"input_preservation_verified": True, "t1_provenance_verified": 1},
        {"input_preservation_verified": True, "t1_difference": float("nan")},
        {"input_preservation_verified": True, "t1_difference": True},
        {"input_preservation_verified": True, "t1_difference": "0.1"},
    ],
)
def test_g4_rejects_bogus_inputs(kwargs):
    with pytest.raises(P06P11ChecklistError):
        g4_checklist(
            pd.DataFrame([_primary_summary()]),
            pd.DataFrame([_hierarchical_interval()]),
            **kwargs,
        )


def test_g4_requires_a_unique_primary_row_and_interval_key():
    with pytest.raises(P06P11ChecklistError):
        g4_checklist(
            pd.DataFrame([_primary_summary(), _primary_summary()]),
            pd.DataFrame([_hierarchical_interval()]),
            input_preservation_verified=True,
        )
    with pytest.raises(P06P11ChecklistError):
        g4_checklist(
            pd.DataFrame([_primary_summary()]),
            pd.DataFrame([_hierarchical_interval(), _hierarchical_interval()]),
            input_preservation_verified=True,
        )


def test_minimal_single_domain_fixture_is_valid():
    domains = {"DOM-A": _spec("INST-1", "STA-1", [(True, 0.7, 0.5)])}
    result = domain_diagnostics(pd.DataFrame(_endpoint_rows(PAIR_X, "M01", domains)))
    table = result["sign_flip"]
    assert len(table) == 2 * len(ALL_ENDPOINTS)
    assert len(EXPLORATORY_ENDPOINTS) == len(ALL_ENDPOINTS) - 1
    primary = table[
        table.model_id.eq(PRIMARY_ENDPOINT[0])
        & table.reference_model_id.eq(PRIMARY_ENDPOINT[1])
        & table.aggregation_id.eq(PRIMARY_ENDPOINT[2])
    ]
    assert (primary.family == "primary_unadjusted_descriptive").all()
