"""Tests for atlas_sers.evaluation.p06p11_release_tables."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from atlas_sers.evaluation import p06p11_release_tables as subject

PRIMARY = "P05-SELECTED"
REFERENCE = "C-SELECTED"
SECONDARY_MODEL = "D3"
SECONDARY_REFERENCE = "D0-M"
OTHER_MODEL = "P05-SELECTED"
OTHER_REFERENCE = "D0-ERM"


def _row(model, reference, endpoint, domain, contexts, model_ba, reference_ba, delta):
    return {
        "model_id": model,
        "reference_model_id": reference,
        "aggregation_id": endpoint,
        "domain": domain,
        "complete_contexts": contexts,
        "model_ba": model_ba,
        "reference_ba": reference_ba,
        "delta": delta,
    }


def _fixture():
    summary = pd.DataFrame(
        [
            {
                "model_id": PRIMARY,
                "reference_model_id": REFERENCE,
                "aggregation_id": "M01",
                "supported_domains": 2,
                "positive_domains": 1,
                "negative_domains": 1,
                "tied_domains": 0,
                "note": "primary-m01",
            },
            {
                "model_id": PRIMARY,
                "reference_model_id": REFERENCE,
                "aggregation_id": "M06",
                "supported_domains": 2,
                "positive_domains": 2,
                "negative_domains": 0,
                "tied_domains": 0,
                "note": "primary-m06",
            },
            {
                "model_id": SECONDARY_MODEL,
                "reference_model_id": SECONDARY_REFERENCE,
                "aggregation_id": "M01",
                "supported_domains": 2,
                "positive_domains": 1,
                "negative_domains": 1,
                "tied_domains": 0,
                "note": "secondary-d3",
            },
            {
                "model_id": OTHER_MODEL,
                "reference_model_id": OTHER_REFERENCE,
                "aggregation_id": "M01",
                "supported_domains": 2,
                "positive_domains": 1,
                "negative_domains": 1,
                "tied_domains": 0,
                "note": "secondary-p05-vs-erm",
            },
        ]
    )
    domain_metrics = pd.DataFrame(
        [
            _row(PRIMARY, REFERENCE, "M01", "D1", 10, 0.80, 0.70, 0.10),
            _row(PRIMARY, REFERENCE, "M01", "D2", 12, 0.60, 0.75, -0.15),
            _row(PRIMARY, REFERENCE, "M06", "D1", 8, 0.70, 0.65, 0.05),
            _row(PRIMARY, REFERENCE, "M06", "D2", 9, 0.80, 0.70, 0.10),
            _row(SECONDARY_MODEL, SECONDARY_REFERENCE, "M01", "D1", 7, 0.55, 0.50, 0.05),
            _row(SECONDARY_MODEL, SECONDARY_REFERENCE, "M01", "D2", 6, 0.40, 0.40, -2e-16),
            _row(OTHER_MODEL, OTHER_REFERENCE, "M01", "D1", 5, 0.50, 0.50, -1e-16),
            _row(OTHER_MODEL, OTHER_REFERENCE, "M01", "D2", 4, 0.60, 0.50, 0.10),
        ]
    )
    tables = {
        "domain_metrics": domain_metrics,
        "summary": summary,
        "intervals": pd.DataFrame(
            [{"aggregation_id": "M01", "domain": "D1", "lower": 0.1, "upper": 0.9}]
        ),
        "feasibility": pd.DataFrame([{"model_id": PRIMARY, "feasible": True}]),
        "leave_one_out": pd.DataFrame([{"model_id": REFERENCE, "loo": 0.3}]),
        "sign_flip": pd.DataFrame([{"model_id": SECONDARY_MODEL, "sign_flips": 2}]),
        "g4_criteria": pd.DataFrame([{"criterion": "g4", "passed": True}]),
    }
    panels = {"M01": pd.DataFrame({"x": [1.0]}), "M06": pd.DataFrame({"x": [2.0]})}
    return tables, panels


def _metrics_domain():
    rows = [
        ("primary_common", "M01", PRIMARY, "D1", 0.80),
        ("primary_common", "M01", PRIMARY, "D2", 0.60),
        ("primary_common", "M01", REFERENCE, "D1", 0.70),
        ("primary_common", "M01", REFERENCE, "D2", 0.75),
        ("primary_common", "M06", PRIMARY, "D1", 0.70),
        ("primary_common", "M06", PRIMARY, "D2", 0.80),
        ("primary_common", "M06", REFERENCE, "D1", 0.65),
        ("primary_common", "M06", REFERENCE, "D2", 0.70),
        ("full_support", "M01", SECONDARY_MODEL, "D1", 0.55),
        ("full_support", "M01", SECONDARY_MODEL, "D2", 0.40),
        ("full_support", "M01", SECONDARY_REFERENCE, "D1", 0.50),
        ("full_support", "M01", SECONDARY_REFERENCE, "D2", 0.40),
        ("full_support", "M01", OTHER_MODEL, "D1", 0.50),
        ("full_support", "M01", OTHER_MODEL, "D2", 0.60),
        ("full_support", "M01", OTHER_REFERENCE, "D1", 0.50),
        ("full_support", "M01", OTHER_REFERENCE, "D2", 0.50),
    ]
    return pd.DataFrame(
        rows,
        columns=["scope", "aggregation_id", "model_id", "domain", "balanced_accuracy"],
    )


def _metrics_payload(domain=None):
    return {
        "domain_metrics": _metrics_domain() if domain is None else domain,
        "model_summary": pd.DataFrame(),
        "confusion": pd.DataFrame(),
        "class_sensitivity": pd.DataFrame(),
        "reliability_bins": pd.DataFrame(),
        "reliability_summary": pd.DataFrame(),
    }


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(subject, "build_metrics", lambda panels: _metrics_payload())


def test_prepare_tables_success(patched):
    tables, panels = _fixture()
    original = copy.deepcopy(tables)

    result = subject.prepare_tables(tables, panels)

    assert set(result) == {
        "inference_tables",
        "metrics",
        "tie_corrections",
        "crosscheck_count",
        "crosscheck_max_error",
    }
    assert set(result["metrics"]) == {
        "domain_metrics",
        "model_summary",
        "confusion",
        "class_sensitivity",
        "reliability_bins",
        "reliability_summary",
    }
    inference = result["inference_tables"]
    assert set(inference) == set(tables)
    for key in subject.TABLE_KEYS:
        if key != "summary":
            pd.testing.assert_frame_equal(inference[key], tables[key])

    corrected = inference["summary"]
    assert list(corrected["note"]) == list(tables["summary"]["note"])

    def counts(model, reference, endpoint):
        row = corrected[
            (corrected["model_id"] == model)
            & (corrected["reference_model_id"] == reference)
            & (corrected["aggregation_id"] == endpoint)
        ].iloc[0]
        return (row["positive_domains"], row["negative_domains"], row["tied_domains"])

    assert counts(PRIMARY, REFERENCE, "M01") == (1, 1, 0)
    assert counts(PRIMARY, REFERENCE, "M06") == (2, 0, 0)
    assert counts(SECONDARY_MODEL, SECONDARY_REFERENCE, "M01") == (1, 0, 1)
    assert counts(OTHER_MODEL, OTHER_REFERENCE, "M01") == (1, 0, 1)

    tie_corrections = result["tie_corrections"]
    assert list(tie_corrections.columns) == list(subject.TIE_CORRECTION_COLUMNS)
    assert len(tie_corrections) == 2
    bug_case = tie_corrections[
        (tie_corrections["model_id"] == OTHER_MODEL)
        & (tie_corrections["reference_model_id"] == OTHER_REFERENCE)
    ].iloc[0]
    assert (bug_case["before_positive"], bug_case["before_negative"], bug_case["before_tied"]) == (
        1,
        1,
        0,
    )
    assert (bug_case["after_positive"], bug_case["after_negative"], bug_case["after_tied"]) == (
        1,
        0,
        1,
    )

    assert result["crosscheck_count"] == 16
    assert result["crosscheck_max_error"] == 0.0

    for key in tables:
        pd.testing.assert_frame_equal(tables[key], original[key])


@pytest.mark.parametrize("endpoint", ["M01", "M06"])
def test_primary_count_change_rejected_both_endpoints(patched, endpoint):
    tables, panels = _fixture()
    mask = (tables["summary"]["model_id"] == PRIMARY) & (
        tables["summary"]["aggregation_id"] == endpoint
    )
    tables["summary"].loc[mask, "tied_domains"] = 1
    with pytest.raises(ValueError):
        subject.prepare_tables(tables, panels)


def test_duplicate_summary_key_rejected(patched):
    tables, panels = _fixture()
    duplicate = tables["summary"].iloc[[1]].copy()
    tables["summary"] = pd.concat([tables["summary"], duplicate], ignore_index=True)
    with pytest.raises(ValueError):
        subject.prepare_tables(tables, panels)


def test_duplicate_domain_key_rejected(patched):
    tables, panels = _fixture()
    duplicate = tables["domain_metrics"].iloc[[2]].copy()
    tables["domain_metrics"] = pd.concat([tables["domain_metrics"], duplicate], ignore_index=True)
    with pytest.raises(ValueError):
        subject.prepare_tables(tables, panels)


def test_duplicate_header_rejected(patched):
    tables, panels = _fixture()
    tables["domain_metrics"] = tables["domain_metrics"].rename(columns={"domain": "model_id"})
    with pytest.raises(ValueError):
        subject.prepare_tables(tables, panels)


@pytest.mark.parametrize(
    "table,column,value",
    [
        ("domain_metrics", "delta", "0.10"),
        ("domain_metrics", "model_ba", True),
        ("domain_metrics", "reference_ba", False),
        ("summary", "tied_domains", 0.5),
        ("summary", "positive_domains", True),
    ],
)
def test_illegal_typed_values_rejected(patched, table, column, value):
    tables, panels = _fixture()
    frame = tables[table].astype(object)
    frame.at[0, column] = value
    tables[table] = frame
    with pytest.raises(ValueError):
        subject.prepare_tables(tables, panels)


def test_delta_inconsistency_rejected(patched):
    tables, panels = _fixture()
    tables["domain_metrics"].loc[0, "delta"] = 0.20
    with pytest.raises(ValueError):
        subject.prepare_tables(tables, panels)


def test_mismatched_keyset_rejected(patched):
    tables, panels = _fixture()
    tables["domain_metrics"] = tables["domain_metrics"][
        tables["domain_metrics"]["reference_model_id"] != SECONDARY_REFERENCE
    ].reset_index(drop=True)
    with pytest.raises(ValueError):
        subject.prepare_tables(tables, panels)


def test_missing_metrics_match_rejected(monkeypatch):
    tables, panels = _fixture()
    domain = _metrics_domain()
    domain = domain[
        ~(
            (domain["scope"] == "full_support")
            & (domain["model_id"] == OTHER_REFERENCE)
            & (domain["domain"] == "D2")
        )
    ]
    monkeypatch.setattr(subject, "build_metrics", lambda panels: _metrics_payload(domain))
    with pytest.raises(ValueError):
        subject.prepare_tables(tables, panels)


def test_ba_mismatch_rejected(monkeypatch):
    tables, panels = _fixture()
    domain = _metrics_domain()
    domain.loc[0, "balanced_accuracy"] = 0.80 + 1e-9
    monkeypatch.setattr(subject, "build_metrics", lambda panels: _metrics_payload(domain))
    with pytest.raises(ValueError):
        subject.prepare_tables(tables, panels)


def test_partial_selected_reference_scope_is_not_full_support(patched):
    tables, panels = _fixture()
    extra_summary = tables["summary"].iloc[[2]].copy()
    extra_summary["reference_model_id"] = REFERENCE
    extra_summary["positive_domains"] = 1
    extra_summary["negative_domains"] = 0
    extra_summary["supported_domains"] = 1
    tables["summary"] = pd.concat([tables["summary"], extra_summary], ignore_index=True)
    extra_domain = pd.DataFrame([_row("D3", REFERENCE, "M01", "D1", 2, 0.33, 0.22, 0.11)])
    tables["domain_metrics"] = pd.concat(
        [tables["domain_metrics"], extra_domain], ignore_index=True
    )
    assert subject.prepare_tables(tables, panels)["crosscheck_count"] == 16
