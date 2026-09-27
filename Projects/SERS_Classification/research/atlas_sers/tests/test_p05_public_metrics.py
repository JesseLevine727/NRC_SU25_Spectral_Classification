"""CPU-only tests for the strict anonymous P05 public metric tables.

Every input is produced by the real ``p05_comparison.compare_predictions`` from
the synthetic ``tests.test_p05_comparison._fixture``. Only the private station
label is rewritten to a public one. No file, torch, private data, fit or
selection is touched here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p05_comparison as p05c
from atlas_sers.evaluation import p05_public_metrics as p05m
from tests.test_p05_comparison import STATION, _fixture

PUBLIC_STATION = "cwa"
BASE_CONTEXT = "ctx-held"
PRIVATE_SENTINEL = "PRIVATE-SENTINEL-VALUE"
MISSING_POLICY = "missing assigned zero; not imputed valid scores"


def _rename_station(frame):
    if "station" not in frame.columns:
        return frame
    renamed = frame.copy()
    renamed["station"] = renamed["station"].astype(str).replace({STATION: PUBLIC_STATION})
    return renamed


def _cwa_fixture():
    fixture = dict(_fixture())
    for key in ("p05_ensemble", "p04_ensemble", "p03_predictions", "contexts"):
        fixture[key] = _rename_station(fixture[key])
    aggregate = dict(fixture["aggregate"])
    aggregate["ensemble_predictions"] = _rename_station(aggregate["ensemble_predictions"])
    fixture["aggregate"] = aggregate
    return fixture


def _compare(fixture):
    return p05c.compare_predictions(
        p05_ensemble=fixture["p05_ensemble"],
        p04_ensemble=fixture["p04_ensemble"],
        p03_predictions=fixture["p03_predictions"],
        contexts=fixture["contexts"],
    )


def _observed_class_counts(fixture):
    ensemble = fixture["aggregate"]["ensemble_predictions"]
    return ensemble.groupby("context_id")["true_label"].nunique().astype(int).to_dict()


_AGGREGATION_COLUMNS = [
    "context_id",
    "experiment_id",
    "station",
    "domain",
    "held_instrument",
    "model_id",
    "observed_class_count",
    "observations",
    "physical_masters",
    *p05m.P05_PUBLIC_METRICS,
]


def _aggregation_tables(fixture, comparison):
    endpoint = comparison["endpoint_metrics"]
    observed = _observed_class_counts(fixture)
    p05_rows = endpoint[endpoint["model_id"].isin(p05m.P05_PUBLIC_MODELS)].copy()
    p05_rows["observed_class_count"] = p05_rows["context_id"].map(observed).astype(int)
    tables = {}
    for aggregation, key in (("M01", "spectrum_metrics"), ("M06", "master_metrics")):
        cell = p05_rows[p05_rows["aggregation_id"].eq(aggregation)].copy()
        tables[key] = cell[_AGGREGATION_COLUMNS].reset_index(drop=True)
    return tables


def _comparison_tables(comparison):
    return {
        "paired_metrics": comparison["paired_metrics"].copy(),
        "coverage": comparison["coverage"].copy(),
        "summary": comparison["summary"].copy(),
    }


def _public_inputs(fixture=None):
    fixture = _cwa_fixture() if fixture is None else fixture
    comparison = _compare(fixture)
    return _aggregation_tables(fixture, comparison), _comparison_tables(comparison)


def _build(aggregation, comparison):
    return p05m.build_public_metrics(aggregation_tables=aggregation, comparison_tables=comparison)


def _append_context(
    aggregation,
    comparison,
    *,
    new_id,
    domain,
    held_instrument,
    observed_class_count=None,
    scores=None,
):
    for key in ("spectrum_metrics", "master_metrics"):
        frame = aggregation[key]
        cell = frame[frame["context_id"].eq(BASE_CONTEXT)].copy()
        cell["context_id"] = new_id
        cell["domain"] = domain
        cell["held_instrument"] = held_instrument
        if observed_class_count is not None:
            cell["observed_class_count"] = observed_class_count
        if scores:
            for model, value in scores.items():
                cell.loc[cell["model_id"].eq(model), "balanced_accuracy"] = value
        aggregation[key] = pd.concat([frame, cell], ignore_index=True)
    paired = comparison["paired_metrics"]
    cell = paired[paired["context_id"].eq(BASE_CONTEXT)].copy()
    cell["context_id"] = new_id
    cell["domain"] = domain
    cell["held_instrument"] = held_instrument
    if scores:
        for model, value in scores.items():
            cell.loc[cell["model_id"].eq(model), "model_balanced_accuracy"] = value
            cell.loc[cell["reference_model_id"].eq(model), "reference_balanced_accuracy"] = value
        cell["delta_balanced_accuracy"] = (
            cell["model_balanced_accuracy"] - cell["reference_balanced_accuracy"]
        )
    paired = pd.concat([paired, cell], ignore_index=True)
    comparison["paired_metrics"] = paired
    coverage = comparison["coverage"]
    cell = coverage[coverage["context_id"].eq(BASE_CONTEXT)].copy()
    cell["context_id"] = new_id
    cell["domain"] = domain
    cell["held_instrument"] = held_instrument
    comparison["coverage"] = pd.concat([coverage, cell], ignore_index=True)
    comparison["summary"] = p05c._summary(paired)


def _two_context_inputs():
    aggregation, comparison = _public_inputs()
    for key in ("spectrum_metrics", "master_metrics"):
        frame = aggregation[key].copy()
        frame["observed_class_count"] = 1
        aggregation[key] = frame
    source = aggregation["spectrum_metrics"]
    held_instrument = str(source["held_instrument"].iloc[0])
    domain = str(source["domain"].iloc[0])
    _append_context(
        aggregation,
        comparison,
        new_id="ctx-dup",
        domain=domain,
        held_instrument=held_instrument,
        observed_class_count=2,
    )
    return aggregation, comparison


def _expected_strategy_domain_columns():
    return [
        *p05m._STRATEGY_GROUP,
        "completed_contexts",
        "contexts_with_1_observed_class",
        "contexts_with_2_observed_classes",
        "contexts_with_3_observed_classes",
        "min_test_masters",
        "max_test_masters",
        *(f"mean_{metric}_equal_contexts" for metric in p05m.P05_PUBLIC_METRICS),
    ]


def _expected_strategy_summary_columns():
    return [
        *p05m._STRATEGY_SUMMARY_GROUP,
        "contexts",
        "domains",
        *(f"mean_{metric}_equal_contexts" for metric in p05m.P05_PUBLIC_METRICS),
        *(f"mean_{metric}_equal_domains" for metric in p05m.P05_PUBLIC_METRICS),
        "worst_domain_balanced_accuracy",
    ]


def _expected_paired_domain_columns():
    return [
        *p05m._PAIRED_GROUP,
        "planned_contexts",
        "common_contexts",
        "common_coverage",
        *(f"mean_model_{metric}" for metric in p05m.P05_PUBLIC_PAIRED_METRICS),
        *(f"mean_reference_{metric}" for metric in p05m.P05_PUBLIC_PAIRED_METRICS),
        *(f"mean_delta_{metric}" for metric in p05m.P05_PUBLIC_PAIRED_METRICS),
    ]


def test_public_tables_whitelist_and_known_scores():
    aggregation, comparison = _two_context_inputs()
    out = _build(aggregation, comparison)
    assert set(out) == set(p05m.P05_PUBLIC_TABLE_NAMES)
    expected = {
        "strategy_contexts": list(p05m._STRATEGY_CONTEXT_COLUMNS),
        "strategy_domains": _expected_strategy_domain_columns(),
        "strategy_summary": _expected_strategy_summary_columns(),
        "paired_contexts": list(p05m._PAIRED_CONTEXT_COLUMNS),
        "paired_domains": _expected_paired_domain_columns(),
        "comparison_summary": list(p05m._COMPARISON_SUMMARY_COLUMNS),
        "coverage": list(p05m._COVERAGE_COLUMNS),
    }
    forbidden = (
        "context_id",
        "observation_uid",
        "master_sample_id",
        "test_uid",
        "true_label",
        "class_vocabulary",
        "probability",
        "path",
        "PRIVATE",
    )
    for name, table in out.items():
        assert list(table.columns) == expected[name]
        for column in table.columns:
            for token in forbidden:
                assert token not in column
    contexts = out["strategy_contexts"]
    d0m = contexts[contexts["model_id"].eq("D0-M") & contexts["aggregation_id"].eq("M01")]
    assert np.allclose(d0m["balanced_accuracy"].to_numpy(), 1.0)
    d3 = contexts[contexts["model_id"].eq("D3") & contexts["aggregation_id"].eq("M01")]
    assert np.allclose(d3["balanced_accuracy"].to_numpy(), 0.5)
    d3_master = contexts[contexts["model_id"].eq("D3") & contexts["aggregation_id"].eq("M06")]
    assert np.allclose(d3_master["balanced_accuracy"].to_numpy(), 0.5)
    assert len(out["strategy_contexts"]) == 12
    assert len(out["paired_contexts"]) == 68
    assert len(out["coverage"]) == 8
    assert out["coverage"].expected_contexts.eq(2).all()


def test_point_index_consistency_between_strategy_and_paired():
    aggregation, comparison = _two_context_inputs()
    out = _build(aggregation, comparison)
    assert set(out["strategy_contexts"]["point_index"]) == {1, 2}
    assert set(out["paired_contexts"]["point_index"]) == {1, 2}
    for point_index, cell in out["strategy_contexts"].groupby("point_index"):
        paired_cell = out["paired_contexts"][out["paired_contexts"]["point_index"].eq(point_index)]
        assert set(paired_cell["station"]) == set(cell["station"])
        assert set(paired_cell["domain"]) == set(cell["domain"])
        assert set(paired_cell["held_instrument"]) == set(cell["held_instrument"])


def test_common_pair_means_use_only_common_contexts():
    aggregation, comparison = _two_context_inputs()
    out = _build(aggregation, comparison)
    domains = out["paired_domains"]
    row = domains[
        domains["model_id"].eq("D3")
        & domains["reference_model_id"].eq("D0-M")
        & domains["aggregation_id"].eq("M01")
    ].iloc[0]
    assert row["common_contexts"] == 2
    assert row["common_coverage"] == pytest.approx(1.0)
    assert row["mean_model_balanced_accuracy"] == pytest.approx(0.5)
    assert row["mean_reference_balanced_accuracy"] == pytest.approx(1.0)
    assert row["mean_delta_balanced_accuracy"] == pytest.approx(-0.5)


def test_strategy_domain_observed_class_counts_and_master_range():
    aggregation, comparison = _two_context_inputs()
    out = _build(aggregation, comparison)
    contexts = out["strategy_contexts"]
    row = out["strategy_domains"][
        out["strategy_domains"]["model_id"].eq("D0-M")
        & out["strategy_domains"]["aggregation_id"].eq("M01")
    ].iloc[0]
    assert row["completed_contexts"] == 2
    assert row["contexts_with_1_observed_class"] == 1
    assert row["contexts_with_2_observed_classes"] == 1
    assert row["contexts_with_3_observed_classes"] == 0
    mask = (
        contexts["station"].eq(row["station"])
        & contexts["phase"].eq(row["phase"])
        & contexts["domain"].eq(row["domain"])
        & contexts["held_instrument"].eq(row["held_instrument"])
        & contexts["model_id"].eq(row["model_id"])
        & contexts["aggregation_id"].eq(row["aggregation_id"])
    )
    assert row["min_test_masters"] == int(contexts.loc[mask, "physical_masters"].min())
    assert row["max_test_masters"] == int(contexts.loc[mask, "physical_masters"].max())
    assert row["min_test_masters"] <= row["max_test_masters"]


def test_partial_random_forest_reference_is_never_scored_as_zero():
    fixture = _cwa_fixture()
    classical = fixture["p03_predictions"]
    rf = classical[classical["model_id"].eq("C-RANDOM-FOREST")]
    fixture["p03_predictions"] = classical.drop(index=rf.index[0]).reset_index(drop=True)
    comparison = _compare(fixture)
    out = _build(
        _aggregation_tables(fixture, comparison),
        _comparison_tables(comparison),
    )
    coverage = out["coverage"]
    assert int(coverage["complete_contexts"].sum()) == 7
    rf_row = coverage[coverage["model_id"].eq("C-RANDOM-FOREST")].iloc[0]
    assert rf_row["complete_contexts"] == 0
    assert rf_row["complete_coverage"] == pytest.approx(0.0)
    paired = out["paired_contexts"]
    rf_pair = paired[
        paired["model_id"].eq("D0-M")
        & paired["reference_model_id"].eq("C-RANDOM-FOREST")
        & paired["aggregation_id"].eq("M01")
    ]
    assert not rf_pair["common_complete"].any()
    rf_domain = out["paired_domains"][
        out["paired_domains"]["model_id"].eq("D0-M")
        & out["paired_domains"]["reference_model_id"].eq("C-RANDOM-FOREST")
        & out["paired_domains"]["aggregation_id"].eq("M01")
    ].iloc[0]
    assert rf_domain["common_contexts"] == 0
    assert np.isnan(rf_domain["mean_model_balanced_accuracy"])
    assert np.isnan(rf_domain["mean_reference_balanced_accuracy"])
    assert np.isnan(rf_domain["mean_delta_balanced_accuracy"])
    assert not rf_domain["mean_delta_balanced_accuracy"] == 0.0
    summary = out["comparison_summary"]
    row = summary[
        summary["model_id"].eq("D0-M")
        & summary["reference_model_id"].eq("C-RANDOM-FOREST")
        & summary["aggregation_id"].eq("M01")
    ].iloc[0]
    assert row[
        "reference_failure_sensitive_mean_balanced_accuracy_missing_as_zero"
    ] == pytest.approx(0.0)
    assert row["model_failure_sensitive_mean_balanced_accuracy_missing_as_zero"] == pytest.approx(
        1.0
    )
    assert row["failure_sensitive_missing_policy"] == MISSING_POLICY
    contexts = out["strategy_contexts"]
    assert np.allclose(
        contexts.loc[contexts["model_id"].eq("D0-M"), "balanced_accuracy"].to_numpy(), 1.0
    )
    assert np.allclose(
        contexts.loc[contexts["model_id"].eq("D3"), "balanced_accuracy"].to_numpy(), 0.5
    )


def test_two_domains_unequal_contexts_distinct_equal_context_and_equal_domain():
    aggregation, comparison = _public_inputs()
    source = aggregation["spectrum_metrics"]
    held_instrument = str(source["held_instrument"].iloc[0])
    domain = str(source["domain"].iloc[0])
    _append_context(
        aggregation,
        comparison,
        new_id="ctx-dup",
        domain=domain,
        held_instrument=held_instrument,
    )
    _append_context(
        aggregation,
        comparison,
        new_id="ctx-dup2",
        domain="DOM-2",
        held_instrument=held_instrument,
        scores={"D0-M": 0.5},
    )
    out = _build(aggregation, comparison)
    row = out["strategy_summary"][
        out["strategy_summary"]["model_id"].eq("D0-M")
        & out["strategy_summary"]["aggregation_id"].eq("M01")
    ].iloc[0]
    assert row["contexts"] == 3
    assert row["domains"] == 2
    equal_contexts = row["mean_balanced_accuracy_equal_contexts"]
    equal_domains = row["mean_balanced_accuracy_equal_domains"]
    assert equal_contexts == pytest.approx((1.0 + 1.0 + 0.5) / 3.0)
    assert equal_domains == pytest.approx((1.0 + 0.5) / 2.0)
    assert equal_contexts != pytest.approx(equal_domains)
    assert row["worst_domain_balanced_accuracy"] == pytest.approx(0.5)


def test_shuffled_rows_yield_semantically_identical_tables():
    aggregation, comparison = _public_inputs()
    expected = _build(aggregation, comparison)
    shuffled_aggregation = {
        key: value.sample(frac=1.0, random_state=7).reset_index(drop=True)
        for key, value in aggregation.items()
    }
    shuffled_comparison = {
        key: value.sample(frac=1.0, random_state=7).reset_index(drop=True)
        for key, value in comparison.items()
    }
    result = _build(shuffled_aggregation, shuffled_comparison)
    for name in p05m.P05_PUBLIC_TABLE_NAMES:
        pd.testing.assert_frame_equal(
            expected[name], result[name], check_exact=False, rtol=1e-9, atol=1e-12
        )


def test_private_sentinel_never_serialized_in_public_tables():
    aggregation, comparison = _public_inputs()
    spectrum = aggregation["spectrum_metrics"].copy()
    spectrum["PRIVATE_NOTE"] = PRIVATE_SENTINEL
    aggregation["spectrum_metrics"] = spectrum
    coverage = comparison["coverage"].copy()
    coverage["observation_uid"] = PRIVATE_SENTINEL
    comparison["coverage"] = coverage
    out = _build(aggregation, comparison)
    serialized = "\n".join(table.to_csv(index=False) for table in out.values())
    assert PRIVATE_SENTINEL not in serialized
    assert "PRIVATE_NOTE" not in serialized


def test_unknown_private_context_is_not_given_a_public_index():
    aggregation, comparison = _public_inputs()
    paired = comparison["paired_metrics"]
    cell = paired[paired["context_id"].eq(BASE_CONTEXT)].copy()
    cell["context_id"] = "ctx-unknown"
    comparison["paired_metrics"] = pd.concat([paired, cell], ignore_index=True)
    with pytest.raises(p05m.P05PublicMetricsError):
        _build(aggregation, comparison)


def test_build_does_not_mutate_input_frames():
    aggregation, comparison = _public_inputs()
    aggregation_before = {key: value.copy(deep=True) for key, value in aggregation.items()}
    comparison_before = {key: value.copy(deep=True) for key, value in comparison.items()}
    _build(aggregation, comparison)
    for key, before in aggregation_before.items():
        pd.testing.assert_frame_equal(before, aggregation[key])
    for key, before in comparison_before.items():
        pd.testing.assert_frame_equal(before, comparison[key])


def test_actual_aggregation_tables_retain_development_and_held_phases():
    fixture = _cwa_fixture()
    aggregation = {name: _rename_station(frame) for name, frame in fixture["aggregate"].items()}
    comparison = _compare(fixture)
    result = _build(aggregation, comparison)
    assert set(result["strategy_contexts"].phase) == {"development", "held_evaluation"}
    held_points = set(
        result["strategy_contexts"].loc[
            result["strategy_contexts"].phase.eq("held_evaluation"), "point_index"
        ]
    )
    assert set(result["paired_contexts"].point_index) == held_points
    assert set(result["strategy_summary"].phase) == {"development", "held_evaluation"}


def _mut_unknown_station(aggregation, comparison):
    frame = aggregation["spectrum_metrics"].copy()
    frame["station"] = frame["station"].astype(object)
    frame.loc[frame.index[0], "station"] = "STN1"
    aggregation["spectrum_metrics"] = frame


def _mut_unknown_phase(aggregation, comparison):
    frame = aggregation["spectrum_metrics"].copy()
    frame["experiment_id"] = frame["experiment_id"].astype(object)
    frame.loc[frame.index[0], "experiment_id"] = "EXP-UNKNOWN"
    aggregation["spectrum_metrics"] = frame


def _mut_unknown_model(aggregation, comparison):
    frame = aggregation["spectrum_metrics"].copy()
    frame["model_id"] = frame["model_id"].astype(object)
    frame.loc[frame.index[0], "model_id"] = "D9"
    aggregation["spectrum_metrics"] = frame


def _mut_unknown_reference(aggregation, comparison):
    frame = comparison["paired_metrics"].copy()
    frame["reference_model_id"] = frame["reference_model_id"].astype(object)
    frame.loc[frame.index[0], "reference_model_id"] = "C-UNKNOWN"
    comparison["paired_metrics"] = frame


def _mut_duplicate_context_key(aggregation, comparison):
    frame = aggregation["spectrum_metrics"]
    aggregation["spectrum_metrics"] = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)


def _mut_missing_model_endpoint(aggregation, comparison):
    frame = aggregation["spectrum_metrics"]
    aggregation["spectrum_metrics"] = frame.drop(index=frame.index[0]).reset_index(drop=True)


def _mut_bool_count(aggregation, comparison):
    frame = aggregation["spectrum_metrics"].copy()
    frame["observations"] = frame["observations"].astype(object)
    frame.loc[frame.index[0], "observations"] = True
    aggregation["spectrum_metrics"] = frame


def _mut_bool_metric(aggregation, comparison):
    frame = aggregation["spectrum_metrics"].copy()
    frame["balanced_accuracy"] = frame["balanced_accuracy"].astype(object)
    frame.loc[frame.index[0], "balanced_accuracy"] = True
    aggregation["spectrum_metrics"] = frame


def _mut_nonfinite_metric(aggregation, comparison):
    frame = aggregation["spectrum_metrics"].copy()
    frame.loc[frame.index[0], "balanced_accuracy"] = np.inf
    aggregation["spectrum_metrics"] = frame


def _mut_out_of_range_metric(aggregation, comparison):
    frame = aggregation["spectrum_metrics"].copy()
    frame.loc[frame.index[0], "ece"] = 1.5
    aggregation["spectrum_metrics"] = frame


def _mut_wrong_aggregation(aggregation, comparison):
    frame = aggregation["spectrum_metrics"].copy()
    frame["aggregation_id"] = "M06"
    aggregation["spectrum_metrics"] = frame


def _mut_conflicting_counts(aggregation, comparison):
    frame = aggregation["spectrum_metrics"].copy()
    frame.loc[frame.index[0], "observations"] = int(frame["observations"].iloc[0]) + 1
    aggregation["spectrum_metrics"] = frame


def _mut_conflicting_model_metadata(aggregation, comparison):
    frame = aggregation["spectrum_metrics"].copy()
    frame["held_instrument"] = frame["held_instrument"].astype(object)
    frame.loc[frame.index[0], "held_instrument"] = "OTHER-INSTRUMENT"
    aggregation["spectrum_metrics"] = frame


def _mut_paired_completeness(aggregation, comparison):
    frame = comparison["paired_metrics"].copy()
    index = frame.index[frame["reference_model_id"].eq("C-RANDOM-FOREST")][0]
    frame["reference_complete"] = frame["reference_complete"].astype(object)
    frame["common_complete"] = frame["common_complete"].astype(object)
    frame.loc[index, "reference_complete"] = False
    frame.loc[index, "common_complete"] = True
    comparison["paired_metrics"] = frame


def _mut_paired_delta(aggregation, comparison):
    frame = comparison["paired_metrics"].copy()
    frame.loc[frame.index[0], "delta_balanced_accuracy"] = 999.0
    comparison["paired_metrics"] = frame


def _mut_paired_model_score(aggregation, comparison):
    frame = comparison["paired_metrics"].copy()
    index = frame.index[frame["model_id"].isin(p05m.P05_PUBLIC_MODELS)][0]
    reference = float(frame.loc[index, "reference_balanced_accuracy"])
    frame.loc[index, "model_balanced_accuracy"] = 0.123
    frame.loc[index, "delta_balanced_accuracy"] = 0.123 - reference
    comparison["paired_metrics"] = frame


def _mut_paired_reference_score(aggregation, comparison):
    frame = comparison["paired_metrics"].copy()
    index = frame.index[frame["reference_model_id"].eq("D0-M")][0]
    model = float(frame.loc[index, "model_balanced_accuracy"])
    frame.loc[index, "reference_balanced_accuracy"] = 0.321
    frame.loc[index, "delta_balanced_accuracy"] = model - 0.321
    comparison["paired_metrics"] = frame


def _mut_coverage_context_set(aggregation, comparison):
    frame = comparison["coverage"]
    comparison["coverage"] = frame[frame["context_id"].ne(BASE_CONTEXT)].reset_index(drop=True)


def _mut_coverage_duplicate(aggregation, comparison):
    frame = comparison["coverage"]
    comparison["coverage"] = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)


def _mut_coverage_second_row_metadata(aggregation, comparison):
    frame = comparison["coverage"].copy()
    frame["domain"] = frame["domain"].astype(object)
    frame.loc[frame.index[1], "domain"] = "OTHER-DOMAIN"
    comparison["coverage"] = frame


def _mut_coverage_completeness(aggregation, comparison):
    frame = comparison["coverage"].copy()
    frame["complete"] = frame["complete"].astype(object)
    frame.loc[frame.index[0], "complete"] = False
    comparison["coverage"] = frame


def _mut_summary_counts(aggregation, comparison):
    frame = comparison["summary"].copy()
    frame.loc[frame.index[0], "planned_contexts"] = int(frame["planned_contexts"].iloc[0]) + 1
    comparison["summary"] = frame


def _mut_summary_policy(aggregation, comparison):
    frame = comparison["summary"].copy()
    frame["failure_sensitive_missing_policy"] = frame["failure_sensitive_missing_policy"].astype(
        object
    )
    frame.loc[frame.index[0], "failure_sensitive_missing_policy"] = "changed policy"
    comparison["summary"] = frame


def _mut_summary_numbers(aggregation, comparison):
    frame = comparison["summary"].copy()
    frame.loc[frame.index[0], "mean_delta_balanced_accuracy"] = 123.0
    comparison["summary"] = frame


_MUTATORS = (
    ("unknown_station", _mut_unknown_station),
    ("unknown_phase", _mut_unknown_phase),
    ("unknown_model", _mut_unknown_model),
    ("unknown_reference", _mut_unknown_reference),
    ("duplicate_context_key", _mut_duplicate_context_key),
    ("missing_model_endpoint", _mut_missing_model_endpoint),
    ("invalid_bool_count", _mut_bool_count),
    ("invalid_bool_metric", _mut_bool_metric),
    ("nonfinite_metric", _mut_nonfinite_metric),
    ("out_of_range_metric", _mut_out_of_range_metric),
    ("wrong_aggregation", _mut_wrong_aggregation),
    ("conflicting_counts", _mut_conflicting_counts),
    ("conflicting_model_metadata", _mut_conflicting_model_metadata),
    ("conflicting_paired_completeness", _mut_paired_completeness),
    ("conflicting_delta", _mut_paired_delta),
    ("conflicting_model_score", _mut_paired_model_score),
    ("conflicting_reference_score", _mut_paired_reference_score),
    ("coverage_context_set", _mut_coverage_context_set),
    ("coverage_duplicate", _mut_coverage_duplicate),
    ("coverage_second_row_metadata", _mut_coverage_second_row_metadata),
    ("coverage_completeness_mismatch", _mut_coverage_completeness),
    ("summary_counts", _mut_summary_counts),
    ("summary_policy", _mut_summary_policy),
    ("summary_numbers", _mut_summary_numbers),
)


@pytest.mark.parametrize(("label", "mutate"), _MUTATORS, ids=[label for label, _ in _MUTATORS])
def test_malformed_inputs_are_rejected(label, mutate):
    aggregation, comparison = _public_inputs()
    mutate(aggregation, comparison)
    with pytest.raises(p05m.P05PublicMetricsError):
        _build(aggregation, comparison)
