"""CPU synthetic tests for the pure P05 comparison module.

Every input is generated in memory from ``tests.test_p05_results._base`` and
``p05_results.aggregate_predictions``. No torch, no real reference files and no
bootstrap p-values are exercised here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p05_comparison as p05c
from atlas_sers.evaluation import p05_results as p05
from atlas_sers.evaluation.p04_comparison import CLASSICAL_MODELS
from tests.test_p05_results import _base

HELD_EXPERIMENT = "EXP-N00-T3"
DEV_EXPERIMENT = "EXP-N00-DEV"
HISTORICAL_MODEL = "D0-ERM"
P05_HELD = "P05-CORE-T3"
HELD_CONTEXT = "ctx-held"
STATION = "STN1"
DOMAIN = "DOM"


def _raw_fixture():
    fx = _base()
    for context in fx["contexts"]:
        if context["phase_gate"] == "held_evaluation":
            context["experiment_id"] = HELD_EXPERIMENT
        else:
            context["experiment_id"] = DEV_EXPERIMENT
    return fx


def _aggregate(fx):
    return p05.aggregate_predictions(
        plan=fx["plan"],
        contexts=fx["contexts"],
        manifest=fx["manifest"],
        predictions=fx["predictions"],
    )


def _contexts_frame(fx):
    frame = pd.DataFrame(fx["contexts"])
    return frame[
        [
            "context_id",
            "experiment_id",
            "domain",
            "station",
            "held_instrument",
            "outer_repeat",
            "outer_fold",
            "phase_gate",
            "outer_test_uid_sha256",
        ]
    ].copy()


def _held_p05(ensemble, model="D0-M"):
    return ensemble[ensemble["experiment_id"].eq(P05_HELD) & ensemble["model_id"].eq(model)].copy()


def _historical(ensemble, *, vocabulary="ndarray"):
    frame = _held_p05(ensemble)
    frame["experiment_id"] = HELD_EXPERIMENT
    frame["model_id"] = HISTORICAL_MODEL
    frame["comparison_model_id"] = HISTORICAL_MODEL
    if vocabulary == "ndarray":
        frame["class_vocabulary"] = frame["class_vocabulary"].map(
            lambda value: np.array(json.loads(value))
        )
    else:
        frame["class_vocabulary"] = frame["class_vocabulary"].map(json.loads)
    return frame


def _classical(ensemble):
    base = _held_p05(ensemble)
    frames = []
    for model in CLASSICAL_MODELS:
        cell = base.copy()
        cell["experiment_id"] = "EXP-C09-T3" if model == "C-SELECTED" else "EXP-C10-T3"
        cell["model_id"] = model
        cell["class_vocabulary"] = cell["class_vocabulary"].map(json.loads)
        cell["probabilities"] = cell.apply(
            lambda row: [
                float(row["probability_0"]),
                float(row["probability_1"]),
                float(row["probability_2"]),
            ],
            axis=1,
        )
        frames.append(cell)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop(columns=["context_id", "probability_0", "probability_1", "probability_2"])
    frame["domain"] = frame["domain"].astype(str)
    frame["outer_repeat"] = frame["outer_repeat"].astype(int)
    frame["outer_fold"] = frame["outer_fold"].astype(int)
    return frame


def _fixture():
    raw = _raw_fixture()
    aggregate = _aggregate(raw)
    ensemble = aggregate["ensemble_predictions"]
    return {
        "raw": raw,
        "aggregate": aggregate,
        "p05_ensemble": ensemble.copy(),
        "p04_ensemble": _historical(ensemble),
        "p03_predictions": _classical(ensemble),
        "contexts": _contexts_frame(raw),
    }


def _run(fixture=None, **overrides):
    fixture = _fixture() if fixture is None else fixture
    arguments = {
        "p05_ensemble": fixture["p05_ensemble"],
        "p04_ensemble": fixture["p04_ensemble"],
        "p03_predictions": fixture["p03_predictions"],
        "contexts": fixture["contexts"],
    }
    arguments.update(overrides)
    return p05c.compare_predictions(**arguments)


def _assert_rejected(fixture, **overrides):
    with pytest.raises(p05c.P05ComparisonError):
        _run(fixture, **overrides)


def _mutate_classical(fixture, **changes):
    frame = fixture["p03_predictions"].copy()
    for column, value in changes.items():
        frame.loc[frame.index[0], column] = value
    fixture["p03_predictions"] = frame


def test_complete_baseline_structure_and_scores():
    out = _run()
    coverage = out["coverage"]
    endpoint = out["endpoint_metrics"]
    paired = out["paired_metrics"]
    assert len(coverage) == 8
    assert len(endpoint) == 16
    assert len(paired) == 34
    assert set(coverage["model_id"]) == set(p05c.ALL_MODELS)
    assert coverage["complete"].all()
    assert set(coverage["source_context_experiment_id"]) == {HELD_EXPERIMENT}
    assert set(paired["aggregation_id"]) == {"M01", "M06"}
    assert len(p05c.PAIRS) == 17
    pairs = set(zip(paired["model_id"], paired["reference_model_id"], strict=True))
    expected = {(m, r) for m in p05c.P05_MODELS for r in p05c.REFERENCE_MODELS}
    expected |= set(p05c.SELECTED_PAIRS)
    assert pairs == expected
    scored = {
        (row.model_id, row.aggregation_id): row.balanced_accuracy
        for row in endpoint.itertuples(index=False)
    }
    for model in p05c.REFERENCE_MODELS:
        assert scored[(model, "M01")] == pytest.approx(1.0)
        assert scored[(model, "M06")] == pytest.approx(1.0)
    assert scored[("D0-M", "M01")] == pytest.approx(1.0)
    assert scored[("P05-SELECTED", "M01")] == pytest.approx(1.0)
    assert scored[("D3", "M01")] == pytest.approx(0.5)
    assert scored[("D3", "M06")] == pytest.approx(0.5)


def test_inputs_are_not_mutated():
    fixture = _fixture()
    snapshot = {
        key: fixture[key].copy(deep=True)
        for key in ("p05_ensemble", "p04_ensemble", "p03_predictions", "contexts")
    }
    _run(fixture)
    for key, before in snapshot.items():
        pd.testing.assert_frame_equal(before, fixture[key])


def test_partial_reference_missing_uid_is_incomplete():
    fixture = _fixture()
    classical = fixture["p03_predictions"]
    rf_rows = classical[classical["model_id"].eq("C-RANDOM-FOREST")]
    fixture["p03_predictions"] = classical.drop(index=rf_rows.index[0]).reset_index(drop=True)
    out = _run(fixture)
    coverage = out["coverage"]
    rf = coverage[coverage["model_id"].eq("C-RANDOM-FOREST")].iloc[0]
    assert not rf["complete"]
    paired = out["paired_metrics"]
    pair = paired[
        paired["model_id"].eq("D0-M")
        & paired["reference_model_id"].eq("C-RANDOM-FOREST")
        & paired["aggregation_id"].eq("M01")
    ].iloc[0]
    assert pair["model_complete"]
    assert not pair["reference_complete"]
    assert not pair["common_complete"]
    assert np.isnan(pair["delta_balanced_accuracy"])
    summary = out["summary"]
    row = summary[
        summary["model_id"].eq("D0-M")
        & summary["reference_model_id"].eq("C-RANDOM-FOREST")
        & summary["aggregation_id"].eq("M01")
    ].iloc[0]
    assert row["model_failure_sensitive_mean_balanced_accuracy_missing_as_zero"] == pytest.approx(
        1.0
    )
    assert row[
        "reference_failure_sensitive_mean_balanced_accuracy_missing_as_zero"
    ] == pytest.approx(0.0)
    assert row["model_failure_sensitive_mean_balanced_accuracy_missing_as_zero"] != 0.0


def test_missing_references_still_keep_p05_endpoints():
    fixture = _fixture()
    empty = pd.DataFrame(columns=fixture["p04_ensemble"].columns)
    out = _run(fixture, p04_ensemble=empty, p03_predictions=empty)
    coverage = out["coverage"]
    assert coverage[coverage["model_id"].isin(p05c.P05_MODELS)]["complete"].all()
    assert not coverage[coverage["model_id"].isin(p05c.REFERENCE_MODELS)]["complete"].any()
    endpoint = out["endpoint_metrics"]
    assert set(endpoint["model_id"]) == set(p05c.P05_MODELS)
    assert len(endpoint) == 6
    paired = out["paired_metrics"]
    assert len(paired) == 34
    reference_pairs = paired[paired["reference_model_id"].isin(p05c.REFERENCE_MODELS)]
    assert not reference_pairs["reference_complete"].any()
    assert not reference_pairs["common_complete"].any()


def test_p03_without_selected_rows_is_missing():
    fixture = _fixture()
    p03 = fixture["p03_predictions"].copy()
    p03["experiment_id"] = "EXP-C11-T3"
    out = _run(fixture, p03_predictions=p03)
    coverage = out["coverage"]
    assert not coverage[coverage["model_id"].isin(CLASSICAL_MODELS)]["complete"].any()
    assert coverage[coverage["model_id"].isin(p05c.P05_MODELS)]["complete"].all()


@pytest.mark.parametrize("vocabulary", ["ndarray", "list"])
def test_reference_class_vocabulary_containers(vocabulary):
    fixture = _fixture()
    fixture["p04_ensemble"] = _historical(
        fixture["aggregate"]["ensemble_predictions"], vocabulary=vocabulary
    )
    out = _run(fixture)
    row = out["coverage"][out["coverage"]["model_id"].eq(HISTORICAL_MODEL)].iloc[0]
    assert row["complete"]


def test_context_strings_merge_integer_p03():
    fixture = _fixture()
    assert all(isinstance(value, str) for value in fixture["contexts"]["outer_repeat"])
    assert all(isinstance(value, str) for value in fixture["contexts"]["outer_fold"])
    assert all(isinstance(value, int) for value in fixture["p03_predictions"]["outer_repeat"])
    out = _run(fixture)
    assert out["coverage"]["complete"].all()


def test_duplicate_uid_within_one_classical_model_rejected():
    fixture = _fixture()
    classical = fixture["p03_predictions"]
    rf = classical[classical["model_id"].eq("C-RANDOM-FOREST")]
    fixture["p03_predictions"] = pd.concat([classical, rf.iloc[[0]]], ignore_index=True)
    _assert_rejected(fixture)


def test_shared_uids_across_four_classical_models_accepted():
    fixture = _fixture()
    grouped = (
        fixture["p03_predictions"]
        .groupby("model_id")["observation_uid"]
        .apply(lambda values: tuple(sorted(values)))
    )
    assert len(set(grouped)) == 1
    out = _run(fixture)
    assert out["coverage"]["complete"].all()


def test_classical_unknown_uid_rejected():
    fixture = _fixture()
    _mutate_classical(fixture, observation_uid="zzz")
    _assert_rejected(fixture)


@pytest.mark.parametrize(
    ("column", "value"),
    [("true_label", "ZZZ"), ("master_sample_id", "mzzz"), ("instrument", "INST-ZZZ")],
)
def test_classical_conflicting_metadata_rejected(column, value):
    fixture = _fixture()
    _mutate_classical(fixture, **{column: value})
    _assert_rejected(fixture)


def test_classical_conflicting_class_order_rejected():
    fixture = _fixture()
    frame = fixture["p03_predictions"].copy()
    frame.at[frame.index[0], "class_vocabulary"] = ["B", "A", "C"]
    fixture["p03_predictions"] = frame
    _assert_rejected(fixture)


@pytest.mark.parametrize("kind", ["normalization", "nonfinite"])
def test_classical_invalid_probabilities_rejected(kind):
    fixture = _fixture()
    frame = fixture["p03_predictions"].copy()
    if kind == "normalization":
        frame.at[frame.index[0], "probabilities"] = [0.5, 0.5, 0.5]
    else:
        frame.at[frame.index[0], "probabilities"] = [np.nan, 0.5, 0.5]
    fixture["p03_predictions"] = frame
    _assert_rejected(fixture)


def test_p05_parent_experiment_mismatch_rejected():
    fixture = _fixture()
    frame = fixture["p05_ensemble"].copy()
    held = frame["experiment_id"].eq(P05_HELD)
    frame.loc[held, "source_context_experiment_id"] = "EXP-OTHER"
    fixture["p05_ensemble"] = frame
    _assert_rejected(fixture)


def test_p05_missing_strategy_rejected():
    fixture = _fixture()
    frame = fixture["p05_ensemble"]
    keep = ~(frame["context_id"].eq(HELD_CONTEXT) & frame["model_id"].eq("D3"))
    fixture["p05_ensemble"] = frame[keep].reset_index(drop=True)
    _assert_rejected(fixture)


def test_p05_missing_held_context_rejected():
    fixture = _fixture()
    frame = fixture["p05_ensemble"]
    fixture["p05_ensemble"] = frame[frame["context_id"].ne(HELD_CONTEXT)].reset_index(drop=True)
    _assert_rejected(fixture)


def test_p05_uid_hash_mismatch_rejected():
    fixture = _fixture()
    contexts = fixture["contexts"].copy()
    contexts.loc[contexts["context_id"].eq(HELD_CONTEXT), "outer_test_uid_sha256"] = "0" * 64
    fixture["contexts"] = contexts
    _assert_rejected(fixture)


def test_context_phase_field_required():
    fixture = _fixture()
    fixture["contexts"] = fixture["contexts"].drop(columns=["phase_gate"])
    _assert_rejected(fixture)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("outer_repeat", 1.5),
        ("outer_fold", True),
        ("outer_repeat", "01"),
        ("outer_fold", -1.5),
    ],
)
def test_context_invalid_coordinates_rejected(column, value):
    fixture = _fixture()
    contexts = fixture["contexts"].copy()
    contexts[column] = contexts[column].astype(object)
    contexts.loc[contexts["context_id"].eq(HELD_CONTEXT), column] = value
    fixture["contexts"] = contexts
    _assert_rejected(fixture)


def test_reference_conflicting_metadata_second_row_rejected():
    fixture = _fixture()
    frame = fixture["p04_ensemble"].copy()
    frame["domain"] = DOMAIN
    frame.loc[frame.index[1], "domain"] = "OTHER"
    fixture["p04_ensemble"] = frame
    _assert_rejected(fixture)


def test_explicit_summary_unequal_domain_context_counts():
    paired = pd.DataFrame(
        {
            "station": [STATION] * 3,
            "model_id": ["D0-M"] * 3,
            "reference_model_id": ["C-SELECTED"] * 3,
            "aggregation_id": ["M01"] * 3,
            "common_complete": [True, True, True],
            "domain": ["D1", "D1", "D2"],
            "model_balanced_accuracy": [0.2, 0.4, 1.0],
            "reference_balanced_accuracy": [0.5, 0.5, 0.5],
            "delta_balanced_accuracy": [-0.3, -0.1, 0.5],
            "delta_macro_f1": [0.0, 0.0, 0.0],
            "delta_negative_log_likelihood": [0.0, 0.0, 0.0],
            "delta_ece": [0.0, 0.0, 0.0],
        }
    )
    summary = p05c._summary(paired)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["planned_contexts"] == 3
    assert row["planned_domains"] == 2
    assert row["contributing_common_domains"] == 2
    assert row["model_mean_balanced_accuracy_equal_contexts"] == pytest.approx(1.6 / 3.0)
    assert row["model_mean_balanced_accuracy_equal_domains"] == pytest.approx(0.65)
    assert (
        row["model_mean_balanced_accuracy_equal_contexts"]
        != row["model_mean_balanced_accuracy_equal_domains"]
    )


def test_no_bootstrap_or_pvalue_columns():
    out = _run()
    for table in out.values():
        for column in table.columns:
            assert "p_value" not in column
            assert "bootstrap" not in column
