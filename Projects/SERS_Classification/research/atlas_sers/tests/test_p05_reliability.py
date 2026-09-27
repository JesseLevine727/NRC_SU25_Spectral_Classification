"""CPU-only tests for the strict anonymous P05 reliability diagnostics.

Every fixture here is synthetic and in-memory. One end-to-end handoff test
feeds the real ``p05_results.aggregate_predictions`` ensemble and the real
``p05_public_metrics.build_public_metrics`` strategy table into
``p05_reliability.build_reliability``; every other case uses small local
fixtures built with the same public helpers. No file, torch, fit, inference,
calibration or selection is touched.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p05_reliability as p05r
from atlas_sers.evaluation.classical import classification_metrics

CLASSES = ("A", "B", "C")
MODELS = p05r.P05_PUBLIC_MODELS
DEV_EXPERIMENT = "P05-CORE-DEV"
HELD_EXPERIMENT = "P05-CORE-T3"
PUBLIC_STATION = "cwa"
SENTINEL = "PRIVATE-SENTINEL-VALUE"
FORBIDDEN_TOKENS = (
    "context_id",
    "observation_uid",
    "master_sample_id",
    "true_label",
    "predicted_label",
    "probability",
    "class_vocabulary",
    "test_uid",
    "path",
    "PRIVATE",
)


def _prob(true_label, *, correct=True, confidence=0.9):
    index = CLASSES.index(true_label)
    values = np.full(3, (1.0 - confidence) / 2.0)
    if correct:
        values[index] = confidence
    else:
        values[(index + 1) % 3] = confidence
    return values


def _model_predictions(observations, model_id, mode):
    predictions = []
    for position, observation in enumerate(observations):
        label = observation["true_label"]
        if model_id in ("D0-M", "P05-SELECTED"):
            predictions.append(_prob(label, correct=(mode != "d0_wrong"), confidence=0.9))
        elif position % 2 == 0:
            predictions.append(_prob(label, correct=True, confidence=0.6))
        else:
            predictions.append(_prob(label, correct=False, confidence=0.7))
    return predictions


def _observations(prefix, specs):
    return [
        {
            "observation_uid": f"{prefix}-{position:02d}",
            "master_sample_id": master,
            "instrument": instrument,
            "true_label": label,
        }
        for position, (master, instrument, label) in enumerate(specs, start=1)
    ]


def _context(context_id, experiment_id, station, domain, held_instrument, observations, mode):
    return {
        "context_id": context_id,
        "experiment_id": experiment_id,
        "station": station,
        "domain": domain,
        "held_instrument": held_instrument,
        "observations": observations,
        "predictions": {
            model_id: _model_predictions(observations, model_id, mode) for model_id in MODELS
        },
    }


def _ensemble(contexts):
    records = []
    vocabulary = json.dumps(list(CLASSES))
    for context in contexts:
        for model_id in MODELS:
            for observation, probability in zip(
                context["observations"], context["predictions"][model_id], strict=True
            ):
                records.append(
                    {
                        "context_id": context["context_id"],
                        "experiment_id": context["experiment_id"],
                        "station": context["station"],
                        "domain": context["domain"],
                        "held_instrument": context["held_instrument"],
                        "model_id": model_id,
                        "observation_uid": observation["observation_uid"],
                        "master_sample_id": observation["master_sample_id"],
                        "instrument": observation["instrument"],
                        "true_label": observation["true_label"],
                        "predicted_label": CLASSES[int(np.argmax(probability))],
                        "class_vocabulary": vocabulary,
                        "probability_0": float(probability[0]),
                        "probability_1": float(probability[1]),
                        "probability_2": float(probability[2]),
                    }
                )
    return pd.DataFrame(records, columns=list(p05r._ENSEMBLE_REQUIRED))


def _classes_of(value):
    return tuple(json.loads(value) if isinstance(value, str) else value)


def _strategy_row(
    point_index,
    station,
    phase,
    domain,
    held_instrument,
    model_id,
    aggregation_id,
    ece,
    observations,
    physical_masters,
    observed_class_count,
):
    return {
        "point_index": point_index,
        "station": station,
        "phase": phase,
        "domain": domain,
        "held_instrument": held_instrument,
        "model_id": model_id,
        "aggregation_id": aggregation_id,
        "ece": ece,
        "observations": observations,
        "physical_masters": physical_masters,
        "observed_class_count": observed_class_count,
    }


def _strategy_rows(ensemble):
    frame = ensemble.copy()
    index = {
        identifier: position
        for position, identifier in enumerate(
            sorted({str(value) for value in frame["context_id"]}), start=1
        )
    }
    rows = []
    for context_id, cell in frame.groupby("context_id", sort=True):
        classes = _classes_of(cell["class_vocabulary"].iloc[0])
        class_index = {label: position for position, label in enumerate(classes)}
        station = str(cell["station"].iloc[0])
        phase = p05r.PHASE_BY_EXPERIMENT[str(cell["experiment_id"].iloc[0])]
        domain = str(cell["domain"].iloc[0])
        held_instrument = cell["held_instrument"].iloc[0]
        for model_id, model_cell in cell.groupby("model_id", sort=True):
            values = model_cell[list(p05r.PROBABILITY_COLUMNS)].to_numpy(dtype=float)
            truth = model_cell["true_label"].astype(str).to_numpy()
            indices = np.asarray([class_index[label] for label in truth], dtype=int)
            spectrum_values = values / values.sum(axis=1, keepdims=True)
            spectrum_ece = p05r.expected_calibration_error(
                spectrum_values, indices, bins=p05r.RELIABILITY_BINS
            )
            rows.append(
                _strategy_row(
                    index[str(context_id)],
                    station,
                    phase,
                    domain,
                    held_instrument,
                    str(model_id),
                    "M01",
                    spectrum_ece,
                    len(model_cell),
                    model_cell["master_sample_id"].astype(str).nunique(),
                    len(set(truth.tolist())),
                )
            )
            master = p05r.instrument_balanced_master_probabilities(
                probabilities=values,
                true_labels=truth,
                master_ids=model_cell["master_sample_id"].astype(str).to_numpy(),
                instruments=model_cell["instrument"].astype(str).to_numpy(),
                class_vocabulary=list(classes),
            )
            master_values = np.asarray(master["probabilities"].tolist(), dtype=float)
            master_values = master_values / master_values.sum(axis=1, keepdims=True)
            master_truth = master["true_label"].astype(str).to_numpy()
            master_indices = np.asarray([class_index[label] for label in master_truth], dtype=int)
            master_ece = p05r.expected_calibration_error(
                master_values, master_indices, bins=p05r.RELIABILITY_BINS
            )
            rows.append(
                _strategy_row(
                    index[str(context_id)],
                    station,
                    phase,
                    domain,
                    held_instrument,
                    str(model_id),
                    "M06",
                    master_ece,
                    len(master),
                    len(master),
                    len(set(master_truth.tolist())),
                )
            )
    return pd.DataFrame(rows, columns=list(p05r._STRATEGY_REQUIRED))


def _fixture():
    dev_one = _observations(
        "d1",
        [
            ("dm1", "INST-A", "A"),
            ("dm2", "INST-A", "A"),
            ("dm3", "INST-B", "B"),
            ("dm4", "INST-A", "B"),
            ("dm5", "INST-B", "C"),
            ("dm6", "INST-A", "C"),
            ("dm7", "INST-A", "A"),
            ("dm8", "INST-B", "B"),
            ("dm9", "INST-A", "C"),
            ("dm10", "INST-B", "A"),
        ],
    )
    dev_two = _observations(
        "d2",
        [
            ("dm11", "INST-A", "A"),
            ("dm12", "INST-B", "B"),
            ("dm13", "INST-A", "A"),
            ("dm14", "INST-B", "B"),
            ("dm15", "INST-A", "A"),
        ],
    )
    held_one = _observations(
        "h1",
        [
            ("hm1", "INST-H", "A"),
            ("hm2", "INST-H", "B"),
            ("hm3", "INST-H", "A"),
            ("hm4", "INST-H", "B"),
        ],
    )
    held_two = _observations(
        "h2",
        [("hm5", "INST-A", "C"), ("hm6", "INST-B", "C"), ("hm7", "INST-A", "C")],
    )
    contexts = [
        _context("ctx-dev-1", DEV_EXPERIMENT, PUBLIC_STATION, "D1", "", dev_one, "d0_correct"),
        _context("ctx-dev-2", DEV_EXPERIMENT, PUBLIC_STATION, "D1", "", dev_two, "d0_wrong"),
        _context("ctx-held-1", HELD_EXPERIMENT, "pills", "D2", "INST-H", held_one, "d0_correct"),
        _context("ctx-held-2", HELD_EXPERIMENT, "surfaces", "D2", "INST-H", held_two, "d0_correct"),
    ]
    ensemble = _ensemble(contexts)
    return ensemble, _strategy_rows(ensemble)


def _build(ensemble, strategy):
    return p05r.build_reliability(ensemble_predictions=ensemble, strategy_contexts=strategy)


def test_outputs_columns_counts_and_policies():
    ensemble, strategy = _fixture()
    result = _build(ensemble, strategy)
    assert set(result) == {"reliability_bins", "reliability_summary"}
    bins = result["reliability_bins"]
    summary = result["reliability_summary"]
    assert list(bins.columns) == list(p05r._RELIABILITY_BIN_COLUMNS)
    assert list(summary.columns) == list(p05r._RELIABILITY_SUMMARY_COLUMNS)
    assert set(summary["pooled_vs_mean_context_policy"]) == {p05r.POOLED_VS_MEAN_CONTEXT_POLICY}
    assert set(summary["independence_policy"]) == {p05r.INDEPENDENCE_POLICY}
    assert set(summary["endpoint_policy"]) == {p05r.ENDPOINT_POLICY}
    assert set(summary["diagnostic_policy"]) == {p05r.DIAGNOSTIC_POLICY}
    dev = bins[bins["station"].eq(PUBLIC_STATION) & bins["phase"].eq("development")]
    for _, cell in dev.groupby(["model_id", "aggregation_id"]):
        assert len(cell) == p05r.RELIABILITY_BINS
        assert int(cell["count"].sum()) == 15
        assert float(cell["bin_weight"].sum()) == pytest.approx(1.0, abs=1e-12)


def test_pooled_ece_matches_recomputation_and_differs_from_mean_context():
    ensemble, strategy = _fixture()
    result = _build(ensemble, strategy)
    bins = result["reliability_bins"]
    summary = result["reliability_summary"]
    for _, row in summary.iterrows():
        cell = bins[
            bins["station"].eq(row["station"])
            & bins["phase"].eq(row["phase"])
            & bins["model_id"].eq(row["model_id"])
            & bins["aggregation_id"].eq(row["aggregation_id"])
        ]
        recomputed = float((cell["bin_weight"] * cell["signed_gap"].abs()).sum())
        assert row["pooled_reliability_ece"] == pytest.approx(recomputed, abs=1e-12)
        assert int(row["total_appearances"]) == int(cell["count"].sum())
    dev = summary[
        summary["station"].eq(PUBLIC_STATION)
        & summary["phase"].eq("development")
        & summary["model_id"].eq("D0-M")
        & summary["aggregation_id"].eq("M01")
    ].iloc[0]
    assert dev["contributing_contexts"] == 2
    assert dev["total_appearances"] == 15
    assert dev["mean_context_ece"] == pytest.approx(0.5, abs=1e-12)
    assert dev["pooled_reliability_ece"] == pytest.approx(5.5 / 15.0, abs=1e-12)
    assert dev["pooled_reliability_ece"] != pytest.approx(dev["mean_context_ece"], abs=1e-12)


def test_equal_mass_bins_are_stable_under_confidence_ties():
    ensemble, strategy = _fixture()
    result = _build(ensemble, strategy)
    cell = result["reliability_bins"][
        result["reliability_bins"]["station"].eq(PUBLIC_STATION)
        & result["reliability_bins"]["phase"].eq("development")
        & result["reliability_bins"]["model_id"].eq("D0-M")
        & result["reliability_bins"]["aggregation_id"].eq("M01")
    ].reset_index(drop=True)
    assert cell["bin_index"].tolist() == list(range(1, 11))
    assert cell["count"].tolist() == [2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    assert np.allclose(cell["mean_confidence"].to_numpy(dtype=float), 0.9, atol=1e-12)
    assert cell["observed_accuracy"].tolist() == [1.0] * 5 + [0.0] * 5
    assert np.allclose(cell["signed_gap"].to_numpy(dtype=float), [0.1] * 5 + [-0.9] * 5, atol=1e-12)


def test_both_phases_all_models_and_sparse_class_counts():
    ensemble, strategy = _fixture()
    result = _build(ensemble, strategy)
    summary = result["reliability_summary"]
    assert set(summary["phase"]) == {"development", "held_evaluation"}
    assert set(summary["model_id"]) == set(MODELS)
    assert set(summary["aggregation_id"]) == {"M01", "M06"}
    assert {1, 2, 3} <= set(int(value) for value in strategy["observed_class_count"])
    assert set(strategy["phase"]) == {"development", "held_evaluation"}


def test_m06_unequal_instrument_counts_use_equal_instrument_average():
    observations = [
        {
            "observation_uid": "m06-1",
            "master_sample_id": "mm1",
            "instrument": "INST-A",
            "true_label": "A",
        },
        {
            "observation_uid": "m06-2",
            "master_sample_id": "mm1",
            "instrument": "INST-A",
            "true_label": "A",
        },
        {
            "observation_uid": "m06-3",
            "master_sample_id": "mm1",
            "instrument": "INST-B",
            "true_label": "A",
        },
    ]
    probabilities = {
        "D0-M": [(0.9, 0.1, 0.0), (0.8, 0.2, 0.0), (0.2, 0.7, 0.1)],
        "P05-SELECTED": [(0.9, 0.1, 0.0), (0.8, 0.2, 0.0), (0.2, 0.7, 0.1)],
        "D3": [(0.9, 0.1, 0.0), (0.8, 0.2, 0.0), (0.2, 0.7, 0.1)],
    }
    context = {
        "context_id": "ctx-m06",
        "experiment_id": DEV_EXPERIMENT,
        "station": PUBLIC_STATION,
        "domain": "D1",
        "held_instrument": "",
        "observations": observations,
        "predictions": {
            model_id: [np.asarray(row, dtype=float) for row in rows]
            for model_id, rows in probabilities.items()
        },
    }
    ensemble = _ensemble([context])
    strategy = _strategy_rows(ensemble)
    values = ensemble[ensemble["model_id"].eq("D0-M")][list(p05r.PROBABILITY_COLUMNS)].to_numpy(
        dtype=float
    )
    master = p05r.instrument_balanced_master_probabilities(
        probabilities=values,
        true_labels=np.asarray(["A", "A", "A"]),
        master_ids=np.asarray(["mm1", "mm1", "mm1"]),
        instruments=np.asarray(["INST-A", "INST-A", "INST-B"]),
        class_vocabulary=list(CLASSES),
    )
    expected = np.asarray([0.525, 0.425, 0.05])
    assert np.allclose(master["probabilities"].iloc[0], expected, atol=1e-12)
    row_average = values.mean(axis=0)
    vote = np.asarray([2.0 / 3.0, 1.0 / 3.0, 0.0])
    assert not np.allclose(expected, row_average, atol=1e-12)
    assert not np.allclose(expected, vote, atol=1e-12)
    result = _build(ensemble, strategy)
    summary = result["reliability_summary"]
    m06 = summary[summary["model_id"].eq("D0-M") & summary["aggregation_id"].eq("M06")].iloc[0]
    m01 = summary[summary["model_id"].eq("D0-M") & summary["aggregation_id"].eq("M01")].iloc[0]
    assert m06["total_appearances"] == 1
    assert m06["pooled_reliability_ece"] == pytest.approx(0.475, abs=1e-12)
    assert m01["total_appearances"] == 3
    assert m01["pooled_reliability_ece"] == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_private_sentinels_are_never_serialized():
    ensemble, strategy = _fixture()
    ensemble = ensemble.copy()
    strategy = strategy.copy()
    ensemble["PRIVATE_NOTE"] = SENTINEL
    ensemble["observation_uid_copy"] = SENTINEL
    strategy["PRIVATE_NOTE"] = SENTINEL
    result = _build(ensemble, strategy)
    serialized = "\n".join(table.to_csv(index=False) for table in result.values())
    assert SENTINEL not in serialized
    for table in result.values():
        for column in table.columns:
            for token in FORBIDDEN_TOKENS:
                assert token not in column


def test_build_does_not_mutate_inputs():
    ensemble, strategy = _fixture()
    ensemble_before = ensemble.copy(deep=True)
    strategy_before = strategy.copy(deep=True)
    _build(ensemble, strategy)
    pd.testing.assert_frame_equal(ensemble, ensemble_before)
    pd.testing.assert_frame_equal(strategy, strategy_before)


def test_near_normalized_probabilities_match_classification_metrics():
    observations = _observations(
        "n1",
        [
            ("nm1", "INST-A", "A"),
            ("nm2", "INST-B", "B"),
            ("nm3", "INST-A", "C"),
            ("nm4", "INST-B", "A"),
        ],
    )
    context = _context(
        "ctx-near", DEV_EXPERIMENT, PUBLIC_STATION, "D1", "", observations, "d0_correct"
    )
    ensemble = _ensemble([context])
    scaled = ensemble.copy()
    for column in p05r.PROBABILITY_COLUMNS:
        scaled[column] = scaled[column].astype(float) * (1.0 + 1e-7)
    strategy = _strategy_rows(scaled)
    result = _build(scaled, strategy)
    summary = result["reliability_summary"]
    model_cell = scaled[scaled["model_id"].eq("D0-M")]
    values = model_cell[list(p05r.PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    truth = model_cell["true_label"].astype(str).to_numpy()
    predicted = model_cell["predicted_label"].astype(str).to_numpy()
    metrics = classification_metrics(
        truth, predicted, class_vocabulary=CLASSES, probabilities=values
    )
    m01 = summary[summary["model_id"].eq("D0-M") & summary["aggregation_id"].eq("M01")].iloc[0]
    assert m01["pooled_reliability_ece"] == pytest.approx(metrics["ece"], abs=1e-12)
    master = p05r.instrument_balanced_master_probabilities(
        probabilities=values,
        true_labels=truth,
        master_ids=model_cell["master_sample_id"].astype(str).to_numpy(),
        instruments=model_cell["instrument"].astype(str).to_numpy(),
        class_vocabulary=list(CLASSES),
    )
    master_values = np.asarray(master["probabilities"].tolist(), dtype=float)
    master_truth = master["true_label"].astype(str).to_numpy()
    master_predicted = master["predicted_label"].astype(str).to_numpy()
    master_metrics = classification_metrics(
        master_truth, master_predicted, class_vocabulary=CLASSES, probabilities=master_values
    )
    m06 = summary[summary["model_id"].eq("D0-M") & summary["aggregation_id"].eq("M06")].iloc[0]
    assert m06["pooled_reliability_ece"] == pytest.approx(master_metrics["ece"], abs=1e-12)


def test_tie_probabilities_are_deterministic_and_accepted():
    observations = _observations(
        "t1", [("tm1", "INST-A", "A"), ("tm2", "INST-B", "B"), ("tm3", "INST-A", "C")]
    )
    ties = [
        np.asarray([0.5, 0.5, 0.0]),
        np.asarray([0.0, 0.5, 0.5]),
        np.asarray([0.5, 0.0, 0.5]),
    ]
    context = {
        "context_id": "ctx-tie",
        "experiment_id": DEV_EXPERIMENT,
        "station": PUBLIC_STATION,
        "domain": "D1",
        "held_instrument": "",
        "observations": observations,
        "predictions": {model_id: [row.copy() for row in ties] for model_id in MODELS},
    }
    ensemble = _ensemble([context])
    d0 = ensemble[ensemble["model_id"].eq("D0-M")]
    assert d0["predicted_label"].tolist() == ["A", "B", "A"]
    strategy = _strategy_rows(ensemble)
    result = _build(ensemble, strategy)
    summary = result["reliability_summary"]
    m01 = summary[summary["model_id"].eq("D0-M") & summary["aggregation_id"].eq("M01")].iloc[0]
    assert m01["pooled_reliability_ece"] == pytest.approx(0.5, abs=1e-12)


def test_handoff_from_p05_results_and_public_metrics():
    from tests.test_p05_public_metrics import _build as build_public
    from tests.test_p05_public_metrics import _compare, _cwa_fixture, _rename_station

    fixture = _cwa_fixture()
    # Preserve both development and held contexts from the actual aggregator.
    # The comparison-only convenience fixture contains held metrics alone.
    aggregation = {
        name: _rename_station(frame) for name, frame in fixture["aggregate"].items()
    }
    comparison = _compare(fixture)
    public = build_public(aggregation, comparison)
    strategy = public["strategy_contexts"].copy()
    ensemble = aggregation["ensemble_predictions"].copy()
    ensemble = ensemble[ensemble["model_id"].isin(p05r.P05_PUBLIC_MODELS)].reset_index(drop=True)
    result = _build(ensemble, strategy)
    assert set(result) == {"reliability_bins", "reliability_summary"}
    assert not result["reliability_bins"].empty
    assert set(result["reliability_summary"]["phase"]) == {"development", "held_evaluation"}


def test_handoff_from_aggregate_predictions_output():
    from tests.test_p05_results import _base, _run

    out = _run(_base())
    ensemble = out["ensemble_predictions"].copy()
    ensemble["station"] = PUBLIC_STATION
    ensemble["experiment_id"] = ensemble["experiment_id"].replace(
        {"P04-CORE-DEV": DEV_EXPERIMENT, "P04-CORE-T3": HELD_EXPERIMENT}
    )
    ensemble = ensemble[ensemble["model_id"].isin(p05r.P05_PUBLIC_MODELS)].reset_index(drop=True)
    strategy = _strategy_rows(ensemble)
    result = _build(ensemble, strategy)
    assert not result["reliability_summary"].empty


def _mut_ensemble_missing_column(ensemble, strategy):
    ensemble.drop(columns=["predicted_label"], inplace=True)


def _mut_ensemble_duplicate_key(ensemble, strategy):
    ensemble.loc[len(ensemble.index)] = ensemble.iloc[0]


def _mut_ensemble_uid_mismatch(ensemble, strategy):
    mask = (
        ensemble["context_id"].eq("ctx-dev-1")
        & ensemble["model_id"].eq("D3")
        & ensemble["observation_uid"].eq("d1-01")
    )
    ensemble.loc[mask, "observation_uid"] = "d1-99"


def _mut_ensemble_master_conflict(ensemble, strategy):
    mask = ensemble["context_id"].eq("ctx-dev-1") & ensemble["observation_uid"].eq("d1-02")
    ensemble.loc[mask, "master_sample_id"] = "dm1"
    ensemble.loc[mask, "true_label"] = "B"


def _mut_ensemble_wrong_class_order(ensemble, strategy):
    ensemble["class_vocabulary"] = json.dumps(["B", "A", "C"])


def _mut_ensemble_invalid_vocab_json(ensemble, strategy):
    ensemble.loc[ensemble.index[0], "class_vocabulary"] = "not-json"


def _mut_ensemble_vocab_string(ensemble, strategy):
    ensemble["class_vocabulary"] = json.dumps("A")


def _mut_ensemble_bool_probability(ensemble, strategy):
    ensemble["probability_0"] = ensemble["probability_0"].astype(object)
    ensemble.loc[ensemble.index[0], "probability_0"] = True


def _mut_ensemble_nan_probability(ensemble, strategy):
    ensemble.loc[ensemble.index[0], "probability_1"] = np.nan


def _mut_ensemble_nonfinite_probability(ensemble, strategy):
    ensemble.loc[ensemble.index[0], "probability_0"] = np.inf


def _mut_ensemble_negative_probability(ensemble, strategy):
    ensemble.loc[ensemble.index[0], "probability_0"] = -0.1


def _mut_ensemble_unnormalized_probability(ensemble, strategy):
    for column in p05r.PROBABILITY_COLUMNS:
        ensemble.loc[ensemble.index[0], column] = 0.5


def _mut_ensemble_predicted_mismatch(ensemble, strategy):
    index = ensemble.index[0]
    current = ensemble.loc[index, "predicted_label"]
    ensemble.loc[index, "predicted_label"] = next(label for label in CLASSES if label != current)


def _mut_ensemble_true_label_unknown(ensemble, strategy):
    ensemble.loc[ensemble.index[0], "true_label"] = "Z"


def _mut_ensemble_station_unknown(ensemble, strategy):
    ensemble.loc[ensemble.index[0], "station"] = "STN1"


def _mut_ensemble_phase_unknown(ensemble, strategy):
    ensemble.loc[ensemble.index[0], "experiment_id"] = "P05-UNKNOWN"


def _mut_ensemble_model_unknown(ensemble, strategy):
    ensemble.loc[ensemble.index[0], "model_id"] = "D9"


def _mut_ensemble_held_invalid(ensemble, strategy):
    ensemble["held_instrument"] = ensemble["held_instrument"].astype(object)
    ensemble.loc[ensemble.index[0], "held_instrument"] = 1


def _mut_ensemble_empty(ensemble, strategy):
    ensemble.drop(ensemble.index, inplace=True)


def _mut_ensemble_context_model_incomplete(ensemble, strategy):
    mask = ensemble["context_id"].eq("ctx-dev-1") & ensemble["model_id"].eq("D3")
    ensemble.drop(ensemble.index[mask], inplace=True)


def _mut_ensemble_context_metadata_conflict(ensemble, strategy):
    mask = ensemble["context_id"].eq("ctx-dev-1") & ensemble["model_id"].eq("D3")
    ensemble.loc[mask, "domain"] = "D9"


def _mut_ensemble_context_vocab_conflict(ensemble, strategy):
    mask = ensemble["context_id"].eq("ctx-dev-1") & ensemble["model_id"].eq("D3")
    ensemble.loc[mask, "class_vocabulary"] = json.dumps(["A", "B", "D"])


def _mut_strategy_missing_column(ensemble, strategy):
    strategy.drop(columns=["ece"], inplace=True)


def _mut_strategy_duplicate_key(ensemble, strategy):
    strategy.loc[len(strategy.index)] = strategy.iloc[0]


def _mut_strategy_context_incomplete(ensemble, strategy):
    strategy.drop(strategy.index[0], inplace=True)


def _mut_strategy_point_index(ensemble, strategy):
    strategy.loc[strategy.index[0], "point_index"] = 99


def _mut_strategy_station_unknown(ensemble, strategy):
    strategy.loc[strategy.index[0], "station"] = "STN1"


def _mut_strategy_phase_unknown(ensemble, strategy):
    strategy.loc[strategy.index[0], "phase"] = "bogus"


def _mut_strategy_model_unknown(ensemble, strategy):
    strategy.loc[strategy.index[0], "model_id"] = "D9"


def _mut_strategy_aggregation_unknown(ensemble, strategy):
    strategy.loc[strategy.index[0], "aggregation_id"] = "M99"


def _mut_strategy_ece_wrong(ensemble, strategy):
    index = strategy.index[0]
    strategy.loc[index, "ece"] = float(strategy.loc[index, "ece"]) + 0.01


def _mut_strategy_ece_nonfinite(ensemble, strategy):
    strategy.loc[strategy.index[0], "ece"] = np.inf


def _mut_strategy_ece_bool(ensemble, strategy):
    strategy["ece"] = strategy["ece"].astype(object)
    strategy.loc[strategy.index[0], "ece"] = True


def _mut_strategy_ece_negative(ensemble, strategy):
    strategy.loc[strategy.index[0], "ece"] = -0.1


def _mut_strategy_observations(ensemble, strategy):
    index = strategy.index[strategy["aggregation_id"].eq("M01")][0]
    strategy.loc[index, "observations"] = int(strategy.loc[index, "observations"]) + 1


def _mut_strategy_masters(ensemble, strategy):
    index = strategy.index[strategy["aggregation_id"].eq("M01")][0]
    strategy.loc[index, "physical_masters"] = int(strategy.loc[index, "physical_masters"]) - 1


def _mut_strategy_class_count(ensemble, strategy):
    index = strategy.index[strategy["aggregation_id"].eq("M01")][0]
    strategy.loc[index, "observed_class_count"] = 2


def _mut_strategy_m06_count(ensemble, strategy):
    index = strategy.index[strategy["aggregation_id"].eq("M06")][0]
    strategy.loc[index, "observations"] = int(strategy.loc[index, "observations"]) + 1


def _mut_strategy_count_nonpositive(ensemble, strategy):
    index = strategy.index[strategy["aggregation_id"].eq("M01")][0]
    strategy.loc[index, "observations"] = 0


def _mut_strategy_metadata_conflict(ensemble, strategy):
    index = strategy.index[strategy["point_index"].eq(1) & strategy["model_id"].eq("D3")][0]
    strategy.loc[index, "domain"] = "D9"


def _mut_strategy_alignment_mismatch(ensemble, strategy):
    mask = strategy["point_index"].eq(1)
    strategy.loc[mask, "station"] = "pills"


def _mut_strategy_held_invalid(ensemble, strategy):
    strategy["held_instrument"] = strategy["held_instrument"].astype(object)
    strategy.loc[strategy.index[0], "held_instrument"] = 1


REJECTIONS = (
    ("ensemble_missing_column", _mut_ensemble_missing_column),
    ("ensemble_duplicate_key", _mut_ensemble_duplicate_key),
    ("ensemble_uid_mismatch", _mut_ensemble_uid_mismatch),
    ("ensemble_master_conflict", _mut_ensemble_master_conflict),
    ("ensemble_wrong_class_order", _mut_ensemble_wrong_class_order),
    ("ensemble_invalid_vocab_json", _mut_ensemble_invalid_vocab_json),
    ("ensemble_vocab_string", _mut_ensemble_vocab_string),
    ("ensemble_bool_probability", _mut_ensemble_bool_probability),
    ("ensemble_nan_probability", _mut_ensemble_nan_probability),
    ("ensemble_nonfinite_probability", _mut_ensemble_nonfinite_probability),
    ("ensemble_negative_probability", _mut_ensemble_negative_probability),
    ("ensemble_unnormalized_probability", _mut_ensemble_unnormalized_probability),
    ("ensemble_predicted_mismatch", _mut_ensemble_predicted_mismatch),
    ("ensemble_true_label_unknown", _mut_ensemble_true_label_unknown),
    ("ensemble_station_unknown", _mut_ensemble_station_unknown),
    ("ensemble_phase_unknown", _mut_ensemble_phase_unknown),
    ("ensemble_model_unknown", _mut_ensemble_model_unknown),
    ("ensemble_held_invalid", _mut_ensemble_held_invalid),
    ("ensemble_empty", _mut_ensemble_empty),
    ("ensemble_context_model_incomplete", _mut_ensemble_context_model_incomplete),
    ("ensemble_context_metadata_conflict", _mut_ensemble_context_metadata_conflict),
    ("ensemble_context_vocab_conflict", _mut_ensemble_context_vocab_conflict),
    ("strategy_missing_column", _mut_strategy_missing_column),
    ("strategy_duplicate_key", _mut_strategy_duplicate_key),
    ("strategy_context_incomplete", _mut_strategy_context_incomplete),
    ("strategy_point_index", _mut_strategy_point_index),
    ("strategy_station_unknown", _mut_strategy_station_unknown),
    ("strategy_phase_unknown", _mut_strategy_phase_unknown),
    ("strategy_model_unknown", _mut_strategy_model_unknown),
    ("strategy_aggregation_unknown", _mut_strategy_aggregation_unknown),
    ("strategy_ece_wrong", _mut_strategy_ece_wrong),
    ("strategy_ece_nonfinite", _mut_strategy_ece_nonfinite),
    ("strategy_ece_bool", _mut_strategy_ece_bool),
    ("strategy_ece_negative", _mut_strategy_ece_negative),
    ("strategy_observations", _mut_strategy_observations),
    ("strategy_masters", _mut_strategy_masters),
    ("strategy_class_count", _mut_strategy_class_count),
    ("strategy_m06_count", _mut_strategy_m06_count),
    ("strategy_count_nonpositive", _mut_strategy_count_nonpositive),
    ("strategy_metadata_conflict", _mut_strategy_metadata_conflict),
    ("strategy_alignment_mismatch", _mut_strategy_alignment_mismatch),
    ("strategy_held_invalid", _mut_strategy_held_invalid),
)


@pytest.mark.parametrize(
    "mutate", [case for _, case in REJECTIONS], ids=[name for name, _ in REJECTIONS]
)
def test_malformed_inputs_are_rejected(mutate):
    ensemble, strategy = _fixture()
    mutate(ensemble, strategy)
    with pytest.raises(p05r.P05ReliabilityError):
        _build(ensemble, strategy)
