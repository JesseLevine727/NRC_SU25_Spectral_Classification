"""Focused synthetic tests for deterministic P06/P11 frozen-panel metrics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.classical import classification_metrics
from atlas_sers.evaluation.p06p11_metrics import MODEL_IDS, build_metrics

MODELS = list(MODEL_IDS)
VOCAB = ("alpha", "beta", "gamma")
VOCAB_JSON = json.dumps(list(VOCAB))

CONTEXTS = [
    ("c1", "d1", "cwa", "i1", ["m1", "m2"]),
    ("c2", "d1", "cwa", "i1", ["m3"]),
    ("c3", "d2", "cwa", "i2", ["m4", "m5"]),
    ("c4", "d3", "pills", "i3", ["m6"]),
]
LABELS = {"m1": "alpha", "m2": "beta", "m3": "alpha", "m4": "beta", "m5": "alpha", "m6": "alpha"}
PREDICTED = {"M01": {("D3", "m2"): "alpha"}, "M06": {("D3", "m5"): "beta"}}
MISSING = {"P05-SELECTED": {"c4"}}


def _units(endpoint):
    result = []
    for context_id, _, _, _, masters in CONTEXTS:
        for master in masters:
            replicas = 2 if (endpoint == "M01" and master == "m1") else 1
            for replica in range(replicas):
                suffix = f"{replica}" if endpoint == "M01" else "0"
                result.append((context_id, master, f"{endpoint}-{context_id}-{master}-{suffix}"))
    return result


def _row_values(endpoint, model, master):
    true_label = LABELS[master]
    predicted = PREDICTED[endpoint].get((model, master), true_label)
    if model == "C-RBF-SVM":
        probabilities = {name: 0.0 for name in VOCAB}
        probabilities[predicted] = 1.0
        return true_label, predicted, probabilities
    if model == "C-RANDOM-FOREST" and master == "m1":
        return true_label, "alpha", {"alpha": 0.6, "beta": 0.2, "gamma": 0.2}
    if model == "C-RANDOM-FOREST" and master == "m2":
        return true_label, "beta", {"alpha": 0.175, "beta": 0.65, "gamma": 0.175}
    if predicted == true_label:
        probabilities = {name: 0.05 for name in VOCAB}
        probabilities[predicted] = 0.9
    else:
        probabilities = {name: 0.05 for name in VOCAB if name not in (predicted, true_label)}
        probabilities[predicted] = 0.6
        probabilities[true_label] = 0.35
    return true_label, predicted, probabilities


def _panel(endpoint):
    context_lookup = {
        name: (domain, station, instrument) for name, domain, station, instrument, _ in CONTEXTS
    }
    rows = []
    for context_id, master, unit_id in _units(endpoint):
        domain, station, instrument = context_lookup[context_id]
        for model in MODELS:
            if context_id in MISSING.get(model, set()):
                continue
            true_label, predicted, probabilities = _row_values(endpoint, model, master)
            rows.append(
                {
                    "context_id": context_id,
                    "domain": domain,
                    "station": station,
                    "instrument": instrument,
                    "master_sample_id": master,
                    "unit_id": unit_id,
                    "true_label": true_label,
                    "model_id": model,
                    "class_vocabulary": VOCAB_JSON,
                    "probability_0": probabilities[VOCAB[0]],
                    "probability_1": probabilities[VOCAB[1]],
                    "probability_2": probabilities[VOCAB[2]],
                    "predicted_label": predicted,
                    "correct": int(predicted == true_label),
                }
            )
    return pd.DataFrame(rows)


def _panels():
    return {"M01": _panel("M01"), "M06": _panel("M06")}


def _confusion_cell(outputs, endpoint, model, station, true, predicted):
    frame = outputs["confusion"]
    row = frame[
        frame.scope.eq("full_support")
        & frame.aggregation_id.eq(endpoint)
        & frame.model_id.eq(model)
        & frame.station.eq(station)
        & frame.true_chemical.eq(true)
        & frame.predicted_chemical.eq(predicted)
    ]
    return row.iloc[0]


def _context_ece(frame):
    classes = tuple(json.loads(frame["class_vocabulary"].iloc[0]))
    return classification_metrics(
        frame["true_label"].to_numpy(),
        frame["predicted_label"].to_numpy(),
        class_vocabulary=list(classes),
        probabilities=frame[["probability_0", "probability_1", "probability_2"]].to_numpy(
            dtype=float
        ),
    )["ece"]


def test_six_tables_scopes_and_endpoints():
    outputs = build_metrics(_panels())
    assert set(outputs) == {
        "domain_metrics",
        "model_summary",
        "confusion",
        "class_sensitivity",
        "reliability_bins",
        "reliability_summary",
    }
    domain = outputs["domain_metrics"]
    assert set(domain.scope) == {"full_support", "primary_common"}
    assert set(domain.aggregation_id) == {"M01", "M06"}
    assert set(domain.model_id) == set(MODELS)
    summary = outputs["model_summary"]
    assert set(summary[summary.scope.eq("primary_common")].model_id) == {
        "P05-SELECTED",
        "C-SELECTED",
    }


def test_endpoints_differ_and_counts_not_doubled():
    panels = _panels()
    outputs = build_metrics(panels)
    summary = outputs["model_summary"]
    full = summary[summary.scope.eq("full_support")]
    for endpoint in ("M01", "M06"):
        expected_rows = len(panels[endpoint][panels[endpoint].model_id.eq("D0-M")])
        expected_masters = panels[endpoint][panels[endpoint].model_id.eq("D0-M")][
            "master_sample_id"
        ].nunique()
        row = full[full.aggregation_id.eq(endpoint) & full.model_id.eq("D0-M")].iloc[0]
        assert row.unit_appearances == expected_rows
        assert row.physical_masters == expected_masters
        assert row.distinct_units == expected_rows
    assert (
        full[full.aggregation_id.eq("M01") & full.model_id.eq("D0-M")].iloc[0].unit_appearances == 7
    )
    assert (
        full[full.aggregation_id.eq("M06") & full.model_id.eq("D0-M")].iloc[0].unit_appearances == 6
    )
    assert _confusion_cell(outputs, "M01", "D3", "cwa", "beta", "alpha")["count"] == 1
    assert _confusion_cell(outputs, "M06", "D3", "cwa", "beta", "alpha")["count"] == 0
    assert _confusion_cell(outputs, "M06", "D3", "cwa", "alpha", "beta")["count"] == 1
    assert _confusion_cell(outputs, "M01", "D3", "cwa", "alpha", "beta")["count"] == 0


def test_hand_known_confusion_and_absent_row():
    outputs = build_metrics(_panels())
    assert _confusion_cell(outputs, "M01", "D3", "cwa", "alpha", "alpha")["count"] == 4
    assert _confusion_cell(outputs, "M01", "D3", "cwa", "beta", "beta")["count"] == 1
    cell = _confusion_cell(outputs, "M01", "D3", "cwa", "beta", "alpha")
    assert cell["true_appearances"] == 2
    assert cell["row_fraction"] == pytest.approx(0.5)
    absent = _confusion_cell(outputs, "M01", "D3", "cwa", "gamma", "alpha")
    assert absent["count"] == 0
    assert absent["true_appearances"] == 0
    assert np.isnan(absent["row_fraction"])


def test_class_sensitivity_absent_and_present():
    outputs = build_metrics(_panels())
    frame = outputs["class_sensitivity"]
    present = frame[
        frame.scope.eq("full_support")
        & frame.aggregation_id.eq("M01")
        & frame.model_id.eq("D3")
        & frame.station.eq("cwa")
        & frame.chemical.eq("beta")
    ].iloc[0]
    assert present.true_appearances == 2
    assert present.physical_masters == 2
    assert present.contributing_contexts == 2
    assert present.contributing_domains == 2
    assert present.pooled_recall == pytest.approx(0.5)
    assert present.mean_context_recall == pytest.approx(0.5)
    assert present.mean_domain_recall == pytest.approx(0.5)
    absent = frame[
        frame.scope.eq("full_support")
        & frame.aggregation_id.eq("M01")
        & frame.model_id.eq("D3")
        & frame.station.eq("cwa")
        & frame.chemical.eq("gamma")
    ].iloc[0]
    assert absent.true_appearances == 0
    assert absent.physical_masters == 0
    assert absent.contributing_contexts == 0
    assert absent.contributing_domains == 0
    assert np.isnan(absent.pooled_recall)
    assert np.isnan(absent.mean_context_recall)
    assert np.isnan(absent.mean_domain_recall)


def test_repeated_masters_counted_once():
    outputs = build_metrics(_panels())
    domain = outputs["domain_metrics"]
    row = domain[
        domain.scope.eq("full_support")
        & domain.aggregation_id.eq("M01")
        & domain.model_id.eq("D0-M")
        & domain.domain.eq("d1")
    ].iloc[0]
    assert row.contexts == 2
    assert row.unit_appearances == 4
    assert row.physical_masters == 3


def test_model_summary_uses_equal_domain_mean():
    outputs = build_metrics(_panels())
    domain = outputs["domain_metrics"]
    summary = outputs["model_summary"]
    row = summary[
        summary.scope.eq("full_support")
        & summary.aggregation_id.eq("M01")
        & summary.model_id.eq("D3")
    ].iloc[0]
    subset = domain[
        domain.scope.eq("full_support") & domain.aggregation_id.eq("M01") & domain.model_id.eq("D3")
    ]
    equal = subset.balanced_accuracy.mean()
    weighted = (subset.balanced_accuracy * subset.contexts).sum() / subset.contexts.sum()
    assert row.balanced_accuracy == pytest.approx(equal)
    assert row.balanced_accuracy != pytest.approx(weighted)
    assert row.domains == 3


def test_primary_common_excludes_missing_context_and_pairs():
    outputs = build_metrics(_panels())
    summary = outputs["model_summary"]
    for endpoint in ("M01", "M06"):
        selected = summary[summary.scope.eq("primary_common") & summary.aggregation_id.eq(endpoint)]
        p05 = selected[selected.model_id.eq("P05-SELECTED")].iloc[0]
        c_selected = selected[selected.model_id.eq("C-SELECTED")].iloc[0]
        assert p05.contexts == 3
        assert c_selected.contexts == 3
    full = outputs["model_summary"]
    c_full = full[
        full.scope.eq("full_support")
        & full.aggregation_id.eq("M01")
        & full.model_id.eq("C-SELECTED")
    ].iloc[0]
    assert c_full.contexts == 4


def test_reliability_bins_fixed_width_edge_and_empty():
    panels = _panels()
    outputs = build_metrics(panels)
    bins = outputs["reliability_bins"]
    subset = bins[
        bins.scope.eq("full_support")
        & bins.aggregation_id.eq("M06")
        & bins.model_id.eq("C-RBF-SVM")
        & bins.station.eq("cwa")
    ]
    assert len(subset) == 10
    assert list(subset.bin_index) == list(range(10))
    top = subset[subset.bin_index.eq(9)].iloc[0]
    assert top.lower == pytest.approx(0.9)
    assert top.upper == pytest.approx(1.0)
    expected = len(
        panels["M06"][panels["M06"].model_id.eq("C-RBF-SVM") & panels["M06"].station.eq("cwa")]
    )
    assert top["count"] == expected
    empty = subset[subset["count"].eq(0)]
    assert len(empty) > 0
    assert empty.mean_confidence.isna().all()
    assert empty.observed_accuracy.isna().all()


def test_reliability_summary_estimators_and_difference():
    panels = _panels()
    outputs = build_metrics(panels)
    summary = outputs["reliability_summary"]
    bins = outputs["reliability_bins"]
    row = summary[
        summary.scope.eq("full_support")
        & summary.aggregation_id.eq("M01")
        & summary.model_id.eq("D3")
        & summary.station.eq("cwa")
    ].iloc[0]
    subset = bins[
        bins.scope.eq("full_support")
        & bins.aggregation_id.eq("M01")
        & bins.model_id.eq("D3")
        & bins.station.eq("cwa")
    ]
    total = subset["count"].sum()
    expected_width = sum(
        item.count / total * abs(item.observed_accuracy - item.mean_confidence)
        for item in subset.itertuples()
        if item.count
    )
    assert row.pooled_ece_equal_width == pytest.approx(expected_width)
    frame = panels["M01"]
    frame = frame[frame.model_id.eq("D3") & frame.station.eq("cwa")]
    expected_mass = np.mean(
        [_context_ece(frame[frame.context_id.eq(c)]) for c in sorted(frame.context_id.unique())]
    )
    assert row.mean_context_ece_equal_mass == pytest.approx(expected_mass)
    assert row.pooled_ece_equal_width != pytest.approx(row.mean_context_ece_equal_mass)


def test_no_private_fields_and_integer_counts():
    outputs = build_metrics(_panels())
    private = {
        "context_id",
        "unit_id",
        "master_sample_id",
        "class_vocabulary",
        "probability_0",
        "probability_1",
        "probability_2",
        "source",
        "source_path",
    }
    for table in outputs.values():
        assert not (set(table.columns) & private)
    for column in ("contexts", "unit_appearances", "physical_masters"):
        assert outputs["domain_metrics"][column].dtype == np.int64
    assert outputs["reliability_bins"]["count"].dtype == np.int64
    assert outputs["reliability_bins"]["bin_index"].dtype == np.int64


def test_deterministic_under_row_shuffle():
    panels = _panels()
    expected = build_metrics(panels)
    shuffled = {
        endpoint: panels[endpoint].sample(frac=1.0, random_state=11).reset_index(drop=True)
        for endpoint in panels
    }
    actual = build_metrics(shuffled)
    for name in expected:
        pd.testing.assert_frame_equal(expected[name], actual[name])


def test_inputs_not_mutated():
    panels = _panels()
    copies = {endpoint: panels[endpoint].copy(deep=True) for endpoint in panels}
    build_metrics(panels)
    for endpoint in panels:
        pd.testing.assert_frame_equal(panels[endpoint], copies[endpoint])


def test_reject_duplicate_columns_and_missing_columns():
    panels = _panels()
    panels["M01"] = pd.concat([panels["M01"], panels["M01"][["predicted_label"]]], axis=1)
    with pytest.raises(ValueError):
        build_metrics(panels)
    panels = _panels()
    panels["M06"] = panels["M06"].drop(columns=["correct"])
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_reject_probability_sum_type_range_boolean():
    panels = _panels()
    panels["M01"].loc[0, "probability_0"] = 0.5
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    frame = panels["M01"]
    frame["probability_1"] = ["bad"] * len(frame)
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    frame = panels["M01"]
    frame["probability_2"] = [1.5] * len(frame)
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    frame = panels["M01"]
    frame["probability_0"] = np.ones(len(frame), dtype=bool)
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_reject_prediction_label_and_inconsistent_correct():
    panels = _panels()
    panels["M01"].loc[0, "predicted_label"] = "beta"
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    panels["M01"].loc[0, "correct"] = 1 - panels["M01"].loc[0, "correct"]
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    frame = panels["M01"]
    frame["correct"] = np.ones(len(frame), dtype=bool)
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_boolean_correct_accepted_across_endpoints():
    expected = build_metrics(_panels())
    panels = _panels()
    for endpoint in panels:
        panels[endpoint]["correct"] = panels[endpoint]["correct"].astype(bool)
    actual = build_metrics(panels)
    assert set(actual) == set(expected)
    for name in expected:
        pd.testing.assert_frame_equal(expected[name], actual[name])


def test_object_float_correct_accepted():
    expected = build_metrics(_panels())
    panels = _panels()
    for endpoint in panels:
        panels[endpoint]["correct"] = panels[endpoint]["correct"].astype(float).astype(object)
    actual = build_metrics(panels)
    for name in expected:
        pd.testing.assert_frame_equal(expected[name], actual[name])


def test_reject_complex_and_datetime_correct():
    panels = _panels()
    panels["M01"]["correct"] = panels["M01"]["correct"].astype(np.complex128)
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    panels["M01"]["correct"] = panels["M01"]["correct"].astype("datetime64[s]")
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_reject_complex_probability():
    panels = _panels()
    panels["M06"]["probability_1"] = panels["M06"]["probability_1"].astype(np.complex128)
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_reject_string_correct_and_probability_even_if_convertible():
    panels = _panels()
    panels["M01"]["correct"] = panels["M01"]["correct"].astype(str)
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    panels["M01"]["probability_2"] = panels["M01"]["probability_2"].astype(str)
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_reject_untrimmed_vocabulary_names():
    panels = _panels()
    panels["M01"]["class_vocabulary"] = json.dumps(["alpha", "beta", "gamma "])
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_reject_true_label_outside_vocabulary():
    panels = _panels()
    frame = panels["M01"]
    row = frame.index[0]
    frame.loc[row, "true_label"] = "omega"
    frame.loc[row, "correct"] = int(
        frame.loc[row, "true_label"] == frame.loc[row, "predicted_label"]
    )
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_reject_cross_endpoint_vocabulary_mismatch():
    panels = _panels()
    frame = panels["M06"]
    mutated = False
    for station in sorted(frame["station"].unique()):
        rows = frame["station"].eq(station)
        observed = set(frame.loc[rows, "true_label"])
        current = json.loads(frame.loc[rows, "class_vocabulary"].iloc[0])
        unobserved = [name for name in current if name not in observed]
        if unobserved:
            new_vocabulary = sorted((set(current) - {unobserved[0]}) | {"omega"})
            frame.loc[rows, "class_vocabulary"] = json.dumps(new_vocabulary)
            mutated = True
            break
    assert mutated
    with pytest.raises(ValueError):
        build_metrics(panels)


def _hand_classification_metrics(subset):
    vocabulary = json.loads(subset["class_vocabulary"].iloc[0])
    probabilities = subset[["probability_0", "probability_1", "probability_2"]].to_numpy(
        dtype=np.float64
    )
    truths = subset["true_label"].tolist()
    predicted = subset["predicted_label"].tolist()
    class_index = {name: position for position, name in enumerate(vocabulary)}
    true_index = np.asarray([class_index[name] for name in truths])
    nll = float(-np.log(probabilities[np.arange(len(subset)), true_index]).mean())
    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(subset)), true_index] = 1.0
    brier = float(((probabilities - onehot) ** 2).sum(axis=1).mean())
    f1_scores = []
    for name in vocabulary:
        tp = sum(1 for t, p in zip(truths, predicted, strict=True) if t == name and p == name)
        fp = sum(1 for t, p in zip(truths, predicted, strict=True) if t != name and p == name)
        fn = sum(1 for t, p in zip(truths, predicted, strict=True) if t == name and p != name)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_scores.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    return nll, brier, float(np.mean(f1_scores))


def test_explicit_hand_metrics():
    frame = _panels()["M01"]
    subset = frame[frame["model_id"].eq("D3") & frame["context_id"].eq("c1")].reset_index(drop=True)
    hand_nll, hand_brier, hand_macro_f1 = _hand_classification_metrics(subset)
    computed = classification_metrics(
        subset["true_label"].to_numpy(),
        subset["predicted_label"].to_numpy(),
        class_vocabulary=VOCAB,
        probabilities=subset[["probability_0", "probability_1", "probability_2"]].to_numpy(),
    )
    assert computed["negative_log_likelihood"] == pytest.approx(hand_nll)
    assert computed["brier_score"] == pytest.approx(hand_brier)
    assert computed["macro_f1"] == pytest.approx(hand_macro_f1)


def test_explicit_equal_context_metrics():
    panels = _panels()
    frame = panels["M01"]
    subset = frame[frame["model_id"].eq("D3") & frame["domain"].eq("d1")]
    assert sorted(subset["context_id"].unique()) == ["c1", "c2"]
    per_context = {}
    for context_id, group in subset.groupby("context_id", sort=True):
        per_context[context_id] = classification_metrics(
            group["true_label"].to_numpy(),
            group["predicted_label"].to_numpy(),
            class_vocabulary=VOCAB,
            probabilities=group[["probability_0", "probability_1", "probability_2"]].to_numpy(),
        )
    rows = build_metrics(panels)["domain_metrics"]
    selected = rows[
        rows["model_id"].eq("D3")
        & rows["domain"].eq("d1")
        & rows["aggregation_id"].eq("M01")
        & rows["scope"].eq("full_support")
    ]
    assert len(selected) == 1
    row = selected.iloc[0]
    for output_metric in ("negative_log_likelihood", "brier_score", "macro_f1"):
        context_values = [
            per_context["c1"][output_metric],
            per_context["c2"][output_metric],
        ]
        assert row[output_metric] == pytest.approx(np.mean(context_values))


def test_reject_vocabulary_and_text():
    panels = _panels()
    panels["M01"]["class_vocabulary"] = json.dumps(["beta", "alpha", "gamma"])
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    panels["M01"].loc[0, "context_id"] = " c1"
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_reject_identity_and_enum_errors():
    panels = _panels()
    duplicated = pd.concat([panels["M01"], panels["M01"].iloc[[0]]], ignore_index=True)
    panels["M01"] = duplicated
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    panels["M01"].loc[0, "master_sample_id"] = "m6"
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    panels["M01"]["model_id"] = panels["M01"]["model_id"].replace("D0-M", "UNKNOWN")
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    panels["M01"].loc[0, "station"] = "mars"
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_reject_m06_master_duplicate():
    panels = _panels()
    frame = panels["M06"]
    row = (
        frame[
            frame.model_id.eq("D0-M") & frame.context_id.eq("c1") & frame.master_sample_id.eq("m1")
        ]
        .iloc[[0]]
        .copy()
    )
    row["unit_id"] = "extra-unit"
    panels["M06"] = pd.concat([frame, row], ignore_index=True)
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_cross_endpoint_guards():
    panels = _panels()
    panels["M06"] = panels["M06"][
        ~(panels["M06"].model_id.eq("P05-SELECTED") & panels["M06"].context_id.eq("c1"))
    ].reset_index(drop=True)
    with pytest.raises(ValueError):
        build_metrics(panels)

    panels = _panels()
    panels["M06"] = panels["M06"][
        ~(
            panels["M06"].model_id.eq("D0-M")
            & panels["M06"].context_id.eq("c1")
            & panels["M06"].master_sample_id.eq("m2")
        )
    ].reset_index(drop=True)
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_primary_common_rejects_partial_unit_overlap():
    panels = _panels()
    mask = panels["M06"].model_id.eq("C-SELECTED") & panels["M06"].context_id.eq("c1")
    panels["M06"].loc[mask, "unit_id"] = "x-" + panels["M06"].loc[mask, "unit_id"].astype(str)
    with pytest.raises(ValueError):
        build_metrics(panels)


def test_class_vocabulary_accepts_supported_containers() -> None:
    expected = build_metrics(_panels())
    assert len(expected) == 6
    assert all(isinstance(value, pd.DataFrame) for value in expected.values())
    for convert in (list, tuple, np.asarray):
        panels = _panels()
        for panel in panels.values():
            panel["class_vocabulary"] = panel["class_vocabulary"].map(
                lambda _value, _convert=convert: _convert(VOCAB)
            )
        actual = build_metrics(panels)
        assert set(actual) == set(expected)
        for key, frame in actual.items():
            assert isinstance(frame, pd.DataFrame)
            assert frame.equals(expected[key])


def test_class_vocabulary_rejects_malformed_containers() -> None:
    for malformed in (np.zeros((2, 2)), [1, 2, 3]):
        panels = _panels()
        for panel in panels.values():
            panel["class_vocabulary"] = panel["class_vocabulary"].map(
                lambda _value, _malformed=malformed: _malformed
            )
        with pytest.raises(ValueError):
            build_metrics(panels)
