"""CPU synthetic tests for the pure P06/P11 prediction adapter.

Primary fixtures come from the attached ``tests.test_p05_comparison`` module:
``_fixture`` builds the frozen inputs and ``_run`` produces the frozen endpoint
metrics used as the authenticated audit reference. A few small hand-built
panels exercise pairing and the M06 average semantics without any real data or
hardcoded scientific outcomes.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p05_comparison as p05c
from atlas_sers.evaluation import p06p11_predictions as p06
from atlas_sers.governance.canonical import sha256_value
from tests.test_p05_comparison import _fixture, _run

HELD_CONTEXT = "ctx-held"
HELD_EXPERIMENT = "EXP-N00-T3"
DOMAIN = "DOM"


def _prepare(fixture):
    return p06.prepare_panel(
        p05_ensemble=fixture["p05_ensemble"],
        p04_ensemble=fixture["p04_ensemble"],
        p03_predictions=fixture["p03_predictions"],
        contexts=fixture["contexts"],
    )


def _frozen(fixture):
    return _run(fixture)["endpoint_metrics"]


def test_reproduction_all_models_both_endpoints_and_audit():
    fixture = _fixture()
    frozen = _frozen(fixture)
    panel = _prepare(fixture)
    coverage = panel["coverage"]
    assert set(coverage["model_id"]) == set(p05c.ALL_MODELS)
    assert coverage["complete"].all()
    m01 = panel["M01"]
    m06 = panel["M06"]
    assert set(m01["model_id"]) == set(p05c.ALL_MODELS)
    assert set(m06["model_id"]) == set(p05c.ALL_MODELS)
    assert len(m01) == len(p05c.ALL_MODELS) * 3
    assert len(m06) == len(p05c.ALL_MODELS) * 2

    selected = m01[m01["model_id"].eq("C-SELECTED")]
    assert len(selected) == 3
    assert set(selected["model_id"]) == {"C-SELECTED"}
    assert set(selected["context_id"]) == {HELD_CONTEXT}

    audit = p06.audit_point_estimates(panel, frozen)
    assert set(audit["model_id"]) == set(p05c.ALL_MODELS)
    assert set(audit["aggregation_id"]) == {"M01", "M06"}
    assert len(audit) == len(p05c.ALL_MODELS) * 2
    assert audit["absolute_error"].abs().max() == pytest.approx(0.0, abs=1e-15)


def test_panel_unit_schema_and_sorting():
    panel = _prepare(_fixture())
    for aggregation in p06.AGGREGATIONS:
        frame = panel[aggregation]
        assert list(frame.columns) == list(p06.UNIT_COLUMNS)
        ordered = frame.sort_values(
            ["model_id", "context_id", "unit_id"], kind="stable"
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(frame, ordered)
    assert set(panel["M01"]["unit_id"]) == {"h1", "h2", "h3"}
    assert all(unit.startswith(p06.MASTER_UNIT_PREFIX) for unit in panel["M06"]["unit_id"])


def test_no_mutation_and_row_permutation_invariance():
    fixture = _fixture()
    snapshot = {
        key: fixture[key].copy(deep=True)
        for key in ("p05_ensemble", "p04_ensemble", "p03_predictions", "contexts")
    }
    baseline = _prepare(fixture)
    for key, before in snapshot.items():
        pd.testing.assert_frame_equal(before, fixture[key])

    permuted = _fixture()
    for key in ("p05_ensemble", "p04_ensemble", "p03_predictions", "contexts"):
        permuted[key] = permuted[key].sample(frac=1.0, random_state=7).reset_index(drop=True)
    shuffled = _prepare(permuted)
    pd.testing.assert_frame_equal(baseline["M01"], shuffled["M01"])
    pd.testing.assert_frame_equal(baseline["M06"], shuffled["M06"])


def test_inherited_probability_class_and_truth_guards_reject():
    fixture = _fixture()
    frame = fixture["p05_ensemble"].copy()
    held_index = frame.index[frame["experiment_id"].eq(p05c.P05_EXPERIMENT)][0]
    frame.loc[held_index, ["probability_0", "probability_1", "probability_2"]] = [0.5, 0.5, 0.5]
    broken = dict(fixture)
    broken["p05_ensemble"] = frame
    with pytest.raises(p05c.P05ComparisonError):
        _prepare(broken)

    fixture = _fixture()
    frame = fixture["p05_ensemble"].copy()
    held_index = frame.index[frame["experiment_id"].eq(p05c.P05_EXPERIMENT)][0]
    frame.loc[held_index, "class_vocabulary"] = json.dumps(["B", "A", "C"])
    broken = dict(fixture)
    broken["p05_ensemble"] = frame
    with pytest.raises(p05c.P05ComparisonError):
        _prepare(broken)

    fixture = _fixture()
    contexts = fixture["contexts"].copy()
    row = contexts["context_id"].eq(HELD_CONTEXT)
    contexts.loc[row, "outer_test_uid_sha256"] = "0" * 64
    broken = dict(fixture)
    broken["contexts"] = contexts
    with pytest.raises(p05c.P05ComparisonError):
        _prepare(broken)


def test_partial_selected_reference_excluded_whole():
    fixture = _fixture()
    classical = fixture["p03_predictions"]
    selected = classical[classical["model_id"].eq("C-SELECTED")]
    fixture["p03_predictions"] = classical.drop(index=selected.index[0]).reset_index(drop=True)
    panel = _prepare(fixture)
    coverage = panel["coverage"]
    row = coverage[coverage["model_id"].eq("C-SELECTED")].iloc[0]
    assert not bool(row["complete"])
    assert row["reason_code"] != "complete"
    assert not panel["M01"]["model_id"].eq("C-SELECTED").any()
    assert coverage[coverage["model_id"].eq("D0-M")]["complete"].all()
    with pytest.raises(p06.PredictionAuditError, match="no_common_complete_contexts"):
        p06.pair_units(
            panel,
            model_id="D0-M",
            reference_model_id="C-SELECTED",
            aggregation_id="M01",
        )


def test_prepare_panel_rejects_duplicate_input_columns():
    fixture = _fixture()
    frame = pd.concat([fixture["p05_ensemble"], fixture["p05_ensemble"][["model_id"]]], axis=1)
    with pytest.raises(p06.PredictionAuditError, match="panel_duplicate_columns"):
        p06.prepare_panel(
            p05_ensemble=frame,
            p04_ensemble=fixture["p04_ensemble"],
            p03_predictions=fixture["p03_predictions"],
            contexts=fixture["contexts"],
        )


def test_pair_units_exact_correctness_from_fixture():
    panel = _prepare(_fixture())
    paired = p06.pair_units(
        panel, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M01"
    )
    assert list(paired.columns) == list(p06.PAIR_COLUMNS)
    assert len(paired) == 3
    assert list(paired["unit_id"]) == ["h1", "h2", "h3"]
    assert set(paired["context_id"]) == {HELD_CONTEXT}
    assert set(paired["domain"]) == {DOMAIN}
    assert paired["correct_model"].all()
    assert paired["correct_reference"].all()


def test_pair_units_m06_master_units_from_fixture():
    panel = _prepare(_fixture())
    paired = p06.pair_units(
        panel, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M06"
    )
    assert len(paired) == 2
    assert all(unit.startswith(p06.MASTER_UNIT_PREFIX) for unit in paired["unit_id"])
    assert set(paired["master_sample_id"]) == {"m4", "m5"}


def _unit_row(model, context, unit, *, domain, instrument, master="mz", true="A", correct=True):
    return {
        "context_id": context,
        "domain": domain,
        "station": "STN1",
        "instrument": instrument,
        "master_sample_id": master,
        "unit_id": unit,
        "true_label": true,
        "model_id": model,
        "class_vocabulary": ("A", "B", "C"),
        "probability_0": 0.6,
        "probability_1": 0.3,
        "probability_2": 0.1,
        "predicted_label": true,
        "correct": correct,
    }


def _two_context_panel():
    records = []
    coverage_rows = []
    for context, domain, instrument, unit in (
        ("c1", "DOM1", "INST-1", "u1"),
        ("c2", "DOM2", "INST-2", "u2"),
    ):
        for model in ("D0-M", "C-SELECTED"):
            records.append(
                _unit_row(
                    model,
                    context,
                    unit,
                    domain=domain,
                    instrument=instrument,
                    correct=(model == "D0-M"),
                )
            )
            coverage_rows.append({"model_id": model, "context_id": context, "complete": True})
    frame = pd.DataFrame(records, columns=list(p06.UNIT_COLUMNS))
    coverage = pd.DataFrame(coverage_rows, columns=["model_id", "context_id", "complete"])
    return {"M01": frame, "M06": frame.copy(), "coverage": coverage}


def test_pair_units_no_cross_context_collapse():
    panel = _two_context_panel()
    paired = p06.pair_units(
        panel, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M01"
    )
    assert len(paired) == 2
    assert set(paired["context_id"]) == {"c1", "c2"}
    assert paired["correct_model"].all()
    assert not paired["correct_reference"].any()
    assert set(paired["unit_id"]) == {"u1", "u2"}
    domains = paired.set_index("context_id")["domain"].to_dict()
    assert domains == {"c1": "DOM1", "c2": "DOM2"}


def test_pair_units_rejects_unit_set_identity_and_correctness_tampering():
    panel = _two_context_panel()
    frame = panel["M01"]

    drop = frame[(frame["model_id"].eq("C-SELECTED")) & (frame["context_id"].eq("c2"))].index
    broken = {**panel, "M01": frame.drop(index=drop).reset_index(drop=True)}
    with pytest.raises(p06.PredictionAuditError, match="coverage_unit_disagreement"):
        p06.pair_units(
            broken, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M01"
        )

    frame = panel["M01"].copy()
    mask = frame["model_id"].eq("C-SELECTED") & frame["context_id"].eq("c1")
    frame.loc[mask, "instrument"] = "INST-ZZZ"
    broken = {**panel, "M01": frame}
    with pytest.raises(p06.PredictionAuditError, match="unit_identity_mismatch"):
        p06.pair_units(
            broken, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M01"
        )

    frame = panel["M01"].copy()
    frame["correct"] = frame["correct"].astype(object)
    frame.loc[frame.index[0], "correct"] = "yes"
    broken = {**panel, "M01": frame}
    with pytest.raises(p06.PredictionAuditError, match="unit_correctness_invalid"):
        p06.pair_units(
            broken, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M01"
        )

    frame = panel["M01"].copy()
    frame["correct"] = frame["correct"].astype(object)
    frame.loc[frame.index[0], "correct"] = 2
    broken = {**panel, "M01": frame}
    with pytest.raises(p06.PredictionAuditError, match="unit_correctness_invalid"):
        p06.pair_units(
            broken, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M01"
        )


def test_pair_units_rejects_unexpected_units_and_maltyped_coverage():
    panel = _two_context_panel()
    frame = panel["M01"].copy()
    frame.loc[frame.index[0], "model_id"] = "ZZZ"
    broken = {**panel, "M01": frame}
    with pytest.raises(p06.PredictionAuditError, match="unit_unknown_model"):
        p06.pair_units(
            broken, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M01"
        )

    panel = _two_context_panel()
    frame = panel["M01"].copy()
    frame.loc[frame.index[0], "context_id"] = "c-missing"
    broken = {**panel, "M01": frame}
    with pytest.raises(p06.PredictionAuditError, match="unexpected_unit_context"):
        p06.pair_units(
            broken, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M01"
        )

    panel = _two_context_panel()
    coverage = panel["coverage"].copy()
    coverage.loc[coverage["model_id"].eq("D0-M") & coverage["context_id"].eq("c2"), "complete"] = (
        False
    )
    broken = {**panel, "coverage": coverage}
    with pytest.raises(p06.PredictionAuditError, match="unit_on_incomplete_context"):
        p06.pair_units(
            broken, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M01"
        )

    panel = _two_context_panel()
    coverage = panel["coverage"].copy()
    coverage["complete"] = coverage["complete"].astype(object)
    coverage.loc[coverage.index[0], "complete"] = 1
    broken = {**panel, "coverage": coverage}
    with pytest.raises(p06.PredictionAuditError, match="coverage_flag_not_bool"):
        p06.pair_units(
            broken, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M01"
        )


def test_pair_units_rejects_unknown_pair_and_aggregation():
    panel = _prepare(_fixture())
    with pytest.raises(p06.PredictionAuditError, match="unknown_aggregation"):
        p06.pair_units(
            panel, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M99"
        )
    with pytest.raises(p06.PredictionAuditError, match="unknown_pair"):
        p06.pair_units(
            panel, model_id="C-SELECTED", reference_model_id="D0-M", aggregation_id="M01"
        )


def test_pair_units_rejects_duplicate_columns():
    panel = _prepare(_fixture())
    units = panel["M01"]
    duplicated = pd.concat([units, units[["unit_id"]]], axis=1)
    broken = {**panel, "M01": duplicated}
    with pytest.raises(p06.PredictionAuditError, match="panel_duplicate_columns"):
        p06.pair_units(
            broken, model_id="D0-M", reference_model_id="C-SELECTED", aggregation_id="M01"
        )


def test_audit_rejects_mismatch_extra_missing_and_nonfinite():
    fixture = _fixture()
    panel = _prepare(fixture)
    frozen = _frozen(fixture)
    audit = p06.audit_point_estimates(panel, frozen)
    assert len(audit) == len(p05c.ALL_MODELS) * 2

    tampered = frozen.copy()
    tampered.loc[tampered.index[0], "balanced_accuracy"] = 0.123
    with pytest.raises(p06.PredictionAuditError, match="frozen_point_mismatch"):
        p06.audit_point_estimates(panel, tampered)

    nonfinite = frozen.copy()
    nonfinite.loc[nonfinite.index[0], "balanced_accuracy"] = np.nan
    with pytest.raises(p06.PredictionAuditError, match="frozen_balanced_accuracy_invalid"):
        p06.audit_point_estimates(panel, nonfinite)

    duplicate = pd.concat([frozen, frozen.iloc[[0]]], ignore_index=True)
    with pytest.raises(p06.PredictionAuditError, match="frozen_duplicate_key"):
        p06.audit_point_estimates(panel, duplicate)

    extra = frozen.iloc[[0]].copy()
    extra["context_id"] = "ctx-elsewhere"
    with pytest.raises(p06.PredictionAuditError, match="frozen_extra_key"):
        p06.audit_point_estimates(panel, pd.concat([frozen, extra], ignore_index=True))

    missing = frozen.iloc[1:].reset_index(drop=True)
    with pytest.raises(p06.PredictionAuditError, match="frozen_missing_key"):
        p06.audit_point_estimates(panel, missing)


def _hand_p05(context_id, domain, instrument, uids, probabilities, masters, true_labels):
    classes = ("A", "B", "C")
    rows = []
    for model in p05c.P05_MODELS:
        for uid in uids:
            probability = probabilities[uid]
            rows.append(
                {
                    "context_id": context_id,
                    "experiment_id": p05c.P05_EXPERIMENT,
                    "source_context_experiment_id": p05c.SOURCE_HELD_EXPERIMENT,
                    "model_id": model,
                    "observation_uid": uid,
                    "master_sample_id": masters[uid],
                    "instrument": instrument,
                    "true_label": true_labels[uid],
                    "class_vocabulary": json.dumps(list(classes), separators=(",", ":")),
                    "probability_0": probability[0],
                    "probability_1": probability[1],
                    "probability_2": probability[2],
                }
            )
    return rows


def _hand_contexts(spec):
    rows = []
    for entry in spec:
        context_id, domain, station, instrument, uids = entry[:5]
        outer_repeat = entry[5] if len(entry) > 5 else "0"
        rows.append(
            {
                "context_id": context_id,
                "experiment_id": HELD_EXPERIMENT,
                "domain": domain,
                "station": station,
                "held_instrument": instrument,
                "outer_repeat": outer_repeat,
                "outer_fold": "0",
                "phase_gate": "held_evaluation",
                "outer_test_uid_sha256": sha256_value(sorted(uids)),
            }
        )
    return pd.DataFrame(rows)


def _hand_panel(records, contexts):
    return p06.prepare_panel(
        p05_ensemble=pd.DataFrame(records),
        p04_ensemble=None,
        p03_predictions=None,
        contexts=contexts,
    )


def test_m06_average_not_majority_vote_or_average_correctness():
    uids = ["h1", "h2", "h3"]
    probabilities = {
        "h1": (0.55, 0.30, 0.15),
        "h2": (0.55, 0.30, 0.15),
        "h3": (0.05, 0.90, 0.05),
    }
    masters = {uid: "m5" for uid in uids}
    true_labels = {uid: "B" for uid in uids}
    records = _hand_p05("ctx-hand", "DOM", "INST-H", uids, probabilities, masters, true_labels)
    contexts = _hand_contexts([("ctx-hand", "DOM", "STN1", "INST-H", uids)])
    panel = _hand_panel(records, contexts)

    m06 = panel["M06"]
    row = m06[m06["model_id"].eq("D0-M") & m06["context_id"].eq("ctx-hand")].iloc[0]
    assert row["master_sample_id"] == "m5"
    assert row["predicted_label"] == "B"
    assert bool(row["correct"]) is True
    assert row["probability_1"] > row["probability_0"]
    assert row["probability_0"] == pytest.approx(1.15 / 3.0)
    assert row["probability_1"] == pytest.approx(1.5 / 3.0)
    assert row["probability_2"] == pytest.approx(0.35 / 3.0)

    m01 = panel["M01"]
    cell = m01[m01["model_id"].eq("D0-M") & m01["context_id"].eq("ctx-hand")]
    assert list(cell["unit_id"]) == uids
    assert cell["correct"].tolist() == [False, False, True]
    assert float(cell["correct"].astype(float).mean()) == pytest.approx(1.0 / 3.0)


def test_m06_unit_id_stable_and_domain_scoped():
    probabilities = {"u1": (0.6, 0.3, 0.1), "u2": (0.6, 0.3, 0.1)}
    masters = {"u1": "mz", "u2": "mz"}
    true_labels = {"u1": "A", "u2": "A"}
    records = _hand_p05("c1", "DOM1", "INST-1", ["u1"], probabilities, masters, true_labels)
    records += _hand_p05("c2", "DOM2", "INST-2", ["u2"], probabilities, masters, true_labels)
    records += _hand_p05("c3", "DOM1", "INST-1", ["u1"], probabilities, masters, true_labels)
    contexts = _hand_contexts(
        [
            ("c1", "DOM1", "STN1", "INST-1", ["u1"]),
            ("c2", "DOM2", "STN1", "INST-2", ["u2"]),
            ("c3", "DOM1", "STN1", "INST-1", ["u1"], "1"),
        ]
    )
    first = _hand_panel(list(records), contexts)["M06"]
    second = _hand_panel(list(records), contexts)["M06"]
    pd.testing.assert_frame_equal(first, second)
    models = first[first["model_id"].eq("D0-M")].set_index("context_id")
    assert models.loc["c1", "unit_id"] == p06.MASTER_UNIT_PREFIX + sha256_value(["DOM1", "mz"])
    assert models.loc["c3", "unit_id"] == p06.MASTER_UNIT_PREFIX + sha256_value(["DOM1", "mz"])
    assert models.loc["c2", "unit_id"] == p06.MASTER_UNIT_PREFIX + sha256_value(["DOM2", "mz"])
    assert models.loc["c1", "unit_id"] == models.loc["c3", "unit_id"]
    assert models.loc["c1", "unit_id"] != models.loc["c2", "unit_id"]
    assert models.loc["c1", "instrument"] == "INST-1"
    assert models.loc["c2", "instrument"] == "INST-2"


def test_m06_vocabulary_tie_uses_first_sorted_class():
    records = _hand_p05(
        "ctx-tie",
        "DOM",
        "INST-H",
        ["t1"],
        {"t1": (0.4, 0.4, 0.2)},
        {"t1": "m1"},
        {"t1": "A"},
    )
    contexts = _hand_contexts([("ctx-tie", "DOM", "STN1", "INST-H", ["t1"])])
    panel = _hand_panel(records, contexts)
    row = panel["M06"][panel["M06"]["model_id"].eq("D0-M")].iloc[0]
    assert row["class_vocabulary"] == ("A", "B", "C")
    assert row["predicted_label"] == "A"
    assert bool(row["correct"]) is True
