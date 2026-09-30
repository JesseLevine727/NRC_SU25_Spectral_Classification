"""Checks for the tightened P06/P11 prediction panel validators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p05_comparison as _p05c
from atlas_sers.evaluation import p06p11_predictions as predictions
from atlas_sers.evaluation.p06p11_predictions import (
    FROZEN_ENDPOINT_COLUMNS,
    PredictionAuditError,
    _require_panel,
    audit_point_estimates,
)

MODEL = sorted(_p05c.ALL_MODELS)[0]
CONTEXT = "ctx-1"
VOCABULARY = ["a", "b", "c"]
LABELS = ["a", "a", "b", "b"]
MASTERS = ["m-a", "m-a", "m-b", "m-b"]


def _unit_rows(correctness):
    rows = []
    for index, (label, master, correct) in enumerate(
        zip(LABELS, MASTERS, correctness, strict=True)
    ):
        rows.append(
            {
                "context_id": CONTEXT,
                "domain": "d1",
                "station": "s1",
                "instrument": "i1",
                "master_sample_id": master,
                "unit_id": f"u{index + 1}",
                "true_label": label,
                "model_id": MODEL,
                "class_vocabulary": VOCABULARY,
                "correct": correct,
            }
        )
    return rows


M01_CORRECT = [True, False, True, True]
M06_CORRECT = [True, True, False, True]


def _panel():
    return {
        "M01": pd.DataFrame(_unit_rows(M01_CORRECT)),
        "M06": pd.DataFrame(_unit_rows(M06_CORRECT)),
        "coverage": pd.DataFrame([{"model_id": MODEL, "context_id": CONTEXT, "complete": True}]),
    }


def _put(panel, aggregation, row, column, value):
    frame = panel[aggregation]
    frame[column] = frame[column].astype(object)
    frame.loc[row, column] = value


def _balanced_accuracy(correctness):
    by_class = {}
    for label, correct in zip(LABELS, correctness, strict=True):
        by_class.setdefault(label, []).append(1.0 if correct else 0.0)
    return float(np.mean([np.mean(values) for values in by_class.values()]))


def _frozen(balanced_accuracy=None):
    records = []
    for aggregation, correctness in (("M01", M01_CORRECT), ("M06", M06_CORRECT)):
        record = {column: None for column in FROZEN_ENDPOINT_COLUMNS}
        record.update(
            {
                "model_id": MODEL,
                "context_id": CONTEXT,
                "aggregation_id": aggregation,
                "balanced_accuracy": _balanced_accuracy(correctness)
                if balanced_accuracy is None
                else balanced_accuracy,
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=list(FROZEN_ENDPOINT_COLUMNS))


def test_consistent_panel_passes_validation():
    _require_panel(_panel())


def test_duplicate_columns_raise_fixed_code():
    panel = _panel()
    frame = panel["M01"]
    panel["M01"] = pd.concat([frame, frame[["unit_id"]]], axis=1)
    with pytest.raises(PredictionAuditError) as error:
        _require_panel(panel)
    assert str(error.value) == "panel_duplicate_columns"


@pytest.mark.parametrize("bad", [None, 7, " x", "", True])
def test_identity_fields_require_strict_strings(bad):
    panel = _panel()
    _put(panel, "M01", 0, "domain", bad)
    with pytest.raises(PredictionAuditError):
        _require_panel(panel)


def test_unknown_model_rejected():
    panel = _panel()
    _put(panel, "M01", 0, "model_id", "unknown-model")
    with pytest.raises(PredictionAuditError, match="unit_unknown_model"):
        _require_panel(panel)


def test_duplicate_units_rejected():
    panel = _panel()
    panel["M01"] = pd.concat([panel["M01"], panel["M01"].iloc[[0]]], ignore_index=True)
    with pytest.raises(PredictionAuditError, match="panel_duplicate_unit"):
        _require_panel(panel)


@pytest.mark.parametrize("bad", [2, -1, "yes", np.nan])
def test_correctness_must_be_binary(bad):
    panel = _panel()
    _put(panel, "M01", 0, "correct", bad)
    with pytest.raises(PredictionAuditError, match="unit_correctness_invalid"):
        _require_panel(panel)


def test_unit_identity_conflict_rejected():
    panel = _panel()
    _put(panel, "M01", 0, "station", "s2")
    with pytest.raises(PredictionAuditError, match="unit_identity_mismatch"):
        _require_panel(panel)


def test_master_identity_conflict_rejected():
    panel = _panel()
    frame = panel["M01"]
    frame.loc[frame["unit_id"] == "u2", "station"] = "s2"
    with pytest.raises(PredictionAuditError, match="unit_identity_mismatch"):
        _require_panel(panel)


def test_coverage_flag_must_be_bool():
    panel = _panel()
    coverage = panel["coverage"].copy()
    coverage["complete"] = "yes"
    panel["coverage"] = coverage
    with pytest.raises(PredictionAuditError, match="coverage_flag_not_bool"):
        _require_panel(panel)


def test_coverage_duplicate_key_rejected():
    panel = _panel()
    panel["coverage"] = pd.concat([panel["coverage"], panel["coverage"]], ignore_index=True)
    with pytest.raises(PredictionAuditError, match="coverage_duplicate_key"):
        _require_panel(panel)


def test_point_audit_accepts_matching_frozen_table():
    audit = audit_point_estimates(_panel(), _frozen())
    assert len(audit) == 2
    assert np.allclose(audit["absolute_error"], 0.0)


@pytest.mark.parametrize("bad", [True, "0.5", 1.5, -0.1, np.nan])
def test_point_audit_rejects_bad_frozen_balanced_accuracy(bad):
    frozen = _frozen()
    frozen["balanced_accuracy"] = frozen["balanced_accuracy"].astype(object)
    frozen.loc[0, "balanced_accuracy"] = bad
    with pytest.raises(PredictionAuditError, match="frozen_balanced_accuracy_invalid"):
        audit_point_estimates(_panel(), frozen)


def test_deleting_group_while_coverage_complete_is_rejected():
    panel = _panel()
    for aggregation in ("M01", "M06"):
        panel[aggregation] = panel[aggregation].iloc[0:0]
    with pytest.raises(PredictionAuditError, match="coverage_unit_disagreement"):
        audit_point_estimates(panel, _frozen())


def test_shared_fixture_panel_is_consistent():
    from tests.test_p05_comparison import _fixture

    fixture = _fixture()
    panel = predictions.prepare_panel(
        p05_ensemble=fixture["p05_ensemble"],
        p04_ensemble=fixture["p04_ensemble"],
        p03_predictions=fixture["p03_predictions"],
        contexts=fixture["contexts"],
    )
    _require_panel(panel)
