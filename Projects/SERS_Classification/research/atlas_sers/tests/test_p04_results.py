from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.p04_results import (
    endpoint_coverage,
    endpoint_metrics,
    ensemble_seed_predictions,
    normalize_p04_predictions,
    validate_prediction_schema,
)
from atlas_sers.governance.canonical import sha256_value


def _seed_predictions() -> pd.DataFrame:
    rows = []
    labels = ["a", "b", "c"]
    probabilities = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    for seed in (1, 2, 3):
        for index, (label, probability) in enumerate(zip(labels, probabilities, strict=True)):
            rows.append(
                {
                    "context_id": "ctx",
                    "experiment_id": "EXP-N00-T3",
                    "domain": "station:instrument",
                    "held_instrument": "instrument",
                    "outer_repeat": 0,
                    "outer_fold": 0,
                    "observation_uid": f"uid-{index}",
                    "master_sample_id": f"master-{index}",
                    "instrument": "instrument",
                    "station": "station",
                    "true_label": label,
                    "class_vocabulary": json.dumps(labels),
                    "candidate_id": "candidate",
                    "seed": seed,
                    "probability_0": probability[0],
                    "probability_1": probability[1],
                    "probability_2": probability[2],
                }
            )
    return pd.DataFrame(rows)


def test_ensemble_precedes_spectrum_and_master_metrics() -> None:
    ensemble = ensemble_seed_predictions(_seed_predictions())
    spectrum, master = endpoint_metrics(ensemble)

    assert len(ensemble) == 3
    assert spectrum.iloc[0].balanced_accuracy == 1.0
    assert master.iloc[0].balanced_accuracy == 1.0
    assert spectrum.iloc[0].aggregation_id == "M01"
    assert master.iloc[0].aggregation_id == "M06"


def test_ensemble_rejects_repeated_seed_rows() -> None:
    frame = _seed_predictions()
    with pytest.raises(ValueError, match="duplicate observation/seed"):
        ensemble_seed_predictions(pd.concat([frame, frame.iloc[:1]], ignore_index=True))


@pytest.mark.parametrize("invalid", [float("nan"), -0.1, 1.1, 0.5])
def test_ensemble_rejects_invalid_probabilities(invalid: float) -> None:
    frame = _seed_predictions()
    frame.loc[0, "probability_0"] = invalid
    with pytest.raises(ValueError, match="finite and normalized"):
        ensemble_seed_predictions(frame)


def test_endpoint_coverage_reconciles_exact_uid_set_and_seed_count() -> None:
    ensemble = ensemble_seed_predictions(_seed_predictions())
    expected_uids = sorted(ensemble.observation_uid.tolist())
    expected = pd.DataFrame(
        [
            {
                "context_id": "ctx",
                "experiment_id": "EXP-N00-T3",
                "domain": "station:instrument",
                "station": "station",
                "held_instrument": "instrument",
                "outer_repeat": 0,
                "outer_fold": 0,
                "expected_test_rows": 3,
                "expected_test_uid_sha256": sha256_value(expected_uids),
                "expected_seed_count": 3,
            }
        ]
    )
    status = pd.DataFrame(
        [
            {"context_id": "ctx", "stage": "final_selected_refit", "status": "complete"}
            for _ in range(3)
        ]
    )
    coverage = endpoint_coverage(expected, status, ensemble)

    assert coverage.iloc[0].status == "complete"
    assert coverage.iloc[0].observed_test_uid_sha256 == sha256_value(expected_uids)


@pytest.mark.parametrize("ensemble", [False, True])
def test_normalization_derives_platform_family_missing_from_p01_manifest(ensemble: bool) -> None:
    predictions = _seed_predictions()
    predictions["station"] = "cwa"
    predictions["instrument"] = "Mira-1"
    predictions["predicted_label"] = predictions.true_label
    for index in range(3):
        predictions[f"logit_{index}"] = 0.0
    if ensemble:
        predictions = ensemble_seed_predictions(predictions)
    contexts = pd.DataFrame(
        [
            {
                "context_id": "ctx",
                "task_id": "T3",
                "selection_mode": "source_pseudo_domain",
                "outer_fit_uid_sha256": "0" * 64,
            }
        ]
    )
    manifest = predictions[["observation_uid", "instrument"]].drop_duplicates()
    manifest["sensor_family"] = "test-substrate"
    status = pd.DataFrame(
        [
            {
                "context_id": "ctx",
                "seed": seed,
                "hyperparameter_sha256": "0" * 64,
                "best_epoch": 30,
                "status": "complete",
                "stage": "final_selected_refit",
            }
            for seed in (1, 2, 3)
        ]
    )
    identity = {
        key: "0" * 64
        for key in (
            "code_sha256",
            "config_sha256",
            "input_sha256",
            "preprocessing_policy_sha256",
        )
    }
    identity["run_id"] = "P04-test"
    normalized = normalize_p04_predictions(
        predictions,
        contexts=contexts,
        manifest=manifest,
        fit_status=status,
        identity=identity,
        ensemble=ensemble,
    )
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "plan/contracts/result_schema.json").read_text()
    )
    audit = validate_prediction_schema(normalized, schema)
    assert audit["status"] == "pass", audit
    assert normalized.instrument_family.eq("Mira").all()
    assert "instrument_family" not in manifest


def test_cli_serializes_numpy_validation_flags(monkeypatch, capsys, tmp_path) -> None:
    from atlas_sers.governance import p04, p04_cli

    monkeypatch.setattr(p04_cli, "artifact_root", lambda: tmp_path)
    monkeypatch.setattr(
        p04,
        "validate_latest_p04_plan",
        lambda _: {
            "status": "pass",
            "checks": {"schema_valid": np.bool_(True)},
        },
    )
    assert p04_cli.main(["validate-plan"]) == 0
    assert json.loads(capsys.readouterr().out)["checks"]["schema_valid"] is True
