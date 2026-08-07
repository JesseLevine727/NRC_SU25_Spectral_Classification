from __future__ import annotations

from pathlib import Path

from atlas_sers.governance.registries import load_governance, validate_governance

PROJECT = Path(__file__).resolve().parents[1]


def test_full_governance_registry_cross_reference_passes() -> None:
    bundle = load_governance(PROJECT / "plan")
    report = validate_governance(bundle)
    assert report["status"] == "pass", report["errors"]
    assert report["counts"]["research_question_registry.csv"] == 8
    assert report["counts"]["preprocessing_policy_registry.csv"] == 6
    assert report["counts"]["experiment_registry.csv"] == 43
    assert report["counts"]["model_registry.csv"] == 39
    assert report["counts"]["artifact_registry.csv"] == 39
    assert report["counts"]["deviations.csv"] == 1


def test_every_experiment_has_registered_model_and_artifacts() -> None:
    bundle = load_governance(PROJECT / "plan")
    models = {row["model_id"] for row in bundle.rows("model_registry.csv")}
    artifacts = {row["artifact_id"] for row in bundle.rows("artifact_registry.csv")}
    for experiment in bundle.rows("experiment_registry.csv"):
        assert experiment["model_id"] in models
        assert set(experiment["artifact_ids"].split("|")) <= artifacts
