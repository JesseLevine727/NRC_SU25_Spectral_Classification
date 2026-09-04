import json
from pathlib import Path

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p13_results import (
    BOOTSTRAP_RESAMPLES,
    _interval,
    build_master_view_predictions,
)

PROJECT = Path(__file__).resolve().parents[1]
PUBLIC_TABLES = PROJECT / "results/p13_portability/tables"


def _prediction(
    *, repeat: int, observation: str, master: str, probabilities: list[float]
) -> dict[str, object]:
    return {
        "domain_id": "D1",
        "station": "s",
        "substrate_family": "substrate",
        "held_instrument": "held",
        "support_tier": "confirmatory",
        "policy_id": "PP-U-MIN",
        "procedure_id": "C-SELECTED",
        "candidate_id": f"candidate-{repeat}",
        "model_id": "model",
        "prediction_role": "held_test",
        "observation_uid": observation,
        "master_sample_id": master,
        "instrument": "held",
        "true_label": "a",
        "class_vocabulary": '["a","b","c"]',
        "probabilities": json.dumps(probabilities),
        "outer_repeat": repeat,
    }


def test_master_view_prediction_averages_repeats_before_technical_repeats() -> None:
    rows = [
        _prediction(repeat=1, observation="o1", master="m1", probabilities=[0.8, 0.1, 0.1]),
        _prediction(repeat=2, observation="o1", master="m1", probabilities=[0.6, 0.2, 0.2]),
        _prediction(repeat=1, observation="o2", master="m1", probabilities=[0.4, 0.3, 0.3]),
        _prediction(repeat=2, observation="o2", master="m1", probabilities=[0.2, 0.4, 0.4]),
    ]
    views = build_master_view_predictions(pd.DataFrame(rows))
    assert len(views) == 1
    assert np.allclose(json.loads(views.probabilities.iloc[0]), [0.5, 0.25, 0.25])
    assert views.predicted_label.iloc[0] == "a"
    assert views.technical_repeat_count.iloc[0] == 2
    assert views.outer_repeat_predictions_min.iloc[0] == 2
    assert json.loads(views.candidate_ids_json.iloc[0]) == ["candidate-1", "candidate-2"]


def test_bca_interval_is_deterministic_and_retains_10000_resamples() -> None:
    rng = np.random.default_rng(123)
    bootstrap = rng.normal(0.5, 0.05, size=BOOTSTRAP_RESAMPLES)
    jackknife = np.asarray([0.48, 0.49, 0.51, 0.52])
    first = _interval(0.5, bootstrap, jackknife)
    second = _interval(0.5, bootstrap, jackknife)
    assert first == second
    assert first["bootstrap_resamples"] == 10_000
    assert first["interval_method"] in {"BCa", "percentile"}
    assert first["lower_95"] < first["estimate"] < first["upper_95"]


def test_public_p13_tables_retain_locked_denominators_and_primary_states() -> None:
    expected_rows = {
        "domain_metrics.csv": 336,
        "interval_table.csv": 570,
        "domain_claims.csv": 34,
        "class_cell_claims.csv": 102,
        "substrate_claims.csv": 4,
        "preprocessing_sensitivity.csv": 181,
        "procedure_comparison.csv": 7,
        "crossover_effects.csv": 238,
        "field_log_results.csv": 35,
        "failure_table.csv": 210,
    }
    for name, expected in expected_rows.items():
        assert len(pd.read_csv(PUBLIC_TABLES / name, low_memory=False)) == expected

    metrics = pd.read_csv(PUBLIC_TABLES / "domain_metrics.csv")
    primary = metrics[
        metrics.policy_id.eq("PP-U-MIN")
        & metrics.procedure_id.eq("C-SELECTED")
        & metrics.support_tier.eq("confirmatory")
    ]
    assert primary.bounded_state.value_counts().to_dict() == {
        "unavailable_terminal_failure": 6,
        "inconclusive": 5,
        "inferior_portability": 2,
    }

    rbf = metrics[
        metrics.policy_id.eq("PP-U-MIN")
        & metrics.procedure_id.eq("C-RBF-SVM")
        & metrics.support_tier.eq("confirmatory")
    ]
    assert rbf.bounded_state.value_counts().to_dict() == {
        "inconclusive": 8,
        "inferior_portability": 3,
        "supports_portability": 2,
    }


def test_public_p13_claims_and_sensitivity_are_bounded() -> None:
    claims = pd.read_csv(PUBLIC_TABLES / "domain_claims.csv")
    confirmatory = claims[claims.support_tier.eq("confirmatory")]
    assert len(confirmatory) == 13
    assert not confirmatory.completion_state.eq("supports_portability").any()

    substrates = pd.read_csv(PUBLIC_TABLES / "substrate_claims.csv").set_index(
        "substrate_family"
    )
    assert (
        substrates.loc["pSERS_Metrohm_silver", "intersection_union_state"]
        == "inferior_portability"
    )
    assert (
        substrates.loc["H_SERS_H_Kit", "intersection_union_state"]
        == "unavailable_terminal_failure"
    )

    comparison = pd.read_csv(PUBLIC_TABLES / "procedure_comparison.csv").set_index(
        "procedure_id"
    )
    assert bool(comparison.loc["C-RBF-SVM", "positive_comparison_claim_allowed"])
    assert not bool(
        comparison.loc["C-LOGREG-EN", "positive_comparison_claim_allowed"]
    )

    crossover = pd.read_csv(PUBLIC_TABLES / "crossover_effects.csv")
    assert len(crossover[crossover.predictive_status.eq("complete")]) == 13
    assert crossover.groupby("procedure_id").crossover_block_id.nunique().eq(34).all()
