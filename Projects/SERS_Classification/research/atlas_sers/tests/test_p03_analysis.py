from __future__ import annotations

import json

import pandas as pd

from atlas_sers.evaluation.p03_analysis import (
    build_confusion_summary,
    build_cost_summary,
    build_endpoint_coverage,
    build_reliability_summary,
    build_selection_diagnostics,
    build_spectrum_master_comparison,
    build_t1_t3_comparison,
)


def _selection_evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    selections: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for repeat in range(1, 6):
        selected = "candidate-a" if repeat < 5 else "candidate-b"
        outer = f"outer-{repeat}"
        selections.append(
            {
                "experiment_id": "EXP-C09-T3",
                "task_id": "T3-ZS",
                "outer_run_id": outer,
                "station": "cwa",
                "domain": "cwa:unit-1",
                "held_instrument": "unit-1",
                "outer_repeat": repeat,
                "outer_fold": 0,
                "selection_kind": "standard",
                "status": "complete",
                "selected_model_id": "C-PCA-LDA",
                "selected_candidate_id": selected,
                "selection_state_sha256": f"{repeat:064x}",
            }
        )
        for order, candidate, mean in (
            (0, "candidate-a", 0.8),
            (1, "candidate-b", 0.7),
        ):
            traces.append(
                {
                    **selections[-1],
                    "candidate_id": candidate,
                    "complete_support": True,
                    "selected": candidate == selected,
                    "mean_balanced_accuracy": mean if repeat < 5 else 1.5 - mean,
                    "worst_balanced_accuracy": mean - 0.1,
                    "mean_macro_f1": mean - 0.05,
                    "complexity_rank": order,
                    "declared_candidate_order": order,
                }
            )
    return pd.DataFrame(selections), pd.DataFrame(traces)


def test_selection_diagnostics_preserve_frequency_repeat_agreement_and_margin() -> None:
    selections, traces = _selection_evidence()
    frequency, stability, margins = build_selection_diagnostics(selections, traces)
    selected_a = frequency[
        frequency.selection_outcome_candidate.eq("candidate-a")
    ].iloc[0]
    assert selected_a.selection_count == 4
    assert selected_a.selection_denominator == 5
    assert selected_a.selection_fraction == 0.8
    assert stability.modal_fraction.tolist() == [0.8]
    assert stability.normalized_entropy.between(0, 1).all()
    assert len(margins) == 5
    assert margins.margin_status.eq("complete").all()


def _metric_row(
    *,
    experiment: str,
    procedure: str,
    domain: str,
    repeat: int,
    level: str,
    balanced_accuracy: float,
    confusion: list[list[int]] | None = None,
) -> dict[str, object]:
    matrix = confusion or [[3, 1], [1, 3]]
    return {
        "experiment_id": experiment,
        "task_id": "T1-CWA" if experiment.endswith("T1") else "T3-ZS",
        "domain": domain,
        "station": "cwa",
        "procedure_id": procedure,
        "outer_repeat": repeat,
        "aggregation_level": level,
        "endpoint_status": "complete",
        "expected_observations": 8,
        "expected_masters": 4,
        "balanced_accuracy": balanced_accuracy,
        "class_vocabulary_json": json.dumps(["a", "b"], separators=(",", ":")),
        "confusion_matrix_json": json.dumps(matrix, separators=(",", ":")),
    }


def _metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for repeat in (1, 2):
        rows.extend(
            [
                _metric_row(
                    experiment="EXP-C03-T1",
                    procedure="C-PCA-LDA",
                    domain="cwa:within",
                    repeat=repeat,
                    level=level,
                    balanced_accuracy=0.9,
                )
                for level in ("spectrum", "instrument_balanced_master")
            ]
        )
        for domain, score in (("cwa:unit-1", 0.6), ("cwa:unit-2", 0.7)):
            rows.append(
                _metric_row(
                    experiment="EXP-C09-T3",
                    procedure="C-SELECTED",
                    domain=domain,
                    repeat=repeat,
                    level="spectrum",
                    balanced_accuracy=score,
                )
            )
            rows.append(
                _metric_row(
                    experiment="EXP-C09-T3",
                    procedure="C-SELECTED",
                    domain=domain,
                    repeat=repeat,
                    level="instrument_balanced_master",
                    balanced_accuracy=score + 0.05,
                )
            )
            for level in ("spectrum", "instrument_balanced_master"):
                rows.append(
                    _metric_row(
                        experiment="EXP-C10-T3",
                        procedure="C-PCA-LDA",
                        domain=domain,
                        repeat=repeat,
                        level=level,
                        balanced_accuracy=score,
                    )
                )
    return pd.DataFrame(rows)


def test_metric_diagnostics_pair_levels_regimes_and_confusion() -> None:
    metrics = _metrics()
    coverage = build_endpoint_coverage(metrics)
    assert coverage.completion_fraction.eq(1.0).all()
    levels = build_spectrum_master_comparison(metrics)
    assert len(levels) == 4
    assert levels.master_minus_spectrum.round(8).eq(0.05).all()
    regimes = build_t1_t3_comparison(metrics)
    assert len(regimes) == 4
    assert regimes.zero_shot_domain_count.eq(2).all()
    assert regimes.zero_shot_minus_within_station.lt(0).all()
    confusion = build_confusion_summary(metrics)
    assert len(confusion) == 8
    assert confusion.repeat_count.eq(2).all()
    row_sums = confusion.groupby(
        ["station", "aggregation_level", "true_label"]
    ).mean_row_fraction.sum()
    assert row_sums.round(8).eq(1.0).all()


def _prediction_rows(probabilities_as_json: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for repeat in (1, 2):
        for label, probabilities in (("a", [0.8, 0.2]), ("b", [0.3, 0.7])):
            rows.append(
                {
                    "experiment_id": "EXP-C09-T3",
                    "station": "cwa",
                    "outer_repeat": repeat,
                    "true_label": label,
                    "predicted_label": label,
                    "probabilities": (
                        json.dumps(probabilities, separators=(",", ":"))
                        if probabilities_as_json
                        else probabilities
                    ),
                    "probability_status": "cross_fitted_temperature",
                }
            )
    return pd.DataFrame(rows)


def test_reliability_bins_accept_row_json_and_master_arrays() -> None:
    reliability = build_reliability_summary(
        _prediction_rows(True), _prediction_rows(False)
    )
    assert set(reliability.aggregation_level) == {
        "spectrum",
        "instrument_balanced_master",
    }
    assert reliability.repeat_count.eq(2).all()
    assert reliability.mean_empirical_accuracy.eq(1.0).all()


def test_cost_summary_uses_actual_final_prediction_denominator() -> None:
    manifest = pd.DataFrame(
        [
            {
                "fit_id": "selection",
                "experiment_id": "EXP-C03-T1",
                "task_id": "T1-CWA",
                "model_id": "C-PCA-LDA",
                "stage": "inner_selection",
                "accounting": "new_fit",
                "validation_rows": 10,
                "test_rows": 0,
            },
            {
                "fit_id": "final",
                "experiment_id": "EXP-C03-T1",
                "task_id": "T1-CWA",
                "model_id": "C-PCA-LDA",
                "stage": "final_selected_refit",
                "accounting": "new_fit",
                "validation_rows": 0,
                "test_rows": 5,
            },
        ]
    )
    ledger = pd.DataFrame(
        [
            {
                "fit_id": "selection",
                "experiment_id": "EXP-C03-T1",
                "task_id": "T1-CWA",
                "status": "complete",
                "model_id": "C-PCA-LDA",
                "stage": "inner_selection",
                "elapsed_seconds": 2.0,
                "inference_seconds": 0.01,
                "prediction_rows": None,
                "serialized_model_bytes": 100,
            },
            {
                "fit_id": "final",
                "experiment_id": None,
                "task_id": None,
                "status": "complete",
                "model_id": "C-PCA-LDA",
                "stage": None,
                "elapsed_seconds": 3.0,
                "inference_seconds": 0.02,
                "prediction_rows": 5,
                "serialized_model_bytes": 120,
            },
        ]
    )
    cost = build_cost_summary(ledger, manifest)
    selection = cost[cost.stage_group.eq("selection")].iloc[0]
    final = cost[cost.stage_group.eq("final_refit_and_prediction")].iloc[0]
    assert selection.median_milliseconds_per_prediction == 1.0
    assert final.median_milliseconds_per_prediction == 4.0
    assert cost.terminal_failure_records.eq(0).all()
