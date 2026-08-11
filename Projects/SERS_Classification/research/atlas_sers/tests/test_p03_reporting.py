from __future__ import annotations

import pandas as pd

from atlas_sers.evaluation.p03_analysis import (
    build_control_summary,
    build_endpoint_coverage,
    build_selection_diagnostics,
)
from atlas_sers.evaluation.p03_reporting import (
    build_p04_comparator_freeze,
    build_selected_model_specs,
    render_p03_report,
    render_p04_handoff,
)
from atlas_sers.governance.canonical import sha256_value
from tests.test_p03_analysis import _metrics, _selection_evidence


def test_selected_model_specs_resolve_exact_candidate_parameters() -> None:
    selections, _ = _selection_evidence()
    registry = pd.DataFrame(
        [
            {
                "candidate_id": candidate,
                "model_id": "C-PCA-LDA",
                "parameters_json": parameters,
                "hyperparameter_sha256": hp_hash,
                "complexity_rank": order,
                "declared_candidate_order": order,
                "technical_seeds": "deterministic",
                "seed_count": 1,
            }
            for order, (candidate, parameters, hp_hash) in enumerate(
                [
                    ("candidate-a", '{"pca_components":2}', "a" * 64),
                    ("candidate-b", '{"pca_components":3}', "b" * 64),
                ]
            )
        ]
    )
    selections["selected_hyperparameter_sha256"] = selections.selected_candidate_id.map(
        {"candidate-a": "a" * 64, "candidate-b": "b" * 64}
    )
    specs = build_selected_model_specs(
        selections=selections,
        candidate_registry=registry,
        coral_candidate_registry=pd.DataFrame(columns=registry.columns),
        metadata_candidate_registry=pd.DataFrame(columns=registry.columns),
    )
    assert len(specs) == 5
    assert specs.parameters_json.notna().all()
    assert specs.selected_hyperparameter_sha256.eq(specs.apply(
        lambda row: "a" * 64 if row.selected_candidate_id == "candidate-a" else "b" * 64,
        axis=1,
    )).all()


def _comparator_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs: list[dict[str, object]] = []
    endpoints: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    index = 0
    for repeat in range(1, 6):
        for domain_index in range(13):
            domain = f"station:unit-{domain_index}"
            for fold in range(4):
                outer = f"outer-{index}"
                specs.append(
                    {
                        "experiment_id": "EXP-C09-T3",
                        "outer_run_id": outer,
                        "domain": domain,
                        "station": "cwa",
                        "outer_repeat": repeat,
                        "outer_fold": fold,
                        "status": "complete",
                        "selected_candidate_id": "candidate-a",
                        "selected_model_id": "C-PCA-LDA",
                        "selected_hyperparameter_sha256": "a" * 64,
                        "selection_state_sha256": f"{index:064x}",
                    }
                )
                endpoints.append(
                    {
                        "experiment_id": "EXP-C09-T3",
                        "outer_run_id": outer,
                        "domain": domain,
                        "station": "cwa",
                        "outer_repeat": repeat,
                        "outer_fold": fold,
                        "procedure_id": "C-SELECTED",
                        "expected_test_rows": 2,
                        "expected_test_masters": 2,
                        "expected_test_uid_sha256": sha256_value([outer]),
                    }
                )
                index += 1
            for level in ("spectrum", "instrument_balanced_master"):
                metrics.append(
                    {
                        "experiment_id": "EXP-C09-T3",
                        "domain": domain,
                        "station": "cwa",
                        "procedure_id": "C-SELECTED",
                        "outer_repeat": repeat,
                        "aggregation_level": level,
                        "endpoint_status": "complete",
                        "observation_uid_sha256": sha256_value([domain, repeat, level]),
                    }
                )
    return pd.DataFrame(specs), pd.DataFrame(endpoints), pd.DataFrame(metrics)


def test_p04_comparator_freeze_is_complete_deterministic_and_human_readable() -> None:
    specs, endpoints, metrics = _comparator_inputs()
    first = build_p04_comparator_freeze(
        execution_run_id="P03-test",
        protected_state_sha256="f" * 64,
        selected_model_specs=specs,
        expected_endpoints=endpoints,
        pooled_metrics=metrics,
    )
    second = build_p04_comparator_freeze(
        execution_run_id="P03-test",
        protected_state_sha256="f" * 64,
        selected_model_specs=specs,
        expected_endpoints=endpoints,
        pooled_metrics=metrics,
    )
    assert first == second
    assert first["outer_cell_count"] == 260
    assert first["pooled_domain_repeat_count"] == 65
    handoff = render_p04_handoff(first)
    assert "C-SELECTED" in handoff
    assert "Exact next action" in handoff
    assert first["selection_mapping_sha256"] in handoff


def test_private_report_contains_results_controls_cost_figures_and_limits() -> None:
    metrics = _metrics()
    selections, traces = _selection_evidence()
    frequency, stability, _ = build_selection_diagnostics(selections, traces)
    domain_summaries = pd.DataFrame(
        [
            {
                "experiment_id": "EXP-C09-T3",
                "procedure_id": "C-SELECTED",
                "aggregation_level": level,
                "outer_repeat": repeat,
                "mean_domain_balanced_accuracy": 0.65,
                "worst_domain_balanced_accuracy": 0.55,
                "summary_status": "complete",
            }
            for level in ("spectrum", "instrument_balanced_master")
            for repeat in (1, 2)
        ]
    )
    report = render_p03_report(
        execution_run_id="P03-test",
        plan_run_id="P03PLAN-test",
        protected_state_sha256="f" * 64,
        fit_manifest_rows=10,
        terminal_fit_ledger=pd.DataFrame(
            [{"fit_id": "fit-1", "status": "complete"}]
        ),
        pooled_metrics=metrics,
        domain_summaries=domain_summaries,
        selection_frequency=frequency,
        selection_stability=stability,
        endpoint_coverage=build_endpoint_coverage(metrics),
        control_summary=build_control_summary(metrics),
        cost_summary=pd.DataFrame(
            [{"effective_model_id": "C-PCA-LDA", "total_training_seconds": 1.0}]
        ),
        figure_manifest=pd.DataFrame(
            [
                {
                    "figure_id": "F12",
                    "title": "Selection",
                    "data_sha256": "a" * 64,
                    "vector_only": True,
                    "png_dpi": 300,
                }
            ]
        ),
    )
    assert "Endpoint coverage" in report
    assert "Negative and confounding controls" in report
    assert "Compute evidence (M23–M25)" in report
    assert "Interpretation limits" in report
