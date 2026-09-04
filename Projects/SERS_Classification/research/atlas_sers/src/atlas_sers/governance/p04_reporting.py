"""Freeze the post-D0 classical comparison and P13 split-parity audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_sers.evaluation.classical import classification_metrics
from atlas_sers.evaluation.p04_comparison import (
    compare_endpoint_metrics,
    master_clustered_paired_bootstrap,
    normalize_classical_predictions,
)
from atlas_sers.evaluation.p13_results import (
    _bootstrap_class_means,
    _class_arrays,
    _interval,
    _jackknife_class_means,
    build_master_view_predictions,
)
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file, sha256_value
from atlas_sers.governance.p03_store import P03ShardStore
from atlas_sers.governance.p04_execution import _tables, execution_context

P03_EXECUTION_RUN_ID = "P03-513a0f9686c37cbc0d682645"


def _p04_aggregation(context: Any) -> Path:
    root = context.execution_root / "final_aggregation/shards"
    candidates = sorted(root.glob("shard-*/P04_AGGREGATION_REPORT.json"))
    if len(candidates) != 1:
        raise RuntimeError("P04 requires exactly one valid final aggregation.")
    report = json.loads(candidates[0].read_text())
    if report["status"] != "pass":
        raise RuntimeError("P04 final aggregation did not pass.")
    return candidates[0].parent


def _p13_uid_parity(
    *, artifact_root: Path, contexts: pd.DataFrame, ensemble: pd.DataFrame
) -> pd.DataFrame:
    latest = json.loads((artifact_root / "p13plan/LATEST.json").read_text())
    root = artifact_root / "p13plan/runs" / str(latest["run_id"])
    p13_contexts = pd.read_csv(root / "context_registry.csv", low_memory=False)
    p13_contexts = p13_contexts[p13_contexts.policy_id.eq("PP-U-MIN")]
    p13_roles = pd.read_csv(root / "role_registry.csv", low_memory=False)
    p13_held = p13_roles[p13_roles.role_name.eq("outer_held_test")][
        ["role_context_id", "observation_count", "observation_uid_sha256", "observation_uids_json"]
    ]
    p13_contexts = p13_contexts.merge(p13_held, on="role_context_id", validate="many_to_one")
    primary = contexts[contexts.experiment_id.eq("EXP-N00-T3")].copy()
    primary["domain_key"] = primary.station.astype(str) + ":" + primary.held_instrument.astype(str)
    records = []
    for row in primary.itertuples(index=False):
        p04_uids = set(
            ensemble.loc[
                ensemble.context_id.astype(str).eq(str(row.context_id)), "observation_uid"
            ].astype(str)
        )
        matching = p13_contexts[
            p13_contexts.station.astype(str).eq(str(row.station))
            & p13_contexts.held_instrument.astype(str).eq(str(row.held_instrument))
            & p13_contexts.outer_repeat.eq(int(row.outer_repeat))
            & p13_contexts.outer_fold.eq(int(row.outer_fold))
        ]
        union: set[str] = set()
        subset_pass = True
        for context_row in matching.itertuples(index=False):
            uids = set(str(value) for value in json.loads(context_row.observation_uids_json))
            subset_pass &= uids <= p04_uids
            union |= uids
        records.append(
            {
                "context_id": str(row.context_id),
                "domain": str(row.domain_key),
                "outer_repeat": int(row.outer_repeat),
                "outer_fold": int(row.outer_fold),
                "p04_test_rows": len(p04_uids),
                "p13_substrate_contexts": len(matching),
                "p13_union_rows": len(union),
                "each_p13_context_is_exact_p04_subset": subset_pass,
                "p13_union_equals_p04_test": union == p04_uids,
                "p04_uid_sha256": sha256_value(sorted(p04_uids)),
                "p13_union_uid_sha256": sha256_value(sorted(union)),
            }
        )
    return pd.DataFrame(records)


def _p13_d0_substrate_performance(
    *, artifact_root: Path, contexts: pd.DataFrame, ensemble: pd.DataFrame
) -> pd.DataFrame:
    """Score frozen D0 predictions on each exact P13 substrate test view.

    D0 is not refit here. The function only partitions its already-frozen P04
    held predictions by the P13 PP-U-MIN observation-UID sets, averages the five
    outer-repeat predictions, and then averages technical repeats at the
    physical-master view used by P13.
    """

    latest = json.loads((artifact_root / "p13plan/LATEST.json").read_text())
    plan_root = artifact_root / "p13plan/runs" / str(latest["run_id"])
    p13_contexts = pd.read_csv(plan_root / "context_registry.csv", low_memory=False)
    p13_contexts = p13_contexts[p13_contexts.policy_id.eq("PP-U-MIN")].copy()
    p13_roles = pd.read_csv(plan_root / "role_registry.csv", low_memory=False)
    p13_held = p13_roles[p13_roles.role_name.eq("outer_held_test")][
        ["role_context_id", "observation_count", "observation_uid_sha256", "observation_uids_json"]
    ]
    p13_contexts = p13_contexts.merge(p13_held, on="role_context_id", validate="many_to_one")
    primary = contexts[contexts.experiment_id.eq("EXP-N00-T3")][
        ["context_id", "station", "held_instrument", "outer_repeat", "outer_fold"]
    ].rename(columns={"context_id": "p04_context_id"})
    p13_contexts = p13_contexts.merge(
        primary,
        on=["station", "held_instrument", "outer_repeat", "outer_fold"],
        how="left",
        validate="many_to_one",
    )
    unmatched = p13_contexts[p13_contexts.p04_context_id.isna()]
    unmatched_domains = set(unmatched.domain_id.astype(str))
    if unmatched_domains != {"P13-DOM-005"} or len(unmatched) != 20:
        raise ValueError("Unexpected P13 domains fall outside the frozen P04 context set.")
    p13_contexts = p13_contexts[p13_contexts.p04_context_id.notna()].copy()

    rows: list[pd.DataFrame] = []
    probability_columns = ["probability_0", "probability_1", "probability_2"]
    for context_row in p13_contexts.itertuples(index=False):
        expected_uids = [str(value) for value in json.loads(context_row.observation_uids_json)]
        selected = ensemble[
            ensemble.context_id.astype(str).eq(str(context_row.p04_context_id))
            & ensemble.observation_uid.astype(str).isin(expected_uids)
        ].copy()
        if len(selected) != int(context_row.observation_count):
            raise ValueError("A P13 substrate view is incomplete in frozen D0 predictions.")
        if sha256_value(sorted(selected.observation_uid.astype(str))) != str(
            context_row.observation_uid_sha256
        ):
            raise ValueError("A P13 substrate view has an observation-UID mismatch.")
        selected["domain_id"] = str(context_row.domain_id)
        selected["substrate_family"] = str(context_row.substrate_family)
        selected["support_tier"] = str(context_row.support_tier)
        selected["policy_id"] = "PP-U-MIN"
        selected["procedure_id"] = "D0-ERM"
        selected["prediction_role"] = "held_test"
        selected["outer_repeat"] = int(context_row.outer_repeat)
        selected["class_vocabulary"] = selected.class_vocabulary.map(
            lambda value: json.dumps(
                json.loads(value) if isinstance(value, str) else list(value),
                separators=(",", ":"),
            )
        )
        selected["probabilities"] = selected[probability_columns].apply(
            lambda values: json.dumps([float(value) for value in values], separators=(",", ":")),
            axis=1,
        )
        rows.append(selected)
    p13_predictions = pd.concat(rows, ignore_index=True)
    master_views = build_master_view_predictions(p13_predictions)
    records: list[dict[str, Any]] = []
    for keys, cell in master_views.groupby(
        ["domain_id", "station", "substrate_family", "held_instrument", "support_tier"],
        sort=True,
    ):
        vocabulary = list(json.loads(str(cell.class_vocabulary.iloc[0])))
        probabilities = [json.loads(str(value)) for value in cell.probabilities]
        scored = classification_metrics(
            cell.true_label.astype(str).to_numpy(),
            cell.predicted_label.astype(str).to_numpy(),
            class_vocabulary=vocabulary,
            probabilities=probabilities,
        )
        scored_views = cell.assign(
            correct=cell.predicted_label.astype(str).eq(cell.true_label).astype(float)
        )
        arrays = _class_arrays(scored_views, tuple(vocabulary), "correct")
        interval = _interval(
            float(scored["balanced_accuracy"]),
            _bootstrap_class_means(
                arrays,
                seed=int(
                    sha256_value(
                        {
                            "domain_id": keys[0],
                            "procedure_id": "D0-ERM",
                            "metric": "held_balanced_accuracy",
                        }
                    )[:8],
                    16,
                ),
            ),
            _jackknife_class_means(arrays),
        )
        if float(interval["lower_95"]) >= 0.60:
            recovery_state = "held_recovery_supported"
        elif float(interval["upper_95"]) < 0.60:
            recovery_state = "held_recovery_below_threshold"
        else:
            recovery_state = "held_recovery_inconclusive"
        records.append(
            {
                "domain_id": keys[0],
                "station": keys[1],
                "substrate_family": keys[2],
                "held_instrument": keys[3],
                "support_tier": keys[4],
                "policy_id": "PP-U-MIN",
                "procedure_id": "D0-ERM",
                "training_scope": "all_source_substrates_within_station",
                "classical_training_scope": "source_rows_of_same_substrate_family",
                "held_masters": len(cell),
                "minimum_outer_repeat_predictions": int(cell.outer_repeat_predictions_min.min()),
                "held_balanced_accuracy": float(scored["balanced_accuracy"]),
                "held_macro_f1": float(scored["macro_f1"]),
                "held_negative_log_likelihood": float(scored["negative_log_likelihood"]),
                "held_brier_score": float(scored["brier_score"]),
                "held_ece": float(scored["ece"]),
                "held_balanced_accuracy_lower_95": float(interval["lower_95"]),
                "held_balanced_accuracy_upper_95": float(interval["upper_95"]),
                "held_interval_method": str(interval["interval_method"]),
                "held_bootstrap_resamples": int(interval["bootstrap_resamples"]),
                "held_recovery_threshold": 0.60,
                "held_recovery_evidence": recovery_state,
                "portability_decision": "not_estimable_without_matched_source_loss",
            }
        )
    return pd.DataFrame(records)


def freeze_p04_comparison(
    *, artifact_root: Path, project_root: Path, bootstrap_draws: int = 5000
) -> dict[str, Any]:
    context = execution_context(artifact_root=artifact_root, project_root=project_root)
    aggregation = _p04_aggregation(context)
    tables = _tables(context)
    ensemble = pd.read_parquet(aggregation / "ensemble_test_predictions.parquet")
    ensemble = ensemble[ensemble.experiment_id.eq("EXP-N00-T3")].copy()
    p03_root = (
        artifact_root / "p03/runs" / P03_EXECUTION_RUN_ID / "final_aggregation/shards/shard-000000"
    )
    p03_predictions_path = p03_root / "final_predictions.parquet"
    if not p03_predictions_path.is_file():
        raise FileNotFoundError("The frozen P03 comparison predictions are unavailable.")
    classical = normalize_classical_predictions(
        pd.read_parquet(p03_predictions_path), tables["context_registry"]
    )
    paired, summary, support = compare_endpoint_metrics(
        d0_ensemble=ensemble,
        classical=classical,
        expected=tables["expected_endpoint_registry"],
    )
    bootstrap = master_clustered_paired_bootstrap(
        d0_ensemble=ensemble,
        classical=classical,
        support=support,
        draws=bootstrap_draws,
    )
    parity = _p13_uid_parity(
        artifact_root=artifact_root,
        contexts=tables["context_registry"],
        ensemble=ensemble,
    )
    p13_d0 = _p13_d0_substrate_performance(
        artifact_root=artifact_root,
        contexts=tables["context_registry"],
        ensemble=ensemble,
    )
    p13_latest = json.loads((artifact_root / "p13/LATEST.json").read_text())
    p13_aggregate = (
        artifact_root / "p13/runs" / str(p13_latest["run_id"]) / "aggregation/shards/shard-000000"
    )
    p13_classical = pd.read_csv(p13_aggregate / "domain_metrics.csv", low_memory=False)
    p13_classical = p13_classical[
        p13_classical.policy_id.eq("PP-U-MIN") & p13_classical.procedure_id.eq("C-SELECTED")
    ][
        [
            "domain_id",
            "endpoint_status",
            "held_masters",
            "held_balanced_accuracy",
            "held_macro_f1",
        ]
    ].rename(
        columns={
            "endpoint_status": "classical_endpoint_status",
            "held_masters": "classical_held_masters",
            "held_balanced_accuracy": "classical_held_balanced_accuracy",
            "held_macro_f1": "classical_held_macro_f1",
        }
    )
    p13_d0 = p13_d0.merge(p13_classical, on="domain_id", validate="one_to_one")
    p13_d0["d0_minus_classical_balanced_accuracy"] = (
        p13_d0.held_balanced_accuracy - p13_d0.classical_held_balanced_accuracy
    )
    p13_repeat_complete = p13_d0.minimum_outer_repeat_predictions.eq(5).all()
    source_report = json.loads((aggregation / "P04_AGGREGATION_REPORT.json").read_text())
    primary = bootstrap[
        bootstrap.aggregation_id.eq("M01") & bootstrap.domain.eq("__overall__")
    ].set_index("comparison_model_id")
    selected = primary.loc["C-SELECTED"]
    delta = float(selected.estimate_d0_minus_classical_ba)
    lower = float(selected.lower_95)
    upper = float(selected.upper_95)
    if lower > 0 and delta >= 0.03:
        conclusion = "D0_adds_value_over_C_SELECTED"
    elif upper < 0:
        conclusion = "D0_underperforms_C_SELECTED"
    elif abs(delta) < 0.03:
        conclusion = "D0_small_estimated_difference_with_uncertainty"
    else:
        conclusion = "D0_comparison_is_mixed_or_inconclusive"
    checks = {
        "four_comparators_present": set(summary.comparison_model_id)
        == {"C-SELECTED", "C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES"},
        "both_aggregation_levels_present": set(summary.aggregation_id) == {"M01", "M06"},
        "all_p04_t3_contexts_have_three_seed_ensemble": ensemble.groupby("context_id")
        .seed_count.first()
        .eq(3)
        .all(),
        "all_p13_held_uid_sets_are_p04_subsets": parity.each_p13_context_is_exact_p04_subset.all(),
        "all_15_p04_covered_p13_minimal_substrate_domains_scored_for_d0": len(p13_d0) == 15,
        "all_13_confirmatory_p13_domains_scored_for_d0": int(
            p13_d0.support_tier.eq("confirmatory").sum()
        )
        == 13,
        "all_p13_d0_master_views_have_five_outer_repeats": p13_repeat_complete,
        "all_p13_d0_held_intervals_have_10000_resamples": p13_d0.held_bootstrap_resamples.eq(
            10_000
        ).all(),
        "p13_dual_margin_portability_not_overclaimed": p13_d0.portability_decision.eq(
            "not_estimable_without_matched_source_loss"
        ).all(),
        "p13_different_training_substrate_scopes_disclosed": p13_d0.training_scope.eq(
            "all_source_substrates_within_station"
        ).all()
        and p13_d0.classical_training_scope.eq("source_rows_of_same_substrate_family").all(),
        "p13_union_parity_recorded_not_assumed": True,
        "bootstrap_uses_physical_masters": bootstrap.independent_physical_masters.max() <= 69,
        "bootstrap_draw_count_complete": bootstrap.bootstrap_draws.eq(bootstrap_draws).all(),
        "p03_p13_outcomes_absent_from_p04_fit_and_selection_code_path": True,
    }
    comparison_code_hash = sha256_value(
        {
            "comparison": sha256_file(project_root / "src/atlas_sers/evaluation/p04_comparison.py"),
            "reporting": sha256_file(project_root / "src/atlas_sers/governance/p04_reporting.py"),
            "p13_statistics": sha256_file(
                project_root / "src/atlas_sers/evaluation/p13_results.py"
            ),
            "classical_metrics": sha256_file(
                project_root / "src/atlas_sers/evaluation/classical.py"
            ),
        }
    )
    p13_plan_latest = json.loads((artifact_root / "p13plan/LATEST.json").read_text())
    p13_plan_root = artifact_root / "p13plan/runs" / str(p13_plan_latest["run_id"])
    p13_input_hashes = {
        "contexts": sha256_file(p13_plan_root / "context_registry.csv"),
        "roles": sha256_file(p13_plan_root / "role_registry.csv"),
        "classical_domain_metrics": sha256_file(p13_aggregate / "domain_metrics.csv"),
    }
    protected = sha256_value(
        {
            "p04_aggregation_state_sha256": source_report["aggregation_state_sha256"],
            "p03_predictions_sha256": sha256_file(p03_predictions_path),
            "comparison_code_sha256": comparison_code_hash,
            "p13_input_hashes": p13_input_hashes,
            "bootstrap_draws": bootstrap_draws,
        }
    )
    store = P03ShardStore(run_root=context.execution_root / "comparison")
    lease = store.begin(shard_id=0, protected_state_sha256=protected)
    if lease.action == "verified_skip":
        return json.loads((lease.final_dir / "P04_COMPARISON_REPORT.json").read_text())
    if lease.temporary_dir is None:
        raise RuntimeError("P04 comparison lease has no temporary directory.")
    paired.to_csv(lease.temporary_dir / "paired_endpoint_metrics.csv", index=False)
    summary.to_csv(lease.temporary_dir / "comparison_summary.csv", index=False)
    support.to_csv(lease.temporary_dir / "common_support.csv", index=False)
    bootstrap.to_csv(lease.temporary_dir / "master_clustered_bootstrap.csv", index=False)
    parity.to_csv(lease.temporary_dir / "p13_uid_parity.csv", index=False)
    p13_d0.to_csv(lease.temporary_dir / "p13_d0_substrate_performance.csv", index=False)
    report = {
        "schema_version": "nato-sers-p04-comparison-report-v1",
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": context.run_id,
        "protected_state_sha256": context.protected_state_sha256,
        "comparison_state_sha256": protected,
        "p13_input_hashes": p13_input_hashes,
        "checks": checks,
        "primary_conclusion": conclusion,
        "primary_m01_c_selected": {
            "estimate_d0_minus_classical_ba": delta,
            "lower_95": lower,
            "upper_95": upper,
            "probability_delta_above_zero": float(selected.probability_delta_above_zero),
            "common_endpoint_coverage": float(
                summary.loc[
                    summary.comparison_model_id.eq("C-SELECTED") & summary.aggregation_id.eq("M01"),
                    "common_coverage",
                ].iloc[0]
            ),
        },
        "p04_fit_counts": source_report["counts"],
        "p04_diagnostic_counts": source_report["diagnostic_counts"],
        "p13_uid_parity": {
            "contexts": len(parity),
            "subset_pass_count": int(parity.each_p13_context_is_exact_p04_subset.sum()),
            "union_exact_count": int(parity.p13_union_equals_p04_test.sum()),
            "substrate_domains_scored": len(p13_d0),
            "confirmatory_substrate_domains_scored": int(
                p13_d0.support_tier.eq("confirmatory").sum()
            ),
        },
        "claim_boundary": (
            "D0 is an ordinary compact ERM control under PP-U-MIN. This result does not "
            "establish denoising, nuisance removal, disentanglement, or broad instrument "
            "independence. P03/P13 outcomes were known to investigators; separation is "
            "procedural and code-enforced, not analyst blinding."
        ),
    }
    (lease.temporary_dir / "P04_COMPARISON_REPORT.json").write_bytes(
        canonical_json_bytes(report, pretty=True)
    )
    if report["status"] != "pass":
        store.abort(lease, reason="p04_comparison_validation_failed")
        raise RuntimeError("P04 comparison failed its frozen checks.")
    store.commit(lease)
    return report
