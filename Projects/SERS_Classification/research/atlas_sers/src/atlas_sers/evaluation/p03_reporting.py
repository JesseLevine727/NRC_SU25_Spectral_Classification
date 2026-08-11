"""Private P03 report, selected-model specification, and P04 comparator freeze."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.governance.canonical import canonical_json_bytes, sha256_value


def _normal(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _records(frame: pd.DataFrame, fields: list[str]) -> list[dict[str, Any]]:
    return [
        {field: _normal(row[field]) for field in fields}
        for _, row in frame[fields].iterrows()
    ]


def _markdown_table(frame: pd.DataFrame, *, maximum_rows: int = 40) -> str:
    if frame.empty:
        return "_No complete rows are available; see the terminal failure ledger._"
    table = frame.head(maximum_rows).copy()
    for column in table.select_dtypes(include=["float"]).columns:
        table[column] = table[column].map(
            lambda value: "NA" if pd.isna(value) else f"{value:.4f}"
        )
    table = table.fillna("NA").astype(str)
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in table.to_numpy().tolist()
    ]
    suffix = (
        f"\n\n_Table truncated to {maximum_rows} of {len(frame)} rows._"
        if len(frame) > maximum_rows
        else ""
    )
    return "\n".join([header, separator, *rows]) + suffix


def build_selected_model_specs(
    *,
    selections: pd.DataFrame,
    candidate_registry: pd.DataFrame,
    coral_candidate_registry: pd.DataFrame,
    metadata_candidate_registry: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve every selected candidate to its exact frozen parameter record."""

    registries: list[pd.DataFrame] = []
    for kind, registry in (
        ("standard", candidate_registry),
        ("source_covariance", coral_candidate_registry),
        ("metadata_control", metadata_candidate_registry),
    ):
        fields = [
            field
            for field in (
                "candidate_id",
                "model_id",
                "parameters_json",
                "hyperparameter_sha256",
                "complexity_rank",
                "declared_candidate_order",
                "technical_seeds",
                "seed_count",
                "base_model_id",
                "base_candidate_id",
                "method_id",
                "method_status",
            )
            if field in registry
        ]
        normalized = registry[fields].copy()
        normalized["candidate_registry_kind"] = kind
        registries.append(normalized)
    candidates = pd.concat(registries, ignore_index=True, sort=False)
    if candidates.candidate_id.astype(str).duplicated().any():
        raise ValueError("P03 selected-model registries contain duplicate candidate IDs.")
    specs = selections.merge(
        candidates,
        left_on="selected_candidate_id",
        right_on="candidate_id",
        how="left",
        validate="many_to_one",
        suffixes=("_selection", "_registry"),
    )
    complete = specs.status.astype(str).eq("complete")
    if specs.loc[complete, "candidate_id"].isna().any():
        raise ValueError("A complete P03 selection lacks its candidate specification.")
    mismatch = complete & ~specs.selected_hyperparameter_sha256.astype(str).eq(
        specs.hyperparameter_sha256.astype(str)
    )
    if mismatch.any():
        raise ValueError("A selected P03 hyperparameter hash differs from its registry.")
    keep = [
        field
        for field in (
            "experiment_id",
            "task_id",
            "outer_run_id",
            "domain",
            "station",
            "held_instrument",
            "outer_repeat",
            "outer_fold",
            "selection_mode",
            "selection_kind",
            "status",
            "reason_code",
            "selected_candidate_id",
            "selected_model_id",
            "selected_hyperparameter_sha256",
            "selection_unit_count",
            "selection_state_sha256",
            "candidate_registry_kind",
            "parameters_json",
            "complexity_rank",
            "declared_candidate_order",
            "technical_seeds",
            "seed_count",
            "base_model_id",
            "base_candidate_id",
            "method_id",
            "method_status",
        )
        if field in specs
    ]
    return specs[keep].sort_values(
        ["experiment_id", "outer_run_id"], kind="stable"
    ).reset_index(drop=True)


def build_p04_comparator_freeze(
    *,
    execution_run_id: str,
    protected_state_sha256: str,
    selected_model_specs: pd.DataFrame,
    expected_endpoints: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Freeze the algorithmic C09 comparator without using it to tune P04."""

    selected = selected_model_specs[
        selected_model_specs.experiment_id.astype(str).eq("EXP-C09-T3")
    ].copy()
    endpoints = expected_endpoints[
        expected_endpoints.experiment_id.astype(str).eq("EXP-C09-T3")
    ].copy()
    metrics = pooled_metrics[
        pooled_metrics.experiment_id.astype(str).eq("EXP-C09-T3")
    ].copy()
    if len(selected) != 260 or len(endpoints) != 260:
        raise RuntimeError("P04 comparator freeze does not contain all 260 C09 outer cells.")
    selection_fields = [
        "outer_run_id",
        "domain",
        "station",
        "outer_repeat",
        "outer_fold",
        "status",
        "selected_candidate_id",
        "selected_model_id",
        "selected_hyperparameter_sha256",
        "selection_state_sha256",
    ]
    endpoint_fields = [
        "outer_run_id",
        "domain",
        "station",
        "outer_repeat",
        "outer_fold",
        "procedure_id",
        "expected_test_rows",
        "expected_test_masters",
        "expected_test_uid_sha256",
    ]
    metric_identity_fields = [
        "domain",
        "station",
        "procedure_id",
        "outer_repeat",
        "aggregation_level",
        "endpoint_status",
        "observation_uid_sha256",
    ]
    return {
        "schema_version": "p04-classical-comparator-freeze-v1",
        "source_phase": "P03",
        "source_execution_run_id": execution_run_id,
        "source_protected_state_sha256": protected_state_sha256,
        "research_question_id": "RQ-P01",
        "classical_procedure_id": "C-SELECTED",
        "population_id": "primary_598",
        "preprocessing_policy_id": "PP-U-MIN",
        "representation_id": "R_MIN_400_1800",
        "information_regime": "zero_shot_source_only",
        "primary_metric_id": "M01",
        "primary_aggregation_level": "spectrum",
        "secondary_master_metric_id": "M06",
        "selection_rule": (
            "registered source-only candidate suite with P02 pseudo-domain or exact "
            "source-master-CV fallback and frozen lexicographic objective"
        ),
        "outer_cell_count": len(endpoints),
        "pooled_domain_repeat_count": int(
            metrics[["domain", "outer_repeat"]].drop_duplicates().shape[0]
        ),
        "complete_selection_count": int(selected.status.astype(str).eq("complete").sum()),
        "terminal_selection_failure_count": int(
            (~selected.status.astype(str).eq("complete")).sum()
        ),
        "selection_mapping_sha256": sha256_value(
            _records(selected.sort_values("outer_run_id"), selection_fields)
        ),
        "outer_test_endpoint_sha256": sha256_value(
            _records(endpoints.sort_values("outer_run_id"), endpoint_fields)
        ),
        "pooled_metric_identity_sha256": sha256_value(
            _records(
                metrics.sort_values(
                    ["domain", "outer_repeat", "aggregation_level"]
                ),
                metric_identity_fields,
            )
        ),
        "neural_development_prohibition": (
            "No P03 outer-test score, error, selected family, control, or figure may choose "
            "P04 architecture, optimizer, epoch, preprocessing, threshold, or fallback."
        ),
        "paired_comparison_rule": (
            "Later deep-versus-classical inference must use identical P02 test UIDs and "
            "domain/repeat identities; failed classical endpoints remain in denominators."
        ),
        "next_action": (
            "Execute P04 D0-ERM source-development validity and learning-curve checks under "
            "the immutable P02 roles before any definitive paired comparison."
        ),
    }


def render_p04_handoff(comparator: dict[str, Any]) -> str:
    """Render a human-readable counterpart to the machine comparator freeze."""

    return "\n".join(
        [
            "# P04 frozen classical comparator handoff",
            "",
            "**Private protected artifact — do not publish without disclosure review.**",
            "",
            f"Source execution: `{comparator['source_execution_run_id']}`",
            f"Protected state: `{comparator['source_protected_state_sha256']}`",
            "",
            "## Comparator",
            "",
            "- Procedure: `C-SELECTED` under `PP-U-MIN` / `R_MIN_400_1800`.",
            "- Regime: zero-shot source-only; no held-target statistic is permitted.",
            f"- Frozen outer cells: {comparator['outer_cell_count']}.",
            (
                "- Complete/terminally failed selections: "
                f"{comparator['complete_selection_count']}/"
                f"{comparator['terminal_selection_failure_count']}."
            ),
            f"- Primary endpoint: `{comparator['primary_metric_id']}` at spectrum level.",
            f"- Secondary master endpoint: `{comparator['secondary_master_metric_id']}`.",
            "",
            "## Immutable hashes",
            "",
            f"- Selection mapping: `{comparator['selection_mapping_sha256']}`",
            f"- Outer-test endpoint registry: `{comparator['outer_test_endpoint_sha256']}`",
            f"- Pooled metric identities: `{comparator['pooled_metric_identity_sha256']}`",
            "",
            "## P04 information boundary",
            "",
            comparator["neural_development_prohibition"],
            "",
            comparator["paired_comparison_rule"],
            "",
            "## Exact next action",
            "",
            comparator["next_action"],
            "",
        ]
    )


def render_p03_report(
    *,
    execution_run_id: str,
    plan_run_id: str,
    protected_state_sha256: str,
    fit_manifest_rows: int,
    terminal_fit_ledger: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    domain_summaries: pd.DataFrame,
    selection_frequency: pd.DataFrame,
    selection_stability: pd.DataFrame,
    endpoint_coverage: pd.DataFrame,
    control_summary: pd.DataFrame,
    cost_summary: pd.DataFrame,
    figure_manifest: pd.DataFrame,
) -> str:
    """Render the complete private classical benchmark report from frozen tables."""

    t1 = pooled_metrics[
        pooled_metrics.task_id.astype(str).str.startswith("T1-")
        & pooled_metrics.endpoint_status.astype(str).eq("complete")
    ]
    t1_summary = (
        t1.groupby(
            ["station", "procedure_id", "aggregation_level"],
            sort=True,
            as_index=False,
        )
        .agg(
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            minimum_repeat_balanced_accuracy=("balanced_accuracy", "min"),
            maximum_repeat_balanced_accuracy=("balanced_accuracy", "max"),
            repeat_count=("outer_repeat", "nunique"),
        )
    )
    c09 = domain_summaries[
        domain_summaries.experiment_id.astype(str).eq("EXP-C09-T3")
    ]
    c09_summary = (
        c09.groupby(["procedure_id", "aggregation_level"], as_index=False, sort=True)
        .agg(
            mean_domain_balanced_accuracy=("mean_domain_balanced_accuracy", "mean"),
            minimum_repeat_mean=("mean_domain_balanced_accuracy", "min"),
            maximum_repeat_mean=("mean_domain_balanced_accuracy", "max"),
            worst_observed_domain=("worst_domain_balanced_accuracy", "min"),
            complete_repeat_count=(
                "summary_status",
                lambda values: int(values.eq("complete").sum()),
            ),
        )
    )
    c09_selection = selection_frequency[
        selection_frequency.experiment_id.astype(str).eq("EXP-C09-T3")
    ][
        [
            "station",
            "selection_outcome_model",
            "selection_count",
            "selection_denominator",
            "selection_fraction",
        ]
    ]
    c09_stability = selection_stability[
        selection_stability.experiment_id.astype(str).eq("EXP-C09-T3")
    ]
    stability_summary = (
        c09_stability.groupby("station", as_index=False, sort=True)
        .agg(
            median_modal_fraction=("modal_fraction", "median"),
            minimum_modal_fraction=("modal_fraction", "min"),
            median_normalized_entropy=("normalized_entropy", "median"),
            decision_count=("modal_fraction", "size"),
        )
    )
    coverage = endpoint_coverage[
        endpoint_coverage.experiment_id.astype(str).str.startswith("EXP-C")
    ][
        [
            "experiment_id",
            "aggregation_level",
            "planned_endpoint_count",
            "complete_endpoint_count",
            "unavailable_endpoint_count",
            "completion_fraction",
        ]
    ]
    status_counts = terminal_fit_ledger.status.astype(str).value_counts().sort_index()
    figure_rows = figure_manifest[
        ["figure_id", "title", "data_sha256", "vector_only", "png_dpi"]
    ]
    return "\n".join(
        [
            "# ATLAS P03 classical benchmark report",
            "",
            "**Private protected artifact — do not publish without disclosure review.**",
            "",
            "## Frozen identity",
            "",
            f"- Execution run: `{execution_run_id}`",
            f"- No-fit plan: `{plan_run_id}`",
            f"- Protected state: `{protected_state_sha256}`",
            f"- Fit-manifest rows: {fit_manifest_rows:,}",
            f"- Terminal fit-ledger rows: {len(terminal_fit_ledger):,}",
            "- Primary population/policy: 598 spectra, 69 physical masters, `PP-U-MIN`.",
            "- Primary representation: `R_MIN_400_1800` (400–1,800 cm⁻¹, row min–max).",
            "",
            "## Terminal accounting",
            "",
            _markdown_table(
                status_counts.rename_axis("terminal_status").rename("fit_records").reset_index()
            ),
            "",
            "Every planned fit ID has exactly one terminal status. Unsupported, failed, inactive, "
            "and dependency-excluded records remain in the ledger and endpoint denominator.",
            "",
            "## Endpoint coverage",
            "",
            _markdown_table(coverage, maximum_rows=36),
            "",
            "## Within-station classical results",
            "",
            _markdown_table(t1_summary, maximum_rows=60),
            "",
            "Values summarize five technical split repeats; ranges are not confidence intervals.",
            "",
            "## Primary unseen-instrument classical comparator",
            "",
            _markdown_table(c09_summary),
            "",
            "Domain means are unweighted across the 13 eligible domains. The worst-domain value "
            "is descriptive. P03 alone makes no classical-versus-deep claim.",
            "",
            "## Source-only candidate selection",
            "",
            _markdown_table(c09_selection, maximum_rows=40),
            "",
            _markdown_table(stability_summary),
            "",
            "Selection used only P02 source pseudo-domains or the exact source-master-CV fallback. "
            "No held-target spectrum, statistic, label, QC summary, or outcome selected a "
            "candidate.",
            "",
            "## Negative and confounding controls",
            "",
            _markdown_table(control_summary, maximum_rows=55),
            "",
            "Permutation results are leakage/chance diagnostics, not a formal permutation-test "
            "p-value. Metadata and prior controls cannot select or promote a model.",
            "",
            "## Compute evidence (M23–M25)",
            "",
            _markdown_table(cost_summary, maximum_rows=50),
            "",
            "Training time and per-prediction inference latency are observed wall-clock "
            "diagnostics "
            "on this workstation. They are not hardware-independent complexity measures.",
            "",
            "## Figures",
            "",
            _markdown_table(figure_rows),
            "",
            "Each figure is generated from one frozen CSV as native TikZ/PGFPlots, vector PDF, "
            "300-DPI PNG, and standalone self-contained HTML. Figure tables and HTML remain "
            "private "
            "until disclosure review.",
            "",
            "## Interpretation limits",
            "",
            "- Spectra, folds, and technical repeats are not independent chemical samples.",
            "- The endpoint concerns the tested 13 station/instrument domains and three-class "
            "station tasks; it does not establish arbitrary instrument or chemistry "
            "generalization.",
            "- `PP-U-MIN` is the fixed primary policy. This phase does not establish universal "
            "superiority over smoothing, baseline correction, or adaptive preprocessing.",
            "- C12 is a secondary source-to-source covariance augmentation control, not target "
            "adaptation and not evidence of causal nuisance disentanglement.",
            "- Probability metrics are reported only where source-development cross-fitted "
            "calibration is valid; permutation controls intentionally omit them.",
            "- P03 freezes a classical comparator for P04/P06. It does not compare against a deep "
            "model or justify a deep-learning claim by itself.",
            "",
        ]
    )


def comparator_json_bytes(comparator: dict[str, Any]) -> bytes:
    return canonical_json_bytes(comparator, pretty=True)
