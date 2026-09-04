# ruff: noqa: E501
"""Publish disclosure-safe aggregate P04 evidence and regenerate its figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file
from atlas_sers.governance.p04_execution import execution_context
from atlas_sers.visualization.p04_figures import generate_p04_figures


def _single_shard(root: Path, report_name: str) -> Path:
    matches = sorted(root.glob(f"shards/shard-*/{report_name}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {report_name} artifact.")
    report = json.loads(matches[0].read_text())
    if report["status"] != "pass":
        raise RuntimeError(f"{report_name} did not pass.")
    return matches[0].parent


def _domain_performance(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, cell in metrics.groupby(
        ["experiment_id", "aggregation_id", "station", "domain"], sort=True
    ):
        rows.append(
            {
                "experiment_id": keys[0],
                "aggregation_id": keys[1],
                "station": keys[2],
                "domain": keys[3],
                "outer_cells": len(cell),
                "mean_balanced_accuracy": float(cell.balanced_accuracy.mean()),
                "minimum_balanced_accuracy": float(cell.balanced_accuracy.min()),
                "mean_macro_f1": float(cell.macro_f1.mean()),
                "mean_negative_log_likelihood": float(cell.negative_log_likelihood.mean()),
                "mean_brier_score": float(cell.brier_score.mean()),
                "mean_ece": float(cell.ece.mean()),
            }
        )
    return pd.DataFrame(rows)


def _overall_performance(metrics: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    merged = coverage.merge(
        metrics[
            [
                "context_id",
                "aggregation_id",
                "balanced_accuracy",
                "macro_f1",
                "negative_log_likelihood",
                "brier_score",
                "ece",
            ]
        ],
        on="context_id",
        how="left",
        validate="one_to_many",
    )
    rows = []
    for keys, cell in merged.groupby(["experiment_id", "aggregation_id"], sort=True):
        complete = cell[cell.status.eq("complete")]
        domain = complete.groupby("domain", as_index=False).balanced_accuracy.mean()
        rows.append(
            {
                "experiment_id": keys[0],
                "aggregation_id": keys[1],
                "planned_outer_cells": len(cell),
                "complete_outer_cells": len(complete),
                "coverage": len(complete) / len(cell),
                "mean_balanced_accuracy": float(complete.balanced_accuracy.mean()),
                "mean_domain_balanced_accuracy": float(domain.balanced_accuracy.mean()),
                "worst_domain_balanced_accuracy": float(domain.balanced_accuracy.min()),
                "failure_sensitive_mean_balanced_accuracy": float(
                    cell.balanced_accuracy.fillna(0).mean()
                ),
                "mean_macro_f1": float(complete.macro_f1.mean()),
                "mean_negative_log_likelihood": float(complete.negative_log_likelihood.mean()),
                "mean_brier_score": float(complete.brier_score.mean()),
                "mean_ece": float(complete.ece.mean()),
            }
        )
    return pd.DataFrame(rows)


def _fit_summary(status: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        status.assign(is_complete=status.status.eq("complete"))
        .groupby(["experiment_id", "stage"], as_index=False)
        .agg(
            planned_or_terminal_fits=("fit_id", "size"),
            complete_fits=("is_complete", "sum"),
            mean_epochs=("epochs_completed", "mean"),
            median_epochs=("epochs_completed", "median"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
            total_elapsed_seconds=("elapsed_seconds", "sum"),
            peak_cuda_megabytes=("peak_cuda_bytes", lambda values: values.max() / 1e6),
        )
    )
    summary["completion_fraction"] = summary.complete_fits / summary.planned_or_terminal_fits
    diagnostics = (
        status.assign(diagnostic=status.diagnostic.fillna("terminal_failure"))
        .groupby(["experiment_id", "stage", "diagnostic"], as_index=False)
        .size()
        .rename(columns={"size": "fit_count"})
    )
    selections = (
        status[status.stage.eq("inner_selection")]
        .groupby(
            ["experiment_id", "candidate_id", "learning_rate", "weight_decay"],
            as_index=False,
        )
        .agg(
            fit_count=("fit_id", "size"),
            complete_fit_count=("status", lambda values: values.eq("complete").sum()),
            mean_best_validation_ba=("best_validation_balanced_accuracy", "mean"),
            mean_best_validation_nll=("best_validation_nll", "mean"),
            median_best_epoch=("best_epoch", "median"),
        )
    )
    return summary, diagnostics, selections


def _write_report(
    *,
    path: Path,
    overall: pd.DataFrame,
    comparison_report: dict[str, Any],
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    diagnostics: pd.DataFrame,
    epochs: pd.DataFrame,
    fit_summary: pd.DataFrame,
    p13_d0: pd.DataFrame,
) -> None:
    dev = overall[overall.experiment_id.eq("EXP-N00-DEV") & overall.aggregation_id.eq("M01")].iloc[
        0
    ]
    t3 = overall[overall.experiment_id.eq("EXP-N00-T3") & overall.aggregation_id.eq("M01")].iloc[0]
    t3_master = overall[
        overall.experiment_id.eq("EXP-N00-T3") & overall.aggregation_id.eq("M06")
    ].iloc[0]
    selected = comparison[
        comparison.comparison_model_id.eq("C-SELECTED") & comparison.aggregation_id.eq("M01")
    ].iloc[0]
    overfit = int(diagnostics.loc[diagnostics.diagnostic.eq("overfit"), "fit_count"].sum())
    collapse = int(diagnostics.loc[diagnostics.diagnostic.eq("collapse"), "fit_count"].sum())
    selected_epoch = epochs.set_index("experiment_id").selected_epoch_median.to_dict()
    final_epochs = (
        fit_summary[fit_summary.stage.eq("final_selected_refit")]
        .set_index("experiment_id")
        .median_epochs.to_dict()
    )
    comparison_lines = [
        "| Classical comparator | Complete paired cells | D0 BA | Classical BA | Difference |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison[comparison.aggregation_id.eq("M01")].itertuples(index=False):
        comparison_lines.append(
            f"| {row.comparison_model_id} | {row.common_complete_endpoints}/{row.planned_endpoints} "
            f"| {row.d0_mean_ba_common:.3f} | {row.classical_mean_ba_common:.3f} "
            f"| {row.mean_paired_delta_d0_minus_classical:+.3f} |"
        )
    diagnostic_lines = [
        "| Inner-fit diagnostic | Development fits | T3 source-selection fits |",
        "| --- | ---: | ---: |",
    ]
    diagnostic_counts = diagnostics[diagnostics.stage.eq("inner_selection")].set_index(
        ["experiment_id", "diagnostic"]
    ).fit_count.to_dict()
    for label in ("none", "overfit", "collapse", "underfit", "optimization_instability"):
        diagnostic_lines.append(
            f"| {label} | {diagnostic_counts.get(('EXP-N00-DEV', label), 0)} "
            f"| {diagnostic_counts.get(('EXP-N00-T3', label), 0)} |"
        )
    p13_confirmatory = p13_d0[p13_d0.support_tier.eq("confirmatory")]
    interval_lines = [
        "| Classical comparator | Pooled M01 difference | Conditional 95% interval |",
        "| --- | ---: | ---: |",
    ]
    for row in bootstrap[
        bootstrap.aggregation_id.eq("M01") & bootstrap.domain.eq("__overall__")
    ].itertuples(index=False):
        interval_lines.append(
            f"| {row.comparison_model_id} | {row.estimate_d0_minus_classical_ba:+.3f} "
            f"| [{row.lower_95:+.3f}, {row.upper_95:+.3f}] |"
        )
    p13_recovery_supported = int(
        p13_confirmatory.held_recovery_evidence.eq("held_recovery_supported").sum()
    )
    text = f"""# P04 compact deep baseline results

P04 passed its locked training-validity gate and produced a complete compact ordinary deep-learning baseline. The one-dimensional residual model has **208,691 trainable parameters**, consumes the frozen 1,401-channel `R_MIN_400_1800` representation, and uses no BatchNorm. All endpoint selection and stopping evidence came from authorized source-only inner roles. P03/P13 results were already known to investigators, but the P04 fitting and selection path could not load them; they entered only the separate post-freeze comparison. This is procedural separation, not analyst blinding.

## Main result

- Within-station development (`EXP-N00-DEV`, M01): mean balanced accuracy **{dev.mean_balanced_accuracy:.3f}** across {int(dev.complete_outer_cells)}/{int(dev.planned_outer_cells)} complete outer cells.
- Unseen-instrument evaluation (`EXP-N00-T3`, M01): mean domain-balanced accuracy **{t3.mean_domain_balanced_accuracy:.3f}**, worst-domain balanced accuracy **{t3.worst_domain_balanced_accuracy:.3f}**, and endpoint coverage **{t3.coverage:.1%}**.
- After instrument-balanced physical-master aggregation (M06), unseen-instrument mean domain-balanced accuracy was **{t3_master.mean_domain_balanced_accuracy:.3f}**, with worst-domain balanced accuracy **{t3_master.worst_domain_balanced_accuracy:.3f}**.
- Unseen-instrument spectrum probability quality: mean negative log likelihood **{t3.mean_negative_log_likelihood:.3f}**, Brier score **{t3.mean_brier_score:.3f}**, and expected calibration error **{t3.mean_ece:.3f}**; lower is better for all three.
- Against frozen C-SELECTED on the common M01 denominator: D0 **{selected.d0_mean_ba_common:.3f}** versus classical **{selected.classical_mean_ba_common:.3f}**, paired difference **{selected.mean_paired_delta_d0_minus_classical:+.3f}**.
- Pooled out-of-fold, physical-master-clustered D0-minus-C-SELECTED M01 difference: **{comparison_report["primary_m01_c_selected"]["estimate_d0_minus_classical_ba"]:+.3f}**, with 95% interval **[{comparison_report["primary_m01_c_selected"]["lower_95"]:+.3f}, {comparison_report["primary_m01_c_selected"]["upper_95"]:+.3f}]**.
- Frozen conclusion: **`{comparison_report["primary_conclusion"]}`**.
- This advantage is comparator-specific: the paired intervals do **not** establish an advantage over fixed Random Forest or Extra Trees. Extra Trees has a slightly higher fold-mean BA, while the pooled D0-minus-Extra-Trees estimate is slightly positive; both are consistent with a small, uncertain difference rather than a general deep-learning win.
- On the exact P13 substrate-stratified PP-U-MIN test views, frozen D0 was scored in **{len(p13_d0)}/16** domains, including **{len(p13_confirmatory)}/13** confirmatory domains; **{p13_recovery_supported}/13** confirmatory domains had a 95% lower bound at or above the P13 held-recovery threshold of 0.60. P13-DOM-005 (CWA/Agilent-3, exploratory low support) is outside the 13-domain P04 eligibility set. This is a post-freeze stratification of the same D0 predictions, not a second model-selection exercise. It cannot issue the full P13 portability verdict because P04 did not generate the matched-source loss endpoint required by that dual-margin rule.

The P13 reuse also has a training-support difference: D0 learned from all source substrates within the station, whereas P13 classical models fitted only the substrate family being evaluated. Their common held-test views permit descriptive recovery comparisons, but any D0-minus-P13-classical difference combines learning strategy with training-data scope. A controlled P13 learner comparison requires new deep refits on the exact substrate-restricted P13 source roles. The primary P04-versus-P03 comparison above uses the shared P02 design and is unaffected by this limitation.

## Training behavior

The source-selected median best epoch was **{selected_epoch.get("EXP-N00-DEV", float("nan")):.0f}** in development and **{selected_epoch.get("EXP-N00-T3", float("nan")):.0f}** in T3 source-only selection. These checkpoint epochs differ from the number of epochs executed while waiting for early stopping. Final refits used a per-seed median of the selected candidate's inner checkpoint epochs, clipped to 30–200; their median durations were **{final_epochs.get("EXP-N00-DEV", float("nan")):.0f}** epochs in development and **{final_epochs.get("EXP-N00-T3", float("nan")):.0f}** epochs in T3. No outer-test result chose these durations.

Diagnostic labels describe optimization traces rather than discarded runs: {overfit} fits met the locked overfit diagnostic and {collapse} met the collapse diagnostic. A finite fit can still overfit or collapse. Every failed or collapsed fit remains in coverage and failure-sensitive accounting. The full fit counts, diagnostic categories, optimizer selections, and execution costs are in `tables/fit_summary.csv`, `tables/training_diagnostics.csv`, and `tables/selected_candidate_frequency.csv`.

{chr(10).join(diagnostic_lines)}

The locked diagnostics mean: **overfit**, checkpoint training BA exceeds validation BA by more than 0.20; **collapse**, the best validation prediction contains fewer than two predicted classes; **underfit**, best training BA is at most chance plus 0.05; **optimization instability**, last-ten validation BA standard deviation exceeds 0.15 or more than half the optimizer steps are gradient-clipped. These are assigned diagnostic categories, not independent hypothesis tests, and concern source-validation runs. Final refits have no test-based diagnostic or early stopping. A `none` label means none of these rules fired, not that generalization is guaranteed.

The registered execution contains **{int(fit_summary.planned_or_terminal_fits.sum()):,}** fits, of which **{int(fit_summary.complete_fits.sum()):,}** completed. The three retained pre-outcome implementation smoke failures are recorded separately in the deviations registry. Timing columns sum elapsed time within individual fits; because fits run concurrently, that sum is neither end-to-end elapsed time nor a measurement of exclusive GPU hours.

This is repeated evaluation of one architecture. The 15,498 inner fits compare six optimizer settings across source-validation units and three seeds. The 960 final refits are 320 evaluation contexts times three seeds; their calibrated probabilities are averaged within context. The 208,691 parameters belong to each individual model. More fits or repeated spectra do not create additional independent physical samples.

## Fair classical comparison and metric definitions

{chr(10).join(comparison_lines)}

The table averages the paired outer-fold balanced accuracies on each comparator's exact common-success cells. Classical scores are recomputed from frozen P03 row predictions using the same P04 aggregation as D0; they can differ from P03's earlier pooled-repeat summaries. Missing classical endpoints are retained in the separate coverage and failure-sensitive columns of `tables/comparison_summary.csv`; common-success means alone cannot establish operational reliability.

{chr(10).join(interval_lines)}

Balanced accuracy (BA) averages class recall. Each station has three classes, for which uniform random prediction has expected BA 1/3 and perfect classification scores 1. The inherited endpoint metric averages recall only over classes actually present in that outer-test cell; some small instrument-specific folds lack one or more classes. The pooled out-of-fold domain comparison below instead requires all three classes. Consequently, the fold-mean table is not a substitute for that pooled three-class comparison. M01 scores spectra. M06 first averages probabilities within each instrument view of a physical master, then weights the available instrument views equally. The same 69 physical masters underlie both summaries. Macro-F1 uses the full station class vocabulary, assigning zero to an undefined class F1. Negative log likelihood, Brier score, and expected calibration error assess predicted probabilities; lower values are better. Calibration on small test cells is uncertain and is not a substitute for discrimination.

The P04 interval and F48 use 5,000 paired bootstrap draws of physical masters, stratified by station and class, carrying each sampled master's repeated predictions and instrument views together. They condition on the observed domain set; draws missing a required class in any included domain are rejected, so the intervals additionally condition on retaining three-class support. Repeats are averaged before inference, and each domain receives equal weight. Their point estimate pools out-of-fold class recalls within each domain, so it can differ from the table's mean of fold-level BAs. This baseline diagnostic does not implement the P11 final primary interval, which additionally resamples domains with 10,000 draws. It cannot by itself pass G4, demonstrate equivalence, or justify a general claim about arbitrary unseen instruments.

## Interpretation

D0 tests whether a compact location-preserving convolutional model adds predictive value over classical methods under the same minimal preprocessing and master-grouped held-instrument design. It does **not** test whether a network removed noise, recovered clean Raman spectra, or disentangled chemistry from acquisition nuisance. Passing G2 means the architecture trained reproducibly enough to serve as the D0 control for P05; it does not by itself establish superiority.

Probability quality is a separate limitation. Uniform three-class probabilities have negative log likelihood 1.099 and Brier score 0.667. D0's held-instrument spectrum NLL of {t3.mean_negative_log_likelihood:.3f} is worse than that uniform reference, although its Brier score of {t3.mean_brier_score:.3f} is better. Log loss penalizes highly confident mistakes particularly strongly. Source-fitted temperature scaling therefore did not deliver uniformly reliable probabilities under acquisition shift, and classification improvement must not be presented as solved calibration. D0's mean NLL was also worse than each of the four classical comparators on their common endpoint sets.

P05 is the next planned phase: test the predeclared supervised-contrastive and paired-consistency successors against this frozen D0. Its exact no-fit expansion and source-only advancement checks must precede fitting. Those models must improve source pseudo-instrument performance without sacrificing worst-domain or within-source performance before any definitive held comparison.

## Figures

| Figure | View | Editable source | Vector export |
| --- | --- | --- | --- |
| F19: architecture and tensor flow | [HTML](../../plan/figures/html/F19_deep_architecture.html) | [TikZ](../../plan/figures/tikz/F19_deep_architecture.tex) | [PDF](../../plan/figures/pdf/F19_deep_architecture.pdf) |
| F20: source-only learning curves | [HTML](../../plan/figures/html/F20_learning_curves.html) | [TikZ](../../plan/figures/tikz/F20_learning_curves.tex) | [PDF](../../plan/figures/pdf/F20_learning_curves.pdf) |
| F48: D0-minus-classical held-domain effects | [HTML](../../plan/figures/html/F48_deep_classical_comparison.html) | [TikZ](../../plan/figures/tikz/F48_deep_classical_comparison.tex) | [PDF](../../plan/figures/pdf/F48_deep_classical_comparison.pdf) |

In F20, only fits still running contribute at later epochs, so changes in the curve can reflect which fits remain. The interquartile band describes variation across fits, not a confidence interval. The vertical line marks the median selected inner checkpoint epoch. In F48, points to the right of zero favour D0, points to the left favour the named classical comparator, and each horizontal interval resamples physical masters.

## Boundaries

The independent chemical evidence remains 69 physical masters, not 598 independent spectra. Results apply to the observed stations, analytes, substrates, and instruments under `PP-U-MIN`; they do not establish broad instrument independence or substrate superiority.
"""
    path.write_text(text)


def publish_p04(*, artifact_root: Path, project_root: Path) -> dict[str, Any]:
    context = execution_context(artifact_root=artifact_root, project_root=project_root)
    aggregation = _single_shard(
        context.execution_root / "final_aggregation", "P04_AGGREGATION_REPORT.json"
    )
    comparison_root = _single_shard(
        context.execution_root / "comparison", "P04_COMPARISON_REPORT.json"
    )
    aggregation_report = json.loads((aggregation / "P04_AGGREGATION_REPORT.json").read_text())
    comparison_report = json.loads((comparison_root / "P04_COMPARISON_REPORT.json").read_text())
    fit_status = pd.read_csv(aggregation / "fit_status.csv", low_memory=False)
    metrics = pd.read_csv(aggregation / "endpoint_metrics.csv", low_memory=False)
    coverage = pd.read_csv(aggregation / "endpoint_coverage.csv", low_memory=False)
    curves = pd.read_csv(aggregation / "learning_curve_summary.csv", low_memory=False)
    epochs = pd.read_csv(aggregation / "selected_epoch_summary.csv", low_memory=False)
    selections = pd.read_csv(aggregation / "selection_trace.csv", low_memory=False)
    comparison = pd.read_csv(comparison_root / "comparison_summary.csv", low_memory=False)
    bootstrap = pd.read_csv(comparison_root / "master_clustered_bootstrap.csv", low_memory=False)
    parity = pd.read_csv(comparison_root / "p13_uid_parity.csv", low_memory=False)
    p13_d0 = pd.read_csv(comparison_root / "p13_d0_substrate_performance.csv", low_memory=False)
    overall = _overall_performance(metrics, coverage)
    domains = _domain_performance(metrics)
    fit_summary, diagnostics, candidate_summary = _fit_summary(fit_status)
    selected_frequency = (
        selections[selections.selected]
        .groupby(["experiment_id", "candidate_id", "learning_rate", "weight_decay"], as_index=False)
        .size()
        .rename(columns={"size": "outer_context_count"})
    )
    parity_summary = pd.DataFrame(
        [
            {
                "primary_p04_contexts": len(parity),
                "p13_subset_pass_contexts": int(parity.each_p13_context_is_exact_p04_subset.sum()),
                "p13_union_exact_contexts": int(parity.p13_union_equals_p04_test.sum()),
                "interpretation": (
                    "Every P13 substrate-stratified held-test set is an exact subset of its "
                    "corresponding P04 held-instrument test set; union parity is reported separately. "
                    "This verifies test-view parity only: P04 fits all source substrates, whereas "
                    "P13 classical fits are substrate restricted."
                ),
            }
        ]
    )
    result_root = project_root / "results/p04_deep"
    tables_root = result_root / "tables"
    semantic_root = result_root / "semantic"
    tables_root.mkdir(parents=True, exist_ok=True)
    semantic_root.mkdir(parents=True, exist_ok=True)
    tables = {
        "overall_performance.csv": overall,
        "domain_performance.csv": domains,
        "fit_summary.csv": fit_summary,
        "training_diagnostics.csv": diagnostics,
        "candidate_summary.csv": candidate_summary,
        "selected_candidate_frequency.csv": selected_frequency,
        "selected_epoch_summary.csv": epochs,
        "learning_curve_summary.csv": curves,
        "comparison_summary.csv": comparison,
        "comparison_domain_effects.csv": bootstrap,
        "p13_uid_parity_summary.csv": parity_summary,
        "p13_d0_substrate_performance.csv": p13_d0,
    }
    for name, frame in tables.items():
        frame.to_csv(
            tables_root / name,
            index=False,
            lineterminator="\n",
            float_format="%.12g",
        )
    figure_manifest = generate_p04_figures(
        learning_curves=curves,
        selected_epochs=epochs,
        bootstrap_effects=bootstrap,
        results_root=result_root,
        plan_root=project_root / "plan",
    )
    figure_manifest.to_csv(
        result_root / "p04_figure_manifest.csv", index=False, lineterminator="\n"
    )
    _write_report(
        path=result_root / "P04_RESULTS.md",
        overall=overall,
        comparison_report=comparison_report,
        comparison=comparison,
        bootstrap=bootstrap,
        diagnostics=diagnostics,
        epochs=epochs,
        fit_summary=fit_summary,
        p13_d0=p13_d0,
    )
    (result_root / "README.md").write_text(
        "# P04 compact deep evidence\n\n"
        "Disclosure-safe aggregate D0 training, held-instrument comparison, and figure data. "
        "Row predictions, master IDs, fold identities, and checkpoints remain outside the public tree.\n\n"
        "[Read the results and figure guide](P04_RESULTS.md) · "
        "[Open the project dashboard](../../plan/index.html) · "
        "[Completion audit](../../plan/P04_COMPLETION_AUDIT.md)\n"
    )
    public_files = sorted(
        path
        for path in result_root.rglob("*")
        if path.is_file() and path.name != "release_manifest.json"
    )
    release = {
        "schema_version": "nato-sers-p04-public-release-v1",
        "p04_run_id": context.run_id,
        "p04_execution_state_sha256": context.protected_state_sha256,
        "p04_aggregation_state_sha256": aggregation_report["aggregation_state_sha256"],
        "p04_comparison_state_sha256": comparison_report["comparison_state_sha256"],
        "privacy": {
            "row_predictions_published": False,
            "master_ids_published": False,
            "fold_identities_published": False,
            "checkpoints_published": False,
        },
        "files": {
            path.relative_to(project_root).as_posix(): {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in public_files
        },
    }
    (result_root / "release_manifest.json").write_bytes(canonical_json_bytes(release, pretty=True))
    return {
        "status": "pass",
        "run_id": context.run_id,
        "primary_conclusion": comparison_report["primary_conclusion"],
        "public_file_count": len(public_files) + 1,
        "figure_count": len(figure_manifest),
    }
