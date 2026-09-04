#!/usr/bin/env python3
"""Validate aggregate-only P04 publication artifacts without private inputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from atlas_sers.governance.canonical import sha256_file

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results/p04_deep"
PLAN = PROJECT / "plan"
PROHIBITED_COLUMNS = {
    "observation_uid",
    "master_sample_id",
    "fit_id",
    "context_id",
    "outer_repeat",
    "outer_fold",
}


def main() -> int:
    errors: list[str] = []
    required = {
        "README.md",
        "P04_RESULTS.md",
        "release_manifest.json",
        "p04_figure_manifest.csv",
        "tables/overall_performance.csv",
        "tables/domain_performance.csv",
        "tables/fit_summary.csv",
        "tables/training_diagnostics.csv",
        "tables/candidate_summary.csv",
        "tables/selected_candidate_frequency.csv",
        "tables/selected_epoch_summary.csv",
        "tables/learning_curve_summary.csv",
        "tables/comparison_summary.csv",
        "tables/comparison_domain_effects.csv",
        "tables/p13_uid_parity_summary.csv",
        "tables/p13_d0_substrate_performance.csv",
    }
    for name in required:
        if not (RESULTS / name).is_file():
            errors.append(f"missing P04 release file: {name}")
    if errors:
        print("P04 public release: FAIL\n- " + "\n- ".join(errors))
        return 1
    release = json.loads((RESULTS / "release_manifest.json").read_text())
    if any(release["privacy"].values()):
        errors.append("P04 release privacy flags expose protected artifacts")
    for relative, descriptor in release["files"].items():
        path = PROJECT / relative
        if not path.is_file() or sha256_file(path) != descriptor["sha256"]:
            errors.append(f"P04 release hash mismatch: {relative}")
    for path in sorted((RESULTS / "tables").glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        overlap = PROHIBITED_COLUMNS & set(frame)
        if overlap:
            errors.append(f"P04 table {path.name} exposes columns: {sorted(overlap)}")
        if re.search(r"\bOBS-[0-9a-f]{20}\b", path.read_text()):
            errors.append(f"P04 table {path.name} exposes an observation UID")
    overall = pd.read_csv(RESULTS / "tables/overall_performance.csv")
    if set(zip(overall.experiment_id, overall.aggregation_id, strict=True)) != {
        ("EXP-N00-DEV", "M01"),
        ("EXP-N00-DEV", "M06"),
        ("EXP-N00-T3", "M01"),
        ("EXP-N00-T3", "M06"),
    }:
        errors.append("P04 overall table lacks its four registered endpoint summaries")
    if not overall.coverage.eq(1.0).all():
        errors.append("P04 public endpoint coverage is not complete")
    fit_summary = pd.read_csv(RESULTS / "tables/fit_summary.csv")
    if int(fit_summary.planned_or_terminal_fits.sum()) != 16_458:
        errors.append("P04 public fit count does not reconcile to 16,458")
    inner = fit_summary[fit_summary.stage.eq("inner_selection")]
    final = fit_summary[fit_summary.stage.eq("final_selected_refit")]
    if (inner.completion_fraction < 0.95).any():
        errors.append("P04 inner-fit completion is below the locked G2 threshold")
    if not final.complete_fits.eq(final.planned_or_terminal_fits).all():
        errors.append("P04 final-refit completion is incomplete")
    p13 = pd.read_csv(RESULTS / "tables/p13_d0_substrate_performance.csv")
    if len(p13) != 15 or int(p13.support_tier.eq("confirmatory").sum()) != 13:
        errors.append("P04/P13 substrate-view coverage is not 15 total and 13 confirmatory")
    if not p13.minimum_outer_repeat_predictions.eq(5).all():
        errors.append("P04/P13 substrate views lack five outer-repeat predictions")
    if not p13.held_bootstrap_resamples.eq(10_000).all():
        errors.append("P04/P13 held-recovery intervals lack 10,000 resamples")
    if not p13.portability_decision.eq("not_estimable_without_matched_source_loss").all():
        errors.append("P04 release overstates the P13 dual-margin portability decision")
    if not p13.training_scope.eq("all_source_substrates_within_station").all() or not (
        p13.classical_training_scope.eq("source_rows_of_same_substrate_family").all()
    ):
        errors.append("P04/P13 release fails to disclose differing training substrate scopes")
    figures = pd.read_csv(RESULTS / "p04_figure_manifest.csv")
    if set(figures.figure_id) != {"F19", "F20", "F48"}:
        errors.append("P04 figure manifest does not contain F19, F20, and F48")
    for row in figures.itertuples(index=False):
        stem = {
            "F19": "F19_deep_architecture",
            "F20": "F20_learning_curves",
            "F48": "F48_deep_classical_comparison",
        }[row.figure_id]
        data = PLAN / "figures/data" / f"{stem}.csv"
        outputs = {
            "tikz_sha256": PLAN / "figures/tikz" / f"{stem}.tex",
            "pdf_sha256": PLAN / "figures/pdf" / f"{stem}.pdf",
            "png_sha256": PLAN / "figures/png" / f"{stem}.png",
            "html_sha256": PLAN / "figures/html" / f"{stem}.html",
        }
        if not data.is_file() or sha256_file(data) != row.semantic_sha256:
            errors.append(f"{row.figure_id} semantic data hash mismatch")
        for field, path in outputs.items():
            if not path.is_file() or sha256_file(path) != getattr(row, field):
                errors.append(f"{row.figure_id} {field} mismatch")
        if data.is_file():
            digest = sha256_file(data)
            tex = outputs["tikz_sha256"]
            html = outputs["html_sha256"]
            if tex.is_file() and digest not in tex.read_text():
                errors.append(f"{row.figure_id} TikZ semantic parity mismatch")
            if html.is_file() and digest not in html.read_text():
                errors.append(f"{row.figure_id} HTML semantic parity mismatch")
    if errors:
        print("P04 public release: FAIL\n- " + "\n- ".join(errors))
        return 1
    print(f"P04 public release: PASS ({len(release['files']) + 1} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
