#!/usr/bin/env python3
"""Validate the aggregate-only P13 public release without private source data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results/p13_portability"
PLAN = PROJECT / "plan"

EXPECTED_TABLE_ROWS = {
    "class_cell_claims.csv": 102,
    "crossover_effects.csv": 238,
    "domain_claims.csv": 34,
    "domain_metrics.csv": 336,
    "failure_table.csv": 210,
    "field_log_results.csv": 35,
    "interval_table.csv": 570,
    "preprocessing_sensitivity.csv": 181,
    "procedure_comparison.csv": 7,
    "substrate_claims.csv": 4,
}
PROHIBITED_COLUMNS = {
    "observation_uid",
    "source_observation_uid",
    "master_sample_id",
    "source_logical_id",
    "source_primary_reference",
}
PROHIBITED_BYTES = (
    bytes((47, 104, 111, 109, 101, 47)),
    bytes((92, 117, 115, 101, 114, 115, 92)),
    b"github_pat_",
    b"ghp_",
    b"gho_",
)
FIGURES = {
    "F45": "F45_substrate_recoverability",
    "F46": "F46_substrate_instrument_crossover",
    "F47": "F47_recorded_detection_agreement",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    errors: list[str] = []
    release_path = RESULTS / "release_manifest.json"
    figure_manifest_path = RESULTS / "p13_figure_manifest.csv"
    report_path = RESULTS / "P13_RESULTS.md"
    readme_path = RESULTS / "README.md"
    for path in (release_path, figure_manifest_path, report_path, readme_path):
        if not path.is_file():
            errors.append(f"missing required release file: {path.relative_to(PROJECT)}")
    if errors:
        return _finish(errors)

    release = json.loads(release_path.read_text())
    if release.get("protocol_version") != "nato-sers-p13-v1-locked":
        errors.append("unexpected P13 protocol version")
    if release.get("run_id") != "P13-3d21aa17c7d6cd750ca9d286":
        errors.append("unexpected P13 execution run")

    public_hashes = release.get("public_files", {})
    for name, expected_rows in EXPECTED_TABLE_ROWS.items():
        path = RESULTS / "tables" / name
        if not path.is_file():
            errors.append(f"missing P13 table: {name}")
            continue
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != expected_rows:
            errors.append(f"{name} has {len(frame)} rows; expected {expected_rows}")
        if PROHIBITED_COLUMNS & set(frame):
            errors.append(f"{name} exposes a private identifier column")
        if public_hashes.get(f"tables/{name}") != sha256(path):
            errors.append(f"release-manifest hash mismatch: {name}")

    claims_path = RESULTS / "tables/domain_claims.csv"
    metrics_path = RESULTS / "tables/domain_metrics.csv"
    intervals_path = RESULTS / "tables/interval_table.csv"
    if claims_path.is_file():
        claims = pd.read_csv(claims_path)
        expected_states = {
            "supports_portability",
            "inferior_portability",
            "inconclusive",
            "unsupported_by_design",
            "unavailable_terminal_failure",
        }
        if not set(claims.completion_state.astype(str)) <= expected_states:
            errors.append("domain claims contain an unregistered completion state")
        if len(claims[claims.support_tier.astype(str).eq("confirmatory")]) != 13:
            errors.append("domain claims do not retain all 13 confirmatory domains")
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        primary = metrics[
            metrics.policy_id.astype(str).eq("PP-U-MIN")
            & metrics.procedure_id.astype(str).eq("C-SELECTED")
            & metrics.support_tier.astype(str).eq("confirmatory")
        ]
        state_counts = primary.bounded_state.value_counts().to_dict()
        if state_counts != {
            "unavailable_terminal_failure": 6,
            "inconclusive": 5,
            "inferior_portability": 2,
        }:
            errors.append(f"primary P13 bounded-state counts changed: {state_counts}")
    if intervals_path.is_file():
        intervals = pd.read_csv(intervals_path)
        if not intervals.bootstrap_resamples.eq(10_000).all():
            errors.append("an interval does not use the locked 10,000 resamples")

    figures = pd.read_csv(figure_manifest_path)
    if set(figures.figure_id.astype(str)) != set(FIGURES):
        errors.append("figure manifest does not contain exactly F45--F47")
    for row in figures.itertuples(index=False):
        stem = FIGURES.get(str(row.figure_id))
        if stem is None:
            continue
        semantic_result = PROJECT / str(row.semantic_path)
        semantic_plan = PLAN / "figures/data" / f"{stem}.csv"
        if not semantic_result.is_file() or not semantic_plan.is_file():
            errors.append(f"{row.figure_id} semantic table is missing")
            continue
        digest = sha256(semantic_result)
        if (
            digest != str(row.semantic_sha256)
            or semantic_result.read_bytes() != semantic_plan.read_bytes()
        ):
            errors.append(f"{row.figure_id} semantic hash/copy mismatch")
        for kind, suffix, manifest_field in (
            ("tikz", ".tex", "tikz_sha256"),
            ("pdf", ".pdf", "pdf_sha256"),
            ("png", ".png", "png_sha256"),
            ("html", ".html", "html_sha256"),
        ):
            path = PLAN / "figures" / kind / f"{stem}{suffix}"
            if not path.is_file():
                errors.append(f"missing {row.figure_id} {kind} artifact")
                continue
            if sha256(path) != str(getattr(row, manifest_field)):
                errors.append(f"{row.figure_id} {kind} manifest hash mismatch")
        tex_path = PLAN / "figures/tikz" / f"{stem}.tex"
        html_path = PLAN / "figures/html" / f"{stem}.html"
        if tex_path.is_file():
            tex = tex_path.read_text(errors="ignore")
            if digest not in tex or "\\includegraphics" in tex:
                errors.append(f"{row.figure_id} TikZ is not native/hash-matched")
        if html_path.is_file():
            html = html_path.read_text(errors="ignore").lower()
            if digest not in html or "<html" not in html or "</html>" not in html:
                errors.append(f"{row.figure_id} HTML is not standalone/hash-matched")
            if "cdn.plot.ly" in html or "<script src=\"http" in html:
                errors.append(f"{row.figure_id} HTML requires an external CDN")

    for path in RESULTS.rglob("*"):
        if path.is_file() and any(token in path.read_bytes().lower() for token in PROHIBITED_BYTES):
            errors.append(f"private path or credential pattern in {path.relative_to(PROJECT)}")
    return _finish(errors)


def _finish(errors: list[str]) -> int:
    if errors:
        print("P13 public release validation: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print("P13 public release validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
