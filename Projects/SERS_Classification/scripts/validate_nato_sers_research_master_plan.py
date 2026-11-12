#!/usr/bin/env python3
"""Validate and freeze the NATO SERS research master plan v1."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
WORKSPACE = REPOSITORY / "Workspace" / "nato_sers_research_master_plan_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(name: str) -> list[dict[str, str]]:
    with (WORKSPACE / "registries" / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def standalone_html(path: Path) -> bool:
    text = path.read_text(errors="ignore")
    lower = text.lower()
    return (
        "<html" in lower[:2000]
        and "</html>" in lower[-2000:]
        and not re.search(r"<script[^>]+src\s*=\s*[\"']https?://", lower)
        and "cdn.plot.ly" not in lower
    )


def pdf_has_raster(path: Path) -> bool:
    result = subprocess.run(
        ["pdfimages", "-list", str(path)], capture_output=True, text=True, check=True
    )
    return any(re.match(r"\s*\d+\s+\d+\s+", line) for line in result.stdout.splitlines())


def compile_native_tikz(source: Path, shared_style: Path) -> tuple[bool, str, bool]:
    with tempfile.TemporaryDirectory(prefix="nato_sers_plan_tex_") as directory:
        temp = Path(directory)
        shutil.copy2(source, temp / source.name)
        shutil.copy2(shared_style, temp / shared_style.name)
        result = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                source.name,
            ],
            cwd=temp,
            capture_output=True,
            text=True,
        )
        pdf = temp / f"{source.stem}.pdf"
        raster = pdf_has_raster(pdf) if result.returncode == 0 and pdf.is_file() else True
        return result.returncode == 0 and pdf.is_file(), result.stdout[-4000:], raster


def main() -> None:
    checks: dict[str, bool] = {}
    diagnostics: dict[str, object] = {}

    master_plan = WORKSPACE / "MASTER_PLAN.md"
    style_plan = WORKSPACE / "FIGURE_STYLE_AND_REGENERATION.md"
    plan_text = master_plan.read_text()
    checks["master_plan_is_extremely_detailed"] = (
        master_plan.stat().st_size > 50_000
        and len(plan_text.splitlines()) >= 900
        and all(f"p{index:02d}" in plan_text.lower() for index in range(13))
        and all(gate in plan_text.lower() for gate in ["g0", "g1", "g2", "g3", "g4", "g5", "g6"])
    )
    required_concepts = [
        "unseen-instrument",
        "master_sample_id",
        "supervised contrastive",
        "paired-view consistency",
        "conditional CORAL",
        "domain-adversarial",
        "hierarchical bootstrap",
        "open-set",
        "native TikZ",
        "standalone self-contained HTML",
        "prohibited claim",
        "Route A",
        "Route B",
        "Route C",
    ]
    checks["all_required_scientific_concepts_are_explicit"] = all(
        concept.lower() in plan_text.lower() for concept in required_concepts
    )
    checks["figure_style_contract_is_detailed"] = (
        style_plan.stat().st_size > 8_000
        and "includegraphics" in style_plan.read_text()
        and "semantic parity" in style_plan.read_text().lower()
    )

    phases = read_csv("phase_registry.csv")
    tasks = read_csv("task_registry.csv")
    metrics = read_csv("metric_registry.csv")
    experiments = read_csv("experiment_registry.csv")
    figures = read_csv("figure_registry.csv")
    gates = read_csv("decision_gate_registry.csv")
    checks["registry_counts_are_locked"] = (
        len(phases) == 13
        and len(tasks) == 12
        and len(metrics) == 25
        and len(experiments) == 39
        and len(figures) == 35
        and len(gates) == 8
    )
    registry_id_fields = [
        (phases, "phase_id"),
        (tasks, "task_id"),
        (metrics, "metric_id"),
        (experiments, "experiment_id"),
        (figures, "figure_id"),
        (gates, "gate_id"),
    ]
    checks["all_registry_ids_are_unique_and_nonempty"] = all(
        all(row[field] for row in rows) and len({row[field] for row in rows}) == len(rows)
        for rows, field in registry_id_fields
    )
    phase_ids = {row["phase_id"] for row in phases}
    checks["experiment_phase_references_are_valid"] = all(
        row["phase"] in phase_ids for row in experiments
    )
    checks["scope_labels_are_valid"] = all(
        row.get("scope") in {"P", "S", "E", "primary", "secondary"}
        for rows in [phases, tasks, metrics, experiments, figures, gates]
        for row in rows
    )

    contract_names = [
        "research_contract.json",
        "split_contract.json",
        "hyperparameter_registry.json",
        "result_schema.json",
        "figure_contract.json",
        "compute_budget.json",
    ]
    contracts = {name: json.loads((WORKSPACE / "contracts" / name).read_text()) for name in contract_names}
    research = contracts["research_contract.json"]
    split = contracts["split_contract.json"]
    checks["contract_core_counts_and_endpoint_are_locked"] = (
        research["population_counts"]["primary_spectra"] == 598
        and research["population_counts"]["physical_masters"] == 69
        and research["primary_representation"]["axis_start_cm1"] == 400
        and research["primary_representation"]["axis_end_cm1"] == 1800
        and research["primary_representation"]["n_features"] == 1401
        and research["primary_endpoint"]["eligible_domains"] == 13
        and research["minimum_meaningful_effect"] == 0.03
    )
    checks["split_contract_is_exact_and_separates_regimes"] = (
        split["outer_repeat_seeds"] == [20260805, 20260817, 20260829, 20260910, 20260922]
        and split["neural_training_seeds"] == [20260805, 20260817, 20260829]
        and split["outer_folds_per_station"] == 4
        and len(split["primary_domain_eligibility"]["domains"]) == 13
        and len(split["exploratory_low_support_domains"]) == 4
        and len(split["tasks"]["T4_OPEN"]["held_tasks"]) == 8
        and set(split["adaptation_regimes"]) == {"zero_shot", "UDA", "paired_calibration", "few_shot"}
    )
    input_failures: list[str] = []
    for item in research["authoritative_inputs"]:
        path = REPOSITORY / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            input_failures.append(item["path"])
    checks["authoritative_input_hashes_match"] = not input_failures
    diagnostics["input_hash_failures"] = input_failures
    restart_validation = json.loads(
        (REPOSITORY / "Workspace/nato_sers_raw_reexploration_v1/FINAL_VALIDATION_REPORT.json").read_text()
    )
    checks["completed_restart_validation_is_pass"] = restart_validation.get("status") == "pass"

    figure_ids = {row["figure_id"] for row in figures}
    checks["figure_registry_is_complete_and_paired"] = (
        figure_ids == {f"F{index:02d}" for index in range(35)}
        and all(row["tikz_path"].endswith(".tex") and row["html_path"].endswith(".html") for row in figures)
        and len({row["tikz_path"] for row in figures}) == 35
        and len({row["html_path"] for row in figures}) == 35
        and sum(row["status"] == "complete" for row in figures) == 2
        and sum(row["status"] == "planned" for row in figures) == 33
    )
    completed = [row for row in figures if row["status"] == "complete"]
    complete_figure_ok = True
    parity_details: dict[str, object] = {}
    style_source = WORKSPACE / "figures/tikz/sers_plan_style.tex"
    for row in completed:
        tikz = WORKSPACE / row["tikz_path"]
        html = WORKSPACE / row["html_path"]
        base = tikz.stem
        data = WORKSPACE / "figures" / "data" / f"{base}.csv"
        pdf = WORKSPACE / "figures" / "pdf" / f"{base}.pdf"
        log = WORKSPACE / "figures" / "logs" / f"{base}.pdflatex.log"
        paths_exist = all(path.is_file() for path in [tikz, html, data, pdf, log])
        data_hash = sha256(data) if data.is_file() else ""
        tikz_text = tikz.read_text(errors="ignore") if tikz.is_file() else ""
        html_text = html.read_text(errors="ignore") if html.is_file() else ""
        parity = data_hash in tikz_text and data_hash in html_text
        native = "\\includegraphics" not in tikz_text
        html_ok = standalone_html(html)
        compile_ok, compile_tail, compiled_raster = compile_native_tikz(tikz, style_source)
        stored_pdf_vector = pdf.is_file() and not pdf_has_raster(pdf)
        item_ok = paths_exist and parity and native and html_ok and compile_ok and not compiled_raster and stored_pdf_vector
        complete_figure_ok &= item_ok
        parity_details[row["figure_id"]] = {
            "paths_exist": paths_exist,
            "data_sha256": data_hash,
            "hash_in_tikz_and_html": parity,
            "native_tikz_no_includegraphics": native,
            "standalone_html": html_ok,
            "fresh_compile_pass": compile_ok,
            "fresh_pdf_has_raster": compiled_raster,
            "stored_pdf_has_raster": not stored_pdf_vector,
            "compile_tail_if_failed": "" if compile_ok else compile_tail,
        }
    checks["completed_figures_pass_native_tikz_html_and_vector_parity"] = complete_figure_ok
    diagnostics["completed_figure_audit"] = parity_details
    raster_files = [
        str(path.relative_to(WORKSPACE))
        for path in WORKSPACE.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}
    ]
    checks["plan_workspace_contains_no_raster_figure_files"] = not raster_files
    diagnostics["raster_files"] = raster_files
    checks["plan_index_is_standalone_and_links_core_artifacts"] = (
        standalone_html(WORKSPACE / "index.html")
        and all(
            token in (WORKSPACE / "index.html").read_text()
            for token in ["MASTER_PLAN.md", "experiment_registry.csv", "F00_research_workflow", "F01_decision_gates"]
        )
    )

    result_schema = contracts["result_schema.json"]
    required_prediction_fields = {
        "experiment_id", "run_id", "task_id", "information_regime", "observation_uid",
        "master_sample_id", "instrument", "target_analyte", "probabilities", "run_status"
    }
    checks["result_schema_preserves_prediction_provenance"] = required_prediction_fields.issubset(
        set(result_schema["required"])
    )
    figure_contract = contracts["figure_contract.json"]
    checks["figure_contract_prohibits_raster_wrappers_and_cdns"] = (
        "includegraphics" in figure_contract["native_tikz_rules"]["prohibited"]
        and figure_contract["native_tikz_rules"]["vector_only"] is True
        and figure_contract["html_rules"]["external_cdn"] is False
        and len(figure_contract["semantic_parity"]) >= 8
    )

    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "diagnostics": diagnostics,
        "counts": {
            "master_plan_lines": len(plan_text.splitlines()),
            "phases": len(phases),
            "tasks": len(tasks),
            "metrics": len(metrics),
            "experiments": len(experiments),
            "figures_registered": len(figures),
            "figures_complete_now": len(completed),
            "decision_gates": len(gates),
        },
        "workspace": str(WORKSPACE),
    }

    excluded = {WORKSPACE / "FINAL_VALIDATION_REPORT.json", WORKSPACE / "FINAL_ARTIFACT_HASHES.json"}
    artifact_paths = [
        path for path in WORKSPACE.rglob("*")
        if path.is_file() and path not in excluded
    ] + [Path(__file__).resolve()]
    artifact_paths = sorted({path.resolve() for path in artifact_paths})
    hashes = {str(path.relative_to(REPOSITORY)): sha256(path) for path in artifact_paths}
    (WORKSPACE / "FINAL_ARTIFACT_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )
    report["hashed_artifacts"] = len(hashes)
    (WORKSPACE / "FINAL_VALIDATION_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
