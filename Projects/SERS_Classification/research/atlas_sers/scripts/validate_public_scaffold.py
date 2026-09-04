#!/usr/bin/env python3
"""Validate the public NATO SERS research package without loading source data."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PLAN = PROJECT / "plan"
sys.path.insert(0, str(PROJECT / "src"))

from atlas_sers.governance.registries import load_governance, validate_governance  # noqa: E402

POSIX_USER_PREFIX = bytes((47, 104, 111, 109, 101, 47))
WINDOWS_USER_PREFIX = bytes((92, 117, 115, 101, 114, 115, 92))

PROHIBITED_ARTIFACT_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".prb",
    ".pt",
    ".pth",
    ".spa",
    ".spc",
}
BUILD_SUFFIXES = {".aux", ".fdb_latexmk", ".fls", ".log", ".out", ".synctex.gz"}
IGNORED_LOCAL_PARTS = {".pytest_cache", ".ruff_cache", "__pycache__", "build", "dist"}

REQUIRED_FILES = {
    "README.md",
    "PUBLICATION_POLICY.md",
    "CONTRIBUTING.md",
    "REPOSITORY_ARCHITECTURE.md",
    "pyproject.toml",
    "data/README.md",
    "artifacts/README.md",
    "plan/MASTER_PLAN.md",
    "plan/RESEARCH_QUESTION_MAP.md",
    "plan/P00_EXECUTION.md",
    "plan/P01_EXECUTION.md",
    "plan/P02_EXECUTION.md",
    "plan/P03_HANDOFF.md",
    "plan/P03_DECISION_MEMO.md",
    "plan/P03_EXECUTION.md",
    "plan/P03_COMPLETION_AUDIT.md",
    "plan/P13_FREEZE_MEMO.md",
    "plan/P13_PROTOCOL.md",
    "plan/FIGURE_STYLE_AND_REGENERATION.md",
    "plan/index.html",
    "plan/contracts/research_contract.json",
    "plan/contracts/split_contract.json",
    "plan/contracts/hyperparameter_registry.json",
    "plan/contracts/result_schema.json",
    "plan/contracts/figure_contract.json",
    "plan/contracts/compute_budget.json",
    "plan/contracts/preprocessing_policy_contract.json",
    "plan/contracts/p00_governance_contract.json",
    "plan/contracts/p00_validation_schema.json",
    "plan/contracts/p01_governance_contract.json",
    "plan/contracts/p01_validation_schema.json",
    "plan/contracts/p02_governance_contract.json",
    "plan/contracts/p02_validation_schema.json",
    "plan/contracts/p03_governance_contract.json",
    "plan/registries/model_registry.csv",
    "plan/registries/research_question_registry.csv",
    "plan/registries/preprocessing_policy_registry.csv",
    "plan/registries/artifact_registry.csv",
    "plan/registries/deviations.csv",
    "plan/registries/p13_decision_registry.csv",
    "plan/registries/p13_domain_support_registry.csv",
    "plan/registries/p13_experiment_registry.csv",
    "plan/registries/p13_figure_registry.csv",
    "plan/registries/p13_metric_registry.csv",
    "plan/registries/p13_phase_registry.csv",
    "plan/registries/p13_research_question_registry.csv",
    "plan/registries/p13_split_registry.csv",
    "plan/registries/p13_crossover_support_registry.csv",
    "plan/registries/p13_support_freeze_summary.json",
    "plan/registries/p13_support_policy_registry.csv",
    "plan/registries/public_release_registry.csv",
    "scripts/run_p00.py",
    "scripts/run_p01.py",
    "scripts/run_p02.py",
    "scripts/run_p03.py",
    "scripts/publish_p02_figures.py",
    "scripts/build_p13_support_freeze.py",
}

REGISTRY_COUNTS = {
    "phase_registry.csv": 13,
    "research_question_registry.csv": 8,
    "preprocessing_policy_registry.csv": 6,
    "task_registry.csv": 15,
    "metric_registry.csv": 31,
    "experiment_registry.csv": 46,
    "figure_registry.csv": 44,
    "model_registry.csv": 41,
    "artifact_registry.csv": 46,
    "decision_gate_registry.csv": 15,
    "deviations.csv": 1,
    "p13_decision_registry.csv": 16,
    "p13_domain_support_registry.csv": 34,
    "p13_experiment_registry.csv": 7,
    "p13_figure_registry.csv": 4,
    "p13_metric_registry.csv": 13,
    "p13_phase_registry.csv": 1,
    "p13_research_question_registry.csv": 1,
    "p13_split_registry.csv": 3,
    "p13_crossover_support_registry.csv": 34,
    "p13_support_policy_registry.csv": 10,
    "public_release_registry.csv": 2,
}


def relative(path: Path) -> str:
    return path.relative_to(PROJECT).as_posix()


def all_files() -> list[Path]:
    return sorted(
        path
        for path in PROJECT.rglob("*")
        if path.is_file()
        and not IGNORED_LOCAL_PARTS.intersection(path.relative_to(PROJECT).parts)
        and not any(part.endswith(".egg-info") for part in path.relative_to(PROJECT).parts)
    )


def validate_required_files(errors: list[str]) -> None:
    missing = sorted(name for name in REQUIRED_FILES if not (PROJECT / name).is_file())
    errors.extend(f"missing required file: {name}" for name in missing)


def validate_publication_boundary(errors: list[str]) -> None:
    for path in all_files():
        content = path.read_bytes().lower()
        if POSIX_USER_PREFIX in content or WINDOWS_USER_PREFIX in content:
            errors.append(f"absolute workstation path found: {relative(path)}")
        if path.suffix.lower() in PROHIBITED_ARTIFACT_SUFFIXES:
            errors.append(f"private artifact format found: {relative(path)}")
        if any(path.name.lower().endswith(suffix) for suffix in BUILD_SUFFIXES):
            errors.append(f"build by-product found: {relative(path)}")
        size_limit = 10_000_000 if path.suffix.lower() == ".html" else 5_000_000
        if path.stat().st_size > size_limit:
            errors.append(
                f"unexpected file larger than {size_limit // 1_000_000} MB: {relative(path)}"
            )
        if path.is_symlink():
            errors.append(f"symbolic links are prohibited: {relative(path)}")

    for guarded_dir in (PROJECT / "data", PROJECT / "artifacts"):
        unexpected = [
            path for path in guarded_dir.rglob("*") if path.is_file() and path.name != "README.md"
        ]
        errors.extend(
            f"guarded directory contains content: {relative(path)}" for path in unexpected
        )


def validate_contracts(errors: list[str]) -> None:
    contracts: dict[str, object] = {}
    for path in sorted((PLAN / "contracts").glob("*.json")):
        try:
            contracts[path.name] = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON {relative(path)}: {exc}")

    research = contracts.get("research_contract.json")
    if not isinstance(research, dict):
        return
    protocol = str(research.get("protocol_version", ""))
    if not protocol.startswith(("atlas-sers-", "nato-sers-")):
        errors.append("research protocol does not use a recognized NATO SERS namespace")
    inputs = research.get("authoritative_inputs", [])
    if not isinstance(inputs, list) or not inputs:
        errors.append("research contract has no authoritative input declarations")
        return
    for item in inputs:
        path = item.get("path", "") if isinstance(item, dict) else ""
        if not str(path).startswith(("${ATLAS_PRIVATE_ROOT}/", "${NATO_SERS_PRIVATE_ROOT}/")):
            errors.append(f"authoritative input is not private-root relative: {path}")


def validate_registries(errors: list[str]) -> None:
    for name, expected in REGISTRY_COUNTS.items():
        path = PLAN / "registries" / name
        if not path.is_file():
            errors.append(f"missing registry: {relative(path)}")
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected:
            errors.append(f"{name} has {len(rows)} rows; expected {expected}")

    try:
        report = validate_governance(load_governance(PLAN))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"governance validation could not run: {type(exc).__name__}: {exc}")
        return
    if report["status"] != "pass":
        errors.extend(f"governance: {message}" for message in report["errors"])


def validate_p13_freeze(errors: list[str]) -> None:
    registry_dir = PLAN / "registries"
    try:
        with (registry_dir / "p13_decision_registry.csv").open(newline="") as handle:
            decisions = list(csv.DictReader(handle))
        with (registry_dir / "p13_support_policy_registry.csv").open(newline="") as handle:
            support_policies = list(csv.DictReader(handle))
        domain_path = registry_dir / "p13_domain_support_registry.csv"
        crossover_path = registry_dir / "p13_crossover_support_registry.csv"
        with domain_path.open(newline="") as handle:
            domains = list(csv.DictReader(handle))
        with crossover_path.open(newline="") as handle:
            crossovers = list(csv.DictReader(handle))
        summary = json.loads(
            (registry_dir / "p13_support_freeze_summary.json").read_text()
        )
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"P13 freeze validation could not run: {type(exc).__name__}: {exc}")
        return

    if {row["status"] for row in decisions} != {"locked"}:
        errors.append("P13 decision registry contains an unlocked decision")
    if {row["status"] for row in support_policies} != {"locked"}:
        errors.append("P13 support-policy registry contains an unlocked rule")

    expected_domain_counts = {
        "confirmatory": 13,
        "exploratory_low_support": 3,
        "unsupported_by_design": 18,
    }
    expected_crossover_counts = {
        "confirmatory": 8,
        "exploratory_low_support": 7,
        "descriptive_singleton": 19,
    }
    domain_counts = {
        tier: sum(row["support_tier"] == tier for row in domains)
        for tier in {row["support_tier"] for row in domains}
    }
    crossover_counts = {
        tier: sum(row["support_tier"] == tier for row in crossovers)
        for tier in {row["support_tier"] for row in crossovers}
    }
    if domain_counts != expected_domain_counts:
        errors.append(f"P13 domain support tiers changed: {domain_counts}")
    if crossover_counts != expected_crossover_counts:
        errors.append(f"P13 crossover support tiers changed: {crossover_counts}")

    hashes = summary.get("registry_hashes", {})
    expected_hashes = {
        "p13_domain_support_registry_sha256": hashlib.sha256(
            domain_path.read_bytes()
        ).hexdigest(),
        "p13_crossover_support_registry_sha256": hashlib.sha256(
            crossover_path.read_bytes()
        ).hexdigest(),
    }
    if hashes != expected_hashes:
        errors.append("P13 support registry hashes do not match the freeze summary")


def validate_figures(errors: list[str]) -> None:
    html_paths = sorted((PLAN / "figures" / "html").glob("*.html"))
    tikz_paths = sorted(
        path
        for path in (PLAN / "figures" / "tikz").glob("*.tex")
        if path.name != "sers_plan_style.tex"
    )

    html_stems = {path.stem for path in html_paths}
    tikz_stems = {path.stem for path in tikz_paths}
    for stem in sorted(html_stems | tikz_stems):
        required = [
            PLAN / "figures" / "html" / f"{stem}.html",
            PLAN / "figures" / "tikz" / f"{stem}.tex",
            PLAN / "figures" / "pdf" / f"{stem}.pdf",
        ]
        data_candidates = [
            PLAN / "figures" / "data" / f"{stem}.csv",
            PLAN / "figures" / "data" / f"{stem}.json",
        ]
        for expected in required:
            if not expected.is_file():
                errors.append(f"published figure artifact missing: {relative(expected)}")
        if not any(path.is_file() for path in data_candidates):
            errors.append(
                "published figure semantic data missing: "
                + " or ".join(relative(path) for path in data_candidates)
            )

    external_script = re.compile(r"<script[^>]+src\s*=\s*['\"]https?://", re.IGNORECASE)
    for path in [PLAN / "index.html", *html_paths]:
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        if "<html" not in text[:2000].lower() or "</html>" not in text[-2000:].lower():
            errors.append(f"HTML is not standalone: {relative(path)}")
        if external_script.search(text) or "cdn.plot.ly" in text.lower():
            errors.append(f"HTML uses an external script/CDN: {relative(path)}")

    for path in tikz_paths:
        if "\\includegraphics" in path.read_text(errors="ignore"):
            errors.append(f"TikZ source wraps a raster or external figure: {relative(path)}")

    for tikz in (path for path in tikz_paths if path.name.startswith("F")):
        stem = tikz.stem
        data_candidates = [
            PLAN / "figures" / "data" / f"{stem}.csv",
            PLAN / "figures" / "data" / f"{stem}.json",
        ]
        data = next((path for path in data_candidates if path.is_file()), data_candidates[0])
        html = PLAN / "figures" / "html" / f"{stem}.html"
        if not data.is_file() or not html.is_file():
            continue
        data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
        if data_hash not in tikz.read_text(errors="ignore"):
            errors.append(f"TikZ/data semantic-parity hash mismatch: {relative(tikz)}")
        if data_hash not in html.read_text(errors="ignore"):
            errors.append(f"HTML/data semantic-parity hash mismatch: {relative(html)}")


def main() -> int:
    errors: list[str] = []
    validate_required_files(errors)
    validate_publication_boundary(errors)
    validate_contracts(errors)
    validate_registries(errors)
    validate_p13_freeze(errors)
    validate_figures(errors)

    if errors:
        print("NATO SERS research validation: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"NATO SERS research validation: PASS ({len(all_files())} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
