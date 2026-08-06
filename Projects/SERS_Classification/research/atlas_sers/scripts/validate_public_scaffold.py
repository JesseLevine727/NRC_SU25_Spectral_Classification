#!/usr/bin/env python3
"""Validate the public ATLAS scaffold without accessing private inputs."""

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

# Kept as code points so the restricted source name is not reproduced in the
# public repository. Matching is case-insensitive and covers paths and content.
RESTRICTED_SOURCE_TOKEN = bytes((110, 97, 116, 111))
RESTRICTED_SOURCE_PATTERN = re.compile(rb"(?<![a-z])" + RESTRICTED_SOURCE_TOKEN + rb"(?![a-z])")
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
    "plan/FIGURE_STYLE_AND_REGENERATION.md",
    "plan/index.html",
    "plan/contracts/research_contract.json",
    "plan/contracts/split_contract.json",
    "plan/contracts/hyperparameter_registry.json",
    "plan/contracts/result_schema.json",
    "plan/contracts/figure_contract.json",
    "plan/contracts/compute_budget.json",
    "plan/contracts/p00_governance_contract.json",
    "plan/contracts/p00_validation_schema.json",
    "plan/registries/model_registry.csv",
    "plan/registries/artifact_registry.csv",
    "plan/registries/deviations.csv",
    "scripts/run_p00.py",
}

REGISTRY_COUNTS = {
    "phase_registry.csv": 13,
    "task_registry.csv": 12,
    "metric_registry.csv": 25,
    "experiment_registry.csv": 39,
    "figure_registry.csv": 35,
    "model_registry.csv": 36,
    "artifact_registry.csv": 37,
    "decision_gate_registry.csv": 13,
    "deviations.csv": 0,
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
        name_bytes = relative(path).encode("utf-8", errors="ignore").lower()
        content = path.read_bytes().lower()
        restricted_name = RESTRICTED_SOURCE_PATTERN.search(name_bytes)
        restricted_content = RESTRICTED_SOURCE_PATTERN.search(content)
        if restricted_name or restricted_content:
            errors.append(f"restricted source identifier found: {relative(path)}")
        if POSIX_USER_PREFIX in content or WINDOWS_USER_PREFIX in content:
            errors.append(f"absolute workstation path found: {relative(path)}")
        if path.suffix.lower() in PROHIBITED_ARTIFACT_SUFFIXES:
            errors.append(f"private artifact format found: {relative(path)}")
        if any(path.name.lower().endswith(suffix) for suffix in BUILD_SUFFIXES):
            errors.append(f"build by-product found: {relative(path)}")
        if path.stat().st_size > 5_000_000:
            errors.append(f"unexpected file larger than 5 MB: {relative(path)}")
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
    if not protocol.startswith("atlas-sers-"):
        errors.append("research protocol does not use the ATLAS namespace")
    inputs = research.get("authoritative_inputs", [])
    if not isinstance(inputs, list) or not inputs:
        errors.append("research contract has no authoritative input declarations")
        return
    for item in inputs:
        path = item.get("path", "") if isinstance(item, dict) else ""
        if not str(path).startswith("${ATLAS_PRIVATE_ROOT}/"):
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


def validate_figures(errors: list[str]) -> None:
    html_paths = sorted((PLAN / "figures" / "html").glob("*.html"))
    tikz_paths = sorted((PLAN / "figures" / "tikz").glob("*.tex"))
    if len(html_paths) != 2:
        errors.append(f"expected 2 completed HTML plan figures; found {len(html_paths)}")
    if len(tikz_paths) != 3:
        errors.append(f"expected 2 TikZ figures plus 1 shared style; found {len(tikz_paths)}")

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
        data = PLAN / "figures" / "data" / f"{stem}.csv"
        html = PLAN / "figures" / "html" / f"{stem}.html"
        pdf = PLAN / "figures" / "pdf" / f"{stem}.pdf"
        for expected in (data, html, pdf):
            if not expected.is_file():
                errors.append(f"completed figure artifact missing: {relative(expected)}")
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
    validate_figures(errors)

    if errors:
        print("ATLAS public scaffold validation: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"ATLAS public scaffold validation: PASS ({len(all_files())} files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
