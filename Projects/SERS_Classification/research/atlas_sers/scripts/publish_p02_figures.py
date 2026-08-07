#!/usr/bin/env python3
"""Publish only disclosure-reviewed aggregate P02 figure forms."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from atlas_sers.governance.canonical import sha256_file
from atlas_sers.paths import artifact_root, project_root

FIGURES = {"F10_split_design", "F11_domain_support"}
FORMS = {"data": ".csv", "tikz": ".tex", "pdf": ".pdf", "png": ".png", "html": ".html"}
FORBIDDEN_COLUMNS = {
    "observation_uid",
    "master_sample_id",
    "source_logical_id",
    "target_analyte",
    "qc_feature_value",
    "numeric_cutpoint",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish reviewed ATLAS P02 figures")
    parser.add_argument("--run-dir", type=Path)
    return parser


def _latest_run() -> Path:
    root = artifact_root()
    latest = json.loads((root / "p02" / "LATEST.json").read_text())
    return root / "p02" / "runs" / latest["run_id"]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = (args.run_dir or _latest_run()).resolve()
    report = json.loads((run_dir / "P02_VALIDATION_REPORT.json").read_text())
    if report["status"] != "pass" or not all(report["checks"].values()):
        raise RuntimeError("Only a fully passing P02 run may publish aggregate figures.")
    public = project_root() / "plan" / "figures"
    copied: dict[str, str] = {}
    for stem in sorted(FIGURES):
        data = run_dir / "figures" / "data" / f"{stem}.csv"
        columns = set(pd.read_csv(data, nrows=0).columns)
        if columns & FORBIDDEN_COLUMNS:
            raise RuntimeError(f"Disclosure review failed for {stem}: protected column present.")
        data_hash = sha256_file(data)
        for form, suffix in FORMS.items():
            source = run_dir / "figures" / form / f"{stem}{suffix}"
            if not source.is_file():
                raise FileNotFoundError(source)
            if form in {"tikz", "html"} and data_hash not in source.read_text():
                raise RuntimeError(f"Semantic-parity hash missing from {source.name}.")
            destination = public / form / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied[destination.relative_to(project_root()).as_posix()] = sha256_file(destination)
    print(json.dumps({"status": "pass", "files": copied}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
