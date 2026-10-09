"""Data-free acceptance checks for the actual reviewed U1 public release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
RELEASE = PACKAGE / "results/p08_universal/release"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_inventory_and_review():
    manifest = json.loads((RELEASE / "release_manifest.json").read_text())
    assert manifest["status"] == "reviewed_for_publication"
    assert manifest["training_counts"] == {
        "unique_fits": 195202,
        "scalar_calibrations": 3354,
        "operations": 404814,
        "remaining": 0,
        "running": 0,
    }
    files = {p.relative_to(RELEASE).as_posix() for p in RELEASE.rglob("*") if p.is_file()}
    assert files == set(manifest["files"]) | {"release_manifest.json"}
    for relative, spec in manifest["files"].items():
        path = RELEASE / relative
        assert not path.is_symlink()
        assert digest(path) == spec["sha256"], relative
        assert path.stat().st_size == spec["bytes"]
        assert path.suffix in {".json", ".csv", ".html", ".tex", ".pdf", ".png", ".md"}
    for relative, expected in manifest["public_code_sha256"].items():
        assert digest(PACKAGE / relative) == expected
    report = PACKAGE / "reports/NATO_SERS_UNIVERSAL_PREPROCESSING_REPORT.md"
    assert digest(report) == manifest["report_sha256"]
    for link in re.findall(r"\]\(([^)]+)\)", report.read_text()):
        assert (report.parent / link).resolve().is_file(), link


def test_figure_manifests_and_four_formats():
    root = RELEASE / "figures"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["figure_count"] == 6 and manifest["total_panels"] == 117
    assert manifest["reviewed"] and manifest["visual_reviewed"] and manifest["disclosure_reviewed"]
    for figure in manifest["figures"]:
        directory = root / figure["figure_id"]
        path = directory / "manifest.json"
        assert digest(path) == figure["manifest_sha256"]
        child = json.loads(path.read_text())
        assert child["semantic_sha256"] == figure["semantic_sha256"]
        assert child["reviewed"] and child["visual_reviewed"] and child["disclosure_reviewed"]
        assert (directory / child["review_record"]).resolve().is_file()
        for item in child["outputs"]:
            assert digest(directory / item["path"]) == item["sha256"]
        groups = [
            {p.stem for p in (directory / sub).glob("*." + extension)}
            for sub, extension in (
                ("tikz", "tex"),
                ("pdf", "pdf"),
                ("png", "png"),
                ("html", "html"),
            )
        ]
        assert all(group == groups[0] for group in groups)
        assert len(groups[0]) == figure["panel_count"]
        for source in (directory / "tikz").glob("*.tex"):
            assert "\\includegraphics" not in source.read_text()


def test_public_spectral_support_and_actual_scores():
    semantic = json.loads((RELEASE / "figures/P08-F01/data/semantic.json").read_text())
    assert len(semantic["cells"]) == 49
    assert sum(cell["available"] for cell in semantic["cells"]) == 46
    assert sum(domain["exploratory"] for domain in semantic["domains"]) == 4
    for cell in semantic["cells"]:
        if cell["available"]:
            assert cell["n_masters"] >= 2 and len(cell["curves"]) == 3
        else:
            assert not cell["curves"] and cell["reason"]
    scores = pd.read_csv(RELEASE / "tables/model_summary.csv")
    assert len(scores) == 60
    primary = scores[scores.estimand.eq("equal_context")]
    paired = primary.pivot(
        index=["model_id", "endpoint"], columns="policy_id", values="balanced_accuracy"
    )
    assert (paired["PP-U-ARPLS"] > paired["PP-U-MIN"]).all()
    assert abs(paired.loc[("C-RANDOM-FOREST", "M01"), "PP-U-ARPLS"] - 0.768062) < 1e-6
    families = pd.read_csv(RELEASE / "family_sensitivity/family_deletions.csv")
    assert len(families) == 352


def test_conservative_accounting_within_approved_limits():
    budget = json.loads((RELEASE / "release_manifest.json").read_text())["accounting"]
    assert 42398 <= budget["charged_active_seconds"] <= 48 * 3600
    assert 73668334526 <= budget["charged_artifact_bytes"] <= 80 * 1024**3
    assert budget["finalization_time_reserve_seconds"] == 3600
    assert budget["no_counter_reset"] is True
