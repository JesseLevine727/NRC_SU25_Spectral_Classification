"""Focused real-boundary tests for the T362 private release-table writer.

Uses the actual analyze_panel fixture from test_p08_release_contrasts with the
four-draw override and the real metric and contrast adapters.  Writer, CSV/JSON
bytes and exclusive creation run for real; only guard/deadline come from tests.
"""

from __future__ import annotations

import hashlib
import json
import time

import pandas as pd
import pytest

from atlas_sers.evaluation import p08_release_metrics
from atlas_sers.evaluation import p08_release_write as writer
from atlas_sers.visualization import p08_figure_build as build
from tests.test_p08_release_contrasts import computed as computed
from tests.test_p08_release_contrasts import small as small
from tests.test_p08_universal_analysis import GLOBAL_MASTERS

REFS = {"analysis_receipt_sha256": "a" * 64, "bundle_manifest_sha256": "b" * 64}


def _deadline():
    return time.monotonic() + 60.0


def _guard():
    return True


def _write(analysis, output):
    return writer.write_release_tables(
        analysis,
        output=output,
        source_refs=dict(REFS),
        check=_guard,
        deadline=_deadline(),
    )


def test_writes_exact_nine_tables_manifest_and_hashes(small, tmp_path):
    output = tmp_path / "candidate_tables"
    manifest = _write(small, output)
    assert manifest["schema"] == writer.SCHEMA
    assert manifest["status"] == "built_unreviewed"
    assert set(manifest["tables"]) == set(writer.TABLE_NAMES)
    assert len(set(writer.TABLE_NAMES)) == 9
    assert manifest["input_source_hashes"] == REFS
    assert manifest["reviewed"] is False
    assert manifest["disclosure_reviewed"] is False
    assert manifest["published"] is False
    assert manifest["external_authentication"] is False
    for name in writer.TABLE_NAMES:
        path = output / (name + ".csv")
        assert path.is_file()
        entry = manifest["tables"][name]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["bytes"] == path.stat().st_size
        assert entry["rows"] >= 0
        assert entry["columns"]
    assert (
        manifest["diagnostics"]["sha256"]
        == hashlib.sha256((output / "diagnostics.json").read_bytes()).hexdigest()
    )
    assert (output / "manifest.json").is_file()


def test_csv_point_values_nulls_and_no_identity_leak(small, tmp_path):
    output = tmp_path / "candidate_tables"
    manifest = _write(small, output)
    metrics = p08_release_metrics.prepare_metric_tables(small)
    frame = metrics["model_summary"]
    reread = pd.read_csv(output / "model_summary.csv", float_precision="round_trip")
    assert len(reread) == len(frame)
    for column in frame.columns:
        text = str(column)
        assert text in reread.columns
        if pd.api.types.is_numeric_dtype(frame[column]):
            left = pd.to_numeric(frame[column], errors="coerce")
            right = pd.to_numeric(reread[text], errors="coerce")
            assert (left.eq(right) | (left.isna() & right.isna())).all()
    for name in writer.TABLE_NAMES:
        raw = (output / (name + ".csv")).read_text("utf-8")
        assert raw.splitlines()[0].split(",")[0] != ""
        assert "Unnamed" not in raw
        assert "NaN" not in raw
        table = pd.read_csv(output / (name + ".csv"))
        assert len(table) == manifest["tables"][name]["rows"]
        assert not {"master_id", "sample_id", "observation_id", "context_id", "unit_id"} & set(
            table
        )
        for column in table:
            if not pd.api.types.is_numeric_dtype(table[column]):
                assert not set(table[column].dropna().astype(str)) & set(GLOBAL_MASTERS)
    contrasts = pd.read_csv(output / "contrast_summary.csv")
    assert contrasts["point_effect"].isna().sum() == 16
    assert (output.stat().st_mode & 0o777) == 0o700
    assert all((path.stat().st_mode & 0o777) == 0o600 for path in output.iterdir())
    assert str(tmp_path) not in (output / "manifest.json").read_text("utf-8")


def test_existing_and_dotdot_output_refused_untouched(small, tmp_path):
    output = tmp_path / "candidate_tables"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(build.FigureBuildError):
        _write(small, output)
    assert sentinel.read_text("utf-8") == "keep"
    assert not (output / "failure.json").exists()
    assert not (output / "manifest.json").exists()
    with pytest.raises(build.FigureBuildError):
        writer.write_release_tables(
            small,
            output=tmp_path / ".." / (tmp_path.name + "_dot"),
            source_refs=dict(REFS),
            check=_guard,
            deadline=_deadline(),
        )


def test_required_guard_and_expired_deadline(small, tmp_path):
    refused = tmp_path / "guard_refused"
    with pytest.raises(build.FigureBuildError):
        writer.write_release_tables(
            small,
            output=refused,
            source_refs=dict(REFS),
            check=lambda: False,
            deadline=_deadline(),
        )
    assert not refused.exists()
    expired = tmp_path / "deadline_expired"
    with pytest.raises(build.FigureBuildError):
        writer.write_release_tables(
            small,
            output=expired,
            source_refs=dict(REFS),
            check=_guard,
            deadline=time.monotonic() - 1.0,
        )
    assert not expired.exists()


def test_forced_write_failure_preserves_partial_without_success(small, tmp_path, monkeypatch):
    output = tmp_path / "partial"
    real_write = writer._write_exclusive

    def boom(path, data):
        if str(path).endswith(writer.DIAGNOSTICS_FILE):
            raise RuntimeError("write_boom")
        return real_write(path, data)

    monkeypatch.setattr(writer, "_write_exclusive", boom)
    with pytest.raises(RuntimeError):
        _write(small, output)
    assert (output / "model_summary.csv").is_file()
    assert (output / "failure.json").is_file()
    assert not (output / "manifest.json").exists()
    assert json.loads((output / "failure.json").read_text("utf-8"))["status"] == "failed"


def test_final_guard_failure_cannot_publish_success(small, tmp_path):
    output = tmp_path / "final_guard"

    def check():
        return not (output / "diagnostics.json").exists()

    with pytest.raises(build.FigureBuildError):
        writer.write_release_tables(
            small,
            output=output,
            source_refs=dict(REFS),
            check=check,
            deadline=_deadline(),
        )
    assert (output / "diagnostics.json").is_file()
    assert (output / "failure.json").is_file()
    assert not (output / "manifest.json").exists()
