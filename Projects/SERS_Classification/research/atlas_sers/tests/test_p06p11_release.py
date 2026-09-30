"""Synthetic contract tests for the deterministic private release writer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from atlas_sers.evaluation import p06p11_release as rel


def _spec(figure_id: str, rows: int) -> dict:
    return {
        "semantic": pd.DataFrame({"x": list(range(rows))}),
        "title": "title " + figure_id,
        "caption": "caption " + figure_id,
        "kind": "scatter",
    }


def _fake_read_csv(path, **kwargs):
    return pd.DataFrame({"value": [1, 2]})


def _fake_read_parquet(path):
    return {"sentinel": str(path)}


def _fake_prepare(tables, panels):
    assert set(tables) == set(rel.TABLE_NAMES)
    assert set(panels) == {"M01", "M06"}, set(panels)
    assert "panel_M01.parquet" in panels["M01"]["sentinel"]
    assert "panel_M06.parquet" in panels["M06"]["sentinel"]
    inference_tables = {name: pd.DataFrame({"value": [1, 2]}) for name in rel.TABLE_NAMES}
    metrics = {name: pd.DataFrame({"value": [1]}) for name in rel.METRIC_NAMES}
    return {
        "inference_tables": inference_tables,
        "metrics": metrics,
        "tie_corrections": pd.DataFrame({"value": list(range(rel.EXPECTED_TIE_ROWS))}),
        "crosscheck_count": rel.EXPECTED_CROSSCHECK_COUNT,
        "crosscheck_max_error": 0.0,
    }


def _fake_build_semantics(inference_tables):
    assert set(inference_tables) == set(rel.TABLE_NAMES)
    fid = rel.FIGURE_IDS[0]
    return {fid: _spec(fid, rel.FIGURE_ROWS[0])}


def _fake_build_interval_semantics(inference_tables):
    fid = rel.FIGURE_IDS[1]
    return {fid: _spec(fid, rel.FIGURE_ROWS[1])}


def _fake_build_deletion_semantics(inference_tables):
    first, second = rel.FIGURE_IDS[2], rel.FIGURE_IDS[3]
    return {
        first: _spec(first, rel.FIGURE_ROWS[2]),
        second: _spec(second, rel.FIGURE_ROWS[3]),
    }


def _fake_semantic_sha(frame):
    return hashlib.sha256(str(len(frame)).encode("utf-8")).hexdigest()


def _fake_render_tex(figure_id, frame, title, caption, kind, semantic_sha):
    assert semantic_sha
    return "tex " + figure_id


def _fake_render_html(figure_id, frame, title, caption, kind, semantic_sha):
    return "html " + figure_id


def _fake_compile(tex_path, pdf_path, png_path, log_path, deadline):
    for path in (tex_path, pdf_path, png_path, log_path):
        assert isinstance(path, Path), path
    assert deadline > 0.0
    pdf_path.write_bytes(b"%PDF-1.4 tiny")
    png_path.write_bytes(b"\x89PNG tiny")
    log_path.write_bytes(b"log")


def _patch(monkeypatch, compile_hook=None):
    monkeypatch.setattr(rel.pd, "read_csv", _fake_read_csv)
    monkeypatch.setattr(rel.pd, "read_parquet", _fake_read_parquet)
    monkeypatch.setattr(rel, "prepare_tables", _fake_prepare)
    monkeypatch.setattr(rel, "build_semantics", _fake_build_semantics)
    monkeypatch.setattr(rel, "build_interval_semantics", _fake_build_interval_semantics)
    monkeypatch.setattr(rel, "build_deletion_semantics", _fake_build_deletion_semantics)
    monkeypatch.setattr(rel, "_semantic_sha", _fake_semantic_sha)
    monkeypatch.setattr(rel, "_render_tex", _fake_render_tex)
    monkeypatch.setattr(rel, "_render_html", _fake_render_html)
    monkeypatch.setattr(rel, "_compile", compile_hook or _fake_compile)


def _populate_analysis(analysis: Path):
    code_hashes = {"src/atlas_sers/evaluation/p06p11_release.py": "a" * 64}
    input_hashes = {"data/original_input.csv": "b" * 64}
    for name in rel.WRITTEN_FILES:
        if name == "start.json":
            continue
        path = analysis / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("payload:" + name).encode("utf-8"))
    (analysis / "start.json").write_bytes(
        json.dumps({"code_hashes": code_hashes, "input_hashes": input_hashes}).encode("utf-8")
    )
    written = {
        name: hashlib.sha256((analysis / name).read_bytes()).hexdigest()
        for name in rel.WRITTEN_FILES
    }
    receipt = {
        "status": "success",
        "mode": "full",
        "draws": rel.DRAW_COUNT,
        "protocol_sha256": rel.PROTOCOL_SHA256,
        "code_hashes": code_hashes,
        "input_hashes": input_hashes,
        "written_files": written,
        "seeds": {"global": 11},
        "software_versions": {"python": "3.11"},
    }
    (analysis / "receipt.json").write_bytes(json.dumps(receipt).encode("utf-8"))
    return code_hashes, input_hashes


def _make_analysis(tmp_path, name="analysis"):
    analysis = tmp_path / name
    analysis.mkdir(parents=True)
    code_hashes, input_hashes = _populate_analysis(analysis)
    return analysis, code_hashes, input_hashes


def test_run_release_success(tmp_path, monkeypatch):
    analysis, code_hashes, input_hashes = _make_analysis(tmp_path)
    output = tmp_path / "out"
    _patch(monkeypatch)
    manifest = rel.run_release(analysis, output)
    assert manifest["status"] == "requires_supervisor_review"
    assert manifest["original_code_hashes"] == code_hashes
    assert manifest["original_input_hashes"] == input_hashes
    assert manifest["input_preservation"] is True
    assert manifest["release_code_hashes"] != code_hashes
    assert manifest["protocol_sha256"] == rel.PROTOCOL_SHA256
    receipt = json.loads((analysis / "receipt.json").read_bytes())
    assert manifest["original_written_files"] == receipt["written_files"]
    assert (
        manifest["original_receipt_sha256"]
        == hashlib.sha256((analysis / "receipt.json").read_bytes()).hexdigest()
    )
    assert (output / "release_manifest.json").is_file()
    assert len(manifest["figures"]) == len(rel.FIGURE_IDS)
    for figure, fid in zip(manifest["figures"], rel.FIGURE_IDS, strict=True):
        assert figure["id"] == fid
        suffixes = {Path(name).suffix for name in figure["files"]}
        assert {".pdf", ".png"} <= suffixes


def test_output_exists_rejected(tmp_path, monkeypatch):
    analysis, _, _ = _make_analysis(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    _patch(monkeypatch)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, output)


def test_relative_path_rejected(tmp_path, monkeypatch):
    _make_analysis(tmp_path)
    _patch(monkeypatch)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(Path("analysis"), tmp_path / "out")


def test_output_inside_analysis_rejected(tmp_path, monkeypatch):
    analysis, _, _ = _make_analysis(tmp_path)
    _patch(monkeypatch)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, analysis / "out")


def test_output_dotdot_bypass_rejected(tmp_path, monkeypatch):
    analysis, _, _ = _make_analysis(tmp_path)
    _patch(monkeypatch)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, tmp_path / "sub" / ".." / "analysis" / "nested")


def test_output_inside_git_rejected(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    analysis = repo / "analysis"
    analysis.mkdir()
    _populate_analysis(analysis)
    _patch(monkeypatch)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, repo / "out")


def test_symlinked_ancestor_rejected(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    analysis = real / "analysis"
    analysis.mkdir()
    _populate_analysis(analysis)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    _patch(monkeypatch)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(link / "analysis", tmp_path / "out")


def test_tampered_artifact_hash_rejected(tmp_path, monkeypatch):
    analysis, _, _ = _make_analysis(tmp_path)
    (analysis / "arrays.npz").write_bytes(b"tampered")
    _patch(monkeypatch)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, tmp_path / "out")


def test_bad_receipt_rejected(tmp_path, monkeypatch):
    analysis, _, _ = _make_analysis(tmp_path)
    receipt = json.loads((analysis / "receipt.json").read_bytes())
    receipt["status"] = "failed"
    (analysis / "receipt.json").write_bytes(json.dumps(receipt).encode("utf-8"))
    _patch(monkeypatch)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, tmp_path / "out")


def test_missing_artifact_rejected(tmp_path, monkeypatch):
    analysis, _, _ = _make_analysis(tmp_path)
    (analysis / "arrays.npz").unlink()
    _patch(monkeypatch)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, tmp_path / "out")


def test_input_mutation_during_compile_rejected(tmp_path, monkeypatch):
    analysis, _, _ = _make_analysis(tmp_path)
    output = tmp_path / "out"

    def mutating_compile(tex_path, pdf_path, png_path, log_path, deadline):
        for path in (tex_path, pdf_path, png_path, log_path):
            assert isinstance(path, Path)
        pdf_path.write_bytes(b"%PDF tiny")
        png_path.write_bytes(b"PNG tiny")
        log_path.write_bytes(b"log")
        (analysis / "arrays.npz").write_bytes(b"mutated during compile")

    _patch(monkeypatch, compile_hook=mutating_compile)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, output)
    assert not (output / "release_manifest.json").exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 1.5])
def test_bad_crosscheck_error_rejected(tmp_path, monkeypatch, bad):
    analysis, _, _ = _make_analysis(tmp_path)
    _patch(monkeypatch)

    def bad_prepare(tables, panels):
        prepared = _fake_prepare(tables, panels)
        prepared["crosscheck_max_error"] = bad
        return prepared

    monkeypatch.setattr(rel, "prepare_tables", bad_prepare)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, tmp_path / "out")


def test_bool_crosscheck_error_rejected(tmp_path, monkeypatch):
    analysis, _, _ = _make_analysis(tmp_path)
    _patch(monkeypatch)

    def bad_prepare(tables, panels):
        prepared = _fake_prepare(tables, panels)
        prepared["crosscheck_max_error"] = False
        return prepared

    monkeypatch.setattr(rel, "prepare_tables", bad_prepare)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, tmp_path / "out")


def test_missing_metrics_key_rejected(tmp_path, monkeypatch):
    analysis, _, _ = _make_analysis(tmp_path)
    _patch(monkeypatch)

    def bad_prepare(tables, panels):
        prepared = _fake_prepare(tables, panels)
        prepared["metrics"].pop("confusion")
        return prepared

    monkeypatch.setattr(rel, "prepare_tables", bad_prepare)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, tmp_path / "out")


def test_extra_inference_key_rejected(tmp_path, monkeypatch):
    analysis, _, _ = _make_analysis(tmp_path)
    _patch(monkeypatch)

    def bad_prepare(tables, panels):
        prepared = _fake_prepare(tables, panels)
        prepared["inference_tables"]["arbitrary_path"] = pd.DataFrame({"value": [1]})
        return prepared

    monkeypatch.setattr(rel, "prepare_tables", bad_prepare)
    with pytest.raises(rel.ReleaseError):
        rel.run_release(analysis, tmp_path / "out")


def test_merge_figure_specs_order():
    parts = [{fid: _spec(fid, 1)} for fid in rel.FIGURE_IDS]
    merged = rel._merge_figure_specs(parts)
    assert [spec["title"] for spec in merged] == ["title " + fid for fid in rel.FIGURE_IDS]


def test_merge_figure_specs_rejects_duplicates():
    parts = [
        {rel.FIGURE_IDS[0]: _spec(rel.FIGURE_IDS[0], 1)},
        {rel.FIGURE_IDS[0]: _spec(rel.FIGURE_IDS[0], 1)},
    ]
    with pytest.raises(rel.ReleaseError):
        rel._merge_figure_specs(parts)


def test_merge_figure_specs_rejects_unknown_id():
    parts = [{"F_P06_unknown": _spec("F_P06_unknown", 1)}]
    with pytest.raises(rel.ReleaseError):
        rel._merge_figure_specs(parts)


def test_merge_figure_specs_rejects_wrong_keys():
    parts = [
        {
            rel.FIGURE_IDS[0]: {
                "semantic": pd.DataFrame({"x": [1]}),
                "title": "t",
                "caption": "c",
            }
        }
    ]
    with pytest.raises(rel.ReleaseError):
        rel._merge_figure_specs(parts)


def test_merge_figure_specs_requires_all_ids():
    parts = [{rel.FIGURE_IDS[0]: _spec(rel.FIGURE_IDS[0], 1)}]
    with pytest.raises(rel.ReleaseError):
        rel._merge_figure_specs(parts)
