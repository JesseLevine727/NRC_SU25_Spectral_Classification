"""Synthetic tests for the P05 diagnostic-figure rendering module.

All inputs are anonymous aggregate DataFrames assembled in memory by the
reusable public fixtures of the P05 benchmark, source-diagnostic and
reliability test modules. No spectra, training runs, torch or private records
are touched. The real LaTeX/PDF/PNG toolchain is exercised only by the guarded
integration test.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from atlas_sers.visualization import p05_diagnostic_figures as mod
from tests import test_p05_benchmark_figures as bench
from tests import test_p05_reliability as rel
from tests import test_p05_source_diagnostics as src

FAMILIES = (
    mod.FAMILY_PAIRED,
    mod.FAMILY_SOURCE,
    mod.FAMILY_EPOCH,
    mod.FAMILY_RELIABILITY,
)
FORBIDDEN_TOKENS = (
    "context_id",
    "slot_id",
    "selection_unit_id",
    "refit_id",
    "observation_uid",
    "private",
    "sentinel",
    "path",
    "uid",
    "label",
    "prediction",
)


def _paired_frame() -> pd.DataFrame:
    return bench._base_frame()


def _source_tables() -> dict:
    bundle = src._combined_bundle()
    plan = src._plan(bundle)
    out = src._run(plan, bundle, src._public_frame())
    return {
        "source_vs_held": out["source_vs_held"],
        "best_epoch_distribution": out["best_epoch_distribution"],
    }


def _reliability_tables() -> dict:
    ensemble, strategy = rel._fixture()
    out = rel._build(ensemble, strategy)
    return {"reliability_bins": out["reliability_bins"]}


def _semantic() -> pd.DataFrame:
    return mod._build_semantic(
        {"paired_domains": _paired_frame()},
        _source_tables(),
        _reliability_tables(),
    )


def _spec(frame: pd.DataFrame, family: str) -> dict:
    return next(spec for spec in mod._figure_specs(frame) if spec["family"] == family)


def _points(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    return frame[frame.figure.eq(spec["figure"])].reset_index(drop=True)


def _inputs() -> dict:
    return {
        "public_metrics": {"paired_domains": _paired_frame()},
        "source_diagnostics": _source_tables(),
        "reliability": _reliability_tables(),
    }


@pytest.fixture()
def stub_compile(monkeypatch):
    def _compile(tex_path, pdf_path, png_path, log_path):
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        log_path.write_text("stub\n", encoding="utf-8")

    monkeypatch.setattr(mod, "_compile", _compile)


def test_semantic_allowlist_and_all_families():
    frame = _semantic()
    assert tuple(frame.columns) == mod.SEMANTIC_COLUMNS
    assert not frame.empty
    for column in frame.columns:
        lowered = column.lower()
        assert not any(token in lowered for token in FORBIDDEN_TOKENS)
    present = {str(value).split("|", 1)[0] for value in frame["figure"]}
    assert present == set(FAMILIES)


def test_private_sentinels_stripped(tmp_path, stub_compile):
    paired = _paired_frame()
    paired["private_uid"] = "SENTINEL-UID"
    source = _source_tables()
    source["source_vs_held"] = source["source_vs_held"].assign(private_note="SENTINEL-NOTE")
    reliability = _reliability_tables()
    reliability["reliability_bins"] = reliability["reliability_bins"].assign(
        private_note="SENTINEL-NOTE"
    )
    root = tmp_path / "sentinel"
    manifest = mod.generate_diagnostic_figures(
        public_metrics={"paired_domains": paired},
        source_diagnostics=source,
        reliability=reliability,
        output_root=root,
    )
    assert "SENTINEL" not in (root / mod.SEMANTIC_NAME).read_text(encoding="utf-8")
    for record in manifest["figures"]:
        for suffix in (".tex", ".html"):
            text = (root / f"{record['stem']}{suffix}").read_text(encoding="utf-8")
            assert "SENTINEL" not in text
            assert "private_uid" not in text


def test_projected_coordinates_exact_in_tikz_and_plotly():
    frame = _semantic()
    for family in FAMILIES:
        spec = _spec(frame, family)
        points = _points(frame, spec)
        plotted = points[points["count"] > 0]
        tex = mod._tikz_figure(spec, points, "deadbeef")
        figure = mod._plotly_figure(spec, points)
        coords = []
        for line in tex.splitlines():
            if "\\addplot[" not in line or "mark=" not in line:
                continue
            body = line.split("coordinates {", 1)[1].rsplit("}", 1)[0]
            coords.extend(re.findall(r"\(([^,]+),([^)]+)\)", body))
        assert len(coords) == len(plotted)
        for row in plotted.itertuples(index=False):
            assert f"({float(row.x):.17g},{float(row.y):.17g})" in tex
        trace_x = [
            float(value)
            for trace in figure.data
            if trace.mode in ("markers", "lines+markers")
            for value in trace.x
        ]
        assert sorted(trace_x) == sorted(float(value) for value in plotted["x"])


def test_best_epoch_fractions_use_count_and_denominator():
    frame = _semantic()
    epoch = frame[frame.figure.str.startswith(mod.FAMILY_EPOCH + "|")]
    assert not epoch.empty
    for _, cell in epoch.groupby(["figure", "station", "recipe"]):
        assert cell["denominator"].nunique() == 1
        assert int(cell["count"].sum()) == int(cell["denominator"].iloc[0])
        for row in cell.itertuples(index=False):
            assert float(row.y) == pytest.approx(int(row.count) / int(row.denominator), abs=1e-12)


def test_source_six_endpoints_and_selected_d1_accepted():
    frame = _semantic()
    source = frame[frame.figure.str.startswith(mod.FAMILY_SOURCE + "|")]
    expected = {
        (model, aggregation) for model in mod.MODEL_ORDER for aggregation in mod.AGGREGATION_ORDER
    }
    assert len(expected) == 6
    assert not source.empty
    for _, cell in source.groupby("point_index"):
        assert {
            (row.series, row.aggregation_id) for row in cell.itertuples(index=False)
        } == expected
    assert not source[source.series.eq("P05-SELECTED") & source.recipe.eq("D1")].empty


def _source_mutate_recipe(table):
    table.loc[table.index[table["model_id"].eq("D0-M")][0], "recipe"] = "D1"


def _source_mutate_flag(table):
    index = table.index[table["selection_mode"].eq("pseudo_domain")][0]
    table.loc[index, "source_transfer_validation_available"] = False


def _source_mutate_guard(table):
    index = table.index[table["selection_mode"].eq("pseudo_domain")][0]
    table.loc[index, "guard_unit_count"] = 0


@pytest.mark.parametrize(
    "mutate",
    [_source_mutate_recipe, _source_mutate_flag, _source_mutate_guard],
    ids=["wrong_recipe", "wrong_flag", "wrong_guard_count"],
)
def test_source_wrong_recipe_flags_guard_counts_rejected(mutate):
    tables = _source_tables()
    mutate(tables["source_vs_held"])
    with pytest.raises(mod.DiagnosticFigureError):
        mod._build_semantic({"paired_domains": _paired_frame()}, tables, _reliability_tables())


def test_reliability_bins_ordinal_and_ties_survive_shuffle():
    frame = _semantic()
    reliability = _reliability_tables()
    shuffled = {
        "reliability_bins": reliability["reliability_bins"]
        .sample(frac=1.0, random_state=13)
        .reset_index(drop=True)
    }
    other = mod._build_semantic({"paired_domains": _paired_frame()}, _source_tables(), shuffled)
    key = ["figure", "panel_row", "panel_col", "series", "bin_index"]
    left = (
        frame[frame.figure.str.startswith(mod.FAMILY_RELIABILITY + "|")]
        .sort_values(key)
        .reset_index(drop=True)
    )
    right = (
        other[other.figure.str.startswith(mod.FAMILY_RELIABILITY + "|")]
        .sort_values(key)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(left, right)
    dev = left[left.figure.eq(mod._figure_id(mod.FAMILY_RELIABILITY, "development", "M01"))]
    dev = dev[dev.series.eq("D0-M")]
    assert dev["bin_index"].tolist() == list(range(1, 11))
    assert dev["x"].nunique() == 1


def _reliability_mutate_weight(table):
    index = table.index[0]
    table.loc[index, "bin_weight"] = float(table.loc[index, "bin_weight"]) + 0.5


def _reliability_mutate_gap(table):
    index = table.index[0]
    table.loc[index, "signed_gap"] = float(table.loc[index, "signed_gap"]) + 0.5


def _reliability_mutate_mass(table):
    index = table.index[0]
    table.loc[index, "count"] = int(table.loc[index, "count"]) + 1


@pytest.mark.parametrize(
    "mutate",
    [_reliability_mutate_weight, _reliability_mutate_gap, _reliability_mutate_mass],
    ids=["wrong_weight", "wrong_signed_gap", "wrong_bin_mass"],
)
def test_reliability_wrong_weight_gap_mass_rejected(mutate):
    tables = _reliability_tables()
    mutate(tables["reliability_bins"])
    with pytest.raises(mod.DiagnosticFigureError):
        mod._build_semantic({"paired_domains": _paired_frame()}, _source_tables(), tables)


def test_zero_common_paired_retained_missing_and_not_plotted():
    paired = _paired_frame()
    index = paired.index[0]
    planned = int(paired.loc[index, "planned_contexts"])
    domain = str(paired.loc[index, "domain"])
    model = str(paired.loc[index, "model_id"])
    paired.loc[index, "common_contexts"] = 0
    paired.loc[index, "common_coverage"] = 0.0
    paired.loc[index, "mean_model_balanced_accuracy"] = np.nan
    paired.loc[index, "mean_reference_balanced_accuracy"] = np.nan
    paired.loc[index, "mean_delta_balanced_accuracy"] = np.nan
    frame = mod._build_semantic({"paired_domains": paired}, _source_tables(), _reliability_tables())
    stored = frame[
        frame.figure.str.startswith(mod.FAMILY_PAIRED + "|")
        & frame.domain.eq(domain)
        & frame.series.eq(model)
    ].iloc[0]
    assert int(stored["count"]) == 0
    assert int(stored["denominator"]) == planned
    assert pd.isna(stored["x"]) and pd.isna(stored["y"])
    spec = _spec(frame, mod.FAMILY_PAIRED)
    counts = mod._figure_counts(spec, _points(frame, spec))
    assert counts["point_count"] < counts["semantic_rows"]


def test_all_missing_paired_figure_still_rendered_with_caption():
    paired = _paired_frame()
    for index in paired.index:
        paired.loc[index, "common_contexts"] = 0
        paired.loc[index, "common_coverage"] = 0.0
        paired.loc[index, "mean_model_balanced_accuracy"] = np.nan
        paired.loc[index, "mean_reference_balanced_accuracy"] = np.nan
        paired.loc[index, "mean_delta_balanced_accuracy"] = np.nan
    frame = mod._build_semantic({"paired_domains": paired}, _source_tables(), _reliability_tables())
    spec = _spec(frame, mod.FAMILY_PAIRED)
    points = _points(frame, spec)
    assert mod._figure_counts(spec, points)["point_count"] == 0
    assert mod._caption_lines(spec)
    tex = mod._tikz_figure(spec, points, "deadbeef")
    assert mod.PAIRED_CAPTION[0] in tex
    assert mod._empty_panel_label(spec) in tex
    figure = mod._plotly_figure(spec, points)
    assert any("no common" in (annotation.text or "") for annotation in figure.layout.annotations)


def test_figure_counts_use_count_column_not_method():
    frame = _semantic()
    spec = _spec(frame, mod.FAMILY_PAIRED)
    points = _points(frame, spec)
    counts = mod._figure_counts(spec, points)
    assert counts["semantic_rows"] == len(points)
    assert counts["point_count"] == int((points["count"] > 0).sum())
    assert len(counts["panel_counts"]) == 3
    assert sum(counts["panel_counts"].values()) == counts["point_count"]


def test_source_and_reliability_scaleanchor_and_fixed_ranges():
    frame = _semantic()
    for family in (mod.FAMILY_SOURCE, mod.FAMILY_RELIABILITY):
        spec = _spec(frame, family)
        figure = mod._plotly_figure(spec, _points(frame, spec))
        rows, cols = mod._panel_grid(spec)
        axes = mod._panel_axes(spec)
        assert (axes["xmin"], axes["xmax"]) == (0.0, 1.0)
        assert (axes["ymin"], axes["ymax"]) == (0.0, 1.0)
        for index in range(1, rows * cols + 1):
            xaxis = "xaxis" if index == 1 else f"xaxis{index}"
            yaxis = "yaxis" if index == 1 else f"yaxis{index}"
            assert tuple(getattr(figure.layout, xaxis).range) == (0.0, 1.0)
            assert tuple(getattr(figure.layout, yaxis).range) == (0.0, 1.0)
            assert getattr(figure.layout, yaxis).scaleanchor == ("x" if index == 1 else f"x{index}")
            assert getattr(figure.layout, xaxis).constrain == "domain"
            assert getattr(figure.layout, yaxis).constrain == "domain"


def test_source_six_panel_selected_d1_empty_panels_and_reliability_lines():
    frame = _semantic()
    source = frame[frame.figure.str.startswith(mod.FAMILY_SOURCE + "|")]
    target = source[source.series.eq("P05-SELECTED") & source.recipe.eq("D1")]
    assert not target.empty
    figure_id = target["figure"].iloc[0]
    spec = next(spec for spec in mod._figure_specs(frame) if spec["figure"] == figure_id)
    assert mod._panel_grid(spec) == (2, 3)
    points = _points(frame, spec)
    counts = mod._figure_counts(spec, points)
    assert len(counts["panel_counts"]) == 6
    assert counts["empty_panels"]
    titles = mod._panel_titles(spec, 2, 3)
    assert titles[(0, 0)].startswith("(A)")
    assert titles[(1, 0)].startswith("(D)")
    rel_spec = _spec(frame, mod.FAMILY_RELIABILITY)
    rel_points = _points(frame, rel_spec)
    rel_figure = mod._plotly_figure(rel_spec, rel_points)
    assert any(trace.mode == "lines+markers" for trace in rel_figure.data)
    assert "mark=" in mod._tikz_figure(rel_spec, rel_points, "deadbeef")


def test_html_offline_resource_parser(tmp_path, stub_compile):
    class ResourceParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.external_resources = []

        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                resource = key == "src" or (tag == "link" and key == "href")
                if resource and value and value.startswith(("http:", "https:", "//")):
                    self.external_resources.append((tag, key, value))

    root = tmp_path / "offline"
    manifest = mod.generate_diagnostic_figures(output_root=root, **_inputs())
    for record in manifest["figures"]:
        html = (root / f"{record['stem']}.html").read_text(encoding="utf-8")
        assert "cdn.plot.ly" not in html
        parser = ResourceParser()
        parser.feed(html)
        assert parser.external_resources == []
        assert "plotly" in html.lower()


def test_overwrite_prevention_manifest_counts_and_digest(tmp_path, stub_compile):
    inputs = _inputs()
    root = tmp_path / "bundle"
    manifest = mod.generate_diagnostic_figures(output_root=root, **inputs)
    digest = hashlib.sha256((root / mod.SEMANTIC_NAME).read_bytes()).hexdigest()
    assert manifest["data_sha256"] == digest
    assert manifest["semantic_path"] == mod.SEMANTIC_NAME
    assert manifest["plotted_rows"] <= manifest["semantic_rows"]
    assert manifest["figures"]
    for record in manifest["figures"]:
        assert record["semantic_sha256"] == digest
        assert record["point_count"] == sum(record["panel_counts"].values())
        assert digest in (root / f"{record['stem']}.tex").read_text(encoding="utf-8")
        assert digest in (root / f"{record['stem']}.html").read_text(encoding="utf-8")
        for suffix in ("tex", "pdf", "png", "html"):
            path = root / f"{record['stem']}.{suffix}"
            assert record[f"{suffix}_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    names = {entry["path"] for entry in manifest["files"]}
    assert mod.SEMANTIC_NAME in names
    assert mod.MANIFEST_NAME not in names
    with pytest.raises(FileExistsError):
        mod.generate_diagnostic_figures(output_root=root, **inputs)


@pytest.mark.skipif(
    shutil.which("pdflatex") is None or shutil.which("pdftocairo") is None,
    reason="TeX toolchain unavailable",
)
def test_real_compile_all_families(tmp_path):
    root = tmp_path / "compiled"
    manifest = mod.generate_diagnostic_figures(output_root=root, **_inputs())
    families = {record["family"] for record in manifest["figures"]}
    assert families == set(FAMILIES)
    for record in manifest["figures"]:
        assert (root / f"{record['stem']}.pdf").read_bytes().startswith(b"%PDF")
        assert (root / f"{record['stem']}.png").read_bytes().startswith(b"\x89PNG")
