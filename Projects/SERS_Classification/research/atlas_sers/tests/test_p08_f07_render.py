# ruff: noqa: E501, UP031
# Assertions mirror literal TeX/HTML renderer templates.
"""Tests for the P08-F07 renderer."""

from __future__ import annotations

import copy
import csv
import io
import json
import math

import pytest

from atlas_sers.visualization import p08_f07_data
from atlas_sers.visualization.p08_f02_render import _canonical_sha
from atlas_sers.visualization.p08_f07_render import (
    N_PANELS,
    N_POINTS,
    N_RECORDS_PER_PANEL,
    N_SUMMARIES,
    POLICIES,
    prepare_f07_render,
)
from tests.test_p08_f07_data import _bundle, _preservation


def _prepared():
    return p08_f07_data.prepare_f07(_preservation(), _bundle())


def _render():
    return prepare_f07_render(_prepared())


def _rehash(prepared):
    prepared["semantic_sha256"] = _canonical_sha(prepared["semantic"])
    return prepared


def _embedded_json(html, element_id):
    marker = '<script type="application/json" id="%s">' % element_id
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def test_returns_mapping_with_expected_keys():
    rendered = _render()
    assert set(rendered) == {"semantic", "semantic_sha256", "panels", "manifest"}
    assert isinstance(rendered["panels"], list)
    assert rendered["semantic_sha256"] == rendered["manifest"]["semantic_sha256"]


def test_preservation_and_counts():
    assert _preservation() is not None
    rendered = _render()
    semantic = rendered["semantic"]
    panels = rendered["panels"]
    manifest = rendered["manifest"]
    assert len(semantic["points"]) == N_POINTS
    assert len(semantic["summaries"]) == N_SUMMARIES
    assert len(panels) == N_PANELS
    assert manifest["figure_id"] == "P08-F07"
    assert manifest["status"] == "prepared"
    assert manifest["reviewed"] is False and manifest["published"] is False
    assert len(manifest["slugs"]) == N_PANELS
    assert semantic["source_semantic_sha256"] == _prepared()["semantic_sha256"]


def test_csv_rows_per_panel():
    rendered = _render()
    for panel in rendered["panels"]:
        rows = list(csv.reader(io.StringIO(panel["csv"])))
        assert rows[0] == list(p08_f07_data.POINT_FIELDS)
        assert len(rows) - 1 == N_RECORDS_PER_PANEL


def test_axes_global_fixed_and_frozen():
    rendered = _render()
    points = rendered["semantic"]["points"]
    axes = rendered["manifest"]["axes"]
    finite = [
        p["peak_displacement_median_cm1"]
        for p in points
        if p["peak_displacement_median_cm1"] is not None
    ]
    expected = max(1.0, float(math.ceil(max(finite))))
    assert axes["x_displacement_max"] == expected
    assert axes["x_recall_max"] == 1.0
    assert axes["ticks"]["A"] == [
        0.0,
        0.2 * expected,
        0.4 * expected,
        0.6 * expected,
        0.8 * expected,
        expected,
    ]
    assert axes["ticks"]["B"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert axes["ticks"]["y"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for panel in rendered["panels"]:
        assert "181.86mm" in panel["tex"]
        assert (
            "xmax=%s" % (int(expected) if expected.is_integer() else "%.6g" % expected)
            in panel["tex"]
        )


def test_grid_and_domain_counts():
    points = _render()["semantic"]["points"]
    domains = {(p["station"], p["instrument"]) for p in points}
    assert len(domains) == 17
    held = {(p["station"], p["instrument"]) for p in points if p["held_comparison_domain"]}
    assert len(held) == 13
    assert len(domains) - len(held) == 4
    grid = {
        (p["station"], p["instrument"], p["policy_id"], p["estimand"], p["endpoint"], p["model_id"])
        for p in points
    }
    assert len(grid) == N_POINTS


def test_summary_grid_and_identities():
    summaries = _render()["semantic"]["summaries"]
    assert len(summaries) == N_SUMMARIES
    for row in summaries:
        assert p08_f07_data.ACTION_BY_REPRESENTATION[row["representation_id"]] == row["action"]
    keys = {(r["station"], r["instrument"], r["representation_id"], r["metric"]) for r in summaries}
    assert len(keys) == N_SUMMARIES


def test_undefined_metric_retained():
    points = _render()["semantic"]["points"]
    assert any(
        (p["peak_displacement_undefined_count"] or 0) > 0
        or (p["peak_recall_undefined_count"] or 0) > 0
        for p in points
    )
    exploratory = [p for p in points if p["held_comparison_domain"] is False]
    assert exploratory
    assert all(p["available"] is False for p in exploratory)
    assert all(p["balanced_accuracy"] is None for p in exploratory)
    assert any(p["balanced_accuracy"] is None for p in points)


def test_tex_native_and_markers():
    tex = _render()["panels"][0]["tex"]
    assert r"\documentclass[border=0pt]{standalone}" in tex
    assert "181.86mm" in tex
    assert "\\begin{minipage}" in tex
    assert "\\usepgfplotslibrary{groupplots}" in tex
    assert "\\includegraphics" not in tex
    assert "semantic-sha256:" in tex
    assert "triangle*" in tex and "square*" in tex
    assert "Peak displacement (cm$^{-1}$)" in tex
    assert "Peak recall ($\\pm$5 cm$^{-1}$)" in tex
    assert "\\definecolor{f07sg}{HTML}{0072B2}" in tex
    assert "\\definecolor{f07arpls}{HTML}{D55E00}" in tex
    assert "P08-F07 / RQ-S05" in tex
    assert "Primary (P)" in tex or "Pooled sensitivity (S)" in tex
    assert "\\addlegendentry{MIN}" in tex
    assert "\\addlegendentry{SG}" in tex
    assert "\\addlegendentry{arPLS}" in tex


def test_html_standalone_no_external_assets():
    rendered = _render()
    for panel in rendered["panels"]:
        low = panel["html"].lower()
        assert "<!doctype html>" in low
        assert "<svg" in low
        # The SVG namespace is not a remote asset and must remain allowed.
        assert 'xmlns="http://www.w3.org/2000/svg"' in low
        assert "<image" not in low and "data:image" not in low
        assert "<link" not in low and "@import" not in low and "cdn" not in low
        assert 'src="http' not in low and "src='http" not in low
        assert 'href="http' not in low and "href='http" not in low
        assert "NA=" in panel["html"]
        assert "#0072b2" in low and "#d55e00" in low


def test_html_font_floor_and_tooltip_hooks():
    rendered = _render()
    for panel in rendered["panels"]:
        low = panel["html"].lower()
        assert 'font-size="16"' in low
        assert "font-size:16px" in low
        assert "font-size:14px" in low
        assert 'id="f07-tooltip"' in low
        assert "mouseover" in low and "focusin" in low
        assert "aria-label" in low
        assert 'class="f07-pt"' in low


def test_downloads_use_frozen_csv_and_full_semantic():
    rendered = _render()
    for panel in rendered["panels"]:
        assert "data.csv" in panel["html"]
        assert "createObjectURL" in panel["html"]
        assert "revokeObjectURL" in panel["html"]
        assert 'id="f07-semantic"' in panel["html"]
    sem = _embedded_json(rendered["panels"][0]["html"], "f07-semantic")
    assert len(sem["points"]) == N_POINTS
    assert len(sem["summaries"]) == N_SUMMARIES
    assert sem["source_semantic_sha256"] == _prepared()["semantic_sha256"]


def test_coordinate_parity_records_csv_and_sources():
    rendered = _render()
    semantic = rendered["semantic"]
    sha = rendered["semantic_sha256"]

    source_points = _prepared()["semantic"]["points"]
    key_fields = ("station", "instrument", "policy_id", "estimand", "endpoint", "model_id")
    src = {tuple(p[field] for field in key_fields): p for p in source_points}
    assert len(semantic["points"]) == N_POINTS
    for point in semantic["points"]:
        ref = src[tuple(point[field] for field in key_fields)]
        for field in p08_f07_data.POINT_FIELDS:
            assert point[field] == ref[field]

    for panel in rendered["panels"]:
        assert sha in panel["tex"]
        assert sha in panel["html"]
        data = _embedded_json(panel["html"], "f07-data")
        assert data["semantic_sha256"] == sha
        assert data["csv"] == panel["csv"]  # exact frozen Python CSV string
        rows = list(csv.DictReader(io.StringIO(panel["csv"])))
        assert len(rows) == N_RECORDS_PER_PANEL == len(data["records"])
        for row, record in zip(rows, data["records"], strict=True):
            for field in p08_f07_data.POINT_FIELDS:
                expected = "" if record[field] is None else str(record[field])
                assert row[field] == expected
        plotted = 0
        for record in data["records"]:
            if record["balanced_accuracy"] is not None:
                if record["peak_displacement_median_cm1"] is not None:
                    plotted += 1
                if record["peak_recall_median"] is not None:
                    plotted += 1
        assert panel["html"].count('class="f07-pt"') == plotted


def test_refuses_missing_hash():
    prepared = copy.deepcopy(_prepared())
    del prepared["semantic_sha256"]
    with pytest.raises(ValueError):
        prepare_f07_render(prepared)


def test_refuses_hash_mismatch():
    prepared = copy.deepcopy(_prepared())
    prepared["semantic_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        prepare_f07_render(prepared)


def test_refuses_non_hex_hash():
    prepared = copy.deepcopy(_prepared())
    prepared["semantic_sha256"] = "z" * 64
    with pytest.raises(ValueError):
        prepare_f07_render(prepared)


def test_refuses_unsigned_top_level_table_override():
    prepared = copy.deepcopy(_prepared())
    prepared["points"] = copy.deepcopy(prepared["semantic"]["points"])
    prepared["summaries"] = copy.deepcopy(prepared["semantic"]["summaries"])
    with pytest.raises(ValueError):
        prepare_f07_render(prepared)


def test_refuses_unknown_semantic_key():
    prepared = _rehash(copy.deepcopy(_prepared()))
    prepared["semantic"]["extra"] = "nope"
    _rehash(prepared)
    with pytest.raises(ValueError):
        prepare_f07_render(prepared)


def test_refuses_duplicate_row_unchanged_count():
    prepared = _rehash(copy.deepcopy(_prepared()))
    points = prepared["semantic"]["points"]
    points[1] = copy.deepcopy(points[0])
    _rehash(prepared)
    with pytest.raises(ValueError):
        prepare_f07_render(prepared)


def test_refuses_bad_coordinate_values():
    for bad in (float("nan"), float("inf"), True):
        prepared = _rehash(copy.deepcopy(_prepared()))
        points = prepared["semantic"]["points"]
        index = next(i for i, point in enumerate(points) if point["held_comparison_domain"])
        points[index]["balanced_accuracy"] = bad
        with pytest.raises(ValueError):
            _rehash(prepared)
            prepare_f07_render(prepared)


def test_refuses_nonfinite_coordinate():
    prepared = _rehash(copy.deepcopy(_prepared()))
    points = prepared["semantic"]["points"]
    index = next(i for i, point in enumerate(points) if point["held_comparison_domain"])
    points[index]["peak_displacement_median_cm1"] = float("inf")
    with pytest.raises(ValueError):
        _rehash(prepared)
        prepare_f07_render(prepared)


def test_refuses_wrong_flag_types():
    for field, bad in (("held_comparison_domain", "held"), ("available", 1)):
        prepared = _rehash(copy.deepcopy(_prepared()))
        prepared["semantic"]["points"][0][field] = bad
        _rehash(prepared)
        with pytest.raises(ValueError):
            prepare_f07_render(prepared)


def test_refuses_flag_disagreement():
    prepared = _rehash(copy.deepcopy(_prepared()))
    points = prepared["semantic"]["points"]
    index = next(i for i, point in enumerate(points) if point["held_comparison_domain"])
    points[index]["available"] = False
    _rehash(prepared)
    with pytest.raises(ValueError):
        prepare_f07_render(prepared)


def test_refuses_unknown_policy_and_grid_gap():
    prepared = _rehash(copy.deepcopy(_prepared()))
    prepared["semantic"]["points"][0]["policy_id"] = "PP-U-UNKNOWN"
    _rehash(prepared)
    with pytest.raises(ValueError):
        prepare_f07_render(prepared)


def test_policies_cover_all_three():
    assert set(POLICIES) == {"PP-U-MIN", "PP-U-SG", "PP-U-ARPLS"}
    rendered = _render()
    for panel in rendered["panels"]:
        for policy in POLICIES:
            assert 'data-policy="%s"' % policy in panel["html"]
