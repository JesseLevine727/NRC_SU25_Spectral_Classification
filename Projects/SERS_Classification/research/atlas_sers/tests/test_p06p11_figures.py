"""Focused tests for the T035 pure renderer (no I/O, no metric maths)."""

# Match the renderer's literal-brace TeX string convention.
# ruff: noqa: UP031

import hashlib
import subprocess

import pandas as pd
import pytest

from atlas_sers.visualization import p06p11_figures as figures

COLUMNS = [
    "figure_id",
    "panel",
    "label",
    "series",
    "x",
    "y",
    "lower",
    "upper",
    "domain",
    "instrument",
    "complete_contexts",
    "remaining_domains",
    "color",
    "marker",
]

NAN = float("nan")


def _row(**values):
    row = {name: NAN for name in COLUMNS}
    row["complete_contexts"] = 0
    row["remaining_domains"] = 0
    row.update(values)
    return row


def _frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def _specs():
    primary = []
    for panel in ("M01", "M06"):
        for series, color, marker in (
            ("cwa", "#0072B2", "circle"),
            ("pills", "#D55E00", "square"),
        ):
            primary.append(
                _row(
                    figure_id="F_P06_primary_scatter",
                    panel=panel,
                    label="ctx & <b>%s</b>" % series,
                    series=series,
                    x=10.5,
                    y=20.0,
                    lower=NAN,
                    upper=NAN,
                    domain="D1",
                    instrument="I1",
                    complete_contexts=3,
                    remaining_domains=5,
                    color=color,
                    marker=marker,
                )
            )

    effect = []
    for pi, panel in enumerate(("M01", "M06")):
        for j in range(3):
            effect.append(
                _row(
                    figure_id="F_P06_effect_intervals",
                    panel=panel,
                    label="Row %d & B_C 50%%" % j,
                    series="conditional",
                    x=50.0 if (pi == 0 and j == 0) else 5.0,
                    y=3 - j,
                    lower=10.0 if (pi == 0 and j == 0) else 3.0,
                    upper=20.0 if (pi == 0 and j == 0) else 7.0,
                    domain="D",
                    instrument="I",
                    complete_contexts=4,
                    remaining_domains=2,
                    color="#009E73",
                    marker="diamond",
                )
            )

    weight = []
    for panel in ("M01", "M06"):
        for j in range(2):
            weight.append(
                _row(
                    figure_id="F_P06_weight_sensitivity",
                    panel=panel,
                    label="W%d" % j,
                    series="weight",
                    x=5.0,
                    y=2 - j,
                    lower=2.5,
                    upper=12.5,
                    domain="D",
                    instrument="I",
                    complete_contexts=6,
                    remaining_domains=1,
                    color="#CC79A7",
                    marker="triangle-up",
                )
            )

    deletion = []
    for i in range(2):
        deletion.append(
            _row(
                figure_id="F_P06_deletion_stability",
                panel="joint",
                label="d%d" % i,
                series="removed_domain",
                x=3.0 + i,
                y=-2.0 - i,
                lower=NAN,
                upper=NAN,
                domain="D%d" % i,
                instrument="I",
                complete_contexts=1,
                remaining_domains=i,
                color="#0072B2",
                marker="circle",
            )
        )
    for i in range(2):
        deletion.append(
            _row(
                figure_id="F_P06_deletion_stability",
                panel="joint",
                label="i%d" % i,
                series="removed_instrument",
                x=1.0 + i,
                y=1.0 + i,
                lower=NAN,
                upper=NAN,
                domain="D",
                instrument="I%d" % i,
                complete_contexts=1,
                remaining_domains=1,
                color="#E69F00",
                marker="diamond",
            )
        )
    deletion.append(
        _row(
            figure_id="F_P06_deletion_stability",
            panel="joint",
            label="full",
            series="full_reference",
            x=0.0,
            y=0.0,
            lower=NAN,
            upper=NAN,
            domain="D",
            instrument="I",
            complete_contexts=1,
            remaining_domains=1,
            color="#000000",
            marker="cross",
        )
    )

    return {
        "F_P06_primary_scatter": {
            "semantic": _frame(primary),
            "title": "Primary scatter",
            "caption": "Primary caption.",
            "kind": "scatter",
        },
        "F_P06_effect_intervals": {
            "semantic": _frame(effect),
            "title": "Effect & intervals",
            "caption": "Caption 50% _x_ <b>bold</b> & more.",
            "kind": "interval",
        },
        "F_P06_weight_sensitivity": {
            "semantic": _frame(weight),
            "title": "Weight sensitivity",
            "caption": "Original hierarchy unavailable from the builder.",
            "kind": "interval",
        },
        "F_P06_deletion_stability": {
            "semantic": _frame(deletion),
            "title": "Deletion stability",
            "caption": "Deletion caption.",
            "kind": "scatter",
        },
    }


@pytest.fixture
def specs():
    return _specs()


@pytest.fixture
def built(monkeypatch, specs):
    monkeypatch.setattr(figures, "build_semantics", lambda tables: specs)
    return figures.build_figures({})


def test_figure_ids_are_fixed():
    assert figures.FIGURE_IDS == (
        "F_P06_primary_scatter",
        "F_P06_effect_intervals",
        "F_P06_weight_sensitivity",
        "F_P06_deletion_stability",
    )


def test_equal_aspect_scatter_constrains_domains_not_data_ranges(specs):
    for fid in ("F_P06_primary_scatter", "F_P06_deletion_stability"):
        fig = figures._html_scatter(fid, specs[fid]["semantic"], "title")
        assert fig.layout.xaxis.constrain == "domain"
        assert fig.layout.yaxis.constrain == "domain"


def test_returns_unchanged_semantic_and_matching_hash(built, specs):
    for fid in figures.FIGURE_IDS:
        entry = built[fid]
        assert entry["semantic"] is specs[fid]["semantic"]
        expected = hashlib.sha256(
            entry["semantic"].to_csv(index=False, lineterminator="\n").encode("utf-8")
        ).hexdigest()
        assert entry["sha256"] == expected
        assert entry["title"] == specs[fid]["title"]
        assert entry["caption"] == specs[fid]["caption"]


def test_primary_tex_is_native_complete_scatter(built):
    tex = built["F_P06_primary_scatter"]["tex"]
    assert r"\documentclass[tikz,border=5pt]{standalone}" in tex
    assert r"\usepackage{pgfplots}" in tex
    assert r"\usepgfplotslibrary{groupplots}" in tex
    assert r"\pgfplotsset{compat=1.18}" in tex
    assert r"\begin{groupplot}" in tex and r"\end{groupplot}" in tex
    assert r"\includegraphics" not in tex
    assert r"% sha256: " + built["F_P06_primary_scatter"]["sha256"] in tex
    assert r"\definecolor{p06color0}{HTML}{0072B2}" in tex
    assert "(0,0) (100,100)" in tex
    assert "A Individual spectra" in tex
    assert "B Combined sample predictions" in tex
    assert r"\addlegendentry{cwa}" in tex
    assert r"\addlegendentry{pills}" in tex
    assert "mark=square*" in tex
    assert "errorbar" not in tex.lower()


def test_deletion_tex_zero_lines_equal_scale_and_legend(built):
    tex = built["F_P06_deletion_stability"]["tex"]
    assert "axis equal" in tex
    assert r"\addplot[forget plot, black, thin] coordinates {(0," in tex
    assert r"\addlegendentry{removed domain}" in tex
    assert r"\addlegendentry{removed instrument}" in tex
    assert r"\addlegendentry{full reference}" in tex


def test_interval_tex_endpoints_and_dot_outside_interval(built):
    tex = built["F_P06_effect_intervals"]["tex"]
    assert "(10,3) (20,3)" in tex
    assert "(50,3)" in tex
    assert "(0,0.4) (0,3.6)" in tex
    assert "Balanced accuracy difference (percentage points)" in tex


def test_tex_escapes_special_text_without_double_escaping(built):
    tex = built["F_P06_effect_intervals"]["tex"]
    assert r"Row 0 \& B\_C 50\%" in tex
    assert r"Effect \& intervals" in tex
    assert r"50\% \_x\_" in tex
    assert "Row 0 & B_C 50%" not in tex
    assert "Caption 50% _x_" not in tex
    assert r"\\&" not in tex


def test_html_is_self_contained_and_caption_visible(built):
    for fid, entry in built.items():
        document = entry["html"]
        assert "<html" in document and "</html>" in document
        assert 'id="%s"' % fid in document
        assert "plotly" in document
        assert "<script src=" not in document
        assert 'id="%s-caption"' % fid in document
        assert entry["sha256"] in document
        assert document.index(entry["sha256"]) < document.index("</body>")


def test_html_escapes_special_text(built):
    document = built["F_P06_effect_intervals"]["html"]
    assert "&lt;b&gt;bold&lt;/b&gt;" in document
    assert "Caption 50% _x_ <b>bold</b> & more." not in document
    assert "&amp;" in document


def test_outputs_are_deterministic_with_stable_div_id(monkeypatch):
    monkeypatch.setattr(figures, "build_semantics", lambda tables: _specs())
    first = figures.build_figures({})
    second = figures.build_figures({})
    for fid in figures.FIGURE_IDS:
        assert first[fid]["sha256"] == second[fid]["sha256"]
        assert first[fid]["tex"] == second[fid]["tex"]
        assert first[fid]["html"] == second[fid]["html"]


def test_values_are_not_rescaled_by_one_hundred(built):
    effect = built["F_P06_effect_intervals"]["tex"]
    assert "(10,3) (20,3)" in effect
    assert "1000" not in effect and "2000" not in effect
    weight = built["F_P06_weight_sensitivity"]["tex"]
    assert "12.5" in weight
    assert "1250" not in weight
    primary = built["F_P06_primary_scatter"]["tex"]
    assert "(10.5,20)" in primary
    assert "1050" not in primary and "2000" not in primary


def test_build_does_not_shell_out(monkeypatch):
    calls = []

    def _boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("unexpected subprocess use")

    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, _boom)
    monkeypatch.setattr(figures, "build_semantics", lambda tables: _specs())
    result = figures.build_figures({})
    assert set(result) == set(figures.FIGURE_IDS)
    assert calls == []


def test_interval_tex_uses_semantic_y_ticks(built):
    effect = built["F_P06_effect_intervals"]["tex"]
    assert "ytick={1,2,3}" in effect
    assert ", ymin=0.4, ymax=3.6" in effect
    weight = built["F_P06_weight_sensitivity"]["tex"]
    assert "ytick={1,2}" in weight
    assert ", ymin=0.4, ymax=2.6" in weight
    assert "xmajorgrids=true" in effect
    assert "ymajorgrids=false" in effect
    assert "grid=x" not in effect


def test_interval_html_uses_semantic_y_ticks(built):
    document = built["F_P06_effect_intervals"]["html"]
    assert "A Individual spectra" in document
    assert "B Combined sample predictions" in document
    assert "Balanced accuracy difference (percentage points)" in document
    assert "x: 50" in document
    assert "lower: 10" in document
    assert "upper: 20" in document
    assert "x: 500" not in document
    assert "lower: 1000" not in document
    assert "upper: 2000" not in document


def test_interval_html_human_panel_titles_for_weights(built):
    document = built["F_P06_weight_sensitivity"]["html"]
    assert "A Individual spectra" in document
    assert "B Combined sample predictions" in document
    assert "lower: 1000" not in document


def test_scatter_html_hover_includes_xy_and_axis_titles(built):
    primary = built["F_P06_primary_scatter"]["html"]
    assert "x: 10.5" in primary
    assert "y: 20" in primary
    assert "Selected classical balanced accuracy (%)" in primary
    assert "Selected CNN balanced accuracy (%)" in primary
    deletion = built["F_P06_deletion_stability"]["html"]
    assert "M01 balanced accuracy difference (percentage points)" in deletion
    assert "M06 balanced accuracy difference (percentage points)" in deletion


def test_interval_coordinates_follow_semantic_y_when_shuffled(specs):
    effect = (
        specs["F_P06_effect_intervals"]["semantic"]
        .sample(frac=1, random_state=0)
        .reset_index(drop=True)
    )
    tex = figures._tex_interval("F_P06_effect_intervals", effect, "t", "c", "0" * 64)
    assert "(50,3)" in tex
    assert "(0,0.4) (0,3.6)" in tex
    fig = figures._html_interval("F_P06_effect_intervals", effect, "t")
    ys = sorted(float(trace.y[0]) for trace in fig.data)
    assert ys == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    horizontal = {
        float(shape.y0)
        for shape in fig.layout.shapes
        if shape.type == "line" and float(shape.x0) != float(shape.x1)
    }
    assert horizontal == {1.0, 2.0, 3.0}
