from __future__ import annotations

from pathlib import Path

import pandas as pd

from atlas_sers.evaluation.p03_analysis import (
    P03DiagnosticTables,
    build_confusion_summary,
    build_control_summary,
    build_endpoint_coverage,
    build_reliability_summary,
    build_selection_diagnostics,
    build_spectrum_master_comparison,
    build_t1_t3_comparison,
)
from atlas_sers.visualization.p03_figures import (
    build_p03_figure_definitions,
    generate_p03_figures,
)
from tests.test_p03_analysis import (
    _metrics,
    _prediction_rows,
    _selection_evidence,
)


def _diagnostics() -> tuple[pd.DataFrame, P03DiagnosticTables]:
    metrics = _metrics()
    selections, traces = _selection_evidence()
    frequency, stability, margins = build_selection_diagnostics(selections, traces)
    diagnostics = P03DiagnosticTables(
        selection_frequency=frequency,
        selection_stability=stability,
        selection_margins=margins,
        endpoint_coverage=build_endpoint_coverage(metrics),
        t1_t3_comparison=build_t1_t3_comparison(metrics),
        spectrum_master_comparison=build_spectrum_master_comparison(metrics),
        confusion=build_confusion_summary(metrics),
        reliability=build_reliability_summary(
            _prediction_rows(True), _prediction_rows(False)
        ),
        control_summary=build_control_summary(metrics),
        cost_summary=pd.DataFrame(),
    )
    return metrics, diagnostics


def test_p03_figure_forms_are_native_paired_vector_and_reproducible(
    tmp_path: Path,
) -> None:
    metrics, diagnostics = _diagnostics()
    definitions = build_p03_figure_definitions(
        pooled_metrics=metrics, diagnostics=diagnostics
    )
    assert set(definitions) == {
        "F12",
        "F13",
        "F38",
        "F39",
        "F40",
        "F41",
        "F42",
        "F43",
    }
    first = generate_p03_figures(tmp_path / "first", definitions)
    second = generate_p03_figures(tmp_path / "second", definitions)
    assert first.data_sha256.tolist() == second.data_sha256.tolist()
    assert first.tikz_sha256.tolist() == second.tikz_sha256.tolist()
    assert first.pdf_sha256.tolist() == second.pdf_sha256.tolist()
    assert first.html_sha256.tolist() == second.html_sha256.tolist()
    assert first[
        [
            "semantic_parity",
            "compiled",
            "native_tikz",
            "standalone_html",
            "colorblind_safe",
            "vector_only",
        ]
    ].all(axis=None)
    assert first.png_dpi.eq(300).all()
    for row in first.itertuples(index=False):
        root = tmp_path / "first"
        assert (root / row.png_path).stat().st_size > 0
        assert "\\includegraphics" not in (root / row.tikz_path).read_text()
        html = (root / row.html_path).read_text()
        assert "cdn.plot.ly" not in html
        assert "Frozen plot-data SHA-256" in html
