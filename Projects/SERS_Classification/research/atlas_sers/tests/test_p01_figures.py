from __future__ import annotations

import pandas as pd

from atlas_sers.governance.canonical import sha256_file
from atlas_sers.visualization.p01_figures import Panel, generate_p01_figures


def _table() -> dict:
    frame = pd.DataFrame(
        {"stage": ["all", "selected", "clear"], "count": [12, 8, 6], "series": "rows"}
    )
    return {
        "F02": (
            "Synthetic flow",
            frame,
            [Panel("Flow", "stage", "count", "series", "Stage", "Count", "bar")],
        )
    }


def test_tikz_pdf_html_and_table_are_semantically_paired_and_reproducible(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = generate_p01_figures(first_root, _table())
    second = generate_p01_figures(second_root, _table())
    assert first[["semantic_parity", "compiled", "native_tikz", "standalone_html"]].all(axis=None)
    assert second[["semantic_parity", "compiled", "native_tikz", "standalone_html"]].all(axis=None)
    for relative in (
        first.iloc[0].data_path,
        first.iloc[0].tikz_path,
        first.iloc[0].pdf_path,
        first.iloc[0].html_path,
    ):
        assert sha256_file(first_root / relative) == sha256_file(second_root / relative)
