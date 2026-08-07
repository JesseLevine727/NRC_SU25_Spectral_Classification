from __future__ import annotations

import json
from pathlib import Path

from atlas_sers.splits.p02 import build_p02_tables
from atlas_sers.visualization.p02_figures import (
    build_p02_figure_tables,
    generate_p02_figures,
)
from tests.p02_fixtures import synthetic_manifest

PROJECT = Path(__file__).resolve().parents[1]


def test_p02_figure_forms_are_paired_native_and_reproducible(tmp_path: Path) -> None:
    contracts = PROJECT / "plan" / "contracts"
    split = json.loads((contracts / "split_contract.json").read_text())
    policy = json.loads((contracts / "preprocessing_policy_contract.json").read_text())
    p02 = json.loads((contracts / "p02_governance_contract.json").read_text())
    tables = build_p02_tables(synthetic_manifest(), split, policy, p02)
    figure_tables = build_p02_figure_tables(tables)
    first = generate_p02_figures(tmp_path / "first", figure_tables)
    second = generate_p02_figures(tmp_path / "second", figure_tables)
    assert set(first.figure_id) == {"F10", "F11"}
    assert first.data_sha256.tolist() == second.data_sha256.tolist()
    assert first.tikz_sha256.tolist() == second.tikz_sha256.tolist()
    assert first.html_sha256.tolist() == second.html_sha256.tolist()
    assert first.pdf_sha256.tolist() == second.pdf_sha256.tolist()
    assert first[["semantic_parity", "compiled", "native_tikz", "standalone_html"]].all(
        axis=None
    )
    assert first.png_dpi.eq(300).all()
    for row in first.itertuples(index=False):
        assert (tmp_path / "first" / row.png_path).stat().st_size > 0
        assert "\\includegraphics" not in (tmp_path / "first" / row.tikz_path).read_text()
        html = (tmp_path / "first" / row.html_path).read_text()
        assert "cdn.plot.ly" not in html
        assert "Frozen plot-data SHA-256" in html
