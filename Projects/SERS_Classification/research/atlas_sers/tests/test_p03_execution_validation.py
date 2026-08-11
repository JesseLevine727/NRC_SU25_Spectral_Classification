from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from atlas_sers.governance.canonical import sha256_value
from atlas_sers.governance.p03_execution import _validate_final_aggregation_directory
from atlas_sers.visualization.p03_figures import FIGURE_SLUGS


def test_final_execution_validator_reconciles_every_required_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "atlas_sers.governance.p03_execution.EXPECTED_FIT_MANIFEST_ROWS", 1
    )
    monkeypatch.setattr(
        "atlas_sers.governance.p03_execution.EXPECTED_FINAL_ENDPOINTS", 1
    )
    context = SimpleNamespace(
        execution_run_id="P03-test",
        execution_protected_state_sha256="a" * 64,
        fit_manifest=pd.DataFrame([{"fit_id": "fit-1"}]),
    )
    terminal = pd.DataFrame([{"fit_id": "fit-1", "status": "complete"}])
    terminal.to_parquet(tmp_path / "terminal_fit_ledger.parquet", index=False)
    endpoint = pd.DataFrame(
        [
            {
                "experiment_id": "EXP-C09-T3",
                "task_id": "T3-ZS",
                "domain": "cwa:unit",
                "station": "cwa",
                "procedure_id": "C-SELECTED",
                "outer_repeat": 1,
                "outer_fold": 0,
                "outer_run_id": "outer-1",
            }
        ]
    )
    endpoint.to_parquet(tmp_path / "expected_endpoint_registry.parquet", index=False)
    pd.DataFrame(
        [
            {"endpoint_status": "complete", "aggregation_level": level}
            for level in ("spectrum", "instrument_balanced_master")
        ]
    ).to_csv(tmp_path / "pooled_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "figure_id": figure_id,
                "semantic_parity": True,
                "compiled": True,
                "native_tikz": True,
                "standalone_html": True,
                "colorblind_safe": True,
                "vector_only": True,
                "png_dpi": 300,
            }
            for figure_id in FIGURE_SLUGS
        ]
    ).to_csv(tmp_path / "figure_manifest.csv", index=False)
    (tmp_path / "prediction_schema_validation.json").write_text(
        json.dumps({"status": "pass", "validated_rows": 1})
    )
    (tmp_path / "p04_comparator_freeze.json").write_text(
        json.dumps({"source_execution_run_id": "P03-test", "outer_cell_count": 260})
    )
    (tmp_path / "P03_REPORT.md").write_text("report")
    (tmp_path / "P04_HANDOFF.md").write_text("handoff")
    (tmp_path / "final_aggregation_descriptor.json").write_text(
        json.dumps(
            {
                "execution_run_id": "P03-test",
                "protected_state_sha256": "a" * 64,
                "terminal_fit_id_sha256": sha256_value(["fit-1"]),
                "final_prediction_rows": 1,
            }
        )
    )
    report = _validate_final_aggregation_directory(tmp_path, context=context)
    assert report["status"] == "pass"
    assert all(report["checks"].values())
    json.dumps(report)
