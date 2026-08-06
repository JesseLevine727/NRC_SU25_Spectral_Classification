from __future__ import annotations

import json
from pathlib import Path

from atlas_sers.governance.p01 import p01_dry_run

PROJECT = Path(__file__).resolve().parents[1]


def test_p01_contract_declares_exact_representations_figures_and_no_training() -> None:
    contract = json.loads(
        (PROJECT / "plan" / "contracts" / "p01_governance_contract.json").read_text()
    )
    assert len(contract["representations"]) == 8
    assert contract["required_figures"] == [f"F{index:02d}" for index in range(2, 10)]
    assert contract["model_fitting_authorized"] is False
    assert contract["split_construction_authorized"] is False
    report = p01_dry_run(PROJECT)
    assert report["status"] == "pass"
    assert report["p02_fields"] == "explicitly unresolved"
