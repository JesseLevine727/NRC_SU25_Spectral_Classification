from __future__ import annotations

import csv
import json
from pathlib import Path

from jsonschema import Draft202012Validator

PROJECT = Path(__file__).resolve().parents[1]
PLAN = PROJECT / "plan"


def _rows(name: str) -> list[dict[str, str]]:
    with (PLAN / "registries" / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_p02_contract_freezes_splits_access_and_no_model_boundary() -> None:
    contracts = PLAN / "contracts"
    split = json.loads((contracts / "split_contract.json").read_text())
    p02 = json.loads((contracts / "p02_governance_contract.json").read_text())
    assert split["outer_repeat_seeds"] == [
        20260805,
        20260817,
        20260829,
        20260910,
        20260922,
    ]
    assert split["outer_folds_per_station"] == 4
    assert len(split["primary_domain_eligibility"]["domains"]) == 13
    assert len(split["exploratory_low_support_domains"]) == 4
    assert p02["predictive_model_fitting_authorized"] is False
    assert p02["inner_master_folds"] == 3
    assert p02["qc_gate_enumeration"]["expected_candidates"] == 124
    assert p02["target_access_draws"]["supervised_few_shot_masters_per_class"] == [
        1,
        2,
        3,
        5,
    ]
    assert set(p02["protected_payloads"]) < set(p02["required_top_level_outputs"])


def test_p02_validation_schema_accepts_only_no_fit_report_shape() -> None:
    schema = json.loads(
        (PLAN / "contracts" / "p02_validation_schema.json").read_text()
    )
    instance = {
        "schema_version": "p02-validation-report-v1",
        "protocol_version": "atlas-sers-research-master-plan-v1",
        "phase": "P02",
        "status": "pass",
        "run_id": "P02-" + "a" * 24,
        "protected_state_sha256": "b" * 64,
        "checks": {"example": True},
        "diagnostics": {},
        "fit_invocations": 0,
        "split_invocations": 275,
        "primary_domains": 13,
        "outer_repeats": 5,
        "outer_folds": 4,
        "figures": ["F10", "F11"],
    }
    assert not list(Draft202012Validator(schema).iter_errors(instance))
    instance["fit_invocations"] = 1
    assert list(Draft202012Validator(schema).iter_errors(instance))


def test_p02_registries_are_complete_frozen_and_run_scoped() -> None:
    phase = {row["phase_id"]: row for row in _rows("phase_registry.csv")}
    models = {row["model_id"]: row for row in _rows("model_registry.csv")}
    figures = {row["figure_id"]: row for row in _rows("figure_registry.csv")}
    artifacts = {row["artifact_id"]: row for row in _rows("artifact_registry.csv")}
    assert phase["P02"]["execution_status"] == "complete"
    assert all(
        models[name]["status"] == "frozen"
        for name in ("SYS-OUTER-SPLIT", "SYS-T3-PARTITION", "SYS-PP-SUPPORT")
    )
    assert figures["F10"]["status"] == figures["F11"]["status"] == "complete"
    expected = {
        "ART-P02-MASTER-SPLITS",
        "ART-P02-T3-PARTITIONS",
        "ART-P02-PP-ROLES",
        "ART-P02-TARGET-ACCESS",
        "ART-P02-OPEN-ROLES",
        "ART-P02-VALIDATION",
    }
    assert expected <= set(artifacts)
    assert all("/p02/runs/<run_id>/" in artifacts[name]["logical_path"] for name in expected)
