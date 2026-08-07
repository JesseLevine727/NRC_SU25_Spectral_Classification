from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path

from atlas_sers.governance.runs import RunIdentity

PROJECT = Path(__file__).resolve().parents[1]
PLAN = PROJECT / "plan"


def _json(name: str) -> dict[str, object]:
    return json.loads((PLAN / "contracts" / name).read_text())


def _csv(name: str) -> list[dict[str, str]]:
    with (PLAN / "registries" / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_parallel_rq_hierarchy_preserves_one_primary_question() -> None:
    research = _json("research_contract.json")
    hierarchy = research["research_question_hierarchy"]
    questions = {row["research_question_id"]: row for row in _csv("research_question_registry.csv")}

    assert hierarchy["primary"] == ["RQ-P01"]
    assert set(questions) == {
        "RQ-P01",
        "RQ-S01",
        "RQ-S02",
        "RQ-S03",
        "RQ-S04",
        "RQ-S05",
        "RQ-S06",
        "RQ-E01",
    }
    assert questions["RQ-P01"]["preprocessing_policy_ids"] == "PP-U-MIN"
    assert research["primary_effect"].startswith("acquisition-aware deep")


def test_policy_actions_are_frozen_p01_representations_on_one_scale() -> None:
    policy = _json("preprocessing_policy_contract.json")
    p01 = _json("p01_governance_contract.json")
    frozen = {item["representation_id"] for item in p01["representations"]}

    assert set(policy["candidate_actions"]) <= frozen
    assert policy["common_output_contract"]["row_scale"] == "[0,1]"
    assert policy["common_output_contract"]["n_features"] == 1401
    assert policy["common_output_contract"]["composition_rule"].startswith(
        "Policies select only among immutable P01 arrays"
    )


def test_family_policy_is_source_only_with_explicit_support_and_fallback() -> None:
    family = _json("preprocessing_policy_contract.json")["family_aware_policy"]

    assert family["support_rule"]["minimum_distinct_supported_source_units"] == 2
    threshold = family["support_rule"]["minimum_masters_per_class"]
    assert threshold["status"] == "unresolved_until_P02_metadata_support_audit"
    assert threshold["candidate_values"] == [2, 3, 4]
    assert family["fallbacks"]["unknown_family"] == "PP-U-MIN"
    assert family["fallbacks"]["known_but_unsupported_family"] == "PP-U-MIN"
    assert "held-unit outcomes" in family["forbidden_target_information"]


def test_qc_policy_is_identity_blind_row_local_and_source_frozen() -> None:
    qc = _json("preprocessing_policy_contract.json")["qc_adaptive_policy"]
    forbidden = set(qc["forbidden_features"])

    assert "instrument or serial identity" in forbidden
    assert "instrument platform family" in forbidden
    assert "batch or target-population aggregates" in forbidden
    assert qc["gate_library"]["source_quantile_candidates"] == [0.5, 0.75, 0.9]
    assert qc["gate_library"]["threshold_status"].startswith("numerical cut points unresolved")
    assert qc["fallbacks"]["missing_or_nonfinite_qc"] == "R_MIN_400_1800"


def test_post_test_instrument_hybrid_is_machine_readably_prohibited() -> None:
    policies = {row["policy_id"]: row for row in _csv("preprocessing_policy_registry.csv")}
    policy = _json("preprocessing_policy_contract.json")

    assert policies["PP-POSTTEST-HYBRID"]["status"] == "prohibited"
    assert policy["factorial_evaluation"]["prohibited_cell"].startswith(
        "Any combination created after test outcomes"
    )
    assert any("Held-test labels" in rule for rule in policy["fatal_leakage_rules"])


def test_prediction_schema_identifies_question_policy_action_and_access() -> None:
    schema = _json("result_schema.json")
    required = set(schema["required"])

    assert {
        "research_question_id",
        "preprocessing_information_regime",
        "target_access_role",
        "preprocessing_policy_id",
        "preprocessing_action_representation_id",
        "preprocessing_policy_sha256",
        "preprocessing_fallback_reason",
        "preprocessing_policy_support_status",
        "preprocessing_gate_candidate_id",
        "instrument_family",
    } <= required


def test_run_identity_contract_separates_question_policy_and_model() -> None:
    p00 = _json("p00_governance_contract.json")
    identity_fields = [field.name for field in fields(RunIdentity)]

    assert identity_fields == p00["run_identity_fields"]
    assert {
        "research_question_id",
        "preprocessing_information_regime",
        "preprocessing_policy_id",
        "model_id",
    } <= set(identity_fields)
