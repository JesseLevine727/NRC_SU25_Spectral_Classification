import json
from pathlib import Path

import pandas as pd

from atlas_sers.evaluation.p13_plan import ELIGIBLE_TIERS, POLICY_REPRESENTATIONS
from atlas_sers.governance.canonical import sha256_file

PLAN = Path(__file__).resolve().parents[1] / "plan"


def test_p13_contract_and_support_freeze_are_consistent() -> None:
    contract = json.loads((PLAN / "contracts/p13_execution_contract.json").read_text())
    summary = json.loads(
        (PLAN / "registries/p13_support_freeze_summary.json").read_text()
    )
    domains_path = PLAN / "registries/p13_domain_support_registry.csv"
    crossovers_path = PLAN / "registries/p13_crossover_support_registry.csv"
    domains = pd.read_csv(domains_path)
    crossovers = pd.read_csv(crossovers_path)
    assert contract["protocol_version"] == "nato-sers-p13-v1-locked"
    assert set(contract["population"]["eligible_domain_tiers"]) == ELIGIBLE_TIERS
    assert set(contract["preprocessing"]) == set(POLICY_REPRESENTATIONS)
    assert domains.support_tier.value_counts().to_dict() == {
        "unsupported_by_design": 18,
        "confirmatory": 13,
        "exploratory_low_support": 3,
    }
    assert crossovers.support_tier.value_counts().to_dict() == {
        "descriptive_singleton": 19,
        "confirmatory": 8,
        "exploratory_low_support": 7,
    }
    assert summary["registry_hashes"]["p13_domain_support_registry_sha256"] == (
        sha256_file(domains_path)
    )
    assert summary["registry_hashes"]["p13_crossover_support_registry_sha256"] == (
        sha256_file(crossovers_path)
    )


def test_p13_candidate_and_crossover_rules_are_fully_resolved() -> None:
    contract = json.loads((PLAN / "contracts/p13_execution_contract.json").read_text())
    assert contract["procedure_order"][0] == "C-SELECTED"
    assert contract["candidate_resolution"]["unrepresented_exploratory_domain"].startswith(
        "When no P03 source-only selection exists"
    )
    assert "correct_B,instrument_B" in contract["crossover"]["correctness_contrast"]
    assert contract["field_log"]["unit"] == "master-substrate-instrument view"
