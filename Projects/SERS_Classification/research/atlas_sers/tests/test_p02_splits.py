from __future__ import annotations

import json
from pathlib import Path

from atlas_sers.splits.p02 import build_p02_tables, validate_p02_tables
from tests.p02_fixtures import synthetic_manifest

PROJECT = Path(__file__).resolve().parents[1]


def _contracts():
    root = PROJECT / "plan" / "contracts"
    return (
        json.loads((root / "split_contract.json").read_text()),
        json.loads((root / "preprocessing_policy_contract.json").read_text()),
        json.loads((root / "p02_governance_contract.json").read_text()),
    )


def test_p02_tables_are_deterministic_complete_and_leakage_free() -> None:
    manifest = synthetic_manifest()
    split, policy, p02 = _contracts()
    first = build_p02_tables(manifest, split, policy, p02)
    second = build_p02_tables(manifest, split, policy, p02)
    assert set(first) == set(p02["protected_payloads"])
    for name in first:
        assert first[name].equals(second[name]), name
    checks = validate_p02_tables(first, manifest, split, p02)
    assert all(checks.values()), {name: value for name, value in checks.items() if not value}
    assert len(first["master_split_registry.csv"]) == 69 * 5
    assert len(first["preprocessing_policy_roles.csv"]) == 13 * 5 * 4
    assert len(first["qc_gate_candidate_registry.csv"]) == 124
    t3 = first["t3_partition_registry.csv"]
    assert not t3[t3.role == "train_source"].instrument.eq(
        t3[t3.role == "train_source"].held_instrument
    ).any()
    assert first["leakage_audit.csv"].query("severity == 'fatal'").status.eq("pass").all()


def test_target_access_and_open_set_roles_are_master_disjoint() -> None:
    manifest = synthetic_manifest()
    split, policy, p02 = _contracts()
    tables = build_p02_tables(manifest, split, policy, p02)
    assignments = tables["target_access_assignment_registry.csv"]
    for _, group in assignments.groupby("scenario_id"):
        calibration = set(
            group[
                group.target_access_role.isin(
                    [
                        "unlabeled_adaptation_master",
                        "paired_calibration_master",
                        "labelled_calibration_master",
                    ]
                )
            ].master_sample_id
        )
        evaluation = set(group[group.target_access_role == "evaluation_only"].master_sample_id)
        assert calibration.isdisjoint(evaluation)
        assert group.master_sample_id.is_unique
    open_set = tables["open_set_partition_registry.csv"]
    for _, group in open_set.groupby("open_partition_id"):
        held = group.held_target.iloc[0]
        assert not group[group.role == "train_known"].target_analyte.eq(held).any()


def test_qc_registry_contains_only_declared_features_and_no_numeric_cutpoints() -> None:
    manifest = synthetic_manifest()
    split, policy, p02 = _contracts()
    tables = build_p02_tables(manifest, split, policy, p02)
    gates = tables["qc_gate_candidate_registry.csv"]
    assert gates.gate_kind.value_counts().to_dict() == {
        "dual_trigger": 108,
        "single_trigger": 15,
        "baseline": 1,
    }
    assert gates.numeric_cutpoints_status.eq("future_fold_local_source_training").all()
    roles = tables["preprocessing_policy_roles.csv"]
    forbidden = {"instrument", "sensor_family", "target_analyte", "master_sample_id"}
    assert forbidden.isdisjoint(roles.columns)
    assert roles.test_outcomes_used.eq(False).all()  # noqa: E712
