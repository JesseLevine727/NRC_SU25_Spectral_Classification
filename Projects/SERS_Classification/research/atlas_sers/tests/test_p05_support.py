from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

import atlas_sers.evaluation.p05_support as p05_support
from atlas_sers.evaluation.p05_readiness import build_readiness_report
from atlas_sers.evaluation.p05_support import (
    AUDIT_STATUS,
    CONTEXT_COLUMNS,
    MANIFEST_COLUMNS,
    ROLE_COLUMNS,
    SCHEMA_VERSION,
    SupportAuditError,
    audit_identity,
    build_support_inputs,
    build_support_report,
    load_support_inputs,
    main,
    pair_identity,
    render_report,
    sha256_value,
    uid_set_hash,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "atlas_sers" / "evaluation" / "p05_support.py"
WRAPPER_PATH = PROJECT_ROOT / "scripts" / "audit_p05_support.py"

FORBIDDEN_IMPORT_ROOTS = {"torch", "numpy", "pandas", "sklearn", "scipy", "matplotlib"}


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _t1_specs() -> list[tuple[str, str, str, str, str, str]]:
    """(uid, master, station, chemical, instrument, family)."""

    return [
        ("u1", "M1", "cwa", "A", "instr1", "fam1"),
        ("u2", "M1", "cwa", "A", "instr1", "fam1"),
        ("u3", "M2", "cwa", "A", "instr1", "fam1"),
        ("u4", "M3", "cwa", "A", "instr2", "fam2"),
        ("u5", "M4", "cwa", "A", "instr1", "fam1"),
        ("u6", "M4", "cwa", "A", "instr2", "fam2"),
        ("u7", "M5", "cwa", "B", "instr1", "fam1"),
        ("u8", "M6", "cwa", "B", "instr1", "fam1"),
        ("u9", "M7", "cwa", "A", "instr1", "fam1"),
        ("u10", "M8", "cwa", "A", "instr1", "fam1"),
        ("u11", "M9", "cwa", "A", "instr1", "fam1"),
    ]


def _t3_specs() -> list[tuple[str, str, str, str, str, str]]:
    return [
        ("t1", "P1", "cwa", "A", "pseudo1", "fam1"),
        ("t2", "P1", "cwa", "A", "trainInstr", "fam2"),
        ("t3", "P2", "cwa", "A", "trainInstr", "fam1"),
        ("t9", "T1", "cwa", "A", "heldX", "fam1"),
    ]


def _t1_definition(**overrides: object) -> dict:
    definition: dict = {
        "context_id": "P04CTX-t1-a",
        "station": "cwa",
        "task_id": "T1-CWA",
        "domain": "cwa:within",
        "held_instrument": "not_applicable",
        "selection_mode": "inner_master_cv",
        "phase_gate": "development",
        "outer_fit_uids": [f"u{i}" for i in range(1, 11)],
        "outer_test_uids": ["u11"],
        "outer_test_masters": 1,
        "fit_role_id": "R-T1-FIT-",
        "validation_role_id": "R-T1-VAL-",
        "outer_fit_role_id": "R-T1-OF",
        "outer_test_role_id": "R-T1-OT",
        "units": [
            ("outer_fold_as_inner:0", [f"u{i}" for i in range(1, 9)], ["u9", "u10"])
        ],
    }
    definition.update(overrides)
    return definition


def _t3_definition(**overrides: object) -> dict:
    definition: dict = {
        "context_id": "P04CTX-t3-a",
        "station": "cwa",
        "task_id": "T3-ZS",
        "domain": "cwa:heldX",
        "held_instrument": "heldX",
        "selection_mode": "pseudo_domain",
        "phase_gate": "held_evaluation",
        "outer_fit_uids": ["t1", "t2", "t3"],
        "outer_test_uids": ["t9"],
        "fit_role_id": "R-T3-FIT-",
        "validation_role_id": "R-T3-VAL-",
        "outer_fit_role_id": "R-T3-OF",
        "outer_test_role_id": "R-T3-OT",
        "units": [("pseudo:pseudo1", ["t3"], ["t1"])],
    }
    definition.update(overrides)
    return definition


def _manifest_rows(specs: list[tuple[str, str, str, str, str, str]]) -> list[dict]:
    return [
        {
            "observation_uid": uid,
            "master_sample_id": master,
            "station": station,
            "target_analyte": chemical,
            "instrument": instrument,
            "sensor_family": family,
        }
        for uid, master, station, chemical, instrument, family in specs
    ]


def _role_rows(
    context_id: str,
    role_id: str,
    role: str,
    unit_id: str,
    uids: list[str],
    by_uid: dict[str, dict],
) -> list[dict]:
    return [
        {
            "context_id": context_id,
            "role_id": role_id,
            "role": role,
            "selection_unit_id": unit_id,
            "observation_uid": uid,
            "master_sample_id": by_uid[uid]["master_sample_id"],
            "target_analyte": by_uid[uid]["target_analyte"],
            "instrument": by_uid[uid]["instrument"],
        }
        for uid in uids
    ]


def _assemble(
    specs: list[tuple[str, str, str, str, str, str]],
    definitions: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    manifest = _manifest_rows(specs)
    by_uid = {row["observation_uid"]: row for row in manifest}
    contexts: list[dict] = []
    roles: list[dict] = []
    for definition in definitions:
        fit = list(definition["outer_fit_uids"])
        test = list(definition["outer_test_uids"])
        context = {
            "context_id": definition["context_id"],
            "station": definition["station"],
            "task_id": definition["task_id"],
            "domain": definition["domain"],
            "held_instrument": definition["held_instrument"],
            "selection_mode": definition["selection_mode"],
            "phase_gate": definition["phase_gate"],
            "selection_unit_count": len(definition["units"]),
            "outer_fit_rows": len(fit),
            "outer_fit_masters": len(
                {by_uid[uid]["master_sample_id"] for uid in fit}
            ),
            "outer_test_rows": len(test),
            "outer_fit_uid_sha256": uid_set_hash(fit),
            "outer_test_uid_sha256": uid_set_hash(test),
        }
        if definition.get("outer_test_masters") is not None:
            context["outer_test_masters"] = definition["outer_test_masters"]
        contexts.append(context)
        roles.extend(
            _role_rows(
                context["context_id"],
                definition["outer_fit_role_id"],
                "outer_fit",
                "outer_fit",
                fit,
                by_uid,
            )
        )
        roles.extend(
            _role_rows(
                context["context_id"],
                definition["outer_test_role_id"],
                "outer_test",
                "outer_test",
                test,
                by_uid,
            )
        )
        for unit_id, unit_fit, unit_validation in definition["units"]:
            roles.extend(
                _role_rows(
                    context["context_id"],
                    definition["fit_role_id"] + unit_id,
                    "selection_fit",
                    unit_id,
                    unit_fit,
                    by_uid,
                )
            )
            roles.extend(
                _role_rows(
                    context["context_id"],
                    definition["validation_role_id"] + unit_id,
                    "selection_validation",
                    unit_id,
                    unit_validation,
                    by_uid,
                )
            )
    return manifest, contexts, roles


def _t1_inputs() -> tuple[list[dict], list[dict], list[dict]]:
    return _assemble(_t1_specs(), [_t1_definition()])


def _t3_inputs() -> tuple[list[dict], list[dict], list[dict]]:
    return _assemble(_t3_specs(), [_t3_definition()])


def _combined_inputs() -> tuple[list[dict], list[dict], list[dict]]:
    return _assemble(
        _t1_specs() + _t3_specs(),
        [_t1_definition(), _t3_definition()],
    )


def _replace_master(
    specs: list[tuple[str, str, str, str, str, str]],
    uids: set[str],
    master: str,
) -> list[tuple[str, str, str, str, str, str]]:
    return [
        (uid, master if uid in uids else row_master, *rest)
        for uid, row_master, *rest in specs
    ]


@lru_cache(maxsize=1)
def _readiness() -> dict:
    return build_readiness_report(project_root=PROJECT_ROOT)


def _report(
    manifest: list[dict],
    contexts: list[dict],
    roles: list[dict],
    *,
    readiness: dict | None = None,
) -> dict:
    inputs = build_support_inputs(manifest=manifest, contexts=contexts, roles=roles)
    return build_support_report(
        inputs, readiness=_readiness() if readiness is None else readiness
    )


def _rejects(
    manifest: list[dict],
    contexts: list[dict],
    roles: list[dict],
) -> None:
    with pytest.raises(SupportAuditError):
        _report(manifest, contexts, roles)


def _entry(report: dict, role: str, unit: str) -> dict:
    matches = [
        entry
        for entry in report["source_role_audit"]
        if entry["role"] == role and entry["selection_unit_id"] == unit
    ]
    assert len(matches) == 1
    return matches[0]


def _summary(report: dict, phase_gate: str, role: str) -> dict:
    matches = [
        summary
        for summary in report["summaries"]
        if summary["station"] == "cwa"
        and summary["phase_gate"] == phase_gate
        and summary["role"] == role
    ]
    assert len(matches) == 1
    return matches[0]


def _expected_pair_digest(
    context_id: str,
    role_id: str,
    role_rows: list[dict],
) -> str:
    """Independent reimplementation of the documented streaming convention."""

    digest = hashlib.sha256()
    ordered = sorted(role_rows, key=lambda row: row["observation_uid"])
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left["target_analyte"] != right["target_analyte"]:
                continue
            digest.update(
                pair_identity(
                    context_id=context_id,
                    role_id=role_id,
                    uid_a=left["observation_uid"],
                    uid_b=right["observation_uid"],
                ).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def _write_table(path: Path, columns: tuple[str, ...], rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _loader_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest, contexts, roles = _t1_inputs()
    return (
        _write_table(tmp_path / "manifest.csv", MANIFEST_COLUMNS, manifest),
        _write_table(tmp_path / "contexts.csv", CONTEXT_COLUMNS, contexts),
        _write_table(tmp_path / "roles.csv", ROLE_COLUMNS, roles),
    )


def _loader_argv(tmp_path: Path) -> list[str]:
    manifest, contexts, roles = _loader_paths(tmp_path)
    return [
        "--manifest",
        str(manifest),
        "--manifest-sha256",
        _sha(manifest),
        "--contexts",
        str(contexts),
        "--contexts-sha256",
        _sha(contexts),
        "--roles",
        str(roles),
        "--roles-sha256",
        _sha(roles),
    ]


def _fake_readiness(*, loss: int = 2, optimizer: int = 3, seed: int = 4) -> dict:
    return {
        "illustrative_full_crossing": {
            "fits_per_selection_unit": loss * optimizer * seed
        },
        "loss_configuration_total": loss,
        "optimizer_candidate_count": optimizer,
        "training_seed_count": seed,
        "unresolved_decisions": [{"decision_id": "P05-U01"}],
        "input_hashes": {"plan/contracts/test.json": "0" * 64},
    }


# --------------------------------------------------------------------------- #
# Counting, categories, cells, digest
# --------------------------------------------------------------------------- #


def test_selection_fit_categories_cells_and_support() -> None:
    report = _report(*_t1_inputs())
    entry = _entry(report, "selection_fit", "outer_fold_as_inner:0")

    assert entry["observation_count"] == 8
    assert entry["master_count"] == 6
    assert entry["instrument_count"] == 2
    assert entry["class_count"] == 2
    assert entry["same_chemical_pair_categories"] == {
        "same_master_same_instrument": 1,
        "same_master_different_instrument": 1,
        "different_master_same_instrument": 7,
        "different_master_different_instrument": 7,
    }
    assert entry["same_chemical_pair_count"] == 16
    assert entry["different_chemical_pair_count"] == 12
    assert entry["same_chemical_cross_substrate_pair_count"] == 8
    assert entry["master_counts_per_chemical"] == {"A": 4, "B": 2}
    assert entry["masters_with_multiple_instruments"] == 1
    assert entry["masters_with_multiple_substrate_families"] == 1
    assert entry["anchors_without_same_chemical_peer"] == 0
    assert entry["chemicals_with_two_or_more_masters"] == 2
    assert entry["two_chemical_two_master_support"] is True
    assert entry["all_chemicals_have_two_or_more_masters"] is True
    assert entry["class_instrument_cells"] == [
        {"target_analyte": "A", "instrument": "instr1", "spectra": 4, "masters": 3},
        {"target_analyte": "A", "instrument": "instr2", "spectra": 2, "masters": 2},
        {"target_analyte": "B", "instrument": "instr1", "spectra": 2, "masters": 2},
    ]
    assert entry["cells_with_two_or_more_spectra"] == 3
    assert entry["cells_with_two_or_more_masters"] == 3


def test_repeated_spectra_from_one_master_do_not_qualify_as_two_masters() -> None:
    specs = _replace_master(_t1_specs(), {"u3", "u4", "u5", "u6"}, "M1")
    report = _report(*_assemble(specs, [_t1_definition()]))
    entry = _entry(report, "selection_fit", "outer_fold_as_inner:0")
    cells = {
        (cell["target_analyte"], cell["instrument"]): cell
        for cell in entry["class_instrument_cells"]
    }

    assert cells[("A", "instr1")]["spectra"] == 4
    assert cells[("A", "instr1")]["masters"] == 1
    assert cells[("A", "instr2")]["spectra"] == 2
    assert cells[("A", "instr2")]["masters"] == 1
    assert cells[("B", "instr1")]["masters"] == 2
    assert entry["cells_with_two_or_more_spectra"] == 3
    assert entry["cells_with_two_or_more_masters"] == 1
    assert entry["master_counts_per_chemical"]["A"] == 1
    assert entry["two_chemical_two_master_support"] is False


def test_positive_pair_digest_matches_documented_convention() -> None:
    manifest, contexts, roles = _t1_inputs()
    report = _report(manifest, contexts, roles)
    entry = _entry(report, "selection_fit", "outer_fold_as_inner:0")
    fit_rows = [row for row in roles if row["role"] == "selection_fit"]

    assert entry["positive_pair_digest"] == _expected_pair_digest(
        entry["parent_context_id"], entry["parent_role_id"], fit_rows
    )
    assert entry["positive_pair_digest"] != _entry(
        report, "outer_fit", "outer_fit"
    )["positive_pair_digest"]


def test_singleton_role_has_empty_streaming_digest() -> None:
    report = _report(*_t3_inputs())
    entry = _entry(report, "selection_fit", "pseudo:pseudo1")

    assert entry["observation_count"] == 1
    assert entry["anchors_without_same_chemical_peer"] == 1
    assert entry["same_chemical_pair_count"] == 0
    assert entry["positive_pair_digest"] == hashlib.sha256(b"").hexdigest()


def test_permutation_invariance() -> None:
    manifest, contexts, roles = _combined_inputs()
    first = _report(manifest, contexts, roles)
    shuffled_manifest = list(manifest)
    shuffled_contexts = list(contexts)
    shuffled_roles = list(roles)
    rng = random.Random(20260824)
    rng.shuffle(shuffled_manifest)
    rng.shuffle(shuffled_contexts)
    rng.shuffle(shuffled_roles)

    second = _report(shuffled_manifest, shuffled_contexts, shuffled_roles)

    assert first == second
    assert render_report(first) == render_report(second)


# --------------------------------------------------------------------------- #
# Budget counting and T3 incomplete-union boundary
# --------------------------------------------------------------------------- #


def test_budget_counts_selection_fit_units_only() -> None:
    report = _report(*_combined_inputs())
    readiness = _readiness()
    crossing = readiness["illustrative_full_crossing"]["fits_per_selection_unit"]

    assert report["counts"]["selection_units"] == 2
    assert report["counts"]["selection_units_development"] == 1
    assert report["counts"]["selection_units_held_evaluation"] == 1
    # Four fitting-role entries exist, but outer roles are not selection units.
    assert report["counts"]["fitting_role_ids"] == 4
    assert report["cost_scenario"]["source_selection_units"]["overall"] == 2
    assert report["cost_scenario"]["crossing_per_selection_unit"] == crossing
    assert report["cost_scenario"]["illustrative_fits"]["overall"] == 2 * crossing
    assert report["cost_scenario"]["authorizing"] is False
    assert report["total_required_fits"] is None


def test_t3_incomplete_fit_union_is_permitted() -> None:
    report = _report(*_t3_inputs())
    fit = _entry(report, "selection_fit", "pseudo:pseudo1")
    outer = _entry(report, "outer_fit", "outer_fit")

    assert fit["observation_count"] == 1
    assert outer["observation_count"] == 3
    # Fit plus validation is two UIDs; the excluded second instrument is only
    # visible in outer_fit, so the union is intentionally incomplete.
    assert fit["observation_count"] + 1 < outer["observation_count"]
    outer_cells = {
        (cell["target_analyte"], cell["instrument"])
        for cell in outer["class_instrument_cells"]
    }
    assert ("A", "trainInstr") in outer_cells
    assert all(
        cell["instrument"] != "heldX"
        for cell in fit["class_instrument_cells"]
    )


def test_held_only_label_changes_do_not_affect_fitting_audit() -> None:
    base = _report(*_combined_inputs())
    specs = [
        (
            uid,
            master,
            station,
            "B" if uid == "t9" else chemical,
            instrument,
            "fam9" if uid == "t9" else family,
        )
        for uid, master, station, chemical, instrument, family in (
            _t1_specs() + _t3_specs()
        )
    ]
    changed = _report(*_assemble(specs, [_t1_definition(), _t3_definition()]))

    assert changed["source_role_audit"] == base["source_role_audit"]
    assert changed["summaries"] == base["summaries"]
    assert changed["cost_scenario"] == base["cost_scenario"]


def test_counts_reconcile_combined_fixture() -> None:
    report = _report(*_combined_inputs())
    counts = report["counts"]

    assert counts["manifest_rows"] == 15
    assert counts["master_samples"] == 12
    assert counts["contexts"] == 2
    assert counts["development_contexts"] == 1
    assert counts["held_evaluation_contexts"] == 1
    assert counts["role_rows"] == 27
    assert counts["role_ids"] == 8
    assert counts["roles_by_name"] == {
        "outer_fit": 2,
        "outer_test": 2,
        "selection_fit": 2,
        "selection_validation": 2,
    }


def test_summary_denominators() -> None:
    report = _report(*_combined_inputs())
    development = _summary(report, "development", "selection_fit")
    held = _summary(report, "held_evaluation", "selection_fit")

    assert development["role_count"] == 1
    assert development["observations"] == {"min": 8, "median": 8, "max": 8}
    assert development["masters"] == {"min": 6, "median": 6, "max": 6}
    assert development["same_master_cross_instrument_pairs"] == {
        "min": 1,
        "median": 1,
        "max": 1,
    }
    assert development["roles_lacking_same_master_cross_instrument_positives"] == 0
    assert (
        development["roles_lacking_different_master_cross_instrument_positives"] == 0
    )
    assert development["roles_lacking_two_chemical_two_master_support"] == 0
    assert development["roles_containing_zero_positive_anchors"] == 0

    assert held["role_count"] == 1
    assert held["roles_lacking_same_master_cross_instrument_positives"] == 1
    assert held["roles_lacking_different_master_cross_instrument_positives"] == 1
    assert held["roles_lacking_two_chemical_two_master_support"] == 1
    assert held["roles_containing_zero_positive_anchors"] == 1


def test_source_role_audit_ordering_and_provenance() -> None:
    report = _report(*_combined_inputs())
    entries = report["source_role_audit"]

    assert [entry["role"] for entry in entries] == [
        "selection_fit",
        "outer_fit",
        "selection_fit",
        "outer_fit",
    ]
    assert [entry["parent_context_id"] for entry in entries] == [
        "P04CTX-t1-a",
        "P04CTX-t1-a",
        "P04CTX-t3-a",
        "P04CTX-t3-a",
    ]
    assert all(entry["audit_id"].startswith("P05AUDIT-") for entry in entries)
    assert len({entry["audit_id"] for entry in entries}) == 4


def test_report_contract_fields() -> None:
    report = _report(*_combined_inputs())

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["audit_status"] == AUDIT_STATUS == "pass"
    assert report["scope"] == "inherited_role_metadata_only"
    assert report["scientific_execution_authorized"] is False
    assert report["scientific_fits_performed"] == 0
    assert report["total_required_fits"] is None
    assert report["resource_estimates"] is None
    assert report["unresolved_decisions_resolved"] is False
    assert len(report["unresolved_decision_ids"]) == 14
    assert report["unresolved_decision_ids"][0] == "P05-U01"
    assert report["claim_boundary"]
    assert report["integrity_note"]
    assert (
        report["input_hashes"]["readiness_contracts"]
        == _readiness()["input_hashes"]
    )
    assert report["input_hashes"]["caller_pinned"] == {
        "manifest": None,
        "contexts": None,
        "roles": None,
    }


def test_render_report_is_deterministic() -> None:
    first = _report(*_combined_inputs())
    second = _report(*_combined_inputs())

    assert render_report(first) == render_report(second)
    assert json.loads(render_report(first)) == first


# --------------------------------------------------------------------------- #
# Readiness boundary
# --------------------------------------------------------------------------- #


def test_readiness_is_required_keyword() -> None:
    inputs = build_support_inputs(*_t1_inputs()) if False else None
    manifest, contexts, roles = _t1_inputs()
    inputs = build_support_inputs(manifest=manifest, contexts=contexts, roles=roles)

    with pytest.raises(TypeError):
        build_support_report(inputs)  # type: ignore[call-arg]


def test_tampered_readiness_crossing_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    bad = _fake_readiness()
    bad["illustrative_full_crossing"]["fits_per_selection_unit"] = 99

    with pytest.raises(SupportAuditError):
        _report(manifest, contexts, roles, readiness=bad)


def test_readiness_missing_fields_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()

    with pytest.raises(SupportAuditError):
        _report(manifest, contexts, roles, readiness={})


# --------------------------------------------------------------------------- #
# Boundary validation
# --------------------------------------------------------------------------- #


def test_inner_master_leakage_rejected() -> None:
    specs = _replace_master(_t1_specs(), {"u9", "u10"}, "M1")
    _rejects(*_assemble(specs, [_t1_definition()]))


def test_outer_test_master_overlap_rejected() -> None:
    specs = _replace_master(_t1_specs(), {"u11"}, "M1")
    _rejects(*_assemble(specs, [_t1_definition()]))


def test_master_station_contradiction_rejected() -> None:
    specs = _t1_specs() + [("u12", "M7", "pills", "A", "instr1", "fam1")]
    _rejects(*_assemble(specs, [_t1_definition()]))


def test_master_chemical_contradiction_rejected() -> None:
    specs = _t1_specs() + [("u12", "M7", "cwa", "B", "instr1", "fam1")]
    _rejects(*_assemble(specs, [_t1_definition()]))


def test_unknown_station_rejected() -> None:
    _rejects(*_assemble(_t1_specs(), [_t1_definition(station="mars")]))


def test_unknown_phase_and_mode_mismatch_rejected() -> None:
    _rejects(*_assemble(_t1_specs(), [_t1_definition(phase_gate="prod")]))
    _rejects(
        *_assemble(_t1_specs(), [_t1_definition(selection_mode="pseudo_domain")])
    )


def test_task_and_domain_mismatch_rejected() -> None:
    _rejects(*_assemble(_t1_specs(), [_t1_definition(task_id="T1-PILLS")]))
    _rejects(*_assemble(_t1_specs(), [_t1_definition(domain="pills:within")]))
    _rejects(
        *_assemble(
            _t3_specs(), [_t3_definition(task_id="T1-CWA")]
        )
    )
    _rejects(
        *_assemble(
            _t3_specs(), [_t3_definition(domain="cwa:something-else")]
        )
    )


def test_held_instrument_in_outer_fit_rejected() -> None:
    _rejects(
        *_assemble(
            _t3_specs(),
            [_t3_definition(held_instrument="trainInstr", domain="cwa:trainInstr")],
        )
    )


def test_outer_test_instrument_contamination_rejected() -> None:
    specs = _t3_specs() + [("t10", "P3", "cwa", "A", "trainInstr", "fam1")]
    _rejects(
        *_assemble(
            specs,
            [_t3_definition(outer_test_uids=["t10"])],
        )
    )


def test_pseudo_validation_instrument_mismatch_rejected() -> None:
    _rejects(
        *_assemble(
            _t3_specs(),
            [_t3_definition(units=[("pseudo:trainInstr", ["t3"], ["t1"])])],
        )
    )


def test_pseudo_instrument_contamination_in_fitting_rejected() -> None:
    specs = _t3_specs() + [("t4", "P3", "cwa", "A", "pseudo1", "fam1")]
    definition = _t3_definition(
        outer_fit_uids=["t1", "t2", "t3", "t4"],
        units=[("pseudo:pseudo1", ["t3", "t4"], ["t1"])],
    )
    manifest, contexts, roles = _assemble(specs, [definition])
    by_uid = {row["observation_uid"]: row for row in manifest}
    assert by_uid["t4"]["instrument"] == "pseudo1"

    _rejects(manifest, contexts, roles)


def test_selection_uid_outside_outer_fit_rejected() -> None:
    _rejects(
        *_assemble(
            _t3_specs(),
            [_t3_definition(units=[("pseudo:pseudo1", ["t3"], ["t9"])])],
        )
    )


def test_whitespace_identifier_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    manifest[0] = {**manifest[0], "observation_uid": "u1 "}
    _rejects(manifest, contexts, roles)


def test_unknown_role_name_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    roles[0] = {**roles[0], "role": "bogus"}
    _rejects(manifest, contexts, roles)


def test_orphan_role_context_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    orphan = {**roles[0], "context_id": "P04CTX-orphan"}
    _rejects(manifest, contexts, roles + [orphan])


def test_duplicate_uid_within_role_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    fit_row = next(row for row in roles if row["role"] == "selection_fit")
    _rejects(manifest, contexts, roles + [dict(fit_row)])


def test_duplicate_unit_role_id_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    fit_row = next(row for row in roles if row["role"] == "selection_fit")
    conflicting = {**fit_row, "role_id": "R-CONFLICT"}
    _rejects(manifest, contexts, roles + [conflicting])


def test_duplicate_manifest_uid_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    manifest = [dict(row) for row in manifest]
    manifest.append(dict(manifest[0]))
    _rejects(manifest, contexts, roles)


def test_duplicate_context_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    contexts = [dict(row) for row in contexts]
    contexts.append(dict(contexts[0]))
    _rejects(manifest, contexts, roles)


def test_missing_required_role_field_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    roles = [dict(row) for row in roles]
    del roles[0]["instrument"]
    _rejects(manifest, contexts, roles)


def test_role_id_spanning_units_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    moved = False
    for index, row in enumerate(roles):
        if row["role"] == "selection_fit":
            roles[index] = {**row, "selection_unit_id": "outer_fold_as_inner:1"}
            moved = True
            break
    assert moved
    _rejects(manifest, contexts, roles)


def test_missing_selection_role_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    _rejects(
        manifest,
        contexts,
        [row for row in roles if row["role"] != "selection_validation"],
    )
    _rejects(
        manifest,
        contexts,
        [row for row in roles if row["role"] != "outer_test"],
    )


def test_role_manifest_metadata_mismatch_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    for index, row in enumerate(roles):
        if row["role"] == "selection_fit":
            roles[index] = {**row, "instrument": "instrX"}
            break
    _rejects(manifest, contexts, roles)


def test_uid_set_hash_mismatch_rejected() -> None:
    manifest, contexts, roles = _t1_inputs()
    contexts[0] = {**contexts[0], "outer_fit_uid_sha256": "0" * 64}
    _rejects(manifest, contexts, roles)

    contexts[0] = {**contexts[0], "outer_fit_uid_sha256": "nothex"}
    _rejects(manifest, contexts, roles)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("selection_unit_count", "0"),
        ("selection_unit_count", 0),
        ("selection_unit_count", True),
        ("selection_unit_count", "\u00b2"),
        ("outer_fit_rows", "abc"),
        ("outer_fit_rows", "1.0"),
        ("outer_fit_masters", " 1"),
        ("outer_test_rows", ""),
    ],
)
def test_invalid_counts_rejected(key: str, value: object) -> None:
    manifest, contexts, roles = _t1_inputs()
    contexts[0] = {**contexts[0], key: value}
    _rejects(manifest, contexts, roles)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("selection_unit_count", 5),
        ("outer_fit_rows", 9),
        ("outer_fit_masters", 7),
        ("outer_test_rows", 5),
        ("outer_test_masters", 3),
    ],
)
def test_context_count_mismatch_rejected(key: str, value: int) -> None:
    manifest, contexts, roles = _t1_inputs()
    contexts[0] = {**contexts[0], key: value}
    _rejects(manifest, contexts, roles)


def test_optional_outer_test_masters_absent_or_empty_accepted() -> None:
    manifest, contexts, roles = _t3_inputs()
    assert "outer_test_masters" not in contexts[0]
    _report(manifest, contexts, roles)

    contexts[0] = {**contexts[0], "outer_test_masters": ""}
    _report(manifest, contexts, roles)


def test_no_input_mutation() -> None:
    manifest, contexts, roles = _combined_inputs()
    snapshot = copy.deepcopy((manifest, contexts, roles))

    _report(manifest, contexts, roles)

    assert (manifest, contexts, roles) == snapshot


def test_report_does_not_leak_uids_masters_or_paths() -> None:
    manifest, contexts, roles = _combined_inputs()
    excluded = {
        "source_path": "/srv/private/raw/spec_0001.spc",
        "operator": "operator-alice",
        "intensity": "9876543.21",
        "qc_flag": "QC-INTERNAL-ONLY",
    }
    manifest = [{**row, **excluded} for row in manifest]
    text = render_report(_report(manifest, contexts, roles))

    for uid in ["u1", "u5", "u11", "t1", "t3", "t9"]:
        assert f'"{uid}"' not in text
    for master in ["M1", "M4", "M9", "P1", "T1"]:
        assert f'"{master}"' not in text
    assert str(PROJECT_ROOT) not in text
    assert "observation_uid" not in text
    for value in excluded.values():
        assert value not in text


# --------------------------------------------------------------------------- #
# Pair identity helpers
# --------------------------------------------------------------------------- #


def test_pair_identity_symmetry_and_sensitivity() -> None:
    forward = pair_identity(context_id="ctx", role_id="role", uid_a="a", uid_b="b")
    reverse = pair_identity(context_id="ctx", role_id="role", uid_a="b", uid_b="a")

    assert forward == reverse
    assert forward.startswith("P05PAIR-")
    assert forward != pair_identity(
        context_id="other", role_id="role", uid_a="a", uid_b="b"
    )
    assert forward != pair_identity(
        context_id="ctx", role_id="other", uid_a="a", uid_b="b"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"context_id": "", "role_id": "r", "uid_a": "a", "uid_b": "b"},
        {"context_id": "c", "role_id": "", "uid_a": "a", "uid_b": "b"},
        {"context_id": "c", "role_id": "r", "uid_a": "", "uid_b": "b"},
        {"context_id": "c", "role_id": "r", "uid_a": "a", "uid_b": ""},
        {"context_id": "c", "role_id": "r", "uid_a": "a", "uid_b": "a"},
        {"context_id": "c", "role_id": "r", "uid_a": 1, "uid_b": "b"},
        {"context_id": "  ", "role_id": "r", "uid_a": "a", "uid_b": "b"},
        {"context_id": "c ", "role_id": "r", "uid_a": "a", "uid_b": "b"},
        {"context_id": "c", "role_id": "r ", "uid_a": "a", "uid_b": "b"},
        {"context_id": "c", "role_id": "r", "uid_a": " a", "uid_b": "b"},
        {"context_id": "c", "role_id": "r", "uid_a": "a", "uid_b": "b "},
    ],
)
def test_pair_identity_rejections(kwargs: dict) -> None:
    with pytest.raises(SupportAuditError):
        pair_identity(**kwargs)


def test_audit_identity_distinguishes_provenance() -> None:
    base = audit_identity(
        context_id="c", role_id="r", role="selection_fit", selection_unit_id="u"
    )

    assert base.startswith("P05AUDIT-")
    assert base == audit_identity(
        context_id="c", role_id="r", role="selection_fit", selection_unit_id="u"
    )
    assert base != audit_identity(
        context_id="c2", role_id="r", role="selection_fit", selection_unit_id="u"
    )
    assert base != audit_identity(
        context_id="c", role_id="r2", role="selection_fit", selection_unit_id="u"
    )
    assert base != audit_identity(
        context_id="c", role_id="r", role="outer_fit", selection_unit_id="u"
    )
    assert base != audit_identity(
        context_id="c", role_id="r", role="selection_fit", selection_unit_id="u2"
    )


def test_uid_set_hash_matches_canonical_convention() -> None:
    assert uid_set_hash(["b", "a"]) == sha256_value(["a", "b"])
    assert uid_set_hash([]) == sha256_value([])


# --------------------------------------------------------------------------- #
# Loader and CLI
# --------------------------------------------------------------------------- #


def test_loader_verifies_pins_then_parses(tmp_path: Path) -> None:
    manifest, contexts, roles = _t1_inputs()
    manifest_path, contexts_path, roles_path = _loader_paths(tmp_path)
    inputs = load_support_inputs(
        manifest_path=manifest_path,
        manifest_sha256=_sha(manifest_path),
        contexts_path=contexts_path,
        contexts_sha256=_sha(contexts_path),
        roles_path=roles_path,
        roles_sha256=_sha(roles_path),
    )

    assert len(inputs.manifest) == len(manifest)
    assert len(inputs.contexts) == len(contexts)
    assert len(inputs.roles) == len(roles)
    assert inputs.manifest_sha256 == _sha(manifest_path)
    assert "outer_test_masters" not in inputs.contexts[0]
    _report(
        list(inputs.manifest),
        [dict(row) for row in inputs.contexts],
        [dict(row) for row in inputs.roles],
    )


def test_loader_discards_unrelated_manifest_columns(tmp_path: Path) -> None:
    manifest, contexts, roles = _t1_inputs()
    manifest_path = tmp_path / "manifest_extra.csv"
    fieldnames = list(MANIFEST_COLUMNS) + ["source_path", "operator", "intensity"]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            writer.writerow(
                {
                    **{column: row[column] for column in MANIFEST_COLUMNS},
                    "source_path": "/private/raw.spc",
                    "operator": "operator-alice",
                    "intensity": "9876543.21",
                }
            )
    contexts_path = _write_table(
        tmp_path / "contexts_extra.csv", CONTEXT_COLUMNS, contexts
    )
    roles_path = _write_table(tmp_path / "roles_extra.csv", ROLE_COLUMNS, roles)

    inputs = load_support_inputs(
        manifest_path=manifest_path,
        manifest_sha256=_sha(manifest_path),
        contexts_path=contexts_path,
        contexts_sha256=_sha(contexts_path),
        roles_path=roles_path,
        roles_sha256=_sha(roles_path),
    )

    assert all(set(row) == set(MANIFEST_COLUMNS) for row in inputs.manifest)
    text = render_report(
        _report(
            [dict(row) for row in inputs.manifest],
            [dict(row) for row in inputs.contexts],
            [dict(row) for row in inputs.roles],
        )
    )
    assert "/private/raw.spc" not in text
    assert "operator-alice" not in text
    assert "9876543.21" not in text


def test_loader_hash_mismatch_and_format_rejected(tmp_path: Path) -> None:
    manifest_path, contexts_path, roles_path = _loader_paths(tmp_path)
    with pytest.raises(SupportAuditError):
        load_support_inputs(
            manifest_path=manifest_path,
            manifest_sha256="0" * 64,
            contexts_path=contexts_path,
            contexts_sha256=_sha(contexts_path),
            roles_path=roles_path,
            roles_sha256=_sha(roles_path),
        )
    with pytest.raises(SupportAuditError):
        load_support_inputs(
            manifest_path=manifest_path,
            manifest_sha256="not-a-hash",
            contexts_path=contexts_path,
            contexts_sha256=_sha(contexts_path),
            roles_path=roles_path,
            roles_sha256=_sha(roles_path),
        )


def test_loader_missing_file_rejected(tmp_path: Path) -> None:
    _, contexts_path, roles_path = _loader_paths(tmp_path)
    with pytest.raises(SupportAuditError):
        load_support_inputs(
            manifest_path=tmp_path / "absent.csv",
            manifest_sha256="0" * 64,
            contexts_path=contexts_path,
            contexts_sha256=_sha(contexts_path),
            roles_path=roles_path,
            roles_sha256=_sha(roles_path),
        )


def _malformed_manifest(tmp_path: Path, payload: bytes) -> Path:
    manifest_path, _, _ = _loader_paths(tmp_path)
    manifest_path.write_bytes(payload)
    return manifest_path


def test_loader_malformed_tables_rejected(tmp_path: Path) -> None:
    _, contexts_path, roles_path = _loader_paths(tmp_path)
    header = ",".join(MANIFEST_COLUMNS)
    payloads = [
        b"",
        f"{header}\n".encode(),
        f"{header}\n\n".encode(),
        f"{header},station\n".encode(),
        f"{header}\nu1,m1,cwa,A,i1\n".encode(),
        f'{header}\nu1,m1,cwa,A,i1,"f1'.encode(),
        b"\xff\xfe\xfd\n",
    ]
    for payload in payloads:
        manifest_path = _malformed_manifest(tmp_path, payload)
        with pytest.raises(SupportAuditError):
            load_support_inputs(
                manifest_path=manifest_path,
                manifest_sha256=_sha(manifest_path),
                contexts_path=contexts_path,
                contexts_sha256=_sha(contexts_path),
                roles_path=roles_path,
                roles_sha256=_sha(roles_path),
            )


def test_cli_audits_pinned_csv(tmp_path: Path, capsys) -> None:
    argv = _loader_argv(tmp_path) + ["--project-root", str(PROJECT_ROOT)]

    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["audit_status"] == "pass"
    assert payload["counts"]["selection_units"] == 1


def test_cli_hash_mismatch_and_malformed_csv_exit_one(
    tmp_path: Path, capsys
) -> None:
    argv = _loader_argv(tmp_path)
    manifest_path = Path(argv[1])
    argv[3] = "0" * 64  # wrong manifest sha

    assert main(argv + ["--project-root", str(PROJECT_ROOT)]) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""

    malformed = b"not,a,manifest\n"
    manifest_path.write_bytes(malformed)
    argv[3] = _sha(manifest_path)
    assert manifest_path.read_bytes() == malformed

    assert main(argv + ["--project-root", str(PROJECT_ROOT)]) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_default_readiness_uses_public_root(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    calls: list[object] = []

    def fake_readiness(*, project_root=None):
        calls.append(project_root)
        return _fake_readiness()

    monkeypatch.setattr(p05_support, "build_readiness_report", fake_readiness)
    argv = _loader_argv(tmp_path)

    assert main(argv) == 0
    assert calls == [None]
    payload = json.loads(capsys.readouterr().out)
    assert payload["audit_status"] == "pass"
    assert payload["cost_scenario"]["crossing_per_selection_unit"] == 24


def test_cli_invalid_arguments_return_one_without_paths(capsys) -> None:
    assert main([]) == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "error: invalid command-line arguments"
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_does_not_write_artifacts(tmp_path: Path) -> None:
    argv = _loader_argv(tmp_path)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    assert main(argv + ["--project-root", str(PROJECT_ROOT)]) == 0

    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert before == after


def test_input_csv_hashes_unchanged_after_cli(tmp_path: Path) -> None:
    argv = _loader_argv(tmp_path)
    csv_paths = [Path(argv[1]), Path(argv[5]), Path(argv[9])]
    before = [_sha(path) for path in csv_paths]

    assert main(argv + ["--project-root", str(PROJECT_ROOT)]) == 0

    assert [_sha(path) for path in csv_paths] == before


# --------------------------------------------------------------------------- #
# Structural checks
# --------------------------------------------------------------------------- #


def _import_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_numeric_or_training_runtime_imports() -> None:
    for path in (MODULE_PATH, WRAPPER_PATH):
        source = path.read_text(encoding="utf-8")
        assert FORBIDDEN_IMPORT_ROOTS.isdisjoint(_import_roots(source))


def test_wrapper_is_a_thin_main_binding() -> None:
    source = WRAPPER_PATH.read_text(encoding="utf-8")

    assert "from atlas_sers.evaluation.p05_support import main" in source
    assert "raise SystemExit(main())" in source


def _subprocess_env() -> dict[str, str]:
    source_path = str(PROJECT_ROOT / "src")
    existing = os.environ.get("PYTHONPATH", "")
    return {
        **os.environ,
        "PYTHONPATH": source_path + (os.pathsep + existing if existing else ""),
    }


def test_fresh_interpreter_imports_no_numeric_runtime() -> None:
    code = (
        "import sys; sys.path.insert(0, {path!r});"
        "import atlas_sers.evaluation.p05_support;"
        "bad=sorted(name for name in {forbidden!r} if name in sys.modules);"
        "print(','.join(bad))"
    ).format(
        path=str(PROJECT_ROOT / "src"),
        forbidden=tuple(sorted(FORBIDDEN_IMPORT_ROOTS)),
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""


def test_wrapper_subprocess_help_and_missing_arguments() -> None:
    help_run = subprocess.run(
        [sys.executable, str(WRAPPER_PATH), "--help"],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert help_run.returncode == 0
    assert "audit" in help_run.stdout.lower()

    missing = subprocess.run(
        [sys.executable, str(WRAPPER_PATH)],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
    )
    assert missing.returncode == 1
    assert "error:" in missing.stderr
    assert "Traceback" not in missing.stderr
    assert missing.stdout == ""
