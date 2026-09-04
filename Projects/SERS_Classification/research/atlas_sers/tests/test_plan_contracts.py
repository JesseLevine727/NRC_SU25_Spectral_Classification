from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PLAN = PROJECT / "plan"


def test_primary_contract_is_locked() -> None:
    contract = json.loads((PLAN / "contracts/research_contract.json").read_text())
    representation = contract["primary_representation"]

    assert contract["protocol_version"].startswith("atlas-sers-")
    assert contract["population_counts"]["primary_spectra"] == 598
    assert contract["population_counts"]["physical_masters"] == 69
    assert representation["axis_start_cm1"] == 400
    assert representation["axis_end_cm1"] == 1800
    assert representation["n_features"] == 1401
    assert representation["scaling"] == "per-spectrum min-max to [0,1]"
    assert all(
        item["path"].startswith("${ATLAS_PRIVATE_ROOT}/")
        for item in contract["authoritative_inputs"]
    )


def test_split_contract_keeps_information_regimes_separate() -> None:
    contract = json.loads((PLAN / "contracts/split_contract.json").read_text())

    assert contract["outer_folds_per_station"] == 4
    assert len(contract["primary_domain_eligibility"]["domains"]) == 13
    assert set(contract["adaptation_regimes"]) == {
        "zero_shot",
        "UDA",
        "paired_calibration",
        "few_shot",
    }
    assert contract["leakage_assertions_are_fatal"] is True


def test_figure_registry_has_complete_native_and_html_pairs() -> None:
    with (PLAN / "registries/figure_registry.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 44
    assert {row["figure_id"] for row in rows} == {f"F{index:02d}" for index in range(44)}
    assert all(row["tikz_path"].endswith(".tex") for row in rows)
    assert all(row["html_path"].endswith(".html") for row in rows)


def test_p13_amendment_is_separate_locked_and_outcome_disclosed() -> None:
    registry_dir = PLAN / "registries"
    with (registry_dir / "p13_research_question_registry.csv").open(newline="") as handle:
        questions = list(csv.DictReader(handle))
    with (registry_dir / "p13_decision_registry.csv").open(newline="") as handle:
        decisions = {row["decision_id"]: row for row in csv.DictReader(handle)}
    with (registry_dir / "p13_figure_registry.csv").open(newline="") as handle:
        figures = {row["figure_id"]: row for row in csv.DictReader(handle)}
    with (registry_dir / "p13_support_policy_registry.csv").open(newline="") as handle:
        support = {row["support_rule_id"]: row for row in csv.DictReader(handle)}

    phase = next(csv.DictReader((registry_dir / "p13_phase_registry.csv").open(newline="")))

    assert [row["research_question_id"] for row in questions] == ["RQ-S07"]
    assert questions[0]["status"] == "locked_2026-09-04"
    assert len(decisions) == 16
    assert {row["status"] for row in decisions.values()} == {"locked"}
    assert "0.60" in decisions["P13-D01"]["decision"]
    assert "0.10" in decisions["P13-D02"]["decision"]
    assert phase["amendment_version"] == "nato-sers-p13-v1-locked"
    assert phase["freeze_gate"] == "P13-FREEZE-PASSED"
    assert support["P13-SUP02"]["dimension"] == "evaluation_domain"
    assert "station substrate family and held instrument" in support["P13-SUP02"]["rule"]
    assert support["P13-SUP03"]["dimension"] == "class_support_cell"
    assert {row["status"] for row in support.values()} == {"locked"}
    assert figures["F44"]["status"] == "complete"
    assert {figures[key]["status"] for key in ("F45", "F46", "F47")} == {
        "locked_planned"
    }


def test_p13_support_registries_match_the_locked_metadata_audit() -> None:
    registry_dir = PLAN / "registries"
    domain_path = registry_dir / "p13_domain_support_registry.csv"
    crossover_path = registry_dir / "p13_crossover_support_registry.csv"
    with domain_path.open(newline="") as handle:
        domains = list(csv.DictReader(handle))
    with crossover_path.open(newline="") as handle:
        crossovers = list(csv.DictReader(handle))
    summary = json.loads((registry_dir / "p13_support_freeze_summary.json").read_text())

    domain_tiers = {
        tier: sum(row["support_tier"] == tier for row in domains)
        for tier in {row["support_tier"] for row in domains}
    }
    crossover_tiers = {
        tier: sum(row["support_tier"] == tier for row in crossovers)
        for tier in {row["support_tier"] for row in crossovers}
    }

    assert len(domains) == 34
    assert domain_tiers == {
        "confirmatory": 13,
        "exploratory_low_support": 3,
        "unsupported_by_design": 18,
    }
    assert len(crossovers) == 34
    assert crossover_tiers == {
        "confirmatory": 8,
        "exploratory_low_support": 7,
        "descriptive_singleton": 19,
    }
    assert {row["protocol_version"] for row in domains + crossovers} == {
        "nato-sers-p13-v1-locked"
    }
    assert summary["domain_counts"] == domain_tiers
    assert summary["crossover_block_counts"] == crossover_tiers
    assert summary["registry_hashes"]["p13_domain_support_registry_sha256"] == (
        hashlib.sha256(domain_path.read_bytes()).hexdigest()
    )
    assert summary["registry_hashes"]["p13_crossover_support_registry_sha256"] == (
        hashlib.sha256(crossover_path.read_bytes()).hexdigest()
    )
    assert "No P13" in summary["p13_outcome_access_disclosure"]


def test_f44_is_one_canonical_master_id_publication_set() -> None:
    stem = "F44_sample_substrate_instrument_matrix"
    data_path = PLAN / "figures" / "data" / f"{stem}.csv"
    html_path = PLAN / "figures" / "html" / f"{stem}.html"
    tikz_path = PLAN / "figures" / "tikz" / f"{stem}.tex"

    with data_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()

    assert len(rows) == 69 * 4 * 10
    assert "master_sample_id" in rows[0]
    assert "sample_code" not in rows[0]
    assert len({row["master_sample_id"] for row in rows}) == 69
    assert digest in html_path.read_text()
    assert digest in tikz_path.read_text()
    assert not (PLAN / "figures" / "html" / f"{stem}_master_ids.html").exists()
    assert not (PLAN / "figures" / "png" / f"{stem}_master_ids.png").exists()


def test_p03_publication_figure_hashes_match_files() -> None:
    manifest = PROJECT / "results" / "p03_classical" / "tables" / "publication_figure_manifest.csv"
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["figure_id"] for row in rows} == {
        "F12",
        "F13",
        "F38",
        "F39",
        "F40",
        "F41",
        "F42",
        "F43",
    }
    slugs = {
        "F12": "classical_selection",
        "F13": "classical_t1",
        "F38": "classical_t3_domains",
        "F39": "classical_t1_t3",
        "F40": "classical_spectrum_master",
        "F41": "classical_confusion",
        "F42": "classical_calibration",
        "F43": "classical_controls",
    }
    for row in rows:
        stem = f"{row['figure_id']}_{slugs[row['figure_id']]}"
        for kind, suffix in (
            ("data", "csv"),
            ("tikz", "tex"),
            ("pdf", "pdf"),
            ("png", "png"),
            ("html", "html"),
        ):
            path = PLAN / "figures" / kind / f"{stem}.{suffix}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row[f"{kind}_sha256"]
