from __future__ import annotations

import csv
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

    assert len(rows) == 35
    assert {row["figure_id"] for row in rows} == {f"F{index:02d}" for index in range(35)}
    assert all(row["tikz_path"].endswith(".tex") for row in rows)
    assert all(row["html_path"].endswith(".html") for row in rows)
