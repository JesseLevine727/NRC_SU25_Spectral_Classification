"""Acceptance tests for the P08 population resource proposal calculator."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from atlas_sers.evaluation.p08_population_resources import (
    build_population_resource_proposal,
)
from atlas_sers.governance.canonical import sha256_value

PACKAGE = Path(__file__).resolve().parents[1]
READINESS = PACKAGE / "results" / "p08_readiness"
INPUT_FILES = {
    "slot_audit": "population_slot_ledger_audit.json",
    "historical_classical": "historical_timing_basis.json",
    "historical_neural": "population_historical_neural_cost.json",
}
SCHEMA = "nato-sers-p08-population-resource-proposal-v1"
STAGE_IDS = ("POP-NOTES-4", "POP-NOTES-5", "POP-MIRA-4", "POP-MIRA-5")
MODEL_FIT = (170277, 258387, 184749, 280878)
SCALAR_CAL = (4071, 4716, 4347, 5067)
TOTAL_CEILING = (355521, 536970, 386493, 585150)
TOTAL_JOBS = (367935, 549384, 396951, 595608)
WALLTIME_HOURS = (120, 168, 144, 192)
ALLOWANCE_GIB = (64, 96, 80, 96)
GIB, DAY = 2**30, 86400


@pytest.fixture
def inputs():
    return {
        name: json.loads((READINESS / filename).read_text())
        for name, filename in INPUT_FILES.items()
    }


def _build(inputs):
    return build_population_resource_proposal(
        slot_audit=inputs["slot_audit"],
        historical_classical=inputs["historical_classical"],
        historical_neural=inputs["historical_neural"],
    )


def _catalog_for(catalogs, stage_id):
    population = "notes_clear_500" if "NOTES" in stage_id else "mira1_excluded_575"
    extra = stage_id.endswith("-5")
    for catalog in catalogs:
        if catalog["population_id"] == population and catalog["include_extra_trees"] is extra:
            return catalog
    raise AssertionError(f"Missing catalog for {stage_id}")


def test_nominal_ceilings_and_flags(inputs):
    proposals = _build(inputs)["proposals"]
    assert [p["stage_id"] for p in proposals] == list(STAGE_IDS)
    for index, proposal in enumerate(proposals):
        assert proposal["selected_panel"] is None
        assert proposal["resource_proposal_approved"] is False
        assert proposal["auto_launch"] is False
        assert proposal["model_fit_ceiling"] == MODEL_FIT[index]
        assert proposal["scalar_calibration_ceiling"] == SCALAR_CAL[index]
        assert proposal["total_job_ceiling"] == TOTAL_CEILING[index]
        assert proposal["total_catalog_jobs"] == TOTAL_JOBS[index]
        assert proposal["active_walltime_ceiling_seconds"] == WALLTIME_HOURS[index] * 3600
        assert proposal["new_artifact_allowance_bytes"] == ALLOWANCE_GIB[index] * GIB
        assert (
            proposal["starting_free_space_requirement_bytes"]
            == ALLOWANCE_GIB[index] * GIB + 30 * GIB
        )
        assert proposal["logical_neural_strategies_in_panel"] == 2
        assert (
            proposal["neural_recipes_charged_in_fresh_source_selection"] == 4
            and "panel_models" not in proposal
        )
        bonus = {"C-EXTRA-TREES"} if proposal["stage_id"].endswith("-5") else set()
        accounted = {"C-RANDOM-FOREST", "C-RBF-SVM", "D0-M", "D1", "D2", "D3"}
        assert set(proposal["accounted_fit_models"]) == accounted | bonus


def test_report_flags_and_hashes(inputs):
    report = _build(inputs)
    assert report["schema_version"] == SCHEMA
    assert report["scientific_operations"] == 0
    flags = (
        "execution_authorized",
        "resource_proposal_approved",
        "input_provenance_independently_verified",
        "actual_arrays_loaded",
        "resources_live_inspected",
        "capacities_reserved",
    )
    assert all(report[flag] is False for flag in flags)
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == sha256_value(payload)
    for name, content in inputs.items():
        assert report["input_canonical_sha256"][name] == sha256_value(content)


def test_repeatability_and_immutability(inputs):
    snapshot = copy.deepcopy(inputs)
    assert _build(inputs) == _build(inputs)
    assert inputs == snapshot


def test_reordering_preserves_sorted_proposals(inputs):
    baseline = _build(inputs)
    shuffled = copy.deepcopy(inputs)
    shuffled["slot_audit"]["catalogs"] = list(reversed(shuffled["slot_audit"]["catalogs"]))
    shuffled["historical_classical"]["classical"] = list(
        reversed(shuffled["historical_classical"]["classical"])
    )
    shuffled["historical_neural"]["neural"] = list(
        reversed(shuffled["historical_neural"]["neural"])
    )
    other = _build(shuffled)
    assert other["proposals"] == baseline["proposals"]
    assert other["input_canonical_sha256"] != baseline["input_canonical_sha256"]
    assert other["report_sha256"] != baseline["report_sha256"]


def test_cost_identities_and_ceil_margins(inputs):
    for proposal in _build(inputs)["proposals"]:
        assert proposal["neural_envelope_seconds"] == pytest.approx(
            proposal["neural_base_seconds"] + proposal["neural_conditional_envelope_seconds"]
        )
        assert proposal["serial_kernel_seconds_envelope"] == pytest.approx(
            proposal["classical_seconds"] + proposal["neural_envelope_seconds"]
        )
        assert (
            proposal["active_walltime_ceiling_seconds"]
            == math.ceil(2 * proposal["serial_kernel_seconds_envelope"] / DAY) * DAY
        )
        assert (
            proposal["new_artifact_allowance_bytes"]
            == math.ceil(2 * proposal["component_storage_bytes_envelope"] / (16 * GIB)) * 16 * GIB
        )
        envelope = proposal["conditional_envelope"]
        assert envelope["guard_source_fit"]["count"] == 0
        assert envelope["calibration_model_fit"]["count"] == 0
        assert envelope["is_future_runtime_upper_bound"] is False
        assert envelope["single_selection_map_realizability_asserted"] is False
        required = (
            "count",
            "max_mean_seconds",
            "max_mean_seconds_recipe",
            "max_checkpoint_bytes",
            "max_checkpoint_bytes_recipe",
        )
        for key in ("source_fit", "final_refit"):
            assert all(field in envelope[key] for field in required)


def test_classical_seconds_recomputed_from_recorded_rates(inputs):
    stage_map = {
        "inner_selection": "source_fit",
        "calibration_crossfit": "calibration_model_fit",
        "final_family_refit": "final_refit",
    }
    for proposal in _build(inputs)["proposals"]:
        catalog = _catalog_for(inputs["slot_audit"]["catalogs"], proposal["stage_id"])
        lower = catalog["summary"]["per_model"]["lower"]
        expected = 0.0
        for row in inputs["historical_classical"]["classical"]:
            model = row["model"]
            if model not in lower:
                continue
            count = lower[model]["stage_counts"].get(stage_map[row["stage"]], 0)
            expected += count * row["seconds"] / row["seconds_recorded"]
        assert proposal["classical_seconds"] == pytest.approx(expected)


def test_neural_conditional_recomputed_from_aggregate_deltas(inputs):
    neural = inputs["historical_neural"]["neural"]
    for proposal in _build(inputs)["proposals"]:
        catalog = _catalog_for(inputs["slot_audit"]["catalogs"], proposal["stage_id"])
        summary = catalog["summary"]
        envelope = proposal["conditional_envelope"]
        expected = 0.0
        for key, stage in (("source_fit", "source_fit"), ("final_refit", "final_refit")):
            rate = max(
                row["elapsed_seconds_sum"] / row["fits"]
                for row in neural
                if row["model"] in {"D1", "D2", "D3"} and row["stage"] == stage
            )
            delta = (
                summary["upper"]["stage_counts"][stage] - summary["lower"]["stage_counts"][stage]
            )
            assert envelope[key]["count"] == delta
            assert envelope[key]["max_mean_seconds"] == pytest.approx(rate)
            expected += delta * rate
        assert proposal["neural_conditional_envelope_seconds"] == pytest.approx(expected)


def test_fractional_checkpoint_mean_is_accepted(inputs):
    row = inputs["historical_neural"]["neural"][0]
    fits = row["fits"]
    assert fits > 1
    row["retained_checkpoint_bytes_max"] = 10
    row["retained_checkpoint_bytes_sum"] = 10 * fits - 1
    row["retained_checkpoint_bytes_mean"] = (10 * fits - 1) / fits
    report = _build(inputs)
    assert report["proposals"]
    cost_fields = (
        "classical_seconds",
        "neural_base_seconds",
        "neural_conditional_envelope_seconds",
        "neural_envelope_seconds",
        "serial_kernel_seconds_envelope",
    )
    for proposal in report["proposals"]:
        assert all(math.isfinite(proposal[field]) for field in cost_fields)
