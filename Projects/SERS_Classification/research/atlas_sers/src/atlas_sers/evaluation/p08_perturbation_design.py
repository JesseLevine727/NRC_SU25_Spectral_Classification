"""Metadata-only case validation for the P08 perturbation design.

This helper validates declared metadata only. It grants no scientific
authority and does not validate geometry, inference, resources or provenance.
"""

from __future__ import annotations

from collections.abc import Mapping

from atlas_sers.governance.canonical import sha256_value

_CASE_DEFINITION = {
    "family_order": [
        "shift",
        "slope",
        "quadratic",
        "gaussian",
        "impulse",
        "clipping",
    ],
    "shift_cm1": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "slope_range_fraction": [-0.1, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.1],
    "quadratic_range_fraction": [0.0, 0.025, 0.05, 0.075, 0.1],
    "gaussian_source_quantiles": [0.5, 0.75, 0.9, 0.95],
    "impulse_counts": [0, 1, 3, 5],
    "clipping_upper_quantiles": [1.0, 0.999, 0.995, 0.99],
    "stochastic_replicates": 10,
    "impulse_height_range_fraction": 1.0,
    "impulse_minimum_index_separation": 3,
    "shared_zero_case_id": "P08-STRESS-CLEAN",
}
_EXPECTED_COUNTS = {
    "unique_cases_with_shared_zero": 96,
    "nonzero_cases": 95,
    "family_memberships_including_zero_aliases": 101,
}
_CLEAN = "P08-STRESS-CLEAN"
_SCHEMA_VERSION = "nato-sers-p08-perturbation-design-v1"
_INVALID = "invalid_perturbation_design"


def _validated_definition(design: Mapping[str, object]) -> Mapping[str, object]:
    """Validate metadata-only P08 design; no scientific authority is granted."""
    if not isinstance(design, Mapping):
        raise ValueError(_INVALID)
    try:
        sha256_value(design)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError(_INVALID) from exc
    if design.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(_INVALID)
    if design.get("execution_authorized") is not False:
        raise ValueError(_INVALID)
    if design.get("numerical_execution_accepted") is not False:
        raise ValueError(_INVALID)
    if design.get("resource_proposal_approved") is not False:
        raise ValueError(_INVALID)
    operations = design.get("authorized_scientific_operations")
    if type(operations) is not int or operations != 0:
        raise ValueError(_INVALID)
    if sha256_value(design.get("case_definition")) != sha256_value(_CASE_DEFINITION):
        raise ValueError(_INVALID)
    if sha256_value(design.get("expected_case_counts")) != sha256_value(_EXPECTED_COUNTS):
        raise ValueError(_INVALID)
    return design["case_definition"]


def build_perturbation_case_manifest(*, design) -> dict:
    definition = _validated_definition(design)

    family_order = definition["family_order"]
    stochastic_replicates = definition["stochastic_replicates"]
    family_levels = {
        "shift": definition["shift_cm1"],
        "slope": definition["slope_range_fraction"],
        "quadratic": definition["quadratic_range_fraction"],
        "gaussian": [0, *definition["gaussian_source_quantiles"]],
        "impulse": definition["impulse_counts"],
        "clipping": definition["clipping_upper_quantiles"],
    }

    cases = [{"case_id": _CLEAN, "family": "clean", "value": 0, "replicate_index": None}]
    family_cases = {family: [] for family in family_order}

    for family in family_order:
        zero_level = 1.0 if family == "clipping" else 0
        for level in family_levels[family]:
            if level == zero_level:
                family_cases[family].append(_CLEAN)
                continue
            if family in ("gaussian", "impulse"):
                replicate_indices = range(stochastic_replicates)
            else:
                replicate_indices = (None,)
            for replicate_index in replicate_indices:
                payload = {"family": family, "value": level, "replicate_index": replicate_index}
                case_id = "P08-STRESS-" + sha256_value(payload)
                cases.append({"case_id": case_id, **payload})
                family_cases[family].append(case_id)

    family_case_counts = {family: len(entries) for family, entries in family_cases.items()}
    distinct_case_ids = {case["case_id"] for case in cases}
    nonzero_case_ids = {case_id for case_id in distinct_case_ids if case_id != _CLEAN}

    summary = {
        "unique_cases_with_shared_zero": len(distinct_case_ids),
        "nonzero_cases": len(nonzero_case_ids),
        "family_memberships_including_zero_aliases": sum(family_case_counts.values()),
        "family_case_counts": family_case_counts,
    }

    canonical_keys = [
        "unique_cases_with_shared_zero",
        "nonzero_cases",
        "family_memberships_including_zero_aliases",
    ]
    if [summary[key] for key in canonical_keys] != [
        _EXPECTED_COUNTS[key] for key in canonical_keys
    ]:
        raise ValueError(_INVALID)
    if [family_case_counts[family] for family in family_order] != [11, 9, 5, 41, 31, 4]:
        raise ValueError(_INVALID)
    if len(distinct_case_ids) != 96:
        raise ValueError(_INVALID)
    for family in family_order:
        if family_cases[family].count(_CLEAN) != 1:
            raise ValueError(_INVALID)

    result = {
        "schema_version": "nato-sers-p08-perturbation-cases-v1",
        "execution_authorized": False,
        "scientific_operations": 0,
        "numerical_perturbations_computed": False,
        "job_ledger_complete": False,
        "input_provenance_independently_verified": False,
        "design_sha256": sha256_value(design),
        "case_definition_sha256": sha256_value(definition),
        "cases": cases,
        "family_cases": family_cases,
        "summary": summary,
    }
    result["report_sha256"] = sha256_value(result)
    return result
