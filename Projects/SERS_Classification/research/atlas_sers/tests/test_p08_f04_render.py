# ruff: noqa: E501, UP031
# Assertions mirror literal renderer templates.
import copy
import csv
import io
import json
import re

import pytest

from atlas_sers.visualization.p08_f02_render import ENDPOINTS, ESTIMANDS, _canonical_sha
from atlas_sers.visualization.p08_f04_render import (
    COMPARATORS,
    DEEP_MODELS,
    POLICIES,
    QC_POLICY,
    prepare_f04,
)


def _cid(policy, deep, comparator, endpoint):
    return "policy_model_interaction::%s::%s::%s::%s" % (policy, deep, comparator, endpoint)


def _meta():
    return {
        "population": {
            "primary_spectra": 598,
            "held_spectra": 557,
            "masters": 69,
            "instruments": 10,
            "held_domains": 13,
            "contexts": 260,
        },
        "independent_unit": "physical_master",
        "selection": "source-only selection and calibration",
        "interval_caption": "95% marginal conditional intervals",
        "interaction_caption": "positive interaction is relative",
        "counts_reference": "598 original spectra",
        "display": "offline",
    }


def _row(estimand, policy, deep, comparator, endpoint, available=True):
    labels = [
        [policy, deep, endpoint],
        ["PP-U-MIN", deep, endpoint],
        [policy, comparator, endpoint],
        ["PP-U-MIN", comparator, endpoint],
    ]
    base = {
        "estimand": estimand,
        "contrast_id": _cid(policy, deep, comparator, endpoint),
        "family_id": "policy_model_interaction",
        "endpoint": endpoint,
        "policy_id": policy,
        "deep_model_id": deep,
        "comparator_model_id": comparator,
        "model_id": None,
        "procedure_labels": labels,
        "family_size": 32,
        "adjustment": "holm",
    }
    if available:
        base.update(
            {
                "available": True,
                "reason": None,
                "point_effect": 0.2,
                "domain_raw_p": 0.01,
                "domain_adjusted_p": 0.02,
                "instrument_raw_p": 0.03,
                "instrument_adjusted_p": 0.04,
                "hierarchy_planned": 5,
                "hierarchy_defined": 4,
                "hierarchy_undefined": 1,
                "hierarchy_lower": -0.1,
                "hierarchy_upper": 0.5,
                "hierarchy_reason": None,
                "crossed_lower": None,
                "crossed_upper": None,
                "crossed_reason": "not estimated",
                "master_only_lower": -0.2,
                "master_only_upper": 0.4,
                "master_only_reason": None,
                "instrument_only_lower": -0.3,
                "instrument_only_upper": 0.3,
                "instrument_only_reason": None,
                "procedure_balanced_accuracies": [0.8, 0.5, 0.4, 0.3],
            }
        )
    else:
        base.update(
            {
                "available": False,
                "reason": "outside_universal_execution_scope",
                "point_effect": None,
                "domain_raw_p": None,
                "domain_adjusted_p": None,
                "instrument_raw_p": None,
                "instrument_adjusted_p": None,
                "hierarchy_planned": None,
                "hierarchy_defined": None,
                "hierarchy_undefined": None,
                "hierarchy_lower": None,
                "hierarchy_upper": None,
                "hierarchy_reason": None,
                "crossed_lower": None,
                "crossed_upper": None,
                "crossed_reason": None,
                "master_only_lower": None,
                "master_only_upper": None,
                "master_only_reason": None,
                "instrument_only_lower": None,
                "instrument_only_upper": None,
                "instrument_only_reason": None,
                "procedure_balanced_accuracies": None,
            }
        )
    return base


def _build_prepared():
    overall = []
    for estimand in ESTIMANDS:
        for policy in POLICIES:
            for deep in DEEP_MODELS:
                for comparator in COMPARATORS:
                    for endpoint in ENDPOINTS:
                        overall.append(_row(estimand, policy, deep, comparator, endpoint))
    for estimand in ESTIMANDS:
        for deep in DEEP_MODELS:
            for comparator in ("C-RBF-SVM", "C-RANDOM-FOREST"):
                for endpoint in ENDPOINTS:
                    overall.append(
                        _row(estimand, QC_POLICY, deep, comparator, endpoint, available=False)
                    )
    domains = []
    for row in overall:
        if not row["available"]:
            continue
        for i in range(13):
            effect = -0.1234567890123456 if i == 0 else 0.1 * i - 0.3
            domains.append(
                {
                    "estimand": row["estimand"],
                    "contrast_id": row["contrast_id"],
                    "family_id": row["family_id"],
                    "endpoint": row["endpoint"],
                    "policy_id": row["policy_id"],
                    "deep_model_id": row["deep_model_id"],
                    "comparator_model_id": row["comparator_model_id"],
                    "domain": "DOM%02d" % i,
                    "station": "ST%02d" % i,
                    "instrument": "IN%02d" % i,
                    "point_effect": effect,
                    "contexts": 20 if row["estimand"] == "equal_context" else 5,
                    "unit_appearances": (i + 2) * (2 if row["endpoint"] == "M01" else 1),
                    "physical_masters": i + 1,
                    "distinct_units": i + 1,
                    "procedure_balanced_accuracies": [0.5 + effect / 2, 0.5 - effect / 2, 0.5, 0.5],
                }
            )
    semantic = {"f04_interactions": overall, "f04_domains": domains, "metadata": _meta()}
    return {
        "semantic": semantic,
        "semantic_sha256": _canonical_sha(semantic),
        "manifest": {"status": "accepted", "figure_id": "P08-F04"},
    }


def _raised(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_prepare_full_grid():
    result = prepare_f04(_build_prepared())
    assert set(result) == {"semantic", "semantic_sha256", "panels", "manifest"}
    manifest = result["manifest"]
    assert manifest["status"] == "prepared"
    assert manifest["reviewed"] is False and manifest["published"] is False
    assert manifest["figure_id"] == "P08-F04"
    assert manifest["research_question_id"] == "RQ-S01"
    assert len(result["panels"]) == 24
    slugs = [panel["slug"] for panel in result["panels"]]
    assert len(set(slugs)) == 24
    for panel in result["panels"]:
        assert set(panel) == {"slug", "semantic_sha256", "tex", "html", "csv"}
    assert len(result["semantic"]["f04_interactions"]) == 64
    assert len(result["semantic"]["f04_domains"]) == 624
    for panel in result["panels"]:
        rows = list(csv.reader(io.StringIO(panel["csv"])))
        assert rows[0][:3] == ["row_type", "semantic_sha256", "interval_mode"]
        assert len(rows) == 1 + 8 + 78
        kinds = [row[0] for row in rows[1:]]
        assert kinds.count("overall") == 8 and kinds.count("domain") == 78
        match = re.search(
            r'<script type="application/json" id="[^"]+">(.*?)</script>', panel["html"], re.S
        )
        assert match
        payload = json.loads(match.group(1).replace("<\\/", "</"))
        assert payload["slug"] == panel["slug"]
        assert len(payload["rows"]) == 6 and len(payload["overall"]) == 8


def test_rejects_missing_grid():
    prepared = copy.deepcopy(_build_prepared())
    prepared["semantic"]["f04_interactions"] = prepared["semantic"]["f04_interactions"][:-1]
    prepared["semantic_sha256"] = _canonical_sha(prepared["semantic"])
    _raised(lambda: prepare_f04(prepared))


def test_rejects_bad_types_and_ranges():
    prepared = copy.deepcopy(_build_prepared())
    prepared["semantic"]["f04_interactions"][0]["family_size"] = "32"
    prepared["semantic_sha256"] = _canonical_sha(prepared["semantic"])
    _raised(lambda: prepare_f04(prepared))
    prepared = copy.deepcopy(_build_prepared())
    prepared["semantic"]["f04_domains"][0]["point_effect"] = 5.0
    prepared["semantic_sha256"] = _canonical_sha(prepared["semantic"])
    _raised(lambda: prepare_f04(prepared))


def test_rejects_hash_mismatch():
    prepared = copy.deepcopy(_build_prepared())
    prepared["semantic_sha256"] = "0" * 64
    _raised(lambda: prepare_f04(prepared))


def test_qc_not_zero_and_negatives_precision():
    result = prepare_f04(_build_prepared())
    panel = result["panels"][0]
    rows = list(csv.reader(io.StringIO(panel["csv"])))
    header = rows[0]
    point_index = header.index("point_effect")
    qc = [row for row in rows if row[0] == "overall" and QC_POLICY in row]
    assert qc
    for row in qc:
        assert row[point_index] == ""
    values = [row["point_effect"] for row in result["semantic"]["f04_domains"]]
    assert any(value < 0 for value in values)
    assert -0.1234567890123456 in values


def test_support_is_specific_to_estimand_and_endpoint():
    prepared = _build_prepared()
    result = prepare_f04(prepared)
    # The fixture deliberately has different M01/M06 and pooled/context counts.
    assert {r["contexts"] for r in result["semantic"]["f04_domains"]} == {5, 20}
    prepared["semantic"]["f04_domains"][0]["unit_appearances"] += 1
    prepared["semantic_sha256"] = _canonical_sha(prepared["semantic"])
    with pytest.raises(ValueError, match="support mismatch"):
        prepare_f04(prepared)


@pytest.mark.parametrize(
    "field,value",
    [
        ("family_id", "wrong"),
        ("available", 1),
        ("point_effect", 0.123),
        ("procedure_labels", [["private", "payload"]] * 4),
    ],
)
def test_refuses_corrupt_overall_semantics(field, value):
    prepared = _build_prepared()
    prepared["semantic"]["f04_interactions"][0][field] = value
    prepared["semantic_sha256"] = _canonical_sha(prepared["semantic"])
    with pytest.raises(ValueError):
        prepare_f04(prepared)


def test_output_lists_do_not_alias_input():
    prepared = _build_prepared()
    before = copy.deepcopy(prepared)
    result = prepare_f04(prepared)
    result["semantic"]["f04_interactions"][0]["procedure_labels"][0][0] = "changed"
    result["semantic"]["f04_domains"][0]["procedure_balanced_accuracies"][0] = 0.99
    assert prepared == before


def test_domain_arithmetic_and_unavailable_policy_refusals():
    prepared = _build_prepared()
    prepared["semantic"]["f04_domains"][0]["point_effect"] = 0.123
    prepared["semantic_sha256"] = _canonical_sha(prepared["semantic"])
    with pytest.raises(ValueError, match="a-b-c"):
        prepare_f04(prepared)
    prepared = _build_prepared()
    row = next(r for r in prepared["semantic"]["f04_interactions"] if not r["available"])
    row["policy_id"] = "wrong"
    prepared["semantic_sha256"] = _canonical_sha(prepared["semantic"])
    with pytest.raises(ValueError, match="planned QC"):
        prepare_f04(prepared)


def test_visible_model_identity_and_native_scope():
    result = prepare_f04(_build_prepared())
    assert _canonical_sha(result["semantic"]) == result["semantic_sha256"]
    for panel in result["panels"]:
        assert "RQ-S01" in panel["tex"] and "RQ-S01" in panel["html"]
        assert "<strong>Deep strategy:</strong>" in panel["html"]
        assert "data-table" in panel["html"]
        assert "not that the deep absolute" in panel["html"]
        assert "Two planned QC contrasts" in panel["tex"]
