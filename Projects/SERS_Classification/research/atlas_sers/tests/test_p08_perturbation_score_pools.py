"""T241 bounded complete-pooling tests for P08 stress score pools.

Metadata-only: no file IO, arrays, fits, predictions, scores, quantiles,
statistics or rendering.  All parent artifacts are invented public fixtures.
"""

from __future__ import annotations

from functools import cache

import pytest

from atlas_sers.evaluation import p08_perturbation_scores as scores_mod
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256

from .test_p08_perturbation_scores import (
    MIXED4,
    PSEUDO4,
    parent_index,
    support_for,
)

ALLFALLBACK = [{"kind": "master", "context_id": f"CTX-F{i}"} for i in range(4)]

_FIXTURES = {"pseudo4": PSEUDO4, "mixed4": MIXED4, "allfallback": ALLFALLBACK}
_KEYS = ("pseudo4", "mixed4", "allfallback")

_EXPECTED = {
    "pseudo4": {
        "physical": 76,
        "pooled": 19,
        "views": 92,
        "pooled_views": 31,
        "jobs": 19380,
        "aliases": 25092,
        "eligible_contexts": 4,
    },
    "mixed4": {
        "physical": 69,
        "pooled": 19,
        "views": 92,
        "pooled_views": 23,
        "jobs": 17952,
        "aliases": 23460,
        "eligible_contexts": 3,
    },
    "allfallback": {
        "physical": 48,
        "pooled": 12,
        "views": 92,
        "pooled_views": 23,
        "jobs": 12240,
        "aliases": 23460,
        "eligible_contexts": 0,
    },
}

_CASES = 96
_FAMILIES = 6
_ENDPOINTS = 2

_STAGE_OF_MODE = {
    "context_case": "context_case_score",
    "context_family": "context_family_curve",
    "pooled_case": "pooled_case_score",
    "pooled_family": "pooled_family_curve",
}


@cache
def _graph(key):
    scenario, catalog = support_for(_FIXTURES[key])
    records = tuple(scores_mod.iter_stress_score_records(catalog))
    return scenario, catalog, records


def _split(records):
    jobs, aliases = [], []
    for record in records:
        (jobs if record["record_type"] == "job" else aliases).append(record)
    return jobs, aliases


def _hashed(prefix, body):
    return prefix + canonical_sha256(body)


def _body(record, id_key):
    return {k: v for k, v in record.items() if k != id_key}


def _parent_ids(scenario):
    units, parity = parent_index(scenario)
    return units, parity, set(units.values()) | set(parity.values())


@pytest.mark.parametrize("key", _KEYS)
def test_pool_counts_and_stage_summary(key):
    _, catalog, records = _graph(key)
    exp = _EXPECTED[key]
    summary = catalog["summary"]
    assert summary["context_count"] == 4
    assert summary["eligible_qc_context_count"] == exp["eligible_contexts"]
    assert summary["procedure_count"] == exp["physical"]
    assert summary["pooled_procedure_count"] == exp["pooled"]
    assert summary["context_view_count"] == exp["views"]
    assert summary["pooled_view_count"] == exp["pooled_views"]
    assert summary["case_count"] == _CASES
    assert summary["endpoint_count"] == _ENDPOINTS
    assert summary["disturbance_family_count"] == _FAMILIES
    assert summary["stage_counts"] == {
        "context_case_score": exp["physical"] * _CASES * _ENDPOINTS,
        "context_family_curve": exp["physical"] * _FAMILIES * _ENDPOINTS,
        "pooled_case_score": exp["pooled"] * _CASES * _ENDPOINTS,
        "pooled_family_curve": exp["pooled"] * _FAMILIES * _ENDPOINTS,
    }
    assert summary["alias_counts"] == {
        "context_case": exp["views"] * _CASES * _ENDPOINTS,
        "context_family": exp["views"] * _FAMILIES * _ENDPOINTS,
        "pooled_case": exp["pooled_views"] * _CASES * _ENDPOINTS,
        "pooled_family": exp["pooled_views"] * _FAMILIES * _ENDPOINTS,
    }
    assert summary["score_job_count"] == exp["jobs"]
    assert summary["reporting_alias_count"] == exp["aliases"]
    jobs, aliases = _split(records)
    assert len(jobs) == exp["jobs"]
    assert len(aliases) == exp["aliases"]
    stage_counts = {}
    for job in jobs:
        stage_counts[job["stage"]] = stage_counts.get(job["stage"], 0) + 1
    assert stage_counts == summary["stage_counts"]
    assert all(r["record_type"] == "job" for r in records[: len(jobs)])
    assert all(r["record_type"] == "alias" for r in records[len(jobs) :])
    counts = {}
    for alias in aliases:
        counts[alias["mode"]] = counts.get(alias["mode"], 0) + 1
    assert counts == summary["alias_counts"]


@pytest.mark.parametrize(
    "key,operational,eligible",
    [("pseudo4", 1, 1), ("mixed4", 1, 0), ("allfallback", 1, 0)],
)
def test_complete_pool_group_counts(key, operational, eligible):
    _, catalog, _ = _graph(key)
    supports = catalog["supports"]
    assert (
        supports["operational_260"]["summary"]["complete_four_fold_domain_repeat_groups"]
        == operational
    )
    assert (
        supports["qc_eligible_54"]["summary"]["complete_four_fold_domain_repeat_groups"] == eligible
    )


def test_allfallback_eligible_support_is_empty_synthetic():
    _, catalog, _ = _graph("allfallback")
    support = catalog["supports"]["qc_eligible_54"]
    assert support["context_ids"] == []
    assert support["complete_pool_group_ids"] == []
    assert support["summary"]["stations"] == []
    for name, value in support["summary"].items():
        if name == "stations":
            continue
        assert value in (0, []), name
    assert [v for v in catalog["pooled_views"] if v["support_id"] == "qc_eligible_54"] == []


@pytest.mark.parametrize("key", _KEYS)
def test_pooled_procedure_bodies_and_physical_identity(key):
    _, catalog, _ = _graph(key)
    records = {c["context_id"]: c for c in catalog["context_records"]}
    groups = {
        (g["domain"], g["station"], g["instrument"], g["outer_repeat"]): g
        for g in catalog["pool_groups"]
        if g["complete_four_fold"]
    }
    for proc in catalog["pooled_procedures"]:
        assert proc["pooled_procedure_id"] == _hashed(
            "P08STRESSPOOLPROC-", _body(proc, "pooled_procedure_id")
        )
        assert set(proc) == {
            "domain",
            "station",
            "instrument",
            "outer_repeat",
            "members",
            "pooled_uid_sha256",
            "pooled_master_count",
            "pooled_procedure_id",
        }
        members = proc["members"]
        ctxs = [m["context_id"] for m in members]
        assert ctxs == sorted(ctxs)
        assert len(set(ctxs)) == 4
        assert {records[c]["outer_fold"] for c in ctxs} == {0, 1, 2, 3}
        masters = [set(m["master_id"] for m in records[c]["master_units"]) for c in ctxs]
        uids = [set(records[c]["test_uids"]) for c in ctxs]
        for i in range(4):
            for j in range(i + 1, 4):
                assert masters[i].isdisjoint(masters[j])
                assert uids[i].isdisjoint(uids[j])
        union = sorted(set().union(*uids))
        assert proc["pooled_uid_sha256"] == canonical_sha256(union)
        assert proc["pooled_master_count"] == sum(len(m) for m in masters)
        group = groups[(proc["domain"], proc["station"], proc["instrument"], proc["outer_repeat"])]
        assert union == group["test_uids"]


@pytest.mark.parametrize("key", _KEYS)
def test_context_and_pooled_views_are_hashed_and_complete(key):
    _, catalog, _ = _graph(key)
    views = catalog["context_views"]
    assert len(views) == 92
    per_context = {}
    for view in views:
        assert view["view_id"] == _hashed("P08STRESSVIEW-", _body(view, "view_id"))
        assert set(view) == {
            "policy_id",
            "context_id",
            "strategy",
            "recipe_id",
            "target_procedure_id",
            "mode",
            "upstream_alias_id",
            "view_id",
        }
        per_context.setdefault(view["context_id"], []).append(view)
    assert len(per_context) == 4
    for context_views in per_context.values():
        assert len(context_views) == 23
        modes = [v["mode"] for v in context_views]
        assert modes.count("universal") == 15
        assert modes.count("family_minimal_fallback") == 4
        assert (modes.count("qc_fixed_route") + modes.count("qc_minimal_fallback")) == 4
    proc_ids = {p["pooled_procedure_id"] for p in catalog["pooled_procedures"]}
    for view in catalog["pooled_views"]:
        assert view["view_id"] == _hashed("P08STRESSPOOLVIEW-", _body(view, "view_id"))
        assert set(view) == {
            "support_id",
            "pool_group_id",
            "domain",
            "outer_repeat",
            "policy_id",
            "strategy",
            "target_pooled_procedure_id",
            "view_id",
        }
        assert view["target_pooled_procedure_id"] in proc_ids


def test_pseudo4_eligible_pooled_views_reuse_operational_targets():
    _, catalog, _ = _graph("pseudo4")
    pooled = catalog["pooled_views"]
    operational = [v for v in pooled if v["support_id"] == "operational_260"]
    eligible = [v for v in pooled if v["support_id"] == "qc_eligible_54"]
    assert len(eligible) == 8
    assert {v["policy_id"] for v in eligible} == {"PP-U-MIN", "PP-QC-SRC"}
    assert {v["strategy"] for v in eligible} == {
        "C-RBF-SVM",
        "C-RANDOM-FOREST",
        "D0-M",
        "P05-SELECTED",
    }
    assert {v["pool_group_id"] for v in eligible}.isdisjoint(
        {v["pool_group_id"] for v in operational}
    )
    index = {
        (v["domain"], v["outer_repeat"], v["policy_id"], v["strategy"]): v[
            "target_pooled_procedure_id"
        ]
        for v in operational
    }
    for view in eligible:
        key = (view["domain"], view["outer_repeat"], view["policy_id"], view["strategy"])
        assert key in index
        assert index[key] == view["target_pooled_procedure_id"]


def test_mixed4_eligible_incomplete_group_has_no_pooled_view():
    _, catalog, _ = _graph("mixed4")
    support = catalog["supports"]["qc_eligible_54"]
    assert support["summary"]["complete_four_fold_domain_repeat_groups"] == 0
    assert len(support["context_ids"]) == 3
    assert support["context_ids"] == sorted(support["context_ids"])
    assert [v for v in catalog["pooled_views"] if v["support_id"] == "qc_eligible_54"] == []
    groups = [g for g in catalog["pool_groups"] if g["support_id"] == "qc_eligible_54"]
    assert len(groups) == 1
    assert groups[0]["complete_four_fold"] is False
    assert groups[0]["folds"] == [0, 1, 2]


def test_allfallback_fallback_views_share_min_targets():
    _, catalog, _ = _graph("allfallback")
    views = catalog["context_views"]
    universal = {}
    for view in views:
        if view["mode"] == "universal":
            universal[(view["context_id"], view["policy_id"], view["strategy"])] = view[
                "target_procedure_id"
            ]
    universal_policies = {key[1] for key in universal}
    assert len(universal_policies) == 3
    flows = {}
    for (context_id, policy_id, strategy), target in universal.items():
        if strategy in ("D0-M", "P05-SELECTED"):
            flows.setdefault((context_id, policy_id), set()).add(target)
    assert flows
    for targets in flows.values():
        assert len(targets) == 1
    min_targets = {
        (view["context_id"], view["strategy"]): view["target_procedure_id"]
        for view in views
        if view["mode"] == "universal" and view["policy_id"] == "PP-U-MIN"
    }
    for view in views:
        if view["mode"] in ("qc_minimal_fallback", "family_minimal_fallback"):
            assert (
                view["target_procedure_id"] == min_targets[(view["context_id"], view["strategy"])]
            )
    keys = [(v["context_id"], v["policy_id"], v["strategy"]) for v in views]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("key", _KEYS)
def test_score_jobs_are_closed_and_topologically_ordered(key):
    scenario, catalog, records = _graph(key)
    jobs, _ = _split(records)
    _, _, parent_ids = _parent_ids(scenario)
    order = {job["job_id"]: i for i, job in enumerate(jobs)}
    assert len(order) == len(jobs)
    known = set(order)
    for job in jobs:
        assert job["job_id"] == _hashed("P08STRESSSCORE-", _body(job, "job_id"))
        assert job["binding_sha256"] == catalog["catalog_sha256"]
        assert job["endpoint"] in ("M01", "M06")
        deps = job["depends_on_job_ids"]
        assert deps == sorted(deps)
        assert len(set(deps)) == len(deps)
        assert all(dep in known for dep in deps)
        assert all(order[dep] < order[job["job_id"]] for dep in deps)
        preds = job["depends_on_prediction_job_ids"]
        assert len(set(preds)) == len(preds)
        assert all(dep in parent_ids for dep in preds)


@pytest.mark.parametrize("key", _KEYS)
def test_context_case_jobs_use_parent_procedure_and_parity(key):
    scenario, _, records = _graph(key)
    jobs, _ = _split(records)
    units, parity, _ = _parent_ids(scenario)
    found = 0
    for job in jobs:
        if job["stage"] != "context_case_score":
            continue
        found += 1
        assert job["depends_on_job_ids"] == []
        expected = sorted({units[(job["target_id"], job["case_id"])], parity[job["target_id"]]})
        assert sorted(set(job["depends_on_prediction_job_ids"])) == expected
    assert found == _EXPECTED[key]["physical"] * _CASES * _ENDPOINTS


@pytest.mark.parametrize("key", _KEYS)
def test_pooled_case_jobs_use_four_member_predictions_not_context_scores(key):
    scenario, catalog, records = _graph(key)
    jobs, _ = _split(records)
    units, parity, _ = _parent_ids(scenario)
    procedures = {p["pooled_procedure_id"]: p for p in catalog["pooled_procedures"]}
    found = 0
    for job in jobs:
        if job["stage"] != "pooled_case_score":
            continue
        found += 1
        assert job["depends_on_job_ids"] == []
        members = procedures[job["target_id"]]["members"]
        assert len(members) == 4
        case = job["case_id"]
        expected = {units[(m["procedure_id"], case)] for m in members}
        expected |= {parity[m["procedure_id"]] for m in members}
        assert sorted(set(job["depends_on_prediction_job_ids"])) == sorted(expected)
    assert found == _EXPECTED[key]["pooled"] * _CASES * _ENDPOINTS


@pytest.mark.parametrize("key", _KEYS)
def test_family_curves_aggregate_case_jobs_without_parent_deps(key):
    scenario, _, records = _graph(key)
    jobs, _ = _split(records)
    family_cases = scenario.input_catalog["case_manifest"]["family_cases"]
    assert len(family_cases) == _FAMILIES
    for cases in family_cases.values():
        assert "P08-STRESS-CLEAN" in cases
    by_id = {j["job_id"]: j for j in jobs}
    context_case = {
        (j["target_id"], j["endpoint"], j["case_id"]): j["job_id"]
        for j in jobs
        if j["stage"] == "context_case_score"
    }
    pooled_case = {
        (j["target_id"], j["endpoint"], j["case_id"]): j["job_id"]
        for j in jobs
        if j["stage"] == "pooled_case_score"
    }
    for job in jobs:
        if job["stage"] not in ("context_family_curve", "pooled_family_curve"):
            continue
        assert job["depends_on_prediction_job_ids"] == []
        source = context_case if job["stage"] == "context_family_curve" else pooled_case
        cases = set(family_cases[job["disturbance_family"]])
        expected = {source[(job["target_id"], job["endpoint"], c)] for c in cases}
        deps = job["depends_on_job_ids"]
        assert set(deps) == expected
        assert {by_id[dep]["case_id"] for dep in deps} == cases


@pytest.mark.parametrize("key", _KEYS)
def test_aliases_resolve_same_case_family_and_endpoint(key):
    _, catalog, records = _graph(key)
    jobs, aliases = _split(records)
    by_id = {j["job_id"]: j for j in jobs}
    context_target = {v["view_id"]: v["target_procedure_id"] for v in catalog["context_views"]}
    pooled_target = {v["view_id"]: v["target_pooled_procedure_id"] for v in catalog["pooled_views"]}
    assert aliases
    for alias in aliases:
        assert alias["alias_id"] == _hashed("P08STRESSSCOREALIAS-", _body(alias, "alias_id"))
        assert alias["binding_sha256"] == catalog["catalog_sha256"]
        job = by_id[alias["target_score_job_id"]]
        assert job["stage"] == _STAGE_OF_MODE[alias["mode"]]
        assert job["endpoint"] == alias["endpoint"]
        if alias["mode"].startswith("context"):
            assert job["target_id"] == context_target[alias["view_id"]]
        else:
            assert job["target_id"] == pooled_target[alias["view_id"]]
        if alias["mode"].endswith("_case"):
            assert job["case_id"] == alias["case_id"]
            assert alias["disturbance_family"] == "not_applicable"
        else:
            assert job["disturbance_family"] == alias["disturbance_family"]
            assert alias["case_id"] == "not_applicable"


def test_pseudo4_eligible_aliases_share_operational_pooled_jobs():
    _, catalog, records = _graph("pseudo4")
    jobs, aliases = _split(records)
    by_id = {j["job_id"]: j for j in jobs}
    pooled = {v["view_id"]: v for v in catalog["pooled_views"]}
    operational = {}
    for alias in aliases:
        view = pooled.get(alias["view_id"])
        if view is None or view["support_id"] != "operational_260":
            continue
        key = (
            alias["mode"],
            alias["endpoint"],
            alias["case_id"],
            alias["disturbance_family"],
            by_id[alias["target_score_job_id"]]["target_id"],
        )
        operational[key] = alias["target_score_job_id"]
    checked = 0
    for alias in aliases:
        view = pooled.get(alias["view_id"])
        if view is None or view["support_id"] != "qc_eligible_54":
            continue
        key = (
            alias["mode"],
            alias["endpoint"],
            alias["case_id"],
            alias["disturbance_family"],
            by_id[alias["target_score_job_id"]]["target_id"],
        )
        assert operational.get(key) == alias["target_score_job_id"]
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("key", _KEYS)
def test_pooled_family_views_reuse_universal_targets(key):
    _, catalog, _ = _graph(key)
    family_policies = {
        v["policy_id"] for v in catalog["context_views"] if v["mode"] == "family_minimal_fallback"
    }
    min_targets = {}
    for view in catalog["pooled_views"]:
        if view["policy_id"] != "PP-U-MIN":
            continue
        min_targets[
            (view["support_id"], view["domain"], view["outer_repeat"], view["strategy"])
        ] = view["target_pooled_procedure_id"]
    for view in catalog["pooled_views"]:
        if view["policy_id"] not in family_policies:
            continue
        key = (view["support_id"], view["domain"], view["outer_repeat"], view["strategy"])
        assert min_targets[key] == view["target_pooled_procedure_id"]
