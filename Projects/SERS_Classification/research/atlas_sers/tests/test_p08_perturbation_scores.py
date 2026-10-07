"""Synthetic metadata-only tests for the P08 stress score-support and DAG.

Invented identities and the public prediction-catalog surface only.  No
science is executed, no scores are computed and no provenance is proved.
"""

from __future__ import annotations

import copy
import json
import unittest
from collections import defaultdict

from atlas_sers.evaluation.p08_perturbation_predictions import (
    iter_stress_prediction_records,
)
from atlas_sers.evaluation.p08_perturbation_score_support import (
    build_stress_score_support,
    validate_stress_score_support,
)
from atlas_sers.evaluation.p08_perturbation_score_support import (
    require_scientific_execution as support_require_scientific_execution,
)
from atlas_sers.evaluation.p08_perturbation_scores import (
    iter_stress_score_records,
)
from atlas_sers.evaluation.p08_perturbation_scores import (
    require_scientific_execution as scores_require_scientific_execution,
)
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256
from tests.test_p08_perturbation_predictions import (
    CLEAN_CASE,
    EXPECTED_CASES,
    EXPECTED_PROCEDURES,
    FAMILY_POLICY,
    NA,
    QC_POLICY,
    independent_hash,
    scenario,
)

SUPPORT_SCHEMA = "nato-sers-p08-stress-score-support-v1"

CONTEXT_CASE = "context_case_score"
CONTEXT_FAMILY = "context_family_curve"
POOLED_CASE = "pooled_case_score"
POOLED_FAMILY = "pooled_family_curve"

ENDPOINTS = ("M01", "M06")
ALIAS_MODES = ("context_case", "context_family", "pooled_case", "pooled_family")

CONTEXT_CASE_RESOLUTION = (
    "registered_context_present_class_scores_and_probability_diagnostics_no_resampling"
)
CONTEXT_FAMILY_RESOLUTION = (
    "all_registered_cases_mean_replicate_scores_signed_directions_normalized_area_"
    "negative_losses_retained"
)
POOLED_CASE_RESOLUTION = (
    "concatenate_four_disjoint_folds_rebuild_M01_M06_present_class_scores_no_cross_repeat_pooling"
)
POOLED_FAMILY_RESOLUTION = CONTEXT_FAMILY_RESOLUTION

SUPPORT_KEYS = {
    "schema_version",
    "execution_authorized",
    "scientific_operations",
    "artifact_provenance_independently_verified",
    "scientific_scores_computed",
    "full_stress_job_ledger_complete",
    "prediction_catalog",
    "memberships",
    "context_records",
    "pool_groups",
    "context_views",
    "pooled_procedures",
    "pooled_views",
    "global_master_ids",
    "global_instrument_ids",
    "supports",
    "summary",
    "catalog_sha256",
}

CONTEXT_RECORD_KEYS = {
    "context_id",
    "domain",
    "station",
    "instrument",
    "outer_repeat",
    "outer_fold",
    "test_uid_sha256",
    "test_uids",
    "classes",
    "class_spectrum_counts",
    "class_master_counts",
    "master_units",
    "context_metadata_sha256",
}

POOL_GROUP_KEYS = {
    "support_id",
    "domain",
    "station",
    "instrument",
    "outer_repeat",
    "folds",
    "context_ids",
    "complete_four_fold",
    "test_uids",
    "master_count",
    "group_id",
}

CONTEXT_VIEW_KEYS = {
    "policy_id",
    "context_id",
    "strategy",
    "recipe_id",
    "target_procedure_id",
    "mode",
    "upstream_alias_id",
    "view_id",
}

POOLED_PROCEDURE_KEYS = {
    "domain",
    "station",
    "instrument",
    "outer_repeat",
    "members",
    "pooled_uid_sha256",
    "pooled_master_count",
    "pooled_procedure_id",
}

POOLED_VIEW_KEYS = {
    "support_id",
    "pool_group_id",
    "domain",
    "outer_repeat",
    "policy_id",
    "strategy",
    "target_pooled_procedure_id",
    "view_id",
}

SUPPORT_SUMMARY_KEYS = {
    "contexts",
    "domains",
    "instruments",
    "stations",
    "distinct_test_spectra",
    "distinct_test_masters",
    "test_spectrum_appearances",
    "master_context_prediction_units",
    "complete_four_fold_domain_repeat_groups",
    "complete_four_fold_contexts",
    "complete_four_fold_distinct_spectra",
    "complete_four_fold_distinct_masters",
    "complete_four_fold_test_appearances",
}

SUMMARY_KEYS = {
    "context_count",
    "eligible_qc_context_count",
    "procedure_count",
    "context_view_count",
    "pooled_procedure_count",
    "pooled_view_count",
    "case_count",
    "endpoint_count",
    "disturbance_family_count",
    "stage_counts",
    "score_job_count",
    "alias_counts",
    "reporting_alias_count",
}

JOB_KEYS = {
    "record_type",
    "binding_sha256",
    "stage",
    "target_id",
    "endpoint",
    "case_id",
    "disturbance_family",
    "depends_on_job_ids",
    "depends_on_prediction_job_ids",
    "resolution",
    "job_id",
}

ALIAS_KEYS = {
    "record_type",
    "binding_sha256",
    "mode",
    "view_id",
    "endpoint",
    "case_id",
    "disturbance_family",
    "target_score_job_id",
    "alias_id",
}

ROW_KEYS = {
    "context_id",
    "domain",
    "station",
    "instrument",
    "outer_repeat",
    "outer_fold",
    "observation_uid",
    "master_id",
    "label",
}

PSEUDO4 = [
    {"kind": "pseudo", "context_id": "CTX-A", "recipe": "D3"},
    {"kind": "pseudo", "context_id": "CTX-B", "recipe": "D3"},
    {"kind": "pseudo", "context_id": "CTX-C", "recipe": "D3"},
    {"kind": "pseudo", "context_id": "CTX-D", "recipe": "D3"},
]

MIXED4 = [
    {"kind": "pseudo", "context_id": "CTX-A", "recipe": "D3"},
    {"kind": "pseudo", "context_id": "CTX-B", "recipe": "D3"},
    {"kind": "pseudo", "context_id": "CTX-C", "recipe": "D3"},
    {"kind": "master", "context_id": "CTX-D"},
]


def _parent_contexts(sc):
    raw = {context["context_id"]: context for context in sc.fixture["contexts"]}
    bound = {context["context_id"]: context for context in sc.input_catalog["contexts"]}
    merged = {}
    for context_id, raw_context in raw.items():
        bound_context = bound[context_id]
        assert sorted(raw_context["outer_test_uids"]) == bound_context["test_uids"]
        merged[context_id] = {**raw_context, **bound_context}
    return merged


def _role_index(sc):
    return {row["observation_uid"]: row for row in sc.fixture["roles"]}


def _eligible_context_ids(sc):
    return {
        procedure["context_id"]
        for procedure in sc.bundle["procedures"]
        if procedure["policy_id"] == QC_POLICY
    }


def build_memberships(sc):
    contexts = _parent_contexts(sc)
    roles = _role_index(sc)
    ordered = sorted(contexts)
    rows = []
    for fold, context_id in enumerate(ordered):
        context = contexts[context_id]
        instrument = context["held_instrument"]
        domain = "toy:" + instrument
        for observation_uid in context["outer_test_uids"]:
            role = roles[observation_uid]
            rows.append(
                {
                    "context_id": context_id,
                    "domain": domain,
                    "station": "toy",
                    "instrument": instrument,
                    "outer_repeat": 1,
                    "outer_fold": fold,
                    "observation_uid": observation_uid,
                    "master_id": role["master_id"],
                    "label": role["label"],
                }
            )
    return {
        "operational_contexts": ordered,
        "qc_eligible_contexts": sorted(_eligible_context_ids(sc)),
        "test_rows": rows,
    }


_SUPPORTS = {}


def support_for(entries):
    key = "default" if entries is None else json.dumps(entries, sort_keys=True)
    if key not in _SUPPORTS:
        sc = scenario(entries)
        _SUPPORTS[key] = (
            sc,
            build_stress_score_support(
                prediction_catalog=sc.catalog,
                memberships=build_memberships(sc),
            ),
        )
    return _SUPPORTS[key]


_PARENT_INDEX = {}


def parent_index(sc):
    key = id(sc)
    if key not in _PARENT_INDEX:
        units = {}
        parity = {}
        for record in iter_stress_prediction_records(sc.catalog):
            if record["record_type"] != "job":
                continue
            if record["stage"] == "prediction_units":
                units[(record["procedure_id"], record["case_id"])] = record["job_id"]
            elif record["stage"] == "clean_probability_parity":
                parity[record["procedure_id"]] = record["job_id"]
        _PARENT_INDEX[key] = (units, parity)
    return _PARENT_INDEX[key]


def _split(records):
    jobs = [record for record in records if record["record_type"] == "job"]
    aliases = [record for record in records if record["record_type"] == "alias"]
    return jobs, aliases


def _by_stage(jobs):
    buckets = defaultdict(list)
    for job in jobs:
        buckets[job["stage"]].append(job)
    return buckets


def _support_ids(support):
    supports = support["supports"]
    operational = next(key for key in supports if "operational" in key)
    eligible = next(key for key in supports if key != operational and "qc" in key)
    return operational, eligible


class SupportIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc, cls.support = support_for(None)

    def test_root_keys_and_denied_authority(self):
        support = self.support
        self.assertEqual(set(support), SUPPORT_KEYS)
        self.assertEqual(support["schema_version"], SUPPORT_SCHEMA)
        for flag in (
            "execution_authorized",
            "artifact_provenance_independently_verified",
            "scientific_scores_computed",
            "full_stress_job_ledger_complete",
        ):
            self.assertIs(support[flag], False)
        self.assertEqual(support["scientific_operations"], 0)

    def test_root_digest_and_catalog_copy(self):
        support = self.support
        body = {name: value for name, value in support.items() if name != "catalog_sha256"}
        self.assertEqual(support["catalog_sha256"], independent_hash(body))
        self.assertEqual(support["prediction_catalog"], self.sc.catalog)
        self.assertIsNot(support["prediction_catalog"], self.sc.catalog)

    def test_literal_summary(self):
        summary = self.support["summary"]
        self.assertEqual(set(summary), SUMMARY_KEYS)
        self.assertEqual(summary["context_count"], 2)
        self.assertEqual(summary["eligible_qc_context_count"], 1)
        self.assertEqual(summary["procedure_count"], EXPECTED_PROCEDURES)
        self.assertEqual(summary["context_view_count"], 46)
        self.assertEqual(summary["pooled_procedure_count"], 0)
        self.assertEqual(summary["pooled_view_count"], 0)
        self.assertEqual(summary["case_count"], EXPECTED_CASES)
        self.assertEqual(summary["endpoint_count"], 2)
        self.assertEqual(summary["disturbance_family_count"], 6)
        self.assertEqual(
            summary["stage_counts"],
            {
                CONTEXT_CASE: 5952,
                CONTEXT_FAMILY: 372,
                POOLED_CASE: 0,
                POOLED_FAMILY: 0,
            },
        )
        self.assertEqual(summary["score_job_count"], 6324)
        self.assertEqual(
            summary["alias_counts"],
            {
                "context_case": 8832,
                "context_family": 552,
                "pooled_case": 0,
                "pooled_family": 0,
            },
        )
        self.assertEqual(summary["reporting_alias_count"], 9384)
        self.assertEqual(sum(summary["stage_counts"].values()), summary["score_job_count"])
        self.assertEqual(sum(summary["alias_counts"].values()), summary["reporting_alias_count"])

    def test_memberships_normalized(self):
        memberships = self.support["memberships"]
        self.assertEqual(
            set(memberships),
            {"operational_contexts", "qc_eligible_contexts", "test_rows"},
        )
        self.assertEqual(memberships["operational_contexts"], ["CTX-F", "CTX-Q"])
        self.assertEqual(memberships["qc_eligible_contexts"], ["CTX-Q"])
        rows = memberships["test_rows"]
        keys = [(row["context_id"], row["observation_uid"]) for row in rows]
        self.assertEqual(keys, sorted(keys))
        for row in rows:
            self.assertEqual(set(row), ROW_KEYS)
            self.assertIsInstance(row["outer_repeat"], int)
            self.assertIsInstance(row["outer_fold"], int)
            self.assertNotIsInstance(row["outer_repeat"], bool)
            self.assertNotIsInstance(row["outer_fold"], bool)

    def test_context_records(self):
        records = self.support["context_records"]
        self.assertEqual([record["context_id"] for record in records], ["CTX-F", "CTX-Q"])
        parent = _parent_contexts(self.sc)
        for record in records:
            self.assertEqual(set(record), CONTEXT_RECORD_KEYS)
            source = parent[record["context_id"]]
            self.assertEqual(record["test_uids"], sorted(source["outer_test_uids"]))
            self.assertEqual(record["test_uid_sha256"], source["test_uid_sha256"])
            self.assertEqual(record["test_uid_sha256"], canonical_sha256(record["test_uids"]))
            self.assertEqual(record["classes"], sorted(record["class_spectrum_counts"]))
            self.assertEqual(
                sum(record["class_spectrum_counts"].values()),
                len(record["test_uids"]),
            )
            self.assertEqual(
                sum(record["class_master_counts"].values()),
                len(record["master_units"]),
            )
            master_ids = [unit["master_id"] for unit in record["master_units"]]
            self.assertEqual(master_ids, sorted(master_ids))
            for unit in record["master_units"]:
                self.assertEqual(unit["observation_uids"], sorted(unit["observation_uids"]))
            body = {
                name: value for name, value in record.items() if name != "context_metadata_sha256"
            }
            self.assertEqual(record["context_metadata_sha256"], independent_hash(body))

    def test_context_views(self):
        views = self.support["context_views"]
        self.assertEqual(len(views), 46)
        ordered = [(view["policy_id"], view["context_id"], view["strategy"]) for view in views]
        self.assertEqual(ordered, sorted(ordered))
        eligible = _eligible_context_ids(self.sc)
        by_context = defaultdict(list)
        for view in views:
            self.assertEqual(set(view), CONTEXT_VIEW_KEYS)
            body = {name: value for name, value in view.items() if name != "view_id"}
            self.assertEqual(view["view_id"], "P08STRESSVIEW-" + independent_hash(body))
            by_context[view["context_id"]].append(view)
        self.assertEqual(set(by_context), {"CTX-F", "CTX-Q"})
        for context_id, context_views in by_context.items():
            self.assertEqual(len(context_views), 23)
            modes = defaultdict(int)
            policies = defaultdict(int)
            for view in context_views:
                modes[view["mode"]] += 1
                policies[view["policy_id"]] += 1
            self.assertEqual(modes["universal"], 15)
            self.assertEqual(modes["family_minimal_fallback"], 4)
            self.assertEqual(modes["qc_fixed_route"] + modes["qc_minimal_fallback"], 4)
            self.assertEqual(policies[FAMILY_POLICY], 4)
            self.assertEqual(policies[QC_POLICY], 4)
            if context_id in eligible:
                self.assertEqual(modes["qc_fixed_route"], 4)
                self.assertEqual(modes["qc_minimal_fallback"], 0)
            else:
                self.assertEqual(modes["qc_fixed_route"], 0)
                self.assertEqual(modes["qc_minimal_fallback"], 4)

    def test_global_ids(self):
        rows = self.support["memberships"]["test_rows"]
        self.assertEqual(
            self.support["global_master_ids"], sorted({row["master_id"] for row in rows})
        )
        self.assertEqual(
            self.support["global_instrument_ids"],
            sorted({row["instrument"] for row in rows}),
        )

    def test_pool_groups_and_supports(self):
        support = self.support
        operational_id, eligible_id = _support_ids(support)
        self.assertEqual(len(support["supports"]), 2)
        for entry in support["supports"].values():
            self.assertEqual(set(entry), {"context_ids", "complete_pool_group_ids", "summary"})
            self.assertEqual(set(entry["summary"]), SUPPORT_SUMMARY_KEYS)
            self.assertEqual(entry["context_ids"], sorted(entry["context_ids"]))
        groups = {group["group_id"]: group for group in support["pool_groups"]}
        self.assertEqual(len(groups), 2)
        for group in groups.values():
            self.assertEqual(set(group), POOL_GROUP_KEYS)
            self.assertFalse(group["complete_four_fold"])
            body = {name: value for name, value in group.items() if name != "group_id"}
            self.assertEqual(group["group_id"], "P08STRESSPOOLGROUP-" + independent_hash(body))
        operational_groups = [
            group for group in groups.values() if group["support_id"] == operational_id
        ]
        eligible_groups = [group for group in groups.values() if group["support_id"] == eligible_id]
        self.assertEqual(len(operational_groups), 1)
        self.assertEqual(len(eligible_groups), 1)
        operational = operational_groups[0]
        self.assertEqual(operational["context_ids"], ["CTX-F", "CTX-Q"])
        self.assertEqual(operational["folds"], [0, 1])
        self.assertEqual(operational["domain"], "toy:held-inst")
        self.assertEqual(operational["outer_repeat"], 1)
        self.assertEqual(len(operational["test_uids"]), 4)
        self.assertEqual(operational["master_count"], 4)
        eligible = eligible_groups[0]
        self.assertEqual(eligible["context_ids"], ["CTX-Q"])
        self.assertEqual(eligible["folds"], [1])
        for support_id in (operational_id, eligible_id):
            self.assertEqual(support["supports"][support_id]["complete_pool_group_ids"], [])
        self.assertEqual(support["pooled_procedures"], [])
        self.assertEqual(support["pooled_views"], [])

    def test_support_summaries_recompute(self):
        support = self.support
        rows = support["memberships"]["test_rows"]
        for entry in support["supports"].values():
            subset = [row for row in rows if row["context_id"] in set(entry["context_ids"])]
            summary = entry["summary"]
            self.assertEqual(summary["contexts"], len(entry["context_ids"]))
            self.assertEqual(summary["domains"], len({row["domain"] for row in subset}))
            self.assertEqual(summary["instruments"], len({row["instrument"] for row in subset}))
            self.assertEqual(summary["stations"], sorted({row["station"] for row in subset}))
            self.assertEqual(
                summary["distinct_test_spectra"],
                len({row["observation_uid"] for row in subset}),
            )
            self.assertEqual(summary["test_spectrum_appearances"], len(subset))
            self.assertEqual(
                summary["distinct_test_masters"],
                len({row["master_id"] for row in subset}),
            )
            self.assertEqual(
                summary["master_context_prediction_units"],
                len({(row["context_id"], row["master_id"]) for row in subset}),
            )
            for key in (
                "complete_four_fold_domain_repeat_groups",
                "complete_four_fold_contexts",
                "complete_four_fold_distinct_spectra",
                "complete_four_fold_distinct_masters",
                "complete_four_fold_test_appearances",
            ):
                self.assertEqual(summary[key], 0)


class ScoreIteratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc, cls.support = support_for(None)
        cls.records = list(iter_stress_score_records(cls.support))
        cls.jobs, cls.aliases = _split(cls.records)

    def test_jobs_precede_aliases_and_are_unique(self):
        jobs, aliases = self.jobs, self.aliases
        self.assertEqual(self.records[: len(jobs)], jobs)
        self.assertEqual(self.records[len(jobs) :], aliases)
        self.assertEqual(len({job["job_id"] for job in jobs}), len(jobs))
        self.assertEqual(len({alias["alias_id"] for alias in aliases}), len(aliases))

    def test_literal_stage_and_alias_counts(self):
        by_stage = _by_stage(self.jobs)
        self.assertEqual(len(self.jobs), 6324)
        self.assertEqual(len(self.aliases), 9384)
        self.assertEqual(set(by_stage), {CONTEXT_CASE, CONTEXT_FAMILY})
        self.assertEqual(len(by_stage[CONTEXT_CASE]), 5952)
        self.assertEqual(len(by_stage[CONTEXT_FAMILY]), 372)
        mode_counts = defaultdict(int)
        for alias in self.aliases:
            mode_counts[alias["mode"]] += 1
        self.assertEqual(dict(mode_counts), {"context_case": 8832, "context_family": 552})
        for mode in mode_counts:
            self.assertIn(mode, ALIAS_MODES)

    def test_job_identity_and_keys(self):
        for job in self.jobs:
            self.assertEqual(set(job), JOB_KEYS)
            self.assertEqual(job["record_type"], "job")
            self.assertEqual(job["binding_sha256"], self.support["catalog_sha256"])
            self.assertIn(job["endpoint"], ENDPOINTS)
            body = {name: value for name, value in job.items() if name != "job_id"}
            self.assertEqual(job["job_id"], "P08STRESSSCORE-" + independent_hash(body))
            self.assertEqual(job["depends_on_job_ids"], sorted(set(job["depends_on_job_ids"])))
            self.assertEqual(
                job["depends_on_prediction_job_ids"],
                sorted(set(job["depends_on_prediction_job_ids"])),
            )

    def test_alias_identity_and_keys(self):
        for alias in self.aliases:
            self.assertEqual(set(alias), ALIAS_KEYS)
            self.assertEqual(alias["record_type"], "alias")
            self.assertEqual(alias["binding_sha256"], self.support["catalog_sha256"])
            self.assertIn(alias["mode"], ALIAS_MODES)
            self.assertIn(alias["endpoint"], ENDPOINTS)
            body = {name: value for name, value in alias.items() if name != "alias_id"}
            self.assertEqual(alias["alias_id"], "P08STRESSSCOREALIAS-" + independent_hash(body))

    def test_internal_dependencies_precede_jobs(self):
        order = {job["job_id"]: index for index, job in enumerate(self.jobs)}
        for job in self.jobs:
            for dependency in job["depends_on_job_ids"]:
                self.assertIn(dependency, order)
                self.assertLess(order[dependency], order[job["job_id"]])

    def test_parent_prediction_dependencies_exist(self):
        units, parity = parent_index(self.sc)
        known = set(units.values()) | set(parity.values())
        for job in self.jobs:
            for dependency in job["depends_on_prediction_job_ids"]:
                self.assertIn(dependency, known)

    def test_stage_order_is_sorted(self):
        for jobs in _by_stage(self.jobs).values():
            keys = [
                (job["target_id"], job["endpoint"], job["case_id"], job["disturbance_family"])
                for job in jobs
            ]
            self.assertEqual(keys, sorted(keys))

    def test_alias_order_is_sorted_within_modes(self):
        for mode in {alias["mode"] for alias in self.aliases}:
            subset = [alias for alias in self.aliases if alias["mode"] == mode]
            keys = [
                (
                    alias["view_id"],
                    alias["endpoint"],
                    alias["case_id"],
                    alias["disturbance_family"],
                )
                for alias in subset
            ]
            self.assertEqual(keys, sorted(keys))

    def test_context_case_wiring(self):
        units, parity = parent_index(self.sc)
        for job in _by_stage(self.jobs)[CONTEXT_CASE]:
            target = job["target_id"]
            case = job["case_id"]
            self.assertNotEqual(case, NA)
            self.assertEqual(job["disturbance_family"], NA)
            self.assertEqual(job["resolution"], CONTEXT_CASE_RESOLUTION)
            self.assertEqual(job["depends_on_job_ids"], [])
            self.assertEqual(
                set(job["depends_on_prediction_job_ids"]),
                {units[(target, case)], parity[target]},
            )

    def test_clean_case_includes_replay_gate(self):
        _, parity = parent_index(self.sc)
        seen = 0
        for job in _by_stage(self.jobs)[CONTEXT_CASE]:
            if job["case_id"] == CLEAN_CASE:
                seen += 1
                self.assertIn(parity[job["target_id"]], job["depends_on_prediction_job_ids"])
        self.assertGreater(seen, 0)

    def test_context_family_curve_wiring(self):
        family_cases = self.sc.input_catalog["case_manifest"]["family_cases"]
        self.assertEqual(len(family_cases), 6)
        shared_clean = set.intersection(*(set(cases) for cases in family_cases.values()))
        self.assertEqual(len(shared_clean), 1)
        shared_clean = shared_clean.pop()
        for cases in family_cases.values():
            self.assertEqual(list(cases).count(shared_clean), 1)
        index = {
            (job["target_id"], job["endpoint"], job["case_id"]): job["job_id"]
            for job in _by_stage(self.jobs)[CONTEXT_CASE]
        }
        for job in _by_stage(self.jobs)[CONTEXT_FAMILY]:
            cases = family_cases[job["disturbance_family"]]
            expected = {index[(job["target_id"], job["endpoint"], case)] for case in cases}
            self.assertIn(index[(job["target_id"], job["endpoint"], shared_clean)], expected)
            self.assertEqual(set(job["depends_on_job_ids"]), expected)
            self.assertEqual(job["depends_on_prediction_job_ids"], [])
            self.assertEqual(job["case_id"], NA)
            self.assertEqual(job["resolution"], CONTEXT_FAMILY_RESOLUTION)

    def test_alias_targets_and_endpoints(self):
        jobs = {job["job_id"]: job for job in self.jobs}
        views = {view["view_id"]: view for view in self.support["context_views"]}
        for alias in self.aliases:
            target = jobs[alias["target_score_job_id"]]
            self.assertEqual(alias["endpoint"], target["endpoint"])
            self.assertIn(alias["view_id"], views)
            if alias["mode"] == "context_case":
                self.assertEqual(alias["case_id"], target["case_id"])
                self.assertEqual(alias["disturbance_family"], NA)
            else:
                self.assertEqual(alias["disturbance_family"], target["disturbance_family"])
                self.assertEqual(alias["case_id"], NA)

    def test_alias_coverage(self):
        views = self.support["context_views"]
        counts = defaultdict(int)
        for alias in self.aliases:
            counts[(alias["view_id"], alias["endpoint"], alias["mode"])] += 1
        case_count = self.support["summary"]["case_count"]
        family_count = self.support["summary"]["disturbance_family_count"]
        for view in views:
            for endpoint in ENDPOINTS:
                self.assertEqual(counts[(view["view_id"], endpoint, "context_case")], case_count)
                self.assertEqual(
                    counts[(view["view_id"], endpoint, "context_family")], family_count
                )

    def test_require_scientific_execution_always_denied(self):
        for function in (
            support_require_scientific_execution,
            scores_require_scientific_execution,
        ):
            for args in ((), ({},), ({"execution_authorized": True},), (self.support,)):
                with self.subTest(function=function.__name__, args=args):
                    with self.assertRaises(ValueError) as caught:
                        function(*args)
                    self.assertEqual(str(caught.exception), "scientific_execution_not_authorized")


class SupportValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc, cls.support = support_for(None)

    def test_validate_roundtrip_snapshot(self):
        snapshot = validate_stress_score_support(self.support)
        self.assertEqual(snapshot, self.support)
        self.assertIsNot(snapshot, self.support)
        self.assertIsNot(snapshot["summary"], self.support["summary"])

    def test_validate_rejects_forged_content(self):
        forged = copy.deepcopy(self.support)
        forged["summary"]["context_count"] += 1
        forged["catalog_sha256"] = independent_hash(
            {name: value for name, value in forged.items() if name != "catalog_sha256"}
        )
        with self.assertRaises(ValueError) as caught:
            validate_stress_score_support(forged)
        self.assertEqual(str(caught.exception), "invalid_stress_score_support")

    def test_validate_rejects_forged_digest(self):
        forged = copy.deepcopy(self.support)
        forged["catalog_sha256"] = "0" * 64
        with self.assertRaises(ValueError) as caught:
            validate_stress_score_support(forged)
        self.assertEqual(str(caught.exception), "invalid_stress_score_support")

    def test_validate_rejects_execution_claim(self):
        forged = copy.deepcopy(self.support)
        forged["execution_authorized"] = True
        forged["catalog_sha256"] = independent_hash(
            {name: value for name, value in forged.items() if name != "catalog_sha256"}
        )
        with self.assertRaises(ValueError) as caught:
            validate_stress_score_support(forged)
        self.assertEqual(str(caught.exception), "invalid_stress_score_support")

    def test_validate_rejects_non_mapping(self):
        for bad in (None, [], "x", 1):
            with self.subTest(kind=type(bad).__name__):
                with self.assertRaises(ValueError) as caught:
                    validate_stress_score_support(bad)
                self.assertEqual(str(caught.exception), "invalid_stress_score_support")


class SupportBuildRejectionTests(unittest.TestCase):
    def _reject(self, mutate):
        sc = scenario(None)
        memberships = build_memberships(sc)
        mutate(memberships)
        with self.assertRaises(ValueError) as caught:
            build_stress_score_support(prediction_catalog=sc.catalog, memberships=memberships)
        self.assertEqual(str(caught.exception), "invalid_stress_score_support")

    def test_invalid_membership_scope(self):
        def missing(memberships):
            memberships["operational_contexts"] = ["CTX-Q"]

        def unknown(memberships):
            memberships["operational_contexts"] = ["CTX-F", "CTX-Q", "CTX-Z"]

        def duplicate(memberships):
            memberships["operational_contexts"] = ["CTX-F", "CTX-Q", "CTX-Q"]

        def eligible_empty(memberships):
            memberships["qc_eligible_contexts"] = []

        def eligible_fallback(memberships):
            memberships["qc_eligible_contexts"] = ["CTX-F"]

        def blank_context(memberships):
            memberships["operational_contexts"] = ["CTX-F", " CTX-Q"]

        def rows_unknown_context(memberships):
            memberships["test_rows"][0]["context_id"] = "CTX-Z"

        def rows_nonstring_key(memberships):
            # guard must reject before json normalization
            memberships["test_rows"][0]["extra"] = {1: "value"}

        def rows_nan_extra(memberships):
            # guard must reject before json normalization
            memberships["test_rows"][0]["extra"] = {"value": float("nan")}

        def crossfold_master_leakage(memberships):
            rows = memberships["test_rows"]
            shared = {row["label"] for row in rows if row["context_id"] == "CTX-F"} & {
                row["label"] for row in rows if row["context_id"] == "CTX-Q"
            }
            label = sorted(shared)[0]
            f_row = next(
                row for row in rows if row["context_id"] == "CTX-F" and row["label"] == label
            )
            q_row = next(
                row for row in rows if row["context_id"] == "CTX-Q" and row["label"] == label
            )
            q_row["master_id"] = f_row["master_id"]

        for name, mutate in (
            ("missing", missing),
            ("unknown", unknown),
            ("duplicate", duplicate),
            ("eligible_empty", eligible_empty),
            ("eligible_fallback", eligible_fallback),
            ("blank_context", blank_context),
            ("rows_unknown_context", rows_unknown_context),
            ("rows_nonstring_key", rows_nonstring_key),
            ("rows_nan_extra", rows_nan_extra),
            ("crossfold_master_leakage", crossfold_master_leakage),
        ):
            with self.subTest(name=name):
                self._reject(mutate)

    def test_invalid_rows(self):
        def missing_row(memberships):
            memberships["test_rows"].pop()

        def duplicate_row(memberships):
            memberships["test_rows"].append(copy.deepcopy(memberships["test_rows"][0]))

        def unknown_uid(memberships):
            memberships["test_rows"][0]["observation_uid"] = "ghost-uid"

        def bool_fold(memberships):
            memberships["test_rows"][0]["outer_fold"] = True

        def float_repeat(memberships):
            memberships["test_rows"][0]["outer_repeat"] = 1.0

        def blank_master(memberships):
            memberships["test_rows"][0]["master_id"] = "  "

        def extra_key(memberships):
            memberships["test_rows"][0]["unexpected"] = 1

        def malformed_json(memberships):
            memberships["test_rows"][0]["outer_repeat"] = {1}

        def not_a_list(memberships):
            memberships["test_rows"] = "not-a-list"

        for name, mutate in (
            ("missing_row", missing_row),
            ("duplicate_row", duplicate_row),
            ("unknown_uid", unknown_uid),
            ("bool_fold", bool_fold),
            ("float_repeat", float_repeat),
            ("blank_master", blank_master),
            ("extra_key", extra_key),
            ("malformed_json", malformed_json),
            ("not_a_list", not_a_list),
        ):
            with self.subTest(name=name):
                self._reject(mutate)

    def test_invalid_relations(self):
        def duplicate_fold(memberships):
            for row in memberships["test_rows"]:
                if row["context_id"] == "CTX-Q":
                    row["outer_fold"] = 0

        def master_label_conflict(memberships):
            rows = memberships["test_rows"]
            first = rows[0]
            for row in rows:
                if (
                    row["context_id"] == first["context_id"]
                    and row["observation_uid"] != first["observation_uid"]
                ):
                    row["master_id"] = first["master_id"]
                    break

        def domain_station_mismatch(memberships):
            memberships["test_rows"][0]["station"] = "other"

        def cross_context_uid(memberships):
            rows = memberships["test_rows"]
            donor = next(row for row in rows if row["context_id"] == "CTX-F")
            victim = next(row for row in rows if row["context_id"] == "CTX-Q")
            victim["observation_uid"] = donor["observation_uid"]

        for name, mutate in (
            ("duplicate_fold", duplicate_fold),
            ("master_label_conflict", master_label_conflict),
            ("domain_station_mismatch", domain_station_mismatch),
            ("cross_context_uid", cross_context_uid),
        ):
            with self.subTest(name=name):
                self._reject(mutate)

    def test_build_does_not_mutate_inputs(self):
        sc = scenario(None)
        catalog = sc.catalog
        memberships = build_memberships(sc)
        catalog_snapshot = copy.deepcopy(catalog)
        memberships_snapshot = copy.deepcopy(memberships)
        build_stress_score_support(prediction_catalog=catalog, memberships=memberships)
        self.assertEqual(catalog, catalog_snapshot)
        self.assertEqual(memberships, memberships_snapshot)


class ScoreIteratorGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc, cls.support = support_for(None)

    def test_forged_support_rejected(self):
        forged = copy.deepcopy(self.support)
        forged["summary"]["score_job_count"] += 1
        forged["catalog_sha256"] = independent_hash(
            {name: value for name, value in forged.items() if name != "catalog_sha256"}
        )
        with self.assertRaises(ValueError) as caught:
            iter_stress_score_records(forged)
        self.assertEqual(str(caught.exception), "invalid_stress_score_metadata")

    def test_non_mapping_support_rejected(self):
        for bad in (None, "nope", [], 3):
            with self.subTest(kind=type(bad).__name__):
                with self.assertRaises(ValueError) as caught:
                    iter_stress_score_records(bad)
                self.assertEqual(str(caught.exception), "invalid_stress_score_metadata")

    def test_lazy_snapshot_immunity(self):
        support = copy.deepcopy(self.support)
        expected = list(iter_stress_score_records(support))
        iterator = iter_stress_score_records(support)
        support["summary"]["score_job_count"] = -1
        support["summary"]["stage_counts"][CONTEXT_CASE] = 0
        support["pooled_procedures"] = ["forged"]
        self.assertEqual(list(iterator), expected)

    def test_output_objects_are_independent(self):
        first = list(iter_stress_score_records(self.support))
        second = list(iter_stress_score_records(self.support))
        self.assertEqual(first, second)
        self.assertIsNot(first[0], second[0])
        original = second[0]["resolution"]
        first[0]["resolution"] = "MUTATED"
        self.assertEqual(second[0]["resolution"], original)
