"""P08-T259 metadata-only tests for the stress inference operation ledger.

Invented parent fixtures, content-derived identifiers, independent hashes and
structural counts only.  No scores, weights, fits, draws, statistics, arrays
or numeric parity are computed, authorized or accepted.  The complete emitted
stream is hashed exactly once in a single cached audit; no private helper is
imported and no filesystem data is written.
"""

from __future__ import annotations

import copy
import unittest

from atlas_sers.evaluation.p08_stress_contrasts import build_stress_contrast_catalog
from atlas_sers.evaluation.p08_stress_inference_plan import (
    build_stress_inference_plan,
    iter_stress_inference_jobs,
    require_scientific_execution,
    validate_stress_inference_plan,
)
from tests.test_p08_perturbation_predictions import independent_hash
from tests.test_p08_perturbation_scores import MIXED4, PSEUDO4
from tests.test_p08_stress_contrasts import build, families_for, parent_for, registry

INVALID = "invalid_stress_inference_plan"
DENIED = "scientific_execution_not_authorized"
SCHEMA = "nato-sers-p08-stress-inference-plan-v1"

BATCH_SIZE = 128
BATCH_COUNT = 79
TOTAL_DRAWS = 10000
TRAILING_DRAWS = 16

FAMILY_SIZES = {
    "universal_robustness_effects": 120,
    "operational_qc_robustness_effects": 48,
    "operational_robustness_model_interactions": 192,
    "eligible_qc_robustness_effects": 48,
    "eligible_qc_robustness_model_interactions": 48,
}
WEIGHT_VIEWS = {"fixed_context", "fixed_pooled", "paired_context", "paired_pooled"}
SIGN_VIEWS = {"fixed_context", "paired_context"}
SIGN_UNITS = {"domain", "instrument_identity"}

STAGES = (
    "global_weight_authentication",
    "support_assessment",
    "point_estimate",
    "unit_weight_parity",
    "weighted_batch",
    "weighted_summary",
    "hierarchy_realization_prepare",
    "hierarchy_batch",
    "hierarchy_summary",
    "sign_sensitivity",
    "holm_adjustment",
    "domain_stability",
    "deletion_stability",
)

_CACHE = {}


def default_catalog():
    if "catalog" not in _CACHE:
        _CACHE["catalog"] = build(None)
    return _CACHE["catalog"]


def default_plan():
    if "plan" not in _CACHE:
        _CACHE["plan"] = build_stress_inference_plan(contrast_catalog=default_catalog())
    return _CACHE["plan"]


def supports_of(catalog):
    return {record["support_id"]: record for record in catalog["support_records"]}


def support_by_contrast(catalog):
    return {c["contrast_id"]: c["support_id"] for c in catalog["contrasts"]}


def binding_by_contrast(catalog):
    return {c["contrast_id"]: c["contrast_binding_sha256"] for c in catalog["contrasts"]}


def family_members(catalog):
    members = {}
    for contrast in catalog["contrasts"]:
        members.setdefault(contrast["multiplicity_family"], []).append(contrast["contrast_id"])
    return {label: sorted(ids) for label, ids in members.items()}


def deletion_count(catalog):
    supports = supports_of(catalog)
    return sum(
        len(supports[c["support_id"]]["domains"])
        + len(supports[c["support_id"]]["instruments"])
        + len(supports[c["support_id"]]["known_platform_families"])
        for c in catalog["contrasts"]
    )


def expected_stage_counts(catalog):
    count = len(catalog["contrasts"])
    return {
        "global_weight_authentication": 2,
        "support_assessment": 2 * count,
        "point_estimate": 4 * count,
        "unit_weight_parity": 12 * count,
        "weighted_batch": 12 * BATCH_COUNT * count,
        "weighted_summary": 12 * count,
        "hierarchy_realization_prepare": count,
        "hierarchy_batch": BATCH_COUNT * count,
        "hierarchy_summary": count,
        "sign_sensitivity": 4 * count,
        "holm_adjustment": len(FAMILY_SIZES) * 2,
        "domain_stability": count,
        "deletion_stability": deletion_count(catalog),
    }


def rehashed(mutate):
    plan = copy.deepcopy(default_plan())
    mutate(plan)
    body = {name: value for name, value in plan.items() if name != "plan_sha256"}
    plan["plan_sha256"] = independent_hash(body)
    return plan


class PlanSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = default_catalog()
        cls.plan = default_plan()

    def test_root_flags_schema_and_source_pin(self):
        plan = self.plan
        self.assertEqual(plan["schema_version"], SCHEMA)
        for flag in (
            "execution_authorized",
            "numerical_inference_accepted",
            "full_stress_job_ledger_complete",
        ):
            self.assertIs(plan[flag], False)
        self.assertIs(type(plan["scientific_operations"]), int)
        self.assertEqual(plan["scientific_operations"], 0)
        self.assertEqual(plan["source_contrast_catalog_sha256"], self.catalog["catalog_sha256"])
        for key in (
            "schema_version",
            "execution_authorized",
            "scientific_operations",
            "source_contrast_catalog_sha256",
            "shared_batches",
            "global_weight_authentication",
            "contrasts",
            "operation_blocks",
            "summary",
            "plan_sha256",
        ):
            self.assertIn(key, plan)

    def test_plan_digest_is_content_derived(self):
        body = {name: value for name, value in self.plan.items() if name != "plan_sha256"}
        self.assertEqual(self.plan["plan_sha256"], independent_hash(body))

    def test_shared_batches_exact(self):
        batches = self.plan["shared_batches"]
        self.assertEqual(batches["batch_size"], BATCH_SIZE)
        self.assertEqual(batches["batch_count"], BATCH_COUNT)
        self.assertEqual(batches["total_draws"], TOTAL_DRAWS)
        self.assertEqual(batches["trailing_draw_count"], TRAILING_DRAWS)
        expected = [
            [index * BATCH_SIZE, min((index + 1) * BATCH_SIZE, TOTAL_DRAWS)]
            for index in range(BATCH_COUNT)
        ]
        self.assertEqual(batches["ranges"], expected)
        self.assertEqual(batches["ranges"][0], [0, BATCH_SIZE])
        self.assertEqual(batches["ranges"][-1], [9984, TOTAL_DRAWS])

    def test_stage_blocks_and_summary_recompute(self):
        expected = expected_stage_counts(self.catalog)
        summary = self.plan["summary"]
        blocks = {block["stage"]: block for block in self.plan["operation_blocks"]}
        self.assertEqual(set(blocks), set(STAGES))
        self.assertEqual(summary["stage_counts"], expected)
        for stage, block in blocks.items():
            self.assertIs(type(block["allocated"]), int)
            self.assertEqual(block["allocated"], expected[stage])
            self.assertEqual(
                block["unconditional_allocated"] + block["conditional_allocated"],
                block["allocated"],
            )
        self.assertEqual(summary["total_operation_descriptors"], sum(expected.values()))
        self.assertEqual(
            summary["stage_unconditional_counts"],
            {stage: block["unconditional_allocated"] for stage, block in blocks.items()},
        )
        self.assertEqual(
            summary["stage_conditional_counts"],
            {stage: block["conditional_allocated"] for stage, block in blocks.items()},
        )
        self.assertEqual(
            summary["unconditional_operation_descriptors"],
            sum(block["unconditional_allocated"] for block in blocks.values()),
        )
        self.assertEqual(
            summary["conditional_operation_descriptors"],
            sum(block["conditional_allocated"] for block in blocks.values()),
        )
        self.assertIsNone(summary["conditional_activated_slots"])
        self.assertEqual(summary["activation_status"], "unresolved_until_runtime")

    def test_sign_and_weight_bounds(self):
        supports = supports_of(self.catalog)
        exact = sum(
            (1 << len(supports[c["support_id"]]["domains"]))
            + (1 << len(supports[c["support_id"]]["instruments"]))
            for c in self.catalog["contrasts"]
        )
        summary = self.plan["summary"]
        self.assertEqual(summary["fixed_sign_assignment_exact_total"], exact)
        self.assertEqual(summary["paired_sign_assignment_upper_bound"], exact)
        count = len(self.catalog["contrasts"])
        self.assertEqual(
            summary["weighted_terminal_scalar_upper_bound"],
            count * len(WEIGHT_VIEWS) * 3 * TOTAL_DRAWS,
        )
        self.assertEqual(summary["holm_family_sizes"], FAMILY_SIZES)

    def test_global_weight_records(self):
        records = self.plan["global_weight_authentication"]
        self.assertEqual(len(records), 2)
        by_weight = {record["weight_id"]: record for record in records}
        self.assertEqual(set(by_weight), {"master", "instrument"})
        self.assertEqual(by_weight["master"]["seed"], 2026093001)
        self.assertEqual(by_weight["instrument"]["seed"], 2026093002)
        for record in records:
            self.assertEqual(record["generator"], "PCG64")
            self.assertEqual(record["dtype"], "float64")
            self.assertEqual(record["draw_count"], TOTAL_DRAWS)
            self.assertEqual(record["identity_order"], "lexicographic")
            self.assertEqual(len(record["identity_sha256"]), 64)
        self.assertEqual(
            by_weight["master"]["identity_count"], len(self.catalog["master_weight_columns"])
        )
        self.assertEqual(
            by_weight["instrument"]["identity_count"],
            len(self.catalog["instrument_weight_columns"]),
        )

    def test_build_does_not_mutate_parent(self):
        catalog = build(None)
        before = copy.deepcopy(catalog)
        build_stress_inference_plan(contrast_catalog=catalog)
        self.assertEqual(catalog, before)


class PlanValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = default_catalog()
        cls.plan = default_plan()

    def test_roundtrip_snapshot_is_independent(self):
        snapshot = validate_stress_inference_plan(self.plan, contrast_catalog=self.catalog)
        self.assertEqual(snapshot, self.plan)
        self.assertIsNot(snapshot, self.plan)
        self.assertIsNot(snapshot["summary"], self.plan["summary"])
        self.assertIsNot(
            snapshot["shared_batches"]["ranges"], self.plan["shared_batches"]["ranges"]
        )
        self.assertEqual(snapshot["plan_sha256"], self.plan["plan_sha256"])

    def test_snapshot_clone_is_deep(self):
        expected = expected_stage_counts(self.catalog)
        snapshot = validate_stress_inference_plan(self.plan, contrast_catalog=self.catalog)
        snapshot["shared_batches"]["ranges"][0][0] = -1
        snapshot["summary"]["stage_counts"]["weighted_batch"] = 0
        snapshot["contrasts"][0]["contrast_id"] = "forged"
        snapshot["global_weight_authentication"][0]["seed"] = 0
        self.assertEqual(self.plan["shared_batches"]["ranges"][0][0], 0)
        self.assertEqual(
            self.plan["summary"]["stage_counts"]["weighted_batch"], expected["weighted_batch"]
        )
        self.assertNotEqual(
            snapshot["contrasts"][0]["contrast_id"], self.plan["contrasts"][0]["contrast_id"]
        )
        self.assertEqual(self.plan["global_weight_authentication"][0]["seed"], 2026093001)

    def test_rejects_non_mapping(self):
        for bad in (None, [], "x", 1):
            with self.subTest(kind=type(bad).__name__):
                with self.assertRaises(ValueError) as caught:
                    validate_stress_inference_plan(bad, contrast_catalog=self.catalog)
                self.assertEqual(str(caught.exception), INVALID)

    def test_rejects_corrupted_digest(self):
        forged = copy.deepcopy(self.plan)
        forged["plan_sha256"] = "0" * 64
        with self.assertRaises(ValueError) as caught:
            validate_stress_inference_plan(forged, contrast_catalog=self.catalog)
        self.assertEqual(str(caught.exception), INVALID)

    def test_rejects_rehashed_forgeries(self):
        cases = {
            "execution_claim": lambda p: p.__setitem__("execution_authorized", True),
            "numerical_claim": lambda p: p.__setitem__("numerical_inference_accepted", True),
            "ledger_claim": lambda p: p.__setitem__("full_stress_job_ledger_complete", True),
            "operations_bool": lambda p: p.__setitem__("scientific_operations", False),
            "operations_float": lambda p: p.__setitem__("scientific_operations", 0.0),
            "draw_range": lambda p: p["shared_batches"]["ranges"][0].__setitem__(1, 64),
            "holm_family": lambda p: p["summary"]["holm_family_sizes"].__setitem__(
                "universal_robustness_effects", 1
            ),
            "summary": lambda p: p["summary"].__setitem__("contrast_count", 1),
            "source_pin": lambda p: p.__setitem__("source_contrast_catalog_sha256", "0" * 64),
            "weight_seed": lambda p: p["global_weight_authentication"][0].__setitem__("seed", 1),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError) as caught:
                    validate_stress_inference_plan(rehashed(mutate), contrast_catalog=self.catalog)
                self.assertEqual(str(caught.exception), INVALID)

    def test_rejects_wrong_catalog_binding(self):
        with self.assertRaises(ValueError) as caught:
            validate_stress_inference_plan(self.plan, contrast_catalog=build(PSEUDO4))
        self.assertEqual(str(caught.exception), INVALID)

    def test_iterator_refuses_forgery_before_iteration(self):
        forged = rehashed(lambda p: p.__setitem__("execution_authorized", True))
        with self.assertRaises(ValueError) as caught:
            iter_stress_inference_jobs(forged, contrast_catalog=self.catalog)
        self.assertEqual(str(caught.exception), INVALID)

    def test_require_scientific_execution_always_denied(self):
        for args in ((), ({},), ({"execution_authorized": True},), (self.plan,)):
            with self.subTest(arity=len(args)):
                with self.assertRaises(ValueError) as caught:
                    require_scientific_execution(*args)
                self.assertEqual(str(caught.exception), DENIED)


class FixtureBuildTests(unittest.TestCase):
    def test_pseudo_and_mixed_catalogs_build_and_validate(self):
        for label, entries in (("pseudo4", PSEUDO4), ("mixed4", MIXED4)):
            with self.subTest(fixture=label):
                catalog = build(entries)
                plan = build_stress_inference_plan(contrast_catalog=catalog)
                snapshot = validate_stress_inference_plan(plan, contrast_catalog=catalog)
                self.assertEqual(snapshot, plan)
                self.assertEqual(plan["summary"]["contrast_count"], len(catalog["contrasts"]))
                self.assertEqual(plan["summary"]["stage_counts"], expected_stage_counts(catalog))

    def test_unknown_family_mapping_drops_known_deletions(self):
        parent = parent_for(None)
        mapping = {instrument: None for instrument in parent["global_instrument_ids"]}
        catalog = build(None, platform_families=mapping)
        plan = build_stress_inference_plan(contrast_catalog=catalog)
        supports = supports_of(catalog)
        expected = sum(
            len(supports[c["support_id"]]["domains"])
            + len(supports[c["support_id"]]["instruments"])
            for c in catalog["contrasts"]
        )
        self.assertEqual(plan["summary"]["stage_counts"]["deletion_stability"], expected)
        self.assertTrue(all(not record["known_platform_families"] for record in supports.values()))

    def test_weight_job_identity_tracks_source_catalog(self):
        parent = parent_for(None)
        catalog_a = build(None)
        catalog_b = build_stress_contrast_catalog(
            score_catalog=parent,
            inference_registry=registry(),
            platform_families=families_for(parent, value="other-platform"),
        )
        plan_a = build_stress_inference_plan(contrast_catalog=catalog_a)
        plan_b = build_stress_inference_plan(contrast_catalog=catalog_b)
        self.assertNotEqual(
            plan_a["source_contrast_catalog_sha256"],
            plan_b["source_contrast_catalog_sha256"],
        )
        first_a = next(iter_stress_inference_jobs(plan_a, contrast_catalog=catalog_a))
        first_b = next(iter_stress_inference_jobs(plan_b, contrast_catalog=catalog_b))
        self.assertEqual(first_a["stage"], "global_weight_authentication")
        self.assertEqual(first_b["stage"], "global_weight_authentication")
        for job in (first_a, first_b):
            body = {name: value for name, value in job.items() if name != "job_id"}
            self.assertEqual(job["job_id"], independent_hash(body))
        self.assertNotEqual(first_a["job_id"], first_b["job_id"])

    def test_iterator_is_lazy_and_outputs_are_independent(self):
        catalog = default_catalog()
        iterator = iter_stress_inference_jobs(default_plan(), contrast_catalog=catalog)
        self.assertTrue(hasattr(iterator, "__next__"))
        first = next(job for job in iterator if job["stage"] == "weighted_batch")
        body = {name: value for name, value in first.items() if name != "job_id"}
        self.assertEqual(first["job_id"], independent_hash(body))
        first_bindings = [
            external
            for external in first["external_dependencies"]
            if external.get("kind") == "contrast_binding"
        ]
        self.assertTrue(first_bindings)
        original_contrast_id = first_bindings[0]["contrast_id"]
        self.assertEqual(
            first_bindings[0]["contrast_binding_sha256"],
            binding_by_contrast(catalog)[original_contrast_id],
        )
        first["external_dependencies"].append({"kind": "forged"})
        first_bindings[0]["contrast_id"] = "forged-contrast"
        second = next(iterator)
        second_body = {name: value for name, value in second.items() if name != "job_id"}
        self.assertEqual(second["job_id"], independent_hash(second_body))
        self.assertNotIn({"kind": "forged"}, second.get("external_dependencies", []))
        second_bindings = [
            external
            for external in second.get("external_dependencies", [])
            if external.get("kind") == "contrast_binding"
        ]
        self.assertTrue(second_bindings)
        self.assertEqual(second_bindings[0]["contrast_id"], original_contrast_id)
        self.assertEqual(
            second_bindings[0]["contrast_binding_sha256"],
            binding_by_contrast(catalog)[second_bindings[0]["contrast_id"]],
        )
        self.assertNotIn(
            "forged-contrast",
            {external.get("contrast_id") for external in second_bindings},
        )


class StreamAuditTests(unittest.TestCase):
    def _activation(self, job):
        view = job["view_id"]
        self.assertIn(view, WEIGHT_VIEWS)
        expected = "conditional" if view.startswith("paired") else "unconditional"
        self.assertEqual(job["activation"], expected)

    def _hierarchy_prepare(self, job):
        self.assertEqual(job["view_id"], "fixed_context")
        self.assertEqual(
            job["class_draw_strategy"],
            "single_multinomial_per_domain_class_size_all_selected_occurrences",
        )
        self.assertIs(job["domain_index_array_first"], True)
        self.assertEqual(job["master_identity_order"], "lexicographic")
        self.assertIs(job["all_realizations_before_score_batches"], True)
        self.assertEqual(job["seed"], 2026093003)
        self.assertIs(job["reset_per_contrast"], True)

    def test_full_stream_audit(self):
        catalog = default_catalog()
        plan = default_plan()
        catalog_hash = catalog["catalog_sha256"]
        plan_hash = plan["plan_sha256"]
        catalog_digest = independent_hash(catalog)
        plan_digest = independent_hash(plan)
        supports = supports_of(catalog)
        support_of = support_by_contrast(catalog)
        bindings = binding_by_contrast(catalog)

        seen = set()
        stage_counts = {}
        prefixes = {}
        sign_index = {}
        support_jobs = {}
        point_ids = {}
        prepare_ids = {}
        holm_jobs = []
        weighted_groups = {}
        hierarchy_groups = {}
        auth_ids = set()
        auth_by_weight = {}
        first_parity = None

        for job in iter_stress_inference_jobs(plan, contrast_catalog=catalog):
            stage = job["stage"]
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            body = {name: value for name, value in job.items() if name != "job_id"}
            assert job["job_id"] == independent_hash(body), stage
            assert job["source_contrast_catalog_sha256"] == catalog_hash, stage
            job_id = job["job_id"]
            assert job_id not in seen, stage
            dependencies = set(job.get("dependencies", []))
            for dependency in dependencies:
                assert dependency in seen, stage
            seen.add(job_id)
            if len(prefixes.setdefault(stage, [])) < 3:
                prefixes[stage].append(job)
            for external in job.get("external_dependencies", []):
                if external.get("kind") == "contrast_binding":
                    assert (
                        external["contrast_binding_sha256"] == bindings[external["contrast_id"]]
                    ), stage

            if stage == "global_weight_authentication":
                auth_ids.add(job_id)
                auth_by_weight[job["weight_id"]] = job
            elif stage == "support_assessment":
                support_jobs[(job["contrast_id"], job["target_kind"])] = job
            elif stage == "point_estimate":
                point_ids[(job["contrast_id"], job["view_id"])] = job_id
                self._activation(job)
            elif stage == "unit_weight_parity":
                self.assertTrue(auth_ids <= dependencies)
                self._activation(job)
                if first_parity is None:
                    first_parity = job
            elif stage == "weighted_batch":
                self.assertTrue(auth_ids <= dependencies)
                self._activation(job)
                assert job["draw_start"] == job["batch_index"] * BATCH_SIZE
                assert job["draw_stop"] == min(job["draw_start"] + BATCH_SIZE, TOTAL_DRAWS)
                weighted_groups.setdefault(
                    (job["contrast_id"], job["view_id"], job["mode"]), set()
                ).add(job["batch_index"])
            elif stage == "weighted_summary":
                self._activation(job)
            elif stage == "hierarchy_realization_prepare":
                prepare_ids[job["contrast_id"]] = job_id
                self._hierarchy_prepare(job)
            elif stage == "hierarchy_batch":
                self.assertEqual(job["view_id"], "fixed_context")
                assert prepare_ids[job["contrast_id"]] in dependencies
                assert point_ids[(job["contrast_id"], "fixed_context")] in dependencies
                hierarchy_groups.setdefault(job["contrast_id"], set()).add(job["batch_index"])
            elif stage == "sign_sensitivity":
                support = supports[support_of[job["contrast_id"]]]
                width = (
                    len(support["domains"])
                    if job["sign_unit"] == "domain"
                    else len(support["instruments"])
                )
                self.assertEqual(job["assignment_count"], 1 << width)
                expected_kind = "exact" if job["view_id"] == "fixed_context" else "upper_bound"
                self.assertEqual(job["assignment_count_kind"], expected_kind)
                sign_index[job_id] = (job["contrast_id"], job["view_id"], job["sign_unit"])
            elif stage == "holm_adjustment":
                holm_jobs.append(job)

        expected = expected_stage_counts(catalog)
        self.assertEqual(set(stage_counts), set(STAGES))
        self.assertEqual(stage_counts, expected)
        self.assertEqual(len(seen), sum(expected.values()))
        self.assertEqual(len(weighted_groups), len(catalog["contrasts"]) * 12)
        for indices in weighted_groups.values():
            self.assertEqual(indices, set(range(BATCH_COUNT)))
        self.assertEqual(len(hierarchy_groups), len(catalog["contrasts"]))
        for indices in hierarchy_groups.values():
            self.assertEqual(indices, set(range(BATCH_COUNT)))
        self.assertEqual(len(sign_index), len(catalog["contrasts"]) * 4)
        self.assertEqual({view for _, view, _ in sign_index.values()}, SIGN_VIEWS)

        self.assertEqual(len(auth_ids), 2)
        self.assertEqual(set(auth_by_weight), {"master", "instrument"})
        records_by_weight = {
            record["weight_id"]: record for record in plan["global_weight_authentication"]
        }
        self.assertEqual(set(records_by_weight), {"master", "instrument"})
        for weight_id, record in records_by_weight.items():
            job = auth_by_weight[weight_id]
            self.assertEqual(job["stage"], "global_weight_authentication")
            self.assertEqual(job["identity_count"], record["identity_count"])
            self.assertEqual(job["identity_sha256"], record["identity_sha256"])
            self.assertEqual(job["seed"], record["seed"])
            self.assertEqual(job["dtype"], record["dtype"])
            self.assertEqual(job["draw_count"], record["draw_count"])
            self.assertEqual(job["generator"], record["generator"])
            self.assertEqual(job["positive_finite_only"], record["positive_finite_only"])
            self.assertEqual(job["identity_order"], record["identity_order"])
            self.assertEqual(job["identity_order"], "lexicographic")

        expected_support_keys = {
            (contrast_id, kind) for contrast_id in support_of for kind in ("context", "pooled")
        }
        self.assertEqual(set(support_jobs), expected_support_keys)
        self.assertEqual(len(support_jobs), 2 * len(catalog["contrasts"]))
        for (contrast_id, target_kind), job in support_jobs.items():
            support = supports[support_of[contrast_id]]
            expected_ids = list(
                support["context_ids"]
                if target_kind == "context"
                else support["complete_pool_group_ids"]
            )
            self.assertEqual(job["fixed_target_ids"], expected_ids)
            self.assertEqual(job["fixed_target_count"], len(expected_ids))
            self.assertEqual(job["fixed_target_sha256"], independent_hash(expected_ids))
            self.assertEqual(job["paired_candidate_ids"], expected_ids)
            self.assertEqual(job["paired_candidate_count"], len(expected_ids))
            self.assertEqual(job["paired_candidate_sha256"], independent_hash(expected_ids))
            self.assertEqual(job["paired_candidate_nonempty"], bool(expected_ids))
        self.assertTrue(
            any(job["paired_candidate_nonempty"] is False for job in support_jobs.values())
        )

        self.assertIsNotNone(first_parity)
        refs = {
            ref["weight_id"]: ref
            for ref in first_parity["external_dependencies"]
            if ref.get("kind") == "shared_global_weight"
        }
        self.assertEqual(set(refs), {"master", "instrument"})
        for weight_id, ref in refs.items():
            record = auth_by_weight[weight_id]
            self.assertEqual(ref["identity_sha256"], record["identity_sha256"])
            self.assertEqual(ref["seed"], record["seed"])
            self.assertEqual(ref["dtype"], record["dtype"])
            self.assertEqual(ref["generator"], record["generator"])
            self.assertEqual(ref["draw_count"], record["draw_count"])

        members = family_members(catalog)
        self.assertEqual(len(holm_jobs), 10)
        self.assertEqual(
            {(job["multiplicity_family"], job["sign_unit"]) for job in holm_jobs},
            {(family, unit) for family in FAMILY_SIZES for unit in SIGN_UNITS},
        )
        for job in holm_jobs:
            family = job["multiplicity_family"]
            unit = job["sign_unit"]
            self.assertEqual(job["family_size"], FAMILY_SIZES[family])
            self.assertEqual(job["slot_count"], len(members[family]))
            family_set = set(members[family])
            expected_ids = sorted(
                sign_id
                for sign_id, (contrast_id, view, sign_unit) in sign_index.items()
                if view == "fixed_context" and sign_unit == unit and contrast_id in family_set
            )
            self.assertEqual(sorted(job["dependencies"]), expected_ids)
            self.assertIsInstance(job["missing_policy"], str)
            self.assertTrue(job["missing_policy"])

        self.assertEqual(independent_hash(catalog), catalog_digest)
        self.assertEqual(independent_hash(plan), plan_digest)
        self.assertEqual(catalog["catalog_sha256"], catalog_hash)
        self.assertEqual(plan["plan_sha256"], plan_hash)


if __name__ == "__main__":
    unittest.main()
