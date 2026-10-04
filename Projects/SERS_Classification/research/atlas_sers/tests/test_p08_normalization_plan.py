"""Tests for the P08-T159 approved N2 normalization/control slot planner."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

from atlas_sers.evaluation.p08_normalization_plan import (
    DETERMINISTIC_SEED,
    FAMILY_CANDIDATE_COUNTS,
    FAMILY_MODELS,
    MASTER_MODE,
    NOT_APPLICABLE,
    POLICIES,
    POLICY_REPRESENTATION,
    PSEUDO_MODE,
    REASON_CODES,
    RESOLVE_SELECTED_CANDIDATE,
    SCHEMA_VERSION,
    SOURCE_SELECTION_DEPENDENT,
    STOCHASTIC_FAMILIES,
    STOCHASTIC_SEEDS,
    NormalizationPlanError,
    build_normalization_plan,
    require_scientific_execution,
)

SENTINEL = "SENTINEL-PRIVATE-424242"

LITERAL_CANDIDATE_COUNTS = {
    "C-SPECTRAL-MATCH": 3,
    "C-NEAREST-CENTROID": 8,
    "C-PCA-LDA": 10,
    "C-PLS-DA": 5,
    "C-LOGREG-EN": 30,
    "C-RBF-SVM": 36,
    "C-RANDOM-FOREST": 16,
    "C-EXTRA-TREES": 16,
}
LITERAL_STOCHASTIC_FAMILIES = frozenset(("C-RANDOM-FOREST", "C-EXTRA-TREES"))
LITERAL_STOCHASTIC_SEEDS = (20260805, 20260817, 20260829)

REQUIRED_JOB_KEYS = (
    "job_id",
    "policy_id",
    "representation_id",
    "array_sha256",
    "context_id",
    "model_id",
    "model_spec_sha256",
    "stage",
    "unit_id",
    "seed",
    "candidate_id",
    "hyperparameter_sha256",
    "fit_uid_sha256",
    "validation_uid_sha256",
    "test_uid_sha256",
    "dependencies",
    "resolution",
    "evidence_status",
    "source_family_mapping_sha256",
    "saved_selection_state_sha256",
)


def recompute(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha(text):
    return hashlib.sha256(f"p08-n2::{text}".encode()).hexdigest()


def family_seeds(model_id):
    if model_id in STOCHASTIC_FAMILIES:
        return tuple(STOCHASTIC_SEEDS)
    return (DETERMINISTIC_SEED,)


def literal_seed_count(model_id):
    if model_id in LITERAL_STOCHASTIC_FAMILIES:
        return len(LITERAL_STOCHASTIC_SEEDS)
    return 1


def literal_expected_stage_counts(model_id, mode, selection_units):
    candidates = LITERAL_CANDIDATE_COUNTS[model_id]
    seeds = literal_seed_count(model_id)
    source = selection_units * candidates * seeds
    counts = {
        "source_fit": source,
        "source_validation_prediction": source,
        "select_hyperparameters": 1,
        "scalar_calibration": 1,
        "final_refit": seeds,
        "held_prediction": seeds,
        "seed_ensemble_prediction": 1,
    }
    if mode == MASTER_MODE:
        counts["calibration_prediction_alias"] = 3 * seeds
    else:
        counts["calibration_model_fit"] = 3 * seeds
        counts["calibration_validation_prediction"] = 3 * seeds
    return counts


def make_candidates():
    candidates = []
    for model_id in FAMILY_MODELS:
        for index in range(FAMILY_CANDIDATE_COUNTS[model_id]):
            candidates.append(
                {
                    "candidate_id": f"{model_id}-{index:03d}",
                    "model_id": model_id,
                    "hyperparameter_sha256": sha(f"hp::{model_id}::{index}"),
                }
            )
    return candidates


def candidate_ids(model_id):
    return {
        f"{model_id}-{index:03d}"
        for index in range(FAMILY_CANDIDATE_COUNTS[model_id])
    }


def make_actions(tag="default"):
    return {
        "R_SNV_400_1800": sha(f"{tag}::array::SNV"),
        "R_VECTOR_400_1800": sha(f"{tag}::array::VECTOR"),
        "R_AREA_400_1800": sha(f"{tag}::array::AREA"),
        "R_D1_400_1800": sha(f"{tag}::array::D1"),
    }


def make_model_spec(tag="default"):
    return {model_id: sha(f"{tag}::spec::{model_id}") for model_id in FAMILY_MODELS}


def make_unit(unit_id, tag):
    return {
        "unit_id": unit_id,
        "fit_uid_sha256": sha(f"{tag}::fit::{unit_id}"),
        "validation_uid_sha256": sha(f"{tag}::val::{unit_id}"),
    }


def make_master_context(context_id, model_id="C-RBF-SVM"):
    units = [make_unit(f"{context_id}-MU-{index}", context_id) for index in range(3)]
    return {
        "context_id": context_id,
        "selected_model_id": model_id,
        "selection_state_sha256": sha(f"state::{context_id}"),
        "selection_mode": MASTER_MODE,
        "outer_fit_uid_sha256": sha(f"outer-fit::{context_id}"),
        "outer_test_uid_sha256": sha(f"outer-test::{context_id}"),
        "selection_units": units,
        "calibration_units": [dict(unit) for unit in units],
        "historical_reference_available": True,
    }


def make_pseudo_context(context_id, model_id="C-RBF-SVM"):
    selection = [make_unit(f"{context_id}-SU-{index}", context_id) for index in range(2)]
    calibration = [make_unit(f"{context_id}-CU-{index}", context_id) for index in range(3)]
    return {
        "context_id": context_id,
        "selected_model_id": model_id,
        "selection_state_sha256": sha(f"state::{context_id}"),
        "selection_mode": PSEUDO_MODE,
        "outer_fit_uid_sha256": sha(f"outer-fit::{context_id}"),
        "outer_test_uid_sha256": sha(f"outer-test::{context_id}"),
        "selection_units": selection,
        "calibration_units": calibration,
        "historical_reference_available": True,
    }


def make_pseudo_context_with_selection(context_id, model_id, selection_count):
    context = make_pseudo_context(context_id, model_id)
    context["selection_units"] = [
        make_unit(f"{context_id}-SU-{index}", context_id)
        for index in range(selection_count)
    ]
    return context


def all_family_contexts():
    return [
        make_master_context("CTX-M-SPECTRAL", "C-SPECTRAL-MATCH"),
        make_pseudo_context("CTX-P-CENTROID", "C-NEAREST-CENTROID"),
        make_master_context("CTX-M-PCA-LDA", "C-PCA-LDA"),
        make_pseudo_context("CTX-P-PLS-DA", "C-PLS-DA"),
        make_master_context("CTX-M-LOGREG-EN", "C-LOGREG-EN"),
        make_pseudo_context("CTX-P-RBF-SVM", "C-RBF-SVM"),
        make_master_context("CTX-M-RANDOM-FOREST", "C-RANDOM-FOREST"),
        make_pseudo_context("CTX-P-EXTRA-TREES", "C-EXTRA-TREES"),
    ]


def expected_counts(model_id, mode):
    candidates = FAMILY_CANDIDATE_COUNTS[model_id]
    seeds = len(family_seeds(model_id))
    units = 3 if mode == MASTER_MODE else 2
    source = units * candidates * seeds
    counts = {
        "source_fit": source,
        "source_validation_prediction": source,
        "select_hyperparameters": 1,
        "scalar_calibration": 1,
        "final_refit": seeds,
        "held_prediction": seeds,
        "seed_ensemble_prediction": 1,
    }
    if mode == MASTER_MODE:
        counts["calibration_prediction_alias"] = 3 * seeds
    else:
        counts["calibration_model_fit"] = 3 * seeds
        counts["calibration_validation_prediction"] = 3 * seeds
    return counts


def build_plan(contexts=None, candidates=None, actions=None, spec=None, source_map=None):
    if contexts is None:
        contexts = [make_master_context("CTX-MASTER"), make_pseudo_context("CTX-PSEUDO")]
    if candidates is None:
        candidates = make_candidates()
    if actions is None:
        actions = make_actions()
    if spec is None:
        spec = make_model_spec()
    if source_map is None:
        source_map = sha("source-family-mapping")
    return build_normalization_plan(contexts, candidates, actions, spec, source_map)


def policy_context_jobs(plan, policy, context_id):
    return [
        job
        for job in plan["jobs"]
        if job["policy_id"] == policy and job["context_id"] == context_id
    ]


def stage_jobs(jobs, stage):
    return [job for job in jobs if job["stage"] == stage]


class PlanConstructionTests(unittest.TestCase):
    def test_top_level_shape_and_denied_execution(self):
        plan = build_plan()
        self.assertEqual(SCHEMA_VERSION, "nato-sers-p08-n2-slot-dag-v1")
        self.assertEqual(plan["schema_version"], SCHEMA_VERSION)
        self.assertIs(plan["execution_authorized"], False)
        self.assertEqual(len(plan["plan_sha256"]), 64)
        self.assertEqual(plan["aliases"], [])
        self.assertIn("source_family_mapping_sha256", plan)

        with self.assertRaises(NormalizationPlanError) as caught:
            require_scientific_execution(plan)
        self.assertEqual(str(caught.exception), "scientific_execution_not_authorized")
        with self.assertRaises(NormalizationPlanError):
            require_scientific_execution({"execution_authorized": True})

    def test_job_schema_exact_and_sorted(self):
        plan = build_plan(contexts=all_family_contexts())
        job_ids = [job["job_id"] for job in plan["jobs"]]
        self.assertEqual(job_ids, sorted(job_ids))
        self.assertTrue(all(job_id.startswith("P08N2JOB-") for job_id in job_ids))
        self.assertTrue(all(len(job_id) == len("P08N2JOB-") + 64 for job_id in job_ids))
        for job in plan["jobs"]:
            self.assertEqual(set(job.keys()), set(REQUIRED_JOB_KEYS))
            self.assertEqual(job["dependencies"], sorted(set(job["dependencies"])))
            self.assertEqual(job["evidence_status"], "unapproved_future_job")

    def test_unique_ids_and_hashes_independently_recomputed(self):
        plan = build_plan(contexts=all_family_contexts())
        job_ids = [job["job_id"] for job in plan["jobs"]]
        self.assertEqual(len(job_ids), len(set(job_ids)))
        for job in plan["jobs"]:
            body = {key: value for key, value in job.items() if key != "job_id"}
            self.assertEqual(job["job_id"], "P08N2JOB-" + recompute(body))
        payload = {
            "schema_version": plan["schema_version"],
            "execution_authorized": plan["execution_authorized"],
            "source_family_mapping_sha256": plan["source_family_mapping_sha256"],
            "jobs": plan["jobs"],
            "aliases": plan["aliases"],
            "summary": plan["summary"],
        }
        self.assertEqual(plan["plan_sha256"], recompute(payload))

    def test_all_families_and_frozen_grids(self):
        contexts = all_family_contexts()
        plan = build_plan(contexts=contexts)
        seen = set()
        for context in contexts:
            model_id = context["selected_model_id"]
            seen.add(model_id)
            seed_count = len(family_seeds(model_id))
            candidate_count = FAMILY_CANDIDATE_COUNTS[model_id]
            units = len(context["selection_units"])
            for policy in POLICIES:
                jobs = policy_context_jobs(plan, policy, context["context_id"])
                source = stage_jobs(jobs, "source_fit")
                self.assertEqual(len(source), units * candidate_count * seed_count)
                self.assertEqual({job["model_id"] for job in source}, {model_id})
                self.assertEqual(
                    {job["candidate_id"] for job in source}, candidate_ids(model_id)
                )
                self.assertEqual(
                    {job["seed"] for job in source}, set(family_seeds(model_id))
                )
        self.assertEqual(seen, set(FAMILY_MODELS))

    def test_stage_counts_arithmetic(self):
        contexts = [
            make_master_context("CTX-M", "C-RANDOM-FOREST"),
            make_pseudo_context("CTX-P", "C-EXTRA-TREES"),
        ]
        plan = build_plan(contexts=contexts)
        for context in contexts:
            for policy in POLICIES:
                jobs = policy_context_jobs(plan, policy, context["context_id"])
                counts = {}
                for job in jobs:
                    counts[job["stage"]] = counts.get(job["stage"], 0) + 1
                self.assertEqual(
                    counts,
                    expected_counts(
                        context["selected_model_id"], context["selection_mode"]
                    ),
                )

    def test_summary_accounting(self):
        plan = build_plan(contexts=all_family_contexts())
        summary = plan["summary"]
        for policy in POLICIES:
            entry = summary["by_policy"][policy]
            policy_jobs = [job for job in plan["jobs"] if job["policy_id"] == policy]
            self.assertTrue(policy_jobs)
            self.assertEqual(
                {job["representation_id"] for job in policy_jobs},
                {POLICY_REPRESENTATION[policy]},
            )
            independently_counted = {}
            for job in policy_jobs:
                independently_counted[job["stage"]] = (
                    independently_counted.get(job["stage"], 0) + 1
                )
            independently_counted = {
                stage: independently_counted[stage]
                for stage in sorted(independently_counted)
            }
            stage_counts = entry["stage_counts"]
            self.assertEqual(stage_counts, independently_counted)
            self.assertEqual(entry["total_jobs"], len(policy_jobs))
            self.assertEqual(
                entry["model_fit_slots"],
                stage_counts.get("source_fit", 0)
                + stage_counts.get("calibration_model_fit", 0)
                + stage_counts.get("final_refit", 0),
            )
            self.assertEqual(
                entry["scalar_calibrations"], stage_counts.get("scalar_calibration", 0)
            )
            self.assertEqual(
                entry["calibration_prediction_aliases"],
                stage_counts.get("calibration_prediction_alias", 0),
            )
        self.assertEqual(summary["authorized_fit_slots"], 0)
        self.assertIs(summary["execution_authorized"], False)
        self.assertEqual(summary["totals"]["total_jobs"], len(plan["jobs"]))

    def test_registered_grid_counts_literal(self):
        contexts = []
        for model_id in LITERAL_CANDIDATE_COUNTS:
            contexts.append(make_master_context(f"LIT-M-{model_id}", model_id))
            contexts.append(make_pseudo_context(f"LIT-P-{model_id}", model_id))
        plan = build_plan(contexts=contexts)

        for model_id in LITERAL_CANDIDATE_COUNTS:
            for mode, prefix, selection_units in (
                (MASTER_MODE, "LIT-M-", 3),
                (PSEUDO_MODE, "LIT-P-", 2),
            ):
                context_id = f"{prefix}{model_id}"
                expected = literal_expected_stage_counts(
                    model_id, mode, selection_units
                )
                for policy in POLICIES:
                    with self.subTest(model=model_id, mode=mode, policy=policy):
                        jobs = policy_context_jobs(plan, policy, context_id)
                        counts = {}
                        for job in jobs:
                            counts[job["stage"]] = counts.get(job["stage"], 0) + 1
                        counts = {stage: counts[stage] for stage in sorted(counts)}
                        self.assertEqual(counts, expected)

                        scalar = stage_jobs(jobs, "scalar_calibration")
                        self.assertEqual(len(scalar), 1)
                        calibration_outputs = [
                            job
                            for job in jobs
                            if job["stage"]
                            in (
                                "calibration_prediction_alias",
                                "calibration_validation_prediction",
                            )
                        ]
                        self.assertEqual(
                            set(scalar[0]["dependencies"]),
                            {job["job_id"] for job in calibration_outputs},
                        )

                        held = stage_jobs(jobs, "held_prediction")
                        ensemble = stage_jobs(jobs, "seed_ensemble_prediction")
                        self.assertEqual(len(ensemble), 1)
                        self.assertEqual(
                            set(ensemble[0]["dependencies"]),
                            {job["job_id"] for job in held}
                            | {scalar[0]["job_id"]},
                        )

                        seeds_seen = {
                            job["seed"] for job in stage_jobs(jobs, "source_fit")
                        }
                        if model_id in LITERAL_STOCHASTIC_FAMILIES:
                            self.assertEqual(
                                seeds_seen, set(LITERAL_STOCHASTIC_SEEDS)
                            )
                        else:
                            self.assertEqual(seeds_seen, {"deterministic"})

        summed = {}
        total_jobs = 0
        for policy in POLICIES:
            entry = plan["summary"]["by_policy"][policy]
            total_jobs += entry["total_jobs"]
            for stage, count in entry["stage_counts"].items():
                summed[stage] = summed.get(stage, 0) + count
        summed = {stage: summed[stage] for stage in sorted(summed)}
        totals = plan["summary"]["totals"]
        self.assertEqual(totals["stage_counts"], summed)
        self.assertEqual(totals["total_jobs"], total_jobs)
        self.assertEqual(total_jobs, len(plan["jobs"]))

    def test_pseudo_four_selection_units_exact_counts(self):
        model_id = "C-RBF-SVM"
        context_id = "PSEUDO-FOUR-SU"
        context = make_pseudo_context_with_selection(context_id, model_id, 4)
        self.assertEqual(len(context["selection_units"]), 4)
        self.assertEqual(len(context["calibration_units"]), 3)
        plan = build_plan(contexts=[context])
        candidates = LITERAL_CANDIDATE_COUNTS[model_id]
        seeds = literal_seed_count(model_id)
        expected = literal_expected_stage_counts(model_id, PSEUDO_MODE, 4)
        for policy in POLICIES:
            with self.subTest(policy=policy):
                jobs = policy_context_jobs(plan, policy, context_id)
                counts = {}
                for job in jobs:
                    counts[job["stage"]] = counts.get(job["stage"], 0) + 1
                counts = {stage: counts[stage] for stage in sorted(counts)}
                self.assertEqual(counts, expected)
                self.assertEqual(counts["source_fit"], 4 * candidates * seeds)
                self.assertEqual(
                    stage_jobs(jobs, "calibration_prediction_alias"), []
                )
                self.assertEqual(
                    len(stage_jobs(jobs, "calibration_model_fit")), 3 * seeds
                )

    def test_pseudo_calibration_equal_selection_still_fits_fresh(self):
        model_id = "C-EXTRA-TREES"
        context_id = "PSEUDO-EQUAL-UNITS"
        context = make_pseudo_context_with_selection(context_id, model_id, 3)
        self.assertEqual(len(context["selection_units"]), 3)
        context["calibration_units"] = [
            dict(unit) for unit in context["selection_units"]
        ]
        plan = build_plan(contexts=[context])
        seeds = literal_seed_count(model_id)
        for policy in POLICIES:
            with self.subTest(policy=policy):
                jobs = policy_context_jobs(plan, policy, context_id)
                lookup = {job["job_id"]: job for job in jobs}
                select = stage_jobs(jobs, "select_hyperparameters")
                self.assertEqual(len(select), 1)
                fits = stage_jobs(jobs, "calibration_model_fit")
                predictions = stage_jobs(jobs, "calibration_validation_prediction")
                self.assertEqual(len(fits), 3 * seeds)
                self.assertEqual(len(predictions), 3 * seeds)
                self.assertEqual(
                    stage_jobs(jobs, "calibration_prediction_alias"), []
                )
                self.assertEqual(
                    {job["unit_id"] for job in fits},
                    {unit["unit_id"] for unit in context["calibration_units"]},
                )
                for fit in fits:
                    self.assertEqual(fit["dependencies"], [select[0]["job_id"]])
                    self.assertEqual(fit["resolution"], SOURCE_SELECTION_DEPENDENT)
                for prediction in predictions:
                    self.assertEqual(len(prediction["dependencies"]), 1)
                    self.assertEqual(
                        lookup[prediction["dependencies"][0]]["stage"],
                        "calibration_model_fit",
                    )
                scalar = stage_jobs(jobs, "scalar_calibration")
                self.assertEqual(
                    set(scalar[0]["dependencies"]),
                    {job["job_id"] for job in predictions},
                )

    def test_model_spec_change_rekeys_one_family_only(self):
        contexts = [
            make_master_context("SPEC-A", "C-PLS-DA"),
            make_master_context("SPEC-B", "C-RBF-SVM"),
        ]
        base = build_plan(contexts=contexts)
        changed_spec = make_model_spec()
        changed_spec["C-PLS-DA"] = sha("mutated-spec::C-PLS-DA")
        changed = build_plan(contexts=contexts, spec=changed_spec)
        self.assertNotEqual(base["plan_sha256"], changed["plan_sha256"])

        def ids_by_model(plan):
            return {
                model_id: {
                    job["job_id"]
                    for job in plan["jobs"]
                    if job["model_id"] == model_id
                }
                for model_id in ("C-PLS-DA", "C-RBF-SVM")
            }

        base_ids = ids_by_model(base)
        changed_ids = ids_by_model(changed)
        self.assertNotEqual(base_ids["C-PLS-DA"], changed_ids["C-PLS-DA"])
        self.assertEqual(base_ids["C-RBF-SVM"], changed_ids["C-RBF-SVM"])

    def test_outer_fit_and_validation_role_hash_propagation(self):
        context_a = make_master_context("ROLL-A", "C-PLS-DA")
        context_b = make_master_context("ROLL-B", "C-RBF-SVM")
        base = build_plan(contexts=[context_a, context_b])

        outer_context = copy.deepcopy(context_a)
        outer_context["outer_fit_uid_sha256"] = sha("mutated-outer-fit::ROLL-A")
        outer_plan = build_plan(contexts=[outer_context, context_b])
        self.assertNotEqual(base["plan_sha256"], outer_plan["plan_sha256"])

        def outer_ids(plan, context_id):
            return {
                job["job_id"]
                for job in plan["jobs"]
                if job["context_id"] == context_id
                and job["stage"] in ("final_refit", "held_prediction")
            }

        self.assertNotEqual(
            outer_ids(base, "ROLL-A"), outer_ids(outer_plan, "ROLL-A")
        )
        self.assertEqual(
            outer_ids(base, "ROLL-B"), outer_ids(outer_plan, "ROLL-B")
        )
        for job in outer_plan["jobs"]:
            if job["context_id"] == "ROLL-A" and job["stage"] in (
                "final_refit",
                "held_prediction",
            ):
                self.assertEqual(
                    job["fit_uid_sha256"], outer_context["outer_fit_uid_sha256"]
                )

        validation_context = copy.deepcopy(context_a)
        new_validation = sha("mutated-validation::ROLL-A-MU-0")
        validation_context["selection_units"][0][
            "validation_uid_sha256"
        ] = new_validation
        validation_context["calibration_units"][0][
            "validation_uid_sha256"
        ] = new_validation
        validation_plan = build_plan(contexts=[validation_context, context_b])
        self.assertNotEqual(base["plan_sha256"], validation_plan["plan_sha256"])
        self.assertNotEqual(
            {job["job_id"] for job in base["jobs"] if job["context_id"] == "ROLL-A"},
            {
                job["job_id"]
                for job in validation_plan["jobs"]
                if job["context_id"] == "ROLL-A"
            },
        )
        self.assertEqual(
            {job["job_id"] for job in base["jobs"] if job["context_id"] == "ROLL-B"},
            {
                job["job_id"]
                for job in validation_plan["jobs"]
                if job["context_id"] == "ROLL-B"
            },
        )
        for job in validation_plan["jobs"]:
            if job["context_id"] == "ROLL-A" and job["unit_id"] == "ROLL-A-MU-0":
                self.assertEqual(job["validation_uid_sha256"], new_validation)

    def test_summary_provenance_rows(self):
        contexts = all_family_contexts()
        contexts[0]["historical_reference_available"] = False
        plan = build_plan(contexts=contexts)
        rows = {row["context_id"]: row for row in plan["summary"]["contexts"]}
        self.assertFalse(rows[contexts[0]["context_id"]]["historical_reference_available"])
        self.assertEqual(plan["summary"]["historical_reference_unavailable_contexts"], 1)
        for context in contexts[1:]:
            self.assertTrue(rows[context["context_id"]]["historical_reference_available"])

    def test_summary_text_contains_no_unit_ids(self):
        contexts = all_family_contexts()
        plan = build_plan(contexts=contexts)
        text = json.dumps(plan["summary"], sort_keys=True)
        for context in contexts:
            for unit in context["selection_units"] + context["calibration_units"]:
                self.assertNotIn(unit["unit_id"], text)

    def test_dependencies_share_context_action_and_seed(self):
        plan = build_plan(contexts=all_family_contexts())
        lookup = {job["job_id"]: job for job in plan["jobs"]}
        for job in plan["jobs"]:
            for dependency_id in job["dependencies"]:
                dependency = lookup[dependency_id]
                self.assertEqual(dependency["context_id"], job["context_id"])
                self.assertEqual(dependency["policy_id"], job["policy_id"])
                self.assertEqual(dependency["representation_id"], job["representation_id"])
                self.assertEqual(dependency["array_sha256"], job["array_sha256"])

    def test_source_prediction_depends_on_its_fit(self):
        plan = build_plan()
        lookup = {job["job_id"]: job for job in plan["jobs"]}
        for job in plan["jobs"]:
            if job["stage"] != "source_validation_prediction":
                continue
            self.assertEqual(len(job["dependencies"]), 1)
            dependency = lookup[job["dependencies"][0]]
            self.assertEqual(dependency["stage"], "source_fit")
            for key in (
                "policy_id",
                "representation_id",
                "array_sha256",
                "context_id",
                "model_id",
                "model_spec_sha256",
                "unit_id",
                "seed",
                "candidate_id",
                "hyperparameter_sha256",
                "fit_uid_sha256",
                "validation_uid_sha256",
                "test_uid_sha256",
                "evidence_status",
                "source_family_mapping_sha256",
                "saved_selection_state_sha256",
            ):
                self.assertEqual(dependency[key], job[key])

    def test_select_depends_on_all_source_predictions(self):
        plan = build_plan()
        for policy in POLICIES:
            for context_id in ("CTX-MASTER", "CTX-PSEUDO"):
                jobs = policy_context_jobs(plan, policy, context_id)
                lookup = {job["job_id"]: job for job in jobs}
                source_predictions = stage_jobs(jobs, "source_validation_prediction")
                select = stage_jobs(jobs, "select_hyperparameters")
                self.assertEqual(len(select), 1)
                self.assertEqual(
                    set(select[0]["dependencies"]),
                    {job["job_id"] for job in source_predictions},
                )
                for dependency_id in select[0]["dependencies"]:
                    self.assertEqual(
                        lookup[dependency_id]["stage"],
                        "source_validation_prediction",
                    )

    def test_master_calibration_alias_selector_gate(self):
        contexts = [make_master_context("CTX-MASTER", "C-RANDOM-FOREST")]
        plan = build_plan(contexts=contexts)
        model_id = "C-RANDOM-FOREST"
        seeds = family_seeds(model_id)
        for policy in POLICIES:
            jobs = policy_context_jobs(plan, policy, "CTX-MASTER")
            lookup = {job["job_id"]: job for job in jobs}
            select = stage_jobs(jobs, "select_hyperparameters")[0]
            aliases = stage_jobs(jobs, "calibration_prediction_alias")
            self.assertEqual(len(aliases), 3 * len(seeds))
            for alias in aliases:
                self.assertEqual(alias["resolution"], RESOLVE_SELECTED_CANDIDATE)
                self.assertEqual(alias["candidate_id"], SOURCE_SELECTION_DEPENDENT)
                self.assertEqual(alias["hyperparameter_sha256"], NOT_APPLICABLE)
                self.assertIn(select["job_id"], alias["dependencies"])
                source_deps = [
                    lookup[dependency_id]
                    for dependency_id in alias["dependencies"]
                    if dependency_id != select["job_id"]
                ]
                self.assertEqual(
                    {dependency["candidate_id"] for dependency in source_deps},
                    candidate_ids(model_id),
                )
                for dependency in source_deps:
                    self.assertEqual(dependency["stage"], "source_validation_prediction")
                    self.assertEqual(dependency["unit_id"], alias["unit_id"])
                    self.assertEqual(dependency["seed"], alias["seed"])
                self.assertEqual(len(source_deps), FAMILY_CANDIDATE_COUNTS[model_id])

    def test_pseudo_calibration_fits(self):
        contexts = [make_pseudo_context("CTX-PSEUDO", "C-EXTRA-TREES")]
        plan = build_plan(contexts=contexts)
        self.assertEqual(stage_jobs(plan["jobs"], "calibration_prediction_alias"), [])
        for policy in POLICIES:
            jobs = policy_context_jobs(plan, policy, "CTX-PSEUDO")
            lookup = {job["job_id"]: job for job in jobs}
            select = stage_jobs(jobs, "select_hyperparameters")[0]
            fits = stage_jobs(jobs, "calibration_model_fit")
            predictions = stage_jobs(jobs, "calibration_validation_prediction")
            self.assertEqual(len(fits), 3 * len(family_seeds("C-EXTRA-TREES")))
            self.assertEqual(len(predictions), len(fits))
            for fit in fits:
                self.assertEqual(fit["dependencies"], [select["job_id"]])
            for prediction in predictions:
                self.assertEqual(len(prediction["dependencies"]), 1)
                self.assertEqual(
                    lookup[prediction["dependencies"][0]]["stage"],
                    "calibration_model_fit",
                )

    def test_scalar_and_held_and_ensemble(self):
        plan = build_plan()
        lookup = {job["job_id"]: job for job in plan["jobs"]}
        for job in plan["jobs"]:
            if job["stage"] == "final_refit":
                self.assertEqual(len(job["dependencies"]), 1)
                self.assertEqual(
                    lookup[job["dependencies"][0]]["stage"], "select_hyperparameters"
                )
            elif job["stage"] == "held_prediction":
                self.assertEqual(len(job["dependencies"]), 1)
                dependency = lookup[job["dependencies"][0]]
                self.assertEqual(dependency["stage"], "final_refit")
                self.assertEqual(dependency["seed"], job["seed"])
                self.assertEqual(dependency["model_id"], job["model_id"])
                self.assertEqual(job["resolution"], "uncalibrated_scores_only")
            elif job["stage"] == "scalar_calibration":
                self.assertTrue(job["dependencies"])
                for dependency_id in job["dependencies"]:
                    self.assertIn(
                        lookup[dependency_id]["stage"],
                        {
                            "calibration_validation_prediction",
                            "calibration_prediction_alias",
                        },
                    )
            elif job["stage"] == "seed_ensemble_prediction":
                self.assertEqual(
                    job["resolution"], "seed_average_then_single_temperature"
                )
                dependencies = [lookup[d] for d in job["dependencies"]]
                held = [
                    dependency
                    for dependency in dependencies
                    if dependency["stage"] == "held_prediction"
                ]
                scalar = [
                    dependency
                    for dependency in dependencies
                    if dependency["stage"] == "scalar_calibration"
                ]
                self.assertEqual(len(scalar), 1)
                self.assertEqual(
                    {dependency["seed"] for dependency in held},
                    set(family_seeds(job["model_id"])),
                )
                self.assertEqual(
                    len(held), len(family_seeds(job["model_id"]))
                )

    def test_no_held_feeds_selection_scalar_or_fit(self):
        plan = build_plan(contexts=all_family_contexts())
        lookup = {job["job_id"]: job for job in plan["jobs"]}
        upstream_stages = {
            "source_fit",
            "select_hyperparameters",
            "calibration_model_fit",
            "calibration_validation_prediction",
            "calibration_prediction_alias",
            "scalar_calibration",
            "final_refit",
        }
        for job in plan["jobs"]:
            if job["stage"] not in upstream_stages:
                continue
            for dependency_id in job["dependencies"]:
                self.assertNotEqual(lookup[dependency_id]["stage"], "held_prediction")

    def test_min_winner_hyperparameters_not_reused(self):
        contexts = [make_master_context("CTX-MASTER", "C-RBF-SVM")]
        plan = build_plan(contexts=contexts)
        for policy in POLICIES:
            select = stage_jobs(
                policy_context_jobs(plan, policy, "CTX-MASTER"),
                "select_hyperparameters",
            )
            self.assertEqual(len(select), 1)
            self.assertEqual(select[0]["candidate_id"], SOURCE_SELECTION_DEPENDENT)
            self.assertEqual(select[0]["hyperparameter_sha256"], NOT_APPLICABLE)
            self.assertNotIn(select[0]["candidate_id"], candidate_ids("C-RBF-SVM"))

    def test_permutation_determinism(self):
        contexts = all_family_contexts()
        base = build_plan(contexts=copy.deepcopy(contexts))
        shuffled_contexts = copy.deepcopy(contexts)
        shuffled_contexts.reverse()
        for context in shuffled_contexts:
            context["selection_units"].reverse()
            context["calibration_units"].reverse()
        candidates = make_candidates()
        shuffled_candidates = copy.deepcopy(candidates)
        shuffled_candidates.reverse()
        actions = make_actions()
        shuffled_actions = {key: actions[key] for key in reversed(list(actions))}
        spec = make_model_spec()
        shuffled_spec = {key: spec[key] for key in reversed(list(spec))}
        permuted = build_plan(
            contexts=shuffled_contexts,
            candidates=shuffled_candidates,
            actions=shuffled_actions,
            spec=shuffled_spec,
        )
        self.assertEqual(base["plan_sha256"], permuted["plan_sha256"])
        self.assertEqual(
            [job["job_id"] for job in base["jobs"]],
            [job["job_id"] for job in permuted["jobs"]],
        )

    def test_inputs_not_mutated(self):
        contexts = all_family_contexts()
        candidates = make_candidates()
        actions = make_actions()
        spec = make_model_spec()
        source_map = sha("source-map")
        snapshots = (
            copy.deepcopy(contexts),
            copy.deepcopy(candidates),
            copy.deepcopy(actions),
            copy.deepcopy(spec),
            source_map,
        )
        build_normalization_plan(contexts, candidates, actions, spec, source_map)
        self.assertEqual(contexts, snapshots[0])
        self.assertEqual(candidates, snapshots[1])
        self.assertEqual(actions, snapshots[2])
        self.assertEqual(spec, snapshots[3])
        self.assertEqual(source_map, snapshots[4])

    def test_single_action_change_rekeys_only_that_policy(self):
        base = build_plan()
        changed_actions = make_actions()
        changed_actions["R_SNV_400_1800"] = sha("changed-snv")
        changed = build_plan(actions=changed_actions)
        self.assertNotEqual(base["plan_sha256"], changed["plan_sha256"])

        def ids_by_policy(plan):
            return {
                policy: {
                    job["job_id"] for job in plan["jobs"] if job["policy_id"] == policy
                }
                for policy in POLICIES
            }

        base_ids = ids_by_policy(base)
        changed_ids = ids_by_policy(changed)
        self.assertNotEqual(base_ids["PP-NORM-SNV"], changed_ids["PP-NORM-SNV"])
        for policy in ("PP-NORM-VECTOR", "PP-NORM-AREA", "PP-NORM-D1"):
            self.assertEqual(base_ids[policy], changed_ids[policy])

    def test_selection_state_change_keeps_family(self):
        contexts = [make_master_context("CTX-M", "C-PLS-DA")]
        base = build_plan(contexts=contexts)
        changed_contexts = copy.deepcopy(contexts)
        changed_contexts[0]["selection_state_sha256"] = sha("changed-state")
        changed = build_plan(contexts=changed_contexts)
        self.assertNotEqual(base["plan_sha256"], changed["plan_sha256"])
        self.assertEqual(
            {job["model_id"] for job in base["jobs"]}, {"C-PLS-DA"}
        )
        self.assertEqual(
            {job["model_id"] for job in changed["jobs"]}, {"C-PLS-DA"}
        )

    def test_source_map_change_rekeys_every_job(self):
        base = build_plan()
        changed = build_plan(source_map=sha("other-map"))
        base_ids = {job["job_id"] for job in base["jobs"]}
        changed_ids = {job["job_id"] for job in changed["jobs"]}
        self.assertNotEqual(base_ids, changed_ids)
        self.assertEqual(base_ids & changed_ids, set())

    def test_candidate_hyperparameter_change_rekeys_select(self):
        base = build_plan()
        changed_candidates = copy.deepcopy(make_candidates())
        for candidate in changed_candidates:
            if candidate["model_id"] == "C-RBF-SVM":
                candidate["hyperparameter_sha256"] = sha("mutated-hp")
                break
        changed = build_plan(candidates=changed_candidates)

        def select_ids(plan):
            return sorted(
                job["job_id"]
                for job in plan["jobs"]
                if job["stage"] == "select_hyperparameters"
                and job["model_id"] == "C-RBF-SVM"
            )

        self.assertNotEqual(select_ids(base), select_ids(changed))
        self.assertEqual(
            {job["model_id"] for job in base["jobs"]},
            {job["model_id"] for job in changed["jobs"]},
        )

    def test_missing_history_preserves_jobs(self):
        available = make_pseudo_context("CTX-P", "C-NEAREST-CENTROID")
        available["historical_reference_available"] = True
        missing = copy.deepcopy(available)
        missing["historical_reference_available"] = False
        plan_available = build_plan(contexts=[available])
        plan_missing = build_plan(contexts=[missing])
        self.assertEqual(plan_available["jobs"], plan_missing["jobs"])
        self.assertTrue(plan_missing["jobs"])
        self.assertEqual(
            plan_available["summary"]["historical_reference_unavailable_contexts"], 0
        )
        self.assertEqual(
            plan_missing["summary"]["historical_reference_unavailable_contexts"], 1
        )

    def test_plan_is_json_safe(self):
        plan = build_plan(contexts=all_family_contexts())
        encoded = json.dumps(plan, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["plan_sha256"], plan["plan_sha256"])
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)

    def test_no_forbidden_scientific_dependencies(self):
        import atlas_sers.evaluation.p08_normalization_plan as module

        for forbidden in ("pandas", "numpy", "sklearn", "torch"):
            self.assertNotIn(forbidden, module.__dict__)
        plan = build_plan()
        self.assertEqual(plan["summary"]["authorized_fit_slots"], 0)
        self.assertIs(plan["summary"]["execution_authorized"], False)


class ValidationTests(unittest.TestCase):
    def base_arguments(self):
        return {
            "contexts": [make_master_context("CTX-M", "C-RBF-SVM")],
            "candidates": make_candidates(),
            "actions": make_actions(),
            "model_spec_sha256": make_model_spec(),
            "source_family_mapping_sha256": sha("source-map"),
        }

    def call(self, arguments):
        return build_normalization_plan(
            arguments["contexts"],
            arguments["candidates"],
            arguments["actions"],
            arguments["model_spec_sha256"],
            arguments["source_family_mapping_sha256"],
        )

    def assertRejected(self, **overrides):
        arguments = self.base_arguments()
        arguments.update(overrides)
        with self.assertRaises(NormalizationPlanError):
            self.call(arguments)

    def assertFixedCode(self, expected_code, **overrides):
        arguments = self.base_arguments()
        arguments.update(overrides)
        with self.assertRaises(NormalizationPlanError) as caught:
            self.call(arguments)
        self.assertIn(str(caught.exception), REASON_CODES)
        self.assertEqual(str(caught.exception), expected_code)
        self.assertNotIn(SENTINEL, str(caught.exception))

    def test_wrong_container_types(self):
        self.assertRejected(contexts="not-a-list")
        self.assertRejected(candidates="not-a-list")
        self.assertRejected(actions=["not-a-mapping"])
        self.assertRejected(model_spec_sha256=None)
        self.assertRejected(source_family_mapping_sha256=123)
        self.assertRejected(contexts=True)
        self.assertRejected(candidates=True)
        self.assertRejected(actions=True)
        self.assertRejected(model_spec_sha256=True)
        self.assertRejected(source_family_mapping_sha256=True)

    def test_actions_schema(self):
        actions = make_actions()
        del actions["R_D1_400_1800"]
        self.assertFixedCode("actions_keys_invalid", actions=actions)

        actions = make_actions()
        actions["R_MIN_400_1800"] = sha("min")
        self.assertFixedCode("actions_keys_invalid", actions=actions)

        for bad in ("A" * 64, "abc", 5, None):
            actions = make_actions()
            actions["R_SNV_400_1800"] = bad
            with self.subTest(bad=bad):
                self.assertFixedCode("action_hash_invalid", actions=actions)

    def test_model_spec_schema(self):
        spec = make_model_spec()
        del spec["C-PLS-DA"]
        self.assertFixedCode("model_spec_keys_invalid", model_spec_sha256=spec)

        for invented in ("D1", "D0-M", "C-PRIOR"):
            spec = make_model_spec()
            spec[invented] = sha("neural")
            with self.subTest(model=invented):
                self.assertFixedCode("model_spec_keys_invalid", model_spec_sha256=spec)

        spec = make_model_spec()
        spec["C-PLS-DA"] = spec["C-PLS-DA"][:-1] + "g"
        self.assertFixedCode("model_spec_hash_invalid", model_spec_sha256=spec)

    def test_source_family_mapping_hash(self):
        self.assertFixedCode(
            "source_family_mapping_hash_invalid",
            source_family_mapping_sha256="xyz",
        )
        self.assertFixedCode(
            "source_family_mapping_hash_invalid",
            source_family_mapping_sha256="A" * 64,
        )

    def test_candidates_schema(self):
        candidates = make_candidates()
        candidates[0]["unexpected"] = "x"
        self.assertFixedCode("candidate_keys_invalid", candidates=candidates)

        candidates = make_candidates()
        del candidates[0]["model_id"]
        self.assertFixedCode("candidate_keys_invalid", candidates=candidates)

        self.assertFixedCode("candidates_empty", candidates=[])

        candidates = make_candidates()
        candidates[0]["candidate_id"] = " "
        self.assertFixedCode("candidate_id_invalid", candidates=candidates)

        candidates = make_candidates()
        candidates[0]["candidate_id"] = SENTINEL
        candidates[1]["candidate_id"] = SENTINEL
        self.assertFixedCode("duplicate_candidate_id", candidates=candidates)

        candidates = make_candidates()
        candidates[0]["hyperparameter_sha256"] = "A" * 64
        self.assertFixedCode("candidate_hyperparameter_hash_invalid", candidates=candidates)

    def test_candidate_unknown_models(self):
        for invented in ("C-PRIOR", "D0-M", "D1", "C-SELECTED", "C-SOURCE-CORAL", SENTINEL):
            candidates = make_candidates()
            candidates[0] = {
                "candidate_id": "invented",
                "model_id": invented,
                "hyperparameter_sha256": sha("x"),
            }
            with self.subTest(model=invented):
                self.assertFixedCode("candidate_model_not_permitted", candidates=candidates)

    def test_candidate_counts(self):
        candidates = make_candidates()
        candidates.pop()
        self.assertFixedCode("candidate_counts_invalid", candidates=candidates)

        candidates = [
            candidate
            for candidate in make_candidates()
            if candidate["model_id"] != "C-SPECTRAL-MATCH"
        ]
        self.assertFixedCode("candidate_counts_invalid", candidates=candidates)

    def test_context_schema(self):
        contexts = [make_master_context("CTX-M")]
        contexts[0]["unexpected"] = "x"
        self.assertFixedCode("context_keys_invalid", contexts=contexts)

        contexts = [make_master_context("CTX-M")]
        del contexts[0]["selection_units"]
        self.assertFixedCode("context_keys_invalid", contexts=contexts)

        self.assertFixedCode("contexts_empty", contexts=[])

        contexts = [make_master_context("CTX-M"), make_master_context("CTX-M")]
        self.assertFixedCode("duplicate_context_id", contexts=contexts)

    def test_context_identifiers_and_hashes(self):
        contexts = [make_master_context(" CTX-M")]
        self.assertFixedCode("context_id_invalid", contexts=contexts)

        contexts = [make_master_context("CTX-M")]
        contexts[0]["selection_state_sha256"] = "A" * 64
        self.assertFixedCode("selection_state_hash_invalid", contexts=contexts)

        contexts = [make_master_context("CTX-M")]
        contexts[0]["outer_fit_uid_sha256"] = "x"
        self.assertFixedCode("outer_fit_uid_hash_invalid", contexts=contexts)

        contexts = [make_master_context("CTX-M")]
        contexts[0]["outer_test_uid_sha256"] = "x"
        self.assertFixedCode("outer_test_uid_hash_invalid", contexts=contexts)

        contexts = [make_master_context("CTX-M")]
        contexts[0]["outer_test_uid_sha256"] = contexts[0]["outer_fit_uid_sha256"]
        self.assertFixedCode("outer_fit_test_equal", contexts=contexts)

    def test_context_selection_rules(self):
        contexts = [make_master_context("CTX-M")]
        contexts[0]["selection_mode"] = "holdout"
        self.assertFixedCode("selection_mode_invalid", contexts=contexts)

        for invented in ("C-PRIOR", "D0-M", "D1", "C-SELECTED", "C-SOURCE-CORAL"):
            contexts = [make_master_context("CTX-M", invented)]
            with self.subTest(model=invented):
                self.assertFixedCode("selected_model_not_permitted", contexts=contexts)

        contexts = [make_master_context("CTX-M")]
        contexts[0]["selected_model_id"] = " "
        self.assertFixedCode("selected_model_id_invalid", contexts=contexts)

        for bad in (1, 0, "true", None):
            contexts = [make_master_context("CTX-M")]
            contexts[0]["historical_reference_available"] = bad
            with self.subTest(bad=bad):
                self.assertFixedCode("historical_reference_invalid", contexts=contexts)

    def test_context_unit_rules(self):
        contexts = [make_master_context("CTX-M")]
        contexts[0]["selection_units"].append(make_unit("CTX-M-MU-3", "CTX-M"))
        self.assertFixedCode("master_selection_units_count_invalid", contexts=contexts)

        contexts = [make_pseudo_context("CTX-P")]
        contexts[0]["selection_units"] = contexts[0]["selection_units"][:1]
        self.assertFixedCode("selection_units_too_few", contexts=contexts)

        contexts = [make_pseudo_context("CTX-P")]
        contexts[0]["calibration_units"] = contexts[0]["calibration_units"][:2]
        self.assertFixedCode("calibration_units_too_few", contexts=contexts)

        contexts = [make_pseudo_context("CTX-P")]
        contexts[0]["calibration_units"].append(make_unit("CTX-P-CU-3", "CTX-P"))
        self.assertFixedCode("calibration_units_count_invalid", contexts=contexts)

        contexts = [make_pseudo_context("CTX-P")]
        contexts[0]["calibration_units"][0]["unit_id"] = " "
        self.assertFixedCode("unit_id_invalid", contexts=contexts)

        contexts = [make_pseudo_context("CTX-P")]
        contexts[0]["selection_units"][0]["unexpected"] = "x"
        self.assertFixedCode("unit_keys_invalid", contexts=contexts)

        contexts = [make_pseudo_context("CTX-P")]
        contexts[0]["selection_units"][0]["fit_uid_sha256"] = "A" * 64
        self.assertFixedCode("fit_uid_hash_invalid", contexts=contexts)

        contexts = [make_pseudo_context("CTX-P")]
        contexts[0]["selection_units"][0]["validation_uid_sha256"] = "A" * 64
        self.assertFixedCode("validation_uid_hash_invalid", contexts=contexts)

        contexts = [make_pseudo_context("CTX-P")]
        contexts[0]["selection_units"][1]["unit_id"] = contexts[0]["selection_units"][0][
            "unit_id"
        ]
        self.assertFixedCode("duplicate_unit_id", contexts=contexts)

    def test_context_role_conflicts(self):
        contexts = [make_pseudo_context("CTX-P")]
        contexts[0]["selection_units"][0]["validation_uid_sha256"] = contexts[0][
            "selection_units"
        ][0]["fit_uid_sha256"]
        self.assertFixedCode("unit_fit_validation_equal", contexts=contexts)

        contexts = [make_pseudo_context("CTX-P")]
        contexts[0]["selection_units"][0]["fit_uid_sha256"] = contexts[0][
            "outer_test_uid_sha256"
        ]
        self.assertFixedCode("unit_conflicts_outer_test", contexts=contexts)

    def test_master_calibration_identity(self):
        contexts = [make_master_context("CTX-M")]
        contexts[0]["calibration_units"][0]["unit_id"] = "OTHER-UNIT"
        self.assertFixedCode("master_calibration_id_mismatch", contexts=contexts)

        contexts = [make_master_context("CTX-M")]
        contexts[0]["calibration_units"][0]["fit_uid_sha256"] = sha("other-fit")
        self.assertFixedCode("master_calibration_hash_mismatch", contexts=contexts)

        contexts = [make_master_context("CTX-M")]
        contexts[0]["calibration_units"][0]["validation_uid_sha256"] = sha("other-val")
        self.assertFixedCode("master_calibration_hash_mismatch", contexts=contexts)

    def test_error_text_never_leaks_inputs(self):
        cases = []

        contexts = [make_master_context(SENTINEL), make_master_context(SENTINEL)]
        cases.append(("duplicate_context_id", {"contexts": contexts}))

        contexts = [make_master_context(f" {SENTINEL}")]
        cases.append(("context_id_invalid", {"contexts": contexts}))

        contexts = [make_pseudo_context(SENTINEL)]
        contexts[0]["selection_mode"] = SENTINEL
        cases.append(("selection_mode_invalid", {"contexts": contexts}))

        contexts = [make_pseudo_context("CTX-P")]
        contexts[0]["selection_units"][0]["unit_id"] = SENTINEL
        contexts[0]["selection_units"][1]["unit_id"] = SENTINEL
        cases.append(("duplicate_unit_id", {"contexts": contexts}))

        candidates = make_candidates()
        candidates[0]["candidate_id"] = SENTINEL
        candidates[1]["candidate_id"] = SENTINEL
        cases.append(("duplicate_candidate_id", {"candidates": candidates}))

        candidates = make_candidates()
        candidates[0]["model_id"] = SENTINEL
        cases.append(("candidate_model_not_permitted", {"candidates": candidates}))

        actions = make_actions()
        actions[SENTINEL] = sha("x")
        cases.append(("actions_keys_invalid", {"actions": actions}))

        spec = make_model_spec()
        spec[SENTINEL] = sha("x")
        cases.append(("model_spec_keys_invalid", {"model_spec_sha256": spec}))

        for expected_code, overrides in cases:
            with self.subTest(code=expected_code):
                self.assertFixedCode(expected_code, **overrides)

    def test_graph_validator_fixed_errors(self):
        from atlas_sers.evaluation.p08_normalization_plan import _validate_graph

        def tiny_job(
            job_id,
            dependencies=(),
            context_id="CTX",
            policy_id="PP",
            representation_id="R",
        ):
            return {
                "job_id": job_id,
                "dependencies": sorted(set(dependencies)),
                "context_id": context_id,
                "policy_id": policy_id,
                "representation_id": representation_id,
            }

        cases = (
            ("duplicate_job_id", [tiny_job("JOB-A"), tiny_job("JOB-A")]),
            ("dependency_missing", [tiny_job("JOB-A", ["JOB-MISSING"])]),
            (
                "dependency_context_mismatch",
                [
                    tiny_job("JOB-A", ["JOB-B"], context_id="CTX-1"),
                    tiny_job("JOB-B", context_id="CTX-2"),
                ],
            ),
            (
                "dependency_cycle",
                [
                    tiny_job("JOB-A", ["JOB-B"]),
                    tiny_job("JOB-B", ["JOB-A"]),
                ],
            ),
        )
        for expected_code, jobs in cases:
            with self.subTest(code=expected_code):
                self.assertIn(expected_code, REASON_CODES)
                with self.assertRaises(NormalizationPlanError) as caught:
                    _validate_graph(jobs)
                self.assertEqual(str(caught.exception), expected_code)


if __name__ == "__main__":
    unittest.main()
