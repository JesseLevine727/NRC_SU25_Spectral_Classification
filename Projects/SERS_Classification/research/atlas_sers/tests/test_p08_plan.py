"""Tests for the P08-T006 universal no-fit slot DAG planner."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

from atlas_sers.evaluation.p08_plan import (
    CLASSICAL_MODELS,
    D0_RECIPE,
    D0_STRATEGY,
    NEURAL_RECIPES,
    POLICIES,
    POLICY_REPRESENTATION,
    REASON_CODES,
    SCHEMA_VERSION,
    SEEDS,
    SELECTED_STRATEGY,
    SVM_MODEL,
    SVM_SEED,
    PlanError,
    build_universal_plan,
    require_scientific_execution,
)

SENTINEL = "SENTINEL-PRIVATE-424242"

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
)

MODEL_SPEC_KEYS = (
    "C-RBF-SVM",
    "C-RANDOM-FOREST",
    "C-EXTRA-TREES",
    "D0-M",
    "D1",
    "D2",
    "D3",
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
    return hashlib.sha256(f"p08-t006::{text}".encode()).hexdigest()


def alpha_id(prefix, index):
    letters = "abcdefghijklmnopqrstuvwxyz"
    text = ""
    value = index
    while True:
        text = letters[value % 26] + text
        value = value // 26
        if value == 0:
            break
        value -= 1
    return f"{prefix}{text}"


def make_unit(unit_id, tag):
    return {
        "unit_id": unit_id,
        "fit_uid_sha256": sha(f"{tag}::fit::{unit_id}"),
        "validation_uid_sha256": sha(f"{tag}::val::{unit_id}"),
    }


def make_master_context(context_id="CTX-MASTER-ALPHA"):
    units = [make_unit(f"MU-{index}", context_id) for index in range(3)]
    return {
        "context_id": context_id,
        "selection_mode": "master_cv",
        "selected_recipe_id": "D0-M",
        "outer_fit_uid_sha256": sha(f"{context_id}::outer-fit"),
        "outer_test_uid_sha256": sha(f"{context_id}::outer-test"),
        "selection_units": units,
        "calibration_units": [dict(unit) for unit in units],
    }


def make_pseudo_context(context_id="CTX-PSEUDO-BETA", recipe="D1"):
    selection = [make_unit(f"PU-{index}", context_id) for index in range(2)]
    calibration = [make_unit(f"CU-{index}", context_id) for index in range(3)]
    return {
        "context_id": context_id,
        "selection_mode": "pseudo_domain",
        "selected_recipe_id": recipe,
        "outer_fit_uid_sha256": sha(f"{context_id}::outer-fit"),
        "outer_test_uid_sha256": sha(f"{context_id}::outer-test"),
        "selection_units": selection,
        "calibration_units": calibration,
    }


def make_candidates():
    candidates = []
    for index in range(36):
        candidates.append(
            {
                "candidate_id": alpha_id("svm-candidate-", index),
                "model_id": "C-RBF-SVM",
                "hyperparameter_sha256": sha(f"svm-hp-{index}"),
            }
        )
    for index in range(16):
        candidates.append(
            {
                "candidate_id": alpha_id("rf-candidate-", index),
                "model_id": "C-RANDOM-FOREST",
                "hyperparameter_sha256": sha(f"rf-hp-{index}"),
            }
        )
    for index in range(16):
        candidates.append(
            {
                "candidate_id": alpha_id("et-candidate-", index),
                "model_id": "C-EXTRA-TREES",
                "hyperparameter_sha256": sha(f"et-hp-{index}"),
            }
        )
    return candidates


def make_actions(tag="default"):
    return {
        "R_MIN_400_1800": sha(f"{tag}::array::MIN"),
        "R_SG_400_1800": sha(f"{tag}::array::SG"),
        "R_ARPLS_400_1800": sha(f"{tag}::array::ARPLS"),
    }


def make_model_spec(tag="default"):
    return {name: sha(f"{tag}::spec::{name}") for name in MODEL_SPEC_KEYS}


def build_plan(contexts=None, candidates=None, actions=None, spec=None):
    if contexts is None:
        contexts = [make_master_context(), make_pseudo_context()]
    if candidates is None:
        candidates = make_candidates()
    if actions is None:
        actions = make_actions()
    if spec is None:
        spec = make_model_spec()
    return build_universal_plan(contexts, candidates, actions, spec)


def policy_context_jobs(plan, policy, context_id):
    return [
        job
        for job in plan["jobs"]
        if job["policy_id"] == policy and job["context_id"] == context_id
    ]


def stage_jobs(plan, policy, context_id, stage):
    return [
        job
        for job in policy_context_jobs(plan, policy, context_id)
        if job["stage"] == stage
    ]


class PlanConstructionTests(unittest.TestCase):
    def test_top_level_shape_and_denied_execution(self):
        plan = build_plan()
        self.assertEqual(SCHEMA_VERSION, "nato-sers-p08-universal-slot-dag-v1")
        self.assertEqual(plan["schema_version"], SCHEMA_VERSION)
        self.assertIs(plan["execution_authorized"], False)
        self.assertEqual(len(plan["plan_sha256"]), 64)
        self.assertIsInstance(plan["jobs"], list)
        self.assertIsInstance(plan["aliases"], list)
        self.assertIsInstance(plan["summary"], dict)

        with self.assertRaises(PlanError) as caught:
            require_scientific_execution(plan)
        self.assertEqual(str(caught.exception), "scientific_execution_not_authorized")
        with self.assertRaises(PlanError):
            require_scientific_execution({"execution_authorized": True})

    def test_policy_and_representation_identifiers(self):
        plan = build_plan()
        for job in plan["jobs"]:
            self.assertIn(job["policy_id"], POLICIES)
            self.assertEqual(job["representation_id"], POLICY_REPRESENTATION[job["policy_id"]])
        for alias in plan["aliases"]:
            self.assertIn(alias["policy_id"], POLICIES)
        self.assertEqual(set(plan["summary"]["by_policy"].keys()), set(POLICIES))

    def test_jobs_and_aliases_sorted_and_well_formed(self):
        plan = build_plan()
        job_ids = [job["job_id"] for job in plan["jobs"]]
        self.assertEqual(job_ids, sorted(job_ids))
        alias_ids = [alias["alias_id"] for alias in plan["aliases"]]
        self.assertEqual(alias_ids, sorted(alias_ids))
        self.assertTrue(all(job_id.startswith("P08JOB-") for job_id in job_ids))
        self.assertTrue(all(alias_id.startswith("P08ALIAS-") for alias_id in alias_ids))
        self.assertTrue(all(len(job_id) == len("P08JOB-") + 64 for job_id in job_ids))
        for job in plan["jobs"]:
            for key in REQUIRED_JOB_KEYS:
                self.assertIn(key, job)
            self.assertIsInstance(job["dependencies"], list)
            self.assertEqual(job["dependencies"], sorted(job["dependencies"]))

    def test_unique_ids(self):
        plan = build_plan()
        job_ids = [job["job_id"] for job in plan["jobs"]]
        alias_ids = [alias["alias_id"] for alias in plan["aliases"]]
        self.assertEqual(len(job_ids), len(set(job_ids)))
        self.assertEqual(len(alias_ids), len(set(alias_ids)))

    def test_job_hash_independently_recomputed(self):
        plan = build_plan()
        for job in plan["jobs"]:
            body = {key: value for key, value in job.items() if key != "job_id"}
            self.assertEqual(job["job_id"], f"P08JOB-{recompute(body)}")

    def test_alias_and_plan_hash_independently_recomputed(self):
        plan = build_plan()
        for alias in plan["aliases"]:
            body = {key: value for key, value in alias.items() if key != "alias_id"}
            self.assertEqual(alias["alias_id"], f"P08ALIAS-{recompute(body)}")
        payload = {
            "schema_version": plan["schema_version"],
            "execution_authorized": plan["execution_authorized"],
            "jobs": plan["jobs"],
            "aliases": plan["aliases"],
            "summary": plan["summary"],
        }
        self.assertEqual(plan["plan_sha256"], recompute(payload))

    def test_strict_json_roundtrip(self):
        plan = build_plan()
        encoded = json.dumps(plan, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["plan_sha256"], plan["plan_sha256"])
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)

    def test_summary_contains_no_context_or_unit_ids(self):
        contexts = [
            make_master_context("CTX-MASTER-SECRET"),
            make_pseudo_context("CTX-PSEUDO-SECRET", "D2"),
        ]
        plan = build_plan(contexts=contexts)
        summary_text = json.dumps(plan["summary"], sort_keys=True)
        for context in contexts:
            self.assertNotIn(context["context_id"], summary_text)
            for unit in context["selection_units"] + context["calibration_units"]:
                self.assertNotIn(unit["unit_id"], summary_text)

    def test_candidate_and_seed_counts(self):
        contexts = [make_master_context()]
        plan = build_plan(contexts=contexts)
        for policy in POLICIES:
            source_fits = stage_jobs(plan, policy, "CTX-MASTER-ALPHA", "source_fit")
            by_model = {}
            for job in source_fits:
                by_model.setdefault(job["model_id"], []).append(job)
            self.assertEqual(len({job["candidate_id"] for job in by_model[SVM_MODEL]}), 36)
            self.assertEqual({job["seed"] for job in by_model[SVM_MODEL]}, {SVM_SEED})
            for tree in ("C-RANDOM-FOREST", "C-EXTRA-TREES"):
                self.assertEqual(len({job["candidate_id"] for job in by_model[tree]}), 16)
                self.assertEqual({job["seed"] for job in by_model[tree]}, set(SEEDS))

    def test_master_calibration_aliases_replace_fits(self):
        contexts = [make_master_context()]
        plan = build_plan(contexts=contexts)
        for policy in POLICIES:
            context_jobs = policy_context_jobs(plan, policy, "CTX-MASTER-ALPHA")
            stages = {job["stage"] for job in context_jobs}
            self.assertNotIn("calibration_model_fit", stages)
            self.assertNotIn("calibration_validation_prediction", stages)
            aliases = [
                job for job in context_jobs if job["stage"] == "calibration_prediction_alias"
            ]
            self.assertEqual(len(aliases), 21)
            for alias in aliases:
                self.assertEqual(
                    alias["resolution"],
                    "resolve_selected_candidate_after_source_selection",
                )

    def test_pseudo_calibration_fits(self):
        contexts = [make_pseudo_context()]
        plan = build_plan(contexts=contexts)
        for policy in POLICIES:
            context_jobs = policy_context_jobs(plan, policy, "CTX-PSEUDO-BETA")
            stages = {job["stage"] for job in context_jobs}
            self.assertIn("calibration_model_fit", stages)
            self.assertIn("calibration_validation_prediction", stages)
            self.assertNotIn("calibration_prediction_alias", stages)
            fits = [
                job for job in context_jobs if job["stage"] == "calibration_model_fit"
            ]
            self.assertEqual(len(fits), 21)

    def test_classical_seed_average_then_temperature_order(self):
        plan = build_plan()
        for policy in POLICIES:
            for context_id in ("CTX-MASTER-ALPHA", "CTX-PSEUDO-BETA"):
                jobs = policy_context_jobs(plan, policy, context_id)
                lookup = {job["job_id"]: job for job in jobs}
                for model_id in CLASSICAL_MODELS:
                    held = [
                        job
                        for job in jobs
                        if job["stage"] == "held_prediction" and job["model_id"] == model_id
                    ]
                    self.assertTrue(held)
                    for job in held:
                        self.assertEqual(job["resolution"], "uncalibrated_scores_only")
                        self.assertEqual(len(job["dependencies"]), 1)
                        dependency = lookup[job["dependencies"][0]]
                        self.assertEqual(dependency["stage"], "final_refit")
                        self.assertEqual(dependency["model_id"], model_id)
                        self.assertEqual(dependency["seed"], job["seed"])
                    scalar = [
                        job
                        for job in jobs
                        if job["stage"] == "scalar_calibration" and job["model_id"] == model_id
                    ]
                    self.assertEqual(len(scalar), 1)
                    self.assertEqual(
                        scalar[0]["resolution"], "calibrate_seed_averaged_source_scores"
                    )
                    ensembles = [
                        job
                        for job in jobs
                        if job["stage"] == "seed_ensemble_prediction"
                        and job["model_id"] == model_id
                    ]
                    self.assertEqual(len(ensembles), 1)
                    ensemble = ensembles[0]
                    self.assertEqual(
                        ensemble["resolution"], "seed_average_then_single_temperature"
                    )
                    expected = sorted(job["job_id"] for job in held)
                    expected.append(scalar[0]["job_id"])
                    self.assertEqual(ensemble["dependencies"], sorted(expected))

    def test_neural_temperature_then_average_order(self):
        plan = build_plan()
        for policy in POLICIES:
            for context_id in ("CTX-MASTER-ALPHA", "CTX-PSEUDO-BETA"):
                jobs = policy_context_jobs(plan, policy, context_id)
                lookup = {job["job_id"]: job for job in jobs}
                neural_models = sorted(
                    {job["model_id"] for job in jobs} & set(NEURAL_RECIPES)
                )
                for model_id in neural_models:
                    held = [
                        job
                        for job in jobs
                        if job["stage"] == "held_prediction" and job["model_id"] == model_id
                    ]
                    self.assertTrue(held)
                    for job in held:
                        self.assertEqual(job["resolution"], "source_epoch_dependent")
                        dependencies = [lookup[dep] for dep in job["dependencies"]]
                        stages = {dependency["stage"] for dependency in dependencies}
                        self.assertEqual(stages, {"final_refit", "scalar_calibration"})
                        for dependency in dependencies:
                            self.assertEqual(dependency["seed"], job["seed"])
                    ensembles = [
                        job
                        for job in jobs
                        if job["stage"] == "seed_ensemble_prediction"
                        and job["model_id"] == model_id
                    ]
                    self.assertEqual(len(ensembles), 1)
                    ensemble = ensembles[0]
                    self.assertEqual(
                        ensemble["dependencies"],
                        sorted(job["job_id"] for job in held),
                    )
                    for dependency_id in ensemble["dependencies"]:
                        self.assertNotEqual(
                            lookup[dependency_id]["stage"], "scalar_calibration"
                        )

    def test_svm_single_seed_follows_classical_order(self):
        contexts = [make_master_context()]
        plan = build_plan(contexts=contexts)
        policy = POLICIES[0]
        jobs = policy_context_jobs(plan, policy, "CTX-MASTER-ALPHA")
        lookup = {job["job_id"]: job for job in jobs}
        held = [
            job
            for job in jobs
            if job["stage"] == "held_prediction" and job["model_id"] == SVM_MODEL
        ]
        self.assertEqual(len(held), 1)
        self.assertEqual(held[0]["resolution"], "uncalibrated_scores_only")
        self.assertEqual(lookup[held[0]["dependencies"][0]]["stage"], "final_refit")
        ensembles = [
            job
            for job in jobs
            if job["stage"] == "seed_ensemble_prediction" and job["model_id"] == SVM_MODEL
        ]
        self.assertEqual(len(ensembles), 1)
        scalar = [
            job
            for job in jobs
            if job["stage"] == "scalar_calibration" and job["model_id"] == SVM_MODEL
        ]
        self.assertEqual(len(scalar), 1)
        self.assertEqual(
            ensembles[0]["dependencies"], sorted([held[0]["job_id"], scalar[0]["job_id"]])
        )

    def test_scalar_calibration_counts(self):
        contexts = [make_pseudo_context()]
        plan = build_plan(contexts=contexts)
        for policy in POLICIES:
            scalars = stage_jobs(plan, policy, "CTX-PSEUDO-BETA", "scalar_calibration")
            counts = {}
            for job in scalars:
                counts[job["model_id"]] = counts.get(job["model_id"], 0) + 1
            for model in CLASSICAL_MODELS:
                self.assertEqual(counts[model], 1)
            self.assertEqual(counts["D0-M"], 3)
            self.assertEqual(counts["D1"], 3)

    def test_aliases_point_at_neural_ensembles(self):
        contexts = [make_master_context(), make_pseudo_context(recipe="D3")]
        plan = build_plan(contexts=contexts)
        lookups = {job["job_id"]: job for job in plan["jobs"]}
        for alias in plan["aliases"]:
            self.assertIn(alias["target_job_id"], lookups)
            target = lookups[alias["target_job_id"]]
            self.assertEqual(target["stage"], "seed_ensemble_prediction")
            self.assertEqual(target["policy_id"], alias["policy_id"])
            self.assertEqual(target["context_id"], alias["context_id"])
            self.assertEqual(target["model_id"], alias["recipe_id"])

    def test_master_d0_has_two_aliases_to_one_job(self):
        contexts = [make_master_context()]
        plan = build_plan(contexts=contexts)
        for policy in POLICIES:
            aliases = [
                alias
                for alias in plan["aliases"]
                if alias["policy_id"] == policy
                and alias["context_id"] == "CTX-MASTER-ALPHA"
            ]
            self.assertEqual(len(aliases), 2)
            strategies = {alias["strategy"] for alias in aliases}
            self.assertEqual(strategies, {D0_STRATEGY, SELECTED_STRATEGY})
            self.assertEqual(len({alias["target_job_id"] for alias in aliases}), 1)
            self.assertEqual({alias["recipe_id"] for alias in aliases}, {D0_RECIPE})

    def test_non_d0_recipe_sets_parametrized(self):
        for recipe in ("D1", "D2", "D3"):
            with self.subTest(recipe=recipe):
                contexts = [make_pseudo_context(recipe=recipe)]
                plan = build_plan(contexts=contexts)
                neural_models = {
                    job["model_id"]
                    for job in policy_context_jobs(plan, POLICIES[0], "CTX-PSEUDO-BETA")
                    if job["model_id"] in NEURAL_RECIPES
                }
                self.assertEqual(neural_models, {D0_RECIPE, recipe})
                aliases = [
                    alias
                    for alias in plan["aliases"]
                    if alias["policy_id"] == POLICIES[0]
                    and alias["context_id"] == "CTX-PSEUDO-BETA"
                ]
                selected = [
                    alias for alias in aliases if alias["strategy"] == SELECTED_STRATEGY
                ]
                self.assertEqual(selected[0]["recipe_id"], recipe)

    def test_dependencies_exist_in_same_context_and_are_acyclic(self):
        plan = build_plan()
        lookup = {job["job_id"]: job for job in plan["jobs"]}
        for job in plan["jobs"]:
            for dependency_id in job["dependencies"]:
                self.assertIn(dependency_id, lookup)
                dependency = lookup[dependency_id]
                self.assertEqual(dependency["context_id"], job["context_id"])
                self.assertEqual(dependency["policy_id"], job["policy_id"])

        visiting = set()
        visited = set()

        def visit(job_id):
            if job_id in visited:
                return
            if job_id in visiting:
                self.fail(f"cycle detected at {job_id}")
            visiting.add(job_id)
            for dependency_id in lookup[job_id]["dependencies"]:
                visit(dependency_id)
            visiting.discard(job_id)
            visited.add(job_id)

        for job_id in lookup:
            visit(job_id)

    def test_provenance_change_propagates(self):
        base = build_plan()
        changed_actions = make_actions(tag="changed")
        changed = build_plan(actions=changed_actions)
        self.assertNotEqual(base["plan_sha256"], changed["plan_sha256"])

        changed_spec = make_model_spec(tag="changed")
        changed_spec_plan = build_plan(spec=changed_spec)
        self.assertNotEqual(base["plan_sha256"], changed_spec_plan["plan_sha256"])

    def test_changed_recipe_changes_plan_hash(self):
        base = build_plan(contexts=[make_master_context(), make_pseudo_context(recipe="D1")])
        changed = build_plan(contexts=[make_master_context(), make_pseudo_context(recipe="D2")])
        self.assertNotEqual(base["plan_sha256"], changed["plan_sha256"])

    def test_candidate_hyperparameter_change_propagates_select_id(self):
        base_candidates = make_candidates()
        base = build_plan(candidates=base_candidates)

        changed_candidates = copy.deepcopy(base_candidates)
        for candidate in changed_candidates:
            if candidate["model_id"] == SVM_MODEL:
                candidate["hyperparameter_sha256"] = sha("mutated-svm-hp")
                break
        changed = build_plan(candidates=changed_candidates)

        def select_ids(plan):
            return sorted(
                job["job_id"]
                for job in plan["jobs"]
                if job["stage"] == "select_hyperparameters" and job["model_id"] == SVM_MODEL
            )

        self.assertNotEqual(select_ids(base), select_ids(changed))

    def test_order_invariance_same_recipe(self):
        base_contexts = [make_master_context(), make_pseudo_context(recipe="D2")]
        base_candidates = make_candidates()
        base = build_plan(contexts=base_contexts, candidates=base_candidates)

        shuffled_contexts = copy.deepcopy(base_contexts)
        shuffled_contexts.reverse()
        for context in shuffled_contexts:
            context["selection_units"].reverse()
            context["calibration_units"].reverse()
        shuffled_candidates = copy.deepcopy(base_candidates)
        shuffled_candidates.reverse()

        shuffled = build_plan(contexts=shuffled_contexts, candidates=shuffled_candidates)
        self.assertEqual(base["plan_sha256"], shuffled["plan_sha256"])

    def test_tuple_inputs_accepted(self):
        contexts = (make_master_context(), make_pseudo_context(recipe="D2"))
        candidates = tuple(make_candidates())
        from_lists = build_plan(
            contexts=list(contexts),
            candidates=list(candidates),
        )
        from_tuples = build_plan(contexts=contexts, candidates=candidates)
        self.assertEqual(from_lists["plan_sha256"], from_tuples["plan_sha256"])

    def test_no_input_mutation(self):
        contexts = [make_master_context(), make_pseudo_context(recipe="D2")]
        candidates = make_candidates()
        actions = make_actions()
        spec = make_model_spec()

        contexts_copy = copy.deepcopy(contexts)
        candidates_copy = copy.deepcopy(candidates)
        actions_copy = copy.deepcopy(actions)
        spec_copy = copy.deepcopy(spec)

        build_universal_plan(contexts, candidates, actions, spec)

        self.assertEqual(contexts, contexts_copy)
        self.assertEqual(candidates, candidates_copy)
        self.assertEqual(actions, actions_copy)
        self.assertEqual(spec, spec_copy)

    def test_distinct_contexts_may_share_unit_ids(self):
        first = make_pseudo_context("CTX-A", "D1")
        second = make_pseudo_context("CTX-B", "D1")
        second["selection_units"] = [dict(unit) for unit in first["selection_units"]]
        second["calibration_units"] = [dict(unit) for unit in first["calibration_units"]]
        second["outer_fit_uid_sha256"] = sha("outer-b-fit")
        second["outer_test_uid_sha256"] = sha("outer-b-test")
        plan = build_plan(contexts=[first, second])
        contexts_in_plan = {job["context_id"] for job in plan["jobs"]}
        self.assertEqual(contexts_in_plan, {"CTX-A", "CTX-B"})

    def test_policy_accounting_relationships(self):
        plan = build_plan()
        summary = plan["summary"]
        for policy in POLICIES:
            entry = summary["by_policy"][policy]
            stage_counts = entry["stage_counts"]
            expected_fits = sum(
                count
                for stage, count in stage_counts.items()
                if stage in ("source_fit", "calibration_model_fit", "final_refit")
            )
            self.assertEqual(entry["model_fit_slots"], expected_fits)
            self.assertEqual(
                entry["scalar_calibrations"], stage_counts.get("scalar_calibration", 0)
            )
            self.assertEqual(
                entry["total_jobs"],
                sum(1 for job in plan["jobs"] if job["policy_id"] == policy),
            )
        self.assertEqual(summary["authorized_fit_slots"], 0)
        self.assertEqual(summary["totals"]["total_jobs"], len(plan["jobs"]))
        self.assertEqual(summary["totals"]["aliases"], len(plan["aliases"]))
        self.assertEqual(
            summary["nonminimal_totals"]["total_jobs"],
            summary["by_policy"]["PP-U-SG"]["total_jobs"]
            + summary["by_policy"]["PP-U-ARPLS"]["total_jobs"],
        )

    def test_fixture_arithmetic_literals(self):
        contexts = [make_master_context(), make_pseudo_context(recipe="D1")]
        plan = build_plan(contexts=contexts)
        for policy in POLICIES:
            entry = plan["summary"]["by_policy"][policy]
            stage_counts = entry["stage_counts"]
            # classical source fits: 5 selection units * (36 + 16*3 + 16*3) = 660
            # neural source fits: (3*1 + 2*2) * 3 = 21
            self.assertEqual(stage_counts["source_fit"], 660 + 21)
            # pseudo calibration fits: 3 calibration units * (1 + 3 + 3) = 21
            self.assertEqual(stage_counts["calibration_model_fit"], 21)
            # final refits: SVM 2 + trees 12 + neural 9 = 23
            self.assertEqual(stage_counts["final_refit"], 23)
            self.assertEqual(entry["model_fit_slots"], 660 + 21 + 21 + 23)
            self.assertEqual(entry["model_fit_slots"], 725)
            # scalar calibrations: classical 6 + neural 9 = 15
            self.assertEqual(entry["scalar_calibrations"], 15)
            # neural strategy aliases: 2 strategies * 2 contexts = 4
            self.assertEqual(entry["aliases"], 4)


class ValidationTests(unittest.TestCase):
    def base_arguments(self):
        return {
            "contexts": [make_master_context()],
            "candidates": make_candidates(),
            "actions": make_actions(),
            "model_spec_sha256": make_model_spec(),
        }

    def assertRejected(self, **overrides):
        arguments = self.base_arguments()
        arguments.update(overrides)
        with self.assertRaises(PlanError):
            build_universal_plan(
                arguments["contexts"],
                arguments["candidates"],
                arguments["actions"],
                arguments["model_spec_sha256"],
            )

    def assertFixedCode(self, expected_code, **overrides):
        arguments = self.base_arguments()
        arguments.update(overrides)
        with self.assertRaises(PlanError) as caught:
            build_universal_plan(
                arguments["contexts"],
                arguments["candidates"],
                arguments["actions"],
                arguments["model_spec_sha256"],
            )
        self.assertIn(str(caught.exception), REASON_CODES)
        self.assertEqual(str(caught.exception), expected_code)
        self.assertNotIn(SENTINEL, str(caught.exception))

    def test_wrong_container_types(self):
        self.assertRejected(contexts="not-a-list")
        self.assertRejected(candidates="not-a-list")
        self.assertRejected(actions=["not-a-mapping"])
        self.assertRejected(model_spec_sha256=None)
        self.assertRejected(contexts=True)
        self.assertRejected(candidates=True)
        self.assertRejected(actions=True)
        self.assertRejected(model_spec_sha256=True)

    def test_missing_and_extra_keys(self):
        actions = make_actions()
        del actions["R_MIN_400_1800"]
        self.assertRejected(actions=actions)

        actions = make_actions()
        actions["R_EXTRA"] = sha("extra")
        self.assertRejected(actions=actions)

        spec = make_model_spec()
        del spec["D3"]
        self.assertRejected(model_spec_sha256=spec)

        spec = make_model_spec()
        spec["D9"] = sha("extra")
        self.assertRejected(model_spec_sha256=spec)

        candidates = make_candidates()
        del candidates[0]["candidate_id"]
        self.assertRejected(candidates=candidates)

        candidates = make_candidates()
        candidates[0]["unexpected"] = "x"
        self.assertRejected(candidates=candidates)

        contexts = [make_master_context()]
        del contexts[0]["selection_units"]
        self.assertRejected(contexts=contexts)

        contexts = [make_master_context()]
        contexts[0]["unexpected"] = "x"
        self.assertRejected(contexts=contexts)

    def test_malformed_hashes_and_identifiers(self):
        actions = make_actions()
        actions["R_MIN_400_1800"] = "A" * 64
        self.assertRejected(actions=actions)

        actions = make_actions()
        actions["R_SG_400_1800"] = "abc"
        self.assertRejected(actions=actions)

        actions = make_actions()
        actions["R_ARPLS_400_1800"] = 123
        self.assertRejected(actions=actions)

        spec = make_model_spec()
        spec["D1"] = spec["D1"][:-1] + "g"
        self.assertRejected(model_spec_sha256=spec)

        candidates = make_candidates()
        candidates[0]["candidate_id"] = "   "
        self.assertRejected(candidates=candidates)

        candidates = make_candidates()
        candidates[0]["model_id"] = "X-UNKNOWN"
        self.assertRejected(candidates=candidates)

        contexts = [make_master_context()]
        contexts[0]["context_id"] = ""
        self.assertRejected(contexts=contexts)

        contexts = [make_master_context()]
        contexts[0]["selection_units"][0]["unit_id"] = 17
        self.assertRejected(contexts=contexts)

    def test_identifiers_reject_padding(self):
        contexts = [make_master_context()]
        contexts[0]["context_id"] = " CTX"
        self.assertFixedCode("context_id_invalid", contexts=contexts)

        contexts = [make_master_context()]
        contexts[0]["selection_units"][0]["unit_id"] = "MU-a "
        self.assertFixedCode("unit_id_invalid", contexts=contexts)

        candidates = make_candidates()
        candidates[0]["candidate_id"] = " svm "
        self.assertFixedCode("candidate_id_invalid", candidates=candidates)

        contexts = [make_master_context()]
        contexts[0]["selection_mode"] = "master_cv "
        self.assertFixedCode("selection_mode_invalid", contexts=contexts)

        contexts = [make_master_context()]
        contexts[0]["selected_recipe_id"] = " D0-M"
        self.assertFixedCode("selected_recipe_id_invalid", contexts=contexts)

    def test_unknown_selection_mode_rejected(self):
        for invented in ("pseudo_cv", "pseudo_dov", "holdout", SENTINEL):
            with self.subTest(mode=invented):
                contexts = [make_pseudo_context()]
                contexts[0]["selection_mode"] = invented
                self.assertFixedCode("selection_mode_invalid", contexts=contexts)

    def test_duplicates_do_not_leak_sentinel(self):
        contexts = [make_pseudo_context(SENTINEL), make_pseudo_context(SENTINEL)]
        self.assertFixedCode("duplicate_context_id", contexts=contexts)

        contexts = [make_pseudo_context()]
        contexts[0]["selection_units"][0]["unit_id"] = SENTINEL
        contexts[0]["selection_units"][1]["unit_id"] = SENTINEL
        self.assertFixedCode("duplicate_unit_id", contexts=contexts)

        candidates = make_candidates()
        candidates[0]["candidate_id"] = SENTINEL
        candidates[1]["candidate_id"] = SENTINEL
        self.assertFixedCode("duplicate_candidate_id", candidates=candidates)

    def test_candidate_counts(self):
        candidates = make_candidates()
        candidates = [c for c in candidates if not c["candidate_id"].startswith("svm-")]
        self.assertRejected(candidates=candidates)

        candidates = make_candidates()
        candidates.pop()
        self.assertRejected(candidates=candidates)

        self.assertRejected(candidates=[])

    def test_context_mode_and_recipe_rules(self):
        contexts = [make_master_context()]
        contexts[0]["selected_recipe_id"] = "D1"
        self.assertRejected(contexts=contexts)

        contexts = [make_pseudo_context()]
        contexts[0]["selected_recipe_id"] = "D9"
        self.assertRejected(contexts=contexts)

        contexts = [make_master_context()]
        contexts[0]["selection_units"] = contexts[0]["selection_units"][:2]
        contexts[0]["calibration_units"] = contexts[0]["calibration_units"][:2]
        self.assertRejected(contexts=contexts)

        contexts = [make_pseudo_context()]
        contexts[0]["selection_units"] = contexts[0]["selection_units"][:1]
        self.assertRejected(contexts=contexts)

        contexts = [make_pseudo_context()]
        contexts[0]["calibration_units"] = contexts[0]["calibration_units"][:2]
        self.assertRejected(contexts=contexts)

    def test_hash_role_conflicts(self):
        contexts = [make_master_context()]
        contexts[0]["outer_test_uid_sha256"] = contexts[0]["outer_fit_uid_sha256"]
        self.assertRejected(contexts=contexts)

        contexts = [make_pseudo_context()]
        contexts[0]["selection_units"][0]["validation_uid_sha256"] = contexts[0][
            "selection_units"
        ][0]["fit_uid_sha256"]
        self.assertRejected(contexts=contexts)

        contexts = [make_pseudo_context()]
        contexts[0]["selection_units"][0]["fit_uid_sha256"] = contexts[0][
            "outer_test_uid_sha256"
        ]
        self.assertRejected(contexts=contexts)

    def test_master_calibration_hash_mismatch(self):
        contexts = [make_master_context()]
        contexts[0]["calibration_units"][0]["validation_uid_sha256"] = sha(
            "mismatched-calibration-validation"
        )
        self.assertFixedCode("master_calibration_hash_mismatch", contexts=contexts)

    def test_master_calibration_unit_id_mismatch(self):
        contexts = [make_master_context()]
        contexts[0]["calibration_units"][0]["unit_id"] = "OTHER-UNIT"
        self.assertFixedCode("master_calibration_id_mismatch", contexts=contexts)

    def test_master_calibration_requires_exact_identity(self):
        contexts = [make_master_context()]
        contexts[0]["calibration_units"][0]["fit_uid_sha256"] = sha("other-fit")
        self.assertFixedCode("master_calibration_hash_mismatch", contexts=contexts)

    def test_all_malformed_errors_are_fixed_codes_without_sentinel(self):
        cases = []

        contexts = [make_pseudo_context(SENTINEL), make_pseudo_context(SENTINEL)]
        cases.append(("duplicate_context_id", {"contexts": contexts}))

        contexts = [make_pseudo_context()]
        contexts[0]["selection_units"][0]["unit_id"] = SENTINEL
        contexts[0]["selection_units"][1]["unit_id"] = SENTINEL
        cases.append(("duplicate_unit_id", {"contexts": contexts}))

        candidates = make_candidates()
        candidates[0]["candidate_id"] = SENTINEL
        candidates[1]["candidate_id"] = SENTINEL
        cases.append(("duplicate_candidate_id", {"candidates": candidates}))

        contexts = [make_master_context()]
        contexts[0]["context_id"] = f" {SENTINEL}"
        cases.append(("context_id_invalid", {"contexts": contexts}))

        contexts = [make_pseudo_context()]
        contexts[0]["selection_mode"] = SENTINEL
        cases.append(("selection_mode_invalid", {"contexts": contexts}))

        candidates = make_candidates()
        candidates[0]["model_id"] = SENTINEL
        cases.append(("candidate_model_not_permitted", {"candidates": candidates}))

        candidates = make_candidates()
        candidates[0][SENTINEL] = SENTINEL
        cases.append(("candidate_keys_invalid", {"candidates": candidates}))

        actions = make_actions()
        actions[SENTINEL] = sha("x")
        cases.append(("actions_keys_invalid", {"actions": actions}))

        spec = make_model_spec()
        spec[SENTINEL] = sha("x")
        cases.append(("model_spec_keys_invalid", {"model_spec_sha256": spec}))

        for expected_code, overrides in cases:
            with self.subTest(code=expected_code):
                self.assertFixedCode(expected_code, **overrides)


if __name__ == "__main__":
    unittest.main()
