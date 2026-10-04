"""Synthetic specification tests for the P08-T180 population operation catalog.

These tests describe the requested specification rather than the provisional
draft behaviour.  They use invented identifiers, hashes, contexts and roles
only, and read the real public locked core contract from the repository.
Nothing here executes science, fits models, writes files or uses private data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from atlas_sers.evaluation.p08_plan import (
    CLASSICAL_MODELS,
    D0_RECIPE,
    D0_STRATEGY,
    EVIDENCE_FUTURE,
    MASTER_MODE,
    NEURAL_RECIPES,
    POLICIES,
    POLICY_REPRESENTATION,
    PSEUDO_MODE,
    SEEDS,
    SELECTED_STRATEGY,
    SVM_MODEL,
)
from atlas_sers.evaluation.p08_population_slots import (
    build_population_slots,
    require_scientific_execution,
)
from atlas_sers.governance.canonical import sha256_value

NON_D0_RECIPES = ("D1", "D2", "D3")
FIT_STAGES = ("source_fit", "guard_source_fit", "calibration_model_fit", "final_refit")
SOURCE_FIT = "source_fit"
SOURCE_PRED = "source_validation_prediction"
GUARD_FIT = "guard_source_fit"
GUARD_PRED = "guard_validation_prediction"
PRED_STAGES = frozenset((SOURCE_PRED, GUARD_PRED))
SELECT_STAGE = "select_neural_recipe"
SELECT_RESOLUTION = "source_only_recipe_selection"
SELECT_REFIT = "select_refit_epochs"
SCALAR_STAGE = "scalar_calibration"
HELD_STAGE = "held_prediction"
ENSEMBLE_STAGE = "seed_ensemble_prediction"
REFIT_STAGE = "final_refit"
CLASSICAL_CAL = "calibration_model_fit"
CLASSICAL_ALIAS = "calibration_prediction_alias"
CAL_PRED = "calibration_validation_prediction"

SENTINEL = "SENTINEL-PRIVATE-424242"

CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "plan" / "contracts" / "p05_core_contract.json"
)
with CONTRACT_PATH.open("r", encoding="utf-8") as _handle:
    CORE_CONTRACT = json.load(_handle)
CORE_CONTRACT_SHA = sha256_value(CORE_CONTRACT)


# ---------------------------------------------------------------------------
# tiny fixtures
# ---------------------------------------------------------------------------


def sha(text):
    return hashlib.sha256(f"p08-t180::{text}".encode()).hexdigest()


def recompute(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


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


def make_guard(unit_id, tag, support=True, exclusion_reason=None):
    guard = make_unit(unit_id, tag)
    guard["support"] = support
    guard["exclusion_reason"] = exclusion_reason
    return guard


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
        representation: sha(f"{tag}::array::{representation}")
        for representation in POLICY_REPRESENTATION.values()
    }


def make_model_spec(tag="default"):
    names = tuple(CLASSICAL_MODELS) + tuple(NEURAL_RECIPES)
    return {name: sha(f"{tag}::spec::{name}") for name in names}


def make_master_context(context_id="CTX-MASTER"):
    units = [make_unit(f"MU-{index}", context_id) for index in range(3)]
    return {
        "context_id": context_id,
        "selection_mode": MASTER_MODE,
        "domain_eligible": True,
        "classical_supported": True,
        "neural_supported": True,
        "g3_comparable": False,
        "unavailable_reasons": [],
        "outer_fit_uid_sha256": sha(f"{context_id}::outer-fit"),
        "outer_test_uid_sha256": sha(f"{context_id}::outer-test"),
        "selection_units": [dict(unit) for unit in units],
        "calibration_units": [dict(unit) for unit in units],
        "guard_units": [],
    }


def make_pseudo_context(context_id="CTX-PSEUDO"):
    selection = [make_unit(f"PU-{index}", context_id) for index in range(2)]
    calibration = [make_unit(f"CU-{index}", context_id) for index in range(3)]
    guards = [make_guard(f"GU-{index}", context_id, support=True) for index in range(3)]
    return {
        "context_id": context_id,
        "selection_mode": PSEUDO_MODE,
        "domain_eligible": True,
        "classical_supported": True,
        "neural_supported": True,
        "g3_comparable": True,
        "unavailable_reasons": [],
        "outer_fit_uid_sha256": sha(f"{context_id}::outer-fit"),
        "outer_test_uid_sha256": sha(f"{context_id}::outer-test"),
        "selection_units": selection,
        "calibration_units": calibration,
        "guard_units": guards,
    }


def make_neural_only_context(context_id="CTX-NEURAL-ONLY"):
    selection = [make_unit(f"NPU-{index}", context_id) for index in range(2)]
    guards = [
        make_guard("NGU-0", context_id, support=True),
        make_guard("NGU-1", context_id, support=True),
        make_guard(
            "NGU-2",
            context_id,
            support=False,
            exclusion_reason="no_comparable_label",
        ),
    ]
    return {
        "context_id": context_id,
        "selection_mode": PSEUDO_MODE,
        "domain_eligible": True,
        "classical_supported": False,
        "neural_supported": True,
        "g3_comparable": False,
        "unavailable_reasons": [],
        "outer_fit_uid_sha256": sha(f"{context_id}::outer-fit"),
        "outer_test_uid_sha256": sha(f"{context_id}::outer-test"),
        "selection_units": selection,
        "calibration_units": [],
        "guard_units": guards,
    }


def make_master_neural_only_context(context_id="CTX-MASTER-NEURAL"):
    context = make_master_context(context_id)
    context["calibration_units"] = []
    context["classical_supported"] = False
    context["neural_supported"] = True
    context["g3_comparable"] = False
    context["guard_units"] = []
    return context


def make_unsupported_context(context_id="CTX-UNSUPPORTED"):
    return {
        "context_id": context_id,
        "selection_mode": "unsupported",
        "domain_eligible": True,
        "classical_supported": False,
        "neural_supported": False,
        "g3_comparable": False,
        "unavailable_reasons": ["unsupported_domain", "no_registered_operation"],
        "outer_fit_uid_sha256": sha(f"{context_id}::outer-fit"),
        "outer_test_uid_sha256": sha(f"{context_id}::outer-test"),
        "selection_units": [],
        "calibration_units": [],
        "guard_units": [],
    }


def make_not_applicable_context(context_id="CTX-NOT-APPLICABLE"):
    return {
        "context_id": context_id,
        "selection_mode": "not_applicable",
        "domain_eligible": False,
        "classical_supported": False,
        "neural_supported": False,
        "g3_comparable": False,
        "unavailable_reasons": ["ineligible_domain"],
        "outer_fit_uid_sha256": sha(f"{context_id}::outer-fit"),
        "outer_test_uid_sha256": sha(f"{context_id}::outer-test"),
        "selection_units": [],
        "calibration_units": [],
        "guard_units": [],
    }


# ---------------------------------------------------------------------------
# plan construction helpers
# ---------------------------------------------------------------------------


def base_arguments():
    return {
        "population_id": "POP-ALPHA",
        "population_sha256": sha("population"),
        "population_plan_sha256": sha("population-plan"),
        "neural_support_sha256": sha("neural-support"),
        "contexts": [make_pseudo_context()],
        "candidates": make_candidates(),
        "actions": make_actions(),
        "model_spec_sha256": make_model_spec(),
        "core_contract": copy.deepcopy(CORE_CONTRACT),
        "include_extra_trees": True,
    }


def build_plan(**overrides):
    arguments = base_arguments()
    arguments.update(overrides)
    return build_population_slots(**arguments)


_CONTEXT_FACTORIES = {
    "master": lambda: [make_master_context()],
    "pseudo": lambda: [make_pseudo_context()],
    "neural_only": lambda: [make_neural_only_context()],
    "unavailable": lambda: [
        make_unsupported_context(),
        make_not_applicable_context(),
    ],
    "mixed": lambda: [
        make_master_context("CTX-M"),
        make_pseudo_context("CTX-P"),
        make_neural_only_context("CTX-N"),
        make_unsupported_context("CTX-U"),
        make_not_applicable_context("CTX-NA"),
    ],
}

_PLAN_CACHE = {}


def cached_plan(kind, include_extra_trees=True):
    key = (kind, include_extra_trees)
    if key not in _PLAN_CACHE:
        contexts = _CONTEXT_FACTORIES[kind]()
        _PLAN_CACHE[key] = build_plan(contexts=contexts, include_extra_trees=include_extra_trees)
    return _PLAN_CACHE[key]


MIN_POLICY = POLICIES[0]


def lookup_jobs(plan):
    return {job["job_id"]: job for job in plan["jobs"]}


def filter_jobs(plan, *, policy=None, context=None, stage=None, model=None):
    result = []
    for job in plan["jobs"]:
        if policy is not None and job["policy_id"] != policy:
            continue
        if context is not None and job["context_id"] != context:
            continue
        if stage is not None and job["stage"] != stage:
            continue
        if model is not None and job["model_id"] != model:
            continue
        result.append(job)
    return result


def aliases_of(plan):
    return plan.get("endpoint_aliases", plan.get("aliases", []))


def excluded_of(plan):
    return plan.get("excluded_source_fit_slots", plan.get("excluded_slots", []))


def select_job(plan, context_id):
    jobs = filter_jobs(plan, context=context_id, stage=SELECT_STAGE)
    assert len(jobs) == 1, f"expected exactly one select job for {context_id}"
    return jobs[0]


def activation_of(job):
    return job.get("activation")


def is_conditional(job):
    return activation_of(job) is not None


def fit_slot_count(stage_counts):
    return sum(count for stage, count in stage_counts.items() if stage in FIT_STAGES)


def alias_target_ids(alias):
    targets = []
    if alias.get("target_job_id") is not None:
        targets.append(alias["target_job_id"])
    for item in alias.get("conditional_targets") or ():
        if isinstance(item, dict):
            targets.append(item["target_job_id"])
        else:
            targets.append(item)
    return targets


def iter_values(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from iter_values(child)


def iter_strings(value):
    for child in iter_values(value):
        if isinstance(child, str):
            yield child


def iter_dicts(value):
    for child in iter_values(value):
        if isinstance(child, dict):
            yield child


# ===========================================================================
# catalogue contract and denials
# ===========================================================================


class CatalogueContractTests(unittest.TestCase):
    def test_schema_version_and_top_level_flags(self):
        plan = cached_plan("pseudo")
        self.assertEqual(plan["schema_version"], "nato-sers-p08-population-slot-catalog-v1")
        self.assertIs(plan["execution_authorized"], False)
        self.assertEqual(plan["scientific_operations"], 0)
        self.assertIs(plan["numerical_readiness_verified"], False)
        self.assertIs(plan["panel_choice_is_planning_only"], True)
        self.assertEqual(plan["summary"]["authorized_scientific_operations"], 0)

    def test_require_scientific_execution_always_valueerror(self):
        plan = cached_plan("pseudo")
        for payload in (
            plan,
            {"execution_authorized": True},
            {"execution_authorized": True, "scientific_operations": 10**9},
            None,
        ):
            with self.subTest(payload=type(payload).__name__):
                with self.assertRaises(ValueError):
                    require_scientific_execution(payload)
        with self.assertRaises(ValueError):
            require_scientific_execution()

    def test_namespaces_and_identifier_lengths(self):
        plan = cached_plan("pseudo")
        for job in plan["jobs"]:
            self.assertTrue(job["job_id"].startswith("P08POPJOB-"))
            self.assertEqual(len(job["job_id"]), len("P08POPJOB-") + 64)
        for alias in aliases_of(plan):
            self.assertTrue(alias["alias_id"].startswith("P08POPALIAS-"))
            self.assertEqual(len(alias["alias_id"]), len("P08POPALIAS-") + 64)
        for slot in excluded_of(plan):
            self.assertTrue(slot["slot_id"].startswith("P08POPEXCL-"))
            self.assertEqual(len(slot["slot_id"]), len("P08POPEXCL-") + 64)

    def test_unique_identifiers(self):
        plan = cached_plan("pseudo")
        job_ids = [job["job_id"] for job in plan["jobs"]]
        self.assertEqual(len(job_ids), len(set(job_ids)))
        alias_ids = [alias["alias_id"] for alias in aliases_of(plan)]
        self.assertEqual(len(alias_ids), len(set(alias_ids)))

    def test_plan_hash_recomputed_from_content(self):
        plan = cached_plan("pseudo")
        payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
        self.assertEqual(plan["plan_sha256"], sha256_value(payload))
        self.assertEqual(plan["plan_sha256"], recompute(payload))

    def test_job_alias_slot_hashes_recomputed_from_content(self):
        plan = cached_plan("neural_only")
        for job in plan["jobs"]:
            body = {key: value for key, value in job.items() if key != "job_id"}
            self.assertEqual(job["job_id"], "P08POPJOB-" + sha256_value(body))
        for alias in aliases_of(plan):
            body = {key: value for key, value in alias.items() if key != "alias_id"}
            self.assertEqual(alias["alias_id"], "P08POPALIAS-" + sha256_value(body))
        for slot in excluded_of(plan):
            body = {key: value for key, value in slot.items() if key != "slot_id"}
            self.assertEqual(slot["slot_id"], "P08POPEXCL-" + sha256_value(body))

    def test_binding_fields_echoed(self):
        arguments = base_arguments()
        plan = build_population_slots(**arguments)
        for field in (
            "population_id",
            "population_sha256",
            "population_plan_sha256",
            "neural_support_sha256",
            "include_extra_trees",
        ):
            self.assertEqual(plan[field], arguments[field])

    def test_all_jobs_are_future_evidence(self):
        for kind in ("pseudo", "master", "neural_only"):
            plan = cached_plan(kind)
            for job in plan["jobs"]:
                with self.subTest(kind=kind, job=job["job_id"]):
                    self.assertEqual(job["evidence_status"], EVIDENCE_FUTURE)

    def test_panel_methods_five_and_four(self):
        five = cached_plan("pseudo", True)
        four = cached_plan("pseudo", False)
        self.assertEqual(len(five["panel_methods"]), 5)
        self.assertEqual(len(four["panel_methods"]), 4)
        for methods, expect_extra in ((five, True), (four, False)):
            self.assertIn("C-RBF-SVM", methods["panel_methods"])
            self.assertIn("C-RANDOM-FOREST", methods["panel_methods"])
            self.assertIn(D0_RECIPE, methods["panel_methods"])
            self.assertIn(SELECTED_STRATEGY, methods["panel_methods"])
            self.assertNotIn("source_selected", methods["panel_methods"])
            self.assertEqual("C-EXTRA-TREES" in methods["panel_methods"], expect_extra)


# ===========================================================================
# immutability, permutation invariance and binding identity
# ===========================================================================


class ImmutabilityAndBindingTests(unittest.TestCase):
    def _mixed_contexts(self):
        return [
            make_master_context("CTX-M"),
            make_pseudo_context("CTX-P"),
            make_neural_only_context("CTX-N"),
            make_unsupported_context("CTX-U"),
        ]

    def test_inputs_not_mutated(self):
        contexts = self._mixed_contexts()
        candidates = make_candidates()
        actions = make_actions()
        spec = make_model_spec()
        contract = copy.deepcopy(CORE_CONTRACT)
        before = (
            copy.deepcopy(contexts),
            copy.deepcopy(candidates),
            copy.deepcopy(actions),
            copy.deepcopy(spec),
            copy.deepcopy(contract),
        )
        build_plan(
            contexts=contexts,
            candidates=candidates,
            actions=actions,
            model_spec_sha256=spec,
            core_contract=contract,
        )
        self.assertEqual(contexts, before[0])
        self.assertEqual(candidates, before[1])
        self.assertEqual(actions, before[2])
        self.assertEqual(spec, before[3])
        self.assertEqual(contract, before[4])

    def test_context_and_unit_permutation_invariance(self):
        contexts = self._mixed_contexts()
        base = build_plan(contexts=copy.deepcopy(contexts))

        shuffled = copy.deepcopy(contexts)
        shuffled.reverse()
        for context in shuffled:
            context["selection_units"].reverse()
            context["calibration_units"].reverse()
            context["guard_units"].reverse()
        candidates = make_candidates()
        candidates.reverse()
        changed = build_plan(contexts=shuffled, candidates=candidates)
        self.assertEqual(base["plan_sha256"], changed["plan_sha256"])

    def test_reason_permutation_invariance(self):
        base_contexts = [make_unsupported_context("CTX-U")]
        base = build_plan(contexts=copy.deepcopy(base_contexts))
        flipped = copy.deepcopy(base_contexts)
        flipped[0]["unavailable_reasons"].reverse()
        changed = build_plan(contexts=flipped)
        self.assertEqual(base["plan_sha256"], changed["plan_sha256"])

    def test_each_population_binding_changes_graph_identity(self):
        base = build_plan()
        replacements = (
            ("population_id", "POP-BETA"),
            ("population_sha256", sha("population-2")),
            ("population_plan_sha256", sha("population-plan-2")),
            ("neural_support_sha256", sha("neural-support-2")),
        )
        for field, value in replacements:
            with self.subTest(field=field):
                changed = build_plan(**{field: value})
                self.assertNotEqual(base["plan_sha256"], changed["plan_sha256"])


# ===========================================================================
# core contract binding
# ===========================================================================


class CoreContractBindingTests(unittest.TestCase):
    def test_original_core_contract_hash_is_bound(self):
        plan = cached_plan("pseudo")
        selection = select_job(plan, "CTX-PSEUDO")
        metadata = selection.get("metadata", {})
        self.assertIn(CORE_CONTRACT_SHA, metadata.values())

    def test_contract_and_input_order_permutation_hash_invariance(self):
        base = build_plan()
        reordered = copy.deepcopy(CORE_CONTRACT)
        if isinstance(reordered, dict):
            reordered = {key: reordered[key] for key in reversed(list(reordered))}
        actions = make_actions()
        actions = {key: actions[key] for key in reversed(list(actions))}
        spec = make_model_spec()
        spec = {key: spec[key] for key in reversed(list(spec))}
        changed = build_plan(
            core_contract=reordered,
            actions=actions,
            model_spec_sha256=spec,
        )
        self.assertEqual(base["plan_sha256"], changed["plan_sha256"])

    def test_audit_hash_is_not_substituted_for_original(self):
        from atlas_sers.evaluation.p05_core_plan import validate_core_contract

        plan = cached_plan("pseudo")
        audit = validate_core_contract(copy.deepcopy(CORE_CONTRACT))
        if isinstance(audit, dict):
            audit_sha = sha256_value(audit)
            if audit_sha != CORE_CONTRACT_SHA:
                self.assertNotIn(audit_sha, set(iter_strings(plan)))

    def test_g3_thresholds_are_copied_exactly(self):
        self.assertIsInstance(CORE_CONTRACT.get("g3"), dict)
        g3 = CORE_CONTRACT["g3"]
        self.assertTrue(g3)
        plan = cached_plan("pseudo")
        selection = select_job(plan, "CTX-PSEUDO")
        metadata = selection.get("metadata", {})
        self.assertEqual(metadata.get("g3"), g3)

    def test_non_mapping_contract_rejected(self):
        arguments = base_arguments()
        arguments["core_contract"] = None
        with self.assertRaises(ValueError):
            build_population_slots(**arguments)

    def test_mutated_contract_rejected(self):
        mutations = []
        extra = copy.deepcopy(CORE_CONTRACT)
        extra[SENTINEL] = SENTINEL
        mutations.append(extra)
        if isinstance(CORE_CONTRACT, dict) and "g3" in CORE_CONTRACT:
            bad_g3 = copy.deepcopy(CORE_CONTRACT)
            bad_g3["g3"] = SENTINEL
            mutations.append(bad_g3)
        for index, mutated in enumerate(mutations):
            with self.subTest(mutation=index):
                arguments = base_arguments()
                arguments["core_contract"] = mutated
                with self.assertRaises(ValueError) as caught:
                    build_population_slots(**arguments)
                self.assertNotIn(SENTINEL, str(caught.exception))


# ===========================================================================
# neural development and guard handling
# ===========================================================================


class NeuralDevelopmentTests(unittest.TestCase):
    def _dev_jobs(self, plan, context_id, *stages):
        return [
            job
            for job in plan["jobs"]
            if job["context_id"] == context_id
            and job["policy_id"] == MIN_POLICY
            and job["stage"] in stages
            and not is_conditional(job)
        ]

    def test_min_development_covers_all_four_recipes_three_seeds(self):
        for kind, context_id, unit_ids in (
            ("pseudo", "CTX-PSEUDO", {f"PU-{i}" for i in range(2)}),
            ("master", "CTX-MASTER", {f"MU-{i}" for i in range(3)}),
        ):
            plan = cached_plan(kind)
            fits = self._dev_jobs(plan, context_id, SOURCE_FIT)
            neural = [job for job in fits if job["model_id"] in NEURAL_RECIPES]
            self.assertEqual(len(neural), len(unit_ids) * len(NEURAL_RECIPES) * len(SEEDS))
            for recipe in NEURAL_RECIPES:
                recipe_jobs = [job for job in neural if job["model_id"] == recipe]
                self.assertEqual(len(recipe_jobs), len(unit_ids) * len(SEEDS))
                self.assertEqual({job["seed"] for job in recipe_jobs}, set(SEEDS))
                self.assertEqual({job["unit_id"] for job in recipe_jobs}, unit_ids)

    def test_supported_guards_get_all_four_recipes(self):
        plan = cached_plan("pseudo")
        guards = filter_jobs(plan, context="CTX-PSEUDO", stage=GUARD_FIT)
        guards = [job for job in guards if job["policy_id"] == MIN_POLICY]
        self.assertEqual(len(guards), 3 * len(NEURAL_RECIPES) * len(SEEDS))
        for recipe in NEURAL_RECIPES:
            recipe_jobs = [job for job in guards if job["model_id"] == recipe]
            self.assertEqual(len(recipe_jobs), 3 * len(SEEDS))
        predictions = filter_jobs(plan, context="CTX-PSEUDO", stage=GUARD_PRED)
        self.assertEqual(len(predictions), len(guards))

    def test_unsupported_guard_excludes_twelve_slots_with_reasons(self):
        plan = cached_plan("neural_only")
        slots = excluded_of(plan)
        self.assertEqual(len(slots), len(NEURAL_RECIPES) * len(SEEDS))
        self.assertEqual({slot["unit_id"] for slot in slots}, {"NGU-2"})
        for recipe in NEURAL_RECIPES:
            recipe_slots = [slot for slot in slots if slot["model_id"] == recipe]
            self.assertEqual(len(recipe_slots), len(SEEDS))
            self.assertEqual({slot["seed"] for slot in recipe_slots}, set(SEEDS))
        for slot in slots:
            self.assertEqual(slot["exclusion_reason"], "no_comparable_label")
            self.assertIs(slot["numerical_job"], False)
            self.assertIs(slot["counted_as_attempted_fit"], False)
            self.assertEqual(slot["stage"], GUARD_FIT)

    def test_no_phantom_operations_or_dependencies_from_excluded_slots(self):
        plan = cached_plan("neural_only")
        slot_ids = {slot["slot_id"] for slot in excluded_of(plan)}
        job_ids = {job["job_id"] for job in plan["jobs"]}
        self.assertTrue(slot_ids.isdisjoint(job_ids))
        dependencies = set()
        for job in plan["jobs"]:
            dependencies.update(job["dependencies"])
        self.assertTrue(slot_ids.isdisjoint(dependencies))
        self.assertEqual(plan["summary"]["excluded_source_fit_slots"], len(slot_ids))

    def test_structural_fallback_keeps_all_min_development_but_no_non_d0_pipeline(self):
        for kind, context_id in (
            ("master", "CTX-MASTER"),
            ("neural_only", "CTX-NEURAL-ONLY"),
        ):
            plan = cached_plan(kind)
            dev_recipes = {
                job["model_id"]
                for job in self._dev_jobs(plan, context_id, SOURCE_FIT)
                if job["model_id"] in NEURAL_RECIPES
            }
            self.assertEqual(dev_recipes, set(NEURAL_RECIPES))
            for job in plan["jobs"]:
                if job["context_id"] != context_id:
                    continue
                if job["model_id"] in NON_D0_RECIPES and job["stage"] in (
                    REFIT_STAGE,
                    SELECT_REFIT,
                    SCALAR_STAGE,
                    HELD_STAGE,
                    ENSEMBLE_STAGE,
                ):
                    self.fail(f"unexpected non-D0 pipeline job {job['stage']}")

    def test_source_recipe_selection_depends_on_predictions_only(self):
        plan = cached_plan("pseudo")
        lookup = lookup_jobs(plan)
        selection = select_job(plan, "CTX-PSEUDO")
        dependency_jobs = [lookup[dep] for dep in selection["dependencies"]]
        self.assertTrue(dependency_jobs)
        for dependency in dependency_jobs:
            self.assertIn(dependency["stage"], PRED_STAGES)
        self.assertEqual(
            len(selection["dependencies"]),
            2 * len(NEURAL_RECIPES) * len(SEEDS) + 3 * len(NEURAL_RECIPES) * len(SEEDS),
        )
        self.assertEqual(selection["resolution"], SELECT_RESOLUTION)
        metadata = selection.get("metadata", {})
        self.assertIs(metadata.get("source_only"), True)
        self.assertIs(metadata.get("requires_all_numerical_status_resolved"), True)
        self.assertNotIn("chosen_recipe", metadata)

    def test_d0_min_source_fits_reused_exactly_once(self):
        plan = cached_plan("pseudo")
        lookup = lookup_jobs(plan)
        fits = filter_jobs(
            plan,
            policy=MIN_POLICY,
            context="CTX-PSEUDO",
            stage=SOURCE_FIT,
            model=D0_RECIPE,
        )
        self.assertEqual(len(fits), 2 * len(SEEDS))
        self.assertEqual(len({job["job_id"] for job in fits}), len(fits))
        predictions = filter_jobs(
            plan,
            policy=MIN_POLICY,
            context="CTX-PSEUDO",
            stage=SOURCE_PRED,
            model=D0_RECIPE,
        )
        self.assertEqual(len(predictions), 2 * len(SEEDS))
        reused = set()
        for job in filter_jobs(
            plan,
            policy=MIN_POLICY,
            context="CTX-PSEUDO",
            stage=SELECT_REFIT,
            model=D0_RECIPE,
        ):
            reused.update(dep for dep in job["dependencies"] if lookup[dep]["stage"] == SOURCE_PRED)
        self.assertEqual(reused, {job["job_id"] for job in predictions})

    def test_sg_and_arpls_fresh_d0_sources_cover_all_units(self):
        plan = cached_plan("pseudo")
        min_ids = {
            job["job_id"]
            for job in filter_jobs(
                plan,
                policy=MIN_POLICY,
                context="CTX-PSEUDO",
                stage=SOURCE_FIT,
                model=D0_RECIPE,
            )
        }
        for policy in POLICIES[1:]:
            fits = filter_jobs(
                plan,
                policy=policy,
                context="CTX-PSEUDO",
                stage=SOURCE_FIT,
                model=D0_RECIPE,
            )
            self.assertEqual(len(fits), 2 * len(SEEDS))
            self.assertEqual({job["unit_id"] for job in fits}, {"PU-0", "PU-1"})
            self.assertTrue({job["job_id"] for job in fits}.isdisjoint(min_ids))

    def test_g3_conditional_fresh_fits_bind_min_selection_decision(self):
        plan = cached_plan("pseudo")
        selection = select_job(plan, "CTX-PSEUDO")
        for policy in POLICIES[1:]:
            for recipe in NON_D0_RECIPES:
                fits = [
                    job
                    for job in filter_jobs(
                        plan,
                        policy=policy,
                        context="CTX-PSEUDO",
                        stage=SOURCE_FIT,
                        model=recipe,
                    )
                    if is_conditional(job)
                ]
                self.assertEqual(len(fits), 2 * len(SEEDS))
                for job in fits:
                    activation = activation_of(job)
                    self.assertEqual(activation["selection_job_id"], selection["job_id"])
                    self.assertEqual(activation["recipe_id"], recipe)

    def test_nn_epochs_and_calibration_depend_on_seed_source_predictions_only(self):
        plan = cached_plan("pseudo")
        lookup = lookup_jobs(plan)
        min_selection = select_job(plan, "CTX-PSEUDO")
        for stage in (SELECT_REFIT, SCALAR_STAGE):
            jobs = [
                job
                for job in filter_jobs(plan, context="CTX-PSEUDO", stage=stage)
                if job["model_id"] in NEURAL_RECIPES and job["seed"] in SEEDS
            ]
            self.assertTrue(jobs)
            for job in jobs:
                source_predictions = [
                    lookup[dependency_id]
                    for dependency_id in job["dependencies"]
                    if lookup[dependency_id]["stage"] == SOURCE_PRED
                ]
                self.assertEqual(len(source_predictions), 2)
                for dependency in source_predictions:
                    self.assertEqual(dependency["policy_id"], job["policy_id"])
                    self.assertEqual(dependency["model_id"], job["model_id"])
                    self.assertEqual(dependency["seed"], job["seed"])
                extra = [
                    lookup[dependency_id]
                    for dependency_id in job["dependencies"]
                    if lookup[dependency_id]["stage"] != SOURCE_PRED
                ]
                for dependency in extra:
                    self.assertEqual(dependency["stage"], SELECT_STAGE)
                    self.assertEqual(dependency["policy_id"], MIN_POLICY)
                    self.assertEqual(dependency["job_id"], min_selection["job_id"])
                if stage == SCALAR_STAGE:
                    self.assertIn(min_selection["job_id"], job["dependencies"])
                    metadata = job.get("metadata", {})
                    self.assertEqual(metadata.get("role_purpose"), "inherited_selection_only")
                    self.assertIs(metadata.get("excludes_guards"), True)
                    self.assertIs(metadata.get("excludes_classical_calibration"), True)
                    self.assertIs(metadata.get("excludes_test_evidence"), True)
                    self.assertEqual(
                        job["resolution"],
                        "master_equal_source_logits_then_seed_probability_average",
                    )

    def test_nn_temperature_applied_before_probability_average(self):
        plan = cached_plan("pseudo")
        lookup = lookup_jobs(plan)
        min_selection = select_job(plan, "CTX-PSEUDO")
        for job in filter_jobs(plan, context="CTX-PSEUDO", stage=HELD_STAGE):
            if job["model_id"] not in NEURAL_RECIPES:
                continue
            dependencies = [lookup[dep] for dep in job["dependencies"]]
            for dependency in dependencies:
                if dependency["stage"] == SELECT_STAGE:
                    self.assertEqual(dependency["policy_id"], MIN_POLICY)
                    self.assertEqual(dependency["job_id"], min_selection["job_id"])
            core = [item for item in dependencies if item["stage"] != SELECT_STAGE]
            self.assertEqual({item["stage"] for item in core}, {REFIT_STAGE, SCALAR_STAGE})
            self.assertEqual({item["seed"] for item in core}, {job["seed"]})
        for job in filter_jobs(plan, context="CTX-PSEUDO", stage=ENSEMBLE_STAGE):
            if job["model_id"] not in NEURAL_RECIPES:
                continue
            dependencies = [lookup[dep] for dep in job["dependencies"]]
            for dependency in dependencies:
                if dependency["stage"] == SELECT_STAGE:
                    self.assertEqual(dependency["policy_id"], MIN_POLICY)
                    self.assertEqual(dependency["job_id"], min_selection["job_id"])
            held = [item for item in dependencies if item["stage"] == HELD_STAGE]
            self.assertEqual(len(held), len(SEEDS))
            self.assertEqual({item["seed"] for item in held}, set(SEEDS))
            self.assertEqual(len({item["seed"] for item in held}), len(SEEDS))

    def test_neural_only_context_has_no_classical_roles(self):
        plan = cached_plan("neural_only")
        context_jobs = filter_jobs(plan, context="CTX-NEURAL-ONLY")
        self.assertTrue(context_jobs)
        for job in context_jobs:
            self.assertNotIn(job["model_id"], CLASSICAL_MODELS)
            self.assertNotIn(
                job["stage"],
                (CLASSICAL_CAL, CLASSICAL_ALIAS, CAL_PRED, "select_hyperparameters"),
            )

    def test_neural_only_pipeline_fully_available(self):
        plan = cached_plan("neural_only")
        ensembles = filter_jobs(
            plan, context="CTX-NEURAL-ONLY", stage=ENSEMBLE_STAGE, model=D0_RECIPE
        )
        self.assertEqual(len(ensembles), len(POLICIES))
        dev = [
            job
            for job in filter_jobs(plan, context="CTX-NEURAL-ONLY", stage=SOURCE_FIT)
            if job["policy_id"] == MIN_POLICY and job["model_id"] in NEURAL_RECIPES
        ]
        self.assertEqual(len(dev), 2 * len(NEURAL_RECIPES) * len(SEEDS))

    def test_non_min_source_fits_depend_on_min_selection(self):
        plan = cached_plan("pseudo")
        min_selection = select_job(plan, "CTX-PSEUDO")
        for policy in POLICIES[1:]:
            fits = [
                job
                for job in filter_jobs(plan, policy=policy, context="CTX-PSEUDO", stage=SOURCE_FIT)
                if job["model_id"] in NEURAL_RECIPES
            ]
            self.assertTrue(fits)
            for job in fits:
                self.assertIn(min_selection["job_id"], job["dependencies"])

    def test_all_neural_calibration_and_refit_jobs_depend_on_min_selection(self):
        plan = cached_plan("pseudo")
        min_selection = select_job(plan, "CTX-PSEUDO")
        for stage in (SCALAR_STAGE, REFIT_STAGE):
            jobs = [
                job
                for job in filter_jobs(plan, context="CTX-PSEUDO", stage=stage)
                if job["model_id"] in NEURAL_RECIPES
            ]
            self.assertTrue(jobs)
            for job in jobs:
                self.assertIn(min_selection["job_id"], job["dependencies"])

    def test_master_neural_only_context_has_no_classical_jobs(self):
        context = make_master_neural_only_context()
        plan = build_plan(contexts=[context])
        context_jobs = filter_jobs(plan, context=context["context_id"])
        self.assertTrue(context_jobs)
        for job in context_jobs:
            self.assertNotIn(job["model_id"], CLASSICAL_MODELS)
            self.assertNotIn(
                job["stage"],
                (CLASSICAL_CAL, CLASSICAL_ALIAS, CAL_PRED, "select_hyperparameters"),
            )


# ===========================================================================
# conditional branches and graph structure
# ===========================================================================


class GraphStructureTests(unittest.TestCase):
    def test_every_conditional_job_directly_depends_on_selection(self):
        for kind in ("pseudo", "neural_only", "master"):
            plan = cached_plan(kind)
            for job in plan["jobs"]:
                activation = activation_of(job)
                if activation is None:
                    continue
                with self.subTest(kind=kind, job=job["job_id"]):
                    self.assertIn(activation["selection_job_id"], job["dependencies"])

    def test_unconditional_jobs_never_depend_on_conditional_jobs(self):
        for kind in ("pseudo", "neural_only", "master"):
            plan = cached_plan(kind)
            lookup = lookup_jobs(plan)
            for job in plan["jobs"]:
                if is_conditional(job):
                    continue
                for dependency_id in job["dependencies"]:
                    with self.subTest(kind=kind, job=job["job_id"]):
                        self.assertFalse(is_conditional(lookup[dependency_id]))

    def test_conditional_edges_stay_within_recipe(self):
        plan = cached_plan("pseudo")
        lookup = lookup_jobs(plan)
        for job in plan["jobs"]:
            if not is_conditional(job):
                continue
            for dependency_id in job["dependencies"]:
                dependency = lookup[dependency_id]
                if is_conditional(dependency):
                    self.assertEqual(dependency["model_id"], job["model_id"])

    def test_cross_policy_edges_only_to_min_select_neural_recipe(self):
        for kind in ("pseudo", "neural_only", "master"):
            plan = cached_plan(kind)
            lookup = lookup_jobs(plan)
            for job in plan["jobs"]:
                for dependency_id in job["dependencies"]:
                    dependency = lookup[dependency_id]
                    if dependency["policy_id"] != job["policy_id"]:
                        with self.subTest(kind=kind, job=job["job_id"]):
                            self.assertEqual(dependency["stage"], SELECT_STAGE)
                            self.assertEqual(dependency["policy_id"], MIN_POLICY)

    def test_dependencies_exist_same_context_and_precede_dependents(self):
        for kind in ("pseudo", "neural_only", "master", "mixed"):
            plan = cached_plan(kind)
            lookup = lookup_jobs(plan)
            positions = {job["job_id"]: index for index, job in enumerate(plan["jobs"])}
            for job in plan["jobs"]:
                for dependency_id in job["dependencies"]:
                    with self.subTest(kind=kind, job=job["job_id"]):
                        self.assertIn(dependency_id, lookup)
                        self.assertEqual(lookup[dependency_id]["context_id"], job["context_id"])
                        self.assertLess(positions[dependency_id], positions[job["job_id"]])

    def test_no_dependency_cycles(self):
        plan = cached_plan("mixed")
        lookup = lookup_jobs(plan)
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
        self.assertEqual(visited, set(lookup))


# ===========================================================================
# classical operations
# ===========================================================================


class ClassicalOperationsTests(unittest.TestCase):
    def _classical(self, plan, *, policy=None, model=None, stage=None):
        return [
            job
            for job in plan["jobs"]
            if job["model_id"] in CLASSICAL_MODELS
            and (policy is None or job["policy_id"] == policy)
            and (model is None or job["model_id"] == model)
            and (stage is None or job["stage"] == stage)
        ]

    def test_classical_counts_pseudo_context(self):
        plan = cached_plan("pseudo")
        expected_source = len(POLICIES) * (2 * 36 + 2 * 16 * 3 + 2 * 16 * 3)
        expected_cal = len(POLICIES) * (3 * 1 + 3 * 3 + 3 * 3)
        expected_refit = len(POLICIES) * (1 + 3 + 3)
        self.assertEqual(len(self._classical(plan, stage=SOURCE_FIT)), expected_source)
        self.assertEqual(len(self._classical(plan, stage=SOURCE_PRED)), expected_source)
        self.assertEqual(len(self._classical(plan, stage=CLASSICAL_CAL)), expected_cal)
        self.assertEqual(len(self._classical(plan, stage=CAL_PRED)), expected_cal)
        self.assertEqual(len(self._classical(plan, stage=REFIT_STAGE)), expected_refit)
        classical_fits = [job for job in self._classical(plan) if job["stage"] in FIT_STAGES]
        self.assertEqual(len(classical_fits), 876)
        self.assertEqual(len(classical_fits), expected_source + expected_cal + expected_refit)

    def test_classical_counts_master_context(self):
        plan = cached_plan("master")
        expected_source = len(POLICIES) * (3 * 36 + 3 * 16 * 3 + 3 * 16 * 3)
        expected_alias = len(POLICIES) * (3 * 1 + 3 * 3 + 3 * 3)
        expected_refit = len(POLICIES) * (1 + 3 + 3)
        self.assertEqual(len(self._classical(plan, stage=SOURCE_FIT)), expected_source)
        self.assertEqual(len(self._classical(plan, stage=CLASSICAL_CAL)), 0)
        self.assertEqual(len(self._classical(plan, stage=CLASSICAL_ALIAS)), expected_alias)
        self.assertEqual(len(self._classical(plan, stage=REFIT_STAGE)), expected_refit)
        classical_fits = [job for job in self._classical(plan) if job["stage"] in FIT_STAGES]
        self.assertEqual(len(classical_fits), 1209)
        self.assertEqual(len(classical_fits), expected_source + expected_refit)

    def test_master_calibration_aliases_have_exact_role_hash_match(self):
        plan = cached_plan("master")
        aliases = self._classical(plan, stage=CLASSICAL_ALIAS)
        units = {unit["unit_id"]: unit for unit in make_master_context()["selection_units"]}
        self.assertTrue(aliases)
        for alias in aliases:
            unit = units[alias["unit_id"]]
            self.assertEqual(alias["fit_uid_sha256"], unit["fit_uid_sha256"])
            self.assertEqual(alias["validation_uid_sha256"], unit["validation_uid_sha256"])
            self.assertEqual(
                alias["resolution"], "resolve_selected_candidate_after_source_selection"
            )

    def test_pseudo_has_fresh_calibration_fits_and_no_aliases(self):
        plan = cached_plan("pseudo")
        self.assertEqual(len(self._classical(plan, stage=CLASSICAL_ALIAS)), 0)
        self.assertTrue(self._classical(plan, stage=CLASSICAL_CAL))
        self.assertTrue(self._classical(plan, stage=CAL_PRED))

    def test_classical_held_uncalibrated_then_single_temperature(self):
        for kind, context_id in (
            ("pseudo", "CTX-PSEUDO"),
            ("master", "CTX-MASTER"),
        ):
            plan = cached_plan(kind)
            lookup = lookup_jobs(plan)
            for policy in POLICIES:
                for model in CLASSICAL_MODELS:
                    held = [
                        job
                        for job in filter_jobs(
                            plan, policy=policy, context=context_id, stage=HELD_STAGE, model=model
                        )
                    ]
                    expected_seeds = 1 if model == SVM_MODEL else len(SEEDS)
                    self.assertEqual(len(held), expected_seeds)
                    for job in held:
                        self.assertEqual(job["resolution"], "uncalibrated_scores_only")
                        self.assertEqual(len(job["dependencies"]), 1)
                        dependency = lookup[job["dependencies"][0]]
                        self.assertEqual(dependency["stage"], REFIT_STAGE)
                        self.assertEqual(dependency["seed"], job["seed"])
                    scalar = filter_jobs(
                        plan, policy=policy, context=context_id, stage=SCALAR_STAGE, model=model
                    )
                    self.assertEqual(len(scalar), 1)
                    self.assertEqual(
                        scalar[0]["resolution"], "calibrate_seed_averaged_source_scores"
                    )
                    ensembles = filter_jobs(
                        plan, policy=policy, context=context_id, stage=ENSEMBLE_STAGE, model=model
                    )
                    self.assertEqual(len(ensembles), 1)
                    self.assertEqual(
                        ensembles[0]["resolution"], "seed_average_then_single_temperature"
                    )
                    expected = sorted([job["job_id"] for job in held] + [scalar[0]["job_id"]])
                    self.assertEqual(ensembles[0]["dependencies"], expected)

    def test_four_method_panel_removes_only_extra_trees_jobs(self):
        five = cached_plan("pseudo", True)
        four = cached_plan("pseudo", False)
        extra_trees_ids = {
            job["job_id"] for job in five["jobs"] if job["model_id"] == "C-EXTRA-TREES"
        }
        five_ids = {job["job_id"] for job in five["jobs"]}
        four_ids = {job["job_id"] for job in four["jobs"]}
        self.assertEqual(four_ids, five_ids - extra_trees_ids)
        extra_trees_fits = [
            job
            for job in five["jobs"]
            if job["model_id"] == "C-EXTRA-TREES" and job["stage"] in FIT_STAGES
        ]
        removed = fit_slot_count(five["summary"]["catalog_stage_counts"]) - fit_slot_count(
            four["summary"]["catalog_stage_counts"]
        )
        self.assertEqual(removed, len(extra_trees_fits))
        self.assertEqual(len(extra_trees_fits), len(POLICIES) * (2 * 16 * 3 + 3 * 3 + 3))
        self.assertEqual(
            len(
                [
                    job
                    for job in five["jobs"]
                    if job["model_id"] == "C-EXTRA-TREES" and job["stage"] == SCALAR_STAGE
                ]
            ),
            len(POLICIES),
        )
        self.assertEqual(
            len(
                [
                    job
                    for job in five["jobs"]
                    if job["model_id"] == "C-EXTRA-TREES" and job["stage"] == ENSEMBLE_STAGE
                ]
            ),
            len(POLICIES),
        )

    def test_master_extra_trees_removal_count(self):
        five = cached_plan("master", True)
        four = cached_plan("master", False)
        extra_trees_fits = [
            job
            for job in five["jobs"]
            if job["model_id"] == "C-EXTRA-TREES" and job["stage"] in FIT_STAGES
        ]
        self.assertEqual(len(extra_trees_fits), len(POLICIES) * (3 * 16 * 3 + 3))
        removed = fit_slot_count(five["summary"]["catalog_stage_counts"]) - fit_slot_count(
            four["summary"]["catalog_stage_counts"]
        )
        self.assertEqual(removed, len(extra_trees_fits))


# ===========================================================================
# endpoints and selection aliases
# ===========================================================================


class EndpointAliasTests(unittest.TestCase):
    def test_alias_targets_are_ensembles_same_context_and_policy(self):
        plan = cached_plan("mixed")
        lookup = lookup_jobs(plan)
        self.assertTrue(aliases_of(plan))
        for alias in aliases_of(plan):
            self.assertIn(alias["strategy"], (D0_STRATEGY, SELECTED_STRATEGY))
            for target_id in alias_target_ids(alias):
                target = lookup[target_id]
                with self.subTest(alias=alias["alias_id"]):
                    self.assertEqual(target["stage"], ENSEMBLE_STAGE)
                    self.assertEqual(target["context_id"], alias["context_id"])
                    self.assertEqual(target["policy_id"], alias["policy_id"])

    def test_two_logical_endpoint_aliases_per_neural_context_policy(self):
        plan = cached_plan("mixed")
        for context_id in ("CTX-M", "CTX-P", "CTX-N"):
            for policy in POLICIES:
                aliases = [
                    alias
                    for alias in aliases_of(plan)
                    if alias["context_id"] == context_id and alias["policy_id"] == policy
                ]
                with self.subTest(context=context_id, policy=policy):
                    self.assertEqual(len(aliases), 2)
                    self.assertEqual(
                        {alias["strategy"] for alias in aliases},
                        {D0_STRATEGY, SELECTED_STRATEGY},
                    )

    def test_g3_selected_alias_covers_all_allowed_recipes_without_chosen_recipe(self):
        plan = cached_plan("pseudo")
        lookup = lookup_jobs(plan)
        min_selection = select_job(plan, "CTX-PSEUDO")
        self.assertIsNotNone(min_selection.get("metadata", {}).get("ready_rule"))
        for policy in POLICIES:
            with self.subTest(policy=policy):
                selected = [
                    alias
                    for alias in aliases_of(plan)
                    if alias["strategy"] == SELECTED_STRATEGY
                    and alias["context_id"] == "CTX-PSEUDO"
                    and alias["policy_id"] == policy
                ]
                self.assertEqual(len(selected), 1)
                self.assertIsNone(selected[0]["recipe_id"])
                target_ids = alias_target_ids(selected[0])
                recipes = {lookup[target_id]["model_id"] for target_id in target_ids}
                self.assertEqual(recipes, set(NEURAL_RECIPES))
                for target_id in target_ids:
                    self.assertEqual(lookup[target_id]["stage"], ENSEMBLE_STAGE)

    def test_non_g3_selected_alias_is_an_explicit_d0_fallback(self):
        for kind, context_id in (
            ("master", "CTX-MASTER"),
            ("neural_only", "CTX-NEURAL-ONLY"),
        ):
            plan = cached_plan(kind)
            lookup = lookup_jobs(plan)
            for policy in POLICIES:
                selected = [
                    alias
                    for alias in aliases_of(plan)
                    if alias["strategy"] == SELECTED_STRATEGY
                    and alias["context_id"] == context_id
                    and alias["policy_id"] == policy
                ]
                with self.subTest(kind=kind, policy=policy):
                    self.assertEqual(len(selected), 1)
                    self.assertEqual(selected[0]["recipe_id"], D0_RECIPE)
                    self.assertIsNone(selected[0].get("conditional_targets"))
                    target_ids = alias_target_ids(selected[0])
                    self.assertTrue(target_ids)
                    for target_id in target_ids:
                        self.assertEqual(lookup[target_id]["model_id"], D0_RECIPE)

    def test_selected_alias_metadata_matches_selection_contract(self):
        plan = cached_plan("pseudo")
        selected = [alias for alias in aliases_of(plan) if alias["strategy"] == SELECTED_STRATEGY]
        self.assertTrue(selected)
        for alias in selected:
            metadata = alias.get("metadata", {})
            with self.subTest(alias=alias["alias_id"]):
                self.assertEqual(metadata.get("ready_rule"), "await_ready_for_refit")
                self.assertIs(metadata.get("no_unknown_or_failure_fallback"), True)
        g3_selected = [alias for alias in selected if alias["context_id"] == "CTX-PSEUDO"]
        self.assertTrue(g3_selected)
        for alias in g3_selected:
            metadata = alias.get("metadata", {})
            with self.subTest(alias=alias["alias_id"]):
                self.assertIs(metadata.get("unknown_recipe_until_selection"), True)
                self.assertIsNone(alias["recipe_id"])

    def test_structural_d0_alias_metadata_matches_selection_contract(self):
        for context in ("master", "neural_only"):
            plan = cached_plan(context)
            structural = [
                alias for alias in aliases_of(plan) if alias["strategy"] == SELECTED_STRATEGY
            ]
            with self.subTest(context=context):
                self.assertTrue(structural)
                for alias in structural:
                    metadata = alias.get("metadata", {})
                    with self.subTest(alias=alias["alias_id"]):
                        self.assertEqual(alias.get("recipe_id"), D0_RECIPE)
                        self.assertIs(metadata.get("d0_default_requires_selection"), True)
                        self.assertIs(metadata.get("d0_default_requires_ready_for_refit"), True)
                        self.assertIs(metadata.get("no_unknown_or_failure_fallback"), True)
                        self.assertEqual(metadata.get("ready_rule"), "await_ready_for_refit")


# ===========================================================================
# summary arithmetic
# ===========================================================================


class SummaryArithmeticTests(unittest.TestCase):
    def test_fully_supported_pseudo_summary_arithmetic(self):
        plan = cached_plan("pseudo")
        summary = plan["summary"]
        self.assertEqual(summary["lower"]["model_fit_slots"], 957)
        self.assertEqual(summary["upper"]["model_fit_slots"], 978)
        self.assertEqual(fit_slot_count(summary["catalog_stage_counts"]), 1020)
        self.assertEqual(summary["lower"]["scalar_calibrations"], 18)
        self.assertEqual(summary["upper"]["scalar_calibrations"], 27)
        self.assertEqual(summary["excluded_source_fit_slots"], 0)
        self.assertEqual(summary["logical_neural_endpoints"], 6)
        self.assertEqual(summary["classical_family_ensembles"], 9)
        self.assertEqual(summary["conditional_context_count"], 1)

    def test_master_summary_arithmetic(self):
        plan = cached_plan("master")
        summary = plan["summary"]
        self.assertEqual(summary["lower"]["model_fit_slots"], 1272)
        self.assertEqual(summary["upper"]["model_fit_slots"], 1272)
        self.assertEqual(summary["lower"]["scalar_calibrations"], 18)
        self.assertEqual(summary["excluded_source_fit_slots"], 0)
        self.assertEqual(summary["conditional_context_count"], 0)
        self.assertEqual(summary["logical_neural_endpoints"], 6)

    def test_neural_only_summary_arithmetic(self):
        plan = cached_plan("neural_only")
        summary = plan["summary"]
        self.assertEqual(summary["lower"]["model_fit_slots"], 69)
        self.assertEqual(summary["upper"]["model_fit_slots"], 69)
        self.assertEqual(summary["lower"]["scalar_calibrations"], 9)
        self.assertEqual(summary["excluded_source_fit_slots"], 12)
        self.assertEqual(summary["conditional_context_count"], 0)
        self.assertEqual(summary["logical_neural_endpoints"], 6)
        self.assertEqual(summary["classical_family_ensembles"], 0)

    def test_upper_bound_counts_a_single_representative_recipe_only(self):
        plan = cached_plan("pseudo")
        summary = plan["summary"]
        difference = summary["upper"]["model_fit_slots"] - summary["lower"]["model_fit_slots"]
        groups = {}
        for job in plan["jobs"]:
            if not is_conditional(job):
                continue
            key = (job["context_id"], job["model_id"])
            groups.setdefault(key, 0)
            if job["stage"] in FIT_STAGES:
                groups[key] += 1
        counts = set(groups.values())
        self.assertEqual(len(counts), 1)
        self.assertEqual(difference, counts.pop())
        total_conditional = sum(groups.values())
        self.assertEqual(total_conditional, len(NON_D0_RECIPES) * difference)
        self.assertGreater(total_conditional, difference)

    def test_per_model_bounds_partition_the_bounds_and_are_not_additive(self):
        plan = cached_plan("pseudo")
        summary = plan["summary"]
        per_model = summary["per_model"]
        self.assertIs(per_model["bounds_are_nonadditive"], True)
        for section in ("catalog", "lower", "upper"):
            self.assertIn(section, per_model)
            self.assertTrue(per_model[section])
        for model, stats in per_model["lower"].items():
            with self.subTest(model=model):
                self.assertEqual(
                    set(stats),
                    {
                        "stage_counts",
                        "model_fit_slots",
                        "scalar_calibrations",
                        "total_jobs",
                    },
                )
                self.assertEqual(stats["model_fit_slots"], fit_slot_count(stats["stage_counts"]))
        self.assertEqual(
            sum(stats["total_jobs"] for stats in per_model["lower"].values()),
            summary["lower"]["total_jobs"],
        )
        self.assertEqual(
            sum(stats["total_jobs"] for stats in per_model["upper"].values()),
            sum(stats["total_jobs"] for stats in per_model["catalog"].values()),
        )
        self.assertGreater(
            sum(stats["total_jobs"] for stats in per_model["upper"].values()),
            summary["upper"]["total_jobs"],
        )
        for recipe in NON_D0_RECIPES:
            self.assertEqual(
                per_model["upper"][recipe]["model_fit_slots"],
                per_model["upper"][D0_RECIPE]["model_fit_slots"],
            )


# ===========================================================================
# availability
# ===========================================================================


class AvailabilityTests(unittest.TestCase):
    def test_availability_entry_per_context(self):
        plan = cached_plan("mixed")
        availability = {entry["context_id"]: entry for entry in plan["availability"]}
        self.assertEqual(set(availability), {"CTX-M", "CTX-P", "CTX-N", "CTX-U", "CTX-NA"})

    def test_unavailable_contexts_are_retained_not_invented(self):
        plan = cached_plan("unavailable")
        availability = {entry["context_id"]: entry for entry in plan["availability"]}
        self.assertEqual(set(availability), {"CTX-UNSUPPORTED", "CTX-NOT-APPLICABLE"})
        for context_id, entry in availability.items():
            with self.subTest(context=context_id):
                self.assertTrue(entry["unavailable_reasons"])
                self.assertFalse(entry["classical_supported"])
                self.assertFalse(entry["neural_supported"])
                self.assertFalse(entry["g3_comparable"])
                self.assertEqual(entry["classical_models"], [])
                self.assertEqual(entry["neural_methods"], [])
                self.assertEqual(filter_jobs(plan, context=context_id), [])
        self.assertEqual(plan["summary"]["logical_neural_endpoints"], 0)

    def test_neural_only_availability_has_no_classical_methods(self):
        plan = cached_plan("neural_only")
        entry = next(
            item for item in plan["availability"] if item["context_id"] == "CTX-NEURAL-ONLY"
        )
        self.assertEqual(entry["classical_models"], [])
        self.assertEqual(entry["neural_methods"], [D0_STRATEGY, SELECTED_STRATEGY])
        self.assertEqual(entry["allowed_recipes"], [D0_RECIPE])
        self.assertFalse(entry["g3_comparable"])

    def test_master_and_pseudo_allowed_recipes(self):
        master = cached_plan("master")
        master_entry = next(
            item for item in master["availability"] if item["context_id"] == "CTX-MASTER"
        )
        self.assertEqual(master_entry["allowed_recipes"], [D0_RECIPE])
        pseudo = cached_plan("pseudo")
        pseudo_entry = next(
            item for item in pseudo["availability"] if item["context_id"] == "CTX-PSEUDO"
        )
        self.assertEqual(set(pseudo_entry["allowed_recipes"]), set(NEURAL_RECIPES))


# ===========================================================================
# malformed inputs
# ===========================================================================


class MalformedInputTests(unittest.TestCase):
    def assert_rejected(self, **overrides):
        arguments = base_arguments()
        arguments.update(overrides)
        with self.assertRaises(ValueError) as caught:
            build_population_slots(**arguments)
        self.assertNotIn(SENTINEL, str(caught.exception))

    def test_wrong_container_types(self):
        self.assert_rejected(contexts="not-a-list")
        self.assert_rejected(contexts=True)
        self.assert_rejected(candidates="not-a-list")
        self.assert_rejected(candidates=True)
        self.assert_rejected(actions=["not-a-mapping"])
        self.assert_rejected(actions=True)
        self.assert_rejected(model_spec_sha256=None)
        self.assert_rejected(model_spec_sha256=True)
        self.assert_rejected(contexts=[])

    def test_binding_fields_invalid(self):
        for bad in ("", " padded", 17, None):
            with self.subTest(field="population_id", bad=repr(bad)):
                self.assert_rejected(population_id=bad)
        for field in (
            "population_sha256",
            "population_plan_sha256",
            "neural_support_sha256",
        ):
            for bad in ("", 17, None, "abc", "A" * 64):
                with self.subTest(field=field, bad=repr(bad)):
                    self.assert_rejected(**{field: bad})
        self.assert_rejected(include_extra_trees="yes")
        self.assert_rejected(include_extra_trees=1)
        self.assert_rejected(include_extra_trees=None)

    def test_bad_context_keys(self):
        context = make_pseudo_context()
        del context["selection_units"]
        self.assert_rejected(contexts=[context])
        context = make_pseudo_context()
        context["unexpected"] = "x"
        self.assert_rejected(contexts=[context])
        context = make_pseudo_context()
        del context["guard_units"]
        self.assert_rejected(contexts=[context])

    def test_bad_modes_and_eligibility_contradictions(self):
        for mode in ("holdout", "master_cv ", SENTINEL, "pseudo_dov"):
            context = make_pseudo_context()
            context["selection_mode"] = mode
            with self.subTest(mode=mode):
                self.assert_rejected(contexts=[context])
        context = make_pseudo_context()
        context["domain_eligible"] = True
        context["selection_mode"] = "not_applicable"
        self.assert_rejected(contexts=[context])
        context = make_pseudo_context()
        context["domain_eligible"] = False
        context["selection_mode"] = PSEUDO_MODE
        self.assert_rejected(contexts=[context])
        context = make_master_context()
        context["g3_comparable"] = True
        self.assert_rejected(contexts=[context])
        context = make_neural_only_context()
        context["g3_comparable"] = True
        self.assert_rejected(contexts=[context])

    def test_bad_support_flags(self):
        for flag in (
            "domain_eligible",
            "classical_supported",
            "neural_supported",
            "g3_comparable",
        ):
            context = make_pseudo_context()
            context[flag] = 1
            with self.subTest(flag=flag):
                self.assert_rejected(contexts=[context])

    def test_duplicate_contexts_and_units(self):
        contexts = [make_pseudo_context(SENTINEL), make_pseudo_context(SENTINEL)]
        self.assert_rejected(contexts=contexts)
        context = make_pseudo_context()
        context["selection_units"][1]["unit_id"] = context["selection_units"][0]["unit_id"]
        self.assert_rejected(contexts=[context])
        context = make_pseudo_context()
        context["guard_units"][1]["unit_id"] = context["guard_units"][0]["unit_id"]
        self.assert_rejected(contexts=[context])

    def test_guard_and_calibration_role_id_collisions(self):
        context = make_neural_only_context()
        context["guard_units"][0]["unit_id"] = context["selection_units"][0]["unit_id"]
        self.assert_rejected(contexts=[context])
        context = make_pseudo_context()
        context["calibration_units"][0]["unit_id"] = context["selection_units"][0]["unit_id"]
        self.assert_rejected(contexts=[context])

    def test_master_calibration_identity_mismatches(self):
        master = make_master_context()
        master["calibration_units"][0]["unit_id"] = "OTHER-UNIT"
        self.assert_rejected(contexts=[master])
        master = make_master_context()
        master["calibration_units"][0]["validation_uid_sha256"] = sha("other-validation")
        self.assert_rejected(contexts=[master])
        master = make_master_context()
        master["calibration_units"][0]["fit_uid_sha256"] = sha("other-fit")
        self.assert_rejected(contexts=[master])

    def test_equal_fit_validation_and_outer_test_conflicts(self):
        for role in ("selection_units", "calibration_units", "guard_units"):
            context = make_pseudo_context()
            context[role][0]["validation_uid_sha256"] = context[role][0]["fit_uid_sha256"]
            with self.subTest(role=role, conflict="fit-validation"):
                self.assert_rejected(contexts=[context])
        for role in ("selection_units", "calibration_units", "guard_units"):
            for key in ("fit_uid_sha256", "validation_uid_sha256"):
                context = make_pseudo_context()
                context[role][0][key] = context["outer_test_uid_sha256"]
                with self.subTest(role=role, key=key, conflict="outer-test"):
                    self.assert_rejected(contexts=[context])
        context = make_neural_only_context()
        context["guard_units"][0]["validation_uid_sha256"] = context["guard_units"][0][
            "fit_uid_sha256"
        ]
        self.assert_rejected(contexts=[context])
        context = make_neural_only_context()
        context["guard_units"][0]["fit_uid_sha256"] = context["outer_test_uid_sha256"]
        self.assert_rejected(contexts=[context])
        context = make_pseudo_context()
        context["outer_test_uid_sha256"] = context["outer_fit_uid_sha256"]
        self.assert_rejected(contexts=[context])

    def test_bad_guard_support_and_reason(self):
        context = make_neural_only_context()
        context["guard_units"][0]["support"] = "yes"
        self.assert_rejected(contexts=[context])
        context = make_neural_only_context()
        context["guard_units"][0]["exclusion_reason"] = "unexpected"
        self.assert_rejected(contexts=[context])
        context = make_neural_only_context()
        context["guard_units"][2]["exclusion_reason"] = None
        self.assert_rejected(contexts=[context])
        context = make_neural_only_context()
        context["guard_units"][2]["exclusion_reason"] = "   "
        self.assert_rejected(contexts=[context])
        context = make_neural_only_context()
        context["guard_units"][2]["exclusion_reason"] = 17
        self.assert_rejected(contexts=[context])
        context = make_neural_only_context()
        context["guard_units"][2]["support"] = False
        context["guard_units"][2]["exclusion_reason"] = "no_comparable_label"
        plan = build_plan(contexts=[context])
        slots = excluded_of(plan)
        self.assertEqual({slot["exclusion_reason"] for slot in slots}, {"no_comparable_label"})

    def test_missing_units_for_claimed_support(self):
        context = make_pseudo_context()
        context["classical_supported"] = False
        self.assert_rejected(contexts=[context])
        context = make_pseudo_context()
        context["guard_units"] = context["guard_units"][:2]
        context["neural_supported"] = True
        self.assert_rejected(contexts=[context])
        context = make_master_context()
        context["guard_units"] = [make_guard("MX-0", "master-extra", support=True)]
        context["neural_supported"] = True
        self.assert_rejected(contexts=[context])
        context = make_neural_only_context()
        context["calibration_units"] = [make_unit(f"NCU-{i}", "neural") for i in range(3)]
        context["classical_supported"] = False
        self.assert_rejected(contexts=[context])

    def test_action_spec_and_candidate_malformations(self):
        actions = make_actions()
        del actions["R_MIN_400_1800"]
        self.assert_rejected(actions=actions)
        actions = make_actions()
        actions[SENTINEL] = sha("extra")
        self.assert_rejected(actions=actions)
        actions = make_actions()
        actions["R_SG_400_1800"] = "A" * 64
        self.assert_rejected(actions=actions)
        actions = make_actions()
        actions["R_ARPLS_400_1800"] = "abc"
        self.assert_rejected(actions=actions)
        actions = make_actions()
        actions["R_ARPLS_400_1800"] = 123
        self.assert_rejected(actions=actions)

        spec = make_model_spec()
        del spec["D3"]
        self.assert_rejected(model_spec_sha256=spec)
        spec = make_model_spec()
        spec[SENTINEL] = sha("extra")
        self.assert_rejected(model_spec_sha256=spec)
        spec = make_model_spec()
        spec["D1"] = spec["D1"][:-1] + "g"
        self.assert_rejected(model_spec_sha256=spec)

        candidates = make_candidates()
        candidates.pop()
        self.assert_rejected(candidates=candidates)
        candidates = make_candidates()
        candidates[0]["hyperparameter_sha256"] = "abc"
        self.assert_rejected(candidates=candidates)
        candidates = make_candidates()
        candidates[1]["candidate_id"] = candidates[0]["candidate_id"]
        self.assert_rejected(candidates=candidates)

    def test_unit_ids_may_repeat_across_contexts(self):
        first = make_pseudo_context("CTX-A")
        second = make_pseudo_context("CTX-B")
        second["selection_units"] = [dict(unit) for unit in first["selection_units"]]
        second["calibration_units"] = [dict(unit) for unit in first["calibration_units"]]
        second["guard_units"] = [dict(unit) for unit in first["guard_units"]]
        second["outer_fit_uid_sha256"] = sha("B::outer-fit")
        second["outer_test_uid_sha256"] = sha("B::outer-test")
        plan = build_plan(contexts=[first, second])
        self.assertEqual({job["context_id"] for job in plan["jobs"]}, {"CTX-A", "CTX-B"})

    def test_master_identical_selection_and_calibration_is_canonical(self):
        plan = build_plan(contexts=[make_master_context()])
        aliases = [job for job in plan["jobs"] if job["stage"] == "calibration_prediction_alias"]
        self.assertTrue(aliases)

    def test_ineligible_contexts_with_roles_are_rejected(self):
        context = make_not_applicable_context()
        context["selection_units"] = [make_unit("NAU-0", "not-applicable")]
        self.assert_rejected(contexts=[context])
        context = make_unsupported_context()
        context["selection_units"] = [make_unit("UU-0", "unsupported")]
        self.assert_rejected(contexts=[context])
        context = make_unsupported_context()
        context["guard_units"] = [make_guard("UG-0", "unsupported", support=True)]
        self.assert_rejected(contexts=[context])

    def test_all_malformed_errors_are_plain_valueerror(self):
        cases = []
        context = make_pseudo_context(SENTINEL)
        cases.append({"contexts": [context, make_pseudo_context(SENTINEL)]})
        context = make_pseudo_context()
        context["selection_units"][0]["unit_id"] = SENTINEL
        context["selection_units"][1]["unit_id"] = SENTINEL
        cases.append({"contexts": [context]})
        actions = make_actions()
        actions[SENTINEL] = sha("x")
        cases.append({"actions": actions})
        spec = make_model_spec()
        spec[SENTINEL] = sha("x")
        cases.append({"model_spec_sha256": spec})
        for overrides in cases:
            with self.subTest(overrides=sorted(overrides)):
                self.assert_rejected(**overrides)


class CoreClassicalGraphParityTests(unittest.TestCase):
    def test_core_classical_graph_parity_with_universal_plan(self):
        from atlas_sers.evaluation import p08_plan

        context = make_pseudo_context()
        universal_context = {
            "context_id": context["context_id"],
            "selection_mode": context["selection_mode"],
            "selected_recipe_id": D0_RECIPE,
            "outer_fit_uid_sha256": context["outer_fit_uid_sha256"],
            "outer_test_uid_sha256": context["outer_test_uid_sha256"],
            "selection_units": context["selection_units"],
            "calibration_units": context["calibration_units"],
        }
        universal = p08_plan.build_universal_plan(
            [universal_context], make_candidates(), make_actions(), make_model_spec()
        )
        plan = build_plan(contexts=[context])

        def normalise(jobs):
            keys = {}
            for job in jobs:
                keys[job["job_id"]] = (
                    job["policy_id"],
                    job["context_id"],
                    job["model_id"],
                    job["stage"],
                    job["unit_id"],
                    job["seed"],
                    job["candidate_id"],
                    job["resolution"],
                )
            graph = set()
            for job in jobs:
                graph.add(
                    (
                        keys[job["job_id"]],
                        tuple(sorted(keys[dep] for dep in job["dependencies"])),
                    )
                )
            return graph

        universal_classical = [
            job for job in universal["jobs"] if job["model_id"] in CLASSICAL_MODELS
        ]
        population_classical = [job for job in plan["jobs"] if job["model_id"] in CLASSICAL_MODELS]
        self.assertEqual(normalise(population_classical), normalise(universal_classical))


if __name__ == "__main__":
    unittest.main()
