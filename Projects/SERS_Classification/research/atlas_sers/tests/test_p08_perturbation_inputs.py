"""Synthetic, metadata-only contract tests for the P08 stress input catalog.

These tests build a bounded toy graph from invented metadata plus the public
design contract. They grant no scientific authority, touch no private inputs,
write no files and execute no numerical work.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from atlas_sers.evaluation.p08_perturbation_design import (
    build_perturbation_case_manifest,
)
from atlas_sers.evaluation.p08_perturbation_inputs import (
    build_stress_input_catalog,
    iter_stress_input_jobs,
    require_scientific_execution,
)
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256

_DESIGN_PATH = (
    Path(__file__).resolve().parents[1] / "plan" / "contracts" / "p08_perturbation_design.json"
)

POPULATION_UIDS = ["a", "b", "c", "d", "e", "f"]
RAW_UIDS = ["raw-a", "raw-b", "raw-c", "raw-d", "raw-e", "raw-f"]
ACTIONS = ("R_MIN_400_1800", "R_SG_400_1800", "R_ARPLS_400_1800")
NA = "not_applicable"
CLEAN_CASE = "P08-STRESS-CLEAN"
_ACTION_COUNT = len(ACTIONS)
_STOCHASTIC_REPLICATES = 10

_CATALOG_KEYS = {
    "schema_version",
    "execution_authorized",
    "scientific_operations",
    "artifact_provenance_independently_verified",
    "numerical_inputs_computed",
    "full_stress_job_ledger_complete",
    "design",
    "case_manifest",
    "input_binding",
    "contexts",
    "binding_sha256",
    "summary",
    "catalog_sha256",
}

_JOB_KEYS = {
    "binding_sha256",
    "stage",
    "context_id",
    "observation_uid",
    "case_id",
    "representation_id",
    "family",
    "replicate_index",
    "depends_on_job_ids",
    "job_id",
}

_STAGE_NAMES = (
    "row_prepare",
    "source_noise_reference",
    "stochastic_realization",
    "raw_case",
    "action_transform",
    "zero_input_parity",
    "context_action_assembly",
)

_LEAK_TOKENS = ("raw-a", "raw-b", "raw-c", "raw-d", "raw-e", "raw-f")


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _load_design() -> dict:
    return json.loads(_DESIGN_PATH.read_text(encoding="utf-8"))


def _design_pins(design: dict) -> dict:
    pins = design["immutable_source_evidence"]
    return {
        "P01_contract_sha256": pins["P01_contract_sha256"],
        "P01_transform_source_sha256": pins["P01_transform_source_sha256"],
        "native_QC_source_sha256": pins["native_QC_source_sha256"],
    }


def _make_binding(design: dict, *, population_uids=None, raw_uids=None) -> dict:
    pins = _design_pins(design)
    pop = list(POPULATION_UIDS if population_uids is None else population_uids)
    raw = list(RAW_UIDS if raw_uids is None else raw_uids)
    return {
        "raw_archive_sha256": _hex("binding.raw_archive_sha256"),
        "native_qc_file_sha256": _hex("binding.native_qc_file_sha256"),
        "P01_contract_sha256": pins["P01_contract_sha256"],
        "P01_transform_source_sha256": pins["P01_transform_source_sha256"],
        "native_QC_source_sha256": pins["native_QC_source_sha256"],
        "primary_manifest_file_sha256": _hex("binding.primary_manifest_file_sha256"),
        "contexts_file_sha256": _hex("binding.contexts_file_sha256"),
        "roles_file_sha256": _hex("binding.roles_file_sha256"),
        "ordered_population_uids": pop,
        "row_order_sha256": canonical_sha256(pop),
        "ordered_raw_uids": raw,
        "raw_row_order_sha256": canonical_sha256(raw),
        "frozen_action_row_order_sha256": hashlib.sha256(
            "\n".join(pop).encode("utf-8")
        ).hexdigest(),
        "actions": {name: _hex("binding.action." + name) for name in ACTIONS},
        "starting_axis_cm1": [400, 1849, 1],
        "output_axis_cm1": [400, 1800, 1],
    }


def _make_context(context_id, fit_uids, test_uids) -> dict:
    fit = sorted(fit_uids)
    test = sorted(test_uids)
    return {
        "context_id": context_id,
        "fit_uid_sha256": canonical_sha256(fit),
        "test_uid_sha256": canonical_sha256(test),
        "fit_uids": fit,
        "test_uids": test,
    }


def _make_contexts() -> list:
    return [
        _make_context("C1", ["a", "b"], ["c", "d"]),
        _make_context("C2", ["a", "e"], ["c", "f"]),
    ]


def _refresh_context_hashes(context: dict) -> dict:
    refreshed = dict(context)
    refreshed["fit_uid_sha256"] = canonical_sha256(sorted(context["fit_uids"]))
    refreshed["test_uid_sha256"] = canonical_sha256(sorted(context["test_uids"]))
    return refreshed


def _build_catalog(*, design=None, binding=None, contexts=None) -> dict:
    if design is None:
        design = _load_design()
    if binding is None:
        binding = _make_binding(design)
    if contexts is None:
        contexts = _make_contexts()
    return build_stress_input_catalog(design=design, input_binding=binding, contexts=contexts)


def _resign(catalog: dict) -> dict:
    catalog["catalog_sha256"] = canonical_sha256(
        {k: v for k, v in catalog.items() if k != "catalog_sha256"}
    )
    return catalog


def _key(job: dict) -> tuple:
    return (
        _STAGE_NAMES.index(job["stage"]) + 1,
        job["context_id"],
        job["observation_uid"],
        job["case_id"],
        job["representation_id"],
        job["family"],
        job["replicate_index"],
    )


def _distinct_test_uids(contexts) -> list:
    uids = set()
    for context in contexts:
        uids.update(context["test_uids"])
    return sorted(uids)


def _shared_case_descriptors(manifest: dict) -> list:
    return [case for case in manifest["cases"] if case["family"] != "gaussian"]


def _gaussian_case_descriptors(manifest: dict) -> list:
    return [case for case in manifest["cases"] if case["family"] == "gaussian"]


def _expected_stage_keys(manifest: dict, contexts) -> dict:
    test_uids = _distinct_test_uids(contexts)
    shared = _shared_case_descriptors(manifest)
    gaussian = _gaussian_case_descriptors(manifest)
    keys = {stage: set() for stage in range(1, 8)}

    for uid in test_uids:
        keys[1].add((1, NA, uid, NA, NA, NA, None))

    for context in contexts:
        keys[2].add((2, context["context_id"], NA, NA, NA, NA, None))

    for uid in test_uids:
        for family in ("gaussian", "impulse"):
            for replicate in range(_STOCHASTIC_REPLICATES):
                keys[3].add((3, NA, uid, NA, NA, family, replicate))

    for case in shared:
        for uid in test_uids:
            keys[4].add(
                (
                    4,
                    NA,
                    uid,
                    case["case_id"],
                    NA,
                    case["family"],
                    case["replicate_index"],
                )
            )
    for context in contexts:
        for uid in context["test_uids"]:
            for case in gaussian:
                keys[4].add(
                    (
                        4,
                        context["context_id"],
                        uid,
                        case["case_id"],
                        NA,
                        "gaussian",
                        case["replicate_index"],
                    )
                )

    for raw_key in list(keys[4]):
        _stage, context_id, uid, case_id, _action, family, replicate = raw_key
        for action in ACTIONS:
            keys[5].add((5, context_id, uid, case_id, action, family, replicate))

    for uid in test_uids:
        for action in ACTIONS:
            keys[6].add((6, NA, uid, CLEAN_CASE, action, "clean", None))

    for context in contexts:
        for case in manifest["cases"]:
            for action in ACTIONS:
                keys[7].add(
                    (
                        7,
                        context["context_id"],
                        NA,
                        case["case_id"],
                        action,
                        case["family"],
                        case["replicate_index"],
                    )
                )

    return keys


def _expected_dependency_keys(key: tuple, manifest: dict, contexts) -> set:
    stage, context_id, uid, case_id, action, family, replicate = key
    if stage in (1, 2, 3):
        return set()
    if stage == 4:
        deps = {(1, NA, uid, NA, NA, NA, None)}
        if family in ("gaussian", "impulse"):
            deps.add((3, NA, uid, NA, NA, family, replicate))
        if family == "gaussian":
            deps.add((2, context_id, NA, NA, NA, NA, None))
        return deps
    if stage == 5:
        return {(4, context_id, uid, case_id, NA, family, replicate)}
    if stage == 6:
        return {(5, NA, uid, CLEAN_CASE, action, "clean", None)}
    if stage == 7:
        context = next(item for item in contexts if item["context_id"] == context_id)
        deps = set()
        for test_uid in context["test_uids"]:
            transform_context = context_id if family == "gaussian" else NA
            deps.add((5, transform_context, test_uid, case_id, action, family, replicate))
            deps.add((6, NA, test_uid, CLEAN_CASE, action, "clean", None))
        return deps
    raise AssertionError(key)


class TestP08StressInputCatalog(unittest.TestCase):
    def _assert_rejected(self, *, design=None, binding=None, contexts=None):
        with self.assertRaises(ValueError) as caught:
            _build_catalog(design=design, binding=binding, contexts=contexts)
        message = str(caught.exception)
        self.assertEqual(message, "invalid_stress_input_metadata")
        for token in _LEAK_TOKENS:
            self.assertNotIn(token, message)
        return caught.exception

    # -- catalog surface -------------------------------------------------

    def test_catalog_schema_authority_and_counters(self):
        catalog = _build_catalog()
        self.assertEqual(set(catalog), _CATALOG_KEYS)
        self.assertEqual(catalog["schema_version"], "nato-sers-p08-stress-input-catalog-v1")
        for field in (
            "execution_authorized",
            "artifact_provenance_independently_verified",
            "numerical_inputs_computed",
            "full_stress_job_ledger_complete",
        ):
            self.assertIs(catalog[field], False)
        self.assertIs(type(catalog["scientific_operations"]), int)
        self.assertEqual(catalog["scientific_operations"], 0)

    def test_catalog_summary_exact_counts(self):
        catalog = _build_catalog()
        summary = catalog["summary"]
        self.assertEqual(summary["context_count"], 2)
        self.assertEqual(summary["distinct_test_uid_count"], 3)
        self.assertEqual(summary["test_uid_appearances"], 4)
        self.assertEqual(summary["case_count"], 96)
        self.assertEqual(summary["context_independent_case_count"], 56)
        self.assertEqual(summary["context_dependent_case_count"], 40)
        self.assertEqual(summary["action_count"], 3)
        self.assertEqual(
            summary["stage_counts"],
            {
                "row_prepare": 3,
                "source_noise_reference": 2,
                "stochastic_realization": 60,
                "raw_case": 328,
                "action_transform": 984,
                "zero_input_parity": 9,
                "context_action_assembly": 576,
            },
        )
        self.assertEqual(summary["total_job_count"], 1962)
        self.assertEqual(set(summary["stage_counts"]), set(_STAGE_NAMES))

    def test_catalog_sha256_and_binding_sha256_are_self_consistent(self):
        catalog = _build_catalog()
        content = {k: v for k, v in catalog.items() if k != "catalog_sha256"}
        self.assertEqual(catalog["catalog_sha256"], canonical_sha256(content))
        self.assertIs(type(catalog["binding_sha256"]), str)
        self.assertEqual(len(catalog["binding_sha256"]), 64)

    # -- job surface -----------------------------------------------------

    def test_jobs_exact_fields_unique_ids_and_recomputed_digests(self):
        catalog = _build_catalog()
        jobs = list(iter_stress_input_jobs(catalog))
        self.assertEqual(len(jobs), 1962)
        seen = set()
        for job in jobs:
            self.assertEqual(set(job), _JOB_KEYS)
            self.assertEqual(job["binding_sha256"], catalog["binding_sha256"])
            body = {k: v for k, v in job.items() if k != "job_id"}
            self.assertEqual(job["job_id"], "P08STRESSINPUT-" + canonical_sha256(body))
            self.assertNotIn(job["job_id"], seen)
            seen.add(job["job_id"])
        self.assertEqual(len(seen), 1962)

    def test_jobs_repeatable_and_stage_ordered(self):
        catalog = _build_catalog()
        first = list(iter_stress_input_jobs(catalog))
        second = list(iter_stress_input_jobs(catalog))
        self.assertEqual(first, second)
        stages = [job["stage"] for job in first]
        self.assertEqual(set(stages), set(_STAGE_NAMES))
        ordinals = [_STAGE_NAMES.index(name) + 1 for name in stages]
        self.assertEqual(ordinals, sorted(ordinals))

    def test_stage_identity_sets_match_independent_expectation(self):
        design = _load_design()
        catalog = _build_catalog(design=design)
        jobs = list(iter_stress_input_jobs(catalog))
        manifest = build_perturbation_case_manifest(design=design)
        expected = _expected_stage_keys(manifest, catalog["contexts"])
        actual = {_key(job): job for job in jobs}
        self.assertEqual(len(actual), 1962)
        for stage in range(1, 8):
            with self.subTest(stage=stage):
                self.assertEqual({key for key in actual if key[0] == stage}, expected[stage])

    def test_dependencies_match_independent_expectation(self):
        design = _load_design()
        catalog = _build_catalog(design=design)
        jobs = list(iter_stress_input_jobs(catalog))
        manifest = build_perturbation_case_manifest(design=design)
        expected = _expected_stage_keys(manifest, catalog["contexts"])
        by_key = {_key(job): job for job in jobs}
        self.assertEqual({key for key in by_key if key[0] == 4}, expected[4])
        order = {job["job_id"]: index for index, job in enumerate(jobs)}
        for key, job in by_key.items():
            expected_deps = sorted(
                by_key[dep_key]["job_id"]
                for dep_key in _expected_dependency_keys(key, manifest, catalog["contexts"])
            )
            self.assertEqual(job["depends_on_job_ids"], expected_deps)
            self.assertEqual(job["depends_on_job_ids"], sorted(set(job["depends_on_job_ids"])))
            for dep in job["depends_on_job_ids"]:
                self.assertLess(order[dep], order[job["job_id"]])

    # -- case shape and sharing -----------------------------------------

    def test_shared_zero_case_appears_once(self):
        catalog = _build_catalog()
        cases = catalog["case_manifest"]["cases"]
        self.assertEqual(len(cases), 96)
        self.assertEqual(len([case for case in cases if case["case_id"] == CLEAN_CASE]), 1)
        self.assertEqual(len(_shared_case_descriptors(catalog["case_manifest"])), 56)
        self.assertEqual(len(_gaussian_case_descriptors(catalog["case_manifest"])), 40)

    def test_shared_and_gaussian_raw_cases_are_split_correctly(self):
        catalog = _build_catalog()
        jobs = list(iter_stress_input_jobs(catalog))
        raw = [job for job in jobs if job["stage"] == "raw_case"]
        test_uids = _distinct_test_uids(catalog["contexts"])
        shared = _shared_case_descriptors(catalog["case_manifest"])
        gaussian = _gaussian_case_descriptors(catalog["case_manifest"])

        for case in shared:
            for uid in test_uids:
                matches = [
                    job
                    for job in raw
                    if job["case_id"] == case["case_id"] and job["observation_uid"] == uid
                ]
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0]["context_id"], NA)

        for case in gaussian:
            self.assertFalse(
                any(job["context_id"] == NA for job in raw if job["case_id"] == case["case_id"])
            )
            for context in catalog["contexts"]:
                for uid in context["test_uids"]:
                    matches = [
                        job
                        for job in raw
                        if job["case_id"] == case["case_id"]
                        and job["observation_uid"] == uid
                        and job["context_id"] == context["context_id"]
                    ]
                    self.assertEqual(len(matches), 1)

    # -- per-stage semantics ---------------------------------------------

    def test_source_noise_reference_is_one_per_context(self):
        catalog = _build_catalog()
        jobs = list(iter_stress_input_jobs(catalog))
        noise = [job for job in jobs if job["stage"] == "source_noise_reference"]
        self.assertEqual(len(noise), len(catalog["contexts"]))
        self.assertEqual(
            {job["context_id"] for job in noise},
            {context["context_id"] for context in catalog["contexts"]},
        )
        for job in noise:
            self.assertEqual(job["observation_uid"], NA)
            self.assertEqual(job["case_id"], NA)
            self.assertEqual(job["representation_id"], NA)
            self.assertEqual(job["depends_on_job_ids"], [])

    def test_stochastic_realization_fields_and_replicates(self):
        catalog = _build_catalog()
        jobs = list(iter_stress_input_jobs(catalog))
        test_uids = _distinct_test_uids(catalog["contexts"])
        stochastic = [job for job in jobs if job["stage"] == "stochastic_realization"]
        self.assertEqual(len(stochastic), len(test_uids) * 2 * _STOCHASTIC_REPLICATES)
        counts = Counter((job["observation_uid"], job["family"]) for job in stochastic)
        for uid in test_uids:
            for family in ("gaussian", "impulse"):
                self.assertEqual(counts[(uid, family)], _STOCHASTIC_REPLICATES)
        for job in stochastic:
            self.assertEqual(job["context_id"], NA)
            self.assertEqual(job["case_id"], NA)
            self.assertEqual(job["representation_id"], NA)
            self.assertEqual(job["depends_on_job_ids"], [])

    def test_zero_input_parity_fields(self):
        catalog = _build_catalog()
        jobs = list(iter_stress_input_jobs(catalog))
        test_uids = _distinct_test_uids(catalog["contexts"])
        parity = [job for job in jobs if job["stage"] == "zero_input_parity"]
        self.assertEqual(len(parity), len(test_uids) * _ACTION_COUNT)
        for job in parity:
            self.assertEqual(job["case_id"], CLEAN_CASE)
            self.assertEqual(job["family"], "clean")
            self.assertEqual(job["context_id"], NA)
            self.assertIsNone(job["replicate_index"])

    def test_context_action_assembly_shape(self):
        catalog = _build_catalog()
        jobs = list(iter_stress_input_jobs(catalog))
        assembly = [job for job in jobs if job["stage"] == "context_action_assembly"]
        self.assertEqual(len(assembly), len(catalog["contexts"]) * 96 * _ACTION_COUNT)
        for context in catalog["contexts"]:
            per_context = [job for job in assembly if job["context_id"] == context["context_id"]]
            self.assertEqual(len(per_context), 96 * _ACTION_COUNT)
            for job in per_context:
                self.assertEqual(job["observation_uid"], NA)

    def test_no_future_payload_or_operational_fields(self):
        catalog = _build_catalog()
        jobs = list(iter_stress_input_jobs(catalog))
        banned = (
            "seed",
            "draw",
            "threshold",
            "array",
            "vector",
            "score",
            "status",
            "file",
            "path",
        )
        for job in jobs:
            for name in job:
                lowered = name.lower()
                for fragment in banned:
                    self.assertNotIn(fragment, lowered)
            for value in job.values():
                self.assertNotIsInstance(value, float)
            self.assertTrue(all(isinstance(item, str) for item in job["depends_on_job_ids"]))

    # -- purity, snapshots and determinism --------------------------------

    def test_inputs_are_not_mutated_and_nested_objects_are_independent(self):
        design = _load_design()
        binding = _make_binding(design)
        contexts = _make_contexts()
        design_snapshot = copy.deepcopy(design)
        binding_snapshot = copy.deepcopy(binding)
        contexts_snapshot = copy.deepcopy(contexts)

        catalog = build_stress_input_catalog(
            design=design, input_binding=binding, contexts=contexts
        )

        self.assertEqual(design, design_snapshot)
        self.assertEqual(binding, binding_snapshot)
        self.assertEqual(contexts, contexts_snapshot)
        self.assertIsNot(catalog["design"], design)
        self.assertIsNot(catalog["input_binding"], binding)
        self.assertIsNot(catalog["contexts"], contexts)

        catalog["design"]["schema_version"] = "mutated"
        catalog["contexts"][0]["context_id"] = "mutated"
        catalog["case_manifest"]["cases"].clear()
        self.assertEqual(design, design_snapshot)
        self.assertEqual(contexts, contexts_snapshot)

    def test_iterator_snapshot_is_immune_to_later_caller_mutation(self):
        design = _load_design()
        binding = _make_binding(design)
        contexts = _make_contexts()
        catalog = build_stress_input_catalog(
            design=design, input_binding=binding, contexts=contexts
        )
        iterator = iter_stress_input_jobs(catalog)
        expected = list(iter_stress_input_jobs(catalog))
        self.assertEqual(len(expected), 1962)

        catalog["summary"]["total_job_count"] = -1
        catalog["contexts"][0]["context_id"] = "MUTATED"
        catalog["contexts"][0]["test_uids"] = []
        catalog["input_binding"]["ordered_population_uids"] = []
        catalog["input_binding"]["actions"]["R_SG_400_1800"] = _hex("mutated.action")
        contexts[0]["context_id"] = "MUTATED"
        contexts[0]["test_uids"] = []
        binding["ordered_population_uids"] = []
        binding["actions"]["R_SG_400_1800"] = _hex("mutated.action")

        self.assertEqual(list(iterator), expected)

    def test_iterator_validation_is_eager(self):
        catalog = _build_catalog()
        forged = copy.deepcopy(catalog)
        forged["schema_version"] = "nato-sers-p08-forged"
        _resign(forged)
        with self.assertRaises(ValueError):
            iter_stress_input_jobs(forged)

    def test_context_order_does_not_change_canonical_output(self):
        forward = _build_catalog(contexts=_make_contexts())
        reversed_contexts = list(reversed(_make_contexts()))
        backward = _build_catalog(contexts=reversed_contexts)
        self.assertEqual(forward, backward)
        self.assertEqual(
            [context["context_id"] for context in backward["contexts"]],
            ["C1", "C2"],
        )

    def test_repeated_inputs_produce_identical_catalog(self):
        first = _build_catalog()
        second = _build_catalog()
        self.assertEqual(first, second)

    # -- forged catalogs -------------------------------------------------

    def test_repaired_hash_forgeries_are_rejected_by_iter(self):
        catalog = _build_catalog()
        forgeries = []

        forged = copy.deepcopy(catalog)
        forged["summary"]["total_job_count"] = 0
        forgeries.append(("summary", forged))

        forged = copy.deepcopy(catalog)
        forged["case_manifest"]["cases"] = []
        forgeries.append(("case_manifest", forged))

        forged = copy.deepcopy(catalog)
        forged["execution_authorized"] = True
        forgeries.append(("execution_authorized", forged))

        forged = copy.deepcopy(catalog)
        forged["numerical_inputs_computed"] = True
        forgeries.append(("numerical_inputs_computed", forged))

        forged = copy.deepcopy(catalog)
        forged["binding_sha256"] = _hex("forged.binding")
        forgeries.append(("binding_sha256", forged))

        forged = copy.deepcopy(catalog)
        forged["schema_version"] = "nato-sers-p08-forged"
        forgeries.append(("schema_version", forged))

        forged = copy.deepcopy(catalog)
        forged["unexpected_root_key"] = "surprise"
        forgeries.append(("unexpected_root_key", forged))

        for name, forged in forgeries:
            with self.subTest(forgery=name):
                _resign(forged)
                with self.assertRaises(ValueError):
                    iter_stress_input_jobs(forged)

    # -- malformed binding -----------------------------------------------

    def test_malformed_binding_is_rejected(self):
        design = _load_design()
        base = _make_binding(design)
        variants = []

        missing = copy.deepcopy(base)
        missing.pop("roles_file_sha256")
        variants.append(("missing_key", missing))

        extra = copy.deepcopy(base)
        extra["unexpected"] = "x"
        variants.append(("extra_key", extra))

        bad_hash = copy.deepcopy(base)
        bad_hash["raw_archive_sha256"] = "XYZ"
        variants.append(("bad_hash", bad_hash))

        pin_mismatch = copy.deepcopy(base)
        pin_mismatch["P01_contract_sha256"] = _hex("wrong.pin")
        variants.append(("pin_mismatch", pin_mismatch))

        missing_action = copy.deepcopy(base)
        del missing_action["actions"]["R_SG_400_1800"]
        variants.append(("missing_action", missing_action))

        extra_action = copy.deepcopy(base)
        extra_action["actions"]["R_UNKNOWN"] = _hex("action.unknown")
        variants.append(("extra_action", extra_action))

        bad_action_hash = copy.deepcopy(base)
        bad_action_hash["actions"]["R_SG_400_1800"] = "not-a-hash"
        variants.append(("bad_action_hash", bad_action_hash))

        axis_wrong = copy.deepcopy(base)
        axis_wrong["starting_axis_cm1"] = [400, 1850, 1]
        variants.append(("axis_wrong", axis_wrong))

        duplicate_population = copy.deepcopy(base)
        duplicate_population["ordered_population_uids"] = [
            "a",
            "a",
            "b",
            "c",
            "d",
            "e",
        ]
        variants.append(("duplicate_population", duplicate_population))

        empty_uid = copy.deepcopy(base)
        empty_uid["ordered_population_uids"] = ["a", "b", "c", "d", "e", ""]
        variants.append(("empty_population_uid", empty_uid))

        bool_axis = copy.deepcopy(base)
        bool_axis["output_axis_cm1"] = [400, 1800, True]
        variants.append(("bool_axis", bool_axis))

        nan_axis = copy.deepcopy(base)
        nan_axis["output_axis_cm1"] = [400, 1800, float("nan")]
        variants.append(("nan_axis", nan_axis))

        for name, binding in variants:
            with self.subTest(binding=name):
                self._assert_rejected(design=design, binding=binding)

    # -- raw id mapping --------------------------------------------------

    def test_raw_mapping_length_duplicate_and_digest_rules(self):
        design = _load_design()
        base = _make_binding(design)

        short = copy.deepcopy(base)
        short["ordered_raw_uids"] = RAW_UIDS[:-1]
        short["raw_row_order_sha256"] = canonical_sha256(short["ordered_raw_uids"])
        self._assert_rejected(design=design, binding=short)

        duplicate = copy.deepcopy(base)
        duplicate["ordered_raw_uids"] = [
            "raw-a",
            "raw-a",
            "raw-c",
            "raw-d",
            "raw-e",
            "raw-f",
        ]
        duplicate["raw_row_order_sha256"] = canonical_sha256(duplicate["ordered_raw_uids"])
        self._assert_rejected(design=design, binding=duplicate)

        digest = copy.deepcopy(base)
        digest["raw_row_order_sha256"] = _hex("wrong.raw.row")
        self._assert_rejected(design=design, binding=digest)

        legacy = copy.deepcopy(base)
        legacy["raw_row_order_sha256"] = hashlib.sha256(
            "\n".join(RAW_UIDS).encode("utf-8")
        ).hexdigest()
        self._assert_rejected(design=design, binding=legacy)

        legacy_actions = copy.deepcopy(base)
        legacy_actions["frozen_action_row_order_sha256"] = canonical_sha256(POPULATION_UIDS)
        self._assert_rejected(design=design, binding=legacy_actions)

    def test_distinct_raw_and_logical_ids_are_accepted_and_bound(self):
        design = _load_design()
        base = _make_binding(design)
        baseline = _build_catalog(design=design, binding=base)

        alternate_raw = [f"raw-alt-{index}" for index in range(len(POPULATION_UIDS))]
        alternate = copy.deepcopy(base)
        alternate["ordered_raw_uids"] = alternate_raw
        alternate["raw_row_order_sha256"] = canonical_sha256(alternate_raw)

        self.assertNotEqual(alternate_raw, POPULATION_UIDS)
        catalog = _build_catalog(design=design, binding=alternate)
        self.assertNotEqual(
            catalog["input_binding"]["ordered_raw_uids"],
            catalog["input_binding"]["ordered_population_uids"],
        )
        self.assertNotEqual(catalog["binding_sha256"], baseline["binding_sha256"])
        self.assertNotEqual(catalog["catalog_sha256"], baseline["catalog_sha256"])
        self.assertEqual(
            catalog["input_binding"]["raw_row_order_sha256"],
            canonical_sha256(alternate_raw),
        )
        for field in (
            "execution_authorized",
            "artifact_provenance_independently_verified",
            "numerical_inputs_computed",
            "full_stress_job_ledger_complete",
        ):
            self.assertIs(catalog[field], False)
        self.assertEqual(catalog["scientific_operations"], 0)
        self.assertEqual(catalog["summary"], baseline["summary"])

    # -- malformed contexts ----------------------------------------------

    def test_malformed_contexts_are_rejected(self):
        design = _load_design()
        base = _make_contexts()

        empty_fit = copy.deepcopy(base)
        empty_fit[0]["fit_uids"] = []
        empty_fit[0] = _refresh_context_hashes(empty_fit[0])
        self._assert_rejected(design=design, contexts=empty_fit)

        empty_test = copy.deepcopy(base)
        empty_test[0]["test_uids"] = []
        empty_test[0] = _refresh_context_hashes(empty_test[0])
        self._assert_rejected(design=design, contexts=empty_test)

        overlap = copy.deepcopy(base)
        overlap[0]["test_uids"] = ["a", "d"]
        overlap[0] = _refresh_context_hashes(overlap[0])
        self._assert_rejected(design=design, contexts=overlap)

        unknown_role = copy.deepcopy(base)
        unknown_role[0]["fit_uids"] = ["a", "z"]
        unknown_role[0] = _refresh_context_hashes(unknown_role[0])
        self._assert_rejected(design=design, contexts=unknown_role)

        duplicate_context = copy.deepcopy(base)
        duplicate_context[1] = copy.deepcopy(base[0])
        self._assert_rejected(design=design, contexts=duplicate_context)

        unsorted_uids = copy.deepcopy(base)
        unsorted_uids[0]["test_uids"] = ["d", "c"]
        self._assert_rejected(design=design, contexts=unsorted_uids)

        reserved_id = copy.deepcopy(base)
        reserved_id[0]["context_id"] = NA
        reserved_id[0] = _refresh_context_hashes(reserved_id[0])
        self._assert_rejected(design=design, contexts=reserved_id)

        whitespace_id = copy.deepcopy(base)
        whitespace_id[0]["context_id"] = " C1 "
        whitespace_id[0] = _refresh_context_hashes(whitespace_id[0])
        self._assert_rejected(design=design, contexts=whitespace_id)

        whitespace_uid = copy.deepcopy(base)
        whitespace_uid[0]["fit_uids"] = [" a", "b"]
        whitespace_uid[0] = _refresh_context_hashes(whitespace_uid[0])
        self._assert_rejected(design=design, contexts=whitespace_uid)

        surrogate_id = copy.deepcopy(base)
        surrogate_id[0]["context_id"] = "\ud800"
        surrogate_id[0] = _refresh_context_hashes(surrogate_id[0])
        self._assert_rejected(design=design, contexts=surrogate_id)

    # -- malformed design / scientific gate ------------------------------

    def test_invalid_design_contract_is_rejected(self):
        base_design = _load_design()

        wrong_panel = copy.deepcopy(base_design)
        wrong_panel["selected_universal_panel"] = ["C-RBF-SVM"]
        self._assert_rejected(design=wrong_panel, binding=_make_binding(wrong_panel))

        wrong_mode = copy.deepcopy(base_design)
        wrong_mode["selected_qc_stress_mode"] = "something_else"
        self._assert_rejected(design=wrong_mode, binding=_make_binding(wrong_mode))

        native_reaction = copy.deepcopy(base_design)
        native_reaction["native_grid_gate_reaction_experiment_included"] = True
        self._assert_rejected(design=native_reaction, binding=_make_binding(native_reaction))

        unauthorized_design = copy.deepcopy(base_design)
        unauthorized_design["execution_authorized"] = True
        self._assert_rejected(
            design=unauthorized_design, binding=_make_binding(unauthorized_design)
        )

        disagreeing_binding = _make_binding(base_design)
        disagreeing_binding["P01_transform_source_sha256"] = _hex("wrong.transform")
        self._assert_rejected(design=base_design, binding=disagreeing_binding)

    def test_require_scientific_execution_is_always_refused(self):
        for args in ((), ("x",), ("x", "y")):
            with self.subTest(args=args):
                with self.assertRaises(ValueError) as caught:
                    require_scientific_execution(*args)
                self.assertEqual(str(caught.exception), "scientific_execution_not_authorized")
        with self.assertRaises(ValueError) as caught:
            require_scientific_execution(design={"execution_authorized": True}, force=True)
        self.assertEqual(str(caught.exception), "scientific_execution_not_authorized")

    # -- numeric type substitution ---------------------------------------

    def test_numeric_type_substitution_cannot_exploit_python_equality(self):
        catalog = _build_catalog()

        stale = copy.deepcopy(catalog)
        self.assertIs(type(stale["summary"]["context_count"]), int)
        stale["summary"]["context_count"] = 2.0
        self.assertEqual(stale["summary"]["context_count"], 2)
        self.assertNotEqual(
            canonical_sha256({k: v for k, v in stale.items() if k != "catalog_sha256"}),
            stale["catalog_sha256"],
        )
        with self.assertRaises(ValueError):
            iter_stress_input_jobs(stale)

        forged = copy.deepcopy(catalog)
        self.assertEqual(forged["scientific_operations"], 0)
        forged["scientific_operations"] = 0.0
        self.assertEqual(forged["scientific_operations"], 0)
        self.assertIs(type(forged["scientific_operations"]), float)
        _resign(forged)
        with self.assertRaises(ValueError):
            iter_stress_input_jobs(forged)

    # -- tuple-typed inputs ----------------------------------------------

    def test_tuple_typed_inputs_are_rejected(self):
        design = _load_design()
        base = _make_binding(design)

        tuple_axis = copy.deepcopy(base)
        tuple_axis["starting_axis_cm1"] = (400, 1849, 1)
        self._assert_rejected(design=design, binding=tuple_axis)

        tuple_raw = copy.deepcopy(base)
        tuple_raw["ordered_raw_uids"] = tuple(RAW_UIDS)
        tuple_raw["raw_row_order_sha256"] = canonical_sha256(RAW_UIDS)
        self._assert_rejected(design=design, binding=tuple_raw)

        self._assert_rejected(design=design, contexts=tuple(_make_contexts()))

        tuple_design = copy.deepcopy(design)
        tuple_design["unexpected_input_axes"] = (400, 1849, 1)
        self._assert_rejected(design=tuple_design, binding=_make_binding(tuple_design))

    def test_jointly_changed_design_and_binding_axes_are_rejected(self):
        base_design = _load_design()
        changed_design = copy.deepcopy(base_design)
        changed_design["input"]["starting_axis_cm1"] = [399, 1849, 1]
        changed_design["input"]["output_axis_cm1"] = [399, 1849, 1]
        changed_binding = _make_binding(changed_design)
        changed_binding["starting_axis_cm1"] = [399, 1849, 1]
        changed_binding["output_axis_cm1"] = [399, 1849, 1]
        self.assertEqual(
            changed_design["input"]["starting_axis_cm1"],
            changed_binding["starting_axis_cm1"],
        )
        self.assertEqual(
            changed_design["input"]["output_axis_cm1"],
            changed_binding["output_axis_cm1"],
        )
        self._assert_rejected(design=changed_design, binding=changed_binding)


if __name__ == "__main__":
    unittest.main()
