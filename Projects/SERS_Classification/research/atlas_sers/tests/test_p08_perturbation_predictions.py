"""Synthetic, metadata-only tests for the P08 stress prediction join and graph.

Only invented identities, the public perturbation design contract and the
public synthetic builders (QC catalog, universal plan/procedure adapter and
stress input catalog) are used.  Nothing here authenticates provenance, reads
private data, writes files, fits a model, draws a realization, selects a QC
route or executes any science.

The two modules under test are authored separately:

* ``atlas_sers.evaluation.p08_perturbation_join.bind_stress_prediction_inputs``
* ``atlas_sers.evaluation.p08_perturbation_predictions``

The graph tests assert the exact descriptor wiring declared by the contract,
including the conservative QC row-receipt semantics.  They deliberately do not
assert numerical parity, which remains a separate validation responsibility.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from collections import defaultdict

from atlas_sers.evaluation.p08_perturbation_inputs import (
    build_stress_input_catalog,
    iter_stress_input_jobs,
)
from atlas_sers.evaluation.p08_perturbation_join import (
    bind_stress_prediction_inputs,
)
from atlas_sers.evaluation.p08_perturbation_predictions import (  # noqa: PLC2701
    _make_job,
    build_stress_prediction_catalog,
    iter_stress_prediction_records,
    require_scientific_execution,
)
from atlas_sers.evaluation.p08_perturbation_procedures import (
    build_universal_procedure_records,
)
from atlas_sers.evaluation.p08_plan import build_universal_plan
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256
from tests.test_p08_perturbation_inputs import (  # noqa: PLC2701
    _load_design,
    _make_binding,
)
from tests.test_p08_perturbation_procedures import build_fixture_bridge
from tests.test_p08_qc_catalog import build, build_fixture

# ---------------------------------------------------------------------------
# Literal contract constants for the bounded toy fixture.
# ---------------------------------------------------------------------------

NA = "not_applicable"
CLEAN_CASE = "P08-STRESS-CLEAN"

MIN_POLICY = "PP-U-MIN"
FUTURE_POLICIES = ("PP-U-SG", "PP-U-ARPLS")
POLICIES = (MIN_POLICY,) + FUTURE_POLICIES
FAMILY_POLICY = "PP-FAMILY-SRC"
QC_POLICY = "PP-QC-SRC"

SVM_MODEL = "C-RBF-SVM"
SVM_SEED = "deterministic"
CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
NEURAL_MODELS = ("D0-M", "D1", "D2", "D3")
UNIVERSAL_STRATEGIES = (
    "C-RBF-SVM",
    "C-RANDOM-FOREST",
    "C-EXTRA-TREES",
    "D0-M",
    "P05-SELECTED",
)
FAMILY_STRATEGIES = ("C-RBF-SVM", "C-RANDOM-FOREST", "D0-M", "P05-SELECTED")

BUNDLE_SCHEMA = "nato-sers-p08-stress-prediction-binding-v1"
CATALOG_SCHEMA = "nato-sers-p08-stress-prediction-catalog-v1"

STAGE_ORDER = (
    "model_reconstruction",
    "retained_model_authentication",
    "calibrator_authentication",
    "qc_clean_route_authentication",
    "qc_mixed_input_assembly",
    "raw_prediction",
    "classical_seed_average",
    "temperature_apply",
    "neural_seed_average",
    "prediction_units",
    "clean_probability_parity",
)

EXPECTED_STAGE_COUNTS = {
    "model_reconstruction": 14,
    "retained_model_authentication": 65,
    "calibrator_authentication": 53,
    "qc_clean_route_authentication": 1,
    "qc_mixed_input_assembly": 96,
    "raw_prediction": 7584,
    "classical_seed_average": 1920,
    "temperature_apply": 5088,
    "neural_seed_average": 1056,
    "prediction_units": 2976,
    "clean_probability_parity": 31,
}

EXPECTED_ALIAS_COUNTS = {
    "universal": 2880,
    "qc_fixed_route": 384,
    "qc_minimal_fallback": 384,
    "family_minimal_fallback": 768,
}

EXPECTED_PROCEDURES = 31
EXPECTED_SEED_ESTIMATORS = 79
EXPECTED_NEURAL_PROCEDURES = 11
EXPECTED_CLASSICAL_PROCEDURES = 20
EXPECTED_CASES = 96
EXPECTED_PREDICTION_JOBS = 18884
EXPECTED_REPORTING_ALIASES = 4416
EXPECTED_HISTORICAL_FITS = 14
EXPECTED_ELIGIBLE_QC = 1

MODEL_RECONSTRUCTION_RESOLUTION = "fixed_saved_selection_source_rows_seed_no_retuning"
RETAINED_RESOLUTION = "retained_upstream_estimator_no_new_fit"
CALIBRATOR_RESOLUTION = "source_fitted_temperature_no_new_fit"
QC_ROUTE_RESOLUTION = "fixed_native_clean_route_not_gate_reaction"
RAW_PREDICTION_RESOLUTION = "uncalibrated_same_model_case_scores"
CLASSICAL_AVERAGE_RESOLUTION = "average_uncalibrated_seed_scores"
NEURAL_AVERAGE_RESOLUTION = "average_calibrated_seed_probabilities"
CLASSICAL_TEMPERATURE_RESOLUTION = "seed_average_then_single_temperature_logclip1e_7"
NEURAL_TEMPERATURE_RESOLUTION = "same_seed_temperature_before_ensemble"
PREDICTION_UNIT_RESOLUTION = (
    "M01_spectrum_and_M06_master_instrument_probability_combination_no_repeat_ensemble"
)
CLEAN_PARITY_RESOLUTION = "absolute_1e_7_relative_0_exact_classes_M01_M06_and_retained_seed_values"
QC_MIXED_RESOLUTION = (
    "candidate_row_receipts_select_clean_route_action_or_invalid_action_MIN_input_"
    "require_selected_clean_parity_unselected_invalid_not_fatal_invalid_MIN_fatal"
)

BUNDLE_KEYS = {
    "schema_version",
    "execution_authorized",
    "scientific_operations",
    "artifact_provenance_independently_verified",
    "input_catalog",
    "universal_procedures",
    "qc_catalog",
    "minimal_bridge",
    "qc_procedures",
    "procedures",
    "family_aliases",
    "binding_sha256",
    "bundle_sha256",
}

CATALOG_KEYS = {
    "schema_version",
    "execution_authorized",
    "scientific_operations",
    "artifact_provenance_independently_verified",
    "scientific_predictions_computed",
    "clean_probability_parity_accepted",
    "full_stress_job_ledger_complete",
    "bundle",
    "summary",
    "catalog_sha256",
}

SUMMARY_KEYS = {
    "case_count",
    "procedure_count",
    "seed_estimator_count",
    "historical_reconstruction_fit_count",
    "new_calibration_fit_count",
    "eligible_qc_context_count",
    "stage_counts",
    "prediction_job_count",
    "reporting_alias_count",
    "alias_counts_by_mode",
}

JOB_KEYS = {
    "record_type",
    "binding_sha256",
    "stage",
    "procedure_id",
    "context_id",
    "policy_id",
    "model_id",
    "case_id",
    "seed",
    "depends_on_job_ids",
    "depends_on_input_job_ids",
    "upstream_references",
    "resolution",
    "job_id",
}

ALIAS_KEYS = {
    "record_type",
    "binding_sha256",
    "policy_id",
    "context_id",
    "strategy",
    "recipe_id",
    "case_id",
    "target_procedure_id",
    "target_prediction_units_job_id",
    "mode",
    "upstream_alias_id",
    "alias_id",
}

FAMILY_ALIAS_KEYS = {
    "alias_id",
    "policy_id",
    "context_id",
    "strategy",
    "recipe_id",
    "fit_uid_sha256",
    "test_uid_sha256",
    "array_sha256",
    "representation_id",
    "model_spec_sha256",
    "target_job_id",
    "target_binding_sha256",
    "new_fits",
    "new_predictions",
    "training_and_test_pipeline",
    "reason",
    "target_evidence",
}

INPUT_DEPENDENCY_STAGES = (
    "action_transform",
    "zero_input_parity",
    "context_action_assembly",
)


def independent_hash(value):
    """A stdlib strict canonical JSON SHA-256, independent of production code."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _family_alias(context_id, strategy, recipe_id, record):
    endpoint = record["clean_endpoint_reference"]
    body = {
        "policy_id": FAMILY_POLICY,
        "context_id": context_id,
        "strategy": strategy,
        "recipe_id": recipe_id,
        "fit_uid_sha256": record["fit_uid_sha256"],
        "test_uid_sha256": record["test_uid_sha256"],
        "array_sha256": record["array_sha256"],
        "representation_id": record["representation_id"],
        "model_spec_sha256": record["model_spec_sha256"],
        "target_job_id": endpoint["job_id"],
        "target_binding_sha256": endpoint["historical_binding_sha256"],
        "new_fits": 0,
        "new_predictions": 0,
        "training_and_test_pipeline": "complete_minimal_pipeline",
        "reason": "synthetic_unsupported_family",
        "target_evidence": [{"synthetic_reference": "invented-only"}],
    }
    body["alias_id"] = "SYNTH-FAMILY-" + independent_hash(body)
    return body


def build_input_index(input_catalog):
    index = {}
    for job in iter_stress_input_jobs(input_catalog):
        if job["stage"] in INPUT_DEPENDENCY_STAGES:
            index[
                (
                    job["stage"],
                    job["context_id"],
                    job["observation_uid"],
                    job["case_id"],
                    job["representation_id"],
                )
            ] = job["job_id"]
    return index


class Scenario:
    """One frozen bounded toy fixture plus its cached derived artifacts."""

    def __init__(self, entries=None):
        if entries is None:
            entries = [
                {"kind": "pseudo", "context_id": "CTX-Q", "recipe": "D3"},
                {"kind": "master", "context_id": "CTX-F"},
            ]
        self.fixture = build_fixture(entries)
        self.plan = build_universal_plan(
            self.fixture["universal_contexts"],
            self.fixture["candidates"],
            self.fixture["actions"],
            self.fixture["model_spec_sha256"],
        )
        self.bridge = build_fixture_bridge(self.plan)
        universal0 = build_universal_procedure_records(plan=self.plan, minimal_bridge=self.bridge)
        self.family_aliases = self._make_family_aliases(universal0)
        self.bridge["fallback_endpoint_aliases"] = self.family_aliases
        self.universal = build_universal_procedure_records(
            plan=self.plan, minimal_bridge=self.bridge
        )
        self.qc_catalog = build(self.fixture)
        self.input_catalog = self._make_input_catalog()
        self._bundle = None
        self._catalog = None
        self._records = None
        self._input_index = None
        self._input_ids = None

    # -- construction helpers -------------------------------------------

    def _make_family_aliases(self, universal):
        aliases = []
        for raw in self.fixture["contexts"]:
            context_id = raw["context_id"]
            for strategy in FAMILY_STRATEGIES:
                model_id = self._min_model_for(universal, context_id, strategy)
                record = self._min_record(universal, context_id, model_id)
                aliases.append(_family_alias(context_id, strategy, model_id, record))
        return aliases

    @staticmethod
    def _min_model_for(universal, context_id, strategy):
        if strategy in CLASSICAL_MODELS:
            return strategy
        alias = next(
            item
            for item in universal["strategy_aliases"]
            if item["policy_id"] == MIN_POLICY
            and item["context_id"] == context_id
            and item["strategy"] == strategy
        )
        return alias["recipe_id"]

    @staticmethod
    def _min_record(universal, context_id, model_id):
        return next(
            record
            for record in universal["records"]
            if record["policy_id"] == MIN_POLICY
            and record["context_id"] == context_id
            and record["model_id"] == model_id
        )

    def _make_input_catalog(self):
        design = _load_design()
        contexts = []
        ordered = []
        seen = set()
        for raw in self.fixture["contexts"]:
            fit = sorted(raw["outer_fit_uids"])
            test = sorted(raw["outer_test_uids"])
            contexts.append(
                {
                    "context_id": raw["context_id"],
                    "fit_uids": fit,
                    "test_uids": test,
                    "fit_uid_sha256": canonical_sha256(fit),
                    "test_uid_sha256": canonical_sha256(test),
                }
            )
            for uid in fit + test:
                if uid not in seen:
                    seen.add(uid)
                    ordered.append(uid)
        raw_uids = [f"raw-{index:04d}" for index in range(len(ordered))]
        binding = _make_binding(design, population_uids=ordered, raw_uids=raw_uids)
        binding["actions"] = dict(self.fixture["actions"])
        return build_stress_input_catalog(design=design, input_binding=binding, contexts=contexts)

    # -- public surface --------------------------------------------------

    def args(self):
        return {
            "input_catalog": copy.deepcopy(self.input_catalog),
            "universal_procedures": copy.deepcopy(self.universal),
            "qc_catalog": copy.deepcopy(self.qc_catalog),
            "minimal_bridge": copy.deepcopy(self.bridge),
        }

    def case_ids(self):
        return [case["case_id"] for case in self.input_catalog["case_manifest"]["cases"]]

    def case_families(self):
        return {
            case["case_id"]: case["family"] for case in self.input_catalog["case_manifest"]["cases"]
        }

    @property
    def bundle(self):
        if self._bundle is None:
            self._bundle = bind_stress_prediction_inputs(**self.args())
        return self._bundle

    @property
    def catalog(self):
        if self._catalog is None:
            self._catalog = build_stress_prediction_catalog(**self.args())
        return self._catalog

    @property
    def records(self):
        if self._records is None:
            self._records = list(iter_stress_prediction_records(self.catalog))
        return self._records

    @property
    def input_index(self):
        if self._input_index is None:
            self._input_index = build_input_index(self.input_catalog)
        return self._input_index

    @property
    def input_ids(self):
        if self._input_ids is None:
            self._input_ids = {job["job_id"] for job in iter_stress_input_jobs(self.input_catalog)}
        return self._input_ids


_SCENARIOS: dict = {}


def scenario(entries=None):
    key = "__default__" if entries is None else json.dumps(entries, sort_keys=True)
    if key not in _SCENARIOS:
        _SCENARIOS[key] = Scenario(entries)
    return _SCENARIOS[key]


def rehash(record, key="catalog_sha256"):
    record[key] = independent_hash({name: value for name, value in record.items() if name != key})
    return record


def rebind_bridge(args):
    args["universal_procedures"]["minimal_bridge_metadata_sha256"] = independent_hash(
        args["minimal_bridge"]
    )
    rehash(args["universal_procedures"])
    return args


def _refresh_procedure_identity(universal, record, previous_id):
    """Recompute one procedure id and retarget aliases so only the body error remains."""

    prefix = previous_id.rsplit("-", 1)[0] + "-"
    body = {name: value for name, value in record.items() if name != "procedure_id"}
    record["procedure_id"] = prefix + independent_hash(body)
    for alias in universal["strategy_aliases"]:
        if alias["target_procedure_id"] == previous_id:
            alias["target_procedure_id"] = record["procedure_id"]


def with_universal_record_mutation(args, policy_id, model_id, mutate_record):
    """Mutate one universal record after refreshing its canonical identity graph."""

    universal = args["universal_procedures"]
    record = next(
        item
        for item in universal["records"]
        if item["policy_id"] == policy_id and item["model_id"] == model_id
    )
    previous_id = record["procedure_id"]
    mutate_record(record)
    _refresh_procedure_identity(universal, record, previous_id)
    rehash(universal)
    return args


class _Base(unittest.TestCase):
    @staticmethod
    def split(records):
        jobs = [record for record in records if record["record_type"] == "job"]
        aliases = [record for record in records if record["record_type"] == "alias"]
        return jobs, aliases

    @staticmethod
    def by_stage(jobs):
        buckets = defaultdict(list)
        for job in jobs:
            buckets[job["stage"]].append(job)
        return buckets

    @staticmethod
    def procedures(sc):
        return {record["procedure_id"]: record for record in sc.bundle["procedures"]}

    @staticmethod
    def units_index(by_stage):
        return {(job["procedure_id"], job["case_id"]): job for job in by_stage["prediction_units"]}


# ---------------------------------------------------------------------------
# 1. Bundle / catalog identity, authority and literal counts.
# ---------------------------------------------------------------------------


class BundleCatalogTests(_Base):
    def test_bundle_identity_and_authority(self):
        sc = scenario()
        bundle = sc.bundle
        self.assertEqual(set(bundle), BUNDLE_KEYS)
        self.assertEqual(bundle["schema_version"], BUNDLE_SCHEMA)
        self.assertIs(bundle["execution_authorized"], False)
        self.assertEqual(bundle["scientific_operations"], 0)
        self.assertIs(bundle["artifact_provenance_independently_verified"], False)

        binding = {
            "namespace": "nato-sers-p08-stress-prediction-v1",
            "input_catalog_sha256": sc.input_catalog["catalog_sha256"],
            "universal_procedure_sha256": sc.universal["catalog_sha256"],
            "qc_catalog_sha256": sc.qc_catalog["catalog_sha256"],
            "qc_procedure_sha256": bundle["qc_procedures"]["catalog_sha256"],
            "minimal_bridge_sha256": independent_hash(sc.bridge),
        }
        self.assertEqual(bundle["binding_sha256"], independent_hash(binding))
        body = {k: v for k, v in bundle.items() if k != "bundle_sha256"}
        self.assertEqual(bundle["bundle_sha256"], independent_hash(body))

        self.assertEqual(len(bundle["family_aliases"]), 8)
        self.assertEqual(len(bundle["procedures"]), EXPECTED_PROCEDURES)
        for alias in bundle["family_aliases"]:
            self.assertEqual(set(alias), FAMILY_ALIAS_KEYS)

    def test_catalog_shape_and_literal_summary(self):
        sc = scenario()
        catalog = sc.catalog
        self.assertEqual(set(catalog), CATALOG_KEYS)
        self.assertEqual(catalog["schema_version"], CATALOG_SCHEMA)
        for flag in (
            "execution_authorized",
            "artifact_provenance_independently_verified",
            "scientific_predictions_computed",
            "clean_probability_parity_accepted",
            "full_stress_job_ledger_complete",
        ):
            self.assertIs(catalog[flag], False)
        self.assertEqual(catalog["scientific_operations"], 0)
        body = {k: v for k, v in catalog.items() if k != "catalog_sha256"}
        self.assertEqual(catalog["catalog_sha256"], independent_hash(body))
        self.assertEqual(catalog["bundle"]["binding_sha256"], sc.bundle["binding_sha256"])

        summary = catalog["summary"]
        self.assertEqual(set(summary), SUMMARY_KEYS)
        self.assertEqual(summary["case_count"], EXPECTED_CASES)
        self.assertEqual(summary["procedure_count"], EXPECTED_PROCEDURES)
        self.assertEqual(summary["seed_estimator_count"], EXPECTED_SEED_ESTIMATORS)
        self.assertEqual(summary["historical_reconstruction_fit_count"], EXPECTED_HISTORICAL_FITS)
        self.assertEqual(summary["new_calibration_fit_count"], 0)
        self.assertEqual(summary["eligible_qc_context_count"], EXPECTED_ELIGIBLE_QC)
        self.assertEqual(summary["stage_counts"], EXPECTED_STAGE_COUNTS)
        self.assertEqual(sum(summary["stage_counts"].values()), EXPECTED_PREDICTION_JOBS)
        self.assertEqual(summary["prediction_job_count"], EXPECTED_PREDICTION_JOBS)
        self.assertEqual(summary["reporting_alias_count"], EXPECTED_REPORTING_ALIASES)
        self.assertEqual(summary["alias_counts_by_mode"], EXPECTED_ALIAS_COUNTS)
        self.assertEqual(sum(summary["alias_counts_by_mode"].values()), EXPECTED_REPORTING_ALIASES)


# ---------------------------------------------------------------------------
# 2. Full expansion: identifiers, digests, ordering and input linkage.
# ---------------------------------------------------------------------------


class PredictionStructureTests(_Base):
    def test_expansion_structure_and_dependencies(self):
        sc = scenario()
        records = sc.records
        jobs, aliases = self.split(records)
        self.assertEqual(len(jobs), EXPECTED_PREDICTION_JOBS)
        self.assertEqual(len(aliases), EXPECTED_REPORTING_ALIASES)
        self.assertEqual(records[: len(jobs)], jobs)
        self.assertEqual(records[len(jobs) :], aliases)

        seen = set()
        for job in jobs:
            self.assertEqual(set(job), JOB_KEYS)
            self.assertEqual(job["record_type"], "job")
            self.assertEqual(job["binding_sha256"], sc.bundle["binding_sha256"])
            body = {k: v for k, v in job.items() if k != "job_id"}
            self.assertEqual(job["job_id"], "P08STRESSPRED-" + independent_hash(body))
            self.assertNotIn(job["job_id"], seen)
            seen.add(job["job_id"])
            self.assertEqual(job["depends_on_job_ids"], sorted(set(job["depends_on_job_ids"])))
            self.assertIsInstance(job["upstream_references"], dict)
        self.assertEqual(len(seen), EXPECTED_PREDICTION_JOBS)

        alias_ids = set()
        for alias in aliases:
            self.assertEqual(set(alias), ALIAS_KEYS)
            body = {k: v for k, v in alias.items() if k != "alias_id"}
            self.assertEqual(alias["alias_id"], "P08STRESSALIAS-" + independent_hash(body))
            self.assertNotIn(alias["alias_id"], alias_ids)
            alias_ids.add(alias["alias_id"])

        by_stage = self.by_stage(jobs)
        self.assertEqual(set(by_stage), set(STAGE_ORDER))
        for stage in STAGE_ORDER:
            self.assertEqual(len(by_stage[stage]), EXPECTED_STAGE_COUNTS[stage])

        order = {job["job_id"]: index for index, job in enumerate(jobs)}
        input_ids = sc.input_ids
        cases = set(sc.case_ids())
        for job in jobs:
            for dependency in job["depends_on_job_ids"]:
                self.assertIn(dependency, order)
                self.assertLess(order[dependency], order[job["job_id"]])
            for dependency in job["depends_on_input_job_ids"]:
                self.assertIn(dependency, input_ids)
            if job["case_id"] != NA:
                self.assertIn(job["case_id"], cases)


# ---------------------------------------------------------------------------
# 3. Raw prediction wiring.
# ---------------------------------------------------------------------------


class RawPredictionWiringTests(_Base):
    def test_raw_prediction_wiring(self):
        sc = scenario()
        jobs, _ = self.split(sc.records)
        by_stage = self.by_stage(jobs)
        procedures = self.procedures(sc)

        ready = {}
        for stage in ("model_reconstruction", "retained_model_authentication"):
            for job in by_stage[stage]:
                ready[(job["procedure_id"], job["seed"])] = job
        raw = {
            (job["procedure_id"], job["case_id"], job["seed"]): job
            for job in by_stage["raw_prediction"]
        }
        self.assertEqual(len(raw), EXPECTED_STAGE_COUNTS["raw_prediction"])
        parity = {job["procedure_id"]: job for job in by_stage["clean_probability_parity"]}
        mixed = {
            (job["context_id"], job["case_id"]): job for job in by_stage["qc_mixed_input_assembly"]
        }
        index = sc.input_index

        for (procedure_id, case, seed), job in raw.items():
            procedure = procedures[procedure_id]
            self.assertEqual(job["policy_id"], procedure["policy_id"])
            self.assertEqual(job["model_id"], procedure["model_id"])
            self.assertEqual(job["resolution"], RAW_PREDICTION_RESOLUTION)
            self.assertIn(ready[(procedure_id, seed)]["job_id"], job["depends_on_job_ids"])
            if case == CLEAN_CASE:
                self.assertNotIn(parity[procedure_id]["job_id"], job["depends_on_job_ids"])
            else:
                self.assertIn(parity[procedure_id]["job_id"], job["depends_on_job_ids"])
            if procedure["policy_id"] == QC_POLICY:
                self.assertEqual(job["depends_on_input_job_ids"], [])
                self.assertIn(
                    mixed[(procedure["context_id"], case)]["job_id"],
                    job["depends_on_job_ids"],
                )
            else:
                if procedure["context_id"] == "CTX-Q":
                    self.assertNotIn(
                        mixed[(procedure["context_id"], case)]["job_id"],
                        job["depends_on_job_ids"],
                    )
                expected = index[
                    (
                        "context_action_assembly",
                        procedure["context_id"],
                        NA,
                        case,
                        procedure["representation_id"],
                    )
                ]
                self.assertEqual(job["depends_on_input_job_ids"], [expected])


# ---------------------------------------------------------------------------
# 4. Calibration and averaging chains.
# ---------------------------------------------------------------------------


class CalibrationChainTests(_Base):
    def test_calibration_chain_wiring(self):
        sc = scenario()
        jobs, _ = self.split(sc.records)
        by_stage = self.by_stage(jobs)

        raw = {(j["procedure_id"], j["case_id"], j["seed"]): j for j in by_stage["raw_prediction"]}
        classical = {
            (j["procedure_id"], j["case_id"]): j for j in by_stage["classical_seed_average"]
        }
        temperature = {
            (j["procedure_id"], j["case_id"], j["seed"]): j for j in by_stage["temperature_apply"]
        }
        neural = {(j["procedure_id"], j["case_id"]): j for j in by_stage["neural_seed_average"]}
        units = self.units_index(by_stage)
        calibrator = {
            (j["procedure_id"], j["seed"]): j for j in by_stage["calibrator_authentication"]
        }
        calibrator_ref = {}
        for procedure in sc.bundle["procedures"]:
            for reference in procedure["calibration_references"]:
                calibrator_ref[(procedure["procedure_id"], reference["seed"])] = reference

        self.assertEqual(len(calibrator), EXPECTED_STAGE_COUNTS["calibrator_authentication"])
        self.assertEqual(len(classical), EXPECTED_CLASSICAL_PROCEDURES * EXPECTED_CASES)
        self.assertEqual(len(neural), EXPECTED_NEURAL_PROCEDURES * EXPECTED_CASES)

        for job in by_stage["calibrator_authentication"]:
            self.assertEqual(job["resolution"], CALIBRATOR_RESOLUTION)
            reference = calibrator_ref[(job["procedure_id"], job["seed"])]
            self.assertEqual(job["upstream_references"]["calibration"], reference)
        for job in by_stage["model_reconstruction"]:
            self.assertEqual(job["resolution"], MODEL_RECONSTRUCTION_RESOLUTION)
        for job in by_stage["retained_model_authentication"]:
            self.assertEqual(job["resolution"], RETAINED_RESOLUTION)
        self.assertFalse([stage for stage in by_stage if "calibration_fit" in stage])

        cases = sc.case_ids()
        for procedure in sc.bundle["procedures"]:
            procedure_id = procedure["procedure_id"]
            seeds = list(procedure["seeds"])
            neural_procedure = procedure["model_id"] in NEURAL_MODELS
            for case in cases:
                unit = units[(procedure_id, case)]
                self.assertEqual(unit["resolution"], PREDICTION_UNIT_RESOLUTION)
                if neural_procedure:
                    self.assertNotIn((procedure_id, case), classical)
                    averaged = neural[(procedure_id, case)]
                    self.assertEqual(averaged["resolution"], NEURAL_AVERAGE_RESOLUTION)
                    seed_temperatures = [temperature[(procedure_id, case, seed)] for seed in seeds]
                    self.assertEqual(
                        averaged["depends_on_job_ids"],
                        sorted(job["job_id"] for job in seed_temperatures),
                    )
                    self.assertEqual(unit["depends_on_job_ids"], [averaged["job_id"]])
                    for seed in seeds:
                        applied = temperature[(procedure_id, case, seed)]
                        self.assertEqual(applied["resolution"], NEURAL_TEMPERATURE_RESOLUTION)
                        self.assertIn(
                            raw[(procedure_id, case, seed)]["job_id"],
                            applied["depends_on_job_ids"],
                        )
                        self.assertIn(
                            calibrator[(procedure_id, seed)]["job_id"],
                            applied["depends_on_job_ids"],
                        )
                else:
                    self.assertNotIn((procedure_id, case), neural)
                    averaged = classical[(procedure_id, case)]
                    self.assertEqual(averaged["resolution"], CLASSICAL_AVERAGE_RESOLUTION)
                    self.assertEqual(
                        averaged["depends_on_job_ids"],
                        sorted(raw[(procedure_id, case, seed)]["job_id"] for seed in seeds),
                    )
                    applied = temperature[(procedure_id, case, NA)]
                    self.assertEqual(applied["resolution"], CLASSICAL_TEMPERATURE_RESOLUTION)
                    self.assertIn(averaged["job_id"], applied["depends_on_job_ids"])
                    self.assertIn(
                        calibrator[(procedure_id, NA)]["job_id"],
                        applied["depends_on_job_ids"],
                    )
                    self.assertEqual(unit["depends_on_job_ids"], [applied["job_id"]])

    def test_model_ready_upstream_refit_references(self):
        sc = scenario()
        jobs, _ = self.split(sc.records)
        by_stage = self.by_stage(jobs)
        procedures = self.procedures(sc)
        for stage in ("model_reconstruction", "retained_model_authentication"):
            for job in by_stage[stage]:
                procedure = procedures[job["procedure_id"]]
                reference = next(
                    ref for ref in procedure["refit_references"] if ref["seed"] == job["seed"]
                )
                self.assertEqual(job["upstream_references"]["refit"], reference)


class PrivateHelperContractTests(_Base):
    def test_make_job_deepcopies_nested_upstream_references(self):
        references = {
            "refit": {
                "job_id": "P08STRESSPRED-" + "0" * 64,
                "seed": SVM_SEED,
                "resolved_source_values": {"alpha": [1, 2, {"beta": 3}]},
            },
            "held_reference_jobs": [{"job_id": "P08STRESSPRED-" + "1" * 64}],
        }
        original = copy.deepcopy(references)
        fields = {
            "binding_sha256": "b" * 64,
            "stage": "model_reconstruction",
            "procedure_id": "P08PROC-" + "a" * 64,
            "context_id": "CTX-Q",
            "policy_id": MIN_POLICY,
            "model_id": SVM_MODEL,
            "case_id": NA,
            "seed": SVM_SEED,
            "depends_on_job_ids": [],
            "depends_on_input_job_ids": [],
            "resolution": MODEL_RECONSTRUCTION_RESOLUTION,
        }
        first = _make_job(upstream_references=references, **fields)
        second = _make_job(upstream_references=references, **fields)

        self.assertIsNot(first, second)
        self.assertIsNot(first["upstream_references"], references)
        self.assertIsNot(second["upstream_references"], references)
        self.assertIsNot(first["upstream_references"], second["upstream_references"])
        self.assertEqual(first["upstream_references"], references)
        self.assertEqual(second["upstream_references"], references)

        for job in (first, second):
            body = {name: value for name, value in job.items() if name != "job_id"}
            self.assertEqual(job["job_id"], "P08STRESSPRED-" + independent_hash(body))

        second_snapshot = copy.deepcopy(second)
        first["upstream_references"]["refit"]["job_id"] = "P08STRESSPRED-" + "f" * 64
        first["upstream_references"]["refit"]["resolved_source_values"]["alpha"].append("x")
        first["upstream_references"]["held_reference_jobs"][0]["job_id"] = "MUTATED"

        self.assertEqual(references, original)
        self.assertEqual(second, second_snapshot)


# ---------------------------------------------------------------------------
# 5. QC route authentication and candidate row receipts.
# ---------------------------------------------------------------------------


class QcRouteAndAssemblyTests(_Base):
    def test_qc_route_authentication(self):
        sc = scenario()
        jobs, _ = self.split(sc.records)
        by_stage = self.by_stage(jobs)
        route_jobs = by_stage["qc_clean_route_authentication"]
        self.assertEqual(len(route_jobs), EXPECTED_ELIGIBLE_QC)
        route = route_jobs[0]
        self.assertEqual(route["case_id"], NA)
        self.assertEqual(route["seed"], NA)
        self.assertEqual(route["policy_id"], QC_POLICY)
        self.assertEqual(route["resolution"], QC_ROUTE_RESOLUTION)
        self.assertEqual(
            set(route["upstream_references"]),
            {
                "source_gate_reference",
                "source_threshold_reference",
                "source_route_reference",
                "clean_route_reference",
            },
        )

        qc_procedures = [
            record
            for record in sc.bundle["procedures"]
            if record["policy_id"] == QC_POLICY and record["context_id"] == route["context_id"]
        ]
        self.assertEqual(len(qc_procedures), 4)
        ready = {}
        for stage in ("model_reconstruction", "retained_model_authentication"):
            for job in by_stage[stage]:
                ready[(job["procedure_id"], job["seed"])] = job
        expected_ready = {
            ready[(record["procedure_id"], seed)]["job_id"]
            for record in qc_procedures
            for seed in record["seeds"]
        }
        self.assertEqual(set(route["depends_on_job_ids"]), expected_ready)

        for key in (
            "source_gate_reference",
            "source_threshold_reference",
            "source_route_reference",
            "clean_route_reference",
        ):
            serialized = {json.dumps(record[key], sort_keys=True) for record in qc_procedures}
            self.assertEqual(len(serialized), 1)
            self.assertEqual(route["upstream_references"][key], qc_procedures[0][key])

    def test_qc_mixed_input_assemblies_are_row_receipts(self):
        sc = scenario()
        jobs, _ = self.split(sc.records)
        by_stage = self.by_stage(jobs)
        route = by_stage["qc_clean_route_authentication"][0]
        mixed = by_stage["qc_mixed_input_assembly"]
        self.assertEqual(len(mixed), EXPECTED_STAGE_COUNTS["qc_mixed_input_assembly"])

        context_id = route["context_id"]
        context_input = next(
            item for item in sc.input_catalog["contexts"] if item["context_id"] == context_id
        )
        test_uids = list(context_input["test_uids"])
        actions = sorted(sc.input_catalog["input_binding"]["actions"])
        families = sc.case_families()
        index = sc.input_index

        for job in mixed:
            self.assertEqual(job["context_id"], context_id)
            self.assertEqual(job["procedure_id"], NA)
            self.assertEqual(job["model_id"], NA)
            self.assertEqual(job["seed"], NA)
            self.assertEqual(job["resolution"], QC_MIXED_RESOLUTION)
            self.assertEqual(job["depends_on_job_ids"], [route["job_id"]])
            transform_context = context_id if families[job["case_id"]] == "gaussian" else NA
            expected = set()
            for uid in test_uids:
                for action in actions:
                    expected.add(
                        index[
                            (
                                "action_transform",
                                transform_context,
                                uid,
                                job["case_id"],
                                action,
                            )
                        ]
                    )
                    expected.add(index[("zero_input_parity", NA, uid, CLEAN_CASE, action)])
            self.assertEqual(set(job["depends_on_input_job_ids"]), expected)

        self.assertTrue(any(families[job["case_id"]] == "gaussian" for job in mixed))
        self.assertTrue(any(families[job["case_id"]] != "gaussian" for job in mixed))


# ---------------------------------------------------------------------------
# 6. Clean-probability parity gate.
# ---------------------------------------------------------------------------


class CleanParityTests(_Base):
    def test_clean_probability_parity(self):
        sc = scenario()
        jobs, _ = self.split(sc.records)
        by_stage = self.by_stage(jobs)
        parity = {job["procedure_id"]: job for job in by_stage["clean_probability_parity"]}
        self.assertEqual(len(parity), EXPECTED_PROCEDURES)
        units = self.units_index(by_stage)
        raw = {(j["procedure_id"], j["case_id"], j["seed"]): j for j in by_stage["raw_prediction"]}
        temperature = {
            (j["procedure_id"], j["case_id"], j["seed"]): j for j in by_stage["temperature_apply"]
        }

        for procedure in sc.bundle["procedures"]:
            procedure_id = procedure["procedure_id"]
            job = parity[procedure_id]
            self.assertEqual(job["case_id"], CLEAN_CASE)
            self.assertEqual(job["seed"], NA)
            self.assertEqual(job["resolution"], CLEAN_PARITY_RESOLUTION)
            dependencies = set(job["depends_on_job_ids"])
            self.assertIn(units[(procedure_id, CLEAN_CASE)]["job_id"], dependencies)
            seeds = list(procedure["seeds"])
            if procedure["model_id"] in NEURAL_MODELS:
                for seed in seeds:
                    self.assertIn(
                        temperature[(procedure_id, CLEAN_CASE, seed)]["job_id"],
                        dependencies,
                    )
            else:
                for seed in seeds:
                    self.assertIn(raw[(procedure_id, CLEAN_CASE, seed)]["job_id"], dependencies)
            references = job["upstream_references"]
            self.assertEqual(
                references["clean_endpoint_reference"],
                procedure["clean_endpoint_reference"],
            )
            self.assertEqual(references["held_reference_jobs"], procedure["held_reference_jobs"])


# ---------------------------------------------------------------------------
# 7. Reporting aliases.
# ---------------------------------------------------------------------------


class AliasTests(_Base):
    def test_alias_logical_identity_and_targets(self):
        sc = scenario()
        records = sc.records
        jobs, aliases = self.split(records)
        self.assertEqual(len(aliases), EXPECTED_REPORTING_ALIASES)

        mode_counts = defaultdict(int)
        views = defaultdict(set)
        for alias in aliases:
            self.assertEqual(alias["record_type"], "alias")
            self.assertEqual(alias["binding_sha256"], sc.bundle["binding_sha256"])
            mode_counts[alias["mode"]] += 1
            views[
                (
                    alias["mode"],
                    alias["policy_id"],
                    alias["context_id"],
                    alias["strategy"],
                )
            ].add(alias["case_id"])
        self.assertEqual(dict(mode_counts), EXPECTED_ALIAS_COUNTS)
        self.assertEqual(len(views), 46)
        for case_set in views.values():
            self.assertEqual(len(case_set), EXPECTED_CASES)

        by_stage = self.by_stage(jobs)
        units = self.units_index(by_stage)
        procedures = self.procedures(sc)
        cases = set(sc.case_ids())
        first_case = sorted(cases)[0]

        for alias in aliases:
            self.assertIn(alias["case_id"], cases)
            target = units[(alias["target_procedure_id"], alias["case_id"])]
            self.assertEqual(alias["target_prediction_units_job_id"], target["job_id"])
            self.assertEqual(target["context_id"], alias["context_id"])
            procedure = procedures[alias["target_procedure_id"]]
            self.assertEqual(procedure["model_id"], alias["recipe_id"])
            if alias["mode"] == "universal":
                self.assertEqual(procedure["policy_id"], alias["policy_id"])
                if alias["strategy"] not in ("D0-M", "P05-SELECTED"):
                    self.assertEqual(alias["upstream_alias_id"], NA)
            elif alias["mode"] == "qc_fixed_route":
                self.assertEqual(procedure["policy_id"], QC_POLICY)
            else:
                self.assertEqual(procedure["policy_id"], MIN_POLICY)

        for policy in POLICIES:
            selected = next(
                alias
                for alias in aliases
                if alias["policy_id"] == policy
                and alias["context_id"] == "CTX-F"
                and alias["strategy"] == "P05-SELECTED"
                and alias["case_id"] == first_case
            )
            direct = next(
                alias
                for alias in aliases
                if alias["policy_id"] == policy
                and alias["context_id"] == "CTX-F"
                and alias["strategy"] == "D0-M"
                and alias["case_id"] == first_case
            )
            self.assertEqual(
                selected["target_prediction_units_job_id"],
                direct["target_prediction_units_job_id"],
            )
            self.assertNotEqual(selected["alias_id"], direct["alias_id"])

        for alias in sc.bundle["family_aliases"]:
            self.assertEqual(alias["new_fits"], 0)
            self.assertEqual(alias["new_predictions"], 0)
            self.assertEqual(alias["policy_id"], FAMILY_POLICY)
        non_universal_recipes = {
            alias["recipe_id"] for alias in aliases if alias["mode"] != "universal"
        }
        self.assertNotIn("C-EXTRA-TREES", non_universal_recipes)
        self.assertFalse([alias for alias in aliases if "adaptive" in alias["mode"]])


# ---------------------------------------------------------------------------
# 8. Purity, snapshot immunity and denied execution.
# ---------------------------------------------------------------------------


class SnapshotAndAuthorityTests(_Base):
    def test_inputs_not_mutated_and_output_not_aliased(self):
        sc = scenario()
        input_snapshot = copy.deepcopy(sc.input_catalog)
        universal_snapshot = copy.deepcopy(sc.universal)
        qc_snapshot = copy.deepcopy(sc.qc_catalog)
        bridge_snapshot = copy.deepcopy(sc.bridge)
        _ = sc.records
        self.assertEqual(sc.input_catalog, input_snapshot)
        self.assertEqual(sc.universal, universal_snapshot)
        self.assertEqual(sc.qc_catalog, qc_snapshot)
        self.assertEqual(sc.bridge, bridge_snapshot)

        first = list(iter_stress_prediction_records(sc.catalog))
        second = list(iter_stress_prediction_records(sc.catalog))
        self.assertIsNot(first[0], second[0])
        original = second[0]["resolution"]
        first[0]["resolution"] = "MUTATED"
        self.assertEqual(second[0]["resolution"], original)

    def test_iterator_yields_deep_copied_nested_references(self):
        sc = scenario()
        catalog = copy.deepcopy(sc.catalog)
        snapshot = copy.deepcopy(catalog)
        expected = list(iter_stress_prediction_records(catalog))

        iterator = iter_stress_prediction_records(catalog)
        first = next(iterator)
        self.assertEqual(first["record_type"], "job")
        self.assertIn(
            first["stage"],
            ("model_reconstruction", "retained_model_authentication"),
        )
        references = first["upstream_references"]
        self.assertIsInstance(references, dict)
        refit = references["refit"]
        self.assertIsInstance(refit, dict)

        refit["job_id"] = "P08STRESSPRED-" + "0" * 64
        resolved = refit.get("resolved_source_values")
        if isinstance(resolved, dict):
            resolved["mutated"] = True
        elif isinstance(resolved, list):
            resolved.append("mutated")

        self.assertEqual(list(iterator), expected[1:])
        self.assertEqual(catalog, snapshot)

    def test_bundle_outputs_do_not_share_nested_objects(self):
        sc = scenario()
        bundle = sc.bundle
        bridge_aliases = {
            alias["alias_id"]: alias for alias in sc.bridge["fallback_endpoint_aliases"]
        }
        for alias in bundle["family_aliases"]:
            source = bridge_aliases[alias["alias_id"]]
            self.assertIsNot(alias, source)
            self.assertIsNot(alias["target_evidence"], source["target_evidence"])

        universal_records = {record["procedure_id"]: record for record in sc.universal["records"]}
        compared = 0
        for procedure in bundle["procedures"]:
            source = universal_records.get(procedure["procedure_id"])
            if source is None:
                continue
            compared += 1
            self.assertIsNot(procedure, source)
            self.assertIsNot(procedure["seeds"], source["seeds"])
            self.assertIsNot(procedure["refit_references"], source["refit_references"])
            self.assertIsNot(procedure["calibration_references"], source["calibration_references"])
        self.assertGreater(compared, 0)

    def test_iterator_snapshot_is_immune_to_catalog_mutation(self):
        sc = scenario()
        catalog = copy.deepcopy(sc.catalog)
        iterator = iter_stress_prediction_records(catalog)
        expected = list(iter_stress_prediction_records(catalog))
        self.assertEqual(len(expected), EXPECTED_PREDICTION_JOBS + EXPECTED_REPORTING_ALIASES)

        catalog["summary"]["prediction_job_count"] = -1
        catalog["summary"]["stage_counts"]["raw_prediction"] = 0
        catalog["bundle"]["execution_authorized"] = True
        self.assertEqual(list(iterator), expected)

    def test_eager_hash_invalid_refusal(self):
        sc = scenario()
        forged = copy.deepcopy(sc.catalog)
        forged["summary"]["prediction_job_count"] += 1
        rehash(forged)
        with self.assertRaises(ValueError) as caught:
            iter_stress_prediction_records(forged)
        self.assertEqual(str(caught.exception), "invalid_stress_prediction_metadata")

    def test_require_scientific_execution_always_denied(self):
        calls = ((), ({},), ({"execution_authorized": True},), (scenario().catalog,))
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError) as caught:
                    require_scientific_execution(*call)
                self.assertEqual(str(caught.exception), "scientific_execution_not_authorized")


# ---------------------------------------------------------------------------
# 9. Rejections: join versus graph scope.
# ---------------------------------------------------------------------------


class RejectionTests(_Base):
    def _assert_join_rejected(self, mutate):
        args = scenario().args()
        mutate(args)
        with self.assertRaises(ValueError) as caught:
            bind_stress_prediction_inputs(**args)
        self.assertEqual(str(caught.exception), "invalid_stress_prediction_binding")

    def _assert_graph_rejected(self, mutate):
        args = scenario().args()
        mutate(args)
        with self.assertRaises(ValueError) as caught:
            build_stress_prediction_catalog(**args)
        self.assertEqual(str(caught.exception), "invalid_stress_prediction_metadata")

    def test_join_rejects_bindings_and_family_aliases(self):
        def input_context_id(args):
            args["input_catalog"]["contexts"][0]["context_id"] = "CTX-OTHER"

        def input_fit_hash(args):
            args["input_catalog"]["contexts"][0]["fit_uid_sha256"] = "0" * 64

        def input_actions(args):
            actions = args["input_catalog"]["input_binding"]["actions"]
            key = sorted(actions)[0]
            actions[key] = "1" * 64

        def missing_family(args):
            args["minimal_bridge"]["fallback_endpoint_aliases"].pop()
            rebind_bridge(args)

        def duplicate_family(args):
            aliases = args["minimal_bridge"]["fallback_endpoint_aliases"]
            aliases.append(copy.deepcopy(aliases[0]))
            rebind_bridge(args)

        def family_target_binding(args):
            alias = args["minimal_bridge"]["fallback_endpoint_aliases"][0]
            alias["target_binding_sha256"] = "0" * 64
            rebind_bridge(args)

        def family_target_job(args):
            alias = args["minimal_bridge"]["fallback_endpoint_aliases"][0]
            alias["target_job_id"] = "P08JOB-" + "0" * 64
            rebind_bridge(args)

        def family_context(args):
            alias = args["minimal_bridge"]["fallback_endpoint_aliases"][0]
            alias["context_id"] = "CTX-OTHER"
            rebind_bridge(args)

        def classical_bool_seed(args):
            def mutate_record(record):
                record["refit_references"][0]["seed"] = True

            return with_universal_record_mutation(args, MIN_POLICY, SVM_MODEL, mutate_record)

        def unknown_reference_scope(args):
            def mutate_record(record):
                record["refit_references"][0]["unexpected_scope"] = "x"

            return with_universal_record_mutation(args, MIN_POLICY, SVM_MODEL, mutate_record)

        def wrong_clean_endpoint(args):
            record = next(
                item
                for item in args["universal_procedures"]["records"]
                if item["policy_id"] == MIN_POLICY and item["model_id"] == SVM_MODEL
            )
            record["clean_endpoint_reference"]["job_id"] = "P08JOB-" + "0" * 64
            rehash(args["universal_procedures"])

        for name, mutate in (
            ("input_context_id", input_context_id),
            ("input_fit_hash", input_fit_hash),
            ("input_actions", input_actions),
            ("missing_family", missing_family),
            ("duplicate_family", duplicate_family),
            ("family_target_binding", family_target_binding),
            ("family_target_job", family_target_job),
            ("family_context", family_context),
            ("classical_bool_seed", classical_bool_seed),
            ("unknown_reference_scope", unknown_reference_scope),
            ("wrong_clean_endpoint", wrong_clean_endpoint),
        ):
            with self.subTest(case=name):
                self._assert_join_rejected(mutate)

    def test_graph_rejects_tampered_catalog(self):
        sc = scenario()

        def forged_summary(catalog):
            catalog["summary"]["prediction_job_count"] += 1

        def forged_stage_count(catalog):
            catalog["summary"]["stage_counts"]["raw_prediction"] = 0

        def forged_alias_count(catalog):
            catalog["summary"]["reporting_alias_count"] += 1

        def forged_flag(catalog):
            catalog["scientific_predictions_computed"] = True

        def forged_bundle_flag(catalog):
            catalog["bundle"]["execution_authorized"] = True

        def forged_bundle_binding(catalog):
            catalog["bundle"]["binding_sha256"] = "0" * 64

        for name, mutate in (
            ("forged_summary", forged_summary),
            ("forged_stage_count", forged_stage_count),
            ("forged_alias_count", forged_alias_count),
            ("forged_flag", forged_flag),
            ("forged_bundle_flag", forged_bundle_flag),
            ("forged_bundle_binding", forged_bundle_binding),
        ):
            with self.subTest(case=name):
                forged = copy.deepcopy(sc.catalog)
                mutate(forged)
                rehash(forged)
                with self.assertRaises(ValueError) as caught:
                    iter_stress_prediction_records(forged)
                self.assertEqual(str(caught.exception), "invalid_stress_prediction_metadata")

        stale = copy.deepcopy(sc.catalog)
        stale["catalog_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            iter_stress_prediction_records(stale)

    def test_graph_rejects_malformed_join_inputs(self):
        for name, mutate in (
            (
                "missing_bridge_alias_type",
                lambda args: args["minimal_bridge"].__setitem__("fallback_endpoint_aliases", {}),
            ),
            ("non_mapping_bridge", lambda args: args.__setitem__("minimal_bridge", [])),
        ):
            with self.subTest(case=name):
                self._assert_graph_rejected(mutate)

    def test_bad_json_bridge_payloads_are_rejected_before_normalization(self):
        payloads = (
            ("nonstring_mapping_key", {1: "value"}),
            ("nan_float_value", float("nan")),
            ("tuple_value", (1, 2)),
            ("object_value", object()),
            ("nested_nonstring_mapping_key", {"outer": {2: "value"}}),
            ("nested_nan_float_value", {"outer": [float("nan")]}),
        )

        def mutate(args, payload):
            # No canonical rehash: the guard must reject malformed JSON before
            # any normalization or hashing of the bridge payload.
            args["minimal_bridge"]["unexpected_extra_field"] = payload

        for name, payload in payloads:
            with self.subTest(case=name, scope="join"):
                self._assert_join_rejected(lambda args, payload=payload: mutate(args, payload))
            with self.subTest(case=name, scope="graph"):
                self._assert_graph_rejected(lambda args, payload=payload: mutate(args, payload))

    def test_universal_record_mutations_are_rejected_after_identity_refresh(self):
        def set_refit_seed_bool(record):
            record["refit_references"][0]["seed"] = True

        def set_refit_seed_float(record):
            record["refit_references"][0]["seed"] = 0.5

        def wrong_calibration_order(record):
            changed = False
            order = record.get("calibration_order")
            if isinstance(order, list) and order:
                record["calibration_order"] = list(reversed(order))
                changed = True
            references = record.get("calibration_references")
            if isinstance(references, list) and len(references) > 1:
                record["calibration_references"] = list(reversed(references))
                changed = True
            if not changed:
                record["calibration_order"] = ["unsupported_order"]

        def set_wrong_reuse_mode(record):
            for key in (
                "reuse_mode",
                "model_reuse_mode",
                "refit_reuse_mode",
                "calibration_reuse_mode",
            ):
                if key in record:
                    record[key] = "unsupported_reuse_mode"
                    return
            record["reuse_mode"] = "unsupported_reuse_mode"

        def add_unknown_refit_field(record):
            record["refit_references"][0]["unexpected_scope"] = "x"

        def duplicate_refit_reference(record):
            record["refit_references"].append(copy.deepcopy(record["refit_references"][0]))

        cases = (
            ("ref_seed_bool", MIN_POLICY, SVM_MODEL, set_refit_seed_bool),
            ("ref_seed_float", MIN_POLICY, SVM_MODEL, set_refit_seed_float),
            ("calibration_order", MIN_POLICY, SVM_MODEL, wrong_calibration_order),
            ("reuse_mode", MIN_POLICY, SVM_MODEL, set_wrong_reuse_mode),
            ("unknown_ref_field", MIN_POLICY, SVM_MODEL, add_unknown_refit_field),
            ("ref_list_length", MIN_POLICY, SVM_MODEL, duplicate_refit_reference),
            (
                "future_sg_classical_ref_seed_bool",
                FUTURE_POLICIES[0],
                SVM_MODEL,
                set_refit_seed_bool,
            ),
        )

        for name, policy_id, model_id, mutate_record in cases:

            def mutate(args, policy_id=policy_id, model_id=model_id, mutate_record=mutate_record):
                return with_universal_record_mutation(args, policy_id, model_id, mutate_record)

            with self.subTest(case=name, scope="join"):
                self._assert_join_rejected(mutate)
            with self.subTest(case=name, scope="graph"):
                self._assert_graph_rejected(mutate)


# ---------------------------------------------------------------------------
# 10. Optional all-fallback fixture.
# ---------------------------------------------------------------------------


class AllFallbackTests(_Base):
    def test_all_fallback_has_no_qc_routes(self):
        sc = scenario(
            [
                {"kind": "master", "context_id": "CTX-F1"},
                {"kind": "master", "context_id": "CTX-F2"},
            ]
        )
        catalog = sc.catalog
        summary = catalog["summary"]
        self.assertEqual(summary["stage_counts"]["qc_clean_route_authentication"], 0)
        self.assertEqual(summary["stage_counts"]["qc_mixed_input_assembly"], 0)
        self.assertEqual(summary["alias_counts_by_mode"]["qc_fixed_route"], 0)
        self.assertGreater(summary["alias_counts_by_mode"]["qc_minimal_fallback"], 0)
        self.assertEqual(summary["eligible_qc_context_count"], 0)

        jobs, aliases = self.split(sc.records)
        by_stage = self.by_stage(jobs)
        self.assertNotIn("qc_clean_route_authentication", by_stage)
        self.assertNotIn("qc_mixed_input_assembly", by_stage)
        units = self.units_index(by_stage)
        procedures = self.procedures(sc)
        for alias in aliases:
            if alias["mode"] in ("qc_minimal_fallback", "family_minimal_fallback"):
                target = units[(alias["target_procedure_id"], alias["case_id"])]
                self.assertEqual(
                    procedures[alias["target_procedure_id"]]["policy_id"],
                    MIN_POLICY,
                )
                self.assertEqual(target["context_id"], alias["context_id"])


if __name__ == "__main__":
    unittest.main()
