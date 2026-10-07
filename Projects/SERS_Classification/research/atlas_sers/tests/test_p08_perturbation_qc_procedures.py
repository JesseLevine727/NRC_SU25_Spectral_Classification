"""Independent contract tests for the T218 fixed-route QC stress adapter.

Synthetic-only: these tests never read evidence, fit, route, score or predict.
They build invented QC catalogs / universal procedures through the public
synthetic fixtures and assert the metadata contract of
``build_qc_procedure_records`` plus the unconditional refusal hook.
"""

from __future__ import annotations

import copy
import hashlib
import unittest

from atlas_sers.evaluation.p08_perturbation_procedures import (
    build_universal_procedure_records,
)
from atlas_sers.evaluation.p08_perturbation_qc_procedures import (
    build_qc_procedure_records,
    require_scientific_execution,
)
from atlas_sers.evaluation.p08_plan import SEEDS, build_universal_plan
from atlas_sers.evaluation.p08_qc_blocks import (
    BLOCK_PREFIX,
    NOT_APPLICABLE,
    SEED_DETERMINISTIC,
    SLOT_PREFIX,
    canonical_sha256,
    iter_slots,
    make_alias,
    make_block,
    seal_catalog,
)
from tests.test_p08_perturbation_procedures import build_fixture_bridge
from tests.test_p08_qc_catalog import (
    build,
    build_fixture,
    fallback_fixture,
    small_fixture,
)

SCHEMA = "nato-sers-p08-qc-stress-procedures-v1"
QC_SCHEMA = "nato-sers-p08-qc-block-catalog-v1"
UNIV_PROC_PREFIX = "P08STRESSPROC-"
PROC_PREFIX = "P08QCSTRESSPROC-"
QC_POLICY = "PP-QC-SRC"
MIN_POLICY = "PP-U-MIN"
GATE_ID = "source_selected_gate"
NA = NOT_APPLICABLE
DETERMINISTIC = SEED_DETERMINISTIC
GENERIC_ERROR = "invalid_qc_stress_procedure_metadata"
REFUSAL_ERROR = "scientific_execution_not_authorized"

ALIAS_STRATEGIES = ("C-RBF-SVM", "C-RANDOM-FOREST", "D0-M", "P05-SELECTED")
CLASSICAL = frozenset({"C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES"})
BASE_MODELS = frozenset({"C-RBF-SVM", "C-RANDOM-FOREST", "D0-M"})

OUTPUT_KEYS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "scientific_operations",
        "artifact_provenance_independently_verified",
        "exact_prediction_ledger_complete",
        "clean_routes_resolved",
        "parent_qc_catalog_sha256",
        "parent_universal_catalog_sha256",
        "records",
        "strategy_aliases",
        "summary",
        "catalog_sha256",
    }
)

SUMMARY_KEYS = frozenset(
    {
        "context_count",
        "eligible_context_count",
        "fallback_context_count",
        "procedure_count",
        "seed_estimator_count",
        "calibration_reference_count",
        "reporting_alias_count",
        "eligible_reporting_alias_count",
        "fallback_reporting_alias_count",
        "unique_fallback_minimal_procedure_count",
    }
)

RECORD_KEYS = frozenset(
    {
        "context_id",
        "policy_id",
        "model_id",
        "model_spec_sha256",
        "action_array_sha256",
        "fit_uid_sha256",
        "test_uid_sha256",
        "seeds",
        "parent_qc_catalog_sha256",
        "parent_universal_catalog_sha256",
        "binding_sha256",
        "model_reuse_mode",
        "calibration_order",
        "clean_route_mode",
        "refit_references",
        "calibration_references",
        "held_reference_jobs",
        "clean_endpoint_reference",
        "source_gate_reference",
        "source_threshold_reference",
        "source_route_reference",
        "clean_route_reference",
        "procedure_id",
    }
)

ALIAS_KEYS = frozenset(
    {
        "context_id",
        "policy_id",
        "strategy",
        "recipe_id",
        "mode",
        "target_policy_id",
        "target_procedure_id",
        "upstream_alias_id",
        "reason_code",
        "same_case_target_required",
    }
)

REFERENCE_KEYS = frozenset({"block_id", "slot_id", "seed", "resolution"})
REFERENCE_LIST_FIELDS = (
    "refit_references",
    "calibration_references",
    "held_reference_jobs",
)
REFERENCE_SINGLE_FIELDS = (
    "clean_endpoint_reference",
    "source_gate_reference",
    "source_threshold_reference",
    "source_route_reference",
    "clean_route_reference",
)


def _all_eligible_fixture():
    return build_fixture(
        [
            {"kind": "pseudo", "context_id": "EL-D0", "recipe": "D0-M"},
            {"kind": "pseudo", "context_id": "EL-D3", "recipe": "D3"},
        ]
    )


_FIXTURE_FACTORIES = {
    "small": small_fixture,
    "fallback": fallback_fixture,
    "eligible": _all_eligible_fixture,
}

_CACHE: dict = {}


def base_fixture(name="small"):
    key = ("fixture", name)
    if key not in _CACHE:
        _CACHE[key] = _FIXTURE_FACTORIES[name]()
    return copy.deepcopy(_CACHE[key])


def base_qc(name="small"):
    key = ("qc", name)
    if key not in _CACHE:
        _CACHE[key] = build(base_fixture(name))
    return copy.deepcopy(_CACHE[key])


def base_universal(name="small"):
    key = ("universal", name)
    if key not in _CACHE:
        fixture = base_fixture(name)
        plan = build_universal_plan(
            fixture["universal_contexts"],
            fixture["candidates"],
            fixture["actions"],
            fixture["model_spec_sha256"],
        )
        _CACHE[key] = build_universal_procedure_records(
            plan=plan, minimal_bridge=build_fixture_bridge(plan)
        )
    return copy.deepcopy(_CACHE[key])


def base_output(name="small"):
    key = ("output", name)
    if key not in _CACHE:
        _CACHE[key] = build_qc_procedure_records(
            qc_catalog=base_qc(name), universal_procedures=base_universal(name)
        )
    return copy.deepcopy(_CACHE[key])


SUMMARY_EXPECTED = {
    "small": {
        "context_count": 4,
        "eligible_context_count": 2,
        "fallback_context_count": 2,
        "procedure_count": 7,
        "seed_estimator_count": 17,
        "calibration_reference_count": 13,
        "reporting_alias_count": 16,
        "eligible_reporting_alias_count": 8,
        "fallback_reporting_alias_count": 8,
        "unique_fallback_minimal_procedure_count": 7,
    },
    "fallback": {
        "context_count": 2,
        "eligible_context_count": 0,
        "fallback_context_count": 2,
        "procedure_count": 0,
        "seed_estimator_count": 0,
        "calibration_reference_count": 0,
        "reporting_alias_count": 8,
        "eligible_reporting_alias_count": 0,
        "fallback_reporting_alias_count": 8,
        "unique_fallback_minimal_procedure_count": 7,
    },
    "eligible": {
        "context_count": 2,
        "eligible_context_count": 2,
        "fallback_context_count": 0,
        "procedure_count": 7,
        "seed_estimator_count": 17,
        "calibration_reference_count": 13,
        "reporting_alias_count": 8,
        "eligible_reporting_alias_count": 8,
        "fallback_reporting_alias_count": 0,
        "unique_fallback_minimal_procedure_count": 0,
    },
}


def compact_qc(qc, context_id):
    return next(
        compact
        for compact in qc["bindings"]["universal_contexts"]
        if compact["context_id"] == context_id
    )


def qc_alias_map(qc):
    return {(a["context_id"], a["strategy"]): a for a in qc["aliases"]}


def eligible_contexts(qc):
    return {block["context_id"] for block in qc["blocks"]}


def is_classical(model_id):
    return model_id in CLASSICAL


def expected_seeds(model_id):
    return [DETERMINISTIC] if model_id == "C-RBF-SVM" else list(SEEDS)


def expected_calibration_seeds(model_id):
    return [NA] if is_classical(model_id) else list(SEEDS)


def expected_candidate_id(model_id):
    return "source_selected_candidate" if is_classical(model_id) else "fixed_spec"


def expected_calibration_order(model_id):
    return (
        "seed_average_then_single_temperature"
        if is_classical(model_id)
        else "per_seed_temperature_then_seed_average"
    )


def expected_scalar_resolution(model_id):
    return (
        "seed_average_then_single_temperature_master_equal"
        if is_classical(model_id)
        else "same_gate_same_seed_source_logits_master_equal"
    )


def expected_held_resolution(model_id):
    return "raw_scores" if is_classical(model_id) else "same_seed_calibrated_scores"


def expected_ensemble_resolution(model_id):
    return (
        "seed_average_then_single_temperature_logclip1e_7"
        if is_classical(model_id)
        else "average_calibrated_seed_probabilities"
    )


def expected_models(qc, context_id):
    recipe = compact_qc(qc, context_id)["selected_recipe_id"]
    models = set(BASE_MODELS)
    if recipe not in models:
        models.add(recipe)
    return models


def slot_id_for(block, seed):
    return SLOT_PREFIX + canonical_sha256(
        {
            "block_id": block["block_id"],
            "gate_id": GATE_ID,
            "candidate": dict(block["axes"]["candidate"][0]),
            "seed": seed,
        }
    )


def blocks_by_stage(qc, context_id):
    stages: dict = {}
    for block in qc["blocks"]:
        if block["context_id"] == context_id:
            stages.setdefault(block["stage"], []).append(block)
    return stages


def record_model(record):
    return record.get("model_id") or record.get("recipe_id")


def clone_block(block, **changes):
    """Rebuild a fresh block from an existing one, applying field changes."""
    fields = {
        "binding_sha256": block["binding_sha256"],
        "context_id": block["context_id"],
        "stage": block["stage"],
        "model_id": block["model_id"],
        "role_id": block["role_id"],
        "fit_uid_sha256": block["fit_uid_sha256"],
        "validation_uid_sha256": block["validation_uid_sha256"],
        "test_uid_sha256": block["test_uid_sha256"],
        "axes": copy.deepcopy(block["axes"]),
        "depends_on_blocks": list(block["depends_on_blocks"]),
        "resolution": block["resolution"],
    }
    fields.update(changes)
    return make_block(**fields)


def _flip_first_hash(value, state=None):
    if state is None:
        state = {"done": False}
    if state["done"]:
        return value
    if isinstance(value, str):
        if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
            state["done"] = True
            return hashlib.sha256(b"p08-t219-mutated").hexdigest()
        return value
    if isinstance(value, dict):
        return {key: _flip_first_hash(item, state) for key, item in value.items()}
    if isinstance(value, list):
        return [_flip_first_hash(item, state) for item in value]
    return value


def variant_qc(key):
    fixture = small_fixture()
    fixture[key] = _flip_first_hash(copy.deepcopy(fixture[key]))
    return build(fixture)


def _topo_order(blocks):
    by_id = {block["block_id"]: block for block in blocks}
    order, state = [], {}

    def visit(block_id):
        if state.get(block_id) == 2:
            return
        if state.get(block_id) == 1:
            raise AssertionError("cycle detected in remap_qc input graph")
        state[block_id] = 1
        for dep in by_id[block_id]["depends_on_blocks"]:
            visit(dep)
        state[block_id] = 2
        order.append(block_id)

    for block_id in by_id:
        visit(block_id)
    return order


def remap_qc(qc, mutator):
    """Rebuild a QC catalog with a semantic block mutation and fresh DAG IDs."""
    originals = [copy.deepcopy(block) for block in qc["blocks"]]
    by_id = {block["block_id"]: block for block in originals}
    order = _topo_order(originals)
    new_id, new_blocks = {}, []
    for original_id in order:
        block = copy.deepcopy(by_id[original_id])
        original_deps = list(block["depends_on_blocks"])
        block["depends_on_blocks"] = [new_id.get(dep, dep) for dep in original_deps]
        mutator(block, original_id, original_deps)
        rebuilt = make_block(
            binding_sha256=block["binding_sha256"],
            context_id=block["context_id"],
            stage=block["stage"],
            model_id=block["model_id"],
            role_id=block["role_id"],
            fit_uid_sha256=block["fit_uid_sha256"],
            validation_uid_sha256=block["validation_uid_sha256"],
            test_uid_sha256=block["test_uid_sha256"],
            axes=block["axes"],
            depends_on_blocks=block["depends_on_blocks"],
            resolution=block["resolution"],
        )
        new_id[original_id] = rebuilt["block_id"]
        new_blocks.append(rebuilt)
    new_aliases = []
    for alias in qc["aliases"]:
        target = new_id.get(alias["target_block_id"], alias["target_block_id"])
        new_aliases.append(
            make_alias(
                context_id=alias["context_id"],
                strategy=alias["strategy"],
                recipe_id=alias["recipe_id"],
                target_block_id=target,
                evidence_status=alias["evidence_status"],
                reason_code=alias["reason_code"],
                metadata=alias["metadata"],
            )
        )
    return seal_catalog(qc["bindings"], new_blocks, new_aliases)


def rebuild_qc_alias(qc, predicate, **changes):
    aliases = []
    for alias in qc["aliases"]:
        fields = {
            "context_id": alias["context_id"],
            "strategy": alias["strategy"],
            "recipe_id": alias["recipe_id"],
            "target_block_id": alias["target_block_id"],
            "evidence_status": alias["evidence_status"],
            "reason_code": alias["reason_code"],
            "metadata": copy.deepcopy(alias["metadata"]),
        }
        if predicate(alias):
            fields.update(changes)
        aliases.append(make_alias(**fields))
    return seal_catalog(qc["bindings"], qc["blocks"], aliases)


def force_block_change(qc, block_id, mutator):
    """Rehash a raw catalog payload without block-level validation.

    Used only to reach the API with a value the block constructor refuses
    (for example a boolean seed).  The API is then expected to reject the
    payload at its own validation boundary.
    """
    body = {key: copy.deepcopy(value) for key, value in qc.items() if key != "catalog_sha256"}
    for block in body["blocks"]:
        if block["block_id"] == block_id:
            mutator(block)
            payload = {key: value for key, value in block.items() if key != "block_id"}
            block["block_id"] = BLOCK_PREFIX + canonical_sha256(payload)
    body["blocks"] = sorted(body["blocks"], key=lambda b: b["block_id"])
    body["catalog_sha256"] = canonical_sha256(body)
    return body


def rehash_universal(universal):
    clone = copy.deepcopy(universal)
    remap = {}
    for record in clone["records"]:
        old_id = record["procedure_id"]
        payload = {key: value for key, value in record.items() if key != "procedure_id"}
        new_id = UNIV_PROC_PREFIX + canonical_sha256(payload)
        remap[old_id] = new_id
        record["procedure_id"] = new_id
    aliases = clone.get("strategy_aliases")
    if aliases is not None:
        kept = []
        for alias in aliases:
            target = alias.get("target_procedure_id")
            if target in remap:
                alias["target_procedure_id"] = remap[target]
                kept.append(alias)
            elif target is None:
                kept.append(alias)
            # A target whose record was removed is dropped so a stale alias
            # cannot mask the intended missing-group semantics downstream.
        clone["strategy_aliases"] = kept
    clone.pop("catalog_sha256", None)
    clone["catalog_sha256"] = canonical_sha256(clone)
    for record in clone["records"]:
        payload = {key: value for key, value in record.items() if key != "procedure_id"}
        assert record["procedure_id"] == UNIV_PROC_PREFIX + canonical_sha256(payload)
    body = {key: value for key, value in clone.items() if key != "catalog_sha256"}
    assert clone["catalog_sha256"] == canonical_sha256(body)
    return clone


class _Base(unittest.TestCase):
    def assert_rejected(self, qc=None, universal=None, fixture="small"):
        if qc is None:
            qc = base_qc(fixture)
        if universal is None:
            universal = base_universal(fixture)
        with self.assertRaises(ValueError) as caught:
            build_qc_procedure_records(qc_catalog=qc, universal_procedures=universal)
        self.assertEqual(str(caught.exception), GENERIC_ERROR)
        return caught.exception

    @staticmethod
    def one(stages, stage, model=None):
        found = [block for block in stages[stage] if model is None or block["model_id"] == model]
        if len(found) != 1:
            raise AssertionError((stage, model, len(found)))
        return found[0]


class _ContractMixin(_Base):
    def check_summary(self, out, expected):
        self.assertEqual(set(out), OUTPUT_KEYS)
        self.assertEqual(out["schema_version"], SCHEMA)
        self.assertIs(out["execution_authorized"], False)
        self.assertEqual(out["scientific_operations"], 0)
        self.assertIs(out["artifact_provenance_independently_verified"], False)
        self.assertIs(out["exact_prediction_ledger_complete"], False)
        self.assertIs(out["clean_routes_resolved"], False)
        self.assertEqual(set(out["summary"]), SUMMARY_KEYS)
        for key, value in expected.items():
            self.assertEqual(out["summary"][key], value, key)
        body = {key: value for key, value in out.items() if key != "catalog_sha256"}
        self.assertEqual(out["catalog_sha256"], canonical_sha256(body))
        self.assertEqual(
            out["records"],
            sorted(out["records"], key=lambda r: (r["context_id"], r["model_id"])),
        )
        self.assertEqual(
            out["strategy_aliases"],
            sorted(
                out["strategy_aliases"],
                key=lambda a: (a["context_id"], a["strategy"]),
            ),
        )
        self.assertEqual(out["summary"]["procedure_count"], len(out["records"]))
        self.assertEqual(out["summary"]["reporting_alias_count"], len(out["strategy_aliases"]))
        self.assertEqual(
            out["summary"]["eligible_reporting_alias_count"]
            + out["summary"]["fallback_reporting_alias_count"],
            out["summary"]["reporting_alias_count"],
        )
        self.assertEqual(
            out["summary"]["seed_estimator_count"],
            sum(len(record["seeds"]) for record in out["records"]),
        )
        self.assertEqual(
            out["summary"]["calibration_reference_count"],
            sum(len(record["calibration_references"]) for record in out["records"]),
        )
        fallback = [
            alias
            for alias in out["strategy_aliases"]
            if alias["mode"] == "disturbed_minimal_pipeline"
        ]
        self.assertEqual(
            out["summary"]["unique_fallback_minimal_procedure_count"],
            len({alias["target_procedure_id"] for alias in fallback}),
        )

    def check_reference(self, qc, reference):
        self.assertEqual(set(reference), REFERENCE_KEYS)
        by_id = {block["block_id"]: block for block in qc["blocks"]}
        self.assertIn(reference["block_id"], by_id)
        block = by_id[reference["block_id"]]
        self.assertEqual(reference["slot_id"], slot_id_for(block, reference["seed"]))
        self.assertEqual(reference["resolution"], block["resolution"])
        return block

    def check_record_references(self, qc, record):
        context_id = record["context_id"]
        model = record["model_id"]
        stages = blocks_by_stage(qc, context_id)
        compact = compact_qc(qc, context_id)

        gate = self.one(stages, "gate_selection")
        threshold = self.one(stages, "final_refit_quantile_fit")
        route = self.one(stages, "final_source_route")
        test_route = self.one(stages, "final_test_route")
        refit = self.one(stages, "final_refit", model)
        scalar = self.one(stages, "final_scalar_calibration", model)
        held = self.one(stages, "final_held_prediction", model)
        endpoint = self.one(stages, "final_seed_ensemble_prediction", model)

        for reference in record["refit_references"]:
            self.assertEqual(self.check_reference(qc, reference)["block_id"], refit["block_id"])
        for reference in record["calibration_references"]:
            self.assertEqual(self.check_reference(qc, reference)["block_id"], scalar["block_id"])
        for reference in record["held_reference_jobs"]:
            self.assertEqual(self.check_reference(qc, reference)["block_id"], held["block_id"])
        self.assertEqual(
            self.check_reference(qc, record["clean_endpoint_reference"])["block_id"],
            endpoint["block_id"],
        )
        self.assertEqual(
            self.check_reference(qc, record["source_gate_reference"])["block_id"],
            gate["block_id"],
        )
        self.assertEqual(
            self.check_reference(qc, record["source_threshold_reference"])["block_id"],
            threshold["block_id"],
        )
        self.assertEqual(
            self.check_reference(qc, record["source_route_reference"])["block_id"],
            route["block_id"],
        )
        self.assertEqual(
            self.check_reference(qc, record["clean_route_reference"])["block_id"],
            test_route["block_id"],
        )

        # Seed ordering and calibration ordering.
        self.assertEqual([ref["seed"] for ref in record["refit_references"]], expected_seeds(model))
        self.assertEqual(
            [ref["seed"] for ref in record["calibration_references"]],
            expected_calibration_seeds(model),
        )
        self.assertEqual(
            [ref["seed"] for ref in record["held_reference_jobs"]], expected_seeds(model)
        )
        self.assertEqual(record["clean_endpoint_reference"]["seed"], NA)
        self.assertEqual(
            record["calibration_references"][0]["resolution"],
            expected_scalar_resolution(model),
        )
        self.assertEqual(
            {ref["resolution"] for ref in record["held_reference_jobs"]},
            {expected_held_resolution(model)},
        )
        self.assertEqual(
            record["clean_endpoint_reference"]["resolution"],
            expected_ensemble_resolution(model),
        )

        # Block axes for model-local blocks.
        for block, seeds in (
            (refit, expected_seeds(model)),
            (scalar, expected_calibration_seeds(model)),
            (held, expected_seeds(model)),
            (endpoint, [NA]),
        ):
            self.assertEqual(block["axes"]["gate_id"], [GATE_ID])
            self.assertEqual(block["axes"]["seed"], seeds)
            self.assertEqual(
                block["axes"]["candidate"][0],
                {
                    "candidate_id": expected_candidate_id(model),
                    "hyperparameter_sha256": NA,
                },
            )
            self.assertEqual(block["role_id"], context_id)
            self.assertEqual(block["fit_uid_sha256"], compact["outer_fit_uid_sha256"])
            self.assertEqual(block["validation_uid_sha256"], NA)

        self.assertEqual(refit["test_uid_sha256"], NA)
        self.assertEqual(scalar["test_uid_sha256"], NA)
        self.assertEqual(held["test_uid_sha256"], compact["outer_test_uid_sha256"])
        self.assertEqual(endpoint["test_uid_sha256"], compact["outer_test_uid_sha256"])

        # Shared source blocks.
        for block in (gate, threshold, route):
            self.assertEqual(block["axes"]["gate_id"], [GATE_ID])
            self.assertEqual(block["axes"]["seed"], [NA])
            self.assertEqual(
                block["axes"]["candidate"][0],
                {"candidate_id": NA, "hyperparameter_sha256": NA},
            )
            self.assertEqual(block["fit_uid_sha256"], compact["outer_fit_uid_sha256"])
            self.assertEqual(block["validation_uid_sha256"], NA)
            self.assertEqual(block["test_uid_sha256"], NA)
        self.assertEqual(test_route["fit_uid_sha256"], NA)
        self.assertEqual(test_route["validation_uid_sha256"], NA)
        self.assertEqual(test_route["test_uid_sha256"], compact["outer_test_uid_sha256"])

        # Source DAG (preserved, never recomputed here).
        self.assertEqual(set(threshold["depends_on_blocks"]), {gate["block_id"]})
        self.assertEqual(
            set(route["depends_on_blocks"]),
            {gate["block_id"], threshold["block_id"]},
        )
        refits = {
            self.one(stages, "final_refit", other)["block_id"]
            for other in expected_models(qc, context_id)
        }
        self.assertEqual(
            set(test_route["depends_on_blocks"]),
            {gate["block_id"], threshold["block_id"]} | refits,
        )

    def check_records(self, out, qc, universal):
        by_cm = {}
        for record in out["records"]:
            self.assertEqual(set(record), RECORD_KEYS)
            context_id = record["context_id"]
            model = record["model_id"]
            compact = compact_qc(qc, context_id)
            self.assertEqual(record["policy_id"], QC_POLICY)
            self.assertEqual(record["binding_sha256"], canonical_sha256(qc["bindings"]))
            self.assertEqual(record["parent_qc_catalog_sha256"], qc["catalog_sha256"])
            self.assertEqual(record["parent_universal_catalog_sha256"], universal["catalog_sha256"])
            self.assertEqual(record["seeds"], expected_seeds(model))
            self.assertEqual(
                record["model_spec_sha256"], qc["bindings"]["model_spec_sha256"][model]
            )
            self.assertEqual(
                record["action_array_sha256"],
                qc["bindings"]["actions"],
            )
            self.assertEqual(record["fit_uid_sha256"], compact["outer_fit_uid_sha256"])
            self.assertEqual(record["test_uid_sha256"], compact["outer_test_uid_sha256"])
            self.assertEqual(record["model_reuse_mode"], "future_retained_mixed_route_estimator")
            self.assertEqual(
                record["clean_route_mode"],
                "fixed_native_clean_route_not_gate_reaction",
            )
            self.assertEqual(record["calibration_order"], expected_calibration_order(model))
            payload = {key: value for key, value in record.items() if key != "procedure_id"}
            self.assertEqual(record["procedure_id"], PROC_PREFIX + canonical_sha256(payload))
            self.check_record_references(qc, record)
            by_cm[(context_id, model)] = record
        return by_cm

    def check_aliases(self, out, qc, universal):
        eligible = eligible_contexts(qc)
        by_cm = {
            (record["context_id"], record["model_id"]): record["procedure_id"]
            for record in out["records"]
        }
        upstream = qc_alias_map(qc)
        universal_ids = {record["procedure_id"]: record for record in universal["records"]}
        by_context: dict = {}
        for alias in out["strategy_aliases"]:
            self.assertEqual(set(alias), ALIAS_KEYS)
            self.assertEqual(alias["policy_id"], QC_POLICY)
            self.assertIs(alias["same_case_target_required"], True)
            self.assertIn(alias["strategy"], ALIAS_STRATEGIES)
            context_id = alias["context_id"]
            source_alias = upstream[(context_id, alias["strategy"])]
            self.assertEqual(alias["upstream_alias_id"], source_alias["alias_id"])
            self.assertEqual(alias["recipe_id"], source_alias["recipe_id"])
            if context_id in eligible:
                self.assertEqual(alias["mode"], "fixed_clean_route")
                self.assertEqual(alias["target_policy_id"], QC_POLICY)
                self.assertEqual(
                    alias["target_procedure_id"],
                    by_cm[(context_id, alias["recipe_id"])],
                )
                self.assertEqual(alias["reason_code"], "eligible")
            else:
                self.assertEqual(alias["mode"], "disturbed_minimal_pipeline")
                self.assertEqual(alias["target_policy_id"], MIN_POLICY)
                self.assertEqual(
                    alias["target_procedure_id"],
                    universal_ids[alias["target_procedure_id"]]["procedure_id"],
                )
                target = universal_ids[alias["target_procedure_id"]]
                self.assertEqual(target["context_id"], context_id)
                self.assertEqual(record_model(target), alias["recipe_id"])
                self.assertNotIn(alias["target_procedure_id"], set(by_cm.values()))
            by_context.setdefault(context_id, []).append(alias)
        for _context_id, aliases in by_context.items():
            self.assertEqual({alias["strategy"] for alias in aliases}, set(ALIAS_STRATEGIES))
        return by_context


class MixedOutputTests(_ContractMixin):
    def test_schema_summary_records_and_aliases(self):
        qc = base_qc("small")
        universal = base_universal("small")
        out = base_output("small")
        self.check_summary(out, SUMMARY_EXPECTED["small"])
        self.check_records(out, qc, universal)
        by_context = self.check_aliases(out, qc, universal)
        self.assertEqual(set(by_context), {"CTX-D0", "CTX-D3", "CTX-MASTER", "CTX-THIN"})
        eligible = eligible_contexts(qc)
        self.assertEqual(eligible, {"CTX-D0", "CTX-D3"})
        for context_id, models in (
            ("CTX-D0", {"C-RBF-SVM", "C-RANDOM-FOREST", "D0-M"}),
            ("CTX-D3", {"C-RBF-SVM", "C-RANDOM-FOREST", "D0-M", "D3"}),
        ):
            self.assertEqual(
                {r["model_id"] for r in out["records"] if r["context_id"] == context_id},
                models,
            )

    def test_shared_references_are_shared_and_preserved(self):
        qc = base_qc("small")
        out = base_output("small")
        for context_id in ("CTX-D0", "CTX-D3"):
            records = [r for r in out["records"] if r["context_id"] == context_id]
            for field in (
                "source_gate_reference",
                "source_threshold_reference",
                "source_route_reference",
                "clean_route_reference",
            ):
                self.assertEqual(len({record[field]["block_id"] for record in records}), 1)
            stages = blocks_by_stage(qc, context_id)
            self.assertEqual(stages["gate_selection"][0]["axes"]["gate_id"], [GATE_ID])

    def test_selected_alias_sharing(self):
        by_context = {}
        for alias in base_output("small")["strategy_aliases"]:
            by_context.setdefault(alias["context_id"], {})[alias["strategy"]] = alias
        d0 = by_context["CTX-D0"]
        self.assertEqual(
            d0["D0-M"]["target_procedure_id"],
            d0["P05-SELECTED"]["target_procedure_id"],
        )
        d3 = by_context["CTX-D3"]
        self.assertNotEqual(
            d3["D0-M"]["target_procedure_id"],
            d3["P05-SELECTED"]["target_procedure_id"],
        )
        self.assertEqual(d3["P05-SELECTED"]["recipe_id"], "D3")

    def test_no_extratrees_blocks_or_records(self):
        qc = base_qc("small")
        out = base_output("small")
        self.assertFalse([b for b in qc["blocks"] if b["model_id"] == "C-EXTRA-TREES"])
        self.assertFalse([r for r in out["records"] if r["model_id"] == "C-EXTRA-TREES"])

    def test_no_orphan_final_blocks(self):
        qc = base_qc("small")
        out = base_output("small")
        final_stages = {
            "final_refit",
            "final_scalar_calibration",
            "final_held_prediction",
            "final_seed_ensemble_prediction",
        }
        records = {(r["context_id"], r["model_id"]) for r in out["records"]}
        for block in qc["blocks"]:
            if block["stage"] in final_stages:
                self.assertIn((block["context_id"], block["model_id"]), records)


class FallbackOnlyTests(_ContractMixin):
    def test_all_fallback_output(self):
        qc = base_qc("fallback")
        universal = base_universal("fallback")
        out = base_output("fallback")
        self.assertEqual(qc["blocks"], [])
        self.assertEqual(out["records"], [])
        self.check_summary(out, SUMMARY_EXPECTED["fallback"])
        by_context = self.check_aliases(out, qc, universal)
        self.assertEqual(set(by_context), {"CTX-M1", "CTX-T1"})
        for alias in out["strategy_aliases"]:
            self.assertEqual(alias["mode"], "disturbed_minimal_pipeline")
            self.assertEqual(alias["target_policy_id"], MIN_POLICY)
            self.assertTrue(alias["target_procedure_id"].startswith(UNIV_PROC_PREFIX))
            self.assertNotIn(
                alias["target_procedure_id"],
                {record["procedure_id"] for record in out["records"]},
            )


class EligibleOnlyTests(_ContractMixin):
    def test_all_eligible_output(self):
        qc = base_qc("eligible")
        universal = base_universal("eligible")
        out = base_output("eligible")
        self.check_summary(out, SUMMARY_EXPECTED["eligible"])
        self.check_records(out, qc, universal)
        self.check_aliases(out, qc, universal)
        self.assertFalse(
            [
                alias
                for alias in out["strategy_aliases"]
                if alias["mode"] == "disturbed_minimal_pipeline"
            ]
        )
        self.assertEqual(out["summary"]["fallback_context_count"], 0)


class DeterminismTests(_Base):
    def test_deterministic_no_mutation_and_deep_independence(self):
        qc = base_qc("small")
        universal = base_universal("small")
        qc_snapshot = copy.deepcopy(qc)
        universal_snapshot = copy.deepcopy(universal)

        # Pass the real inputs; the adapter must not mutate them.
        first = build_qc_procedure_records(qc_catalog=qc, universal_procedures=universal)
        second = build_qc_procedure_records(qc_catalog=qc, universal_procedures=universal)
        self.assertEqual(first, second)
        self.assertEqual(qc, qc_snapshot)
        self.assertEqual(universal, universal_snapshot)

        baseline = copy.deepcopy(first)

        # Snapshot every other record's action-array mapping before injection.
        other_action_arrays_before = {
            record["procedure_id"]: copy.deepcopy(record["action_array_sha256"])
            for record in first["records"][1:]
        }

        # Mutate a returned record's action-array mapping in place.
        first["records"][0]["action_array_sha256"]["INJECTED"] = "0" * 64
        for record in first["records"][1:]:
            self.assertEqual(
                record["action_array_sha256"],
                other_action_arrays_before[record["procedure_id"]],
            )
        same_context = [
            record
            for record in first["records"]
            if record["context_id"] == first["records"][0]["context_id"]
        ]
        other = next(record for record in same_context if record is not first["records"][0])
        other_reference_before = copy.deepcopy(other["clean_route_reference"])

        # Mutate the shared clean-route reference on the first record.
        first["records"][0]["clean_route_reference"]["slot_id"] = "MUTATED-SLOT"
        self.assertEqual(other["clean_route_reference"], other_reference_before)
        self.assertNotEqual(
            first["records"][0]["clean_route_reference"]["slot_id"],
            baseline["records"][0]["clean_route_reference"]["slot_id"],
        )

        self.assertEqual(qc, qc_snapshot)
        self.assertEqual(universal, universal_snapshot)

        third = build_qc_procedure_records(qc_catalog=qc, universal_procedures=universal)
        self.assertEqual(third, baseline)

    def test_iter_slots_validates_then_streams(self):
        qc = base_qc("small")
        stream = iter_slots(qc)
        first = next(stream)
        self.assertIn("slot_id", first)
        self.assertTrue(first["slot_id"].startswith(SLOT_PREFIX))

    def test_require_scientific_execution_always_refuses(self):
        for args, kwargs in (
            ((), {}),
            ((base_qc("small"),), {}),
            ((), {"qc_catalog": {"execution_authorized": True}}),
        ):
            with self.assertRaises(ValueError) as caught:
                require_scientific_execution(*args, **kwargs)
            self.assertEqual(str(caught.exception), REFUSAL_ERROR)


class NegativeInputTests(_Base):
    def test_rehash_helper_matches_producer(self):
        universal = base_universal("small")
        self.assertEqual(rehash_universal(copy.deepcopy(universal)), universal)

    def test_authority_schema_and_stale_hash_either_input(self):
        for key, value in (
            ("execution_authorized", True),
            ("schema_version", "bogus-schema"),
        ):
            with self.subTest(input="qc", key=key):
                qc = base_qc("small")
                qc[key] = value
                self.assert_rejected(qc=qc)
            with self.subTest(input="universal", key=key):
                universal = base_universal("small")
                universal[key] = value
                self.assert_rejected(universal=universal)
        qc = base_qc("small")
        qc["catalog_sha256"] = "0" * 64
        self.assert_rejected(qc=qc)
        universal = base_universal("small")
        universal["catalog_sha256"] = "f" * 64
        self.assert_rejected(universal=universal)

    def test_cross_catalog_binding_mismatch(self):
        # Only the action array and model-spec bindings are cross-bound to the
        # universal minimal-pipeline records.  Candidate grids and gate
        # libraries are authenticated by the caller, never by this adapter, so
        # changing them is not a cross-binding error.
        for key in ("actions", "model_spec_sha256"):
            with self.subTest(key=key):
                self.assert_rejected(qc=variant_qc(key))

    def test_changed_candidate_or_gate_qc_extracts_within_boundary(self):
        universal = base_universal("small")
        baseline = base_output("small")
        changed_catalogs = (
            ("candidates", variant_qc("candidates")),
            (
                "gate_library",
                build(
                    small_fixture(),
                    gate_library_sha256=hashlib.sha256(b"p08-t221-gate").hexdigest(),
                ),
            ),
        )
        for label, qc in changed_catalogs:
            with self.subTest(catalog=label):
                self.assertNotEqual(qc["catalog_sha256"], base_qc("small")["catalog_sha256"])
                out = build_qc_procedure_records(qc_catalog=qc, universal_procedures=universal)
                self.assertIs(out["execution_authorized"], False)
                self.assertIs(out["artifact_provenance_independently_verified"], False)
                self.assertIs(out["exact_prediction_ledger_complete"], False)
                self.assertIs(out["clean_routes_resolved"], False)
                self.assertEqual(out["summary"], baseline["summary"])
                self.assertNotEqual(
                    out["parent_qc_catalog_sha256"],
                    baseline["parent_qc_catalog_sha256"],
                )
                self.assertEqual(
                    out["parent_universal_catalog_sha256"],
                    universal["catalog_sha256"],
                )
                self.assertNotEqual(
                    [r["procedure_id"] for r in out["records"]],
                    [r["procedure_id"] for r in baseline["records"]],
                )

    def test_cross_catalog_outer_hash_mismatch(self):
        for key in ("outer_fit_uid_sha256", "outer_test_uid_sha256"):
            with self.subTest(key=key):
                fixture = small_fixture()
                fixture["universal_contexts"][0] = dict(
                    fixture["universal_contexts"][0],
                    **{key: hashlib.sha256(b"p08-t219-outer").hexdigest()},
                )
                universal = base_universal("small")
                plan = build_universal_plan(
                    fixture["universal_contexts"],
                    fixture["candidates"],
                    fixture["actions"],
                    fixture["model_spec_sha256"],
                )
                changed = build_universal_procedure_records(
                    plan=plan, minimal_bridge=build_fixture_bridge(plan)
                )
                self.assertNotEqual(changed, universal)
                self.assert_rejected(universal=changed)

    def test_cross_catalog_selected_recipe_mismatch(self):
        fixture = small_fixture()
        for compact in fixture["universal_contexts"]:
            if compact["context_id"] == "CTX-D0":
                compact["selected_recipe_id"] = "D3"
        plan = build_universal_plan(
            fixture["universal_contexts"],
            fixture["candidates"],
            fixture["actions"],
            fixture["model_spec_sha256"],
        )
        changed = build_universal_procedure_records(
            plan=plan, minimal_bridge=build_fixture_bridge(plan)
        )
        self.assert_rejected(universal=changed)

    def test_missing_context(self):
        fixture = small_fixture()
        fixture["universal_contexts"] = fixture["universal_contexts"][:-1]
        plan = build_universal_plan(
            fixture["universal_contexts"],
            fixture["candidates"],
            fixture["actions"],
            fixture["model_spec_sha256"],
        )
        changed = build_universal_procedure_records(
            plan=plan, minimal_bridge=build_fixture_bridge(plan)
        )
        self.assert_rejected(universal=changed)

    def test_missing_model_group(self):
        universal = base_universal("small")
        clone = copy.deepcopy(universal)
        index = next(
            index
            for index, record in enumerate(clone["records"])
            if record.get("context_id") == "CTX-D0"
        )
        del clone["records"][index]
        clone = rehash_universal(clone)
        self.assert_rejected(universal=clone)

    def test_extra_universal_context_record(self):
        universal = base_universal("small")
        clone = copy.deepcopy(universal)
        template = copy.deepcopy(
            next(record for record in clone["records"] if record["policy_id"] == "PP-U-SG")
        )
        template["context_id"] = "CTX-SG-INVENTED"
        payload = {key: value for key, value in template.items() if key != "procedure_id"}
        template["procedure_id"] = UNIV_PROC_PREFIX + canonical_sha256(payload)

        original_records = copy.deepcopy(clone["records"])
        original_aliases = copy.deepcopy(clone.get("strategy_aliases"))

        clone["records"].append(template)
        clone = rehash_universal(clone)

        self.assertNotEqual(clone, universal)
        self.assertEqual(len(clone["records"]), len(original_records) + 1)
        self.assertEqual(clone["records"][: len(original_records)], original_records)
        self.assertEqual(clone.get("strategy_aliases"), original_aliases)
        for record in clone["records"]:
            payload = {key: value for key, value in record.items() if key != "procedure_id"}
            self.assertEqual(record["procedure_id"], UNIV_PROC_PREFIX + canonical_sha256(payload))
        self.assert_rejected(universal=clone)

    def test_extra_universal_strategy_alias(self):
        universal = base_universal("small")
        clone = copy.deepcopy(universal)
        aliases = clone["strategy_aliases"]
        self.assertTrue(aliases)
        extra = copy.deepcopy(aliases[0])
        extra["upstream_alias_id"] = "P08SYNTH-UNIQUE-UPSTREAM"
        aliases.append(extra)
        clone = rehash_universal(clone)
        self.assert_rejected(universal=clone)

    def test_duplicate_universal_upstream_alias_id(self):
        universal = base_universal("small")
        clone = copy.deepcopy(universal)
        aliases = clone["strategy_aliases"]
        self.assertGreaterEqual(len(aliases), 2)
        aliases[1]["upstream_alias_id"] = aliases[0]["upstream_alias_id"]
        clone = rehash_universal(clone)
        self.assert_rejected(universal=clone)

    def test_partial_fallback_context(self):
        qc = base_qc("small")
        changed = rebuild_qc_alias(
            qc,
            lambda alias: alias["context_id"] == "CTX-THIN" and alias["strategy"] == "D0-M",
            reason_code="eligible",
        )
        self.assert_rejected(qc=changed)

    def test_alias_extra_extratrees(self):
        qc = base_qc("small")
        changed = rebuild_qc_alias(
            qc,
            lambda alias: alias["context_id"] == "CTX-D0" and alias["strategy"] == "D0-M",
            strategy="C-EXTRA-TREES",
        )
        self.assert_rejected(qc=changed)

    def test_duplicate_alias_and_block_payloads(self):
        qc = base_qc("small")
        duplicated_alias = {
            key: copy.deepcopy(value) for key, value in qc.items() if key != "catalog_sha256"
        }
        duplicated_alias["aliases"].append(copy.deepcopy(duplicated_alias["aliases"][0]))
        duplicated_alias["catalog_sha256"] = canonical_sha256(duplicated_alias)
        self.assert_rejected(qc=duplicated_alias)

        duplicated_block = {
            key: copy.deepcopy(value) for key, value in qc.items() if key != "catalog_sha256"
        }
        duplicated_block["blocks"].append(copy.deepcopy(duplicated_block["blocks"][0]))
        duplicated_block["catalog_sha256"] = canonical_sha256(duplicated_block)
        self.assert_rejected(qc=duplicated_block)

    def test_duplicate_universal_procedure(self):
        universal = base_universal("small")
        clone = copy.deepcopy(universal)
        clone["records"].append(copy.deepcopy(clone["records"][0]))
        clone = rehash_universal(clone)
        self.assert_rejected(universal=clone)

    def test_route_dependency_missing_model_refit(self):
        qc = base_qc("small")
        stages = blocks_by_stage(qc, "CTX-D0")
        test_route = self.one(stages, "final_test_route")
        refit = self.one(stages, "final_refit", "C-RBF-SVM")

        def mutator(block, original_id, original_deps):
            if original_id == test_route["block_id"]:
                index = original_deps.index(refit["block_id"])
                del block["depends_on_blocks"][index]

        self.assert_rejected(qc=remap_qc(qc, mutator))

    def test_test_bearing_refit(self):
        qc = base_qc("small")
        stages = blocks_by_stage(qc, "CTX-D0")
        refit = self.one(stages, "final_refit", "C-RBF-SVM")
        outer_test = compact_qc(qc, "CTX-D0")["outer_test_uid_sha256"]

        def mutator(block, original_id, original_deps):
            if original_id == refit["block_id"]:
                block["test_uid_sha256"] = outer_test

        self.assert_rejected(qc=remap_qc(qc, mutator))

    def test_wrong_calibration_resolution(self):
        qc = base_qc("small")
        stages = blocks_by_stage(qc, "CTX-D0")
        scalar = self.one(stages, "final_scalar_calibration", "D0-M")

        def mutator(block, original_id, original_deps):
            if original_id == scalar["block_id"]:
                block["resolution"] = "bogus_calibration_resolution"

        self.assert_rejected(qc=remap_qc(qc, mutator))

    def test_missing_scalar_calibration_parents(self):
        for model in ("C-RBF-SVM", "D0-M"):
            with self.subTest(model=model):
                qc = base_qc("small")
                stages = blocks_by_stage(qc, "CTX-D0")
                scalar = self.one(stages, "final_scalar_calibration", model)
                self.assertTrue(scalar["depends_on_blocks"])

                def mutator(block, original_id, original_deps, target=scalar["block_id"]):
                    if original_id == target:
                        block["depends_on_blocks"] = []

                self.assert_rejected(qc=remap_qc(qc, mutator))

    def test_remap_qc_identity_is_noop(self):
        qc = base_qc("small")
        self.assertEqual(remap_qc(qc, lambda block, original_id, original_deps: None), qc)

    def test_threshold_resolution_guard(self):
        qc = base_qc("small")
        stages = blocks_by_stage(qc, "CTX-D0")
        threshold = self.one(stages, "final_refit_quantile_fit")
        self.assertEqual(threshold["resolution"], "calibration_quantile_fit_role_only")

        def mutator(block, original_id, original_deps):
            if original_id == threshold["block_id"]:
                block["resolution"] = "bogus_threshold_resolution"

        self.assert_rejected(qc=remap_qc(qc, mutator))

    def test_wrong_scalar_calibration_parent_hash(self):
        qc = base_qc("small")
        stages = blocks_by_stage(qc, "CTX-D0")

        def dependency_for(model_id, parent_stage):
            scalar = self.one(stages, "final_scalar_calibration", model_id)
            matches = [
                block
                for block in stages[parent_stage]
                if block["model_id"] == model_id
                and block["block_id"] in scalar["depends_on_blocks"]
            ]
            self.assertTrue(
                matches,
                f"no {parent_stage} dependency on {model_id} final_scalar_calibration",
            )
            parent = matches[0]
            self.assertIn(parent["block_id"], scalar["depends_on_blocks"])
            return parent

        def locate(catalog, model_id, stage, role_id):
            for block in blocks_by_stage(catalog, "CTX-D0")[stage]:
                if block["model_id"] == model_id and block.get("role_id") == role_id:
                    return block
            raise AssertionError(f"missing {model_id}/{stage}/{role_id} block")

        for model_id, parent_stage in (
            ("C-RBF-SVM", "final_calibration_model_prediction"),
            ("D0-M", "final_source_prediction"),
        ):
            parent = dependency_for(model_id, parent_stage)
            parent_id = parent["block_id"]
            role_id = parent.get("role_id")

            for field in ("fit_uid_sha256", "validation_uid_sha256"):
                other_field = (
                    "validation_uid_sha256" if field == "fit_uid_sha256" else "fit_uid_sha256"
                )
                original_other = parent[other_field]
                replacement = hashlib.sha256(
                    f"p08-t222-wrong-parent-{model_id}-{field}".encode()
                ).hexdigest()

                def mutator(
                    block,
                    original_id,
                    original_deps,
                    _parent=parent_id,
                    _field=field,
                    _new=replacement,
                ):
                    if original_id != _parent:
                        return
                    block[_field] = _new

                with self.subTest(model=model_id, field=field):
                    self.assertNotEqual(parent[field], replacement)

                    mutated = remap_qc(qc, mutator)

                    # Re-sealed graph must pass the eager validator; the
                    # returned iterator is lazy and must not be consumed.
                    iter_slots(mutated)

                    target = locate(mutated, model_id, parent_stage, role_id)
                    self.assertEqual(target["stage"], parent_stage)
                    self.assertEqual(target[field], replacement)
                    self.assertEqual(target[other_field], original_other)

                    self.assert_rejected(qc=mutated)

    def test_qc_extra_model_stage_and_orphan_select_rejected(self):
        qc = base_qc("small")
        stages = blocks_by_stage(qc, "CTX-D0")
        base_ids = {block["block_id"] for block in qc["blocks"]}

        test_route = self.one(stages, "final_test_route")
        extra_test_route = clone_block(test_route, model_id="D0-M")
        extras = seal_catalog(
            qc["bindings"],
            list(qc["blocks"]) + [extra_test_route],
            qc["aliases"],
        )
        self.assertTrue(base_ids <= {b["block_id"] for b in extras["blocks"]})
        with self.subTest(case="extra_shared_stage_block"):
            self.assert_rejected(qc=extras)

        template = next(
            (
                block
                for block in qc["blocks"]
                if block["stage"] == "final_select_hyperparameters"
                and block["context_id"] == "CTX-D0"
            ),
            None,
        )
        if template is None:
            template = self.one(stages, "final_refit", "C-RBF-SVM")
        orphan = clone_block(
            template,
            stage="final_select_hyperparameters",
            model_id="C-EXTRA-TREES",
            depends_on_blocks=[],
        )
        orphan_catalog = seal_catalog(
            qc["bindings"],
            list(qc["blocks"]) + [orphan],
            qc["aliases"],
        )
        self.assertTrue(base_ids <= {b["block_id"] for b in orphan_catalog["blocks"]})
        with self.subTest(case="orphan_select_hyperparameters"):
            self.assert_rejected(qc=orphan_catalog)

    def test_malformed_seed_bool(self):
        qc = base_qc("small")
        target = next(block for block in qc["blocks"] if block["stage"] == "final_refit")

        def mutator(block):
            block["axes"]["seed"] = [True]

        forged = force_block_change(qc, target["block_id"], mutator)
        # The block constructor refuses boolean seeds; this payload bypasses it
        # and is rejected at the adapter validation boundary.
        self.assert_rejected(qc=forged)


if __name__ == "__main__":
    unittest.main()
