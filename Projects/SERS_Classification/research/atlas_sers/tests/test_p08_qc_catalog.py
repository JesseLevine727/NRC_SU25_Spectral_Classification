"""Tests for the P08-T022 no-fit QC catalog planner."""

from __future__ import annotations

import copy
import hashlib
import unittest

from atlas_sers.evaluation.p08_qc_blocks import (
    FALLBACK_TARGET,
    BlockError,
    canonical_sha256,
    iter_slots,
    seal_catalog,
)
from atlas_sers.evaluation.p08_qc_catalog import (
    QCPlanError,
    build_qc_catalog,
    require_scientific_execution,
)
from atlas_sers.evaluation.p08_qc_roles import build_nested_qc_roles
from tests.test_p08_plan import make_actions, make_candidates, make_model_spec
from tests.test_p08_qc_roles import Corpus, add_master_context, add_pseudo_context

SENTINEL = "SENTINEL-PRIVATE-424242"
GATE_LIBRARY_SHA256 = hashlib.sha256(b"p08-t022-gate-library").hexdigest()
ALIAS_STRATEGIES = {"C-RBF-SVM", "C-RANDOM-FOREST", "D0-M", "P05-SELECTED"}
METADATA_KEYS = {
    "binding_sha256",
    "outer_fit_uid_sha256",
    "outer_test_uid_sha256",
    "minimal_array_sha256",
    "model_spec_sha256",
}
BLOCK_KEYS = {
    "stage",
    "block_id",
    "context_id",
    "model_id",
    "role_id",
    "fit_uid_sha256",
    "validation_uid_sha256",
    "test_uid_sha256",
    "axes",
    "depends_on_blocks",
    "resolution",
    "slot_count",
    "binding_sha256",
    "schema_version",
}
BINDING_KEYS = {
    "protocol_namespace",
    "nested_registry_sha256",
    "universal_contexts",
    "candidates",
    "actions",
    "model_spec_sha256",
    "gate_library_sha256",
}
NA = "not_applicable"
FINAL_STAGES = {
    "final_test_route",
    "final_held_prediction",
    "final_seed_ensemble_prediction",
}
MODEL_FIT_STAGES = (
    "inner_source_fit",
    "policy_refit",
    "final_source_fit",
    "final_calibration_model_fit",
    "final_refit",
)
SCALAR_STAGES = ("policy_scalar_calibration", "final_scalar_calibration")


def _unit_hashes(unit):
    return {
        "unit_id": unit["unit_id"],
        "fit_uid_sha256": canonical_sha256(sorted(unit["fit_uids"])),
        "validation_uid_sha256": canonical_sha256(sorted(unit["validation_uids"])),
    }


def _outer(raw):
    return {
        "context_id": raw["context_id"],
        "selection_mode": raw["selection_mode"],
        "outer_fit_uid_sha256": canonical_sha256(sorted(raw["outer_fit_uids"])),
        "outer_test_uid_sha256": canonical_sha256(sorted(raw["outer_test_uids"])),
    }


def compact_pseudo(raw, recipe):
    compact = _outer(raw)
    compact["selected_recipe_id"] = recipe
    compact["selection_units"] = [_unit_hashes(unit) for unit in raw["selection_units"]]
    compact["calibration_units"] = [
        {
            "unit_id": f"{raw['context_id']}-cal-{index}",
            "fit_uid_sha256": canonical_sha256(
                [f"t022::{raw['context_id']}::cf::{index}"]
            ),
            "validation_uid_sha256": canonical_sha256(
                [f"t022::{raw['context_id']}::cv::{index}"]
            ),
        }
        for index in range(3)
    ]
    return compact


def compact_master(raw):
    compact = _outer(raw)
    compact["selected_recipe_id"] = "D0-M"
    compact["selection_units"] = [_unit_hashes(unit) for unit in raw["selection_units"]]
    compact["calibration_units"] = [dict(unit) for unit in compact["selection_units"]]
    return compact


def build_fixture(entries):
    corpus = Corpus()
    contexts = []
    universal = []
    for entry in entries:
        if entry["kind"] == "master":
            raw = add_master_context(corpus, entry["context_id"])
            universal.append(compact_master(raw))
        else:
            raw = add_pseudo_context(
                corpus,
                entry["context_id"],
                per_class=entry.get("per_class", 3),
                n_units=entry.get("n_units", 2),
            )
            universal.append(compact_pseudo(raw, entry["recipe"]))
        contexts.append(raw)
    return {
        "contexts": contexts,
        "roles": corpus.roles,
        "universal_contexts": universal,
        "candidates": make_candidates(),
        "actions": make_actions(),
        "model_spec_sha256": make_model_spec(),
        "gate_library_sha256": GATE_LIBRARY_SHA256,
    }


def build(fixture, **changes):
    inputs = dict(fixture)
    inputs.update(changes)
    return build_qc_catalog(
        inputs["contexts"],
        inputs["roles"],
        inputs["universal_contexts"],
        inputs["candidates"],
        inputs["actions"],
        inputs["model_spec_sha256"],
        inputs["gate_library_sha256"],
    )


def compact_for(fixture, context_id):
    return next(
        compact
        for compact in fixture["universal_contexts"]
        if compact["context_id"] == context_id
    )


def binding_variants():
    candidates = make_candidates()
    candidates[0] = dict(
        candidates[0],
        hyperparameter_sha256=hashlib.sha256(b"mutated").hexdigest(),
    )
    return {
        "gate": {"gate_library_sha256": hashlib.sha256(b"other-gate").hexdigest()},
        "actions": {"actions": make_actions("mutated")},
        "spec": {"model_spec_sha256": make_model_spec("mutated")},
        "candidates": {"candidates": candidates},
    }


def small_fixture():
    return build_fixture(
        [
            {"kind": "pseudo", "context_id": "CTX-D0", "recipe": "D0-M"},
            {"kind": "pseudo", "context_id": "CTX-D3", "recipe": "D3"},
            {"kind": "master", "context_id": "CTX-MASTER"},
            {"kind": "pseudo", "context_id": "CTX-THIN", "recipe": "D1", "per_class": 1},
        ]
    )


def fallback_fixture():
    return build_fixture(
        [
            {"kind": "master", "context_id": "CTX-M1"},
            {"kind": "pseudo", "context_id": "CTX-T1", "recipe": "D1", "per_class": 1},
        ]
    )


def synthetic_fixture():
    entries = []
    for index in range(40):
        entries.append(
            {"kind": "pseudo", "context_id": f"EL-D0-{index:03d}", "recipe": "D0-M"}
        )
    entries.append({"kind": "pseudo", "context_id": "EL-D1-000", "recipe": "D1"})
    for index in range(7):
        entries.append(
            {"kind": "pseudo", "context_id": f"EL-D2-{index:03d}", "recipe": "D2"}
        )
    for index in range(6):
        entries.append(
            {"kind": "pseudo", "context_id": f"EL-D3-{index:03d}", "recipe": "D3"}
        )
    for index in range(74):
        entries.append(
            {
                "kind": "pseudo",
                "context_id": f"UN-{index:03d}",
                "recipe": "D1",
                "per_class": 1,
            }
        )
    for index in range(132):
        entries.append({"kind": "master", "context_id": f"MC-{index:03d}"})
    return build_fixture(entries)


def aliases_for(catalog, context_id):
    return [a for a in catalog["aliases"] if a["context_id"] == context_id]


def alias_map(catalog, context_id):
    return {a["strategy"]: a for a in aliases_for(catalog, context_id)}


def blocks_for(catalog, context_id):
    return [b for b in catalog["blocks"] if b["context_id"] == context_id]


def context_ids(items):
    return {item["context_id"] for item in items}


class _Base(unittest.TestCase):
    def assert_qc_error(self, fixture=None, **changes):
        with self.assertRaises(QCPlanError) as caught:
            build(small_fixture() if fixture is None else fixture, **changes)
        self.assertNotIn(SENTINEL, str(caught.exception))
        return caught.exception


class CatalogStructureTests(_Base):
    def test_supported_and_fallback_structure(self):
        catalog = build(small_fixture())
        self.assertEqual(
            catalog["schema_version"], "nato-sers-p08-qc-block-catalog-v1"
        )
        self.assertFalse(catalog["execution_authorized"])
        self.assertEqual(FALLBACK_TARGET, "PP-U-MIN-COMPLETE-PIPELINE")
        self.assertEqual(set(catalog["bindings"]), BINDING_KEYS)
        self.assertEqual(
            catalog["bindings"]["protocol_namespace"], "nato-sers-p08-qc-catalog-v1"
        )
        self.assertNotIn("roles", catalog["bindings"])
        self.assertNotIn("contexts", catalog["bindings"])
        self.assertEqual(len(catalog["aliases"]), 16)
        for block in catalog["blocks"]:
            self.assertEqual(set(block), BLOCK_KEYS)
            self.assertTrue(block["block_id"].startswith("P08QCBLOCK-"))
            self.assertNotEqual(block["model_id"], "C-EXTRA-TREES")
        self.assertEqual(
            context_ids(catalog["blocks"]), {"CTX-D0", "CTX-D3"}
        )
        block_ids = {block["block_id"] for block in catalog["blocks"]}
        spec = make_model_spec()
        for alias in catalog["aliases"]:
            self.assertEqual(alias["strategy"] in ALIAS_STRATEGIES, True)
            self.assertEqual(set(alias["metadata"]), METADATA_KEYS)
            for key in ("binding_sha256", "outer_fit_uid_sha256", "outer_test_uid_sha256"):
                self.assertRegex(alias["metadata"][key], r"^[0-9a-f]{64}$")
            if alias["context_id"] in ("CTX-D0", "CTX-D3"):
                self.assertEqual(alias["evidence_status"], "unapproved_future_job")
                self.assertEqual(alias["reason_code"], "eligible")
                self.assertIn(alias["target_block_id"], block_ids)
                self.assertEqual(
                    alias["metadata"]["model_spec_sha256"],
                    spec[alias["recipe_id"]],
                )
            else:
                self.assertEqual(alias["target_block_id"], FALLBACK_TARGET)
                self.assertEqual(
                    alias["evidence_status"],
                    "requires_authenticated_minimal_endpoint",
                )
                self.assertTrue(alias["reason_code"])
        d0 = alias_map(catalog, "CTX-D0")
        d3 = alias_map(catalog, "CTX-D3")
        self.assertEqual(set(d0), ALIAS_STRATEGIES)
        self.assertEqual(set(d3), ALIAS_STRATEGIES)
        self.assertEqual(d0["D0-M"]["recipe_id"], "D0-M")
        self.assertEqual(d0["P05-SELECTED"]["recipe_id"], "D0-M")
        self.assertEqual(
            d0["D0-M"]["target_block_id"], d0["P05-SELECTED"]["target_block_id"]
        )
        self.assertEqual(d3["P05-SELECTED"]["recipe_id"], "D3")
        self.assertNotEqual(
            d3["D0-M"]["target_block_id"], d3["P05-SELECTED"]["target_block_id"]
        )
        self.assertNotEqual(
            d0["P05-SELECTED"]["target_block_id"], d3["P05-SELECTED"]["target_block_id"]
        )
        self.assertEqual(
            {alias["recipe_id"] for alias in aliases_for(catalog, "CTX-D0")},
            {"C-RBF-SVM", "C-RANDOM-FOREST", "D0-M"},
        )
        self.assertEqual(
            {alias["recipe_id"] for alias in aliases_for(catalog, "CTX-D3")},
            {"C-RBF-SVM", "C-RANDOM-FOREST", "D0-M", "D3"},
        )
        self.assertEqual(
            {
                alias["strategy"]
                for alias in catalog["aliases"]
                if alias["context_id"] in ("CTX-D0", "CTX-D3")
            },
            ALIAS_STRATEGIES,
        )

    def test_alias_metadata_matches_compact_and_recipe(self):
        fixture = small_fixture()
        catalog = build(fixture)
        spec = make_model_spec()
        compact_by_id = {
            compact["context_id"]: compact
            for compact in catalog["bindings"]["universal_contexts"]
        }
        fresh = build_nested_qc_roles(fixture["contexts"], fixture["roles"])
        fresh_by_id = {
            entry["context_id"]: entry for entry in fresh["contexts"]
        }
        by_id = {block["block_id"]: block for block in catalog["blocks"]}
        strategies_by_context = {}
        for alias in catalog["aliases"]:
            strategies_by_context.setdefault(alias["context_id"], set()).add(
                alias["strategy"]
            )
        for context_id, strategies in strategies_by_context.items():
            self.assertEqual(strategies, ALIAS_STRATEGIES)
            self.assertEqual(len(aliases_for(catalog, context_id)), 4)
        for alias in catalog["aliases"]:
            compact = compact_by_id[alias["context_id"]]
            metadata = alias["metadata"]
            entry = fresh_by_id[alias["context_id"]]
            expected_recipe = (
                compact["selected_recipe_id"]
                if alias["strategy"] == "P05-SELECTED"
                else alias["strategy"]
            )
            self.assertEqual(set(metadata), METADATA_KEYS)
            self.assertEqual(alias["recipe_id"], expected_recipe)
            self.assertEqual(
                metadata["outer_fit_uid_sha256"], compact["outer_fit_uid_sha256"]
            )
            self.assertEqual(
                metadata["outer_test_uid_sha256"], compact["outer_test_uid_sha256"]
            )
            self.assertEqual(metadata["model_spec_sha256"], spec[alias["recipe_id"]])
            self.assertEqual(
                metadata["binding_sha256"], canonical_sha256(catalog["bindings"])
            )
            self.assertEqual(
                metadata["minimal_array_sha256"],
                catalog["bindings"]["actions"]["R_MIN_400_1800"],
            )
            if entry["eligible"]:
                self.assertEqual(alias["reason_code"], "eligible")
                self.assertEqual(alias["evidence_status"], "unapproved_future_job")
                target = by_id[alias["target_block_id"]]
                self.assertEqual(target["stage"], "final_seed_ensemble_prediction")
                self.assertEqual(target["context_id"], alias["context_id"])
                self.assertEqual(target["model_id"], expected_recipe)
            else:
                self.assertEqual(alias["reason_code"], entry["reason_code"])
                self.assertEqual(
                    alias["evidence_status"],
                    "requires_authenticated_minimal_endpoint",
                )


class SealAndInvarianceTests(_Base):
    def test_seal_roundtrip_and_fresh_nested_sha(self):
        fixture = small_fixture()
        catalog = build(fixture)
        sealed = seal_catalog(
            catalog["bindings"], catalog["blocks"], catalog["aliases"]
        )
        self.assertEqual(sealed, catalog)
        fresh = build_nested_qc_roles(fixture["contexts"], fixture["roles"])
        self.assertEqual(
            catalog["bindings"]["nested_registry_sha256"], fresh["registry_sha256"]
        )
        self.assertEqual(
            fresh["registry_sha256"],
            canonical_sha256(
                {
                    key: value
                    for key, value in fresh.items()
                    if key != "registry_sha256"
                }
            ),
        )
        self.assertRegex(
            catalog["bindings"]["nested_registry_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertEqual(
            catalog["bindings"]["gate_library_sha256"], GATE_LIBRARY_SHA256
        )
        for compact in catalog["bindings"]["universal_contexts"]:
            self.assertIn("selected_recipe_id", compact)

    def test_ordered_invariance_and_non_mutation(self):
        fixture = small_fixture()
        baseline = build(fixture)
        snapshot = copy.deepcopy(fixture)
        build(fixture)
        self.assertEqual(fixture, snapshot)
        reordered = copy.deepcopy(fixture)
        reordered["contexts"].reverse()
        reordered["universal_contexts"].reverse()
        reordered["roles"].reverse()
        reordered["candidates"] = list(reversed(reordered["candidates"]))
        for raw in reordered["contexts"]:
            raw["outer_fit_uids"].reverse()
            raw["outer_test_uids"].reverse()
            raw["selection_units"].reverse()
            raw["classes"].reverse()
            for unit in raw["selection_units"]:
                unit["fit_uids"].reverse()
                unit["validation_uids"].reverse()
        for compact in reordered["universal_contexts"]:
            compact["selection_units"].reverse()
            compact["calibration_units"].reverse()
        changed = build(reordered)
        self.assertEqual(baseline, changed)

    def test_binding_identity_changes(self):
        baseline = build(small_fixture())
        base_fallback = aliases_for(baseline, "CTX-MASTER")[0]
        for name, change in binding_variants().items():
            with self.subTest(name=name):
                changed = build(small_fixture(), **change)
                self.assertNotEqual(
                    baseline["catalog_sha256"], changed["catalog_sha256"]
                )
                changed_fallback = aliases_for(changed, "CTX-MASTER")[0]
                self.assertNotEqual(
                    base_fallback["metadata"]["binding_sha256"],
                    changed_fallback["metadata"]["binding_sha256"],
                )

    def test_fallback_only_still_bound(self):
        catalog = build(fallback_fixture())
        self.assertEqual(catalog["blocks"], [])
        self.assertEqual(catalog["summary"]["block_count"], 0)
        self.assertEqual(len(catalog["aliases"]), 8)
        self.assertTrue(catalog["bindings"])
        for alias in catalog["aliases"]:
            self.assertEqual(alias["target_block_id"], FALLBACK_TARGET)

    def test_fallback_only_mutations_change_binding(self):
        baseline = build(fallback_fixture())
        self.assertEqual(baseline["blocks"], [])
        base_bindings = {
            (alias["context_id"], alias["strategy"]): alias["metadata"][
                "binding_sha256"
            ]
            for alias in baseline["aliases"]
        }
        base_alias_ids = {alias["alias_id"] for alias in baseline["aliases"]}
        for name, change in binding_variants().items():
            with self.subTest(variant=name):
                changed = build(fallback_fixture(), **change)
                self.assertEqual(changed["blocks"], [])
                self.assertNotEqual(
                    baseline["catalog_sha256"], changed["catalog_sha256"]
                )
                changed_bindings = {
                    (alias["context_id"], alias["strategy"]): alias["metadata"][
                        "binding_sha256"
                    ]
                    for alias in changed["aliases"]
                }
                changed_alias_ids = {
                    alias["alias_id"] for alias in changed["aliases"]
                }
                self.assertEqual(set(changed_bindings), set(base_bindings))
                self.assertTrue(base_alias_ids.isdisjoint(changed_alias_ids))
                for key, value in base_bindings.items():
                    self.assertNotEqual(changed_bindings[key], value)


class InvalidInputTests(_Base):
    def test_invalid_membership_and_hashes(self):
        bad = copy.deepcopy(small_fixture())
        bad["universal_contexts"][0]["context_id"] = "CTX-OTHER"
        self.assert_qc_error(bad)

        bad = copy.deepcopy(small_fixture())
        bad["contexts"][0]["context_id"] = "CTX-OTHER"
        self.assert_qc_error(bad)

        bad = copy.deepcopy(small_fixture())
        bad["universal_contexts"][0]["selection_mode"] = "bogus_mode"
        self.assert_qc_error(bad)

        bad = copy.deepcopy(small_fixture())
        bad["universal_contexts"][0]["outer_test_uid_sha256"] = "f" * 64
        self.assert_qc_error(bad)

        bad = copy.deepcopy(small_fixture())
        bad["universal_contexts"][0]["selection_units"][0][
            "fit_uid_sha256"
        ] = "e" * 64
        self.assert_qc_error(bad)

        bad = copy.deepcopy(small_fixture())
        bad["contexts"][0]["selection_units"][0]["fit_uids"].pop()
        self.assert_qc_error(bad)

    def test_invalid_context_sets(self):
        self.assert_qc_error(build_fixture([]))

        bad = copy.deepcopy(small_fixture())
        bad["contexts"].append(copy.deepcopy(bad["contexts"][0]))
        bad["universal_contexts"].append(copy.deepcopy(bad["universal_contexts"][0]))
        self.assert_qc_error(bad)

    def test_invalid_calibration(self):
        bad = copy.deepcopy(small_fixture())
        master = next(
            c for c in bad["universal_contexts"] if c["selection_mode"] == "master_cv"
        )
        master["calibration_units"][0]["validation_uid_sha256"] = "d" * 64
        self.assert_qc_error(bad)

        bad = copy.deepcopy(small_fixture())
        pseudo = next(
            c
            for c in bad["universal_contexts"]
            if c["selection_mode"] == "pseudo_domain"
        )
        pseudo["calibration_units"] = pseudo["calibration_units"][:1]
        self.assert_qc_error(bad)

        bad = copy.deepcopy(small_fixture())
        pseudo = next(
            c
            for c in bad["universal_contexts"]
            if c["selection_mode"] == "pseudo_domain"
        )
        pseudo["calibration_units"][1] = copy.deepcopy(pseudo["calibration_units"][0])
        self.assert_qc_error(bad)

    def test_fallback_raw_compact_alignment(self):
        def mutate_outer(compact, key):
            compact[key] = canonical_sha256([f"mutated::{key}"])

        def mutate_unit(compact, key, mirror):
            if key == "unit_id":
                compact["selection_units"][0][key] = "MUTATED-UNIT"
            else:
                compact["selection_units"][0][key] = canonical_sha256(
                    [f"mutated::{key}"]
                )
            if mirror:
                compact["calibration_units"][0][key] = compact["selection_units"][0][
                    key
                ]

        for context_id, mirror in (("CTX-M1", True), ("CTX-T1", False)):
            for key in ("outer_fit_uid_sha256", "outer_test_uid_sha256"):
                with self.subTest(context=context_id, mutation=key):
                    fixture = fallback_fixture()
                    mutate_outer(compact_for(fixture, context_id), key)
                    self.assert_qc_error(fixture)
            for key in ("unit_id", "fit_uid_sha256", "validation_uid_sha256"):
                with self.subTest(context=context_id, mutation=key):
                    fixture = fallback_fixture()
                    mutate_unit(compact_for(fixture, context_id), key, mirror)
                    self.assert_qc_error(fixture)

    def test_fallback_allowed_mode_mismatch_rejected(self):
        fixture = fallback_fixture()
        raw_master = next(
            context
            for context in fixture["contexts"]
            if context["selection_mode"] == "master_cv"
        )
        pseudo_compact = compact_pseudo(
            dict(raw_master, selection_mode="pseudo_domain"), "D0-M"
        )
        for index, compact in enumerate(fixture["universal_contexts"]):
            if compact["context_id"] == raw_master["context_id"]:
                fixture["universal_contexts"][index] = pseudo_compact
        self.assert_qc_error(fixture)

    def test_invalid_assembler_inputs(self):
        for name, change in (
            ("candidates", {"candidates": None}),
            ("actions", {"actions": None}),
            ("spec", {"model_spec_sha256": None}),
        ):
            with self.subTest(input=name):
                self.assert_qc_error(small_fixture(), **change)

    def test_raw_leakage(self):
        bad = copy.deepcopy(small_fixture())
        master = next(c for c in bad["contexts"] if c["selection_mode"] == "master_cv")
        leaked_uid = master["selection_units"][0]["validation_uids"][0]
        for role in bad["roles"]:
            if role["observation_uid"] == leaked_uid:
                role["instrument"] = master["held_instrument"]
        self.assert_qc_error(bad)

        bad = copy.deepcopy(small_fixture())
        master = next(c for c in bad["contexts"] if c["selection_mode"] == "master_cv")
        unit = master["selection_units"][0]
        fit_uid = unit["fit_uids"][0]
        validation_uid = unit["validation_uids"][0]
        fit_master = next(
            role["master_id"]
            for role in bad["roles"]
            if role["observation_uid"] == fit_uid
        )
        for role in bad["roles"]:
            if role["observation_uid"] == validation_uid:
                role["master_id"] = fit_master
        self.assert_qc_error(bad)

        bad = copy.deepcopy(small_fixture())
        first = bad["contexts"][0]["selection_units"][0]["fit_uids"][0]
        bad["roles"] = [
            role for role in bad["roles"] if role["observation_uid"] != first
        ]
        self.assert_qc_error(bad)

    def test_repeated_master_across_contexts_allowed(self):
        fixture = small_fixture()
        first = fixture["contexts"][0]["selection_units"][0]["fit_uids"][0]
        second = fixture["contexts"][1]["selection_units"][0]["fit_uids"][0]
        shared_master = next(
            role["master_id"]
            for role in fixture["roles"]
            if role["observation_uid"] == first
        )
        for role in fixture["roles"]:
            if role["observation_uid"] == second:
                role["master_id"] = shared_master
        catalog = build(fixture)
        self.assertTrue(catalog["blocks"])

    def test_bad_gate_library(self):
        for bad_gate in (None, True, "A" * 64, "a" * 63 + " ", "\ud800" * 64, "a" * 63):
            with self.subTest(gate=repr(bad_gate)):
                self.assert_qc_error(
                    small_fixture(), gate_library_sha256=bad_gate
                )

    def test_static_payload_and_unknown_code(self):
        renamed = copy.deepcopy(small_fixture())
        renamed["contexts"][0]["context_id"] = SENTINEL
        renamed["universal_contexts"][0]["context_id"] = SENTINEL
        self.assertTrue(build(renamed)["blocks"])

        bad = copy.deepcopy(small_fixture())
        bad["contexts"][0]["context_id"] = SENTINEL
        self.assert_qc_error(bad)

        error = QCPlanError(SENTINEL)
        self.assertNotIn(SENTINEL, str(error))
        self.assertNotIn(SENTINEL, repr(error))
        self.assertNotIn(SENTINEL, getattr(error, "code", ""))


class DependencyTests(_Base):
    def test_combined_block_ancestor_walk(self):
        catalog = build(small_fixture())
        by_id = {block["block_id"]: block for block in catalog["blocks"]}
        for block in catalog["blocks"]:
            for parent in block["depends_on_blocks"]:
                self.assertIn(parent, by_id)

        def ancestors(block_id):
            seen = set()
            stack = list(by_id[block_id]["depends_on_blocks"])
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                stack.extend(by_id[current]["depends_on_blocks"])
            return seen

        compact_by_id = {
            compact["context_id"]: compact
            for compact in catalog["bindings"]["universal_contexts"]
        }
        for context_id in ("CTX-D0", "CTX-D3"):
            ctx_blocks = blocks_for(catalog, context_id)
            self.assertTrue(ctx_blocks)
            final_refits = {
                b["block_id"] for b in ctx_blocks if b["stage"] == "final_refit"
            }
            self.assertTrue(final_refits)
            quantile_fits = {
                b["block_id"]
                for b in ctx_blocks
                if b["stage"] == "final_refit_quantile_fit"
            }
            gate_selections = {
                b["block_id"] for b in ctx_blocks if b["stage"] == "gate_selection"
            }
            self.assertTrue(quantile_fits)
            self.assertTrue(gate_selections)
            route = [b for b in ctx_blocks if b["stage"] == "final_test_route"]
            self.assertEqual(len(route), 1)
            self.assertEqual(
                set(route[0]["depends_on_blocks"]),
                final_refits | quantile_fits | gate_selections,
            )
            outer_test = compact_by_id[context_id]["outer_test_uid_sha256"]
            protected = [b for b in ctx_blocks if b["stage"] not in FINAL_STAGES]
            self.assertTrue(protected)
            for block in protected:
                members = {block["block_id"]} | ancestors(block["block_id"])
                for member_id in members:
                    member = by_id[member_id]
                    self.assertEqual(member["test_uid_sha256"], NA)
                    self.assertEqual(member["context_id"], context_id)
                    self.assertNotEqual(member["fit_uid_sha256"], outer_test)
                    self.assertNotEqual(member["validation_uid_sha256"], outer_test)


class TamperTests(_Base):
    def test_iter_slots_rejects_tampering(self):
        catalog = build(small_fixture())
        forged = copy.deepcopy(catalog)
        forged["execution_authorized"] = True
        with self.assertRaises(BlockError):
            iter_slots(forged)

        summary = copy.deepcopy(catalog)
        summary["summary"]["block_count"] += 1
        with self.assertRaises(BlockError):
            iter_slots(summary)

        digest = copy.deepcopy(catalog)
        digest["catalog_sha256"] = "0" * 64
        with self.assertRaises(BlockError):
            iter_slots(digest)

        blocks = copy.deepcopy(catalog)
        blocks["blocks"][0]["slot_count"] += 1
        with self.assertRaises(BlockError):
            iter_slots(blocks)

    def test_require_scientific_execution_always_denies(self):
        for args in ((), ({"execution_authorized": True},)):
            with self.subTest(args=args):
                with self.assertRaises(QCPlanError) as caught:
                    require_scientific_execution(*args)
                self.assertEqual(
                    caught.exception.code, "scientific_execution_not_authorized"
                )


class SyntheticScaleTests(_Base):
    def test_synthetic_scale_compact_counts(self):
        catalog = build(synthetic_fixture())
        summary = catalog["summary"]
        self.assertEqual(summary["block_count"], 6282)
        self.assertEqual(len(catalog["blocks"]), 6282)
        self.assertEqual(summary["expanded_operation_slots"], 3471416)
        self.assertEqual(summary["alias_count"], 1040)
        self.assertEqual(len(catalog["aliases"]), 1040)
        stage_counts = summary["stage_counts"]
        self.assertEqual(
            sum(stage_counts[stage] for stage in MODEL_FIT_STAGES), 1630980
        )
        self.assertEqual(sum(stage_counts[stage] for stage in SCALAR_STAGES), 53880)
        self.assertEqual(
            sum(summary["stage_block_counts"].values()), summary["block_count"]
        )
        fallback = [
            alias
            for alias in catalog["aliases"]
            if alias["target_block_id"] == FALLBACK_TARGET
        ]
        self.assertEqual(len(fallback), 824)
        fallback_contexts = context_ids(fallback)
        self.assertEqual(len(fallback_contexts), 206)
        block_contexts = context_ids(catalog["blocks"])
        self.assertEqual(len(block_contexts), 54)
        self.assertFalse(fallback_contexts & block_contexts)
        self.assertEqual(len(context_ids(catalog["aliases"])), 260)
