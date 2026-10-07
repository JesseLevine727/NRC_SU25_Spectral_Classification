"""P08-T248 metadata-only tests for the stress contrast binding catalog.

Invented identities and the public prediction/score-support surface only.
No fits, predictions, scores, weights, resampling, routes or quantiles are
executed or produced: these tests only bind the frozen public inference
registry to the public parent support catalog and check structure, signs,
identities and independently recomputed hashes.  Scientific execution is
never authorized and no numerical result is accepted.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from atlas_sers.evaluation.p08_perturbation_score_support import (
    build_stress_score_support,
)
from atlas_sers.evaluation.p08_stress_contrasts import (
    build_stress_contrast_catalog,
    require_scientific_execution,
    validate_stress_contrast_catalog,
)
from tests.test_p08_perturbation_predictions import independent_hash
from tests.test_p08_perturbation_scores import (
    MIXED4,
    PSEUDO4,
    build_memberships,
    support_for,
)

INVALID = "invalid_stress_contrast_metadata"
DENIED = "scientific_execution_not_authorized"
SCHEMA = "nato-sers-p08-stress-contrast-catalog-v1"
REGISTRY_RELATIVE = Path("plan") / "contracts" / "p08_perturbation_inference.json"

ROOT_KEYS = {
    "schema_version",
    "execution_authorized",
    "scientific_operations",
    "artifact_provenance_independently_verified",
    "numerical_inference_accepted",
    "full_stress_job_ledger_complete",
    "score_catalog",
    "inference_registry",
    "platform_families",
    "master_weight_columns",
    "instrument_weight_columns",
    "support_records",
    "contrasts",
    "summary",
    "catalog_sha256",
}
CONTRAST_KEYS = {
    "contrast_id",
    "kind",
    "panel",
    "support_id",
    "disturbance_family",
    "endpoint",
    "multiplicity_family",
    "terms",
    "context_bindings",
    "pooled_bindings",
    "contrast_binding_sha256",
}
TERM_KEYS = {"coefficient", "policy_id", "strategy"}
CONTEXT_BINDING_KEYS = {"context_id", "view_ids", "target_procedure_ids"}
POOLED_BINDING_KEYS = {"pool_group_id", "view_ids", "target_pooled_procedure_ids"}
SUPPORT_KEYS = {
    "support_id",
    "context_ids",
    "complete_pool_group_ids",
    "master_weight_columns",
    "instrument_weight_columns",
    "domains",
    "instruments",
    "known_platform_families",
    "unknown_family_instruments",
}
SUMMARY_KEYS = {
    "contrast_count",
    "kind_counts",
    "multiplicity_family_counts",
    "support_contrast_counts",
    "context_comparison_units",
    "pooled_comparison_units",
    "signed_context_term_references",
    "signed_pooled_term_references",
    "global_masters",
    "global_instruments",
    "platform_family_mapping_sha256",
    "all_authorization_flags_false",
}

_REGISTRY = None


def registry():
    """The frozen public registry, read once; callers must deep-copy to mutate."""

    global _REGISTRY
    if _REGISTRY is None:
        path = Path(__file__).resolve().parents[1] / REGISTRY_RELATIVE
        _REGISTRY = json.loads(path.read_text(encoding="utf-8"))
    return _REGISTRY


def parent_for(entries=None):
    return support_for(entries)[1]


def support_ids(parent):
    supports = parent["supports"]
    operational = next(key for key in supports if "operational" in key)
    eligible = next(key for key in supports if key != operational and "qc" in key)
    return operational, eligible


def families_for(parent, value="toy-platform"):
    return {instrument: value for instrument in parent["global_instrument_ids"]}


def build(entries=None, *, registry_obj=None, platform_families=None):
    parent = parent_for(entries)
    return build_stress_contrast_catalog(
        score_catalog=parent,
        inference_registry=registry() if registry_obj is None else registry_obj,
        platform_families=(
            families_for(parent) if platform_families is None else platform_families
        ),
    )


_CACHE = {}


def default_catalog():
    if "default" not in _CACHE:
        _CACHE["default"] = build(None)
    return _CACHE["default"]


def fourfold_catalog():
    if "fourfold" not in _CACHE:
        _CACHE["fourfold"] = build(PSEUDO4)
    return _CACHE["fourfold"]


def _summary_counts(catalog):
    kind, multiplicity, supports = {}, {}, {}
    context_units = pooled_units = signed_context = signed_pooled = 0
    for contrast in catalog["contrasts"]:
        kind[contrast["kind"]] = kind.get(contrast["kind"], 0) + 1
        multiplicity[contrast["multiplicity_family"]] = (
            multiplicity.get(contrast["multiplicity_family"], 0) + 1
        )
        supports[contrast["support_id"]] = supports.get(contrast["support_id"], 0) + 1
        width = len(contrast["terms"])
        context_units += len(contrast["context_bindings"])
        pooled_units += len(contrast["pooled_bindings"])
        signed_context += width * len(contrast["context_bindings"])
        signed_pooled += width * len(contrast["pooled_bindings"])
    return {
        "contrast_count": len(catalog["contrasts"]),
        "kind_counts": kind,
        "multiplicity_family_counts": multiplicity,
        "support_contrast_counts": supports,
        "context_comparison_units": context_units,
        "pooled_comparison_units": pooled_units,
        "signed_context_term_references": signed_context,
        "signed_pooled_term_references": signed_pooled,
    }


class ContrastCatalogIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = default_catalog()
        cls.parent = parent_for(None)
        cls.registry = registry()

    def test_root_schema_and_denied_authority(self):
        self.assertEqual(set(self.catalog), ROOT_KEYS)
        self.assertEqual(self.catalog["schema_version"], SCHEMA)
        for flag in (
            "execution_authorized",
            "artifact_provenance_independently_verified",
            "numerical_inference_accepted",
            "full_stress_job_ledger_complete",
        ):
            self.assertIs(self.catalog[flag], False)
        self.assertEqual(self.catalog["scientific_operations"], 0)

    def test_root_digest_and_snapshots(self):
        catalog = self.catalog
        body = {name: value for name, value in catalog.items() if name != "catalog_sha256"}
        self.assertEqual(catalog["catalog_sha256"], independent_hash(body))
        self.assertEqual(catalog["inference_registry"], self.registry)
        self.assertIsNot(catalog["inference_registry"], self.registry)
        self.assertEqual(catalog["score_catalog"], self.parent)
        self.assertIsNot(catalog["score_catalog"], self.parent)

    def test_registry_contrasts_bind_by_id(self):
        declared = sorted(c["contrast_id"] for c in self.registry["contrasts"])
        emitted = [c["contrast_id"] for c in self.catalog["contrasts"]]
        self.assertEqual(emitted, declared)
        self.assertEqual(len(emitted), 456)
        self.assertEqual(len(set(emitted)), len(emitted))

    def test_contrast_keys_and_digests(self):
        for contrast in self.catalog["contrasts"]:
            self.assertEqual(set(contrast), CONTRAST_KEYS)
            body = {
                name: value for name, value in contrast.items() if name != "contrast_binding_sha256"
            }
            self.assertEqual(contrast["contrast_binding_sha256"], independent_hash(body))

    def test_terms_and_signs(self):
        source = {c["contrast_id"]: c for c in self.registry["contrasts"]}
        reference = self.registry["reference_policy"]
        for contrast in self.catalog["contrasts"]:
            original = source[contrast["contrast_id"]]
            terms = contrast["terms"]
            for term in terms:
                self.assertEqual(set(term), TERM_KEYS)
                self.assertIn(term["coefficient"], (-1, 1))
            if contrast["kind"] == "effect":
                self.assertEqual([t["coefficient"] for t in terms], [1, -1])
                self.assertEqual(terms[0]["policy_id"], reference)
                self.assertEqual(terms[1]["policy_id"], original["policy"])
                self.assertEqual(terms[0]["strategy"], terms[1]["strategy"])
                self.assertEqual(terms[0]["strategy"], original["method"])
            else:
                self.assertEqual([t["coefficient"] for t in terms], [1, -1, -1, 1])
                self.assertEqual(terms[0]["policy_id"], reference)
                self.assertEqual(terms[2]["policy_id"], reference)
                self.assertEqual(terms[1]["policy_id"], original["policy"])
                self.assertEqual(terms[3]["policy_id"], original["policy"])
                self.assertEqual(terms[0]["strategy"], terms[1]["strategy"])
                self.assertEqual(terms[2]["strategy"], terms[3]["strategy"])
                self.assertEqual(terms[0]["strategy"], original["neural"])
                self.assertEqual(terms[1]["strategy"], original["neural"])
                self.assertEqual(terms[2]["strategy"], original["classical"])
                self.assertEqual(terms[3]["strategy"], original["classical"])

    def test_context_and_pool_bindings_resolve(self):
        views = {}
        for view in self.parent["context_views"]:
            key = (view["context_id"], view["policy_id"], view["strategy"])
            self.assertNotIn(key, views)
            views[key] = view
        pooled = {}
        for view in self.parent["pooled_views"]:
            key = (
                view["support_id"],
                view["pool_group_id"],
                view["policy_id"],
                view["strategy"],
            )
            self.assertNotIn(key, pooled)
            pooled[key] = view
        for contrast in self.catalog["contrasts"]:
            for binding in contrast["context_bindings"]:
                self.assertEqual(set(binding), CONTEXT_BINDING_KEYS)
                self.assertEqual(len(binding["view_ids"]), len(contrast["terms"]))
                self.assertEqual(len(binding["target_procedure_ids"]), len(contrast["terms"]))
                for term, view_id, target in zip(
                    contrast["terms"],
                    binding["view_ids"],
                    binding["target_procedure_ids"],
                    strict=True,
                ):
                    view = views[(binding["context_id"], term["policy_id"], term["strategy"])]
                    self.assertEqual(view["view_id"], view_id)
                    self.assertEqual(view["target_procedure_id"], target)
            for binding in contrast["pooled_bindings"]:
                self.assertEqual(set(binding), POOLED_BINDING_KEYS)
                for term, view_id, target in zip(
                    contrast["terms"],
                    binding["view_ids"],
                    binding["target_pooled_procedure_ids"],
                    strict=True,
                ):
                    view = pooled[
                        (
                            contrast["support_id"],
                            binding["pool_group_id"],
                            term["policy_id"],
                            term["strategy"],
                        )
                    ]
                    self.assertEqual(view["view_id"], view_id)
                    self.assertEqual(view["target_pooled_procedure_id"], target)

    def test_fourfold_and_mixed_pool_bindings_resolve(self):
        for label, entries in (("pseudo4", PSEUDO4), ("mixed4", MIXED4)):
            with self.subTest(fixture=label):
                parent = parent_for(entries)
                catalog = build(entries)
                pooled = {}
                for view in parent["pooled_views"]:
                    key = (
                        view["support_id"],
                        view["pool_group_id"],
                        view["policy_id"],
                        view["strategy"],
                    )
                    self.assertNotIn(key, pooled)
                    pooled[key] = view
                groups = {
                    support_id: set(support["complete_pool_group_ids"])
                    for support_id, support in parent["supports"].items()
                }
                bindings_seen = 0
                signed_terms_seen = 0
                for contrast in catalog["contrasts"]:
                    self.assertIn(contrast["support_id"], groups)
                    for binding in contrast["pooled_bindings"]:
                        self.assertIn(
                            binding["pool_group_id"],
                            groups[contrast["support_id"]],
                        )
                        self.assertEqual(len(binding["view_ids"]), len(contrast["terms"]))
                        self.assertEqual(
                            len(binding["target_pooled_procedure_ids"]),
                            len(contrast["terms"]),
                        )
                        bindings_seen += 1
                        for term, view_id, target in zip(
                            contrast["terms"],
                            binding["view_ids"],
                            binding["target_pooled_procedure_ids"],
                            strict=True,
                        ):
                            view = pooled[
                                (
                                    contrast["support_id"],
                                    binding["pool_group_id"],
                                    term["policy_id"],
                                    term["strategy"],
                                )
                            ]
                            self.assertEqual(view["view_id"], view_id)
                            self.assertEqual(view["target_pooled_procedure_id"], target)
                            signed_terms_seen += 1
                self.assertEqual(
                    bindings_seen,
                    catalog["summary"]["pooled_comparison_units"],
                )
                self.assertEqual(
                    signed_terms_seen,
                    catalog["summary"]["signed_pooled_term_references"],
                )

    def test_qc_fallback_operational_context_not_collapsed(self):
        parent = parent_for(MIXED4)
        catalog = build(MIXED4)
        operational, eligible = support_ids(parent)
        operational_contexts = set(parent["supports"][operational]["context_ids"])
        eligible_contexts = set(parent["supports"][eligible]["context_ids"])
        fallback_contexts = operational_contexts - eligible_contexts
        self.assertTrue(fallback_contexts)
        qc_policy = registry()["comparison_qc_policy"]
        views = {
            (view["context_id"], view["policy_id"], view["strategy"]): view
            for view in parent["context_views"]
        }
        checked = 0
        for contrast in catalog["contrasts"]:
            if contrast["support_id"] != operational:
                continue
            if contrast["kind"] != "effect":
                continue
            if qc_policy not in {term["policy_id"] for term in contrast["terms"]}:
                continue
            for binding in contrast["context_bindings"]:
                if binding["context_id"] not in fallback_contexts:
                    continue
                self.assertEqual(len(binding["view_ids"]), len(contrast["terms"]))
                self.assertEqual(
                    len(binding["target_procedure_ids"]),
                    len(contrast["terms"]),
                )
                self.assertEqual(len(set(binding["view_ids"])), len(binding["view_ids"]))
                self.assertEqual(len(set(binding["target_procedure_ids"])), 1)
                for term, view_id, target in zip(
                    contrast["terms"],
                    binding["view_ids"],
                    binding["target_procedure_ids"],
                    strict=True,
                ):
                    view = views[
                        (
                            binding["context_id"],
                            term["policy_id"],
                            term["strategy"],
                        )
                    ]
                    self.assertEqual(view["view_id"], view_id)
                    self.assertEqual(view["target_procedure_id"], target)
                    if term["policy_id"] == qc_policy:
                        self.assertEqual(view["mode"], "qc_minimal_fallback")
                checked += 1
        self.assertGreater(checked, 0)

    def test_support_records_match_parent_groups(self):
        records = {record["support_id"]: record for record in self.catalog["support_records"]}
        self.assertEqual(set(records), set(self.parent["supports"]))
        for support_id, support in self.parent["supports"].items():
            record = records[support_id]
            self.assertEqual(set(record), SUPPORT_KEYS)
            self.assertEqual(record["context_ids"], support["context_ids"])
            self.assertEqual(record["complete_pool_group_ids"], support["complete_pool_group_ids"])
            self.assertEqual(record["context_ids"], sorted(record["context_ids"]))
            self.assertEqual(
                record["complete_pool_group_ids"], sorted(record["complete_pool_group_ids"])
            )

    def test_instrument_columns(self):
        entries = self.catalog["instrument_weight_columns"]
        self.assertEqual(
            [entry["instrument"] for entry in entries],
            sorted(self.parent["global_instrument_ids"]),
        )
        self.assertEqual([entry["column"] for entry in entries], list(range(len(entries))))

    def test_summary_literals_and_recompute(self):
        summary = self.catalog["summary"]
        self.assertEqual(set(summary), SUMMARY_KEYS)
        self.assertEqual(summary["contrast_count"], 456)
        self.assertEqual(summary["kind_counts"], {"effect": 216, "interaction": 240})
        self.assertEqual(summary["context_comparison_units"], 816)
        self.assertEqual(summary["pooled_comparison_units"], 0)
        self.assertEqual(summary["signed_context_term_references"], 2496)
        self.assertEqual(summary["signed_pooled_term_references"], 0)
        self.assertEqual(
            summary["platform_family_mapping_sha256"],
            independent_hash(self.catalog["platform_families"]),
        )
        self.assertEqual(sorted(summary["support_contrast_counts"].values()), [96, 360])
        self.assertEqual(len(summary["multiplicity_family_counts"]), 5)
        self.assertTrue(summary["all_authorization_flags_false"])
        for key, value in _summary_counts(self.catalog).items():
            self.assertEqual(summary[key], value)

    def test_fourfold_summary_literals(self):
        summary = fourfold_catalog()["summary"]
        self.assertEqual(summary["contrast_count"], 456)
        self.assertEqual(summary["context_comparison_units"], 1824)
        self.assertEqual(summary["pooled_comparison_units"], 456)
        self.assertEqual(summary["signed_context_term_references"], 5568)
        self.assertEqual(summary["signed_pooled_term_references"], 1392)

    def test_mixed_incomplete_pool_has_no_eligible_bindings(self):
        catalog = build(MIXED4)
        _, eligible = support_ids(parent_for(MIXED4))
        summary = catalog["summary"]
        self.assertEqual(summary["context_comparison_units"], 1728)
        self.assertEqual(summary["pooled_comparison_units"], 360)
        self.assertEqual(summary["signed_context_term_references"], 5280)
        self.assertEqual(summary["signed_pooled_term_references"], 1104)
        for contrast in catalog["contrasts"]:
            if contrast["support_id"] == eligible:
                self.assertEqual(contrast["pooled_bindings"], [])


class ContrastValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = default_catalog()

    def test_validate_roundtrip_snapshot(self):
        snapshot = validate_stress_contrast_catalog(self.catalog)
        self.assertEqual(snapshot, self.catalog)
        self.assertIsNot(snapshot, self.catalog)
        self.assertIsNot(snapshot["summary"], self.catalog["summary"])

    def test_validate_snapshot_deeply_independent(self):
        snapshot = validate_stress_contrast_catalog(self.catalog)
        self.assertIsNot(snapshot, self.catalog)
        self.assertIsNot(snapshot["summary"], self.catalog["summary"])
        first = snapshot["contrasts"][0]
        first["terms"][0]["coefficient"] = 0
        first["terms"].append({"coefficient": 9, "policy_id": "x", "strategy": "y"})
        snapshot["summary"]["kind_counts"]["effect"] = 0
        snapshot["support_records"][0]["context_ids"].append("forged")
        self.assertEqual(self.catalog["contrasts"][0]["terms"][0]["coefficient"], 1)
        self.assertEqual(len(self.catalog["contrasts"][0]["terms"]), len(first["terms"]) - 1)
        self.assertNotEqual(snapshot["summary"], self.catalog["summary"])
        if first["context_bindings"]:
            first["context_bindings"][0]["view_ids"].append(-1)
            self.assertNotIn(
                -1,
                self.catalog["contrasts"][0]["context_bindings"][0]["view_ids"],
            )
        revalidated = validate_stress_contrast_catalog(self.catalog)
        self.assertEqual(revalidated, self.catalog)

    def test_validate_rejects_non_mapping(self):
        for bad in (None, [], "x", 1):
            with self.subTest(kind=type(bad).__name__):
                with self.assertRaises(ValueError) as caught:
                    validate_stress_contrast_catalog(bad)
                self.assertEqual(str(caught.exception), INVALID)

    def test_validate_rejects_forged_digest(self):
        forged = copy.deepcopy(self.catalog)
        forged["catalog_sha256"] = "0" * 64
        with self.assertRaises(ValueError) as caught:
            validate_stress_contrast_catalog(forged)
        self.assertEqual(str(caught.exception), INVALID)

    def test_validate_rejects_rehashed_execution_claim(self):
        forged = copy.deepcopy(self.catalog)
        forged["execution_authorized"] = True
        forged["catalog_sha256"] = independent_hash(
            {k: v for k, v in forged.items() if k != "catalog_sha256"}
        )
        with self.assertRaises(ValueError) as caught:
            validate_stress_contrast_catalog(forged)
        self.assertEqual(str(caught.exception), INVALID)

    def test_validate_rejects_rehashed_summary_forgery(self):
        forged = copy.deepcopy(self.catalog)
        forged["summary"]["contrast_count"] = 1
        forged["catalog_sha256"] = independent_hash(
            {k: v for k, v in forged.items() if k != "catalog_sha256"}
        )
        with self.assertRaises(ValueError) as caught:
            validate_stress_contrast_catalog(forged)
        self.assertEqual(str(caught.exception), INVALID)

    def test_require_scientific_execution_always_denied(self):
        cases = ((), ({},), ({"execution_authorized": True},), (self.catalog,))
        for index, args in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError) as caught:
                    require_scientific_execution(*args)
                self.assertEqual(str(caught.exception), DENIED)


class ContrastBuildRejectionTests(unittest.TestCase):
    def assert_invalid(self, *, registry_obj=None, platform_families=None):
        parent = parent_for(None)
        with self.assertRaises(ValueError) as caught:
            build_stress_contrast_catalog(
                score_catalog=parent,
                inference_registry=registry() if registry_obj is None else registry_obj,
                platform_families=(
                    families_for(parent) if platform_families is None else platform_families
                ),
            )
        self.assertEqual(str(caught.exception), INVALID)

    def _registry_case(self, mutate):
        reg = copy.deepcopy(registry())
        mutate(reg)
        self.assert_invalid(registry_obj=reg)

    def test_bad_registry_semantics(self):
        cases = {
            "missing": lambda r: r["contrasts"].pop(),
            "duplicate": lambda r: r["contrasts"].append(copy.deepcopy(r["contrasts"][0])),
            "blank_id": lambda r: r["contrasts"][0].__setitem__("contrast_id", "  "),
            "duplicate_id": lambda r: r["contrasts"][1].__setitem__(
                "contrast_id", r["contrasts"][0]["contrast_id"]
            ),
            "unknown_method": lambda r: r["contrasts"][0].__setitem__("method", "NOPE"),
            "unknown_policy": lambda r: r["contrasts"][0].__setitem__("policy", "NOPE"),
            "unknown_endpoint": lambda r: r["contrasts"][0].__setitem__("endpoint", "M99"),
            "unknown_family": lambda r: r["contrasts"][0].__setitem__("disturbance_family", "nope"),
            "unknown_support": lambda r: r["contrasts"][0].__setitem__("support", "nope"),
            "counter_claim": lambda r: r.__setitem__("authorized_model_fits", 1),
            "bool_counter": lambda r: r.__setitem__("authorized_model_fits", True),
            "flag_claim": lambda r: r.__setitem__("execution_authorized", True),
            "extra_key": lambda r: r["contrasts"][0].__setitem__("extra", 1),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                self._registry_case(mutate)

    def test_adversarial_registry_semantics(self):
        def rename_panel(reg):
            for contrast in reg["contrasts"]:
                if contrast.get("panel") == "universal":
                    contrast["panel"] = "evil"

        def rename_families(reg):
            families = reg["multiplicity"]["families"]
            mapping = {label: "evil-" + str(label) for label in families}
            for contrast in reg["contrasts"]:
                label = contrast.get("multiplicity_family")
                if label in mapping:
                    contrast["multiplicity_family"] = mapping[label]
            reg["multiplicity"]["families"] = {
                mapping.get(label, "evil-" + str(label)): count for label, count in families.items()
            }

        def bool_family_count(reg):
            families = reg["multiplicity"]["families"]
            key = sorted(families)[0]
            families[key] = True

        def float_family_count(reg):
            families = reg["multiplicity"]["families"]
            key = sorted(families)[0]
            families[key] = float(families[key])

        def duplicate_semantics_new_id(reg):
            contrasts = reg["contrasts"]
            clone = copy.deepcopy(contrasts[0])
            clone["contrast_id"] = "renamed-identical-semantics"
            contrasts[-1] = clone

        def swap_support_labels(reg):
            names = sorted(reg["supports"])
            operational = next(name for name in names if "operational" in name)
            eligible = next(name for name in names if name != operational and "qc" in name)
            for contrast in reg["contrasts"]:
                if contrast["support"] == operational:
                    contrast["support"] = eligible
                elif contrast["support"] == eligible:
                    contrast["support"] = operational

        def change_protocol(reg):
            reg["protocol_version"] = "diverted"

        def accept_inference(reg):
            reg["inference_implementation_accepted"] = True

        def accept_perturbation(reg):
            reg["perturbation_runtime_accepted"] = True

        def approve_resource(reg):
            reg["resource_proposal_approved"] = True

        def imply_g4(reg):
            reg["original_G4_pass_implied"] = True

        cases = {
            "panel_renamed": rename_panel,
            "family_labels_renamed": rename_families,
            "family_count_bool": bool_family_count,
            "family_count_float": float_family_count,
            "semantics_replaced_new_id": duplicate_semantics_new_id,
            "support_labels_swapped": swap_support_labels,
            "protocol_version_changed": change_protocol,
            "inference_implementation_accepted": accept_inference,
            "perturbation_runtime_accepted": accept_perturbation,
            "resource_proposal_approved": approve_resource,
            "original_G4_pass_implied": imply_g4,
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                self._registry_case(mutate)

    def test_bad_registry_strict_json(self):
        cases = {
            "nan": lambda r: r.__setitem__("date", float("nan")),
            "tuple": lambda r: r.__setitem__("scope_approvals", ("P08-A10", "P08-A11")),
            "nonstring_key": lambda r: r["supports"].__setitem__(1, {}),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                self._registry_case(mutate)

    def test_bad_platform_families(self):
        parent = parent_for(None)
        base = families_for(parent)
        cases = {
            "missing": {},
            "extra": {**base, "extra-inst": "toy-platform"},
            "empty": {key: "" for key in base},
            "bool": {key: True for key in base},
            "integer": {key: 1 for key in base},
        }
        for name, platform_families in cases.items():
            with self.subTest(name=name):
                self.assert_invalid(platform_families=platform_families)

    def test_null_family_marks_unknown_instrument(self):
        parent = parent_for(None)
        platform_families = {instrument: None for instrument in parent["global_instrument_ids"]}
        catalog = build_stress_contrast_catalog(
            score_catalog=parent,
            inference_registry=registry(),
            platform_families=platform_families,
        )
        self.assertEqual(catalog["platform_families"], platform_families)
        for record in catalog["support_records"]:
            self.assertEqual(record["known_platform_families"], [])
            self.assertEqual(record["unknown_family_instruments"], sorted(record["instruments"]))


class MasterWeightColumnTests(unittest.TestCase):
    def _lexical_parent(self):
        sc, _ = support_for(None)
        memberships = build_memberships(sc)
        masters = sorted({row["master_id"] for row in memberships["test_rows"]})
        chosen = [2, 10, 100] + [900000 + index for index in range(len(masters) - 3)]
        mapping = dict(zip(masters, chosen, strict=True))
        mutated = copy.deepcopy(memberships)
        for row in mutated["test_rows"]:
            row["master_id"] = mapping[row["master_id"]]
        parent = build_stress_score_support(prediction_catalog=sc.catalog, memberships=mutated)
        return parent

    def test_lexical_identity_and_raw_types(self):
        parent = self._lexical_parent()
        catalog = build_stress_contrast_catalog(
            score_catalog=parent,
            inference_registry=registry(),
            platform_families=families_for(parent),
        )
        columns = catalog["master_weight_columns"]
        identities = [entry["weight_identity"] for entry in columns]
        self.assertEqual(identities, sorted(identities))
        self.assertEqual(identities[:4], ["10", "100", "2", "900000"])
        self.assertEqual([entry["column"] for entry in columns], list(range(len(columns))))
        self.assertTrue(all(type(entry["master_id"]) is int for entry in columns))
        self.assertTrue(
            all(entry["weight_identity"] == str(entry["master_id"]) for entry in columns)
        )

    def test_support_columns_are_global_indices(self):
        parent = self._lexical_parent()
        catalog = build_stress_contrast_catalog(
            score_catalog=parent,
            inference_registry=registry(),
            platform_families=families_for(parent),
        )
        columns = catalog["master_weight_columns"]
        index = {entry["weight_identity"]: entry["column"] for entry in columns}
        rows = parent["memberships"]["test_rows"]
        for record in catalog["support_records"]:
            scoped = [row for row in rows if row["context_id"] in set(record["context_ids"])]
            expected = sorted({index[str(row["master_id"])] for row in scoped})
            self.assertEqual(record["master_weight_columns"], expected)
            self.assertTrue(set(record["master_weight_columns"]) <= set(range(len(columns))))

    def test_eligible_columns_are_global_indices(self):
        parent = self._lexical_parent()
        catalog = build_stress_contrast_catalog(
            score_catalog=parent,
            inference_registry=registry(),
            platform_families=families_for(parent),
        )
        _, eligible = support_ids(parent)
        records = {record["support_id"]: record for record in catalog["support_records"]}
        record = records[eligible]
        columns = record["master_weight_columns"]
        self.assertEqual(columns, sorted(columns))
        self.assertNotEqual(columns, list(range(len(columns))))
        index = {
            entry["weight_identity"]: entry["column"] for entry in catalog["master_weight_columns"]
        }
        rows = parent["memberships"]["test_rows"]
        contexts = set(record["context_ids"])
        scoped = sorted(
            {index[str(row["master_id"])] for row in rows if row["context_id"] in contexts}
        )
        self.assertEqual(columns, scoped)

    def test_numeric_fixture_mapping_is_noncontiguous(self):
        parent = self._lexical_parent()
        catalog = build_stress_contrast_catalog(
            score_catalog=parent,
            inference_registry=registry(),
            platform_families=families_for(parent),
        )
        entries = catalog["master_weight_columns"]
        self.assertEqual([entry["column"] for entry in entries], list(range(len(entries))))
        identities = [entry["weight_identity"] for entry in entries]
        self.assertEqual(identities[:4], ["10", "100", "2", "900000"])
        master_ids = [entry["master_id"] for entry in entries]
        self.assertTrue(all(type(value) is int for value in master_ids))
        self.assertNotEqual(master_ids, sorted(master_ids))

    def test_build_does_not_mutate_inputs(self):
        parent = parent_for(None)
        reg = registry()
        families = families_for(parent)
        before_parent = copy.deepcopy(parent)
        before_registry = copy.deepcopy(reg)
        before_families = copy.deepcopy(families)
        build_stress_contrast_catalog(
            score_catalog=parent,
            inference_registry=reg,
            platform_families=families,
        )
        self.assertEqual(parent, before_parent)
        self.assertEqual(reg, before_registry)
        self.assertEqual(families, before_families)


if __name__ == "__main__":
    unittest.main()
