"""Tests for the P08-T009 no-fit nested QC role registry."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest

from atlas_sers.evaluation.p08_qc_roles import (
    ALGORITHM,
    MASTER_NAMESPACE,
    REQUIRED_INNER_FOLDS,
    ROLE_PAIR_NAMESPACE,
    ROLE_REASON_CODES,
    SALT,
    SCHEMA_VERSION,
    QCRoleError,
    build_nested_qc_roles,
    require_scientific_execution,
)
from atlas_sers.evaluation.p08_qc_support import (
    ERROR_REASON_CODES as SUPPORT_REASON_CODES,
)
from atlas_sers.evaluation.p08_qc_support import (
    MASTER_MODE,
    PSEUDO_MODE,
    audit_nested_qc_support,
)
from atlas_sers.governance.canonical import sha256_value

SENTINEL = "SENTINEL-PRIVATE-424242"


def canonical_hash(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class Corpus:
    def __init__(self):
        self.roles = []
        self._counter = 0

    def obs(self, master_id, instrument, label):
        self._counter += 1
        uid = f"obs-{self._counter:08d}"
        self.roles.append(
            {
                "observation_uid": uid,
                "master_id": master_id,
                "instrument": instrument,
                "label": label,
            }
        )
        return uid


def add_pseudo_context(
    corpus,
    context_id,
    per_class=3,
    n_units=2,
    classes=("A", "B"),
    test_classes=None,
    extra_views=0,
    short_second_unit=False,
):
    units = []
    outer_fit = []
    for unit_index in range(n_units):
        fit_uids = []
        for class_id in classes:
            limit = per_class
            if short_second_unit and unit_index == 1 and class_id == classes[-1]:
                limit = 1
            for master_index in range(limit):
                master_id = f"{context_id}-u{unit_index}-{class_id}-m{master_index}"
                fit_uids.append(
                    corpus.obs(master_id, f"{context_id}-u{unit_index}-fit", class_id)
                )
                for view in range(extra_views):
                    fit_uids.append(
                        corpus.obs(
                            master_id,
                            f"{context_id}-u{unit_index}-fit-v{view}",
                            class_id,
                        )
                    )
        validation_uids = [
            corpus.obs(
                f"{context_id}-u{unit_index}-val-{class_id}-m",
                f"{context_id}-u{unit_index}-val",
                class_id,
            )
            for class_id in classes
        ]
        units.append(
            {
                "unit_id": f"{context_id}-u{unit_index}",
                "fit_uids": fit_uids,
                "validation_uids": validation_uids,
            }
        )
        outer_fit.extend(fit_uids)
        outer_fit.extend(validation_uids)
    if test_classes is None:
        test_classes = classes
    outer_test = [
        corpus.obs(f"{context_id}-test-{class_id}", "held-inst", class_id)
        for class_id in test_classes
    ]
    return {
        "context_id": context_id,
        "selection_mode": PSEUDO_MODE,
        "classes": list(classes),
        "outer_fit_uids": outer_fit,
        "outer_test_uids": outer_test,
        "held_instrument": "held-inst",
        "selection_units": units,
    }


def add_master_context(
    corpus, context_id, per_class=1, classes=("A", "B"), test_classes=None
):
    folds = []
    outer_fit = []
    for fold_index in range(3):
        fold = []
        for class_id in classes:
            for master_index in range(per_class):
                fold.append(
                    corpus.obs(
                        f"{context_id}-f{fold_index}-{class_id}-m{master_index}",
                        f"{context_id}-fit",
                        class_id,
                    )
                )
        folds.append(fold)
        outer_fit.extend(fold)
    units = []
    for fold_index in range(3):
        fit_uids = [
            uid for other in range(3) if other != fold_index for uid in folds[other]
        ]
        units.append(
            {
                "unit_id": f"{context_id}-u{fold_index}",
                "fit_uids": fit_uids,
                "validation_uids": list(folds[fold_index]),
            }
        )
    if test_classes is None:
        test_classes = classes
    outer_test = [
        corpus.obs(f"{context_id}-test-{class_id}", "held-inst", class_id)
        for class_id in test_classes
    ]
    return {
        "context_id": context_id,
        "selection_mode": MASTER_MODE,
        "classes": list(classes),
        "outer_fit_uids": outer_fit,
        "outer_test_uids": outer_test,
        "held_instrument": "held-inst",
        "selection_units": units,
    }


def role_index(roles):
    return {row["observation_uid"]: row for row in roles}


def unit_by_id(context):
    return {unit["unit_id"]: unit for unit in context["selection_units"]}


def all_folds(plan, index=0):
    return [
        fold
        for unit in plan["contexts"][index]["policy_units"]
        for fold in unit["inner_folds"]
    ]


class RegistryShapeTests(unittest.TestCase):
    def test_top_level_shape_and_denied_execution(self):
        corpus = Corpus()
        contexts = [add_pseudo_context(corpus, "CTX-P", per_class=3)]
        plan = build_nested_qc_roles(contexts, corpus.roles)
        self.assertEqual(SCHEMA_VERSION, "nato-sers-p08-qc-nested-roles-v1")
        self.assertEqual(plan["schema_version"], SCHEMA_VERSION)
        self.assertIs(plan["execution_authorized"], False)
        self.assertEqual(plan["algorithm"], ALGORITHM)
        self.assertEqual(
            plan["algorithm"], "classwise_canonical_hash_rank_round_robin_3"
        )
        self.assertEqual(plan["salt"], SALT)
        self.assertEqual(plan["salt"], 2026093004)
        self.assertEqual(REQUIRED_INNER_FOLDS, 3)
        self.assertEqual(len(plan["support_audit_sha256"]), 64)
        self.assertEqual(len(plan["registry_sha256"]), 64)
        self.assertIsInstance(plan["contexts"], list)
        self.assertIsInstance(plan["summary"], dict)

    def test_support_audit_sha256_matches_auditor(self):
        corpus = Corpus()
        contexts = [add_pseudo_context(corpus, "CTX-P", per_class=3)]
        audit = audit_nested_qc_support(contexts, corpus.roles)
        plan = build_nested_qc_roles(contexts, corpus.roles)
        self.assertEqual(plan["support_audit_sha256"], audit["audit_sha256"])

    def test_registry_hash_independently_recomputed(self):
        corpus = Corpus()
        contexts = [
            add_pseudo_context(corpus, "CTX-P", per_class=4),
            add_master_context(corpus, "CTX-M", per_class=1),
        ]
        plan = build_nested_qc_roles(contexts, corpus.roles)
        payload = {
            key: value for key, value in plan.items() if key != "registry_sha256"
        }
        self.assertEqual(plan["registry_sha256"], canonical_hash(payload))
        self.assertEqual(plan["registry_sha256"], sha256_value(payload))

    def test_contexts_sorted(self):
        corpus = Corpus()
        contexts = [
            add_pseudo_context(corpus, "CTX-Z", per_class=3),
            add_master_context(corpus, "CTX-A", per_class=1),
        ]
        plan = build_nested_qc_roles(contexts, corpus.roles)
        ids = [entry["context_id"] for entry in plan["contexts"]]
        self.assertEqual(ids, ["CTX-A", "CTX-Z"])


class InnerFoldTests(unittest.TestCase):
    def test_three_folds_per_policy_unit(self):
        corpus = Corpus()
        contexts = [add_pseudo_context(corpus, "CTX-P", per_class=3, n_units=2)]
        plan = build_nested_qc_roles(contexts, corpus.roles)
        detail = plan["contexts"][0]
        self.assertTrue(detail["eligible"])
        self.assertEqual(detail["reason_code"], "nested_three_fold_supported")
        self.assertEqual(len(detail["policy_units"]), 2)
        for unit in detail["policy_units"]:
            self.assertEqual(len(unit["inner_folds"]), 3)
            self.assertEqual(
                [fold["fold_index"] for fold in unit["inner_folds"]], [0, 1, 2]
            )
        summary = plan["summary"]
        self.assertEqual(summary["policy_validation_units"], 2)
        self.assertEqual(summary["nested_estimator_folds"], 6)
        self.assertEqual(summary["unique_role_pair_ids"], 6)

    def test_fold_partition_class_support_and_hashes(self):
        corpus = Corpus()
        contexts = [add_pseudo_context(corpus, "CTX-P", per_class=3)]
        plan = build_nested_qc_roles(contexts, corpus.roles)
        detail = plan["contexts"][0]
        classes = contexts[0]["classes"]
        for unit in detail["policy_units"]:
            parent_fit = set(unit["policy_fit_uids"])
            self.assertEqual(
                unit["policy_fit_uid_sha256"], canonical_hash(unit["policy_fit_uids"])
            )
            self.assertEqual(
                unit["policy_validation_uid_sha256"],
                canonical_hash(unit["policy_validation_uids"]),
            )
            self.assertEqual(
                unit["policy_refit_quantile_fit_uid_sha256"],
                unit["policy_fit_uid_sha256"],
            )
            for fold in unit["inner_folds"]:
                fit = set(fold["fit_uids"])
                validation = set(fold["validation_uids"])
                self.assertEqual(fit | validation, parent_fit)
                self.assertTrue(fit.isdisjoint(validation))
                self.assertEqual(fit, parent_fit - validation)
                self.assertEqual(
                    fold["fit_uid_sha256"], canonical_hash(fold["fit_uids"])
                )
                self.assertEqual(
                    fold["validation_uid_sha256"],
                    canonical_hash(fold["validation_uids"]),
                )
                self.assertEqual(
                    fold["quantile_fit_uid_sha256"], fold["fit_uid_sha256"]
                )
                for class_id in classes:
                    self.assertGreaterEqual(
                        fold["fit_class_master_counts"][class_id], 2
                    )
                    self.assertGreaterEqual(
                        fold["validation_class_master_counts"][class_id], 1
                    )
        self.assertEqual(
            detail["final_refit_quantile_fit_uid_sha256"],
            detail["outer_fit_uid_sha256"],
        )

    def test_master_counts_three_four_five(self):
        for masters_per_class in (3, 4, 5):
            with self.subTest(masters=masters_per_class):
                corpus = Corpus()
                context = add_pseudo_context(
                    corpus, f"CTX-{masters_per_class}", per_class=masters_per_class
                )
                plan = build_nested_qc_roles([context], corpus.roles)
                detail = plan["contexts"][0]
                unit = detail["policy_units"][0]
                by_uid = role_index(corpus.roles)
                for class_id in context["classes"]:
                    parent_masters = {
                        by_uid[uid]["master_id"]
                        for uid in unit["policy_fit_uids"]
                        if by_uid[uid]["label"] == class_id
                    }
                    self.assertEqual(len(parent_masters), masters_per_class)
                    seen = set()
                    for fold in unit["inner_folds"]:
                        fit_count = fold["fit_class_master_counts"][class_id]
                        val_count = fold["validation_class_master_counts"][class_id]
                        self.assertEqual(fit_count + val_count, masters_per_class)
                        self.assertGreaterEqual(fit_count, 2)
                        validation_for_class = {
                            master
                            for master in fold["validation_masters"]
                            if master in parent_masters
                        }
                        self.assertEqual(len(validation_for_class), val_count)
                        self.assertTrue(seen.isdisjoint(validation_for_class))
                        seen |= validation_for_class
                    self.assertEqual(seen, parent_masters)

    def test_literal_hash_rank_fold_assignment(self):
        corpus = Corpus()
        context = add_pseudo_context(
            corpus, "CTX-GOLD", per_class=5, classes=("A", "B", "C")
        )
        plan = build_nested_qc_roles([context], corpus.roles)
        detail = plan["contexts"][0]
        unit = next(
            item
            for item in detail["policy_units"]
            if item["parent_unit_id"] == "CTX-GOLD-u0"
        )
        expected = {
            "CTX-GOLD-u0-A-m2": 0,
            "CTX-GOLD-u0-A-m0": 1,
            "CTX-GOLD-u0-A-m3": 2,
            "CTX-GOLD-u0-A-m4": 0,
            "CTX-GOLD-u0-A-m1": 1,
            "CTX-GOLD-u0-B-m4": 0,
            "CTX-GOLD-u0-B-m0": 1,
            "CTX-GOLD-u0-B-m3": 2,
            "CTX-GOLD-u0-B-m2": 0,
            "CTX-GOLD-u0-B-m1": 1,
            "CTX-GOLD-u0-C-m0": 0,
            "CTX-GOLD-u0-C-m4": 1,
            "CTX-GOLD-u0-C-m3": 2,
            "CTX-GOLD-u0-C-m1": 0,
            "CTX-GOLD-u0-C-m2": 1,
        }
        actual = {}
        for fold in unit["inner_folds"]:
            for master in fold["validation_masters"]:
                self.assertNotIn(master, actual)
                actual[master] = fold["fold_index"]
        self.assertEqual(actual, expected)

    def test_role_pair_ids_are_full_sha256(self):
        corpus = Corpus()
        contexts = [add_pseudo_context(corpus, "CTX-P", per_class=3)]
        plan = build_nested_qc_roles(contexts, corpus.roles)
        detail = plan["contexts"][0]
        for unit in detail["policy_units"]:
            for fold in unit["inner_folds"]:
                identity = {
                    "namespace": ROLE_PAIR_NAMESPACE,
                    "context_id": detail["context_id"],
                    "parent_unit_id": unit["parent_unit_id"],
                    "fold_index": fold["fold_index"],
                    "fit_uid_sha256": fold["fit_uid_sha256"],
                    "validation_uid_sha256": fold["validation_uid_sha256"],
                }
                self.assertEqual(
                    fold["role_pair_id"], "P08QCROLE-" + canonical_hash(identity)
                )
                suffix = fold["role_pair_id"][len("P08QCROLE-") :]
                self.assertRegex(suffix, r"^[0-9a-f]{64}$")

    def test_unique_role_pair_ids_match_fold_count(self):
        corpus = Corpus()
        contexts = [
            add_pseudo_context(corpus, "CTX-P1", per_class=4, n_units=2),
            add_pseudo_context(corpus, "CTX-P2", per_class=5, n_units=3),
        ]
        plan = build_nested_qc_roles(contexts, corpus.roles)
        ids = [fold["role_pair_id"] for fold in all_folds(plan, 0)]
        ids += [fold["role_pair_id"] for fold in all_folds(plan, 1)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(plan["summary"]["unique_role_pair_ids"], len(ids))


class RepetitionTests(unittest.TestCase):
    def test_repeated_views_kept_in_same_fold(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", per_class=3, extra_views=1)
        plan = build_nested_qc_roles([context], corpus.roles)
        by_uid = role_index(corpus.roles)
        detail = plan["contexts"][0]
        raw_units = unit_by_id(context)
        for unit in detail["policy_units"]:
            raw = raw_units[unit["parent_unit_id"]]
            master_to_uids = {}
            for uid in raw["fit_uids"]:
                master_to_uids.setdefault(by_uid[uid]["master_id"], set()).add(uid)
            for master, uids in master_to_uids.items():
                validation_folds = [
                    fold["fold_index"]
                    for fold in unit["inner_folds"]
                    if master in fold["validation_masters"]
                ]
                self.assertEqual(len(validation_folds), 1)
                validation_fold = unit["inner_folds"][validation_folds[0]]
                for uid in uids:
                    self.assertIn(uid, validation_fold["validation_uids"])
                for fold in unit["inner_folds"]:
                    if fold["fold_index"] == validation_folds[0]:
                        continue
                    for uid in uids:
                        self.assertIn(uid, fold["fit_uids"])

    def test_repeated_measurement_count_does_not_change_master_fold(self):
        base_corpus = Corpus()
        base_context = add_pseudo_context(
            base_corpus, "CTX-P", per_class=3, extra_views=0
        )
        base = build_nested_qc_roles([base_context], base_corpus.roles)

        variant_corpus = Corpus()
        variant_context = add_pseudo_context(
            variant_corpus, "CTX-P", per_class=3, extra_views=1
        )
        variant = build_nested_qc_roles([variant_context], variant_corpus.roles)

        def validation_master_map(plan):
            result = {}
            for fold in all_folds(plan):
                for master in fold["validation_masters"]:
                    self.assertNotIn(master, result)
                    result[master] = fold["fold_index"]
            return result

        self.assertEqual(
            validation_master_map(base), validation_master_map(variant)
        )


class UnsupportedTests(unittest.TestCase):
    def test_mastercv_builds_no_folds(self):
        corpus = Corpus()
        context = add_master_context(corpus, "CTX-M", per_class=1)
        plan = build_nested_qc_roles([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertFalse(detail["eligible"])
        self.assertEqual(detail["reason_code"], "no_source_pseudo_domains")
        self.assertEqual(detail["policy_units"], [])
        self.assertIsNone(detail["final_refit_quantile_fit_uid_sha256"])
        self.assertEqual(plan["summary"]["nested_estimator_folds"], 0)
        self.assertIsNone(plan["summary"]["minimum_fit_masters_per_class"])
        self.assertIsNone(plan["summary"]["minimum_validation_masters_per_class"])

    def test_mastercv_sufficient_counts_still_no_folds(self):
        corpus = Corpus()
        context = add_master_context(corpus, "CTX-M", per_class=2)
        plan = build_nested_qc_roles([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertEqual(detail["reason_code"], "no_source_pseudo_domains")
        self.assertEqual(detail["policy_units"], [])

    def test_min_one_master_pseudo_no_folds(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", per_class=1)
        plan = build_nested_qc_roles([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertFalse(detail["eligible"])
        self.assertEqual(
            detail["reason_code"], "insufficient_nested_class_masters"
        )
        self.assertEqual(detail["policy_units"], [])
        self.assertIsNone(detail["final_refit_quantile_fit_uid_sha256"])

    def test_mixed_units_fall_back_whole_context(self):
        corpus = Corpus()
        context = add_pseudo_context(
            corpus, "CTX-P", per_class=3, short_second_unit=True
        )
        plan = build_nested_qc_roles([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertFalse(detail["eligible"])
        self.assertEqual(
            detail["reason_code"], "insufficient_nested_class_masters"
        )
        self.assertEqual(detail["policy_units"], [])
        self.assertIsNone(detail["final_refit_quantile_fit_uid_sha256"])


class IsolationTests(unittest.TestCase):
    def test_held_test_class_distribution_does_not_change_folds(self):
        corpus_a = Corpus()
        context_a = add_pseudo_context(
            corpus_a, "CTX-P", per_class=4, test_classes=("A", "B")
        )
        plan_a = build_nested_qc_roles([context_a], corpus_a.roles)

        corpus_b = Corpus()
        context_b = add_pseudo_context(
            corpus_b, "CTX-P", per_class=4, test_classes=("A",)
        )
        plan_b = build_nested_qc_roles([context_b], corpus_b.roles)

        self.assertEqual(
            [fold["role_pair_id"] for fold in all_folds(plan_a)],
            [fold["role_pair_id"] for fold in all_folds(plan_b)],
        )

    def test_inner_masters_disjoint_from_policy_validation_and_outer_test(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", per_class=4)
        by_uid = role_index(corpus.roles)
        plan = build_nested_qc_roles([context], corpus.roles)
        detail = plan["contexts"][0]
        raw_units = unit_by_id(context)
        outer_test_masters = {
            by_uid[uid]["master_id"] for uid in context["outer_test_uids"]
        }
        for unit in detail["policy_units"]:
            raw = raw_units[unit["parent_unit_id"]]
            policy_validation_masters = {
                by_uid[uid]["master_id"] for uid in raw["validation_uids"]
            }
            for fold in unit["inner_folds"]:
                inner = set(fold["fit_masters"]) | set(fold["validation_masters"])
                self.assertTrue(inner.isdisjoint(policy_validation_masters))
                self.assertTrue(inner.isdisjoint(outer_test_masters))
                self.assertTrue(
                    set(fold["validation_masters"]).isdisjoint(outer_test_masters)
                )


class InvarianceTests(unittest.TestCase):
    def test_inputs_not_mutated(self):
        corpus = Corpus()
        contexts = [add_pseudo_context(corpus, "CTX-P", per_class=3)]
        roles = corpus.roles
        contexts_copy = copy.deepcopy(contexts)
        roles_copy = copy.deepcopy(roles)
        build_nested_qc_roles(contexts, roles)
        self.assertEqual(contexts, contexts_copy)
        self.assertEqual(roles, roles_copy)

    def test_permutation_invariance(self):
        corpus = Corpus()
        contexts = [
            add_pseudo_context(corpus, "CTX-B", per_class=3),
            add_pseudo_context(corpus, "CTX-A", per_class=4),
        ]
        roles = corpus.roles
        base = build_nested_qc_roles(contexts, roles)

        shuffled_contexts = copy.deepcopy(contexts)
        shuffled_contexts.reverse()
        for context in shuffled_contexts:
            context["classes"].reverse()
            context["selection_units"].reverse()
            for unit in context["selection_units"]:
                unit["fit_uids"].reverse()
                unit["validation_uids"].reverse()
            context["outer_fit_uids"].reverse()
            context["outer_test_uids"].reverse()
        shuffled_roles = copy.deepcopy(roles)
        shuffled_roles.reverse()

        shuffled = build_nested_qc_roles(shuffled_contexts, shuffled_roles)
        self.assertEqual(base["registry_sha256"], shuffled["registry_sha256"])


class SummaryTests(unittest.TestCase):
    def test_summary_arithmetic(self):
        corpus = Corpus()
        contexts = [
            add_pseudo_context(corpus, "CTX-P1", per_class=3, n_units=2),
            add_pseudo_context(corpus, "CTX-P2", per_class=1, n_units=2),
            add_master_context(corpus, "CTX-M", per_class=1),
        ]
        plan = build_nested_qc_roles(contexts, corpus.roles)
        summary = plan["summary"]
        self.assertEqual(summary["context_count"], 3)
        self.assertEqual(summary["eligible_contexts"], 1)
        self.assertEqual(summary["fallback_contexts"], 2)
        self.assertEqual(summary["policy_validation_units"], 2)
        self.assertEqual(summary["nested_estimator_folds"], 6)
        self.assertEqual(summary["unique_role_pair_ids"], 6)
        self.assertEqual(summary["minimum_fit_masters_per_class"], 2)
        self.assertEqual(summary["minimum_validation_masters_per_class"], 1)

    def test_aggregate_mixed_corpus_counts(self):
        corpus = Corpus()
        contexts = []
        for index in range(54):
            contexts.append(
                add_pseudo_context(
                    corpus, f"CTX-ELIG-{index:03d}", per_class=3, n_units=2
                )
            )
        for index in range(74):
            contexts.append(
                add_pseudo_context(
                    corpus, f"CTX-PSEUDO-{index:03d}", per_class=1, n_units=2
                )
            )
        for index in range(132):
            contexts.append(
                add_master_context(corpus, f"CTX-MASTER-{index:03d}", per_class=1)
            )
        plan = build_nested_qc_roles(contexts, corpus.roles)
        summary = plan["summary"]
        self.assertEqual(summary["context_count"], 260)
        self.assertEqual(summary["eligible_contexts"], 54)
        self.assertEqual(summary["fallback_contexts"], 206)
        self.assertEqual(summary["policy_validation_units"], 108)
        self.assertEqual(summary["nested_estimator_folds"], 324)
        self.assertEqual(summary["unique_role_pair_ids"], 324)

    def test_summary_contains_no_identities(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-SECRET", per_class=3)
        plan = build_nested_qc_roles([context], corpus.roles)
        text = json.dumps(plan["summary"], sort_keys=True)
        self.assertNotIn("CTX-SECRET", text)
        for unit in plan["contexts"][0]["policy_units"]:
            self.assertNotIn(unit["parent_unit_id"], text)
            for fold in unit["inner_folds"]:
                for uid in fold["fit_uids"] + fold["validation_uids"]:
                    self.assertNotIn(uid, text)


class CanonicalHashTests(unittest.TestCase):
    def test_unicode_canonical_hash(self):
        corpus = Corpus()
        context = add_pseudo_context(
            corpus, "CTX-ÜNI", per_class=3, classes=("Ä", "Ω")
        )
        plan = build_nested_qc_roles([context], corpus.roles)
        payload = {
            key: value for key, value in plan.items() if key != "registry_sha256"
        }
        self.assertEqual(plan["registry_sha256"], sha256_value(payload))
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.assertIn("Ω", canonical)
        escaped = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.assertNotEqual(
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            hashlib.sha256(escaped.encode("utf-8")).hexdigest(),
        )


class ErrorPropagationTests(unittest.TestCase):
    def test_malformed_inputs_propagate_support_codes(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", per_class=3)
        roles = corpus.roles

        bad_mode = copy.deepcopy([context])
        bad_mode[0]["selection_mode"] = "holdout"
        bad_uid = copy.deepcopy([context])
        bad_uid[0]["selection_units"][0]["fit_uids"].append("ghost-uid")

        cases = [
            ("contexts_must_be_sequence", "not-a-list", roles),
            ("contexts_empty", [], roles),
            ("roles_must_be_sequence", [context], "not-a-list"),
            ("selection_mode_invalid", bad_mode, roles),
            ("unknown_observation_uid", bad_uid, roles),
        ]
        for code, bad_contexts, bad_roles in cases:
            with self.subTest(code=code):
                with self.assertRaises(QCRoleError) as caught:
                    build_nested_qc_roles(bad_contexts, bad_roles)
                self.assertEqual(str(caught.exception), code)
                self.assertEqual(caught.exception.reason_code, code)
                self.assertIn(code, SUPPORT_REASON_CODES)
                self.assertNotIn(SENTINEL, str(caught.exception))

    def test_errors_do_not_echo_private_values(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", per_class=3)
        bad = copy.deepcopy([context])
        bad[0]["context_id"] = f" {SENTINEL}"
        with self.assertRaises(QCRoleError) as caught:
            build_nested_qc_roles(bad, corpus.roles)
        self.assertEqual(caught.exception.reason_code, "context_id_invalid")
        self.assertNotIn(SENTINEL, str(caught.exception))

    def test_refusal_for_forged_authorization(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", per_class=3)
        plan = build_nested_qc_roles([context], corpus.roles)
        self.assertIs(plan["execution_authorized"], False)
        with self.assertRaises(QCRoleError) as caught:
            require_scientific_execution(plan)
        self.assertEqual(
            caught.exception.reason_code, "scientific_execution_not_authorized"
        )
        forged = dict(plan)
        forged["execution_authorized"] = True
        with self.assertRaises(QCRoleError):
            require_scientific_execution(forged)
        with self.assertRaises(QCRoleError):
            require_scientific_execution()

    def test_internal_constants_registered(self):
        self.assertIn("scientific_execution_not_authorized", ROLE_REASON_CODES)
        self.assertEqual(MASTER_NAMESPACE, "p08-qc-inner-master-v1")
        self.assertEqual(ROLE_PAIR_NAMESPACE, "p08-qc-inner-role-pair-v1")


if __name__ == "__main__":
    unittest.main()
