"""Tests for the P08-T007 metadata-only nested-QC feasibility auditor."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from types import MappingProxyType

from atlas_sers.evaluation.p08_qc_support import (
    AMENDMENT_STATUS,
    ERROR_REASON_CODES,
    MASTER_MODE,
    PSEUDO_MODE,
    REASON_CODES,
    REQUIRED_INNER_FOLDS,
    SCHEMA_VERSION,
    QCSupportError,
    audit_nested_qc_support,
    require_scientific_execution,
)
from atlas_sers.governance.canonical import sha256_value

SENTINEL = "SENTINEL-PRIVATE-424242"


def recompute(value):
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
        self._seen = set()
        self._counter = 0

    def obs(self, master_id, instrument, label):
        self._counter += 1
        uid = f"obs-{self._counter:06d}"
        if uid in self._seen:
            raise AssertionError("duplicate generated observation uid")
        self._seen.add(uid)
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
    corpus, context_id, viable=True, n_units=2, classes=("A", "B"), test_classes=None
):
    units = []
    outer_fit = []
    for unit_index in range(n_units):
        fit_uids = []
        for class_id in classes:
            if viable or class_id != "B":
                masters_needed = 3
            else:
                masters_needed = 2
            for master_index in range(masters_needed):
                fit_uids.append(
                    corpus.obs(
                        f"{context_id}-u{unit_index}-fit-{class_id}-{master_index}",
                        f"{context_id}-fit-inst-{unit_index}",
                        class_id,
                    )
                )
            if not viable and class_id == "B":
                # Repeated observation of an already-counted master.
                fit_uids.append(
                    corpus.obs(
                        f"{context_id}-u{unit_index}-fit-{class_id}-0",
                        f"{context_id}-fit-inst-{unit_index}",
                        class_id,
                    )
                )
        validation_uids = [
            corpus.obs(
                f"{context_id}-u{unit_index}-val-{class_id}",
                f"{context_id}-val-inst-{unit_index}",
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
                        f"{context_id}-f{fold_index}-{class_id}-{master_index}",
                        f"{context_id}-fit-inst",
                        class_id,
                    )
                )
        folds.append(fold)
        outer_fit.extend(fold)
    units = []
    for fold_index in range(3):
        fit_uids = [
            uid
            for other in range(3)
            if other != fold_index
            for uid in folds[other]
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


def role_by_uid(roles, uid):
    for row in roles:
        if row["observation_uid"] == uid:
            return row
    raise KeyError(uid)


def label_index(roles):
    return {row["observation_uid"]: row["label"] for row in roles}


def viable_pseudo_case(context_id="CTX-P"):
    corpus = Corpus()
    context = add_pseudo_context(corpus, context_id, viable=True)
    return [context], corpus.roles


def full_case():
    corpus = Corpus()
    contexts = [
        add_pseudo_context(corpus, "CTX-P", viable=True),
        add_master_context(corpus, "CTX-M", per_class=1),
    ]
    return contexts, corpus.roles


class QCTestCase(unittest.TestCase):
    def assert_code(self, code, contexts, roles):
        with self.assertRaises(QCSupportError) as caught:
            audit_nested_qc_support(contexts, roles)
        self.assertEqual(str(caught.exception), code)
        self.assertEqual(caught.exception.reason_code, code)
        self.assertIn(code, ERROR_REASON_CODES)
        self.assertNotIn(SENTINEL, str(caught.exception))


class PlanConstructionTests(QCTestCase):
    def test_top_level_shape_and_denied_execution(self):
        contexts, roles = full_case()
        plan = audit_nested_qc_support(contexts, roles)
        self.assertEqual(SCHEMA_VERSION, "nato-sers-p08-qc-nested-support-v1")
        self.assertEqual(plan["schema_version"], SCHEMA_VERSION)
        self.assertIs(plan["execution_authorized"], False)
        self.assertEqual(plan["amendment_status"], AMENDMENT_STATUS)
        self.assertEqual(
            plan["amendment_status"], "owner_approved_support_rule_no_execution_permit"
        )
        self.assertEqual(plan["required_inner_folds"], 3)
        self.assertEqual(REQUIRED_INNER_FOLDS, 3)
        self.assertEqual(len(plan["audit_sha256"]), 64)
        self.assertIsInstance(plan["contexts"], list)
        self.assertIsInstance(plan["summary"], dict)

        with self.assertRaises(QCSupportError) as caught:
            require_scientific_execution(plan)
        self.assertEqual(str(caught.exception), "scientific_execution_not_authorized")
        with self.assertRaises(QCSupportError):
            require_scientific_execution({"execution_authorized": True})
        with self.assertRaises(QCSupportError):
            require_scientific_execution()

    def test_reason_code_set_is_exact(self):
        self.assertEqual(
            REASON_CODES,
            {
                "nested_three_fold_supported",
                "insufficient_nested_class_masters",
                "no_source_pseudo_domains",
            },
        )

    def test_audit_hash_independently_recomputed(self):
        contexts, roles = full_case()
        plan = audit_nested_qc_support(contexts, roles)
        payload = {key: value for key, value in plan.items() if key != "audit_sha256"}
        self.assertEqual(plan["audit_sha256"], recompute(payload))

    def test_strict_json_roundtrip(self):
        contexts, roles = full_case()
        plan = audit_nested_qc_support(contexts, roles)
        encoded = json.dumps(plan, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["audit_sha256"], plan["audit_sha256"])
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)

    def test_contexts_sorted(self):
        corpus = Corpus()
        contexts = [
            add_pseudo_context(corpus, "CTX-Z", viable=True),
            add_master_context(corpus, "CTX-A", per_class=1),
        ]
        plan = audit_nested_qc_support(contexts, corpus.roles)
        ids = [detail["context_id"] for detail in plan["contexts"]]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(ids, ["CTX-A", "CTX-Z"])

    def test_summary_contains_no_identities(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-SECRET-ALPHA", viable=True)
        plan = audit_nested_qc_support([context], corpus.roles)
        text = json.dumps(plan["summary"], sort_keys=True)
        self.assertNotIn("CTX-SECRET-ALPHA", text)
        for unit in context["selection_units"]:
            self.assertNotIn(unit["unit_id"], text)
            for uid in unit["fit_uids"] + unit["validation_uids"]:
                self.assertNotIn(uid, text)

    def test_viable_pseudo_context_is_eligible(self):
        contexts, roles = viable_pseudo_case()
        plan = audit_nested_qc_support(contexts, roles)
        detail = plan["contexts"][0]
        self.assertEqual(detail["reason_code"], "nested_three_fold_supported")
        self.assertTrue(detail["eligible"])
        self.assertIs(plan["execution_authorized"], False)
        for unit in detail["units"]:
            self.assertTrue(unit["meets_inner_three_fold"])
            self.assertEqual(unit["min_class_masters"], 3)
            self.assertEqual(unit["class_master_counts"], {"A": 3, "B": 3})

    def test_min_one_master_is_insufficient(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        unit = context["selection_units"][0]
        # Keep A0, A1, A2, B0 only: class B collapses to a single master.
        unit["fit_uids"] = unit["fit_uids"][:4]
        plan = audit_nested_qc_support([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertEqual(detail["reason_code"], "insufficient_nested_class_masters")
        self.assertFalse(detail["eligible"])
        first = detail["units"][0]
        self.assertEqual(first["min_class_masters"], 1)
        self.assertFalse(first["meets_inner_three_fold"])

    def test_master_cv_always_fallback(self):
        corpus = Corpus()
        context = add_master_context(corpus, "CTX-MC", per_class=1)
        plan = audit_nested_qc_support([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertEqual(detail["reason_code"], "no_source_pseudo_domains")
        self.assertFalse(detail["eligible"])

    def test_master_cv_fallback_even_when_counts_sufficient(self):
        corpus = Corpus()
        context = add_master_context(corpus, "CTX-MC", per_class=2)
        plan = audit_nested_qc_support([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertEqual(detail["reason_code"], "no_source_pseudo_domains")
        self.assertFalse(detail["eligible"])
        for unit in detail["units"]:
            self.assertTrue(unit["meets_inner_three_fold"])
            self.assertGreaterEqual(unit["min_class_masters"], 3)

    def test_two_pseudo_units_one_not_viable_falls_back(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        second = context["selection_units"][1]
        second["fit_uids"] = second["fit_uids"][:5]
        plan = audit_nested_qc_support([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertEqual(detail["reason_code"], "insufficient_nested_class_masters")
        self.assertFalse(detail["eligible"])
        by_unit = {unit["unit_id"]: unit for unit in detail["units"]}
        self.assertTrue(by_unit["CTX-P-u0"]["meets_inner_three_fold"])
        self.assertFalse(by_unit["CTX-P-u1"]["meets_inner_three_fold"])

    def test_repeated_master_observations_do_not_inflate_support(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=False)
        labels = label_index(corpus.roles)
        unit = context["selection_units"][0]
        b_uid_count = sum(1 for uid in unit["fit_uids"] if labels[uid] == "B")
        self.assertEqual(b_uid_count, 3)
        plan = audit_nested_qc_support([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertEqual(detail["reason_code"], "insufficient_nested_class_masters")
        self.assertEqual(detail["units"][0]["class_master_counts"]["B"], 2)
        self.assertEqual(detail["units"][0]["min_class_masters"], 2)

    def test_three_master_folds_exact_partition(self):
        corpus = Corpus()
        context = add_master_context(corpus, "CTX-MC", per_class=1)
        plan = audit_nested_qc_support([context], corpus.roles)
        self.assertEqual(plan["contexts"][0]["reason_code"], "no_source_pseudo_domains")
        outer_fit = set(context["outer_fit_uids"])
        for unit in context["selection_units"]:
            self.assertEqual(
                set(unit["fit_uids"]), outer_fit - set(unit["validation_uids"])
            )

    def test_tuple_inputs_accepted_and_order_invariance(self):
        contexts, roles = full_case()
        from_lists = audit_nested_qc_support(list(contexts), list(roles))
        from_tuples = audit_nested_qc_support(tuple(contexts), tuple(roles))
        self.assertEqual(from_lists["audit_sha256"], from_tuples["audit_sha256"])

    def test_ordering_invariance_rows_units_and_classes(self):
        contexts, roles = full_case()
        base = audit_nested_qc_support(contexts, roles)

        shuffled_contexts = copy.deepcopy(contexts)
        shuffled_contexts.reverse()
        for context in shuffled_contexts:
            context["classes"].reverse()
            context["selection_units"].reverse()
        shuffled_roles = copy.deepcopy(roles)
        shuffled_roles.reverse()

        shuffled = audit_nested_qc_support(shuffled_contexts, shuffled_roles)
        self.assertEqual(base["audit_sha256"], shuffled["audit_sha256"])

    def test_inputs_unchanged(self):
        contexts, roles = full_case()
        contexts_copy = copy.deepcopy(contexts)
        roles_copy = copy.deepcopy(roles)
        audit_nested_qc_support(contexts, roles)
        self.assertEqual(contexts, contexts_copy)
        self.assertEqual(roles, roles_copy)

    def test_extra_unreferenced_rows_allowed(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        corpus.obs("orphan-master", "orphan-instrument", "ORPHAN")
        plan = audit_nested_qc_support([context], corpus.roles)
        self.assertEqual(plan["summary"]["eligible_contexts"], 1)

    def test_summary_arithmetic_and_histogram(self):
        corpus = Corpus()
        contexts = [
            add_pseudo_context(corpus, "CTX-P1", viable=True),
            add_pseudo_context(corpus, "CTX-P2", viable=False),
            add_master_context(corpus, "CTX-M", per_class=1),
        ]
        plan = audit_nested_qc_support(contexts, corpus.roles)
        summary = plan["summary"]
        self.assertEqual(summary["total_contexts"], 3)
        self.assertEqual(summary["pseudo_contexts"], 2)
        self.assertEqual(summary["mastercv_contexts"], 1)
        self.assertEqual(summary["eligible_contexts"], 1)
        self.assertEqual(summary["fallback_contexts"], 2)
        self.assertEqual(sum(summary["reason_counts"].values()), 3)
        self.assertEqual(summary["total_pseudo_units"], 4)
        self.assertEqual(summary["individually_supported_pseudo_units"], 2)
        self.assertEqual(summary["eligible_pseudo_units"], 2)
        self.assertEqual(
            summary["pseudo_unit_min_masters_histogram"], {"2": 2, "3": 2}
        )

    def test_aggregate_fixture_without_private_ids(self):
        corpus = Corpus()
        contexts = []
        for index in range(128):
            viable = index < 54
            contexts.append(
                add_pseudo_context(corpus, f"PSEUDO-{index:03d}", viable=viable)
            )
        for index in range(132):
            contexts.append(add_master_context(corpus, f"MASTER-{index:03d}", per_class=1))

        plan = audit_nested_qc_support(contexts, corpus.roles)
        summary = plan["summary"]
        self.assertEqual(summary["total_contexts"], 260)
        self.assertEqual(summary["pseudo_contexts"], 128)
        self.assertEqual(summary["mastercv_contexts"], 132)
        self.assertEqual(summary["eligible_contexts"], 54)
        self.assertEqual(summary["fallback_contexts"], 206)
        self.assertEqual(
            summary["reason_counts"],
            {
                "nested_three_fold_supported": 54,
                "insufficient_nested_class_masters": 74,
                "no_source_pseudo_domains": 132,
            },
        )
        self.assertEqual(summary["total_pseudo_units"], 256)
        self.assertEqual(summary["individually_supported_pseudo_units"], 108)
        self.assertEqual(summary["eligible_pseudo_units"], 108)
        self.assertEqual(
            summary["pseudo_unit_min_masters_histogram"], {"2": 148, "3": 108}
        )

    def test_outer_test_may_hold_two_of_three_classes(self):
        corpus = Corpus()
        context = add_pseudo_context(
            corpus,
            "CTX-P23",
            viable=True,
            classes=("A", "B", "C"),
            test_classes=("A", "C"),
        )
        plan = audit_nested_qc_support([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertEqual(detail["reason_code"], "nested_three_fold_supported")
        self.assertTrue(detail["eligible"])
        self.assertEqual(detail["classes"], ["A", "B", "C"])

    def test_outer_test_may_hold_single_class(self):
        corpus = Corpus()
        context = add_pseudo_context(
            corpus,
            "CTX-P1C",
            viable=True,
            classes=("A", "B", "C"),
            test_classes=("B",),
        )
        plan = audit_nested_qc_support([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertEqual(detail["reason_code"], "nested_three_fold_supported")
        self.assertTrue(detail["eligible"])

    def test_master_outer_test_may_hold_subset(self):
        corpus = Corpus()
        context = add_master_context(
            corpus,
            "CTX-MSU",
            per_class=1,
            classes=("A", "B", "C"),
            test_classes=("A",),
        )
        plan = audit_nested_qc_support([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertEqual(detail["reason_code"], "no_source_pseudo_domains")
        self.assertFalse(detail["eligible"])

    def test_mapping_proxy_inputs_accepted(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-MP", viable=True)
        context_proxy = MappingProxyType(
            {
                **context,
                "selection_units": [
                    MappingProxyType(unit) for unit in context["selection_units"]
                ],
            }
        )
        role_proxies = [MappingProxyType(row) for row in corpus.roles]
        plan = audit_nested_qc_support([context_proxy], role_proxies)
        self.assertEqual(plan["summary"]["eligible_contexts"], 1)

    def test_audit_hash_matches_governance_canonical_json(self):
        corpus = Corpus()
        context = add_pseudo_context(
            corpus, "CTX-ÜNI", viable=True, classes=("Ä", "B", "Ω")
        )
        plan = audit_nested_qc_support([context], corpus.roles)
        payload = {key: value for key, value in plan.items() if key != "audit_sha256"}
        self.assertEqual(plan["audit_sha256"], sha256_value(payload))
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

    def test_summary_distinguishes_individual_from_context_eligibility(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        second = context["selection_units"][1]
        second["fit_uids"] = second["fit_uids"][:5]
        plan = audit_nested_qc_support([context], corpus.roles)
        summary = plan["summary"]
        self.assertEqual(summary["total_pseudo_units"], 2)
        self.assertEqual(summary["individually_supported_pseudo_units"], 1)
        self.assertEqual(summary["eligible_pseudo_units"], 0)
        self.assertEqual(summary["eligible_contexts"], 0)

    def test_same_master_two_instruments_in_same_fit_role_counts_once(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        unit = context["selection_units"][0]
        original = role_by_uid(corpus.roles, unit["fit_uids"][0])
        extra_uid = corpus.obs(
            original["master_id"], "CTX-P-alt-inst-0", original["label"]
        )
        unit["fit_uids"].append(extra_uid)
        context["outer_fit_uids"].append(extra_uid)
        plan = audit_nested_qc_support([context], corpus.roles)
        detail = plan["contexts"][0]
        self.assertEqual(detail["reason_code"], "nested_three_fold_supported")
        first_unit = detail["units"][0]
        self.assertEqual(first_unit["class_master_counts"][original["label"]], 3)
        self.assertTrue(first_unit["meets_inner_three_fold"])


class ValidationTests(QCTestCase):
    def test_wrong_container_types(self):
        contexts, roles = viable_pseudo_case()
        self.assert_code("contexts_must_be_sequence", "not-a-list", roles)
        self.assert_code("contexts_must_be_sequence", True, roles)
        self.assert_code("contexts_must_be_sequence", {"contexts": []}, roles)
        self.assert_code("roles_must_be_sequence", contexts, "not-a-list")
        self.assert_code("roles_must_be_sequence", contexts, True)
        self.assert_code("contexts_empty", [], roles)

        bad = copy.deepcopy(contexts)
        bad[0] = True
        self.assert_code("context_must_be_mapping", bad, roles)

    def test_str_as_sequence_and_bool_fields(self):
        contexts, roles = viable_pseudo_case()

        bad = copy.deepcopy(contexts)
        bad[0]["classes"] = "AB"
        self.assert_code("classes_must_be_sequence", bad, roles)

        bad = copy.deepcopy(contexts)
        bad[0]["outer_fit_uids"] = "abc"
        self.assert_code("outer_fit_uids_must_be_sequence", bad, roles)

        bad = copy.deepcopy(contexts)
        bad[0]["selection_units"] = "units"
        self.assert_code("selection_units_must_be_sequence", bad, roles)

        bad = copy.deepcopy(contexts)
        bad[0]["selection_units"][0]["fit_uids"] = "abc"
        self.assert_code("fit_uids_must_be_sequence", bad, roles)

        bad = copy.deepcopy(contexts)
        bad[0]["selection_units"][0]["validation_uids"] = True
        self.assert_code("validation_uids_must_be_sequence", bad, roles)

        bad = copy.deepcopy(contexts)
        bad[0]["held_instrument"] = True
        self.assert_code("held_instrument_invalid", bad, roles)

        bad_roles = copy.deepcopy(roles)
        bad_roles[0]["master_id"] = True
        self.assert_code("role_field_invalid", contexts, bad_roles)

        bad = copy.deepcopy(contexts)
        bad[0]["selection_units"][0]["unit_id"] = 17
        self.assert_code("unit_id_invalid", bad, roles)

    def test_duplicate_ids_and_keys(self):
        contexts, roles = viable_pseudo_case()

        bad = copy.deepcopy(contexts)
        bad.append(copy.deepcopy(contexts[0]))
        self.assert_code("duplicate_context_id", bad, roles)

        bad = copy.deepcopy(contexts)
        bad[0]["selection_units"][1]["unit_id"] = bad[0]["selection_units"][0]["unit_id"]
        self.assert_code("duplicate_unit_id", bad, roles)

        bad = copy.deepcopy(contexts)
        bad[0]["selection_units"][0]["fit_uids"].append(
            bad[0]["selection_units"][0]["fit_uids"][0]
        )
        self.assert_code("duplicate_uid", bad, roles)

        bad = copy.deepcopy(contexts)
        bad[0]["classes"] = ["A", "B", "A"]
        self.assert_code("duplicate_class_id", bad, roles)

        bad_roles = copy.deepcopy(roles)
        bad_roles.append(copy.deepcopy(roles[0]))
        self.assert_code("duplicate_observation_uid", contexts, bad_roles)

        bad = copy.deepcopy(contexts)
        bad[0]["extra"] = 1
        self.assert_code("context_keys_invalid", bad, roles)

        bad = copy.deepcopy(contexts)
        bad[0]["selection_units"][0]["extra"] = 1
        self.assert_code("unit_keys_invalid", bad, roles)

        bad = copy.deepcopy(contexts)
        del bad[0]["held_instrument"]
        self.assert_code("context_keys_invalid", bad, roles)

        bad_roles = copy.deepcopy(roles)
        bad_roles[0]["extra"] = 1
        self.assert_code("role_keys_invalid", contexts, bad_roles)

    def test_missing_referenced_rows(self):
        contexts, roles = viable_pseudo_case()
        missing = contexts[0]["selection_units"][0]["fit_uids"][0]
        trimmed = [row for row in roles if row["observation_uid"] != missing]
        self.assert_code("unknown_observation_uid", contexts, trimmed)
        self.assert_code("unknown_observation_uid", contexts, [])

    def test_ambiguous_master_labels(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        master = corpus.roles[0]["master_id"]
        corpus.obs(master, "other-instrument", "Z")
        self.assert_code("ambiguous_master_label", [context], corpus.roles)

    def test_same_master_crossing_role_boundary(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        unit = context["selection_units"][0]
        fit_role = role_by_uid(corpus.roles, unit["fit_uids"][0])
        validation_role = role_by_uid(corpus.roles, unit["validation_uids"][0])
        validation_role["master_id"] = fit_role["master_id"]
        self.assert_code("selection_fit_validation_master_overlap", [context], corpus.roles)

        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        fit_role = role_by_uid(corpus.roles, context["outer_fit_uids"][0])
        test_role = role_by_uid(corpus.roles, context["outer_test_uids"][0])
        test_role["master_id"] = fit_role["master_id"]
        self.assert_code("fit_test_master_overlap", [context], corpus.roles)

    def test_held_instrument_leakage(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        role_by_uid(corpus.roles, context["selection_units"][0]["fit_uids"][0])[
            "instrument"
        ] = "held-inst"
        self.assert_code("held_instrument_in_outer_fit", [context], corpus.roles)

        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        role_by_uid(corpus.roles, context["outer_test_uids"][0])[
            "instrument"
        ] = "somewhere-else"
        self.assert_code("outer_test_instrument_mismatch", [context], corpus.roles)

    def test_pseudo_instrument_leakage(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        unit = context["selection_units"][0]
        for uid in unit["validation_uids"]:
            role_by_uid(corpus.roles, uid)["instrument"] = "CTX-P-fit-inst-0"
        self.assert_code(
            "pseudo_validation_instrument_in_selection_fit", [context], corpus.roles
        )

        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        unit = context["selection_units"][0]
        for uid in unit["validation_uids"]:
            role_by_uid(corpus.roles, uid)["instrument"] = "mixed-instrument-" + uid
        self.assert_code(
            "pseudo_validation_multiple_instruments", [context], corpus.roles
        )

        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        second = context["selection_units"][1]
        shared = "CTX-P-val-inst-0"
        for uid in second["validation_uids"]:
            role_by_uid(corpus.roles, uid)["instrument"] = shared
        self.assert_code(
            "pseudo_validation_instruments_not_distinct", [context], corpus.roles
        )

    def test_incomplete_class_support(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        labels = label_index(corpus.roles)
        unit = context["selection_units"][0]
        unit["fit_uids"] = [uid for uid in unit["fit_uids"] if labels[uid] != "B"]
        self.assert_code("selection_fit_missing_classes", [context], corpus.roles)

        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        labels = label_index(corpus.roles)
        unit = context["selection_units"][0]
        unit["validation_uids"] = [
            uid for uid in unit["validation_uids"] if labels[uid] != "B"
        ]
        self.assert_code("selection_validation_missing_classes", [context], corpus.roles)

        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        labels = label_index(corpus.roles)
        context["outer_fit_uids"] = [
            uid for uid in context["outer_fit_uids"] if labels[uid] != "B"
        ]
        self.assert_code("outer_fit_missing_classes", [context], corpus.roles)

    def test_invalid_selection_mode(self):
        contexts, roles = viable_pseudo_case()
        for invented in ("holdout", "pseudo_cv", SENTINEL):
            with self.subTest(mode=invented):
                bad = copy.deepcopy(contexts)
                bad[0]["selection_mode"] = invented
                self.assert_code("selection_mode_invalid", bad, roles)

    def test_master_folds_partition_break(self):
        corpus = Corpus()
        context = add_master_context(corpus, "CTX-MC", per_class=1)
        extra_uid = corpus.obs("CTX-MC-extra", "CTX-MC-fit-inst", "A")
        context["outer_fit_uids"].append(extra_uid)
        self.assert_code("master_folds_not_partition", [context], corpus.roles)

    def test_master_fit_not_outer_minus_fold(self):
        corpus = Corpus()
        context = add_master_context(corpus, "CTX-MC", per_class=1)
        unit = context["selection_units"][0]
        unit["fit_uids"] = unit["fit_uids"][1:]
        self.assert_code("master_fit_not_outer_minus_fold", [context], corpus.roles)

    def test_outer_fit_test_uid_overlap(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        context["outer_test_uids"].append(context["outer_fit_uids"][0])
        self.assert_code("fit_test_uid_overlap", [context], corpus.roles)

    def test_label_not_in_context_classes(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        roles = copy.deepcopy(corpus.roles)
        role_by_uid(roles, context["selection_units"][0]["fit_uids"][0])[
            "label"
        ] = "NOT-A-CLASS"
        self.assert_code("label_not_in_context_classes", [context], roles)

    def test_malformed_errors_do_not_leak_sentinel(self):
        contexts, roles = viable_pseudo_case()
        cases = []

        bad = copy.deepcopy(contexts)
        bad[0]["context_id"] = f" {SENTINEL}"
        cases.append(("context_id_invalid", bad, roles))

        bad = copy.deepcopy(contexts)
        bad[0]["classes"] = [SENTINEL, SENTINEL]
        cases.append(("duplicate_class_id", bad, roles))

        bad = copy.deepcopy(contexts)
        bad[0]["selection_units"][0]["fit_uids"].append(SENTINEL)
        cases.append(("unknown_observation_uid", bad, roles))

        bad_roles = copy.deepcopy(roles)
        bad_roles[0][SENTINEL] = SENTINEL
        cases.append(("role_keys_invalid", contexts, bad_roles))

        for code, bad_contexts, bad_roles in cases:
            with self.subTest(code=code):
                self.assert_code(code, bad_contexts, bad_roles)

    def test_execution_refusal_even_with_forged_flag(self):
        corpus = Corpus()
        context = add_pseudo_context(corpus, "CTX-P", viable=True)
        plan = audit_nested_qc_support([context], corpus.roles)
        forged = dict(plan)
        forged["execution_authorized"] = True
        self.assertIs(plan["execution_authorized"], False)
        with self.assertRaises(QCSupportError) as caught:
            require_scientific_execution(forged)
        self.assertEqual(caught.exception.reason_code, "scientific_execution_not_authorized")


if __name__ == "__main__":
    unittest.main()
