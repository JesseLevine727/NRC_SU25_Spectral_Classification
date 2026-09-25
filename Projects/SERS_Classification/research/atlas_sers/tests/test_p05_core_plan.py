"""Standard-library tests for the metadata-only P05 core plan module.

No NumPy, torch, spectra, or outcome data are used. Contract-dependent tests
load the public locked contract file; no test bypasses actual build validation.
"""

from __future__ import annotations

import copy
import json
import random
import unittest
from pathlib import Path

from atlas_sers.evaluation import p05_core_plan as core

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "plan" / "contracts" / "p05_core_contract.json"
CONTRACT_AVAILABLE = CONTRACT_PATH.is_file()
requires_contract = unittest.skipUnless(
    CONTRACT_AVAILABLE, "locked P05 core contract file is unavailable"
)


def load_contract() -> dict:
    with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dense_entry(
    entry_id: str, station: str, *, instruments: int = 3, cross: int = 1, phase: str = "development"
) -> dict:
    return {
        "role": "selection_fit",
        "phase_gate": phase,
        "station": station,
        "instrument_count": instruments,
        "master_count": 10,
        "parent_role_id": entry_id,
        "selection_unit_id": "unit:" + entry_id,
        "audit_id": "P05AUDIT-" + entry_id,
        "same_chemical_pair_categories": {"same_master_different_instrument": cross},
    }


def sparse_entry(
    entry_id: str, masters: int, *, phase: str = "development", instruments: int = 1
) -> dict:
    return {
        "role": "selection_fit",
        "phase_gate": phase,
        "station": core.SPARSE_STATION,
        "instrument_count": instruments,
        "master_count": masters,
        "parent_role_id": entry_id,
        "selection_unit_id": "unit:" + entry_id,
        "audit_id": "P05AUDIT-" + entry_id,
        "same_chemical_pair_categories": {"same_master_different_instrument": 0},
    }


def observation(
    uid: str,
    master: str,
    target: str,
    instrument: str,
    *,
    station: str = "cwa",
    substrate: str = "glass",
) -> dict:
    return {
        "uid": uid,
        "master": master,
        "station": station,
        "target": target,
        "instrument": instrument,
        "substrate": substrate,
    }


class SelectSmokeRolesTest(unittest.TestCase):
    def _dense(self) -> list:
        return [
            dense_entry("cwa_role_b", "cwa"),
            dense_entry("cwa_role_a", "cwa"),
            dense_entry("pills_role_a", "pills"),
            dense_entry("surfaces_role_a", "surfaces"),
        ]

    def test_declared_role_label_order(self):
        entries = self._dense() + [sparse_entry("sparse_z", 9)]
        selected = core.select_smoke_roles(entries)
        self.assertEqual(
            [record["role_label"] for record in selected],
            ["cwa_dense", "pills_dense", "surfaces_dense", "surfaces_sparse"],
        )

    def test_dense_lexicographic_first(self):
        entries = self._dense() + [sparse_entry("sparse_z", 4, phase="held_evaluation")]
        selected = core.select_smoke_roles(entries)
        by_label = {record["role_label"]: record["audit"] for record in selected}
        self.assertEqual(by_label["cwa_dense"]["parent_role_id"], "cwa_role_a")
        self.assertEqual(by_label["cwa_dense"]["phase_gate"], "development")

    def test_sparse_held_evaluation_selected_for_minimum_masters(self):
        entries = [
            dense_entry("cwa_role_a", "cwa"),
            dense_entry("pills_role_a", "pills"),
            dense_entry("surfaces_role_a", "surfaces"),
            sparse_entry("dev_sparse", 9, phase="development"),
            sparse_entry("held_sparse", 4, phase="held_evaluation"),
        ]
        sparse = core.select_smoke_roles(entries)[-1]
        self.assertEqual(sparse["role_label"], "surfaces_sparse")
        self.assertEqual(sparse["audit"]["parent_role_id"], "held_sparse")
        self.assertEqual(sparse["audit"]["phase_gate"], "held_evaluation")

    def test_sparse_tie_breaks_on_parent_role_id(self):
        entries = [
            dense_entry("cwa_role_a", "cwa"),
            dense_entry("pills_role_a", "pills"),
            dense_entry("surfaces_role_a", "surfaces"),
            sparse_entry("z_sparse", 4, phase="held_evaluation"),
            sparse_entry("a_sparse", 4, phase="development"),
        ]
        sparse = core.select_smoke_roles(entries)[-1]
        self.assertEqual(sparse["audit"]["parent_role_id"], "a_sparse")

    def test_missing_dense_support_fails(self):
        entries = [dense_entry("cwa_role_a", "cwa"), sparse_entry("s", 4)]
        with self.assertRaises(core.CorePlanError):
            core.select_smoke_roles(entries)

    def test_missing_sparse_support_fails(self):
        entries = [
            dense_entry("cwa_role_a", "cwa"),
            dense_entry("pills_role_a", "pills"),
            dense_entry("surfaces_role_a", "surfaces"),
        ]
        with self.assertRaises(core.CorePlanError):
            core.select_smoke_roles(entries)

    def test_row_order_invariance(self):
        base = self._dense() + [sparse_entry("held_sparse", 4, phase="held_evaluation")]
        first = [
            (record["role_label"], record["audit"]["parent_role_id"])
            for record in core.select_smoke_roles(base)
        ]
        shuffled = list(base)
        random.Random(7).shuffle(shuffled)
        second = [
            (record["role_label"], record["audit"]["parent_role_id"])
            for record in core.select_smoke_roles(shuffled)
        ]
        self.assertEqual(first, second)


@requires_contract
class ContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()

    def test_locked_contract_authenticates(self):
        summary = core.validate_core_contract(self.contract)
        self.assertEqual(summary["contract_sha256"], core.LOCKED_CONTRACT_SHA256)
        self.assertFalse(summary["full_development_authorized"])
        self.assertFalse(summary["outer_evaluation_authorized"])
        self.assertEqual(summary["authorized_stage"], "bounded_source_fit_numerical_smoke_only")

    def test_budget_expansion_rejected(self):
        mutated = copy.deepcopy(self.contract)
        mutated["later_core_plan"]["inner_fit_slot_ceiling"] += 1
        with self.assertRaises(core.CorePlanError):
            core.validate_core_contract(mutated)

    def test_recipe_tampering_rejected(self):
        mutated = copy.deepcopy(self.contract)
        mutated["recipes"][1]["lambda_supcon"] = 0.4
        with self.assertRaises(core.CorePlanError):
            core.validate_core_contract(mutated)

    def test_added_unknown_recipe_rejected(self):
        mutated = copy.deepcopy(self.contract)
        mutated["recipes"].append(
            {"recipe_id": "D4", "lambda_supcon": 0.5, "lambda_pair": 0.5, "projection": True}
        )
        with self.assertRaises(core.CorePlanError):
            core.validate_core_contract(mutated)

    def test_missing_field_rejected(self):
        mutated = copy.deepcopy(self.contract)
        del mutated["g3"]["fallback"]
        with self.assertRaises(core.CorePlanError):
            core.validate_core_contract(mutated)

    def test_smoke_schedule_arithmetic(self):
        summary = core.check_smoke_schedule(self.contract["smoke"], self.contract["sampler"])
        self.assertEqual(summary["primary_fit_count"], 32)
        self.assertEqual(summary["replay_count"], 2)
        self.assertEqual(summary["maximum_fit_executions"], 34)
        self.assertEqual(summary["maximum_optimizer_steps"], 1088)

    def test_later_ledger_arithmetic(self):
        ledger = core.check_later_ledger(self.contract["later_core_plan"])
        self.assertEqual(ledger["guard_unit_slots"], 384)
        self.assertEqual(ledger["inner_fit_slot_ceiling"], 14940)
        self.assertEqual(ledger["later_neural_fit_ceiling"], 17820)
        self.assertEqual(ledger["including_smoke_neural_execution_ceiling"], 17854)

    def test_ledger_rejects_nondecomposable_ceiling(self):
        mutated = copy.deepcopy(self.contract["later_core_plan"])
        mutated["inner_fit_slot_ceiling"] += 12
        with self.assertRaises(core.CorePlanError):
            core.check_later_ledger(mutated)


@requires_contract
class SmokeFitsTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()

    def _selected_roles(self) -> dict:
        roles = {}
        for label in self.contract["smoke"]["role_labels"]:
            roles[label] = {
                "p05_role_id": core.core_role_identity(
                    contract_sha256=core.LOCKED_CONTRACT_SHA256,
                    role_label=label,
                    parent_context_id="ctx:" + label,
                    parent_role_id="role:" + label,
                    role="selection_fit",
                    selection_unit_id="unit:" + label,
                )
            }
        return roles

    def test_primary_and_replay_identities(self):
        fits = core.expand_smoke_fits(
            contract=self.contract,
            contract_sha256=core.LOCKED_CONTRACT_SHA256,
            input_pins=self.contract["input_pins"],
            selected_roles=self._selected_roles(),
        )
        self.assertEqual(len(fits), 34)
        primary = [row for row in fits if row["execution_kind"] == "primary"]
        replay = [row for row in fits if row["execution_kind"] == "replay"]
        self.assertEqual(len(primary), 32)
        self.assertEqual(len(replay), 2)
        by_key = {(row["role_label"], row["recipe_id"], row["seed"]): row for row in primary}
        for row in replay:
            base = by_key[(row["role_label"], row["recipe_id"], row["seed"])]
            self.assertEqual(row["fit_id"], base["fit_id"])
            self.assertEqual(row["replay_of"], base["execution_id"])
            self.assertNotEqual(row["execution_id"], base["execution_id"])
            self.assertIsNone(base["replay_of"])

    def test_optimizer_step_totals(self):
        fits = core.expand_smoke_fits(
            contract=self.contract,
            contract_sha256=core.LOCKED_CONTRACT_SHA256,
            input_pins=self.contract["input_pins"],
            selected_roles=self._selected_roles(),
        )
        self.assertEqual(sum(row["optimizer_steps"] for row in fits), 1088)
        self.assertTrue(all(row["optimizer_steps"] == 32 for row in fits))

    def test_shared_rng_group_is_role_and_seed_only(self):
        self.assertEqual(
            core.shared_rng_group(p05_role_id="r", seed=1),
            core.shared_rng_group(p05_role_id="r", seed=1),
        )
        self.assertNotEqual(
            core.shared_rng_group(p05_role_id="r", seed=1),
            core.shared_rng_group(p05_role_id="r", seed=2),
        )
        self.assertNotEqual(
            core.shared_rng_group(p05_role_id="r", seed=1),
            core.shared_rng_group(p05_role_id="r2", seed=1),
        )

    def test_fit_identity_sensitivity(self):
        base = {
            "contract_sha256": core.LOCKED_CONTRACT_SHA256,
            "input_pins": self.contract["input_pins"],
            "p05_role_id": "r",
            "recipe_id": "D3",
            "seed": 20260805,
        }
        original = core.fit_identity(**base)
        self.assertNotEqual(original, core.fit_identity(**{**base, "seed": 20260817}))
        self.assertNotEqual(original, core.fit_identity(**{**base, "recipe_id": "D1"}))
        self.assertNotEqual(original, core.fit_identity(**{**base, "p05_role_id": "r2"}))

    def test_execution_identity_kinds(self):
        primary = core.execution_identity(fit_id="f", execution_kind="primary")
        replay0 = core.execution_identity(fit_id="f", execution_kind="replay", replay_index=0)
        replay1 = core.execution_identity(fit_id="f", execution_kind="replay", replay_index=1)
        self.assertEqual(len({primary, replay0, replay1}), 3)


@requires_contract
class PairIdentityTest(unittest.TestCase):
    def setUp(self):
        self.pins = load_contract()["input_pins"]

    def _pair(
        self, uid_a: str, uid_b: str, *, role: str = "role", contract_sha256: str | None = None
    ) -> str:
        return core.core_pair_identity(
            contract_sha256=contract_sha256 or core.LOCKED_CONTRACT_SHA256,
            input_pins=self.pins,
            p05_role_id=role,
            uid_a=uid_a,
            uid_b=uid_b,
        )

    def test_symmetry(self):
        self.assertEqual(self._pair("u1", "u2"), self._pair("u2", "u1"))

    def test_role_and_contract_sensitivity(self):
        base = self._pair("u1", "u2")
        self.assertNotEqual(base, self._pair("u1", "u2", role="other"))
        self.assertNotEqual(base, self._pair("u1", "u2", contract_sha256="0" * 64))

    def test_repeated_observation_distinction(self):
        self.assertNotEqual(self._pair("u1", "u2"), self._pair("u1", "u3"))

    def test_self_and_bad_uids_rejected(self):
        with self.assertRaises(core.CorePlanError):
            self._pair("u1", "u1")
        with self.assertRaises(core.CorePlanError):
            self._pair("u1", " u2 ")
        with self.assertRaises(core.CorePlanError):
            self._pair("", "u2")


class SubstrateTest(unittest.TestCase):
    def test_unknown_family_set(self):
        for token in (
            "",
            "  ",
            "NA",
            "na",
            "n/a",
            "N/A",
            "none",
            "UNKNOWN",
            "not_applicable",
            "Unspecified",
        ):
            self.assertIsNone(core.known_substrate_family(token))
        self.assertIsNone(core.known_substrate_family(None))
        self.assertIsNone(core.known_substrate_family(7))
        self.assertEqual(core.known_substrate_family(" Glass "), "glass")

    def test_different_known_substrate_requires_both_known(self):
        self.assertTrue(
            core.different_known_substrate_family({"substrate": "glass"}, {"substrate": "quartz"})
        )
        self.assertFalse(
            core.different_known_substrate_family({"substrate": "NA"}, {"substrate": "quartz"})
        )
        self.assertFalse(
            core.different_known_substrate_family({"substrate": "glass"}, {"substrate": "Glass"})
        )
        self.assertFalse(
            core.different_known_substrate_family({"substrate": "unknown"}, {"substrate": "quartz"})
        )


class ClassifyPairTest(unittest.TestCase):
    def setUp(self):
        self.base = {"target": "A", "master": "m1", "instrument": "i1"}

    def test_categories(self):
        self.assertIsNone(core.classify_positive_pair(self.base, dict(self.base)))
        self.assertEqual(
            core.classify_positive_pair(
                self.base, {"target": "A", "master": "m1", "instrument": "i2"}
            ),
            "same_master_different_instrument",
        )
        self.assertEqual(
            core.classify_positive_pair(
                self.base, {"target": "A", "master": "m2", "instrument": "i1"}
            ),
            "different_master_same_instrument",
        )
        self.assertEqual(
            core.classify_positive_pair(
                self.base, {"target": "A", "master": "m2", "instrument": "i2"}
            ),
            "different_master_different_instrument",
        )
        self.assertIsNone(
            core.classify_positive_pair(
                self.base, {"target": "B", "master": "m2", "instrument": "i2"}
            )
        )


class AssignGuardFoldsTest(unittest.TestCase):
    def test_deterministic_round_robin(self):
        masters = [f"m{index}" for index in range(6)]
        first = core.assign_guard_folds(context_id="ctx", masters=masters)
        shuffled = list(masters)
        random.Random(3).shuffle(shuffled)
        second = core.assign_guard_folds(context_id="ctx", masters=shuffled)
        self.assertEqual(first, second)
        self.assertTrue(all(fold in (0, 1, 2) for fold in first.values()))
        self.assertEqual(sorted({fold for fold in first.values()}), [0, 1, 2])


class GuardFoldTest(unittest.TestCase):
    def _rows(self) -> list:
        rows = []
        for clazz, prefix in zip(("A", "B", "C"), ("a", "b", "c"), strict=False):
            for index in range(3):
                master = f"{prefix}{index}"
                rows.append(
                    {"uid": master, "master": master, "target": clazz, "instrument": f"i{index}"}
                )
        return rows

    def _build(self, rows: list, *, outer_test_masters=()) -> list:
        return core.build_guard_fold_roles(
            contract_sha256=core.LOCKED_CONTRACT_SHA256,
            context_id="ctx1",
            station="surfaces",
            outer_fit_rows=rows,
            held_instrument="HELD",
            outer_test_masters=outer_test_masters,
        )

    def test_class_stratified_disjoint_contained(self):
        rows = self._rows()
        units = self._build(rows)
        self.assertEqual(len(units), 3)
        all_masters = {row["master"] for row in rows}
        for unit in units:
            fitting = set(unit["fitting_role"]["masters"])
            validation = set(unit["validation_role"]["masters"])
            self.assertEqual(fitting & validation, set())
            self.assertEqual(fitting | validation, all_masters)
            self.assertTrue(unit["support_ok"])
            self.assertIsNone(unit["exclusion_reason"])
            self.assertEqual(unit["fitting_role"]["class_count"], 3)
            self.assertEqual(unit["validation_role"]["class_count"], 3)

    def test_observations_follow_master(self):
        rows = []
        for clazz, prefix in zip(("A", "B", "C"), ("a", "b", "c"), strict=False):
            for index in range(2):
                master = f"{prefix}{index}"
                rows.append(
                    {"uid": f"{master}-0", "master": master, "target": clazz, "instrument": "i0"}
                )
                rows.append(
                    {"uid": f"{master}-1", "master": master, "target": clazz, "instrument": "i1"}
                )
        units = self._build(rows)
        placement: dict[str, set] = {}
        for unit in units:
            for role_name in ("fitting_role", "validation_role"):
                for uid in unit[role_name]["fitting_uids"]:
                    placement.setdefault(uid, set()).add((unit["guard_fold"], role_name))
        for uid in list(placement):
            master = uid.split("-")[0]
            self.assertEqual(placement[uid], placement[f"{master}-0"])
            self.assertEqual(placement[uid], placement[f"{master}-1"])

    def test_unavailable_class_support_is_reason_coded(self):
        rows = [
            {"uid": "a0", "master": "a0", "target": "A", "instrument": "i0"},
            {"uid": "b0", "master": "b0", "target": "B", "instrument": "i0"},
            {"uid": "c0", "master": "c0", "target": "C", "instrument": "i0"},
            {"uid": "c1", "master": "c1", "target": "C", "instrument": "i1"},
        ]
        units = self._build(rows)
        self.assertEqual(len(units), 3)
        self.assertFalse(all(unit["support_ok"] for unit in units))
        for unit in units:
            if unit["support_ok"]:
                self.assertIsNone(unit["exclusion_reason"])
            else:
                self.assertIsNotNone(unit["exclusion_reason"])

    def test_held_instrument_rejected(self):
        rows = [{"uid": "x", "master": "m", "target": "A", "instrument": "HELD"}]
        with self.assertRaises(core.CorePlanError):
            self._build(rows)

    def test_outer_test_master_rejected(self):
        rows = [{"uid": "x", "master": "m", "target": "A", "instrument": "i0"}]
        with self.assertRaises(core.CorePlanError):
            self._build(rows, outer_test_masters=["m"])


@requires_contract
class EnumeratePairsTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()

    def _roles(self, rows: list) -> dict:
        return {
            label: {"p05_role_id": "role:" + label, "rows": rows}
            for label in self.contract["smoke"]["role_labels"]
        }

    def _pairs(self, rows: list) -> list:
        return core.enumerate_smoke_pairs(
            contract=self.contract,
            contract_sha256=core.LOCKED_CONTRACT_SHA256,
            input_pins=self.contract["input_pins"],
            selected_roles=self._roles(rows),
        )

    def test_set_invariant_under_row_shuffle(self):
        rows = [
            observation("u1", "m1", "A", "i1"),
            observation("u2", "m1", "A", "i2"),
            observation("u3", "m2", "A", "i1", substrate="quartz"),
            observation("u4", "m3", "B", "i1"),
        ]
        first = sorted(pair["pair_id"] for pair in self._pairs(rows))
        shuffled = list(rows)
        random.Random(11).shuffle(shuffled)
        second = sorted(pair["pair_id"] for pair in self._pairs(shuffled))
        self.assertEqual(first, second)

    def test_same_master_same_instrument_excluded(self):
        rows = [
            observation("u1", "m1", "A", "i1"),
            observation("u2", "m1", "A", "i1"),
        ]
        self.assertEqual(self._pairs(rows), [])

    def test_paired_consistency_eligible_flag(self):
        rows = [
            observation("u1", "m1", "A", "i1"),
            observation("u2", "m1", "A", "i2"),
            observation("u3", "m2", "A", "i1"),
        ]
        pairs = self._pairs(rows)
        for pair in pairs:
            if pair["pair_category"] == "same_master_different_instrument":
                self.assertTrue(pair["paired_consistency_eligible"])
            else:
                self.assertFalse(pair["paired_consistency_eligible"])
        categories = {pair["pair_category"] for pair in pairs}
        self.assertIn("same_master_different_instrument", categories)
        self.assertIn("different_master_same_instrument", categories)


@requires_contract
class DevelopmentSlotTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_contract()

    def test_recipe_seed_expansion_and_exclusion_flag(self):
        specs = [
            {
                "slot_kind": "inherited_selection_fit",
                "context_id": "c1",
                "selection_unit_id": "u1",
                "fitting_role_id": "f1",
                "validation_role_id": "v1",
                "guard_fold": None,
                "excluded_by_protocol": False,
                "exclusion_reason": None,
            },
            {
                "slot_kind": "guard_selection_fit",
                "context_id": "c1",
                "selection_unit_id": "gu1",
                "fitting_role_id": "gf1",
                "validation_role_id": "gv1",
                "guard_fold": 2,
                "excluded_by_protocol": True,
                "exclusion_reason": "guard_fold_lacks_three_classes_in_one_or_both_roles",
            },
        ]
        slots = core.expand_development_slots(
            contract=self.contract,
            contract_sha256=core.LOCKED_CONTRACT_SHA256,
            specs=specs,
        )
        self.assertEqual(len(slots), 24)
        self.assertEqual(sum(1 for slot in slots if slot["excluded_by_protocol"]), 12)
        self.assertEqual(sum(1 for slot in slots if not slot["excluded_by_protocol"]), 12)
        self.assertTrue(all(slot["planned"] for slot in slots))
        self.assertTrue(all(slot["guard_fold"] == 2 for slot in slots[12:]))


@requires_contract
class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.plan = load_contract()["later_core_plan"]

    def _counts(self, **overrides) -> dict:
        base = {
            "outer_contexts": 320,
            "inherited_inner_units": 861,
            "pseudo_domain_contexts": 128,
            "master_cv_fallback_t3_contexts": 132,
            "guard_unit_slots": 384,
            "inner_fit_slots": 14940,
            "inner_fit_slots_eligible": 14400,
            "inner_fit_slots_excluded": 540,
        }
        base.update(overrides)
        return base

    def test_reconciles_locked_ledger(self):
        budget = core.reconcile_budget(counts=self._counts(), later_plan=self.plan)
        self.assertEqual(budget["inner_fit_slots"], 14940)
        self.assertEqual(budget["later_neural_fits"], 17820)
        self.assertEqual(budget["including_smoke_neural_executions"], 17854)
        self.assertEqual(budget["retry_slots"], 0)
        self.assertEqual(budget["d4_slots"], 0)
        self.assertEqual(budget["d5_slots"], 0)
        self.assertEqual(budget["authorization"], "ceiling_only_not_execution_permission")

    def test_rejects_mismatched_inner_total(self):
        with self.assertRaises(core.CorePlanError):
            core.reconcile_budget(counts=self._counts(inner_fit_slots=14000), later_plan=self.plan)

    def test_rejects_eligible_excluded_sum_mismatch(self):
        with self.assertRaises(core.CorePlanError):
            core.reconcile_budget(
                counts=self._counts(inner_fit_slots_eligible=1, inner_fit_slots_excluded=1),
                later_plan=self.plan,
            )

    def test_rejects_expanded_guard_slots(self):
        with self.assertRaises(core.CorePlanError):
            core.reconcile_budget(counts=self._counts(guard_unit_slots=999), later_plan=self.plan)


class NoFittingImportTest(unittest.TestCase):
    def test_module_source_has_no_numerical_imports(self):
        source = Path(core.__file__).read_text(encoding="utf-8")
        for token in ("import numpy", "import torch", "from numpy", "from torch"):
            self.assertNotIn(token, source)
        self.assertFalse(hasattr(core, "numpy"))
        self.assertFalse(hasattr(core, "torch"))


if __name__ == "__main__":
    unittest.main()
