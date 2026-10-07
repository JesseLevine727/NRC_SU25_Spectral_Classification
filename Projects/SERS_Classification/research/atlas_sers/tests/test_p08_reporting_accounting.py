"""Public-metadata tests for the T263 P08 reporting accounting declaration.

These tests read only the public contract, the frozen P08 figure-plan CSV and
the pinned public audit/protocol files. They perform no fitting, prediction,
resampling, statistics, preprocessing or rendering, and load no private data.
"""

import csv
import hashlib
import json
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PACKAGE_ROOT / "plan" / "contracts" / "p08_reporting_accounting.json"
FIGURE_PLAN_PATH = PACKAGE_ROOT / "plan" / "P08_FIGURE_PLAN.csv"

EXPECTED_SCHEMA = "nato-sers-p08-reporting-accounting-v1"
EXPECTED_PROTOCOL_VERSION = "nato-sers-p08-reporting-accounting-20261007-v1"
EXPECTED_FIGURE_IDS = [f"P08-F{i:02d}" for i in range(1, 12)]
EXPECTED_FORMATS = {"native_tikz", "offline_html", "vector_pdf", "png"}
ALLOWED_ALLOCATIONS = {"unconditional", "conditional", "alias_only"}
ALLOWED_BLOCK_STAGES = {"U1", "Q1", "S1", "R1", "N2", "population", "per_figure_ownership"}
EXPECTED_BLOCK_IDS = [
    "preservation_authentication",
    "preservation_rows",
    "preservation_domain_action_summary",
    "public_spectral_cell_prepare",
    "private_example_prepare",
    "fixed_route_qc_case_summary",
    "source_noise_display_summary",
    "public_figure_delivery",
    "private_example_delivery",
]

EXPECTED_STAGE_ORDER = [
    "semantic_data",
    "disclosure_review",
    "native_tikz",
    "offline_html",
    "vector_pdf",
    "png_review",
    "visual_semantic_review",
    "release_manifest",
]
EXPECTED_STAGE_EDGES = {
    "semantic_data": [],
    "disclosure_review": ["semantic_data"],
    "native_tikz": ["disclosure_review"],
    "offline_html": ["disclosure_review"],
    "vector_pdf": ["native_tikz"],
    "png_review": ["vector_pdf"],
    "visual_semantic_review": ["offline_html", "vector_pdf", "png_review"],
    "release_manifest": ["visual_semantic_review"],
}

EXPECTED_EVIDENCE = {
    "preservation_reporting_support_audit": (
        "results/p08_readiness/preservation_reporting_support_audit.json",
        "afa64083f5cb78a1c6d0f256b40cda8565ce72ed8c86febec59cb4862bc919e0",
    ),
    "stress_score_catalog_audit": (
        "results/p08_readiness/stress_score_catalog_audit.json",
        "1be38c92b63b9de5c33c0a6a49dd794a2a23c95dcb42277aed590f3880abee39",
    ),
    "P08_STATISTICAL_PROTOCOL": (
        "plan/P08_STATISTICAL_PROTOCOL.md",
        "edf4b6e4ab303b2b5910b0285cd07e4ffdbaa782373f6d2e074faea07d8b2907",
    ),
    "P08_FIGURE_PLAN": (
        "plan/P08_FIGURE_PLAN.csv",
        "d921b877fceba9ba286ee8da1e6b7f8d4f78395900e7f0beb1396afde7dcac0e",
    ),
    "P08_FIGURE_PROTOCOL": (
        "plan/P08_FIGURE_PROTOCOL.md",
        "73eebbbb13dbaccdceec1c28d4124573ca59f63537c0681872aacd9150ec8c45",
    ),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestP08ReportingAccounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
            cls.contract = json.load(handle)
        with FIGURE_PLAN_PATH.open("r", encoding="utf-8", newline="") as handle:
            cls.figure_rows = list(csv.DictReader(handle))
        cls.blocks = {block["id"]: block for block in cls.contract["blocks"]}

    def test_schema_and_authority_flags(self):
        contract = self.contract
        self.assertEqual(contract["schema"], EXPECTED_SCHEMA)
        self.assertEqual(contract["status"], "prospective_metadata_inventory")
        authority = contract["authority"]
        self.assertIs(authority["execution_authorized"], False)
        self.assertIs(authority["numerical_reporting_accepted"], False)
        self.assertIs(authority["full_readiness_complete"], False)
        self.assertIs(authority["new_figures_rendered"], False)
        self.assertNotIsInstance(authority["authorized_scientific_operations"], bool)
        self.assertIsInstance(authority["authorized_scientific_operations"], int)
        self.assertEqual(authority["authorized_scientific_operations"], 0)

    def test_exact_protocol_version(self):
        self.assertEqual(self.contract["protocol_version"], EXPECTED_PROTOCOL_VERSION)
        self.assertEqual(
            self.contract["protocol_version"],
            "nato-sers-p08-reporting-accounting-20261007-v1",
        )

    def test_declaration_boundary(self):
        boundary = self.contract["declaration_boundary"]
        self.assertFalse(boundary["is_numerical_executor"])
        self.assertFalse(boundary["is_generic_planner"])
        self.assertTrue(boundary["is_finite_declarative_inventory"])
        self.assertTrue(boundary["allocated_future_operations_not_results"])
        self.assertFalse(boundary["qc_gate_reaction_evaluated"])
        self.assertFalse(boundary["completes_readiness_goal"])
        self.assertFalse(boundary["accepts_numerical_implementation"])
        self.assertTrue(boundary["later_branch_inference_launch_acceptance_separate"])

    def test_block_inventory(self):
        blocks = self.contract["blocks"]
        self.assertEqual(len(blocks), 9)
        ids = [block["id"] for block in blocks]
        self.assertEqual(ids, EXPECTED_BLOCK_IDS)
        self.assertEqual(len(set(ids)), 9)
        self.assertEqual([block["index"] for block in blocks], list(range(1, 10)))

        for block in blocks:
            with self.subTest(block=block["id"]):
                self.assertIn(block["allocation"], ALLOWED_ALLOCATIONS)
                self.assertNotIsInstance(block["new_operations"], bool)
                self.assertIsInstance(block["new_operations"], int)
                self.assertGreaterEqual(block["new_operations"], 0)
                self.assertNotIsInstance(block["aliases"], bool)
                self.assertIsInstance(block["aliases"], int)
                self.assertGreaterEqual(block["aliases"], 0)
                self.assertIn(block["stage"], ALLOWED_BLOCK_STAGES)

        conditional = [block["id"] for block in blocks if block["allocation"] == "conditional"]
        self.assertEqual(conditional, ["private_example_prepare", "private_example_delivery"])
        alias_only = [block["id"] for block in blocks if block["allocation"] == "alias_only"]
        self.assertEqual(alias_only, ["preservation_rows"])

        self.assertEqual(sum(block["new_operations"] for block in blocks), 25603)
        self.assertEqual(
            sum(
                block["new_operations"] for block in blocks if block["allocation"] == "conditional"
            ),
            441,
        )
        self.assertEqual(sum(block["aliases"] for block in blocks), 1794)
        self.assertEqual(1 + 0 + 51 + 49 + 49 + 24960 + 13 + 88 + 392, 25603)
        self.assertEqual(49 + 392, 441)

        owner_stages = set(self.contract["stage_owner_map"].values())
        for block in blocks:
            if block["stage"] != "per_figure_ownership":
                self.assertIn(block["stage"], owner_stages, block["id"])
        self.assertEqual(self.blocks["public_figure_delivery"]["stage"], "per_figure_ownership")

    def test_totals_are_independently_derived(self):
        blocks = self.contract["blocks"]
        for block in blocks:
            self.assertIn(block["allocation"], ALLOWED_ALLOCATIONS)
            self.assertNotIsInstance(block["new_operations"], bool)
            self.assertIsInstance(block["new_operations"], int)
            self.assertIsInstance(block["aliases"], int)

        nonalias = sum(block["new_operations"] for block in blocks)
        conditional = sum(
            block["new_operations"] for block in blocks if block["allocation"] == "conditional"
        )
        aliases = sum(block["aliases"] for block in blocks)

        self.assertEqual(nonalias, 25603)
        self.assertEqual(conditional, 441)
        self.assertEqual(nonalias - conditional, 25162)
        self.assertEqual(aliases, 1794)

        # Independent reconstruction of the declared component arithmetic.
        self.assertEqual(1 + 0 + 51 + 49 + 49 + 24960 + 13 + 88 + 392, 25603)
        self.assertEqual(49 + 392, 441)

        totals = self.contract["totals"]
        self.assertEqual(totals["nonalias_planned_slots"], nonalias)
        self.assertEqual(totals["conditional_slots"], conditional)
        self.assertEqual(totals["unconditional_slots"], nonalias - conditional)
        self.assertEqual(totals["aliases"], aliases)
        self.assertTrue(totals["aliases_excluded_from_nonalias"])
        self.assertFalse(totals["missing_prerequisites_reduce_slots"])
        self.assertFalse(totals["automatic_retries"])

    def test_preservation_rows_are_alias_only(self):
        block = self.blocks["preservation_rows"]
        self.assertEqual(block["new_operations"], 0)
        self.assertEqual(block["aliases"], 598 * 3)
        self.assertEqual(block["composition"]["observations"], 598)
        self.assertEqual(block["composition"]["actions"], 3)
        self.assertTrue(block["aliases_excluded_from_totals"])
        self.assertEqual(
            block["private_catalog_digest"],
            "4e77dcad7978f5f6fe48260f4336711ae17213917ffbfd1ba93dddf905e9d24f",
        )
        self.assertEqual(len(block["private_catalog_digest"]), 64)

    def test_preservation_domain_action_summary(self):
        block = self.blocks["preservation_domain_action_summary"]
        self.assertEqual(block["new_operations"], 17 * 3)
        self.assertEqual(block["new_operations"], 51)
        self.assertEqual(block["composition"]["domains"], 17)
        self.assertEqual(block["composition"]["actions"], 3)
        self.assertEqual(block["stored_numeric_column_count"], 11)
        self.assertEqual(block["quantile_method"], "linear")
        for name in ["finite_count", "undefined_count", "median", "q10", "q90"]:
            self.assertIn(name, block["summary_statistics"])
        constraints = block["constraints"]
        self.assertTrue(constraints["no_new_spectral_diagnostic_computation"])
        self.assertTrue(constraints["no_clean_chemical_reference"])
        self.assertTrue(constraints["no_binary_preservation_threshold"])
        self.assertTrue(constraints["row_level_values_used_as_recorded"])
        self.assertTrue(constraints["master_equal_applies_to_display_curves_only"])

    def test_public_spectral_cell_prepare(self):
        block = self.blocks["public_spectral_cell_prepare"]
        composition = block["composition"]
        self.assertEqual(composition["planned_cells"], 49)
        self.assertEqual(composition["eligible_cells"], 46)
        self.assertEqual(composition["unavailable_cells"], 3)
        self.assertEqual(
            composition["eligible_cells"] + composition["unavailable_cells"],
            composition["planned_cells"],
        )
        self.assertEqual(composition["curves_max"], composition["eligible_cells"] * 3)
        self.assertEqual(46 * 3, 138)
        self.assertTrue(block["privacy"]["public"])
        self.assertEqual(block["privacy"]["min_masters_per_public_cell"], 2)
        self.assertFalse(block["aggregation"]["aggregate_renormalization"])
        self.assertFalse(block["aggregation"]["is_averaged_model_input"])

    def test_private_preparation_not_blocked_by_public_availability(self):
        prepare = self.blocks["private_example_prepare"]
        public_prepare = self.blocks["public_spectral_cell_prepare"]
        self.assertNotIn("public_spectral_cell_prepare", prepare["depends_on"])
        for dependency in [
            "preservation_authentication",
            "frozen_primary_action_arrays_and_validity_receipts",
            "fixed_private_cell_membership",
            "explicit later private-reporting approval",
        ]:
            self.assertIn(dependency, prepare["depends_on"])
        for dependency in [
            "preservation_authentication",
            "frozen_primary_action_arrays_and_validity_receipts",
            "fixed_private_cell_membership",
        ]:
            self.assertIn(dependency, public_prepare["depends_on"])

    def test_private_example_blocks(self):
        prepare = self.blocks["private_example_prepare"]
        self.assertEqual(prepare["new_operations"], 49)
        self.assertEqual(prepare["allocation"], "conditional")
        self.assertFalse(prepare["outcome_selected"])
        self.assertEqual(prepare["visibility"], "private")
        self.assertFalse(prepare["ever_public"])

        delivery = self.blocks["private_example_delivery"]
        self.assertEqual(delivery["new_operations"], 49 * 8)
        self.assertEqual(delivery["composition"]["cells"], 49)
        self.assertEqual(delivery["composition"]["delivery_stages"], 8)
        self.assertEqual(delivery["allocation"], "conditional")
        self.assertEqual(delivery["new_public_figure_ids"], 0)
        self.assertTrue(delivery["private_f01_subviews"])

    def test_fixed_route_qc_case_summary(self):
        block = self.blocks["fixed_route_qc_case_summary"]
        self.assertEqual(block["new_operations"], 260 * 96)
        self.assertEqual(block["new_operations"], 24960)
        self.assertEqual(block["composition"]["contexts"], 260)
        self.assertEqual(block["composition"]["cases"], 96)
        self.assertEqual(block["eligible_contexts"], 54)
        self.assertEqual(block["unsupported_contexts"], 206)
        self.assertEqual(block["eligible_contexts"] + block["unsupported_contexts"], 260)
        self.assertEqual(
            set(block["qc_methods"]),
            {"RBF-SVM", "Random Forest", "D0-M", "P05-SELECTED"},
        )
        self.assertFalse(block["extra_trees_in_qc_panel"])
        self.assertTrue(block["one_gate_per_context"])
        self.assertFalse(block["method_multiplied"])
        self.assertEqual(block["route_summary_unit"], "context x case")
        self.assertTrue(block["no_regating"])
        self.assertFalse(block["qc_gate_reaction_evaluated"])
        fallback = block["fallback"]
        self.assertEqual(fallback["invalid_selected_action"], "MIN-input under same QC estimator")
        self.assertEqual(fallback["invalid_unselected_action"], "not fatal")
        self.assertEqual(fallback["invalid_MIN"], "fatal")
        self.assertEqual(fallback["structurally_unsupported_206"], "complete-MIN estimator")

    def test_source_noise_registered_quantiles(self):
        block = self.blocks["source_noise_display_summary"]
        self.assertEqual(block["new_operations"], 13)
        self.assertEqual(block["quantile_labels"], [0.5, 0.75, 0.9, 0.95])
        for obsolete in (
            "protocol_quantile_labels",
            "quantile_labels_match_protocol",
            "mismatch_reported",
            "mismatch_action",
        ):
            self.assertNotIn(obsolete, block)
        self.assertFalse(block["held_data_quantile_fit"])

    def test_figure_ids_match_frozen_csv(self):
        csv_ids = [row["figure_id"] for row in self.figure_rows]
        self.assertEqual(csv_ids, EXPECTED_FIGURE_IDS)
        self.assertEqual(len(csv_ids), 11)
        block = self.blocks["public_figure_delivery"]
        self.assertEqual(block["figure_ids"], EXPECTED_FIGURE_IDS)
        self.assertEqual(block["composition"]["figures"], 11)
        self.assertEqual(block["composition"]["delivery_stages"], 8)
        self.assertEqual(block["new_operations"], 11 * 8)
        self.assertEqual(block["stage"], "per_figure_ownership")

    def test_formats_and_stage_ownership_match_csv(self):
        for row in self.figure_rows:
            formats = [fmt for fmt in row["formats"].split(";") if fmt]
            self.assertEqual(set(formats), EXPECTED_FORMATS, row["figure_id"])

        owner_map = self.contract["stage_owner_map"]
        ownership = self.blocks["public_figure_delivery"]["ownership"]
        owned = {}
        for owner, figures in ownership.items():
            for figure in figures:
                self.assertNotIn(figure, owned)
                owned[figure] = owner
        self.assertEqual(sorted(owned), sorted(EXPECTED_FIGURE_IDS))

        for row in self.figure_rows:
            stage = row["stage"]
            self.assertIn(stage, owner_map, row["figure_id"])
            self.assertEqual(owned[row["figure_id"]], owner_map[stage], row["figure_id"])

    def test_delivery_stage_dag(self):
        stages = self.contract["delivery_stages"]
        ids = [stage["id"] for stage in stages]
        self.assertEqual(ids, EXPECTED_STAGE_ORDER)
        self.assertEqual(len(ids), 8)
        by_id = {stage["id"]: stage for stage in stages}
        for stage in stages:
            for dependency in stage["depends_on"]:
                self.assertIn(dependency, by_id, stage["id"])
                self.assertLess(
                    ids.index(dependency),
                    ids.index(stage["id"]),
                    stage["id"],
                )
        for stage_id, dependencies in EXPECTED_STAGE_EDGES.items():
            self.assertEqual(
                sorted(by_id[stage_id]["depends_on"]),
                sorted(dependencies),
                stage_id,
            )
        self.assertEqual(by_id["semantic_data"]["depends_on"], [])
        self.assertIs(by_id["semantic_data"]["local_predecessor"], False)

    def test_format_constraints(self):
        constraints = self.contract["format_constraints"]
        self.assertEqual(set(constraints["formats"]), EXPECTED_FORMATS)
        self.assertIs(constraints["raster_embedded_in_tikz"], False)
        self.assertIs(constraints["external_scripts_or_fonts_in_html"], False)
        self.assertTrue(constraints["same_semantic_data"])
        self.assertEqual(
            set(constraints["final_publication_requires"]),
            {"disclosure_review", "visual_semantic_review"},
        )

    def test_probability_diagnostic_reuse_has_no_double_count(self):
        reuse = self.contract["probability_diagnostic_reuse"]
        jobs = reuse["existing_case_jobs"]
        self.assertEqual(jobs["context_case_scoring"], 655296)
        self.assertEqual(jobs["pooled_case_scoring"], 174912)
        self.assertEqual(jobs["context_case_scoring"] + jobs["pooled_case_scoring"], 830208)
        self.assertEqual(jobs["total"], 830208)
        self.assertEqual(reuse["new_jobs"], 0)
        self.assertIs(reuse["added_to_new_operations"], False)
        self.assertEqual(self.contract["totals"]["nonalias_planned_slots"], 25603)
        self.assertNotEqual(self.contract["totals"]["nonalias_planned_slots"], jobs["total"])
        outputs = set(reuse["outputs"])
        for name in [
            "balanced_accuracy",
            "supported_macro_f1",
            "nll",
            "brier",
            "confusion",
            "class_recall",
        ]:
            self.assertIn(name, outputs)
        self.assertEqual(
            len([name for name in outputs if "bins" in name]),
            1,
        )
        weakest = reuse["weakest_domain_diagnostics"]
        self.assertTrue(weakest["belongs_to_existing_inference_descriptors"])
        self.assertEqual(weakest["additional_reporting_jobs"], 0)
        self.assertEqual(weakest["extra_significance_families"], 0)

    def test_evidence_bindings_and_pins(self):
        bindings = self.contract["evidence_bindings"]
        for name, (rel_path, expected_sha) in EXPECTED_EVIDENCE.items():
            with self.subTest(binding=name):
                self.assertIn(name, bindings)
                self.assertEqual(bindings[name]["path"], rel_path)
                self.assertFalse(rel_path.startswith("/"))
                self.assertEqual(bindings[name]["sha256"], expected_sha)
                path = PACKAGE_ROOT / rel_path
                self.assertTrue(path.is_file(), "missing evidence file: " + rel_path)
                self.assertEqual(sha256_file(path), expected_sha)

    def test_preservation_audit_cross_check(self):
        rel_path = EXPECTED_EVIDENCE["preservation_reporting_support_audit"][0]
        with (PACKAGE_ROOT / rel_path).open("r", encoding="utf-8") as handle:
            audit = json.load(handle)

        rows = self.blocks["preservation_rows"]
        domain_action = self.blocks["preservation_domain_action_summary"]
        public_prepare = self.blocks["public_spectral_cell_prepare"]
        private_prepare = self.blocks["private_example_prepare"]
        figure_block = self.blocks["public_figure_delivery"]
        digest = self.contract["private_catalog_bindings"]["preservation_rows"]["digest"]

        expected_rows = rows["composition"]["observations"] * rows["composition"]["actions"]
        self.assertEqual(expected_rows, 598 * 3)
        self.assertEqual(audit["primary_action_diagnostic_rows"], expected_rows)
        self.assertEqual(audit["primary_action_diagnostic_rows"], 1794)
        self.assertEqual(len(audit["primary_actions"]), rows["composition"]["actions"])
        self.assertEqual(len(audit["primary_actions"]), 3)
        self.assertEqual(audit["primary_domain_action_groups"], domain_action["new_operations"])
        self.assertEqual(audit["primary_domain_action_groups"], 51)
        self.assertEqual(audit["domain_count"], domain_action["composition"]["domains"])
        self.assertEqual(audit["domain_count"], 17)
        self.assertEqual(
            len(audit["primary_action_diagnostic_columns"]),
            domain_action["stored_numeric_column_count"],
        )
        self.assertEqual(len(audit["primary_action_diagnostic_columns"]), 11)
        self.assertEqual(
            audit["domain_analyte_cells"], public_prepare["composition"]["planned_cells"]
        )
        self.assertEqual(audit["domain_analyte_cells"], 49)
        self.assertEqual(
            audit["public_spectral_eligible_cells"],
            public_prepare["composition"]["eligible_cells"],
        )
        self.assertEqual(audit["public_spectral_eligible_cells"], 46)
        self.assertEqual(
            audit["public_spectral_unavailable_cells"],
            public_prepare["composition"]["unavailable_cells"],
        )
        self.assertEqual(audit["public_spectral_unavailable_cells"], 3)
        self.assertEqual(
            audit["private_preselected_example_rows"], private_prepare["new_operations"]
        )
        self.assertEqual(audit["private_preselected_example_rows"], 49)
        self.assertEqual(audit["registered_figure_bundles"], figure_block["composition"]["figures"])
        self.assertEqual(audit["registered_figure_bundles"], 11)
        self.assertEqual(audit["private_catalog_sha256"], digest)

    def test_score_audit_cross_check(self):
        rel_path = EXPECTED_EVIDENCE["stress_score_catalog_audit"][0]
        with (PACKAGE_ROOT / rel_path).open("r", encoding="utf-8") as handle:
            audit = json.load(handle)

        summary = audit["summary"]
        stage_counts = summary["stage_counts"]
        qc = self.blocks["fixed_route_qc_case_summary"]
        jobs = self.contract["probability_diagnostic_reuse"]["existing_case_jobs"]

        self.assertEqual(summary["context_count"], qc["composition"]["contexts"])
        self.assertEqual(summary["context_count"], 260)
        self.assertEqual(summary["case_count"], qc["composition"]["cases"])
        self.assertEqual(summary["case_count"], 96)
        self.assertEqual(summary["eligible_qc_context_count"], qc["eligible_contexts"])
        self.assertEqual(summary["eligible_qc_context_count"], 54)
        self.assertEqual(stage_counts["context_case_score"], jobs["context_case_scoring"])
        self.assertEqual(stage_counts["context_case_score"], 655296)
        self.assertEqual(stage_counts["pooled_case_score"], jobs["pooled_case_scoring"])
        self.assertEqual(stage_counts["pooled_case_score"], 174912)
        self.assertEqual(
            stage_counts["context_case_score"] + stage_counts["pooled_case_score"],
            830208,
        )
        self.assertEqual(jobs["total"], 830208)

    def test_privacy_flags(self):
        privacy = self.contract["privacy"]
        self.assertIs(privacy["public_identities"], False)
        self.assertIs(privacy["public_row_identifiers"], False)
        self.assertIs(privacy["private_examples_public"], False)
        self.assertEqual(privacy["public_cells_min_masters"], 2)

    def test_resource_charging_has_no_new_allowance(self):
        charging = self.contract["resource_charging"]
        self.assertTrue(charging["charged_to_producing_stage_ceilings"])
        self.assertTrue(charging["includes_failures_and_logs"])
        self.assertEqual(charging["new_resource_allowance"], 0)
        self.assertIs(charging["permission_inheritance"], False)

    def test_no_absolute_paths(self):
        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    yield key
                    yield from walk(value)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    yield from walk(item)
            elif isinstance(node, str):
                yield node

        for value in walk(self.contract):
            self.assertFalse(value.startswith("/"), value)
            self.assertFalse(
                len(value) >= 3
                and value[0].isalpha()
                and value[1] == ":"
                and value[2] in ("\\", "/"),
                value,
            )


if __name__ == "__main__":
    unittest.main()
