"""P08-T253 metadata-only declaration tests for the finite stress resource proposal.

Reads the frozen public resource proposal at plan/contracts/p08_stress_resources.json,
the historical timing basis and the public input/prediction/score catalog audits.
No fits, predictions, scores, weights, resampling, routes, quantiles or scientific
operations are executed or produced.  The proposal declares finite ceilings and
public payload arithmetic only; it grants no authority and reserves no capacity.
No executor or runner exists or is claimed.
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "p08_readiness"
CONTRACT = ROOT / "plan" / "contracts" / "p08_stress_resources.json"
TIMING = RESULTS / "historical_timing_basis.json"
INPUT_AUDIT = RESULTS / "stress_input_catalog_audit.json"
PREDICTION_AUDIT = RESULTS / "stress_prediction_catalog_audit.json"
SCORE_AUDIT = RESULTS / "stress_score_catalog_audit.json"

SCHEMA = "nato-sers-p08-stress-resource-proposal-v1"
GIB = 1024**3
DRAWS = 10000


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StressResourceDeclarationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal = _load(CONTRACT)
        cls.timing = _load(TIMING)
        cls.input_audit = _load(INPUT_AUDIT)
        cls.prediction_audit = _load(PREDICTION_AUDIT)
        cls.score_audit = _load(SCORE_AUDIT)
        cls.payload = cls.proposal["payload_arithmetic_not_total_storage_forecast"]

    def test_identity_scope_and_no_authority(self):
        p = self.proposal
        self.assertEqual(p["schema_version"], SCHEMA)
        self.assertEqual(
            p["protocol_version"],
            "nato-sers-p08-stress-resource-proposal-20261007-v1",
        )
        self.assertEqual(p["stage_id"], "S1")
        self.assertEqual(p["required_upstream_stages"], ["U1", "Q1"])
        self.assertEqual(p["status"], "finite_proposal_not_execution_permission")
        self.assertEqual(
            p["scope"],
            "registered_primary_population_universal_and_fixed_clean_route_QC_stress_test",
        )
        self.assertIs(p["other_population_range_or_normalization_crosses"], False)
        for flag in (
            "execution_authorized",
            "resource_proposal_approved",
            "numerical_runtime_accepted",
            "full_inference_ledger_accepted",
        ):
            self.assertIs(p[flag], False)
        self.assertIs(type(p["authorized_scientific_operations"]), int)
        self.assertEqual(p["authorized_scientific_operations"], 0)

    def test_fit_ceilings_and_reconstruction_scope(self):
        p = self.proposal
        self.assertEqual(p["model_fit_ceiling"], 1820)
        self.assertEqual(
            p["model_fits_by_family"],
            {"C-RBF-SVM": 260, "C-RANDOM-FOREST": 780, "C-EXTRA-TREES": 780},
        )
        self.assertEqual(sum(p["model_fits_by_family"].values()), 1820)
        self.assertIs(p["fits_are_exact_historical_MIN_reconstruction_only"], True)
        for key in (
            "neural_optimization_fits",
            "hyperparameter_search_fits",
            "scalar_calibration_fits",
            "automatic_retries",
        ):
            self.assertIs(type(p[key]), int)
            self.assertEqual(p[key], 0)

    def test_ceilings_retention_and_capacity_are_not_reserved(self):
        p = self.proposal
        self.assertEqual(p["active_wall_seconds"], 72 * 3600)
        self.assertEqual(p["active_wall_seconds"], 259200)
        self.assertEqual(p["artifact_ceiling_bytes"], 64 * GIB)
        self.assertEqual(p["process_tree_ram_ceiling_bytes"], 24 * GIB)
        self.assertEqual(p["allocated_gpu_ceiling_bytes"], 8 * GIB)
        self.assertEqual(p["maximum_CPU_workers"], 4)
        self.assertEqual(p["maximum_GPU_workers"], 1)
        self.assertEqual(p["threads_per_worker"], 1)
        self.assertEqual(p["reserve_free_bytes"], 30 * GIB)
        self.assertEqual(p["initial_free_bytes_required"], 94 * GIB)
        self.assertIs(p["retained_upstream_model_or_temperature_replacement_allowed"], False)
        retention = p["retention"]
        for key in (
            "preserve_existing_and_failed_evidence",
            "reconstructed_MIN_estimators",
            "upstream_models_referenced_not_copied_or_refitted",
            "input_transform_realization_and_parity_receipts",
            "transformed_rows_and_seed_predictions",
            "calibrated_prediction_units_and_case_scores",
            "terminal_contrast_draws_and_missingness_reasons",
            "shared_weights_referenced_by_identity_and_hash",
            "diagnostic_summaries_and_figure_provenance",
            "immutable_shards_require_individual_record_identity_and_hash",
            "control_files_logs_failed_attempts_count_toward_artifacts",
        ):
            self.assertIs(retention[key], True)
        self.assertIs(retention["ceiling_breach_authorizes_deletion_or_budget_extension"], False)
        snapshot = p["capacity_snapshot"]
        self.assertEqual(snapshot["observed_at_UTC"], "2026-10-07 18:09:10 UTC")
        self.assertIs(snapshot["resources_reserved"], False)
        self.assertIs(snapshot["cleanup_performed"], False)
        for key in (
            "filesystem_available_bytes",
            "host_MemAvailable_kB",
            "GPU_total_MiB",
            "GPU_free_MiB",
        ):
            self.assertIn(key, snapshot)

    def test_cost_basis_pins_stage_and_historical_sums(self):
        basis = self.proposal["historical_cost_basis"]
        self.assertEqual(basis["file"], "results/p08_readiness/historical_timing_basis.json")
        self.assertEqual(_sha256(TIMING), basis["sha256"])
        for filename, declared in self.proposal["input_audits"].items():
            self.assertEqual(_sha256(RESULTS / filename), declared)
        self.assertEqual(basis["stage"], "final_family_refit")
        self.assertEqual(basis["recorded_fits"], 1820)
        records = [r for r in self.timing["classical"] if r["stage"] == basis["stage"]]
        self.assertEqual(sum(r["slots"] for r in records), basis["recorded_fits"])
        self.assertAlmostEqual(
            sum(r["seconds"] for r in records), basis["summed_fit_seconds"], places=9
        )
        self.assertEqual(
            sum(r["serialized_model_size_bytes_sum"] for r in records),
            basis["summed_in_memory_serialized_estimator_bytes"],
        )
        self.assertAlmostEqual(basis["summed_fit_seconds"], 1786.6997406235314, places=9)
        self.assertEqual(basis["summed_in_memory_serialized_estimator_bytes"], 6142752908)
        self.assertIs(basis["historical_estimator_files_retained"], False)
        self.assertIs(
            basis["includes_stress_transforms_predictions_inference_or_orchestration"],
            False,
        )
        self.assertIs(basis["measured_new_stress_runtime"], False)
        self.assertIs(basis["wall_ceiling_is_completion_forecast"], False)
        self.assertIs(self.timing["measured_p08_timing"], False)

    def test_payload_row_weight_and_contrast_arithmetic(self):
        payload = self.payload
        grid = payload["primary_grid_values_per_row"]
        rows = payload["raw_case_rows"]
        transformed = payload["transformed_rows"]
        self.assertEqual(grid, 1401)
        self.assertEqual(rows, self.input_audit["summary"]["stage_counts"]["raw_case"])
        self.assertEqual(
            transformed,
            self.input_audit["summary"]["stage_counts"]["action_transform"],
        )
        self.assertEqual(payload["all_raw_cases_float64_bytes"], rows * grid * 8)
        self.assertEqual(payload["all_transformed_rows_float32_bytes"], transformed * grid * 4)
        self.assertEqual(
            payload["all_transformed_rows_float64_workspace_bytes"],
            transformed * grid * 8,
        )
        self.assertEqual(payload["draw_batch_ceiling"], 128)
        self.assertEqual(
            payload["batches_per_10000_draws"],
            math.ceil(DRAWS / payload["draw_batch_ceiling"]),
        )
        self.assertEqual(payload["batches_per_10000_draws"], 79)
        operational = self.score_audit["supports"]["operational_260"]
        self.assertEqual(operational["distinct_test_masters"], 69)
        self.assertEqual(operational["instruments"], 10)
        self.assertEqual(
            payload["existing_shared_weight_float64_bytes"],
            DRAWS * (operational["distinct_test_masters"] + operational["instruments"]) * 8,
        )
        contrasts = payload["original_hierarchy_defined_mask_boolean_bytes"] // DRAWS
        self.assertEqual(contrasts, 456)
        self.assertEqual(payload["original_hierarchy_scalar_float64_bytes"], contrasts * DRAWS * 8)
        one_view = payload["one_contrast_view_three_factor_terminal_draw_float64_bytes"]
        self.assertEqual(one_view, contrasts * 3 * DRAWS * 8)
        self.assertEqual(
            payload["four_contrast_views_three_factor_terminal_draw_float64_bytes"],
            4 * one_view,
        )
        self.assertIs(
            payload["conditional_view_allowance_is_not_additional_inference_authority"],
            True,
        )
        self.assertIs(payload["persist_draw_by_context_by_case_tensor"], False)

    def test_declared_counts_match_public_audits(self):
        self.assertEqual(
            self.prediction_audit["summary"]["historical_reconstruction_fit_count"],
            self.proposal["model_fit_ceiling"],
        )
        self.assertEqual(
            self.prediction_audit["input_graph_job_count"],
            self.input_audit["summary"]["total_job_count"],
        )
        self.assertEqual(self.input_audit["summary"]["context_count"], 260)
        self.assertEqual(self.score_audit["summary"]["context_count"], 260)
        for audit in (self.input_audit, self.prediction_audit, self.score_audit):
            self.assertEqual(audit["summary"]["case_count"], 96)
            self.assertIs(audit["execution_authorized"], False)
            operations = audit["new_scientific_operations"]
            self.assertIs(type(operations), int)
            self.assertEqual(operations, 0)


if __name__ == "__main__":
    unittest.main()
