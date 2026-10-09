import copy
import hashlib
import json
import unittest

from atlas_sers.visualization.p08_training_figure_data import (
    MONITOR_SCHEMA,
    prepare_training_diagnostics,
)


def _full(label):
    return "P08JOB-" + hashlib.sha256(label.encode()).hexdigest()


def _suffix(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _expected(label, policy, model, seed, stage):
    return {
        "job_id": _full(label),
        "policy_id": policy,
        "model_id": model,
        "seed": seed,
        "stage": stage,
    }


def _source_rows(n, value, supcon=True, paired=False):
    rows = []
    for e in range(1, n + 1):
        rows.append(
            {
                "epoch": e,
                "elapsed_seconds": e * 60.0,
                "chemical_ce": value(e) if callable(value) else value,
                "total_loss": 1.5,
                "supcon_enabled": supcon,
                "paired_enabled": paired,
                "supcon_loss": 0.1 * e if supcon else None,
                "paired_loss": 0.2 * e if paired else None,
                "train_nll": 0.5,
                "validation_nll": 0.6,
                "train_balanced_accuracy": 0.8,
                "validation_balanced_accuracy": 0.7,
                "best_epoch": 1,
                "nonimproving_epochs": 0,
                "total_optimizer_steps": e * 10,
            }
        )
    return rows


def _source_monitor(label, policy, model, seed, n, value, supcon=True, paired=False):
    return {
        "schema": MONITOR_SCHEMA,
        "job_id": _suffix(label),
        "model_id": model,
        "policy_id": policy,
        "seed": seed,
        "stage": "source_fit",
        "validation_available": True,
        "epoch_budget": 200,
        "status": "complete",
        "stop_reason": "patience",
        "epochs_completed": n,
        "elapsed_seconds": n * 60.0,
        "rows": _source_rows(n, value, supcon, paired),
    }


def _refit_monitor(label, policy, model, seed, n):
    rows = []
    for e in range(1, n + 1):
        rows.append(
            {
                "epoch": e,
                "elapsed_seconds": e * 30.0,
                "chemical_ce": 1.0,
                "total_loss": 1.5,
                "supcon_enabled": False,
                "paired_enabled": True,
                "supcon_loss": None,
                "paired_loss": 0.2 * e,
            }
        )
    return {
        "schema": MONITOR_SCHEMA,
        "job_id": _suffix(label),
        "model_id": model,
        "policy_id": policy,
        "seed": seed,
        "stage": "final_refit",
        "validation_available": False,
        "epoch_budget": 50,
        "status": "complete",
        "stop_reason": "fixed_duration",
        "epochs_completed": n,
        "elapsed_seconds": n * 30.0,
        "rows": rows,
    }


def _valid():
    a = _expected("a", "PP-U-SG", "D1", 20260805, "source_fit")
    b = _expected("b", "PP-U-SG", "D1", 20260817, "source_fit")
    c = _expected("c", "PP-U-SG", "D0-M", 20260805, "source_fit")
    d = _expected("d", "PP-U-ARPLS", "D2", 20260829, "final_refit")
    monitors = {
        a["job_id"]: _source_monitor("a", "PP-U-SG", "D1", 20260805, 32, lambda e: 1.0 - 0.01 * e),
        b["job_id"]: _source_monitor("b", "PP-U-SG", "D1", 20260817, 30, lambda e: 2.0 - 0.02 * e),
        d["job_id"]: _refit_monitor("d", "PP-U-ARPLS", "D2", 20260829, 30),
    }
    return [a, b, c, d], monitors


def _curve(result, policy, recipe, stage, epoch, metric):
    for row in result["semantic"]["curves"]:
        if (
            row["policy_id"] == policy
            and row["recipe"] == recipe
            and row["stage"] == stage
            and row["epoch"] == epoch
            and row["metric"] == metric
        ):
            return row
    raise AssertionError("curve row not found")


def _group(result, policy, recipe, stage):
    for group in result["semantic"]["groups"]:
        if (group["policy_id"], group["recipe"], group["stage"]) == (
            policy,
            recipe,
            stage,
        ):
            return group
    raise AssertionError("group not found")


class TrainingDiagnosticsTest(unittest.TestCase):
    def test_semantic_shape_and_manifest(self):
        result = prepare_training_diagnostics(*_valid())
        semantic = result["semantic"]
        self.assertEqual(semantic["schema_version"], "nato-sers-p08-training-figure-data-v1")
        self.assertEqual(semantic["figure_id"], "P08-U1-training-diagnostics")
        self.assertEqual(len(semantic["groups"]), 24)
        self.assertEqual(
            semantic["population"],
            {"spectra": 598, "masters": 69, "instruments": 10},
        )
        self.assertEqual(len(semantic["metric_order"]), 8)
        manifest = result["manifest"]
        self.assertEqual(manifest["status"], "prepared")
        for flag in (
            "reviewed",
            "published",
            "external_authentication_verified",
        ):
            self.assertFalse(manifest[flag])
        self.assertEqual(manifest["counts"]["total_expected"], 4)
        self.assertEqual(manifest["counts"]["monitored"], 3)
        self.assertEqual(manifest["counts"]["missing_history"], 1)
        self.assertEqual(manifest["counts"]["groups"], 24)
        self.assertEqual(manifest["counts"]["curve_rows"], len(semantic["curves"]))
        self.assertEqual(result["semantic_sha256"], manifest["semantic_sha256"])

    def test_quantiles_and_attrition(self):
        result = prepare_training_diagnostics(*_valid())
        row = _curve(result, "PP-U-SG", "D1", "source_fit", 30, "chemical_ce")
        self.assertEqual(row["n_runs_at_epoch"], 2)
        self.assertEqual(row["finite_count"], 2)
        self.assertEqual(row["undefined_count"], 0)
        self.assertIsNone(row["reason"])
        self.assertAlmostEqual(row["median"], 1.05)
        self.assertAlmostEqual(row["q10"], 0.77)
        self.assertAlmostEqual(row["q90"], 1.33)
        late = _curve(result, "PP-U-SG", "D1", "source_fit", 31, "chemical_ce")
        self.assertEqual(late["n_runs_at_epoch"], 1)
        self.assertAlmostEqual(late["median"], 0.69)
        last = _curve(result, "PP-U-SG", "D1", "source_fit", 32, "chemical_ce")
        self.assertEqual(last["n_runs_at_epoch"], 1)
        self.assertAlmostEqual(last["median"], 0.68)

    def test_absent_aux_and_validation_are_none_not_zero(self):
        result = prepare_training_diagnostics(*_valid())
        aux = _curve(result, "PP-U-SG", "D1", "source_fit", 5, "paired_loss")
        self.assertEqual(aux["n_runs_at_epoch"], 2)
        self.assertEqual(aux["finite_count"], 0)
        self.assertEqual(aux["undefined_count"], 2)
        self.assertIsNone(aux["median"])
        self.assertEqual(aux["reason"], "metric_not_recorded")
        nll = _curve(result, "PP-U-ARPLS", "D2", "final_refit", 5, "train_nll")
        self.assertEqual(nll["n_runs_at_epoch"], 1)
        self.assertIsNone(nll["median"])
        self.assertEqual(nll["reason"], "metric_not_recorded")
        paired = _curve(result, "PP-U-ARPLS", "D2", "final_refit", 5, "paired_loss")
        self.assertAlmostEqual(paired["median"], 1.0)
        self.assertIsNone(paired["reason"])

    def test_group_coverage(self):
        result = prepare_training_diagnostics(*_valid())
        full = _group(result, "PP-U-SG", "D1", "source_fit")
        self.assertEqual((full["planned_jobs"], full["monitored_jobs"]), (2, 2))
        self.assertEqual(full["missing_history_jobs"], 0)
        self.assertEqual(full["max_recorded_epoch"], 32)
        self.assertEqual(full["status"], "complete_history_coverage")
        missing = _group(result, "PP-U-SG", "D0-M", "source_fit")
        self.assertEqual(missing["planned_jobs"], 1)
        self.assertEqual(missing["monitored_jobs"], 0)
        self.assertEqual(missing["missing_history_jobs"], 1)
        self.assertIsNone(missing["max_recorded_epoch"])
        self.assertEqual(missing["status"], "missing_histories")
        self.assertEqual(missing["coverage"], "NA")
        empty = _group(result, "PP-U-ARPLS", "D1", "source_fit")
        self.assertEqual(empty["status"], "no_registered_jobs")
        refit = _group(result, "PP-U-ARPLS", "D2", "final_refit")
        self.assertEqual(refit["status"], "complete_history_coverage")
        self.assertEqual(refit["max_recorded_epoch"], 30)

    def test_individual_identifiers_and_seeds_are_absent(self):
        real, monitors = _valid()
        dump = json.dumps(prepare_training_diagnostics(real, monitors)["semantic"], sort_keys=True)
        for job in real:
            self.assertNotIn(job["job_id"], dump)
        for seed in ("20260805", "20260817", "20260829"):
            self.assertNotIn(seed, dump)
        for token in ("P08JOB-", "role_id", "context_id", "private"):
            self.assertNotIn(token, dump)

        jobs = []
        monitors = {}
        for index, seed in enumerate((20260805, 20260817, 20260829)):
            label = f"seed{index}"
            job = _expected(label, "PP-U-SG", "D3", seed, "source_fit")
            jobs.append(job)
            monitors[job["job_id"]] = _source_monitor(
                label, "PP-U-SG", "D3", seed, 30, 1.0, paired=True
            )
        grouped = json.dumps(
            prepare_training_diagnostics(jobs, monitors)["semantic"], sort_keys=True
        )
        for job in jobs:
            self.assertNotIn(job["job_id"], grouped)
        for seed in ("20260805", "20260817", "20260829"):
            self.assertNotIn(seed, grouped)

    def test_no_mutation_and_input_shuffle_invariance(self):
        expected, monitors = _valid()
        expected_copy = copy.deepcopy(expected)
        monitors_copy = copy.deepcopy(monitors)
        first = prepare_training_diagnostics(expected, monitors)
        self.assertEqual(expected, expected_copy)
        self.assertEqual(monitors, monitors_copy)
        second = prepare_training_diagnostics(
            list(reversed(expected)), dict(reversed(list(monitors.items())))
        )
        self.assertEqual(first["semantic"], second["semantic"])
        self.assertEqual(first["semantic_sha256"], second["semantic_sha256"])

    def test_optional_optimizer_steps(self):
        expected, monitors = _valid()
        del monitors[_full("a")]["rows"][0]["total_optimizer_steps"]
        result = prepare_training_diagnostics(expected, monitors)
        self.assertEqual(result["manifest"]["counts"]["monitored"], 3)

    def test_rejections(self):
        expected, monitors = _valid()

        def E():
            return copy.deepcopy(expected)

        def M():
            return copy.deepcopy(monitors)

        def rec(store, label):
            return store[_full(label)]

        cases = [
            ([], {}, "empty expected"),
            (E()[:1] + [copy.deepcopy(E()[0])], M(), "duplicate expected"),
        ]
        e = E()
        e[0]["job_id"] = "P08JOB-zz"
        cases.append((e, M(), "bad hash"))
        e = E()
        e[0]["seed"] = True
        cases.append((e, M(), "bool seed"))
        e = E()
        e[0]["seed"] = 1
        cases.append((e, M(), "unknown seed"))
        e = E()
        e[0]["policy_id"] = "PP-U-XX"
        cases.append((e, M(), "unknown policy"))
        e = E()
        e[0]["model_id"] = "D9"
        cases.append((e, M(), "unknown model"))
        e = E()
        e[0]["stage"] = "eval"
        cases.append((e, M(), "unknown stage"))
        e = E()
        e[0]["extra"] = 1
        cases.append((e, M(), "expected extra field"))

        m = M()
        m["P08JOB-" + "0" * 64] = {}
        cases.append((E(), m, "extra monitor job"))
        m = M()
        m[_full("a")] = None
        cases.append((E(), m, "explicit null is not a missing mapping"))
        m = M()
        rec(m, "a")["epoch_budget"] = 100
        cases.append((E(), m, "wrong source stopping budget"))

        for status in ("failed", "interrupted"):
            m = M()
            rec(m, "a")["status"] = status
            cases.append((E(), m, "non-complete status"))
        m = M()
        rec(m, "a")["stop_reason"] = "bogus"
        cases.append((E(), m, "bad stop_reason"))
        m = M()
        rec(m, "a")["schema"] = "other"
        cases.append((E(), m, "bad schema"))
        m = M()
        rec(m, "a")["validation_available"] = False
        cases.append((E(), m, "source validation unavailable"))
        m = M()
        rec(m, "d")["validation_available"] = True
        cases.append((E(), m, "refit validation available"))
        m = M()
        rec(m, "a")["epoch_budget"] = 201
        cases.append((E(), m, "budget out of range"))
        m = M()
        rec(m, "a")["epochs_completed"] = 29
        cases.append((E(), m, "completed mismatch"))
        m = M()
        rec(m, "a")["elapsed_seconds"] = 1.0
        cases.append((E(), m, "monitor elapsed too small"))
        m = M()
        rec(m, "a")["extra"] = 1
        cases.append((E(), m, "monitor extra field"))
        m = M()
        del rec(m, "a")["rows"]
        cases.append((E(), m, "monitor missing field"))
        m = M()
        rec(m, "a")["model_id"] = "D2"
        cases.append((E(), m, "monitor model mismatch"))
        m = M()
        rec(m, "b")["job_id"] = "0" * 64
        cases.append((E(), m, "monitor job_id mismatch"))
        m = M()
        rec(m, "a")["rows"][0]["epoch"] = 2
        cases.append((E(), m, "non-contiguous epochs"))
        m = M()
        rec(m, "a")["rows"][1]["elapsed_seconds"] = 0.0
        cases.append((E(), m, "decreasing row elapsed"))
        m = M()
        rec(m, "a")["rows"][0]["chemical_ce"] = float("nan")
        cases.append((E(), m, "non-finite metric"))
        m = M()
        rec(m, "a")["rows"][0]["total_loss"] = {"x": 1}
        cases.append((E(), m, "nested metric"))
        m = M()
        rec(m, "a")["rows"][0]["validation_balanced_accuracy"] = 1.5
        cases.append((E(), m, "balanced accuracy out of range"))
        m = M()
        rec(m, "a")["rows"][0]["best_epoch"] = 999
        cases.append((E(), m, "best_epoch out of range"))
        m = M()
        rec(m, "a")["rows"][0]["nonimproving_epochs"] = -1
        cases.append((E(), m, "negative nonimproving_epochs"))
        m = M()
        rec(m, "a")["rows"][0]["total_optimizer_steps"] = -1
        cases.append((E(), m, "negative optimizer steps"))
        m = M()
        rec(m, "a")["rows"][0]["total_optimizer_steps"] = True
        cases.append((E(), m, "bool optimizer steps"))
        m = M()
        del rec(m, "a")["rows"][0]["validation_nll"]
        cases.append((E(), m, "source missing validation field"))
        m = M()
        rec(m, "a")["rows"][0]["paired_loss"] = 0.1
        cases.append((E(), m, "disabled aux loss finite"))
        m = M()
        rec(m, "a")["rows"][0]["supcon_loss"] = None
        cases.append((E(), m, "enabled aux loss missing"))
        m = M()
        rec(m, "a")["rows"][0]["paired_enabled"] = True
        cases.append((E(), m, "inconsistent recipe flags"))
        m = M()
        rec(m, "a")["rows"][0]["supcon_enabled"] = "yes"
        cases.append((E(), m, "non-bool recipe flag"))
        m = M()
        rec(m, "d")["rows"][0]["train_nll"] = 0.5
        cases.append((E(), m, "refit validation field present"))

        for exp, mon, label in cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    prepare_training_diagnostics(exp, mon)


def test_actual_live_monitor_saved_document(tmp_path):
    """No model: exercise the real frozen callback/finalization schema."""
    from atlas_sers.visualization.p08_live_monitor import EpochMonitor

    jobs, records = [], {}
    for stage in ("source_fit", "calibration_model_fit", "final_refit"):
        model = "D1" if stage == "source_fit" else "D2"
        job = _expected(stage, "PP-U-SG", model, 20260805, stage)
        source = stage == "source_fit"
        rows = (
            _source_rows(30, 1.0)
            if source
            else _refit_monitor(stage, "PP-U-SG", model, 20260805, 30)["rows"]
        )
        directory = tmp_path / stage
        monitor = EpochMonitor(
            str(directory),
            job_id=job["job_id"].removeprefix("P08JOB-"),
            model_id=model,
            policy_id=job["policy_id"],
            seed=job["seed"],
            stage=stage,
            validation_available=source,
            epoch_budget=200 if source else 30,
            stream=None,
        )
        try:
            for row in rows:
                monitor(row)
            monitor.finish("complete", stop_reason="patience" if source else "fixed_duration")
        finally:
            monitor.close()
        jobs.append(job)
        records[job["job_id"]] = json.loads((directory / "semantic.json").read_text())
    result = prepare_training_diagnostics(jobs, records)
    assert result["manifest"]["counts"]["monitored"] == 3
    assert result["manifest"]["counts"]["missing_history"] == 0
    assert result["manifest"]["external_authentication_verified"] is False


if __name__ == "__main__":
    unittest.main()
