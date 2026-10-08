"""Synthetic tests for the P08-U1 source-only selection adapter."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from atlas_sers.evaluation import p08_plan
from atlas_sers.evaluation import p08_u1_selection as u1
from atlas_sers.evaluation.classical import select_lexicographic_candidate
from atlas_sers.governance.canonical import sha256_value

try:
    from tests import test_p08_plan as planner_fixture
except ImportError:  # pragma: no cover
    import test_p08_plan as planner_fixture

A = "a" * 64
M = "b" * 64
T = "c" * 64
FIT_UID = "d" * 64
VAL_UID = "e" * 64
MASTER_UID = "f" * 64


def _job(**overrides):
    fields = {
        "policy_id": "PP-U-SG",
        "representation_id": "R_SG_400_1800",
        "array_sha256": A,
        "context_id": "ctx-synth",
        "model_id": "C-RBF-SVM",
        "model_spec_sha256": M,
        "stage": "source_fit",
        "unit_id": "unit",
        "seed": "deterministic",
        "candidate_id": "candidate",
        "hyperparameter_sha256": "0" * 64,
        "fit_uid_sha256": FIT_UID,
        "validation_uid_sha256": VAL_UID,
        "test_uid_sha256": T,
        "resolution": "fixed_spec",
        "evidence_status": "unapproved_future_job",
    }
    dependencies = overrides.pop("dependencies", [])
    fields.update(overrides)
    return p08_plan._new_job(fields, dependencies)


def _registry_row(
    candidate_id,
    model_id,
    family_order,
    family_candidate_order,
    declared_candidate_order,
    params,
    complexity_rank,
    seeds,
    seed_count,
):
    return {
        "candidate_id": candidate_id,
        "model_id": model_id,
        "family_order": str(family_order),
        "family_candidate_order": str(family_candidate_order),
        "declared_candidate_order": str(declared_candidate_order),
        "parameters_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
        "hyperparameter_sha256": sha256_value(params),
        "complexity_rank": str(complexity_rank),
        "stochastic": "False" if model_id == "C-RBF-SVM" else "True",
        "technical_seeds": "|".join(str(seed) for seed in seeds),
        "seed_count": str(seed_count),
    }


def _install_registry(monkeypatch, rows):
    frame = pd.DataFrame(rows, columns=list(u1.REGISTRY_COLUMNS))
    raw = frame.to_csv(index=False).encode("utf-8")
    monkeypatch.setattr(u1, "REGISTRY_SHA256", hashlib.sha256(raw).hexdigest())
    return raw


def _classical_scenario(model_id, specs, units, seeds, metric, context_by_unit=None):
    context_by_unit = context_by_unit or {}
    dependencies = {}
    for index, unit in enumerate(units):
        for spec in specs:
            for seed in seeds:
                context = context_by_unit.get(unit, "ctx-synth")
                fit_uid = hashlib.sha256(f"fit-{index}".encode()).hexdigest()
                validation_uid = hashlib.sha256(f"val-{index}".encode()).hexdigest()
                hyperparameter_sha256 = sha256_value(spec["params"])
                fit = _job(
                    stage="source_fit",
                    model_id=model_id,
                    context_id=context,
                    candidate_id=spec["candidate_id"],
                    hyperparameter_sha256=hyperparameter_sha256,
                    unit_id=unit,
                    seed=seed,
                    fit_uid_sha256=fit_uid,
                    validation_uid_sha256=validation_uid,
                )
                prediction = _job(
                    stage="source_validation_prediction",
                    model_id=model_id,
                    context_id=context,
                    candidate_id=spec["candidate_id"],
                    hyperparameter_sha256=hyperparameter_sha256,
                    unit_id=unit,
                    seed=seed,
                    fit_uid_sha256=fit_uid,
                    validation_uid_sha256=validation_uid,
                    dependencies=[fit["job_id"]],
                )
                balanced_accuracy, macro_f1 = metric(unit, spec["candidate_id"], seed)
                dependencies[prediction["job_id"]] = {
                    "fit_job": fit,
                    "prediction_job": prediction,
                    "summary": {
                        "status": "complete",
                        "fit_id": fit["job_id"],
                        "model_id": model_id,
                        "candidate_id": spec["candidate_id"],
                        "seed": seed,
                        "fit_uid_sha256": fit_uid,
                        "validation_uid_sha256": validation_uid,
                        "fit_master_sha256": MASTER_UID,
                        "validation_metrics": {
                            "balanced_accuracy": balanced_accuracy,
                            "macro_f1": macro_f1,
                        },
                    },
                }
    selection = _job(
        stage="select_hyperparameters",
        model_id=model_id,
        unit_id=u1.NOT_APPLICABLE,
        seed=u1.NOT_APPLICABLE,
        candidate_id=u1.SELECTED_CANDIDATE_DEPENDENT,
        hyperparameter_sha256=u1.NOT_APPLICABLE,
        fit_uid_sha256=u1.NOT_APPLICABLE,
        validation_uid_sha256=u1.NOT_APPLICABLE,
        resolution=u1.SELECTED_CANDIDATE_DEPENDENT,
        dependencies=sorted(dependencies),
    )
    rows = [
        _registry_row(
            spec["candidate_id"],
            model_id,
            spec.get("family_order", 1),
            spec.get("family_candidate_order", index),
            spec["declared_candidate_order"],
            spec["params"],
            spec["complexity_rank"],
            seeds,
            len(seeds),
        )
        for index, spec in enumerate(specs)
    ]
    return selection, dependencies, rows


def _neural_scenario(
    recipe, seed, units, best_epochs, completed=None, context_by_unit=None, fit_seed=None
):
    context_by_unit = context_by_unit or {}
    fit_seed = seed if fit_seed is None else fit_seed
    dependencies = {}
    for index, (unit, best_epoch) in enumerate(zip(units, best_epochs, strict=True)):
        context = context_by_unit.get(unit, "ctx-synth")
        fit_uid = hashlib.sha256(f"nfit-{index}".encode()).hexdigest()
        validation_uid = hashlib.sha256(f"nval-{index}".encode()).hexdigest()
        fit = _job(
            stage="source_fit",
            model_id=recipe,
            context_id=context,
            candidate_id="fixed_recipe",
            hyperparameter_sha256=M,
            unit_id=unit,
            seed=fit_seed,
            fit_uid_sha256=fit_uid,
            validation_uid_sha256=validation_uid,
        )
        prediction = _job(
            stage="source_validation_prediction",
            model_id=recipe,
            context_id=context,
            candidate_id="fixed_recipe",
            hyperparameter_sha256=M,
            unit_id=unit,
            seed=fit_seed,
            fit_uid_sha256=fit_uid,
            validation_uid_sha256=validation_uid,
            dependencies=[fit["job_id"]],
        )
        dependencies[prediction["job_id"]] = {
            "fit_job": fit,
            "prediction_job": prediction,
            "summary": {
                "status": "complete",
                "best_epoch": best_epoch,
                "epochs_completed": 200 if completed is None else completed,
                "history": [],
                "seed": fit_seed,
                "recipe_id": recipe,
                "slot_id": fit["job_id"],
                "unit_id": unit,
            },
        }
    selection = _job(
        stage="select_refit_epochs",
        model_id=recipe,
        unit_id=u1.NOT_APPLICABLE,
        seed=seed,
        candidate_id=u1.SELECTED_CANDIDATE_DEPENDENT,
        hyperparameter_sha256=u1.NOT_APPLICABLE,
        fit_uid_sha256=u1.NOT_APPLICABLE,
        validation_uid_sha256=u1.NOT_APPLICABLE,
        resolution=u1.EPOCH_DEPENDENT,
        dependencies=sorted(dependencies),
    )
    return selection, dependencies


def _specs(*pairs, complexity=None, declared=None):
    specs = []
    for index, (candidate_id, params) in enumerate(pairs):
        specs.append(
            {
                "candidate_id": candidate_id,
                "params": params,
                "complexity_rank": (complexity or {}).get(candidate_id, index),
                "declared_candidate_order": (declared or {}).get(candidate_id, index),
            }
        )
    return specs


# --------------------------------------------------------------------------- #
# Classical selection
# --------------------------------------------------------------------------- #


def test_classical_svm_selects_best_mean(monkeypatch):
    specs = _specs(("svm-a", {"k": 1}), ("svm-b", {"k": 2}))

    def metric(unit, cand, seed):
        return (0.9, 0.8) if cand == "svm-a" else (0.5, 0.4)

    selection, dependencies, rows = _classical_scenario(
        "C-RBF-SVM", specs, ["u1"], ("deterministic",), metric
    )
    raw = _install_registry(monkeypatch, rows)
    result = u1.select_classical(
        selection_job=selection,
        dependencies=dependencies,
        candidate_registry_bytes=raw,
    )
    assert result["selected_candidate_id"] == "svm-a"
    assert result["selected_parameters"] == {"k": 1}
    assert result["model_id"] == "C-RBF-SVM"
    assert len(result["trace"]) == 2
    assert result["counts"]["seed_count_per_family_candidate"] == 1
    json.dumps(result, allow_nan=False)


def test_classical_extra_trees_family_selected(monkeypatch):
    specs = _specs(("et-a", {"n": 1}), ("et-b", {"n": 2}))

    def metric(unit, cand, seed):
        return (0.4, 0.3) if cand == "et-a" else (0.8, 0.7)

    selection, dependencies, rows = _classical_scenario(
        "C-EXTRA-TREES", specs, ["u1"], u1.SEEDS, metric
    )
    raw = _install_registry(monkeypatch, rows)
    result = u1.select_classical(
        selection_job=selection, dependencies=dependencies, candidate_registry_bytes=raw
    )
    assert result["selected_candidate_id"] == "et-b"
    assert result["counts"]["seed_count_per_family_candidate"] == 3


def test_classical_family_filter_excludes_other_families(monkeypatch):
    specs = _specs(("rf-a", {"n": 1}), ("rf-b", {"n": 2}))

    def metric(unit, cand, seed):
        return (0.9, 0.8) if cand == "rf-a" else (0.5, 0.4)

    selection, dependencies, rows = _classical_scenario(
        "C-RANDOM-FOREST", specs, ["u1"], u1.SEEDS, metric
    )
    rows.append(
        _registry_row(
            "et-foreign", "C-EXTRA-TREES", 2, 0, 99, {"z": 1}, 0, u1.SEEDS, 3
        )
    )
    raw = _install_registry(monkeypatch, rows)
    result = u1.select_classical(
        selection_job=selection, dependencies=dependencies, candidate_registry_bytes=raw
    )
    assert result["counts"]["family_candidate_count"] == 2
    assert all(
        row["candidate_id"] != "et-foreign" for row in result["trace"]
    )


def test_classical_tie_break_worst_unit_ba(monkeypatch):
    specs = _specs(("rf-a", {"n": 1}), ("rf-b", {"n": 2}))
    def metric(unit, cand, seed):
        if cand == "rf-a":
            return (0.9, 0.1) if unit == "u1" else (0.5, 0.1)
        return (0.85, 0.1) if unit == "u1" else (0.55, 0.1)
    selection, dependencies, rows = _classical_scenario(
        "C-RANDOM-FOREST", specs, ["u1", "u2"], u1.SEEDS, metric
    )
    raw = _install_registry(monkeypatch, rows)
    result = u1.select_classical(
        selection_job=selection, dependencies=dependencies, candidate_registry_bytes=raw
    )
    assert result["selected_candidate_id"] == "rf-b"


def test_classical_tie_break_macro_f1(monkeypatch):
    specs = _specs(("rf-a", {"n": 1}), ("rf-b", {"n": 2}))

    def metric(unit, cand, seed):
        return (0.7, 0.1 if cand == "rf-a" else 0.2)

    selection, dependencies, rows = _classical_scenario(
        "C-RANDOM-FOREST", specs, ["u1", "u2"], u1.SEEDS, metric
    )
    raw = _install_registry(monkeypatch, rows)
    result = u1.select_classical(
        selection_job=selection, dependencies=dependencies, candidate_registry_bytes=raw
    )
    assert result["selected_candidate_id"] == "rf-b"


def test_classical_tie_break_complexity_rank(monkeypatch):
    specs = _specs(
        ("rf-a", {"n": 1}),
        ("rf-b", {"n": 2}),
        complexity={"rf-a": 5, "rf-b": 2},
    )

    def metric(unit, cand, seed):
        return (0.7, 0.2)

    selection, dependencies, rows = _classical_scenario(
        "C-RANDOM-FOREST", specs, ["u1"], u1.SEEDS, metric
    )
    raw = _install_registry(monkeypatch, rows)
    result = u1.select_classical(
        selection_job=selection, dependencies=dependencies, candidate_registry_bytes=raw
    )
    assert result["selected_candidate_id"] == "rf-b"


def test_classical_tie_break_declared_order(monkeypatch):
    specs = _specs(
        ("rf-a", {"n": 1}),
        ("rf-b", {"n": 2}),
        declared={"rf-a": 1, "rf-b": 0},
        complexity={"rf-a": 0, "rf-b": 0},
    )

    def metric(unit, cand, seed):
        return (0.7, 0.2)

    selection, dependencies, rows = _classical_scenario(
        "C-RANDOM-FOREST", specs, ["u1"], u1.SEEDS, metric
    )
    raw = _install_registry(monkeypatch, rows)
    result = u1.select_classical(
        selection_job=selection, dependencies=dependencies, candidate_registry_bytes=raw
    )
    assert result["selected_candidate_id"] == "rf-b"


def test_classical_collapsed_zero_metrics_still_valid(monkeypatch):
    specs = _specs(("svm-a", {"k": 1}), ("svm-b", {"k": 2}))

    def metric(unit, cand, seed):
        return (0.0, 0.0)

    selection, dependencies, rows = _classical_scenario(
        "C-RBF-SVM", specs, ["u1"], ("deterministic",), metric
    )
    raw = _install_registry(monkeypatch, rows)
    result = u1.select_classical(
        selection_job=selection, dependencies=dependencies, candidate_registry_bytes=raw
    )
    assert result["selected_candidate_id"] in ("svm-a", "svm-b")


# --------------------------------------------------------------------------- #
# Classical failures
# --------------------------------------------------------------------------- #


def _classical_ready(monkeypatch):
    specs = _specs(("svm-a", {"k": 1}), ("svm-b", {"k": 2}))

    def metric(unit, cand, seed):
        return (0.9, 0.8) if cand == "svm-a" else (0.5, 0.4)

    selection, dependencies, rows = _classical_scenario(
        "C-RBF-SVM", specs, ["u1"], ("deterministic",), metric
    )
    raw = _install_registry(monkeypatch, rows)
    return selection, dependencies, raw


def test_classical_missing_dependency_fails(monkeypatch):
    selection, dependencies, raw = _classical_ready(monkeypatch)
    trimmed = dict(dependencies)
    trimmed.pop(next(iter(trimmed)))
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=selection, dependencies=trimmed, candidate_registry_bytes=raw
        )
    assert info.value.reason_code == "dependency_missing"


def test_classical_extra_dependency_fails(monkeypatch):
    selection, dependencies, raw = _classical_ready(monkeypatch)
    augmented = dict(dependencies)
    augmented["P08JOB-" + "9" * 64] = next(iter(dependencies.values()))
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=selection, dependencies=augmented, candidate_registry_bytes=raw
        )
    assert info.value.reason_code == "dependency_extra"


def test_classical_svm_wrong_seed_fails(monkeypatch):
    specs = _specs(("svm-a", {"k": 1}))

    def metric(unit, cand, seed):
        return (0.9, 0.8)

    selection, dependencies, rows = _classical_scenario(
        "C-RBF-SVM", specs, ["u1"], ("other",), metric
    )
    raw = _install_registry(monkeypatch, rows)
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=selection, dependencies=dependencies, candidate_registry_bytes=raw
        )
    assert info.value.reason_code == "seed_invalid"


def test_classical_cross_context_fails(monkeypatch):
    specs = _specs(("svm-a", {"k": 1}))

    def metric(unit, cand, seed):
        return (0.9, 0.8)

    selection, dependencies, rows = _classical_scenario(
        "C-RBF-SVM", specs, ["u1"], ("deterministic",), metric,
        context_by_unit={"u1": "ctx-other"},
    )
    raw = _install_registry(monkeypatch, rows)
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=selection, dependencies=dependencies, candidate_registry_bytes=raw
        )
    assert info.value.reason_code == "selection_job_mismatch"


def test_classical_incomplete_summary_fails(monkeypatch):
    selection, dependencies, raw = _classical_ready(monkeypatch)
    mutated = dict(dependencies)
    key = next(iter(mutated))
    entry = dict(mutated[key])
    entry["summary"] = dict(entry["summary"], status="failed")
    mutated[key] = entry
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=selection, dependencies=mutated, candidate_registry_bytes=raw
        )
    assert info.value.reason_code == "summary_status_not_complete"


def test_classical_nonfinite_metric_fails(monkeypatch):
    specs = _specs(("svm-a", {"k": 1}))

    def metric(unit, cand, seed):
        return (float("nan"), 0.5)

    selection, dependencies, rows = _classical_scenario(
        "C-RBF-SVM", specs, ["u1"], ("deterministic",), metric
    )
    raw = _install_registry(monkeypatch, rows)
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=selection, dependencies=dependencies, candidate_registry_bytes=raw
        )
    assert info.value.reason_code == "summary_metric_invalid"


def _real_selection_job(plan, stage, minimum_dependencies=1, policy="PP-U-SG"):
    for job in plan["jobs"]:
        if (
            job["policy_id"] == policy
            and job["stage"] == stage
            and len(job["dependencies"]) >= minimum_dependencies
        ):
            return job
    raise AssertionError(f"no frozen {stage} job with enough dependencies")


def test_integration_real_planner_select_epochs():
    plan = planner_fixture.build_plan()
    jobs = {job["job_id"]: job for job in plan["jobs"]}
    selection = _real_selection_job(plan, "select_refit_epochs")
    dependencies = {}
    for index, prediction_id in enumerate(selection["dependencies"]):
        prediction = jobs[prediction_id]
        fit_id = prediction["dependencies"][0]
        fit = jobs[fit_id]
        dependencies[prediction_id] = {
            "fit_job": fit,
            "prediction_job": prediction,
            "summary": {
                "status": "complete",
                "best_epoch": 40 + index,
                "epochs_completed": 200,
                "history": [],
                "seed": fit["seed"],
                "recipe_id": fit["model_id"],
                "slot_id": fit_id,
                "unit_id": fit["unit_id"],
            },
        }
    result = u1.select_epochs(selection_job=selection, dependencies=dependencies)
    assert result["execution_authorized"] is False
    assert result["model_id"] == selection["model_id"]
    assert result["seed"] == selection["seed"]
    assert len(result["dependency_job_ids"]) == len(dependencies)
    json.dumps(result, allow_nan=False)


def test_real_selection_job_unsorted_dependencies_rejected():
    plan = planner_fixture.build_plan()
    selection = _real_selection_job(plan, "select_refit_epochs", minimum_dependencies=2)
    mutated = dict(selection)
    mutated["dependencies"] = list(reversed(selection["dependencies"]))
    mutated["job_id"] = u1._recompute_job_id(mutated)
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_epochs(selection_job=mutated, dependencies={})
    assert info.value.reason_code == "dependency_invalid"


def test_real_selection_job_hash_mismatch_rejected():
    plan = planner_fixture.build_plan()
    selection = _real_selection_job(plan, "select_refit_epochs")
    mutated = dict(selection, array_sha256="0" * 64)
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_epochs(selection_job=mutated, dependencies={})
    assert info.value.reason_code == "selection_job_invalid"


def test_classical_parity_with_frozen_objective(monkeypatch):
    specs = _specs(("rf-a", {"n": 1}), ("rf-b", {"n": 2}))

    def metric(unit, cand, seed):
        if cand == "rf-a":
            return (0.9, 0.1) if unit == "u1" else (0.5, 0.1)
        return (0.85, 0.1) if unit == "u1" else (0.55, 0.1)

    selection, dependencies, rows = _classical_scenario(
        "C-RANDOM-FOREST", specs, ["u1", "u2"], u1.SEEDS, metric
    )
    raw = _install_registry(monkeypatch, rows)
    result = u1.select_classical(
        selection_job=selection,
        dependencies=dependencies,
        candidate_registry_bytes=raw,
    )
    records = []
    for entry in dependencies.values():
        summary = entry["summary"]
        records.append(
            {
                "candidate_id": summary["candidate_id"],
                "selection_unit_id": entry["fit_job"]["unit_id"],
                "seed": summary["seed"],
                "status": "complete",
                "balanced_accuracy": summary["validation_metrics"]["balanced_accuracy"],
                "macro_f1": summary["validation_metrics"]["macro_f1"],
            }
        )
    registry = pd.DataFrame(
        [
            {
                "candidate_id": row["candidate_id"],
                "model_id": row["model_id"],
                "complexity_rank": int(row["complexity_rank"]),
                "declared_candidate_order": int(row["declared_candidate_order"]),
                "seed_count": int(row["seed_count"]),
            }
            for row in rows
        ]
    )
    winner, _ = select_lexicographic_candidate(pd.DataFrame(records), registry)
    assert result["selected_candidate_id"] == winner["candidate_id"]


def test_forbidden_fit_job_field_rejected(monkeypatch):
    selection, dependencies, raw = _classical_ready(monkeypatch)
    mutated = dict(dependencies)
    key = next(iter(mutated))
    entry = dict(mutated[key])
    entry["fit_job"] = dict(entry["fit_job"], held_prediction=[1, 2, 3])
    mutated[key] = entry
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=selection, dependencies=mutated, candidate_registry_bytes=raw
        )
    assert info.value.reason_code == "fit_job_invalid"


def test_classical_policy_not_permitted(monkeypatch):
    selection, dependencies, raw = _classical_ready(monkeypatch)
    changed = dict(selection, policy_id="PP-U-MIN")
    changed["job_id"] = u1._recompute_job_id(changed)
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=changed, dependencies=dependencies, candidate_registry_bytes=raw
        )
    assert info.value.reason_code == "policy_not_permitted"


def test_registry_sha256_mismatch(monkeypatch):
    selection, dependencies, raw = _classical_ready(monkeypatch)
    monkeypatch.setattr(u1, "REGISTRY_SHA256", "0" * 64)
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=selection, dependencies=dependencies, candidate_registry_bytes=raw
        )
    assert info.value.reason_code == "registry_sha256_mismatch"


def test_registry_header_invalid(monkeypatch):
    selection, dependencies, raw = _classical_ready(monkeypatch)
    bad = b"candidate_id,model_id\nx,y\n"
    monkeypatch.setattr(u1, "REGISTRY_SHA256", hashlib.sha256(bad).hexdigest())
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=selection, dependencies=dependencies, candidate_registry_bytes=bad
        )
    assert info.value.reason_code == "registry_header_invalid"


def test_registry_parameters_hash_mismatch(monkeypatch):
    selection, dependencies, raw = _classical_ready(monkeypatch)
    frame = pd.read_csv(u1.io.BytesIO(raw), dtype=str, keep_default_na=False)
    frame.loc[0, "parameters_json"] = json.dumps({"k": 999})
    tampered = frame.to_csv(index=False).encode("utf-8")
    monkeypatch.setattr(u1, "REGISTRY_SHA256", hashlib.sha256(tampered).hexdigest())
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=selection, dependencies=dependencies, candidate_registry_bytes=tampered
        )
    assert info.value.reason_code == "registry_candidate_hash_mismatch"


def test_classical_coverage_mismatch(monkeypatch):
    selection, dependencies, raw = _classical_ready(monkeypatch)
    frame = pd.read_csv(u1.io.BytesIO(raw), dtype=str, keep_default_na=False)
    extra = {name: "" for name in frame.columns}
    extra["candidate_id"] = "svm-c"
    extra["model_id"] = "C-RBF-SVM"
    extra["family_order"] = "1"
    extra["family_candidate_order"] = "9"
    extra["declared_candidate_order"] = "9"
    extra["parameters_json"] = json.dumps({"k": 3})
    extra["hyperparameter_sha256"] = sha256_value({"k": 3})
    extra["complexity_rank"] = "9"
    extra["stochastic"] = "False"
    extra["technical_seeds"] = "deterministic"
    extra["seed_count"] = "1"
    extended = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)
    tampered = extended.to_csv(index=False).encode("utf-8")
    monkeypatch.setattr(u1, "REGISTRY_SHA256", hashlib.sha256(tampered).hexdigest())
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_classical(
            selection_job=selection, dependencies=dependencies, candidate_registry_bytes=tampered
        )
    assert info.value.reason_code == "dependency_coverage_mismatch"


# --------------------------------------------------------------------------- #
# Refit epochs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "best_epochs,expected",
    [
        ([30, 31], 30),
        ([31, 32], 32),
        ([101, 102], 102),
        ([5, 6], 30),
        ([200, 200], 200),
        ([12], 30),
        ([60, 60, 60], 60),
    ],
)
def test_select_epochs_median_and_clipping(best_epochs, expected):
    units = [f"u{index}" for index in range(len(best_epochs))]
    selection, dependencies = _neural_scenario(
        "D1", 20260805, units, best_epochs
    )
    result = u1.select_epochs(selection_job=selection, dependencies=dependencies)
    assert result["duration"] == expected
    assert result["epochs"] == expected
    assert result["best_epochs"] == best_epochs
    assert result["recipe_id"] == "D1"
    assert result["seed"] == 20260805
    json.dumps(result, allow_nan=False)


def test_select_epochs_long_fit_best_below_minimum():
    selection, dependencies = _neural_scenario(
        "D2", 20260817, ["u1"], [7], completed=200
    )
    result = u1.select_epochs(selection_job=selection, dependencies=dependencies)
    assert result["duration"] == 30
    assert result["epochs"] == 30


def test_select_epochs_clips_above_maximum():
    selection, dependencies = _neural_scenario(
        "D3", 20260829, ["u1", "u2"], [200, 200]
    )
    result = u1.select_epochs(selection_job=selection, dependencies=dependencies)
    assert result["duration"] == 200


def test_select_epochs_wrong_seed_fails():
    selection, dependencies = _neural_scenario(
        "D1", 20260805, ["u1"], [40], fit_seed=20260817
    )
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_epochs(selection_job=selection, dependencies=dependencies)
    assert info.value.reason_code == "seed_mismatch"


def test_select_epochs_cross_context_fails():
    selection, dependencies = _neural_scenario(
        "D1", 20260805, ["u1"], [40], context_by_unit={"u1": "ctx-other"}
    )
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_epochs(selection_job=selection, dependencies=dependencies)
    assert info.value.reason_code == "selection_job_mismatch"


def test_select_epochs_duplicate_unit_fails():
    selection, dependencies = _neural_scenario(
        "D1", 20260805, ["u1", "u1"], [40, 50]
    )
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_epochs(selection_job=selection, dependencies=dependencies)
    assert info.value.reason_code == "dependency_duplicate"


def test_select_epochs_missing_dependency_fails():
    selection, dependencies = _neural_scenario("D1", 20260805, ["u1", "u2"], [40, 50])
    trimmed = dict(dependencies)
    trimmed.pop(next(iter(trimmed)))
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_epochs(selection_job=selection, dependencies=trimmed)
    assert info.value.reason_code == "dependency_missing"


def test_select_epochs_incomplete_fails():
    selection, dependencies = _neural_scenario("D1", 20260805, ["u1"], [40])
    mutated = dict(dependencies)
    key = next(iter(mutated))
    entry = dict(mutated[key])
    entry["summary"] = dict(entry["summary"], status="failed")
    mutated[key] = entry
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_epochs(selection_job=selection, dependencies=mutated)
    assert info.value.reason_code == "summary_status_not_complete"


def test_select_epochs_best_exceeds_completed_fails():
    selection, dependencies = _neural_scenario(
        "D1", 20260805, ["u1"], [50], completed=40
    )
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_epochs(selection_job=selection, dependencies=dependencies)
    assert info.value.reason_code == "summary_metric_invalid"


def test_select_epochs_completed_below_minimum_fails():
    selection, dependencies = _neural_scenario(
        "D1", 20260805, ["u1"], [20], completed=29
    )
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_epochs(selection_job=selection, dependencies=dependencies)
    assert info.value.reason_code == "summary_metric_invalid"


def test_select_epochs_nonneural_recipe_rejected():
    selection, dependencies = _neural_scenario("D1", 20260805, ["u1"], [40])
    changed = dict(selection, model_id="C-RBF-SVM")
    changed["job_id"] = u1._recompute_job_id(changed)
    with pytest.raises(u1.SelectionAdapterError) as info:
        u1.select_epochs(selection_job=changed, dependencies=dependencies)
    assert info.value.reason_code == "model_not_permitted"
