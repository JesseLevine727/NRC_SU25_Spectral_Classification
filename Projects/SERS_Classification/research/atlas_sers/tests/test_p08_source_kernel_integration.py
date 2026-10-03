"""Synthetic compatibility integration of inherited P05 kernels with P08 (T091).

This is a bounded test-only slice.  It drives the real, unmodified P05
development training kernel and the real P03 classical runtime over invented
spectra, persists through the real P05 artifact writers, and feeds the
resulting installed bytes into the accepted P08 semantic verifiers.

Non-claims
----------
* No controller, stage binding, scientific permit, budget or new graph edge is
  added, parsed or exercised.  Execution through the P08 verifiers stays
  unauthorized.
* The PP-U-SG / PP-U-ARPLS job labels are invented placeholders; the synthetic
  scores below are not produced by SG/arPLS preprocessing and this is not a
  preprocessing comparison.
* ``source_validation_prediction`` is treated as a separate materialization and
  acceptance stage, not as a promise that no forward evaluation happened inside
  fitting.  These checks never refit to manufacture scores.
* A finite, structurally consistent score archive can still disagree with the
  restored-best forward pass; the tamper cases demonstrate that limit.
* Everything runs on CPU under the inherited deterministic thread/RNG contract.
  This is synthetic optimization, not real-data training or a scientific result.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p05_pilot  # noqa: E402
from atlas_sers.evaluation import p08_neural_bundle as bundle  # noqa: E402
from atlas_sers.evaluation import p08_source_predictions as source_predictions  # noqa: E402
from atlas_sers.evaluation import p08_training_record as training_record  # noqa: E402
from atlas_sers.evaluation.p08_plan import SEEDS  # noqa: E402
from atlas_sers.governance.canonical import sha256_value  # noqa: E402
from tests.test_p03_runtime import _dataset, _role_uids  # noqa: E402
from tests.test_p05_development import _small_roles  # noqa: E402
from tests.test_p08_source_predictions import make_pair  # noqa: E402

_NEURAL_RECIPES = ("D0-M", "D1", "D2", "D3")
_CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
_POLICIES = ("PP-U-SG", "PP-U-ARPLS")
_CLASS_VOCABULARY = ("a", "b", "c")
_NEURAL_SEED = SEEDS[0]
_CLASSICAL_SEED = SEEDS[1]
_NEURAL_TAMPER_DELTA = 0.125
_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "plan" / "contracts" / "p05_core_contract.json"
)


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _training_record(result: object) -> dict:
    """Project one fit result onto the exact P08 training-record schema."""

    summary = p05_pilot._result_private_summary(result)
    record = {name: summary[name] for name in training_record.RECORD_FIELDS}
    record["history"] = [
        {name: entry[name] for name in training_record.HISTORY_FIELDS}
        for entry in summary["history"]
    ]
    return record


def _score_matrix(frame: object) -> np.ndarray:
    return np.vstack([np.asarray(json.loads(value), dtype=np.float64) for value in frame.scores])


@pytest.fixture(autouse=True)
def _preserve_cpu_rng():
    """Keep the caller CPU RNG across each function-scoped verification test."""

    with torch.random.fork_rng(devices=[]):
        yield


@pytest.fixture(autouse=True)
def _verification_phase_guards(monkeypatch):
    """Fail verification tests that refit a model or initialize CUDA.

    Module-scoped model fixtures are set up before this function-scoped
    fixture, so their real fitting runs unguarded.  Attempts made later are
    recorded here and the teardown assertions fail even when a calling helper
    swallows the raised error.
    """

    attempts = {"neural_fit": 0, "classical_fit": 0, "cuda_init": 0, "cuda_lazy_init": 0}
    development = importlib.import_module("atlas_sers.evaluation.p05_development")
    classical = importlib.import_module("atlas_sers.models.classical")

    def forbidden_train_development_fit(*args, **kwargs):
        attempts["neural_fit"] += 1
        raise AssertionError("train_development_fit called during verification")

    def forbidden_audited_fit(self, *args, **kwargs):
        attempts["classical_fit"] += 1
        raise AssertionError("AuditedClassifier.fit called during verification")

    def guarded_cuda_init(*args, **kwargs):
        attempts["cuda_init"] += 1
        raise AssertionError("torch.cuda.init called during verification")

    monkeypatch.setattr(development, "train_development_fit", forbidden_train_development_fit)
    monkeypatch.setattr(classical.AuditedClassifier, "fit", forbidden_audited_fit)
    monkeypatch.setattr(torch.cuda, "init", guarded_cuda_init)
    if hasattr(torch.cuda, "_lazy_init"):

        def guarded_cuda_lazy_init(*args, **kwargs):
            attempts["cuda_lazy_init"] += 1
            raise AssertionError("torch.cuda._lazy_init called during verification")

        monkeypatch.setattr(torch.cuda, "_lazy_init", guarded_cuda_lazy_init)

    yield

    assert attempts["neural_fit"] == 0
    assert attempts["classical_fit"] == 0
    assert attempts["cuda_init"] == 0
    assert attempts["cuda_lazy_init"] == 0


# --------------------------------------------------------------------------- #
# Neural path: real CPU kernel, real persistence, accepted P08 checks
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _NeuralCase:
    recipe: str
    role_id: str
    result: object
    unit: dict
    slot: dict
    unit_inputs: dict
    validation_uids: list
    classes: tuple
    run_dir: Path
    exec_dir: Path
    best_bytes: bytes
    terminal_bytes: bytes
    source_bytes: bytes
    record: dict
    fit_calls: int


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module", params=_NEURAL_RECIPES)
def neural_case(request, tmp_path_factory) -> _NeuralCase:
    recipe = request.param
    with torch.random.fork_rng(devices=[]):
        fit, validation = _small_roles()
        fit_values, fit_observations, fit_metadata = fit
        validation_values, validation_observations, _ = validation
        station = str(fit_observations[0].station)
        role_id = f"source-fit-{recipe}"
        unit = {
            "unit_id": f"unit-{recipe}",
            "station": station,
            "fitting_role_id": role_id,
            "validation_role_id": f"validation-{recipe}",
            "validation_uids": sorted(str(row.uid) for row in validation_observations),
            "validation_classes": sorted({str(row.target) for row in validation_observations}),
        }
        slot = {"slot_id": f"{recipe}-slot", "recipe_id": recipe, "seed": _NEURAL_SEED}
        unit_inputs = {
            "fitting_values": fit_values,
            "fitting_observations": list(fit_observations),
            "noise": fit_metadata,
            "validation_values": validation_values,
            "validation_observations": list(validation_observations),
        }

        development = importlib.import_module("atlas_sers.evaluation.p05_development")
        real_fit = development.train_development_fit
        counters = {"fit": 0}

        def counting_fit(*args, **kwargs):
            counters["fit"] += 1
            return real_fit(*args, **kwargs)

        monkey = pytest.MonkeyPatch()
        monkey.setattr(development, "train_development_fit", counting_fit)
        try:
            result = p05_pilot.train_fit(
                unit_inputs,
                unit,
                slot,
                "cpu",
                time.perf_counter() + 3600.0,
                lambda record: None,
            )
        finally:
            monkey.undo()

        assert counters["fit"] == 1
        assert result.status == "complete"

        run_dir = tmp_path_factory.mktemp(f"neural-{recipe}")
        p05_pilot.persist_result(torch, run_dir, unit, slot, result)
        exec_dir = run_dir / "executions" / p05_pilot.execution_id(unit, slot)
        return _NeuralCase(
            recipe=recipe,
            role_id=role_id,
            result=result,
            unit=unit,
            slot=slot,
            unit_inputs=unit_inputs,
            validation_uids=list(unit["validation_uids"]),
            classes=tuple(result.classes),
            run_dir=run_dir,
            exec_dir=exec_dir,
            best_bytes=(exec_dir / "best.pt").read_bytes(),
            terminal_bytes=(exec_dir / "terminal.pt").read_bytes(),
            source_bytes=(exec_dir / "validation_logits.npz").read_bytes(),
            record=_training_record(result),
            fit_calls=counters["fit"],
        )


def _verify_neural_bundle(
    case: _NeuralCase,
    policy: str,
    *,
    source_bytes: bytes | None = None,
    record: dict | None = None,
    classes: tuple | None = None,
    uids: list | None = None,
) -> dict:
    source_bytes = case.source_bytes if source_bytes is None else source_bytes
    record = case.record if record is None else record
    classes = case.classes if classes is None else classes
    uids = case.validation_uids if uids is None else uids
    fit_job, prediction_job = make_pair(case.recipe, policy, uids=tuple(uids))
    return bundle.verify_neural_source_bundle(
        fit_job=fit_job,
        prediction_job=prediction_job,
        expected_fit_job_id=fit_job["job_id"],
        expected_prediction_job_id=prediction_job["job_id"],
        expected_role_id=case.role_id,
        expected_validation_uids=list(uids),
        expected_classes=list(classes),
        training_record=record,
        expected_training_record_sha256=sha256_value(record),
        best_checkpoint_bytes=case.best_bytes,
        expected_best_checkpoint_file_sha256=_sha256_bytes(case.best_bytes),
        terminal_checkpoint_bytes=case.terminal_bytes,
        expected_terminal_checkpoint_file_sha256=_sha256_bytes(case.terminal_bytes),
        source_prediction_bytes=source_bytes,
        expected_source_prediction_file_sha256=_sha256_bytes(source_bytes),
    )


def test_neural_kernel_persists_and_restores(neural_case, contract):
    case = neural_case
    assert case.fit_calls == 1
    result = case.result
    observations = case.unit_inputs["validation_observations"]
    assert result.status == "complete"
    assert list(result.validation_uids) == [str(row.uid) for row in observations]
    assert list(result.classes) == sorted({str(row.target) for row in observations})
    assert result.validation_logits.dtype == np.float64
    assert np.isfinite(result.validation_logits).all()
    assert p05_pilot.MINIMUM_EPOCHS <= len(result.history) <= p05_pilot.MAXIMUM_EPOCHS
    assert case.best_bytes and case.terminal_bytes and case.source_bytes
    p05_pilot.check_completed_result(
        result, case.run_dir, case.unit, case.slot, contract, case.unit_inputs, torch, "cpu"
    )


def test_neural_bundle_accepts_both_policies(neural_case):
    case = neural_case
    for policy in _POLICIES:
        report = _verify_neural_bundle(case, policy)
        assert report["bundle_consistency_verified"] is True
        assert report["supplied_file_hashes_verified"] is True
        assert report["training_record_pin_verified"] is True
        assert report["execution_authorized"] is False
        assert report["external_registry_membership_verified"] is False
        assert report["physical_role_isolation_verified"] is False
        assert report["training_completion_verified"] is False
        assert report["prediction_parity_verified"] is False
        assert report["live_resources_verified"] is False
        assert report["row_count"] == len(case.validation_uids)
        assert report["class_count"] == 3
    assert case.fit_calls == 1


def test_neural_finite_tamper_structural_accept_restored_reject(
    neural_case, contract, tmp_path_factory
):
    case = neural_case
    original = np.array(case.result.validation_logits, dtype=np.float64, copy=True)
    tampered_logits = np.array(original, dtype=np.float64, copy=True)
    tampered_logits[0, 0] += _NEURAL_TAMPER_DELTA
    assert np.isfinite(tampered_logits).all()
    tampered_result = dataclasses.replace(case.result, validation_logits=tampered_logits)

    tamper_dir = tmp_path_factory.mktemp(f"tamper-{case.recipe}")
    p05_pilot.persist_result(torch, tamper_dir, case.unit, case.slot, tampered_result)
    exec_dir = tamper_dir / "executions" / p05_pilot.execution_id(case.unit, case.slot)
    tampered_case = dataclasses.replace(
        case,
        result=tampered_result,
        best_bytes=(exec_dir / "best.pt").read_bytes(),
        terminal_bytes=(exec_dir / "terminal.pt").read_bytes(),
        source_bytes=(exec_dir / "validation_logits.npz").read_bytes(),
    )

    # The altered archive is finite and structurally consistent.
    report = _verify_neural_bundle(tampered_case, _POLICIES[0])
    assert report["bundle_consistency_verified"] is True

    # The inherited restored-best check compares against the saved best.pt.
    with pytest.raises(p05_pilot.P05PilotError) as exc:
        p05_pilot.check_completed_result(
            tampered_result,
            tamper_dir,
            case.unit,
            case.slot,
            contract,
            case.unit_inputs,
            torch,
            "cpu",
        )
    assert exc.value.reason_code == "restored_best_logits_mismatch"
    assert np.array_equal(case.result.validation_logits, original)
    assert case.fit_calls == 1


def test_neural_bundle_rejects_changed_record_and_class_order(neural_case):
    case = neural_case
    broken_record = copy.deepcopy(case.record)
    broken_record["epochs_completed"] = 0
    with pytest.raises(bundle.BundleError) as exc:
        _verify_neural_bundle(case, _POLICIES[0], record=broken_record)
    assert exc.value.reason_code == "invalid_counters"

    swapped = list(case.classes)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(bundle.BundleError) as exc:
        _verify_neural_bundle(case, _POLICIES[0], classes=tuple(swapped))
    assert exc.value.reason_code == "class_order_mismatch"
    assert case.fit_calls == 1


# --------------------------------------------------------------------------- #
# Classical path: real P03 runtime, real estimator fit and score alignment
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _ClassicalCase:
    model_id: str
    outcome: object
    fit_uids: list
    validation_uids: list
    vocabulary: tuple
    validation_values: np.ndarray
    fit_calls: int
    alignment_calls: int
    real_aligned: object


@pytest.fixture(scope="module", params=_CLASSICAL_MODELS)
def classical_case(request) -> _ClassicalCase:
    model_id = request.param
    dataset = _dataset()
    fit_uids, validation_uids = _role_uids(dataset)
    runtime = importlib.import_module("atlas_sers.evaluation.p03_runtime")
    classical = importlib.import_module("atlas_sers.models.classical")
    counters = {"fit": 0, "align": 0}
    real_fit = classical.AuditedClassifier.fit
    real_aligned = runtime._aligned_scores

    def counting_fit(self, *args, **kwargs):
        counters["fit"] += 1
        return real_fit(self, *args, **kwargs)

    def counting_aligned(model, values, vocabulary):
        counters["align"] += 1
        return real_aligned(model, values, vocabulary)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(classical.AuditedClassifier, "fit", counting_fit)
    monkey.setattr(runtime, "_aligned_scores", counting_aligned)
    try:
        if model_id == "C-RBF-SVM":
            parameters = {"C": 1.0, "gamma": "scale", "class_weight": "balanced"}
            seed = "deterministic"
            candidate_id = "cand-svm"
        else:
            parameters = {
                "n_estimators": 5,
                "max_features": "sqrt",
                "min_samples_leaf": 1,
                "class_weight": "balanced",
            }
            if model_id == "C-EXTRA-TREES":
                parameters["bootstrap"] = False
            seed = _CLASSICAL_SEED
            candidate_id = "cand-" + model_id
        outcome = runtime.run_candidate_fit(
            dataset=dataset,
            fit_id=f"integration-{model_id}",
            model_id=model_id,
            candidate_id=candidate_id,
            parameters=parameters,
            seed=seed,
            fit_uids=fit_uids,
            validation_uids=validation_uids,
            class_vocabulary=list(_CLASS_VOCABULARY),
        )
    finally:
        monkey.undo()
    return _ClassicalCase(
        model_id=model_id,
        outcome=outcome,
        fit_uids=list(fit_uids),
        validation_uids=list(validation_uids),
        vocabulary=_CLASS_VOCABULARY,
        validation_values=dataset.subset(validation_uids)[0],
        fit_calls=counters["fit"],
        alignment_calls=counters["align"],
        real_aligned=real_aligned,
    )


def test_classical_fit_spies_and_source_prediction(classical_case):
    case = classical_case
    assert case.fit_calls == 1
    assert case.alignment_calls == 1
    outcome = case.outcome
    assert outcome.status == "complete"
    assert outcome.estimator is not None
    frame = outcome.validation_predictions
    assert list(frame.observation_uid) == case.validation_uids
    stored = _score_matrix(frame)
    recomputed = np.asarray(
        case.real_aligned(outcome.estimator, case.validation_values, list(case.vocabulary)),
        dtype=np.float64,
    )
    assert np.array_equal(stored, recomputed)

    for policy in _POLICIES:
        fit_job, prediction_job = make_pair(case.model_id, policy, uids=tuple(case.validation_uids))
        report = source_predictions.verify_source_prediction_values(
            stored,
            observed_uids=list(case.validation_uids),
            observed_classes=list(case.vocabulary),
            expected_validation_uids=list(case.validation_uids),
            expected_classes=list(case.vocabulary),
            fit_job=fit_job,
            prediction_job=prediction_job,
            expected_fit_job_id=fit_job["job_id"],
            expected_prediction_job_id=prediction_job["job_id"],
        )
        assert report["source_prediction_structure_verified"] is True
        assert report["declared_job_pair_verified"] is True
        assert report["execution_authorized"] is False
    assert case.fit_calls == 1
