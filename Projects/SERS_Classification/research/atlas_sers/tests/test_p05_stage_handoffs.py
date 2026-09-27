"""CPU-only producer/consumer handoff tests for the P05 stages.

The synthetic ``world`` fixture is reused from
``tests.test_p05_comprehensive_refits``. Training, calibration fitting, prior
selection, input/device/provenance probes and source-pair counts remain stubbed.
Evidence persistence and the refit/evaluation producers and consumers use real
temporary files. The inference stub reads the persisted checkpoint and
temperature but does not construct a model or touch scientific data.
"""

from __future__ import annotations

# ruff: noqa: E402
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import classical
from atlas_sers.evaluation import p05_calibration as calibration
from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_comprehensive_refits as runner
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_evaluation_authority as consumer
from atlas_sers.evaluation import p05_frozen_predictions as frozen
from atlas_sers.evaluation import p05_outer_inputs as outer_inputs
from atlas_sers.evaluation import p05_prediction as prediction
from atlas_sers.evaluation import p05_prediction_io as prediction_io
from atlas_sers.evaluation import p05_refit_authority as refit_authority
from atlas_sers.evaluation import p05_refit_evidence as evidence
from tests import test_p05_comprehensive_refits as refit_fixtures
from tests.test_p05_comprehensive_refits import (
    CLASSES,
    PRIOR_SECONDS,
    SOURCE_OPTIMIZER_STEPS,
)

world = refit_fixtures.world
REAL_PERSIST_CALIBRATION = evidence.persist_calibration
REAL_SUMMARIZE_RESULT = evidence.summarize_result
REAL_CHECK_RECIPE_GROUP = evidence.check_recipe_group
ALIAS_COUNT = 9
EXPECTED_REFITS = 6
EXPECTED_REFIT_STEPS = 720
DEADLINE_SECONDS = 3600.0
CONTEXT_ID = "ctx-1"
TEST_UIDS = ("001", "NA")


def _calibrate(*, spec, ledger, manifest, load_logits):
    calibrated = calibration.TemperatureCalibration(
        temperature=1.0,
        class_vocabulary=tuple(spec["classes"]),
        observations=3,
        masters=3,
        fit_observation_uid_sha256="a" * 64,
        fit_master_uid_sha256="b" * 64,
        optimizer_success=True,
        optimizer_objective=0.5,
    )
    audit = {
        "calibration_state_sha256": calibrated.state_sha256,
        "refit_id": spec["refit_id"],
        "context_id": spec["context_id"],
        "recipe_id": spec["recipe_id"],
        "seed": spec["seed"],
        "calibration_slot_ids": list(spec["calibration_slot_ids"]),
        "temperature": calibrated.temperature,
        "optimizer_success": True,
        "optimizer_objective": calibrated.optimizer_objective,
    }
    return calibrated, audit


def _train_handler(world):
    def handler(spec, on_epoch):
        result = world.build_result(spec, elapsed=0.0)
        for item in result.history:
            item["sampling_digest"] = result.sampling_digest
            item["augmentation_digest"] = result.augmentation_digest
            item["pair_digest"] = result.pair_digest
            on_epoch(item)
        return result

    return handler


def _selection(world):
    def authenticate_selection(bundle, *, deadline):
        return {
            "prior_seconds": PRIOR_SECONDS,
            "plan": world.plan,
            "source_optimizer_steps": SOURCE_OPTIMIZER_STEPS,
            "selection_receipt": {},
        }

    return authenticate_selection


def _prepare_refit_stage(world, monkeypatch):
    monkeypatch.setattr(evidence, "persist_calibration", REAL_PERSIST_CALIBRATION)
    monkeypatch.setattr(evidence, "summarize_result", REAL_SUMMARIZE_RESULT)
    monkeypatch.setattr(evidence, "check_recipe_group", REAL_CHECK_RECIPE_GROUP)
    monkeypatch.setattr(consumer, "STRATEGY_ALIAS_COUNT", ALIAS_COUNT)
    monkeypatch.setattr(consumer, "MAXIMUM_REFITS", ALIAS_COUNT)
    monkeypatch.setattr(calibration, "calibrate_spec", _calibrate)
    monkeypatch.setattr(refit_authority, "authenticate_selection", _selection(world))
    world.train_handler = _train_handler(world)
    return runner.run_refits(**world.run_kwargs())


def _prepare_outer(bundle, endpoint):
    return {
        "classes": list(CLASSES),
        "values": np.zeros((len(endpoint["test_uids"]), 1401), dtype=np.float32),
        "observation_uids": list(endpoint["test_uids"]),
    }


def _fake_predict(*, spec, unit_dir, values, observation_uids, device, deadline):
    _state, model_sha = prediction._load_result(Path(unit_dir), spec)
    calibrated, calibration_sha = prediction._load_calibration(Path(unit_dir), spec)
    uids = list(observation_uids)
    logits = np.zeros((len(uids), prediction.EXPECTED_CLASS_COUNT), dtype=np.float64)
    logits[:, 0] = 1.0
    probabilities = classical.apply_temperature(logits, calibrated)
    frame = pd.DataFrame({"observation_uid": uids})
    for index in range(prediction.EXPECTED_CLASS_COUNT):
        frame[f"logit_{index}"] = logits[:, index]
        frame[f"probability_{index}"] = probabilities[:, index]
    audit = {
        "refit_id": spec["refit_id"],
        "classes": list(spec["classes"]),
        "model_state_sha256": model_sha,
        "calibration_state_sha256": calibration_sha,
        "test_uid_set_sha256": core._canon().sha256_value(sorted(uids)),
        "rows": len(uids),
        "elapsed_seconds": 0.0,
        "peak_cuda_bytes": 0,
        "optimizer_steps": 0,
    }
    return frame, audit


def test_refit_producer_consumer_handoff(world, monkeypatch):
    receipt = _prepare_refit_stage(world, monkeypatch)

    assert receipt["status"] == "complete"
    authenticated = consumer.authenticate_refits(
        world.bundle, deadline=time.perf_counter() + DEADLINE_SECONDS
    )

    assert authenticated["plan"] is world.plan
    assert authenticated["source_optimizer_steps"] == SOURCE_OPTIMIZER_STEPS
    assert authenticated["refit_optimizer_steps"] == EXPECTED_REFIT_STEPS
    assert isinstance(authenticated["prior_seconds"], float)
    assert authenticated["prior_seconds"] > PRIOR_SECONDS
    assert authenticated["prior_seconds"] == pytest.approx(
        receipt["scientific_seconds_cumulative_bound"]
    )
    assert dict(authenticated["refit_receipt"]) == dict(receipt)
    assert len(world.record["train"]) == EXPECTED_REFITS


def test_evaluation_producer_consumer_handoff(world, monkeypatch):
    specs = list(world.plan["unique_refits"].values())
    endpoint = {"context_id": CONTEXT_ID, "test_uids": list(TEST_UIDS)}
    world.plan["endpoints"] = [endpoint]
    by_recipe_seed = {(s["recipe_id"], s["seed"]): s["refit_id"] for s in specs}
    world.plan["strategy_aliases"] = [
        {
            "context_id": CONTEXT_ID,
            "strategy": strategy,
            "seed": seed,
            "refit_id": by_recipe_seed[("D3" if strategy == "D3" else "D0-M", seed)],
        }
        for seed in sorted({s["seed"] for s in specs})
        for strategy in ("D0-M", "P05-SELECTED", "D3")
    ]
    refit_receipt = _prepare_refit_stage(world, monkeypatch)
    monkeypatch.setattr(prediction_io, "EXPECTED_CONTEXTS", 1)
    monkeypatch.setattr(prediction_io, "EXPECTED_ALIASES", ALIAS_COUNT)
    monkeypatch.setattr(
        outer_inputs, "load_context_rows", lambda bundle: [{"context_id": CONTEXT_ID}]
    )
    monkeypatch.setattr(outer_inputs, "prepare_outer_inputs", _prepare_outer)
    monkeypatch.setattr(prediction, "predict_refit", _fake_predict)

    receipt = evaluation.run_evaluation(**world.run_kwargs())
    authenticated = frozen.authenticate_predictions(
        world.bundle, deadline=time.perf_counter() + DEADLINE_SECONDS
    )

    assert receipt["status"] == "complete"
    assert set(authenticated["predictions"]) == {spec["refit_id"] for spec in specs}
    assert authenticated["source_optimizer_steps"] == SOURCE_OPTIMIZER_STEPS
    assert authenticated["refit_optimizer_steps"] == EXPECTED_REFIT_STEPS
    assert isinstance(authenticated["prior_seconds"], float)
    assert authenticated["prior_seconds"] > refit_receipt["scientific_seconds_cumulative_bound"]
    assert len(world.record["train"]) == EXPECTED_REFITS
