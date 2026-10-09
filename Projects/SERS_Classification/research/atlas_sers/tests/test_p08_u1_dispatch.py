"""Whole classical DAG on invented spectra; never the field-trial dataset."""

import hashlib
import io
import json
import time

import pandas as pd
import pytest
import torch

from atlas_sers.evaluation import p08_plan, p08_u0_arrays, p08_u1_selection
from atlas_sers.evaluation import p08_u1_dispatch as dispatch
from atlas_sers.evaluation import p08_u1_post_inputs as post
from atlas_sers.evaluation.p08_u1_artifacts import ArtifactStore
from atlas_sers.evaluation.p08_u1_device import prepare_worker_device
from atlas_sers.governance.canonical import sha256_value
from tests import test_p08_u0_runtime_inputs as fixture
from tests import test_p08_u1_source_inputs as source_fixture


@pytest.mark.parametrize(
    "model_id,device",
    [
        ("C-RBF-SVM", "cpu"),
        ("D0-M", "cpu"),
        ("D0-M", "cuda:0"),
        ("D3", "cuda:0"),
    ],
)
def test_source_to_held_end_to_end(tmp_path, monkeypatch, model_id, device):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("GPU integration requires usable CUDA")
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    prepare_worker_device(device)
    ctx = fixture._build_context(monkeypatch, "master_cv")
    plan, source = source_fixture._prepare(monkeypatch, ctx)
    metadata = ctx["metadata_bytes"]

    def parse(name):
        return pd.read_csv(io.BytesIO(metadata[name]), dtype=str, keep_default_na=False)

    inputs = post.PostSourceInputs(
        source,
        parse("manifest_bytes"),
        parse("contexts_bytes"),
        parse("roles_bytes"),
        pd.DataFrame(),
        {},
    )
    monkeypatch.setattr(post, "_ACTION_PINS", p08_u0_arrays._ACTION_PINS)
    jobs = [j for j in plan["jobs"] if j["policy_id"] == "PP-U-SG" and j["model_id"] == model_id]
    assert jobs and all(j["stage"] != "calibration_model_fit" for j in jobs)
    template = jobs[0]
    source_jobs = [j for j in jobs if j["stage"] == "source_fit"]
    units = list(
        {
            j["unit_id"]: {k: j[k] for k in ("unit_id", "fit_uid_sha256", "validation_uid_sha256")}
            for j in source_jobs
        }.values()
    )
    outer = inputs._roles[
        (inputs._roles.context_id == template["context_id"]) & (inputs._roles.role == "outer_fit")
    ].observation_uid.tolist()
    context = dict(
        context_id=template["context_id"],
        selection_mode="master_cv",
        selected_recipe_id=model_id if model_id in p08_plan.NEURAL_RECIPES else "D0-M",
        outer_fit_uid_sha256=sha256_value(sorted(outer)),
        outer_test_uid_sha256=template["test_uid_sha256"],
        selection_units=units,
        calibration_units=units,
    )
    candidates = {
        m: [r for r in fixture.CANDIDATE_ROWS if r["model_id"] == m]
        for m in p08_plan.CLASSICAL_MODELS
    }
    full_jobs, _ = p08_plan._build_context(
        "PP-U-SG",
        template["representation_id"],
        template["array_sha256"],
        p08_plan.EVIDENCE_FUTURE,
        context,
        candidates,
        source._specification_hashes,
    )
    jobs = [j for j in full_jobs if j["model_id"] == model_id]
    row = candidates["C-RBF-SVM"][0] | dict(
        family_order="0",
        family_candidate_order="0",
        complexity_rank="0",
        stochastic="false",
        technical_seeds="deterministic",
        seed_count="1",
    )
    registry = (
        pd.DataFrame([row], columns=p08_u1_selection.REGISTRY_COLUMNS).to_csv(index=False).encode()
    )
    monkeypatch.setattr(p08_u1_selection, "REGISTRY_SHA256", hashlib.sha256(registry).hexdigest())
    artifacts = ArtifactStore(tmp_path, binding_sha256="a" * 64)
    engine = dispatch.UniversalDispatcher(
        jobs=jobs,
        source_factory=source,
        post_inputs=inputs,
        artifacts=artifacts,
        candidate_registry_bytes=registry,
        monitor_root=tmp_path / "monitor",
        stream=io.StringIO(),
        device=device,
        global_deadline=time.perf_counter() + 300,
    )
    complete = set()
    pending = {j["job_id"]: j for j in jobs}
    while pending:
        ready = [j for j in pending.values() if set(j["dependencies"]) <= complete]
        assert ready
        if engine.retained is not None:
            ready = [
                j
                for j in ready
                if j["stage"] == "source_validation_prediction"
                and j["dependencies"] == [engine.retained[0]]
            ]
        assert ready
        job = sorted(ready, key=lambda j: j["job_id"])[0]
        result = engine.execute(job)
        receipt, payload = artifacts.verify(job)
        assert result["status"] == "complete", payload.get("error.json", b"").decode()
        assert dispatch.receipt_result_verifier(artifacts, job, result) == receipt
        complete.add(job["job_id"])
        del pending[job["job_id"]]
    final = [j for j in jobs if j["stage"] == "seed_ensemble_prediction"]
    assert final
    for job in final:
        _, payload = artifacts.verify(job)
        frame = pd.read_csv(io.BytesIO(payload["predictions.csv"]), keep_default_na=False)
        assert not frame.empty
        expected_status = (
            "cross_fitted_temperature"
            if model_id in p08_plan.CLASSICAL_MODELS
            else "seedwise_temperature_ensemble"
        )
        assert frame.probability_status.eq(expected_status).all()
        assert all(len(json.loads(p)) == 3 for p in frame.probabilities)
    assert engine.retained is None


def test_dispatcher_rejects_unregistered_job_before_kernel(tmp_path):
    engine = dispatch.UniversalDispatcher(
        jobs=[],
        source_factory=None,
        post_inputs=None,
        artifacts=None,
        candidate_registry_bytes=b"",
        monitor_root=tmp_path,
        stream=None,
        device="cpu",
        global_deadline=time.perf_counter() + 10,
    )
    with pytest.raises(ValueError, match="unregistered"):
        engine.execute({"job_id": "absent"})
