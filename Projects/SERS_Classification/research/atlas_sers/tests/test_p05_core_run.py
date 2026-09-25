"""Boundary tests for the governed P05-T009 plan/smoke surface.

No real data, no real fits.  Numerical-stack tests are guarded with
``importorskip`` so the suite runs where torch/numpy/pandas are absent.
"""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from atlas_sers.evaluation import p05_core_run as core

ROLES = ["cwa_dense", "pills_dense", "surfaces_dense", "surfaces_sparse"]
RECIPES = [
    {"recipe_id": "D0-M", "lambda_supcon": 0.0, "lambda_pair": 0.0, "projection": False},
    {"recipe_id": "D1", "lambda_supcon": 0.3, "lambda_pair": 0.0, "projection": True},
    {"recipe_id": "D2", "lambda_supcon": 0.0, "lambda_pair": 0.3, "projection": False},
    {"recipe_id": "D3", "lambda_supcon": 0.3, "lambda_pair": 0.3, "projection": True},
]
SEEDS = [20260805, 20260817]


def _digest(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _contract() -> dict:
    return {
        "sampler": {"batch_size_ceiling": 48},
        "recipes": [dict(recipe) for recipe in RECIPES],
        "model": {"base_parameters": 208691, "projection_model_parameters": 212851},
        "smoke": {
            "role_labels": list(ROLES),
            "seeds": list(SEEDS),
            "replays": [
                {"role_label": "cwa_dense", "recipe_id": "D3", "seed": 20260805},
                {"role_label": "surfaces_sparse", "recipe_id": "D2", "seed": 20260805},
            ],
            "expected_sparse_equivalences": [["D0-M", "D2"], ["D1", "D3"]],
        },
        "later_core_plan": {"inner_fit_slot_ceiling": 14940, "extra_guard_unit_slots": 384},
    }


def _fit(role, recipe, seed, kind="primary", execution_id=None, replay_of=None):
    execution_id = execution_id or f"{role}|{recipe}|{seed}"
    return {
        "execution_id": execution_id,
        "fit_id": execution_id,
        "execution_kind": kind,
        "replay_of": replay_of,
        "role_label": role,
        "p05_role_id": f"R-{role}",
        "recipe_id": recipe,
        "seed": seed,
        "epochs": core.EPOCHS,
        "batches_per_epoch": core.DRAWS_PER_EPOCH,
        "optimizer_steps": core.OPTIMIZER_STEPS,
    }


def _plan(contract) -> dict:
    fits = [
        _fit(role, recipe["recipe_id"], seed)
        for role in ROLES
        for recipe in RECIPES
        for seed in SEEDS
    ]
    for replay in contract["smoke"]["replays"]:
        primary = _fit(replay["role_label"], replay["recipe_id"], replay["seed"])
        fits.append(
            _fit(
                replay["role_label"],
                replay["recipe_id"],
                replay["seed"],
                "replay",
                execution_id=f"replay|{replay['role_label']}|{replay['recipe_id']}|{replay['seed']}",
                replay_of=primary["execution_id"],
            )
        )
    return {
        "smoke_roles": [
            {
                "role_label": role,
                "p05_role_id": f"R-{role}",
                "fitting_uids": [f"{role}-1"],
                "uid_set_sha256": _digest(role),
                "support": {},
            }
            for role in ROLES
        ],
        "smoke_observations": [
            {
                "role_label": role,
                "p05_role_id": f"R-{role}",
                "uid": f"{role}-1",
                "master": "m",
                "station": "s",
                "target": "t",
                "instrument": "i",
                "substrate": "sub",
            }
            for role in ROLES
        ],
        "smoke_fits": fits,
        "development_slots": [None] * 14940,
        "guard_roles": [None] * 384,
    }


def _history(recipe, sparse) -> list[dict]:
    supcon_on = recipe["lambda_supcon"] > 0.0
    pair_on = recipe["lambda_pair"] > 0.0
    records = []
    for epoch in range(1, core.EPOCHS + 1):
        records.append(
            {
                "epoch": epoch,
                "chemical_ce": 1.0 + epoch * 0.01,
                "total_loss": 2.0,
                "supcon_loss": 0.0 if sparse else (0.5 if supcon_on else 0.0),
                "paired_loss": 0.0 if sparse else (0.5 if pair_on else 0.0),
                "supcon_enabled": supcon_on,
                "paired_enabled": pair_on,
                "supcon_available_batches": core.DRAWS_PER_EPOCH if supcon_on else 0,
                "paired_available_batches": core.DRAWS_PER_EPOCH if (pair_on and not sparse) else 0,
                "eligible_anchor_count": 2 if supcon_on else 0,
                "zero_positive_anchor_count": 2 if supcon_on else 0,
                "paired_master_count": 8 if (pair_on and not sparse) else 0,
                "gradient_norm_mean": 0.5,
                "gradient_norm_max": 0.8,
                "head_gradient_norm_mean": 0.5 if recipe["projection"] else 0.0,
                "backbone_gradient_norm_mean": 0.5,
                "clipped_fraction": 0.0,
                "embedding_variance": 0.1,
                "embedding_norm_mean": 1.0,
                "train_ba": 0.4 + epoch * 0.001,
                "train_nll": 1.0,
                "train_predicted_class_count": 3,
                "optimizer_steps": epoch * core.DRAWS_PER_EPOCH,
            }
        )
    return records


def _result(role, recipe, seed):
    sparse = role.endswith("_sparse")
    recipe_def = next(item for item in RECIPES if item["recipe_id"] == recipe)
    group = recipe if not sparse else ("A" if recipe in ("D0-M", "D2") else "B")
    supcon_on = recipe_def["lambda_supcon"] > 0.0
    pair_on = recipe_def["lambda_pair"] > 0.0
    return SimpleNamespace(
        status="complete",
        reason_code=None,
        history=_history(recipe_def, sparse),
        parameter_count=212851 if recipe_def["projection"] else 208691,
        initial_state_digest=_digest(f"si|{role}|{seed}|{group}"),
        final_state_digest=_digest(f"sf|{role}|{seed}|{group}"),
        initial_backbone_digest=_digest(f"bb|{role}|{seed}"),
        final_backbone_digest=_digest(f"bf|{role}|{seed}"),
        initial_head_digest=_digest(f"hi|{role}|{seed}|{group}")
        if recipe_def["projection"]
        else None,
        final_head_digest=_digest(f"hf|{role}|{seed}|{group}")
        if recipe_def["projection"]
        else None,
        augmentation_digest=_digest(f"aug|{role}|{seed}"),
        sampling_digest=_digest(f"samp|{role}|{seed}"),
        pair_digest=_digest(f"pair|{role}|{seed}"),
        optimizer_steps=core.OPTIMIZER_STEPS,
        elapsed_seconds=1.0,
        peak_cuda_bytes=0,
        finite_gradient_batches=32,
        nonzero_gradient_elements=10,
        traceback_digest=_digest("trace"),
        supcon_support=(
            {
                "enabled": True,
                "available_batches": 32,
                "eligible_anchors": 64 if sparse else 32,
                "zero_positive_anchors": 64 if sparse else 0,
            }
            if supcon_on
            else {"enabled": False, "available_batches": 0}
        ),
        paired_support={
            "enabled": pair_on,
            "available_batches": 32 if (pair_on and not sparse) else 0,
            "eligible_masters": 8 if (pair_on and not sparse) else 0,
            "pairs": 16 if (pair_on and not sparse) else 0,
        },
    )


def _executions():
    items = [
        {
            "fit": _fit(role, recipe["recipe_id"], seed),
            "result": _result(role, recipe["recipe_id"], seed),
        }
        for role in ROLES
        for recipe in RECIPES
        for seed in SEEDS
    ]
    for replay in _contract()["smoke"]["replays"]:
        primary = _fit(replay["role_label"], replay["recipe_id"], replay["seed"])
        items.append(
            {
                "fit": _fit(
                    replay["role_label"],
                    replay["recipe_id"],
                    replay["seed"],
                    "replay",
                    execution_id=f"replay|{replay['role_label']}|{replay['recipe_id']}|{replay['seed']}",
                    replay_of=primary["execution_id"],
                ),
                "result": _result(replay["role_label"], replay["recipe_id"], replay["seed"]),
            }
        )
    return items


def test_plan_checks_accept_synthetic_plan():
    core._minimal_plan_checks(_plan(_contract()), _contract())


def test_plan_missing_slot_rejected():
    plan = _plan(_contract())
    plan["smoke_fits"] = [
        fit for fit in plan["smoke_fits"] if fit["execution_id"] != "cwa_dense|D0-M|20260805"
    ]
    with pytest.raises(core.P05CoreError):
        core._minimal_plan_checks(plan, _contract())


def test_plan_duplicate_slot_rejected():
    plan = _plan(_contract())
    duplicate = _fit("cwa_dense", "D0-M", SEEDS[0])
    plan["smoke_fits"] = [
        duplicate if fit["execution_id"] == "cwa_dense|D1|20260805" else fit
        for fit in plan["smoke_fits"]
    ]
    with pytest.raises(core.P05CoreError):
        core._minimal_plan_checks(plan, _contract())


def test_replay_target_missing_rejected():
    plan = _plan(_contract())
    plan["smoke_fits"] = [
        {**fit, "replay_of": "absent"} if fit["execution_kind"] == "replay" else fit
        for fit in plan["smoke_fits"]
    ]
    with pytest.raises(core.P05CoreError):
        core._minimal_plan_checks(plan, _contract())


def test_acceptance_passes_for_complete_synthetic_run():
    core._check_acceptance(_contract(), _executions())


def test_replay_mutation_is_detected():
    executions = _executions()
    replay = next(item for item in executions if item["fit"]["execution_kind"] == "replay")
    replay["result"].final_state_digest = _digest("mutated")
    with pytest.raises(core.P05CoreError):
        core._check_acceptance(_contract(), executions)


def test_sparse_supcon_available_paired_absent_allowed():
    contract = _contract()
    executions = _executions()
    core._check_acceptance(contract, executions)


def test_history_missing_field_and_bad_types_rejected():
    recipe = RECIPES[0]
    good = {"fit": None, "result": SimpleNamespace(history=_history(recipe, False))}
    core._check_history({}, good["result"], recipe, False)
    broken = _history(recipe, False)
    broken[0].pop("chemical_ce")
    with pytest.raises(core.P05CoreError):
        core._check_history({}, SimpleNamespace(history=broken), recipe, False)
    broken = _history(recipe, False)
    broken[0]["chemical_ce"] = "not-a-number"
    with pytest.raises(core.P05CoreError):
        core._check_history({}, SimpleNamespace(history=broken), recipe, False)


def test_symlink_ancestor_rejected(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(core.P05CoreError):
        core._reject_symlink_chain(link / "child")


def test_artifact_inside_repository_rejected(tmp_path):
    repository = tmp_path / "repo"
    package = repository / "pkg"
    package.mkdir(parents=True)
    with pytest.raises(core.P05CoreError):
        core._assert_artifact_location(package, repository, repository / "artifacts")
    with pytest.raises(core.P05CoreError):
        core._assert_artifact_location(package, repository, package / "artifacts")


def test_lease_exclusive_blocks_rerun(tmp_path):
    lease = tmp_path / "leases" / "abc"
    core._mkdir_exclusive(lease, "lease_exists")
    with pytest.raises(core.P05CoreError):
        core._mkdir_exclusive(lease, "lease_exists")


def test_authenticate_phase_hashes_report_bytes(tmp_path):
    phase = tmp_path / "p01"
    run = phase / "runs" / "R1"
    run.mkdir(parents=True)
    report = run / "P01_VALIDATION_REPORT.json"
    report.write_bytes(b'{"ok":true}')
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    (run / "table.csv").write_bytes(b"a,b\n1,2\n")
    table_digest = hashlib.sha256((run / "table.csv").read_bytes()).hexdigest()
    (phase / "LATEST.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "run_id": "R1",
                "protected_state_sha256": "p",
                "report_sha256": digest,
            }
        )
    )
    (run / "_STATE.json").write_text(
        json.dumps(
            {
                "execution_status": "complete",
                "scientific_status": "pass",
                "run_id": "R1",
                "protected_state_sha256": "p",
                "files": {"P01_VALIDATION_REPORT.json": digest, "table.csv": table_digest},
            }
        )
    )
    core._authenticate_phase(
        phase, run, "R1", "P01_VALIDATION_REPORT.json", digest, {"table.csv": table_digest}
    )
    report.write_bytes(b'{"ok":false}')
    with pytest.raises(core.P05CoreError):
        core._authenticate_phase(
            phase, run, "R1", "P01_VALIDATION_REPORT.json", digest, {"table.csv": table_digest}
        )


def test_state_schema_rejection(tmp_path):
    phase = tmp_path / "p01"
    run = phase / "runs" / "R1"
    run.mkdir(parents=True)
    (phase / "LATEST.json").write_text(
        json.dumps({"status": "pass", "run_id": "R1", "report_sha256": "x"})
    )
    (run / "_STATE.json").write_text(json.dumps({"run_id": "R1", "status": "pass"}))
    with pytest.raises(core.P05CoreError):
        core._authenticate_phase(phase, run, "R1", "report.json", "x", {})


def test_noise_frame_preserves_uid_column_and_order(tmp_path):
    pandas = pytest.importorskip("pandas")
    path = tmp_path / "primary_manifest.csv"
    path.write_bytes(b"observation_uid,first_difference_noise_mad,intensity_range\nb,1,2\na,3,4\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = core._noise_frame(path, digest, ["a", "b"], pandas)
    assert list(frame["observation_uid"]) == ["a", "b"]
    assert list(frame.columns)[0] == "observation_uid"


def test_load_representation_population_and_axis(tmp_path):
    numpy = pytest.importorskip("numpy")
    path = tmp_path / "R_MIN_400_1800.npz"
    axis = numpy.arange(core.REPRESENTATION_AXIS_START, core.REPRESENTATION_AXIS_STOP)
    intensity = numpy.zeros((2, core.REPRESENTATION_FEATURES), dtype="float32")
    uids = numpy.array(["u1", "u2"])
    numpy.savez(path, axis_cm1=axis, intensity=intensity, observation_uid=uids)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, labels = core._load_representation(path, digest, ["u1", "u2"], 2)
    assert labels == ["u1", "u2"] and loaded.shape[0] == 2
    with pytest.raises(core.P05CoreError):
        core._load_representation(path, digest, ["u1"], 1)


def test_import_stack_brings_smoke_without_metric_stub():
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    core._import_stack()
    assert "atlas_sers.evaluation._metric_values" not in sys.modules


def test_train_one_pins_fixed_budget_and_single_attempt(tmp_path):
    calls = []

    def fake_train_smoke_fit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="complete")

    smoke = SimpleNamespace(train_smoke_fit=fake_train_smoke_fit)
    fit = _fit("cwa_dense", "D3", SEEDS[0])
    values = object()
    result = core._train_one(
        fit,
        values,
        [object()],
        object(),
        smoke,
        "cpu",
        123.0,
        tmp_path / "history.jsonl",
    )
    assert result.status == "complete" and len(calls) == 1
    kwargs = calls[0]
    assert kwargs["epochs"] == core.EPOCHS and kwargs["batches_per_epoch"] == core.DRAWS_PER_EPOCH
    assert kwargs["maximum_fit_seconds"] == core.MAX_FIT_SECONDS
    assert kwargs["global_deadline"] == 123.0 and kwargs["values"] is values


def test_reserve_row_is_exclusive(tmp_path):
    rows = tmp_path / "rows"
    fit = _fit("cwa_dense", "D0-M", SEEDS[0])
    core._reserve_row(rows, fit)
    with pytest.raises(core.P05CoreError):
        core._reserve_row(rows, fit)


def test_persist_error_retains_traceback_digest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fit = _fit("cwa_dense", "D0-M", SEEDS[0])
    core._persist_error(run_dir, fit, core.P05CoreError("boom"))
    payload = json.loads((run_dir / "executions" / fit["execution_id"] / "error.json").read_text())
    assert payload["status"] == "fail" and core._is_hex64(payload["traceback_sha256"])


def test_validate_source_inputs_real_prepare_inputs():
    pytest.importorskip("torch")
    smoke = pytest.importorskip("atlas_sers.evaluation.p05_smoke")
    from tests.test_p05_smoke import _dense

    values, observations, noise = _dense()
    role_inputs = {"cwa_dense": (values, observations, noise)}
    core._validate_source_inputs(smoke, role_inputs)


def test_sample_capacity_ceiling_enforced():
    pytest.importorskip("torch")
    smoke = pytest.importorskip("atlas_sers.evaluation.p05_smoke")
    sampling = pytest.importorskip("atlas_sers.evaluation.p05_sampling")
    observations = [
        sampling.Observation(
            uid=f"u{master}-{view}",
            master=f"m{master}",
            station="s",
            target="t",
            instrument=f"i{view}",
            substrate="sub",
        )
        for master in range(25)
        for view in range(2)
    ]
    role_inputs = {"cwa_dense": (object(), observations, None)}
    with pytest.raises(core.P05CoreError):
        core._validate_sample_capacity(smoke, role_inputs, _contract())


def test_source_preflight_rejects_missing_class():
    pytest.importorskip("torch")
    smoke = pytest.importorskip("atlas_sers.evaluation.p05_smoke")
    sampling = pytest.importorskip("atlas_sers.evaluation.p05_sampling")
    numpy = pytest.importorskip("numpy")
    observations = [
        sampling.Observation(
            uid=f"u{index}",
            master=f"m{index}",
            station="s",
            target=f"t{index % 2}",
            instrument="i",
            substrate="sub",
        )
        for index in range(6)
    ]
    values = numpy.zeros((6, core.REPRESENTATION_FEATURES), dtype="float32")
    values[:, -1] = 1.0
    role_inputs = {"cwa_dense": (values, observations, None)}
    with pytest.raises(core.P05CoreError) as caught:
        core._validate_source_inputs(smoke, role_inputs)
    assert "fitting roles must contain exactly three chemicals" in str(caught.value.__cause__)


def _persist_result_fixture(recipe_id: str) -> SimpleNamespace:
    recipe_def = next(item for item in RECIPES if item["recipe_id"] == recipe_id)
    return SimpleNamespace(
        status="complete",
        reason_code=None,
        history=[
            {"epoch": epoch, "chemical_ce": 1.0 + epoch * 0.01, "optimizer_steps": epoch * 4}
            for epoch in range(1, core.EPOCHS + 1)
        ],
        parameter_count=212851 if recipe_def["projection"] else 208691,
        optimizer_steps=core.OPTIMIZER_STEPS,
        elapsed_seconds=1.0,
        peak_cuda_bytes=0,
        finite_gradient_batches=32,
        nonzero_gradient_elements=10,
        traceback_digest=None,
        augmentation_digest=_digest("aug"),
        sampling_digest=_digest("samp"),
        pair_digest=_digest("pair"),
        supcon_support={},
        paired_support={},
        initial_state_digest=_digest("s0"),
        final_state_digest=_digest("s1"),
        initial_backbone_digest=_digest("b0"),
        final_backbone_digest=_digest("b1"),
        initial_head_digest=_digest("h0"),
        final_head_digest=_digest("h1"),
        state_dict=None,
    )


def test_save_state_cpu_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "state.pt"
    state = {"weight": torch.tensor([1.0, 2.0, 3.0]), "bias": torch.tensor([[0.5]])}
    core._save_state(torch, state, path)
    assert path.is_file() and not path.is_symlink()
    loaded = torch.load(path, weights_only=True)
    assert set(loaded["state_dict"]) == set(state)
    for key, tensor in state.items():
        assert torch.equal(loaded["state_dict"][key], tensor)
    assert [entry.name for entry in tmp_path.iterdir()] == ["state.pt"]


def test_save_state_failure_leaves_no_residue(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "state.pt"
    with pytest.raises(AttributeError):
        core._save_state(torch, {"bad": lambda: None}, path)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_persist_execution_consistent_artifacts(tmp_path):
    torch = pytest.importorskip("torch")
    runtime = pytest.importorskip("atlas_sers.evaluation.p04_runtime")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fit = _fit("cwa_dense", "D1", SEEDS[0])
    result = _persist_result_fixture("D1")
    result.state_dict = {"weight": torch.tensor([1.0, 2.0])}
    result.final_state_digest = runtime._state_hash(result.state_dict)
    core._persist_execution(run_dir, fit, result, torch)
    directory = run_dir / "executions" / fit["execution_id"]
    record = json.loads((directory / "result.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    assert record["status"] == "complete"
    assert record["initial_state_digest"] == result.initial_state_digest
    assert history == result.history
    checkpoint = torch.load(directory / "state.pt", weights_only=True)
    assert torch.equal(checkpoint["state_dict"]["weight"], result.state_dict["weight"])
    assert runtime._state_hash(checkpoint["state_dict"]) == record["final_state_digest"]
    assert record["initial_state_digest"] == result.initial_state_digest
    assert sorted(entry.name for entry in directory.iterdir()) == [
        "history.json",
        "result.json",
        "state.pt",
    ]


def test_checkpoint_preflight_passes(tmp_path):
    torch = pytest.importorskip("torch")
    core._checkpoint_preflight(torch, tmp_path)
    preflight = tmp_path / core.P05CORE_NAMESPACE / "preflight"
    assert preflight.is_dir()
    assert list(preflight.iterdir()) == []


def test_checkpoint_preflight_controlled_failure(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    def failing_save(_torch, _state, _path):
        raise core.P05CoreError("checkpoint_preflight_failed")

    monkeypatch.setattr(core, "_save_state", failing_save)
    with pytest.raises(core.P05CoreError):
        core._checkpoint_preflight(torch, tmp_path)
