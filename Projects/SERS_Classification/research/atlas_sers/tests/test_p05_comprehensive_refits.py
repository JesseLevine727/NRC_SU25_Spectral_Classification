"""CPU-only orchestration tests for the comprehensive P05 refit runner.

These tests never fit a scientific model. They drive the runner against a
synthetic fixture world backed by real temporary filesystem components and the
real ``p05_refit_io`` persistence/acceptance helpers, while the training kernel,
calibration, authority, provenance and device probes are replaced by
deterministic stubs.
"""

from __future__ import annotations

# The optional torch import must precede torch-dependent project modules.
# ruff: noqa: E402
import hashlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p04_runtime as runtime
from atlas_sers.evaluation import p05_calibration as calibration
from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_comprehensive_refits as runner
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_refit as refit
from atlas_sers.evaluation import p05_refit_authority as authority
from atlas_sers.evaluation import p05_refit_evidence as evidence
from atlas_sers.evaluation import p05_refit_io as refit_io
from atlas_sers.evaluation import p05_refit_plan as refit_plan
from atlas_sers.evaluation import p05_selection as selection
from atlas_sers.evaluation.p05_refit import RefitResult

RECIPES = ("D0-M", "D3")
FITTING_UIDS = ("fA", "fB", "fC")
CLASSES = ("A", "B", "C")
CALIBRATION_SLOTS = ("calib-1",)
EPOCHS = 30
SOURCE_OPTIMIZER_STEPS = 96
PRIOR_SECONDS = 3720.0


def _forbidden_loader(*args, **kwargs):
    raise AssertionError("calibration stub must not load logits")


def _temperature_calibration():
    return calibration.TemperatureCalibration(
        temperature=1.0,
        class_vocabulary=CLASSES,
        observations=3,
        masters=3,
        fit_observation_uid_sha256="a" * 64,
        fit_master_uid_sha256="b" * 64,
        optimizer_success=True,
        optimizer_objective=0.5,
    )


def _build_result(
    spec,
    *,
    status="complete",
    reason_code=None,
    optimizer_steps=None,
    epochs_completed=None,
    history_epochs=None,
    terminal=True,
    elapsed=0.1,
    peak=0,
):
    epochs = int(spec["epochs"])
    complete = status == "complete"
    if history_epochs is None:
        history_epochs = epochs if complete else 0
    if optimizer_steps is None:
        optimizer_steps = epochs * refit_io.UPDATES_PER_EPOCH if complete else 0
    if epochs_completed is None:
        epochs_completed = epochs if complete else history_epochs
    history = [
        {
            "epoch": index,
            "chemical_ce": 0.5,
            "total_loss": 0.6,
            "supcon_loss": 0.0,
            "paired_loss": 0.0,
            "epoch_optimizer_steps": refit_io.UPDATES_PER_EPOCH,
            "total_optimizer_steps": index * refit_io.UPDATES_PER_EPOCH,
        }
        for index in range(1, history_epochs + 1)
    ]
    state = {
        "weight": torch.zeros(2, 2, dtype=torch.float32),
        "bias": torch.zeros(2, dtype=torch.float32),
    }
    digest = runtime._state_hash(state)
    seed_digest = hashlib.sha256(str(spec["seed"]).encode("utf-8")).hexdigest()
    architecture_digest = hashlib.sha256(f"{spec['seed']}:{spec['recipe_id']}".encode()).hexdigest()
    head_digest = hashlib.sha256(f"head:{spec['seed']}".encode()).hexdigest()
    return RefitResult(
        status=status,
        reason_code=reason_code,
        history=history,
        epochs=epochs,
        epochs_completed=epochs_completed,
        parameter_count=refit_io._expected_parameter_count(spec["recipe_id"]),
        optimizer_steps=optimizer_steps,
        zero_gradient_batches=0,
        initial_state_digest=architecture_digest,
        terminal_state_digest=digest,
        initial_backbone_digest=seed_digest,
        terminal_backbone_digest=seed_digest,
        initial_head_digest=head_digest,
        terminal_head_digest=head_digest,
        terminal_state_dict=state if terminal else None,
        state_capture_failed=False,
        classes=tuple(spec["classes"]),
        source_noise_levels=(0.0,),
        augmentation_digest=seed_digest,
        sampling_digest=seed_digest,
        pair_digest=seed_digest,
        finite_gradient_batches=optimizer_steps,
        nonzero_gradient_elements=1,
        supcon_support={
            "enabled": 0,
            "available_batches": 0,
            "eligible_anchors": 0,
            "zero_positive_anchors": 0,
        },
        paired_support={
            "enabled": 1 if spec["recipe_id"] == "D3" else 0,
            "available_batches": 0,
            "eligible_masters": 0,
            "pairs": 0,
        },
        role_id=spec["fitting_role_id"],
        recipe=spec["recipe_id"],
        seed=spec["seed"],
        elapsed_seconds=elapsed,
        peak_cuda_bytes=peak,
        traceback_digest=None,
    )


def _stage_path(world):
    run_root = runner._run_root(world.artifact_root, world.bundle["permit_sha256"])
    return run_root / runner.STAGE_NAME


def _failure_summary(world):
    return core._read_json(_stage_path(world) / "summary.json", "summary")


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    torch.manual_seed(0)

    artifact_root = tmp_path / "artifacts"
    project_root = tmp_path / "project"
    repository_root = tmp_path / "repository"
    contract_path = tmp_path / "contract.json"
    permit_path = tmp_path / "permit.json"
    for directory in (artifact_root, project_root, repository_root):
        directory.mkdir(parents=True, exist_ok=True)
    contract_path.write_text("{}", encoding="utf-8")
    permit_path.write_text("{}", encoding="utf-8")

    permit_sha256 = inputs.COMPREHENSIVE_PERMIT_SHA256
    context_id = "ctx-1"
    fitting_role_id = "outer_fit"
    fitting_uids = list(FITTING_UIDS)
    classes = list(CLASSES)
    calibration_slots = list(CALIBRATION_SLOTS)
    source_uid_set_sha256 = core._canon().sha256_value(fitting_uids)
    seeds = list(selection.SEEDS)[:3]

    specs = []
    for seed in seeds:
        for recipe in RECIPES:
            identity = {
                "context_id": context_id,
                "fitting_role_id": fitting_role_id,
                "source_uid_set_sha256": source_uid_set_sha256,
                "recipe_id": recipe,
                "seed": seed,
                "epochs": EPOCHS,
                "calibration_slot_ids": list(calibration_slots),
                "permit_sha256": permit_sha256,
            }
            specs.append(
                {
                    **identity,
                    "refit_id": refit_plan._sha256_canonical(identity),
                    "fitting_uids": list(fitting_uids),
                    "classes": list(classes),
                }
            )
    unique_refits = {spec["refit_id"]: spec for spec in specs}
    aliases = [
        {"refit_id": specs[index % len(specs)]["refit_id"], "strategy_id": "D0-M"}
        for index in range(9)
    ]
    plan = {"plan_id": "plan-1", "unique_refits": unique_refits, "strategy_aliases": aliases}

    support = SimpleNamespace(
        contexts=[{"context_id": context_id, "station": "S1", "held_instrument": "not_applicable"}],
        roles=[
            {
                "context_id": context_id,
                "role": "outer_fit",
                "role_id": fitting_role_id,
                "observation_uid": uid,
                "master_sample_id": f"m-{uid}",
                "instrument": "A",
                "target_analyte": target,
            }
            for uid, target in zip(FITTING_UIDS, CLASSES, strict=True)
        ],
        manifest={
            uid: {
                "master": f"m-{uid}",
                "instrument": "A",
                "target": target,
                "station": "S1",
            }
            for uid, target in zip(FITTING_UIDS, CLASSES, strict=True)
        },
    )
    ledger = {"ledger_id": "ledger-1"}
    bundle = {
        "support": support,
        "ledger": ledger,
        "contract": {"population": {"rows": 3}},
        "p01_path": tmp_path / "p01",
        "repository_root": repository_root,
        "project_root": project_root,
        "artifact_root": artifact_root,
        "permit_sha256": permit_sha256,
        "contract_sha256": "c" * 64,
        "core_plan_id": "core-plan-1",
    }

    selection_receipt = tmp_path / "selection_receipt.json"

    def write_receipt(prior):
        selection_receipt.write_bytes(
            core._canon().canonical_json_bytes({"scientific_seconds_cumulative_bound": prior})
        )

    write_receipt(PRIOR_SECONDS)

    record = {
        "prepare": [],
        "auth": [],
        "calibrate": [],
        "train": [],
        "results": [],
        "persist_calibration": [],
        "groups": [],
    }

    monkeypatch.setattr(runner, "MAXIMUM_REFITS", 9)
    monkeypatch.setattr(freeze, "STRATEGY_ALIAS_COUNT", 9)
    monkeypatch.setattr(freeze, "_paths", lambda bundle: {"selection_receipt": selection_receipt})
    monkeypatch.setattr(core, "_configure_environment", lambda: None)
    monkeypatch.setattr(core, "_capture_provenance", lambda *args: {"marker": "before"})
    monkeypatch.setattr(pilot, "_free_cuda_bytes", lambda torch_module: 2 * 10**10)
    monkeypatch.setattr(pilot, "_enforce_cuda_cap", lambda torch_module, device: None)
    monkeypatch.setattr(pilot, "_checkpoint_preflight", lambda torch_module, artifact: None)
    monkeypatch.setattr(pilot, "_post_run_reauth", lambda *args: {"marker": "after"})

    def fake_prepare(project_root, artifact_root, contract_path, permit_path, require_unstarted):
        record["prepare"].append(require_unstarted)
        return bundle

    monkeypatch.setattr(inputs, "prepare", fake_prepare)

    def fake_auth(bundle, deadline):
        record["auth"].append(deadline)
        return {
            "prior_seconds": PRIOR_SECONDS,
            "plan": plan,
            "source_optimizer_steps": SOURCE_OPTIMIZER_STEPS,
        }

    monkeypatch.setattr(authority, "authenticate_selection", fake_auth)
    monkeypatch.setattr(evidence, "make_logit_loader", lambda bundle: _forbidden_loader)

    def fake_persist_calibration(unit_dir, calibrated, audit, spec):
        record["persist_calibration"].append(spec["refit_id"])

    monkeypatch.setattr(evidence, "persist_calibration", fake_persist_calibration)

    def fake_summarize(spec, result):
        return {"refit_id": spec["refit_id"], "recipe_id": spec["recipe_id"], "seed": spec["seed"]}

    monkeypatch.setattr(evidence, "summarize_result", fake_summarize)
    monkeypatch.setattr(evidence, "cross_instrument_master_count", lambda support, spec: 0)
    monkeypatch.setattr(
        evidence,
        "check_recipe_group",
        lambda bucket, cross_instrument_pairs: record["groups"].append(len(bucket)),
    )

    def fake_calibrate(*, spec, ledger, manifest, load_logits):
        record["calibrate"].append(spec["refit_id"])
        return _temperature_calibration(), {"temperature": 1.0, "classes": list(CLASSES)}

    monkeypatch.setattr(calibration, "calibrate_spec", fake_calibrate)

    def fake_prepare_inputs(bundle, spec):
        return {
            "spec": spec,
            "role_id": spec["fitting_role_id"],
            "recipe": spec["recipe_id"],
            "seed": spec["seed"],
            "epochs": spec["epochs"],
        }

    monkeypatch.setattr(refit_io, "prepare_refit_inputs", fake_prepare_inputs)

    def build_result(spec, **kwargs):
        return _build_result(spec, **kwargs)

    def ordered():
        return sorted(
            specs,
            key=lambda spec: (
                str(spec["context_id"]),
                int(spec["seed"]),
                str(spec["recipe_id"]),
                str(spec["refit_id"]),
            ),
        )

    def run_kwargs(device="cuda"):
        return {
            "project_root": project_root,
            "artifact_root": artifact_root,
            "contract_path": contract_path,
            "permit_path": permit_path,
            "device": device,
        }

    world = SimpleNamespace(
        artifact_root=artifact_root,
        project_root=project_root,
        repository_root=repository_root,
        contract_path=contract_path,
        permit_path=permit_path,
        bundle=bundle,
        support=support,
        ledger=ledger,
        plan=plan,
        specs=specs,
        record=record,
        build_result=build_result,
        ordered=ordered,
        run_kwargs=run_kwargs,
        set_prior=write_receipt,
    )

    def default_handler(spec, on_epoch):
        result = world.build_result(spec)
        for item in result.history:
            on_epoch(dict(item))
        return result

    world.train_handler_default = default_handler
    world.train_handler = default_handler

    def fake_train(**kwargs):
        spec = kwargs.pop("spec")
        on_epoch = kwargs["on_epoch"]
        record["train"].append(spec["refit_id"])
        result = world.train_handler(spec, on_epoch)
        record["results"].append(result)
        return result

    monkeypatch.setattr(refit, "train_refit", fake_train)
    return world


def test_success_orchestration_accounts_and_persists(world):
    receipt = runner.run_refits(**world.run_kwargs())

    assert receipt["status"] == "complete"
    assert len(world.record["train"]) == 6
    assert len(world.record["calibrate"]) == 6
    assert len(world.record["persist_calibration"]) == 6
    assert world.record["prepare"] == [False, False]

    counters = receipt["counters"]
    assert counters["optimizer_steps"] == 720
    assert counters["optimizer_steps_exact"] is True
    assert counters["calibration_started"] == 6
    assert counters["calibration_completed"] == 6
    assert counters["calibration_failed"] == 0
    assert counters["neural_started"] == 6
    assert counters["neural_completed"] == 6
    assert counters["neural_failed"] == 0
    assert counters["sum_fit_elapsed_seconds"] == pytest.approx(0.6)
    assert counters["elapsed_seconds"] > 0.0
    assert receipt["source_optimizer_steps"] == 96
    assert receipt["total_new_optimizer_steps"] == 96 + 720

    stage = _stage_path(world)
    run_root = stage.parent
    pilot._verify_manifest(stage)
    assert (run_root / runner.RECEIPT_NAME).is_file()
    saved_receipt = core._read_json(run_root / runner.RECEIPT_NAME, "receipt")
    assert saved_receipt["stage_manifest_sha256"] == core._canon().sha256_file(
        stage / "manifest.json"
    )
    assert core._read_json(stage / "provenance_before.json", "before") == {"marker": "before"}
    assert core._read_json(stage / "provenance_after.json", "after") == {"marker": "after"}

    for spec in world.specs:
        unit = stage / "units" / spec["refit_id"]
        assert (unit / "lease.json").is_file()
        assert (unit / "summary.json").is_file()
        assert (unit / "terminal.pt").is_file()
        pilot._verify_manifest(unit)

    for result in world.record["results"]:
        for name in ("sampling_digest", "augmentation_digest", "pair_digest"):
            value = getattr(result, name)
            assert isinstance(value, str) and len(value) == 64
        assert result.paired_support["enabled"] == (1 if result.recipe == "D3" else 0)
        assert result.paired_support["available_batches"] == 0
        assert result.paired_support["eligible_masters"] == 0
        assert result.paired_support["pairs"] == 0

    by_seed = {}
    for result in world.record["results"]:
        by_seed.setdefault(result.seed, []).append(result)
    for group in by_seed.values():
        assert len({result.sampling_digest for result in group}) == 1
        assert len({result.augmentation_digest for result in group}) == 1
        assert len({result.pair_digest for result in group}) == 1
        assert len({result.initial_backbone_digest for result in group}) == 1
        assert len({result.initial_state_digest for result in group}) == 2


def test_partial_result_is_persisted_then_fails(world):
    def handler(spec, on_epoch):
        return world.build_result(
            spec,
            status="fail",
            reason_code="deadline_exceeded",
            optimizer_steps=13,
            epochs_completed=3,
            history_epochs=3,
        )

    world.train_handler = handler
    with pytest.raises(refit_io.P05RefitIOError):
        runner.run_refits(**world.run_kwargs())

    first = world.ordered()[0]
    unit = _stage_path(world) / "units" / first["refit_id"]
    saved = core._read_json(unit / "summary.json", "summary")
    assert saved["status"] == "fail"
    assert saved["optimizer_steps"] == 13
    assert (unit / "terminal.pt").is_file()
    assert len(world.record["train"]) == 1

    counters = _failure_summary(world)["counters"]
    assert counters["neural_started"] == 1
    assert counters["neural_failed"] == 1
    assert counters["optimizer_steps"] == 13


def test_kernel_failure_charges_lower_bound_once(world):
    def handler(spec, on_epoch):
        on_epoch(
            {
                "epoch": 1,
                "chemical_ce": 0.5,
                "total_loss": 0.6,
                "supcon_loss": 0.0,
                "paired_loss": 0.0,
                "epoch_optimizer_steps": 4,
                "total_optimizer_steps": 4,
            }
        )
        raise RuntimeError("kernel boom")

    world.train_handler = handler
    with pytest.raises(RuntimeError):
        runner.run_refits(**world.run_kwargs())

    counters = _failure_summary(world)["counters"]
    assert counters["optimizer_steps"] == 4
    assert counters["optimizer_steps_exact"] is False
    assert counters["neural_started"] == 1
    assert counters["neural_failed"] == 1
    assert len(world.record["train"]) == 1
    units = list((_stage_path(world) / "units").iterdir())
    assert len(units) == 1
    assert any(units[0].iterdir())


def test_calibration_failure_skips_neural(world, monkeypatch):
    def boom(*, spec, ledger, manifest, load_logits):
        raise RuntimeError("calibration boom")

    monkeypatch.setattr(calibration, "calibrate_spec", boom)
    with pytest.raises(RuntimeError):
        runner.run_refits(**world.run_kwargs())

    counters = _failure_summary(world)["counters"]
    assert counters["calibration_started"] == 1
    assert counters["calibration_completed"] == 0
    assert counters["calibration_failed"] == 1
    assert counters["neural_started"] == 0
    assert counters["neural_failed"] == 0
    assert world.record["train"] == []


def test_saver_failure_after_kernel_charges_exact_steps(world, monkeypatch):
    def boom(torch_module, run_dir, spec, result):
        raise refit_io.P05RefitIOError("save_failed")

    monkeypatch.setattr(refit_io, "persist_refit_result", boom)
    with pytest.raises(refit_io.P05RefitIOError):
        runner.run_refits(**world.run_kwargs())

    counters = _failure_summary(world)["counters"]
    assert counters["optimizer_steps"] == 120
    assert counters["optimizer_steps_exact"] is True
    assert counters["neural_started"] == 1
    assert counters["neural_failed"] == 1


def test_existing_stage_sentinel_is_not_overwritten(world):
    stage = _stage_path(world)
    stage.mkdir(parents=True)
    sentinel = stage / "sentinel.bin"
    sentinel.write_bytes(b"keep-me")

    with pytest.raises(core.P05CoreError):
        runner.run_refits(**world.run_kwargs())

    assert sentinel.read_bytes() == b"keep-me"


def test_existing_receipt_symlink_rejected_without_stage_write(world):
    run_root = runner._run_root(world.artifact_root, world.bundle["permit_sha256"])
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / runner.RECEIPT_NAME).symlink_to(run_root / "absent.json")

    with pytest.raises(core.P05CoreError):
        runner.run_refits(**world.run_kwargs())

    assert not (run_root / runner.STAGE_NAME).exists()


def test_cpu_device_rejected_before_preparation(world):
    with pytest.raises(runner.P05ComprehensiveRefitError) as info:
        runner.run_refits(**world.run_kwargs(device="cpu"))

    assert info.value.reason_code == "device_invalid"
    assert world.record["prepare"] == []
    assert world.record["auth"] == []


def test_exhausted_clock_reserve_precedes_authentication(world):
    world.set_prior(development.MAXIMUM_TOTAL_SECONDS)

    with pytest.raises(core.P05CoreError):
        runner.run_refits(**world.run_kwargs())

    assert world.record["auth"] == []


def test_source_helper_pins_are_not_mutated(world):
    before = getattr(development, "MAXIMUM_FIT_STEPS", None)

    runner.run_refits(**world.run_kwargs())

    assert getattr(development, "MAXIMUM_FIT_STEPS", None) == before
