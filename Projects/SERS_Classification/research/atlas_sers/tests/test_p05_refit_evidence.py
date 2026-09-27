"""CPU-only tests for bounded P05 refit evidence helpers."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_refit_evidence as evidence
from atlas_sers.evaluation.classical import TemperatureCalibration


def _digest(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def _fails(code, func, *args, **kwargs):
    with pytest.raises(evidence.P05RefitEvidenceError) as exc:
        func(*args, **kwargs)
    assert exc.value.reason_code == code


@pytest.fixture
def fake_core(monkeypatch):
    # Real canonical serialization, atomic I/O and symlink-chain checks.
    return core


class _FakeInputs:
    loaded = []

    @staticmethod
    def pilot_slot_ids(bundle):
        return ["p1"]

    @staticmethod
    def _pilot_run_dir(artifact_root):
        return Path(artifact_root) / "p05pilot" / "runs" / "pilot"

    @staticmethod
    def _load_logits(numpy, path):
        _FakeInputs.loaded.append(Path(path))
        return {"path": str(path)}


class _FakePilot:
    @staticmethod
    def execution_id(unit, slot):
        return f"exec-{unit['unit_id']}-{slot['slot_id']}"


@pytest.fixture
def loader_env(monkeypatch, tmp_path):
    _FakeInputs.loaded = []
    inputs = types.ModuleType("atlas_sers.evaluation.p05_comprehensive_inputs")
    inputs.pilot_slot_ids = _FakeInputs.pilot_slot_ids
    inputs._pilot_run_dir = _FakeInputs._pilot_run_dir
    inputs._load_logits = _FakeInputs._load_logits
    pilot = types.ModuleType("atlas_sers.evaluation.p05_pilot")
    pilot.execution_id = _FakePilot.execution_id
    monkeypatch.setitem(sys.modules, "atlas_sers.evaluation.p05_comprehensive_inputs", inputs)
    monkeypatch.setitem(sys.modules, "atlas_sers.evaluation.p05_pilot", pilot)
    return tmp_path


def _bundle(root, slot_ids):
    unit = {
        "unit_id": "u1",
        "context_id": "c1",
        "selection_unit_id": "su1",
        "fitting_role_id": "fit",
        "validation_role_id": "val",
    }
    slots = [{**unit, "slot_id": slot_id} for slot_id in slot_ids]
    return {
        "artifact_root": str(root),
        "permit_sha256": "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8",
        "ledger": {"units": [unit], "slots": slots},
    }


def test_logit_loader_uses_pilot_base_without_unit(loader_env):
    bundle = _bundle(loader_env, ["p1"])
    loader = evidence.make_logit_loader(bundle)
    result = loader(bundle["ledger"]["slots"][0])
    expected = (
        loader_env
        / "p05pilot"
        / "runs"
        / "pilot"
        / "executions"
        / "exec-u1-p1"
        / "validation_logits.npz"
    )
    assert _FakeInputs.loaded == [expected]
    assert result == {"path": str(expected)}


def test_logit_loader_uses_develop_base_with_unit(loader_env):
    bundle = _bundle(loader_env, ["d1"])
    loader = evidence.make_logit_loader(bundle)
    loader(bundle["ledger"]["slots"][0])
    expected = (
        loader_env
        / "p05comprehensive"
        / "runs"
        / bundle["permit_sha256"]
        / "develop"
        / "units"
        / "u1"
        / "executions"
        / "exec-u1-d1"
        / "validation_logits.npz"
    )
    assert _FakeInputs.loaded == [expected]


def test_logit_loader_rejects_unknown_and_tampered_before_loading(loader_env):
    bundle = _bundle(loader_env, ["p1"])
    loader = evidence.make_logit_loader(bundle)
    _fails("slot_ledger_mismatch", loader, {"slot_id": "nope", "unit_id": "u1"})
    tampered = dict(bundle["ledger"]["slots"][0])
    tampered["context_id"] = "other"
    _fails("slot_ledger_mismatch", loader, tampered)
    assert _FakeInputs.loaded == []


def test_logit_loader_rejects_symlink(loader_env):
    bundle = _bundle(loader_env, ["p1"])
    target = loader_env / "p05pilot" / "runs" / "pilot" / "executions" / "exec-u1-p1"
    target.mkdir(parents=True)
    (target / "validation_logits.npz").symlink_to(loader_env / "missing.npz")
    loader = evidence.make_logit_loader(bundle)
    with pytest.raises(core.P05CoreError):
        loader(bundle["ledger"]["slots"][0])
    assert _FakeInputs.loaded == []


def _calibration(**overrides):
    values = dict(
        temperature=1.5,
        class_vocabulary=("a", "b", "c"),
        observations=10,
        masters=5,
        fit_observation_uid_sha256="a" * 64,
        fit_master_uid_sha256="b" * 64,
        optimizer_success=True,
        optimizer_objective=0.5,
    )
    values.update(overrides)
    return TemperatureCalibration(**values)


def _cal_audit(cal, **overrides):
    values = dict(
        calibration_state_sha256=cal.state_sha256,
        refit_id="r1",
        context_id="c1",
        recipe_id="D0-M",
        seed=7,
        calibration_slot_ids=["s1"],
        temperature=cal.temperature,
        optimizer_success=True,
        optimizer_objective=cal.optimizer_objective,
    )
    values.update(overrides)
    return values


def _cal_spec(cal, **overrides):
    values = dict(
        refit_id="r1",
        context_id="c1",
        recipe_id="D0-M",
        seed=7,
        classes=list(cal.class_vocabulary),
        calibration_slot_ids=["s1"],
    )
    values.update(overrides)
    return values


def test_persist_calibration_roundtrip_hashes(fake_core, tmp_path):
    cal = _calibration()
    unit_dir = tmp_path / "unit"
    unit_dir.mkdir()
    result = evidence.persist_calibration(unit_dir, cal, _cal_audit(cal), _cal_spec(cal))
    cal_bytes = (unit_dir / "calibration.json").read_bytes()
    audit_bytes = (unit_dir / "calibration_audit.json").read_bytes()
    assert result["state_sha256"] == cal.state_sha256
    assert result["calibration_sha256"] == hashlib.sha256(cal_bytes).hexdigest()
    assert result["audit_sha256"] == hashlib.sha256(audit_bytes).hexdigest()
    stored = json.loads(cal_bytes)
    assert stored["state_sha256"] == cal.state_sha256
    assert stored["state"]["temperature"] == 1.5


def test_persist_calibration_rejects_audit_mismatches(fake_core, tmp_path):
    cal = _calibration()
    unit_dir = tmp_path / "unit"
    unit_dir.mkdir()
    for overrides, code in (
        ({"calibration_state_sha256": "0" * 64}, "calibration_audit_sha_mismatch"),
        ({"refit_id": "other"}, "calibration_audit_refit_id_mismatch"),
        ({"seed": 99}, "calibration_audit_seed_mismatch"),
        ({"temperature": 9.0}, "calibration_audit_temperature_mismatch"),
        ({"optimizer_success": False}, "calibration_audit_status_mismatch"),
    ):
        _fails(
            code,
            evidence.persist_calibration,
            unit_dir,
            cal,
            _cal_audit(cal, **overrides),
            _cal_spec(cal),
        )


def test_persist_calibration_rejects_spec_mismatches(fake_core, tmp_path):
    cal = _calibration()
    unit_dir = tmp_path / "unit"
    unit_dir.mkdir()
    _fails(
        "calibration_classes_mismatch",
        evidence.persist_calibration,
        unit_dir,
        cal,
        _cal_audit(cal),
        _cal_spec(cal, classes=["x", "y"]),
    )
    _fails(
        "calibration_audit_refit_id_mismatch",
        evidence.persist_calibration,
        unit_dir,
        cal,
        _cal_audit(cal),
        _cal_spec(cal, refit_id="other"),
    )


def test_persist_calibration_rejects_existing_and_symlink(fake_core, tmp_path):
    cal = _calibration()
    unit_dir = tmp_path / "unit"
    unit_dir.mkdir()
    (unit_dir / "calibration.json").write_text("{}")
    _fails(
        "calibration_output_exists",
        evidence.persist_calibration,
        unit_dir,
        cal,
        _cal_audit(cal),
        _cal_spec(cal),
    )
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(core.P05CoreError):
        evidence.persist_calibration(link, cal, _cal_audit(cal), _cal_spec(cal))


def test_persist_calibration_rejects_failed_optimizer_and_nonfinite(fake_core, tmp_path):
    unit_dir = tmp_path / "unit"
    unit_dir.mkdir()
    failed = _calibration(optimizer_success=False)
    _fails(
        "calibration_failed",
        evidence.persist_calibration,
        unit_dir,
        failed,
        _cal_audit(failed),
        _cal_spec(failed),
    )
    hot = _calibration(temperature=float("nan"))
    _fails(
        "calibration_temperature_invalid",
        evidence.persist_calibration,
        unit_dir,
        hot,
        _cal_audit(_calibration()),
        _cal_spec(hot),
    )


def test_persist_calibration_detects_persisted_audit_tamper(fake_core, tmp_path, monkeypatch):
    cal = _calibration()
    unit_dir = tmp_path / "unit"
    unit_dir.mkdir()
    original = core._atomic_write

    def tampering(path, data):
        if Path(path).name == "calibration_audit.json":
            payload = json.loads(data)
            payload["seed"] += 1
            data = json.dumps(payload).encode()
        original(path, data)

    monkeypatch.setattr(evidence.core, "_atomic_write", tampering)
    _fails(
        "calibration_audit_reload_mismatch",
        evidence.persist_calibration,
        unit_dir,
        cal,
        _cal_audit(cal),
        _cal_spec(cal),
    )


@pytest.fixture
def refit_io(monkeypatch):
    # The real identity check is torch-free; do not replace it with a weaker fake.
    from atlas_sers.evaluation import p05_refit_io

    return p05_refit_io


def _result(**overrides):
    values = dict(
        refit_id="r1",
        status="complete",
        role_id="fit",
        recipe="D0-M",
        seed=7,
        epochs=2,
        history=[
            {"loss": 0.5, "sampling_digest": _digest("s0"), "weights": [1, 2, 3]},
            {"loss": 0.4, "sampling_digest": _digest("s1"), "weights": [4, 5, 6]},
        ],
        initial_state_digest=_digest("init"),
        initial_backbone_digest=_digest("backbone"),
        terminal_state_digest=_digest("term"),
        paired_support={"available_batches": 0, "pairs": 0},
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _result_spec():
    return {
        "refit_id": "r1",
        "context_id": "c1",
        "fitting_role_id": "fit",
        "recipe_id": "D0-M",
        "seed": 7,
        "epochs": 2,
    }


def test_summarize_result_is_tensor_free(refit_io):
    summary = evidence.summarize_result(_result_spec(), _result())
    assert summary["recipe_id"] == "D0-M"
    assert summary["epochs"] == 2
    assert summary["paired_support"] == {"available_batches": 0, "pairs": 0}
    assert all(set(record) == {"loss", "sampling_digest"} for record in summary["history"])


def test_summarize_result_rejects_incomplete_and_malformed(refit_io):
    spec = _result_spec()
    _fails("result_not_complete", evidence.summarize_result, spec, _result(status="failed"))
    _fails("result_history_length_mismatch", evidence.summarize_result, spec, _result(history=[]))
    _fails(
        "result_initial_state_missing",
        evidence.summarize_result,
        spec,
        _result(initial_state_digest="bad"),
    )


def _history(epochs, *, value=0.5):
    return [
        {
            "sampling_digest": _digest(f"sampling-{i}"),
            "augmentation_digest": _digest(f"augmentation-{i}"),
            "pair_digest": _digest(f"pair-{i}"),
            "loss": value + i,
        }
        for i in range(epochs)
    ]


def _item(
    recipe,
    *,
    epochs=2,
    context="c1",
    seed=7,
    role="outer_fit",
    initial_state=None,
    initial_backbone=None,
    terminal_state=None,
    support=None,
    history=None,
):
    return {
        "recipe_id": recipe,
        "context_id": context,
        "seed": seed,
        "fitting_role_id": role,
        "epochs": epochs,
        "initial_state_digest": initial_state or _digest(f"state-{recipe in ('D1', 'D3')}"),
        "initial_backbone_digest": initial_backbone or _digest("backbone"),
        "terminal_state_digest": terminal_state or _digest(f"terminal-{recipe in ('D1', 'D3')}"),
        "paired_support": support
        if support is not None
        else {
            "enabled": int(recipe in ("D2", "D3")),
            "available_batches": 0,
            "eligible_masters": 0,
            "pairs": 0,
        },
        "history": history if history is not None else _history(epochs),
    }


def test_group_accepts_required_plus_optional_d1():
    out = evidence.check_recipe_group(
        [_item("D0-M"), _item("D1"), _item("D3")], cross_instrument_pairs=0
    )
    assert out == {"recipes": ["D0-M", "D1", "D3"], "epochs_prefix": 2}


def test_group_accepts_optional_d2_with_equivalence():
    state, terminal = _digest("state-shared"), _digest("terminal-shared")
    group = [
        _item("D0-M", initial_state=state, terminal_state=terminal),
        _item("D2", initial_state=state, terminal_state=terminal),
        _item("D3"),
    ]
    assert evidence.check_recipe_group(group, cross_instrument_pairs=0)["recipes"] == [
        "D0-M",
        "D2",
        "D3",
    ]


def test_group_rejects_malformed_shapes():
    def base():
        return [_item("D0-M"), _item("D1"), _item("D3")]

    _fails("group_size", evidence.check_recipe_group, [_item("D0-M")], cross_instrument_pairs=0)
    _fails(
        "group_recipe_invalid",
        evidence.check_recipe_group,
        [_item("D0-M"), _item("X"), _item("D3")],
        cross_instrument_pairs=0,
    )
    _fails(
        "group_required_recipe_missing",
        evidence.check_recipe_group,
        [_item("D0-M"), _item("D1")],
        cross_instrument_pairs=0,
    )
    for mutate, code in (
        (lambda g: g[1].__setitem__("context_id", "c2"), "group_context_mismatch"),
        (lambda g: g[1].__setitem__("seed", 8), "group_seed_mismatch"),
        (lambda g: g[1].__setitem__("fitting_role_id", "other"), "group_role_mismatch"),
        (
            lambda g: g[1].__setitem__("initial_backbone_digest", _digest("other")),
            "group_initial_backbone_mismatch",
        ),
        (lambda g: g[1].__setitem__("initial_state_digest", "xyz"), "group_state_digest_malformed"),
        (lambda g: g[1].pop("initial_state_digest"), "group_state_digest_malformed"),
        (lambda g: g[1]["history"][0].pop("pair_digest"), "group_history_digest_malformed"),
        (lambda g: g[1].__setitem__("history", []), "group_history_length_mismatch"),
        (lambda g: g[1].__setitem__("epochs", 0), "group_epochs_malformed"),
    ):
        group = base()
        mutate(group)
        _fails(code, evidence.check_recipe_group, group, cross_instrument_pairs=0)


@pytest.mark.parametrize("field", ["sampling_digest", "augmentation_digest", "pair_digest"])
def test_group_rejects_each_digest_field_prefix_mismatch(field):
    group = [_item("D0-M"), _item("D1"), _item("D3")]
    group[1]["history"][0][field] = _digest("tampered")
    _fails(
        "group_digest_prefix_mismatch", evidence.check_recipe_group, group, cross_instrument_pairs=0
    )


def test_group_cross_pairs_positive_allows_divergent_terminals():
    group = [
        _item("D0-M", initial_state=_digest("s-a"), terminal_state=_digest("t-a")),
        _item("D2", initial_state=_digest("s-b"), terminal_state=_digest("t-b")),
        _item("D3"),
    ]
    assert evidence.check_recipe_group(group, cross_instrument_pairs=2)["recipes"] == [
        "D0-M",
        "D2",
        "D3",
    ]


def test_group_cross_pairs_zero_requires_equivalent_d0m_d2():
    group = [
        _item("D0-M", initial_state=_digest("s-a"), terminal_state=_digest("t-a")),
        _item("D2", initial_state=_digest("s-b"), terminal_state=_digest("t-b")),
        _item("D3"),
    ]
    _fails(
        "group_equivalent_state_mismatch",
        evidence.check_recipe_group,
        group,
        cross_instrument_pairs=0,
    )


def test_group_unequal_epochs_allows_shared_prefix():
    group = [
        _item("D0-M", epochs=3, initial_state=_digest("s-a"), terminal_state=_digest("t-a")),
        _item("D2", epochs=2, initial_state=_digest("s-b"), terminal_state=_digest("t-b")),
        _item("D3"),
    ]
    assert evidence.check_recipe_group(group, cross_instrument_pairs=0)["epochs_prefix"] == 2


def test_group_d1_d3_share_initial_state_distinct_from_d0m():
    shared, terminal = _digest("state-d1d3"), _digest("terminal-d1d3")
    group = [
        _item("D0-M", initial_state=_digest("state-d0m"), terminal_state=_digest("t-d0m")),
        _item("D1", initial_state=shared, terminal_state=terminal),
        _item("D3", initial_state=shared, terminal_state=terminal),
    ]
    assert evidence.check_recipe_group(group, cross_instrument_pairs=0)["recipes"] == [
        "D0-M",
        "D1",
        "D3",
    ]


def test_group_sparse_support_missing_and_positive_fail():
    missing = _item("D3")
    del missing["paired_support"]
    _fails(
        "group_paired_support_malformed",
        evidence.check_recipe_group,
        [_item("D0-M"), _item("D1"), missing],
        cross_instrument_pairs=0,
    )
    positive = _item(
        "D3", support={"enabled": 1, "available_batches": 0, "eligible_masters": 0, "pairs": 1}
    )
    _fails(
        "group_sparse_pairs_present",
        evidence.check_recipe_group,
        [_item("D0-M"), _item("D1"), positive],
        cross_instrument_pairs=0,
    )


def _support(rows):
    return SimpleNamespace(roles=rows)


def test_cross_instrument_master_count_ignores_other_contexts_and_roles():
    rows = [
        {
            "context_id": "c1",
            "role_id": "fit",
            "role": "outer_fit",
            "master_sample_id": "m1",
            "instrument": "A",
        },
        {
            "context_id": "c1",
            "role_id": "fit",
            "role": "outer_fit",
            "master_sample_id": "m1",
            "instrument": "B",
        },
        {
            "context_id": "c1",
            "role_id": "fit",
            "role": "outer_fit",
            "master_sample_id": "m2",
            "instrument": "A",
        },
        {
            "context_id": "c2",
            "role_id": "fit",
            "role": "outer_fit",
            "master_sample_id": "m3",
            "instrument": "A",
        },
        {
            "context_id": "c2",
            "role_id": "fit",
            "role": "outer_fit",
            "master_sample_id": "m3",
            "instrument": "B",
        },
        {
            "context_id": "c1",
            "role_id": "outer_test",
            "role": "outer_test",
            "master_sample_id": "m4",
            "instrument": "A",
        },
        {
            "context_id": "c1",
            "role_id": "inner",
            "role": "inner",
            "master_sample_id": "m5",
            "instrument": "A",
        },
    ]
    spec = {"context_id": "c1", "fitting_role_id": "fit"}
    assert evidence.cross_instrument_master_count(_support(rows), spec) == 1


def test_cross_instrument_master_count_requires_matching_outer_fit():
    spec = {"context_id": "c1", "fitting_role_id": "fit"}
    _fails("support_outer_fit_empty", evidence.cross_instrument_master_count, _support([]), spec)
    bad = [
        {
            "context_id": "c1",
            "role_id": "fit",
            "role": "outer_test",
            "master_sample_id": "m1",
            "instrument": "A",
        }
    ]
    _fails("support_not_outer_fit", evidence.cross_instrument_master_count, _support(bad), spec)
