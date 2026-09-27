"""Synthetic, torch-free boundary tests for the P05 selection freeze stage.

No scientific data, arrays, checkpoints or fits: ``inputs.prepare``, the
private provenance/pilot re-auth boundaries and the deterministic refit-plan
builder are stubbed, while the real ``StorageBudget`` accounting, canonical
JSON writers, manifest write/verify and ``inputs.selector_record``
reconstruction are exercised.  Constants are scaled to 4 units / 48 slots /
12 new fits / 36 reused pilot slots and 1 context / 9 strategy aliases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation.p05_comprehensive_storage import P05StorageError
from atlas_sers.evaluation.p05_core_run import P05CoreError

PERMIT = inputs.COMPREHENSIVE_PERMIT_SHA256
CONTRACT = inputs.CORE_CONTRACT_SHA256
CORE_PLAN = inputs.CORE_PLAN_ID
LEDGER_ID = inputs.LEDGER_ID
PLAN_ID = "f" * 64

NEW_UNITS = 1
PILOT_UNITS = 3
TOTAL_UNITS = NEW_UNITS + PILOT_UNITS
SLOTS_PER_UNIT = 12
TOTAL_SLOTS = TOTAL_UNITS * SLOTS_PER_UNIT
PILOT_SLOTS = PILOT_UNITS * SLOTS_PER_UNIT
NEW_SLOTS = NEW_UNITS * SLOTS_PER_UNIT
CONTEXTS = 1
ALIASES = 9
STATIONS = ("cwa", "pills", "surfaces", "extra")
PRIOR_STAGE_SECONDS = 120.0
PRIOR_CUMULATIVE_SECONDS = PRIOR_STAGE_SECONDS + 3600.0
PROTECTED = {"protected_environment_sha256": "fixed"}


class _Support:
    def __init__(self, contexts, roles):
        self.contexts = contexts
        self.roles = roles


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    core._atomic_write(path, core._canon().canonical_json_bytes(payload))


def _forbidden(label):
    def _fail(*args, **kwargs):
        raise AssertionError(f"{label} must not be called")

    return _fail


def _unit_by_id(units, unit_id):
    for unit in units:
        if unit["unit_id"] == unit_id:
            return unit
    raise KeyError(unit_id)


def _make_units():
    return [
        {
            "unit_id": f"unit-{index}",
            "station": STATIONS[index],
            "context_id": f"ctx-{index}",
            "selection_unit_id": f"sel-{index}",
            "fitting_role_id": f"fit-{index}",
            "validation_role_id": f"val-{index}",
            "fitting_uid_set_sha256": f"{index + 1:064x}",
            "validation_uid_set_sha256": f"{index + 101:064x}",
        }
        for index in range(TOTAL_UNITS)
    ]


def _make_slots(units):
    slots = []
    counter = 0
    for unit in units:
        for recipe in pilot.PILOT_RECIPES:
            for seed in pilot.PILOT_SEEDS:
                slots.append(
                    {
                        "slot_id": f"{counter:064x}",
                        "unit_id": unit["unit_id"],
                        "recipe_id": recipe,
                        "seed": seed,
                        "slot_kind": "inherited",
                        "fitting_role_id": unit["fitting_role_id"],
                        "validation_role_id": unit["validation_role_id"],
                        "excluded_by_protocol": False,
                    }
                )
                counter += 1
    return slots


def _execution_summary(unit, slot):
    return {
        "execution_id": pilot.execution_id(unit, slot),
        "unit_id": unit["unit_id"],
        "slot_id": slot["slot_id"],
        "recipe_id": slot["recipe_id"],
        "seed": int(slot["seed"]),
        "role_id": unit["fitting_role_id"],
        "recipe": slot["recipe_id"],
        "status": "complete",
        "reason_code": None,
        "epochs_completed": 30,
        "optimizer_steps": 120,
        "best_epoch": 5,
        "best_validation_balanced_accuracy": 0.75,
        "best_validation_nll": 0.5,
        "best_validation_macro_f1": 0.7,
        "best_validation_predicted_class_count": 3,
        "best_training_balanced_accuracy": 0.8,
        "elapsed_seconds": 0.25,
        "peak_cuda_bytes": 0,
    }


def _expected_source_ledger(bundle):
    ledger = bundle["ledger"]
    pilot_unit_ids = {unit["unit_id"] for unit in bundle["pilot_bundle"]["units"]}
    by_unit = {}
    for slot in ledger["slots"]:
        by_unit.setdefault(slot["unit_id"], []).append(slot)
    units = []
    for unit in ledger["units"]:
        if unit["unit_id"] in pilot_unit_ids:
            continue
        group = sorted(by_unit[unit["unit_id"]], key=lambda item: (item["recipe_id"], item["seed"]))
        units.append(
            {
                "unit_id": unit["unit_id"],
                "station": unit["station"],
                "fitting_uid_set_sha256": unit["fitting_uid_set_sha256"],
                "validation_uid_set_sha256": unit["validation_uid_set_sha256"],
                "slot_ids": [item["slot_id"] for item in group],
            }
        )
    return {
        "permit_sha256": PERMIT,
        "core_contract_sha256": CONTRACT,
        "core_plan_id": CORE_PLAN,
        "ledger_id": LEDGER_ID,
        "schema_version": development.SCHEMA_VERSION,
        "units": units,
    }


def _dev_summary():
    return {
        "status": "complete",
        "command": "run_development",
        "schema_version": development.SCHEMA_VERSION,
        "protocol_version": development.PROTOCOL_VERSION,
        "permit_sha256": PERMIT,
        "core_contract_sha256": CONTRACT,
        "core_plan_id": CORE_PLAN,
        "ledger_id": LEDGER_ID,
        "device": "cuda",
        "reused_pilot_slots": PILOT_SLOTS,
        "units_total": TOTAL_UNITS,
        "units_completed": NEW_UNITS,
        "started": NEW_SLOTS,
        "completed": NEW_SLOTS,
        "failed": 0,
        "new_completions": NEW_SLOTS,
        "selector_records": TOTAL_SLOTS,
        "optimizer_steps": 96,
        "maximum_optimizer_steps": 9600,
        "sum_elapsed_seconds": 1.0,
        "maximum_peak_cuda_bytes": 0,
        "scientific_seconds_this_stage": PRIOR_STAGE_SECONDS,
        "prelaunch_audit_reserve_seconds": 3600.0,
        "scientific_seconds_cumulative_bound": PRIOR_CUMULATIVE_SECONDS,
        "live_bytes": 0,
        "storage_ceiling_bytes": 107374182400,
        "claim": development.COMPREHENSIVE_CLAIM,
        "source_fits_only": True,
        "selection_authorized": False,
        "refit_authorized": False,
        "calibration_authorized": False,
        "outer_evaluation_authorized": False,
    }


def _receipt_payload(
    manifest_sha, *, this_stage=PRIOR_STAGE_SECONDS, cumulative=PRIOR_CUMULATIVE_SECONDS
):
    return {
        "schema_version": development.SCHEMA_VERSION,
        "protocol_version": development.PROTOCOL_VERSION,
        "stage": "develop",
        "permit_sha256": PERMIT,
        "core_contract_sha256": CONTRACT,
        "core_plan_id": CORE_PLAN,
        "ledger_id": LEDGER_ID,
        "device": "cuda",
        "stage_manifest_sha256": manifest_sha,
        "scientific_seconds_this_stage": this_stage,
        "prelaunch_audit_reserve_seconds": 3600.0,
        "scientific_seconds_cumulative_bound": cumulative,
        "units_completed": NEW_UNITS,
        "new_completions": NEW_SLOTS,
        "selector_records": TOTAL_SLOTS,
        "optimizer_steps": 96,
        "maximum_total_seconds": 172800.0,
        "claim": development.COMPREHENSIVE_CLAIM,
        "source_fits_only": True,
        "outer_evaluation_authorized": False,
    }


def _plan_payload(
    *,
    decisions=CONTEXTS,
    aliases=ALIASES,
    expected=ALIASES,
    unique_count=1,
    unique_map_len=1,
    plan_id=PLAN_ID,
):
    return {
        "decisions": [{"context_id": f"ctx-{index}"} for index in range(decisions)],
        "unique_refits": {f"refit-{index}": {"index": index} for index in range(unique_map_len)},
        "strategy_aliases": [{"index": index} for index in range(aliases)],
        "endpoints": [{"context_id": "ctx-0"}],
        "counts": {
            "context_count": decisions,
            "strategy_alias_count": aliases,
            "expected_strategy_alias_count": expected,
            "unique_refit_count": unique_count,
        },
        "plan_id": plan_id,
    }


def _build_world(tmp_path, monkeypatch, *, plan=None):
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    units = _make_units()
    slots = _make_slots(units)
    pilot_units = units[:PILOT_UNITS]
    pilot_unit_ids = {unit["unit_id"] for unit in pilot_units}
    pilot_slots = [slot for slot in slots if slot["unit_id"] in pilot_unit_ids]
    ledger = {
        "ledger_id": LEDGER_ID,
        "schema_version": "nato-sers-p05-development-ledger-v1",
        "units": units,
        "slots": slots,
    }
    bundle = {
        "project_root": tmp_path,
        "artifact_root": artifact,
        "repository_root": tmp_path,
        "permit": {"pilot_manifest_sha256": inputs.PILOT_MANIFEST_SHA256},
        "permit_sha256": PERMIT,
        "contract": {},
        "contract_sha256": CONTRACT,
        "support": _Support([{"context_id": "ctx-0"}], [{"role_id": "fit-0"}]),
        "p01_path": tmp_path,
        "core_plan": {},
        "core_plan_id": CORE_PLAN,
        "ledger": ledger,
        "units": units,
        "slots": slots,
        "pilot_bundle": {"units": pilot_units, "slots": pilot_slots},
    }
    pilot_run = inputs._pilot_run_dir(artifact)
    for slot in pilot_slots:
        unit = _unit_by_id(units, slot["unit_id"])
        _write_json(
            pilot_run / "executions" / pilot.execution_id(unit, slot) / "summary.json",
            _execution_summary(unit, slot),
        )
    core._write_manifest(pilot_run)
    pilot_manifest_sha = core._canon().sha256_file(pilot_run / "manifest.json")
    monkeypatch.setattr(inputs, "PILOT_MANIFEST_SHA256", pilot_manifest_sha)
    bundle["permit"]["pilot_manifest_sha256"] = pilot_manifest_sha

    run_root = artifact / freeze.NAMESPACE / "runs" / PERMIT
    develop = run_root / "develop"
    develop.mkdir(parents=True)
    _write_json(develop / "ledger.json", ledger)
    _write_json(develop / "source_ledger.json", _expected_source_ledger(bundle))
    _write_json(develop / "provenance_before.json", dict(PROTECTED))
    _write_json(develop / "provenance_after.json", dict(PROTECTED))
    records = []
    for slot in slots:
        unit = _unit_by_id(units, slot["unit_id"])
        records.append(inputs.selector_record(unit, slot, _execution_summary(unit, slot)))
    _write_selector(develop, records)
    new_unit = units[PILOT_UNITS]
    for slot in slots:
        if slot["unit_id"] != new_unit["unit_id"]:
            continue
        _write_json(
            develop
            / "units"
            / new_unit["unit_id"]
            / "executions"
            / pilot.execution_id(new_unit, slot)
            / "summary.json",
            _execution_summary(new_unit, slot),
        )
    _write_json(develop / "summary.json", _dev_summary())
    core._write_manifest(develop)
    manifest_sha = core._canon().sha256_file(develop / "manifest.json")
    _write_json(run_root / "development_receipt.json", _receipt_payload(manifest_sha))

    payload = _plan_payload() if plan is None else plan
    monkeypatch.setattr(inputs, "prepare", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(core, "_capture_provenance", lambda *args, **kwargs: dict(PROTECTED))
    monkeypatch.setattr(pilot, "_post_run_reauth", lambda *args, **kwargs: dict(PROTECTED))
    monkeypatch.setattr(freeze, "build_refit_plan", lambda **kwargs: dict(payload))
    for module, name in (
        (core, "_load_representation"),
        (core, "_noise_frame"),
        (inputs, "_load_logits"),
        (inputs, "import_pilot"),
        (inputs, "load_development_result"),
        (pilot, "train_fit"),
        (pilot, "run"),
        (pilot, "preflight"),
        (pilot, "prepare_role_inputs"),
    ):
        monkeypatch.setattr(module, name, _forbidden(f"{module.__name__}.{name}"))
    return bundle, run_root, develop


def _finalize_develop(run_root, develop):
    core._write_manifest(develop)
    manifest_sha = core._canon().sha256_file(develop / "manifest.json")
    receipt_path = run_root / "development_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["stage_manifest_sha256"] = manifest_sha
    _write_json(receipt_path, receipt)


def _read_selector(develop):
    path = develop / "selector.jsonl"
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_selector(develop, records):
    path = develop / "selector.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for record in records:
            stream.write(core._canon().canonical_json_bytes(record) + b"\n")


def _invoke(bundle, tmp_path):
    return freeze.freeze_selection(
        project_root=tmp_path,
        artifact_root=bundle["artifact_root"],
        contract_path=tmp_path / "contract.json",
        permit_path=tmp_path / "permit.json",
    )


@pytest.fixture
def scaled(monkeypatch):
    monkeypatch.setattr(freeze, "UNIT_COUNT", TOTAL_UNITS)
    monkeypatch.setattr(freeze, "INNER_SLOT_COUNT", TOTAL_SLOTS)
    monkeypatch.setattr(freeze, "MAXIMUM_NEW_FITS", NEW_SLOTS)
    monkeypatch.setattr(freeze, "REUSED_PILOT_SLOTS", PILOT_SLOTS)
    monkeypatch.setattr(freeze, "SLOTS_PER_UNIT", SLOTS_PER_UNIT)
    monkeypatch.setattr(freeze, "CONTEXT_COUNT", CONTEXTS)
    monkeypatch.setattr(freeze, "STRATEGY_ALIAS_COUNT", ALIASES)
    return freeze


def test_real_bounds_and_zero_fits_reporting():
    payload = freeze._base_payload(
        {
            "ledger": {"ledger_id": LEDGER_ID},
            "permit_sha256": PERMIT,
            "contract_sha256": CONTRACT,
            "core_plan_id": CORE_PLAN,
        }
    )
    assert payload["source_new_fits"] == 14904
    assert payload["fits_started"] == 0
    assert payload["selector_records"] == 14940
    assert payload["context_count"] == 320
    assert payload["unit_count"] == 1245
    assert payload["reused_pilot_slots"] == 36
    assert development.MAXIMUM_TOTAL_SECONDS == 172800.0
    assert development.PRELAUNCH_AUDIT_RESERVE_SECONDS == 3600.0
    assert freeze.STRATEGY_ALIAS_COUNT == 2880


def test_freeze_completes_and_binds_artifacts(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    summary = _invoke(bundle, tmp_path)

    assert summary["status"] == "complete"
    assert summary["command"] == "freeze_selection"
    assert summary["fits_started"] == 0
    assert summary["source_new_fits"] == NEW_SLOTS
    assert summary["selector_records"] == TOTAL_SLOTS
    assert summary["context_count"] == CONTEXTS
    assert summary["unit_count"] == TOTAL_UNITS
    assert summary["reused_pilot_slots"] == PILOT_SLOTS
    assert summary["plan_id"] == PLAN_ID
    assert summary["selection_only"] is True
    assert summary["refit_authorized"] is False
    assert summary["outer_evaluation_authorized"] is False
    prior = summary["prior_scientific_seconds_cumulative_bound"]
    assert prior == pytest.approx(PRIOR_CUMULATIVE_SECONDS)
    assert prior == pytest.approx(PRIOR_STAGE_SECONDS + freeze.PRELAUNCH_AUDIT_RESERVE_SECONDS)
    assert summary["scientific_seconds_cumulative_bound"] == pytest.approx(
        prior + summary["scientific_seconds_this_stage"]
    )

    stage = run_root / "selection"
    assert json.loads((stage / "plan.json").read_text(encoding="utf-8")) == _plan_payload()
    bindings = json.loads((stage / "source_bindings.json").read_text(encoding="utf-8"))
    assert bindings["develop_manifest_sha256"] == core._canon().sha256_file(
        develop / "manifest.json"
    )
    assert bindings["source_ledger_sha256"] == core._canon().sha256_file(
        develop / "source_ledger.json"
    )
    assert bindings["selector_sha256"] == core._canon().sha256_file(develop / "selector.jsonl")
    assert bindings["source_new_fits"] == NEW_SLOTS
    assert bindings["fits_started"] == 0
    assert (stage / "provenance_before.json").is_file()
    assert (stage / "provenance_after.json").is_file()
    assert (stage / "summary.json").is_file()
    pilot._verify_manifest(stage)

    receipt = json.loads((run_root / "selection_receipt.json").read_text(encoding="utf-8"))
    assert receipt["stage"] == "selection"
    assert receipt["selection_plan_id"] == PLAN_ID
    assert receipt["selection_manifest_sha256"] == core._canon().sha256_file(
        stage / "manifest.json"
    )
    assert receipt["prior_scientific_seconds_cumulative_bound"] == pytest.approx(
        PRIOR_CUMULATIVE_SECONDS
    )
    assert receipt["scientific_seconds_cumulative_bound"] == pytest.approx(
        receipt["prior_scientific_seconds_cumulative_bound"]
        + receipt["scientific_seconds_this_stage"]
    )
    assert receipt["fits_started"] == 0
    assert receipt["refit_authorized"] is False

    assert (develop / "ledger.json").read_bytes() == core._canon().canonical_json_bytes(
        bundle["ledger"]
    )
    assert (develop / "source_ledger.json").read_bytes() == core._canon().canonical_json_bytes(
        _expected_source_ledger(bundle)
    )
    assert not any(path.suffix in {".pt", ".npz"} for path in stage.rglob("*") if path.is_file())


@pytest.mark.parametrize(
    "override,reason",
    [
        (
            {"scientific_seconds_cumulative_bound": float("nan")},
            "receipt_cumulative_seconds_out_of_range",
        ),
        ({"scientific_seconds_cumulative_bound": -1.0}, "receipt_cumulative_seconds_out_of_range"),
        (
            {
                "scientific_seconds_this_stage": 196400.0,
                "scientific_seconds_cumulative_bound": 200000.0,
            },
            "receipt_cumulative_exceeds_total",
        ),
        ({"scientific_seconds_cumulative_bound": "abc"}, "receipt_cumulative_seconds_malformed"),
        (
            {
                "scientific_seconds_this_stage": 1000.0,
                "scientific_seconds_cumulative_bound": 5000.0,
            },
            "receipt_cumulative_inconsistent",
        ),
        ({"maximum_total_seconds": 100.0}, "receipt_maximum_total_mismatch"),
    ],
)
def test_prior_bound_rejections_before_expensive_verify(
    scaled, monkeypatch, tmp_path, override, reason
):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    receipt_path = run_root / "development_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(override)
    # Deliberately forge noncanonical NaN JSON to test hostile input rejection.
    receipt_path.write_text(json.dumps(receipt, allow_nan=True), encoding="utf-8")

    called = {"manifest": False, "selector": False}

    def manifest_spy(*args, **kwargs):
        called["manifest"] = True
        raise AssertionError("manifest verification must not run")

    def selector_spy(*args, **kwargs):
        called["selector"] = True
        raise AssertionError("selector reconstruction must not run")

    monkeypatch.setattr(freeze, "_verify_develop_manifest", manifest_spy)
    monkeypatch.setattr(freeze, "_authenticate_selector", selector_spy)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == reason
    assert called == {"manifest": False, "selector": False}
    assert not (run_root / "selection").exists()


def test_occupied_selection_stage_cannot_overwrite(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    (run_root / "selection").mkdir()
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "selection_stage_exists"


def test_occupied_selection_receipt_cannot_overwrite(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    (run_root / "selection_receipt.json").write_bytes(b"{}")
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "selection_receipt_exists"
    assert not (run_root / "selection").exists()


def test_develop_manifest_tamper_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    with (develop / "manifest.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "develop_manifest_digest_mismatch"


def test_develop_integrity_tamper_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    _write_json(develop / "provenance_before.json", {"tampered": True})
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "manifest_integrity_mismatch"


def test_pilot_manifest_tamper_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    pilot_run = inputs._pilot_run_dir(bundle["artifact_root"])
    with (pilot_run / "manifest.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "pilot_manifest_digest_mismatch"


def test_ledger_mismatch_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    bundle["ledger"]["schema_version"] = "tampered"
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "develop_ledger_mismatch"


def test_source_ledger_mismatch_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    path = develop / "source_ledger.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["units"][0]["station"] = "tampered"
    _write_json(path, payload)
    _finalize_develop(run_root, develop)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "source_ledger_mismatch"


def test_duplicate_selector_slot_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    records = _read_selector(develop)
    records[1] = dict(records[0])
    _write_selector(develop, records)
    _finalize_develop(run_root, develop)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "selector_slot_duplicate"


def test_missing_selector_record_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    records = _read_selector(develop)
    records.pop()
    _write_selector(develop, records)
    _finalize_develop(run_root, develop)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "selector_count_mismatch"


def test_foreign_selector_slot_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    records = _read_selector(develop)
    records[0]["slot_id"] = "e" * 64
    _write_selector(develop, records)
    _finalize_develop(run_root, develop)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "selector_slot_unknown"


def test_stale_selector_record_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    records = _read_selector(develop)
    records[0]["best_epoch"] = 999
    _write_selector(develop, records)
    _finalize_develop(run_root, develop)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "selector_reconstruction_mismatch"


def test_pilot_summary_missing_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    pilot_run = inputs._pilot_run_dir(bundle["artifact_root"])
    victim = next(iter(pilot_run.glob("executions/*/summary.json")))
    victim.unlink()
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "execution_summary_missing"


def test_pilot_summary_stale_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    pilot_run = inputs._pilot_run_dir(bundle["artifact_root"])
    victim = next(iter(pilot_run.glob("executions/*/summary.json")))
    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["best_epoch"] = 999
    _write_json(victim, payload)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "selector_reconstruction_mismatch"


def test_new_unit_summary_stale_rejected(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    new_unit = bundle["ledger"]["units"][PILOT_UNITS]["unit_id"]
    victim = next(iter((develop / "units" / new_unit / "executions").glob("*/summary.json")))
    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["best_validation_nll"] = 99.0
    _write_json(victim, payload)
    _finalize_develop(run_root, develop)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "selector_reconstruction_mismatch"


@pytest.mark.parametrize(
    "plan,reason",
    [
        (_plan_payload(decisions=2), "refit_decision_count_mismatch"),
        (_plan_payload(aliases=10, expected=10), "refit_alias_count_mismatch"),
        (_plan_payload(aliases=9, expected=8), "refit_alias_count_mismatch"),
        (_plan_payload(unique_count=0, unique_map_len=0), "refit_unique_count_mismatch"),
        (_plan_payload(unique_count=10, unique_map_len=10), "refit_unique_count_mismatch"),
        (_plan_payload(plan_id="zz"), "refit_plan_id_malformed"),
    ],
)
def test_refit_plan_limits(scaled, monkeypatch, tmp_path, plan, reason):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch, plan=plan)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == reason


def test_storage_failure_preserves_failed_stage(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    monkeypatch.setattr(freeze, "STORAGE_CEILING_BYTES", 1)
    with pytest.raises(P05StorageError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "storage_ceiling_exceeded"
    stage = run_root / "selection"
    summary = json.loads((stage / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "fail"
    assert summary["prior_scientific_seconds_cumulative_bound"] == pytest.approx(
        PRIOR_CUMULATIVE_SECONDS
    )
    assert summary["scientific_seconds_cumulative_bound"] >= PRIOR_CUMULATIVE_SECONDS
    assert (stage / "manifest.json").is_file()
    pilot._verify_manifest(stage)


def test_late_deadline_preserves_failed_stage(scaled, monkeypatch, tmp_path):
    bundle, run_root, develop = _build_world(tmp_path, monkeypatch)
    stage = run_root / "selection"
    real_check = freeze._check_deadline

    def late(deadline):
        if stage.exists():
            raise freeze.FreezeSelectionError("global_deadline_exceeded")
        return real_check(deadline)

    monkeypatch.setattr(freeze, "_check_deadline", late)
    with pytest.raises(P05CoreError) as exc:
        _invoke(bundle, tmp_path)
    assert exc.value.reason_code == "global_deadline_exceeded"
    summary = json.loads((stage / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "fail"
    assert summary["prior_scientific_seconds_cumulative_bound"] == pytest.approx(
        PRIOR_CUMULATIVE_SECONDS
    )
    assert summary["scientific_seconds_this_stage"] >= 0.0
    assert summary["scientific_seconds_cumulative_bound"] >= PRIOR_CUMULATIVE_SECONDS
    pilot._verify_manifest(stage)
