"""Synthetic, model-free integration tests for P05 recovery evidence."""

import hashlib
import time
import types

import pytest

from atlas_sers.evaluation import p05_comprehensive_inputs as base_inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_recovery_authority as authority
from atlas_sers.evaluation import p05_recovery_evidence as evidence
from atlas_sers.evaluation import p05_recovery_inputs as inputs
from atlas_sers.evaluation import p05_recovery_plan as recovery_plan


def _future():
    return time.perf_counter() + 3600.0


def _h64(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _ReasonError(Exception):
    def __init__(self, reason_code):
        super().__init__(reason_code)
        self.reason_code = reason_code


def _make_result(slot):
    seed = int(slot["seed"])
    history = [
        {
            "sampling_digest": _h64(f"sampling:{seed}"),
            "augmentation_digest": _h64(f"augmentation:{seed}"),
            "pair_digest": _h64(f"pair:{seed}"),
            "index": index,
        }
        for index in range(4)
    ]
    return types.SimpleNamespace(
        history=history,
        initial_backbone_digest=_h64(f"init:{seed}"),
        optimizer_steps=120,
        seed=seed,
    )


def _slot_rows(unit_id):
    return [
        {
            "unit_id": unit_id,
            "slot_id": f"{recipe}-{seed}",
            "recipe_id": recipe,
            "seed": int(seed),
        }
        for recipe in recovery_plan.RECIPES
        for seed in recovery_plan.SEEDS
    ]


def _prepare(tmp_path, monkeypatch, *, full=True):
    unit_id = "unit-001"
    slots = _slot_rows(unit_id)
    if full:
        completed = list(slots)
        interrupted = "slot-interrupted"
        plan = {
            "sealed_unit_ids": [unit_id],
            "incomplete_unit_id": "",
            "reused_original_slot_ids": [slot["slot_id"] for slot in slots],
            "interrupted_slot_id": interrupted,
        }
    else:
        completed = slots[: recovery_plan.PARTIAL_COMPLETED]
        interrupted = slots[recovery_plan.PARTIAL_COMPLETED]["slot_id"]
        plan = {
            "sealed_unit_ids": [],
            "incomplete_unit_id": unit_id,
            "reused_original_slot_ids": [slot["slot_id"] for slot in completed],
            "interrupted_slot_id": interrupted,
        }
    unit = {"unit_id": unit_id, "auxiliary_support": {"cross_instrument_master_pairs": 0}}
    stage = inputs._original_run_root(str(tmp_path)) / inputs.DEVELOP_STAGE_NAME
    unit_dir = stage / "units" / unit_id
    unit_dir.mkdir(parents=True)
    deadline = _future()
    files = {}
    for slot in completed:
        stem = f"executions/{slot['slot_id']}"
        for name in inputs.SLOT_EXECUTION_FILES:
            files[f"{stem}/{name}"] = f"opaque-{slot['slot_id']}-{name}"
        files[f"histories/{slot['slot_id']}.jsonl"] = f"history-{slot['slot_id']}"
    if not full:
        files[f"histories/{interrupted}.jsonl"] = "interrupted-history"
    inventory = {}
    for relative, text in files.items():
        path = unit_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        inventory[f"units/{unit_id}/{relative}"] = inputs._hash_file_record(path, deadline)
    if full:
        prefix = f"units/{unit_id}/"
        manifest = {"files": {key[len(prefix) :]: record for key, record in inventory.items()}}
        path = unit_dir / "manifest.json"
        path.write_bytes(core._canon().canonical_json_bytes(manifest))
        key = prefix + "manifest.json"
        inventory[key] = inputs._hash_file_record(path, deadline)
        anchor_files = {key: inventory[key]}
    else:
        anchor_files = {key: dict(value) for key, value in inventory.items()}
    anchor = {"schema_version": "synthetic", "files": anchor_files}
    anchor_sha = core._canon().sha256_value(dict(anchor))
    monkeypatch.setattr(authority, "ORIGINAL_EVIDENCE_ANCHOR_SHA256", anchor_sha, raising=False)
    monkeypatch.setattr(authority, "validate_recovery_permit", lambda permit: None, raising=False)
    calls = {
        "guard": [],
        "roles": [],
        "load": [],
        "history": [],
        "accept": [],
        "sparse": [],
        "prefix": [],
        "equiv_unit": [],
        "pairs": [],
    }
    monkeypatch.setattr(
        authority,
        "check_resources",
        lambda torch, phase="fit": calls["guard"].append(phase),
        raising=False,
    )

    def _role_inputs(bundle):
        calls["roles"].append(bundle["units"][0]["unit_id"])
        return {unit_id: types.SimpleNamespace(role="synthetic")}

    monkeypatch.setattr(pilot, "prepare_role_inputs", _role_inputs, raising=False)
    monkeypatch.setattr(pilot, "execution_id", lambda unit, slot: slot["slot_id"], raising=False)
    results = {slot["slot_id"]: _make_result(slot) for slot in slots}

    def _load_result(unit_dir_arg, unit_arg, slot_arg):
        calls["load"].append(slot_arg["slot_id"])
        return results[slot_arg["slot_id"]]

    monkeypatch.setattr(base_inputs, "load_development_result", _load_result, raising=False)
    monkeypatch.setattr(
        base_inputs, "_read_pilot_summary", lambda *a, **k: {"summary": "synthetic"}, raising=False
    )
    monkeypatch.setattr(
        base_inputs,
        "_check_history_matches",
        lambda *a, **k: calls["history"].append("h"),
        raising=False,
    )
    monkeypatch.setattr(
        pilot, "check_completed_result", lambda *a, **k: calls["accept"].append("a"), raising=False
    )
    monkeypatch.setattr(
        pilot, "check_sparse_support", lambda *a, **k: calls["sparse"].append("s"), raising=False
    )
    monkeypatch.setattr(
        pilot, "check_shared_prefixes", lambda *a, **k: calls["prefix"].append("p"), raising=False
    )
    monkeypatch.setattr(
        pilot,
        "check_sparse_equivalences",
        lambda *a, **k: calls["equiv_unit"].append("e"),
        raising=False,
    )

    def _equivalent(left, right):
        assert left.seed == right.seed
        calls["pairs"].append(left.seed)

    monkeypatch.setattr(pilot, "_check_equivalent_results", _equivalent, raising=False)

    def _selector_record(unit_arg, slot_arg, summary_arg):
        return {
            "slot_id": slot_arg["slot_id"],
            "recipe_id": slot_arg["recipe_id"],
            "seed": int(slot_arg["seed"]),
            "unit_id": unit_arg["unit_id"],
            "status": "complete",
        }

    monkeypatch.setattr(base_inputs, "selector_record", _selector_record, raising=False)
    expected_selectors = {
        slot["slot_id"]: {
            "slot_id": slot["slot_id"],
            "recipe_id": slot["recipe_id"],
            "seed": int(slot["seed"]),
            "unit_id": unit_id,
            "status": "complete",
        }
        for slot in completed
    }
    base_bundle = {
        "permit_sha256": authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        "artifact_root": str(tmp_path),
        "ledger": {"units": [dict(unit)], "slots": [dict(slot) for slot in slots]},
        "contract": {},
    }
    bundle = {
        "recovery_permit_sha256": authority.RECOVERY_PERMIT_SHA256,
        "recovery_permit": {"permit": "synthetic"},
        "base_bundle": base_bundle,
        "original_anchor_sha256": anchor_sha,
        "original_anchor": anchor,
        "original_stage": str(stage),
        "original_inventory": inventory,
        "plan": plan,
    }
    return {
        "unit_id": unit_id,
        "slots": slots,
        "completed_slots": completed,
        "interrupted": interrupted,
        "unit": unit,
        "stage": stage,
        "unit_dir": unit_dir,
        "inventory": inventory,
        "bundle": bundle,
        "base_bundle": base_bundle,
        "plan": plan,
        "expected_selectors": expected_selectors,
        "torch": object(),
        "deadline": deadline,
        "device": "cuda",
        "results": results,
        "calls": calls,
    }


def _invoke(env, **overrides):
    kwargs = {
        "recovery_bundle": env["bundle"],
        "unit": env["unit"],
        "expected_selectors": env["expected_selectors"],
        "torch": env["torch"],
        "device": env["device"],
        "deadline": env["deadline"],
    }
    kwargs.update(overrides)
    return evidence.load_verified_original_unit(**kwargs)


def _expect(env, code, **overrides):
    with pytest.raises(evidence.RecoveryEvidenceError) as info:
        _invoke(env, **overrides)
    assert info.value.reason_code == code
    return info.value


def _prove_success(env):
    out = _invoke(env)
    assert out["complete_unit_cross_recipe_checks"] is True
    for bucket in env["calls"].values():
        bucket.clear()
    return out


def _assert_inventory_unchanged(env):
    prefix = f"units/{env['unit_id']}/"
    for key, record in env["inventory"].items():
        observed = inputs._hash_file_record(env["unit_dir"] / key[len(prefix) :], env["deadline"])
        assert observed == record


@pytest.fixture
def full_env(tmp_path, monkeypatch):
    return _prepare(tmp_path, monkeypatch, full=True)


@pytest.fixture
def partial_env(tmp_path, monkeypatch):
    return _prepare(tmp_path, monkeypatch, full=False)


def test_full_success_accepts_all_and_leaves_files(full_env):
    env = full_env
    out = _invoke(env)
    assert len(out["items"]) == recovery_plan.SLOTS_PER_UNIT
    assert len(out["selector_records"]) == recovery_plan.SLOTS_PER_UNIT
    assert set(out["completed_slots"]) == set(env["plan"]["reused_original_slot_ids"])
    assert out["optimizer_updates_exact"] == recovery_plan.SLOTS_PER_UNIT * 120
    assert out["complete_unit_cross_recipe_checks"] is True
    assert out["deferred_until_replay"] is False
    assert out["fits_started"] == 0
    assert out["files_written"] == 0
    assert env["calls"]["prefix"] == ["p"]
    assert env["calls"]["equiv_unit"] == ["e"]
    assert env["calls"]["roles"] == [env["unit_id"]]
    assert len(env["calls"]["load"]) == recovery_plan.SLOTS_PER_UNIT
    _assert_inventory_unchanged(env)


def test_partial_success_defers_full_checks(partial_env):
    env = partial_env
    out = _invoke(env)
    assert len(out["items"]) == recovery_plan.PARTIAL_COMPLETED
    assert set(out["completed_slots"]) == set(env["plan"]["reused_original_slot_ids"])
    assert env["interrupted"] not in out["completed_slots"]
    assert out["deferred_until_replay"] is True
    assert out["complete_unit_cross_recipe_checks"] is False
    assert env["calls"]["prefix"] == []
    assert env["calls"]["equiv_unit"] == []
    assert out["optimizer_updates_exact"] == recovery_plan.PARTIAL_COMPLETED * 120


def test_partial_pair_equivalences_for_available_seeds(partial_env):
    env = partial_env
    _invoke(env)
    by_seed = {}
    for slot in env["completed_slots"]:
        by_seed.setdefault(int(slot["seed"]), set()).add(slot["recipe_id"])
    expected = sorted(
        seed for seed, recipes in by_seed.items() if "D0-M" in recipes and "D2" in recipes
    )
    assert expected
    assert sorted(env["calls"]["pairs"]) == expected


def test_device_cpu_rejected(full_env):
    _expect(full_env, "device_invalid", device="cpu")


def test_expired_deadline_rejected(full_env):
    _expect(full_env, "deadline_exceeded", deadline=-1.0)


def test_stale_checkpoint_failure_is_path_free(full_env, monkeypatch, tmp_path):
    env = full_env
    _prove_success(env)

    def _stale(*args, **kwargs):
        raise _ReasonError("stale_checkpoint")

    monkeypatch.setattr(pilot, "check_completed_result", _stale, raising=False)
    before = len(env["calls"]["load"])
    error = _expect(env, "result_acceptance_failed")
    assert "stale_checkpoint" in str(error)
    assert str(tmp_path) not in str(error)
    assert len(env["calls"]["load"]) > before


def test_malformed_digest_rejected(partial_env):
    env = partial_env
    out = _invoke(env)
    assert out["deferred_until_replay"] is True
    slot_id = env["completed_slots"][0]["slot_id"]
    env["results"][slot_id].history[0]["sampling_digest"] = "nothex"
    _expect(env, "digest_invalid")


def test_host_guard_failure_blocks_loader(full_env, monkeypatch):
    env = full_env
    _prove_success(env)

    def _host_guard(torch, phase="fit"):
        raise authority.RecoveryAuthorityError("host_guard")

    monkeypatch.setattr(authority, "check_resources", _host_guard, raising=False)
    before = len(env["calls"]["load"])
    _expect(env, "resources_host_guard")
    assert len(env["calls"]["load"]) == before


def _case_unit_malformed(env):
    return {"unit": ["not", "a", "mapping"]}


def _case_ledger_malformed(env):
    env["base_bundle"]["ledger"]["units"] = "not-a-list"
    return {}


def _case_seed_not_int(env):
    env["base_bundle"]["ledger"]["slots"][0]["seed"] = "20260805"
    return {}


def _case_foreign_recipe(env):
    env["base_bundle"]["ledger"]["slots"][0]["recipe_id"] = "D9"
    return {}


def _case_not_in_reused(env):
    env["plan"]["reused_original_slot_ids"] = env["plan"]["reused_original_slot_ids"][:-1]
    return {}


def _case_wrong_stage(env):
    env["bundle"]["original_stage"] = str(env["stage"].parent / "elsewhere")
    return {}


def _case_wrong_unit(env):
    return {
        "unit": {
            "unit_id": env["unit_id"],
            "auxiliary_support": {"cross_instrument_master_pairs": 1},
        }
    }


def _case_wrong_anchor(env):
    env["bundle"]["original_anchor"] = {"schema_version": "other", "files": {}}
    return {}


def _case_selector_mismatch(env):
    slot_id = env["completed_slots"][0]["slot_id"]
    env["expected_selectors"][slot_id]["status"] = "failed"
    return {}


def _case_unexpected_root_file(env):
    (env["unit_dir"] / "rogue-manifest.json").write_text("x", encoding="utf-8")
    return {}


def _case_extra_empty_dir(env):
    (env["unit_dir"] / "extra-empty").mkdir()
    return {}


def _case_byte_corruption(env):
    (env["unit_dir"] / "manifest.json").write_text("tampered", encoding="utf-8")
    return {}


_AUTH_CASES = (
    ("unit_malformed", _case_unit_malformed),
    ("ledger_units_malformed", _case_ledger_malformed),
    ("unit_slot_malformed", _case_seed_not_int),
    ("unit_slot_count_mismatch", _case_foreign_recipe),
    ("unit_slot_not_completed", _case_not_in_reused),
    ("original_stage_mismatch", _case_wrong_stage),
    ("unit_identity_mismatch", _case_wrong_unit),
    ("original_anchor_digest_mismatch", _case_wrong_anchor),
    ("selector_status_mismatch", _case_selector_mismatch),
    ("unit_inventory_mismatch", _case_unexpected_root_file),
    ("unit_directory_mismatch", _case_extra_empty_dir),
    ("unit_file_digest_mismatch", _case_byte_corruption),
)


@pytest.mark.parametrize("code,mutate", _AUTH_CASES)
def test_auth_boundary_errors(full_env, code, mutate):
    env = full_env
    _prove_success(env)
    overrides = mutate(env)
    _expect(env, code, **overrides)


@pytest.mark.parametrize("fixture_name", ["full_env", "partial_env"])
def test_inventory_cannot_be_replaced_outside_original_anchor(request, fixture_name):
    env = request.getfixturevalue(fixture_name)
    _invoke(env)
    slot = env["completed_slots"][0]
    relative = f"executions/{slot['slot_id']}/best.pt"
    path = env["unit_dir"] / relative
    path.write_bytes(b"forged-saved-checkpoint")
    env["inventory"][f"units/{env['unit_id']}/{relative}"] = inputs._hash_file_record(
        path, env["deadline"]
    )
    _expect(env, "unit_inventory_anchor_mismatch")
