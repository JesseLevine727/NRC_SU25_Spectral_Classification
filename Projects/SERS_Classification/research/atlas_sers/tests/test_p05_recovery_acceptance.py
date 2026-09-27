"""Acceptance tests for the post-completion recovery source verifier.

Numerical input preparation, the fixed full-1242 accounting metadata and the
pinned recovery permit/anchor constants are substituted only at the module
boundary so a small synthetic run root can be driven through the real
hash/JSON/manifest/layout/digest checks of ``p05_recovery_acceptance``.
Full-size receipt validation is covered by the dedicated receipt tests.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

import pytest

from atlas_sers.evaluation import p05_recovery_acceptance as mod

TOP_FILES = mod.recovery_inputs.ORIGINAL_TOP_FILES
SLOT_FILES = mod.recovery_inputs.SLOT_EXECUTION_FILES


def _slot_payload(rel):
    if rel.endswith("summary.json"):
        return b'{"optimizer_steps":120,"status":"complete"}'
    return f"bytes:{rel}".encode()


def _write(root, rel, data, records):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    records[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    return records[rel]


def _read_pilot_summary(unit_dir, unit, slot):
    execution = f"{unit['unit_id']}-{slot['slot_id']}"
    return json.loads((unit_dir / "executions" / execution / "summary.json").read_bytes())


def _build_completed(tmp_path, monkeypatch):
    canon = mod.core._canon()
    monkeypatch.setattr(mod.recovery_plan, "SLOTS_PER_UNIT", 12)
    monkeypatch.setattr(mod.recovery_plan, "PARTIAL_COMPLETED", 8)
    monkeypatch.setattr(mod.recovery_plan, "LEASE_COUNT", 21)
    monkeypatch.setattr(mod.recovery_plan, "ORIGINAL_UNSTARTED", 3)
    monkeypatch.setattr(mod.recovery_plan, "BATCH_DRAWS_PER_EPOCH", 4)
    monkeypatch.setattr(mod.recovery_plan, "MINIMUM_FIT_UPDATES", 120)
    monkeypatch.setattr(mod.recovery_plan, "MAXIMUM_FIT_UPDATES", 800)
    monkeypatch.setattr(mod.receipt, "REUSED_ORIGINAL_OPTIMIZER_STEPS", 2400)
    monkeypatch.setattr(mod.source, "DEVELOP_STAGE_NAME", "develop")
    monkeypatch.setattr(mod.source, "RECOVERED_UNIT_COUNT", 1242)
    monkeypatch.setattr(mod, "ANCHOR_FILE_COUNT", 49)
    monkeypatch.setattr(
        mod.pilot, "execution_id", lambda unit, slot: f"{unit['unit_id']}-{slot['slot_id']}"
    )
    monkeypatch.setattr(mod.authority, "validate_recovery_permit", lambda permit: dict(permit))
    monkeypatch.setattr(
        mod.recovery_inputs, "_expected_source_ledger", lambda bundle: {"source": "ledger"}
    )
    monkeypatch.setattr(mod.base_inputs, "_read_pilot_summary", _read_pilot_summary)

    units = [{"unit_id": "A"}, {"unit_id": "B"}]
    slots = []
    for unit_id in ("A", "B"):
        for index in range(12):
            slots.append(
                {
                    "unit_id": unit_id,
                    "slot_id": f"{unit_id}{index}",
                    "recipe_id": f"r{index:02d}",
                    "seed": index,
                }
            )
    ledger = {"ledger_id": "0" * 64, "units": units, "slots": slots}
    bundle = {
        "artifact_root": tmp_path / "artifacts",
        "permit_sha256": "b" * 64,
        "contract_sha256": "c" * 64,
        "core_plan_id": "core-plan",
        "ledger_id": "0" * 64,
        "ledger": ledger,
        "pilot_bundle": {"units": [], "slots": []},
    }
    bundle["artifact_root"].mkdir(parents=True, exist_ok=True)

    b_ids = [f"B{index}" for index in range(12)]
    counts = {
        "source_slots": 24,
        "pilot_reused": 0,
        "original_started": 21,
        "original_completed": 20,
        "original_interrupted": 1,
        "unstarted": 3,
        "recovery_fits": 4,
        "final_new_source_successes": 24,
        "final_new_source_attempts": 25,
    }
    updates = {
        "original_completed_exact": 20 * 120,
        "interrupted_observed_lower_bound": 2,
        "interrupted_charged_upper_bound": 800,
    }
    stored_plan = {
        "plan_id": "a" * 64,
        "sealed_unit_ids": ["A"],
        "incomplete_unit_id": "B",
        "reused_original_slot_ids": [f"A{i}" for i in range(12)] + b_ids[:8],
        "unstarted_slot_ids": b_ids[9:],
        "interrupted_slot_id": b_ids[8],
        "counts": counts,
        "optimizer_updates": updates,
    }
    stored_permit = {
        "final_source_evidence_slots": 24,
        "reused_pilot_slots": 0,
        "original_new_attempts_started": 21,
        "original_new_fits_completed": 20,
        "original_interrupted_attempts": 1,
        "original_unstarted_slots": 3,
        "maximum_recovery_fit_executions": 4,
        "final_successful_new_source_slots": 24,
        "final_new_source_attempts": 25,
        "original_completed_optimizer_updates": 20 * 120,
        "original_interrupted_observed_update_lower_bound": 2,
        "original_interrupted_charged_update_upper_bound": 800,
        "original_sealed_units": 1,
    }
    monkeypatch.setattr(
        mod.authority, "RECOVERY_PERMIT_SHA256", canon.sha256_value(dict(stored_permit))
    )

    run_root = tmp_path / "run"
    original_stage = run_root / "develop"
    recovered_stage = run_root / "recovered"
    lease_root = tmp_path / "leases"
    lease_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod.base_inputs, "_slot_lease_root", lambda artifact: lease_root)

    interrupted_id = b_ids[8]
    interrupted_execution = f"B-{interrupted_id}"
    anchored_history = b'{"step":0}\n{"step":1}\n'
    replayed_history = b'{"step":0}\n{"step":1}\n{"step":2}\n{"step":3}\n'

    original_records = {}
    for unit_id in ("A", "B"):
        for index in range(12):
            slot_id = f"{unit_id}{index}"
            execution = f"{unit_id}-{slot_id}"
            if unit_id == "B" and slot_id not in set(b_ids[:8]) and slot_id != interrupted_id:
                continue
            if slot_id == interrupted_id:
                rel = f"units/B/histories/{execution}.jsonl"
                _write(original_stage, rel, anchored_history, original_records)
                continue
            for name in SLOT_FILES:
                rel = f"units/{unit_id}/executions/{execution}/{name}"
                _write(original_stage, rel, _slot_payload(rel), original_records)
            rel = f"units/{unit_id}/histories/{execution}.jsonl"
            _write(original_stage, rel, b'{"step":0}\n', original_records)

    manifest_a = canon.canonical_json_bytes(
        {
            "files": {
                rel[len("units/A/") :]: record
                for rel, record in original_records.items()
                if rel.startswith("units/A/")
            }
        }
    )
    _write(original_stage, "units/A/manifest.json", manifest_a, original_records)

    original_top = {
        "ledger.json": canon.canonical_json_bytes(ledger),
        "source_ledger.json": canon.canonical_json_bytes({"source": "ledger"}),
        "events.jsonl": (
            json.dumps(
                {
                    "event": "started",
                    "unit_id": "B",
                    "slot_id": interrupted_id,
                    "execution_id": interrupted_execution,
                }
            )
            + "\n"
        ).encode(),
        "selector.jsonl": b"",
        "progress.json": b"{}",
        "input_manifest.json": b"{}",
        "provenance_before.json": b"{}",
    }
    for name, data in original_top.items():
        _write(original_stage, name, data, original_records)

    anchor_files = {name: original_records[name] for name in TOP_FILES}
    anchor_files["units/A/manifest.json"] = original_records["units/A/manifest.json"]
    for rel, record in original_records.items():
        if rel.startswith("units/B/"):
            anchor_files[rel] = record
    stored_anchor = {
        "schema_version": mod.recovery_inputs.INTERRUPTION_ANCHOR_SCHEMA_VERSION,
        "files": anchor_files,
    }
    monkeypatch.setattr(
        mod.authority,
        "ORIGINAL_EVIDENCE_ANCHOR_SHA256",
        canon.sha256_value(stored_anchor),
    )

    recovered_records = {}
    for unit_id in ("A", "B"):
        for index in range(12):
            slot_id = f"{unit_id}{index}"
            execution = f"{unit_id}-{slot_id}"
            for name in SLOT_FILES:
                rel = f"units/{unit_id}/executions/{execution}/{name}"
                _write(recovered_stage, rel, _slot_payload(rel), recovered_records)
            rel = f"units/{unit_id}/histories/{execution}.jsonl"
            data = replayed_history if slot_id == interrupted_id else b'{"step":0}\n'
            _write(recovered_stage, rel, data, recovered_records)
    _write(recovered_stage, "units/A/manifest.json", manifest_a, recovered_records)
    manifest_b = canon.canonical_json_bytes(
        {
            "files": {
                rel[len("units/B/") :]: record
                for rel, record in recovered_records.items()
                if rel.startswith("units/B/")
            }
        }
    )
    _write(recovered_stage, "units/B/manifest.json", manifest_b, recovered_records)

    lease_inventory = {}
    for slot_id in [*stored_plan["reused_original_slot_ids"], interrupted_id]:
        rel = f"{slot_id}/lease.json"
        data = canon.canonical_json_bytes({"slot_id": slot_id})
        path = lease_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        lease_inventory[rel] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    for slot in slots:
        if slot["slot_id"] not in stored_plan["unstarted_slot_ids"]:
            continue
        value = {
            **slot,
            "contract_sha256": bundle["contract_sha256"],
            "core_plan_id": bundle["core_plan_id"],
            "permit_sha256": mod.authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        }
        path = lease_root / slot["slot_id"] / "lease.json"
        path.parent.mkdir()
        path.write_bytes(canon.canonical_json_bytes(value))
    interrupted_slot = next(slot for slot in slots if slot["slot_id"] == interrupted_id)
    replay_lease = {
        "schema_version": mod.outer.persistence.REPLAY_LEASE_SCHEMA_VERSION,
        "attempt_kind": mod.outer.persistence.REPLAY_LEASE_ATTEMPT_KIND,
        "base_comprehensive_permit_sha256": mod.authority.BASECOMPREHENSIVE_PERMIT_SHA256,
        "recovery_permit_sha256": mod.authority.RECOVERY_PERMIT_SHA256,
        "core_contract_sha256": bundle["contract_sha256"],
        "core_plan_id": bundle["core_plan_id"],
        **interrupted_slot,
        "original_lease_sha256": lease_inventory[f"{interrupted_id}/lease.json"]["sha256"],
    }
    (run_root / mod.outer.persistence.REPLAY_LEASE_NAME).write_bytes(
        canon.canonical_json_bytes(replay_lease)
    )

    root_manifest = canon.canonical_json_bytes({"files": {}})
    _write(recovered_stage, "manifest.json", root_manifest, recovered_records)
    unit_manifest_inventory = {
        "A": recovered_records["units/A/manifest.json"],
        "B": recovered_records["units/B/manifest.json"],
    }
    summary = {
        "optimizer_steps": 2880,
        "recovery_optimizer_steps_exact": 480,
        "recovery_completed": 4,
        "recovery_plan_id": stored_plan["plan_id"],
        "original_evidence_anchor_sha256": canon.sha256_value(stored_anchor),
    }
    receipt_record = {"stage": "develop", "payload": "receipt"}
    receipt_path = run_root / mod.receipt.RECEIPT_NAME

    top_data = {
        "events.jsonl": b'{"event":"completed"}\n',
        "selector.jsonl": b"",
        "progress.json": b"{}",
        "ledger.json": canon.canonical_json_bytes(ledger),
        "source_ledger.json": canon.canonical_json_bytes({"source": "ledger"}),
        "plan.json": canon.canonical_json_bytes(stored_plan),
        "original_inventory.json": canon.canonical_json_bytes(original_records),
        "original_anchor.json": canon.canonical_json_bytes(stored_anchor),
        "original_lease_inventory.json": canon.canonical_json_bytes(lease_inventory),
        "recovery_permit.json": canon.canonical_json_bytes(stored_permit),
        "input_manifest.json": canon.canonical_json_bytes({"input": "manifest"}),
        "provenance_before.json": b"{}",
        "provenance_after.json": b"{}",
        "unit_manifest_inventory.json": canon.canonical_json_bytes(unit_manifest_inventory),
        "summary.json": canon.canonical_json_bytes(summary),
    }
    for name, data in top_data.items():
        _write(recovered_stage, name, data, recovered_records)
    mod.core._write_manifest(recovered_stage)
    receipt_record["stage_manifest_sha256"] = canon.sha256_file(recovered_stage / "manifest.json")
    _write(run_root, mod.receipt.RECEIPT_NAME, canon.canonical_json_bytes(receipt_record), {})

    monkeypatch.setattr(
        mod.source,
        "resolve_paths",
        lambda bundle: {
            "run_root": run_root,
            "develop": recovered_stage,
            "receipt": receipt_path,
        },
    )
    monkeypatch.setattr(
        mod.source,
        "accounting_from_recovered",
        lambda summary_value, receipt_value: {"accounted": True},
    )
    monkeypatch.setattr(
        mod.receipt,
        "validate_pair",
        lambda summary_value, receipt_value, units=0: {
            **summary_value,
            "stage_manifest_sha256": receipt_value["stage_manifest_sha256"],
        },
    )
    monkeypatch.setattr(mod.recovery_plan, "build_recovery_plan", lambda **kwargs: stored_plan)
    monkeypatch.setattr(mod.outer, "_check_plan", lambda plan: None)
    monkeypatch.setattr(
        mod.outer, "_input_manifest", lambda bundle, plan, view: {"input": "manifest"}
    )

    return {
        "canon": canon,
        "bundle": bundle,
        "paths": {"run_root": run_root, "develop": recovered_stage, "receipt": receipt_path},
        "run_root": run_root,
        "original_stage": original_stage,
        "stage": recovered_stage,
        "lease_root": lease_root,
        "receipt_path": receipt_path,
        "summary": summary,
        "receipt_record": receipt_record,
        "stored_plan": stored_plan,
        "stored_anchor": stored_anchor,
        "stored_permit": stored_permit,
        "original_records": original_records,
        "anchor_sha": canon.sha256_value(stored_anchor),
    }


def _run_acceptance(fx):
    return mod.authenticate_completed_source(
        fx["bundle"],
        paths=fx["paths"],
        summary=fx["summary"],
        receipt_record=fx["receipt_record"],
        deadline=time.perf_counter() + 60,
    )


def _reseal(fx):
    """Rebind deliberate tampering to test the deeper independent proof."""
    mod.core._write_manifest(fx["stage"])
    fx["receipt_record"]["stage_manifest_sha256"] = fx["canon"].sha256_file(
        fx["stage"] / "manifest.json"
    )
    fx["receipt_path"].write_bytes(fx["canon"].canonical_json_bytes(fx["receipt_record"]))


def _rewrite_recovered_unit_file(fx, rel, data):
    canon = fx["canon"]
    stage = fx["stage"]
    (stage / rel).write_bytes(data)
    unit_id = rel.split("/", 2)[1]
    manifest_path = stage / "units" / unit_id / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][rel.split("/", 2)[2]] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    manifest_bytes = canon.canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    inventory_path = stage / "unit_manifest_inventory.json"
    inventory = json.loads(inventory_path.read_bytes())
    inventory[unit_id] = {
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "size_bytes": len(manifest_bytes),
    }
    inventory_path.write_bytes(canon.canonical_json_bytes(inventory))
    _reseal(fx)


def test_authenticate_completed_source_success(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    assert _run_acceptance(fx) == {"accounted": True}


def test_authenticate_completed_source_is_read_only(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in fx["run_root"].rglob("*") if path.is_file()}
    modules_before = set(sys.modules)
    _run_acceptance(fx)
    after = {path: path.read_bytes() for path in fx["run_root"].rglob("*") if path.is_file()}
    assert after == before
    assert {"torch", "numpy"} & (set(sys.modules) - modules_before) == set()


def test_summary_bytes_mismatch(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    (fx["stage"] / "summary.json").write_bytes(b'{"tampered":true}')
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "summary_mismatch"


def test_receipt_bytes_mismatch_before_scans(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    fx["receipt_path"].write_bytes(b'{"tampered":true}')
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "receipt_changed"


def test_recovery_plan_id_mismatch(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    fx["summary"]["recovery_plan_id"] = "f" * 64
    (fx["stage"] / "summary.json").write_bytes(fx["canon"].canonical_json_bytes(fx["summary"]))
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "recovery_plan_id_mismatch"


def test_anchor_digest_mismatch(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    monkeypatch.setattr(mod.authority, "ORIGINAL_EVIDENCE_ANCHOR_SHA256", "0" * 64)
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "original_anchor_digest_mismatch"


def test_extra_stage_file_rejected(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    (fx["stage"] / "unexpected.json").write_bytes(b"{}")
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "stage_layout_rejected"


def test_extra_unit_directory_rejected(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    (fx["stage"] / "units" / "C").mkdir()
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "unit_directory_mismatch"


def test_unit_manifest_inventory_mismatch(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    path = fx["stage"] / "unit_manifest_inventory.json"
    inventory = json.loads(path.read_bytes())
    inventory["C"] = {"sha256": "0" * 64, "size_bytes": 0}
    path.write_bytes(fx["canon"].canonical_json_bytes(inventory))
    _reseal(fx)
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "unit_manifest_inventory_mismatch"


def test_reused_slot_changed(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    (fx["stage"] / "units" / "B" / "executions" / "B-B0" / "best.pt").write_bytes(b"corrupt")
    _reseal(fx)
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "reused_slot_changed"


def test_missing_lease_rejected(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    (fx["lease_root"] / "B0" / "lease.json").unlink()
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "original_lease_unreadable"


def test_changed_lease_rejected(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    (fx["lease_root"] / "B1" / "lease.json").write_bytes(b'{"slot_id":"B1","extra":1}')
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "original_lease_changed"


def test_replay_prefix_mismatch(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    _rewrite_recovered_unit_file(
        fx,
        "units/B/histories/B-B8.jsonl",
        b'{"step":99}\n{"step":1}\n{"step":2}\n{"step":3}\n',
    )
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "replay_history_prefix_mismatch"


def test_source_ledger_uses_recovery_inputs_api(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    monkeypatch.setattr(mod.recovery_inputs, "_expected_source_ledger", lambda bundle: {"other": 1})
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "source_ledger_mismatch"


def test_undercharged_optimizer_steps_rejected(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    fx["summary"]["optimizer_steps"] = 0
    (fx["stage"] / "summary.json").write_bytes(fx["canon"].canonical_json_bytes(fx["summary"]))
    _reseal(fx)
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "optimizer_steps_total_mismatch"


def test_top_file_changed_during_final_hash(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    original = mod.recovery_inputs._hash_file_record
    target = fx["stage"] / "progress.json"

    def racing(path, deadline):
        if path == target:
            target.write_bytes(b'{"race":true}')
        return original(path, deadline)

    monkeypatch.setattr(mod.recovery_inputs, "_hash_file_record", racing)
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "recovery_evidence_changed"


def test_original_anchor_inventory_mismatch(tmp_path, monkeypatch):
    fx = _build_completed(tmp_path, monkeypatch)
    canon = fx["canon"]
    anchor = json.loads((fx["stage"] / "original_anchor.json").read_bytes())
    anchor["files"]["bogus.json"] = anchor["files"].pop("progress.json")
    (fx["stage"] / "original_anchor.json").write_bytes(canon.canonical_json_bytes(anchor))
    new_sha = canon.sha256_value(anchor)
    monkeypatch.setattr(mod.authority, "ORIGINAL_EVIDENCE_ANCHOR_SHA256", new_sha)
    fx["summary"]["original_evidence_anchor_sha256"] = new_sha
    (fx["stage"] / "summary.json").write_bytes(canon.canonical_json_bytes(fx["summary"]))
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        _run_acceptance(fx)
    assert exc.value.reason_code == "original_anchor_inventory_mismatch"


def test_check_paths_rejects_key_and_value_mismatch(monkeypatch):
    monkeypatch.setattr(mod.source, "resolve_paths", lambda bundle: {"a": 1})
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        mod._check_paths({}, {"b": 2})
    assert exc.value.reason_code == "paths_keys_mismatch"

    monkeypatch.setattr(mod.source, "resolve_paths", lambda bundle: {"a": "x"})
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        mod._check_paths({}, {"a": "y"})
    assert exc.value.reason_code == "paths_mismatch"


def test_index_groups_and_sorts_slots():
    bundle = {
        "ledger": {
            "units": [{"unit_id": "U"}],
            "slots": [
                {"unit_id": "U", "slot_id": "s2", "recipe_id": "r2", "seed": 2},
                {"unit_id": "U", "slot_id": "s1", "recipe_id": "r1", "seed": 1},
            ],
        }
    }
    unit_by_id, slots_by_unit = mod._index(bundle)
    assert set(unit_by_id) == {"U"}
    assert [slot["slot_id"] for slot in slots_by_unit["U"]] == ["s1", "s2"]
    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        mod._index({})
    assert exc.value.reason_code == "base_bundle_malformed"


def test_call_wraps_unexpected_exception():
    def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(mod.RecoveryAcceptanceError) as exc:
        mod._call("wrapped", boom)
    assert exc.value.reason_code == "wrapped"
    assert "kaboom" not in str(exc.value)
