import hashlib
import json
import os
import time

import pytest

from atlas_sers.evaluation import p05_recovery_inputs as mod

TOP_FILES = (
    "ledger.json",
    "source_ledger.json",
    "events.jsonl",
    "selector.jsonl",
    "progress.json",
    "input_manifest.json",
    "provenance_before.json",
)
SLOT_FILES = ("best.pt", "terminal.pt", "summary.json", "validation_logits.npz")
SEALED_MANIFEST = "units/A/manifest.json"


class _Canon:
    def canonical_json_bytes(self, value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    def sha256_value(self, value):
        return hashlib.sha256(self.canonical_json_bytes(value)).hexdigest()


def test_check_deadline_accepts_and_rejects():
    mod._check_deadline(time.perf_counter() + 30)
    for bad in (True, False, "1", None, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(mod.RecoveryInputsError):
            mod._check_deadline(bad)
    with pytest.raises(mod.RecoveryInputsError):
        mod._check_deadline(time.perf_counter() - 1)


def test_check_deadline_rejects_huge_int():
    with pytest.raises(mod.RecoveryInputsError):
        mod._check_deadline(10**400)


def test_parse_json_strictness():
    for raw in (
        b'{"a": 1, "a": 2}',
        b'{"a": NaN}',
        b'{"a": Infinity}',
        b'{"a": -Infinity}',
        b"\xff\xfe",
        ("[" * 5000 + "]" * 5000).encode(),
    ):
        with pytest.raises(mod.RecoveryInputsError):
            mod._parse_json(raw, "j")


def test_parse_json_rejects_overflowed_float():
    with pytest.raises(mod.RecoveryInputsError):
        mod._parse_json(b'{"a": 1e999}', "j")


def test_parse_jsonl_strictness_and_records():
    with pytest.raises(mod.RecoveryInputsError):
        mod._parse_jsonl(b'{"a": 1, "a": 2}\n', "j")
    assert mod._parse_jsonl(b'{"a": 1}\n\n{"b": 2}\n', "j") == [{"a": 1}, {"b": 2}]


def test_record_equal_exact_only():
    record = {"sha256": "a" * 64, "size_bytes": 5}
    assert mod._record_equal({"sha256": "a" * 64, "size_bytes": 5}, record)
    for bad in (
        {"sha256": "a" * 64, "size_bytes": 5, "extra": 1},
        {"sha256": "a" * 64, "size_bytes": True},
        {"sha256": "a" * 64},
        "a" * 64,
        None,
        {"sha256": "b" * 64, "size_bytes": 5},
    ):
        assert not mod._record_equal(bad, record)


def test_check_component_rules():
    assert mod._check_component("good-name") == "good-name"
    for bad in ("", ".", "..", "a/b", "a\\b", "a\x00b", "a\x1fb", "a\x7fb", 5):
        with pytest.raises(mod.RecoveryInputsError):
            mod._check_component(bad)


def test_open_regular_boundary(tmp_path):
    regular = tmp_path / "file.bin"
    regular.write_bytes(b"data")
    with mod._open_regular(regular, "code") as handle:
        assert handle.read() == b"data"

    link = tmp_path / "link.bin"
    os.symlink(regular, link)
    with pytest.raises(mod.RecoveryInputsError):
        mod._open_regular(link, "code")

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "inner.bin").write_bytes(b"x")
    linked_dir = tmp_path / "linked"
    os.symlink(real_dir, linked_dir)
    with pytest.raises(mod.RecoveryInputsError):
        mod._open_regular(linked_dir / "inner.bin", "code")

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(mod.RecoveryInputsError):
        mod._open_regular(fifo, "code")


def test_hash_and_bounded_reads(tmp_path):
    path = tmp_path / "data.bin"
    payload = b"payload" * 10
    path.write_bytes(payload)
    assert mod._hash_file_record(path, time.perf_counter() + 30) == {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    assert mod._read_bytes_bounded(path, len(payload), "code", time.perf_counter() + 30) == payload
    with pytest.raises(mod.RecoveryInputsError) as info:
        mod._read_bytes_bounded(path, len(payload) - 1, "code", time.perf_counter() + 30)
    assert info.value.reason_code == "code_too_large"
    with pytest.raises(mod.RecoveryInputsError):
        mod._hash_file_record(path, time.perf_counter() - 1)


def test_collect_tree_boundary(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "manifest.json").write_bytes(b"{}")
    files, directories = mod._collect_tree(tmp_path, time.perf_counter() + 30)
    assert "nested/manifest.json" in files
    assert "nested" in directories

    link_root = tmp_path / "links"
    link_root.mkdir()
    (link_root / "real.bin").write_bytes(b"x")
    os.symlink(link_root / "real.bin", link_root / "alias.bin")
    with pytest.raises(mod.RecoveryInputsError):
        mod._collect_tree(link_root, time.perf_counter() + 30)

    fifo_root = tmp_path / "fifos"
    fifo_root.mkdir()
    os.mkfifo(fifo_root / "pipe")
    with pytest.raises(mod.RecoveryInputsError):
        mod._collect_tree(fifo_root, time.perf_counter() + 30)


def test_error_reason_code_and_path_free():
    with pytest.raises(mod.RecoveryInputsError) as info:
        mod._check_component("..")
    assert info.value.reason_code == "path_component_rejected"
    assert ".." not in str(info.value)


def _write_lease(root, slot_id, payload=None):
    directory = root / slot_id
    directory.mkdir(parents=True)
    body = json.dumps(payload if payload is not None else {"slot_id": slot_id})
    (directory / "lease.json").write_bytes(body.encode())


def test_read_original_leases_success(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.recovery_plan, "LEASE_COUNT", 2)
    root = tmp_path / "leases"
    _write_lease(root, "s1")
    _write_lease(root, "s2")
    records, inventory = mod._read_original_leases(root, time.perf_counter() + 10)
    assert {record["slot_id"] for record in records} == {"s1", "s2"}
    assert set(inventory) == {"s1/lease.json", "s2/lease.json"}


def test_read_original_leases_rejects_count(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.recovery_plan, "LEASE_COUNT", 3)
    root = tmp_path / "leases"
    _write_lease(root, "s1")
    _write_lease(root, "s2")
    with pytest.raises(mod.RecoveryInputsError):
        mod._read_original_leases(root, time.perf_counter() + 10)


def test_read_original_leases_rejects_extra_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.recovery_plan, "LEASE_COUNT", 2)
    root = tmp_path / "leases"
    _write_lease(root, "s1")
    _write_lease(root, "s2")
    (root / "stray.json").write_bytes(b"{}")
    with pytest.raises(mod.RecoveryInputsError):
        mod._read_original_leases(root, time.perf_counter() + 10)


def test_read_original_leases_rejects_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.recovery_plan, "LEASE_COUNT", 2)
    root = tmp_path / "leases"
    _write_lease(root, "s1")
    _write_lease(root, "s2", {"slot_id": "other"})
    with pytest.raises(mod.RecoveryInputsError):
        mod._read_original_leases(root, time.perf_counter() + 10)


def test_read_original_leases_rejects_link(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.recovery_plan, "LEASE_COUNT", 2)
    root = tmp_path / "leases"
    _write_lease(root, "s1")
    os.symlink(root / "s1", root / "s2")
    with pytest.raises(mod.RecoveryInputsError):
        mod._read_original_leases(root, time.perf_counter() + 10)


def test_authenticate_pilot_passes_permit_not_contract(monkeypatch):
    seen = {}
    monkeypatch.setattr(mod.base_inputs, "_pilot_run_dir", lambda root: root)
    monkeypatch.setattr(
        mod.base_inputs,
        "_check_pilot_run_manifest",
        lambda run_dir, arg: seen.setdefault("manifest", arg),
    )
    monkeypatch.setattr(mod.base_inputs, "_check_pilot_run_summary", lambda run_dir: None)
    monkeypatch.setattr(mod.base_inputs, "_check_pilot_slot_leases", lambda bundle: None)
    monkeypatch.setattr(mod, "_verify_pilot_inventory", lambda *_args: None)
    bundle = {
        "artifact_root": "/root",
        "permit": {"token": "permit"},
        "contract": {"token": "contract"},
    }
    mod._authenticate_pilot(bundle, time.perf_counter() + 10)
    assert seen["manifest"] == {"token": "permit"}


def _build(tmp_path, monkeypatch):
    canon = _Canon()
    monkeypatch.setattr(mod.core, "_canon", lambda: canon)
    monkeypatch.setattr(
        mod.pilot,
        "execution_id",
        lambda unit, slot: f"{unit['unit_id']}-{slot['slot_id']}",
    )
    monkeypatch.setattr(mod, "_authenticate_pilot", lambda bundle, deadline: None)
    monkeypatch.setattr(mod, "_read_original_leases", lambda root, deadline: ([], {}))
    monkeypatch.setattr(mod, "_expected_source_ledger", lambda bundle: {"reproduced": True})
    monkeypatch.setattr(mod, "_check_plan_against_permit", lambda plan, permit: None)
    monkeypatch.setattr(
        mod.authority, "validate_recovery_permit", lambda permit: dict(permit), raising=False
    )
    monkeypatch.setattr(mod.authority, "BASECOMPREHENSIVE_PERMIT_SHA256", "b" * 64, raising=False)

    artifact_root = tmp_path / "artifacts"
    lease_root = tmp_path / "leases"
    lease_root.mkdir()
    monkeypatch.setattr(mod.base_inputs, "_slot_lease_root", lambda _artifact: lease_root)
    run_root = artifact_root / mod.COMPREHENSIVE_NAMESPACE / "runs" / ("b" * 64)
    stage = run_root / mod.DEVELOP_STAGE_NAME

    units = []
    slots = []
    for unit_id in ("A", "B"):
        units.append({"unit_id": unit_id})
        for index in range(12):
            slots.append(
                {
                    "unit_id": unit_id,
                    "slot_id": f"{unit_id}{index}",
                    "recipe_id": f"r{index:02d}",
                    "seed": index,
                }
            )
    ledger = {"units": units, "slots": slots}
    bundle = {
        "permit_sha256": "b" * 64,
        "artifact_root": artifact_root,
        "ledger": ledger,
        "pilot_bundle": {"units": [], "slots": []},
    }
    b_ids = [f"B{index}" for index in range(12)]
    plan = {
        "sealed_unit_ids": ["A"],
        "incomplete_unit_id": "B",
        "reused_original_slot_ids": b_ids[:8],
        "interrupted_slot_id": b_ids[8],
        "execution_authorized": False,
        "fits_started": 0,
    }
    monkeypatch.setattr(mod.recovery_plan, "build_recovery_plan", lambda **kwargs: plan)

    records = {}

    def write(relative, data):
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        records[relative] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
        return records[relative]

    for unit_id in ("A", "B"):
        for index in range(12):
            slot_id = f"{unit_id}{index}"
            execution = f"{unit_id}-{slot_id}"
            completed = unit_id == "A" or slot_id in plan["reused_original_slot_ids"]
            if completed:
                for name in SLOT_FILES:
                    rel = f"units/{unit_id}/executions/{execution}/{name}"
                    write(rel, rel.encode())
                rel = f"units/{unit_id}/histories/{execution}.jsonl"
                write(rel, b'{"step": 0}\n')
            elif slot_id == plan["interrupted_slot_id"]:
                rel = f"units/{unit_id}/histories/{execution}.jsonl"
                write(rel, b'{"step": 0}\n')

    manifest_files = {
        rel.split("/", 2)[2]: record
        for rel, record in records.items()
        if rel.startswith("units/A/")
    }
    write(SEALED_MANIFEST, canon.canonical_json_bytes({"files": manifest_files}))

    top_bytes = {
        "ledger.json": canon.canonical_json_bytes(ledger),
        "source_ledger.json": canon.canonical_json_bytes({"reproduced": True}),
        "events.jsonl": (
            json.dumps(
                {
                    "event": "started",
                    "unit_id": "B",
                    "slot_id": plan["interrupted_slot_id"],
                    "execution_id": f"B-{plan['interrupted_slot_id']}",
                }
            )
            + "\n"
        ).encode(),
        "selector.jsonl": b"",
        "progress.json": b"{}",
        "input_manifest.json": b"{}",
        "provenance_before.json": b"{}",
    }
    for name, data in top_bytes.items():
        write(name, data)

    anchor_entries = {name: records[name] for name in TOP_FILES}
    anchor_entries[SEALED_MANIFEST] = records[SEALED_MANIFEST]
    for rel in sorted(records):
        if rel.startswith("units/B/"):
            anchor_entries[rel] = records[rel]
    anchor = {
        "schema_version": mod.INTERRUPTION_ANCHOR_SCHEMA_VERSION,
        "files": anchor_entries,
    }
    anchor_sha = canon.sha256_value(anchor)
    monkeypatch.setattr(mod.authority, "ORIGINAL_EVIDENCE_ANCHOR_SHA256", anchor_sha, raising=False)
    permit = {
        "original_events_sha256": hashlib.sha256(top_bytes["events.jsonl"]).hexdigest(),
        "original_selector_sha256": hashlib.sha256(top_bytes["selector.jsonl"]).hexdigest(),
        "original_evidence_anchor_files": len(anchor_entries),
        "original_evidence_anchor_sha256": anchor_sha,
    }
    return {
        "stage": stage,
        "run_root": run_root,
        "bundle": bundle,
        "plan": plan,
        "permit": permit,
        "anchor_sha": anchor_sha,
    }


def test_expected_layout_exact(tmp_path, monkeypatch):
    fx = _build(tmp_path, monkeypatch)
    files, directories, manifests, sealed_files, unsealed = mod._expected_layout(
        fx["bundle"], fx["plan"]
    )
    assert len(files) == 7 + 61 + 41
    assert len(directories) == 27
    assert manifests == {SEALED_MANIFEST}
    assert len(sealed_files["A"]) == 60
    assert len(unsealed) == 41


def test_authenticate_original_success(tmp_path, monkeypatch):
    fx = _build(tmp_path, monkeypatch)
    result = mod.authenticate_original_stage(fx["bundle"], fx["permit"], time.perf_counter() + 60)
    assert result["original_run_root"] == fx["run_root"]
    assert result["original_anchor_sha256"] == fx["anchor_sha"]
    assert result["inventory_total_bytes"] > 0


def test_authenticate_original_stage_calls_validate_recovery_permit(tmp_path, monkeypatch):
    fx = _build(tmp_path, monkeypatch)
    seen = {}
    monkeypatch.setattr(
        mod.authority,
        "validate_recovery_permit",
        lambda permit: seen.setdefault("permit", permit),
        raising=False,
    )
    mod.authenticate_original_stage(fx["bundle"], fx["permit"], time.perf_counter() + 60)
    assert seen.get("permit") is fx["permit"]


def _apply_mutation(fx, mutation, monkeypatch):
    stage = fx["stage"]
    if mutation == "missing_file":
        (stage / "progress.json").unlink()
    elif mutation == "extra_empty_dir":
        (stage / "units" / "empty").mkdir()
    elif mutation == "extra_file":
        (stage / "unexpected.json").write_bytes(b"{}")
    elif mutation == "nested_manifest":
        path = stage / SEALED_MANIFEST
        data = json.loads(path.read_text())
        data["files"]["extra/entry.pt"] = {"sha256": "0" * 64, "size_bytes": 1}
        path.write_bytes(json.dumps(data).encode())
    elif mutation == "manifest_entry":
        path = stage / SEALED_MANIFEST
        data = json.loads(path.read_text())
        data["files"][sorted(data["files"])[0]]["sha256"] = "0" * 64
        path.write_bytes(json.dumps(data).encode())
    elif mutation == "checkpoint_byte":
        (stage / "units/A/executions/A-A0/best.pt").write_bytes(b"corrupt")
    elif mutation == "source_ledger":
        (stage / "source_ledger.json").write_bytes(b'{"reproduced": false}')
    elif mutation == "raw_journal":
        (stage / "events.jsonl").write_bytes(b'{"event": "other"}\n')
    elif mutation == "anchor":
        monkeypatch.setattr(
            mod.authority, "ORIGINAL_EVIDENCE_ANCHOR_SHA256", "0" * 64, raising=False
        )
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_file",
        "extra_empty_dir",
        "extra_file",
        "nested_manifest",
        "manifest_entry",
        "checkpoint_byte",
        "source_ledger",
        "raw_journal",
        "anchor",
    ],
)
def test_authenticate_original_rejections(tmp_path, monkeypatch, mutation):
    fx = _build(tmp_path, monkeypatch)
    _apply_mutation(fx, mutation, monkeypatch)
    with pytest.raises(mod.RecoveryInputsError) as caught:
        mod.authenticate_original_stage(fx["bundle"], fx["permit"], time.perf_counter() + 60)
    expected = {
        "missing_file": "original_progress.json",
        "extra_empty_dir": "original_stage_directory_mismatch",
        "extra_file": "original_stage_inventory_mismatch",
        "nested_manifest": "original_unit_manifest_inventory_mismatch",
        "manifest_entry": "original_unit_manifest_digest_mismatch",
        "checkpoint_byte": "original_unit_manifest_digest_mismatch",
        "source_ledger": "original_source_ledger_mismatch",
        "raw_journal": "original_events_digest_mismatch",
        "anchor": "original_anchor_digest_mismatch",
    }
    assert caught.value.reason_code == expected[mutation]


def test_changed_evidence_during_second_hash_pass(tmp_path, monkeypatch):
    fx = _build(tmp_path, monkeypatch)
    original = mod._hash_file_record
    target = fx["stage"] / "units/A/executions/A-A0/best.pt"
    visits = 0

    def changing(path, deadline):
        nonlocal visits
        if path == target:
            visits += 1
            if visits == 2:
                target.write_bytes(b"changed-after-first-pass")
        return original(path, deadline)

    monkeypatch.setattr(mod, "_hash_file_record", changing)
    with pytest.raises(mod.RecoveryInputsError, match="original_evidence_changed"):
        mod.authenticate_original_stage(fx["bundle"], fx["permit"], time.perf_counter() + 60)


def test_change_between_parse_and_hash_is_rejected(tmp_path, monkeypatch):
    fx = _build(tmp_path, monkeypatch)
    original = mod._hash_file_record
    target = fx["stage"] / "progress.json"

    def changing(path, deadline):
        if path == target:
            target.write_bytes(b'{"changed":true}')
        return original(path, deadline)

    monkeypatch.setattr(mod, "_hash_file_record", changing)
    with pytest.raises(mod.RecoveryInputsError, match="original_evidence_changed"):
        mod.authenticate_original_stage(fx["bundle"], fx["permit"], time.perf_counter() + 60)


@pytest.mark.parametrize("kind", ["nested_manifest", "extra_directory", "changed_bytes", "symlink"])
def test_strict_pilot_inventory_rejects_mutation(tmp_path, monkeypatch, kind):
    payload = b"opaque-weights"
    data = tmp_path / "weights.pt"
    data.write_bytes(payload)
    raw = json.dumps(
        {
            "files": {
                "weights.pt": {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            }
        }
    ).encode()
    (tmp_path / "manifest.json").write_bytes(raw)
    monkeypatch.setattr(mod.base_inputs, "PILOT_MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())
    mod._verify_pilot_inventory(tmp_path, time.perf_counter() + 30)
    if kind == "nested_manifest":
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested/manifest.json").write_bytes(b"{}")
    elif kind == "extra_directory":
        (tmp_path / "empty").mkdir()
    elif kind == "changed_bytes":
        data.write_bytes(b"other")
    else:
        (tmp_path / "alias.pt").symlink_to(data)
    with pytest.raises(mod.RecoveryInputsError):
        mod._verify_pilot_inventory(tmp_path, time.perf_counter() + 30)


def test_standalone_helper_rejects_bad_permit_before_reading(monkeypatch):
    def reject(_permit):
        raise mod.authority.RecoveryAuthorityError("permit_digest_mismatch")

    monkeypatch.setattr(mod.authority, "validate_recovery_permit", reject)
    with pytest.raises(mod.RecoveryInputsError, match="recovery_permit_rejected"):
        mod.authenticate_original_stage({}, {}, time.perf_counter() + 30)


def test_root_exclusivity_is_checked(tmp_path):
    (tmp_path / "develop").mkdir()
    mod._assert_root_exclusive(tmp_path)
    (tmp_path / "selection").mkdir()
    with pytest.raises(mod.RecoveryInputsError, match="original_run_root_not_exclusive"):
        mod._assert_root_exclusive(tmp_path)


def test_prepare_recovery_orchestration(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(mod.authority, "load_recovery_permit", lambda path: {"p": 1})
    monkeypatch.setattr(mod.authority, "BASECOMPREHENSIVE_PERMIT_SHA256", "b" * 64, raising=False)
    monkeypatch.setattr(mod.authority, "RECOVERY_PERMIT_SHA256", "r" * 64, raising=False)

    def fake_prepare(**kwargs):
        captured["prepare"] = kwargs
        return {"permit_sha256": "b" * 64, "artifact_root": tmp_path}

    monkeypatch.setattr(mod.base_inputs, "prepare", fake_prepare)
    monkeypatch.setattr(mod, "_original_run_root", lambda root: tmp_path / "run")
    monkeypatch.setattr(mod, "_assert_root_exclusive", lambda root: None)
    monkeypatch.setattr(
        mod,
        "authenticate_original_stage",
        lambda bundle, permit, deadline: {"original_run_root": tmp_path / "run"},
    )
    result = mod.prepare_recovery(
        project_root=tmp_path,
        artifact_root=tmp_path,
        contract_path=tmp_path / "contract.json",
        base_permit_path=tmp_path / "base.json",
        recovery_permit_path=tmp_path / "recovery.json",
        deadline=time.perf_counter() + 60,
    )
    assert captured["prepare"]["require_unstarted"] is False
    assert result["fits_started"] == 0
    assert result["files_written"] == 0
    assert result["scientific_model_reauthentication_complete"] is False
    assert result["recovery_permit_sha256"] == "r" * 64


def test_prepare_recovery_rejects_base_permit_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(mod.authority, "load_recovery_permit", lambda path: {})
    monkeypatch.setattr(mod.authority, "BASECOMPREHENSIVE_PERMIT_SHA256", "b" * 64, raising=False)
    monkeypatch.setattr(mod.base_inputs, "prepare", lambda **kwargs: {"permit_sha256": "x" * 64})
    with pytest.raises(mod.RecoveryInputsError) as info:
        mod.prepare_recovery(
            project_root=tmp_path,
            artifact_root=tmp_path,
            contract_path=tmp_path / "c",
            base_permit_path=tmp_path / "bp",
            recovery_permit_path=tmp_path / "rp",
            deadline=time.perf_counter() + 10,
        )
    assert info.value.reason_code == "base_permit_hash_mismatch"
