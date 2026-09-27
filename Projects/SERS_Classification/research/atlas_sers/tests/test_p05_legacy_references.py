"""CPU-only read-only tests for the P05 legacy-reference loader.

These tests never read private scientific data or touch a GPU.  They build a
temporary artifact root containing genuine P03/P04 pinned shard directories,
descriptor / report / validation records and ``_STATE.json`` inventories, then
drive the real
:func:`atlas_sers.evaluation.p05_legacy_references.load_references` against
real parquet IO.  ``pandas.read_parquet`` is wrapped (not replaced) so the
actual hash gate is exercised while the tests can observe and mutate reads.
"""

from __future__ import annotations

# The optional torch import must precede torch-dependent project modules.
# ruff: noqa: E402
import hashlib
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p05_comprehensive_evaluation as evaluation
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_legacy_references as legacy

P03_PROTECTED = legacy.P03_EXECUTION_ID + "0" * 40
PERMIT_SHA256 = "a" * 64
DEADLINE = time.monotonic() + 10**6
_UNSET = object()


def _sha(path: Path) -> str:
    return core._canon().sha256_file(Path(path))


def _write_json(path: Path, value) -> None:
    core._atomic_write(Path(path), core._canon().canonical_json_bytes(value))


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def _snapshot(root: Path) -> dict:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _state(shard_dir: Path) -> dict:
    return _read_json(shard_dir / legacy.STATE_NAME)


def _write_state(shard_dir: Path, state) -> None:
    _write_json(shard_dir / legacy.STATE_NAME, state)


def _set_state(shard_dir: Path, **changes) -> None:
    state = _state(shard_dir)
    state.update(changes)
    _write_state(shard_dir, state)


def _set_state_files(shard_dir: Path, mutate) -> None:
    state = _state(shard_dir)
    state["files"] = mutate(dict(state["files"]))
    _write_state(shard_dir, state)


def _write_shard_json(shard_dir: Path, name: str, value) -> None:
    _write_json(shard_dir / name, value)
    state = _state(shard_dir)
    state["files"][name] = _sha(shard_dir / name)
    _write_state(shard_dir, state)


def _write_shard_raw(shard_dir: Path, name: str, raw: bytes) -> None:
    (shard_dir / name).write_bytes(raw)
    state = _state(shard_dir)
    state["files"][name] = _sha(shard_dir / name)
    _write_state(shard_dir, state)


def _build_p03_shard(artifact_root: Path, frame: pd.DataFrame) -> Path:
    shard_dir = (
        artifact_root
        / "p03"
        / "runs"
        / legacy.P03_RUN_ID
        / legacy.FINAL_AGGREGATION_DIR
        / legacy.SHARDS_DIR
        / legacy.SHARD_NAME
    )
    shard_dir.mkdir(parents=True)
    _write_parquet(shard_dir / legacy.P03_PREDICTIONS_NAME, frame)
    _write_json(
        shard_dir / legacy.P03_DESCRIPTOR_NAME,
        {
            "schema_version": legacy.P03_DESCRIPTOR_SCHEMA,
            "execution_run_id": legacy.P03_RUN_ID,
            "protected_state_sha256": P03_PROTECTED,
        },
    )
    _write_json(shard_dir / legacy.P03_VALIDATION_NAME, {"status": "pass"})
    files = {
        name: _sha(shard_dir / name)
        for name in (
            legacy.P03_PREDICTIONS_NAME,
            legacy.P03_DESCRIPTOR_NAME,
            legacy.P03_VALIDATION_NAME,
        )
    }
    _write_json(
        shard_dir / legacy.STATE_NAME,
        {
            "schema_version": legacy.SHARD_STATE_SCHEMA,
            "execution_status": "complete",
            "shard_id": legacy.SHARD_ID,
            "protected_state_sha256": P03_PROTECTED,
            "files": files,
        },
    )
    return shard_dir


def _build_p04_shard(artifact_root: Path, frame: pd.DataFrame) -> Path:
    shard_dir = (
        artifact_root
        / "p04"
        / "runs"
        / legacy.P04_RUN_ID
        / legacy.FINAL_AGGREGATION_DIR
        / legacy.SHARDS_DIR
        / legacy.SHARD_NAME
    )
    shard_dir.mkdir(parents=True)
    _write_parquet(shard_dir / legacy.P04_PREDICTIONS_NAME, frame)
    _write_json(
        shard_dir / legacy.P04_REPORT_NAME,
        {
            "schema_version": legacy.P04_REPORT_SCHEMA,
            "status": "pass",
            "run_id": legacy.P04_RUN_ID,
            "protected_state_sha256": legacy.P04_PROTECTED_STATE_SHA256,
            "aggregation_state_sha256": legacy.P04_AGGREGATION_STATE_SHA256,
        },
    )
    files = {
        name: _sha(shard_dir / name)
        for name in (legacy.P04_PREDICTIONS_NAME, legacy.P04_REPORT_NAME)
    }
    _write_json(
        shard_dir / legacy.STATE_NAME,
        {
            "schema_version": legacy.SHARD_STATE_SCHEMA,
            "execution_status": "complete",
            "shard_id": legacy.SHARD_ID,
            "protected_state_sha256": legacy.P04_AGGREGATION_STATE_SHA256,
            "files": files,
        },
    )
    return shard_dir


def _build_world(tmp_path: Path) -> SimpleNamespace:
    artifact_root = tmp_path / "artifacts"
    run_root = artifact_root / evaluation.COMPREHENSIVE_DIR / evaluation.RUNS_DIR / PERMIT_SHA256
    stage = run_root / evaluation.STAGE_NAME
    stage.mkdir(parents=True)
    manifest_path = stage / evaluation.MANIFEST_NAME
    manifest_path.write_bytes(b'{"stage": "manifest"}\n')
    manifest_sha256 = _sha(manifest_path)
    receipt = {
        "status": "complete",
        "predictions_frozen": True,
        "predictions_complete": True,
        "all_complete": True,
        "permit_sha256": PERMIT_SHA256,
        "stage_manifest_sha256": manifest_sha256,
    }
    receipt_path = run_root / evaluation.RECEIPT_NAME
    _write_json(receipt_path, receipt)

    p03_frame = pd.DataFrame({"observation_uid": ["p03-uid-1", "p03-uid-2"], "score": [0.25, 0.75]})
    p04_frame = pd.DataFrame(
        {
            "observation_uid": ["p04-uid-1", "p04-uid-2", "p04-uid-3"],
            "score": [0.1, 0.5, 0.9],
        }
    )
    p03_dir = _build_p03_shard(artifact_root, p03_frame)
    p04_dir = _build_p04_shard(artifact_root, p04_frame)

    return SimpleNamespace(
        tmp=tmp_path,
        artifact_root=artifact_root,
        run_root=run_root,
        stage=stage,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        receipt_path=receipt_path,
        receipt=receipt,
        p03_dir=p03_dir,
        p04_dir=p04_dir,
        p03_frame=p03_frame,
        p04_frame=p04_frame,
        bundle={"permit_sha256": PERMIT_SHA256, "artifact_root": artifact_root},
        authenticated={"evaluation_receipt": dict(receipt)},
        deadline=DEADLINE,
    )


def _load(world: SimpleNamespace, deadline=_UNSET):
    return legacy.load_references(
        world.bundle,
        authenticated=world.authenticated,
        deadline=world.deadline if deadline is _UNSET else deadline,
    )


@pytest.fixture
def reads(monkeypatch):
    calls: list[Path] = []
    real = pd.read_parquet

    def wrapper(path, *args, **kwargs):
        calls.append(Path(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", wrapper)
    return calls


# --------------------------------------------------------------------------- #
# Mutators used by the failure tables.
# --------------------------------------------------------------------------- #
def _receipt_update(world: SimpleNamespace, **changes) -> None:
    world.authenticated["evaluation_receipt"].update(changes)


def _receipt_sync(world: SimpleNamespace, **changes) -> None:
    world.authenticated["evaluation_receipt"].update(changes)
    _write_json(world.receipt_path, world.authenticated["evaluation_receipt"])


def _receipt_drop(world: SimpleNamespace) -> None:
    world.authenticated.pop("evaluation_receipt", None)


def _receipt_flag_drop(world: SimpleNamespace, name: str) -> None:
    world.authenticated["evaluation_receipt"].pop(name)


def _p03_descriptor(world: SimpleNamespace, **changes) -> None:
    value = {
        "schema_version": legacy.P03_DESCRIPTOR_SCHEMA,
        "execution_run_id": legacy.P03_RUN_ID,
        "protected_state_sha256": P03_PROTECTED,
    }
    value.update(changes)
    _write_shard_json(world.p03_dir, legacy.P03_DESCRIPTOR_NAME, value)


def _p03_validation(world: SimpleNamespace, **changes) -> None:
    value = {"status": "pass"}
    value.update(changes)
    _write_shard_json(world.p03_dir, legacy.P03_VALIDATION_NAME, value)


def _p04_report(world: SimpleNamespace, **changes) -> None:
    value = {
        "schema_version": legacy.P04_REPORT_SCHEMA,
        "status": "pass",
        "run_id": legacy.P04_RUN_ID,
        "protected_state_sha256": legacy.P04_PROTECTED_STATE_SHA256,
        "aggregation_state_sha256": legacy.P04_AGGREGATION_STATE_SHA256,
    }
    value.update(changes)
    _write_shard_json(world.p04_dir, legacy.P04_REPORT_NAME, value)


def _remove_p03_descriptor(world: SimpleNamespace) -> None:
    (world.p03_dir / legacy.P03_DESCRIPTOR_NAME).unlink()
    _set_state_files(
        world.p03_dir,
        lambda files: {
            key: value for key, value in files.items() if key != legacy.P03_DESCRIPTOR_NAME
        },
    )


def _remove_p04_report(world: SimpleNamespace) -> None:
    (world.p04_dir / legacy.P04_REPORT_NAME).unlink()
    _set_state_files(
        world.p04_dir,
        lambda files: {key: value for key, value in files.items() if key != legacy.P04_REPORT_NAME},
    )


def _unregister_p03_predictions(world: SimpleNamespace) -> None:
    (world.p03_dir / legacy.P03_PREDICTIONS_NAME).unlink()
    _set_state_files(
        world.p03_dir,
        lambda files: {
            key: value for key, value in files.items() if key != legacy.P03_PREDICTIONS_NAME
        },
    )


def _unsafe(name: str):
    return lambda world: _set_state_files(world.p03_dir, lambda files: {**files, name: "0" * 64})


GATE_CASES = [
    pytest.param(_receipt_drop, "evaluation_receipt_malformed", id="receipt_missing"),
    pytest.param(
        lambda world: _receipt_update(world, status="failed"),
        "evaluation_receipt_incomplete",
        id="receipt_status",
    ),
    pytest.param(
        lambda world: _receipt_flag_drop(world, "predictions_frozen"),
        "evaluation_receipt_predictions_frozen_not_true",
        id="frozen_missing",
    ),
    pytest.param(
        lambda world: _receipt_update(world, predictions_frozen=False),
        "evaluation_receipt_predictions_frozen_not_true",
        id="frozen_false",
    ),
    pytest.param(
        lambda world: _receipt_update(world, predictions_frozen=1),
        "evaluation_receipt_predictions_frozen_not_true",
        id="frozen_truthy",
    ),
    pytest.param(
        lambda world: _receipt_update(world, predictions_complete=False),
        "evaluation_receipt_predictions_complete_not_true",
        id="complete_false",
    ),
    pytest.param(
        lambda world: _receipt_update(world, all_complete=False),
        "evaluation_receipt_all_complete_not_true",
        id="all_complete_false",
    ),
    pytest.param(
        lambda world: world.bundle.update({"permit_sha256": "nothex"}),
        "permit_sha256_malformed",
        id="permit_malformed",
    ),
    pytest.param(
        lambda world: world.bundle.update({"permit_sha256": "b" * 64}),
        "evaluation_permit_mismatch",
        id="permit_mismatch",
    ),
    pytest.param(
        lambda world: world.bundle.pop("artifact_root"),
        "artifact_root_missing",
        id="artifact_root_missing",
    ),
    pytest.param(
        lambda world: _write_json(world.receipt_path, {**world.receipt, "extra": True}),
        "evaluation_receipt_changed",
        id="receipt_changed",
    ),
    pytest.param(
        lambda world: world.receipt_path.unlink(),
        "evaluation_receipt_missing",
        id="receipt_missing_on_disk",
    ),
    pytest.param(
        lambda world: _receipt_sync(world, stage_manifest_sha256="nothex"),
        "stage_manifest_sha256_malformed",
        id="stage_manifest_sha256_malformed",
    ),
    pytest.param(
        lambda world: world.manifest_path.unlink(),
        "evaluation_manifest_missing",
        id="manifest_missing",
    ),
    pytest.param(
        lambda world: world.manifest_path.write_bytes(b"changed"),
        "evaluation_manifest_changed",
        id="manifest_changed",
    ),
]


EVIDENCE_CASES = [
    pytest.param(
        lambda world: shutil.rmtree(world.p03_dir),
        "legacy_shard_missing",
        id="p03_shard_missing",
    ),
    pytest.param(
        lambda world: (world.p03_dir / legacy.STATE_NAME).unlink(),
        "legacy_state_missing",
        id="p03_state_missing",
    ),
    pytest.param(
        lambda world: (world.p03_dir / legacy.STATE_NAME).write_bytes(b"{"),
        "legacy_state_malformed",
        id="p03_state_malformed",
    ),
    pytest.param(
        lambda world: _set_state(world.p03_dir, schema_version="other"),
        "legacy_state_schema_mismatch",
        id="p03_schema",
    ),
    pytest.param(
        lambda world: _set_state(world.p03_dir, execution_status="running"),
        "legacy_state_incomplete",
        id="p03_incomplete",
    ),
    pytest.param(
        lambda world: _set_state(world.p03_dir, shard_id=False),
        "legacy_state_shard_mismatch",
        id="p03_shard_bool",
    ),
    pytest.param(
        lambda world: _set_state(world.p03_dir, shard_id="0"),
        "legacy_state_shard_mismatch",
        id="p03_shard_str",
    ),
    pytest.param(
        lambda world: _set_state(world.p03_dir, protected_state_sha256="xyz"),
        "legacy_state_protected_malformed",
        id="p03_protected_malformed",
    ),
    pytest.param(
        lambda world: _set_state(world.p03_dir, protected_state_sha256="f" * 64),
        "p03_protected_mismatch",
        id="p03_protected_prefix",
    ),
    pytest.param(
        lambda world: _set_state(world.p03_dir, files=["nope"]),
        "legacy_state_files_malformed",
        id="p03_files_type",
    ),
    pytest.param(
        lambda world: _set_state(world.p03_dir, files={}),
        "legacy_state_files_malformed",
        id="p03_files_empty",
    ),
    pytest.param(
        lambda world: _set_state_files(
            world.p03_dir,
            lambda files: {**files, legacy.P03_DESCRIPTOR_NAME: "nothex"},
        ),
        "legacy_state_digest_malformed",
        id="p03_digest_malformed",
    ),
    pytest.param(
        lambda world: _set_state_files(
            world.p03_dir, lambda files: {**files, "missing.bin": "0" * 64}
        ),
        "legacy_state_file_missing",
        id="p03_file_missing",
    ),
    pytest.param(
        lambda world: (world.p03_dir / "extra.bin").write_bytes(b"extra"),
        "legacy_state_inventory_mismatch",
        id="p03_inventory_extra",
    ),
    pytest.param(
        lambda world: (world.p03_dir / legacy.P03_PREDICTIONS_NAME).write_bytes(b"tampered"),
        "legacy_state_hash_mismatch",
        id="p03_tamper",
    ),
    pytest.param(_unsafe("../evil"), "legacy_state_path_unsafe", id="unsafe_dotdot"),
    pytest.param(_unsafe("/abs/evil"), "legacy_state_path_unsafe", id="unsafe_absolute"),
    pytest.param(_unsafe("a//b"), "legacy_state_path_unsafe", id="unsafe_redundant_slash"),
    pytest.param(_unsafe("./evil"), "legacy_state_path_unsafe", id="unsafe_dot"),
    pytest.param(_unsafe("a\\b"), "legacy_state_path_unsafe", id="unsafe_backslash"),
    pytest.param(
        _remove_p03_descriptor,
        "p03_descriptor_missing",
        id="p03_descriptor_missing",
    ),
    pytest.param(
        lambda world: _write_shard_raw(world.p03_dir, legacy.P03_DESCRIPTOR_NAME, b"{"),
        "p03_descriptor_malformed",
        id="p03_descriptor_malformed",
    ),
    pytest.param(
        lambda world: _p03_descriptor(world, schema_version="other"),
        "p03_descriptor_schema_mismatch",
        id="p03_descriptor_schema",
    ),
    pytest.param(
        lambda world: _p03_descriptor(world, execution_run_id="P03-other"),
        "p03_descriptor_run_mismatch",
        id="p03_descriptor_run",
    ),
    pytest.param(
        lambda world: _p03_descriptor(world, protected_state_sha256="f" * 64),
        "p03_descriptor_protected_mismatch",
        id="p03_descriptor_protected",
    ),
    pytest.param(
        lambda world: _p03_validation(world, status="fail"),
        "p03_prediction_validation_failed",
        id="p03_validation_status",
    ),
    pytest.param(
        lambda world: _set_state(world.p04_dir, protected_state_sha256="f" * 64),
        "p04_protected_mismatch",
        id="p04_protected",
    ),
    pytest.param(_remove_p04_report, "p04_report_missing", id="p04_report_missing"),
    pytest.param(
        lambda world: _write_shard_raw(world.p04_dir, legacy.P04_REPORT_NAME, b"{"),
        "p04_report_malformed",
        id="p04_report_malformed",
    ),
    pytest.param(
        lambda world: _p04_report(world, schema_version="other"),
        "p04_report_schema_mismatch",
        id="p04_report_schema",
    ),
    pytest.param(
        lambda world: _p04_report(world, status="fail"),
        "p04_report_status_failed",
        id="p04_report_status",
    ),
    pytest.param(
        lambda world: _p04_report(world, run_id="P04-other"),
        "p04_report_run_mismatch",
        id="p04_report_run",
    ),
    pytest.param(
        lambda world: _p04_report(world, protected_state_sha256="f" * 64),
        "p04_report_protected_mismatch",
        id="p04_report_protected",
    ),
    pytest.param(
        lambda world: _p04_report(world, aggregation_state_sha256="f" * 64),
        "p04_report_aggregation_mismatch",
        id="p04_report_aggregation",
    ),
    pytest.param(
        _unregister_p03_predictions,
        "legacy_predictions_unregistered",
        id="p03_predictions_unregistered",
    ),
]


# --------------------------------------------------------------------------- #
# Success path.
# --------------------------------------------------------------------------- #
def test_success_roundtrip_exact_frames_bindings_and_read_only(tmp_path, reads):
    world = _build_world(tmp_path)
    before = _snapshot(world.artifact_root)

    result = _load(world)

    assert isinstance(result, dict)
    assert set(result) == {"p03_predictions", "p04_ensemble", "bindings"}
    pd.testing.assert_frame_equal(result["p03_predictions"], world.p03_frame, check_exact=True)
    pd.testing.assert_frame_equal(result["p04_ensemble"], world.p04_frame, check_exact=True)

    bindings = result["bindings"]
    assert bindings["p03_run_id"] == legacy.P03_RUN_ID
    assert bindings["p04_run_id"] == legacy.P04_RUN_ID
    assert bindings["p03_state_sha256"] == _sha(world.p03_dir / legacy.STATE_NAME)
    assert bindings["p04_state_sha256"] == _sha(world.p04_dir / legacy.STATE_NAME)
    assert bindings["p03_predictions_sha256"] == _sha(world.p03_dir / legacy.P03_PREDICTIONS_NAME)
    assert bindings["p04_predictions_sha256"] == _sha(world.p04_dir / legacy.P04_PREDICTIONS_NAME)
    assert bindings["p03_protected_state_sha256"] == P03_PROTECTED
    assert bindings["p04_shard_protected_state_sha256"] == legacy.P04_AGGREGATION_STATE_SHA256
    assert bindings["p04_execution_protected_state_sha256"] == legacy.P04_PROTECTED_STATE_SHA256
    assert bindings["evaluation_receipt_sha256"] == _sha(world.receipt_path)

    assert reads == [
        world.p03_dir / legacy.P03_PREDICTIONS_NAME,
        world.p04_dir / legacy.P04_PREDICTIONS_NAME,
    ]
    assert _snapshot(world.artifact_root) == before

    encoded = json.dumps(bindings)
    assert str(world.artifact_root) not in encoded
    assert str(tmp_path) not in encoded
    assert all("uid" not in str(value) for value in bindings.values())
    assert all("uid" not in key for key in bindings)


# --------------------------------------------------------------------------- #
# Receipt / manifest gate failures: no parquet read may happen.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutate, code", GATE_CASES)
def test_receipt_and_manifest_gate_failures(tmp_path, reads, mutate, code):
    world = _build_world(tmp_path)
    mutate(world)

    with pytest.raises(core.P05CoreError) as info:
        _load(world)

    assert info.value.reason_code == code
    assert str(tmp_path) not in info.value.reason_code
    assert reads == []


def test_bundle_must_be_mapping(tmp_path, reads):
    world = _build_world(tmp_path)
    with pytest.raises(legacy.P05LegacyReferenceError) as info:
        legacy.load_references(
            "not-a-mapping",
            authenticated=world.authenticated,
            deadline=world.deadline,
        )
    assert info.value.reason_code == "bundle_malformed"
    assert reads == []


def test_authenticated_must_be_mapping(tmp_path, reads):
    world = _build_world(tmp_path)
    with pytest.raises(legacy.P05LegacyReferenceError) as info:
        legacy.load_references(
            world.bundle, authenticated=["not-a-mapping"], deadline=world.deadline
        )
    assert info.value.reason_code == "authenticated_malformed"
    assert reads == []


# --------------------------------------------------------------------------- #
# Shard / evidence failures: both shards must be validated before any read.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mutate, code", EVIDENCE_CASES)
def test_legacy_shard_and_evidence_failures(tmp_path, reads, mutate, code):
    world = _build_world(tmp_path)
    mutate(world)

    with pytest.raises(core.P05CoreError) as info:
        _load(world)

    assert info.value.reason_code == code
    assert str(tmp_path) not in info.value.reason_code
    assert reads == []


def test_both_shards_validated_before_any_parquet_read(tmp_path, reads):
    world = _build_world(tmp_path)
    _p04_report(world, status="fail")

    with pytest.raises(legacy.P05LegacyReferenceError) as info:
        _load(world)

    assert info.value.reason_code == "p04_report_status_failed"
    assert reads == []


# --------------------------------------------------------------------------- #
# Symlinks.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kind, code",
    [
        ("shard_dir", "symlink_path_rejected"),
        ("listed_file", "symlink_path_rejected"),
        ("unlisted_file", "legacy_shard_symlink_rejected"),
    ],
)
def test_symlink_entries_rejected(tmp_path, reads, kind, code):
    world = _build_world(tmp_path)
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")

    if kind == "shard_dir":
        moved = tmp_path / "p03-moved"
        shutil.move(str(world.p03_dir), str(moved))
        world.p03_dir.symlink_to(moved, target_is_directory=True)
    elif kind == "listed_file":
        path = world.p03_dir / legacy.P03_PREDICTIONS_NAME
        path.unlink()
        path.symlink_to(target)
    else:
        (world.p03_dir / "unlisted.bin").symlink_to(target)

    with pytest.raises(core.P05CoreError) as info:
        _load(world)

    assert info.value.reason_code == code
    assert reads == []


# --------------------------------------------------------------------------- #
# Read failures.
# --------------------------------------------------------------------------- #
def test_read_exception_reason_code_is_path_free(tmp_path, monkeypatch):
    world = _build_world(tmp_path)

    def boom(path, *args, **kwargs):
        raise OSError(f"cannot open {path}")

    monkeypatch.setattr(pd, "read_parquet", boom)
    with pytest.raises(legacy.P05LegacyReferenceError) as info:
        _load(world)

    assert info.value.reason_code == "legacy_predictions_read_failed"
    assert str(tmp_path) not in info.value.reason_code


def test_non_frame_read_result_rejected(tmp_path, monkeypatch):
    world = _build_world(tmp_path)

    def not_a_frame(path, *args, **kwargs):
        return {"rows": []}

    monkeypatch.setattr(pd, "read_parquet", not_a_frame)
    with pytest.raises(legacy.P05LegacyReferenceError) as info:
        _load(world)

    assert info.value.reason_code == "legacy_predictions_malformed"
    assert str(tmp_path) not in info.value.reason_code


# --------------------------------------------------------------------------- #
# Mutation while the second parquet read is in flight.
# --------------------------------------------------------------------------- #
def _mutate_p03_predictions(world: SimpleNamespace) -> None:
    path = world.p03_dir / legacy.P03_PREDICTIONS_NAME
    path.write_bytes(path.read_bytes() + b"x")


def _mutate_p03_state(world: SimpleNamespace) -> None:
    path = world.p03_dir / legacy.STATE_NAME
    path.write_bytes(path.read_bytes() + b" ")


def _mutate_receipt(world: SimpleNamespace) -> None:
    world.receipt_path.write_bytes(world.receipt_path.read_bytes() + b" ")


def _mutate_manifest(world: SimpleNamespace) -> None:
    world.manifest_path.write_bytes(world.manifest_path.read_bytes() + b" ")


@pytest.mark.parametrize(
    "mutate, code",
    [
        (_mutate_p03_predictions, "legacy_predictions_hash_changed"),
        (_mutate_p03_state, "legacy_state_changed"),
        (_mutate_receipt, "evaluation_receipt_changed"),
        (_mutate_manifest, "evaluation_manifest_changed"),
    ],
    ids=["p03_predictions", "p03_state", "receipt", "manifest"],
)
def test_mutation_during_second_read_rejected(tmp_path, monkeypatch, mutate, code):
    world = _build_world(tmp_path)
    real = pd.read_parquet
    calls: list[Path] = []

    def wrapper(path, *args, **kwargs):
        calls.append(Path(path))
        if len(calls) == 2:
            mutate(world)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", wrapper)
    with pytest.raises(legacy.P05LegacyReferenceError) as info:
        _load(world)

    assert info.value.reason_code == code
    assert calls == [
        world.p03_dir / legacy.P03_PREDICTIONS_NAME,
        world.p04_dir / legacy.P04_PREDICTIONS_NAME,
    ]


# --------------------------------------------------------------------------- #
# Deadlines.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "deadline",
    [None, "soon", float("nan"), float("inf"), True],
    ids=["none", "string", "nan", "inf", "bool"],
)
def test_invalid_deadline_rejected_without_reads(tmp_path, reads, deadline):
    world = _build_world(tmp_path)

    with pytest.raises(legacy.P05LegacyReferenceError) as info:
        _load(world, deadline=deadline)

    assert info.value.reason_code == "deadline_malformed"
    assert str(tmp_path) not in info.value.reason_code
    assert reads == []


def test_expired_deadline_rejected_without_reads(tmp_path, reads):
    world = _build_world(tmp_path)

    with pytest.raises(core.P05CoreError):
        _load(world, deadline=time.monotonic() - 10**6)

    assert reads == []
