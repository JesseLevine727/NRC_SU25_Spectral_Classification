"""T295 tests for compact durable per-job artifact IO.

Only temporary directories are used.  No science, no fitting and no private
filesystem access is exercised.
"""

from __future__ import annotations

import hashlib
import os
import stat

import pytest

from atlas_sers.evaluation import p08_u1_artifacts as artifacts_module
from atlas_sers.evaluation.p08_plan import JOB_FIELDS
from atlas_sers.evaluation.p08_u1_artifacts import ArtifactError, ArtifactStore
from atlas_sers.governance.canonical import sha256_value


def _hex(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _fields():
    return {
        "policy_id": "PP-U-MIN",
        "representation_id": "R_MIN_400_1800",
        "array_sha256": _hex("array"),
        "context_id": "ctx-1",
        "model_id": "C-RBF-SVM",
        "model_spec_sha256": _hex("spec"),
        "stage": "source_fit",
        "unit_id": "u1",
        "seed": "deterministic",
        "candidate_id": "c1",
        "hyperparameter_sha256": _hex("hp"),
        "fit_uid_sha256": _hex("fit"),
        "validation_uid_sha256": _hex("val"),
        "test_uid_sha256": _hex("test"),
        "resolution": "fixed_spec",
        "evidence_status": "unapproved_future_job",
    }


def _new_job(fields=None, dependencies=()):
    base = _fields()
    if fields:
        base.update(fields)
    base["dependencies"] = sorted(dependencies)
    job_id = "P08JOB-" + sha256_value({name: base[name] for name in JOB_FIELDS})
    return {**base, "job_id": job_id}


def _store(tmp_path, binding=None):
    root = tmp_path / "root"
    root.mkdir()
    return ArtifactStore(root, binding_sha256=binding or _hex("binding"))


def test_roundtrip_and_canonical_receipt(tmp_path):
    store = _store(tmp_path, "a" * 64)
    job = _new_job()
    receipt = store.write(job, {"a.bin": b"alpha", "b.json": b'{"k":1}'}, status="complete")
    assert receipt["schema_version"] == "nato-sers-p08-u1-artifacts-v1"
    assert receipt["job_id"] == job["job_id"]
    assert receipt["sha256"] == sha256_value(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    loaded, files = store.verify(job, expected_receipt_sha256=receipt["sha256"])
    assert loaded == receipt
    assert files == {"a.bin": b"alpha", "b.json": b'{"k":1}'}
    raw = (store._job_dir(job["job_id"]) / "receipt.json").read_bytes()
    assert raw == artifacts_module.canonical_json_bytes(receipt)
    with pytest.raises(ArtifactError):
        store.verify(job, expected_receipt_sha256="0" * 64)


def test_wrong_binding_and_wrong_job_rejected(tmp_path):
    store = _store(tmp_path, "a" * 64)
    job = _new_job()
    store.write(job, {"a.bin": b"x"}, status="complete")
    other = ArtifactStore(store._root, binding_sha256="b" * 64)
    with pytest.raises(ArtifactError):
        other.verify(job)
    with pytest.raises(ArtifactError):
        store.verify(_new_job({"unit_id": "u2"}))
    tampered = dict(job)
    tampered["seed"] = 1
    with pytest.raises(ArtifactError):
        store.verify(tampered)


def test_write_input_validation(tmp_path):
    store = _store(tmp_path)
    job = _new_job()
    with pytest.raises(ArtifactError):
        store.write(job, {}, status="complete")
    with pytest.raises(ArtifactError):
        store.write(job, {"a.bin": "text"}, status="complete")
    with pytest.raises(ArtifactError):
        store.write(job, {"a.bin": b"x"}, status="running")
    with pytest.raises(ArtifactError):
        store.write(job, {"a.bin": b"x"}, status="complete", extra={"v": float("nan")})
    assert not store._job_dir(job["job_id"]).exists()


def test_content_tamper_missing_extra_and_unlisted(tmp_path):
    store = _store(tmp_path)
    job = _new_job()
    receipt = store.write(job, {"a.bin": b"alpha"}, status="failed", extra={"note": "x"})
    assert receipt["extra"] == {"note": "x"}
    _, files = store.verify(job)
    assert files == {"a.bin": b"alpha"}
    job_dir = store._job_dir(job["job_id"])
    (job_dir / "a.bin").write_bytes(b"tampered")
    with pytest.raises(ArtifactError):
        store.verify(job)
    (job_dir / "a.bin").write_bytes(b"alpha")
    (job_dir / "a.bin").unlink()
    with pytest.raises(ArtifactError):
        store.verify(job)
    (job_dir / "a.bin").write_bytes(b"alpha")
    (job_dir / "unlisted.bin").write_bytes(b"z")
    with pytest.raises(ArtifactError):
        store.verify(job)


def test_suspicious_names_and_symlinks_rejected(tmp_path):
    store = _store(tmp_path)
    job = _new_job()
    for name in ("../x", "..", ".", "receipt.json", "a/b"):
        with pytest.raises(ArtifactError):
            store.write(job, {name: b"x"}, status="complete")
    store.write(job, {"a.bin": b"x"}, status="complete")
    job_dir = store._job_dir(job["job_id"])
    target = tmp_path / "target.bin"
    target.write_bytes(b"x")
    (job_dir / "a.bin").unlink()
    (job_dir / "a.bin").symlink_to(target)
    with pytest.raises(ArtifactError):
        store.verify(job)


def test_root_shard_and_jobdir_symlinks_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ArtifactError):
        ArtifactStore(link, binding_sha256=_hex("binding"))

    store = _store(tmp_path)
    job = _new_job()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shard = store._jobs / job["job_id"][7:9]
    shard.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ArtifactError):
        store.write(job, {"a.bin": b"x"}, status="complete")
    shard.unlink()
    shard.mkdir()
    (shard / job["job_id"]).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ArtifactError):
        store.write(job, {"a.bin": b"x"}, status="complete")


def test_hardlinked_artifact_rejected(tmp_path):
    store = _store(tmp_path)
    job = _new_job()
    store.write(job, {"a.bin": b"x"}, status="complete")
    job_dir = store._job_dir(job["job_id"])
    target = tmp_path / "target.bin"
    target.write_bytes(b"x")
    (job_dir / "a.bin").unlink()
    os.link(target, job_dir / "a.bin")
    with pytest.raises(ArtifactError):
        store.verify(job)


def test_duplicate_write_does_not_overwrite(tmp_path):
    store = _store(tmp_path)
    job = _new_job()
    store.write(job, {"a.bin": b"first"}, status="complete")
    with pytest.raises(ArtifactError):
        store.write(job, {"a.bin": b"second"}, status="complete")
    assert (store._job_dir(job["job_id"]) / "a.bin").read_bytes() == b"first"


def test_receipt_last_failure_preserves_artifacts(tmp_path, monkeypatch):
    store = _store(tmp_path)
    job = _new_job()
    real = artifacts_module._write_exclusive

    def fake(directory, name, data):
        if name == "receipt.json":
            raise ArtifactError("artifact_create_failed")
        return real(directory, name, data)

    monkeypatch.setattr(artifacts_module, "_write_exclusive", fake)
    with pytest.raises(ArtifactError):
        store.write(job, {"a.bin": b"data"}, status="complete")
    job_dir = store._job_dir(job["job_id"])
    assert (job_dir / "a.bin").read_bytes() == b"data"
    assert not (job_dir / "receipt.json").exists()


def test_permissions_are_private(tmp_path):
    store = _store(tmp_path)
    job = _new_job()
    store.write(job, {"a.bin": b"x"}, status="complete")
    job_dir = store._job_dir(job["job_id"])
    assert stat.S_IMODE(os.stat(job_dir).st_mode) == 0o700
    for path in (job_dir / "a.bin", job_dir / "receipt.json"):
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
