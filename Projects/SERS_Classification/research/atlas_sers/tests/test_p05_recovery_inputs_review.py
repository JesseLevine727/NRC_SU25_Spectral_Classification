"""Independent regressions for the read-only recovery evidence boundary."""

from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import pytest

from atlas_sers.evaluation import p05_recovery_inputs as module


@pytest.mark.parametrize("raw", [b'{"x":1e999}', b'{"x":-1e999}', b'{"x":"\\ud800"}'])
def test_reject_json_overflow_or_invalid_unicode(raw):
    with pytest.raises(module.RecoveryInputsError):
        module._parse_json(raw, "synthetic")


def test_huge_deadline_has_stable_error():
    with pytest.raises(module.RecoveryInputsError, match="deadline_invalid"):
        module._check_deadline(10**400)


@pytest.mark.parametrize("path", [None, "bad\0path", "one/../two"])
def test_invalid_raw_path_has_stable_error(path):
    with pytest.raises(module.RecoveryInputsError):
        module._reject_symlink_chain(path)


def test_stream_read_failure_has_stable_error():
    class Broken(io.BytesIO):
        def read(self, *_args):
            raise OSError("private-file-path")

    with pytest.raises(module.RecoveryInputsError) as caught:
        module._hash_stream(Broken(), time.perf_counter() + 10)
    assert "private-file-path" not in str(caught.value)


@pytest.mark.parametrize("kind", ["bare", "extra", "bool"])
def test_manifest_record_has_exact_schema(kind):
    record = {"sha256": hashlib.sha256(b"test").hexdigest(), "size_bytes": 1}
    candidate = {**record}
    if kind == "bare":
        candidate = record["sha256"]
    elif kind == "extra":
        candidate["ignored"] = "no"
    else:
        candidate["size_bytes"] = True
    assert module._record_equal(candidate, record) is False


def test_pilot_manifest_check_gets_base_permit(monkeypatch, tmp_path):
    permit = {"pilot_manifest_sha256": "synthetic"}
    bundle = {"artifact_root": tmp_path, "permit": permit, "contract": {}}
    monkeypatch.setattr(module.base_inputs, "_pilot_run_dir", lambda _root: Path(tmp_path))
    calls = []
    monkeypatch.setattr(
        module.base_inputs, "_check_pilot_run_manifest", lambda _root, p: calls.append(p)
    )
    monkeypatch.setattr(module.base_inputs, "_check_pilot_run_summary", lambda _root: None)
    monkeypatch.setattr(module.base_inputs, "_check_pilot_slot_leases", lambda _bundle: None)
    # Any stricter whole-pilot verifier is tested at the filesystem seam separately.
    monkeypatch.setattr(module, "_verify_pilot_inventory", lambda *_args: None, raising=False)
    module._authenticate_pilot(bundle, time.perf_counter() + 10)
    assert calls == [permit]
