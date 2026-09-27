import copy
import hashlib
import json
import os
import stat
import types
from pathlib import Path

import pytest

from atlas_sers.evaluation import p05_recovery_authority as auth

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "plan" / "contracts" / "p05_comprehensive_recovery.json"
OLD_CONTRACT_PATH = ROOT / "plan" / "contracts" / "p05_comprehensive.json"
GIB = 1024**3


def _canonical(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@pytest.fixture(scope="module")
def contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_canonical_pin(contract):
    assert hashlib.sha256(_canonical(contract)).hexdigest() == auth.RECOVERY_PERMIT_SHA256


def test_validate_returns_independent_copy(contract):
    result = auth.validate_recovery_permit(contract)
    assert result == contract
    result["schema_version"] = "mutated"
    assert contract["schema_version"] != "mutated"


def test_validate_does_not_mutate_input(contract):
    before = copy.deepcopy(contract)
    auth.validate_recovery_permit(contract)
    assert contract == before


def test_load_actual_contract(contract):
    loaded = auth.load_recovery_permit(CONTRACT_PATH)
    assert loaded == contract


@pytest.mark.skipif(not OLD_CONTRACT_PATH.exists(), reason="old permit absent")
def test_old_permit_rejected():
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.load_recovery_permit(OLD_CONTRACT_PATH)


@pytest.mark.parametrize("field", ["schema_version", "maximum_total_seconds"])
def test_tampered_field_rejected(contract, field):
    bad = copy.deepcopy(contract)
    if isinstance(bad.get(field), int) and not isinstance(bad.get(field), bool):
        bad[field] = bad[field] + 1
    else:
        bad[field] = "tampered"
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.validate_recovery_permit(bad)


@pytest.mark.parametrize("bad", [None, [], "text", 5, True, (1, 2)])
def test_non_mapping_rejected(bad):
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.validate_recovery_permit(bad)


def test_non_string_key_rejected(contract):
    bad = copy.deepcopy(contract)
    bad[1] = "x"
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.validate_recovery_permit(bad)


def test_nested_non_string_key_rejected(contract):
    bad = copy.deepcopy(contract)
    bad["__nested__"] = {"ok": [1, 2, {3: "x"}]}
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.validate_recovery_permit(bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rejected(contract, value):
    bad = copy.deepcopy(contract)
    bad["__value__"] = value
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.validate_recovery_permit(bad)


def test_bool_numeric_rejected(contract):
    bad = copy.deepcopy(contract)
    bad["maximum_total_seconds"] = True
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.validate_recovery_permit(bad)


def test_bogus_payload_rejected():
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.validate_recovery_permit({"schema_version": auth.RECOVERY_SCHEMA_VERSION})


def test_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "dupe.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.load_recovery_permit(path)


def test_non_finite_json_rejected(tmp_path):
    path = tmp_path / "nan.json"
    path.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.load_recovery_permit(path)


def test_symlink_file_rejected(tmp_path):
    target = tmp_path / "real.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.load_recovery_permit(link)


def test_symlink_parent_rejected(tmp_path):
    real_dir = tmp_path / "realdir"
    real_dir.mkdir()
    (real_dir / "permit.json").write_text("{}", encoding="utf-8")
    link_dir = tmp_path / "linkdir"
    link_dir.symlink_to(real_dir)
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.load_recovery_permit(link_dir / "permit.json")


def test_directory_rejected(tmp_path):
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.load_recovery_permit(tmp_path)


def test_missing_rejected(tmp_path):
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.load_recovery_permit(tmp_path / "absent.json")


def test_oversized_rejected(tmp_path):
    path = tmp_path / "big.json"
    path.write_bytes(b"{" + b" " * (64 * 1024 + 8) + b"}")
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.load_recovery_permit(path)


def test_invalid_utf8_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(b'{"a": "\xff"}')
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.load_recovery_permit(path)


def _meminfo_file(tmp_path, available_bytes, name="meminfo"):
    path = tmp_path / name
    path.write_text(f"MemAvailable: {available_bytes // 1024} kB\n", encoding="utf-8")
    return path


def test_meminfo_legit(tmp_path):
    path = _meminfo_file(tmp_path, 12345 * 1024)
    assert auth.read_host_available_bytes(path) == 12345 * 1024


def test_meminfo_zero_st_size_simulated(tmp_path, monkeypatch):
    path = tmp_path / "meminfo"
    path.write_text("MemAvailable: 2048 kB\n", encoding="utf-8")
    real_stat = os.stat

    class _Stat:
        st_mode = stat.S_IFREG | 0o644
        st_size = 0

    def fake_stat(target, *args, **kwargs):
        if os.fspath(target) == os.fspath(path):
            return _Stat()
        return real_stat(target, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fake_stat)
    assert auth.read_host_available_bytes(path) == 2048 * 1024


@pytest.mark.parametrize(
    "text",
    [
        "MemTotal: 1 kB\nMemFree: 1 kB\n",
        "MemAvailable: -5 kB\n",
        "MemAvailable: 12.5 kB\n",
        "MemAvailable: 123 KB\n",
        "MemAvailable: 123 kb\n",
        "MemAvailable: 123 MB\n",
        "MemAvailable: kB\n",
        "MemAvailable: 123\n",
        "MemAvailable: 12 3 kB\n",
    ],
)
def test_meminfo_malformed(tmp_path, text):
    path = tmp_path / "meminfo"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.read_host_available_bytes(path)


def test_meminfo_duplicate(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemAvailable: 1 kB\nMemAvailable: 2 kB\n", encoding="utf-8")
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.read_host_available_bytes(path)


def test_meminfo_oversized(tmp_path):
    path = tmp_path / "meminfo"
    path.write_bytes(b"MemAvailable: 1 kB\n" + b" " * (1024 * 1024 + 8))
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.read_host_available_bytes(path)


def test_meminfo_symlink_rejected(tmp_path):
    target = _meminfo_file(tmp_path, 1024 * 1024, name="real_meminfo")
    link = tmp_path / "link_meminfo"
    link.symlink_to(target)
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.read_host_available_bytes(link)


class _FakeCuda:
    def __init__(self, *, available=True, allocated=0, peak=0, free=0, total=0, raise_on=()):
        self._available = available
        self._allocated = allocated
        self._peak = peak
        self._free = free
        self._total = total
        self._raise_on = (raise_on,) if isinstance(raise_on, str) else tuple(raise_on)

    def _maybe_raise(self, name):
        if name in self._raise_on:
            raise RuntimeError("boom")

    def is_available(self):
        self._maybe_raise("is_available")
        return self._available

    def memory_allocated(self):
        self._maybe_raise("memory_allocated")
        return self._allocated

    def max_memory_allocated(self):
        self._maybe_raise("max_memory_allocated")
        return self._peak

    def mem_get_info(self):
        self._maybe_raise("mem_get_info")
        return (self._free, self._total)


def _fake_torch(**kwargs):
    return types.SimpleNamespace(cuda=_FakeCuda(**kwargs))


def _launch_meminfo(tmp_path):
    return _meminfo_file(tmp_path, auth.MINIMUM_HOST_AVAILABLE_BYTES_BEFORE_LAUNCH)


def test_launch_exact_boundaries_pass(tmp_path):
    path = _launch_meminfo(tmp_path)
    torch = _fake_torch(
        available=True,
        allocated=auth.MAXIMUM_CUDA_ALLOCATED_BYTES,
        peak=auth.MAXIMUM_CUDA_ALLOCATED_BYTES,
        free=auth.MINIMUM_FREE_CUDA_BYTES,
        total=auth.MINIMUM_FREE_CUDA_BYTES + auth.MAXIMUM_CUDA_ALLOCATED_BYTES,
    )
    result = auth.check_resources(torch, phase="launch", meminfo_path=path)
    assert result["phase"] == "launch"
    assert result["cuda_free_bytes"] == auth.MINIMUM_FREE_CUDA_BYTES


def test_launch_low_host_fails(tmp_path):
    path = _meminfo_file(tmp_path, auth.MINIMUM_HOST_AVAILABLE_BYTES_BEFORE_LAUNCH - 1024)
    torch = _fake_torch(free=auth.MINIMUM_FREE_CUDA_BYTES, total=auth.MINIMUM_FREE_CUDA_BYTES)
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(torch, phase="launch", meminfo_path=path)


def test_launch_low_free_fails(tmp_path):
    path = _launch_meminfo(tmp_path)
    torch = _fake_torch(
        free=auth.MINIMUM_FREE_CUDA_BYTES - 1,
        total=auth.MINIMUM_FREE_CUDA_BYTES,
    )
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(torch, phase="launch", meminfo_path=path)


def test_launch_free_exceeds_total_fails(tmp_path):
    path = _launch_meminfo(tmp_path)
    torch = _fake_torch(
        free=auth.MINIMUM_FREE_CUDA_BYTES + 1,
        total=auth.MINIMUM_FREE_CUDA_BYTES,
    )
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(torch, phase="launch", meminfo_path=path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allocated": auth.MAXIMUM_CUDA_ALLOCATED_BYTES + 1},
        {"peak": auth.MAXIMUM_CUDA_ALLOCATED_BYTES + 1},
    ],
)
def test_overallocated_fails(tmp_path, kwargs):
    path = _launch_meminfo(tmp_path)
    torch = _fake_torch(
        free=auth.MINIMUM_FREE_CUDA_BYTES, total=auth.MINIMUM_FREE_CUDA_BYTES, **kwargs
    )
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(torch, phase="launch", meminfo_path=path)


def test_unavailable_fails(tmp_path):
    path = _launch_meminfo(tmp_path)
    torch = _fake_torch(available=False)
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(torch, phase="launch", meminfo_path=path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allocated": True},
        {"allocated": 1.5},
        {"peak": True},
        {"peak": 2.5},
    ],
)
def test_malformed_metrics_fail(tmp_path, kwargs):
    path = _launch_meminfo(tmp_path)
    torch = _fake_torch(
        free=auth.MINIMUM_FREE_CUDA_BYTES, total=auth.MINIMUM_FREE_CUDA_BYTES, **kwargs
    )
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(torch, phase="launch", meminfo_path=path)


@pytest.mark.parametrize(
    "name", ["is_available", "memory_allocated", "max_memory_allocated", "mem_get_info"]
)
def test_query_runtime_error_fails(tmp_path, name):
    path = _launch_meminfo(tmp_path)
    torch = _fake_torch(
        free=auth.MINIMUM_FREE_CUDA_BYTES,
        total=auth.MINIMUM_FREE_CUDA_BYTES,
        raise_on=name,
    )
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(torch, phase="launch", meminfo_path=path)


def test_invalid_phase_fails(tmp_path):
    path = _launch_meminfo(tmp_path)
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(_fake_torch(), phase="bogus", meminfo_path=path)


@pytest.mark.parametrize("phase", ["fit", "epoch"])
def test_fit_epoch_no_free_requirement(tmp_path, phase):
    path = _meminfo_file(tmp_path, auth.MINIMUM_HOST_AVAILABLE_BYTES_DURING_EXECUTION)
    torch = _fake_torch(allocated=0, peak=0, free=0, total=0)
    result = auth.check_resources(torch, phase=phase, meminfo_path=path)
    assert result["phase"] == phase
    assert "cuda_free_bytes" not in result


def test_during_execution_low_host_fails(tmp_path):
    path = _meminfo_file(tmp_path, auth.MINIMUM_HOST_AVAILABLE_BYTES_DURING_EXECUTION - 1024)
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(_fake_torch(), phase="fit", meminfo_path=path)


def test_no_torch_or_numpy_imports():
    import ast

    source = Path(auth.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"torch", "numpy"}
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"torch", "numpy"}


@pytest.mark.parametrize("path", [None, [], 42, "bad\x00path"])
@pytest.mark.parametrize("loader", [auth.load_recovery_permit, auth.read_host_available_bytes])
def test_bad_path_maps_to_authority_error(path, loader):
    with pytest.raises(auth.RecoveryAuthorityError):
        loader(path)


def test_recursive_permit_maps_to_authority_error(contract):
    malformed = copy.deepcopy(contract)
    malformed["cycle"] = malformed
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.validate_recovery_permit(malformed)


def test_overdeep_json_maps_to_authority_error(tmp_path):
    path = tmp_path / "deep.json"
    path.write_text("[" * 1100 + "0" + "]" * 1100)
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.load_recovery_permit(path)


@pytest.mark.parametrize("digits", ["\u0661\u0662", "9" * 5000], ids=["unicode", "too_long"])
def test_unreasonable_meminfo_integer_rejected(tmp_path, digits):
    path = tmp_path / "meminfo"
    path.write_text(f"MemAvailable: {digits} kB\n")
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.read_host_available_bytes(path)


def test_cuda_peak_cannot_be_less_than_current(tmp_path):
    path = _launch_meminfo(tmp_path)
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(_fake_torch(allocated=2, peak=1), phase="fit", meminfo_path=path)


def test_cuda_current_plus_free_cannot_exceed_total(tmp_path):
    path = _launch_meminfo(tmp_path)
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(
            _fake_torch(allocated=1, peak=1, free=10 * GIB, total=10 * GIB),
            phase="launch",
            meminfo_path=path,
        )


@pytest.mark.parametrize("field", ["free", "total"])
@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_cuda_free_total_metrics_strict(tmp_path, field, value):
    path = _launch_meminfo(tmp_path)
    metrics = {"free": 10 * GIB, "total": 16 * GIB, field: value}
    with pytest.raises(auth.RecoveryAuthorityError):
        auth.check_resources(_fake_torch(**metrics), phase="launch", meminfo_path=path)
