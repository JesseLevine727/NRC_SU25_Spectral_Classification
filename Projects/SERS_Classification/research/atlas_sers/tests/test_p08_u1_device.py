"""T297: regression tests for the worker-device initialization adapter."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas_sers.evaluation.p08_u1_device import prepare_worker_device

torch = pytest.importorskip("torch")

_CUBLAS = ":4096:8"


class FakeCuda:
    def __init__(self, *, available=True, initialized_after_init=True, allocated=0, reserved=0):
        self.calls = []
        self._available = available
        self._initialized = False
        self._initialized_after_init = initialized_after_init
        self._allocated = allocated
        self._reserved = reserved

    def is_available(self):
        self.calls.append("is_available")
        return self._available

    def init(self):
        self.calls.append("init")
        self._initialized = self._initialized_after_init

    def set_device(self, device):
        self.calls.append(("set_device", str(device)))

    def is_initialized(self):
        self.calls.append("is_initialized")
        return self._initialized

    def reset_peak_memory_stats(self, device):
        self.calls.append(("reset_peak_memory_stats", str(device)))

    def synchronize(self, device):
        self.calls.append(("synchronize", str(device)))

    def memory_allocated(self, device):
        self.calls.append(("memory_allocated", str(device)))
        return self._allocated

    def memory_reserved(self, device):
        self.calls.append(("memory_reserved", str(device)))
        return self._reserved


def _names(calls):
    return [call if isinstance(call, str) else call[0] for call in calls]


def _fake_torch(cuda):
    return SimpleNamespace(device=torch.device, cuda=cuda)


def test_cpu_never_touches_cuda(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    cuda = FakeCuda()
    result = prepare_worker_device("cpu", torch_module=_fake_torch(cuda))
    assert result == {
        "device": "cpu",
        "initialized": False,
        "allocated_gpu_bytes": 0,
        "reserved_gpu_bytes": 0,
    }
    assert cuda.calls == []


def test_gpu_call_order_and_telemetry(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", _CUBLAS)
    cuda = FakeCuda(allocated=10, reserved=20)
    result = prepare_worker_device("cuda:0", torch_module=_fake_torch(cuda))
    assert result == {
        "device": "cuda:0",
        "initialized": True,
        "allocated_gpu_bytes": 10,
        "reserved_gpu_bytes": 20,
    }
    assert _names(cuda.calls) == [
        "is_available",
        "init",
        "set_device",
        "is_initialized",
        "reset_peak_memory_stats",
        "synchronize",
        "memory_allocated",
        "memory_reserved",
    ]


@pytest.mark.parametrize("value", [None, "4096:8", ":4096:9"])
def test_missing_or_wrong_cublas_rejected_before_init(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
    cuda = FakeCuda()
    with pytest.raises(RuntimeError):
        prepare_worker_device("cuda:0", torch_module=_fake_torch(cuda))
    assert cuda.calls == []


def test_unavailable_cuda_rejected(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", _CUBLAS)
    cuda = FakeCuda(available=False)
    with pytest.raises(RuntimeError):
        prepare_worker_device("cuda:0", torch_module=_fake_torch(cuda))
    assert _names(cuda.calls) == ["is_available"]


def test_failed_initialization_rejected(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", _CUBLAS)
    cuda = FakeCuda(initialized_after_init=False)
    with pytest.raises(RuntimeError):
        prepare_worker_device("cuda:0", torch_module=_fake_torch(cuda))
    assert _names(cuda.calls) == ["is_available", "init", "set_device", "is_initialized"]


def test_invalid_device_rejected(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    cuda = FakeCuda()
    with pytest.raises((ValueError, RuntimeError)):
        prepare_worker_device("meta", torch_module=_fake_torch(cuda))
    assert cuda.calls == []


@pytest.mark.parametrize("bad", [-1, True, "0"])
def test_invalid_telemetry_rejected(monkeypatch, bad):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", _CUBLAS)
    cuda = FakeCuda(allocated=bad)
    with pytest.raises(RuntimeError):
        prepare_worker_device("cuda:0", torch_module=_fake_torch(cuda))
    assert "reset_peak_memory_stats" in _names(cuda.calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no usable CUDA")
def test_fresh_subprocess_cuda_init_and_reset():
    src = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS
    env["PYTHONPATH"] = os.pathsep.join([str(src), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    child = textwrap.dedent(
        """
        import torch
        from atlas_sers.evaluation.p08_u1_device import prepare_worker_device
        assert torch.cuda.is_available()
        assert not torch.cuda.is_initialized()
        prepare_worker_device("cuda:0")
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        """
    )
    subprocess.run([sys.executable, "-c", child], env=env, check=True, timeout=30)
