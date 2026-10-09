"""T297: worker-device initialization adapter for p08_u1."""

from __future__ import annotations

import os
from typing import Any

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_ALLOWED_TYPES = ("cpu", "cuda")


def _import_torch() -> Any:
    import torch

    return torch


def _require_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"invalid {name}: {value!r}")
    return value


def prepare_worker_device(device, *, torch_module=None) -> dict:
    """Parse a worker device and initialize CUDA without allocating tensors."""
    torch = _import_torch() if torch_module is None else torch_module
    parsed = torch.device(device)
    if parsed.type not in _ALLOWED_TYPES:
        raise ValueError(f"unsupported device type: {parsed.type!r}")
    if parsed.type == "cpu":
        return {
            "device": str(parsed),
            "initialized": False,
            "allocated_gpu_bytes": 0,
            "reserved_gpu_bytes": 0,
        }
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be exactly ':4096:8' for deterministic CUDA"
        )
    cuda = torch.cuda
    if cuda.is_available() is not True:
        raise RuntimeError("CUDA is not available")
    cuda.init()
    cuda.set_device(parsed)
    if cuda.is_initialized() is not True:
        raise RuntimeError("CUDA initialization failed")
    cuda.reset_peak_memory_stats(parsed)
    cuda.synchronize(parsed)
    allocated = _require_nonnegative_int("allocated_gpu_bytes", cuda.memory_allocated(parsed))
    reserved = _require_nonnegative_int("reserved_gpu_bytes", cuda.memory_reserved(parsed))
    return {
        "device": str(parsed),
        "initialized": True,
        "allocated_gpu_bytes": allocated,
        "reserved_gpu_bytes": reserved,
    }
