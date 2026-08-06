"""Canonical serialization and streaming hashes for protected scientific state."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _normalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, Path):
        raise TypeError(
            "Filesystem paths must be converted to sanitized logical identifiers first."
        )
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Canonical mappings require string keys.")
            normalized[key] = _normalize(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Canonical JSON prohibits NaN and infinite values.")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported canonical value type: {type(value).__name__}")


def canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    """Serialize JSON-compatible state deterministically as UTF-8."""

    normalized = _normalize(value)
    if pretty:
        text = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        return f"{text}\n".encode()
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Create a compressed NumPy archive with stable member order and timestamps."""

    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError("NPZ member names must be simple non-empty identifiers.")
            member = io.BytesIO()
            np.save(member, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def hash_relative_files(base: Path, paths: Sequence[Path]) -> dict[str, dict[str, int | str]]:
    """Hash files and expose only paths relative to an approved base."""

    result: dict[str, dict[str, int | str]] = {}
    base_resolved = base.resolve()
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix()):
        relative = path.relative_to(base_resolved).as_posix()
        result[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return result
