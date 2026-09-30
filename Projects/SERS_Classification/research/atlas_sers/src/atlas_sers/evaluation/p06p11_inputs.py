"""Pinned-byte consistency reader for reviewed P06-P11 evidence artifacts.

This module is a read-only, pinned mapping reader. It does not constitute
independent provenance; it only confirms that the reviewed evidence bytes
match the explicit SHA-256 pins fixed at review time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

PINS: dict[str, list[str]] = {
    "manifest": [
        "p01/runs/P01-ba26c5a50087107e314912f7/primary_manifest.csv",
        "db1f298a76aeb9962db004776a9f41d6c9afe5b76c39aa9277a24848108d5f90",
    ],
    "contexts": [
        "p04plan/runs/P04PLAN-ef3155b83c14ab242a47067a/context_registry.csv",
        "12be22701e6c9847301bff7978bd6e4a4cd2f4abade84c4a025f1ed2c24810fb",
    ],
    "roles": [
        "p04plan/runs/P04PLAN-ef3155b83c14ab242a47067a/role_registry.csv",
        "224891579d5df84c42a1c3c827590b6515ec83a13bfe4f07d48f249432f38d91",
    ],
    "p03_predictions": [
        "p03/runs/P03-513a0f9686c37cbc0d682645/final_aggregation/"
        "shards/shard-000000/final_predictions.parquet",
        "ddc1620ffb67335c63806e778abf13d830f5bd68a39b2ecc5fb05d87e898aa9a",
    ],
    "p04_ensemble": [
        "p04/runs/P04-e845290bb15d37882f29da6f/final_aggregation/"
        "shards/shard-000000/ensemble_test_predictions.parquet",
        "186897fd977640bd98249dae107b046d657feedc25822d3fedb63ddb1101815d",
    ],
    "p05_ensemble": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "aggregation/ensemble_predictions.parquet",
        "d8b8d342e4249fd4953f8df9e73940b2468b36449c7f069738a607cbb1e4ff95",
    ],
    "p05_seed": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "aggregation/seed_predictions.parquet",
        "dd85d5164d54ce7c534374c709f5dac36fe3e07c77c012ce814d1ae410b2f786",
    ],
    "reporting_receipt": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "reporting_receipt.json",
        "f09f066bbc46b7c1369b340b69ebed4758cf2587723392d80b665085ca0ecb9e",
    ],
    "comparison_receipt": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "comparison_receipt.json",
        "c286da307e2be5bfd7be004b6984cab3e1290b6fed324cb32a8ce1fd7c1c5980",
    ],
    "aggregation_receipt": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "aggregation_receipt.json",
        "e00554ba017281eaef236dead29e57cc01a890d4e3c45e12204bf138faeaeedf",
    ],
    "aggregation_manifest": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "aggregation/manifest.json",
        "bed71a16f7cc29633a673cf9b5ba1173b7d39a3239f275b8f8f6de96e4e6abc3",
    ],
    "comparison_manifest": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "comparison/manifest.json",
        "ad85de4fe5ae567f99ae3be2966845d95a6f25481d9039cfdceff983a452c75c",
    ],
    "endpoint_metrics": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "comparison/endpoint_metrics.parquet",
        "2facc4cf08c187d2b0d2c074909da5494ceb7899c8f8e6285ec9b8cf3c946a69",
    ],
    "paired_metrics": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "comparison/paired_metrics.parquet",
        "0d753c9d15127f9537440d21cbc09ff56e6fb0fe08ce3b0fb5c07d88ebe2f84e",
    ],
    "coverage": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "comparison/coverage.parquet",
        "7868c0d8801a110c3a5191ed40ba75e7b81f7a14e2dd37f7cd2185ac1f3fb3d7",
    ],
    "summary": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "comparison/summary.parquet",
        "84f340c072607c57386ef712ab4a4406d79825b4446d5e0e6e271327290ef204",
    ],
    "reference_bindings": [
        "p05comprehensive/runs/"
        "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8/"
        "comparison/reference_bindings.json",
        "1ca9688f37df37399f6733587c0d0b417a582c0156432747345eabd536b72efe",
    ],
}

_CHUNK = 1 << 20
_CSV = ".csv"
_PARQUET = ".parquet"
_JSON = ".json"


class InputIntegrityError(RuntimeError):
    """Fail-closed, path-free integrity error carrying a fixed reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _validated_root(artifact_root: Any) -> Path:
    try:
        root = Path(artifact_root)
        absolute = root.absolute()
    except (TypeError, OSError):
        raise InputIntegrityError("root_invalid") from None
    try:
        current = absolute
        while True:
            if current.is_symlink():
                raise InputIntegrityError("symlink_rejected")
            parent = current.parent
            if parent == current:
                break
            current = parent
    except InputIntegrityError:
        raise
    except OSError:
        raise InputIntegrityError("root_invalid") from None
    try:
        if not absolute.exists():
            raise InputIntegrityError("missing_root")
        if not absolute.is_dir():
            raise InputIntegrityError("not_directory")
    except InputIntegrityError:
        raise
    except OSError:
        raise InputIntegrityError("root_invalid") from None
    return absolute


def _safe_parts(rel: str) -> tuple[str, ...]:
    if not rel or rel.startswith("/") or "\\" in rel:
        raise InputIntegrityError("unsafe_relative_path")
    parts = rel.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise InputIntegrityError("unsafe_relative_path")
    return tuple(parts)


def _safe_target(root: Path, rel: str) -> Path:
    try:
        current = root
        for part in _safe_parts(rel):
            current = current / part
            if current.is_symlink():
                raise InputIntegrityError("symlink_rejected")
        if not current.is_file():
            raise InputIntegrityError("not_regular_file")
    except InputIntegrityError:
        raise
    except OSError:
        raise InputIntegrityError("target_invalid") from None
    return current


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
    except OSError:
        raise InputIntegrityError("hash_read_failure") from None
    return digest.hexdigest()


def verify_inputs(artifact_root: Any) -> dict[str, str]:
    root = _validated_root(artifact_root)
    observed: dict[str, str] = {}
    for key, (rel, expected) in PINS.items():
        path = _safe_target(root, rel)
        actual = _sha256_file(path)
        if actual != expected:
            raise InputIntegrityError("hash_mismatch")
        observed[key] = actual
    return observed


def _read_csv(path: Path) -> Any:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        raise InputIntegrityError("read_failure") from None


def _read_parquet(path: Path) -> Any:
    try:
        return pd.read_parquet(path)
    except Exception:
        raise InputIntegrityError("read_failure") from None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        raise InputIntegrityError("read_failure") from None
    if not isinstance(payload, dict):
        raise InputIntegrityError("json_not_object")
    return payload


def load_inputs(artifact_root: Any) -> dict[str, Any]:
    root = _validated_root(artifact_root)
    pre = verify_inputs(root)
    loaded: dict[str, Any] = {}
    for key, (rel, _expected) in PINS.items():
        path = _safe_target(root, rel)
        suffix = path.suffix.lower()
        if suffix == _CSV:
            loaded[key] = _read_csv(path)
        elif suffix == _PARQUET:
            loaded[key] = _read_parquet(path)
        elif suffix == _JSON:
            loaded[key] = _read_json(path)
        else:
            raise InputIntegrityError("unsupported_kind")
    post = verify_inputs(root)
    if post != pre:
        raise InputIntegrityError("post_read_mismatch")
    loaded["hashes"] = post
    return loaded
