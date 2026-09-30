"""Bounded P06/P11 analysis child runner (probe and full modes).

Supervised child only: no subprocesses, no torch, no fitting, no network.
The independent parent enforces hard wall/RSS/output guards; this child also
checks a monotonic wall clock at stage and contrast boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os as _os
import resource
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ[_thread_var] = "1"

PROTOCOL_SHA256 = "14e0de1094411b0b3d63ebc98d3caa3350945e224c5f963ac9ddf109656061c0"
PROTOCOL_RELATIVE = "plan/P06P11_INFERENCE_PROTOCOL.md"
PROBE_WALL_SECONDS = 60.0
FULL_WALL_SECONDS = 1800.0
MAX_RSS_BYTES = 2 * 1024**3
MAX_OUTPUT_BYTES = 1024**3
PROBE_DRAWS = 100
FULL_DRAWS = 10000
PROBE_SEED = 2026092904
MASTER_SEED = 2026092901
INSTRUMENT_SEED = 2026092902
HIERARCHY_SEED = 2026092903
EXPECTED_MASTERS = 69
EXPECTED_INSTRUMENTS = 10

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "atlas_sers"
PROTOCOL_PATH = PROJECT_ROOT / PROTOCOL_RELATIVE

_SYNTH_VOCAB = ("A", "B", "C")
_ARRAY_KEY_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")

__all__ = ["RunError", "run_analysis", "synthetic_panel", "synthetic_paired_metrics", "main"]


class RunError(RuntimeError):
    def __init__(self, code: str, detail: str | None = None) -> None:
        super().__init__(detail or code)
        self.code = code


def _load_module(name: str):
    return importlib.import_module("atlas_sers.evaluation." + name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_code() -> dict[str, str]:
    hashes: dict[str, str] = {}
    if SRC_ROOT.is_dir():
        for path in sorted(SRC_ROOT.rglob("*.py")):
            hashes[path.relative_to(PROJECT_ROOT).as_posix()] = _sha256_file(path)
    hashes[PROTOCOL_RELATIVE] = _sha256_file(PROTOCOL_PATH)
    return hashes


def _check_protocol() -> str:
    digest = _sha256_file(PROTOCOL_PATH)
    if digest != PROTOCOL_SHA256:
        raise RunError("protocol_hash_mismatch")
    return digest


def _software_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if type(obj).__module__ == "numpy":
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
    return obj


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(_jsonable(payload), allow_nan=False, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _validate_output(output: Path, artifact_root: Path | None = None) -> Path:
    candidate = Path(output).absolute()
    for path in (candidate, *candidate.parents):
        if path.is_symlink():
            raise RunError("output_symlink_rejected")
    if candidate.exists():
        raise RunError("output_exists")
    parent = candidate.parent
    if not parent.exists():
        raise RunError("output_parent_missing")
    if not parent.is_dir():
        raise RunError("output_parent_not_directory")
    resolved = candidate.resolve()
    package = PACKAGE_ROOT.resolve()
    project = PROJECT_ROOT.resolve()
    if resolved == package or resolved.is_relative_to(package):
        raise RunError("output_inside_package")
    if resolved == project or resolved.is_relative_to(project):
        raise RunError("output_inside_project")
    if artifact_root is not None:
        art = Path(artifact_root).resolve()
        if resolved == art or resolved.is_relative_to(art):
            raise RunError("output_inside_artifact_root")
    return candidate


def _wall_check(start: float, budget: float) -> Callable[[], None]:
    def check() -> None:
        if time.monotonic() - start > budget:
            raise RunError("wall_limit_exceeded")

    return check


def _safe_name(name: Any) -> str:
    text = str(name)
    if not text or not all(char.isalnum() or char == "_" for char in text):
        raise RunError("unsafe_result_name")
    return text


def _safe_array_key(name: Any) -> str:
    text = str(name)
    if not text:
        raise RunError("unsafe_array_key")
    if text.startswith(".") or ".." in text:
        raise RunError("unsafe_array_key")
    if not set(text) <= _ARRAY_KEY_CHARS:
        raise RunError("unsafe_array_key")
    return text


def _global_counts(panel: Mapping[str, Any]) -> dict[str, int]:
    frame = panel["M01"]
    return {
        "masters": int(frame["master_sample_id"].astype(str).nunique()),
        "instruments": int(frame["instrument"].astype(str).nunique()),
    }


def _assert_frames_equal(produced: Any, pinned: Any, label: str) -> None:
    import pandas as pd

    try:
        pd.testing.assert_frame_equal(
            produced.reset_index(drop=True),
            pinned.reset_index(drop=True),
            check_exact=True,
            check_dtype=True,
        )
    except AssertionError as exc:
        raise RunError(f"gate1_reproduction_mismatch:{label}") from exc


def _input_hashes(inputs: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("hashes", "input_hashes", "pins"):
        value = inputs.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _enforce_resources(output: Path, start: float, budget: float) -> None:
    if time.monotonic() - start > budget:
        raise RunError("wall_limit_exceeded")
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if maxrss > MAX_RSS_BYTES:
        raise RunError("memory_limit_exceeded")
    total = 0
    for path in output.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    if total > MAX_OUTPUT_BYTES:
        raise RunError("output_limit_exceeded")


def _hash_tree(output: Path, exclude: frozenset[str] = frozenset()) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in exclude:
            hashes[path.relative_to(output).as_posix()] = _sha256_file(path)
    return hashes


def _record_failure(
    output: Path,
    mode: str,
    start: float,
    reason: str,
    exception_class: str | None = None,
) -> None:
    payload = {
        "mode": mode,
        "status": "failed",
        "reason": str(reason),
        "exception_class": str(exception_class) if exception_class else None,
        "elapsed_seconds": time.monotonic() - start,
    }
    try:
        _write_json(output / "failure.json", payload)
    except Exception:
        pass


def _iter_pairs(pairs: Any):
    import pandas as pd

    if isinstance(pairs, pd.DataFrame):
        for row in pairs.itertuples(index=False):
            yield str(row.model_id), str(row.reference_model_id)
        return
    for pair in pairs:
        if isinstance(pair, Mapping):
            yield str(pair["model_id"]), str(pair["reference_model_id"])
        elif hasattr(pair, "model_id") and hasattr(pair, "reference_model_id"):
            yield str(pair.model_id), str(pair.reference_model_id)
        else:
            yield str(pair[0]), str(pair[1])


def _context_balanced_accuracy(frame: Any, model_id: Any, context_id: Any) -> float:
    subset = frame[
        (frame["model_id"].astype(str) == str(model_id))
        & (frame["context_id"].astype(str) == str(context_id))
    ]
    if subset.empty:
        return float("nan")
    recalls = [
        float(group["correct"].mean()) for _, group in subset.groupby("true_label", sort=True)
    ]
    return float(sum(recalls) / len(recalls)) if recalls else float("nan")


def synthetic_panel() -> dict[str, Any]:
    import pandas as pd

    comparison = _load_module("p05_comparison")
    models = [str(model) for model in comparison.ALL_MODELS]
    domains = (("d0", "i0"), ("d1", "i1"))
    masters = [f"m{index}" for index in range(6)]
    contexts = []
    for domain, instrument in domains:
        contexts.append((f"{domain}_c0", domain, "s0", instrument, (0, 2, 4)))
        contexts.append((f"{domain}_c1", domain, "s0", instrument, (1, 3, 5)))
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        for context_id, domain, station, instrument, indices in contexts:
            coverage.append(
                {
                    "model_id": model,
                    "context_id": context_id,
                    "domain": domain,
                    "station": station,
                    "held_instrument": instrument,
                    "complete": True,
                    "covered": True,
                }
            )
            for master_index in indices:
                master = masters[master_index]
                class_index = master_index // 2
                true_label = _SYNTH_VOCAB[class_index]
                correct = (model_index + master_index) % 3 == 0
                predicted_label = true_label if correct else _SYNTH_VOCAB[(class_index + 1) % 3]
                rows.append(
                    {
                        "model_id": model,
                        "context_id": context_id,
                        "domain": domain,
                        "station": station,
                        "instrument": instrument,
                        "master_sample_id": master,
                        "unit_id": f"{instrument}_{master}",
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "class_vocabulary": _SYNTH_VOCAB,
                        "correct": bool(correct),
                    }
                )
    m01 = pd.DataFrame(rows)
    return {"M01": m01, "M06": m01.copy(deep=True), "coverage": pd.DataFrame(coverage)}


def synthetic_paired_metrics(panel: Mapping[str, Any]) -> Any:
    import pandas as pd

    comparison = _load_module("p05_comparison")
    frame = panel["M01"]
    meta: dict[str, tuple[str, str, str]] = {}
    for row in frame.drop_duplicates("context_id").itertuples(index=False):
        meta[row.context_id] = (row.domain, row.station, row.instrument)
    rows: list[dict[str, Any]] = []
    for model_id, reference_id in _iter_pairs(comparison.PAIRS):
        for aggregation in ("M01", "M06"):
            for context_id in sorted(meta):
                model_ba = _context_balanced_accuracy(frame, model_id, context_id)
                reference_ba = _context_balanced_accuracy(frame, reference_id, context_id)
                domain, station, instrument = meta[context_id]
                rows.append(
                    {
                        "model_id": model_id,
                        "reference_model_id": reference_id,
                        "aggregation_id": aggregation,
                        "context_id": context_id,
                        "domain": domain,
                        "station": station,
                        "held_instrument": instrument,
                        "common_complete": True,
                        "model_balanced_accuracy": model_ba,
                        "reference_balanced_accuracy": reference_ba,
                        "delta_balanced_accuracy": model_ba - reference_ba,
                    }
                )
    return pd.DataFrame(rows)


def run_analysis(
    *,
    output: Path,
    mode: str,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    mode = str(mode)
    if mode not in ("probe", "full"):
        raise RunError("invalid_mode")
    if mode == "full" and artifact_root is None:
        raise RunError("artifact_root_required")

    budget = PROBE_WALL_SECONDS if mode == "probe" else FULL_WALL_SECONDS
    check = _wall_check(start, budget)

    output = _validate_output(Path(output), artifact_root=artifact_root)
    protocol_sha = _check_protocol()
    code_hashes = _snapshot_code()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)

    if mode == "probe":
        draws = PROBE_DRAWS
        master_seed = instrument_seed = hierarchy_seed = PROBE_SEED
    else:
        draws = FULL_DRAWS
        master_seed, instrument_seed, hierarchy_seed = MASTER_SEED, INSTRUMENT_SEED, HIERARCHY_SEED

    try:
        input_hashes: dict[str, Any] = {}
        point_audit = None
        if mode == "full":
            inputs = _load_module("p06p11_inputs").load_inputs(Path(artifact_root))
            input_hashes = _input_hashes(inputs)

        _write_json(
            output / "start.json",
            {
                "mode": mode,
                "code_hashes": code_hashes,
                "protocol_sha256": protocol_sha,
                "input_hashes": input_hashes,
                "software_versions": _software_versions(),
                "seeds": {
                    "master_seed": master_seed,
                    "instrument_seed": instrument_seed,
                    "hierarchy_seed": hierarchy_seed,
                },
            },
        )
        check()

        if mode == "full":
            produced = _load_module("p05_comparison").compare_predictions(
                p05_ensemble=inputs["p05_ensemble"],
                p04_ensemble=inputs["p04_ensemble"],
                p03_predictions=inputs["p03_predictions"],
                contexts=inputs["contexts"],
            )
            for name in ("endpoint_metrics", "paired_metrics", "coverage", "summary"):
                _assert_frames_equal(produced[name], inputs[name], name)
            check()

            predictions = _load_module("p06p11_predictions")
            panel = predictions.prepare_panel(
                p05_ensemble=inputs["p05_ensemble"],
                p04_ensemble=inputs["p04_ensemble"],
                p03_predictions=inputs["p03_predictions"],
                contexts=inputs["contexts"],
            )
            point_audit = predictions.audit_point_estimates(panel, inputs["endpoint_metrics"])
            counts = _global_counts(panel)
            if counts["masters"] != EXPECTED_MASTERS:
                raise RunError("master_identity_count_mismatch")
            if counts["instruments"] != EXPECTED_INSTRUMENTS:
                raise RunError("instrument_identity_count_mismatch")
            paired_metrics = inputs["paired_metrics"]
        else:
            panel = synthetic_panel()
            paired_metrics = synthetic_paired_metrics(panel)
            counts = _global_counts(panel)

        check()
        result = _load_module("p06p11_analysis").analyze_panel(
            panel,
            paired_metrics,
            draws=draws,
            master_seed=master_seed,
            instrument_seed=instrument_seed,
            hierarchy_seed=hierarchy_seed,
            check=check,
        )
        check()

        if _snapshot_code() != code_hashes:
            raise RunError("source_changed_during_run")
        if mode == "full":
            verified = _load_module("p06p11_inputs").verify_inputs(Path(artifact_root))
            if not isinstance(verified, Mapping) or dict(verified) != input_hashes:
                raise RunError("input_hash_mismatch")

        tables = dict(result.get("tables") or {})
        g4_state = "not_run_probe"
        g4_decision = None
        if mode == "full":
            gate = _load_module("p06p11_diagnostics").g4_checklist(
                tables["summary"],
                tables["intervals"],
                input_preservation_verified=True,
                t1_difference=None,
                t1_provenance_verified=False,
            )
            tables["g4_criteria"] = gate["criteria"]
            g4_decision = gate["decision"]
            g4_state = "ran"

        tables_dir = output / "tables"
        tables_dir.mkdir(mode=0o700)
        for name, frame in tables.items():
            frame.to_csv(tables_dir / f"{_safe_name(name)}.csv", index=False)

        arrays = result.get("arrays") or {}
        import numpy as np

        array_payload: dict[str, Any] = {}
        for key, value in arrays.items():
            arr = np.asarray(value)
            if arr.dtype == object or arr.dtype.kind not in "biuf":
                raise RunError("unsupported_array_dtype")
            array_payload[_safe_array_key(key)] = arr
        np.savez_compressed(output / "arrays.npz", **array_payload)
        _write_json(output / "registry.json", result.get("registry") or {})
        if g4_decision is not None:
            _write_json(output / "g4_decision.json", g4_decision)

        if mode == "full":
            panel["M01"].to_parquet(output / "panel_M01.parquet", index=False)
            panel["M06"].to_parquet(output / "panel_M06.parquet", index=False)
            if point_audit is not None:
                point_audit.to_csv(output / "point_audit.csv", index=False)
            verified = _load_module("p06p11_inputs").verify_inputs(Path(artifact_root))
            if not isinstance(verified, Mapping) or dict(verified) != input_hashes:
                raise RunError("input_hash_mismatch")
            check()

        if _snapshot_code() != code_hashes:
            raise RunError("source_changed_during_run")

        _enforce_resources(output, start, budget)

        file_hashes = _hash_tree(output, exclude=frozenset({"receipt.json"}))
        receipt = {
            "mode": mode,
            "status": "success",
            "draws": draws,
            "seeds": {
                "master_seed": master_seed,
                "instrument_seed": instrument_seed,
                "hierarchy_seed": hierarchy_seed,
            },
            "elapsed_seconds": time.monotonic() - start,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "resources": {
                "maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            },
            "software_versions": _software_versions(),
            "protocol_sha256": protocol_sha,
            "code_hashes": code_hashes,
            "input_hashes": input_hashes,
            "written_files": file_hashes,
            "global_counts": counts,
            "g4": g4_state,
            "publication": "not_claimed",
        }
        _write_json(output / "receipt.json", receipt)
        _enforce_resources(output, start, budget)
        return receipt
    except Exception as exc:
        reason = exc.code if isinstance(exc, RunError) else "analysis_failed"
        _record_failure(output, mode, start, reason, exception_class=type(exc).__name__)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="p06p11_run", description="Bounded P06/P11 analysis child runner."
    )
    parser.add_argument("--mode", choices=("probe", "full"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-root", default=None)
    args = parser.parse_args(argv)
    artifact_root = Path(args.artifact_root) if args.artifact_root else None
    try:
        receipt = run_analysis(
            output=Path(args.output), mode=args.mode, artifact_root=artifact_root
        )
    except Exception as exc:
        print(exc.code if isinstance(exc, RunError) else "analysis_failed", file=sys.stderr)
        return 1
    print(f"mode={receipt['mode']} status={receipt['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
