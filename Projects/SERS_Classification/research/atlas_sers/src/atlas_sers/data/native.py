"""Read native ATLAS instrument exports without exposing resolved source paths."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, percentile_filter

from atlas_sers.governance.canonical import sha256_file, sha256_value
from atlas_sers.paths import is_relative_to


def _scan_index(directory: Path, glob: str, pattern: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(directory.glob(glob)):
        match = re.search(pattern, path.name, re.IGNORECASE)
        if match:
            result[int(match.group(1))] = path
    return result


def _relative(root: Path, path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    if not is_relative_to(path, root):
        raise ValueError("A native source escaped ATLAS_NATIVE_ROOT.")
    return path.resolve().relative_to(root.resolve()).as_posix()


def index_native_sources(native_root: Path) -> dict[tuple[str, int], dict[str, str]]:
    """Index the four supported vendor export layouts using relative paths only."""

    root = native_root.resolve()
    records: dict[tuple[str, int], dict[str, str]] = {}
    for instrument, child in (("Agilent-1", "Agilent 1"), ("Agilent-3", "Agilent 3")):
        directory = root / "Agilent" / child
        if not directory.is_dir():
            continue
        spc = _scan_index(directory, "*.spc", r"Scan\s+(\d+)")
        aggregates = sorted(directory.glob("*.csv"))
        if len(aggregates) != 1:
            raise ValueError(f"{instrument} requires exactly one aggregate numeric export.")
        for scan_id, binary in spc.items():
            records[(instrument, scan_id)] = {
                "numeric_relative_path": _relative(root, aggregates[0]),
                "binary_relative_path": _relative(root, binary),
                "report_relative_path": "",
                "probe_relative_path": "",
                "dark_relative_path": "",
                "source_format": "aggregate_csv",
            }

    for instrument, child in (
        ("Mira-1", "Mira 1"),
        ("Mira-2", "Mira 2"),
        ("Mira-3", "Mira 3"),
    ):
        directory = root / "Mira" / child
        if not directory.is_dir():
            continue
        csv_index = _scan_index(directory, "*.csv", r"Scan\s+(\d+)")
        spc_index = _scan_index(directory, "*.spc", r"Scan\s+(\d+)")
        pdf_index = _scan_index(directory, "*.pdf", r"Scan\s+(\d+)")
        for scan_id, numeric in csv_index.items():
            records[(instrument, scan_id)] = {
                "numeric_relative_path": _relative(root, numeric),
                "binary_relative_path": _relative(root, spc_index.get(scan_id)),
                "report_relative_path": _relative(root, pdf_index.get(scan_id)),
                "probe_relative_path": "",
                "dark_relative_path": "",
                "source_format": "two_column_csv",
            }

    for instrument, child in (
        ("Pendar-1", "Pendar 1"),
        ("Pendar-2", "Pendar 2"),
        ("Pendar-3", "Pendar 3"),
    ):
        directory = root / "Pendar" / child
        if not directory.is_dir():
            continue
        csv_index = _scan_index(directory, "*.csv", r"^[^-]+-(\d+)-")
        for scan_id, numeric in csv_index.items():
            records[(instrument, scan_id)] = {
                "numeric_relative_path": _relative(root, numeric),
                "binary_relative_path": _relative(root, numeric.with_suffix(".spc")),
                "report_relative_path": _relative(root, numeric.with_suffix(".pdf")),
                "probe_relative_path": _relative(root, numeric.with_suffix(".prb")),
                "dark_relative_path": "",
                "source_format": "headered_two_column_csv",
            }

    for instrument, child in (("RMX-1", "RMX 1"), ("RMX-2", "RMX 2")):
        directory = root / "RMX" / child
        if not directory.is_dir():
            continue
        txt_index = _scan_index(directory, "Scan*.txt", r"Scan(\d+)")
        for scan_id, numeric in txt_index.items():
            records[(instrument, scan_id)] = {
                "numeric_relative_path": _relative(root, numeric),
                "binary_relative_path": _relative(root, numeric.with_suffix(".spc")),
                "report_relative_path": "",
                "probe_relative_path": "",
                "dark_relative_path": _relative(
                    root, numeric.with_name(f"{numeric.stem}-dark.spc")
                ),
                "source_format": "tagged_text",
            }
    return records


def _resolved(native_root: Path, relative: str) -> Path:
    path = (native_root / relative).resolve()
    if not relative or not is_relative_to(path, native_root) or not path.is_file():
        raise ValueError("A declared native source is missing or outside ATLAS_NATIVE_ROOT.")
    return path


def load_native_spectrum(
    native_root: Path,
    record: dict[str, Any],
    aggregate_cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one native spectrum from a sanitized indexed record."""

    cache = aggregate_cache if aggregate_cache is not None else {}
    path = _resolved(native_root, str(record["numeric_relative_path"]))
    source_format = str(record["source_format"])
    scan_id = int(record["source_scan_id"])
    if source_format == "aggregate_csv":
        key = record["numeric_relative_path"]
        if key not in cache:
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                first_row = next(csv.reader(handle))
            values = np.loadtxt(path, delimiter=",", skiprows=2)
            scans: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for index in range(values.shape[1] // 4):
                match = re.search(r"\d+", first_row[4 * index + 3])
                if match:
                    scans[int(match.group())] = (
                        values[:, 4 * index],
                        values[:, 4 * index + 1],
                    )
            cache[key] = scans
        return cache[key][scan_id]
    if source_format == "two_column_csv":
        values = np.loadtxt(path, delimiter=",")
        return values[:, 0], values[:, 1]
    if source_format == "headered_two_column_csv":
        values = np.loadtxt(path, delimiter=",", skiprows=1)
        return values[:, 0], values[:, 1]
    if source_format == "tagged_text":
        lines = path.read_text(errors="replace").splitlines()
        start = next(index for index, line in enumerate(lines) if line.startswith("spectrum "))
        count = int(lines[start].split()[1])
        values = np.asarray(
            [
                [float(value) for value in line.split()]
                for line in lines[start + 1 : start + 1 + count]
            ]
        )
        return values[:, 0], values[:, 1]
    raise ValueError(f"Unsupported native source format: {source_format}")


def _constant_edge_count(values: np.ndarray, *, right: bool = False) -> int:
    work = np.asarray(values, dtype=float)[::-1] if right else np.asarray(values, dtype=float)
    if len(work) < 2:
        return len(work)
    tolerance = max(float(np.nanmax(np.abs(work))), 1.0) * 1e-12
    changes = np.flatnonzero(np.abs(np.diff(work)) > tolerance)
    return int(changes[0] + 1) if len(changes) else len(work)


def spectrum_diagnostics(axis: np.ndarray, intensity: np.ndarray) -> dict[str, Any]:
    """Compute the frozen native-axis, source-hash, and numeric-QC record."""

    x = np.asarray(axis, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    step = np.diff(x)
    leading = _constant_edge_count(y)
    trailing = _constant_edge_count(y, right=True)
    effective_min = x[min(leading, max(len(x) - 1, 0))] if leading >= 5 else x[0]
    effective_max = x[max(0, len(x) - 1 - trailing)] if trailing >= 5 else x[-1]
    baseline_window = min(101, len(y) if len(y) % 2 else len(y) - 1)
    baseline_window = max(3, baseline_window)
    baseline = gaussian_filter1d(
        percentile_filter(y, percentile=10, size=baseline_window, mode="nearest"),
        sigma=max(1.0, baseline_window / 5.0),
    )
    first = np.diff(y)
    second = np.diff(y, n=2)
    second_median = float(np.median(second)) if len(second) else 0.0
    second_mad = float(np.median(np.abs(second - second_median))) if len(second) else 0.0
    spike_threshold = max(12.0 * second_mad, np.finfo(float).eps)
    spike_count = int(np.sum(np.abs(second - second_median) > spike_threshold))
    dynamic = max(float(np.ptp(y)), np.finfo(float).eps)
    if not finite.all():
        status = "non_finite"
    elif len(x) < 2 or not np.all(step > 0):
        status = "invalid_axis"
    elif float(np.ptp(y)) == 0.0:
        status = "constant"
    else:
        status = "pass"
    return {
        "numeric_qc_status": status,
        "n_points": len(x),
        "finite_fraction": float(np.mean(finite)),
        "axis_min_cm1": float(x.min()),
        "axis_max_cm1": float(x.max()),
        "axis_step_median_cm1": float(np.median(step)),
        "axis_step_min_cm1": float(step.min()),
        "axis_step_max_cm1": float(step.max()),
        "axis_strictly_increasing": bool(np.all(step > 0)),
        "axis_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "intensity_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "leading_constant_points": leading,
        "trailing_constant_points": trailing,
        "effective_axis_min_cm1": float(effective_min),
        "effective_axis_max_cm1": float(effective_max),
        "intensity_min": float(y.min()),
        "intensity_max": float(y.max()),
        "intensity_range": dynamic,
        "negative_fraction": float(np.mean(y < 0)),
        "first_difference_noise_mad": float(
            np.median(np.abs(first - np.median(first))) / 0.67448975
        ),
        "spike_fraction_proxy": float(spike_count / max(len(y) - 2, 1)),
        "baseline_energy_fraction_proxy": float(
            np.linalg.norm(baseline) / max(np.linalg.norm(y), np.finfo(float).eps)
        ),
        "baseline_span_fraction_proxy": float(np.ptp(baseline) / dynamic),
    }


def native_source_audit(native_root: Path) -> dict[str, Any]:
    """Return aggregate-only archive facts suitable for a sanitized private report."""

    files = sorted(path for path in native_root.rglob("*") if path.is_file())
    extensions: dict[str, int] = {}
    total_bytes = 0
    for path in files:
        key = path.suffix.lower() or "[none]"
        extensions[key] = extensions.get(key, 0) + 1
        total_bytes += path.stat().st_size
    notes = [path for path in files if path.name.lower() == "notes.txt"]
    note_text = notes[0].read_text(errors="replace").lower() if len(notes) == 1 else ""
    return {
        "schema_version": "p01-private-source-audit-v1",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "extension_counts": dict(sorted(extensions.items())),
        "note_file_count": len(notes),
        "note_semantics": {
            "declares_na_as_no_sensor": "na (no sers sensor" in note_text,
            "declares_four_sensor_families": "total of 4" in note_text,
            "nonempty_line_count": len([line for line in note_text.splitlines() if line.strip()]),
        },
        "workbook_count": sum(path.suffix.lower() == ".xlsx" for path in files),
        "indexed_numeric_observations": len(index_native_sources(native_root)),
    }


def build_native_registry(
    manifest: pd.DataFrame,
    native_root: Path,
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Re-read, hash, and register every selected native source in manifest order."""

    index = index_native_sources(native_root)
    cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    file_hash_cache: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    spectra: list[tuple[np.ndarray, np.ndarray]] = []
    axis_hash_failures = 0
    intensity_hash_failures = 0
    for item in manifest.itertuples(index=False):
        key = (str(item.instrument), int(item.source_scan_id))
        if key not in index:
            raise ValueError(f"Native source index does not contain {key}.")
        source = {**index[key], "source_scan_id": key[1]}
        axis, intensity = load_native_spectrum(native_root, source, cache)
        diagnostics = spectrum_diagnostics(axis, intensity)
        axis_hash_failures += diagnostics["axis_sha256"] != item.axis_sha256
        intensity_hash_failures += diagnostics["intensity_sha256"] != item.intensity_sha256
        numeric_relative = source["numeric_relative_path"]
        if numeric_relative not in file_hash_cache:
            file_hash_cache[numeric_relative] = sha256_file(
                _resolved(native_root, numeric_relative)
            )
        source_id = (
            "SRC-"
            + sha256_value(
                {
                    "instrument": key[0],
                    "scan": key[1],
                    "axis_sha256": diagnostics["axis_sha256"],
                    "intensity_sha256": diagnostics["intensity_sha256"],
                }
            )[:20]
        )
        rows.append(
            {
                "observation_uid": item.observation_uid,
                "source_logical_id": source_id,
                "instrument": key[0],
                "source_scan_id": key[1],
                **source,
                "numeric_file_sha256": file_hash_cache[numeric_relative],
                **diagnostics,
            }
        )
        spectra.append((np.asarray(axis, dtype=float), np.asarray(intensity, dtype=float)))
    report = {
        "selected_rows": len(rows),
        "unique_numeric_files": len(file_hash_cache),
        "axis_hash_failures": int(axis_hash_failures),
        "intensity_hash_failures": int(intensity_hash_failures),
    }
    return pd.DataFrame(rows), spectra, report
