#!/usr/bin/env python3
"""Compare canonical and clean-rebuild SERS open-world study artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import pandas as pd


NEURAL = [
    "siamese_triplet",
    "supervised_contrastive",
    "standard_vae",
    "structured_vae",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("Workspace/sers_open_world_shift/open_world_v1"),
    )
    parser.add_argument(
        "--rebuild",
        type=Path,
        default=Path(
            "Workspace/sers_open_world_shift/open_world_v1_clean_rebuild_20260730"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_csvs(root: Path) -> list[Path]:
    files = list(root.glob("*.csv"))
    files.extend((root / "baselines").glob("*.csv"))
    for method in NEURAL:
        files.extend((root / "neural_controls" / method).glob("*.csv"))
    files.extend((root / "cnn").glob("*.csv"))
    files.extend((root / "summaries").glob("*.csv"))
    return sorted(path for path in files if path.is_file())


def normalize_frame(relative: Path, frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if relative.name == "domain_epoch_registry.csv" and "source_file" in result:
        result["source_file"] = result["source_file"].map(
            lambda value: Path(str(value)).name
        )
    if relative.parts[:1] == ("neural_controls",):
        for column in ("checkpoint", "history"):
            if column in result:
                result[column] = result[column].map(
                    lambda value: (
                        str(value).split("neural_controls/", 1)[-1]
                        if "neural_controls/" in str(value)
                        else str(value)
                    )
                )
    return result


def frames_equal(
    relative: Path, left: Path, right: Path
) -> tuple[bool, dict[str, Any]]:
    first = normalize_frame(relative, pd.read_csv(left))
    second = normalize_frame(relative, pd.read_csv(right))
    if list(first.columns) != list(second.columns) or first.shape != second.shape:
        return False, {
            "canonical_shape": first.shape,
            "rebuild_shape": second.shape,
            "columns_equal": list(first.columns) == list(second.columns),
        }
    numeric = [
        column
        for column in first.columns
        if pd.api.types.is_numeric_dtype(first[column])
        and pd.api.types.is_numeric_dtype(second[column])
    ]
    nonnumeric = [column for column in first.columns if column not in numeric]
    numeric_equal = all(
        np.allclose(
            first[column].to_numpy(float),
            second[column].to_numpy(float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        for column in numeric
    )
    string_equal = all(
        first[column].fillna("<NA>").astype(str).equals(
            second[column].fillna("<NA>").astype(str)
        )
        for column in nonnumeric
    )
    return numeric_equal and string_equal, {
        "shape": first.shape,
        "numeric_columns": len(numeric),
        "nonnumeric_columns": len(nonnumeric),
        "numeric_exact": numeric_equal,
        "nonnumeric_exact": string_equal,
    }


def compare_csvs(
    canonical: Path, rebuild: Path
) -> tuple[list[dict[str, Any]], bool]:
    left = {
        path.relative_to(canonical): path
        for path in deterministic_csvs(canonical)
    }
    right = {
        path.relative_to(rebuild): path
        for path in deterministic_csvs(rebuild)
    }
    inventory_equal = set(left) == set(right)
    rows: list[dict[str, Any]] = []
    for relative in sorted(set(left) | set(right)):
        if relative not in left or relative not in right:
            rows.append(
                {
                    "artifact": str(relative),
                    "type": "csv",
                    "status": "missing",
                    "byte_identical": False,
                    "semantic_exact": False,
                }
            )
            continue
        byte_identical = sha256(left[relative]) == sha256(right[relative])
        semantic, details = frames_equal(
            relative, left[relative], right[relative]
        )
        rows.append(
            {
                "artifact": str(relative),
                "type": "csv",
                "status": "pass" if semantic else "fail",
                "byte_identical": byte_identical,
                "semantic_exact": semantic,
                "details": json.dumps(details, sort_keys=True),
            }
        )
    return rows, inventory_equal and all(
        row["semantic_exact"] for row in rows
    )


def compare_npz(
    canonical: Path, rebuild: Path
) -> tuple[list[dict[str, Any]], bool]:
    def arrays_equal(first: np.ndarray, second: np.ndarray) -> bool:
        if first.dtype.kind in {"f", "c"} and second.dtype.kind in {"f", "c"}:
            return np.array_equal(first, second, equal_nan=True)
        return np.array_equal(first, second)

    left_dir = canonical / "cnn" / "cache" / "domain"
    right_dir = rebuild / "cnn" / "cache" / "domain"
    left = {path.name: path for path in left_dir.glob("*.npz")}
    right = {path.name: path for path in right_dir.glob("*.npz")}
    rows: list[dict[str, Any]] = []
    for name in sorted(set(left) | set(right)):
        exact = False
        keys_equal = False
        if name in left and name in right:
            with (
                np.load(left[name], allow_pickle=False) as first,
                np.load(right[name], allow_pickle=False) as second,
            ):
                keys_equal = set(first.files) == set(second.files)
                exact = keys_equal and all(
                    arrays_equal(first[key], second[key])
                    for key in first.files
                )
        rows.append(
            {
                "artifact": f"cnn/cache/domain/{name}",
                "type": "npz",
                "status": "pass" if exact else "fail",
                "byte_identical": (
                    name in left
                    and name in right
                    and sha256(left[name]) == sha256(right[name])
                ),
                "semantic_exact": exact,
                "details": json.dumps({"keys_equal": keys_equal}),
            }
        )
    return rows, set(left) == set(right) and all(
        row["semantic_exact"] for row in rows
    )


def compare_json_caches(
    canonical: Path, rebuild: Path
) -> tuple[list[dict[str, Any]], bool]:
    directories = [
        Path("cnn/cache/domain"),
        Path("cnn/cache/controls_permutation"),
        Path("cnn/cache/controls_sensitivity"),
    ]
    rows: list[dict[str, Any]] = []
    inventory_equal = True
    for relative_dir in directories:
        left = {
            path.name: path
            for path in (canonical / relative_dir).glob("*.json")
        }
        right = {
            path.name: path
            for path in (rebuild / relative_dir).glob("*.json")
        }
        inventory_equal &= set(left) == set(right)
        for name in sorted(set(left) | set(right)):
            exact = (
                name in left
                and name in right
                and sha256(left[name]) == sha256(right[name])
            )
            rows.append(
                {
                    "artifact": str(relative_dir / name),
                    "type": "json_cache",
                    "status": "pass" if exact else "fail",
                    "byte_identical": exact,
                    "semantic_exact": exact,
                }
            )
    return rows, inventory_equal and all(
        row["semantic_exact"] for row in rows
    )


def compare_surrogate_caches(
    canonical: Path, rebuild: Path
) -> tuple[list[dict[str, Any]], bool]:
    relative_dir = Path("cnn/cache/surrogate")
    left = {
        path.name: path
        for path in (canonical / relative_dir).glob("*.csv")
    }
    right = {
        path.name: path
        for path in (rebuild / relative_dir).glob("*.csv")
    }
    rows: list[dict[str, Any]] = []
    for name in sorted(set(left) | set(right)):
        exact = (
            name in left
            and name in right
            and sha256(left[name]) == sha256(right[name])
        )
        rows.append(
            {
                "artifact": str(relative_dir / name),
                "type": "surrogate_csv_cache",
                "status": "pass" if exact else "fail",
                "byte_identical": exact,
                "semantic_exact": exact,
            }
        )
    return rows, set(left) == set(right) and all(
        row["semantic_exact"] for row in rows
    )


def compare_reports_and_pngs(
    canonical: Path, rebuild: Path
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    reports = [Path("FINAL_REPORT.md")]
    for relative in reports:
        left, right = canonical / relative, rebuild / relative
        exact = left.exists() and right.exists() and left.read_bytes() == right.read_bytes()
        rows.append(
            {
                "artifact": str(relative),
                "type": "report",
                "status": "pass" if exact else "fail",
                "byte_identical": exact,
                "semantic_exact": exact,
            }
        )
    left_png = {
        path.name: path for path in (canonical / "figures").glob("*.png")
    }
    right_png = {
        path.name: path for path in (rebuild / "figures").glob("*.png")
    }
    inventory_equal = set(left_png) == set(right_png)
    for name in sorted(set(left_png) | set(right_png)):
        pixel_exact = False
        if name in left_png and name in right_png:
            first = mpimg.imread(left_png[name])
            second = mpimg.imread(right_png[name])
            pixel_exact = np.array_equal(first, second, equal_nan=True)
        rows.append(
            {
                "artifact": f"figures/{name}",
                "type": "png",
                "status": "pass" if pixel_exact else "fail",
                "byte_identical": (
                    name in left_png
                    and name in right_png
                    and sha256(left_png[name]) == sha256(right_png[name])
                ),
                "semantic_exact": pixel_exact,
            }
        )
    return rows, inventory_equal and all(
        row["semantic_exact"] for row in rows
    )


def validation_passes(root: Path) -> bool:
    partition = json.loads((root / "validation_report.json").read_text())
    cnn = json.loads((root / "cnn" / "validation_report.json").read_text())
    return partition["status"] == "pass" and cnn["status"] == "pass"


def main() -> None:
    args = parse_args()
    csv_rows, csv_pass = compare_csvs(args.canonical, args.rebuild)
    npz_rows, npz_pass = compare_npz(args.canonical, args.rebuild)
    cache_rows, cache_pass = compare_json_caches(
        args.canonical, args.rebuild
    )
    surrogate_cache_rows, surrogate_cache_pass = (
        compare_surrogate_caches(args.canonical, args.rebuild)
    )
    visual_rows, visual_pass = compare_reports_and_pngs(
        args.canonical, args.rebuild
    )
    rows = (
        csv_rows
        + npz_rows
        + cache_rows
        + surrogate_cache_rows
        + visual_rows
    )
    validation = validation_passes(args.canonical) and validation_passes(
        args.rebuild
    )
    status = (
        "pass"
        if csv_pass
        and npz_pass
        and cache_pass
        and surrogate_cache_pass
        and visual_pass
        and validation
        else "fail"
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.canonical / "rebuild_comparison.csv", index=False)
    frame.to_csv(args.rebuild / "rebuild_comparison.csv", index=False)
    report = {
        "status": status,
        "canonical": str(args.canonical.resolve()),
        "rebuild": str(args.rebuild.resolve()),
        "validated_bundles": validation,
        "artifacts_compared": len(rows),
        "semantic_exact": int(frame["semantic_exact"].sum()),
        "byte_identical": int(frame["byte_identical"].sum()),
        "csv_pass": csv_pass,
        "domain_npz_array_pass": npz_pass,
        "cache_json_pass": cache_pass,
        "surrogate_cache_pass": surrogate_cache_pass,
        "report_and_png_pass": visual_pass,
        "expected_byte_variance": (
            "Compressed NPZ container metadata and normalized provenance path "
            "fields may differ; every stored array and normalized table must "
            "remain exactly equal."
        ),
    }
    for root in (args.canonical, args.rebuild):
        (root / "rebuild_comparison.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(json.dumps(report, indent=2))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
