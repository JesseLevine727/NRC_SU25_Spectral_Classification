#!/usr/bin/env python3
"""Write a SHA-256 catalog for an open-world SERS study bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("Workspace/sers_open_world_shift/open_world_v1"),
    )
    parser.add_argument(
        "--output-name", default="study_artifact_hashes.json"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_digest(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.study_dir
    output = root / args.output_name
    files: list[Path] = []
    caches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(root)
        if "logs" in relative.parts:
            continue
        if "cache" in relative.parts or "search_shards" in relative.parts:
            caches.append(path)
        else:
            files.append(path)
    source_files = [
        Path("configs/sers_open_world_shift_v1.json"),
        Path("configs/sers_open_world_baselines_v1.json"),
        Path("configs/sers_open_world_neural_controls_v1.json"),
        Path("configs/sers_open_world_cnn_v1.json"),
        Path("configs/sers_open_world_cnn_domain_v1.json"),
        Path("configs/sers_open_world_cnn_controls_v1.json"),
        Path("scripts/freeze_sers_open_world_partitions.py"),
        Path("scripts/validate_sers_open_world_partitions.py"),
        Path("scripts/freeze_nato_sers_preprocessing.py"),
        Path("scripts/sers_baseline_common.py"),
        Path("scripts/sers_classical_benchmark_common.py"),
        Path("scripts/sers_contrastive_common.py"),
        Path("scripts/sers_vae_common.py"),
        Path("scripts/sers_structured_vae_common.py"),
        Path("scripts/run_sers_standard_vae_final.py"),
        Path("scripts/run_sers_structured_vae_selection.py"),
        Path("scripts/sers_open_world_common.py"),
        Path("scripts/sers_open_world_neural_common.py"),
        Path("scripts/sers_open_world_cnn_common.py"),
        Path("scripts/run_sers_open_world_baselines.py"),
        Path("scripts/run_sers_open_world_neural_controls.py"),
        Path("scripts/run_sers_open_world_cnn.py"),
        Path("scripts/run_sers_open_world_cnn_domain.py"),
        Path("scripts/run_sers_open_world_cnn_controls.py"),
        Path("scripts/validate_sers_open_world_cnn.py"),
        Path("scripts/finalize_sers_open_world_shift_study.py"),
        Path("scripts/plot_sers_open_world_shift_study.py"),
        Path("scripts/rebuild_sers_open_world_shift_study.py"),
        Path("scripts/compare_sers_open_world_shift_rebuild.py"),
        Path("scripts/hash_sers_open_world_shift_study.py"),
    ]
    payload = {
        "study_dir": str(root.resolve()),
        "artifact_count": len(files),
        "artifacts": {
            str(path.relative_to(root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
        },
        "cache_inventory": {
            "files": len(caches),
            "bytes": sum(path.stat().st_size for path in caches),
            "aggregate_name_and_content_sha256": cache_digest(caches, root),
        },
        "source_files": {
            str(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in source_files
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "artifacts": len(files),
                "cache_files": len(caches),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
