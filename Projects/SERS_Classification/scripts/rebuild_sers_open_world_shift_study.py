#!/usr/bin/env python3
"""Execute the complete open-world SERS study in a separate clean directory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


HELD = [
    "4_ANPP",
    "4_nitrophenol",
    "acetaminophen",
    "benzyl_fentanyl",
    "ethanol",
    "ethyl_paraoxon",
]
NEURAL = [
    "siamese_triplet",
    "supervised_contrastive",
    "standard_vae",
    "structured_vae",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild-dir",
        type=Path,
        default=Path(
            "Workspace/sers_open_world_shift/open_world_v1_clean_rebuild_20260730"
        ),
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("Workspace/nato_sers_field_trial/preprocessing_v2"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def environment() -> dict[str, str]:
    value = dict(os.environ)
    value.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    value.setdefault("OMP_NUM_THREADS", "1")
    value.setdefault("MKL_NUM_THREADS", "1")
    value.setdefault("OPENBLAS_NUM_THREADS", "1")
    value["PYTHONPATH"] = (
        "scripts"
        if not value.get("PYTHONPATH")
        else f"scripts:{value['PYTHONPATH']}"
    )
    return value


def command(
    script: str, *arguments: object
) -> list[str]:
    return [sys.executable, f"scripts/{script}", *map(str, arguments)]


def run(
    name: str,
    arguments: Sequence[str],
    log_dir: Path,
    env: dict[str, str],
) -> None:
    log_path = log_dir / f"{name}.log"
    print(f"START {name}", flush=True)
    with log_path.open("w") as handle:
        result = subprocess.run(
            list(arguments),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-30:])
        raise RuntimeError(f"{name} failed:\n{tail}")
    print(f"DONE {name}", flush=True)


def run_parallel(
    tasks: list[tuple[str, list[str]]],
    log_dir: Path,
    env: dict[str, str],
) -> None:
    processes: list[tuple[str, subprocess.Popen[bytes], object, Path]] = []
    for name, arguments in tasks:
        log_path = log_dir / f"{name}.log"
        handle = log_path.open("wb")
        process = subprocess.Popen(
            arguments,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        processes.append((name, process, handle, log_path))
        print(f"START {name} pid={process.pid}", flush=True)
    failures: list[str] = []
    for name, process, handle, log_path in processes:
        returncode = process.wait()
        handle.close()
        if returncode != 0:
            tail = "\n".join(log_path.read_text().splitlines()[-30:])
            failures.append(f"{name}:\n{tail}")
        else:
            print(f"DONE {name}", flush=True)
    if failures:
        raise RuntimeError("\n\n".join(failures))


def main() -> None:
    args = parse_args()
    root = args.rebuild_dir
    audit_dir = root.parent / f"{root.name}_orchestration"
    if root.exists() and not args.resume:
        raise FileExistsError(
            f"Clean rebuild target already exists: {root}; use --resume only "
            "after inspecting the prior attempt"
        )
    if audit_dir.exists() and not args.resume:
        raise FileExistsError(
            f"Clean rebuild audit target already exists: {audit_dir}; remove "
            "the inspected prior attempt before starting a clean rebuild"
        )
    root.mkdir(parents=True, exist_ok=True)
    logs = audit_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = environment()
    provenance = {
        "status": "running",
        "python": sys.executable,
        "rebuild_dir": str(root.resolve()),
        "bundle": str(args.bundle.resolve()),
        "device": args.device,
        "resume": args.resume,
    }
    (audit_dir / "rebuild_registry.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    run(
        "01_freeze_partitions",
        command(
            "freeze_sers_open_world_partitions.py",
            "--bundle",
            args.bundle,
            "--output-dir",
            root,
        ),
        logs,
        env,
    )
    run(
        "02_validate_partitions",
        command(
            "validate_sers_open_world_partitions.py",
            "--bundle",
            args.bundle,
            "--output-dir",
            root,
        ),
        logs,
        env,
    )
    run(
        "03_baselines",
        command(
            "run_sers_open_world_baselines.py",
            "--bundle",
            args.bundle,
            "--partition-dir",
            root,
            "--output-dir",
            root / "baselines",
            "--stage",
            "all",
            "--jobs",
            "16",
        ),
        logs,
        env,
    )
    run_parallel(
        [
            (
                f"04_neural_{method}",
                command(
                    "run_sers_open_world_neural_controls.py",
                    "--method",
                    method,
                    "--stage",
                    "all",
                    "--bundle",
                    args.bundle,
                    "--partition-dir",
                    root,
                    "--output-dir",
                    root / "neural_controls",
                    "--device",
                    args.device,
                ),
            )
            for method in NEURAL
        ],
        logs,
        env,
    )
    run_parallel(
        [
            (
                f"05_cnn_surrogate_{held}",
                command(
                    "run_sers_open_world_cnn.py",
                    "--stage",
                    "surrogate",
                    "--held-unknown",
                    held,
                    "--bundle",
                    args.bundle,
                    "--partition-dir",
                    root,
                    "--output-dir",
                    root / "cnn",
                    "--device",
                    args.device,
                ),
            )
            for held in HELD
        ],
        logs,
        env,
    )
    run(
        "06_cnn_select",
        command(
            "run_sers_open_world_cnn.py",
            "--stage",
            "select",
            "--bundle",
            args.bundle,
            "--partition-dir",
            root,
            "--output-dir",
            root / "cnn",
            "--device",
            args.device,
        ),
        logs,
        env,
    )
    run(
        "07_cnn_domain_epochs",
        command(
            "run_sers_open_world_cnn_domain.py",
            "--stage",
            "epochs",
            "--bundle",
            args.bundle,
            "--partition-dir",
            root,
            "--cnn-dir",
            root / "cnn",
            "--device",
            args.device,
        ),
        logs,
        env,
    )
    run(
        "08_cnn_calibration",
        command(
            "run_sers_open_world_cnn.py",
            "--stage",
            "calibration",
            "--bundle",
            args.bundle,
            "--partition-dir",
            root,
            "--output-dir",
            root / "cnn",
            "--device",
            args.device,
        ),
        logs,
        env,
    )
    run_parallel(
        [
            (
                "09a_cnn_final",
                command(
                    "run_sers_open_world_cnn.py",
                    "--stage",
                    "final",
                    "--bundle",
                    args.bundle,
                    "--partition-dir",
                    root,
                    "--output-dir",
                    root / "cnn",
                    "--device",
                    args.device,
                ),
            ),
            *[
                (
                    f"09b_cnn_domain_shard_{index:02d}",
                    command(
                        "run_sers_open_world_cnn_domain.py",
                        "--stage",
                        "domain",
                        "--shard-index",
                        index,
                        "--shard-count",
                        6,
                        "--bundle",
                        args.bundle,
                        "--partition-dir",
                        root,
                        "--cnn-dir",
                        root / "cnn",
                        "--device",
                        args.device,
                    ),
                )
                for index in range(6)
            ],
            (
                "09c_cnn_controls",
                command(
                    "run_sers_open_world_cnn_controls.py",
                    "--stage",
                    "all",
                    "--bundle",
                    args.bundle,
                    "--partition-dir",
                    root,
                    "--cnn-dir",
                    root / "cnn",
                    "--device",
                    args.device,
                ),
            ),
        ],
        logs,
        env,
    )
    run(
        "10a_cnn_domain_merge",
        command(
            "run_sers_open_world_cnn_domain.py",
            "--stage",
            "merge",
            "--shard-count",
            "6",
            "--bundle",
            args.bundle,
            "--partition-dir",
            root,
            "--cnn-dir",
            root / "cnn",
            "--device",
            args.device,
        ),
        logs,
        env,
    )
    run(
        "10b_cnn_domain_finalize",
        command(
            "run_sers_open_world_cnn_domain.py",
            "--stage",
            "finalize",
            "--bundle",
            args.bundle,
            "--partition-dir",
            root,
            "--cnn-dir",
            root / "cnn",
            "--device",
            args.device,
        ),
        logs,
        env,
    )
    run(
        "11_validate_complete",
        command(
            "validate_sers_open_world_cnn.py",
            "--bundle",
            args.bundle,
            "--partition-dir",
            root,
            "--cnn-dir",
            root / "cnn",
        ),
        logs,
        env,
    )
    run(
        "12_finalize",
        command(
            "finalize_sers_open_world_shift_study.py",
            "--study-dir",
            root,
            "--docs-report",
            root / "FINAL_REPORT_DOC_COPY.md",
        ),
        logs,
        env,
    )
    run(
        "13_figures",
        command(
            "plot_sers_open_world_shift_study.py",
            "--study-dir",
            root,
        ),
        logs,
        env,
    )
    provenance["status"] = "complete"
    (root / "rebuild_registry.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    (audit_dir / "rebuild_registry.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    run(
        "14_hash_catalog",
        command(
            "hash_sers_open_world_shift_study.py",
            "--study-dir",
            root,
        ),
        logs,
        env,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "rebuild_dir": str(root),
                "logs": str(logs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
