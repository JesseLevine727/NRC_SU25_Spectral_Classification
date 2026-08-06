"""Sanitized, path-free capture of the P00 execution environment."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from atlas_sers.governance.canonical import hash_relative_files, sha256_bytes, sha256_value

IGNORED_PARTS = {".pytest_cache", ".ruff_cache", "__pycache__", "build", "dist"}


def _command(args: list[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _public_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in project_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(project_root)
        if IGNORED_PARTS.intersection(relative.parts):
            continue
        if any(part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        files.append(path)
    return sorted(files)


def _dependency_lock() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return {name: packages[name] for name in sorted(packages)}


def _cpu_model() -> str:
    processor = platform.processor().strip()
    if processor:
        return processor
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unavailable"


def _gpu_inventory(repository_root: Path) -> list[dict[str, str]]:
    output = _command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        cwd=repository_root,
    )
    if not output:
        return []
    gpus: list[dict[str, str]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 3:
            gpus.append({"name": values[0], "driver_version": values[1], "memory_mib": values[2]})
    return gpus


def _cuda_compiler(repository_root: Path) -> str | None:
    output = _command(["nvcc", "--version"], cwd=repository_root)
    return output.splitlines()[-1].strip() if output else None


def _blas_inventory() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return []
    allowed = {
        "architecture",
        "internal_api",
        "num_threads",
        "prefix",
        "threading_layer",
        "user_api",
        "version",
    }
    return [
        {key: item[key] for key in sorted(allowed) if key in item} for item in threadpool_info()
    ]


def capture_provenance(
    *, repository_root: Path, project_root: Path, artifact_root: Path
) -> dict[str, Any]:
    """Capture reproducibility state while omitting local paths and environment values."""

    commit = _command(["git", "rev-parse", "HEAD"], cwd=repository_root)
    global_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=repository_root,
        capture_output=True,
        check=True,
    ).stdout
    project_relative = project_root.resolve().relative_to(repository_root.resolve()).as_posix()
    scoped_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--", project_relative],
        cwd=repository_root,
        capture_output=True,
        check=True,
    ).stdout
    public_manifest = hash_relative_files(project_root, _public_files(project_root))
    dependencies = _dependency_lock()
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(artifact_root.parent)
    provenance: dict[str, Any] = {
        "schema_version": "p00-environment-v1",
        "repository": {
            "commit": commit or "unavailable",
            "globally_dirty": bool(global_status),
            "global_dirty_state_sha256": sha256_bytes(global_status),
            "atlas_scaffold_dirty": bool(scoped_status),
            "atlas_dirty_state_sha256": sha256_bytes(scoped_status),
            "atlas_public_tree_sha256": sha256_value(public_manifest),
            "atlas_public_file_count": len(public_manifest),
        },
        "runtime": {
            "operating_system": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "compute": {
            "logical_cpu_count": os.cpu_count(),
            "cpu_model": _cpu_model(),
            "gpus": _gpu_inventory(repository_root),
            "cuda_compiler": _cuda_compiler(repository_root),
            "blas": _blas_inventory(),
        },
        "storage": {
            "artifact_filesystem_total_bytes": disk.total,
            "artifact_filesystem_free_bytes_at_capture": disk.free,
        },
        "dependencies": dependencies,
        "dependency_lock_sha256": sha256_value(dependencies),
    }
    protected_environment = {
        "repository": provenance["repository"],
        "runtime": provenance["runtime"],
        "compute": provenance["compute"],
        "artifact_filesystem_total_bytes": disk.total,
        "dependency_lock_sha256": provenance["dependency_lock_sha256"],
    }
    provenance["protected_environment_sha256"] = sha256_value(protected_environment)
    provenance["environment_sha256"] = sha256_value(provenance)
    return provenance
