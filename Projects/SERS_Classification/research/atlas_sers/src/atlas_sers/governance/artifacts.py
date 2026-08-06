"""Atomic and idempotent private artifact transactions for ATLAS phases."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file
from atlas_sers.paths import validate_private_roots


@dataclass(frozen=True)
class ArtifactLease:
    action: str
    run_id: str
    protected_state_sha256: str
    work_dir: Path | None
    final_dir: Path


class ArtifactStore:
    def __init__(
        self,
        *,
        artifact_root: Path,
        input_root: Path,
        project_root: Path,
        phase: str = "p00",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        validate_private_roots(
            input_root=input_root,
            output_root=artifact_root,
            public_project_root=project_root,
        )
        if not phase or not phase.isascii() or not phase.isalnum():
            raise ValueError("Artifact phase must be a non-empty ASCII alphanumeric value.")
        self.root = artifact_root.resolve()
        self.phase = phase.lower()
        self.phase_root = self.root / self.phase
        self.runs = self.phase_root / "runs"
        self.quarantine = self.phase_root / "quarantine"
        self.clock = clock or (lambda: datetime.now(UTC))
        self.runs.mkdir(parents=True, exist_ok=True)
        self.quarantine.mkdir(parents=True, exist_ok=True)

    def _quarantine_path(self, source: Path, *, run_id: str, reason: str) -> Path:
        stamp = self.clock().strftime("%Y%m%dT%H%M%SZ")
        base = self.quarantine / f"{run_id}--{stamp}--{reason}"
        destination = base
        counter = 1
        while destination.exists():
            destination = self.quarantine / f"{base.name}--{counter}"
            counter += 1
        os.replace(source, destination)
        return destination

    def _quarantine_existing(
        self,
        source: Path,
        *,
        run_id: str,
        reason: str,
        expected_hash: str,
        observed_hash: str | None,
    ) -> None:
        destination = self._quarantine_path(source, run_id=run_id, reason=reason)
        record = {
            "schema_version": "atlas-artifact-quarantine-v1",
            "phase": self.phase.upper(),
            "run_id": run_id,
            "reason": reason,
            "expected_protected_state_sha256": expected_hash,
            "observed_protected_state_sha256": observed_hash,
            "quarantined_at_utc": self.clock().isoformat(),
        }
        (destination / "QUARANTINE_REASON.json").write_bytes(
            canonical_json_bytes(record, pretty=True)
        )

    def begin(self, *, run_id: str, protected_state_sha256: str) -> ArtifactLease:
        final_dir = self.runs / run_id
        if final_dir.exists():
            state_path = final_dir / "_STATE.json"
            observed: str | None = None
            state: dict[str, object] = {}
            try:
                state = json.loads(state_path.read_text())
                observed_value = state.get("protected_state_sha256")
                observed = str(observed_value) if observed_value else None
            except (OSError, json.JSONDecodeError):
                pass
            expected_files = state.get("files", {})
            files_valid = isinstance(expected_files, dict) and all(
                (final_dir / name).is_file() and sha256_file(final_dir / name) == expected_hash
                for name, expected_hash in expected_files.items()
            )
            successful = (
                state.get("execution_status") == "complete"
                and state.get("scientific_status") == "pass"
            )
            if successful and observed == protected_state_sha256 and files_valid:
                return ArtifactLease(
                    "verified_skip", run_id, protected_state_sha256, None, final_dir
                )
            self._quarantine_existing(
                final_dir,
                run_id=run_id,
                reason="conflicting_or_incomplete_final",
                expected_hash=protected_state_sha256,
                observed_hash=observed,
            )

        for stale in sorted(self.runs.glob(f".{run_id}.tmp-*")):
            self._quarantine_existing(
                stale,
                run_id=run_id,
                reason="stale_temporary_run",
                expected_hash=protected_state_sha256,
                observed_hash=None,
            )
        work_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=self.runs))
        return ArtifactLease("new", run_id, protected_state_sha256, work_dir, final_dir)

    def commit(self, lease: ArtifactLease, *, scientific_status: str) -> Path:
        if lease.action != "new" or lease.work_dir is None:
            raise ValueError("Only a new artifact lease can be committed.")
        if lease.final_dir.exists():
            raise FileExistsError("A final run appeared during the artifact transaction.")
        files = {
            path.relative_to(lease.work_dir).as_posix(): sha256_file(path)
            for path in sorted(lease.work_dir.rglob("*"))
            if path.is_file()
        }
        state = {
            "schema_version": "atlas-artifact-state-v1",
            "phase": self.phase.upper(),
            "run_id": lease.run_id,
            "protected_state_sha256": lease.protected_state_sha256,
            "execution_status": "complete",
            "scientific_status": scientific_status,
            "files": files,
        }
        (lease.work_dir / "_STATE.json").write_bytes(canonical_json_bytes(state, pretty=True))
        os.replace(lease.work_dir, lease.final_dir)
        return lease.final_dir

    def quarantine_lease(self, lease: ArtifactLease, *, reason: str) -> Path:
        if lease.work_dir is None or not lease.work_dir.exists():
            raise ValueError("Artifact lease has no active temporary directory.")
        destination = self._quarantine_path(lease.work_dir, run_id=lease.run_id, reason=reason)
        record = {
            "schema_version": "atlas-artifact-quarantine-v1",
            "phase": self.phase.upper(),
            "run_id": lease.run_id,
            "reason": reason,
            "expected_protected_state_sha256": lease.protected_state_sha256,
            "observed_protected_state_sha256": None,
            "quarantined_at_utc": self.clock().isoformat(),
        }
        (destination / "QUARANTINE_REASON.json").write_bytes(
            canonical_json_bytes(record, pretty=True)
        )
        return destination
