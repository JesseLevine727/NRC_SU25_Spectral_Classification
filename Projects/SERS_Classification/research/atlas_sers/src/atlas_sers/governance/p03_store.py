"""Atomic resumable shard storage for the large P03 classical execution."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file


@dataclass(frozen=True)
class ShardLease:
    action: str
    shard_id: int
    protected_state_sha256: str
    temporary_dir: Path | None
    final_dir: Path
    lock_path: Path | None


class P03ShardStore:
    """Commit one bounded fit shard at a time without overwriting evidence."""

    def __init__(self, *, run_root: Path) -> None:
        self.run_root = run_root.resolve()
        self.shards = self.run_root / "shards"
        self.locks = self.run_root / "locks"
        self.quarantine = self.run_root / "quarantine"
        self.shards.mkdir(parents=True, exist_ok=True)
        self.locks.mkdir(parents=True, exist_ok=True)
        self.quarantine.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _name(shard_id: int) -> str:
        if shard_id < 0:
            raise ValueError("Shard ID must be nonnegative.")
        return f"shard-{shard_id:06d}"

    @staticmethod
    def _valid_final(path: Path, protected_state_sha256: str) -> bool:
        try:
            state = json.loads((path / "_STATE.json").read_text())
        except (OSError, json.JSONDecodeError):
            return False
        files = state.get("files")
        return (
            state.get("execution_status") == "complete"
            and state.get("protected_state_sha256") == protected_state_sha256
            and isinstance(files, dict)
            and all(
                (path / name).is_file() and sha256_file(path / name) == digest
                for name, digest in files.items()
            )
        )

    def _quarantine(self, path: Path, *, reason: str) -> Path:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        destination = self.quarantine / f"{path.name}--{stamp}--{reason}"
        os.replace(path, destination)
        return destination

    def begin(self, *, shard_id: int, protected_state_sha256: str) -> ShardLease:
        name = self._name(shard_id)
        final = self.shards / name
        if final.exists():
            if self._valid_final(final, protected_state_sha256):
                return ShardLease(
                    "verified_skip", shard_id, protected_state_sha256, None, final, None
                )
            self._quarantine(final, reason="corrupt_or_conflicting_final")
        lock = self.locks / f"{name}.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise RuntimeError(f"P03 shard {shard_id} is already leased.") from error
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.close(descriptor)
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=self.shards))
        return ShardLease("new", shard_id, protected_state_sha256, temporary, final, lock)

    def commit(self, lease: ShardLease) -> Path:
        if lease.action != "new" or lease.temporary_dir is None or lease.lock_path is None:
            raise ValueError("Only a new shard lease can be committed.")
        files = {
            path.relative_to(lease.temporary_dir).as_posix(): sha256_file(path)
            for path in sorted(lease.temporary_dir.rglob("*"))
            if path.is_file()
        }
        state = {
            "schema_version": "p03-shard-state-v1",
            "shard_id": lease.shard_id,
            "protected_state_sha256": lease.protected_state_sha256,
            "execution_status": "complete",
            "files": files,
        }
        (lease.temporary_dir / "_STATE.json").write_bytes(canonical_json_bytes(state, pretty=True))
        if lease.final_dir.exists():
            self._quarantine(lease.temporary_dir, reason="concurrent_final_appeared")
            lease.lock_path.unlink(missing_ok=True)
            raise FileExistsError("A P03 shard final appeared during commit.")
        os.replace(lease.temporary_dir, lease.final_dir)
        lease.lock_path.unlink(missing_ok=True)
        return lease.final_dir

    def abort(self, lease: ShardLease, *, reason: str) -> Path:
        if lease.temporary_dir is None or lease.lock_path is None:
            raise ValueError("Shard lease has no active temporary directory.")
        destination = self._quarantine(lease.temporary_dir, reason=reason)
        lease.lock_path.unlink(missing_ok=True)
        record = {
            "schema_version": "p03-shard-abort-v1",
            "shard_id": lease.shard_id,
            "protected_state_sha256": lease.protected_state_sha256,
            "reason": reason,
        }
        (destination / "QUARANTINE_REASON.json").write_bytes(
            canonical_json_bytes(record, pretty=True)
        )
        return destination

    def validation_table(self, expected: dict[int, str]) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for shard_id, protected_hash in sorted(expected.items()):
            path = self.shards / self._name(shard_id)
            records.append(
                {
                    "shard_id": shard_id,
                    "protected_state_sha256": protected_hash,
                    "exists": path.is_dir(),
                    "valid": path.is_dir() and self._valid_final(path, protected_hash),
                }
            )
        return records
