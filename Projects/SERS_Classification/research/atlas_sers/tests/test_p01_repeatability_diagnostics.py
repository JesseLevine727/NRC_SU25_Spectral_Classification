"""Test-only diagnostics for P01 artifact-reuse mismatches.

Read-only and side-effect-free: these helpers summarize metadata a synthetic
test run already produced, never write outside fixture creation, never mutate
inputs, never print on successful reuse, and never touch production artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEBUG_PREFIX = "[DEBUG-p01-reuse]"

# Mirrors the protected components in atlas_sers.governance.provenance. Free
# disk space is deliberately excluded because it is volatile between builds.
PROTECTED_ENVIRONMENT_KEYS = (
    "repository",
    "runtime",
    "compute",
    "dependency_lock_sha256",
)

# Bound for per-element diagnostic expansion of large lists.
ELEMENT_LIMIT = 20


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


MISSING = _Missing()


@dataclass(frozen=True)
class ReuseSnapshot:
    state: Any
    protected_state: Any
    environment_lock: Any


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return MISSING


def snapshot_reuse_state(run_dir: Path) -> ReuseSnapshot:
    """Capture reuse metadata from exactly the run's own files (read-only).

    No parent-directory fallback is used, so a stray sibling file cannot be
    mistaken for this run's state.
    """
    return ReuseSnapshot(
        state=_read_json(run_dir / "_STATE.json"),
        protected_state=_read_json(run_dir / "protected_state.json"),
        environment_lock=_read_json(run_dir / "environment_lock.json"),
    )


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=repr
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str | None:
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                hasher.update(block)
        return hasher.hexdigest()
    except OSError:
        return None


def _safe_name(name: Any) -> str:
    """Sanitize a diagnostic key or logical payload name for safe output."""
    text = str(name)
    if (
        not text
        or text in {".", ".."}
        or "/" in text
        or "\\" in text
        or "\x00" in text
        or text.startswith("~")
    ):
        return f"<unsafe:{_digest(text)[:12]}>"
    return text


def _safe_payload(base: Path, name: Any) -> Path | None:
    """Return a resolved payload path confined to ``base`` or ``None``."""
    text = str(name)
    candidate = Path(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        return None
    try:
        base_resolved = base.resolve(strict=True)
    except (OSError, ValueError):
        return None
    try:
        resolved = (base_resolved / candidate).resolve()
    except (OSError, ValueError):
        return None
    if resolved != base_resolved and base_resolved not in resolved.parents:
        return None
    return resolved


def _sanitize(value: Any) -> str:
    if value is MISSING:
        return "<missing>"
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        if "/" in value or "\\" in value:
            return f"<redacted:{_digest(value)[:12]}>"
        return value if len(value) <= 80 else f"{value[:77]}..."
    if isinstance(value, list):
        return f"list(len={len(value)}, sha={_digest(value)[:12]})"
    if isinstance(value, dict):
        keys = ",".join(sorted(_safe_name(key) for key in value))
        return f"dict(keys={keys})"
    return f"<{type(value).__name__}>"


def differences(path: str, before: Any, after: Any) -> list[str]:
    """Return compact, sanitized difference descriptions for two values."""
    if before is MISSING and after is MISSING:
        return []
    if before is MISSING:
        return [f"{path}: absent before, present now"]
    if after is MISSING:
        return [f"{path}: present before, absent now"]
    if isinstance(before, dict) and isinstance(after, dict):
        found: list[str] = []
        for key in sorted(set(before) | set(after), key=str):
            child = f"{path}.{_safe_name(key)}"
            found.extend(
                differences(
                    child,
                    before.get(key, MISSING),
                    after.get(key, MISSING),
                )
            )
        return found
    if isinstance(before, list) and isinstance(after, list):
        return _list_differences(path, before, after)
    if before != after:
        return [f"{path}: {_sanitize(before)} -> {_sanitize(after)}"]
    return []


def _list_differences(path: str, before: list, after: list) -> list[str]:
    if before == after:
        return []
    if sorted(_canonical(item) for item in before) == sorted(
        _canonical(item) for item in after
    ):
        return [f"{path}: identical records in a different order ({len(before)} items)"]
    before_sha = _digest(before)[:12]
    after_sha = _digest(after)[:12]
    found = [f"{path}: content changed (sha {before_sha} -> {after_sha})"]
    deepest = max(len(before), len(after))
    for index in range(min(deepest, ELEMENT_LIMIT)):
        left = before[index] if index < len(before) else MISSING
        right = after[index] if index < len(after) else MISSING
        found.extend(differences(f"{path}[{index}]", left, right))
    if deepest > ELEMENT_LIMIT:
        found.append(f"{path}: truncated after {ELEMENT_LIMIT} elements")
    return found


def _without_files(state: Any) -> Any:
    if not isinstance(state, dict):
        return state
    return {key: value for key, value in state.items() if key != "files"}


def protected_environment(lock: Any) -> Any:
    """Project an environment lock onto its protected, non-volatile components."""
    if not isinstance(lock, dict):
        return lock
    protected = {
        key: lock[key] for key in PROTECTED_ENVIRONMENT_KEYS if key in lock
    }
    storage = lock.get("storage")
    if isinstance(storage, dict) and "artifact_filesystem_total_bytes" in storage:
        total = storage["artifact_filesystem_total_bytes"]
        protected["artifact_filesystem_total_bytes"] = total
    if "artifact_filesystem_total_bytes" in lock:
        total = lock["artifact_filesystem_total_bytes"]
        protected["artifact_filesystem_total_bytes"] = total
    return protected


def payload_findings(previous_state: Any, current_state: Any) -> list[str]:
    """Compare recorded cached-payload hashes across two artifact states."""
    previous = previous_state.get("files") if isinstance(previous_state, dict) else None
    if not isinstance(previous, dict):
        return []
    current = current_state.get("files") if isinstance(current_state, dict) else {}
    if not isinstance(current, dict):
        current = {}
    found: list[str] = []
    for name in sorted(set(previous) | set(current), key=str):
        before = previous.get(name, MISSING)
        after = current.get(name, MISSING)
        label = _safe_name(name)
        if before is MISSING:
            found.append(f"payload {label}: absent before, present now")
        elif after is MISSING:
            found.append(f"payload {label}: missing from current build")
        elif before != after:
            found.append(
                f"payload {label}: hash changed {str(before)[:12]} -> {str(after)[:12]}"
            )
    return found


def quarantine_findings(snapshot: ReuseSnapshot, phase_root: Path) -> list[str]:
    """Verify original payload hashes of the matching quarantined test run."""
    state = snapshot.state
    if not isinstance(state, dict):
        return []
    recorded_files = state.get("files")
    if not isinstance(recorded_files, dict):
        return []
    expected_hash = state.get("protected_state_sha256")
    expected_run_id = state.get("run_id")
    if not expected_hash:
        return ["quarantine: no protected hash recorded; no run matched"]
    quarantine = phase_root / "quarantine"
    try:
        if not quarantine.is_dir():
            return ["quarantine: directory absent at assertion time"]
        entries = sorted(quarantine.iterdir())
    except OSError:
        return ["quarantine: directory unreadable at assertion time"]
    matches = []
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        candidate = _read_json(entry / "_STATE.json")
        if not isinstance(candidate, dict):
            continue
        if candidate.get("protected_state_sha256") != expected_hash:
            continue
        candidate_run = candidate.get("run_id")
        if expected_run_id is not None and candidate_run != expected_run_id:
            continue
        matches.append(entry)
    if not matches:
        return ["quarantine: no prior run matches this protected hash/run_id"]
    found: list[str] = []
    for entry in matches:
        ordered = sorted(recorded_files.items(), key=lambda item: str(item[0]))
        for name, recorded in ordered:
            label = _safe_name(name)
            safe = _safe_payload(entry, name)
            if safe is None:
                found.append(f"quarantined payload {label}: unsafe path rejected")
                continue
            observed = _file_digest(safe)
            if observed is None:
                found.append(
                    f"quarantined payload {label}: missing from quarantined copy"
                )
            elif observed != recorded:
                prefix = str(recorded)[:12]
                found.append(
                    f"quarantined payload {label}: mutated (recorded {prefix})"
                )
    return found or ["quarantine: matching run retains its original payload hashes"]


def reuse_mismatch_diagnostic(
    *,
    snapshot: ReuseSnapshot,
    run_dir: Path,
    observed: Any,
    expected: Any,
) -> str:
    """Build the compact failure message for a repeated-build reuse mismatch."""
    current = snapshot_reuse_state(run_dir)
    lines = [
        f"{DEBUG_PREFIX} repeated P01 build did not verify_skip; "
        "the reuse equality is unchanged and a mismatch still fails.",
    ]
    lines.extend(differences("response", expected, observed))
    lines.extend(
        differences(
            "state",
            _without_files(snapshot.state),
            _without_files(current.state),
        )
    )
    lines.extend(
        differences(
            "protected_state",
            snapshot.protected_state,
            current.protected_state,
        )
    )
    lines.extend(
        differences(
            "environment",
            protected_environment(snapshot.environment_lock),
            protected_environment(current.environment_lock),
        )
    )
    lines.extend(payload_findings(snapshot.state, current.state))
    phase_root = run_dir.parent.parent
    try:
        lines.extend(quarantine_findings(snapshot, phase_root))
    except (OSError, ValueError):
        lines.append("quarantine: diagnostic read failed; original assertion intact")
    return "\n".join(lines)


def _synthetic_lock() -> dict:
    return {
        "schema_version": "p00-environment-v1",
        "repository": {"commit": "commit-abc", "globally_dirty": False},
        "runtime": {"operating_system": "Linux", "python_version": "3.12.0"},
        "compute": {
            "logical_cpu_count": 8,
            "blas": [
                {"internal_api": "openblas", "version": "0.3.27", "num_threads": 8},
                {"internal_api": "blas", "version": "3.11.0", "num_threads": 8},
            ],
        },
        "storage": {
            "artifact_filesystem_total_bytes": 1000,
            "artifact_filesystem_free_bytes_at_capture": 500,
        },
        "dependencies": {"numpy": "2.1.0"},
        "dependency_lock_sha256": "hash-lock",
    }


def test_identical_protected_components_report_no_difference() -> None:
    before = protected_environment(_synthetic_lock())
    after = protected_environment(_synthetic_lock())
    assert before == after
    assert differences("environment", before, after) == []


def test_blas_reordering_is_distinguished_from_changed_value() -> None:
    lock = _synthetic_lock()
    reordered = _synthetic_lock()
    reordered["compute"]["blas"] = list(reversed(lock["compute"]["blas"]))
    reorder_messages = differences(
        "environment", protected_environment(lock), protected_environment(reordered)
    )
    assert any("different order" in message for message in reorder_messages)

    changed = _synthetic_lock()
    changed["compute"]["blas"] = [dict(record) for record in lock["compute"]["blas"]]
    changed["compute"]["blas"][0]["version"] = "9.9.9"
    value_messages = differences(
        "environment", protected_environment(lock), protected_environment(changed)
    )
    assert any("content changed" in message for message in value_messages)
    assert not any("different order" in message for message in value_messages)
    assert any(".version" in message for message in value_messages)
    assert any("9.9.9" in message for message in value_messages)


def test_volatile_free_disk_bytes_are_excluded() -> None:
    later = _synthetic_lock()
    later["storage"]["artifact_filesystem_free_bytes_at_capture"] = 123
    assert protected_environment(_synthetic_lock()) == protected_environment(later)
    assert (
        differences(
            "environment",
            protected_environment(_synthetic_lock()),
            protected_environment(later),
        )
        == []
    )


def test_payload_findings_identify_missing_and_mutated_payloads() -> None:
    previous = {"files": {"figure.bin": "hash-a", "table.bin": "hash-b"}}
    current = {"files": {"figure.bin": "hash-a", "table.bin": "hash-c"}}
    messages = payload_findings(previous, current)
    assert any(
        "table.bin" in message and "hash changed" in message for message in messages
    )
    missing = payload_findings({"files": {"figure.bin": "hash-a"}}, {"files": {}})
    assert any(
        "figure.bin" in message and "missing" in message for message in missing
    )


def test_quarantined_payload_mutation_and_missing_are_detected(tmp_path: Path) -> None:
    phase_root = tmp_path / "p01"
    entry = phase_root / "quarantine" / "run-1--conflicting"
    entry.mkdir(parents=True)
    payload = entry / "figure.bin"
    payload.write_bytes(b"original")
    recorded = hashlib.sha256(payload.read_bytes()).hexdigest()
    (entry / "_STATE.json").write_text(
        json.dumps({"protected_state_sha256": "protected-1"})
    )
    snapshot = ReuseSnapshot(
        state={
            "protected_state_sha256": "protected-1",
            "files": {"figure.bin": recorded, "gone.bin": "0" * 64},
        },
        protected_state=MISSING,
        environment_lock=MISSING,
    )
    unchanged = quarantine_findings(snapshot, phase_root)
    assert not any("figure.bin" in message for message in unchanged)
    assert any("gone.bin" in message and "missing" in message for message in unchanged)

    payload.write_bytes(b"mutated")
    mutated = quarantine_findings(snapshot, phase_root)
    assert any("figure.bin" in message and "mutated" in message for message in mutated)


def test_missing_diagnostic_files_do_not_mask_the_assertion(tmp_path: Path) -> None:
    missing_run = tmp_path / "absent-run"
    assert snapshot_reuse_state(missing_run).state is MISSING

    snapshot = ReuseSnapshot(
        state={"protected_state_sha256": "p"},
        protected_state={"repository": {"commit": "c"}},
        environment_lock=MISSING,
    )
    message = reuse_mismatch_diagnostic(
        snapshot=snapshot,
        run_dir=missing_run,
        observed={"status": "pass", "action": "new"},
        expected={"status": "pass", "action": "verified_skip"},
    )
    assert DEBUG_PREFIX in message
    assert "state: present before, absent now" in message
    assert "response.action" in message
    assert differences("environment", MISSING, MISSING) == []


def test_diagnostics_are_read_only_and_do_not_leak_paths(tmp_path: Path) -> None:
    secret = str(tmp_path / "private" / "spectra.csv")
    lock = _synthetic_lock()
    lock["repository"]["commit"] = secret
    before = json.loads(json.dumps(lock))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "_STATE.json").write_text(json.dumps({"files": {}}))
    (run_dir / "protected_state.json").write_text(json.dumps({"repository": {"commit": "other"}}))
    (run_dir / "environment_lock.json").write_text(json.dumps(_synthetic_lock()))

    message = reuse_mismatch_diagnostic(
        snapshot=ReuseSnapshot(
            state={"x": 1}, protected_state=lock, environment_lock=lock
        ),
        run_dir=run_dir,
        observed=lock,
        expected=_synthetic_lock(),
    )
    assert lock == before
    assert secret not in message
    assert str(tmp_path) not in message
    assert "<redacted:" in message


def test_integrated_diagnostic_reports_quarantined_defect(tmp_path: Path) -> None:
    phase_root = tmp_path / "p01"
    run_dir = phase_root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    state = {
        "schema_version": "atlas-artifact-state-v1",
        "phase": "P01",
        "run_id": "run-1",
        "protected_state_sha256": "protected-1",
        "execution_status": "complete",
        "scientific_status": "pass",
        "files": {"figure.bin": "0" * 64, "gone.bin": "1" * 64},
    }
    (run_dir / "_STATE.json").write_text(json.dumps(state))
    (run_dir / "protected_state.json").write_text(
        json.dumps({"protected_state_sha256": "protected-1"})
    )
    (run_dir / "environment_lock.json").write_text(json.dumps(_synthetic_lock()))

    quarantine = phase_root / "quarantine" / "run-1--conflicting"
    quarantine.mkdir(parents=True)
    (quarantine / "_STATE.json").write_text(json.dumps(state))
    (quarantine / "figure.bin").write_bytes(b"mutated")

    snapshot = snapshot_reuse_state(run_dir)
    message = reuse_mismatch_diagnostic(
        snapshot=snapshot,
        run_dir=run_dir,
        observed={"status": "pass", "action": "new", "run_id": "run-1"},
        expected={"status": "pass", "action": "verified_skip", "run_id": "run-1"},
    )
    assert DEBUG_PREFIX in message
    assert "figure.bin" in message and "mutated" in message
    assert "gone.bin" in message and "missing" in message
    assert str(tmp_path) not in message


def test_unsafe_payload_paths_are_rejected_without_leakage(tmp_path: Path) -> None:
    phase_root = tmp_path / "p01"
    entry = phase_root / "quarantine" / "run-1--dup"
    entry.mkdir(parents=True)
    outside = tmp_path / "secret.bin"
    outside.write_bytes(b"secret")
    (entry / "_STATE.json").write_text(
        json.dumps({"protected_state_sha256": "p", "run_id": "r"})
    )
    link = entry / "link.bin"
    link.symlink_to(outside)
    digest = hashlib.sha256(b"secret").hexdigest()
    snapshot = ReuseSnapshot(
        state={
            "run_id": "r",
            "protected_state_sha256": "p",
            "files": {
                "../secret.bin": digest,
                str(outside): digest,
                "link.bin": digest,
                "ok.bin": hashlib.sha256(b"ok").hexdigest(),
            },
        },
        protected_state=MISSING,
        environment_lock=MISSING,
    )
    messages = quarantine_findings(snapshot, phase_root)
    assert sum("unsafe path rejected" in message for message in messages) == 3
    assert any("ok.bin" in message and "missing" in message for message in messages)
    assert all(str(tmp_path) not in message for message in messages)


def test_unsafe_diagnostic_keys_are_sanitized() -> None:
    messages = differences(
        "environment",
        {"../evil/key": "old", "safe": 1},
        {"../evil/key": "new", "safe": 1},
    )
    joined = "\n".join(messages)
    assert "<unsafe:" in joined
    assert "../evil/key" not in joined
    assert "environment.safe" not in joined


def test_quarantine_matching_requires_hash_and_run_id(tmp_path: Path) -> None:
    phase_root = tmp_path / "p01"
    quarantine = phase_root / "quarantine"
    recorded = hashlib.sha256(b"ok").hexdigest()
    entries = (
        ("wrong-id", "bad", b"mutated"),
        ("missing-id", None, b"mutated"),
        ("correct-id", "good", b"ok"),
    )
    for name, run_id, payload in entries:
        entry = quarantine / f"{name}--conflicting"
        entry.mkdir(parents=True)
        metadata: dict[str, str] = {"protected_state_sha256": "p"}
        if run_id is not None:
            metadata["run_id"] = run_id
        (entry / "_STATE.json").write_text(json.dumps(metadata))
        (entry / "figure.bin").write_bytes(payload)

    snapshot = ReuseSnapshot(
        state={
            "run_id": "good",
            "protected_state_sha256": "p",
            "files": {"figure.bin": recorded},
        },
        protected_state=MISSING,
        environment_lock=MISSING,
    )
    messages = quarantine_findings(snapshot, phase_root)
    assert not any("mutated" in message for message in messages)
    assert any(
        "retains its original payload hashes" in message for message in messages
    )

    no_hash = ReuseSnapshot(
        state={"files": {"figure.bin": recorded}},
        protected_state=MISSING,
        environment_lock=MISSING,
    )
    assert quarantine_findings(no_hash, phase_root) == [
        "quarantine: no protected hash recorded; no run matched"
    ]
