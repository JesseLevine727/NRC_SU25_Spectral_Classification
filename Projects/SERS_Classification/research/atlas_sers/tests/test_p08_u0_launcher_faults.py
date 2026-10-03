"""Independent invented-file outer-entry fault probes for the U0 launcher.

This module drives the outer-entry validation of the referenced smoke script
with mocked capacity observations, mocked GPU observations, and invented files
written under ``tmp_path``.  A test-only permit hash pins the failure path, so
no real permit or spectra are needed.

These probes are not a scientific GPU acceptance test, and they do not run the
full 78-pair smoke.  They only re-check the launcher's portable outer-entry
fault handling against invented inputs.
"""

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_p08_u0_smoke.py"


@pytest.fixture
def subject():
    spec = importlib.util.spec_from_file_location("invented_launcher_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def capacity(monkeypatch):
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda fd: SimpleNamespace(f_bavail=64 * 1024**3, f_frsize=1),
    )


def claim(subject, root):
    budget = subject._Budget(
        {"limits": dict(subject._EXPECTED_LIMITS)}, time.monotonic_ns()
    )
    return subject._claim_output(str(root), budget)


def test_claim_leaves_artifact_creation_to_existing_session(subject, tmp_path):
    out = claim(subject, tmp_path / "run")
    try:
        assert (tmp_path / "run" / "control").is_dir()
        assert not (tmp_path / "run" / "artifacts").exists()
        assert not (tmp_path / "run" / "journal").exists()
    finally:
        out.close()


@pytest.mark.parametrize("which", ["parent", "root", "control"])
def test_output_path_replacement_rejected(subject, tmp_path, which):
    parent = tmp_path / "parent"
    parent.mkdir()
    root = parent / "run"
    out = claim(subject, root)
    try:
        if which == "parent":
            parent.rename(tmp_path / "old-parent")
            parent.mkdir()
        elif which == "root":
            root.rename(parent / "old-run")
            root.mkdir()
        else:
            (root / "control").rename(root / "old-control")
            (root / "control").mkdir()
        with pytest.raises(subject.LaunchError, match="output_identity_changed"):
            out.verify()
    finally:
        out.close()


def test_layout_does_not_ignore_unexpected_root_file(subject, tmp_path):
    out = claim(subject, tmp_path / "run")
    try:
        (tmp_path / "run" / "not-in-inventory.bin").write_bytes(b"123")
        with pytest.raises(subject.LaunchError):
            out.verify()
    finally:
        out.close()


def test_zero_write_is_rejected_without_repeating_write(
    subject, tmp_path, monkeypatch
):
    out = claim(subject, tmp_path / "run")
    count = 0

    def stopped_write(fd, data):
        nonlocal count
        count += 1
        if count == 1:
            return 0
        raise KeyboardInterrupt("prevent an unbounded test loop")

    try:
        monkeypatch.setattr(subject.os, "write", stopped_write)
        with pytest.raises(subject.LaunchError, match="control_write_failed"):
            subject._write_control_file(out, "launch.json", b"{}")
        assert count == 1
    finally:
        out.close()


def test_failure_diagnostics_preserve_original_exception(subject, tmp_path, monkeypatch):
    package = tmp_path / "public-package"
    package.mkdir()
    unused = str(tmp_path / "unused-input")
    permit = {
        "schema_version": subject.PERMIT_SCHEMA,
        "stage": "U0",
        "execution_authorized": True,
        "proposal_sha256": subject.PROPOSAL_SHA256,
        "manifest_sha256": subject.MANIFEST_SHA256,
        "catalog_sha256": "a" * 64,
        "source_revision": "b" * 40,
        "limits": dict(subject._EXPECTED_LIMITS),
        "output_root": str(tmp_path / "run"),
        "metadata_paths": {key: unused for key in subject._METADATA_KEYS},
        "action_paths": {key: unused for key in subject._ACTION_KEYS},
        "specification_audit_path": unused,
        "candidate_registry_path": unused,
    }
    raw = json.dumps(permit, sort_keys=True).encode()
    permit_file = tmp_path / "invented-permit.json"
    permit_file.write_bytes(raw)
    monkeypatch.setattr(
        subject, "APPROVED_PERMIT_SHA256", hashlib.sha256(raw).hexdigest()
    )
    monkeypatch.setattr(subject, "_require_environment", lambda: None)
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: SimpleNamespace(f_bavail=64 * 1024**3, f_frsize=1),
    )
    monkeypatch.setattr(
        os,
        "fstatvfs",
        lambda fd: SimpleNamespace(f_bavail=64 * 1024**3, f_frsize=1),
    )
    original = ValueError("invented initiating failure")

    def broken_run(*args, **kwargs):
        raise original

    def broken_diagnostic(*args, **kwargs):
        raise RuntimeError("invented diagnostic failure")

    monkeypatch.setattr(subject, "_run", broken_run)
    monkeypatch.setattr(subject, "_finalize_failure", broken_diagnostic)
    with pytest.raises(ValueError) as caught:
        subject.launch(str(package), str(permit_file))
    assert caught.value is original


@pytest.mark.parametrize("raw", [b'{"value":1e999}', b'{"value":NaN}'])
def test_nonfinite_json_is_rejected(subject, raw):
    with pytest.raises(subject.LaunchError, match="permit_parse_failed"):
        subject._load_json_bytes(raw, "permit_parse_failed")


@pytest.mark.parametrize("case", ["retained_high_water", "exact_ceiling"])
def test_budget_refuses_exhausted_charge(subject, tmp_path, case):
    permit = {"limits": dict(subject._EXPECTED_LIMITS)}
    budget = subject._Budget(permit, time.monotonic_ns())
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        cap = permit["limits"]["artifact_bytes"]
        actual = cap if case == "exact_ceiling" else 0
        if case == "retained_high_water":
            budget.high_water_bytes = cap + 1
        with pytest.raises(subject.LaunchError, match="resource_limit_exceeded"):
            budget.check_setup(fd, actual, 0)
    finally:
        os.close(fd)


def test_success_cannot_ignore_cleanup_time(subject, tmp_path, monkeypatch):
    stamp = 1_000_000_000
    clock = [stamp]
    monkeypatch.setattr(
        subject, "time", SimpleNamespace(monotonic_ns=lambda: clock[0])
    )
    permit = {
        "limits": dict(subject._EXPECTED_LIMITS),
        "catalog_sha256": "a" * 64,
        "source_revision": "b" * 40,
    }
    budget = subject._Budget(permit, stamp)
    out = subject._claim_output(str(tmp_path / "run"), budget)
    original_close = subject._Output.close

    def slow_close(self):
        original_close(self)
        clock[0] = stamp + permit["limits"]["wall_ns"]

    monkeypatch.setattr(subject._Output, "close", slow_close)
    data = {
        "session_report": {
            "completed_pair_count": 78,
            "fit_attempt_count": 78,
            "prediction_attempt_count": 78,
            "recorded_artifact_bytes": 0,
            "observed_logical_bytes": 0,
        }
    }
    data["resource_record"] = {
        "snapshot": {"within_proposed_limits": True},
        "evidence": subject._gpu_resource_evidence(
            dict(
                initialized=False,
                observed=False,
                allocated=0,
                reserved=0,
                device_used=0,
                peak=0,
            ),
            "post_session_pre_terminal_persistence",
            stamp,
        ),
    }
    try:
        with pytest.raises(subject.LaunchError, match="resource_limit_exceeded"):
            subject._finalize_success(out, permit, "c" * 64, stamp, data, budget)
    finally:
        out.close()


def test_successful_write_does_not_swallow_close_interrupt(
    subject, tmp_path, monkeypatch
):
    out = claim(subject, tmp_path / "run")
    original_open, original_close = os.open, os.close
    target = [-1]
    signal = KeyboardInterrupt("invented writer-close interrupt")

    def opened(path, flags, *args, **kwargs):
        fd = original_open(path, flags, *args, **kwargs)
        if flags & os.O_WRONLY:
            target[0] = fd
        return fd

    def closed(fd):
        original_close(fd)
        if fd == target[0]:
            target[0] = -1
            raise signal

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "close", closed)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            subject._write_control_file(out, "launch.json", b"{}")
        assert caught.value is signal
    finally:
        out.close()


@pytest.mark.parametrize("fault", ["missing_peak", "float_peak", "wrong_flag_type"])
def test_gpu_evidence_refuses_malformed_actual_observer_packet(subject, fault):
    packet = dict(
        initialized=True,
        observed=True,
        allocated=10,
        reserved=20,
        device_used=25,
        peak=12,
    )
    if fault == "missing_peak":
        del packet["peak"]
    elif fault == "float_peak":
        packet["peak"] = 12.0
    else:
        packet["initialized"] = "true"
    with pytest.raises(subject.LaunchError, match="resource_setup_failed"):
        subject._gpu_resource_evidence(packet, "invented_probe", 100)


def test_gpu_peak_at_existing_limit_is_not_a_new_stricter_limit(subject):
    cap = subject._EXPECTED_LIMITS["allocated_gpu_bytes"]
    packet = dict(
        initialized=True,
        observed=True,
        allocated=10,
        reserved=20,
        device_used=25,
        peak=cap,
    )
    evidence = subject._gpu_resource_evidence(packet, "invented_probe", 100)
    assert evidence["gpu_peak_within_limit"] is True


@pytest.mark.parametrize("key", ["allocated", "reserved", "device_used", "peak"])
def test_uninitialized_gpu_packet_must_match_actual_zero_placeholders(subject, key):
    packet = dict(
        initialized=False,
        observed=False,
        allocated=0,
        reserved=0,
        device_used=0,
        peak=0,
    )
    packet[key] = 1
    with pytest.raises(subject.LaunchError, match="resource_setup_failed"):
        subject._gpu_resource_evidence(packet, "invented_probe", 100)


@pytest.mark.parametrize(
    "data",
    [None, {}, {"resource_record": {}}, {"resource_record": {"evidence": {}}}],
)
def test_missing_final_resource_evidence_is_not_success(subject, data):
    with pytest.raises(subject.LaunchError, match="resource_setup_failed"):
        subject._resource_evidence(data)


def test_layout_refuses_inventory_change_during_scan(subject, tmp_path, monkeypatch):
    root = tmp_path / "run"
    out = claim(subject, root)
    record = root / "control" / "launch.json"
    record.write_bytes(b"{}")
    original = subject._stat_at
    hit = []

    def changed(fd, name, reason):
        result = original(fd, name, reason)
        if name == "launch.json" and not hit:
            hit.append(True)
            (root / "control" / "unexpected").write_bytes(b"x")
        return result

    monkeypatch.setattr(subject, "_stat_at", changed)
    try:
        with pytest.raises(subject.LaunchError, match="output_identity_changed"):
            out.verify()
        assert hit
    finally:
        out.close()


def test_layout_refuses_child_replacement_after_scan(subject, tmp_path, monkeypatch):
    root = tmp_path / "run"
    out = claim(subject, root)
    child = root / "artifacts"
    child.mkdir()
    (child / "invented.bin").write_bytes(b"123")
    original = subject._scan_flat
    hit = []

    def changed(fd, *args, **kwargs):
        result = original(fd, *args, **kwargs)
        if not hit:
            hit.append(True)
            child.rename(tmp_path / "old-artifacts")
            child.mkdir()
        return result

    monkeypatch.setattr(subject, "_scan_flat", changed)
    try:
        with pytest.raises(subject.LaunchError, match="output_identity_changed"):
            out.verify()
        assert hit
    finally:
        out.close()


def test_input_refuses_in_place_change_after_read(subject, tmp_path, monkeypatch):
    path = tmp_path / "invented-input"
    path.write_bytes(b"abcd")
    original = subject._read_fd

    def changed(*args):
        result = original(*args)
        path.write_bytes(b"wxyz")
        return result

    monkeypatch.setattr(subject, "_read_fd", changed)
    with pytest.raises(subject.LaunchError, match="input_read_failed"):
        subject._read_absolute_file(str(path), 100, "input_read_failed")


def test_scan_uses_bounded_inventory_iterator(subject, tmp_path, monkeypatch):
    consumed = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            consumed.append(len(consumed))
            assert len(consumed) <= 4, "inventory read past bound plus one"
            return SimpleNamespace(name="invented-" + str(len(consumed)))

    def unbounded(*args):
        raise AssertionError("unbounded os.listdir used")

    monkeypatch.setattr(os, "scandir", lambda fd: Stream())
    monkeypatch.setattr(os, "listdir", unbounded)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(subject.LaunchError, match="output_identity_changed"):
            subject._scan_flat(fd, 3, "output_identity_changed")
        assert len(consumed) == 4
    finally:
        os.close(fd)


@pytest.mark.parametrize("primary", [False, True])
def test_scandir_close_interrupt_propagation(subject, tmp_path, monkeypatch, primary):
    signal = KeyboardInterrupt("invented-scandir-close")
    initiating = ValueError("invented-inventory-error")
    closed = []

    class Stream:
        def __iter__(self):
            if primary:
                raise initiating
            return iter(())

        def close(self):
            closed.append(True)
            raise signal

    monkeypatch.setattr(os, "scandir", lambda fd: Stream())
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        expected = initiating if primary else signal
        with pytest.raises(type(expected)) as caught:
            subject._list_fd(fd, 3, "output_identity_changed")
        assert caught.value is expected
        assert closed == [True]
    finally:
        os.close(fd)


def test_inventory_is_order_independent_without_changes(subject, tmp_path, monkeypatch):
    (tmp_path / "a").write_bytes(b"123")
    (tmp_path / "b").write_bytes(b"45")
    calls = []

    class Stream:
        def __init__(self, names):
            self.names = names

        def __iter__(self):
            return iter(SimpleNamespace(name=name) for name in self.names)

        def close(self):
            pass

    def reordered(fd):
        calls.append(True)
        return Stream(["a", "b"] if len(calls) % 2 else ["b", "a"])

    monkeypatch.setattr(os, "scandir", reordered)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert subject._scan_flat(fd, 2, "output_identity_changed") == 5
        assert len(calls) == 2
    finally:
        os.close(fd)
