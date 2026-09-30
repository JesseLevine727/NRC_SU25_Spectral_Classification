"""Synthetic, no-execution tests for the P08-T063 terminal receipt verifier.

Everything here is invented metadata written under pytest ``tmp_path``.  No
dataset path, device, model or scientific artifact is ever touched.  The tests
exercise read-only byte verification and its exact static reason codes.
"""

from __future__ import annotations

import contextlib
import copy
import errno
import hashlib
import json
import os
import pathlib

import pytest

from atlas_sers.evaluation import p08_terminal_receipt as tr
from atlas_sers.evaluation import p08_u0_admission as admission
from atlas_sers.evaluation import p08_u0_store as store
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256
from tests.p08_store_fixtures import (
    append,
    bind_manifest,
    make_event,
    readtree,
    resources,
)

_RECEIPT_SCHEMA = "nato-sers-p08-terminal-receipt-v1"
_REPORT_SCHEMA = "nato-sers-p08-terminal-receipt-check-v1"


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _reseal(receipt, terminal):
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    terminal["receipt_sha256"] = receipt["receipt_sha256"]
    terminal["event_sha256"] = canonical_sha256(
        {key: value for key, value in terminal.items() if key != "event_sha256"}
    )


def _rewrite_receipt(s):
    (pathlib.Path(s["root"]) / s["receipt_name"]).write_bytes(_canonical_bytes(s["receipt"]))


def _build_scenario(
    tmp_path,
    monkeypatch,
    *,
    owner_box=None,
    fit_id="fit001",
    status="succeeded",
    predict=False,
    artifact_names=("a.bin",),
    receipt_name="receipt.json",
):
    manifest = bind_manifest(monkeypatch)
    jobs = {job["job_id"]: job for job in manifest["jobs"]}
    owner = store.create_store(str(tmp_path / "journal"), manifest)
    if owner_box is not None:
        owner_box["owner"] = owner
    append(owner, "session_open")
    fit_job = jobs[fit_id]
    append(
        owner,
        "attempt_start",
        job_id=fit_job["job_id"],
        resources=resources(),
    )
    target_job = fit_job
    if predict:
        append(
            owner,
            "attempt_finish",
            job_id=fit_job["job_id"],
            status="succeeded",
        )
        target_job = jobs["pred" + fit_id[len("fit") :]]
        append(
            owner,
            "attempt_start",
            job_id=target_job["job_id"],
            resources=resources(),
        )

    state = owner.snapshot()
    terminal = make_event(state, "attempt_finish", job_id=target_job["job_id"], status=status)
    start_event = next(
        event
        for event in state["events"]
        if event["event_type"] == "attempt_start" and event["job_id"] == target_job["job_id"]
    )

    ordered = sorted(artifact_names)
    blobs = {name: (name + ":payload").encode("utf-8") for name in ordered}
    artifacts = [
        {
            "name": name,
            "size_bytes": len(blobs[name]),
            "sha256": hashlib.sha256(blobs[name]).hexdigest(),
        }
        for name in ordered
    ]
    receipt = {
        "schema_version": _RECEIPT_SCHEMA,
        "execution_authorized": False,
        "proposal_sha256": admission.U0_PROPOSAL_SHA256,
        "manifest_sha256": admission.U0_MANIFEST_SHA256,
        "job_id": terminal["job_id"],
        "stage": target_job["stage"],
        "worker": target_job["worker"],
        "session_id": terminal["session_id"],
        "start_event_sha256": start_event["event_sha256"],
        "status": terminal["status"],
        "artifacts": artifacts,
        "receipt_sha256": "",
    }
    _reseal(receipt, terminal)

    root = tmp_path / "artifacts"
    root.mkdir()
    for name, data in blobs.items():
        (root / name).write_bytes(data)
    (root / receipt_name).write_bytes(_canonical_bytes(receipt))

    return {
        "root": str(root),
        "receipt_name": receipt_name,
        "manifest": manifest,
        "events": state["events"],
        "terminal_event": terminal,
        "expected_head_sha256": state["summary"]["head_sha256"],
        "expected_artifact_names": ordered,
        "receipt": receipt,
        "blobs": blobs,
        "owner": owner,
    }


@contextlib.contextmanager
def scenario(tmp_path, monkeypatch, **kwargs):
    owner_box = {}
    try:
        built = _build_scenario(tmp_path, monkeypatch, owner_box=owner_box, **kwargs)
    except BaseException:
        owner = owner_box.get("owner")
        if owner is not None:
            try:
                owner.close()
            except BaseException:
                pass
        raise
    try:
        yield built
    finally:
        built["owner"].close()


def _call(
    s,
    *,
    terminal_event=None,
    expected_head_sha256=None,
    expected_artifact_names=None,
    receipt_name=None,
):
    return tr.verify_terminal_receipt(
        s["root"],
        s["receipt_name"] if receipt_name is None else receipt_name,
        s["manifest"],
        s["events"],
        s["terminal_event"] if terminal_event is None else terminal_event,
        expected_head_sha256=(
            s["expected_head_sha256"] if expected_head_sha256 is None else expected_head_sha256
        ),
        expected_artifact_names=(
            s["expected_artifact_names"]
            if expected_artifact_names is None
            else expected_artifact_names
        ),
    )


def _expect(s, code, **kwargs):
    with pytest.raises(tr.ReceiptError) as info:
        _call(s, **kwargs)
    assert info.value.reason_code == code
    return info.value


def _mutate_and_reseal(s, field, value):
    s["receipt"][field] = value
    _reseal(s["receipt"], s["terminal_event"])
    _rewrite_receipt(s)


def _capture_owned_fds(monkeypatch):
    real_resolve = store._resolve_parent
    real_open = store._open_dir
    seen = {}

    def resolve_root(root):
        parent_fd, root_name = real_resolve(root)
        seen["parent_fd"] = parent_fd
        return parent_fd, root_name

    def open_root(parent_fd, root_name):
        root_fd = real_open(parent_fd, root_name)
        seen["root_fd"] = root_fd
        return root_fd

    monkeypatch.setattr(store, "_resolve_parent", resolve_root)
    monkeypatch.setattr(store, "_open_dir", open_root)
    return seen


def _arm_cleanup_interrupt(monkeypatch, seen, interrupt):
    real_close = store._close_fds
    state = {"armed": True}

    def close_then_raise(*fds):
        result = real_close(*fds)
        owned = (seen.get("root_fd"), seen.get("parent_fd"))
        if state["armed"] and "root_fd" in seen and fds == owned:
            state["armed"] = False
            raise interrupt
        return result

    monkeypatch.setattr(store, "_close_fds", close_then_raise)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fit_id": "fit001"},
        {"fit_id": "fit050"},
        {"fit_id": "fit001", "status": "failed"},
        {"fit_id": "fit001", "status": "interrupted"},
        {"fit_id": "fit001", "predict": True},
        {"fit_id": "fit050", "predict": True},
        {"fit_id": "fit001", "artifact_names": ("a.bin", "b.bin", "c.bin")},
    ],
)
def test_verify_valid(tmp_path, monkeypatch, kwargs):
    with scenario(tmp_path, monkeypatch, **kwargs) as s:
        before = readtree(s["root"])
        report = _call(s)
        after = readtree(s["root"])
        assert before == after
        assert report["schema_version"] == _REPORT_SCHEMA
        assert report["execution_authorized"] is False
        assert report["byte_integrity_verified"] is True
        assert report["scientific_semantics_verified"] is False
        assert report["artifact_count"] == len(s["expected_artifact_names"])
        assert report["artifact_bytes"] == sum(len(value) for value in s["blobs"].values())
        assert report["status"] == s["terminal_event"]["status"]
        assert report["manifest_sha256"] == admission.U0_MANIFEST_SHA256
        assert report["previous_head_sha256"] == s["expected_head_sha256"]
        assert report["proposed_terminal_event_sha256"] == s["terminal_event"]["event_sha256"]
        assert report["receipt_sha256"] == s["receipt"]["receipt_sha256"]
        receipt_bytes = (pathlib.Path(s["root"]) / s["receipt_name"]).read_bytes()
        assert report["receipt_file_sha256"] == hashlib.sha256(receipt_bytes).hexdigest()
        body = {k: v for k, v in report.items() if k != "check_sha256"}
        assert report["check_sha256"] == canonical_sha256(body)


def test_no_mutation(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch, artifact_names=("a.bin", "b.bin")) as s:
        manifest_before = copy.deepcopy(s["manifest"])
        events_before = copy.deepcopy(s["events"])
        terminal_before = copy.deepcopy(s["terminal_event"])
        names_before = list(s["expected_artifact_names"])
        _call(s)
        assert s["manifest"] == manifest_before
        assert s["events"] == events_before
        assert s["terminal_event"] == terminal_before
        assert s["expected_artifact_names"] == names_before


def test_report_hides_names_and_paths(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch, artifact_names=("secret.bin",)) as s:
        report = _call(s)
        text = json.dumps(report, sort_keys=True)
        assert "secret.bin" not in text
        assert s["root"] not in text
        assert "fit001" not in text


def test_wrong_job_id(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _mutate_and_reseal(s, "job_id", "fit002")
        _expect(s, "invalid_receipt")


def test_wrong_start_event(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _mutate_and_reseal(s, "start_event_sha256", "0" * 64)
        _expect(s, "invalid_receipt")


def test_wrong_session_id(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _mutate_and_reseal(s, "session_id", s["terminal_event"]["session_id"] + 1)
        _expect(s, "invalid_receipt")


def test_wrong_stage(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _mutate_and_reseal(s, "stage", "wrong_stage")
        _expect(s, "invalid_receipt")


def test_wrong_worker(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _mutate_and_reseal(s, "worker", "npu")
        _expect(s, "invalid_receipt")


def test_wrong_proposal(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _mutate_and_reseal(s, "proposal_sha256", "0" * 64)
        _expect(s, "invalid_receipt")


def test_wrong_manifest(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _mutate_and_reseal(s, "manifest_sha256", "0" * 64)
        _expect(s, "invalid_receipt")


def test_wrong_status(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _mutate_and_reseal(s, "status", "failed")
        _expect(s, "invalid_receipt")


def test_forged_receipt_digest(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        s["receipt"]["receipt_sha256"] = "0" * 64
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


def test_receipt_digest_not_over_fields(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        s["receipt"]["receipt_sha256"] = "f" * 64
        s["terminal_event"]["receipt_sha256"] = "f" * 64
        s["terminal_event"]["event_sha256"] = canonical_sha256(
            {key: value for key, value in s["terminal_event"].items() if key != "event_sha256"}
        )
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


def test_forged_terminal_event_hash(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        s["terminal_event"]["event_sha256"] = "0" * 64
        _expect(s, "invalid_journal")


def test_forged_current_event(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        s["events"][-1]["event_sha256"] = "0" * 64
        _expect(s, "invalid_journal")


def test_terminal_not_a_dict(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _expect(s, "invalid_journal", terminal_event=["not", "a", "dict"])


def test_terminal_wrong_event_type(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        s["terminal_event"]["event_type"] = "progress"
        _expect(s, "invalid_journal")


def test_stale_head(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _expect(s, "invalid_journal", expected_head_sha256="0" * 64)


def test_smoke_binding_mismatch(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        monkeypatch.setattr(admission, "_u0_binding_ok", lambda manifest: False)
        _expect(s, "smoke_binding_mismatch")


def test_extra_receipt_key(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        s["receipt"]["extra"] = 1
        _reseal(s["receipt"], s["terminal_event"])
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


def test_missing_receipt_key(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        del s["receipt"]["status"]
        _reseal(s["receipt"], s["terminal_event"])
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


def test_duplicate_json_key(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        (pathlib.Path(s["root"]) / s["receipt_name"]).write_bytes(b'{"a":1,"a":2}')
        _expect(s, "invalid_receipt")


def test_nonfinite_json(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        (pathlib.Path(s["root"]) / s["receipt_name"]).write_bytes(b'{"a":Infinity}')
        _expect(s, "invalid_receipt")


def test_bool_size_bytes(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        s["receipt"]["artifacts"][0]["size_bytes"] = True
        _reseal(s["receipt"], s["terminal_event"])
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


def test_bool_session_id(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        s["receipt"]["session_id"] = True
        _reseal(s["receipt"], s["terminal_event"])
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


def test_bad_artifact_sha(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        s["receipt"]["artifacts"][0]["sha256"] = "Z" * 64
        _reseal(s["receipt"], s["terminal_event"])
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


def test_missing_artifact_key(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        del s["receipt"]["artifacts"][0]["sha256"]
        _reseal(s["receipt"], s["terminal_event"])
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


def test_extra_artifact_key(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        s["receipt"]["artifacts"][0]["extra"] = "x"
        _reseal(s["receipt"], s["terminal_event"])
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


def test_forged_execution_authorized(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        s["receipt"]["execution_authorized"] = 0
        _reseal(s["receipt"], s["terminal_event"])
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


def test_unsorted_artifacts(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch, artifact_names=("a.bin", "b.bin")) as s:
        s["receipt"]["artifacts"].reverse()
        _reseal(s["receipt"], s["terminal_event"])
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


def test_expected_names_mismatch(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _expect(s, "invalid_receipt", expected_artifact_names=["z.bin"])


@pytest.mark.parametrize(
    "bad",
    ["a/b", "../x", "..", "", "caf\u00e9", "x" * 129],
)
def test_bad_receipt_name(tmp_path, monkeypatch, bad):
    with scenario(tmp_path, monkeypatch) as s:
        _expect(s, "invalid_input", receipt_name=bad)


@pytest.mark.parametrize(
    "bad",
    [
        "not-a-list",
        [],
        ["a.bin"] * 9,
        ["a.bin", "a.bin"],
        ["b.bin", "a.bin"],
        ["a/b"],
        ["caf\u00e9"],
    ],
)
def test_bad_expected_names(tmp_path, monkeypatch, bad):
    with scenario(tmp_path, monkeypatch, artifact_names=("a.bin", "b.bin")) as s:
        _expect(s, "invalid_input", expected_artifact_names=bad)


def test_receipt_name_in_artifacts(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        _expect(s, "invalid_input", receipt_name="a.bin")


def test_missing_receipt_file(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        (pathlib.Path(s["root"]) / s["receipt_name"]).unlink()
        _expect(s, "receipt_io_error")


def test_missing_artifact_file(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        (pathlib.Path(s["root"]) / "a.bin").unlink()
        _expect(s, "receipt_io_error")


def test_corrupt_artifact(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        (pathlib.Path(s["root"]) / "a.bin").write_bytes(b"other")
        _expect(s, "artifact_mismatch")


def test_truncated_artifact(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        data = s["blobs"]["a.bin"]
        (pathlib.Path(s["root"]) / "a.bin").write_bytes(data[:-1])
        _expect(s, "artifact_mismatch")


def test_symlink_artifact(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        path = pathlib.Path(s["root"]) / "a.bin"
        target = tmp_path / "outside.bin"
        target.write_bytes(s["blobs"]["a.bin"])
        path.unlink()
        path.symlink_to(target)
        _expect(s, "receipt_io_error")


def test_hardlink_artifact(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        path = pathlib.Path(s["root"]) / "a.bin"
        target = tmp_path / "outside-hard.bin"
        target.write_bytes(s["blobs"]["a.bin"])
        path.unlink()
        os.link(target, path)
        _expect(s, "receipt_io_error")


def test_fifo_artifact(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        path = pathlib.Path(s["root"]) / "a.bin"
        path.unlink()
        os.mkfifo(path)
        _expect(s, "receipt_io_error")


def test_receipt_size_cap(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        monkeypatch.setattr(tr, "_MAX_RECEIPT_BYTES", 32)
        (pathlib.Path(s["root"]) / s["receipt_name"]).write_bytes(b"x" * 64)
        _expect(s, "receipt_io_error")


def test_artifact_size_cap(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        monkeypatch.setattr(tr, "_MAX_ARTIFACT_BYTES", 8)
        s["receipt"]["artifacts"][0]["size_bytes"] = 8
        _reseal(s["receipt"], s["terminal_event"])
        _rewrite_receipt(s)
        (pathlib.Path(s["root"]) / "a.bin").write_bytes(b"x" * 16)
        _expect(s, "receipt_io_error")


def test_interrupt_cleanup_and_no_side_calls(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        close_calls = []
        real_close = store._close_fds

        def tracking_close(*fds):
            close_calls.append(fds)
            return real_close(*fds)

        monkeypatch.setattr(store, "_close_fds", tracking_close)

        def boom(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(store, "_read_file", boom)

        def forbidden(*args, **kwargs):
            raise AssertionError("forbidden side call")

        monkeypatch.setattr(store.Store, "append_event", forbidden)
        monkeypatch.setattr(admission, "evaluate_u0_candidate", forbidden)

        with pytest.raises(KeyboardInterrupt):
            _call(s)
        assert close_calls


def test_read_interrupt_closes_exact_owned_fds(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        seen = _capture_owned_fds(monkeypatch)
        interrupt = KeyboardInterrupt("receipt-read")

        def boom(*args, **kwargs):
            raise interrupt

        monkeypatch.setattr(store, "_read_file", boom)

        with pytest.raises(KeyboardInterrupt) as info:
            _call(s)
        assert info.value is interrupt
        assert set(seen) == {"parent_fd", "root_fd"}
        for fd in (seen["root_fd"], seen["parent_fd"]):
            with pytest.raises(OSError) as closed:
                os.fstat(fd)
            assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit])
def test_cleanup_interrupt_on_success_is_not_reported(tmp_path, monkeypatch, exc_type):
    with scenario(tmp_path, monkeypatch) as s:
        seen = _capture_owned_fds(monkeypatch)
        interrupt = exc_type("cleanup")
        _arm_cleanup_interrupt(monkeypatch, seen, interrupt)

        with pytest.raises(exc_type) as info:
            _call(s)
        assert info.value is interrupt
        for fd in (seen["root_fd"], seen["parent_fd"]):
            with pytest.raises(OSError) as closed:
                os.fstat(fd)
            assert closed.value.errno == errno.EBADF


def test_cleanup_oserror_on_success_is_receipt_io_error(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        seen = _capture_owned_fds(monkeypatch)
        _arm_cleanup_interrupt(monkeypatch, seen, OSError("close failed"))
        _expect(s, "receipt_io_error")
        for fd in (seen["root_fd"], seen["parent_fd"]):
            with pytest.raises(OSError) as closed:
                os.fstat(fd)
            assert closed.value.errno == errno.EBADF


def test_read_interrupt_survives_distinct_cleanup_interrupt(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        seen = _capture_owned_fds(monkeypatch)
        read_interrupt = KeyboardInterrupt("receipt-read")
        cleanup_interrupt = SystemExit("cleanup")
        _arm_cleanup_interrupt(monkeypatch, seen, cleanup_interrupt)

        def boom(*args, **kwargs):
            raise read_interrupt

        monkeypatch.setattr(store, "_read_file", boom)

        with pytest.raises(KeyboardInterrupt) as info:
            _call(s)
        assert info.value is read_interrupt
        for fd in (seen["root_fd"], seen["parent_fd"]):
            with pytest.raises(OSError) as closed:
                os.fstat(fd)
            assert closed.value.errno == errno.EBADF


def test_corrupt_artifact_same_length(tmp_path, monkeypatch):
    with scenario(tmp_path, monkeypatch) as s:
        original = s["blobs"]["a.bin"]
        corrupted = bytes([original[0] ^ 0xFF]) + original[1:]
        assert len(corrupted) == len(original)
        assert corrupted != original
        (pathlib.Path(s["root"]) / "a.bin").write_bytes(corrupted)
        _expect(s, "artifact_mismatch")


def _artifact_mutations():
    def entry_of(entries):
        return copy.deepcopy(entries[0])

    def with_field(entries, **changes):
        entry = copy.deepcopy(entries[0])
        entry.update(changes)
        return [entry]

    return {
        "duplicate_entry": lambda entries: [
            entry_of(entries),
            entry_of(entries),
        ],
        "duplicate_name": lambda entries: [
            entry_of(entries),
            dict(entry_of(entries), sha256="1" * 64),
        ],
        "empty": lambda entries: [],
        "too_many": lambda entries: [
            {"name": f"n{i}.bin", "size_bytes": 0, "sha256": "0" * 64}
            for i in range(tr._MAX_ARTIFACTS + 1)
        ],
        "non_dict": lambda entries: ["not-a-dict"],
        "negative_size": lambda entries: with_field(entries, size_bytes=-1),
        "float_size": lambda entries: with_field(entries, size_bytes=1.5),
        "bool_size": lambda entries: with_field(entries, size_bytes=True),
        "over_limit_size": lambda entries: with_field(
            entries, size_bytes=tr._MAX_ARTIFACT_BYTES + 1
        ),
        "unsafe_name": lambda entries: with_field(entries, name="a/b"),
        "upper_sha": lambda entries: with_field(entries, sha256="A" * 64),
        "nonhex_sha": lambda entries: with_field(entries, sha256="Z" * 64),
    }


_ARTIFACT_MUTATION_KINDS = sorted(_artifact_mutations())


@pytest.mark.parametrize("kind", _ARTIFACT_MUTATION_KINDS)
def test_invalid_receipt_artifacts(tmp_path, monkeypatch, kind):
    with scenario(tmp_path, monkeypatch) as s:
        mutate = _artifact_mutations()[kind]
        s["receipt"]["artifacts"] = mutate(s["receipt"]["artifacts"])
        _reseal(s["receipt"], s["terminal_event"])
        _rewrite_receipt(s)
        _expect(s, "invalid_receipt")


@pytest.mark.parametrize(
    "bad_names",
    [
        [],
        [f"n{i}.bin" for i in range(tr._MAX_ARTIFACTS + 1)],
        ["b.bin", "a.bin"],
    ],
)
def test_bad_expected_names_rejected_before_read(tmp_path, monkeypatch, bad_names):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("filesystem access is forbidden")

    with scenario(tmp_path, monkeypatch, artifact_names=("a.bin", "b.bin")) as s:
        monkeypatch.setattr(store, "_read_file", forbidden)
        monkeypatch.setattr(store, "_resolve_parent", forbidden)
        monkeypatch.setattr(store, "_open_dir", forbidden)
        _expect(s, "invalid_input", expected_artifact_names=bad_names)

    assert calls == []


@pytest.mark.parametrize(
    "corrupt",
    ["terminal_hash", "current_event_hash", "stale_head"],
)
def test_malformed_journal_rejected_before_read(tmp_path, monkeypatch, corrupt):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("filesystem access is forbidden")

    with scenario(tmp_path, monkeypatch) as s:
        monkeypatch.setattr(store, "_read_file", forbidden)
        monkeypatch.setattr(store, "_resolve_parent", forbidden)
        monkeypatch.setattr(store, "_open_dir", forbidden)

        if corrupt == "terminal_hash":
            s["terminal_event"]["event_sha256"] = "0" * 64
            _expect(s, "invalid_journal")
        elif corrupt == "current_event_hash":
            s["events"][-1]["event_sha256"] = "0" * 64
            _expect(s, "invalid_journal")
        else:
            _expect(s, "invalid_journal", expected_head_sha256="0" * 64)

    assert calls == []


def test_require_scientific_execution_denies():
    with pytest.raises(tr.ReceiptError) as info:
        tr.require_scientific_execution(execution_authorized=True, root="/tmp")
    assert info.value.reason_code == "scientific_execution_not_authorized"


def test_receipt_error_normalizes_reason():
    assert tr.ReceiptError("not-a-code").reason_code == "invalid_input"
