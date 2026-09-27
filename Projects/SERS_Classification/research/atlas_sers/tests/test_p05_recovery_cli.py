"""Sanitized CLI tests for the P05 recovery commands."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from atlas_sers.evaluation import p05_recovery_receipt as receipt
from tests.test_p05_recovery_source import _recovered_pair

_CLI_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_p05_comprehensive.py"
_BASE = ["--project-root", "/proj", "--artifact-root", "/art", "--contract", "/c", "--permit", "/p"]
_RECOVERY = [*_BASE, "--recovery-permit", "/rp"]


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_p05_recovery_cli", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Canon:
    @staticmethod
    def canonical_json_bytes(payload):
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class _RecoveryError(Exception):
    def __init__(self, reason_code):
        self.reason_code = reason_code
        super().__init__(reason_code)


def _cli():
    cli = _load_cli()
    cli._canon = lambda: _Canon()
    return cli


def _run(cli, argv, capsys):
    code = cli.main(argv)
    return code, json.loads(capsys.readouterr().out.strip())


def _claim():
    claim = receipt.validate_pair(*_recovered_pair())
    claim["uid"] = "secret-uid"
    return claim


def _modules(dev):
    return {
        "atlas_sers.evaluation.p05_recovery_development": dev,
        "atlas_sers.evaluation.p05_recovery_receipt": receipt,
    }.__getitem__


def test_recover_develop_forwards_exact_kwargs(capsys):
    cli = _cli()
    captured = {}
    calls = []

    class _Dev:
        @staticmethod
        def run_recovery_development(**kwargs):
            captured.update(kwargs)
            return _claim()

    def _module(name):
        calls.append(name)
        return _modules(_Dev)(name)

    cli._module = _module
    code, payload = _run(cli, ["recover-develop", *_RECOVERY], capsys)
    assert code == 0 and calls == [
        "atlas_sers.evaluation.p05_recovery_development",
        "atlas_sers.evaluation.p05_recovery_receipt",
    ]
    assert captured == {
        "project_root": "/proj",
        "artifact_root": "/art",
        "contract_path": "/c",
        "base_permit_path": "/p",
        "recovery_permit_path": "/rp",
        "device": "cuda",
    }
    assert payload["status"] == "complete" and payload["command"] == "recover-develop"


def test_recover_develop_output_is_whitelisted(capsys):
    cli = _cli()

    class _Dev:
        @staticmethod
        def run_recovery_development(**kwargs):
            return _claim()

    cli._module = _modules(_Dev)
    code, payload = _run(cli, ["recover-develop", *_RECOVERY], capsys)
    assert code == 0
    assert set(payload) == {"status", "command"} | set(cli._RECOVERY_DEVELOP_FIELDS)
    assert payload["started"] == 14905 and payload["optimizer_steps_exact"] is True
    assert "uid" not in payload and "hash_pins" not in payload and "aggregate" not in payload
    assert "torch" not in json.dumps(payload)


def test_recover_develop_unverified_claim_is_sanitized(capsys):
    cli = _cli()

    class _Dev:
        @staticmethod
        def run_recovery_development(**kwargs):
            return {"started": "8", "completed": 8, "failed": 0, "secret_path": "/secret"}

    cli._module = _modules(_Dev)
    code, payload = _run(cli, ["recover-develop", *_RECOVERY], capsys)
    assert code == 1 and payload["reason_code"] == "recovery_development_unverified"
    assert "/secret" not in json.dumps(payload)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("schema_version", "wrong"),
        ("claim", "complete"),
        ("optimizer_steps_scope", "/private/path"),
        ("recovery_failed", False),
        ("recovery_completed", 6183),
        ("scientific_seconds_this_stage", 10**1000),
        ("total_source_optimizer_steps_exact", True),
    ],
)
def test_recover_develop_rejects_invalid_flat_completion(field, bad, capsys):
    cli = _cli()
    value = _claim()
    value[field] = bad

    class Dev:
        @staticmethod
        def run_recovery_development(**kwargs):
            return value

    cli._module = _modules(Dev)
    code, payload = _run(cli, ["recover-develop", *_RECOVERY], capsys)
    assert code == 1
    assert payload["reason_code"] == "recovery_development_unverified"
    assert "/private/path" not in json.dumps(payload)


def test_recover_develop_does_not_flatten_nested_claim(capsys):
    cli = _cli()

    class Dev:
        @staticmethod
        def run_recovery_development(**kwargs):
            return {"invented": _claim()}

    cli._module = _modules(Dev)
    code, payload = _run(cli, ["recover-develop", *_RECOVERY], capsys)
    assert code == 1 and payload["reason_code"] == "recovery_development_unverified"


def _progress():
    return {
        "new_started": 1,
        "new_completed": 1,
        "new_failed": 0,
        "new_optimizer_steps": 10,
        "new_optimizer_steps_exact": True,
        "new_elapsed_seconds": 1.5,
        "new_peak_cuda_bytes": 1024,
        "reused_completed": 2,
        "reused_optimizer_steps": 3,
        "replay_started": 4,
        "unstarted_started": 5,
        "units_completed": 6,
        "units_total": 7,
        "secret_path": "/secret",
    }


def _status_cli(progress, recorder, *, read_error=None, base_digest="b" * 64):
    class _In:
        @staticmethod
        def _load_permit(permit_path):
            return {"permit": permit_path}, base_digest

    class _Authority:
        BASECOMPREHENSIVE_PERMIT_SHA256 = "b" * 64
        RECOVERY_PERMIT_SHA256 = "r" * 64

        @staticmethod
        def load_recovery_permit(path):
            recorder.append(("permit", path))
            return {"recovery": path}

    class _Pilot:
        @staticmethod
        def _resolve_paths(project_root, artifact_root):
            return Path(project_root), Path(artifact_root), Path(project_root)

    class _RecoveryInputs:
        @staticmethod
        def _read_json_mapping(path, kind, *, deadline):
            recorder.append((Path(path), kind, deadline))
            if read_error is not None:
                raise read_error
            return progress

    modules = {
        "atlas_sers.evaluation.p05_pilot": _Pilot,
        "atlas_sers.evaluation.p05_recovery_authority": _Authority,
        "atlas_sers.evaluation.p05_recovery_inputs": _RecoveryInputs,
    }
    cli = _cli()
    cli._inputs = lambda: _In
    cli._module = modules.__getitem__
    return cli


def test_recovery_status_exact_layout_and_whitelist(capsys):
    recorder = []
    cli = _status_cli(_progress(), recorder)
    code, payload = _run(cli, ["recovery-status", *_RECOVERY], capsys)
    assert code == 0
    assert ("permit", "/rp") in recorder
    read = next(entry for entry in recorder if isinstance(entry[0], Path))
    assert read[0] == (
        Path("/art")
        / "p05comprehensive"
        / "runs"
        / ("b" * 64)
        / "recoveries"
        / ("r" * 64)
        / "develop"
        / "progress.json"
    )
    assert read[1] == "recovery_progress" and read[2] > 0
    assert set(payload) == {"status", "command"} | set(cli._RECOVERY_PROGRESS_FIELDS)
    assert payload["new_optimizer_steps_exact"] is True
    assert payload["new_elapsed_seconds"] == 1.5 and "secret_path" not in payload


def test_recovery_status_rejects_invalid_counters(capsys):
    cases = (
        ("new_started", "1"),
        ("new_optimizer_steps_exact", 1),
        ("new_elapsed_seconds", -1.0),
        ("new_elapsed_seconds", float("nan")),
        ("new_peak_cuda_bytes", 1.5),
        ("units_total", True),
    )
    for field, bad in cases:
        progress = _progress()
        progress[field] = bad
        code, payload = _run(_status_cli(progress, []), ["recovery-status", *_RECOVERY], capsys)
        assert code == 1 and payload["reason_code"] == "recovery_progress_malformed"
        assert "/art" not in json.dumps(payload)


def test_recovery_status_missing_counter_fails(capsys):
    for field in _progress():
        if field == "secret_path":
            continue
        progress = _progress()
        del progress[field]
        code, payload = _run(_status_cli(progress, []), ["recovery-status", *_RECOVERY], capsys)
        assert code == 1 and payload["reason_code"] == "recovery_progress_malformed"


def test_recovery_status_failure_is_path_free(capsys):
    cli = _status_cli({}, [], read_error=_RecoveryError("recovery_artifact_missing"))
    code, payload = _run(cli, ["recovery-status", *_RECOVERY], capsys)
    assert code == 1
    assert payload == {
        "status": "fail",
        "command": "recovery-status",
        "reason_code": "recovery_artifact_missing",
    }
    assert "/art" not in json.dumps(payload)


def test_recovery_status_base_pin_mismatch(capsys):
    recorder = []
    cli = _status_cli(_progress(), recorder, base_digest="x" * 64)
    code, payload = _run(cli, ["recovery-status", *_RECOVERY], capsys)
    assert code == 1 and payload["reason_code"] == "base_permit_mismatch"
    assert recorder == []


def test_recovery_invalid_arguments_sanitized(capsys):
    cli = _cli()
    cases = (
        ["recover-develop", *_BASE],
        ["recovery-status", *_BASE],
        ["recover-develop", *_RECOVERY, "--device", "cpu"],
        ["recover-develop", *_RECOVERY, "--force"],
        ["recovery-status", *_RECOVERY, "--device", "cuda"],
    )
    for argv in cases:
        code, payload = _run(cli, argv, capsys)
        assert code == 1 and set(payload) == {"status", "command", "reason_code"}
        assert payload["reason_code"] == "invalid_arguments" and "/proj" not in json.dumps(payload)
