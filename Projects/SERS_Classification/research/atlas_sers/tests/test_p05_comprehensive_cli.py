"""Sanitized CLI tests for scripts/run_p05_comprehensive.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_CLI_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_p05_comprehensive.py"
_BASE = ["--project-root", "/proj", "--artifact-root", "/art", "--contract", "/c", "--permit", "/p"]


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_p05_comprehensive_cli", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Canon:
    @staticmethod
    def canonical_json_bytes(payload):
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class _InputsError(Exception):
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


def _bundle():
    return {
        "contract_sha256": "c" * 64,
        "permit_sha256": "p" * 64,
        "core_plan_id": "a" * 64,
        "ledger": {
            "schema_version": "nato-sers-p05-comprehensive-v1",
            "ledger_id": "LEDGER",
            "summary": {
                "context_count": 320,
                "contexts_by_selection_mode": {"source_validation": 36},
                "contexts_by_phase_gate": {"primary": 36},
                "eligible_unit_count": 10,
                "excluded_unit_count": 5,
                "slot_count": 20,
                "eligible_slot_count": 15,
                "excluded_slot_count": 5,
            },
        },
        "pilot_bundle": {"slots": [1, 2, 3]},
    }


def test_inspect_metadata_only(capsys):
    cli = _cli()
    calls = []

    class _In:
        ComprehensiveInputsError = _InputsError

        @staticmethod
        def prepare(
            project_root, artifact_root, contract_path, permit_path, *, require_unstarted=False
        ):
            calls.append(
                (project_root, artifact_root, contract_path, permit_path, require_unstarted)
            )
            return _bundle()

    cli._inputs = lambda: _In
    cli._module = lambda name: (_ for _ in ()).throw(AssertionError(name))
    code, payload = _run(cli, ["inspect", *_BASE], capsys)
    assert code == 0 and calls == [("/proj", "/art", "/c", "/p", True)]
    assert payload["command"] == "inspect" and payload["arrays_loaded"] is False
    assert payload["fits_started"] == 0 and payload["execution_authorized"] is False
    assert payload["reused_pilot_slots"] == 3 and "torch" not in json.dumps(payload)


def test_develop_forwards_kwargs_only_cuda(capsys):
    cli = _cli()
    captured = {}

    class _Dev:
        @staticmethod
        def run_development(**kwargs):
            captured.update(kwargs)
            return {"status": "complete", "started": 0, "completed": 36}

    def _module(name):
        assert name == "atlas_sers.evaluation.p05_comprehensive_development", name
        return _Dev

    cli._module = _module
    code, payload = _run(cli, ["develop", *_BASE], capsys)
    assert code == 0
    assert captured == {
        "project_root": "/proj",
        "artifact_root": "/art",
        "contract_path": "/c",
        "permit_path": "/p",
        "device": "cuda",
    }
    assert payload["status"] == "complete" and payload["command"] == "develop"


def test_freeze_selection_forwards_kwargs_without_device(capsys):
    cli = _cli()
    captured = {}

    class _Freeze:
        @staticmethod
        def freeze_selection(**kwargs):
            captured.update(kwargs)
            return {"status": "complete", "frozen_selections": 4}

    def _module(name):
        assert name == "atlas_sers.evaluation.p05_comprehensive_freeze", name
        return _Freeze

    cli._module = _module
    code, payload = _run(cli, ["freeze-selection", *_BASE], capsys)
    assert code == 0
    assert captured == {
        "project_root": "/proj",
        "artifact_root": "/art",
        "contract_path": "/c",
        "permit_path": "/p",
    }
    assert "device" not in captured
    assert payload == {
        "status": "complete",
        "frozen_selections": 4,
        "command": "freeze-selection",
    }


def test_refits_forwards_kwargs_only_cuda(capsys):
    cli = _cli()
    captured = {}

    class _Refits:
        @staticmethod
        def run_refits(**kwargs):
            captured.update(kwargs)
            return {"status": "complete", "refits_started": 4}

    def _module(name):
        assert name == "atlas_sers.evaluation.p05_comprehensive_refits", name
        return _Refits

    cli._module = _module
    code, payload = _run(cli, ["refits", *_BASE], capsys)
    assert code == 0
    assert captured == {
        "project_root": "/proj",
        "artifact_root": "/art",
        "contract_path": "/c",
        "permit_path": "/p",
        "device": "cuda",
    }
    assert payload == {
        "status": "complete",
        "refits_started": 4,
        "command": "refits",
    }


def test_evaluate_forwards_kwargs_only_cuda(capsys):
    cli = _cli()
    captured = {}

    class _Eval:
        @staticmethod
        def run_evaluation(**kwargs):
            captured.update(kwargs)
            return {"status": "complete", "evaluated": 4}

    def _module(name):
        assert name == "atlas_sers.evaluation.p05_comprehensive_evaluation", name
        return _Eval

    cli._module = _module
    code, payload = _run(cli, ["evaluate", *_BASE], capsys)
    assert code == 0
    assert captured == {
        "project_root": "/proj",
        "artifact_root": "/art",
        "contract_path": "/c",
        "permit_path": "/p",
        "device": "cuda",
    }
    assert payload == {"status": "complete", "evaluated": 4, "command": "evaluate"}


def test_aggregate_forwards_kwargs_without_device(capsys):
    cli = _cli()
    captured = {}

    class _Agg:
        @staticmethod
        def run_aggregation(**kwargs):
            captured.update(kwargs)
            return {"status": "complete", "aggregated": 4}

    def _module(name):
        assert name == "atlas_sers.evaluation.p05_comprehensive_aggregation", name
        return _Agg

    cli._module = _module
    code, payload = _run(cli, ["aggregate", *_BASE], capsys)
    assert code == 0
    assert captured == {
        "project_root": "/proj",
        "artifact_root": "/art",
        "contract_path": "/c",
        "permit_path": "/p",
    }
    assert "device" not in captured
    assert payload == {"status": "complete", "aggregated": 4, "command": "aggregate"}


def test_new_command_failures_are_sanitized(capsys):
    class _Boom(Exception):
        def __init__(self):
            self.reason_code = "stage_failed"
            super().__init__("/secret/implementation.py:42")

    def _make_stage(attribute):
        def _fail(**kwargs):
            raise _Boom()

        class _Stage:
            pass

        setattr(_Stage, attribute, staticmethod(_fail))
        return _Stage

    for command, module_name, attribute in (
        (
            "freeze-selection",
            "atlas_sers.evaluation.p05_comprehensive_freeze",
            "freeze_selection",
        ),
        ("refits", "atlas_sers.evaluation.p05_comprehensive_refits", "run_refits"),
        (
            "evaluate",
            "atlas_sers.evaluation.p05_comprehensive_evaluation",
            "run_evaluation",
        ),
        (
            "aggregate",
            "atlas_sers.evaluation.p05_comprehensive_aggregation",
            "run_aggregation",
        ),
    ):
        cli = _cli()
        stage = _make_stage(attribute)

        def _module(name, _module_name=module_name, _stage=stage):
            assert name == _module_name, name
            return _stage

        cli._module = _module
        code, payload = _run(cli, [command, *_BASE], capsys)
        assert code == 1
        assert payload == {"status": "fail", "command": command, "reason_code": "stage_failed"}
        assert "/secret" not in json.dumps(payload)
        assert "Traceback" not in json.dumps(payload)


def test_invalid_arguments_sanitized(capsys):
    cli = _cli()
    cases = (
        ["develop", *_BASE, "--device", "cpu"],
        ["develop", *_BASE, "--force"],
        ["inspect", *_BASE, "--resume"],
        ["inspect", "--project-root", "/proj"],
        ["refits", *_BASE, "--device", "cpu"],
        ["freeze-selection", *_BASE, "--device", "cuda"],
        ["evaluate", *_BASE, "--device", "cpu"],
        ["aggregate", *_BASE, "--device", "cuda"],
    )
    for argv in cases:
        code, payload = _run(cli, argv, capsys)
        assert code == 1 and set(payload) == {"status", "command", "reason_code"}
        assert payload["reason_code"] == "invalid_arguments" and "/proj" not in json.dumps(payload)


def _status_cli(progress, recorder, error=None):
    class _In:
        ComprehensiveInputsError = _InputsError

        @staticmethod
        def _load_permit(permit_path):
            return {"permit": True}, "digest123"

    class _Pilot:
        @staticmethod
        def _resolve_paths(project_root, artifact_root):
            return Path(project_root), Path(artifact_root), Path(project_root)

    class _Core:
        @staticmethod
        def _read_json(path, kind):
            recorder.append(Path(path))
            if error is not None:
                raise error
            return progress

    modules = {
        "atlas_sers.evaluation.p05_pilot": _Pilot,
        "atlas_sers.evaluation.p05_core_run": _Core,
    }
    cli = _cli()
    cli._inputs = lambda: _In
    cli._module = modules.__getitem__
    return cli


def test_status_reads_comprehensive_namespace_and_whitelists(capsys):
    recorder = []
    progress = {
        "started": 5,
        "completed": 3,
        "failed": 0,
        "new_completions": 3,
        "units_completed": 1,
        "units_total": 10,
        "optimizer_steps": 100,
        "optimizer_steps_exact": True,
        "secret_path": "/secret",
        "uid": "u1",
    }
    cli = _status_cli(progress, recorder)
    code, payload = _run(cli, ["status", *_BASE], capsys)
    assert code == 0
    assert recorder[0].parts[-5:] == (
        "p05comprehensive",
        "runs",
        "digest123",
        "develop",
        "progress.json",
    )
    assert "p05development" not in str(recorder[0])
    assert set(payload) == {"status", "command"} | set(cli._STATUS_FIELDS)
    assert (
        "secret_path" not in payload
        and "uid" not in payload
        and payload["optimizer_steps_exact"] is True
    )


def test_status_rejects_wrong_types_and_schema(capsys):
    names = (
        "started",
        "completed",
        "failed",
        "new_completions",
        "units_completed",
        "units_total",
        "optimizer_steps",
    )
    cases = [("started", "5"), ("units_total", 1.5), ("optimizer_steps_exact", 1)]
    for field, bad in cases:
        recorder = []
        progress = {name: 0 for name in names}
        progress["optimizer_steps_exact"] = True
        progress[field] = bad
        code, payload = _run(_status_cli(progress, recorder), ["status", *_BASE], capsys)
        assert code == 1 and payload["reason_code"] == "development_progress_malformed"
        assert "/proj" not in json.dumps(payload)
    recorder = []
    code, payload = _run(_status_cli(["not", "a", "mapping"], recorder), ["status", *_BASE], capsys)
    assert code == 1 and payload["reason_code"] == "development_progress_malformed"


def test_status_missing_progress_is_path_free(capsys):
    recorder = []
    cli = _status_cli({}, recorder, error=_InputsError("artifact_missing"))
    code, payload = _run(cli, ["status", *_BASE], capsys)
    assert code == 1
    assert payload == {"status": "fail", "command": "status", "reason_code": "artifact_missing"}
    assert "/art" not in json.dumps(payload)
