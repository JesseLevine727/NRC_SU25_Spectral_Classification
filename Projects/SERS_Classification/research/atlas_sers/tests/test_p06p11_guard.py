"""Tests for the p06p11 parent guard."""

from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

from atlas_sers.evaluation import p06p11_guard as guard


@pytest.fixture(autouse=True)
def _isolate_process_group_signals(monkeypatch):
    monkeypatch.setattr(guard.os, "killpg", lambda *_args, **_kwargs: None)


class _ScriptedClock:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def __call__(self):
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


class _FakeProc:
    def __init__(self, returns, pid=4242):
        self.pid = pid
        self.returns = list(returns)
        self.term = False

    def poll(self):
        if len(self.returns) > 1:
            return self.returns.pop(0)
        return self.returns[0]

    def wait(self, timeout=None):
        return self.poll()

    def kill(self):
        self.term = True


def test_parse_status_scales_kib():
    text = "Name:\tpython\nVmRSS:\t  1024 kB\nVmHWM:\t 2048 kB\n"
    assert guard._parse_status(text) == {"VmRSS": 1024 * 1024, "VmHWM": 2048 * 1024}


def test_measure_output_counts_nested(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "a.bin").write_bytes(b"x" * 7)
    (tmp_path / "b.bin").write_bytes(b"y" * 3)
    assert guard._measure_output(tmp_path) == 10


def test_measure_output_rejects_symlink(tmp_path):
    (tmp_path / "real").write_bytes(b"x")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    with pytest.raises(guard.GuardError) as exc:
        guard._measure_output(tmp_path)
    assert exc.value.reason_code == "symlink_detected"


def test_preflight_rejects_existing_output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    assert guard._preflight("probe", out, None, project_root=None) == "output_exists"


def test_preflight_rejects_repo_nested_output(tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    out = root / "guard"
    assert guard._preflight("probe", out, None, project_root=root) == "output_in_repository"


def test_preflight_requires_artifact_root_for_full(tmp_path):
    assert (
        guard._preflight("full", tmp_path / "out", None, project_root=None)
        == "artifact_root_required"
    )


def test_preflight_rejects_unknown_mode(tmp_path):
    assert guard._preflight("turbo", tmp_path / "out", None, project_root=None) == "invalid_mode"


def test_preflight_normalizes_dotdot_for_containment(tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / "nested").mkdir()
    out = root / "nested" / ".." / "guard"
    assert guard._preflight("probe", out, None, project_root=root) == "output_in_repository"


def test_preflight_rejects_symlinked_input_root(tmp_path):
    real = tmp_path / "real_in"
    real.mkdir()
    link = tmp_path / "link_in"
    link.symlink_to("real_in")
    out = tmp_path / "guard"
    assert guard._preflight("probe", out, link, project_root=None) == "input_root_symlink_rejected"


def test_monitor_success_returns_metrics(tmp_path):
    proc = _FakeProc([None, None, 0])
    clock = _ScriptedClock([0.0, 0.05, 0.2, 0.3, 0.3])
    statuses = iter([{"VmRSS": 1000, "VmHWM": 2000}, {"VmRSS": 1500, "VmHWM": 2500}])
    result = guard._monitor(
        proc,
        tmp_path,
        wall_limit=10.0,
        clock=clock,
        sleep=lambda _: None,
        status_reader=lambda pid: next(statuses),
        children_reader=lambda pid: "",
        sizer=lambda root: 5,
    )
    assert result["child_returncode"] == 0
    assert result["peak_rss_bytes"] == 2500
    assert result["output_bytes"] == 5
    assert result["elapsed_seconds"] == 0.3


def test_monitor_stops_on_descendant(tmp_path):
    proc = _FakeProc([None, None, 0])
    with pytest.raises(guard.GuardError) as exc:
        guard._monitor(
            proc,
            tmp_path,
            wall_limit=10.0,
            clock=_ScriptedClock([0.0, 0.1, 0.2]),
            sleep=lambda _: None,
            status_reader=lambda pid: {"VmRSS": 1, "VmHWM": 1},
            children_reader=lambda pid: "999",
            sizer=lambda root: 0,
        )
    assert exc.value.reason_code == "unexpected_descendant"


def test_monitor_stops_on_error(tmp_path):
    proc = _FakeProc([None, None, 0])

    def boom(pid):
        raise guard.GuardError("monitor_failed")

    with pytest.raises(guard.GuardError) as exc:
        guard._monitor(
            proc,
            tmp_path,
            wall_limit=10.0,
            clock=_ScriptedClock([0.0, 0.1, 0.2]),
            sleep=lambda _: None,
            status_reader=boom,
            children_reader=lambda pid: "",
            sizer=lambda root: 0,
        )
    assert exc.value.reason_code == "monitor_failed"


def test_monitor_stops_on_wall_budget(tmp_path):
    proc = _FakeProc([None, None])
    with pytest.raises(guard.GuardError) as exc:
        guard._monitor(
            proc,
            tmp_path,
            wall_limit=0.5,
            clock=_ScriptedClock([0.0, 0.6, 0.7]),
            sleep=lambda _: None,
            status_reader=lambda pid: {"VmRSS": 1, "VmHWM": 1},
            children_reader=lambda pid: "",
            sizer=lambda root: 0,
        )
    assert exc.value.reason_code == "wall_exceeded"


def test_terminate_sends_sigterm_to_owned_group(monkeypatch):
    proc = _FakeProc([None, None])
    calls = []
    monkeypatch.setattr(guard.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    guard._terminate(proc)
    assert calls[0] == (proc.pid, signal.SIGTERM)


def test_run_guard_child_failed(tmp_path, monkeypatch):
    out = tmp_path / "guard"
    proc = _FakeProc([1])
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(
        guard,
        "_monitor",
        lambda process, root, **kwargs: {
            "child_returncode": 1,
            "elapsed_seconds": 0.1,
            "peak_rss_bytes": 10,
            "output_bytes": 0,
        },
    )
    receipt = guard.run_guard(mode="probe", output=out)
    assert receipt["status"] == "failed"
    assert receipt["reason_code"] == "child_failed"


def test_run_guard_kills_on_budget(tmp_path, monkeypatch):
    out = tmp_path / "guard"
    proc = _FakeProc([None])
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: proc)

    def fake_terminate(p):
        p.term = True
        p.returns = [-15]

    monkeypatch.setattr(guard, "_terminate", fake_terminate)

    def boom(process, root, **kwargs):
        raise guard.GuardError("wall_exceeded")

    monkeypatch.setattr(guard, "_monitor", boom)
    receipt = guard.run_guard(mode="probe", output=out)
    assert receipt["status"] == "failed"
    assert receipt["reason_code"] == "wall_exceeded"
    assert proc.term is True


def _write_child_receipt(root, mode="probe", peak=50):
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "receipt.json").write_text(
        json.dumps(
            {
                "status": "success",
                "mode": mode,
                "draws": 100 if mode == "probe" else 10000,
                "peak_rss_bytes": peak,
            }
        )
    )


def _success_monitor(peak=100, mode="probe"):
    def monitor(process, path, **kwargs):
        _write_child_receipt(path, mode=mode, peak=peak)
        return {
            "child_returncode": 0,
            "elapsed_seconds": 0.2,
            "peak_rss_bytes": peak,
            "output_bytes": 0,
        }

    return monitor


def test_run_guard_ordinary_success(tmp_path, monkeypatch):
    out = tmp_path / "guard"
    proc = _FakeProc([0])
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(guard, "_monitor", _success_monitor())
    receipt = guard.run_guard(mode="probe", output=out)
    assert receipt["status"] == "passed"
    assert receipt["reason_code"] is None
    assert (out / "guard_receipt.json").is_file()


def test_run_guard_missing_child_receipt_fails(tmp_path, monkeypatch):
    out = tmp_path / "guard"
    proc = _FakeProc([0])
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(
        guard,
        "_monitor",
        lambda process, path, **kwargs: {
            "child_returncode": 0,
            "elapsed_seconds": 0.2,
            "peak_rss_bytes": 100,
            "output_bytes": 0,
        },
    )
    receipt = guard.run_guard(mode="probe", output=out)
    assert receipt["status"] == "failed"
    assert receipt["reason_code"] == "child_receipt_invalid"


def test_run_guard_child_peak_over_limit_fails(tmp_path, monkeypatch):
    out = tmp_path / "guard"
    proc = _FakeProc([0])
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: proc)

    def monitor(process, path, **kwargs):
        _write_child_receipt(path, peak=guard.RSS_LIMIT + 1)
        return {
            "child_returncode": 0,
            "elapsed_seconds": 0.2,
            "peak_rss_bytes": 100,
            "output_bytes": 0,
        }

    monkeypatch.setattr(guard, "_monitor", monitor)
    receipt = guard.run_guard(mode="probe", output=out)
    assert receipt["status"] == "failed"
    assert receipt["reason_code"] == "rss_exceeded"


def test_run_guard_full_wall_finalization_fails(tmp_path, monkeypatch):
    inroot = tmp_path / "input"
    inroot.mkdir()
    out = tmp_path / "guard"
    proc = _FakeProc([0])
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(guard, "_monitor", _success_monitor(mode="full"))
    monkeypatch.setattr(
        guard.time,
        "monotonic",
        _ScriptedClock([0.0, guard.MODES["full"] + 1.0]),
    )
    receipt = guard.run_guard(mode="full", output=out, artifact_root=inroot)
    assert receipt["status"] == "failed"
    assert receipt["reason_code"] == "wall_exceeded"


def test_run_guard_keyboard_interrupt(tmp_path, monkeypatch):
    out = tmp_path / "guard"
    proc = _FakeProc([None])
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: proc)

    def fake_terminate(p):
        p.term = True
        p.returns = [-15]

    monkeypatch.setattr(guard, "_terminate", fake_terminate)

    def interrupt(process, path, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(guard, "_monitor", interrupt)
    receipt = guard.run_guard(mode="probe", output=out)
    assert receipt["status"] == "failed"
    assert receipt["reason_code"] == "interrupted"
    assert proc.term is True
    assert (out / "guard_receipt.json").is_file()


def test_run_guard_termination_unconfirmed(tmp_path, monkeypatch):
    out = tmp_path / "guard"
    proc = _FakeProc([None])
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(guard, "_terminate", lambda p: None)

    def boom(process, root, **kwargs):
        raise guard.GuardError("wall_exceeded")

    monkeypatch.setattr(guard, "_monitor", boom)
    receipt = guard.run_guard(mode="probe", output=out)
    assert receipt["status"] == "failed"
    assert receipt["reason_code"] == "termination_unconfirmed"


def test_read_child_peak_requires_analysis_receipt(tmp_path):
    with pytest.raises(guard.GuardError) as exc:
        guard._read_child_peak(tmp_path, "probe")
    assert exc.value.reason_code == "child_receipt_invalid"


def test_read_child_peak_validates_fields(tmp_path):
    _write_child_receipt(tmp_path, peak=1234)
    assert guard._read_child_peak(tmp_path, "probe") == 1234
    _write_child_receipt(tmp_path, mode="full", peak=7)
    with pytest.raises(guard.GuardError) as exc:
        guard._read_child_peak(tmp_path, "probe")
    assert exc.value.reason_code == "child_receipt_invalid"


def test_monitor_failure_retains_metrics(tmp_path):
    proc = _FakeProc([None, None, None])
    with pytest.raises(guard.GuardError) as exc:
        guard._monitor(
            proc,
            tmp_path,
            wall_limit=10.0,
            rss_limit=100,
            clock=_ScriptedClock([0.0, 0.1, 0.2, 0.3]),
            sleep=lambda _: None,
            status_reader=lambda pid: {"VmRSS": 50, "VmHWM": 500},
            children_reader=lambda pid: "",
            sizer=lambda root: 0,
        )
    assert exc.value.reason_code == "rss_exceeded"
    assert exc.value.metrics is not None
    assert exc.value.metrics["peak_rss_bytes"] == 500


def test_monitor_exit_race_missing_status_is_normal(tmp_path):
    proc = _FakeProc([None, 0])
    result = guard._monitor(
        proc,
        tmp_path,
        wall_limit=10.0,
        clock=_ScriptedClock([0.0, 0.1, 0.2, 0.3]),
        sleep=lambda _: None,
        status_reader=lambda pid: {},
        children_reader=lambda pid: "",
        sizer=lambda root: 0,
    )
    assert result["child_returncode"] == 0


def test_child_command_and_env():
    cmd = guard._child_command("probe", Path("/tmp/x"), Path("/data/in"))
    assert cmd[0] == guard.sys.executable
    assert cmd[1:3] == ["-m", guard.CHILD_MODULE]
    assert "--artifact-root" in cmd
    env = guard._child_env()
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_cli_invalid_args_exit_2():
    with pytest.raises(SystemExit) as exc:
        guard.main([])
    assert exc.value.code == 2


def test_cli_reports_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        guard,
        "run_guard",
        lambda **kwargs: {"status": "failed", "reason_code": "wall_exceeded"},
    )
    assert guard.main(["--mode", "probe", "--output", str(tmp_path / "out")]) == 1
    output = capsys.readouterr().out
    assert "failed" in output
    assert "wall_exceeded" in output
