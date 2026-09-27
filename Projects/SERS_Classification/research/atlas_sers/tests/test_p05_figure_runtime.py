# ruff: noqa: E501
"""Tests for the bounded P05 figure compilation runtime."""

from __future__ import annotations

import subprocess
import types
from pathlib import Path

import pytest

from atlas_sers.visualization import p05_figure_runtime as runtime


class Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class Runner:
    def __init__(
        self,
        *,
        pdflatex_code: int = 0,
        pdftocairo_code: int = 0,
        pdflatex_timeout: bool = False,
        pdftocairo_timeout: bool = False,
        on_pdflatex=None,
    ) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.pdflatex_code = pdflatex_code
        self.pdftocairo_code = pdftocairo_code
        self.pdflatex_timeout = pdflatex_timeout
        self.pdftocairo_timeout = pdftocairo_timeout
        self.on_pdflatex = on_pdflatex

    def __call__(self, command, **kwargs):
        command = list(command)
        self.calls.append((command, kwargs))
        if command[0] == "pdflatex":
            if self.on_pdflatex is not None:
                self.on_pdflatex()
            if self.pdflatex_timeout:
                raise subprocess.TimeoutExpired(
                    command, kwargs["timeout"], output="partial", stderr="boom"
                )
            (Path(kwargs["cwd"]) / f"{Path(command[-1]).stem}.pdf").write_text("pdf")
            return subprocess.CompletedProcess(
                command, self.pdflatex_code, "latex out", "latex err"
            )
        if self.pdftocairo_timeout:
            raise subprocess.TimeoutExpired(
                command, kwargs["timeout"], output="partial", stderr="boom"
            )
        Path(command[-1] + ".png").write_text("png")
        return subprocess.CompletedProcess(command, self.pdftocairo_code, "cairo out", "cairo err")


def _fake(runner: Runner) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        run=runner,
        TimeoutExpired=subprocess.TimeoutExpired,
        SubprocessError=subprocess.SubprocessError,
    )


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    tex = tmp_path / "figure.tex"
    tex.write_text("tex")
    return {
        "tex": tex,
        "pdf": tmp_path / "figure.pdf",
        "png": tmp_path / "figure.png",
        "log": tmp_path / "figure.pdflatex.log",
    }


def _wire(monkeypatch, runner: Runner, clock: Clock) -> None:
    monkeypatch.setattr(runtime, "perf_counter", clock)
    monkeypatch.setattr(runtime, "subprocess", _fake(runner))


def _compile(paths, **kwargs) -> None:
    runtime._compile(paths["tex"], paths["pdf"], paths["png"], paths["log"], **kwargs)


def test_no_deadline_uses_120_and_succeeds(monkeypatch, paths):
    runner, clock = Runner(), Clock()
    _wire(monkeypatch, runner, clock)
    _compile(paths)
    assert [call[0] for call in runner.calls] == [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "figure.tex"],
        [
            "pdftocairo",
            "-png",
            "-singlefile",
            "-r",
            "300",
            str(paths["pdf"]),
            str(paths["png"].with_suffix("")),
        ],
    ]
    assert all(call[1]["timeout"] == 120.0 for call in runner.calls)
    assert paths["pdf"].read_text() == "pdf"
    assert paths["png"].read_text() == "png"
    assert not paths["log"].exists()


def test_deadline_bounds_and_recomputes_before_second_child(monkeypatch, paths):
    clock = Clock(100.0)
    runner = Runner(on_pdflatex=lambda: setattr(clock, "value", 130.0))
    _wire(monkeypatch, runner, clock)
    _compile(paths, deadline=150.0)
    assert runner.calls[0][1]["timeout"] == 50.0
    assert runner.calls[1][1]["timeout"] == 20.0


def test_child_timeout_is_capped_at_120(monkeypatch, paths):
    runner, clock = Runner(), Clock(0.0)
    _wire(monkeypatch, runner, clock)
    _compile(paths, deadline=1000.0)
    assert [call[1]["timeout"] for call in runner.calls] == [120.0, 120.0]


def test_expired_deadline_fails_before_any_child(monkeypatch, paths):
    runner, clock = Runner(), Clock(100.0)
    _wire(monkeypatch, runner, clock)
    with pytest.raises(runtime.P05FigureRuntimeError) as info:
        _compile(paths, deadline=100.0)
    assert info.value.reason_code == "deadline_expired"
    assert runner.calls == []


def test_deadline_checked_after_child(monkeypatch, paths):
    clock = Clock(100.0)
    runner = Runner(on_pdflatex=lambda: setattr(clock, "value", 151.0))
    _wire(monkeypatch, runner, clock)
    with pytest.raises(runtime.P05FigureRuntimeError) as info:
        _compile(paths, deadline=150.0)
    assert info.value.reason_code == "deadline_expired"
    assert len(runner.calls) == 1


def test_pdflatex_timeout_preserves_log_and_stops(monkeypatch, paths):
    runner, clock = Runner(pdflatex_timeout=True), Clock()
    _wire(monkeypatch, runner, clock)
    with pytest.raises(runtime.P05FigureRuntimeError) as info:
        _compile(paths)
    assert info.value.reason_code == "pdflatex_timeout"
    assert len(runner.calls) == 1
    assert "partial" in paths["log"].read_text()


def test_pdflatex_nonzero_stops_and_never_claims_success(monkeypatch, paths):
    runner, clock = Runner(pdflatex_code=1), Clock()
    _wire(monkeypatch, runner, clock)
    with pytest.raises(runtime.P05FigureRuntimeError) as info:
        _compile(paths)
    assert str(info.value) == "pdflatex_failed"
    assert info.value.reason_code == "pdflatex_failed"
    assert len(runner.calls) == 1
    assert paths["log"].exists()
    assert not paths["pdf"].exists() and not paths["png"].exists()


def test_pdftocairo_timeout_after_first_child(monkeypatch, paths):
    runner, clock = Runner(pdftocairo_timeout=True), Clock()
    _wire(monkeypatch, runner, clock)
    with pytest.raises(runtime.P05FigureRuntimeError) as info:
        _compile(paths)
    assert info.value.reason_code == "pdftocairo_timeout"
    assert len(runner.calls) == 2
    assert paths["log"].exists()


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), True, False, "123", {}, []]
)
def test_invalid_deadline_rejected(monkeypatch, paths, value):
    runner, clock = Runner(), Clock()
    _wire(monkeypatch, runner, clock)
    with pytest.raises(runtime.P05FigureRuntimeError) as info:
        _compile(paths, deadline=value)
    assert info.value.reason_code == "invalid_deadline"
    assert runner.calls == []
