# ruff: noqa: E501
"""Bounded runtime helpers for compiling P05 figure artifacts."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

_MAX_COMPILE_SECONDS = 120.0


class P05FigureRuntimeError(RuntimeError):
    """Stable, path-free failure for P05 figure compilation."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _validate_deadline(deadline: float | None) -> None:
    if deadline is None:
        return
    if (
        not isinstance(deadline, (int, float))
        or isinstance(deadline, bool)
        or not math.isfinite(deadline)
    ):
        raise P05FigureRuntimeError("invalid_deadline")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise P05FigureRuntimeError("deadline_expired")


def _child_timeout(deadline: float | None) -> float:
    if deadline is None:
        return _MAX_COMPILE_SECONDS
    remaining = deadline - perf_counter()
    if remaining <= 0.0:
        raise P05FigureRuntimeError("deadline_expired")
    return min(_MAX_COMPILE_SECONDS, remaining)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _write_log(path: Path, stdout: object, stderr: object, *, append: bool) -> None:
    text = _as_text(stdout) + "\n" + _as_text(stderr)
    try:
        if append and path.exists():
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n" + text)
        else:
            path.write_text(text, encoding="utf-8")
    except OSError:
        return


def _run(
    command: tuple[str, ...],
    *,
    cwd: Path | None = None,
    timeout: float,
    log_path: Path,
    timeout_reason: str,
    failure_reason: str,
    append: bool,
) -> None:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        _write_log(log_path, error.stdout, error.stderr, append=append)
        raise P05FigureRuntimeError(timeout_reason) from error
    _write_log(log_path, result.stdout, result.stderr, append=append)
    if result.returncode:
        raise P05FigureRuntimeError(failure_reason)


def _compile(
    tex_path: Path,
    pdf_path: Path,
    png_path: Path,
    log_path: Path,
    *,
    deadline: float | None = None,
) -> None:
    _validate_deadline(deadline)
    _check_deadline(deadline)
    try:
        with tempfile.TemporaryDirectory(prefix="p05-figure-") as temporary_name:
            temporary = Path(temporary_name)
            local = temporary / tex_path.name
            shutil.copy2(tex_path, local)
            _run(
                ("pdflatex", "-interaction=nonstopmode", "-halt-on-error", local.name),
                cwd=temporary,
                timeout=_child_timeout(deadline),
                log_path=log_path,
                timeout_reason="pdflatex_timeout",
                failure_reason="pdflatex_failed",
                append=False,
            )
            _check_deadline(deadline)
            shutil.copy2(temporary / f"{tex_path.stem}.pdf", pdf_path)
        _run(
            (
                "pdftocairo",
                "-png",
                "-singlefile",
                "-r",
                "300",
                str(pdf_path),
                str(png_path.with_suffix("")),
            ),
            timeout=_child_timeout(deadline),
            log_path=log_path,
            timeout_reason="pdftocairo_timeout",
            failure_reason="pdftocairo_failed",
            append=True,
        )
        _check_deadline(deadline)
        if not pdf_path.is_file() or not png_path.is_file():
            raise P05FigureRuntimeError("output_missing")
    except P05FigureRuntimeError:
        raise
    except (subprocess.SubprocessError, OSError) as error:
        raise P05FigureRuntimeError("runtime_error") from error
    log_path.unlink(missing_ok=True)
