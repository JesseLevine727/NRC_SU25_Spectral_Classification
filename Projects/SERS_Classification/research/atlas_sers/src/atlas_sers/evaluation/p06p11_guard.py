"""Parent guard for the approved p06p11 inference child runner.

The guard launches exactly one fixed child module, enforces wall-clock,
resident-memory and output-size budgets, and writes a private receipt.
It never accepts an arbitrary command line.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

CHILD_MODULE = "atlas_sers.evaluation.p06p11_run"
MODES: dict[str, float] = {"probe": 60.0, "full": 1800.0}
RSS_LIMIT = 2 * 1024**3
OUTPUT_LIMIT = 1024**3
RECEIPT_RESERVE = 16 * 1024
POLL_INTERVAL = 0.1
SIZE_INTERVAL = 1.0
KILL_GRACE = 2.0
PROC = Path("/proc")
_AUTO = object()


class GuardError(Exception):
    """Fatal condition observed by the guard."""

    def __init__(self, reason_code: str, metrics: dict[str, Any] | None = None) -> None:
        self.reason_code = reason_code
        self.metrics = metrics
        super().__init__(reason_code)


def _parse_status(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _unit = line.split()
            values[key[:-1]] = int(value) * 1024
    return values


def _read_status(pid: int) -> dict[str, int]:
    try:
        return _parse_status((PROC / str(pid) / "status").read_text())
    except OSError as exc:
        raise GuardError("monitor_failed") from exc


def _read_children(pid: int) -> str:
    try:
        return (PROC / str(pid) / "task" / str(pid) / "children").read_text().strip()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise GuardError("monitor_failed") from exc


def _measure_output(root: Path) -> int:
    def _onerror(exc: OSError) -> None:
        raise GuardError("monitor_failed") from exc

    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=_onerror):
        for name in dirnames:
            if os.path.islink(os.path.join(dirpath, name)):
                raise GuardError("symlink_detected")
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                info = os.lstat(path)
            except OSError as exc:
                raise GuardError("monitor_failed") from exc
            if stat.S_ISLNK(info.st_mode):
                raise GuardError("symlink_detected")
            total += info.st_size
    return total


def _has_symlink(path: Path) -> bool:
    return any(os.path.islink(part) for part in (path, *path.parents))


def _within(child: Path, parent: Path) -> bool:
    try:
        return child == parent or child.is_relative_to(parent)
    except ValueError:
        return False


def _find_project_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def _preflight(mode: str, output, artifact_root=None, *, project_root=_AUTO) -> str | None:
    if mode not in MODES:
        return "invalid_mode"
    if mode == "full" and artifact_root is None:
        return "artifact_root_required"
    if not (sys.platform.startswith("linux") and PROC.is_dir()):
        return "proc_unavailable"
    out = Path(output).absolute()
    if _has_symlink(out):
        return "symlink_rejected"
    if not out.parent.is_dir():
        return "output_parent_missing"
    if out.exists() or out.is_symlink():
        return "output_exists"
    if (out / "analysis").exists():
        return "child_output_exists"
    out = out.resolve()
    root = _find_project_root() if project_root is _AUTO else project_root
    if root is not None and _within(out, Path(root).resolve()):
        return "output_in_repository"
    if artifact_root is not None:
        inr = Path(artifact_root)
        if _has_symlink(inr):
            return "input_root_symlink_rejected"
        inr_resolved = inr.resolve()
        if out == inr_resolved or _within(inr_resolved, out):
            return "output_over_input_root"
    return None


def _receipt(
    mode, status, reason_code, child_returncode, elapsed_seconds, peak_rss_bytes, output_bytes
) -> dict[str, Any]:
    return {
        "status": status,
        "mode": mode,
        "reason_code": reason_code,
        "child_returncode": child_returncode,
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "output_bytes": output_bytes,
        "limits": {
            "wall_seconds": MODES.get(mode),
            "rss_bytes": RSS_LIMIT,
            "output_bytes": OUTPUT_LIMIT,
        },
    }


def _open_private(path: Path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(fd, "wb", buffering=0)


def _write_receipt(root: Path, receipt: dict[str, Any]) -> None:
    data = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with _open_private(root / "guard_receipt.json") as handle:
        handle.write(data)


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def _child_command(mode: str, child_output: Path, artifact_root: Path | None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        CHILD_MODULE,
        "--mode",
        mode,
        "--output",
        str(child_output),
    ]
    if artifact_root is not None:
        cmd += ["--artifact-root", str(artifact_root)]
    return cmd


def _read_child_peak(root: Path, mode: str) -> int:
    path = root / "analysis" / "receipt.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GuardError("child_receipt_invalid") from exc
    if not isinstance(data, dict):
        raise GuardError("child_receipt_invalid")
    if data.get("status") != "success":
        raise GuardError("child_receipt_invalid")
    if data.get("mode") != mode:
        raise GuardError("child_receipt_invalid")
    expected_draws = 100 if mode == "probe" else 10000
    if data.get("draws") != expected_draws:
        raise GuardError("child_receipt_invalid")
    peak = data.get("peak_rss_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
        raise GuardError("child_receipt_invalid")
    return peak


def _safe_rc(proc: Any) -> int | None:
    if proc is None:
        return None
    try:
        return proc.poll()
    except Exception:
        return None


def _terminate(proc: Any) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        proc.wait(timeout=KILL_GRACE)
        return
    except Exception:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait(timeout=KILL_GRACE)
    except Exception:
        pass


def _metrics(
    returncode: int | None,
    elapsed: float,
    peak_rss: int,
    output_bytes: int,
) -> dict[str, Any]:
    return {
        "child_returncode": returncode,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "output_bytes": output_bytes,
    }


def _monitor(
    process: Any,
    root: Path,
    *,
    wall_limit: float,
    rss_limit: int = RSS_LIMIT,
    output_limit: int = OUTPUT_LIMIT,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    status_reader: Callable[[int], Mapping[str, int]] = _read_status,
    children_reader: Callable[[int], str] = _read_children,
    sizer: Callable[[Path], int] = _measure_output,
    poll_interval: float = POLL_INTERVAL,
    size_interval: float = SIZE_INTERVAL,
) -> dict[str, Any]:
    start = clock()
    last_size_at = start
    peak_rss = 0
    output_bytes = 0
    while True:
        returncode = process.poll()
        now = clock()
        elapsed = now - start
        if returncode is not None:
            break
        snapshot = _metrics(None, elapsed, peak_rss, output_bytes)
        if elapsed > wall_limit:
            raise GuardError("wall_exceeded", snapshot)
        if children_reader(process.pid):
            raise GuardError("unexpected_descendant", snapshot)
        try:
            status = status_reader(process.pid)
        except GuardError as exc:
            if process.poll() is not None:
                returncode = process.poll()
                break
            raise GuardError("monitor_failed", snapshot) from exc
        if not status or ("VmRSS" not in status and "VmHWM" not in status):
            if process.poll() is not None:
                returncode = process.poll()
                break
            raise GuardError("monitor_failed", snapshot)
        peak_rss = max(peak_rss, int(status.get("VmRSS", 0)), int(status.get("VmHWM", 0)))
        if peak_rss > rss_limit:
            raise GuardError("rss_exceeded", _metrics(None, elapsed, peak_rss, output_bytes))
        if now - last_size_at >= size_interval:
            output_bytes = sizer(root)
            if output_bytes + RECEIPT_RESERVE > output_limit:
                raise GuardError("output_exceeded", _metrics(None, elapsed, peak_rss, output_bytes))
            last_size_at = now
        remaining = wall_limit - elapsed
        if remaining <= 0:
            raise GuardError("wall_exceeded", _metrics(None, elapsed, peak_rss, output_bytes))
        sleep(min(poll_interval, remaining))
    elapsed = clock() - start
    output_bytes = sizer(root)
    snapshot = _metrics(returncode, elapsed, peak_rss, output_bytes)
    if output_bytes + RECEIPT_RESERVE > output_limit:
        raise GuardError("output_exceeded", snapshot)
    if elapsed > wall_limit:
        raise GuardError("wall_exceeded", snapshot)
    return snapshot


def _metric_value(metrics: Mapping[str, Any] | None, key: str) -> Any:
    if metrics is None:
        return None
    return metrics.get(key)


def run_guard(*, mode: str, output, artifact_root=None) -> dict[str, Any]:
    start = time.monotonic()
    out = Path(output)
    inroot = Path(artifact_root) if artifact_root is not None else None
    reason_code = _preflight(mode, out, inroot)
    if reason_code is not None:
        return _receipt(mode, "failed", reason_code, None, None, None, None)

    os.makedirs(out, mode=0o700)
    try:
        os.chmod(out, 0o700)
    except OSError:
        pass
    child_out = out / "analysis"

    proc: Any = None
    metrics: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    try:
        with contextlib.ExitStack() as stack:
            stdout_f = stack.enter_context(_open_private(out / "child.stdout.log"))
            stderr_f = stack.enter_context(_open_private(out / "child.stderr.log"))
            proc = subprocess.Popen(
                _child_command(mode, child_out, inroot),
                stdout=stdout_f,
                stderr=stderr_f,
                env=_child_env(),
                start_new_session=True,
                close_fds=True,
            )
            remaining = MODES[mode] - (time.monotonic() - start)
            metrics = _monitor(proc, out, wall_limit=remaining)
    except GuardError as exc:
        metrics = exc.metrics
        receipt = _receipt(
            mode,
            "failed",
            exc.reason_code,
            _safe_rc(proc),
            _metric_value(metrics, "elapsed_seconds"),
            _metric_value(metrics, "peak_rss_bytes"),
            _metric_value(metrics, "output_bytes"),
        )
    except KeyboardInterrupt:
        receipt = _receipt(mode, "failed", "interrupted", None, None, None, None)
    except Exception:
        receipt = _receipt(mode, "failed", "monitor_failed", None, None, None, None)
    else:
        assert metrics is not None
        child_rc = int(metrics["child_returncode"])
        receipt = _receipt(
            mode,
            "passed" if child_rc == 0 else "failed",
            None if child_rc == 0 else "child_failed",
            child_rc,
            float(metrics["elapsed_seconds"]),
            int(metrics["peak_rss_bytes"]),
            int(metrics["output_bytes"]),
        )
        if receipt["status"] == "passed":
            try:
                child_peak = _read_child_peak(out, mode)
            except GuardError as exc:
                receipt["status"] = "failed"
                receipt["reason_code"] = exc.reason_code
            else:
                if child_peak > receipt["peak_rss_bytes"]:
                    receipt["peak_rss_bytes"] = child_peak
                if receipt["peak_rss_bytes"] > RSS_LIMIT:
                    receipt["status"] = "failed"
                    receipt["reason_code"] = "rss_exceeded"
    finally:
        if proc is not None and (
            proc.poll() is None or receipt is None or receipt.get("status") != "passed"
        ):
            _terminate(proc)

    if receipt is None:
        receipt = _receipt(mode, "failed", "monitor_failed", None, None, None, None)

    receipt["elapsed_seconds"] = time.monotonic() - start
    if proc is not None:
        returncode = _safe_rc(proc)
        if returncode is not None:
            receipt["child_returncode"] = returncode

    if receipt["output_bytes"] is None:
        with contextlib.suppress(GuardError, OSError):
            receipt["output_bytes"] = _measure_output(out)

    if receipt["status"] == "passed":
        try:
            final_bytes = _measure_output(out)
        except GuardError as exc:
            receipt["status"] = "failed"
            receipt["reason_code"] = exc.reason_code
        else:
            if final_bytes + RECEIPT_RESERVE > OUTPUT_LIMIT:
                receipt["status"] = "failed"
                receipt["reason_code"] = "output_exceeded"
            receipt["output_bytes"] = final_bytes

    if proc is not None and _safe_rc(proc) is None:
        receipt["status"] = "failed"
        receipt["reason_code"] = "termination_unconfirmed"

    receipt["elapsed_seconds"] = time.monotonic() - start
    if receipt["status"] == "passed" and receipt["elapsed_seconds"] > MODES[mode]:
        receipt["status"] = "failed"
        receipt["reason_code"] = "wall_exceeded"

    _write_receipt(out, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="p06p11_guard",
        description="Parent guard for the p06p11 inference child runner.",
    )
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-root", default=None)
    args = parser.parse_args(argv)
    try:
        receipt = run_guard(mode=args.mode, output=args.output, artifact_root=args.artifact_root)
    except KeyboardInterrupt:
        print("failed interrupted")
        return 1
    status = receipt.get("status")
    reason = receipt.get("reason_code")
    if reason is None:
        print(str(status))
    else:
        print(f"{status} {reason}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
