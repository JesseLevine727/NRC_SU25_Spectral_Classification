"""Bounded per-fitting-run epoch monitor for the P08-U1 controller.

This module provides :class:`EpochMonitor`, a standard-library-only observer
that the U1 controller can attach to an existing ``on_epoch(record)`` callback.
It keeps a bounded in-memory view of the epochs it accepted, mirrors them to a
durable append-only JSONL file and to replaceable offline HTML/TikZ snapshots,
and refuses record fields outside its declared allowlist.  It is observational:
it is not an execution journal, resource monitor, model selector or permit.

Guarantees
----------

* A newly created, exclusive, private directory is owned by exactly one
  monitor.  Existing directories (including symlink targets) are refused, so
  there is no resume/retry authority.
* Every accepted epoch is validated before any in-memory or on-disk state
  changes.  Invalid records are rejected without side effects; a rejected
  record is not an I/O failure and does not poison the monitor.
* Accepted epochs are durably appended to ``epochs.jsonl`` (flush + fsync).
  ``index.html`` is replaced each epoch while running.
* Only an allowlisted projection of the callback record is retained; unknown
  input fields are ignored and never copied.
* Finished/closed monitors reject further epochs.  ``close`` without
  ``finish`` is recorded as ``closed-unfinished`` and never implies success.
* The first clock reading must be finite, and each per-epoch duration must be
  finite and non-decreasing.  One frozen elapsed value is persisted at
  finish/close and reused in the final artifacts.
* Filesystem, finalization and stream write errors propagate; a failed write
  leaves the monitor non-successful and closed to further epochs.

Limits
------

* Elapsed time is a monotonic duration measured from construction; the
  injectable clock exists only for deterministic tests.
* Persistence is deliberately not an atomic multi-file transaction.
  ``semantic.json``, ``learning_curves.tex`` and ``index.html`` are written one
  after another; an error between them leaves earlier files as partial
  evidence.  The monitor does not attempt cross-file rollback.
* The monitor makes no hard filesystem isolation claim against a malicious
  concurrent process that replaces files inside the owned directory.
* At finish it writes canonical semantic JSON plus a native pgfplots/TikZ
  source and a final offline HTML.  It never runs LaTeX, performs model
  inference, uses the network or imports training libraries.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any

__all__ = ["EpochMonitor"]

EPOCHS_NAME = "epochs.jsonl"
HTML_NAME = "index.html"
SEMANTIC_NAME = "semantic.json"
TEX_NAME = "learning_curves.tex"
SEMANTIC_SCHEMA = "p08_live_monitor_semantic_v1"

MAXIMUM_EPOCHS = 200

STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"
STATUS_CLOSED = "closed-unfinished"

FINISH_STATUSES = frozenset({STATUS_COMPLETE, STATUS_FAILED, STATUS_INTERRUPTED})
STOP_REASONS = frozenset(
    {
        "patience",
        "epoch_limit",
        "fixed_duration",
        "error",
        "interrupted",
        "unspecified",
    }
)

MODEL_LABELS = {
    "D0-M": "D0-M ordinary classification",
    "D1": "D1 chemical similarity",
    "D2": "D2 matched-sample consistency",
    "D3": "D3 both",
}
POLICY_LABELS = {
    "PP-U-SG": "PP-U-SG Savitzky-Golay",
    "PP-U-ARPLS": "PP-U-ARPLS baseline",
}
STAGE_LABELS = {
    "source_fit": "source fit",
    "calibration_model_fit": "calibration model fit",
    "final_refit": "final refit",
}
STATUS_LABELS = {
    STATUS_RUNNING: "running",
    STATUS_COMPLETE: "complete",
    STATUS_FAILED: "failed",
    STATUS_INTERRUPTED: "interrupted",
    STATUS_CLOSED: "closed unfinished (no success implied)",
}

ALLOWED_SEEDS = frozenset({20260805, 20260817, 20260829})

_VALIDATION_FIELDS = (
    "train_nll",
    "validation_nll",
    "train_balanced_accuracy",
    "validation_balanced_accuracy",
    "best_epoch",
    "nonimproving_epochs",
)

_HTML_DASH = {
    "solid": None,
    "dashed": "7,4",
    "dotted": "2,3",
    "dashdot": "8,3,2,3",
}
_TEX_DASH = {
    "solid": "solid",
    "dashed": "dashed",
    "dotted": "dotted",
    "dashdot": "dash dot",
}
_TEX_MARK = {
    "circle": "*",
    "square": "square*",
    "triangle": "triangle*",
    "diamond": "diamond*",
}


def _is_hex64(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer.")
    return int(value)


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean.")
    return value


def _require_finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite real number.")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{field} must be a finite real number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite real number.")
    return number


def _require_unit(value: object, field: str) -> float:
    number = _require_finite(value, field)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field} must lie in [0, 1].")
    return number


def _esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _tex_escape(text: object) -> str:
    result = str(text)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("_", r"\_"),
    ):
        result = result.replace(source, target)
    return result


def _format_number(value: float) -> str:
    return format(float(value), ".17g")


def _format_tick(value: float) -> str:
    return format(float(value), ".6g")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_line(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _create_private_directory(output_dir: str) -> str:
    parent = os.path.dirname(os.path.abspath(output_dir))
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, mode=0o700, exist_ok=True)
        except OSError as exc:
            raise OSError(
                "the monitor output directory could not be created"
            ) from exc
    try:
        os.mkdir(output_dir, 0o700)
    except FileExistsError:
        raise FileExistsError(
            "an output directory already exists at the monitor target"
        ) from None
    except OSError as exc:
        raise OSError(
            "the monitor output directory could not be created"
        ) from exc
    try:
        os.chmod(output_dir, 0o700)
    except OSError as exc:
        try:
            os.rmdir(output_dir)
        except OSError:
            pass
        raise OSError(
            "the monitor output directory could not be secured"
        ) from exc
    return output_dir


def _open_append_private(path: str):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "a", encoding="utf-8", newline="\n")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _atomic_write_private(path: str, data: bytes) -> None:
    directory = os.path.dirname(path) or "."
    descriptor, temporary = tempfile.mkstemp(prefix=".p08-", dir=directory)
    open_descriptor = descriptor
    try:
        os.fchmod(open_descriptor, 0o600)
        with os.fdopen(open_descriptor, "wb") as handle:
            open_descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if open_descriptor != -1:
            try:
                os.close(open_descriptor)
            except OSError:
                pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _project_record(
    record: object,
    *,
    validation_available: bool,
    expected_epoch: int,
    epoch_budget: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("record must be a mapping.")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise ValueError(
            "clock observations must be finite and non-decreasing."
        )
    epoch = _require_int(record.get("epoch"), "epoch")
    if epoch != expected_epoch:
        raise ValueError("epoch must be the next contiguous integer epoch.")
    if epoch < 1 or epoch > epoch_budget:
        raise ValueError("epoch must lie within the declared epoch budget.")

    row: dict[str, Any] = {
        "epoch": epoch,
        "elapsed_seconds": float(elapsed_seconds),
        "chemical_ce": _require_finite(record.get("chemical_ce"), "chemical_ce"),
        "total_loss": _require_finite(record.get("total_loss"), "total_loss"),
        "supcon_enabled": _require_bool(
            record.get("supcon_enabled"), "supcon_enabled"
        ),
        "paired_enabled": _require_bool(
            record.get("paired_enabled"), "paired_enabled"
        ),
    }

    if row["supcon_enabled"]:
        row["supcon_loss"] = _require_finite(
            record.get("supcon_loss"), "supcon_loss"
        )
    else:
        row["supcon_loss"] = None
    if row["paired_enabled"]:
        row["paired_loss"] = _require_finite(
            record.get("paired_loss"), "paired_loss"
        )
    else:
        row["paired_loss"] = None

    steps = record.get("total_optimizer_steps")
    if steps is not None:
        steps = _require_int(steps, "total_optimizer_steps")
        if steps < 0:
            raise ValueError(
                "total_optimizer_steps must be a nonnegative integer."
            )
        row["total_optimizer_steps"] = steps

    if validation_available:
        row["train_nll"] = _require_finite(
            record.get("train_nll"), "train_nll"
        )
        row["validation_nll"] = _require_finite(
            record.get("validation_nll"), "validation_nll"
        )
        row["train_balanced_accuracy"] = _require_unit(
            record.get("train_balanced_accuracy"),
            "train_balanced_accuracy",
        )
        row["validation_balanced_accuracy"] = _require_unit(
            record.get("validation_balanced_accuracy"),
            "validation_balanced_accuracy",
        )
        best_epoch = _require_int(record.get("best_epoch"), "best_epoch")
        if best_epoch < 1 or best_epoch > epoch:
            raise ValueError("best_epoch must lie in [1, current epoch].")
        row["best_epoch"] = best_epoch
        nonimproving = _require_int(
            record.get("nonimproving_epochs"), "nonimproving_epochs"
        )
        if nonimproving < 0:
            raise ValueError(
                "nonimproving_epochs must be a nonnegative integer."
            )
        row["nonimproving_epochs"] = nonimproving
    else:
        for field in _VALIDATION_FIELDS:
            if record.get(field) is not None:
                raise ValueError(
                    "records without source validation must not carry "
                    "validation metrics."
                )
    return row


def _training_series(rows: list[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
    series: list[tuple[str, str, str, str]] = [
        ("Weighted training CE", "chemical_ce", "solid", "circle"),
        ("Minibatch total objective", "total_loss", "dashed", "square"),
    ]
    if any(row.get("supcon_loss") is not None for row in rows):
        series.append(
            ("Chemical similarity (SupCon)", "supcon_loss", "dotted", "triangle")
        )
    if any(row.get("paired_loss") is not None for row in rows):
        series.append(
            (
                "Matched-sample consistency (paired)",
                "paired_loss",
                "dashdot",
                "diamond",
            )
        )
    return series


def _nll_series(rows: list[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
    del rows
    return [
        ("Clean-evaluation training NLL", "train_nll", "solid", "circle"),
        ("Source-validation NLL", "validation_nll", "dashed", "square"),
    ]


def _ba_series(rows: list[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
    del rows
    return [
        (
            "Clean-evaluation training BA",
            "train_balanced_accuracy",
            "solid",
            "circle",
        ),
        (
            "Source-validation BA",
            "validation_balanced_accuracy",
            "dashed",
            "square",
        ),
    ]


def _panel_axes(
    rows: list[dict[str, Any]],
    series: list[tuple[str, str, str, str]],
    *,
    y_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Return the shared axis description used by both SVG and TeX panels."""
    count = max(len(rows), 1)
    if count <= 1:
        x_min, x_max, x_ticks = 0.5, 1.5, [1]
    else:
        x_min, x_max = 1.0, float(count)
        step = max(1, int(round(count / 5.0)))
        tick_set = {1, count}
        tick_set.update(range(step, count + 1, step))
        x_ticks = sorted(value for value in tick_set if 1 <= value <= count)

    values: list[float] = []
    for row in rows:
        for _label, key, _dash, _marker in series:
            value = row.get(key)
            if value is not None:
                values.append(float(value))
    if y_bounds is not None:
        y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    elif values:
        y_min, y_max = min(values), max(values)
        if y_max <= y_min:
            y_max = y_min + 1.0
        pad = 0.06 * (y_max - y_min)
        y_min -= pad
        y_max += pad
    else:
        y_min, y_max = 0.0, 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_ticks = [
        y_min + (index / 4.0) * (y_max - y_min) for index in range(5)
    ]
    return {
        "x_min": x_min,
        "x_max": x_max,
        "x_ticks": x_ticks,
        "y_min": y_min,
        "y_max": y_max,
        "y_ticks": y_ticks,
    }


def _marker_html(marker: str, x: float, y: float, title: str) -> str:
    hover = f"<title>{_esc(title)}</title>"
    if marker == "circle":
        return (
            f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.0' "
            f"fill='black'>{hover}</circle>"
        )
    if marker == "square":
        return (
            f"<rect x='{x - 2.6:.2f}' y='{y - 2.6:.2f}' width='5.2' "
            f"height='5.2' fill='black'>{hover}</rect>"
        )
    if marker == "triangle":
        return (
            f"<polygon points='{x:.2f},{y - 3.2:.2f} "
            f"{x - 3.0:.2f},{y + 2.4:.2f} {x + 3.0:.2f},{y + 2.4:.2f}' "
            f"fill='black'>{hover}</polygon>"
        )
    return (
        f"<polygon points='{x:.2f},{y - 3.2:.2f} {x + 3.2:.2f},{y:.2f} "
        f"{x:.2f},{y + 3.2:.2f} {x - 3.2:.2f},{y:.2f}' "
        f"fill='black'>{hover}</polygon>"
    )


def _wrap_label(label: str, limit: int = 34) -> list[str]:
    words = str(label).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if current and len(candidate) > limit:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current or not lines:
        lines.append(current)
    return lines


def _svg_panel(
    rows: list[dict[str, Any]],
    series: list[tuple[str, str, str, str]],
    *,
    y_label: str,
    axes: dict[str, Any],
    width: int = 860,
    height: int = 330,
) -> str:
    legend_width = 320
    left, top, bottom = 66, 30, 48
    right = legend_width
    plot_w = max(10.0, float(width - left - right))
    plot_h = max(10.0, float(height - top - bottom))
    x_min = float(axes["x_min"])
    x_max = float(axes["x_max"])
    y_min = float(axes["y_min"])
    y_max = float(axes["y_max"])
    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0

    def sx(epoch: int) -> float:
        return left + (float(epoch) - x_min) / x_span * plot_w

    def sy(value: float) -> float:
        return top + (y_max - float(value)) / y_span * plot_h

    parts: list[str] = [
        f"<svg viewBox='0 0 {width} {height}' width='{width}' "
        f"height='{height}' role='img'>"
    ]
    parts.append(
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white'/>"
    )
    parts.append(
        f"<line x1='{left}' y1='{top}' x2='{left}' "
        f"y2='{top + plot_h:.2f}' stroke='black' stroke-width='1'/>"
    )
    parts.append(
        f"<line x1='{left}' y1='{top + plot_h:.2f}' "
        f"x2='{left + plot_w:.2f}' y2='{top + plot_h:.2f}' "
        "stroke='black' stroke-width='1'/>"
    )
    for value in axes["y_ticks"]:
        y = sy(value)
        parts.append(
            f"<line x1='{left - 4}' y1='{y:.2f}' x2='{left}' "
            f"y2='{y:.2f}' stroke='black' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{left - 8}' y='{y + 4:.2f}' font-family='serif' "
            f"font-size='11' fill='black' text-anchor='end'>"
            f"{_format_tick(value)}</text>"
        )
    for epoch in axes["x_ticks"]:
        x = sx(epoch)
        parts.append(
            f"<line x1='{x:.2f}' y1='{top + plot_h:.2f}' "
            f"x2='{x:.2f}' y2='{top + plot_h + 4:.2f}' "
            "stroke='black' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{x:.2f}' y='{top + plot_h + 18:.2f}' "
            f"font-family='serif' font-size='11' fill='black' "
            f"text-anchor='middle'>{epoch}</text>"
        )
    parts.append(
        f"<text x='{left + plot_w / 2.0:.2f}' y='{height - 8}' "
        "font-family='serif' font-size='12' fill='black' "
        "text-anchor='middle'>epoch</text>"
    )
    parts.append(
        f"<text x='14' y='{top + plot_h / 2.0:.2f}' font-family='serif' "
        f"font-size='12' fill='black' text-anchor='middle' "
        f"transform='rotate(-90 14 {top + plot_h / 2.0:.2f})'>"
        f"{_esc(y_label)}</text>"
    )

    legend_y = top + 12
    for label, key, dash, marker in series:
        dash_value = _HTML_DASH[dash]
        dash_attr = f" stroke-dasharray='{dash_value}'" if dash_value else ""
        plotted = [
            (row["epoch"], float(row[key]))
            for row in rows
            if row.get(key) is not None
        ]
        if plotted:
            point_string = " ".join(
                f"{sx(e):.2f},{sy(v):.2f}" for e, v in plotted
            )
            parts.append(
                f"<polyline points='{point_string}' fill='none' "
                f"stroke='black' stroke-width='1.3'{dash_attr}/>"
            )
            for epoch, value in plotted:
                title = f"epoch {epoch}, {_format_number(value)}"
                parts.append(
                    _marker_html(marker, sx(epoch), sy(value), title)
                )
        legend_x = left + plot_w + 12
        legend_lines = _wrap_label(label)
        parts.append(
            f"<line x1='{legend_x}' y1='{legend_y}' "
            f"x2='{legend_x + 24}' y2='{legend_y}' stroke='black' "
            f"stroke-width='1.3'{dash_attr}/>"
        )
        for offset, text_line in enumerate(legend_lines):
            parts.append(
                f"<text x='{legend_x + 30}' "
                f"y='{legend_y + 4 + offset * 13}' "
                f"font-family='serif' font-size='11' fill='black'>"
                f"{_esc(text_line)}</text>"
            )
        legend_y += 18 * len(legend_lines)
    parts.append("</svg>")
    return "\n".join(parts)


_NOTES = (
    "Panel 1 shows the minibatch training total objective together with the "
    "active chemical/auxiliary components. The weighted training cross-entropy "
    "and the total objective are minibatch training quantities computed under "
    "augmentation and class/master/view weighting; they are not the "
    "clean-evaluation negative log-likelihood and are not interchangeable. The "
    "total objective depends on the recipe through its active auxiliary terms. "
    "Panels 2 and 3 contrast clean-evaluation training metrics with source-only "
    "validation metrics. Source validation is not held-test performance. The "
    "best checkpoint shown was chosen by the inherited score rule; the monitor "
    "does not choose it."
)
_REFIT_NOTE = "No validation during fixed-duration refit."


def _render_html(state: dict[str, Any], digest: str | None = None) -> str:
    rows = state["rows"]
    status = state["status"]
    running = status == STATUS_RUNNING
    budget = int(state["epoch_budget"])
    current = len(rows)
    best_epoch = None
    nonimproving = None
    if rows and state["validation_available"]:
        best_epoch = rows[-1].get("best_epoch")
        nonimproving = rows[-1].get("nonimproving_epochs")

    pieces: list[str] = []
    pieces.append("<!DOCTYPE html>")
    pieces.append("<html lang='en'><head><meta charset='utf-8'>")
    pieces.append("<title>P08 live epoch monitor</title>")
    if running:
        pieces.append(
            f"<meta http-equiv='refresh' "
            f"content='{state['refresh_seconds']:g}'>"
        )
    if digest is not None:
        pieces.append(
            f"<meta name='p08-semantic-sha256' content='{digest}'>"
        )
    pieces.append("<style>")
    pieces.append(
        "body{background:#fff;color:#000;font-family:serif;margin:24px;}"
    )
    pieces.append(
        "h1{font-size:20px;margin:0 0 8px 0;}"
        "h2{font-size:15px;margin:18px 0 6px 0;}"
    )
    pieces.append(
        "table{border-collapse:collapse;margin:6px 0;}"
        "th,td{text-align:left;padding:2px 12px 2px 0;font-size:13px;}"
    )
    pieces.append(".note{font-size:13px;max-width:960px;line-height:1.45;}")
    pieces.append(".panel{margin:6px 0 18px 0;}")
    pieces.append("</style></head><body>")
    pieces.append("<h1>P08 live epoch monitor</h1>")
    pieces.append(
        f"<p class='note'><strong>Status:</strong> "
        f"{_esc(STATUS_LABELS[status])}</p>"
    )
    pieces.append("<table>")
    pieces.append(
        f"<tr><th>Job token</th><td>{_esc(state['job_id'][:12])}</td></tr>"
    )
    pieces.append(
        f"<tr><th>Recipe</th>"
        f"<td>{_esc(MODEL_LABELS[state['model_id']])}</td></tr>"
    )
    pieces.append(
        f"<tr><th>Preprocessing</th>"
        f"<td>{_esc(POLICY_LABELS[state['policy_id']])}</td></tr>"
    )
    pieces.append(f"<tr><th>Seed</th><td>{state['seed']}</td></tr>")
    pieces.append(
        f"<tr><th>Stage</th>"
        f"<td>{_esc(STAGE_LABELS[state['stage']])}</td></tr>"
    )
    pieces.append(
        "<tr><th>Source validation</th><td>"
        + (
            "available (source-only, not held test)"
            if state["validation_available"]
            else "not used"
        )
        + "</td></tr>"
    )
    pieces.append(f"<tr><th>Epoch</th><td>{current} / {budget}</td></tr>")
    pieces.append(
        f"<tr><th>Elapsed seconds</th>"
        f"<td>{state['elapsed_seconds']:.3f}</td></tr>"
    )
    if best_epoch is not None:
        pieces.append(
            f"<tr><th>Best epoch (inherited rule)</th>"
            f"<td>{best_epoch}</td></tr>"
        )
        pieces.append(
            f"<tr><th>Nonimproving epochs</th><td>{nonimproving}</td></tr>"
        )
    if not running:
        pieces.append(
            f"<tr><th>Stop reason</th>"
            f"<td>{_esc(state['stop_reason'])}</td></tr>"
        )
    if running:
        pieces.append(
            "<tr><th>Live epoch data</th>"
            "<td><a href='epochs.jsonl'>epochs.jsonl</a></td></tr>"
        )
    else:
        pieces.append(
            "<tr><th>Final semantic data</th>"
            "<td><a href='semantic.json'>semantic.json</a></td></tr>"
        )
    pieces.append("</table>")
    if digest is not None:
        pieces.append(
            f"<p class='note'>semantic sha256: <code>{digest}</code></p>"
        )
    pieces.append(
        "<p class='note'>The job token is an opaque run key derived from the "
        "recipe, preprocessing and seed; it is not a physical-sample identity."
        "</p>"
    )
    if state["validation_available"]:
        pieces.append(f"<p class='note'>{_esc(_NOTES)}</p>")

    if not rows:
        pieces.append(
            "<p class='note'><strong>No epochs were recorded.</strong></p>"
        )
    else:
        training = _training_series(rows)
        pieces.append(
            "<h2>1. Minibatch training objective and active components</h2>"
        )
        pieces.append(
            "<div class='panel'>"
            + _svg_panel(
                rows,
                training,
                y_label="loss / objective",
                axes=_panel_axes(rows, training),
            )
            + "</div>"
        )
        if state["validation_available"]:
            pieces.append(
                "<h2>2. Clean-evaluation training NLL versus "
                "source-validation NLL</h2>"
            )
            nll = _nll_series(rows)
            pieces.append(
                "<div class='panel'>"
                + _svg_panel(
                    rows,
                    nll,
                    y_label="NLL",
                    axes=_panel_axes(rows, nll),
                )
                + "</div>"
            )
            pieces.append(
                "<h2>3. Clean-evaluation training BA versus "
                "source-validation BA</h2>"
            )
            ba = _ba_series(rows)
            pieces.append(
                "<div class='panel'>"
                + _svg_panel(
                    rows,
                    ba,
                    y_label="balanced accuracy",
                    axes=_panel_axes(rows, ba, y_bounds=(0.0, 1.0)),
                )
                + "</div>"
            )
        else:
            pieces.append(
                f"<p class='note'><strong>{_esc(_REFIT_NOTE)}</strong></p>"
            )
    pieces.append("</body></html>")
    return "\n".join(pieces)


def _tex_title_node(state: dict[str, Any]) -> str:
    token = str(state["job_id"])[:12]
    metadata = (
        "recipe: {recipe} | policy: {policy} | seed: {seed} | "
        "job token: {token} | stage: {stage} | status: {status} | "
        "epochs: {epochs}/{budget} | elapsed: {elapsed:.3f} s"
    ).format(
        recipe=MODEL_LABELS[state["model_id"]],
        policy=POLICY_LABELS[state["policy_id"]],
        seed=state["seed"],
        token=token,
        stage=STAGE_LABELS[state["stage"]],
        status=STATUS_LABELS[state["status"]],
        epochs=len(state["rows"]),
        budget=int(state["epoch_budget"]),
        elapsed=float(state["elapsed_seconds"]),
    )
    return (
        r"\node[anchor=south, font=\bfseries\small, color=black, "
        r"align=center, text width=16cm] at (current bounding box.north) {"
        r"P08 live epoch monitor learning curves\\[2pt]"
        r"\normalfont\small "
        + _tex_escape(metadata)
        + r"};"
    )


def _tex_caption_node(validation_available: bool) -> str:
    if validation_available:
        body = (
            "Panel 1: minibatch training objective and active auxiliary "
            "components. Panels 2-3: clean-evaluation training versus "
            "source-only validation NLL and balanced accuracy. Source "
            "validation is not held-test performance. The best checkpoints "
            "follow the inherited score rule."
        )
    else:
        body = (
            "Panel 1: minibatch training objective and active auxiliary "
            "components. No validation during the fixed-duration refit, so no "
            "validation panels are shown."
        )
    return (
        r"\node[anchor=north, font=\small, color=black, align=left, "
        r"text width=14cm] "
        r"(p08caption) at ([yshift=-2mm]current bounding box.south) {"
        + _tex_escape(body)
        + r"};"
    )


def _tex_axis_options(axes: dict[str, Any]) -> str:
    x_ticks = ",".join(str(int(value)) for value in axes["x_ticks"])
    y_ticks = ",".join(_format_number(value) for value in axes["y_ticks"])
    y_labels = ",".join(_format_tick(value) for value in axes["y_ticks"])
    return (
        f"xmin={_format_number(axes['x_min'])}, "
        f"xmax={_format_number(axes['x_max'])}, "
        f"ymin={_format_number(axes['y_min'])}, "
        f"ymax={_format_number(axes['y_max'])}, "
        f"xtick={{{x_ticks}}}, "
        f"ytick={{{y_ticks}}}, "
        f"yticklabels={{{y_labels}}}"
    )


def _render_tex(state: dict[str, Any], digest: str) -> str:
    rows = state["rows"]
    validation_available = bool(state["validation_available"])
    scope = (
        "source-only validation; not held-test"
        if validation_available
        else "no validation"
    )
    lines: list[str] = [
        "% P08 live epoch monitor learning curves",
        f"% semantic_sha256={digest}",
        f"% job={state['job_id'][:12]}",
        f"% recipe={state['model_id']}",
        f"% policy={state['policy_id']}",
        f"% seed={state['seed']}",
        f"% stage={state['stage']}",
        f"% status={state['status']}",
        f"% epoch={len(rows)}/{int(state['epoch_budget'])}",
        f"% scope={scope}",
        r"\documentclass[tikz,border=8pt]{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\usepgfplotslibrary{groupplots}",
        r"\ifdefined\pdfinfoomitdate\pdfinfoomitdate=1\fi",
        r"\ifdefined\pdfsuppressptexinfo\pdfsuppressptexinfo=-1\fi",
        r"\ifdefined\pdftrailerid\pdftrailerid{}\fi",
        r"\begin{document}",
        r"\begin{tikzpicture}",
    ]
    if not rows:
        lines.append(
            r"\node[font=\bfseries\large, align=center] "
            r"{No epochs were recorded.};"
        )
        lines.extend([r"\end{tikzpicture}", r"\end{document}", ""])
        return "\n".join(lines)

    training = _training_series(rows)
    panels: list[
        tuple[str, str, list[tuple[str, str, str, str]], dict[str, Any]]
    ] = [
        (
            "Training objective",
            "loss",
            training,
            _panel_axes(rows, training),
        )
    ]
    if validation_available:
        nll = _nll_series(rows)
        panels.append(
            (
                "Negative log-likelihood",
                "NLL",
                nll,
                _panel_axes(rows, nll),
            )
        )
        ba = _ba_series(rows)
        panels.append(
            (
                "Balanced accuracy",
                "balanced accuracy",
                ba,
                _panel_axes(rows, ba, y_bounds=(0.0, 1.0)),
            )
        )

    count = len(panels)
    lines.append(r"\begin{groupplot}[")
    lines.append(
        rf"  group style={{group size=1 by {count}, "
        r"xlabels at=edge bottom, vertical sep=1.5cm},"
    )
    lines.append(r"  width=13cm, height=4.2cm,")
    lines.append(r"  every axis plot/.append style={line width=0.9pt},")
    lines.append(r"  tick label style={font=\small, color=black},")
    lines.append(r"  label style={font=\small, color=black},")
    lines.append(r"  title style={font=\small\bfseries, color=black},")
    lines.append(
        r"  legend style={font=\small, color=black, draw=black, fill=white},"
    )
    lines.append(r"]")
    for title, y_label, series, axes in panels:
        options = _tex_axis_options(axes)
        lines.append(
            rf"\nextgroupplot[title={{{_tex_escape(title)}}}, "
            rf"xlabel={{epoch}}, ylabel={{{_tex_escape(y_label)}}}, "
            rf"legend pos=outer north east, {options}]"
        )
        for label, key, dash, marker in series:
            coordinates = " ".join(
                f"({row['epoch']},{_format_number(row[key])})"
                for row in rows
                if row.get(key) is not None
            )
            if not coordinates:
                continue
            lines.append(
                rf"\addplot[black, {_TEX_DASH[dash]}, "
                rf"mark={_TEX_MARK[marker]}, mark size=1.1pt] "
                rf"coordinates {{{coordinates}}};"
            )
            lines.append(rf"\addlegendentry{{{_tex_escape(label)}}}")
    lines.append(r"\end{groupplot}")
    lines.append(_tex_title_node(state))
    lines.append(_tex_caption_node(validation_available))
    lines.extend([r"\end{tikzpicture}", r"\end{document}", ""])
    return "\n".join(lines)


class EpochMonitor:
    """Observational per-fit monitor with bounded, durable, private output.

    The monitor is callable with the record dictionaries produced by the frozen
    P05 development and refit kernels.  It validates an allowlisted projection,
    appends it durably to ``epochs.jsonl``, refreshes an offline ``index.html``
    snapshot and writes a short progress line to ``stream``.  It never imports
    training libraries, performs inference, selects models, runs LaTeX or
    touches the network.
    """

    def __init__(
        self,
        output_dir: Any,
        *,
        job_id: str,
        model_id: str,
        policy_id: str,
        seed: int,
        stage: str,
        validation_available: bool,
        epoch_budget: int = 200,
        stream: Any = None,
        refresh_seconds: float = 5,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(job_id, str) or not _is_hex64(job_id):
            raise ValueError(
                "job_id must be a 64-character lower-case hexadecimal string."
            )
        if not isinstance(model_id, str) or model_id not in MODEL_LABELS:
            raise ValueError("model_id is not an allowed model.")
        if not isinstance(policy_id, str) or policy_id not in POLICY_LABELS:
            raise ValueError(
                "policy_id is not an allowed preprocessing policy."
            )
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in ALLOWED_SEEDS
        ):
            raise ValueError("seed is not an allowed initialization seed.")
        if not isinstance(stage, str) or stage not in STAGE_LABELS:
            raise ValueError("stage is not an allowed stage.")
        if not isinstance(validation_available, bool):
            raise ValueError("validation_available must be a boolean.")
        if stage == "final_refit" and validation_available:
            raise ValueError("final_refit must not declare source validation.")
        if isinstance(epoch_budget, bool) or not isinstance(epoch_budget, int):
            raise ValueError("epoch_budget must be an integer.")
        if epoch_budget < 1 or epoch_budget > MAXIMUM_EPOCHS:
            raise ValueError("epoch_budget must lie in [1, 200].")
        if isinstance(refresh_seconds, bool) or not isinstance(
            refresh_seconds, (int, float)
        ):
            raise ValueError("refresh_seconds must be a positive real number.")
        refresh_seconds = float(refresh_seconds)
        if not math.isfinite(refresh_seconds) or refresh_seconds <= 0.0:
            raise ValueError("refresh_seconds must be a positive real number.")
        if stream is not None and not hasattr(stream, "write"):
            raise TypeError("stream must provide a write method.")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable.")
        try:
            raw_output = os.fspath(output_dir)
        except TypeError:
            raise TypeError("output_dir must be a filesystem path.") from None
        if isinstance(raw_output, bytes):
            raise TypeError("output_dir must be a text filesystem path.")
        if not raw_output:
            raise ValueError("output_dir must not be empty.")

        self._job_id = job_id
        self._model_id = model_id
        self._policy_id = policy_id
        self._seed = seed
        self._stage = stage
        self._validation_available = validation_available
        self._epoch_budget = epoch_budget
        self._refresh_seconds = refresh_seconds
        self._stream = stream if stream is not None else sys.stdout
        self._clock = clock if clock is not None else time.monotonic
        start = float(self._clock())
        if not math.isfinite(start):
            raise ValueError("clock must return a finite initial value.")
        self._start = start
        self._final_elapsed: float | None = None
        self._last_elapsed = 0.0

        self._history: list[dict[str, Any]] = []
        self._status = STATUS_RUNNING
        self._stop_reason: str | None = None
        self._terminal = False
        self._semantic_digest: str | None = None

        self._directory = _create_private_directory(os.fspath(raw_output))
        self._jsonl_path = os.path.join(self._directory, EPOCHS_NAME)
        self._html_path = os.path.join(self._directory, HTML_NAME)
        self._semantic_path = os.path.join(self._directory, SEMANTIC_NAME)
        self._tex_path = os.path.join(self._directory, TEX_NAME)
        self._handle = _open_append_private(self._jsonl_path)
        try:
            self._render_live()
        except BaseException:
            try:
                self._handle.close()
            except BaseException:
                pass
            self._handle = None  # type: ignore[assignment]
            raise

    # -- public properties -------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def epochs_completed(self) -> int:
        return len(self._history)

    @property
    def elapsed_seconds(self) -> float:
        if self._final_elapsed is not None:
            return self._final_elapsed
        return self._last_elapsed

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> EpochMonitor:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            try:
                self.close()
            except BaseException:
                pass
            return None
        self.close()
        return None

    def __call__(self, record: Any) -> None:
        """Validate and accept one epoch record, or raise without side effects."""
        if self._terminal:
            raise RuntimeError("monitor is no longer accepting epochs.")
        if not isinstance(record, Mapping):
            raise ValueError("record must be a mapping.")
        expected = len(self._history) + 1
        if expected > self._epoch_budget:
            raise ValueError("epoch budget is exhausted.")
        elapsed = float(self._clock()) - self._start
        if not math.isfinite(elapsed) or elapsed < self._last_elapsed:
            raise ValueError(
                "clock observations must be finite and non-decreasing."
            )
        row = _project_record(
            record,
            validation_available=self._validation_available,
            expected_epoch=expected,
            epoch_budget=self._epoch_budget,
            elapsed_seconds=elapsed,
        )
        line = _canonical_line(row)
        try:
            self._handle.write(line + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except BaseException:
            self._mark_failed()
            raise
        self._history.append(row)
        self._last_elapsed = float(row["elapsed_seconds"])
        try:
            self._render_live()
            self._emit_progress(row)
        except BaseException:
            self._mark_failed()
            raise

    def finish(self, status: str, *, stop_reason: str = "unspecified") -> None:
        """Finalize the monitor and write semantic/TikZ/HTML artifacts."""
        if self._terminal:
            raise RuntimeError("monitor has already been finished or closed.")
        if not isinstance(status, str) or status not in FINISH_STATUSES:
            raise ValueError(
                "status must be one of complete, failed or interrupted."
            )
        if not isinstance(stop_reason, str) or stop_reason not in STOP_REASONS:
            raise ValueError("stop_reason is not an allowed value.")
        if status == STATUS_COMPLETE and not self._history:
            raise ValueError(
                "cannot finish as complete without an accepted epoch."
            )
        self._terminal = True
        self._status = status
        self._stop_reason = stop_reason
        try:
            self._freeze_elapsed()
            self._finalize()
        except BaseException:
            self._status = STATUS_FAILED
            self._stop_reason = "error"
            self._shutdown_best_effort()
            raise

    def close(self) -> None:
        """Close the monitor; without a prior finish this records no success."""
        if self._terminal:
            self._shutdown()
            return
        self._terminal = True
        self._status = STATUS_CLOSED
        self._stop_reason = "unspecified"
        try:
            self._freeze_elapsed()
            self._finalize()
        except BaseException:
            self._status = STATUS_FAILED
            self._stop_reason = "error"
            self._shutdown_best_effort()
            raise

    # -- internals ---------------------------------------------------------

    def _mark_failed(self) -> None:
        self._status = STATUS_FAILED
        self._stop_reason = "error"
        self._terminal = True
        if self._final_elapsed is None:
            self._final_elapsed = self._last_elapsed
        self._shutdown_best_effort()

    def _shutdown_best_effort(self) -> None:
        try:
            self._shutdown()
        except BaseException:
            pass

    def _freeze_elapsed(self) -> None:
        if self._final_elapsed is not None:
            return
        candidate = float(self._clock()) - self._start
        if not math.isfinite(candidate) or candidate < self._last_elapsed:
            raise ValueError(
                "the final clock observation must be finite and "
                "non-decreasing."
            )
        self._final_elapsed = candidate

    def _view(self) -> dict[str, Any]:
        return {
            "job_id": self._job_id,
            "model_id": self._model_id,
            "policy_id": self._policy_id,
            "seed": self._seed,
            "stage": self._stage,
            "validation_available": self._validation_available,
            "epoch_budget": self._epoch_budget,
            "refresh_seconds": self._refresh_seconds,
            "status": self._status,
            "stop_reason": self._stop_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "rows": self._history,
        }

    def _semantic_document(self) -> dict[str, Any]:
        return {
            "schema": SEMANTIC_SCHEMA,
            "job_id": self._job_id,
            "model_id": self._model_id,
            "policy_id": self._policy_id,
            "seed": self._seed,
            "stage": self._stage,
            "validation_available": self._validation_available,
            "epoch_budget": self._epoch_budget,
            "status": self._status,
            "stop_reason": self._stop_reason,
            "epochs_completed": len(self._history),
            "elapsed_seconds": self.elapsed_seconds,
            "rows": [dict(row) for row in self._history],
        }

    def _render_live(self) -> None:
        html = _render_html(self._view())
        _atomic_write_private(self._html_path, html.encode("utf-8"))

    def _emit_progress(self, row: dict[str, Any]) -> None:
        stream = self._stream
        if stream is None:
            return
        stage = STAGE_LABELS[self._stage]
        parts = [
            f"P08 {self._job_id[:12]} {self._model_id} "
            f"{self._policy_id} seed={self._seed} {stage} "
            f"epoch {row['epoch']}/{self._epoch_budget}",
            f"total={row['total_loss']:.6g}",
            f"CE={row['chemical_ce']:.6g}",
        ]
        if row["supcon_enabled"]:
            parts.append(f"SupCon={row['supcon_loss']:.6g}")
        if row["paired_enabled"]:
            parts.append(f"paired={row['paired_loss']:.6g}")
        if self._validation_available:
            parts.append(f"trainNLL={row['train_nll']:.6g}")
            parts.append(f"sourceNLL={row['validation_nll']:.6g}")
            parts.append(f"trainBA={row['train_balanced_accuracy']:.4f}")
            parts.append(
                f"sourceBA={row['validation_balanced_accuracy']:.4f}"
            )
            parts.append(f"best={row['best_epoch']}")
            parts.append(f"nonimproving={row['nonimproving_epochs']}")
        parts.append(f"elapsed={row['elapsed_seconds']:.3f}s")
        stream.write(" | ".join(parts) + "\n")
        flush = getattr(stream, "flush", None)
        if flush is not None:
            flush()

    def _finalize(self) -> None:
        error: BaseException | None = None
        try:
            semantic = self._semantic_document()
            data = _canonical_bytes(semantic)
            digest = hashlib.sha256(data).hexdigest()
            self._semantic_digest = digest
            _atomic_write_private(self._semantic_path, data)
            tex = _render_tex(self._view(), digest)
            _atomic_write_private(self._tex_path, tex.encode("utf-8"))
            html = _render_html(self._view(), digest=digest)
            _atomic_write_private(self._html_path, html.encode("utf-8"))
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                self._shutdown()
            except BaseException:
                if error is None:
                    raise

    def _shutdown(self) -> None:
        handle = self._handle
        self._handle = None  # type: ignore[assignment]
        if handle is None:
            return
        failure: BaseException | None = None
        try:
            handle.flush()
        except BaseException as exc:
            failure = exc
        try:
            handle.close()
        except BaseException:
            if failure is None:
                raise
        if failure is not None:
            raise failure
