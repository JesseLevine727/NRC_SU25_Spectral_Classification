"""Thin private composition and navigation driver for the fixed P08 figures.

No statistics, no fitting, no data selection, no CLI, no publication, no git,
no networking and no new renderer code.  This driver only routes already
accepted component APIs into the fixed six figure private build and emits a
private build manifest plus a public candidate relative index.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping

from atlas_sers.visualization import (
    p08_f01_render,
    p08_f02_render,
    p08_f03_render,
    p08_f04_render,
    p08_f07_data,
    p08_f07_render,
    p08_model_figure_data,
    p08_training_render,
)
from atlas_sers.visualization import p08_figure_build as _writer

FIGURE_ORDER = (
    "P08-F01",
    "P08-F02",
    "P08-F03",
    "P08-F04",
    "P08-F07",
    p08_training_render.FIGURE_ID,
)

_PRESERVATION_KEYS = frozenset({"spectral", "preservation", "support"})
_MANIFEST_NAME = "manifest.json"
_INDEX_NAME = "index.html"
_FAILURE_NAME = "failure.json"
_MANIFEST_BYTE_LIMIT = 1 << 20
_FAILURE_BYTES = b'{"status":"failed","reason":"universal figure build failed"}'

_FIGURE_DESCRIPTIONS = {
    "P08-F01": "MIN, smoothed and baseline-corrected aggregate spectra.",
    "P08-F02": "Paired held-instrument classification accuracy.",
    "P08-F03": "Accuracy changes relative to minimal preprocessing.",
    "P08-F04": "Preprocessing-by-model interactions.",
    "P08-F07": "Spectral-change diagnostics versus classification accuracy.",
    p08_training_render.FIGURE_ID: "Training diagnostic panels (native vector).",
}


class UniversalFigureError(RuntimeError):
    """Path free universal figure driver failure."""


def _guard(check):
    if not callable(check):
        raise UniversalFigureError("check guard must be callable")
    if check() is False:
        raise UniversalFigureError("check guard refused")


def prepare_universal_figures(*, analysis, preservation, training_prepared, check):
    if not callable(check):
        raise UniversalFigureError("check guard must be callable")
    if not isinstance(preservation, Mapping) or set(preservation) != _PRESERVATION_KEYS:
        raise UniversalFigureError("preservation mapping mismatch")
    _guard(check)
    models = p08_model_figure_data.prepare_model_figures(analysis)
    _guard(check)
    f01 = p08_f01_render.prepare_f01(preservation["spectral"])
    _guard(check)
    f02 = p08_f02_render.prepare_f02(models)
    _guard(check)
    f03 = p08_f03_render.prepare_f03(models)
    _guard(check)
    f04 = p08_f04_render.prepare_f04(models)
    _guard(check)
    f07_data = p08_f07_data.prepare_f07(preservation["preservation"], models)
    _guard(check)
    f07 = p08_f07_render.prepare_f07_render(f07_data)
    _guard(check)
    training = p08_training_render.prepare_training_render(training_prepared)
    _guard(check)
    return {
        "P08-F01": f01,
        "P08-F02": f02,
        "P08-F03": f03,
        "P08-F04": f04,
        "P08-F07": f07,
        p08_training_render.FIGURE_ID: training,
    }


def _bounded_bytes_sha(path, limit=_MANIFEST_BYTE_LIMIT):
    try:
        size = os.path.getsize(str(path))
    except OSError as exc:
        raise UniversalFigureError("figure manifest missing") from exc
    if size <= 0:
        raise UniversalFigureError("figure manifest empty")
    if size > limit:
        raise UniversalFigureError("figure manifest too large")
    with open(str(path), "rb") as handle:
        data = handle.read(limit + 1)
    if len(data) != size or len(data) > limit:
        raise UniversalFigureError("figure manifest truncated")
    return hashlib.sha256(data).hexdigest()


def _index_document(records):
    items = []
    for record in records:
        figure_id = record["figure_id"]
        items.append(
            '<li><a href="'
            + figure_id
            + '/built_index.html">'
            + figure_id
            + "</a> - "
            + _FIGURE_DESCRIPTIONS.get(figure_id, "Prepared figure panels.")
            + "</li>"
        )
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="robots" content="noindex,nofollow">'
        "<title>P08 universal figure set (unreviewed, not published)</title>"
        "<style>html{color:#000;background:#fff;font-family:serif;}"
        "a{color:#000;}</style></head><body>"
        "<h1>P08 universal figure set</h1>"
        "<p>Unreviewed private build candidate. Not published.</p>"
        "<ul>" + "".join(items) + "</ul></body></html>"
    )


def _root_manifest(records, refs, style_sha, index_sha):
    return {
        "status": "built_unreviewed",
        "reviewed": False,
        "published": False,
        "disclosure_reviewed": False,
        "visual_reviewed": False,
        "structural_validation": True,
        "external_authentication_verified": False,
        "figure_count": len(records),
        "total_panels": sum(record["panel_count"] for record in records),
        "source_refs": dict(refs),
        "style_config_sha256": style_sha,
        "index_sha256": index_sha,
        "figures": [
            {
                "figure_id": record["figure_id"],
                "panel_count": record["panel_count"],
                "semantic_sha256": record["semantic_sha256"],
                "manifest_sha256": record["manifest_sha256"],
            }
            for record in records
        ],
    }


def build_universal_figures(
    prepared,
    *,
    output,
    private_logs,
    source_refs,
    style_config_path,
    check,
    deadline,
):
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise UniversalFigureError("deadline must be a finite number")
    if not math.isfinite(deadline):
        raise UniversalFigureError("deadline must be finite")
    if not callable(check):
        raise UniversalFigureError("check guard must be callable")

    _writer._budget_check(check, deadline)
    if not isinstance(prepared, Mapping):
        raise UniversalFigureError("prepared must be a mapping")
    if set(prepared) != set(FIGURE_ORDER):
        raise UniversalFigureError("prepared figure ids mismatch")

    validated = {}
    for figure_id in FIGURE_ORDER:
        _writer._budget_check(check, deadline)
        if figure_id not in _writer.FIGURE_PANEL_COUNTS:
            raise UniversalFigureError("figure id unsupported by writer")
        _semantic, sha, panels, found = _writer._validate_prepared(prepared[figure_id])
        if found != figure_id:
            raise UniversalFigureError("prepared figure id mismatch")
        if len(panels) != _writer.FIGURE_PANEL_COUNTS[figure_id]:
            raise UniversalFigureError("prepared panel count mismatch")
        validated[figure_id] = {
            "panel_count": len(panels),
            "semantic_sha256": sha,
        }

    refs = _writer._validate_source_refs(source_refs)
    out, logs, style = _writer._validate_roots(output, private_logs, style_config_path)
    style_sha = _writer._read_style(style)
    _writer._budget_check(check, deadline)

    records = []
    owned = False
    old_umask = os.umask(0o077)
    try:
        _writer._make_root(out)
        owned = True
        _writer._make_root(logs)
        for figure_id in FIGURE_ORDER:
            _writer._budget_check(check, deadline)
            figure_out = out / figure_id
            figure_logs = logs / figure_id
            _writer.build_figure_bundle(
                prepared[figure_id],
                output=figure_out,
                private_logs=figure_logs,
                source_refs=refs,
                style_config_path=style,
                check=check,
                deadline=deadline,
            )
            _writer._budget_check(check, deadline)
            records.append(
                {
                    "figure_id": figure_id,
                    "panel_count": validated[figure_id]["panel_count"],
                    "semantic_sha256": validated[figure_id]["semantic_sha256"],
                    "manifest_sha256": _bounded_bytes_sha(figure_out / _MANIFEST_NAME),
                }
            )

        _writer._budget_check(check, deadline)
        if _writer._read_style(style) != style_sha:
            raise UniversalFigureError("style config changed during build")
        index_document = _index_document(records)
        _writer._check_html(index_document)
        _writer._write_exclusive(out / _INDEX_NAME, index_document)
        index_sha = hashlib.sha256(index_document.encode("utf-8")).hexdigest()
        manifest = _root_manifest(records, refs, style_sha, index_sha)
        _writer._write_exclusive(out / _MANIFEST_NAME, _writer._canonical_json(manifest))
        _writer._budget_check(check, deadline)
    except BaseException:
        if owned:
            try:
                _writer._write_exclusive(out / _FAILURE_NAME, _FAILURE_BYTES)
            except OSError:
                pass  # Keep the original failure and any partial output.
        raise
    finally:
        os.umask(old_umask)
    return manifest
