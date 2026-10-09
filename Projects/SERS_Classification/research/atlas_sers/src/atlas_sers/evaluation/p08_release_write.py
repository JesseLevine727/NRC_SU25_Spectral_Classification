"""Thin private release-table writer for the fixed P08 nine-table set (T362).

Reuses accepted p08_figure_build boundary helpers; performs no new statistics,
no fitting, no model or policy selection, no publication, no subprocess and no
HTML work.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import pandas as pd

from atlas_sers.evaluation import p08_release_contrasts, p08_release_metrics
from atlas_sers.visualization.p08_figure_build import (
    FigureBuildError,
    _budget_check,
    _in_git_worktree,
    _make_root,
    _normalize_new,
    _sha256_file,
    _validate_source_refs,
    _write_exclusive,
)

SCHEMA = "nato-sers-p08-release-tables-v1"
METRIC_TABLE_NAMES = (
    "model_summary",
    "domain_metrics",
    "confusion",
    "reliability_bins",
    "equal_context_reliability",
    "class_sensitivity",
)
CONTRAST_TABLE_NAMES = ("contrast_summary", "paired_domains", "procedure_domains")
TABLE_NAMES = METRIC_TABLE_NAMES + CONTRAST_TABLE_NAMES
DIAGNOSTICS_FILE = "diagnostics.json"
MANIFEST_FILE = "manifest.json"
FAILURE_FILE = "failure.json"

__all__ = ["SCHEMA", "TABLE_NAMES", "write_release_tables"]


def _canonical_json_bytes(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _prepare(analysis, check, deadline):
    _budget_check(check, deadline)
    metric_tables = p08_release_metrics.prepare_metric_tables(analysis)
    _budget_check(check, deadline)
    contrast = p08_release_contrasts.prepare_contrast_tables(analysis)
    _budget_check(check, deadline)
    if set(metric_tables) != set(METRIC_TABLE_NAMES):
        raise FigureBuildError("metric_table_set_mismatch")
    if set(contrast) != {"tables", "diagnostics", "manifest"}:
        raise FigureBuildError("contrast_export_schema_mismatch")
    if set(contrast["tables"]) != set(CONTRAST_TABLE_NAMES):
        raise FigureBuildError("contrast_table_set_mismatch")
    tables = {**metric_tables, **contrast["tables"]}
    diagnostics = contrast["diagnostics"]
    contrast_manifest = contrast["manifest"]
    for name in TABLE_NAMES:
        if not isinstance(tables[name], pd.DataFrame):
            raise FigureBuildError("prepared_table_not_dataframe:" + name)
    if not isinstance(diagnostics, Mapping):
        raise FigureBuildError("prepared_diagnostics_not_mapping")
    if not isinstance(contrast_manifest, Mapping):
        raise FigureBuildError("prepared_contrast_manifest_not_mapping")
    return tables, diagnostics, contrast_manifest


def _metadata():
    return {
        "null_csv_cells": "empty cells denote undefined or non-computable values",
        "score_units": (
            "accuracy, recall and confidence are proportions; balanced-accuracy "
            "effects are proportion differences, not percentage points. NLL is "
            "negative log likelihood; Brier is squared probability error. "
            "Counts are not scores."
        ),
        "confusion_reliability_counts": (
            "counts reflect repeated appearances of the same underlying units"
        ),
        "class_sensitivity_recall": {
            "pooled_repeated_appearance_recall": "pooled over repeated appearances",
            "mean_supported_context_recall": (
                "mean over supported contexts; a separately labelled quantity"
            ),
        },
        "support_boundaries": ("support-limited quantities are conditional on available contexts"),
        "no_G4_decision": True,
        "no_automatic_selection": True,
    }


def _file_entry(root, name, frame=None):
    path = root / name
    entry = {"file": name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}
    if frame is not None:
        entry["rows"] = int(frame.shape[0])
        entry["columns"] = [str(column) for column in frame.columns]
    return entry


def write_release_tables(analysis, *, output, source_refs, check, deadline):
    refs = _validate_source_refs(source_refs)
    root = _normalize_new(output, "output")
    if _in_git_worktree(root):
        raise FigureBuildError("output must sit outside a git worktree")
    tables, diagnostics, contrast_manifest = _prepare(analysis, check, deadline)
    created = False
    try:
        _budget_check(check, deadline)
        _make_root(root)
        created = True
        for name in TABLE_NAMES:
            _budget_check(check, deadline)
            _write_exclusive(
                root / (name + ".csv"),
                tables[name].to_csv(index=False, float_format="%.17g", na_rep=""),
            )
        _budget_check(check, deadline)
        _write_exclusive(root / DIAGNOSTICS_FILE, _canonical_json_bytes(dict(diagnostics)))
        manifest = {
            "schema": SCHEMA,
            "status": "built_unreviewed",
            "tables": {
                name: _file_entry(root, name + ".csv", tables[name]) for name in TABLE_NAMES
            },
            "diagnostics": _file_entry(root, DIAGNOSTICS_FILE),
            "contrast_manifest": dict(contrast_manifest),
            "input_source_hashes": dict(refs),
            "reviewed": False,
            "disclosure_reviewed": False,
            "published": False,
            "external_authentication": False,
            "metadata": _metadata(),
        }
        _budget_check(check, deadline)
        _write_exclusive(root / MANIFEST_FILE, _canonical_json_bytes(manifest))
        return manifest
    except BaseException as exc:  # noqa: BLE001 -- partial private output is kept.
        if created:
            try:
                _write_exclusive(
                    root / FAILURE_FILE,
                    _canonical_json_bytes(
                        {
                            "status": "failed",
                            "error_code": str(getattr(exc, "code", type(exc).__name__)),
                            "error_type": type(exc).__name__,
                        }
                    ),
                )
            except BaseException:  # noqa: BLE001 -- failure marker is best effort.
                pass
        raise
