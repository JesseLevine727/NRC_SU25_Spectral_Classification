"""Deterministic private release writer for the P06/P11 evaluation.

The module never performs scientific resampling.  It authenticates an existing
private analysis directory, rebuilds the released tables through the existing
table layer and renders already-frozen figure semantics.  No inference engine
or fitter is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import sys
import time
from pathlib import Path

import pandas as pd

from atlas_sers.evaluation.p06p11_release_tables import prepare_tables
from atlas_sers.visualization.p05_figure_runtime import _compile
from atlas_sers.visualization.p06p11_deletion_data import build_deletion_semantics
from atlas_sers.visualization.p06p11_figure_data import build_semantics
from atlas_sers.visualization.p06p11_figures import (
    _render_html,
    _render_tex,
    _semantic_sha,
)
from atlas_sers.visualization.p06p11_interval_data import build_interval_semantics


class ReleaseError(Exception):
    """Raised when the private release contract is violated."""


PROTOCOL_SHA256 = "14e0de1094411b0b3d63ebc98d3caa3350945e224c5f963ac9ddf109656061c0"
DEADLINE_SECONDS = 600.0
MAX_OUTPUT_BYTES = 1024**3
DRAW_COUNT = 10000
EXPECTED_CROSSCHECK_COUNT = 780
MAX_CROSSCHECK_ERROR = 1e-12
EXPECTED_TIE_ROWS = 7

TABLE_NAMES = (
    "domain_metrics",
    "summary",
    "intervals",
    "feasibility",
    "leave_one_out",
    "sign_flip",
    "g4_criteria",
)

METRIC_NAMES = (
    "domain_metrics",
    "model_summary",
    "confusion",
    "class_sensitivity",
    "reliability_bins",
    "reliability_summary",
)

PROTOCOL_RELATIVE = "plan/P06P11_INFERENCE_PROTOCOL.md"

WRITTEN_FILES = (
    "start.json",
    "g4_decision.json",
    "panel_M01.parquet",
    "panel_M06.parquet",
    "arrays.npz",
    "registry.json",
    "point_audit.csv",
    "tables/domain_metrics.csv",
    "tables/summary.csv",
    "tables/intervals.csv",
    "tables/feasibility.csv",
    "tables/leave_one_out.csv",
    "tables/sign_flip.csv",
    "tables/g4_criteria.csv",
)

FIGURE_IDS = (
    "F_P06_primary_scatter",
    "F_P06_effect_intervals",
    "F_P06_weight_sensitivity",
    "F_P06_deletion_stability",
)

FIGURE_ROWS = (26, 12, 6, 24)

_INCLUDEGRAPHICS = re.compile(r"\\includegraphics", re.IGNORECASE)
_EXTERNAL_SCRIPT = re.compile(r"<script\b[^>]*\bsrc\s*=", re.IGNORECASE)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_no_symlink(path: Path) -> None:
    for candidate in (Path(path), *Path(path).parents):
        if candidate.is_symlink():
            raise ReleaseError("symlinked path rejected")


def _is_within(base: Path, target: Path) -> bool:
    base_str = str(Path(base))
    target_str = str(Path(target))
    return target_str == base_str or target_str.startswith(base_str.rstrip(os.sep) + os.sep)


def _has_git_ancestor(path: Path) -> bool:
    for candidate in Path(path).parents:
        if (candidate / ".git").exists():
            return True
    return False


def _validate_paths(analysis: Path, output: Path) -> None:
    analysis = Path(analysis)
    output = Path(output)
    if not analysis.is_absolute() or not output.is_absolute():
        raise ReleaseError("paths must be absolute")
    _assert_no_symlink(analysis)
    _assert_no_symlink(output)
    analysis = analysis.resolve()
    output = output.resolve()
    _assert_no_symlink(analysis)
    _assert_no_symlink(output)
    if not analysis.is_dir():
        raise ReleaseError("analysis directory rejected")
    if output.exists():
        raise ReleaseError("output already exists")
    if not output.parent.is_dir():
        raise ReleaseError("output parent missing")
    if _is_within(analysis, output):
        raise ReleaseError("output inside analysis")
    if _has_git_ancestor(output):
        raise ReleaseError("output inside repository")


def _write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        handle.write(payload)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _write_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _hash_tree(output: Path) -> dict:
    hashes: dict = {}
    for root, _dirs, names in os.walk(output):
        for name in sorted(names):
            path = Path(root) / name
            relative = path.relative_to(output).as_posix()
            if relative == "release_manifest.json":
                continue
            hashes[relative] = _sha256_file(path)
    return hashes


def _figure_hashes(figures_dir: Path, output: Path, figure_id: str) -> dict:
    hashes: dict = {}
    for path in sorted(figures_dir.iterdir()):
        if path.name.startswith(figure_id + "."):
            hashes[path.relative_to(output).as_posix()] = _sha256_file(path)
    return hashes


def _release_code_hashes() -> dict:
    root = Path(__file__).resolve().parents[3]
    src = root / "src"
    hashes: dict = {}
    if src.is_dir():
        for path in sorted(src.rglob("*.py")):
            hashes[path.relative_to(root).as_posix()] = _sha256_file(path)
    protocol = root / PROTOCOL_RELATIVE
    if protocol.is_file():
        hashes[PROTOCOL_RELATIVE] = _sha256_file(protocol)
    return hashes


def _merge_figure_specs(parts) -> list:
    merged: dict = {}
    for part in parts:
        if not isinstance(part, dict):
            raise ReleaseError("figure spec rejected")
        for figure_id, spec in part.items():
            if figure_id not in FIGURE_IDS:
                raise ReleaseError("figure id rejected")
            if figure_id in merged:
                raise ReleaseError("duplicate figure id")
            if not isinstance(spec, dict):
                raise ReleaseError("figure spec rejected")
            if set(spec) != {"semantic", "title", "caption", "kind"}:
                raise ReleaseError("figure spec keys rejected")
            merged[figure_id] = spec
    if set(merged) != set(FIGURE_IDS):
        raise ReleaseError("figure ids rejected")
    return [merged[figure_id] for figure_id in FIGURE_IDS]


def run_release(analysis: Path, output: Path) -> dict:
    started = time.perf_counter()
    analysis = Path(analysis)
    output = Path(output)

    _validate_paths(analysis, output)

    def remaining() -> float:
        left = DEADLINE_SECONDS - (time.perf_counter() - started)
        if left <= 0.0:
            raise ReleaseError("deadline exceeded")
        return left

    def check_budget() -> None:
        remaining()
        total = 0
        if output.is_dir():
            for root, _dirs, names in os.walk(output):
                for name in names:
                    total += (Path(root) / name).stat().st_size
        if total > MAX_OUTPUT_BYTES:
            raise ReleaseError("output budget exceeded")

    release_code = _release_code_hashes()
    if release_code.get(PROTOCOL_RELATIVE) != PROTOCOL_SHA256:
        raise ReleaseError("protocol document rejected")

    receipt_path = analysis / "receipt.json"
    _assert_no_symlink(receipt_path)
    receipt_raw = receipt_path.read_bytes()
    receipt_sha = _sha256_bytes(receipt_raw)
    receipt = json.loads(receipt_raw)

    if receipt.get("status") != "success":
        raise ReleaseError("receipt status rejected")
    if receipt.get("mode") != "full":
        raise ReleaseError("receipt mode rejected")
    draws = receipt.get("draws")
    if isinstance(draws, bool) or draws != DRAW_COUNT:
        raise ReleaseError("receipt draws rejected")
    if receipt.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ReleaseError("protocol rejected")

    code_hashes = receipt.get("code_hashes")
    input_hashes = receipt.get("input_hashes")
    written_files = receipt.get("written_files")
    seeds = receipt.get("seeds")
    software_versions = receipt.get("software_versions")
    if not isinstance(code_hashes, dict) or not isinstance(input_hashes, dict):
        raise ReleaseError("receipt maps rejected")
    if not isinstance(written_files, dict) or set(written_files) != set(WRITTEN_FILES):
        raise ReleaseError("receipt written files rejected")
    if not isinstance(seeds, dict) or not isinstance(software_versions, dict):
        raise ReleaseError("receipt metadata rejected")

    for name in WRITTEN_FILES:
        path = analysis / name
        _assert_no_symlink(path)
        if not path.is_file():
            raise ReleaseError("analysis artifact missing")
        if _sha256_file(path) != written_files[name]:
            raise ReleaseError("analysis artifact mismatch")

    start_payload = json.loads((analysis / "start.json").read_bytes())
    if start_payload.get("code_hashes") != code_hashes:
        raise ReleaseError("start code hashes rejected")
    if start_payload.get("input_hashes") != input_hashes:
        raise ReleaseError("start input hashes rejected")

    remaining()
    check_budget()

    tables = {}
    for name in TABLE_NAMES:
        tables[name] = pd.read_csv(
            analysis / "tables" / (name + ".csv"),
            float_precision="round_trip",
        )
    panels = {
        "M01": pd.read_parquet(analysis / "panel_M01.parquet"),
        "M06": pd.read_parquet(analysis / "panel_M06.parquet"),
    }

    prepared = prepare_tables(tables, panels)
    inference_tables = prepared["inference_tables"]
    metrics = prepared["metrics"]
    tie_corrections = prepared["tie_corrections"]
    crosscheck_count = prepared["crosscheck_count"]
    crosscheck_max_error = prepared["crosscheck_max_error"]

    if not isinstance(inference_tables, dict) or set(inference_tables) != set(TABLE_NAMES):
        raise ReleaseError("inference tables rejected")
    if not isinstance(metrics, dict) or set(metrics) != set(METRIC_NAMES):
        raise ReleaseError("metrics keys rejected")
    if isinstance(crosscheck_count, bool):
        raise ReleaseError("crosscheck count rejected")
    crosscheck_count = int(crosscheck_count)
    if crosscheck_count != EXPECTED_CROSSCHECK_COUNT:
        raise ReleaseError("crosscheck count rejected")
    if isinstance(crosscheck_max_error, bool):
        raise ReleaseError("crosscheck error rejected")
    crosscheck_max_error = float(crosscheck_max_error)
    if not math.isfinite(crosscheck_max_error):
        raise ReleaseError("crosscheck error rejected")
    if crosscheck_max_error < 0.0 or crosscheck_max_error > MAX_CROSSCHECK_ERROR:
        raise ReleaseError("crosscheck error rejected")
    if len(tie_corrections) != EXPECTED_TIE_ROWS:
        raise ReleaseError("tie correction rows rejected")

    remaining()
    output.mkdir()
    (output / "tables").mkdir()
    (output / "metrics").mkdir()
    (output / "figures").mkdir()

    for key in inference_tables:
        _write_csv(output / "tables" / (str(key) + ".csv"), inference_tables[key])
    _write_csv(output / "tables" / "tie_corrections.csv", tie_corrections)
    check_budget()

    for key in metrics:
        _write_csv(output / "metrics" / (str(key) + ".csv"), metrics[key])
    check_budget()

    _write_bytes(output / "g4_decision.json", (analysis / "g4_decision.json").read_bytes())
    check_budget()

    parts = (
        build_semantics(inference_tables),
        build_interval_semantics(inference_tables),
        build_deletion_semantics(inference_tables),
    )
    specs = _merge_figure_specs(parts)

    figure_manifest = []
    figures_dir = output / "figures"
    for figure_id, expected_rows, spec in zip(FIGURE_IDS, FIGURE_ROWS, specs, strict=True):
        frame = spec["semantic"]
        title = spec["title"]
        caption = spec["caption"]
        kind = spec["kind"]
        if int(len(frame)) != expected_rows:
            raise ReleaseError("figure row count rejected")
        semantic_sha = _semantic_sha(frame)
        tex_payload = _render_tex(figure_id, frame, title, caption, kind, semantic_sha)
        html_payload = _render_html(figure_id, frame, title, caption, kind, semantic_sha)
        if not isinstance(tex_payload, str) or not isinstance(html_payload, str):
            raise ReleaseError("figure render rejected")
        if _INCLUDEGRAPHICS.search(tex_payload):
            raise ReleaseError("figure tex rejected")
        if _EXTERNAL_SCRIPT.search(html_payload):
            raise ReleaseError("figure html rejected")

        _write_csv(figures_dir / (figure_id + ".csv"), frame)
        _write_bytes(figures_dir / (figure_id + ".tex"), tex_payload.encode("utf-8"))
        _write_bytes(figures_dir / (figure_id + ".html"), html_payload.encode("utf-8"))

        tex_path = figures_dir / (figure_id + ".tex")
        pdf_path = figures_dir / (figure_id + ".pdf")
        png_path = figures_dir / (figure_id + ".png")
        log_path = figures_dir / (figure_id + ".log")
        _compile(
            tex_path,
            pdf_path,
            png_path,
            log_path,
            deadline=time.perf_counter() + remaining(),
        )

        figure_manifest.append(
            {
                "id": figure_id,
                "semantic_sha256": semantic_sha,
                "rows": int(len(frame)),
                "title": title,
                "caption": caption,
                "files": _figure_hashes(figures_dir, output, figure_id),
            }
        )
        check_budget()

    _assert_no_symlink(receipt_path)
    for name in WRITTEN_FILES:
        _assert_no_symlink(analysis / name)
    if _sha256_file(receipt_path) != receipt_sha:
        raise ReleaseError("input preservation rejected")
    if receipt_path.read_bytes() != receipt_raw:
        raise ReleaseError("input preservation rejected")
    for name in WRITTEN_FILES:
        if _sha256_file(analysis / name) != written_files[name]:
            raise ReleaseError("input preservation rejected")

    if _release_code_hashes() != release_code:
        raise ReleaseError("release code changed")

    remaining()
    check_budget()
    public_files = _hash_tree(output)

    manifest = {
        "status": "requires_supervisor_review",
        "original_code_hashes": code_hashes,
        "original_input_hashes": input_hashes,
        "original_written_files": written_files,
        "original_receipt_sha256": receipt_sha,
        "release_code_hashes": release_code,
        "seeds": seeds,
        "software_versions": software_versions,
        "protocol_sha256": PROTOCOL_SHA256,
        "draw_count": DRAW_COUNT,
        "input_preservation": True,
        "crosscheck_count": crosscheck_count,
        "crosscheck_max_error": crosscheck_max_error,
        "tie_correction_rows": int(len(tie_corrections)),
        "figures": figure_manifest,
        "files": public_files,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }

    _write_bytes(
        output / "release_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
    )
    check_budget()
    return manifest


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="p06p11_release",
        description="Write deterministic private P06/P11 release artifacts.",
    )
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    analysis = Path(args.analysis)
    output = Path(args.output)
    try:
        _validate_paths(analysis, output)
        manifest = run_release(analysis, output)
    except ReleaseError as exc:
        print("release failed: " + str(exc), file=sys.stderr)
        return 1
    except Exception:
        print("release failed", file=sys.stderr)
        return 1
    print(manifest["status"] + " files=" + str(len(manifest["files"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
