"""T349 bounded private figure build writer for already prepared P08 figures.

No CLI, no fitting, no array loading, no new statistics, no publication, no
git, no networking, no data selection, no report prose.  The writer consumes
an already reviewed ``prepared`` mapping and emits private build evidence.
"""

from __future__ import annotations

import hashlib
import html.parser
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from atlas_sers.governance.p08_owned_process import _identity, terminate_owned_child
from atlas_sers.visualization.p08_f02_render import _canonical_json, _canonical_sha

CHILD_SECONDS = 45.0
POLL_SECONDS = 0.1
LOG_LIMIT = 1 << 20
PDF_WIDTH_MM = 181.86
PDF_WIDTH_TOL_MM = 0.2
FIGURE_PANEL_COUNTS = {
    "P08-F01": 17,
    "P08-F02": 20,
    "P08-F03": 12,
    "P08-F04": 24,
    "P08-F07": 20,
    "P08-U1-training-diagnostics": 24,
}

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_TEX_FORBIDDEN = re.compile(
    r"\\(?:includegraphics|pgfimage|includepdf|pdfximage|includestandalone)\b",
    re.IGNORECASE,
)
_CSS_NETWORK = re.compile(r"@import|url\(\s*['\"]?\s*(?:https?:)?//", re.IGNORECASE)
_HTML_DOC = re.compile(r"<!doctype html", re.IGNORECASE)
_HTML_ROOT = re.compile(r"<html[\s>]", re.IGNORECASE)
_ASSET_ATTRS = {
    "script": ("src",),
    "link": ("href",),
    "img": ("src",),
    "iframe": ("src",),
    "object": ("data",),
    "embed": ("src",),
    "source": ("src", "srcset"),
    "video": ("src", "poster"),
    "audio": ("src",),
    "image": ("href", "xlink:href"),
    "use": ("href", "xlink:href"),
    "input": ("src",),
}


class FigureBuildError(RuntimeError):
    """Path-free bounded build failure."""


def _canonical_semantic_sha(semantic):
    return _canonical_sha(semantic)


def _guard(check):
    if not callable(check):
        raise FigureBuildError("check guard must be callable")
    if check() is False:
        raise FigureBuildError("check guard refused")


def _budget_check(check, deadline):
    _guard(check)
    if time.monotonic() >= deadline:
        raise FigureBuildError("global deadline exceeded")


def _is_within(child, parent):
    child = os.path.normpath(str(child))
    parent = os.path.normpath(str(parent))
    if child == parent:
        return False
    return child.startswith(parent.rstrip(os.sep) + os.sep)


def _normalize_new(path, label):
    raw = Path(path)
    if not raw.is_absolute():
        raise FigureBuildError(f"{label} must be absolute")
    if ".." in raw.parts:
        raise FigureBuildError(f"literal '..' refused in {label}")
    norm = Path(os.path.normpath(str(path)))
    if norm.exists() or norm.is_symlink():
        raise FigureBuildError(f"{label} already exists")
    if norm == Path(norm.anchor) or norm == Path.home():
        raise FigureBuildError(f"{label} too broad")
    parent = norm.parent
    if not parent.exists() or not parent.is_dir():
        raise FigureBuildError(f"{label} parent missing")
    cur = parent
    while True:
        if cur.is_symlink():
            raise FigureBuildError(f"{label} symlink ancestor refused")
        if cur == cur.parent:
            break
        cur = cur.parent
    return norm


def _in_git_worktree(path):
    cur = Path(os.path.normpath(str(path)))
    while True:
        if (cur / ".git").exists():
            return True
        if cur == cur.parent:
            return False
        cur = cur.parent


def _validate_roots(output, private_logs, style_config_path):
    out = _normalize_new(output, "output")
    logs = _normalize_new(private_logs, "private_logs")
    if out == logs or _is_within(out, logs) or _is_within(logs, out):
        raise FigureBuildError("output and private_logs must not nest")
    style = Path(style_config_path)
    if not style.is_absolute() or not style.is_file() or style.is_symlink():
        raise FigureBuildError("style config must be an existing regular file")
    if ".." in style.parts or any(p.is_symlink() for p in style.parents):
        raise FigureBuildError("style config ancestor invalid")
    style_norm = os.path.normpath(str(style))
    if _is_within(style_norm, out) or _is_within(style_norm, logs):
        raise FigureBuildError("style config must sit outside the build roots")
    if _in_git_worktree(out) or _in_git_worktree(logs):
        raise FigureBuildError("build roots must not sit in a git worktree")
    return out, logs, style


def _validate_source_refs(source_refs):
    if not isinstance(source_refs, dict) or not source_refs:
        raise FigureBuildError("source_refs must be a nonempty mapping")
    checked = {}
    for label, digest in source_refs.items():
        if (
            not isinstance(label, str)
            or ".." in label
            or "/" in label
            or "\\" in label
            or not _SLUG_RE.match(label)
        ):
            raise FigureBuildError("source_refs label invalid")
        if not isinstance(digest, str) or not _HEX64_RE.match(digest):
            raise FigureBuildError("source_refs digest invalid")
        checked[label] = digest
    return checked


def _read_style(style):
    try:
        raw = style.read_text(encoding="utf-8")
    except OSError as exc:
        raise FigureBuildError("style config unreadable") from exc
    try:
        json.loads(raw)
    except ValueError as exc:
        raise FigureBuildError("style config is not JSON") from exc
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class _AssetParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.problems = []

    def _scan(self, tag, attrs):
        tag = tag.lower()
        missing = _ASSET_ATTRS.get(tag)
        for name, value in attrs:
            if value is None:
                continue
            low = name.lower()
            if low.startswith("xmlns"):
                continue
            text = value.strip().lower()
            if text.startswith(("http:", "https:", "//", "ftp:")):
                self.problems.append((tag, name))
                continue
            if missing and low in missing and not text.startswith(("#", "data:")):
                self.problems.append((tag, name))

    def handle_starttag(self, tag, attrs):
        self._scan(tag, attrs)

    handle_startendtag = handle_starttag


def _check_native_tex(tex):
    if _TEX_FORBIDDEN.search(tex):
        raise FigureBuildError("native TeX contains forbidden inclusion")


def _check_html(text):
    if not _HTML_DOC.search(text) or not _HTML_ROOT.search(text):
        raise FigureBuildError("html must be a full document")
    if _CSS_NETWORK.search(text):
        raise FigureBuildError("html contains an external css reference")
    parser = _AssetParser()
    parser.feed(text)
    if parser.problems:
        raise FigureBuildError("html contains an external asset reference")


def _validate_prepared(prepared):
    if not isinstance(prepared, dict) or set(prepared) != {
        "semantic",
        "semantic_sha256",
        "panels",
        "manifest",
    }:
        raise FigureBuildError("prepared keys mismatch")
    semantic = prepared["semantic"]
    if not isinstance(semantic, dict):
        raise FigureBuildError("semantic must be a mapping")
    figure_id = semantic.get("figure_id")
    if figure_id not in FIGURE_PANEL_COUNTS:
        raise FigureBuildError("figure_id not supported")
    sha = prepared["semantic_sha256"]
    if not isinstance(sha, str) or not _HEX64_RE.match(sha):
        raise FigureBuildError("semantic_sha256 malformed")
    if sha != _canonical_semantic_sha(semantic):
        raise FigureBuildError("semantic hash mismatch")
    incoming = prepared["manifest"]
    if not isinstance(incoming, dict):
        raise FigureBuildError("incoming manifest must be a mapping")
    if (
        incoming.get("status") != "prepared"
        or incoming.get("reviewed") is not False
        or incoming.get("published") is not False
    ):
        raise FigureBuildError("incoming manifest not prepared/unreviewed")
    panels = prepared["panels"]
    if not isinstance(panels, list) or not panels:
        raise FigureBuildError("panels must be a nonempty list")
    if len(panels) != FIGURE_PANEL_COUNTS[figure_id]:
        raise FigureBuildError("panel count mismatch")
    seen = set()
    for panel in panels:
        if not isinstance(panel, dict):
            raise FigureBuildError("panel must be a mapping")
        for key in ("slug", "semantic_sha256", "tex", "html", "csv"):
            if key not in panel:
                raise FigureBuildError("panel missing required key")
        slug = panel["slug"]
        if (
            not isinstance(slug, str)
            or ".." in slug
            or "/" in slug
            or "\\" in slug
            or not _SLUG_RE.match(slug)
        ):
            raise FigureBuildError("panel slug invalid")
        if slug in seen:
            raise FigureBuildError("duplicate panel slug")
        seen.add(slug)
        if panel["semantic_sha256"] != sha:
            raise FigureBuildError("panel hash mismatch")
        for fmt in ("tex", "html", "csv"):
            if not isinstance(panel[fmt], str) or not panel[fmt]:
                raise FigureBuildError("panel format empty")
        if sha not in panel["tex"] or sha not in panel["html"]:
            raise FigureBuildError("sha must occur in tex and html")
        _check_native_tex(panel["tex"])
        _check_html(panel["html"])
    return semantic, sha, panels, figure_id


def _make_root(path):
    os.makedirs(str(path), mode=0o700)
    os.chmod(str(path), 0o700)


def _make_children(root, names):
    for name in names:
        child = root / name
        os.makedirs(str(child), mode=0o700)
        os.chmod(str(child), 0o700)


def _write_exclusive(path, data):
    payload = data if isinstance(data, bytes) else data.encode("utf-8")
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)


def _copy_exclusive(source, destination):
    fd = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out_handle:
        with open(str(source), "rb") as in_handle:
            while True:
                chunk = in_handle.read(1 << 20)
                if not chunk:
                    break
                out_handle.write(chunk)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bounded(path, limit=LOG_LIMIT):
    try:
        size = os.path.getsize(str(path))
    except OSError as exc:
        raise FigureBuildError("log missing") from exc
    if size == 0:
        raise FigureBuildError("log empty")
    if size > limit:
        raise FigureBuildError("log truncated")
    with open(str(path), "rb") as handle:
        data = handle.read(limit + 1)
    if len(data) != size or len(data) > limit:
        raise FigureBuildError("log truncated")
    return data.decode("utf-8", "replace")


def _reject_compile_log(text):
    if "Overfull \\hbox" in text or "Overfull \\vbox" in text:
        raise FigureBuildError("compile log has an overfull box warning")
    if (
        "undefined reference" in text.lower()
        or "LaTeX Warning: Reference" in text
        or "undefined on input line" in text.lower()
    ):
        raise FigureBuildError("compile log has an undefined reference")


def _run_child(argv, cwd, log_path, check, deadline, max_seconds=CHILD_SECONDS):
    _guard(check)
    now = time.monotonic()
    if now >= deadline:
        raise FigureBuildError("global deadline exceeded before child")
    budget = min(max_seconds, deadline - now)
    if budget <= 0:
        raise FigureBuildError("no time budget for child")
    with open(str(log_path), "xb") as log_handle:
        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=True,
        )
    identity = None
    cleanup = {"unexpected": False, "error": None}
    child_deadline = time.monotonic() + budget
    returncode = None
    timed_out = False
    failure = None
    try:
        try:
            _, _, identity = _identity(proc)
        except Exception as exc:
            if proc.poll() is None:
                raise FigureBuildError("owned child identity unavailable") from exc
        while True:
            _guard(check)
            if log_path.stat().st_size > LOG_LIMIT:
                raise FigureBuildError("child log exceeded byte limit")
            if time.monotonic() >= child_deadline:
                timed_out = True
                break
            returncode = proc.poll()
            if returncode is not None:
                break
            time.sleep(POLL_SECONDS)
    except BaseException as exc:
        failure = exc
    finally:
        try:
            cleanup["unexpected"] = terminate_owned_child(proc, identity)
        except BaseException as exc:  # noqa: BLE001 - preserve original error
            cleanup["error"] = exc
    if cleanup["error"] is not None:
        raise FigureBuildError("owned child cleanup failed") from cleanup["error"]
    if failure is not None:
        raise failure
    if timed_out:
        raise FigureBuildError("child exceeded its time budget")
    if cleanup["unexpected"]:
        raise FigureBuildError("unexpected descendant processes observed")
    if returncode != 0:
        raise FigureBuildError("child returned a nonzero status")
    _budget_check(check, deadline)
    return returncode


def _parse_pdfinfo(text):
    pages = None
    width = None
    height = None
    for line in text.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":", 1)[1].strip())
        elif line.startswith("Page size:"):
            match = re.search(r"([0-9.]+)\s+x\s+([0-9.]+)\s+pts", line)
            if match:
                width = float(match.group(1))
                height = float(match.group(2))
    if pages != 1:
        raise FigureBuildError("pdf must have exactly one page")
    if width is None or height is None:
        raise FigureBuildError("pdf page size not reported")
    width_mm = width * 25.4 / 72.0
    if abs(width_mm - PDF_WIDTH_MM) > PDF_WIDTH_TOL_MM:
        raise FigureBuildError("pdf width outside tolerance")
    return {
        "pages": pages,
        "width_mm": round(width_mm, 4),
        "height_mm": round(height * 25.4 / 72.0, 4),
    }


def _parse_pdfimages(text):
    if not re.search(r"(?m)^page\s+num\s+type\s+width\s+height", text):
        raise FigureBuildError("pdfimages header missing")
    rows = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("page"):
            continue
        if stripped.startswith("---"):
            continue
        if re.match(r"^\d+\s+\d+", stripped):
            rows += 1
        else:
            raise FigureBuildError("pdfimages record malformed")
    return rows


def _parse_pdffonts(text):
    fonts = []
    started = False
    for line in text.splitlines():
        if not started:
            if line.startswith("name") and "emb" in line:
                started = True
            continue
        if not line.strip() or line.startswith("---"):
            continue
        tokens = line.split()
        if (
            len(tokens) < 8
            or any(v not in ("yes", "no") for v in tokens[-5:-2])
            or not all(v.isdigit() for v in tokens[-2:])
        ):
            raise FigureBuildError("pdffonts record malformed")
        fonts.append(
            {
                "name": tokens[0],
                "type": " ".join(tokens[1:-6]),
                "encoding": tokens[-6],
                "emb": tokens[-5],
                "sub": tokens[-4],
                "uni": tokens[-3],
                "object_id": [int(v) for v in tokens[-2:]],
            }
        )
    if not started:
        raise FigureBuildError("pdffonts header missing")
    return fonts


def _render_panel(slug, tex, output, log_root, check, deadline):
    panel_log = log_root / slug
    _make_root(panel_log)
    _make_children(panel_log, ("work",))
    work = panel_log / "work"
    tex_path = work / (slug + ".tex")
    _write_exclusive(tex_path, tex)
    for pass_number in (1, 2):
        pass_log = panel_log / ("pdflatex_pass" + str(pass_number) + ".log")
        _run_child(
            [
                "pdflatex",
                "-halt-on-error",
                "-interaction=nonstopmode",
                "-no-shell-escape",
                slug + ".tex",
            ],
            work,
            pass_log,
            check,
            deadline,
        )
        _reject_compile_log(_read_bounded(pass_log))
    pdf_source = work / (slug + ".pdf")
    if not pdf_source.is_file():
        raise FigureBuildError("pdflatex produced no pdf")
    _copy_exclusive(pdf_source, output / "pdf" / (slug + ".pdf"))
    png_stem = work / (slug + "_png")
    _run_child(
        ["pdftocairo", "-png", "-singlefile", "-r", "300", str(pdf_source), str(png_stem)],
        work,
        panel_log / "pdftocairo.log",
        check,
        deadline,
    )
    png_source = work / (slug + "_png.png")
    if not png_source.is_file():
        raise FigureBuildError("pdftocairo produced no png")
    _copy_exclusive(png_source, output / "png" / (slug + ".png"))
    info_log = panel_log / "pdfinfo.log"
    _run_child(["pdfinfo", str(pdf_source)], work, info_log, check, deadline)
    inspection = _parse_pdfinfo(_read_bounded(info_log))
    images_log = panel_log / "pdfimages.log"
    _run_child(["pdfimages", "-list", str(pdf_source)], work, images_log, check, deadline)
    if _parse_pdfimages(_read_bounded(images_log)):
        raise FigureBuildError("pdf contains embedded raster objects")
    fonts_log = panel_log / "pdffonts.log"
    _run_child(["pdffonts", str(pdf_source)], work, fonts_log, check, deadline)
    fonts = _parse_pdffonts(_read_bounded(fonts_log))
    if not fonts:
        raise FigureBuildError("pdf has no font records")
    for font in fonts:
        if font["emb"].lower() != "yes":
            raise FigureBuildError("pdf font is not embedded")
    return {
        "slug": slug,
        "pdf": "pdf/" + slug + ".pdf",
        "png": "png/" + slug + ".png",
        "inspection": inspection,
        "raster_objects": 0,
        "fonts": fonts,
    }


def _write_index(output, figure_id, sha, panels):
    rows = "".join(
        '<li><a href="html/'
        + panel["slug"]
        + '.html">'
        + panel["slug"]
        + '</a> <a href="pdf/'
        + panel["slug"]
        + '.pdf">pdf</a> '
        '<a href="png/' + panel["slug"] + '.png">png</a></li>'
        for panel in panels
    )
    document = (
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>'
        + figure_id
        + " "
        + sha
        + "</title></head><body><h1>"
        + figure_id
        + "</h1><p>"
        + sha
        + "</p><ul>"
        + rows
        + "</ul></body></html>"
    )
    _write_exclusive(output / "built_index.html", document)


def _output_files(output):
    files = []
    for root, _dirs, names in os.walk(str(output)):
        for name in names:
            path = Path(root) / name
            rel = os.path.relpath(str(path), str(output)).replace(os.sep, "/")
            if rel == "manifest.json":
                continue
            files.append(
                {
                    "path": rel,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    files.sort(key=lambda item: item["path"])
    return files


def build_figure_bundle(
    prepared, *, output, private_logs, source_refs, style_config_path, check, deadline
):
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        raise FigureBuildError("deadline must be a finite number")
    if not math.isfinite(deadline):
        raise FigureBuildError("deadline must be finite")
    _budget_check(check, deadline)
    semantic, sha, panels, figure_id = _validate_prepared(prepared)
    refs = _validate_source_refs(source_refs)
    out, logs, style = _validate_roots(output, private_logs, style_config_path)
    style_sha = _read_style(style)
    _budget_check(check, deadline)

    records = []
    fonts = []
    owned = False
    old_umask = os.umask(0o077)
    try:
        _make_root(out)
        owned = True
        _make_children(out, ("data", "tikz", "html", "pdf", "png"))
        _make_root(logs)
        _write_exclusive(out / "data" / "semantic.json", _canonical_json(semantic))
        for panel in panels:
            slug = panel["slug"]
            _write_exclusive(out / "data" / (slug + ".csv"), panel["csv"])
            _write_exclusive(out / "tikz" / (slug + ".tex"), panel["tex"])
            _write_exclusive(out / "html" / (slug + ".html"), panel["html"])
            _budget_check(check, deadline)
        for panel in panels:
            _budget_check(check, deadline)
            record = _render_panel(panel["slug"], panel["tex"], out, logs, check, deadline)
            records.append(record)
            fonts.extend(record["fonts"])
            _budget_check(check, deadline)
        _write_index(out, figure_id, sha, panels)
        outputs = _output_files(out)
        _budget_check(check, deadline)
        manifest = {
            "status": "built_unreviewed",
            "reviewed": False,
            "published": False,
            "disclosure_reviewed": False,
            "visual_reviewed": False,
            "structural_validation": True,
            "external_authentication_verified": False,
            "figure_id": figure_id,
            "semantic_sha256": sha,
            "source_refs": dict(refs),
            "style_config_sha256": style_sha,
            "renderer_source_references": dict(refs),
            "fonts": fonts,
            "pdf_inspections": [record["inspection"] for record in records],
            "outputs": outputs,
            "limits": {
                "child_seconds": CHILD_SECONDS,
                "poll_seconds": POLL_SECONDS,
                "log_bytes": LOG_LIMIT,
                "pdf_width_mm": PDF_WIDTH_MM,
                "pdf_width_tolerance_mm": PDF_WIDTH_TOL_MM,
                "png_dpi": 300,
            },
        }
        _write_exclusive(out / "manifest.json", _canonical_json(manifest))
        _budget_check(check, deadline)
        return manifest
    except BaseException:
        if owned:
            try:
                _write_exclusive(out / "failure.json", '{"status":"failed"}')
            except Exception:
                pass
        raise
    finally:
        os.umask(old_umask)
