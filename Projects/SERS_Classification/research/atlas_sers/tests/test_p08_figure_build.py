import subprocess
import sys
import time

import pytest

from atlas_sers.visualization import p08_figure_build as mod

H = "a" * 64
SEMANTIC = {"figure_id": "P08-F01", "captions": {}, "datatables": {"x": [1, 2]}}
SHA = mod._canonical_semantic_sha(SEMANTIC)


def _tex(value):
    return "\\documentclass{article}\\begin{document}" + value + "\\end{document}"


def _html(value):
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>' + value + "</body></html>"
    )


def _prepared(sha=SHA):
    panels = []
    for slug in ("panel_one", "panel_two"):
        panels.append(
            {
                "slug": slug,
                "semantic_sha256": sha,
                "tex": _tex(sha),
                "html": _html(sha),
                "csv": "x,y\n1,2\n",
                "domain_id": "exp",
                "station": "S",
                "instrument": "I",
                "exploratory": True,
            }
        )
    return {
        "semantic": SEMANTIC,
        "semantic_sha256": sha,
        "panels": panels,
        "manifest": {"status": "prepared", "reviewed": False, "published": False},
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "FIGURE_PANEL_COUNTS", {"P08-F01": 2})
    style = tmp_path / "style.json"
    style.write_text('{"fonts": ["CMR10"]}', encoding="utf-8")
    return {
        "prepared": _prepared(),
        "output": tmp_path / "out",
        "private_logs": tmp_path / "logs",
        "source_refs": {"table_a": H},
        "style_config_path": style,
        "check": lambda: True,
        "deadline": time.monotonic() + 3600.0,
    }


def _fake_render(slug, tex, output, log_root, check, deadline):
    (output / "pdf").mkdir(exist_ok=True)
    (output / "png").mkdir(exist_ok=True)
    (output / "pdf" / (slug + ".pdf")).write_bytes(b"%PDF-1.4\n")
    (output / "png" / (slug + ".png")).write_bytes(b"\x89PNG\r\n")
    return {
        "slug": slug,
        "pdf": "pdf/" + slug + ".pdf",
        "png": "png/" + slug + ".png",
        "inspection": {"pages": 1, "width_mm": 181.86, "height_mm": 250.0},
        "raster_objects": 0,
        "fonts": [{"name": "CMR10", "type": "Type1", "encoding": "Builtin", "emb": "yes"}],
    }


def test_build_success(env, monkeypatch):
    monkeypatch.setattr(mod, "_render_panel", _fake_render)
    manifest = mod.build_figure_bundle(**env)
    assert manifest["status"] == "built_unreviewed"
    assert manifest["structural_validation"] is True
    paths = {item["path"] for item in manifest["outputs"]}
    assert "manifest.json" not in paths
    assert {
        "data/semantic.json",
        "data/panel_one.csv",
        "tikz/panel_one.tex",
        "html/panel_one.html",
        "pdf/panel_one.pdf",
        "png/panel_one.png",
        "built_index.html",
    } <= paths


def _mut_hash(state):
    state["prepared"]["semantic_sha256"] = "0" * 64


def _mut_duplicate(state):
    state["prepared"]["panels"][1]["slug"] = "panel_one"


def _mut_existing(state):
    state["output"].mkdir()


def _mut_refs(state):
    state["source_refs"] = {"../secret": H}


def _mut_raster(state):
    state["prepared"]["panels"][0]["tex"] = "\\includegraphics{x.png}" + SHA


def _mut_remote(state):
    state["prepared"]["panels"][0]["html"] = (
        _html(SHA) + '<script src="https://example.com/x.js"></script>'
    )


@pytest.mark.parametrize(
    "mutate", [_mut_hash, _mut_duplicate, _mut_existing, _mut_refs, _mut_raster, _mut_remote]
)
def test_refusals(env, mutate):
    mutate(env)
    with pytest.raises(mod.FigureBuildError):
        mod.build_figure_bundle(**env)


def test_symlink_parent(env, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    env["output"] = link / "out"
    with pytest.raises(mod.FigureBuildError):
        mod.build_figure_bundle(**env)


def test_git_worktree(env, tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(mod.FigureBuildError):
        mod.build_figure_bundle(**env)


def test_runner_check_cancel_keeps_log(tmp_path):
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return calls["n"] < 2

    log = tmp_path / "child.log"
    with pytest.raises(mod.FigureBuildError):
        mod._run_child(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            log,
            check,
            time.monotonic() + 60.0,
        )
    assert log.is_file()


def test_runner_deadline_fast_exit(tmp_path):
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return True

    log = tmp_path / "fast.log"
    mod._run_child([sys.executable, "-c", "pass"], tmp_path, log, check, time.monotonic() + 60.0)
    assert calls["n"] >= 2


def test_void_guard_and_identical_roots(env, monkeypatch):
    env["check"] = lambda: None
    monkeypatch.setattr(mod, "_render_panel", _fake_render)
    assert mod.build_figure_bundle(**env)["external_authentication_verified"] is False
    env["output"] = env["output"].parent / "identical"
    env["private_logs"] = env["output"]
    with pytest.raises(mod.FigureBuildError, match="must not nest"):
        mod.build_figure_bundle(**env)


def test_font_columns_are_not_confused():
    header = "name type encoding emb sub uni object ID\n----\n"
    font = mod._parse_pdffonts(header + "FONT Type 1 Custom no yes yes 12 0\n")[0]
    assert font == {
        "name": "FONT",
        "type": "Type 1",
        "encoding": "Custom",
        "emb": "no",
        "sub": "yes",
        "uni": "yes",
        "object_id": [12, 0],
    }
    font = mod._parse_pdffonts(header + "FONT CID Type 0C Identity-H yes no no 4 0\n")[0]
    assert font["type"] == "CID Type 0C" and font["emb"] == "yes"
    with pytest.raises(mod.FigureBuildError):
        mod._parse_pdffonts("garbled")


def test_pdf_images_and_undefined_reference_refusals():
    header = (
        "page num type width height color comp bpc enc interp "
        "object ID x-ppi y-ppi size ratio\n-----\n"
    )
    assert mod._parse_pdfimages(header) == 0
    assert mod._parse_pdfimages(header + "1 0 image 1 1 rgb 3 8 image no 4 0 10 10 30B 4%\n") == 1
    with pytest.raises(mod.FigureBuildError):
        mod._parse_pdfimages("garbled")
    with pytest.raises(mod.FigureBuildError):
        mod._reject_compile_log("LaTeX Warning: There were undefined references.")


def test_cancel_reaps_the_actual_owned_child(tmp_path, monkeypatch):
    children = []
    popen = subprocess.Popen

    def capture(*a, **kw):
        proc = popen(*a, **kw)
        children.append(proc)
        return proc

    monkeypatch.setattr(mod.subprocess, "Popen", capture)
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("guard cancellation")

    with pytest.raises(RuntimeError, match="guard cancellation"):
        mod._run_child(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            tmp_path / "cancel.log",
            check,
            time.monotonic() + 60,
        )
    assert len(children) == 1 and children[0].poll() is not None


def test_identity_failure_reaps_owned_child(tmp_path, monkeypatch):
    children = []
    popen = subprocess.Popen

    def capture(*a, **kw):
        proc = popen(*a, **kw)
        children.append(proc)
        return proc

    def fail_identity(proc):
        raise RuntimeError("lookup failed")

    monkeypatch.setattr(mod.subprocess, "Popen", capture)
    monkeypatch.setattr(mod, "_identity", fail_identity)
    with pytest.raises(mod.FigureBuildError, match="identity unavailable"):
        mod._run_child(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            tmp_path / "identity.log",
            lambda: None,
            time.monotonic() + 60,
        )
    assert children[0].poll() is not None


def test_failure_preserves_partial_outputs_and_evidence(env, monkeypatch):
    def fail(*args):
        raise RuntimeError("compile refused")

    monkeypatch.setattr(mod, "_render_panel", fail)
    with pytest.raises(RuntimeError, match="compile refused"):
        mod.build_figure_bundle(**env)
    assert (env["output"] / "failure.json").read_text() == '{"status":"failed"}'
    assert (env["output"] / "tikz/panel_one.tex").is_file()
    assert not (env["output"] / "manifest.json").exists()


def test_expired_deadline_does_not_start_build(env):
    env["deadline"] = time.monotonic() - 1
    with pytest.raises(mod.FigureBuildError, match="deadline exceeded"):
        mod.build_figure_bundle(**env)
    assert not env["output"].exists()
