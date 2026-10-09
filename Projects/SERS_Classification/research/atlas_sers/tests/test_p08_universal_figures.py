import hashlib
import json
import os

import pytest

from atlas_sers.visualization import p08_figure_build as _writer
from atlas_sers.visualization import p08_universal_figures as u


def _style_file(root):
    path = os.path.join(root, "style.json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('{"theme": "black-roman"}')
    return path


def _prepared(figure_id, count):
    semantic = {"figure_id": figure_id, "note": "synthetic"}
    sha = _writer._canonical_semantic_sha(semantic)
    panels = []
    for index in range(count):
        panels.append(
            {
                "slug": "panel" + str(index),
                "semantic_sha256": sha,
                "tex": "% " + sha + "\n",
                "html": (
                    '<!DOCTYPE html><html><head><meta charset="utf-8"></head>'
                    "<body>" + sha + "</body></html>"
                ),
                "csv": "value\n1\n",
            }
        )
    return {
        "semantic": semantic,
        "semantic_sha256": sha,
        "panels": panels,
        "manifest": {"status": "prepared", "reviewed": False, "published": False},
    }


def _all_prepared():
    return {
        figure_id: _prepared(figure_id, _writer.FIGURE_PANEL_COUNTS[figure_id])
        for figure_id in u.FIGURE_ORDER
    }


def _fake_builder(order, fail_on=None):
    def builder(prepared, *, output, private_logs, source_refs, style_config_path, check, deadline):
        figure_id = prepared["semantic"]["figure_id"]
        order.append(figure_id)
        if fail_on is not None and len(order) == fail_on:
            raise RuntimeError("boom-secret-path")
        os.makedirs(str(output), mode=0o700, exist_ok=True)
        os.makedirs(str(private_logs), mode=0o700, exist_ok=True)
        payload = _writer._canonical_json(
            {
                "status": "built_unreviewed",
                "figure_id": figure_id,
                "source_refs": dict(source_refs),
                "outputs": [],
            }
        )
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        with open(os.path.join(str(output), "manifest.json"), "wb") as handle:
            handle.write(payload)
        return {"figure_id": figure_id}

    return builder


def test_prepare_routes_all_six_in_order(monkeypatch):
    order = []
    calls = {}

    def record(name, result):
        def call(*args, **kwargs):
            order.append(name)
            calls[name] = (args, kwargs)
            return result

        return call

    models = {"models": True}
    monkeypatch.setattr(
        u.p08_model_figure_data, "prepare_model_figures", record("models", models)
    )
    monkeypatch.setattr(u.p08_f01_render, "prepare_f01", record("f01", "F01"))
    monkeypatch.setattr(u.p08_f02_render, "prepare_f02", record("f02", "F02"))
    monkeypatch.setattr(u.p08_f03_render, "prepare_f03", record("f03", "F03"))
    monkeypatch.setattr(u.p08_f04_render, "prepare_f04", record("f04", "F04"))
    monkeypatch.setattr(u.p08_f07_data, "prepare_f07", record("f07_data", "F07DATA"))
    monkeypatch.setattr(u.p08_f07_render, "prepare_f07_render", record("f07", "F07"))
    monkeypatch.setattr(
        u.p08_training_render, "prepare_training_render", record("training", "TRAIN")
    )

    checks = {"n": 0}

    def check():
        checks["n"] += 1
        return True

    spectral = object()
    frame = object()
    support = object()
    training_prepared = object()
    analysis = object()
    result = u.prepare_universal_figures(
        analysis=analysis,
        preservation={"spectral": spectral, "preservation": frame, "support": support},
        training_prepared=training_prepared,
        check=check,
    )

    assert order == ["models", "f01", "f02", "f03", "f04", "f07_data", "f07", "training"]
    assert calls["f01"][0] == (spectral,)
    assert calls["f02"][0] == (models,)
    assert calls["f03"][0] == (models,)
    assert calls["f04"][0] == (models,)
    assert calls["f07_data"][0] == (frame, models)
    assert calls["training"][0] == (training_prepared,)
    assert list(result) == list(u.FIGURE_ORDER)
    assert result["P08-F07"] == "F07"
    assert result[u.p08_training_render.FIGURE_ID] == "TRAIN"
    assert checks["n"] >= len(order)


def test_prepare_refuses_when_guard_false(monkeypatch):
    called = []
    monkeypatch.setattr(
        u.p08_model_figure_data,
        "prepare_model_figures",
        lambda analysis: called.append("models") or {},
    )
    monkeypatch.setattr(
        u.p08_f01_render, "prepare_f01", lambda spectral: called.append("f01") or {}
    )
    monkeypatch.setattr(
        u.p08_f07_data,
        "prepare_f07",
        lambda preservation, models: called.append("f07_data") or {},
    )

    state = {"n": 0}

    def check():
        state["n"] += 1
        return state["n"] < 2

    with pytest.raises(u.UniversalFigureError):
        u.prepare_universal_figures(
            analysis=object(),
            preservation={"spectral": 1, "preservation": 2, "support": 3},
            training_prepared=object(),
            check=check,
        )
    assert called == ["models"]


def test_prepare_requires_callable_check_and_exact_preservation():
    with pytest.raises(u.UniversalFigureError):
        u.prepare_universal_figures(
            analysis=object(),
            preservation={"spectral": 1, "preservation": 2, "support": 3},
            training_prepared=object(),
            check=None,
        )
    with pytest.raises(u.UniversalFigureError):
        u.prepare_universal_figures(
            analysis=object(),
            preservation={"spectral": 1, "preservation": 2},
            training_prepared=object(),
            check=lambda: True,
        )


def test_build_requires_exact_six_before_outputs(tmp_path):
    base = str(tmp_path)
    style = _style_file(base)
    out = os.path.join(base, "out")
    logs = os.path.join(base, "logs")

    missing = _all_prepared()
    del missing["P08-F04"]
    with pytest.raises(u.UniversalFigureError):
        u.build_universal_figures(
            missing,
            output=out,
            private_logs=logs,
            source_refs={"input": "a" * 64},
            style_config_path=style,
            check=lambda: True,
            deadline=1e12,
        )

    phantom = _all_prepared()
    phantom["P08-F99"] = phantom["P08-F01"]
    with pytest.raises(u.UniversalFigureError):
        u.build_universal_figures(
            phantom,
            output=out,
            private_logs=logs,
            source_refs={"input": "a" * 64},
            style_config_path=style,
            check=lambda: True,
            deadline=1e12,
        )

    assert not os.path.exists(out)
    assert not os.path.exists(logs)


def test_build_writes_roots_index_and_bound_manifest(monkeypatch, tmp_path):
    base = str(tmp_path)
    style = _style_file(base)
    prepared = _all_prepared()
    out = os.path.join(base, "out")
    logs = os.path.join(base, "logs")
    order = []
    monkeypatch.setattr(_writer, "_in_git_worktree", lambda path: False)
    monkeypatch.setattr(_writer, "build_figure_bundle", _fake_builder(order))
    refs = {"input": "a" * 64, "spec": "b" * 64}

    manifest = u.build_universal_figures(
        prepared,
        output=out,
        private_logs=logs,
        source_refs=refs,
        style_config_path=style,
        check=lambda: True,
        deadline=1e12,
    )

    assert order == list(u.FIGURE_ORDER)
    assert manifest["status"] == "built_unreviewed"
    assert manifest["reviewed"] is False
    assert manifest["published"] is False
    assert manifest["disclosure_reviewed"] is False
    assert manifest["visual_reviewed"] is False
    assert manifest["external_authentication_verified"] is False
    assert manifest["source_refs"] == refs
    assert manifest["figure_count"] == 6
    assert manifest["total_panels"] == 117
    assert set(manifest["figures"][0]) == {
        "figure_id",
        "panel_count",
        "semantic_sha256",
        "manifest_sha256",
    }
    with open(style, "rb") as handle:
        expected_style_sha = hashlib.sha256(handle.read()).hexdigest()
    assert manifest["style_config_sha256"] == expected_style_sha

    for entry in manifest["figures"]:
        figure_manifest = os.path.join(out, entry["figure_id"], "manifest.json")
        with open(figure_manifest, "rb") as handle:
            assert entry["manifest_sha256"] == hashlib.sha256(handle.read()).hexdigest()
        assert entry["semantic_sha256"] == prepared[entry["figure_id"]]["semantic_sha256"]
        assert entry["panel_count"] == _writer.FIGURE_PANEL_COUNTS[entry["figure_id"]]

    with open(os.path.join(out, "index.html"), encoding="utf-8") as handle:
        index = handle.read()
    for figure_id in u.FIGURE_ORDER:
        assert figure_id + "/built_index.html" in index
    assert "unreviewed" in index.lower()
    assert "not published" in index.lower()
    assert manifest["index_sha256"] == hashlib.sha256(index.encode("utf-8")).hexdigest()

    blob = json.dumps(manifest)
    assert "logs" not in blob
    assert base not in blob


def test_build_partial_failure_preserves_roots(monkeypatch, tmp_path):
    base = str(tmp_path)
    style = _style_file(base)
    prepared = _all_prepared()
    out = os.path.join(base, "out")
    logs = os.path.join(base, "logs")
    order = []
    monkeypatch.setattr(_writer, "_in_git_worktree", lambda path: False)
    monkeypatch.setattr(_writer, "build_figure_bundle", _fake_builder(order, fail_on=3))

    with pytest.raises(RuntimeError):
        u.build_universal_figures(
            prepared,
            output=out,
            private_logs=logs,
            source_refs={"input": "a" * 64},
            style_config_path=style,
            check=lambda: True,
            deadline=1e12,
        )

    assert order == list(u.FIGURE_ORDER[:3])
    assert os.path.isfile(os.path.join(out, "failure.json"))
    assert not os.path.isfile(os.path.join(out, "manifest.json"))
    for figure_id in u.FIGURE_ORDER[:2]:
        assert os.path.isfile(os.path.join(out, figure_id, "manifest.json"))
    with open(os.path.join(out, "failure.json"), "rb") as handle:
        failure = handle.read()
    assert failure == u._FAILURE_BYTES
    assert b"boom-secret-path" not in failure


def test_build_refuses_existing_root_before_child_writes(monkeypatch, tmp_path):
    base = str(tmp_path)
    style = _style_file(base)
    prepared = _all_prepared()
    out = os.path.join(base, "out")
    logs = os.path.join(base, "logs")
    os.makedirs(out, mode=0o700)
    calls = []
    monkeypatch.setattr(
        _writer, "build_figure_bundle", lambda *args, **kwargs: calls.append("built")
    )

    with pytest.raises(_writer.FigureBuildError):
        u.build_universal_figures(
            prepared,
            output=out,
            private_logs=logs,
            source_refs={"input": "a" * 64},
            style_config_path=style,
            check=lambda: True,
            deadline=1e12,
        )

    assert calls == []
    assert os.listdir(out) == []
    assert not os.path.exists(logs)
