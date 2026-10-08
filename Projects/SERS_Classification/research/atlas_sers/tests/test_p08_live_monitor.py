"""Focused synthetic tests for the P08-U1 bounded epoch monitor."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

import pytest

from atlas_sers.visualization.p08_live_monitor import EpochMonitor

JOB_ID = "9f" * 32
SEED = 20260805
SECRET = "OPERATOR-DOE-1234"


def monitor_at(root, stream=None, **overrides):
    if stream is None:
        stream = io.StringIO()
    params = {
        "job_id": JOB_ID,
        "model_id": "D3",
        "policy_id": "PP-U-SG",
        "seed": SEED,
        "stage": "source_fit",
        "validation_available": True,
        "epoch_budget": 50,
        "stream": stream,
    }
    params.update(overrides)
    return EpochMonitor(root, **params)


def record(epoch, *, validation=False, supcon=True, paired=True, **extra):
    rec = {
        "epoch": epoch,
        "chemical_ce": 0.4 / epoch,
        "total_loss": 0.9 / epoch,
        "supcon_enabled": supcon,
        "paired_enabled": paired,
        "supcon_loss": 0.2 / epoch,
        "paired_loss": 0.1 / epoch,
        "total_optimizer_steps": epoch * 4,
    }
    if validation:
        rec.update(
            {
                "train_nll": 0.7 / epoch,
                "validation_nll": 0.8 / epoch,
                "train_balanced_accuracy": min(0.99, 0.4 + 0.01 * epoch),
                "validation_balanced_accuracy": min(0.99, 0.3 + 0.01 * epoch),
                "best_epoch": epoch,
                "nonimproving_epochs": 0 if epoch == 1 else 1,
            }
        )
    rec.update(extra)
    return rec


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


class _BrokenStream:
    def write(self, _text):
        raise OSError("synthetic stream failure")

    def flush(self):
        raise OSError("synthetic stream failure")


def test_live_updates_after_each_epoch(tmp_path):
    stream = io.StringIO()
    mon = monitor_at(tmp_path / "run", stream=stream)
    mon(record(1, validation=True))
    root = tmp_path / "run"
    index = (root / "index.html").read_text(encoding="utf-8")
    assert "Status:</strong> running" in index
    assert "1 / 50" in index
    assert len(_lines(root / "epochs.jsonl")) == 1
    row = json.loads(_lines(root / "epochs.jsonl")[0])
    assert row["epoch"] == 1
    assert "epoch 1/50" in stream.getvalue()
    assert row["elapsed_seconds"] >= 0.0


def test_output_directory_is_exclusive(tmp_path):
    monitor_at(tmp_path / "run")
    with pytest.raises(FileExistsError):
        monitor_at(tmp_path / "run")


def test_symlink_output_target_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    with pytest.raises(FileExistsError):
        monitor_at(link)


def test_private_permissions(tmp_path):
    mon = monitor_at(tmp_path / "run")
    mon(record(1, validation=True))
    mon.finish("complete", stop_reason="epoch_limit")
    root = tmp_path / "run"
    assert (os.stat(root).st_mode & 0o777) == 0o700
    for name in (
        "epochs.jsonl",
        "index.html",
        "semantic.json",
        "learning_curves.tex",
    ):
        assert (os.stat(root / name).st_mode & 0o777) == 0o600


def test_monotonic_elapsed_clock(tmp_path):
    stream = io.StringIO()
    now = [100.0]
    mon = monitor_at(tmp_path / "run", stream=stream, clock=lambda: now[0])
    now[0] = 103.5
    mon(record(1, validation=True))
    assert "elapsed=3.500s" in stream.getvalue()
    index = (tmp_path / "run" / "index.html").read_text(encoding="utf-8")
    assert "3.500" in index


def test_refit_rejects_validation_metrics(tmp_path):
    mon = monitor_at(
        tmp_path / "run", validation_available=False, stage="final_refit"
    )
    with pytest.raises(ValueError):
        mon(record(1, validation=True))
    assert _lines(tmp_path / "run" / "epochs.jsonl") == []


def test_refit_accepts_none_validation_fields(tmp_path):
    mon = monitor_at(
        tmp_path / "run", validation_available=False, stage="final_refit"
    )
    rec = record(1, validation=False)
    rec["validation_nll"] = None
    mon(rec)
    assert len(_lines(tmp_path / "run" / "epochs.jsonl")) == 1


def test_refit_panel_and_note(tmp_path):
    mon = monitor_at(
        tmp_path / "run", validation_available=False, stage="final_refit"
    )
    mon(record(1, validation=False))
    index = (tmp_path / "run" / "index.html").read_text(encoding="utf-8")
    assert "No validation during fixed-duration refit" in index
    assert "Clean-evaluation" not in index


def test_validation_required_fields(tmp_path):
    mon = monitor_at(tmp_path / "run", validation_available=True)
    with pytest.raises(ValueError):
        mon(record(1, validation=False))
    assert _lines(tmp_path / "run" / "epochs.jsonl") == []


def test_finish_writes_semantic_tex_and_digest(tmp_path):
    mon = monitor_at(tmp_path / "run")
    mon(record(1, validation=True))
    mon.finish("complete", stop_reason="epoch_limit")
    root = tmp_path / "run"
    semantic_bytes = (root / "semantic.json").read_bytes()
    semantic = json.loads(semantic_bytes)
    assert semantic["status"] == "complete"
    assert semantic["stop_reason"] == "epoch_limit"
    assert len(semantic["rows"]) == 1
    digest = hashlib.sha256(semantic_bytes).hexdigest()
    assert digest in (root / "index.html").read_text(encoding="utf-8")
    assert digest in (root / "learning_curves.tex").read_text(
        encoding="utf-8"
    )


def test_html_and_tex_share_exact_coordinates(tmp_path):
    mon = monitor_at(tmp_path / "run", validation_available=False)
    rec = record(1, validation=False)
    rec["chemical_ce"] = 0.125
    rec["total_loss"] = 0.5
    mon(rec)
    mon.finish("complete", stop_reason="fixed_duration")
    html = (tmp_path / "run" / "index.html").read_text(encoding="utf-8")
    tex = (tmp_path / "run" / "learning_curves.tex").read_text(
        encoding="utf-8"
    )
    assert "epoch 1, 0.125" in html
    assert "(1,0.125)" in tex
    assert "epoch 1, 0.5" in html
    assert "(1,0.5)" in tex


def test_shared_axes_and_tick_labels(tmp_path):
    import atlas_sers.visualization.p08_live_monitor as module

    mon = monitor_at(tmp_path / "run", validation_available=False)
    mon(record(1, validation=False))
    mon(record(2, validation=False))
    mon.finish("complete", stop_reason="fixed_duration")
    root = tmp_path / "run"
    rows = json.loads(
        (root / "semantic.json").read_text(encoding="utf-8")
    )["rows"]
    series = module._training_series(rows)
    axes = module._panel_axes(rows, series)
    html = (root / "index.html").read_text(encoding="utf-8")
    tex = (root / "learning_curves.tex").read_text(encoding="utf-8")
    assert f"xmax={module._format_number(axes['x_max'])}" in tex
    assert f"ymax={module._format_number(axes['y_max'])}" in tex
    for tick in axes["y_ticks"]:
        label = module._format_tick(tick)
        assert label in html
        assert label in tex
    for epoch in axes["x_ticks"]:
        assert f">{epoch}<" in html
        assert str(epoch) in tex


def test_single_epoch_constant_values_render(tmp_path):
    mon = monitor_at(tmp_path / "run", validation_available=False)
    rec = record(1, validation=False, supcon=False, paired=False)
    rec["chemical_ce"] = 0.5
    rec["total_loss"] = 0.5
    mon(rec)
    mon.finish("complete", stop_reason="fixed_duration")
    root = tmp_path / "run"
    html = (root / "index.html").read_text(encoding="utf-8")
    tex = (root / "learning_curves.tex").read_text(encoding="utf-8")
    assert "epoch 1, 0.5" in html
    assert "(1,0.5)" in tex
    saved = json.loads((root / "semantic.json").read_text(encoding="utf-8"))
    assert saved["rows"][0]["elapsed_seconds"] >= 0.0
    assert saved["rows"][0]["supcon_loss"] is None


def test_ba_panel_uses_shared_unit_bounds(tmp_path):
    import atlas_sers.visualization.p08_live_monitor as module

    mon = monitor_at(tmp_path / "run")
    mon(record(1, validation=True))
    mon.finish("complete", stop_reason="epoch_limit")
    rows = json.loads(
        (tmp_path / "run" / "semantic.json").read_text(encoding="utf-8")
    )["rows"]
    ba = module._ba_series(rows)
    axes = module._panel_axes(rows, ba, y_bounds=(0.0, 1.0))
    assert axes["y_min"] == 0.0
    assert axes["y_max"] == 1.0
    tex = (tmp_path / "run" / "learning_curves.tex").read_text(
        encoding="utf-8"
    )
    assert "ymin=0, ymax=1" in tex


def test_live_output_identifies_job_and_links_data(tmp_path):
    stream = io.StringIO()
    mon = monitor_at(tmp_path / "run", stream=stream)
    mon(record(1, validation=True))
    index = (tmp_path / "run" / "index.html").read_text(encoding="utf-8")
    assert JOB_ID[:12] in index
    assert "D3" in index
    assert "PP-U-SG" in index
    assert str(SEED) in index
    assert "href='epochs.jsonl'" in index
    console = stream.getvalue()
    assert JOB_ID[:12] in console
    assert "D3" in console
    assert "PP-U-SG" in console
    assert f"seed={SEED}" in console
    mon.finish("complete", stop_reason="epoch_limit")
    final = (tmp_path / "run" / "index.html").read_text(encoding="utf-8")
    assert "href='semantic.json'" in final


def test_failed_and_closed_unfinished(tmp_path):
    mon_a = monitor_at(tmp_path / "a")
    mon_a(record(1, validation=True))
    mon_a.finish("failed", stop_reason="error")
    semantic_a = json.loads(
        (tmp_path / "a" / "semantic.json").read_text(encoding="utf-8")
    )
    assert semantic_a["status"] == "failed"
    assert "failed" in (tmp_path / "a" / "index.html").read_text(
        encoding="utf-8"
    )

    mon_b = monitor_at(tmp_path / "b")
    mon_b(record(1, validation=True))
    mon_b.close()
    semantic_b = json.loads(
        (tmp_path / "b" / "semantic.json").read_text(encoding="utf-8")
    )
    assert semantic_b["status"] == "closed-unfinished"
    assert "closed unfinished" in (tmp_path / "b" / "index.html").read_text(
        encoding="utf-8"
    )


def test_empty_failed_fit(tmp_path):
    mon = monitor_at(tmp_path / "run")
    mon.finish("failed", stop_reason="error")
    root = tmp_path / "run"
    semantic = json.loads(
        (root / "semantic.json").read_text(encoding="utf-8")
    )
    assert semantic["rows"] == []
    assert "No epochs were recorded" in (root / "index.html").read_text(
        encoding="utf-8"
    )
    assert "No epochs" in (root / "learning_curves.tex").read_text(
        encoding="utf-8"
    )


def test_active_components_shown_and_disabled_hidden(tmp_path):
    active = monitor_at(tmp_path / "active")
    active(record(1, validation=True, supcon=True, paired=True))
    html = (tmp_path / "active" / "index.html").read_text(encoding="utf-8")
    assert "SupCon" in html
    assert "Matched-sample consistency" in html
    assert (
        "source-validation NLL" in html.lower()
        or "Source-validation NLL" in html
    )

    disabled = monitor_at(tmp_path / "disabled", validation_available=False)
    disabled(record(1, validation=False, supcon=False, paired=False))
    html_disabled = (tmp_path / "disabled" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "SupCon" not in html_disabled
    assert "Matched-sample consistency" not in html_disabled
    disabled.finish("complete", stop_reason="fixed_duration")
    rows = json.loads(
        (tmp_path / "disabled" / "semantic.json").read_text(encoding="utf-8")
    )["rows"]
    assert rows[0]["supcon_loss"] is None
    assert rows[0]["paired_loss"] is None


def test_record_is_not_mutated(tmp_path):
    rec = record(1, validation=True)
    snapshot = copy.deepcopy(rec)
    mon = monitor_at(tmp_path / "run")
    mon(rec)
    assert rec == snapshot


def test_rejects_bad_epochs_without_accepting(tmp_path):
    mon = monitor_at(tmp_path / "run", epoch_budget=3)
    good = record(1, validation=True)
    cases = []
    gap = dict(good)
    gap["epoch"] = 2
    cases.append(gap)
    nonfinite = dict(good)
    nonfinite["total_loss"] = float("nan")
    cases.append(nonfinite)
    boolean = dict(good)
    boolean["chemical_ce"] = True
    cases.append(boolean)
    missing = dict(good)
    del missing["total_loss"]
    cases.append(missing)
    bad_flag = dict(good)
    bad_flag["supcon_enabled"] = "yes"
    cases.append(bad_flag)
    bad_unit = dict(good)
    bad_unit["validation_balanced_accuracy"] = 1.5
    cases.append(bad_unit)
    bad_best = dict(good)
    bad_best["best_epoch"] = 5
    cases.append(bad_best)
    bad_nonimproving = dict(good)
    bad_nonimproving["nonimproving_epochs"] = -1
    cases.append(bad_nonimproving)

    for bad in cases:
        with pytest.raises(ValueError):
            mon(bad)
    assert _lines(tmp_path / "run" / "epochs.jsonl") == []

    mon(good)
    assert len(_lines(tmp_path / "run" / "epochs.jsonl")) == 1
    duplicate = dict(good)
    with pytest.raises(ValueError):
        mon(duplicate)
    assert len(_lines(tmp_path / "run" / "epochs.jsonl")) == 1


def test_epoch_budget_bounds(tmp_path):
    mon = monitor_at(tmp_path / "run", epoch_budget=2)
    mon(record(1, validation=True))
    mon(record(2, validation=True))
    with pytest.raises(ValueError):
        mon(record(3, validation=True))
    assert len(_lines(tmp_path / "run" / "epochs.jsonl")) == 2


def test_unknown_fields_and_exceptions_do_not_leak(tmp_path):
    stream = io.StringIO()
    mon = monitor_at(tmp_path / "run", stream=stream)
    mon(record(1, validation=True, operator=SECRET, instrument=SECRET))
    mon.finish("complete", stop_reason="epoch_limit")
    root = tmp_path / "run"
    for name in (
        "index.html",
        "epochs.jsonl",
        "semantic.json",
        "learning_curves.tex",
    ):
        text = (root / name).read_text(encoding="utf-8")
        assert SECRET not in text
    assert SECRET not in stream.getvalue()
    index = (root / "index.html").read_text(encoding="utf-8")
    assert JOB_ID[:12] in index
    assert JOB_ID not in index

    other = monitor_at(tmp_path / "other")
    bad = record(1, validation=True, operator=SECRET)
    bad["total_loss"] = float("inf")
    with pytest.raises(ValueError) as error:
        other(bad)
    assert SECRET not in str(error.value)


def test_offline_vector_html_and_native_tex(tmp_path):
    mon = monitor_at(tmp_path / "run")
    mon(record(1, validation=True))
    mon.finish("complete", stop_reason="epoch_limit")
    html = (tmp_path / "run" / "index.html").read_text(
        encoding="utf-8"
    ).lower()
    assert "<svg" in html
    assert "<img" not in html
    assert "http://" not in html
    assert "https://" not in html
    assert "data:" not in html
    assert ".png" not in html
    assert "xlink:href" not in html
    tex = (tmp_path / "run" / "learning_curves.tex").read_text(
        encoding="utf-8"
    )
    assert "includegraphics" not in tex


def test_meta_refresh_only_while_running(tmp_path):
    mon = monitor_at(tmp_path / "run")
    mon(record(1, validation=True))
    running = (tmp_path / "run" / "index.html").read_text(encoding="utf-8")
    assert "http-equiv='refresh'" in running
    mon.finish("complete", stop_reason="epoch_limit")
    final = (tmp_path / "run" / "index.html").read_text(encoding="utf-8")
    assert "http-equiv='refresh'" not in final


def test_filesystem_failure_propagates(tmp_path, monkeypatch):
    import atlas_sers.visualization.p08_live_monitor as module

    mon = monitor_at(tmp_path / "run")

    def boom(path, data):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(module, "_atomic_write_private", boom)
    with pytest.raises(OSError):
        mon(record(1, validation=True))
    assert mon.status == "failed"
    with pytest.raises(RuntimeError):
        mon(record(2, validation=True))


def test_finish_close_misuse(tmp_path):
    mon = monitor_at(tmp_path / "a")
    mon(record(1, validation=True))
    mon.finish("complete", stop_reason="epoch_limit")
    with pytest.raises(RuntimeError):
        mon.finish("complete", stop_reason="epoch_limit")
    with pytest.raises(RuntimeError):
        mon(record(2, validation=True))
    assert len(_lines(tmp_path / "a" / "epochs.jsonl")) == 1
    mon.close()

    mon_b = monitor_at(tmp_path / "b")
    mon_b.close()
    semantic = json.loads(
        (tmp_path / "b" / "semantic.json").read_text(encoding="utf-8")
    )
    assert semantic["rows"] == []
    with pytest.raises(RuntimeError):
        mon_b.finish("complete", stop_reason="epoch_limit")


@pytest.mark.parametrize(
    "overrides",
    [
        {"job_id": "A" * 64},
        {"job_id": "abc"},
        {"model_id": "D9"},
        {"policy_id": "PP-X"},
        {"seed": 123},
        {"seed": True},
        {"stage": "training"},
        {"epoch_budget": 0},
        {"epoch_budget": 201},
        {"epoch_budget": True},
        {"refresh_seconds": 0},
        {"validation_available": "yes"},
        {"stage": "final_refit", "validation_available": True},
    ],
)
def test_metadata_validation(tmp_path, overrides):
    with pytest.raises((ValueError, TypeError)):
        monitor_at(tmp_path / "run", **overrides)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"job_id": ["9f" * 32]},
        {"model_id": ["D1"]},
        {"policy_id": {"p": "PP-U-SG"}},
        {"stage": {"source_fit"}},
    ],
)
def test_metadata_unhashable_rejected_clearly(tmp_path, overrides):
    with pytest.raises(ValueError):
        monitor_at(tmp_path / "run", **overrides)
    assert not (tmp_path / "run").exists()


def test_finish_rejects_unhashable_status(tmp_path):
    mon = monitor_at(tmp_path / "run")
    with pytest.raises(ValueError):
        mon.finish(["complete"])
    assert mon.status == "running"
    with pytest.raises(ValueError):
        mon.finish("complete", stop_reason=["error"])
    assert mon.status == "running"


def test_accepts_p05_like_records(tmp_path):
    development = monitor_at(
        tmp_path / "dev", model_id="D3", validation_available=True
    )
    development(record(1, validation=True))
    development.finish("complete", stop_reason="epoch_limit")

    refit = monitor_at(
        tmp_path / "refit",
        model_id="D0-M",
        stage="final_refit",
        validation_available=False,
    )
    refit_record = record(1, validation=False, supcon=False, paired=False)
    refit(refit_record)
    refit.finish("complete", stop_reason="fixed_duration")


def test_initial_clock_must_be_finite(tmp_path):
    for value in (float("inf"), float("nan")):
        with pytest.raises(ValueError):
            monitor_at(tmp_path / "run", clock=lambda v=value: v)
        assert not (tmp_path / "run").exists()


def test_backwards_clock_rejected_before_accepting(tmp_path):
    now = [100.0]
    mon = monitor_at(tmp_path / "run", clock=lambda: now[0])
    now[0] = 99.0
    with pytest.raises(ValueError):
        mon(record(1, validation=True))
    assert _lines(tmp_path / "run" / "epochs.jsonl") == []
    assert mon.status == "running"
    assert mon.epochs_completed == 0
    now[0] = 101.0
    mon(record(1, validation=True))
    assert mon.epochs_completed == 1


def test_nonfinite_clock_observation_rejected(tmp_path):
    now = [1.0]
    mon = monitor_at(tmp_path / "run", clock=lambda: now[0])
    now[0] = float("nan")
    with pytest.raises(ValueError):
        mon(record(1, validation=True))
    assert _lines(tmp_path / "run" / "epochs.jsonl") == []


def test_final_elapsed_frozen_and_persisted(tmp_path):
    now = [10.0]
    mon = monitor_at(tmp_path / "run", clock=lambda: now[0])
    now[0] = 12.0
    mon(record(1, validation=True))
    now[0] = 40.0
    mon.finish("complete", stop_reason="epoch_limit")
    assert mon.elapsed_seconds == 30.0
    semantic = json.loads(
        (tmp_path / "run" / "semantic.json").read_text(encoding="utf-8")
    )
    assert semantic["elapsed_seconds"] == 30.0
    index = (tmp_path / "run" / "index.html").read_text(encoding="utf-8")
    assert "30.000" in index
    now[0] = 500.0
    assert mon.elapsed_seconds == 30.0
    final = (tmp_path / "run" / "index.html").read_text(encoding="utf-8")
    assert "30.000" in final
    assert "500" not in final


def test_stream_error_marks_failed(tmp_path):
    mon = monitor_at(tmp_path / "run", stream=_BrokenStream())
    with pytest.raises(OSError):
        mon(record(1, validation=True))
    assert mon.status == "failed"
    with pytest.raises(RuntimeError):
        mon(record(2, validation=True))


def test_fsync_failure_marks_failed(tmp_path, monkeypatch):
    import atlas_sers.visualization.p08_live_monitor as module

    mon = monitor_at(tmp_path / "run")

    def boom(_descriptor):
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(module.os, "fsync", boom)
    with pytest.raises(OSError):
        mon(record(1, validation=True))
    assert mon.status == "failed"
    with pytest.raises(RuntimeError):
        mon(record(2, validation=True))


def test_initialization_failure_closes_handle(tmp_path, monkeypatch):
    import atlas_sers.visualization.p08_live_monitor as module

    opened = {}
    real_open = module._open_append_private

    def tracking_open(path):
        handle = real_open(path)
        opened["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_append_private", tracking_open)

    def boom(path, data):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(module, "_atomic_write_private", boom)
    with pytest.raises(OSError):
        monitor_at(tmp_path / "run")
    assert opened["handle"].closed


def test_fchmod_failure_propagates(tmp_path, monkeypatch):
    import atlas_sers.visualization.p08_live_monitor as module

    def boom(_descriptor, _mode):
        raise OSError("synthetic chmod failure")

    monkeypatch.setattr(module.os, "fchmod", boom)
    with pytest.raises(OSError):
        monitor_at(tmp_path / "run")


def test_finish_complete_requires_epochs(tmp_path):
    mon = monitor_at(tmp_path / "run")
    with pytest.raises(ValueError):
        mon.finish("complete", stop_reason="epoch_limit")
    assert mon.status == "running"
    assert mon.epochs_completed == 0
    assert not (tmp_path / "run" / "semantic.json").exists()
    mon.finish("failed", stop_reason="error")
    assert mon.status == "failed"


def test_finish_failure_marks_failed(tmp_path, monkeypatch):
    import atlas_sers.visualization.p08_live_monitor as module

    mon = monitor_at(tmp_path / "run")
    mon(record(1, validation=True))

    def boom(path, data):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(module, "_atomic_write_private", boom)
    with pytest.raises(OSError):
        mon.finish("complete", stop_reason="epoch_limit")
    assert mon.status == "failed"


def test_invalid_input_does_not_mark_failed(tmp_path):
    mon = monitor_at(tmp_path / "run")
    mon(record(1, validation=True))
    bad = record(2, validation=True)
    del bad["total_loss"]
    with pytest.raises(ValueError):
        mon(bad)
    assert mon.status == "running"
    mon(record(2, validation=True))
    assert mon.epochs_completed == 2


def test_canonical_json_rejects_nan():
    import atlas_sers.visualization.p08_live_monitor as module

    with pytest.raises(ValueError):
        module._canonical_bytes({"value": float("nan")})


def test_require_finite_overflow_is_value_error():
    import atlas_sers.visualization.p08_live_monitor as module

    with pytest.raises(ValueError):
        module._require_finite(10**400, "value")


def test_tex_metadata_states_scope(tmp_path):
    mon = monitor_at(
        tmp_path / "run", validation_available=False, stage="final_refit"
    )
    mon(record(1, validation=False))
    mon.finish("complete", stop_reason="fixed_duration")
    tex = (tmp_path / "run" / "learning_curves.tex").read_text(
        encoding="utf-8"
    )
    for needle in (
        "% recipe=D3",
        "% policy=PP-U-SG",
        f"% seed={SEED}",
        "% stage=final_refit",
        "% status=complete",
        "% epoch=1/50",
        "% scope=no validation",
    ):
        assert needle in tex


def test_refit_caption_omits_best_checkpoints(tmp_path):
    mon = monitor_at(tmp_path / "run", validation_available=False)
    mon(record(1, validation=False))
    mon.finish("complete", stop_reason="fixed_duration")
    tex = (tmp_path / "run" / "learning_curves.tex").read_text(
        encoding="utf-8"
    )
    assert "best checkpoint" not in tex.lower()


def test_fresh_process_import_is_lightweight(tmp_path):
    import atlas_sers.visualization.p08_live_monitor as module

    module_path = os.path.abspath(module.__file__)
    src_root = os.path.dirname(os.path.dirname(os.path.dirname(module_path)))
    code = (
        "import sys\n"
        "import atlas_sers.visualization.p08_live_monitor\n"
        "banned = {'numpy', 'pandas', 'sklearn', 'torch'}\n"
        "loaded = sorted(name for name in banned if name in sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_root + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout.decode("utf-8", "replace")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_tex_compiles_optional(tmp_path):
    mon = monitor_at(tmp_path / "run")
    mon(record(1, validation=True))
    mon(record(2, validation=True))
    mon.finish("complete", stop_reason="epoch_limit")
    result = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "learning_curves.tex",
        ],
        cwd=tmp_path / "run",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout.decode("utf-8", "replace")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_tex_compiles_refit_optional(tmp_path):
    mon = monitor_at(
        tmp_path / "run",
        model_id="D0-M",
        stage="final_refit",
        validation_available=False,
    )
    mon(record(1, validation=False, supcon=False, paired=False))
    mon.finish("complete", stop_reason="fixed_duration")
    result = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "learning_curves.tex",
        ],
        cwd=tmp_path / "run",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout.decode("utf-8", "replace")


def test_positive_time_reversal_rejected(tmp_path):
    now = [100.0]
    mon = monitor_at(tmp_path / "run", clock=lambda: now[0])
    now[0] = 105.0
    mon(record(1, validation=True))
    assert mon.epochs_completed == 1
    now[0] = 104.0
    with pytest.raises(ValueError):
        mon(record(2, validation=True))
    assert mon.epochs_completed == 1
    assert mon.status == "running"
    assert len(_lines(tmp_path / "run" / "epochs.jsonl")) == 1
    now[0] = 106.0
    mon(record(2, validation=True))
    assert mon.epochs_completed == 2


def test_finish_clock_regression_rejected(tmp_path):
    now = [10.0]
    mon = monitor_at(tmp_path / "run", clock=lambda: now[0])
    now[0] = 20.0
    mon(record(1, validation=True))
    now[0] = 15.0
    with pytest.raises(ValueError):
        mon.finish("complete", stop_reason="epoch_limit")
    assert mon.status == "failed"


def test_bad_clock_at_finish_fails_and_closes_handle(tmp_path, monkeypatch):
    import atlas_sers.visualization.p08_live_monitor as module

    opened = {}
    real_open = module._open_append_private

    def tracking_open(path):
        handle = real_open(path)
        opened["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_append_private", tracking_open)
    now = [10.0]
    mon = monitor_at(tmp_path / "run", clock=lambda: now[0])
    now[0] = 12.0
    mon(record(1, validation=True))
    now[0] = float("nan")
    with pytest.raises(ValueError):
        mon.finish("complete", stop_reason="epoch_limit")
    assert mon.status == "failed"
    assert opened["handle"].closed
    assert not (tmp_path / "run" / "semantic.json").exists()
    with pytest.raises(RuntimeError):
        mon.finish("failed", stop_reason="error")


def test_stream_error_closes_handle(tmp_path, monkeypatch):
    import atlas_sers.visualization.p08_live_monitor as module

    opened = {}
    real_open = module._open_append_private

    def tracking_open(path):
        handle = real_open(path)
        opened["handle"] = handle
        return handle

    monkeypatch.setattr(module, "_open_append_private", tracking_open)
    mon = monitor_at(tmp_path / "run", stream=_BrokenStream())
    with pytest.raises(OSError):
        mon(record(1, validation=True))
    assert mon.status == "failed"
    assert opened["handle"].closed


def test_body_exception_preserved_when_cleanup_fails(tmp_path, monkeypatch):
    import atlas_sers.visualization.p08_live_monitor as module

    mon = monitor_at(tmp_path / "run")
    mon(record(1, validation=True))

    def boom(path, data):
        raise OSError("synthetic cleanup failure")

    monkeypatch.setattr(module, "_atomic_write_private", boom)

    class BodyError(Exception):
        pass

    with pytest.raises(BodyError):
        with mon:
            raise BodyError("body failure")


def test_tex_metadata_visible(tmp_path):
    mon = monitor_at(
        tmp_path / "run", validation_available=False, stage="final_refit"
    )
    mon(record(1, validation=False))
    mon.finish("complete", stop_reason="fixed_duration")
    tex = (tmp_path / "run" / "learning_curves.tex").read_text(
        encoding="utf-8"
    )
    visible = "\n".join(
        line
        for line in tex.splitlines()
        if line.strip() and not line.lstrip().startswith("%")
    )
    for needle in (
        "D3 both",
        "PP-U-SG Savitzky-Golay",
        f"seed: {SEED}",
        "final refit",
        "complete",
        "epochs: 1/50",
        "job token:",
        "elapsed:",
    ):
        assert needle in visible, needle
    assert JOB_ID not in visible
    assert JOB_ID[:12] in visible
    assert "best checkpoint" not in visible.lower()


def test_group_style_single_brace_and_outer_legend(tmp_path):
    mon = monitor_at(tmp_path / "run")
    mon(record(1, validation=True))
    mon.finish("complete", stop_reason="epoch_limit")
    tex = (tmp_path / "run" / "learning_curves.tex").read_text(
        encoding="utf-8"
    )
    assert "vertical sep=1.5cm}" in tex
    assert "vertical sep=1.5cm}}" not in tex
    assert "legend pos=outer north east" in tex
