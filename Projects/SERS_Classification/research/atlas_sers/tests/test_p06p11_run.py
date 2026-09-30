import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p06p11_run

_ACTUAL = p06p11_run._load_module("p05_comparison")


def _fake_comparison(frames=None):
    return SimpleNamespace(
        ALL_MODELS=_ACTUAL.ALL_MODELS,
        PAIRS=_ACTUAL.PAIRS,
        compare_predictions=lambda **kwargs: frames if frames is not None else {},
    )


def _loader(mapping):
    def load(name):
        if name not in mapping:
            raise AssertionError(f"unexpected module {name}")
        return mapping[name]

    return load


def _fake_analyze(calls):
    def analyze_panel(
        panel, paired_metrics, *, draws, master_seed, instrument_seed, hierarchy_seed, check=None
    ):
        if check is not None:
            check()
        calls["draws"] = draws
        calls["seeds"] = (master_seed, instrument_seed, hierarchy_seed)
        frame = pd.DataFrame({"metric": [1.0, 2.0], "value": [0.1, 0.2]})
        return {
            "tables": {"summary": frame.copy(), "intervals": frame.copy()},
            "arrays": {"c000_M01.crossed.scores": [0.1, 0.2]},
            "registry": {"arrays": ["c000_M01.crossed.scores"]},
        }

    return analyze_panel


def _require_numpy():
    return pytest.importorskip("numpy")


def test_synthetic_panel_shapes(monkeypatch):
    n_models = len(_ACTUAL.ALL_MODELS)
    monkeypatch.setattr(p06p11_run, "_load_module", _loader({"p05_comparison": _fake_comparison()}))
    panel = p06p11_run.synthetic_panel()
    m01, m06, coverage = panel["M01"], panel["M06"], panel["coverage"]
    assert m01["model_id"].nunique() == n_models
    assert m01["context_id"].nunique() == 4
    assert m01["master_sample_id"].nunique() == 6
    assert m01["instrument"].nunique() == 2
    assert len(m01) == n_models * 4 * 3
    assert m01.equals(m06)
    assert len(coverage) == n_models * 4
    assert set(coverage["complete"]) == {True}
    assert set(m01["class_vocabulary"]) == {("A", "B", "C")}
    pool = m01.groupby(["domain", "true_label"])["master_sample_id"].nunique()
    assert set(pool) == {2}


def test_synthetic_paired_metrics(monkeypatch):
    monkeypatch.setattr(p06p11_run, "_load_module", _loader({"p05_comparison": _fake_comparison()}))
    panel = p06p11_run.synthetic_panel()
    paired = p06p11_run.synthetic_paired_metrics(panel)
    required = {
        "model_id",
        "reference_model_id",
        "aggregation_id",
        "context_id",
        "domain",
        "station",
        "held_instrument",
        "common_complete",
        "model_balanced_accuracy",
        "reference_balanced_accuracy",
        "delta_balanced_accuracy",
    }
    assert required <= set(paired.columns)
    assert len(paired) == len(_ACTUAL.PAIRS) * 2 * 4
    assert paired["common_complete"].all()


def test_probe_run_writes_outputs(monkeypatch, tmp_path):
    _require_numpy()
    calls = {}
    mapping = {
        "p05_comparison": _fake_comparison(),
        "p06p11_analysis": SimpleNamespace(analyze_panel=_fake_analyze(calls)),
    }
    monkeypatch.setattr(p06p11_run, "_load_module", _loader(mapping))
    out = tmp_path / "probe"
    receipt = p06p11_run.run_analysis(output=out, mode="probe")
    assert receipt["status"] == "success"
    assert receipt["draws"] == 100
    assert receipt["seeds"] == {
        "master_seed": 2026092904,
        "instrument_seed": 2026092904,
        "hierarchy_seed": 2026092904,
    }
    assert calls["draws"] == 100
    assert calls["seeds"] == (2026092904, 2026092904, 2026092904)
    assert (out / "start.json").is_file()
    assert (out / "arrays.npz").is_file()
    assert (out / "registry.json").is_file()
    assert (out / "tables" / "summary.csv").is_file()
    assert (out / "receipt.json").is_file()
    assert not (out / "failure.json").exists()
    with np.load(out / "arrays.npz") as data:
        assert set(data.files) == {"c000_M01.crossed.scores"}
    listed = json.loads((out / "receipt.json").read_text())
    assert "receipt.json" not in listed["written_files"]
    assert listed["publication"] == "not_claimed"
    assert listed["peak_rss_bytes"] > 0


def test_full_run_writes_private_outputs(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(p06p11_run, "EXPECTED_MASTERS", 6)
    monkeypatch.setattr(p06p11_run, "EXPECTED_INSTRUMENTS", 2)
    calls = {}
    frames = {
        name: pd.DataFrame({"x": [1.0]})
        for name in ("endpoint_metrics", "paired_metrics", "coverage", "summary")
    }
    inputs = {
        **frames,
        "p05_ensemble": object(),
        "p04_ensemble": object(),
        "p03_predictions": object(),
        "contexts": object(),
        "hashes": {"protocol": "abc"},
    }
    mapping = {
        "p05_comparison": _fake_comparison(frames=frames),
        "p06p11_inputs": SimpleNamespace(
            load_inputs=lambda root: inputs,
            verify_inputs=lambda root: inputs["hashes"],
        ),
        "p06p11_predictions": SimpleNamespace(
            prepare_panel=lambda **kwargs: p06p11_run.synthetic_panel(),
            audit_point_estimates=lambda panel, frozen: pd.DataFrame({"absolute_error": [0.0]}),
        ),
        "p06p11_analysis": SimpleNamespace(analyze_panel=_fake_analyze(calls)),
        "p06p11_diagnostics": SimpleNamespace(
            g4_checklist=lambda *args, **kwargs: {
                "criteria": pd.DataFrame({"criterion": ["m1"]}),
                "decision": {"decision": "supported"},
            },
        ),
    }
    monkeypatch.setattr(p06p11_run, "_load_module", _loader(mapping))
    out = tmp_path / "full"
    receipt = p06p11_run.run_analysis(output=out, mode="full", artifact_root=tmp_path / "art")
    assert receipt["status"] == "success"
    assert receipt["draws"] == 10000
    assert receipt["input_hashes"] == {"protocol": "abc"}
    assert receipt["global_counts"] == {"masters": 6, "instruments": 2}
    assert (out / "start.json").is_file()
    assert (out / "panel_M01.parquet").is_file()
    assert (out / "panel_M06.parquet").is_file()
    assert (out / "point_audit.csv").is_file()
    assert (out / "g4_decision.json").is_file()
    assert json.loads((out / "g4_decision.json").read_text()) == {"decision": "supported"}


def test_existing_output_refused(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(p06p11_run.RunError) as exc:
        p06p11_run.run_analysis(output=out, mode="probe")
    assert exc.value.code == "output_exists"


def test_invalid_mode(tmp_path):
    with pytest.raises(p06p11_run.RunError) as exc:
        p06p11_run.run_analysis(output=tmp_path / "x", mode="quick")
    assert exc.value.code == "invalid_mode"


def test_full_requires_artifact_root(tmp_path):
    with pytest.raises(p06p11_run.RunError) as exc:
        p06p11_run.run_analysis(output=tmp_path / "run", mode="full")
    assert exc.value.code == "artifact_root_required"


def test_output_inside_project_rejected(tmp_path):
    with pytest.raises(p06p11_run.RunError) as exc:
        p06p11_run.run_analysis(output=p06p11_run.PROJECT_ROOT / "run", mode="probe")
    assert exc.value.code == "output_inside_project"


def test_symlink_ancestor_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(p06p11_run.RunError) as exc:
        p06p11_run.run_analysis(output=link / "run", mode="probe")
    assert exc.value.code == "output_symlink_rejected"


def test_protocol_mismatch_rejected_before_analysis(monkeypatch, tmp_path):
    monkeypatch.setattr(p06p11_run, "PROTOCOL_SHA256", "0" * 64)
    used = {"called": False}

    def load(name):
        used["called"] = True
        return _fake_comparison()

    monkeypatch.setattr(p06p11_run, "_load_module", load)
    out = tmp_path / "run"
    with pytest.raises(p06p11_run.RunError) as exc:
        p06p11_run.run_analysis(output=out, mode="probe")
    assert exc.value.code == "protocol_hash_mismatch"
    assert used["called"] is False
    assert not out.exists()


def test_failed_analysis_preserves_artifact(monkeypatch, tmp_path):
    mapping = {
        "p05_comparison": _fake_comparison(),
        "p06p11_analysis": SimpleNamespace(
            analyze_panel=lambda *args, **kwargs: (_ for _ in ()).throw(
                p06p11_run.RunError("wall_limit_exceeded")
            ),
        ),
    }
    monkeypatch.setattr(p06p11_run, "_load_module", _loader(mapping))
    out = tmp_path / "run"
    with pytest.raises(p06p11_run.RunError):
        p06p11_run.run_analysis(output=out, mode="probe")
    failure = json.loads((out / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["reason"] == "wall_limit_exceeded"
    assert failure["exception_class"] == "RunError"


def test_unknown_exception_maps_to_analysis_failed(monkeypatch, tmp_path):
    mapping = {
        "p05_comparison": _fake_comparison(),
        "p06p11_analysis": SimpleNamespace(
            analyze_panel=lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")),
        ),
    }
    monkeypatch.setattr(p06p11_run, "_load_module", _loader(mapping))
    out = tmp_path / "run"
    with pytest.raises(ValueError):
        p06p11_run.run_analysis(output=out, mode="probe")
    failure = json.loads((out / "failure.json").read_text())
    assert failure["reason"] == "analysis_failed"
    assert failure["exception_class"] == "ValueError"


def test_source_change_detected(monkeypatch, tmp_path):
    _require_numpy()
    snapshots = {"count": 0}

    def flaky_snapshot():
        snapshots["count"] += 1
        return {"frame": "1" if snapshots["count"] == 1 else "2"}

    monkeypatch.setattr(p06p11_run, "_snapshot_code", flaky_snapshot)
    mapping = {
        "p05_comparison": _fake_comparison(),
        "p06p11_analysis": SimpleNamespace(analyze_panel=_fake_analyze({})),
    }
    monkeypatch.setattr(p06p11_run, "_load_module", _loader(mapping))
    with pytest.raises(p06p11_run.RunError) as exc:
        p06p11_run.run_analysis(output=tmp_path / "run", mode="probe")
    assert exc.value.code == "source_changed_during_run"


def test_full_input_preservation_detected(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(p06p11_run, "EXPECTED_MASTERS", 6)
    monkeypatch.setattr(p06p11_run, "EXPECTED_INSTRUMENTS", 2)
    frames = {
        name: pd.DataFrame({"x": [1.0]})
        for name in ("endpoint_metrics", "paired_metrics", "coverage", "summary")
    }
    inputs = {
        **frames,
        "p05_ensemble": object(),
        "p04_ensemble": object(),
        "p03_predictions": object(),
        "contexts": object(),
        "hashes": {"protocol": "abc"},
    }
    mapping = {
        "p05_comparison": _fake_comparison(frames=frames),
        "p06p11_inputs": SimpleNamespace(
            load_inputs=lambda root: inputs,
            verify_inputs=lambda root: {"protocol": "changed"},
        ),
        "p06p11_predictions": SimpleNamespace(
            prepare_panel=lambda **kwargs: p06p11_run.synthetic_panel(),
            audit_point_estimates=lambda panel, frozen: pd.DataFrame({"absolute_error": [0.0]}),
        ),
        "p06p11_analysis": SimpleNamespace(analyze_panel=_fake_analyze({})),
        "p06p11_diagnostics": SimpleNamespace(
            g4_checklist=lambda *args, **kwargs: {
                "criteria": pd.DataFrame({"criterion": ["m1"]}),
                "decision": {"decision": "supported"},
            },
        ),
    }
    monkeypatch.setattr(p06p11_run, "_load_module", _loader(mapping))
    with pytest.raises(p06p11_run.RunError) as exc:
        p06p11_run.run_analysis(
            output=tmp_path / "full", mode="full", artifact_root=tmp_path / "art"
        )
    assert exc.value.code == "input_hash_mismatch"
    assert not (tmp_path / "full" / "g4_decision.json").exists()


def test_probe_real_integration(tmp_path):
    out = tmp_path / "probe_real"
    receipt = p06p11_run.run_analysis(output=out, mode="probe")
    assert receipt["status"] == "success"
    assert receipt["draws"] == 100
    assert receipt["seeds"] == {
        "master_seed": 2026092904,
        "instrument_seed": 2026092904,
        "hierarchy_seed": 2026092904,
    }
    assert receipt["peak_rss_bytes"] > 0
    assert (out / "start.json").is_file()
    assert receipt["global_counts"] == {"masters": 6, "instruments": 2}
    intervals = pd.read_csv(out / "tables" / "intervals.csv")
    assert len(intervals) == 136
    feasibility = pd.read_csv(out / "tables" / "feasibility.csv")
    assert len(feasibility) == 34

    keys = ["model_id", "reference_model_id", "aggregation_id"]
    pointwise = intervals[intervals["method"] != "hierarchical"]
    assert set(pointwise["method"]) == {
        "crossed_weight",
        "master_weight",
        "instrument_weight",
    }
    assert np.isfinite(pointwise["lower"]).all()
    assert np.isfinite(pointwise["upper"]).all()
    assert (pointwise["defined_draws"] == 100).all()

    hierarchical = intervals[intervals["method"] == "hierarchical"]
    undefined = hierarchical[hierarchical["undefined_draws"] > 0]
    assert undefined["lower"].isna().all()
    assert undefined["upper"].isna().all()
    assert (undefined["reason_code"] == "hierarchical_fixed_support_undefined").all()

    merged = feasibility.merge(
        hierarchical[keys + ["undefined_draws"]], on=keys, suffixes=("_feas", "_hier")
    )
    assert len(merged) == len(feasibility)
    assert (merged["undefined_draws_feas"] == merged["undefined_draws_hier"]).all()

    with np.load(out / "arrays.npz") as data:
        files = set(data.files)
    assert files
    registry = json.loads((out / "registry.json").read_text())
    assert isinstance(registry, dict)
    declared = [registry["master_weight_array"], registry["instrument_weight_array"]]
    for entry in registry["contrasts"].values():
        declared.append(entry["hierarchical_scores"])
        declared.append(entry["hierarchical_sampled_domains"])
        declared.extend(entry["method_scores"].values())
    for key in declared:
        assert key in files, f"declared array key {key} missing from NPZ"


def test_cli_invalid_mode_exit_code():
    with pytest.raises(SystemExit) as exc:
        p06p11_run.main(["--mode", "quick", "--output", "x"])
    assert exc.value.code == 2
