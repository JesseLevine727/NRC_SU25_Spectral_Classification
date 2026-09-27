"""Synthetic tests for the P05 comprehensive reporting stage."""

from __future__ import annotations

# ruff: noqa: E402
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from atlas_sers.evaluation import p05_comprehensive_reporting as mod
from atlas_sers.evaluation import p05_pilot as real_pilot
from atlas_sers.evaluation import p05_public_metrics, p05_reporting_inputs, p05_source_diagnostics
from atlas_sers.evaluation.p05_comprehensive_storage import P05StorageError
from tests import test_p05_diagnostic_figures as diag_fixtures

_REAL_IMPORT_RUNTIME = mod._import_runtime

PUBLIC_NAMES = p05_public_metrics.P05_PUBLIC_TABLE_NAMES
SOURCE_NAMES = p05_source_diagnostics.P05_SOURCE_DIAGNOSTIC_TABLE_NAMES
RELIABILITY_NAMES = ("reliability_bins", "reliability_summary")
COST_KEYS = ("alpha", "beta", "gamma")
REQUIRED_COST_KEYS = ("alpha", "beta")
BINDINGS = {key: "a" * 64 for key in p05_reporting_inputs.BINDING_KEYS}
COSTS = {"alpha": 1, "beta": 2.5, "gamma": 0}
STAGE_COSTS = {key: 1 for key in p05_reporting_inputs.REQUIRED_PUBLIC_COST_KEYS}


def _tiny() -> pd.DataFrame:
    return pd.DataFrame({"metric": ["a", "b"], "value": [1.0, 2.0]})


def _fixture_frames():
    provided = diag_fixtures._inputs()
    public = {"paired_domains": provided["public_metrics"]["paired_domains"]}
    for name in PUBLIC_NAMES:
        public.setdefault(name, _tiny())
    source = dict(provided["source_diagnostics"])
    for name in SOURCE_NAMES:
        source.setdefault(name, _tiny())
    reliability = dict(provided["reliability"])
    for name in RELIABILITY_NAMES:
        reliability.setdefault(name, _tiny())
    return public, source, reliability


def _load_sources(bundle, *, authenticated, deadline):
    return {
        "bindings": dict(BINDINGS),
        "selector_records": [],
        "public_costs": dict(STAGE_COSTS),
    }


def _verify_sources(bundle, *, bindings, deadline):
    assert dict(bindings) == BINDINGS


def _stub_compile(tex_path, pdf_path, png_path, log_path=None, **kwargs):
    assert isinstance(kwargs.get("deadline"), float)
    Path(pdf_path).write_bytes(b"%PDF-1.4\n%stub\n")
    Path(png_path).write_bytes(b"\x89PNG\r\n\x1a\n")


def _setup(mp, base):
    base.mkdir(parents=True, exist_ok=True)
    artifact_root = base / "artifact"
    artifact_root.mkdir(exist_ok=True)
    project_root = base / "project"
    project_root.mkdir(exist_ok=True)
    repository_root = base / "repo"
    repository_root.mkdir(exist_ok=True)
    contract_path = base / "contract.json"
    contract_path.write_text("{}", encoding="utf-8")
    permit_path = base / "permit.json"
    permit_path.write_text("{}", encoding="utf-8")

    permit_id = "b" * 64
    run_root = artifact_root / mod.COMPREHENSIVE_DIR / mod.RUNS_DIR / permit_id
    comparison_stage = run_root / mod.COMPARISON_STAGE_NAME
    comparison_stage.mkdir(parents=True)
    comparison_manifest = comparison_stage / mod.COMPARISON_MANIFEST_NAME
    mod.core._write_manifest(comparison_stage)
    manifest_sha = hashlib.sha256(comparison_manifest.read_bytes()).hexdigest()

    prior = float(mod.development.PRELAUNCH_AUDIT_RESERVE_SECONDS)
    comparison_receipt = {mod.PRIOR_FIELD: prior, "stage_manifest_sha256": manifest_sha}
    (run_root / mod.COMPARISON_RECEIPT_NAME).write_text(
        json.dumps(comparison_receipt), encoding="utf-8"
    )

    bundle = {
        "permit_sha256": permit_id,
        "artifact_root": str(artifact_root),
        "contract_sha256": "c" * 64,
        "core_plan_id": "core-plan",
        "ledger": {"ledger_id": "ledger-1"},
        "repository_root": str(repository_root),
        "project_root": str(project_root),
        "contract": {},
        "support": {},
        "slots": [],
    }
    public, source, reliability = _fixture_frames()

    def _prepare(
        project_root, artifact_root, contract_path, permit_path, *, require_unstarted=False
    ):
        return bundle

    def _authenticate(bundle, *, deadline):
        return {
            "prior_seconds": prior,
            "comparison_receipt": dict(comparison_receipt),
            "plan": {"plan_id": "plan-1"},
            "comparison_tables": {},
            "aggregation_tables": {"ensemble_predictions": pd.DataFrame({"value": [1.0]})},
            "source_optimizer_steps": 0,
            "refit_optimizer_steps": 0,
        }

    def _post_run_reauth(*args, **kwargs):
        return {"captured": True, "reauthed": True}

    authority = SimpleNamespace(authenticate_comparison=_authenticate)
    pilot = SimpleNamespace(
        _post_run_reauth=_post_run_reauth,
        _verify_manifest=real_pilot._verify_manifest,
    )
    reporting_inputs = SimpleNamespace(
        PUBLIC_COST_KEYS=p05_reporting_inputs.PUBLIC_COST_KEYS,
        REQUIRED_PUBLIC_COST_KEYS=p05_reporting_inputs.REQUIRED_PUBLIC_COST_KEYS,
        RECOVERY_PUBLIC_COST_KEYS=p05_reporting_inputs.RECOVERY_PUBLIC_COST_KEYS,
        load_reporting_sources=_load_sources,
        verify_reporting_sources=_verify_sources,
    )
    public_metrics = SimpleNamespace(
        P05_PUBLIC_TABLE_NAMES=PUBLIC_NAMES,
        build_public_metrics=lambda **kwargs: dict(public),
    )
    source_diagnostics = SimpleNamespace(
        P05_SOURCE_DIAGNOSTIC_TABLE_NAMES=SOURCE_NAMES,
        build_source_diagnostics=lambda **kwargs: dict(source),
    )
    reliability_metrics = SimpleNamespace(build_reliability=lambda **kwargs: dict(reliability))

    from atlas_sers.visualization import p05_benchmark_figures as benchmark_figures
    from atlas_sers.visualization import p05_diagnostic_figures as diagnostic_figures

    mp.setattr(benchmark_figures, "_compile", _stub_compile)
    mp.setattr(diagnostic_figures, "_compile", _stub_compile)

    def _import_runtime():
        return {
            "torch": torch,
            "authority": authority,
            "pilot": pilot,
            "public_metrics": public_metrics,
            "source_diagnostics": source_diagnostics,
            "reliability_metrics": reliability_metrics,
            "reporting_inputs": reporting_inputs,
            "benchmark_figures": benchmark_figures,
            "diagnostic_figures": diagnostic_figures,
        }

    mp.setattr(mod, "_import_runtime", _import_runtime)
    mp.setattr(mod.core, "_capture_provenance", lambda *args, **kwargs: {"captured": True})
    mp.setattr(mod.inputs, "prepare", _prepare)

    return {
        "base": base,
        "artifact_root": artifact_root,
        "run_root": run_root,
        "bundle": bundle,
        "prior": prior,
        "public": public,
        "source": source,
        "reliability": reliability,
        "authority": authority,
        "pilot": pilot,
        "reporting_inputs": reporting_inputs,
        "benchmark_figures": benchmark_figures,
        "diagnostic_figures": diagnostic_figures,
        "kwargs": {
            "project_root": str(project_root),
            "artifact_root": str(artifact_root),
            "contract_path": str(contract_path),
            "permit_path": str(permit_path),
        },
    }


@pytest.fixture(scope="module")
def report_run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    ctx = _setup(mp, tmp_path_factory.mktemp("reporting"))
    ctx["receipt"] = mod.run_reporting(**ctx["kwargs"])
    yield ctx
    mp.undo()


def test_import_runtime_smoke():
    runtime = _REAL_IMPORT_RUNTIME()
    assert set(runtime) == {
        "torch",
        "authority",
        "pilot",
        "public_metrics",
        "source_diagnostics",
        "reliability_metrics",
        "reporting_inputs",
        "benchmark_figures",
        "diagnostic_figures",
    }
    assert hasattr(runtime["authority"], "authenticate_comparison")
    assert hasattr(runtime["reporting_inputs"], "verify_reporting_sources")


def test_receipt_counters_and_manifest(report_run):
    receipt = report_run["receipt"]
    assert receipt["status"] == "complete"
    assert receipt["reporting_complete"] is True
    counters = receipt["counters"]
    assert counters["public_tables"] == 7
    assert counters["source_diagnostic_tables"] == 5
    assert counters["reliability_tables"] == 2
    assert counters["paired_figures"] > 0
    assert counters["diagnostic_figures"] > 0
    for key in ("fits", "calibrations", "outer_predictions", "updates"):
        assert counters[key] == 0
    assert receipt["prior_scientific_seconds_cumulative_bound"] == report_run["prior"]
    assert receipt["scientific_seconds_cumulative_bound"] == pytest.approx(
        report_run["prior"] + receipt["scientific_seconds_this_stage"]
    )
    manifest = report_run["run_root"] / mod.STAGE_NAME / mod.MANIFEST_NAME
    assert manifest.is_file()
    assert receipt["stage_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_tables_costs_and_figure_groups(report_run):
    public_root = report_run["run_root"] / mod.STAGE_NAME / mod.PUBLIC_ROOT_NAME
    tables = public_root / mod.TABLES_DIR_NAME
    assert len(list(tables.glob("*.csv"))) == mod.TOTAL_TABLE_COUNT
    assert (tables / mod.COSTS_NAME).is_file()
    figures = public_root / mod.FIGURES_DIR_NAME
    assert (figures / mod.PAIRED_UNIT_NAME / mod.RENDER_DIR_NAME).is_dir()
    assert (figures / mod.DIAGNOSTIC_UNIT_NAME / mod.RENDER_DIR_NAME).is_dir()


def test_actual_figure_helpers(report_run):
    figures = report_run["run_root"] / mod.STAGE_NAME / mod.PUBLIC_ROOT_NAME / mod.FIGURES_DIR_NAME
    paired_root = figures / mod.PAIRED_UNIT_NAME / mod.RENDER_DIR_NAME
    paired_name = report_run["benchmark_figures"].MANIFEST_NAME
    paired_manifest = json.loads((paired_root / paired_name).read_text(encoding="utf-8"))
    paired_files, paired_listed = mod._verify_figure_files(
        paired_root, paired_manifest, "paired", paired_name
    )
    assert (
        mod._verify_figure_records(paired_manifest, paired_listed, "paired") * 4 + 1 == paired_files
    )

    diagnostic_root = figures / mod.DIAGNOSTIC_UNIT_NAME / mod.RENDER_DIR_NAME
    diagnostic_name = report_run["diagnostic_figures"].MANIFEST_NAME
    diagnostic_manifest = json.loads(
        (diagnostic_root / diagnostic_name).read_text(encoding="utf-8")
    )
    diagnostic_files, diagnostic_listed = mod._verify_figure_files(
        diagnostic_root, diagnostic_manifest, "diagnostic", diagnostic_name
    )
    assert (
        mod._verify_figure_records(diagnostic_manifest, diagnostic_listed, "diagnostic") * 4 + 1
        == diagnostic_files
    )


def test_auth_failure_before_render(tmp_path, monkeypatch):
    ctx = _setup(monkeypatch, tmp_path / "auth")

    def _boom(bundle, *, deadline):
        raise RuntimeError("auth boom")

    monkeypatch.setattr(ctx["authority"], "authenticate_comparison", _boom)
    with pytest.raises(RuntimeError):
        mod.run_reporting(**ctx["kwargs"])
    assert not (ctx["run_root"] / mod.RECEIPT_NAME).exists()
    assert not (ctx["run_root"] / mod.STAGE_NAME).exists()


def test_occupied_stage_preserved(tmp_path, monkeypatch):
    ctx = _setup(monkeypatch, tmp_path / "occupied")
    stage = ctx["run_root"] / mod.STAGE_NAME
    stage.mkdir(parents=True)
    marker = stage / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(mod.core.P05CoreError, match="reporting_stage_exists"):
        mod.run_reporting(**ctx["kwargs"])
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (ctx["run_root"] / mod.RECEIPT_NAME).exists()


def test_occupied_stage_symlink_preserved(tmp_path, monkeypatch):
    ctx = _setup(monkeypatch, tmp_path / "symlink")
    target = ctx["run_root"] / "real_stage"
    target.mkdir()
    link = ctx["run_root"] / mod.STAGE_NAME
    link.symlink_to(target)
    with pytest.raises(mod.core.P05CoreError, match="symlink_path_rejected"):
        mod.run_reporting(**ctx["kwargs"])
    assert link.is_symlink()
    assert target.is_dir()


def test_occupied_receipt(tmp_path, monkeypatch):
    ctx = _setup(monkeypatch, tmp_path / "receipt")
    (ctx["run_root"] / mod.RECEIPT_NAME).write_text("{}", encoding="utf-8")
    with pytest.raises(mod.core.P05CoreError, match="reporting_receipt_exists"):
        mod.run_reporting(**ctx["kwargs"])


def test_changed_prior_receipt(tmp_path, monkeypatch):
    ctx = _setup(monkeypatch, tmp_path / "prior")
    receipt_path = ctx["run_root"] / mod.COMPARISON_RECEIPT_NAME
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload[mod.PRIOR_FIELD] = float(mod.development.MAXIMUM_TOTAL_SECONDS) + 1.0
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod.run_reporting(**ctx["kwargs"])
    assert exc.value.reason_code == "comparison_prior_out_of_range"


def test_storage_failure(tmp_path, monkeypatch):
    ctx = _setup(monkeypatch, tmp_path / "storage")
    monkeypatch.setattr(mod, "MINIMUM_BUDGET_HEADROOM_BYTES", 1 << 62)
    with pytest.raises(P05StorageError, match="storage_ceiling_exceeded"):
        mod.run_reporting(**ctx["kwargs"])
    stage = ctx["run_root"] / mod.STAGE_NAME
    assert (stage / mod.SUMMARY_NAME).is_file()
    assert not (ctx["run_root"] / mod.RECEIPT_NAME).exists()


def test_render_failure_leaves_failed_summary(tmp_path, monkeypatch):
    ctx = _setup(monkeypatch, tmp_path / "render")

    def _boom(*args, **kwargs):
        raise RuntimeError("render boom")

    monkeypatch.setattr(ctx["benchmark_figures"], "generate_pair_figures", _boom)
    with pytest.raises(RuntimeError):
        mod.run_reporting(**ctx["kwargs"])
    stage = ctx["run_root"] / mod.STAGE_NAME
    summary = json.loads((stage / mod.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["status"] == "fail"
    assert summary["reporting_complete"] is False
    assert not (ctx["run_root"] / mod.RECEIPT_NAME).exists()
    with pytest.raises(mod.core.P05CoreError, match="reporting_stage_exists"):
        mod.run_reporting(**ctx["kwargs"])


def test_source_verify_late_failure(tmp_path, monkeypatch):
    ctx = _setup(monkeypatch, tmp_path / "late")

    def _boom(*args, **kwargs):
        raise RuntimeError("verify boom")

    monkeypatch.setattr(ctx["reporting_inputs"], "verify_reporting_sources", _boom)
    with pytest.raises(RuntimeError):
        mod.run_reporting(**ctx["kwargs"])
    stage = ctx["run_root"] / mod.STAGE_NAME
    summary = json.loads((stage / mod.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["status"] == "fail"
    assert not (ctx["run_root"] / mod.RECEIPT_NAME).exists()


def test_costs_json_changed_late(tmp_path):
    original = {"alpha": 1, "beta": 2.5}
    path = tmp_path / mod.COSTS_NAME
    mod._write_json(path, original)
    path.write_text(json.dumps({"alpha": 2}), encoding="utf-8")
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod._verify_costs_json(path, original)
    assert exc.value.reason_code == "costs_json_changed"


def test_table_changed_late(tmp_path):
    frame = pd.DataFrame({"metric": ["a", "b"], "value": [1.0, 2.0]})
    path = tmp_path / "table.csv"
    mod._write_public_csv(path, frame)
    path.write_text("metric,value\nb,2.0\n", encoding="utf-8")
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod._roundtrip_verify(path, frame)
    assert exc.value.reason_code == "roundtrip_row_count_changed"


def test_csv_roundtrip_preserves_types(tmp_path):
    frame = pd.DataFrame(
        {
            "numeric": [1.5, 2.0],
            "numeric_string": ["00123", "45.0"],
            "blank": ["", "text"],
            "missing": [1.0, float("nan")],
            "flag": [True, False],
        }
    )
    path = tmp_path / "values.csv"
    mod._write_public_csv(path, frame)
    mod._roundtrip_verify(path, frame)


def test_csv_rejects_invalid_numeric(tmp_path):
    frame = pd.DataFrame({"numeric": [1.0, 2.0]})
    path = tmp_path / "bad.csv"
    mod._write_public_csv(path, frame)
    path.write_text("numeric\nnot-a-number\n2.0\n", encoding="utf-8")
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod._roundtrip_verify(path, frame)
    assert exc.value.reason_code == "roundtrip_numeric_malformed"


def test_csv_rejects_changed_columns(tmp_path):
    frame = pd.DataFrame({"numeric": [1.0, 2.0]})
    path = tmp_path / "cols.csv"
    mod._write_public_csv(path, frame)
    path.write_text("other\n1.0\n2.0\n", encoding="utf-8")
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod._roundtrip_verify(path, frame)
    assert exc.value.reason_code == "roundtrip_columns_changed"


def test_strict_table_keys_rejected():
    public_metrics = SimpleNamespace(P05_PUBLIC_TABLE_NAMES=PUBLIC_NAMES)
    source_diagnostics = SimpleNamespace(P05_SOURCE_DIAGNOSTIC_TABLE_NAMES=SOURCE_NAMES)
    public = {name: _tiny() for name in PUBLIC_NAMES[:-1]}
    diagnostics = {name: _tiny() for name in SOURCE_NAMES}
    reliability = {name: _tiny() for name in RELIABILITY_NAMES}
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod._collect_frames(public, diagnostics, reliability, public_metrics, source_diagnostics)
    assert exc.value.reason_code == "public_table_keys_mismatch"


def test_private_columns_rejected():
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod._reject_private_columns({"table": pd.DataFrame({"observation_uid": ["x"]})})
    assert exc.value.reason_code == "private_column_rejected_table"
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod._reject_private_columns({"table": pd.DataFrame({"probability_x": [0.5]})})
    assert exc.value.reason_code == "private_column_rejected_table"


def _cost_inputs():
    return SimpleNamespace(PUBLIC_COST_KEYS=COST_KEYS, REQUIRED_PUBLIC_COST_KEYS=REQUIRED_COST_KEYS)


def test_public_cost_bool_rejected():
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod._check_public_costs({"public_costs": {"alpha": True, "beta": 1.0}}, _cost_inputs())
    assert exc.value.reason_code == "public_cost_value_malformed"


def test_public_cost_negative_missing_unknown():
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod._check_public_costs({"public_costs": {"alpha": -1.0, "beta": 1.0}}, _cost_inputs())
    assert exc.value.reason_code == "public_cost_value_negative"
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod._check_public_costs({"public_costs": {"alpha": 1.0}}, _cost_inputs())
    assert exc.value.reason_code == "public_cost_key_missing"
    with pytest.raises(mod.P05ComprehensiveReportingError) as exc:
        mod._check_public_costs({"public_costs": {"alpha": 1.0, "delta": 2.0}}, _cost_inputs())
    assert exc.value.reason_code == "public_cost_key_not_allowed"


def _public_root(ctx):
    return ctx["run_root"] / mod.STAGE_NAME / mod.PUBLIC_ROOT_NAME


def _assert_failed(ctx):
    stage = ctx["run_root"] / mod.STAGE_NAME
    summary = json.loads((stage / mod.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["status"] == "fail"
    assert summary["reporting_complete"] is False
    assert not (ctx["run_root"] / mod.RECEIPT_NAME).exists()


def _mutate_costs(ctx):
    path = _public_root(ctx) / mod.TABLES_DIR_NAME / mod.COSTS_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[next(iter(payload))] = True
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mutate_csv(ctx):
    path = _public_root(ctx) / mod.TABLES_DIR_NAME / "strategy_contexts.csv"
    stored = pd.read_csv(path, keep_default_na=False)
    stored.loc[0, "value"] = 999.0
    stored.to_csv(path, index=False)


def _mutate_figure(ctx):
    root = _public_root(ctx) / mod.FIGURES_DIR_NAME / mod.PAIRED_UNIT_NAME / mod.RENDER_DIR_NAME
    manifest = json.loads((root / ctx["benchmark_figures"].MANIFEST_NAME).read_text("utf-8"))
    (root / manifest["semantic_path"]).write_bytes(b"tampered")


def _mutate_comparison_receipt(ctx):
    path = ctx["run_root"] / mod.COMPARISON_RECEIPT_NAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tampered"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mutate_bindings(ctx):
    (ctx["run_root"] / mod.STAGE_NAME / mod.BINDINGS_NAME).write_text(
        json.dumps({"tampered": "b" * 64}), encoding="utf-8"
    )


def _mutate_unexpected_file(ctx):
    (_public_root(ctx) / "secret.txt").write_text("x", encoding="utf-8")


def _mutate_unexpected_dir(ctx):
    (_public_root(ctx) / "extra").mkdir()


LATE_MUTATIONS = [
    ("costs_bool", _mutate_costs, mod.P05ComprehensiveReportingError),
    ("csv_value", _mutate_csv, (mod.P05ComprehensiveReportingError, AssertionError)),
    ("figure_bytes", _mutate_figure, mod.P05ComprehensiveReportingError),
    ("comparison_receipt", _mutate_comparison_receipt, mod.P05ComprehensiveReportingError),
    ("reporting_bindings", _mutate_bindings, mod.P05ComprehensiveReportingError),
    ("unexpected_file", _mutate_unexpected_file, mod.P05ComprehensiveReportingError),
    ("unexpected_dir", _mutate_unexpected_dir, mod.P05ComprehensiveReportingError),
]


@pytest.mark.parametrize(
    "name,mutate,expected", LATE_MUTATIONS, ids=[item[0] for item in LATE_MUTATIONS]
)
def test_late_mutation_rejected(tmp_path, monkeypatch, name, mutate, expected):
    ctx = _setup(monkeypatch, tmp_path / name)
    original = ctx["pilot"]._post_run_reauth

    def _hook(*args, **kwargs):
        result = original(*args, **kwargs)
        mutate(ctx)
        return result

    monkeypatch.setattr(ctx["pilot"], "_post_run_reauth", _hook)
    with pytest.raises(expected):
        mod.run_reporting(**ctx["kwargs"])
    _assert_failed(ctx)


def test_late_receipt_failure_invalidates_receipt(tmp_path, monkeypatch):
    ctx = _setup(monkeypatch, tmp_path / "receipt_late")
    real_check = mod.freeze._check_deadline

    def _boom(deadline):
        if (ctx["run_root"] / mod.RECEIPT_NAME).exists():
            raise RuntimeError("receipt late boom")
        return real_check(deadline)

    monkeypatch.setattr(mod.freeze, "_check_deadline", _boom)
    with pytest.raises(RuntimeError, match="receipt late boom"):
        mod.run_reporting(**ctx["kwargs"])
    receipt_path = ctx["run_root"] / mod.RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    stage = ctx["run_root"] / mod.STAGE_NAME
    summary = json.loads((stage / mod.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["status"] == "fail"
    manifest_sha = hashlib.sha256((stage / mod.MANIFEST_NAME).read_bytes()).hexdigest()
    assert receipt["stage_manifest_sha256"] != manifest_sha
