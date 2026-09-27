"""Read-only filesystem tests for the P05 refit authority boundary.

These tests drive the real ``authenticate_selection`` reads, hashes, manifest
verification and selector reconstruction against the synthetic scaled world
built by ``tests.test_p05_comprehensive_freeze``.  The pure refit-plan builder
is stubbed by ``_build_world`` with a canonical minimal plan; no arrays,
logits, training or fits are touched.
"""

from __future__ import annotations

import json
import time

import pytest

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_comprehensive_inputs as inputs
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_refit_authority as authority
from atlas_sers.evaluation import p05_refit_plan as refit_plan
from atlas_sers.evaluation.p05_core_run import P05CoreError
from tests.test_p05_comprehensive_freeze import (
    _build_world,
    _invoke,
    _write_json,
)
from tests.test_p05_comprehensive_freeze import (
    scaled as _scaled,
)

SELECTION_STAGE = "selection"
SELECTION_MANIFEST = "manifest.json"
SELECTION_RECEIPT = "selection_receipt.json"
scaled = _scaled


@pytest.fixture
def limits(monkeypatch):
    monkeypatch.setattr(development, "MAXIMUM_FIT_STEPS", 9600)


def _canonical_plan():
    content = {
        "decisions": [{}],
        "strategy_aliases": [{"refit_id": "r"} for _ in range(9)],
        "unique_refits": {"r": {}},
        "counts": {
            "context_count": 1,
            "strategy_alias_count": 9,
            "expected_strategy_alias_count": 9,
            "unique_refit_count": 1,
        },
    }
    return {**content, "plan_id": refit_plan._sha256_canonical(content)}


def _freeze(monkeypatch, tmp_path, *, plan=None):
    bundle, run_root, develop = _build_world(
        tmp_path, monkeypatch, plan=_canonical_plan() if plan is None else plan
    )
    _invoke(bundle, tmp_path)
    return bundle, run_root, develop


def _inventory(stage):
    return {
        str(path.relative_to(stage)): core._canon().sha256_file(path)
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    }


def _refresh_selection_manifest(run_root, stage):
    core._write_manifest(stage)
    receipt_path = run_root / SELECTION_RECEIPT
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["selection_manifest_sha256"] = core._canon().sha256_file(stage / SELECTION_MANIFEST)
    _write_json(receipt_path, receipt)


def _authenticate(bundle):
    return authority.authenticate_selection(bundle, deadline=time.perf_counter() + 3600.0)


def test_authenticate_selection_success_leaves_inventory(limits, scaled, monkeypatch, tmp_path):
    bundle, run_root, _ = _freeze(monkeypatch, tmp_path)
    stage = run_root / SELECTION_STAGE
    before = _inventory(stage)

    result = _authenticate(bundle)

    receipt = json.loads((run_root / SELECTION_RECEIPT).read_text(encoding="utf-8"))
    assert result["plan"] == _canonical_plan()
    assert result["source_optimizer_steps"] == 96
    assert result["prior_seconds"] == pytest.approx(receipt["scientific_seconds_cumulative_bound"])
    assert _inventory(stage) == before


def test_post_freeze_selection_tamper_fails_manifest(limits, scaled, monkeypatch, tmp_path):
    bundle, run_root, _ = _freeze(monkeypatch, tmp_path)
    summary = run_root / SELECTION_STAGE / "summary.json"
    with summary.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(P05CoreError) as exc:
        _authenticate(bundle)
    assert exc.value.reason_code == "manifest_integrity_mismatch"


def test_source_stored_summary_tamper_fails_develop_manifest(limits, scaled, monkeypatch, tmp_path):
    bundle, _, develop = _freeze(monkeypatch, tmp_path)
    with (develop / "summary.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(P05CoreError) as exc:
        _authenticate(bundle)
    assert exc.value.reason_code == "manifest_integrity_mismatch"


def test_pilot_file_tamper_fails_pilot(limits, scaled, monkeypatch, tmp_path):
    bundle, _, _ = _freeze(monkeypatch, tmp_path)
    pilot_run = inputs._pilot_run_dir(bundle["artifact_root"])
    victim = next(iter(pilot_run.glob("executions/*/summary.json")))
    with victim.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(P05CoreError) as exc:
        _authenticate(bundle)
    assert exc.value.reason_code == "manifest_integrity_mismatch"


def test_edited_bindings_fails_binding_expected(limits, scaled, monkeypatch, tmp_path):
    bundle, run_root, _ = _freeze(monkeypatch, tmp_path)
    stage = run_root / SELECTION_STAGE
    path = stage / "source_bindings.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_new_fits"] = int(payload["source_new_fits"]) + 1
    _write_json(path, payload)
    _refresh_selection_manifest(run_root, stage)
    with pytest.raises(P05CoreError) as exc:
        _authenticate(bundle)
    assert exc.value.reason_code == "selection_bindings_mismatch"


def test_edited_plan_fails_reconstructed_match(limits, scaled, monkeypatch, tmp_path):
    bundle, run_root, _ = _freeze(monkeypatch, tmp_path)
    stage = run_root / SELECTION_STAGE
    path = stage / "plan.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decisions"][0]["tampered"] = True
    _write_json(path, payload)
    _refresh_selection_manifest(run_root, stage)
    with pytest.raises(P05CoreError) as exc:
        _authenticate(bundle)
    assert exc.value.reason_code == "selection_plan_mismatch"


def test_forged_selection_prior_fails_before_verify(limits, scaled, monkeypatch, tmp_path):
    bundle, run_root, _ = _freeze(monkeypatch, tmp_path)
    receipt_path = run_root / SELECTION_RECEIPT
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["prior_scientific_seconds_cumulative_bound"] = (
        float(receipt["prior_scientific_seconds_cumulative_bound"]) + 1.0
    )
    _write_json(receipt_path, receipt)

    called = {"manifest": False, "selector": False}

    def manifest_spy(*args, **kwargs):
        called["manifest"] = True
        raise AssertionError("manifest verification must not run")

    def selector_spy(*args, **kwargs):
        called["selector"] = True
        raise AssertionError("selector reconstruction must not run")

    monkeypatch.setattr(freeze, "_verify_develop_manifest", manifest_spy)
    monkeypatch.setattr(freeze, "_authenticate_selector", selector_spy)
    with pytest.raises(P05CoreError) as exc:
        _authenticate(bundle)
    assert exc.value.reason_code == "selection_receipt_prior_seconds_mismatch"
    assert called == {"manifest": False, "selector": False}


def test_missing_selection_receipt_fails(limits, scaled, monkeypatch, tmp_path):
    bundle, run_root, _ = _freeze(monkeypatch, tmp_path)
    (run_root / SELECTION_RECEIPT).unlink()
    with pytest.raises(P05CoreError) as exc:
        _authenticate(bundle)
    assert exc.value.reason_code == "selection_receipt_missing"


def test_expired_finite_deadline_fails_before_files(limits, scaled, monkeypatch, tmp_path):
    bundle, _, _ = _freeze(monkeypatch, tmp_path)

    def read_spy(*args, **kwargs):
        raise AssertionError("filesystem reads must not run")

    monkeypatch.setattr(freeze, "_read_mapping", read_spy)
    with pytest.raises(P05CoreError) as exc:
        authority.authenticate_selection(bundle, deadline=time.perf_counter() - 1.0)
    assert exc.value.reason_code == "global_deadline_exceeded"
