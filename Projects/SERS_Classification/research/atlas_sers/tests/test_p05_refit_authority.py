"""Torch-free refit-authority boundary tests for P05 selection authentication."""

from __future__ import annotations

import pytest

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_refit_authority as authority
from atlas_sers.evaluation import p05_refit_plan as refit_plan

PRIOR = 3720.0
SECONDS = 10.0
MAX_STEPS = development.MAXIMUM_FIT_STEPS
RESERVE = development.PRELAUNCH_AUDIT_RESERVE_SECONDS
TOTAL = development.MAXIMUM_TOTAL_SECONDS


def _bundle():
    return {
        "permit_sha256": "1" * 64,
        "contract_sha256": "2" * 64,
        "core_plan_id": "3" * 64,
        "ledger": {"ledger_id": "4" * 64},
    }


def _plan_payload(decisions=1, aliases=1, unique=1):
    return {
        "decisions": [{} for _ in range(decisions)],
        "strategy_aliases": [{"refit_id": f"refit-{i % unique}"} for i in range(aliases)],
        "counts": {"strategy_alias_count": aliases, "unique_refit_count": unique},
        "unique_refits": {f"refit-{i}": {"i": i} for i in range(unique)},
    }


def _seal_plan(plan):
    plan["plan_id"] = refit_plan._sha256_canonical(
        {key: value for key, value in plan.items() if key != "plan_id"}
    )
    return plan


def _reseal(world, **over):
    plan = world["plan"]
    plan.update(over)
    _seal_plan(plan)
    world["receipt_id"] = plan["plan_id"]
    world["summary"]["plan_id"] = plan["plan_id"]


def _valid_receipt(bundle, prior=PRIOR, seconds=SECONDS, plan_id="a" * 64):
    return {
        **freeze._base_payload(bundle),
        "stage": freeze.SELECTION_STAGE_NAME,
        "selection_plan_id": plan_id,
        "selection_manifest_sha256": "b" * 64,
        "prior_scientific_seconds_cumulative_bound": prior,
        "scientific_seconds_this_stage": seconds,
        "scientific_seconds_cumulative_bound": prior + seconds,
        "prelaunch_audit_reserve_seconds": RESERVE,
        "maximum_total_seconds": TOTAL,
    }


def _valid_summary(bundle, plan_id="a" * 64, prior=PRIOR, seconds=SECONDS):
    return {
        **freeze._base_payload(bundle),
        "status": "complete",
        "command": "freeze_selection",
        "refit_decision_count": 1,
        "strategy_alias_count": 1,
        "unique_refit_count": 1,
        "plan_id": plan_id,
        "prior_scientific_seconds_cumulative_bound": prior,
        "scientific_seconds_this_stage": seconds,
        "scientific_seconds_cumulative_bound": prior + seconds,
        "prelaunch_audit_reserve_seconds": RESERVE,
        "maximum_total_seconds": TOTAL,
    }


@pytest.mark.parametrize(
    "override,reason",
    [
        (
            {"scientific_seconds_this_stage": float("nan")},
            "selection_receipt_stage_seconds_malformed",
        ),
        ({"scientific_seconds_this_stage": True}, "selection_receipt_stage_seconds_malformed"),
        (
            {"prior_scientific_seconds_cumulative_bound": float("inf")},
            "selection_receipt_prior_seconds_malformed",
        ),
        (
            {"prior_scientific_seconds_cumulative_bound": PRIOR + 1.0},
            "selection_receipt_prior_seconds_mismatch",
        ),
        (
            {"scientific_seconds_cumulative_bound": PRIOR + 99.0},
            "selection_receipt_cumulative_inconsistent",
        ),
        (
            {
                "scientific_seconds_this_stage": -1.0,
                "scientific_seconds_cumulative_bound": PRIOR - 1.0,
            },
            "selection_receipt_stage_seconds_out_of_range",
        ),
        ({"selection_plan_id": "zz"}, "selection_receipt_plan_id_malformed"),
        ({"extra": 1}, "selection_receipt_fields_mismatch"),
    ],
)
def test_selection_receipt_rejections(override, reason):
    bundle = _bundle()
    receipt = _valid_receipt(bundle)
    receipt.update(override)
    with pytest.raises(authority.RefitAuthorityError) as exc:
        authority._check_selection_receipt(bundle, receipt, PRIOR)
    assert exc.value.reason_code == reason


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"status": "fail"}, "selection_summary_not_complete"),
        ({"command": "x"}, "selection_summary_command_mismatch"),
        (
            {"prior_scientific_seconds_cumulative_bound": float("nan")},
            "selection_summary_prior_seconds_malformed",
        ),
        (
            {"scientific_seconds_this_stage": float("nan")},
            "selection_summary_stage_seconds_malformed",
        ),
        (
            {
                "scientific_seconds_this_stage": SECONDS + 1.0,
                "scientific_seconds_cumulative_bound": PRIOR + SECONDS + 1.0,
            },
            "selection_summary_elapsed_exceeds_receipt",
        ),
        ({"refit_decision_count": True}, "selection_summary_counts_malformed"),
        ({"plan_id": "zz"}, "selection_summary_plan_id_malformed"),
        ({"extra": 1}, "selection_summary_fields_mismatch"),
    ],
)
def test_selection_summary_rejections(override, reason):
    bundle = _bundle()
    summary = _valid_summary(bundle)
    summary.update(override)
    with pytest.raises(authority.RefitAuthorityError) as exc:
        authority._check_selection_summary(bundle, summary, PRIOR, SECONDS)
    assert exc.value.reason_code == reason


@pytest.mark.parametrize(
    "summary_over,receipt_over,expected",
    [
        ({"optimizer_steps": True}, {}, "develop_optimizer_steps_malformed"),
        ({"optimizer_steps": -1}, {}, "develop_optimizer_steps_out_of_range"),
        ({"optimizer_steps": MAX_STEPS + 1}, {}, "develop_optimizer_steps_out_of_range"),
        ({"maximum_optimizer_steps": 1}, {}, "develop_maximum_optimizer_steps_mismatch"),
        ({"optimizer_steps_exact": False}, {}, "develop_optimizer_steps_inexact"),
        ({}, {"optimizer_steps": 6}, "receipt_optimizer_steps_mismatch"),
        ({}, {"optimizer_steps": True}, "receipt_optimizer_steps_malformed"),
        ({}, {}, 5),
        ({"optimizer_steps_exact": True}, {}, 5),
    ],
)
def test_source_optimizer_steps(summary_over, receipt_over, expected):
    summary = {"optimizer_steps": 5, "maximum_optimizer_steps": MAX_STEPS, **summary_over}
    receipt = {"optimizer_steps": 5, **receipt_over}
    if isinstance(expected, str):
        with pytest.raises(authority.RefitAuthorityError) as exc:
            authority._check_source_optimizer_steps(summary, receipt)
        assert exc.value.reason_code == expected
    else:
        assert authority._check_source_optimizer_steps(summary, receipt) == expected


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda w: w.update(stored=dict(w["plan"], plan_id="c" * 64)), "selection_plan_mismatch"),
        (lambda w: w["plan"].update(plan_id="c" * 64), "selection_plan_id_not_canonical"),
        (lambda w: w.update(receipt_id="c" * 64), "selection_receipt_plan_id_mismatch"),
        (lambda w: w["summary"].update(plan_id="c" * 64), "selection_summary_plan_id_mismatch"),
        (lambda w: _reseal(w, decisions=[]), "selection_decision_count_mismatch"),
        (lambda w: _reseal(w, strategy_aliases=[]), "selection_alias_count_mismatch"),
        (lambda w: w["summary"].update(refit_decision_count=9), "selection_summary_count_mismatch"),
        (
            lambda w: (
                _reseal(w, counts={"strategy_alias_count": 1, "unique_refit_count": 9}),
                w["summary"].update(unique_refit_count=9),
            ),
            "selection_unique_refit_count_mismatch",
        ),
        (
            lambda w: _reseal(w, strategy_aliases=[{"refit_id": "ghost"}]),
            "selection_alias_refit_unknown",
        ),
    ],
)
def test_check_plan_rejections(monkeypatch, mutate, reason):
    monkeypatch.setattr(freeze, "CONTEXT_COUNT", 1)
    monkeypatch.setattr(freeze, "STRATEGY_ALIAS_COUNT", 1)
    bundle = _bundle()
    plan = _seal_plan(_plan_payload())
    world = {
        "plan": plan,
        "stored": plan,
        "receipt_id": plan["plan_id"],
        "summary": _valid_summary(bundle, plan_id=plan["plan_id"]),
    }
    mutate(world)
    with pytest.raises(authority.RefitAuthorityError) as exc:
        authority._check_plan(world["plan"], world["stored"], world["receipt_id"], world["summary"])
    assert exc.value.reason_code == reason


def test_authenticate_selection_authenticates(monkeypatch, tmp_path):
    monkeypatch.setattr(freeze, "CONTEXT_COUNT", 1)
    monkeypatch.setattr(freeze, "STRATEGY_ALIAS_COUNT", 1)
    bundle = _bundle()
    plan = _seal_plan(_plan_payload())
    dev_receipt = {"optimizer_steps": 96}
    sel_receipt = _valid_receipt(bundle, plan_id=plan["plan_id"])
    summary = _valid_summary(bundle, plan_id=plan["plan_id"])
    develop_summary = {"optimizer_steps": 96, "maximum_optimizer_steps": MAX_STEPS}
    run_root, develop, stage = tmp_path / "run", tmp_path / "develop", tmp_path / "selection"
    paths = {
        "receipt": run_root / "development_receipt.json",
        "develop": develop,
        "selection": stage,
        "selection_receipt": run_root / "selection_receipt.json",
    }
    payloads = {
        paths["receipt"]: dev_receipt,
        develop / "summary.json": develop_summary,
        paths["selection_receipt"]: sel_receipt,
        stage / "summary.json": summary,
        stage / "plan.json": plan,
    }
    monkeypatch.setattr(freeze, "_paths", lambda bundle: paths)
    monkeypatch.setattr(freeze, "_expected_new_units", lambda bundle: 1)
    monkeypatch.setattr(freeze, "_read_mapping", lambda path, code: payloads[path])
    monkeypatch.setattr(freeze, "_check_prior_bound", lambda receipt: PRIOR)
    monkeypatch.setattr(freeze, "_read_jsonl", lambda *a, **k: [])
    monkeypatch.setattr(freeze, "_authenticate_selector", lambda *a, **k: [])
    monkeypatch.setattr(freeze, "_build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(authority, "_verify_selection_manifest", lambda *a, **k: None)
    monkeypatch.setattr(authority, "_check_source_bindings", lambda *a, **k: None)
    for name in (
        "_check_deadline",
        "_check_receipt",
        "_check_develop_summary",
        "_verify_develop_manifest",
        "_verify_pilot_manifest",
        "_verify_ledger",
        "_verify_source_ledger",
    ):
        monkeypatch.setattr(freeze, name, lambda *a, **k: None)

    result = authority.authenticate_selection(bundle, deadline=1000000.0)
    assert result["plan"] == plan
    assert result["prior_seconds"] == PRIOR + SECONDS
    assert result["source_optimizer_steps"] == 96
    assert result["development_receipt"] == dev_receipt
    assert result["selection_receipt"] == sel_receipt


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True])
def test_nonfinite_deadline_rejected_before_reads(monkeypatch, deadline):
    def forbidden(*args, **kwargs):
        raise AssertionError("No input reads before deadline validation")

    monkeypatch.setattr(freeze, "_paths", forbidden)
    with pytest.raises(authority.RefitAuthorityError, match="deadline_malformed"):
        authority.authenticate_selection({}, deadline=deadline)
