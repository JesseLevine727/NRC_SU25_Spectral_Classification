"""P05 refit authority read-only selection authentication boundary.

``authenticate_selection`` re-authenticates the completed comprehensive source
development evidence and the frozen read-only selection stage without fitting a
model, loading a logit or array, running an outer evaluation or writing any
filesystem artifact.  It reconstructs every persisted selector record and the
deterministic refit plan from the development execution summaries with the
existing freeze helpers, verifies the stored selection plan, source bindings,
summary, receipt and manifests, and returns the authenticated plan together
with the prior cumulative bound and the exact source optimizer-step count.  No
selection choice is made here and no refit is authorized or launched.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_comprehensive_development as development
from atlas_sers.evaluation import p05_comprehensive_freeze as freeze
from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation import p05_refit_plan as refit_plan

__all__ = ["RefitAuthorityError", "authenticate_selection"]

SELECTION_SUMMARY_NAME = "summary.json"
SELECTION_PLAN_NAME = "plan.json"
SELECTION_BINDINGS_NAME = "source_bindings.json"
SELECTION_MANIFEST_NAME = "manifest.json"

_SELECTION_RECEIPT_FIELDS = frozenset(
    {
        "stage",
        "selection_plan_id",
        "selection_manifest_sha256",
        "prior_scientific_seconds_cumulative_bound",
        "scientific_seconds_this_stage",
        "scientific_seconds_cumulative_bound",
        "prelaunch_audit_reserve_seconds",
        "maximum_total_seconds",
    }
)
_SELECTION_SUMMARY_FIELDS = frozenset(
    {
        "status",
        "command",
        "refit_decision_count",
        "strategy_alias_count",
        "unique_refit_count",
        "plan_id",
        "prior_scientific_seconds_cumulative_bound",
        "scientific_seconds_this_stage",
        "scientific_seconds_cumulative_bound",
        "prelaunch_audit_reserve_seconds",
        "maximum_total_seconds",
    }
)


class RefitAuthorityError(freeze.FreezeSelectionError):
    """Stable refit-authority failure with a path-free reason code."""


def _err(reason_code: str) -> RefitAuthorityError:
    return RefitAuthorityError(reason_code)


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _err(code)
    number = float(value)
    if not math.isfinite(number):
        raise _err(code)
    return number


def _check_base(bundle: Mapping[str, Any], record: Mapping[str, Any], code: str) -> None:
    for key, expected in freeze._base_payload(bundle).items():
        if record.get(key) != expected:
            raise _err(code)


def _check_source_optimizer_steps(summary: Mapping[str, Any], receipt: Mapping[str, Any]) -> int:
    steps = summary.get("optimizer_steps")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise _err("develop_optimizer_steps_malformed")
    maximum = summary.get("maximum_optimizer_steps", development.MAXIMUM_FIT_STEPS)
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise _err("develop_maximum_optimizer_steps_malformed")
    if int(maximum) != int(development.MAXIMUM_FIT_STEPS):
        raise _err("develop_maximum_optimizer_steps_mismatch")
    if steps < 0 or int(steps) > int(maximum):
        raise _err("develop_optimizer_steps_out_of_range")
    if "optimizer_steps_exact" in summary and summary.get("optimizer_steps_exact") is not True:
        raise _err("develop_optimizer_steps_inexact")
    receipt_steps = receipt.get("optimizer_steps")
    if isinstance(receipt_steps, bool) or not isinstance(receipt_steps, int):
        raise _err("receipt_optimizer_steps_malformed")
    if int(receipt_steps) != int(steps):
        raise _err("receipt_optimizer_steps_mismatch")
    return int(steps)


def _check_selection_receipt(
    bundle: Mapping[str, Any], receipt: Mapping[str, Any], prior: float
) -> tuple[str, str, float]:
    base = freeze._base_payload(bundle)
    _check_base(bundle, receipt, "selection_receipt_base_mismatch")
    if set(receipt) != set(base) | _SELECTION_RECEIPT_FIELDS:
        raise _err("selection_receipt_fields_mismatch")
    if str(receipt.get("stage")) != freeze.SELECTION_STAGE_NAME:
        raise _err("selection_receipt_stage_mismatch")
    plan_id = str(receipt.get("selection_plan_id"))
    if not core._is_hex64(plan_id):
        raise _err("selection_receipt_plan_id_malformed")
    manifest_sha256 = str(receipt.get("selection_manifest_sha256"))
    if not core._is_hex64(manifest_sha256):
        raise _err("selection_receipt_manifest_sha_malformed")
    if _finite(
        receipt.get("prelaunch_audit_reserve_seconds"),
        "selection_receipt_reserve_malformed",
    ) != float(development.PRELAUNCH_AUDIT_RESERVE_SECONDS):
        raise _err("selection_receipt_reserve_mismatch")
    if _finite(
        receipt.get("maximum_total_seconds"), "selection_receipt_maximum_total_malformed"
    ) != float(development.MAXIMUM_TOTAL_SECONDS):
        raise _err("selection_receipt_maximum_total_mismatch")
    stage_seconds = _finite(
        receipt.get("scientific_seconds_this_stage"),
        "selection_receipt_stage_seconds_malformed",
    )
    if stage_seconds < 0.0:
        raise _err("selection_receipt_stage_seconds_out_of_range")
    if _finite(
        receipt.get("prior_scientific_seconds_cumulative_bound"),
        "selection_receipt_prior_seconds_malformed",
    ) != float(prior):
        raise _err("selection_receipt_prior_seconds_mismatch")
    cumulative = _finite(
        receipt.get("scientific_seconds_cumulative_bound"),
        "selection_receipt_cumulative_seconds_malformed",
    )
    if cumulative != float(prior) + stage_seconds:
        raise _err("selection_receipt_cumulative_inconsistent")
    if cumulative > float(development.MAXIMUM_TOTAL_SECONDS):
        raise _err("selection_receipt_cumulative_exceeds_total")
    return plan_id, manifest_sha256, stage_seconds


def _check_selection_summary(
    bundle: Mapping[str, Any],
    summary: Mapping[str, Any],
    prior: float,
    receipt_seconds: float,
) -> str:
    base = freeze._base_payload(bundle)
    _check_base(bundle, summary, "selection_summary_base_mismatch")
    if set(summary) != set(base) | _SELECTION_SUMMARY_FIELDS:
        raise _err("selection_summary_fields_mismatch")
    if str(summary.get("status")) != "complete":
        raise _err("selection_summary_not_complete")
    if str(summary.get("command")) != "freeze_selection":
        raise _err("selection_summary_command_mismatch")
    if _finite(
        summary.get("prelaunch_audit_reserve_seconds"),
        "selection_summary_reserve_malformed",
    ) != float(development.PRELAUNCH_AUDIT_RESERVE_SECONDS):
        raise _err("selection_summary_reserve_mismatch")
    if _finite(
        summary.get("maximum_total_seconds"), "selection_summary_maximum_total_malformed"
    ) != float(development.MAXIMUM_TOTAL_SECONDS):
        raise _err("selection_summary_maximum_total_mismatch")
    if _finite(
        summary.get("prior_scientific_seconds_cumulative_bound"),
        "selection_summary_prior_seconds_malformed",
    ) != float(prior):
        raise _err("selection_summary_prior_seconds_mismatch")
    stage_seconds = _finite(
        summary.get("scientific_seconds_this_stage"),
        "selection_summary_stage_seconds_malformed",
    )
    if stage_seconds < 0.0:
        raise _err("selection_summary_stage_seconds_out_of_range")
    if stage_seconds > receipt_seconds:
        raise _err("selection_summary_elapsed_exceeds_receipt")
    if (
        _finite(
            summary.get("scientific_seconds_cumulative_bound"),
            "selection_summary_cumulative_seconds_malformed",
        )
        != float(prior) + stage_seconds
    ):
        raise _err("selection_summary_cumulative_inconsistent")
    for key in ("refit_decision_count", "strategy_alias_count", "unique_refit_count"):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _err("selection_summary_counts_malformed")
    plan_id = str(summary.get("plan_id"))
    if not core._is_hex64(plan_id):
        raise _err("selection_summary_plan_id_malformed")
    return plan_id


def _verify_selection_manifest(stage: Path, manifest_sha256: str) -> None:
    manifest_path = stage / SELECTION_MANIFEST_NAME
    core._reject_symlink_chain(manifest_path)
    if not manifest_path.is_file():
        raise _err("selection_manifest_missing")
    if freeze._canon().sha256_file(manifest_path) != str(manifest_sha256):
        raise _err("selection_manifest_digest_mismatch")
    pilot._verify_manifest(stage)


def _check_source_bindings(
    bundle: Mapping[str, Any], stage: Path, develop_stage: Path, selector_path: Path
) -> None:
    stored = freeze._read_mapping(stage / SELECTION_BINDINGS_NAME, "selection_bindings_missing")
    expected = freeze._source_bindings(
        bundle,
        freeze._canon().sha256_file(develop_stage / "manifest.json"),
        freeze._canon().sha256_file(develop_stage / "source_ledger.json"),
        freeze._canon().sha256_file(selector_path),
    )
    if stored != expected:
        raise _err("selection_bindings_mismatch")


def _check_plan(
    plan: Mapping[str, Any],
    stored_plan: Mapping[str, Any],
    receipt_plan_id: str,
    summary: Mapping[str, Any],
) -> None:
    if stored_plan != plan:
        raise _err("selection_plan_mismatch")
    plan_id = str(plan.get("plan_id"))
    if not core._is_hex64(plan_id):
        raise _err("selection_plan_id_malformed")
    content = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan_id != refit_plan._sha256_canonical(content):
        raise _err("selection_plan_id_not_canonical")
    if plan_id != receipt_plan_id:
        raise _err("selection_receipt_plan_id_mismatch")
    if plan_id != str(summary.get("plan_id")):
        raise _err("selection_summary_plan_id_mismatch")
    decisions = plan.get("decisions")
    aliases = plan.get("strategy_aliases")
    counts = plan.get("counts")
    unique_refits = plan.get("unique_refits")
    if not isinstance(decisions, Sequence) or isinstance(decisions, (str, bytes)):
        raise _err("selection_plan_shape_mismatch")
    if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
        raise _err("selection_plan_shape_mismatch")
    if not isinstance(counts, Mapping) or not isinstance(unique_refits, Mapping):
        raise _err("selection_plan_shape_mismatch")
    if len(decisions) != freeze.CONTEXT_COUNT:
        raise _err("selection_decision_count_mismatch")
    if len(aliases) != freeze.STRATEGY_ALIAS_COUNT:
        raise _err("selection_alias_count_mismatch")
    for key, expected in (
        ("refit_decision_count", len(decisions)),
        ("strategy_alias_count", int(counts.get("strategy_alias_count", -1))),
        ("unique_refit_count", int(counts.get("unique_refit_count", -1))),
    ):
        if int(summary.get(key, -1)) != expected:
            raise _err("selection_summary_count_mismatch")
    if int(counts.get("unique_refit_count", -1)) != len(unique_refits):
        raise _err("selection_unique_refit_count_mismatch")
    for alias in aliases:
        if str(alias.get("refit_id")) not in unique_refits:
            raise _err("selection_alias_refit_unknown")


def authenticate_selection(bundle: Mapping[str, Any], *, deadline: float) -> dict[str, Any]:
    """Re-authenticate the frozen P05 selection evidence read-only."""

    deadline = _finite(deadline, "deadline_malformed")
    freeze._check_deadline(deadline)
    paths = freeze._paths(bundle)
    expected_units = freeze._expected_new_units(bundle)
    development_receipt = freeze._read_mapping(paths["receipt"], "development_receipt_missing")
    freeze._check_receipt(development_receipt, expected_units)
    develop_summary = freeze._read_mapping(
        paths["develop"] / "summary.json", "develop_summary_missing"
    )
    freeze._check_develop_summary(develop_summary, expected_units)
    prior = freeze._check_prior_bound(development_receipt)
    source_optimizer_steps = _check_source_optimizer_steps(develop_summary, development_receipt)
    stage = paths["selection"]
    selection_receipt = freeze._read_mapping(
        paths["selection_receipt"], "selection_receipt_missing"
    )
    receipt_plan_id, manifest_sha256, receipt_seconds = _check_selection_receipt(
        bundle, selection_receipt, prior
    )
    selection_summary = freeze._read_mapping(
        stage / SELECTION_SUMMARY_NAME, "selection_summary_missing"
    )
    summary_plan_id = _check_selection_summary(bundle, selection_summary, prior, receipt_seconds)
    if summary_plan_id != receipt_plan_id:
        raise _err("selection_plan_id_mismatch")
    freeze._check_deadline(deadline)
    freeze._verify_develop_manifest(development_receipt, paths["develop"])
    freeze._verify_pilot_manifest(bundle)
    _verify_selection_manifest(stage, manifest_sha256)
    freeze._check_deadline(deadline)
    freeze._verify_ledger(bundle, paths["develop"])
    freeze._verify_source_ledger(bundle, paths["develop"])
    selector_path = paths["develop"] / "selector.jsonl"
    _check_source_bindings(bundle, stage, paths["develop"], selector_path)
    freeze._check_deadline(deadline)
    records = freeze._authenticate_selector(
        bundle,
        freeze._read_jsonl(selector_path, "selector_missing"),
        paths["develop"],
        deadline,
    )
    freeze._check_deadline(deadline)
    plan = freeze._build_plan(bundle, records)
    stored_plan = freeze._read_mapping(stage / SELECTION_PLAN_NAME, "selection_plan_missing")
    _check_plan(plan, stored_plan, receipt_plan_id, selection_summary)
    freeze._check_deadline(deadline)
    return {
        "plan": plan,
        "prior_seconds": prior + receipt_seconds,
        "source_optimizer_steps": source_optimizer_steps,
        "development_receipt": development_receipt,
        "selection_receipt": selection_receipt,
    }
