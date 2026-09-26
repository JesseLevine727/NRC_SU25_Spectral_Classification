"""P05 comprehensive input preparation and pilot-result import boundary.

``prepare`` is the read-only metadata boundary for the owner-approved P05
comprehensive development run.  It authenticates the new comprehensive permit
by canonical digest, binds the frozen core contract and the rebuilt development
ledger, checks every frozen numerical file pin relative to the project root and
invokes the existing pilot ``prepare`` boundary in metadata-only mode so the 36
approved source-validation fits can be reused.  It never calls the pilot
``run``/``preflight`` boundary, never creates a lease and never writes.

``import_pilot`` re-authenticates the completed pilot run on disk: the pinned
raw run manifest digest, the exact recorded file inventory, the aggregate
summary and the 36 independent slot leases.  It then restores every
``DevelopmentFitResult`` from its private summary, logits NPZ and checkpoint
states and re-runs the pilot completion, history, shared-prefix and
sparse-equivalence checks against the restored results.  It returns
selection-ready records for the complete (including collapsed) pilot fits and
never reads outer-test predictions or metrics.

NumPy and torch are imported lazily; importing this module only touches the
standard library and the existing stdlib-only P05 modules.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_pilot as pilot
from atlas_sers.evaluation.p05_development_plan import (
    DevelopmentLedgerError,
    build_development_ledger,
)

__all__ = [
    "COMPREHENSIVE_PERMIT_SHA256",
    "ComprehensiveInputsError",
    "import_pilot",
    "load_development_result",
    "pilot_slot_ids",
    "prepare",
    "selector_record",
]

COMPREHENSIVE_SCHEMA_VERSION = "nato-sers-p05-comprehensive-v1"
COMPREHENSIVE_PERMIT_SHA256 = "2251916421ca2e94aa5d6acc2883439e6ac29b6872a49461c150d21603f128d8"

CORE_CONTRACT_SHA256 = "60e3a49753c59fb7038c83e50795614ad1cb4ca764dd487ac49692edcaf2ccae"
CORE_PLAN_ID = "a6334b2ed13a92fd953e4202bc2153e1aea4d12419d2a6f891f64f126136fe37"
LEDGER_ID = "P05DEV-8bf60eeca36d4b4663441eda"
PILOT_PERMIT_SHA256 = "652f5c07a1076a907778a9dd80394203ded9084298a95ce791cb5ee2814e576d"
PILOT_PLAN_ID = "7c57ba72a98ee018b30973dc35003ef9232babee7d8965fcbfb77b10ada6ca9c"
PILOT_MANIFEST_SHA256 = "847245673400310422cffbdf8f470eb56c5796d9b17a6b43ba24b68149592de1"

CONTEXT_COUNT = 320
UNIT_COUNT = 1245
INNER_SLOT_COUNT = 14940
REUSED_PILOT_SLOTS = 36

DEVELOPMENT_NAMESPACE = pilot.P05DEVELOPMENT_NAMESPACE
PILOT_PERMIT_RELATIVE = Path("plan") / "contracts" / "p05_development_pilot.json"
COMPREHENSIVE_CONTRACT_RELATIVE = Path("plan") / "contracts" / "p05_comprehensive.json"

SELECTOR_METRIC_FIELDS = (
    "best_epoch",
    "best_validation_balanced_accuracy",
    "best_validation_nll",
    "best_validation_macro_f1",
    "best_validation_predicted_class_count",
)

_UNPERSISTED_FIELD_NAMES = frozenset(
    {
        "state_dict",
        "best_state_dict",
        "terminal_state_dict",
        "validation_logits",
        "validation_uids",
        "classes",
    }
)


class ComprehensiveInputsError(core.P05CoreError):
    """Stable comprehensive-input failure with a path-free reason code."""


def _canon() -> Any:
    return core._canon()


def _pilot_run_dir(artifact_root: Path | str) -> Path:
    return Path(artifact_root) / DEVELOPMENT_NAMESPACE / "runs" / PILOT_PERMIT_SHA256


def _slot_lease_root(artifact_root: Path | str) -> Path:
    return (
        Path(artifact_root)
        / DEVELOPMENT_NAMESPACE
        / "slot_leases"
        / CORE_CONTRACT_SHA256
        / CORE_PLAN_ID
    )


def _execution_dir(run_dir: Path | str, unit: Mapping[str, Any], slot: Mapping[str, Any]) -> Path:
    return Path(run_dir) / "executions" / pilot.execution_id(unit, slot)


def _load_permit(permit_path: Path | str) -> tuple[dict[str, Any], str]:
    permit = core._read_json(Path(permit_path), "permit")
    if not isinstance(permit, Mapping):
        raise ComprehensiveInputsError("permit_malformed")
    observed = _canon().sha256_value(permit)
    if observed != COMPREHENSIVE_PERMIT_SHA256:
        raise ComprehensiveInputsError("permit_digest_mismatch")
    return dict(permit), observed


def _authorization(permit: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    return dict(permit)


def _check_authorization_pins(authorization: Mapping[str, Any]) -> None:
    if str(authorization.get("schema_version")) != COMPREHENSIVE_SCHEMA_VERSION:
        raise ComprehensiveInputsError("authorization_schema_mismatch")
    for key, expected in (
        ("core_contract_sha256", CORE_CONTRACT_SHA256),
        ("core_plan_id", CORE_PLAN_ID),
        ("ledger_id", LEDGER_ID),
        ("pilot_permit_sha256", PILOT_PERMIT_SHA256),
        ("pilot_plan_id", PILOT_PLAN_ID),
        ("pilot_manifest_sha256", PILOT_MANIFEST_SHA256),
    ):
        if str(authorization.get(key)) != expected:
            raise ComprehensiveInputsError("authorization_pin_mismatch")


def _check_frozen_files(project_root: Path, authorization: Mapping[str, Any]) -> None:
    frozen = authorization.get("frozen_numerical_files")
    if not isinstance(frozen, Mapping) or not frozen:
        raise ComprehensiveInputsError("frozen_numerical_files_malformed")
    canon = _canon()
    for relative, expected in frozen.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ComprehensiveInputsError("frozen_numerical_files_malformed")
        path = project_root / relative
        core._reject_symlink_chain(path)
        if not path.is_file():
            raise ComprehensiveInputsError("frozen_numerical_file_missing")
        if canon.sha256_file(path) != expected:
            raise ComprehensiveInputsError("frozen_numerical_file_mismatch")


def _build_ledger(
    plan: Mapping[str, Any], support: Any, contract: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return build_development_ledger(plan=plan, support=support, contract=contract)
    except DevelopmentLedgerError as error:
        raise ComprehensiveInputsError(f"development_ledger_{error.reason_code}") from error


def _check_ledger(ledger: Mapping[str, Any], authorization: Mapping[str, Any]) -> None:
    if str(ledger.get("ledger_id")) != LEDGER_ID:
        raise ComprehensiveInputsError("ledger_identity_mismatch")
    if ledger.get("execution_authorized") is not False:
        raise ComprehensiveInputsError("ledger_authorization_invalid")
    if ledger.get("arrays_loaded") is not False:
        raise ComprehensiveInputsError("ledger_arrays_invalid")
    if int(ledger.get("fits_started", -1)) != 0:
        raise ComprehensiveInputsError("ledger_fits_invalid")
    summary = ledger.get("summary")
    if not isinstance(summary, Mapping):
        raise ComprehensiveInputsError("ledger_summary_malformed")
    if int(summary.get("context_count", -1)) != int(authorization.get("context_count", -1)):
        raise ComprehensiveInputsError("ledger_context_count_mismatch")
    if len(ledger.get("units", ())) != int(authorization.get("unit_count", -1)):
        raise ComprehensiveInputsError("ledger_unit_count_mismatch")
    if len(ledger.get("slots", ())) != int(authorization.get("inner_slot_count", -1)):
        raise ComprehensiveInputsError("ledger_slot_count_mismatch")


def _check_pilot_bundle(pilot_bundle: Mapping[str, Any], authorization: Mapping[str, Any]) -> None:
    if str(pilot_bundle.get("pilot_plan_id")) != PILOT_PLAN_ID:
        raise ComprehensiveInputsError("pilot_plan_identity_mismatch")
    if str(pilot_bundle.get("contract_sha256")) != CORE_CONTRACT_SHA256:
        raise ComprehensiveInputsError("pilot_contract_identity_mismatch")
    ledger = pilot_bundle.get("ledger")
    if not isinstance(ledger, Mapping) or str(ledger.get("ledger_id")) != LEDGER_ID:
        raise ComprehensiveInputsError("pilot_ledger_identity_mismatch")
    slots = pilot_bundle.get("slots")
    if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
        raise ComprehensiveInputsError("pilot_slots_malformed")
    expected = int(authorization.get("reused_pilot_slots", -1))
    if len(slots) != expected or len(slots) != REUSED_PILOT_SLOTS:
        raise ComprehensiveInputsError("pilot_reused_slot_count_mismatch")


def _assert_no_foreign_leases(bundle: Mapping[str, Any]) -> None:
    root = _slot_lease_root(bundle["artifact_root"])
    core._reject_symlink_chain(root)
    if not root.is_dir():
        return
    reused = pilot_slot_ids(bundle)
    for entry in root.iterdir():
        core._reject_symlink_chain(entry)
        if entry.name not in reused:
            raise ComprehensiveInputsError("foreign_slot_lease_exists")


def prepare(
    project_root: Path | str,
    artifact_root: Path | str,
    contract_path: Path | str,
    permit_path: Path | str,
    *,
    require_unstarted: bool = True,
) -> dict[str, Any]:
    """Authenticate metadata and bind the full comprehensive unit/slot ledger."""

    project, artifact, repository_root = pilot._resolve_paths(project_root, artifact_root)
    permit, permit_digest = _load_permit(permit_path)
    authorization = _authorization(permit, project)
    _check_authorization_pins(authorization)
    _check_frozen_files(project, authorization)
    contract, contract_sha256 = core._load_contract(Path(contract_path), CORE_CONTRACT_SHA256)
    support, p01_run, _p04_run = core._authenticate(artifact, contract)
    core._manifest_uids(support, int(contract["population"]["rows"]))
    core_plan = core._build_plan(support, contract, project)
    core._minimal_plan_checks(core_plan, contract)
    core_plan_id = _canon().sha256_bytes(_canon().canonical_json_bytes(core_plan))
    if core_plan_id != CORE_PLAN_ID:
        raise ComprehensiveInputsError("core_plan_identity_mismatch")
    ledger = _build_ledger(core_plan, support, contract)
    _check_ledger(ledger, authorization)
    pilot_bundle = pilot.prepare(project, artifact, contract_path, project / PILOT_PERMIT_RELATIVE)
    _check_pilot_bundle(pilot_bundle, authorization)
    bundle = {
        "project_root": project,
        "artifact_root": artifact,
        "repository_root": repository_root,
        "permit": permit,
        "permit_sha256": permit_digest,
        "authorization": authorization,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "support": support,
        "p01_path": p01_run,
        "core_plan": core_plan,
        "core_plan_id": core_plan_id,
        "ledger": ledger,
        "units": list(ledger["units"]),
        "slots": list(ledger["slots"]),
        "pilot_bundle": pilot_bundle,
    }
    if require_unstarted:
        _assert_no_foreign_leases(bundle)
    return bundle


def pilot_slot_ids(bundle: Mapping[str, Any]) -> set[str]:
    """Return the canonical slot identifiers reused from the completed pilot."""

    return {str(slot["slot_id"]) for slot in bundle["pilot_bundle"]["slots"]}


def _read_pilot_summary(
    run_dir: Path | str, unit: Mapping[str, Any], slot: Mapping[str, Any]
) -> dict[str, Any]:
    summary = core._read_json(
        _execution_dir(run_dir, unit, slot) / "summary.json", "execution_summary"
    )
    if not isinstance(summary, Mapping):
        raise ComprehensiveInputsError("pilot_summary_malformed")
    return dict(summary)


def _load_logits(numpy: Any, path: Path) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    core._reject_symlink_chain(path)
    if not path.is_file():
        raise ComprehensiveInputsError("pilot_logits_missing")
    with numpy.load(path, allow_pickle=False) as data:
        if not {"logits", "classes", "uids"} <= set(data.files):
            raise ComprehensiveInputsError("pilot_logits_malformed")
        logits = numpy.asarray(data["logits"], dtype=numpy.float64)
        classes = tuple(str(item) for item in data["classes"].tolist())
        uids = tuple(str(item) for item in data["uids"].tolist())
    return logits, classes, uids


def _load_state(torch: Any, path: Path) -> dict[str, Any] | None:
    core._reject_symlink_chain(path)
    if not path.is_file():
        return None
    loaded = torch.load(path, weights_only=True, map_location="cpu")
    payload = loaded.get("state_dict") if isinstance(loaded, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ComprehensiveInputsError("pilot_checkpoint_malformed")
    return dict(payload)


def load_development_result(
    run_dir: Path | str, unit: Mapping[str, Any], slot: Mapping[str, Any]
) -> Any:
    """Restore one private ``DevelopmentFitResult`` from its persisted artifacts."""

    directory = _execution_dir(run_dir, unit, slot)
    summary = _read_pilot_summary(run_dir, unit, slot)
    module = importlib.import_module("atlas_sers.evaluation.p05_development")
    result_type = getattr(module, "DevelopmentFitResult", None)
    if result_type is None or not dataclasses.is_dataclass(result_type):
        raise ComprehensiveInputsError("development_result_type_unavailable")
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")
    logits, classes, uids = _load_logits(numpy, directory / "validation_logits.npz")
    best_state = _load_state(torch, directory / "best.pt")
    terminal_state = _load_state(torch, directory / "terminal.pt")
    runtime = importlib.import_module("atlas_sers.evaluation.p04_runtime")
    for label, state in (("best", best_state), ("terminal", terminal_state)):
        if state is None or runtime._state_hash(state) != summary.get(f"{label}_state_digest"):
            raise ComprehensiveInputsError("pilot_checkpoint_digest_mismatch")
    values: dict[str, Any] = {}
    for field in dataclasses.fields(result_type):
        name = field.name
        if name == "validation_logits":
            values[name] = logits
        elif name == "validation_uids":
            values[name] = uids
        elif name == "classes":
            values[name] = classes
        elif name == "best_state_dict":
            values[name] = best_state
        elif name == "terminal_state_dict":
            values[name] = terminal_state
        elif name == "state_dict":
            values[name] = best_state
        elif name in _UNPERSISTED_FIELD_NAMES:
            raise ComprehensiveInputsError("development_result_field_unavailable")
        elif name in summary:
            values[name] = summary[name]
        else:
            raise ComprehensiveInputsError("pilot_summary_field_missing")
    return result_type(**values)


def _read_history_jsonl(path: Path) -> list[dict[str, Any]]:
    core._reject_symlink_chain(path)
    records: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line.decode("utf-8"))
            if not isinstance(record, Mapping):
                raise ComprehensiveInputsError("pilot_history_malformed")
            records.append(dict(record))
    if not records:
        raise ComprehensiveInputsError("pilot_history_missing")
    return records


def _check_history_matches(
    run_dir: Path | str,
    unit: Mapping[str, Any],
    slot: Mapping[str, Any],
    result: Any,
) -> None:
    identifier = pilot.execution_id(unit, slot)
    saved = _read_history_jsonl(Path(run_dir) / "histories" / f"{identifier}.jsonl")
    history = list(result.history)
    if len(saved) != len(history):
        raise ComprehensiveInputsError("pilot_history_length_mismatch")
    for saved_record, observed in zip(saved, history, strict=True):
        if saved_record != dict(observed):
            raise ComprehensiveInputsError("pilot_history_mismatch")


def selector_record(
    unit: Mapping[str, Any], slot: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Return an identity-first selection record, never overwriting identity."""

    identity = {
        "slot_id": str(slot["slot_id"]),
        "context_id": str(unit["context_id"]),
        "selection_unit_id": str(unit["selection_unit_id"]),
        "slot_kind": str(slot["slot_kind"]),
        "fitting_role_id": str(unit["fitting_role_id"]),
        "validation_role_id": str(unit["validation_role_id"]),
        "recipe_id": str(slot["recipe_id"]),
        "seed": int(slot["seed"]),
    }
    for field, expected in identity.items():
        if field in summary and summary[field] != expected:
            raise ComprehensiveInputsError("selector_identity_mismatch")
        if field in slot and slot[field] != expected:
            raise ComprehensiveInputsError("selector_slot_unit_mismatch")
    for field, expected in (
        ("role_id", identity["fitting_role_id"]),
        ("recipe", identity["recipe_id"]),
    ):
        if field in summary and summary[field] != expected:
            raise ComprehensiveInputsError("selector_identity_mismatch")
    if "slot_id" in summary and str(summary["slot_id"]) != identity["slot_id"]:
        raise ComprehensiveInputsError("selector_identity_mismatch")
    if "recipe_id" in summary and str(summary["recipe_id"]) != identity["recipe_id"]:
        raise ComprehensiveInputsError("selector_identity_mismatch")
    if "unit_id" in summary and str(summary["unit_id"]) != str(unit["unit_id"]):
        raise ComprehensiveInputsError("selector_identity_mismatch")
    if "seed" in summary and int(summary["seed"]) != identity["seed"]:
        raise ComprehensiveInputsError("selector_identity_mismatch")
    status = str(summary.get("status"))
    record = dict(identity)
    record["status"] = status
    if status == "complete":
        for field in SELECTOR_METRIC_FIELDS:
            if field not in summary:
                raise ComprehensiveInputsError("selector_metric_missing")
            record[field] = summary[field]
    return record


def _check_pilot_run_manifest(run_dir: Path, contract: Mapping[str, Any]) -> None:
    manifest_path = run_dir / "manifest.json"
    core._reject_symlink_chain(manifest_path)
    if not manifest_path.is_file():
        raise ComprehensiveInputsError("pilot_manifest_missing")
    if _canon().sha256_file(manifest_path) != PILOT_MANIFEST_SHA256:
        raise ComprehensiveInputsError("pilot_manifest_digest_mismatch")
    if str(contract.get("pilot_manifest_sha256")) != PILOT_MANIFEST_SHA256:
        raise ComprehensiveInputsError("pilot_manifest_pin_mismatch")
    pilot._verify_manifest(run_dir)


def _check_pilot_run_summary(run_dir: Path) -> None:
    summary = core._read_json(run_dir / "summary.json", "pilot_summary")
    if not isinstance(summary, Mapping):
        raise ComprehensiveInputsError("pilot_summary_malformed")
    if str(summary.get("status")) != "complete":
        raise ComprehensiveInputsError("pilot_not_complete")
    for key, expected in (
        ("permit_sha256", PILOT_PERMIT_SHA256),
        ("core_contract_sha256", CORE_CONTRACT_SHA256),
        ("core_plan_id", CORE_PLAN_ID),
        ("pilot_plan_id", PILOT_PLAN_ID),
    ):
        if str(summary.get(key)) != expected:
            raise ComprehensiveInputsError("pilot_summary_identity_mismatch")
    for key in ("started", "completed"):
        if int(summary.get(key, -1)) != REUSED_PILOT_SLOTS:
            raise ComprehensiveInputsError("pilot_summary_count_mismatch")
    for key in ("failed", "unstarted"):
        if int(summary.get(key, -1)) != 0:
            raise ComprehensiveInputsError("pilot_summary_count_mismatch")


def _check_pilot_slot_leases(bundle: Mapping[str, Any]) -> None:
    root = _slot_lease_root(bundle["artifact_root"])
    for slot in bundle["pilot_bundle"]["slots"]:
        directory = root / str(slot["slot_id"])
        if not directory.is_dir():
            raise ComprehensiveInputsError("pilot_slot_lease_missing")
        lease = core._read_json(directory / "lease.json", "slot_lease")
        if not isinstance(lease, Mapping):
            raise ComprehensiveInputsError("pilot_slot_lease_malformed")
        for key, expected in (
            ("slot_id", str(slot["slot_id"])),
            ("unit_id", str(slot["unit_id"])),
            ("recipe_id", str(slot["recipe_id"])),
            ("contract_sha256", CORE_CONTRACT_SHA256),
            ("core_plan_id", CORE_PLAN_ID),
            ("permit_sha256", PILOT_PERMIT_SHA256),
        ):
            if str(lease.get(key)) != expected:
                raise ComprehensiveInputsError("pilot_slot_lease_mismatch")
        if int(lease.get("seed", -1)) != int(slot["seed"]):
            raise ComprehensiveInputsError("pilot_slot_lease_mismatch")


def import_pilot(bundle: Mapping[str, Any], *, device: str) -> list[dict[str, Any]]:
    """Re-authenticate the completed pilot and return complete selection records."""

    if device not in ("cpu", "cuda"):
        raise ComprehensiveInputsError("device_invalid")
    pilot_bundle = bundle["pilot_bundle"]
    contract = bundle["contract"]
    run_dir = _pilot_run_dir(bundle["artifact_root"])
    _check_pilot_run_manifest(run_dir, bundle["permit"])
    _check_pilot_run_summary(run_dir)
    _check_pilot_slot_leases(bundle)
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    core._configure_environment()
    importlib.import_module("atlas_sers.evaluation.p05_smoke")._configure_determinism()
    unit_inputs = pilot.prepare_role_inputs(pilot_bundle)
    units = list(pilot_bundle["units"])
    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    items: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for slot in pilot_bundle["slots"]:
        unit = unit_by_id[str(slot["unit_id"])]
        summary = _read_pilot_summary(run_dir, unit, slot)
        result = load_development_result(run_dir, unit, slot)
        _check_history_matches(run_dir, unit, slot, result)
        pilot.check_completed_result(
            result,
            run_dir,
            unit,
            slot,
            contract,
            unit_inputs[str(unit["unit_id"])],
            torch,
            device,
        )
        pilot.check_sparse_support(unit, slot, contract, result)
        record = selector_record(unit, slot, summary)
        if record["status"] != "complete":
            raise ComprehensiveInputsError("pilot_result_not_complete")
        records.append(record)
        items.append(
            {
                "slot": slot,
                "unit": unit,
                "unit_id": str(unit["unit_id"]),
                "seed": int(slot["seed"]),
                "result": result,
            }
        )
    pilot.check_shared_prefixes(items)
    pilot.check_sparse_equivalences(items, units, contract)
    return records
