"""P05 refit I/O: input preparation, private persistence and acceptance checks.

Wraps the frozen source-only refit kernel. ``prepare_refit_inputs`` authenticates
one content-addressed refit specification against the authoritative support
contexts, roles and manifest and exposes only outer source-fitting rows.
``persist_refit_result`` writes private, non-overwriting metadata and any
terminal checkpoint for every result, including partial failures.
``check_completed_refit`` re-verifies a completed refit from disk without
retraining and without any held prediction. No prediction, calibration, early
stopping or checkpoint selection happens here, and no file in the owning
per-refit execution directory is ever created twice.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation.p05_core_run import P05CoreError

OUTER_FIT_ROLE = "outer_fit"
OUTER_TEST_ROLE = "outer_test"
HELD_INSTRUMENT_SENTINELS = frozenset({"", "not_applicable"})
EXPECTED_CLASS_COUNT = 3
UPDATES_PER_EPOCH = 4
MINIMUM_REFIT_EPOCHS = 30
MAXIMUM_REFIT_EPOCHS = 200
LOSS_FIELDS = ("chemical_ce", "total_loss", "supcon_loss", "paired_loss")
BINDING_FIELDS = (
    "refit_id",
    "context_id",
    "fitting_role_id",
    "source_uid_set_sha256",
    "recipe_id",
    "seed",
    "epochs",
)


class P05RefitIOError(P05CoreError):
    """Stable refit I/O failure carrying a path-free reason code."""


def _canon() -> Any:
    return core._canon()


def _module(name: str) -> Any:
    return importlib.import_module(name)


def _require(condition: Any, code: str) -> None:
    if not condition:
        raise P05RefitIOError(code)


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def _text(value: Any, code: str) -> str:
    _require(isinstance(value, str) and bool(value) and value == value.strip(), code)
    return value


def _integer(value: Any, code: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), code)
    return value


def _string_list(value: Any, code: str) -> list[str]:
    _require(not isinstance(value, (str, bytes)) and isinstance(value, Sequence), code)
    _require(all(isinstance(item, str) for item in value), code)
    items = list(value)
    _require(bool(items) and all(item and item == item.strip() for item in items), code)
    return items


def _check_spec(spec: Any) -> dict[str, Any]:
    """Authenticate a refit spec identity and return its canonical fields."""

    _require(isinstance(spec, Mapping), "spec_malformed")
    context_id = _text(spec.get("context_id"), "spec_field_malformed")
    fitting_role_id = _text(spec.get("fitting_role_id"), "spec_field_malformed")
    source_uid_set_sha256 = _text(spec.get("source_uid_set_sha256"), "spec_field_malformed")
    recipe_id = _text(spec.get("recipe_id"), "spec_field_malformed")
    seed = _integer(spec.get("seed"), "spec_field_malformed")
    epochs = _integer(spec.get("epochs"), "spec_field_malformed")
    refit_id = _text(spec.get("refit_id"), "spec_field_malformed")
    permit_sha256 = _text(spec.get("permit_sha256"), "spec_field_malformed")
    comprehensive = _module("atlas_sers.evaluation.p05_comprehensive_inputs")
    _require(permit_sha256 == comprehensive.COMPREHENSIVE_PERMIT_SHA256, "spec_permit_mismatch")
    smoke = _module("atlas_sers.evaluation.p05_smoke")
    _require(recipe_id in smoke.RECIPE_SPECIFICATIONS, "spec_recipe_unknown")
    _require(MINIMUM_REFIT_EPOCHS <= epochs <= MAXIMUM_REFIT_EPOCHS, "spec_epochs_out_of_range")
    selection = _module("atlas_sers.evaluation.p05_selection")
    _require(seed in set(selection.SEEDS), "spec_seed_unknown")
    calibration = _string_list(spec.get("calibration_slot_ids"), "spec_calibration_slots_malformed")
    _require(calibration == sorted(set(calibration)), "spec_calibration_slots_malformed")
    fitting_uids = _string_list(spec.get("fitting_uids"), "spec_fitting_uids_malformed")
    _require(fitting_uids == sorted(set(fitting_uids)), "spec_fitting_uids_malformed")
    _require(
        _canon().sha256_value(fitting_uids) == source_uid_set_sha256, "source_uid_set_mismatch"
    )
    classes = _string_list(spec.get("classes"), "spec_classes_malformed")
    _require(
        len(classes) == EXPECTED_CLASS_COUNT and classes == sorted(set(classes)),
        "spec_classes_malformed",
    )
    identity = {
        "context_id": context_id,
        "fitting_role_id": fitting_role_id,
        "source_uid_set_sha256": source_uid_set_sha256,
        "recipe_id": recipe_id,
        "seed": seed,
        "epochs": epochs,
        "calibration_slot_ids": calibration,
        "permit_sha256": permit_sha256,
    }
    planner = _module("atlas_sers.evaluation.p05_refit_plan")
    _require(planner._sha256_canonical(identity) == refit_id, "refit_id_mismatch")
    return {
        "refit_id": refit_id,
        "context_id": context_id,
        "fitting_role_id": fitting_role_id,
        "source_uid_set_sha256": source_uid_set_sha256,
        "recipe_id": recipe_id,
        "seed": seed,
        "epochs": epochs,
        "permit_sha256": permit_sha256,
        "calibration_slot_ids": calibration,
        "fitting_uids": fitting_uids,
        "classes": classes,
    }


def _context_row(support: Any, context_id: str) -> Any:
    contexts = getattr(support, "contexts", None)
    _require(contexts is not None, "support_contexts_missing")
    matches = [row for row in contexts if str(row.get("context_id")) == context_id]
    _require(len(matches) == 1, "context_cardinality")
    return matches[0]


def _role_rows(
    support: Any, context_id: str, *, role_name: str | None = None, role_id: str | None = None
) -> list[Any]:
    rows = [
        row
        for row in support.roles
        if str(row.get("context_id")) == context_id
        and (role_name is None or str(row.get("role")) == role_name)
        and (role_id is None or str(row.get("role_id")) == role_id)
    ]
    _require(bool(rows), "outer_role_empty")
    return rows


def _role_id_for(support: Any, context_id: str, role_name: str) -> str:
    role_ids = {
        str(row.get("role_id")) for row in _role_rows(support, context_id, role_name=role_name)
    }
    _require(len(role_ids) == 1, "outer_role_cardinality")
    return role_ids.pop()


def _sorted_uids(rows: Sequence[Any]) -> list[str]:
    uids = [_text(row.get("observation_uid"), "role_uid_malformed") for row in rows]
    _require(len(set(uids)) == len(uids), "outer_role_uid_duplicate")
    return sorted(uids)


def _check_role_manifest(rows: Sequence[Any], manifest: Mapping[str, Mapping[str, str]]) -> None:
    for row in rows:
        uid = _text(row.get("observation_uid"), "role_uid_malformed")
        metadata = manifest.get(uid)
        _require(metadata is not None, "role_uid_unknown")
        _require(
            _text(row.get("master_sample_id"), "role_master_malformed") == metadata["master"],
            "role_master_mismatch",
        )
        _require(
            _text(row.get("instrument"), "role_instrument_malformed") == metadata["instrument"],
            "role_instrument_mismatch",
        )
        _require(
            _text(row.get("target_analyte"), "role_target_malformed") == metadata["target"],
            "role_target_mismatch",
        )


def prepare_refit_inputs(bundle: Any, spec: Any) -> dict[str, Any]:
    """Authenticate one refit spec and expose only outer source-fitting rows."""

    _require(isinstance(bundle, Mapping), "bundle_malformed")
    support = bundle.get("support")
    contract = bundle.get("contract")
    p01_run = bundle.get("p01_path")
    _require(
        support is not None and isinstance(contract, Mapping) and p01_run is not None,
        "bundle_incomplete",
    )
    spec = _check_spec(spec)
    context_id = spec["context_id"]
    context = _context_row(support, context_id)
    fitting_role_id = _role_id_for(support, context_id, OUTER_FIT_ROLE)
    _require(fitting_role_id == spec["fitting_role_id"], "outer_fit_role_mismatch")
    fit_rows = _role_rows(support, context_id, role_id=fitting_role_id)
    _require(
        {str(row.get("role")) for row in fit_rows} == {OUTER_FIT_ROLE}, "outer_fit_role_mismatch"
    )
    test_role_id = _role_id_for(support, context_id, OUTER_TEST_ROLE)
    test_rows = _role_rows(support, context_id, role_id=test_role_id)
    fitting_uids = _sorted_uids(fit_rows)
    test_uids = _sorted_uids(test_rows)
    _require(fitting_uids == spec["fitting_uids"], "spec_fitting_uids_mismatch")
    _require(
        _module("atlas_sers.evaluation.p05_refit_plan")._sha256_canonical(fitting_uids)
        == spec["source_uid_set_sha256"],
        "source_uid_set_mismatch",
    )
    manifest = _module("atlas_sers.evaluation.p05_pilot")._manifest_rows(support)
    _check_role_manifest(fit_rows, manifest)
    _check_role_manifest(test_rows, manifest)
    _require(
        {manifest[uid]["station"] for uid in fitting_uids} == {str(context["station"])},
        "source_station_mismatch",
    )
    fit_classes = sorted(
        {_text(row.get("target_analyte"), "role_target_malformed") for row in fit_rows}
    )
    _require(len(fit_classes) == EXPECTED_CLASS_COUNT, "source_class_count_mismatch")
    _require(fit_classes == spec["classes"], "source_class_order_mismatch")
    held_instrument = str(context.get("held_instrument", ""))
    fit_instruments = {manifest[uid]["instrument"] for uid in fitting_uids}
    _require(
        held_instrument in HELD_INSTRUMENT_SENTINELS or held_instrument not in fit_instruments,
        "held_instrument_in_source",
    )
    fit_masters = {_text(row.get("master_sample_id"), "role_master_malformed") for row in fit_rows}
    test_masters = {
        _text(row.get("master_sample_id"), "role_master_malformed") for row in test_rows
    }
    _require(not (set(fitting_uids) & set(test_uids)), "source_test_uid_overlap")
    _require(not (fit_masters & test_masters), "source_test_master_overlap")
    expected_rows = int(contract["population"]["rows"])
    manifest_uids = core._manifest_uids(support, expected_rows)
    intensity, labels = core._load_representation(
        Path(p01_run) / core.REPRESENTATION_REL,
        contract["input_pins"]["representation_sha256"],
        manifest_uids,
        expected_rows,
    )
    uid_index = {uid: index for index, uid in enumerate(labels)}
    try:
        values = intensity[[uid_index[uid] for uid in fitting_uids]]
    except KeyError as error:
        raise P05RefitIOError("refit_uid_missing_representation") from error
    observation_type = _module("atlas_sers.evaluation.p05_sampling").Observation
    observations = [
        _module("atlas_sers.evaluation.p05_pilot")._observation(manifest, uid, observation_type)
        for uid in fitting_uids
    ]
    noise_metadata = core._noise_frame(
        Path(p01_run) / "primary_manifest.csv",
        contract["input_pins"]["manifest_sha256"],
        fitting_uids,
        _module("pandas"),
    )
    return {
        "values": values,
        "observations": observations,
        "noise_metadata": noise_metadata,
        "role_id": fitting_role_id,
        "recipe": spec["recipe_id"],
        "seed": spec["seed"],
        "epochs": spec["epochs"],
    }


def _guard_run_dir(run_dir: Any) -> Path:
    run_dir = Path(run_dir)
    _require(not run_dir.is_symlink(), "run_directory_symlink")
    core._reject_symlink_chain(run_dir)
    _require(run_dir.is_dir(), "run_directory_missing")
    return run_dir


def _write_new(path: Path, content: bytes) -> None:
    core._reject_symlink_chain(path.parent)
    _require(not (path.exists() or path.is_symlink()), "output_exists")
    core._atomic_write(path, content)


def _result_summary(result: Any) -> dict[str, Any]:
    _require(dataclasses.is_dataclass(result), "result_not_dataclass")
    summary: dict[str, Any] = {}
    for field in dataclasses.fields(result):
        if field.name == "terminal_state_dict":
            continue
        value = getattr(result, field.name)
        summary[field.name] = list(value) if isinstance(value, tuple) else value
    return summary


def _binding(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {name: spec[name] for name in BINDING_FIELDS}


def _check_result_identity(spec: Mapping[str, Any], result: Any) -> None:
    _require(
        str(getattr(result, "role_id", None)) == spec["fitting_role_id"], "refit_role_mismatch"
    )
    _require(str(getattr(result, "recipe", None)) == spec["recipe_id"], "refit_recipe_mismatch")
    _require(_as_int(getattr(result, "seed", None)) == spec["seed"], "refit_seed_mismatch")
    _require(
        _as_int(getattr(result, "epochs", None)) == spec["epochs"], "refit_epoch_budget_mismatch"
    )


def _persist_terminal(torch: Any, run_dir: Path, result: Any) -> dict[str, Any] | None:
    state = getattr(result, "terminal_state_dict", None)
    if state is None:
        return None
    path = run_dir / "terminal.pt"
    _require(not (path.exists() or path.is_symlink()), "terminal_checkpoint_exists")
    core._save_state(torch, state, path)
    loaded = torch.load(path, weights_only=True, map_location="cpu")
    payload = loaded.get("state_dict") if isinstance(loaded, Mapping) else None
    _require(
        isinstance(payload, Mapping) and set(payload) == set(state),
        "terminal_checkpoint_reload_mismatch",
    )
    runtime = _module("atlas_sers.evaluation.p04_runtime")
    observed = runtime._state_hash(dict(payload))
    expected = getattr(result, "terminal_state_digest", None)
    _require(expected is not None and observed == expected, "terminal_digest_mismatch")
    return {"file_sha256": _canon().sha256_file(path), "state_sha256": observed}


def persist_refit_result(torch: Any, run_dir: Any, spec: Any, result: Any) -> dict[str, Any]:
    """Persist private metadata and any terminal checkpoint for any result."""

    run_dir = _guard_run_dir(run_dir)
    spec = _check_spec(spec)
    _require(dataclasses.is_dataclass(result), "result_not_dataclass")
    _check_result_identity(spec, result)
    summary = _result_summary(result)
    binding = _binding(spec)
    for name, value in binding.items():
        if name in summary:
            _require(summary[name] == value, "refit_identity_conflict")
    summary.update(binding)
    _write_new(run_dir / "summary.json", _canon().canonical_json_bytes(summary))
    terminal = _persist_terminal(torch, run_dir, result)
    return {
        "refit_id": spec["refit_id"],
        "context_id": spec["context_id"],
        "recipe_id": spec["recipe_id"],
        "seed": spec["seed"],
        "epochs": spec["epochs"],
        "status": str(getattr(result, "status", "unknown")),
        "reason_code": getattr(result, "reason_code", None),
        "epochs_completed": _as_int(getattr(result, "epochs_completed", None)),
        "optimizer_steps": _as_int(getattr(result, "optimizer_steps", None)),
        "terminal": terminal,
        "summary_sha256": _canon().sha256_file(run_dir / "summary.json"),
    }


def _expected_parameter_count(recipe_id: str) -> int:
    smoke = _module("atlas_sers.evaluation.p05_smoke")
    _require(recipe_id in smoke.RECIPE_SPECIFICATIONS, "spec_recipe_unknown")
    _supcon, _pair, use_projection = smoke.RECIPE_SPECIFICATIONS[recipe_id]
    acquisition = _module("atlas_sers.models.acquisition")
    if use_projection:
        return int(acquisition.PROJECTION_MODEL_PARAMETERS)
    return int(acquisition.BASE_PARAMETERS)


def _check_history(result: Any, epochs: int) -> None:
    history = getattr(result, "history", None)
    _require(
        not isinstance(history, (str, bytes)) and isinstance(history, Sequence),
        "refit_history_malformed",
    )
    _require(len(history) == epochs, "refit_history_length_mismatch")
    for index, record in enumerate(history, start=1):
        _require(isinstance(record, Mapping), "refit_history_record_malformed")
        _require(_as_int(record.get("epoch")) == index, "refit_history_epoch_mismatch")
        _require(
            _as_int(record.get("epoch_optimizer_steps")) == UPDATES_PER_EPOCH,
            "refit_history_epoch_steps_mismatch",
        )
        _require(
            _as_int(record.get("total_optimizer_steps")) == index * UPDATES_PER_EPOCH,
            "refit_history_steps_mismatch",
        )
        for name in LOSS_FIELDS:
            value = record.get(name)
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                "refit_history_loss_non_finite",
            )


def _verify_terminal(torch: Any, run_dir: Path, result: Any) -> str:
    expected = getattr(result, "terminal_state_digest", None)
    _require(expected is not None, "refit_terminal_digest_missing")
    path = run_dir / "terminal.pt"
    core._reject_symlink_chain(path)
    _require(path.is_file() and not path.is_symlink(), "refit_terminal_checkpoint_missing")
    loaded = torch.load(path, weights_only=True, map_location="cpu")
    payload = loaded.get("state_dict") if isinstance(loaded, Mapping) else None
    _require(isinstance(payload, Mapping), "refit_terminal_reload_mismatch")
    runtime = _module("atlas_sers.evaluation.p04_runtime")
    observed = runtime._state_hash(dict(payload))
    _require(observed == expected, "refit_terminal_digest_mismatch")
    state = getattr(result, "terminal_state_dict", None)
    if state is not None:
        _require(runtime._state_hash(state) == expected, "refit_terminal_state_digest_mismatch")
    return observed


def check_completed_refit(torch: Any, run_dir: Any, spec: Any, result: Any) -> dict[str, Any]:
    """Verify one completed refit from disk without retraining or held data."""

    run_dir = _guard_run_dir(run_dir)
    spec = _check_spec(spec)
    _require(str(getattr(result, "status", None)) == "complete", "refit_not_complete")
    _require(getattr(result, "reason_code", None) is None, "refit_reason_code_present")
    _require(getattr(result, "state_capture_failed", True) is False, "refit_state_capture_failed")
    _check_result_identity(spec, result)
    epochs = spec["epochs"]
    _require(
        _as_int(getattr(result, "epochs_completed", None)) == epochs,
        "refit_epochs_completed_mismatch",
    )
    _require(
        _as_int(getattr(result, "optimizer_steps", None)) == epochs * UPDATES_PER_EPOCH,
        "refit_optimizer_steps_mismatch",
    )
    _require(
        _as_int(getattr(result, "finite_gradient_batches", None)) == epochs * UPDATES_PER_EPOCH,
        "refit_finite_gradient_mismatch",
    )
    _require(
        _as_int(getattr(result, "parameter_count", None))
        == _expected_parameter_count(spec["recipe_id"]),
        "refit_parameter_count_mismatch",
    )
    _require(list(getattr(result, "classes", ())) == spec["classes"], "refit_class_order_mismatch")
    _check_history(result, epochs)
    expected_summary = _result_summary(result)
    expected_summary.update(_binding(spec))
    saved = core._read_json(run_dir / "summary.json", "refit_summary")
    _require(isinstance(saved, Mapping), "refit_summary_malformed")
    _require(dict(saved) == expected_summary, "refit_summary_mismatch")
    observed = _verify_terminal(torch, run_dir, result)
    return {
        "refit_id": spec["refit_id"],
        "status": "complete",
        "recipe_id": spec["recipe_id"],
        "seed": spec["seed"],
        "epochs": epochs,
        "epochs_completed": epochs,
        "optimizer_steps": epochs * UPDATES_PER_EPOCH,
        "terminal_state_digest": observed,
    }


__all__ = [
    "P05RefitIOError",
    "check_completed_refit",
    "persist_refit_result",
    "prepare_refit_inputs",
]
