"""Compose authenticated P08 U0 source arrays with frozen kernel specifications.

This module is a fixed composition boundary that sits above
:func:`atlas_sers.evaluation.p08_u0_arrays.prepare_u0_source_arrays`.  It

* authenticates the caller-supplied specification-audit and candidate-registry
  byte strings against fixed module pins before any JSON/CSV parsing or source
  array preparation,
* hands the exact authenticated metadata/action snapshots to the inherited
  source-array binder exactly once,
* checks each recorded source-fit / source-prediction job against the frozen
  model-specification hashes and frozen first classical candidate records,
* preserves the recorded original ``sensor_family`` substrate strings, and
* exposes thin constructors that return the exact inherited kernel keyword
  dictionaries for the future controller.

Non-claims
----------
* ``specification_bytes_verified``, ``kernel_arguments_prepared`` and
  ``substrate_metadata_preserved`` are internal consistency statements about
  the supplied snapshots and the module's own recorded pins only.  They are not
  provenance, runtime parity, physical-isolation or execution evidence.
* The caller owns and controls every supplied input dictionary for the whole
  capture window; this module retains no filesystem handles and performs no
  explicit source or artifact filesystem reads.
* ``classical_kwargs`` lazily imports the inherited
  :class:`atlas_sers.evaluation.p03_runtime.P03Dataset`; Python imports may
  read from the filesystem, so this module makes no promise that importing
  runtime code never touches the filesystem.
* Imported runtime code is not re-hashed or re-authenticated here; the
  source byte pins from the authenticated audit identify the supplied strings, not
  already-imported runtime code, and the flag
  ``loaded_runtime_code_verified`` is always false.
* Frozen inputs are not execution capabilities and cannot authorize any
  scientific operation.
* No fitting, prediction, calibration, quantile computation, augmentation or
  model construction happens here.
* Supplied source byte strings are never executed and no supplied source path
  is opened.
* ``require_scientific_execution`` always denies execution.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p08_plan as _plan
from atlas_sers.evaluation import p08_u0_arrays as _arrays
from atlas_sers.evaluation import p08_u0_inputs as _u0_inputs
from atlas_sers.evaluation.p05_sampling import Observation as _SamplerObservation
from atlas_sers.governance.canonical import sha256_value

SCHEMA_VERSION = "nato-sers-p08-u0-runtime-inputs-v1"

_SPEC_AUDIT_SHA256 = "7a6fdd91b33ae63a212c7936599e38b28cb0e88273024ed28f4bf0ec7d42c1a0"
_CANDIDATES_SHA256 = "046ebfa9023591ba91f48b797fe3bb037f6e7e82cb4bfb34800a9aa12b468bba"

_AUDIT_SCHEMA = "nato-sers-p08-universal-ledger-audit-v1"
_AUDIT_KEYS = (
    "schema_version",
    "execution_authorized",
    "plan_sha256",
    "model_specification_sha256",
    "specification_inputs",
)
_KNOWN_MODELS = frozenset(
    {"C-EXTRA-TREES", "C-RANDOM-FOREST", "C-RBF-SVM", "D0-M", "D1", "D2", "D3"}
)
_NEURAL_MODELS = frozenset({"D0-M", "D1", "D2", "D3"})
_SPECIFICATION_SOURCE_COUNT = 18
_CORE_CONTRACT_PATH = "plan/contracts/p05_core_contract.json"
_NEURAL_CANDIDATE_ID = "fixed_recipe"

_METADATA_KEYS = frozenset(
    {
        "proposal_bytes",
        "attempt_manifest_bytes",
        "manifest_bytes",
        "contexts_bytes",
        "roles_bytes",
    }
)
_ACTION_KEYS = frozenset({"R_MIN_400_1800", "R_SG_400_1800", "R_ARPLS_400_1800"})
_CANDIDATE_COLUMNS = (
    "candidate_id",
    "model_id",
    "declared_candidate_order",
    "parameters_json",
    "hyperparameter_sha256",
)

_MAXIMUM_API_BYTES = 1024 * 1024
_MAXIMUM_SOURCE_TOTAL_BYTES = 4 * 1024 * 1024
_MAXIMUM_CANDIDATE_ROWS = 1000
_MAXIMUM_FIT_SECONDS = 120
_MAXIMUM_CUDA_ALLOCATED_BYTES = 4294967296

_REASON_CODES = frozenset(
    {
        "scientific_execution_not_authorized",
        "invalid_arguments",
        "invalid_metadata_bytes",
        "invalid_action_bytes",
        "invalid_specification_audit_bytes",
        "invalid_candidate_registry_bytes",
        "invalid_specification_source_bytes",
        "audit_hash_mismatch",
        "candidate_registry_hash_mismatch",
        "invalid_audit",
        "model_spec_mismatch",
        "specification_source_mismatch",
        "invalid_candidate_registry",
        "candidate_mismatch",
        "missing_sensor_family",
        "invalid_substrate_metadata",
        "input_preparation_failed",
        "wrong_kernel",
        "unlisted_reason_code",
    }
)

__all__ = [
    "SCHEMA_VERSION",
    "InputError",
    "RuntimeInputs",
    "RuntimePair",
    "prepare_u0_runtime_inputs",
    "require_scientific_execution",
]


class InputError(ValueError):
    """ValueError carrying one static allowlisted reason code."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "unlisted_reason_code"
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(reason_code):
    raise InputError(reason_code) from None


def _copy_api_bytes(value, reason_code):
    if type(value) is not bytes or len(value) == 0 or len(value) > _MAXIMUM_API_BYTES:
        _fail(reason_code)
    return value


def _copy_metadata(metadata_bytes):
    if type(metadata_bytes) is not dict or len(metadata_bytes) != len(_METADATA_KEYS):
        _fail("invalid_metadata_bytes")
    copied = {}
    for key in _METADATA_KEYS:
        if key not in metadata_bytes:
            _fail("invalid_metadata_bytes")
        value = metadata_bytes[key]
        if type(value) is not bytes:
            _fail("invalid_metadata_bytes")
        copied[key] = value
    return copied


def _copy_action_bytes(action_bytes):
    if type(action_bytes) is not dict or len(action_bytes) != len(_ACTION_KEYS):
        _fail("invalid_action_bytes")
    copied = {}
    for representation_id in _ACTION_KEYS:
        if representation_id not in action_bytes:
            _fail("invalid_action_bytes")
        value = action_bytes[representation_id]
        if type(value) is not bytes or len(value) == 0:
            _fail("invalid_action_bytes")
        copied[representation_id] = value
    return copied


def _parse_json_object(raw, reason_code):
    try:
        value = json.loads(raw.decode("utf-8"))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail(reason_code)
    if type(value) is not dict:
        _fail(reason_code)
    return value


def _validate_audit(audit):
    if type(audit) is not dict:
        _fail("invalid_audit")
    for key in _AUDIT_KEYS:
        if key not in audit:
            _fail("invalid_audit")
    if audit["schema_version"] != _AUDIT_SCHEMA:
        _fail("invalid_audit")
    if audit["execution_authorized"] is not False:
        _fail("invalid_audit")
    if audit["plan_sha256"] != _u0_inputs._PARENT_PLAN_SHA256:
        _fail("invalid_audit")

    specification_hashes = audit["model_specification_sha256"]
    specification_inputs = audit["specification_inputs"]
    if type(specification_hashes) is not dict or type(specification_inputs) is not dict:
        _fail("invalid_audit")
    if set(specification_hashes) != _KNOWN_MODELS or set(specification_inputs) != _KNOWN_MODELS:
        _fail("model_spec_mismatch")

    source_pins = {}
    for model_id in _KNOWN_MODELS:
        entry = specification_inputs[model_id]
        if type(entry) is not dict or set(entry) != {"model_id", "inherited_sources"}:
            _fail("model_spec_mismatch")
        if entry["model_id"] != model_id:
            _fail("model_spec_mismatch")
        inherited = entry["inherited_sources"]
        if type(inherited) is not dict or not inherited:
            _fail("model_spec_mismatch")
        recorded = specification_hashes[model_id]
        if type(recorded) is not str or sha256_value(entry) != recorded:
            _fail("model_spec_mismatch")
        for path, pin in inherited.items():
            if type(path) is not str or not path or type(pin) is not str or not pin:
                _fail("model_spec_mismatch")
            previous = source_pins.get(path)
            if previous is None:
                source_pins[path] = pin
            elif previous != pin:
                _fail("specification_source_mismatch")
    if len(source_pins) != _SPECIFICATION_SOURCE_COUNT:
        _fail("model_spec_mismatch")
    return specification_hashes, source_pins


def _copy_specification_sources(specification_source_bytes, source_pins):
    if type(specification_source_bytes) is not dict:
        _fail("invalid_specification_source_bytes")
    if len(specification_source_bytes) != len(source_pins):
        _fail("invalid_specification_source_bytes")
    if set(specification_source_bytes) != set(source_pins):
        _fail("invalid_specification_source_bytes")
    copied = {}
    total = 0
    for path in source_pins:
        value = specification_source_bytes[path]
        if type(value) is not bytes or len(value) == 0:
            _fail("invalid_specification_source_bytes")
        total += len(value)
        copied[path] = value
    if total > _MAXIMUM_SOURCE_TOTAL_BYTES:
        _fail("invalid_specification_source_bytes")
    for path, pin in source_pins.items():
        if hashlib.sha256(copied[path]).hexdigest() != pin:
            _fail("specification_source_mismatch")
    return copied


def _read_candidate_registry(raw):
    try:
        text = raw.decode("utf-8-sig")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_candidate_registry")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = reader.fieldnames
    if not header or len(header) != len(set(header)):
        _fail("invalid_candidate_registry")
    for column in _CANDIDATE_COLUMNS:
        if column not in header:
            _fail("invalid_candidate_registry")
    records = []
    seen_ids = set()
    try:
        for row in reader:
            if len(records) >= _MAXIMUM_CANDIDATE_ROWS:
                _fail("invalid_candidate_registry")
            if None in row:
                _fail("invalid_candidate_registry")
            for value in row.values():
                if value is None:
                    _fail("invalid_candidate_registry")
            record = {column: row[column] for column in _CANDIDATE_COLUMNS}
            candidate_id = record["candidate_id"]
            if type(candidate_id) is not str or not candidate_id or candidate_id in seen_ids:
                _fail("invalid_candidate_registry")
            seen_ids.add(candidate_id)
            records.append(record)
    except InputError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_candidate_registry")
    return records


def _parse_declared_order(token):
    if type(token) is not str or not token:
        _fail("invalid_candidate_registry")
    if not token.isascii() or not token.isdigit():
        _fail("invalid_candidate_registry")
    if len(token) > 1 and token[0] == "0":
        _fail("invalid_candidate_registry")
    return int(token)


def _select_classical_candidates(records):
    grouped = {}
    for record in records:
        model_id = record["model_id"]
        if model_id not in _plan.CLASSICAL_MODELS:
            continue
        order = _parse_declared_order(record["declared_candidate_order"])
        grouped.setdefault(model_id, []).append((order, record))

    selected = {}
    for model_id in _plan.CLASSICAL_MODELS:
        candidates = grouped.get(model_id)
        if not candidates:
            _fail("candidate_mismatch")
        candidates.sort(key=lambda item: item[0])
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            _fail("candidate_mismatch")
        record = candidates[0][1]
        try:
            parameters = json.loads(record["parameters_json"])
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("invalid_candidate_registry")
        if type(parameters) is not dict:
            _fail("invalid_candidate_registry")
        if sha256_value(parameters) != record["hyperparameter_sha256"]:
            _fail("candidate_mismatch")
        selected[model_id] = {
            "candidate_id": record["candidate_id"],
            "hyperparameter_sha256": record["hyperparameter_sha256"],
            "parameters_json": json.dumps(
                parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
        }
    return selected


def _read_neural_contract(source_copy):
    raw = source_copy.get(_CORE_CONTRACT_PATH)
    if type(raw) is not bytes or not raw:
        _fail("specification_source_mismatch")
    contract = _parse_json_object(raw, "invalid_specification_source_bytes")
    recipes = contract.get("recipes")
    if type(recipes) is not list:
        _fail("model_spec_mismatch")
    recipes_by_id = {}
    for record in recipes:
        if type(record) is not dict:
            _fail("model_spec_mismatch")
        recipe_id = record.get("recipe_id")
        if type(recipe_id) is not str or recipe_id in recipes_by_id:
            _fail("model_spec_mismatch")
        recipes_by_id[recipe_id] = record
    if len(recipes) != len(_NEURAL_MODELS) or set(recipes_by_id) != _NEURAL_MODELS:
        _fail("model_spec_mismatch")

    later = contract.get("later_core_plan")
    if type(later) is not dict:
        _fail("model_spec_mismatch")
    minimum_epochs = later.get("minimum_epochs")
    maximum_epochs = later.get("maximum_epochs")
    patience = later.get("patience")
    seeds = later.get("seeds")
    if minimum_epochs != 30 or maximum_epochs != 200 or patience != 20:
        _fail("model_spec_mismatch")
    if type(seeds) is not list:
        _fail("model_spec_mismatch")
    for seed in seeds:
        if type(seed) is not int:
            _fail("model_spec_mismatch")
    if seeds != list(_plan.SEEDS):
        _fail("model_spec_mismatch")

    smoke = contract.get("smoke")
    if type(smoke) is not dict:
        _fail("model_spec_mismatch")
    maximum_fit_seconds = smoke.get("maximum_fit_seconds")
    maximum_cuda_allocated_bytes = smoke.get("maximum_cuda_allocated_bytes")
    if type(maximum_fit_seconds) is not int or maximum_fit_seconds != _MAXIMUM_FIT_SECONDS:
        _fail("model_spec_mismatch")
    if (
        type(maximum_cuda_allocated_bytes) is not int
        or maximum_cuda_allocated_bytes != _MAXIMUM_CUDA_ALLOCATED_BYTES
    ):
        _fail("model_spec_mismatch")

    for section in ("model", "sampler", "objective", "optimization"):
        if type(contract.get(section)) is not dict:
            _fail("model_spec_mismatch")
    return {
        "recipes": recipes_by_id,
        "model": contract["model"],
        "sampler": contract["sampler"],
        "objective": contract["objective"],
        "optimization": contract["optimization"],
        "minimum_epochs": minimum_epochs,
        "maximum_epochs": maximum_epochs,
        "patience": patience,
        "maximum_fit_seconds": maximum_fit_seconds,
        "maximum_cuda_allocated_bytes": maximum_cuda_allocated_bytes,
    }


def _read_substrate_metadata(manifest_bytes):
    if type(manifest_bytes) is not bytes or not manifest_bytes:
        _fail("invalid_substrate_metadata")
    try:
        text = manifest_bytes.decode("utf-8-sig")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_substrate_metadata")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = reader.fieldnames
    if not header:
        _fail("invalid_substrate_metadata")
    if "observation_uid" not in header:
        _fail("invalid_substrate_metadata")
    if "sensor_family" not in header:
        _fail("missing_sensor_family")
    mapping = {}
    try:
        for row in reader:
            if None in row:
                _fail("invalid_substrate_metadata")
            uid = row["observation_uid"]
            substrate = row["sensor_family"]
            if type(uid) is not str or not uid or substrate is None:
                _fail("invalid_substrate_metadata")
            if uid in mapping:
                _fail("invalid_substrate_metadata")
            mapping[uid] = substrate
    except InputError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_substrate_metadata")
    return mapping


def _build_observations(source_roles, substrate_map):
    def convert(records):
        converted = []
        for record in records:
            uid = record.observation_uid
            substrate = substrate_map.get(uid)
            if substrate is None:
                _fail("invalid_substrate_metadata")
            converted.append(
                _SamplerObservation(
                    uid=uid,
                    master=record.master_sample_id,
                    station=record.station,
                    target=record.target_analyte,
                    instrument=record.instrument,
                    substrate=substrate,
                )
            )
        return tuple(converted)

    return convert(source_roles.fitting), convert(source_roles.validation)


def _parse_job_json(job_json):
    if type(job_json) is not str or not job_json:
        _fail("input_preparation_failed")
    try:
        job = json.loads(job_json)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("input_preparation_failed")
    if type(job) is not dict:
        _fail("input_preparation_failed")
    return job


def _kernel_for_job(job, specification_hashes, classical_candidates, neural_contract):
    if type(job) is not dict:
        _fail("candidate_mismatch")
    model_id = job.get("model_id")
    recorded = specification_hashes.get(model_id)
    if type(recorded) is not str:
        _fail("model_spec_mismatch")
    if job.get("model_spec_sha256") != recorded:
        _fail("model_spec_mismatch")
    candidate_id = job.get("candidate_id")
    hyperparameter_sha256 = job.get("hyperparameter_sha256")

    if model_id in _plan.CLASSICAL_MODELS:
        candidate = classical_candidates.get(model_id)
        if candidate is None:
            _fail("candidate_mismatch")
        if candidate_id != candidate["candidate_id"]:
            _fail("candidate_mismatch")
        if hyperparameter_sha256 != candidate["hyperparameter_sha256"]:
            _fail("candidate_mismatch")
        configuration = {
            "model_id": model_id,
            "model_spec_sha256": recorded,
            "candidate_id": candidate["candidate_id"],
            "hyperparameter_sha256": candidate["hyperparameter_sha256"],
            "parameters": json.loads(candidate["parameters_json"]),
            "threads": 1,
        }
        return "classical", configuration, candidate["parameters_json"]

    if model_id in _NEURAL_MODELS:
        if candidate_id != _NEURAL_CANDIDATE_ID:
            _fail("candidate_mismatch")
        if hyperparameter_sha256 != recorded:
            _fail("candidate_mismatch")
        recipe = neural_contract["recipes"].get(model_id)
        if type(recipe) is not dict:
            _fail("model_spec_mismatch")
        configuration = {
            "model_id": model_id,
            "model_spec_sha256": recorded,
            "recipe": recipe,
            "model": neural_contract["model"],
            "sampler": neural_contract["sampler"],
            "objective": neural_contract["objective"],
            "optimization": neural_contract["optimization"],
            "stopping": {
                "minimum_epochs": neural_contract["minimum_epochs"],
                "maximum_epochs": neural_contract["maximum_epochs"],
                "patience": neural_contract["patience"],
            },
            "maximum_fit_seconds": neural_contract["maximum_fit_seconds"],
            "maximum_cuda_allocated_bytes": neural_contract["maximum_cuda_allocated_bytes"],
        }
        return "neural", configuration, None

    _fail("model_spec_mismatch")


def _job_field(job, name):
    value = job.get(name)
    if value is None:
        _fail("input_preparation_failed")
    return value


@dataclass(frozen=True, repr=False)
class RuntimePair:
    """One prepared U0 source pair plus its frozen kernel configuration."""

    prepared_pair: _arrays.PreparedPair
    parameters_json: str | None
    configuration_json: str
    fitting_observations: tuple
    validation_observations: tuple

    def configuration(self):
        return json.loads(self.configuration_json)

    def classical_kwargs(self):
        configuration = self.configuration()
        if configuration.get("model_id") not in _plan.CLASSICAL_MODELS:
            _fail("wrong_kernel")
        prepared = self.prepared_pair
        source_roles = prepared.inputs.source_roles
        fit_job = _parse_job_json(prepared.fit_job_json)
        from atlas_sers.evaluation.p03_runtime import P03Dataset

        fitting_values = prepared.inputs.fitting_values()
        validation_values = prepared.inputs.validation_values()
        observations = list(self.fitting_observations) + list(self.validation_observations)
        intensity = np.concatenate([fitting_values, validation_values], axis=0)
        metadata = pd.DataFrame(
            {
                "observation_uid": [row.uid for row in observations],
                "master_sample_id": [row.master for row in observations],
                "target_analyte": [row.target for row in observations],
                "instrument": [row.instrument for row in observations],
                "station": [row.station for row in observations],
            }
        )
        dataset = P03Dataset.from_frozen_representation(
            intensity=intensity,
            representation_uids=np.asarray([row.uid for row in observations]),
            metadata=metadata,
        )
        return {
            "dataset": dataset,
            "fit_id": _job_field(fit_job, "job_id"),
            "model_id": configuration["model_id"],
            "candidate_id": configuration["candidate_id"],
            "parameters": json.loads(self.parameters_json),
            "seed": _job_field(fit_job, "seed"),
            "fit_uids": [row.uid for row in self.fitting_observations],
            "validation_uids": [row.uid for row in self.validation_observations],
            "class_vocabulary": tuple(source_roles.classes),
            "expected_fit_uid_sha256": _job_field(fit_job, "fit_uid_sha256"),
            "expected_validation_uid_sha256": _job_field(fit_job, "validation_uid_sha256"),
        }

    def neural_kwargs(self):
        configuration = self.configuration()
        if configuration.get("model_id") not in _NEURAL_MODELS:
            _fail("wrong_kernel")
        prepared = self.prepared_pair
        source_roles = prepared.inputs.source_roles
        fit_job = _parse_job_json(prepared.fit_job_json)
        return {
            "values": prepared.inputs.fitting_values(),
            "observations": list(self.fitting_observations),
            "noise_metadata": prepared.inputs.fitting_noise_frame(),
            "validation_values": prepared.inputs.validation_values(),
            "validation_observations": list(self.validation_observations),
            "role_id": source_roles.fitting_role_id,
            "recipe": _job_field(fit_job, "model_id"),
            "seed": _job_field(fit_job, "seed"),
            "maximum_fit_seconds": float(_MAXIMUM_FIT_SECONDS),
            "maximum_cuda_allocated_bytes": _MAXIMUM_CUDA_ALLOCATED_BYTES,
        }


@dataclass(frozen=True, repr=False)
class RuntimeInputs:
    """Prepared runtime pairs plus a privacy-safe public report."""

    pairs: tuple
    report_json: str

    def public_report(self):
        return json.loads(self.report_json)


def _prepare(
    *,
    metadata_bytes,
    action_bytes,
    specification_audit_bytes,
    specification_source_bytes,
    candidate_registry_bytes,
):
    audit_bytes = _copy_api_bytes(specification_audit_bytes, "invalid_specification_audit_bytes")
    registry_bytes = _copy_api_bytes(candidate_registry_bytes, "invalid_candidate_registry_bytes")
    if hashlib.sha256(audit_bytes).hexdigest() != _SPEC_AUDIT_SHA256:
        _fail("audit_hash_mismatch")
    if hashlib.sha256(registry_bytes).hexdigest() != _CANDIDATES_SHA256:
        _fail("candidate_registry_hash_mismatch")

    metadata_copy = _copy_metadata(metadata_bytes)
    action_copy = _copy_action_bytes(action_bytes)

    audit = _parse_json_object(audit_bytes, "invalid_audit")
    specification_hashes, source_pins = _validate_audit(audit)
    source_copy = _copy_specification_sources(specification_source_bytes, source_pins)

    prepared = _arrays.prepare_u0_source_arrays(
        metadata_bytes=metadata_copy,
        action_bytes=action_copy,
    )

    classical_candidates = _select_classical_candidates(_read_candidate_registry(registry_bytes))
    neural_contract = _read_neural_contract(source_copy)
    substrate_map = _read_substrate_metadata(metadata_copy["manifest_bytes"])

    pairs = []
    observations_cache = {}
    classical_count = 0
    neural_count = 0
    for prepared_pair in prepared.pairs:
        fit_job = _parse_job_json(prepared_pair.fit_job_json)
        prediction_job = _parse_job_json(prepared_pair.prediction_job_json)
        kind, configuration, parameters_json = _kernel_for_job(
            fit_job, specification_hashes, classical_candidates, neural_contract
        )
        prediction_kind, prediction_configuration, _ = _kernel_for_job(
            prediction_job, specification_hashes, classical_candidates, neural_contract
        )
        if prediction_kind != kind:
            _fail("candidate_mismatch")
        if prediction_configuration.get("model_id") != configuration.get("model_id"):
            _fail("candidate_mismatch")

        source_roles = prepared_pair.inputs.source_roles
        context_id = getattr(source_roles, "context_id", None)
        unit_id = getattr(source_roles, "unit_id", None)
        if type(context_id) is not str or type(unit_id) is not str:
            _fail("input_preparation_failed")
        cache_key = (context_id, unit_id)
        observations = observations_cache.get(cache_key)
        if observations is None:
            observations = _build_observations(source_roles, substrate_map)
            observations_cache[cache_key] = observations
        fitting_observations, validation_observations = observations

        pairs.append(
            RuntimePair(
                prepared_pair=prepared_pair,
                parameters_json=parameters_json,
                configuration_json=json.dumps(
                    configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ),
                fitting_observations=fitting_observations,
                validation_observations=validation_observations,
            )
        )
        if kind == "classical":
            classical_count += 1
        else:
            neural_count += 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "prepared_report_sha256": prepared.public_report()["report_sha256"],
        "specification_audit_sha256": _SPEC_AUDIT_SHA256,
        "candidate_registry_sha256": _CANDIDATES_SHA256,
        "pairs": len(pairs),
        "classical_pairs": classical_count,
        "neural_pairs": neural_count,
        "models": len(_KNOWN_MODELS),
        "candidates": len(classical_candidates),
        "source_files": len(source_pins),
        "specification_bytes_verified": True,
        "kernel_arguments_prepared": True,
        "substrate_metadata_preserved": True,
        "loaded_runtime_code_verified": False,
        "live_controller_verified": False,
        "execution_authorized": False,
        "new_scientific_operations": 0,
    }
    report["report_sha256"] = sha256_value(report)
    report_json = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return RuntimeInputs(pairs=tuple(pairs), report_json=report_json)


def prepare_u0_runtime_inputs(
    *,
    metadata_bytes,
    action_bytes,
    specification_audit_bytes,
    specification_source_bytes,
    candidate_registry_bytes,
):
    """Authenticate inputs and assemble inherited U0 kernel arguments."""

    try:
        return _prepare(
            metadata_bytes=metadata_bytes,
            action_bytes=action_bytes,
            specification_audit_bytes=specification_audit_bytes,
            specification_source_bytes=specification_source_bytes,
            candidate_registry_bytes=candidate_registry_bytes,
        )
    except InputError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("input_preparation_failed")


def require_scientific_execution(*args, **kwargs):
    """Always deny execution, regardless of forged flags or arguments."""

    _fail("scientific_execution_not_authorized")
