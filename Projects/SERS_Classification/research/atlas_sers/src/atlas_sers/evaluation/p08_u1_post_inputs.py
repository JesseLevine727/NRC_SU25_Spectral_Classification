"""Prepare exact post-source U1 inputs for the frozen graph (no execution).

The adapter reshapes already-authenticated factory outputs and recorded
P02/P03/P04 roles into keyword arguments for the frozen numerical kernels.  It
authenticates no receipts, grants no execution and never treats a returned
container as permission.  Every returned object is a defensive copy.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p03_roles, p03_runtime, p05_sampling
from atlas_sers.evaluation.p08_plan import (
    CLASSICAL_MODELS,
    JOB_FIELDS,
    NEURAL_RECIPES,
    POLICY_REPRESENTATION,
    SEEDS,
    SVM_MODEL,
    SVM_SEED,
    _hash,
)
from atlas_sers.evaluation.p08_u0_arrays import _ACTION_PINS
from atlas_sers.evaluation.p08_u0_inputs import _BYTE_PINS
from atlas_sers.governance.canonical import sha256_value

_FIT_MANIFEST_SHA256 = "3a8e22e50cb87c32ff26062d65d35dcd7ae6d339ef344bcedcde9123a33122a8"
_T3_PARTITION_SHA256 = "70488703746e15338cb3fd0f21872a22dd93f74d9b05a31f52b658c6a80a7416"
_INNER_MASTER_SHA256 = "c2b72b26acf11860d53288d45b5d5efa7c4c751436de360096b58b2b6ecc3f2c"

_CALIBRATION_EXPERIMENT_ID = "EXP-C10-T3"
_CALIBRATION_STAGE = "calibration_crossfit"
_CALIBRATION_TEMPLATE_MODEL_ID = "C-RBF-SVM"

STAGE_SOURCE_FIT = "source_fit"
STAGE_SOURCE_VALIDATION_PREDICTION = "source_validation_prediction"
STAGE_CALIBRATION_MODEL_FIT = "calibration_model_fit"
STAGE_CALIBRATION_VALIDATION_PREDICTION = "calibration_validation_prediction"
STAGE_CALIBRATION_ALIAS = "calibration_prediction_alias"
STAGE_FINAL_REFIT = "final_refit"
STAGE_HELD_PREDICTION = "held_prediction"

_SOURCE_STAGES = (STAGE_SOURCE_FIT, STAGE_SOURCE_VALIDATION_PREDICTION)
_CALIBRATION_STAGES = (
    STAGE_CALIBRATION_MODEL_FIT,
    STAGE_CALIBRATION_VALIDATION_PREDICTION,
)
_FINAL_STAGES = (STAGE_FINAL_REFIT, STAGE_HELD_PREDICTION)

_METADATA_KEYS = ("manifest_bytes", "contexts_bytes", "roles_bytes")
_P02_TABLE_KEYS = ("t3_partition_registry.csv", "inner_master_split_registry.csv")
_P02_PINS = {
    "t3_partition_registry.csv": _T3_PARTITION_SHA256,
    "inner_master_split_registry.csv": _INNER_MASTER_SHA256,
}

_EPOCHS_MIN, _EPOCHS_MAX = 30, 200
_MAXIMUM_FIT_SECONDS = 120.0
_MAXIMUM_CUDA_ALLOCATED_BYTES = 4 * 2**30


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(bytes(payload)).hexdigest()


def _check_byte_pin(payload: Any, pin: str, label: str) -> None:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be supplied as raw immutable bytes.")
    if _sha256_bytes(bytes(payload)) != pin:
        raise ValueError(f"{label} does not match its immutable byte pin.")


def _check_metadata_pins(metadata_bytes: Mapping[str, Any]) -> None:
    if not isinstance(metadata_bytes, Mapping):
        raise TypeError("metadata_bytes must be a mapping of pinned payloads.")
    if set(metadata_bytes.keys()) != set(_METADATA_KEYS):
        raise ValueError("metadata_bytes must supply exactly the three post-source payloads.")
    for name in _METADATA_KEYS:
        _check_byte_pin(metadata_bytes[name], _BYTE_PINS[name], name)


def _parse_text_frame(payload: bytes) -> pd.DataFrame:
    return pd.read_csv(
        io.BytesIO(bytes(payload)),
        dtype=str,
        keep_default_na=False,
    )


def _parse_table(payload: bytes) -> pd.DataFrame:
    frame = _parse_text_frame(payload)
    for column in ("outer_repeat", "outer_fold", "inner_fold"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    return frame


def _subset_manifest(manifest: pd.DataFrame, uids: Any) -> pd.DataFrame:
    if "sensor_family" not in manifest.columns:
        raise ValueError("Manifest lacks the required sensor_family column.")
    requested = [str(uid) for uid in uids]
    if len(set(requested)) != len(requested):
        raise ValueError("Role resolution produced duplicate observation UIDs.")
    ordered = sorted(requested)
    wanted = set(ordered)
    frame = manifest[manifest.observation_uid.astype(str).isin(wanted)].copy()
    missing = wanted - set(frame.observation_uid.astype(str))
    if missing:
        raise ValueError(f"Manifest lacks {len(missing)} required role UIDs.")
    frame = frame.assign(_sort_uid=frame.observation_uid.astype(str))
    frame = frame.sort_values("_sort_uid", kind="stable").drop(columns=["_sort_uid"])
    return frame.reset_index(drop=True)


def _uid_digest(uids: Any) -> str:
    return sha256_value(sorted(str(uid) for uid in uids))


def _normalize_parameters(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _validate_epochs(epochs: Any) -> int:
    if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)):
        raise ValueError("Neural refit epochs must be an integer.")
    value = int(epochs)
    if not _EPOCHS_MIN <= value <= _EPOCHS_MAX:
        raise ValueError(
            f"Neural refit epochs must lie within [{_EPOCHS_MIN}, {_EPOCHS_MAX}]."
        )
    return value


def _selection_field(selection: Any, name: str) -> Any:
    if isinstance(selection, Mapping):
        if name not in selection:
            raise ValueError(f"Selection does not record {name!r}.")
        return selection[name]
    if not hasattr(selection, name):
        raise ValueError(f"Selection does not record {name!r}.")
    return getattr(selection, name)


def _noise_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    # Inherited augmentation uses recorded native-grid QC, not recomputed SG/arPLS QC.
    columns = ["observation_uid", "first_difference_noise_mad", "intensity_range"]
    result = metadata[columns].copy(deep=True)
    for column in columns[1:]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    if not np.isfinite(result[columns[1:]].to_numpy()).all():
        raise ValueError("Recorded source noise metadata must be finite.")
    if (result.first_difference_noise_mad < 0).any() or (result.intensity_range <= 0).any():
        raise ValueError("Recorded source noise/range metadata is invalid.")
    return result


@dataclass(frozen=True)
class PostSourceInputs:
    """Read-only snapshot of authenticated source inputs and recorded roles."""

    _source_factory: Any
    _manifest: pd.DataFrame
    _contexts: pd.DataFrame
    _roles: pd.DataFrame
    _fit_manifest: pd.DataFrame
    _p02_tables: dict[str, pd.DataFrame]
    _calibration_cache: dict[tuple[str, str], Any] = field(default_factory=dict)

    def context_metadata(self, context_id: str) -> pd.Series:
        return self._context_row(context_id).copy(deep=True)

    def role_metadata(self, job: Any, role: str) -> pd.DataFrame:
        self._validate_job(job)
        self._validate_context_isolation(job)
        if role not in ("fit", "validation", "test"):
            raise ValueError("role_metadata supports only fit, validation or test.")
        stage = str(job["stage"])
        if role == "test":
            uids, _ = self._outer_role_uids(job["context_id"], "outer_test")
        elif stage in _SOURCE_STAGES:
            uids = self._source_role_uids(job, role)
        elif stage in _CALIBRATION_STAGES:
            roles = self._calibration_roles(job)
            uids = list(getattr(roles, f"{role}_uids"))
        elif stage in _FINAL_STAGES:
            if role == "validation":
                raise ValueError("Final and held jobs register no validation role.")
            uids, _ = self._outer_role_uids(job["context_id"], "outer_fit")
        elif stage == STAGE_CALIBRATION_ALIAS:
            uids = self._alias_role_uids(job, role)
        else:
            raise ValueError(f"Job stage {stage!r} does not expose role metadata.")
        return _subset_manifest(self._manifest, uids)

    def classical_fit_kwargs(self, job: Any, selection: Any) -> dict[str, Any]:
        self._validate_job(job)
        self._validate_context_isolation(job)
        stage = str(job["stage"])
        if stage not in (STAGE_CALIBRATION_MODEL_FIT, STAGE_FINAL_REFIT):
            raise ValueError("Classical fitting needs calibration_model_fit or final_refit.")
        model_id = str(job["model_id"])
        if model_id not in CLASSICAL_MODELS:
            raise ValueError("Classical fitting requires a registered classical model.")
        representation = str(job["representation_id"])
        candidate_id, parameters = self._validate_selection(job, selection)
        if stage == STAGE_CALIBRATION_MODEL_FIT:
            roles = self._calibration_roles(job)
            fit_uids = [str(uid) for uid in roles.fit_uids]
            validation_uids = [str(uid) for uid in roles.validation_uids]
            self._assert_inner_source_roles(job, fit_uids, validation_uids)
            ordered = sorted(set(fit_uids) | set(validation_uids))
            dataset = self._dataset(ordered, representation)
            return {
                "dataset": dataset,
                "fit_id": str(job["job_id"]),
                "model_id": model_id,
                "candidate_id": candidate_id,
                "parameters": parameters,
                "seed": job["seed"],
                "fit_uids": tuple(sorted(fit_uids)),
                "validation_uids": tuple(sorted(validation_uids)),
                "class_vocabulary": self._outer_class_vocabulary(job),
                "expected_fit_uid_sha256": job["fit_uid_sha256"],
                "expected_validation_uid_sha256": job["validation_uid_sha256"],
            }
        fit_uids, _ = self._outer_role_uids(job["context_id"], "outer_fit")
        ordered = sorted(str(uid) for uid in fit_uids)
        dataset = self._dataset(ordered, representation)
        return {
            "dataset": dataset,
            "fit_id": str(job["job_id"]),
            "model_id": model_id,
            "candidate_id": candidate_id,
            "parameters": parameters,
            "seed": job["seed"],
            "fit_uids": tuple(ordered),
            "expected_fit_uid_sha256": job["fit_uid_sha256"],
        }

    def neural_refit_kwargs(self, job: Any, epochs: Any) -> dict[str, Any]:
        self._validate_job(job)
        self._validate_context_isolation(job)
        if str(job["stage"]) != STAGE_FINAL_REFIT:
            raise ValueError("Neural refit requires a final_refit job.")
        model_id = str(job["model_id"])
        if model_id not in NEURAL_RECIPES:
            raise ValueError("Neural refit requires a registered neural recipe.")
        epochs = _validate_epochs(epochs)
        fit_uids, role_id = self._outer_role_uids(job["context_id"], "outer_fit")
        values, metadata, observations = self._fitting_matrix(
            fit_uids, str(job["representation_id"])
        )
        return {
            "values": values,
            "observations": observations,
            "noise_metadata": _noise_metadata(metadata),
            "role_id": role_id,
            "recipe": model_id,
            "seed": job["seed"],
            "epochs": epochs,
            "maximum_fit_seconds": _MAXIMUM_FIT_SECONDS,
            "maximum_cuda_allocated_bytes": _MAXIMUM_CUDA_ALLOCATED_BYTES,
        }

    def held_inputs(
        self, job: Any
    ) -> tuple[np.ndarray, pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
        self._validate_job(job)
        self._validate_context_isolation(job)
        if str(job["stage"]) != STAGE_HELD_PREDICTION:
            raise ValueError("held_inputs requires a held_prediction job.")
        context_id = str(job["context_id"])
        fit_uids, _ = self._outer_role_uids(context_id, "outer_fit")
        test_uids, _ = self._outer_role_uids(context_id, "outer_test")
        ordered, positions = self._ordered_positions(test_uids)
        values = self._action_values(ordered, positions, str(job["representation_id"]))
        metadata = _subset_manifest(self._manifest, ordered)
        return (
            values,
            metadata,
            self._outer_class_vocabulary(job),
            tuple(sorted(str(uid) for uid in fit_uids)),
        )

    # -- internals ---------------------------------------------------------

    def _validate_job(self, job: Any) -> None:
        if not isinstance(job, Mapping):
            raise ValueError("Jobs must be mapping objects.")
        if set(job.keys()) != set(JOB_FIELDS) | {"job_id"}:
            raise ValueError("Job keys are not exactly JOB_FIELDS plus job_id.")
        payload = {key: value for key, value in job.items() if key != "job_id"}
        if str(job["job_id"]) != "P08JOB-" + _hash(payload):
            raise ValueError("Job id does not match its frozen payload hash.")
        policy = str(job["policy_id"])
        if policy not in ("PP-U-SG", "PP-U-ARPLS"):
            raise ValueError("Post-source fitting is restricted to SG/arPLS policies.")
        representation = str(job["representation_id"])
        if POLICY_REPRESENTATION[policy] != representation:
            raise ValueError("Job representation contradicts POLICY_REPRESENTATION.")
        if representation not in _ACTION_PINS:
            raise ValueError("Job representation is absent from the frozen action pins.")
        if str(job["array_sha256"]) != str(_ACTION_PINS[representation]["array_sha256"]):
            raise ValueError("Job array hash differs from the frozen action pin.")
        model_id = str(job["model_id"])
        specifications = getattr(self._source_factory, "_specification_hashes", {})
        if model_id not in specifications:
            raise ValueError("Job model is absent from the factory specification registry.")
        if str(job["model_spec_sha256"]) != str(specifications[model_id]):
            raise ValueError("Job specification hash differs from the factory registry.")
        seed = job["seed"]
        if model_id == SVM_MODEL:
            if seed != SVM_SEED:
                raise ValueError("SVM jobs must record the deterministic seed.")
        elif model_id in CLASSICAL_MODELS or model_id in NEURAL_RECIPES:
            if seed not in SEEDS:
                raise ValueError("Job seed is absent from the frozen seed registry.")
        else:
            raise ValueError("Job model is not a registered classical or neural model.")

    def _context_row(self, context_id: Any) -> pd.Series:
        cell = self._contexts[self._contexts.context_id.astype(str) == str(context_id)]
        if len(cell) != 1:
            raise ValueError("Context does not resolve to exactly one registry row.")
        row = cell.iloc[0]
        if str(row.phase_gate) != "held_evaluation":
            raise ValueError("Only registered held-evaluation contexts are in scope.")
        for name in ("station", "held_instrument"):
            if name not in row.index:
                raise ValueError(f"Context row lacks the required {name!r} column.")
        return row

    def _outer_role_uids(self, context_id: Any, role: str) -> tuple[list[str], str]:
        cell = self._roles[
            (self._roles.context_id.astype(str) == str(context_id))
            & (self._roles.role.astype(str) == role)
        ]
        if cell.empty:
            raise ValueError(f"Context does not register an outer {role} role.")
        role_ids = cell.role_id.astype(str).unique()
        if len(role_ids) != 1:
            raise ValueError(f"Outer {role} role does not carry a unique role_id.")
        uids = cell.observation_uid.astype(str).tolist()
        if len(set(uids)) != len(uids):
            raise ValueError(f"Outer {role} role repeats observation UIDs.")
        return uids, str(role_ids[0])

    def _source_role_uids(self, job: Any, role: str) -> list[str]:
        context_id = str(job["context_id"])
        unit_id = str(job["unit_id"])
        roles = getattr(self._source_factory, "_roles", {}).get((context_id, unit_id))
        if roles is None:
            raise ValueError("Factory has no authenticated source roles for this cell.")
        for field_name, observations in (
            ("fit_uid_sha256", roles.fitting),
            ("validation_uid_sha256", roles.validation),
        ):
            if _uid_digest(obs.observation_uid for obs in observations) != job[field_name]:
                raise ValueError("Source role UID hash differs from the recorded job.")
        observations = roles.fitting if role == "fit" else roles.validation
        return [str(obs.observation_uid) for obs in observations]

    def _alias_role_uids(self, job: Any, role: str) -> list[str]:
        for entry in getattr(self._source_factory, "_roles", {}).values():
            if entry.context_id != job["context_id"]:
                continue
            fit_digest = _uid_digest(obs.observation_uid for obs in entry.fitting)
            val_digest = _uid_digest(obs.observation_uid for obs in entry.validation)
            if (
                fit_digest == str(job["fit_uid_sha256"])
                and val_digest == str(job["validation_uid_sha256"])
            ):
                observations = entry.fitting if role == "fit" else entry.validation
                return [str(obs.observation_uid) for obs in observations]
        raise ValueError("Calibration alias does not match an authenticated source unit.")

    def _calibration_roles(self, job: Any) -> Any:
        context_id = str(job["context_id"])
        unit_id = str(job["unit_id"])
        key = (context_id, unit_id)
        cached = self._calibration_cache.get(key)
        if cached is not None:
            roles, hashes = cached
            self._assert_cached_hashes(job, hashes)
            return roles
        context = self._context_row(context_id)
        frame = self._fit_manifest
        cell = frame[
            (frame.experiment_id.astype(str) == _CALIBRATION_EXPERIMENT_ID)
            & (frame.stage.astype(str) == _CALIBRATION_STAGE)
            & (frame.model_id.astype(str) == _CALIBRATION_TEMPLATE_MODEL_ID)
            & (frame.domain.astype(str) == str(context.domain))
            & (frame.outer_repeat.astype(int) == int(context.outer_repeat))
            & (frame.outer_fold.astype(int) == int(context.outer_fold))
        ]
        matched = cell[cell.selection_unit_id.astype(str) == unit_id]
        if len(matched) != 1:
            raise ValueError("Calibration unit does not resolve uniquely in the P03 plan.")
        row = matched.iloc[0]
        hashes = {
            "fit_uid_sha256": str(row.fit_uid_sha256),
            "validation_uid_sha256": str(row.validation_uid_sha256),
            "test_uid_sha256": str(row.test_uid_sha256),
        }
        self._assert_cached_hashes(job, hashes)
        roles = p03_roles.resolve_fit_roles(
            row, manifest=self._manifest, p02_tables=self._p02_tables
        )
        self._calibration_cache[key] = (roles, hashes)
        return roles

    @staticmethod
    def _assert_cached_hashes(job: Any, hashes: Mapping[str, str]) -> None:
        for name, value in hashes.items():
            if str(job[name]) != str(value):
                raise ValueError("Calibration unit hashes differ from the recorded job.")

    def _validate_selection(self, job: Any, selection: Any) -> tuple[str, Any]:
        dependencies = list(job["dependencies"])
        if len(dependencies) != 1:
            raise ValueError("Classical fitting needs exactly one selection dependency.")
        selection_job_id = str(_selection_field(selection, "selection_job_id"))
        if selection_job_id != str(dependencies[0]):
            raise ValueError("Selection id differs from the job selection dependency.")
        for name in ("policy_id", "context_id", "model_id"):
            if str(_selection_field(selection, name)) != str(job[name]):
                raise ValueError(f"Selection {name!r} differs from the job.")
        model_id = str(job["model_id"])
        candidate_id = str(_selection_field(selection, "selected_candidate_id"))
        hyperparameter_sha256 = str(
            _selection_field(selection, "selected_hyperparameter_sha256")
        )
        parameters = _selection_field(selection, "selected_parameters")
        registry = getattr(self._source_factory, "_candidate_index", {})
        candidate = registry.get((model_id, candidate_id))
        if candidate is None:
            raise ValueError("Selected candidate is absent from the factory registry.")
        if str(candidate["hyperparameter_sha256"]) != hyperparameter_sha256:
            raise ValueError("Selected hyperparameter hash differs from the registry.")
        recorded = candidate.get("parameters_json", candidate.get("parameters"))
        if recorded is not None:
            if _hash(_normalize_parameters(parameters)) != _hash(
                _normalize_parameters(recorded)
            ):
                raise ValueError("Selected parameters differ from the registry.")
        return candidate_id, copy.deepcopy(_normalize_parameters(parameters))

    def _validate_context_isolation(self, job: Any) -> None:
        context_id = str(job["context_id"])
        context = self._context_row(context_id)
        station = str(context.station)
        held_instrument = str(context.held_instrument)
        source_uids, _ = self._outer_role_uids(context_id, "outer_fit")
        test_uids, _ = self._outer_role_uids(context_id, "outer_test")
        source = _subset_manifest(self._manifest, source_uids)
        test = _subset_manifest(self._manifest, test_uids)
        if set(source_uids) & set(test_uids):
            raise ValueError("Outer source and held test observation roles overlap.")
        source_masters = set(source.master_sample_id.astype(str))
        test_masters = set(test.master_sample_id.astype(str))
        if source_masters & test_masters:
            raise ValueError("Outer source and held physical masters overlap.")
        if set(source.station.astype(str)) != {station}:
            raise ValueError("Outer source station differs from the matched context.")
        if set(test.station.astype(str)) != {station}:
            raise ValueError("Held test station differs from the matched context.")
        if held_instrument in set(source.instrument.astype(str)):
            raise ValueError("Held instrument appears in the outer source role.")
        if set(test.instrument.astype(str)) != {held_instrument}:
            raise ValueError("Held test instrument differs from the matched context.")
        if _uid_digest(test_uids) != str(job["test_uid_sha256"]):
            raise ValueError("Outer test UID hash differs from the recorded job.")
        if str(job["stage"]) in _FINAL_STAGES:
            if _uid_digest(source_uids) != str(job["fit_uid_sha256"]):
                raise ValueError("Outer source UID hash differs from the recorded job.")

    def _assert_inner_source_roles(
        self, job: Any, fit_uids: Any, validation_uids: Any
    ) -> None:
        context_id = str(job["context_id"])
        held = str(self._context_row(context_id).held_instrument)
        outer_uids, _ = self._outer_role_uids(context_id, "outer_fit")
        outer_masters = set(
            _subset_manifest(self._manifest, outer_uids).master_sample_id.astype(str)
        )
        fit_frame = _subset_manifest(self._manifest, fit_uids)
        validation_frame = _subset_manifest(self._manifest, validation_uids)
        fit_masters = set(fit_frame.master_sample_id.astype(str))
        validation_masters = set(validation_frame.master_sample_id.astype(str))
        if fit_masters & validation_masters:
            raise ValueError("Inner fit and validation physical masters overlap.")
        if set(str(uid) for uid in fit_uids) & set(str(uid) for uid in validation_uids):
            raise ValueError("Inner fit and validation observation roles overlap.")
        if not fit_masters <= outer_masters or not validation_masters <= outer_masters:
            raise ValueError("Inner roles are not subsets of the outer source role.")
        if not set(fit_uids) <= set(outer_uids) or not set(validation_uids) <= set(outer_uids):
            raise ValueError("Inner observation roles are not subsets of outer source.")
        instruments = set(fit_frame.instrument.astype(str)) | set(
            validation_frame.instrument.astype(str)
        )
        if held in instruments:
            raise ValueError("Held instrument leaks into an inner source role.")

    def _outer_class_vocabulary(self, job: Any) -> tuple[str, ...]:
        uids, _ = self._outer_role_uids(job["context_id"], "outer_fit")
        frame = _subset_manifest(self._manifest, uids)
        return tuple(sorted(set(frame.target_analyte.astype(str))))

    def _labels_for(self, uids: Any) -> tuple[str, ...]:
        frame = _subset_manifest(self._manifest, uids)
        return tuple(str(label) for label in frame.target_analyte)

    def _ordered_positions(self, uids: Any) -> tuple[list[str], list[int]]:
        ordered = sorted(str(uid) for uid in uids)
        if len(set(ordered)) != len(ordered):
            raise ValueError("Requested role repeats observation UIDs.")
        index = getattr(self._source_factory, "_manifest_index", {})
        positions: list[int] = []
        for uid in ordered:
            if uid not in index:
                raise ValueError(f"UID {uid!r} is absent from the factory index.")
            positions.append(int(index[uid]))
        return ordered, positions

    def _action_values(
        self, ordered: list[str], positions: list[int], representation: str
    ) -> np.ndarray:
        action = self._source_factory._actions[representation]
        matrix = np.asarray(action["intensity"])[positions]
        if matrix.shape != (len(ordered), 1401) or not np.isfinite(matrix).all():
            raise ValueError("Factory action matrix must be finite on the 1401-point grid.")
        return np.array(matrix, dtype=np.float32, copy=True)

    def _dataset(self, uids: Any, representation: str) -> Any:
        ordered, positions = self._ordered_positions(uids)
        values = self._action_values(ordered, positions, representation)
        metadata = _subset_manifest(self._manifest, ordered)
        metadata.index = metadata.observation_uid.astype(str)
        metadata = metadata.reindex(ordered)
        return p03_runtime.P03Dataset.from_frozen_representation(
            intensity=values,
            representation_uids=np.asarray(ordered, dtype=object),
            metadata=metadata,
        )

    def _fitting_matrix(
        self, uids: Any, representation: str
    ) -> tuple[np.ndarray, pd.DataFrame, tuple[Any, ...]]:
        ordered, positions = self._ordered_positions(uids)
        values = self._action_values(ordered, positions, representation)
        metadata = _subset_manifest(self._manifest, ordered)
        observations = tuple(self._observation(row) for row in metadata.itertuples())
        return values, metadata, observations

    @staticmethod
    def _observation(row: Any) -> Any:
        return p05_sampling.Observation(
            uid=str(row.observation_uid),
            master=str(row.master_sample_id),
            station=str(row.station),
            target=str(row.target_analyte),
            instrument=str(row.instrument),
            substrate=str(row.sensor_family),
        )


def prepare_post_source_inputs(
    *,
    source_factory: Any,
    metadata_bytes: Mapping[str, Any],
    p03_fit_manifest_bytes: bytes,
    p02_table_bytes: Mapping[str, Any],
) -> PostSourceInputs:
    """Authenticate immutable byte pins, then parse and snapshot the inputs."""

    _check_metadata_pins(metadata_bytes)
    _check_byte_pin(p03_fit_manifest_bytes, _FIT_MANIFEST_SHA256, "P03 fit manifest")
    if not isinstance(p02_table_bytes, Mapping):
        raise TypeError("p02_table_bytes must be a mapping of pinned tables.")
    for name in _P02_TABLE_KEYS:
        if name not in p02_table_bytes:
            raise ValueError(f"p02_table_bytes is missing {name!r}.")
        _check_byte_pin(p02_table_bytes[name], _P02_PINS[name], name)
    manifest = _parse_text_frame(metadata_bytes["manifest_bytes"])
    contexts = _parse_text_frame(metadata_bytes["contexts_bytes"])
    roles = _parse_text_frame(metadata_bytes["roles_bytes"])
    fit_manifest = _parse_text_frame(p03_fit_manifest_bytes)
    p02_tables = {name: _parse_table(p02_table_bytes[name]) for name in _P02_TABLE_KEYS}
    return PostSourceInputs(
        _source_factory=source_factory,
        _manifest=manifest,
        _contexts=contexts,
        _roles=roles,
        _fit_manifest=fit_manifest,
        _p02_tables=p02_tables,
    )


__all__ = [
    "PostSourceInputs",
    "prepare_post_source_inputs",
    "STAGE_CALIBRATION_MODEL_FIT",
    "STAGE_FINAL_REFIT",
    "STAGE_HELD_PREDICTION",
    "STAGE_SOURCE_FIT",
    "STAGE_SOURCE_VALIDATION_PREDICTION",
]
