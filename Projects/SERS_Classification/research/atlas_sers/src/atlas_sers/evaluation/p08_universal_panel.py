"""P08 universal panel adapter (private draft).

This module is a pure in-memory *consistency* adapter between already
authenticated saved P08 outputs and the reviewed universal unit builder
``atlas_sers.evaluation.p08_universal_units.build_units``.

It is explicitly **not** a provenance authentication layer.  The caller must
authenticate the graph archive, the full ``PP-U-MIN`` evidence bridge, every
artifact file and its receipt, and all source/validation/test metadata before
calling :func:`assemble_panel`.  The checks below only confirm that the frozen
tables, graph records and aliases the caller supplies are mutually consistent.
They cannot establish scientific authority, unit disjointness or provenance.

The adapter performs no file IO, no hashing of external bytes, no training, no
calibration, no resampling, no clipping, no renormalisation, no imputation and
no seed averaging.  Inputs are never mutated.  A missing or malformed
endpoint raises an informative, path-free code instead of returning a false
``complete`` headline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p05_comparison, p08_plan, p08_universal_units
from atlas_sers.governance.canonical import sha256_value

__all__ = ["SCHEMA_VERSION", "P08PanelError", "assemble_panel"]

SCHEMA_VERSION = "nato-sers-p08-universal-panel-v1"

POLICIES = ("PP-U-MIN", "PP-U-SG", "PP-U-ARPLS")
NEW_POLICIES = ("PP-U-SG", "PP-U-ARPLS")
CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
NEURAL_RECIPES = ("D0-M", "D1", "D2", "D3")
ENDPOINT_MODELS = CLASSICAL_MODELS + NEURAL_RECIPES
REPORT_MODELS = CLASSICAL_MODELS + ("D0-M", "P05-SELECTED")
D0_STRATEGY = "D0-M"
SELECTED_STRATEGY = "P05-SELECTED"
NEURAL_STRATEGIES = (D0_STRATEGY, SELECTED_STRATEGY)

SEED_ENSEMBLE_PREDICTION = "seed_ensemble_prediction"
SOURCE_ROLE = "outer_fit"
TEST_ROLE = "outer_test"

PROBABILITY_COLUMNS = ("probability_0", "probability_1", "probability_2")

_PROBABILITY_TOLERANCE = 1e-12
_CLASSICAL_STATUS = "cross_fitted_temperature"
_NEURAL_STATUS = "seedwise_temperature_ensemble"
_DETERMINISTIC_SEED_COUNT = 1
_STOCHASTIC_SEED_COUNT = 3
_UNKNOWN_FAMILIES = ("na", "n/a", "none", "nan", "null", "unknown")

_MANIFEST_COLUMNS = (
    "observation_uid",
    "master_sample_id",
    "station",
    "target_analyte",
    "instrument",
)
_ROLE_COLUMNS = (
    "context_id",
    "role_id",
    "role",
    "selection_unit_id",
    "observation_uid",
    "master_sample_id",
    "target_analyte",
    "instrument",
)
_REGISTERED_COLUMNS = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "outer_repeat",
    "outer_fold",
    "observation_uid",
    "master_sample_id",
    "true_label",
    "class_vocabulary",
)
_PREDICTION_COLUMNS = (
    "context_id",
    "policy_id",
    "model_id",
    "observation_uid",
    "master_sample_id",
    "instrument",
    "station",
    "true_label",
    "class_vocabulary",
    "probability_0",
    "probability_1",
    "probability_2",
)
_COVERAGE_COLUMNS = (
    "context_id",
    "policy_id",
    "model_id",
    "endpoint_job_id",
    "recipe_id",
    "reference_count",
    "unique_endpoint",
    "expected_n_spectra",
    "actual_n_spectra",
    "expected_n_masters",
    "actual_n_masters",
    "complete",
)
_ENDPOINT_INDEX_COLUMNS = (
    "context_id",
    "policy_id",
    "model_id",
    "endpoint_job_id",
    "recipe_id",
)
_ALIAS_KEYS = (
    "alias_id",
    "context_id",
    "policy_id",
    "recipe_id",
    "strategy",
    "target_job_id",
)


class P08PanelError(ValueError):
    """Raised when supplied P08 panel inputs are inconsistent."""


def _fail(code: str) -> None:
    raise P08PanelError(code)


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _strict_text(value, code: str) -> str:
    if not isinstance(value, str):
        _fail(code)
    if not value or value != value.strip():
        _fail(code)
    return value


def _master_id(value, code: str) -> str:
    if isinstance(value, (bool, np.bool_)):
        _fail(code)
    if isinstance(value, str):
        return _strict_text(value, code)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    _fail(code)


def _integral(value, code: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        _fail(code)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return int(number)
        _fail(code)
    if isinstance(value, str):
        text = value
        if not text or text != text.strip():
            _fail(code)
        try:
            number = int(text)
        except ValueError:
            _fail(code)
        if str(number) != text:
            _fail(code)
        return number
    _fail(code)


def _seed_count(value, code: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        _fail(code)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
    elif isinstance(value, str):
        text = value
        if not text or text != text.strip():
            _fail(code)
        try:
            number = float(text)
        except ValueError:
            _fail(code)
    else:
        _fail(code)
    if not np.isfinite(number) or not number.is_integer():
        _fail(code)
    return int(number)


def _optional_family(value):
    if _is_missing(value):
        return None
    if not isinstance(value, str):
        _fail("manifest_family_invalid")
    text = value.strip()
    if not text or text.lower() in _UNKNOWN_FAMILIES:
        return None
    return text


def _classes(value, code: str) -> tuple[str, ...]:
    try:
        return tuple(p05_comparison._classes(value))
    except p05_comparison.P05ComparisonError:
        _fail(code)


def _parse_manifest(manifest: pd.DataFrame) -> dict:
    if not isinstance(manifest, pd.DataFrame):
        _fail("manifest_not_frame")
    if manifest.columns.duplicated().any():
        _fail("manifest_duplicate_columns")
    missing = [column for column in _MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        _fail("manifest_columns_missing")
    has_family = "instrument_family" in manifest.columns
    index: dict[str, dict] = {}
    master_station: dict[str, str] = {}
    master_target: dict[str, str] = {}
    station_labels: dict[str, set] = {}
    instruments: set = set()
    families: dict[str, object] = {}
    for row in manifest.itertuples(index=False):
        observation_uid = _strict_text(
            row.observation_uid, "manifest_observation_invalid"
        )
        master = _master_id(row.master_sample_id, "manifest_master_invalid")
        station = _strict_text(row.station, "manifest_station_invalid")
        target = _strict_text(row.target_analyte, "manifest_target_invalid")
        instrument = _strict_text(row.instrument, "manifest_instrument_invalid")
        if observation_uid in index:
            _fail("manifest_observation_duplicate")
        if master_station.get(master, station) != station:
            _fail("manifest_master_station_conflict")
        if master_target.get(master, target) != target:
            _fail("manifest_master_target_conflict")
        master_station[master] = station
        master_target[master] = target
        station_labels.setdefault(station, set()).add(target)
        instruments.add(instrument)
        if has_family:
            family = _optional_family(row.instrument_family)
            if instrument in families and families[instrument] != family:
                _fail("manifest_instrument_family_conflict")
            families[instrument] = family
        index[observation_uid] = {
            "observation_uid": observation_uid,
            "master_sample_id": master,
            "station": station,
            "target_analyte": target,
            "instrument": instrument,
        }
    if not index:
        _fail("manifest_empty")
    vocabulary: dict[str, tuple[str, ...]] = {}
    for station, labels in station_labels.items():
        if len(labels) != 3:
            _fail("manifest_station_vocabulary_size")
        vocabulary[station] = tuple(sorted(labels))
    return {
        "index": index,
        "vocabulary": vocabulary,
        "masters": sorted(set(master_station)),
        "instruments": sorted(instruments),
        "families": families,
        "has_family": has_family,
    }


def _select_contexts(contexts: pd.DataFrame) -> pd.DataFrame:
    held = p05_comparison._held_contexts(contexts)
    held = held.copy()
    for column in (
        "context_id",
        "domain",
        "station",
        "held_instrument",
        "outer_test_uid_sha256",
    ):
        held[column] = held[column].astype(str)
    coordinates = list(
        zip(held.domain, held.outer_repeat, held.outer_fold, strict=True)
    )
    if len(set(coordinates)) != len(coordinates):
        _fail("held_context_coordinate_duplicate")
    return held.sort_values("context_id", kind="stable").reset_index(drop=True)


def _select_role_rows(roles: pd.DataFrame, context_id: str, role: str) -> pd.DataFrame:
    mask = roles.context_id.astype(str).eq(context_id) & roles.role.astype(str).eq(role)
    return roles[mask]


def _role_records(
    rows: pd.DataFrame, manifest_index: Mapping[str, dict], code: str
) -> list:
    records = []
    for row in rows.itertuples(index=False):
        observation_uid = _strict_text(
            row.observation_uid, f"{code}_observation_invalid"
        )
        entry = manifest_index.get(observation_uid)
        if entry is None:
            _fail(f"{code}_observation_unknown")
        master = _master_id(row.master_sample_id, f"{code}_master_invalid")
        target = _strict_text(row.target_analyte, f"{code}_target_invalid")
        instrument = _strict_text(row.instrument, f"{code}_instrument_invalid")
        if master != entry["master_sample_id"]:
            _fail(f"{code}_master_mismatch")
        if target != entry["target_analyte"]:
            _fail(f"{code}_target_mismatch")
        if instrument != entry["instrument"]:
            _fail(f"{code}_instrument_mismatch")
        records.append((observation_uid, master, target, instrument))
    return records


def _parse_context_roles(
    roles: pd.DataFrame, held: pd.DataFrame, manifest: dict
) -> dict:
    if not isinstance(roles, pd.DataFrame):
        _fail("roles_not_frame")
    missing = [column for column in _ROLE_COLUMNS if column not in roles.columns]
    if missing:
        _fail("roles_columns_missing")
    manifest_index = manifest["index"]
    result: dict[str, dict] = {}
    for context in held.itertuples(index=False):
        context_id = str(context.context_id)
        held_instrument = str(context.held_instrument)
        station = str(context.station)
        test_records = _role_records(
            _select_role_rows(roles, context_id, TEST_ROLE), manifest_index, TEST_ROLE
        )
        if not test_records:
            _fail("outer_test_missing")
        test_uids = sorted({record[0] for record in test_records})
        if len(test_uids) != len(test_records):
            _fail("outer_test_duplicate_uid")
        for observation_uid, _master, _target, instrument in test_records:
            if instrument != held_instrument:
                _fail("outer_test_instrument_mismatch")
            if manifest_index[observation_uid]["station"] != station:
                _fail("outer_test_station_mismatch")
        if sha256_value(test_uids) != str(context.outer_test_uid_sha256):
            _fail("outer_test_uid_hash_mismatch")
        test_masters = {record[1] for record in test_records}
        if "outer_test_rows" in held.columns:
            value = context.outer_test_rows
            if not _is_missing(value) and _integral(
                value, "outer_test_rows_invalid"
            ) != len(test_uids):
                _fail("outer_test_rows_mismatch")
        if "outer_test_masters" in held.columns:
            value = context.outer_test_masters
            if not _is_missing(value) and _integral(
                value, "outer_test_masters_invalid"
            ) != len(test_masters):
                _fail("outer_test_masters_mismatch")

        fit_records = _role_records(
            _select_role_rows(roles, context_id, SOURCE_ROLE),
            manifest_index,
            SOURCE_ROLE,
        )
        if not fit_records:
            _fail("outer_fit_missing")
        fit_uids = sorted({record[0] for record in fit_records})
        if len(fit_uids) != len(fit_records):
            _fail("outer_fit_duplicate_uid")
        for observation_uid, _master, _target, instrument in fit_records:
            if instrument == held_instrument:
                _fail("outer_fit_held_instrument_present")
            if manifest_index[observation_uid]["station"] != station:
                _fail("outer_fit_station_mismatch")
        fit_masters = {record[1] for record in fit_records}
        if fit_masters & test_masters:
            _fail("outer_fit_master_overlap")
        if "outer_fit_uid_sha256" in held.columns:
            value = context.outer_fit_uid_sha256
            if not _is_missing(value) and sha256_value(fit_uids) != str(value):
                _fail("outer_fit_uid_hash_mismatch")
        result[context_id] = {
            "test_records": test_records,
            "test_uids": test_uids,
            "test_masters": test_masters,
            "fit_uids": fit_uids,
            "fit_uid_sha256": sha256_value(fit_uids),
        }
    return result


def _build_registered(
    held: pd.DataFrame, context_roles: Mapping[str, dict], manifest: dict
) -> pd.DataFrame:
    rows = []
    for context in held.itertuples(index=False):
        context_id = str(context.context_id)
        vocabulary = manifest["vocabulary"].get(str(context.station))
        if vocabulary is None:
            _fail("registered_station_vocabulary_missing")
        for observation_uid, master, target, _instrument in context_roles[context_id][
            "test_records"
        ]:
            rows.append(
                {
                    "context_id": context_id,
                    "domain": str(context.domain),
                    "station": str(context.station),
                    "instrument": str(context.held_instrument),
                    "outer_repeat": int(context.outer_repeat),
                    "outer_fold": int(context.outer_fold),
                    "observation_uid": observation_uid,
                    "master_sample_id": master,
                    "true_label": target,
                    "class_vocabulary": tuple(vocabulary),
                }
            )
    registered = pd.DataFrame(rows, columns=list(_REGISTERED_COLUMNS))
    if registered.empty:
        _fail("registered_empty")
    if registered.duplicated(["context_id", "observation_uid"]).any():
        _fail("registered_duplicate_observation")
    return registered.sort_values(
        ["context_id", "observation_uid"], kind="stable"
    ).reset_index(drop=True)


def _index_jobs(raw_jobs) -> dict[str, dict]:
    if not isinstance(raw_jobs, (list, tuple)):
        _fail("jobs_must_be_sequence")
    index: dict[str, dict] = {}
    for job in raw_jobs:
        if not isinstance(job, Mapping):
            _fail("job_must_be_mapping")
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            _fail("job_id_invalid")
        if job_id in index:
            _fail("job_id_duplicate")
        index[job_id] = dict(job)
    return index


def _collect_endpoints(job_index: Mapping[str, dict]) -> dict:
    endpoints: dict = {}
    for job_id, job in job_index.items():
        if job.get("stage") != SEED_ENSEMBLE_PREDICTION:
            continue
        policy = job.get("policy_id")
        if policy not in POLICIES:
            continue
        missing = [field for field in p08_plan.JOB_FIELDS if field not in job]
        if missing:
            _fail("endpoint_fields_missing")
        payload = {key: job[key] for key in job if key != "job_id"}
        if "P08JOB-" + sha256_value(payload) != job_id:
            _fail("endpoint_job_hash_invalid")
        model_id = job.get("model_id")
        if model_id not in ENDPOINT_MODELS:
            _fail("endpoint_model_not_permitted")
        if job.get("representation_id") != p08_plan.POLICY_REPRESENTATION[policy]:
            _fail("endpoint_representation_mismatch")
        key = (policy, job.get("context_id"), model_id)
        if key in endpoints:
            _fail("endpoint_key_duplicate")
        endpoints[key] = job
    return endpoints


def _collect_aliases(raw_aliases, job_index: Mapping[str, dict], held_ids: set) -> dict:
    if not isinstance(raw_aliases, (list, tuple)):
        _fail("aliases_must_be_sequence")
    lookup: dict = {}
    for alias in raw_aliases:
        if not isinstance(alias, Mapping):
            _fail("alias_must_be_mapping")
        if set(alias.keys()) != set(_ALIAS_KEYS):
            _fail("alias_keys_invalid")
        policy = _strict_text(alias["policy_id"], "alias_policy_invalid")
        context_id = _strict_text(alias["context_id"], "alias_context_invalid")
        strategy = _strict_text(alias["strategy"], "alias_strategy_invalid")
        recipe = _strict_text(alias["recipe_id"], "alias_recipe_invalid")
        target_job_id = _strict_text(alias["target_job_id"], "alias_target_invalid")
        alias_id = _strict_text(alias["alias_id"], "alias_id_invalid")
        if policy not in POLICIES:
            _fail("alias_policy_invalid")
        if strategy not in NEURAL_STRATEGIES:
            _fail("alias_strategy_invalid")
        if recipe not in NEURAL_RECIPES:
            _fail("alias_recipe_invalid")
        if strategy == D0_STRATEGY and recipe != D0_STRATEGY:
            _fail("alias_d0_recipe_invalid")
        fields = {
            "policy_id": policy,
            "context_id": context_id,
            "strategy": strategy,
            "recipe_id": recipe,
            "target_job_id": target_job_id,
        }
        if alias_id != "P08ALIAS-" + p08_plan._hash(fields):
            _fail("alias_hash_invalid")
        job = job_index.get(target_job_id)
        if job is None:
            _fail("alias_target_missing")
        if job.get("stage") != SEED_ENSEMBLE_PREDICTION:
            _fail("alias_target_stage_invalid")
        if job.get("policy_id") != policy:
            _fail("alias_target_policy_mismatch")
        if job.get("context_id") != context_id:
            _fail("alias_target_context_mismatch")
        if job.get("model_id") != recipe:
            _fail("alias_target_model_mismatch")
        if context_id not in held_ids:
            _fail("alias_context_unregistered")
        key = (policy, context_id, strategy)
        if key in lookup:
            _fail("alias_duplicate")
        lookup[key] = {
            "alias_id": alias_id,
            "policy_id": policy,
            "context_id": context_id,
            "strategy": strategy,
            "recipe_id": recipe,
            "target_job_id": target_job_id,
        }
    for context_id in held_ids:
        for policy in POLICIES:
            for strategy in NEURAL_STRATEGIES:
                if (policy, context_id, strategy) not in lookup:
                    _fail("alias_missing")
    for context_id in held_ids:
        recipes = {
            lookup[(policy, context_id, SELECTED_STRATEGY)]["recipe_id"]
            for policy in POLICIES
        }
        if len(recipes) != 1:
            _fail("selected_recipe_inconsistent")
    return lookup


def _report_cells(
    held: pd.DataFrame, endpoints: Mapping, alias_lookup: Mapping
) -> list:
    cells = []
    for context in held.itertuples(index=False):
        context_id = str(context.context_id)
        for policy in POLICIES:
            for model in CLASSICAL_MODELS:
                job = endpoints.get((policy, context_id, model))
                if job is None:
                    _fail("endpoint_missing")
                cells.append(
                    {
                        "context_id": context_id,
                        "policy_id": policy,
                        "model_id": model,
                        "endpoint_job_id": job["job_id"],
                        "recipe_id": model,
                    }
                )
            for strategy in NEURAL_STRATEGIES:
                alias = alias_lookup[(policy, context_id, strategy)]
                job = endpoints.get((policy, context_id, alias["recipe_id"]))
                if job is None:
                    _fail("endpoint_missing")
                cells.append(
                    {
                        "context_id": context_id,
                        "policy_id": policy,
                        "model_id": strategy,
                        "endpoint_job_id": job["job_id"],
                        "recipe_id": alias["recipe_id"],
                    }
                )
    return cells


def _verify_endpoints(cells, job_index, held_lookup, context_roles) -> None:
    spec_by_model: dict[str, str] = {}
    for cell in cells:
        job = job_index[cell["endpoint_job_id"]]
        context = held_lookup[cell["context_id"]]
        if job["test_uid_sha256"] != str(context.outer_test_uid_sha256):
            _fail("endpoint_test_uid_mismatch")
        fit_uid_sha256 = job["fit_uid_sha256"]
        if (
            fit_uid_sha256 != p08_plan.NOT_APPLICABLE
            and fit_uid_sha256 != context_roles[cell["context_id"]]["fit_uid_sha256"]
        ):
            _fail("endpoint_fit_uid_mismatch")
        model_id = job["model_id"]
        spec = job["model_spec_sha256"]
        if model_id in spec_by_model and spec_by_model[model_id] != spec:
            _fail("model_spec_inconsistent")
        spec_by_model[model_id] = spec


def _parse_probability_scalar(value, code: str, *, allow_string: bool) -> float:
    if isinstance(value, (bool, np.bool_)):
        _fail(code)
    if isinstance(value, (bytes, complex, np.complexfloating)):
        _fail(code)
    if isinstance(value, str):
        if not allow_string:
            _fail(code)
        text = value
        if not text or text != text.strip():
            _fail(code)
        try:
            number = float(text)
        except ValueError:
            _fail(code)
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            _fail(code)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        _fail(code)
    return number


def _parse_probability_vector(value, code: str) -> list:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            _fail(code)
        items = value.tolist()
    elif isinstance(value, str):
        text = value
        if not text or text != text.strip():
            _fail(code)
        try:
            items = json.loads(text)
        except (TypeError, ValueError):
            _fail(code)
        if not isinstance(items, list):
            _fail(code)
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        _fail(code)
    if len(items) != 3:
        _fail(code)
    return [_parse_probability_scalar(item, code, allow_string=False) for item in items]


def _extract_probability_matrix(frame: pd.DataFrame, code: str) -> np.ndarray:
    has_vector = "probabilities" in frame.columns
    has_scalars = all(column in frame.columns for column in PROBABILITY_COLUMNS)
    if not has_vector and not has_scalars:
        _fail(f"{code}_probabilities_missing")
    vector_matrix = None
    scalar_matrix = None
    if has_vector:
        rows = [
            _parse_probability_vector(value, f"{code}_probability_vector_invalid")
            for value in frame["probabilities"].tolist()
        ]
        vector_matrix = np.asarray(rows, dtype=np.float64)
    if has_scalars:
        rows = []
        for position in range(len(frame)):
            rows.append(
                [
                    _parse_probability_scalar(
                        frame[column].iloc[position],
                        f"{code}_probability_scalar_invalid",
                        allow_string=True,
                    )
                    for column in PROBABILITY_COLUMNS
                ]
            )
        scalar_matrix = np.asarray(rows, dtype=np.float64)
    if vector_matrix is not None and scalar_matrix is not None:
        if vector_matrix.shape != scalar_matrix.shape or not np.allclose(
            vector_matrix, scalar_matrix, rtol=0.0, atol=_PROBABILITY_TOLERANCE
        ):
            _fail(f"{code}_probability_representations_differ")
        matrix = vector_matrix
    elif vector_matrix is not None:
        matrix = vector_matrix
    else:
        matrix = scalar_matrix
    if matrix.shape != (len(frame), 3):
        _fail(f"{code}_probability_shape")
    if not np.isfinite(matrix).all():
        _fail(f"{code}_probability_not_finite")
    if (matrix < 0.0).any() or (matrix > 1.0).any():
        _fail(f"{code}_probability_out_of_range")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=_PROBABILITY_TOLERANCE):
        _fail(f"{code}_probability_not_normalised")
    return matrix


def _require_optional_text(
    frame: pd.DataFrame, column: str, expected: str, code: str
) -> None:
    if column not in frame.columns:
        return
    for value in frame[column].tolist():
        if _strict_text(value, code) != expected:
            _fail(code)


def _require_optional_integral(
    frame: pd.DataFrame, column: str, expected: int, code: str
) -> None:
    if column not in frame.columns:
        return
    for value in frame[column].tolist():
        if _integral(value, code) != expected:
            _fail(code)


def _require_optional_seed_count(
    frame: pd.DataFrame, column: str, expected: int, code: str
) -> None:
    if column not in frame.columns:
        return
    for value in frame[column].tolist():
        if _seed_count(value, code) != expected:
            _fail(code)


def _require_optional_choice(
    frame: pd.DataFrame, column: str, allowed: set, code: str
) -> None:
    if column not in frame.columns:
        return
    for value in frame[column].tolist():
        if _strict_text(value, code) not in allowed:
            _fail(code)


def _check_probability_status(frame: pd.DataFrame, policy: str, recipe: str) -> None:
    present = "probability_status" in frame.columns
    if policy in NEW_POLICIES and not present:
        _fail("endpoint_probability_status_missing")
    if not present:
        return
    expected = _NEURAL_STATUS if recipe in NEURAL_RECIPES else _CLASSICAL_STATUS
    for value in frame["probability_status"].tolist():
        if _strict_text(value, "endpoint_probability_status_invalid") != expected:
            _fail("endpoint_probability_status_invalid")


def _validate_endpoint_frame(
    job: Mapping, context, registered_cell: pd.DataFrame, frame
) -> pd.DataFrame:
    context_id = job["context_id"]
    policy = job["policy_id"]
    recipe = job["model_id"]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        _fail("endpoint_frame_invalid")
    if frame.columns.duplicated().any():
        _fail("endpoint_frame_duplicate_columns")
    required = (
        "observation_uid",
        "master_sample_id",
        "instrument",
        "station",
        "true_label",
        "class_vocabulary",
        "predicted_label",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        _fail("endpoint_frame_columns_missing")
    reference = {
        row.observation_uid: row for row in registered_cell.itertuples(index=False)
    }
    frame_uids: list = []
    frame_masters: list = []
    frame_instruments: list = []
    frame_stations: list = []
    frame_labels: list = []
    frame_vocabularies: list = []
    frame_predicted: list = []
    seen: set = set()
    for row in frame.itertuples(index=False):
        observation_uid = _strict_text(
            row.observation_uid, "endpoint_observation_invalid"
        )
        if observation_uid in seen:
            _fail("endpoint_observation_duplicate")
        seen.add(observation_uid)
        reference_row = reference.get(observation_uid)
        if reference_row is None:
            _fail("endpoint_observation_unknown")
        master = _master_id(row.master_sample_id, "endpoint_master_invalid")
        instrument = _strict_text(row.instrument, "endpoint_instrument_invalid")
        station = _strict_text(row.station, "endpoint_station_invalid")
        label = _strict_text(row.true_label, "endpoint_label_invalid")
        vocabulary = _classes(row.class_vocabulary, "endpoint_vocabulary_invalid")
        predicted = _strict_text(row.predicted_label, "endpoint_predicted_invalid")
        if (
            master != reference_row.master_sample_id
            or instrument != reference_row.instrument
            or station != reference_row.station
            or label != reference_row.true_label
            or vocabulary != tuple(reference_row.class_vocabulary)
        ):
            _fail("endpoint_identity_mismatch")
        frame_uids.append(observation_uid)
        frame_masters.append(master)
        frame_instruments.append(instrument)
        frame_stations.append(station)
        frame_labels.append(label)
        frame_vocabularies.append(vocabulary)
        frame_predicted.append(predicted)
    if seen != set(reference):
        _fail("endpoint_uid_set_mismatch")
    matrix = _extract_probability_matrix(frame, "endpoint")
    for position, vocabulary in enumerate(frame_vocabularies):
        expected = vocabulary[int(np.argmax(matrix[position]))]
        if frame_predicted[position] != expected:
            _fail("endpoint_predicted_label_mismatch")

    _require_optional_text(frame, "context_id", context_id, "endpoint_context_mismatch")
    _require_optional_text(
        frame, "domain", str(context.domain), "endpoint_domain_mismatch"
    )
    _require_optional_text(
        frame,
        "held_instrument",
        str(context.held_instrument),
        "endpoint_held_instrument_mismatch",
    )
    _require_optional_text(
        frame,
        "representation_id",
        p08_plan.POLICY_REPRESENTATION[policy],
        "endpoint_representation_mismatch",
    )
    _require_optional_integral(
        frame, "outer_repeat", int(context.outer_repeat), "endpoint_repeat_mismatch"
    )
    _require_optional_integral(
        frame, "outer_fold", int(context.outer_fold), "endpoint_fold_mismatch"
    )
    allowed_models = {recipe}
    if recipe == D0_STRATEGY:
        allowed_models = {D0_STRATEGY, SELECTED_STRATEGY}
    elif recipe in NEURAL_RECIPES:
        allowed_models = {recipe, SELECTED_STRATEGY}
    _require_optional_choice(
        frame, "model_id", allowed_models, "endpoint_model_mismatch"
    )
    if policy in NEW_POLICIES:
        _require_optional_text(
            frame, "fit_id", job["job_id"], "endpoint_fit_id_mismatch"
        )
    expected_seeds = (
        _DETERMINISTIC_SEED_COUNT if recipe == "C-RBF-SVM" else _STOCHASTIC_SEED_COUNT
    )
    for column in ("seed_count", "technical_seed_count"):
        _require_optional_seed_count(
            frame, column, expected_seeds, "endpoint_seed_count_mismatch"
        )
    _check_probability_status(frame, policy, recipe)

    result = pd.DataFrame(
        {
            "context_id": [context_id] * len(frame_uids),
            "observation_uid": frame_uids,
            "master_sample_id": frame_masters,
            "instrument": frame_instruments,
            "station": frame_stations,
            "true_label": frame_labels,
            "class_vocabulary": frame_vocabularies,
            "probability_0": matrix[:, 0],
            "probability_1": matrix[:, 1],
            "probability_2": matrix[:, 2],
        }
    )
    return result.sort_values("observation_uid", kind="stable").reset_index(drop=True)


def assemble_panel(
    *,
    manifest: pd.DataFrame,
    contexts: pd.DataFrame,
    roles: pd.DataFrame,
    jobs,
    aliases,
    endpoint_frames,
) -> dict:
    """Assemble the registered universal panel from authenticated saved outputs.

    This is a consistency adapter, not a provenance authentication layer.  See
    the module docstring.
    """

    manifest_data = _parse_manifest(manifest)
    held = _select_contexts(contexts)
    held_lookup = {str(row.context_id): row for row in held.itertuples(index=False)}
    held_ids = set(held_lookup)
    context_roles = _parse_context_roles(roles, held, manifest_data)
    registered = _build_registered(held, context_roles, manifest_data)
    registered_by_context = {
        str(context_id): cell.reset_index(drop=True)
        for context_id, cell in registered.groupby("context_id", sort=True)
    }

    job_index = _index_jobs(jobs)
    endpoints = _collect_endpoints(job_index)
    alias_lookup = _collect_aliases(aliases, job_index, held_ids)
    cells = _report_cells(held, endpoints, alias_lookup)
    _verify_endpoints(cells, job_index, held_lookup, context_roles)

    if not isinstance(endpoint_frames, Mapping):
        _fail("endpoint_frames_must_be_mapping")
    referenced = sorted({cell["endpoint_job_id"] for cell in cells})
    if set(endpoint_frames.keys()) != set(referenced):
        _fail("endpoint_frames_mismatch")

    cache: dict[str, pd.DataFrame] = {}
    for job_id in referenced:
        job = job_index[job_id]
        context = held_lookup[job["context_id"]]
        cell = registered_by_context[job["context_id"]]
        cache[job_id] = _validate_endpoint_frame(
            job, context, cell, endpoint_frames[job_id]
        )

    prediction_frames = []
    for cell in cells:
        base = cache[cell["endpoint_job_id"]]
        tagged = base.copy()
        tagged["policy_id"] = cell["policy_id"]
        tagged["model_id"] = cell["model_id"]
        prediction_frames.append(tagged[list(_PREDICTION_COLUMNS)])
    predictions = pd.concat(prediction_frames, ignore_index=True)
    for column in PROBABILITY_COLUMNS:
        predictions[column] = predictions[column].astype(np.float64)

    reference_counts: dict[str, int] = {}
    for cell in cells:
        reference_counts[cell["endpoint_job_id"]] = (
            reference_counts.get(cell["endpoint_job_id"], 0) + 1
        )

    coverage_rows = []
    for cell in cells:
        expected = registered_by_context[cell["context_id"]]
        actual = cache[cell["endpoint_job_id"]]
        coverage_rows.append(
            {
                "context_id": cell["context_id"],
                "policy_id": cell["policy_id"],
                "model_id": cell["model_id"],
                "endpoint_job_id": cell["endpoint_job_id"],
                "recipe_id": cell["recipe_id"],
                "reference_count": reference_counts[cell["endpoint_job_id"]],
                "unique_endpoint": reference_counts[cell["endpoint_job_id"]] == 1,
                "expected_n_spectra": len(expected),
                "actual_n_spectra": len(actual),
                "expected_n_masters": int(expected.master_sample_id.nunique()),
                "actual_n_masters": int(actual.master_sample_id.nunique()),
                "complete": True,
            }
        )
    coverage = pd.DataFrame(coverage_rows, columns=list(_COVERAGE_COLUMNS))
    endpoint_index = pd.DataFrame(
        [
            {
                "context_id": cell["context_id"],
                "policy_id": cell["policy_id"],
                "model_id": cell["model_id"],
                "endpoint_job_id": cell["endpoint_job_id"],
                "recipe_id": cell["recipe_id"],
            }
            for cell in cells
        ],
        columns=list(_ENDPOINT_INDEX_COLUMNS),
    )

    domain_families: dict[str, object] = {}
    for context in held.itertuples(index=False):
        family = (
            manifest_data["families"].get(str(context.held_instrument))
            if manifest_data["has_family"]
            else None
        )
        key = str(context.domain)
        if key in domain_families and domain_families[key] != family:
            _fail("domain_family_conflict")
        domain_families[key] = family

    panels = p08_universal_units.build_units(predictions, registered)

    return {
        "registered_test_rows": registered,
        "predictions": predictions,
        "coverage": coverage,
        "global_masters": list(manifest_data["masters"]),
        "global_instruments": list(manifest_data["instruments"]),
        "domain_families": domain_families,
        "endpoint_index": endpoint_index,
        "panels": panels,
    }
