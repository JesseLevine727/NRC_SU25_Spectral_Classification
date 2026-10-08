"""P08-U1 T274: lazy all-candidate SOURCE input factory (metadata preparation).

Composition-only boundary above the frozen U0 source-input helpers.  It
authenticates the full universal plan, the three recorded metadata buffers, the
three recorded action archives, the specification audit and all eighteen
specification sources, binds physical source roles once per context/unit, then
exposes lazy per-fit :class:`p08_u0_runtime_inputs.RuntimePair` containers.

Non-claims
----------
* The factory is a metadata/kernel-argument preparation boundary.  It is not an
  execution permit, does not verify already-imported runtime code and never
  authorizes a scientific operation.
* ``SourceFactory`` holds immutable job snapshots, one role binding per
  context/unit and a bounded array cache; it does not eagerly materialise an
  array payload for every fit.
* No fitting, prediction, calibration, quantile computation, selection or
  filesystem access happens here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from atlas_sers.evaluation import p08_actions as _actions
from atlas_sers.evaluation import p08_plan as _plan
from atlas_sers.evaluation import p08_source_predictions as _sp
from atlas_sers.evaluation import p08_u0_arrays as _arrays
from atlas_sers.evaluation import p08_u0_inputs as _u0_inputs
from atlas_sers.evaluation import p08_u0_runtime_inputs as _runtime
from atlas_sers.governance.canonical import sha256_value

SCHEMA_VERSION = "nato-sers-p08-u1-source-inputs-v1"

_PLAN_SHA256 = "179b95e8011a5f6cc02c65c7fab1acf0f6a6241ba1ef02378aac207b9e19cb03"

_METADATA_KEYS = ("manifest_bytes", "contexts_bytes", "roles_bytes")
_ROLE_CACHE_MAXIMUM = 8
_FIXED_SPEC = "fixed_spec"
_JOB_ID_PREFIX = "P08JOB-"
_JSON_OPTIONS = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": True}

_REASON_CODES = frozenset(
    {
        "scientific_execution_not_authorized",
        "invalid_input",
        "invalid_plan",
        "plan_hash_mismatch",
        "invalid_jobs",
        "job_pair_mismatch",
        "metadata_mismatch",
        "action_invalid",
        "specification_mismatch",
        "candidate_mismatch",
        "role_binding_mismatch",
        "fit_unavailable",
        "source_preparation_failed",
        "unlisted_reason_code",
    }
)

__all__ = [
    "SCHEMA_VERSION",
    "SourceInputError",
    "SourceFactory",
    "prepare_source_factory",
    "require_scientific_execution",
]


class SourceInputError(ValueError):
    """ValueError carrying one static allowlisted reason code."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "unlisted_reason_code"
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(reason_code):
    raise SourceInputError(reason_code) from None


def _translate(error, fallback):
    code = getattr(error, "reason_code", None)
    if type(code) is str and code in _REASON_CODES:
        _fail(code)
    _fail(fallback)


def _stream_sha256(value):
    """Hash value with the canonical JSON encoding without one huge string."""
    encoder = json.JSONEncoder(**_JSON_OPTIONS)
    hasher = hashlib.sha256()
    for chunk in encoder.iterencode(value):
        hasher.update(chunk.encode("utf-8"))
    return hasher.hexdigest()


def _authenticate_plan(plan):
    if type(plan) is not dict:
        _fail("invalid_plan")
    if plan.get("schema_version") != _plan.SCHEMA_VERSION:
        _fail("invalid_plan")
    if plan.get("execution_authorized") is not False:
        _fail("invalid_plan")
    recorded = plan.get("plan_sha256")
    if type(recorded) is not str or recorded != _PLAN_SHA256:
        _fail("plan_hash_mismatch")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    try:
        computed = _stream_sha256(body)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_plan")
    if computed != _PLAN_SHA256:
        _fail("plan_hash_mismatch")
    jobs = plan.get("jobs")
    if type(jobs) is not list or not jobs:
        _fail("invalid_jobs")
    return jobs


def _authenticate_metadata(metadata_bytes):
    if type(metadata_bytes) is not dict or len(metadata_bytes) != len(_METADATA_KEYS):
        _fail("metadata_mismatch")
    for key in _METADATA_KEYS:
        if key not in metadata_bytes:
            _fail("metadata_mismatch")
    copied = {}
    for key in _METADATA_KEYS:
        value = metadata_bytes[key]
        if type(value) is not bytes:
            _fail("metadata_mismatch")
        limit = _u0_inputs._BYTE_LIMITS[key]
        if len(value) < _u0_inputs._BYTE_MINIMUM or len(value) > limit:
            _fail("metadata_mismatch")
        copied[key] = value
    for key in _METADATA_KEYS:
        if hashlib.sha256(copied[key]).hexdigest() != _u0_inputs._BYTE_PINS[key]:
            _fail("metadata_mismatch")
    return copied


def _authenticate_actions(action_bytes):
    try:
        copied = _arrays._copy_action_bytes(action_bytes)
    except _arrays.ArrayInputError:
        _fail("action_invalid")
    for representation_id in _arrays._ALL_REPRESENTATIONS:
        observed = hashlib.sha256(copied[representation_id]).hexdigest()
        if observed != _arrays._ACTION_PINS[representation_id]["file_sha256"]:
            _fail("action_invalid")
    return copied


def _decode_actions(action_copy, manifest_bytes):
    try:
        frame, manifest_uids = _arrays._read_manifest(manifest_bytes)
    except _arrays.ArrayInputError:
        _fail("action_invalid")
    row_order_sha256 = hashlib.sha256("\n".join(manifest_uids).encode("utf-8")).hexdigest()
    manifest_index = {uid: index for index, uid in enumerate(manifest_uids)}
    noise_map = {}
    range_map = {}
    for uid, noise, intensity_range in zip(
        frame["observation_uid"],
        frame["first_difference_noise_mad"],
        frame["intensity_range"],
        strict=True,
    ):
        noise_map[uid] = noise
        range_map[uid] = intensity_range

    actions = {}
    for representation_id in _arrays._ALL_REPRESENTATIONS:
        try:
            members = _arrays._extract_action_members(action_copy[representation_id])
            axis = _arrays._read_plain_action_array(
                members["axis_cm1.npy"], expected_shape=(_arrays.FEATURES,), unicode_member=False
            )
            intensity = _arrays._read_plain_action_array(
                members["intensity.npy"],
                expected_shape=(len(manifest_uids), _arrays.FEATURES),
                unicode_member=False,
            )
            uids_array = _arrays._read_plain_action_array(
                members["observation_uid.npy"],
                expected_shape=(len(manifest_uids),),
                unicode_member=True,
            )
        except _arrays.ArrayInputError:
            _fail("action_invalid")
        action = {"axis_cm1": axis, "intensity": intensity, "observation_uid": uids_array}
        pin = _arrays._ACTION_PINS[representation_id]
        row_meta = {
            "axis_sha256": pin["axis_sha256"],
            "array_sha256": pin["array_sha256"],
            "row_order_sha256": pin["row_order_sha256"],
        }
        try:
            _actions._validate_action(action, list(manifest_uids), row_meta, row_order_sha256)
        except _actions.ActionAuditError:
            _fail("action_invalid")
        actions[representation_id] = action
    return actions, manifest_uids, manifest_index, noise_map, range_map


def _load_specification(
    specification_audit_bytes, specification_source_bytes, candidate_registry_bytes
):
    try:
        audit_bytes = _runtime._copy_api_bytes(specification_audit_bytes, "invalid_input")
        registry_bytes = _runtime._copy_api_bytes(candidate_registry_bytes, "invalid_input")
    except _runtime.InputError:
        _fail("specification_mismatch")
    if hashlib.sha256(audit_bytes).hexdigest() != _runtime._SPEC_AUDIT_SHA256:
        _fail("specification_mismatch")
    if hashlib.sha256(registry_bytes).hexdigest() != _runtime._CANDIDATES_SHA256:
        _fail("candidate_mismatch")
    try:
        audit = _runtime._parse_json_object(audit_bytes, "invalid_input")
        specification_hashes, source_pins = _runtime._validate_audit(audit)
        source_copy = _runtime._copy_specification_sources(specification_source_bytes, source_pins)
        neural_contract = _runtime._read_neural_contract(source_copy)
    except _runtime.InputError:
        _fail("specification_mismatch")
    try:
        records = _runtime._read_candidate_registry(registry_bytes)
    except _runtime.InputError:
        _fail("candidate_mismatch")
    return specification_hashes, neural_contract, records


def _is_job_id(value):
    if type(value) is not str or len(value) != len(_JOB_ID_PREFIX) + 64:
        return False
    if value[: len(_JOB_ID_PREFIX)] != _JOB_ID_PREFIX:
        return False
    return _sp._is_lower_hex64(value[len(_JOB_ID_PREFIX) :])


def _authenticate_all_jobs(jobs):
    """Globally authenticate every graph job identity before filtering."""
    identifiers = set()
    for job in jobs:
        if type(job) is not dict:
            _fail("invalid_jobs")
        job_id = job.get("job_id")
        if not _is_job_id(job_id) or job_id in identifiers:
            _fail("invalid_jobs")
        identifiers.add(job_id)
        body = {key: value for key, value in job.items() if key != "job_id"}
        try:
            computed = _plan._hash(body)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("invalid_jobs")
        if job_id != _JOB_ID_PREFIX + computed:
            _fail("invalid_jobs")
    return identifiers


def _is_source_candidate(job):
    if job.get("stage") not in (_sp._FIT_STAGE, _sp._PREDICTION_STAGE):
        return False
    return job.get("policy_id") in _sp._SUPPORTED_POLICIES


def _validate_source_job(job):
    try:
        snapshot = _sp._validate_job(job)
        _sp._check_job_hash(snapshot)
        _sp._validate_job_semantics(snapshot)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_jobs")
    return snapshot


def _validate_source_configuration(fit, prediction):
    representation_id = fit["representation_id"]
    if prediction["representation_id"] != representation_id:
        _fail("specification_mismatch")
    pin = _arrays._ACTION_PINS.get(representation_id)
    if pin is None:
        _fail("action_invalid")
    if fit["array_sha256"] != pin["array_sha256"]:
        _fail("action_invalid")
    if prediction["array_sha256"] != pin["array_sha256"]:
        _fail("action_invalid")
    if prediction["model_id"] != fit["model_id"]:
        _fail("specification_mismatch")
    if prediction["model_spec_sha256"] != fit["model_spec_sha256"]:
        _fail("specification_mismatch")
    if prediction["candidate_id"] != fit["candidate_id"]:
        _fail("candidate_mismatch")
    if prediction["hyperparameter_sha256"] != fit["hyperparameter_sha256"]:
        _fail("candidate_mismatch")


def _snapshot_jobs(jobs):
    """Authenticate the full graph, then retain only source candidate pairs."""
    _authenticate_all_jobs(jobs)

    fits = {}
    predictions = {}
    for job in jobs:
        if not _is_source_candidate(job):
            continue
        if job["stage"] == _sp._FIT_STAGE:
            fits[job["job_id"]] = job
        else:
            dependencies = job.get("dependencies")
            if type(dependencies) is not list or len(dependencies) != 1:
                _fail("job_pair_mismatch")
            predictions.setdefault(dependencies[0], []).append(job)

    snapshots = []
    for job_id, fit in fits.items():
        matches = predictions.pop(job_id, None)
        if not matches or len(matches) != 1:
            _fail("job_pair_mismatch")
        prediction = matches[0]
        fit_snapshot = _validate_source_job(fit)
        prediction_snapshot = _validate_source_job(prediction)
        _validate_source_configuration(fit_snapshot, prediction_snapshot)
        try:
            _sp._validate_pair(fit_snapshot, prediction_snapshot)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("job_pair_mismatch")
        snapshots.append(fit_snapshot)
        snapshots.append(prediction_snapshot)
    if predictions:
        _fail("job_pair_mismatch")
    return snapshots


def _pair_jobs(snapshots):
    fits = {}
    predictions_by_fit = {}
    for snapshot in snapshots:
        stage = snapshot.get("stage")
        if stage == _u0_inputs._FIT_STAGE:
            fits.setdefault(snapshot["job_id"], snapshot)
        elif stage == _u0_inputs._PREDICTION_STAGE:
            dependencies = snapshot.get("dependencies")
            if type(dependencies) is not list or len(dependencies) != 1:
                _fail("invalid_jobs")
            predictions_by_fit.setdefault(dependencies[0], []).append(snapshot)
    pairs = []
    for job_id, fit in fits.items():
        matches = predictions_by_fit.pop(job_id, None)
        if not matches or len(matches) != 1:
            _fail("job_pair_mismatch")
        prediction = matches[0]
        try:
            _sp._validate_pair(fit, prediction)
        except SourceInputError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("job_pair_mismatch")
        pairs.append((fit, prediction))
    if predictions_by_fit:
        _fail("job_pair_mismatch")
    return pairs


def _retrievable(fit):
    if fit.get("stage") != _u0_inputs._FIT_STAGE:
        return False
    if fit.get("representation_id") not in _arrays._PREPARED_REPRESENTATIONS:
        return False
    if fit.get("policy_id") not in _arrays._PREPARED_POLICIES:
        return False
    return fit.get("resolution") == _FIXED_SPEC


def _parse_metadata(metadata_copy):
    try:
        manifest_rows = _u0_inputs._read_csv(
            metadata_copy["manifest_bytes"],
            _u0_inputs._MANIFEST_COLUMNS,
            _u0_inputs._MANIFEST_MAX_ROWS,
            "invalid_manifest",
        )
        observations = _u0_inputs._parse_manifest(manifest_rows)
        context_rows = _u0_inputs._read_csv(
            metadata_copy["contexts_bytes"],
            _u0_inputs._CONTEXT_COLUMNS,
            _u0_inputs._CONTEXT_MAX_ROWS,
            "invalid_contexts",
        )
        contexts = _u0_inputs._parse_contexts(context_rows)
        role_rows = _u0_inputs._read_csv(
            metadata_copy["roles_bytes"],
            _u0_inputs._ROLE_COLUMNS,
            _u0_inputs._ROLE_MAX_ROWS,
            "invalid_roles",
        )
        roles = _u0_inputs._parse_roles(role_rows)
    except SourceInputError:
        raise
    except _u0_inputs.BindingError:
        _fail("metadata_mismatch")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("metadata_mismatch")
    return observations, contexts, roles


def _bind_source_roles(pairs, contexts, observations, role_rows):
    representatives = {}
    for fit, prediction in pairs:
        key = (fit["context_id"], fit["unit_id"])
        representatives.setdefault(key, (fit, prediction))
    ordered = list(representatives.items())
    rows = [item for _key, item in ordered]
    try:
        source_pairs, _selected = _u0_inputs._bind_roles(rows, contexts, observations, role_rows)
    except SourceInputError:
        raise
    except _u0_inputs.BindingError:
        _fail("role_binding_mismatch")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("role_binding_mismatch")
    roles_by_key = {}
    for (key, _item), source_pair in zip(ordered, source_pairs, strict=True):
        roles_by_key[key] = source_pair.source_roles
    for fit, _prediction in pairs:
        key = (fit["context_id"], fit["unit_id"])
        representative = representatives[key][0]
        for field in ("fit_uid_sha256", "validation_uid_sha256", "test_uid_sha256"):
            if fit.get(field) != representative.get(field):
                _fail("role_binding_mismatch")
    return roles_by_key


def _build_candidate_index(records):
    index = {}
    for record in records:
        model_id = record["model_id"]
        if model_id not in _plan.CLASSICAL_MODELS:
            continue
        try:
            order = _runtime._parse_declared_order(record["declared_candidate_order"])
        except _runtime.InputError:
            _fail("candidate_mismatch")
        candidate_id = record["candidate_id"]
        expected_hash = record["hyperparameter_sha256"]
        try:
            parameters = json.loads(record["parameters_json"])
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("candidate_mismatch")
        if type(parameters) is not dict or sha256_value(parameters) != expected_hash:
            _fail("candidate_mismatch")
        key = (model_id, candidate_id)
        if key in index:
            _fail("candidate_mismatch")
        index[key] = {
            "model_id": model_id,
            "candidate_id": candidate_id,
            "hyperparameter_sha256": expected_hash,
            "parameters_json": json.dumps(parameters, **_JSON_OPTIONS),
            "declared_candidate_order": order,
        }
    return index


def _resolve_kernel(fit, prediction, specification_hashes, candidate_index, neural_contract):
    if prediction.get("representation_id") != fit.get("representation_id"):
        _fail("specification_mismatch")
    if prediction.get("policy_id") != fit.get("policy_id"):
        _fail("specification_mismatch")
    model_id = fit.get("model_id")
    if model_id in _plan.CLASSICAL_MODELS:
        candidate_id = fit.get("candidate_id")
        entry = candidate_index.get((model_id, candidate_id))
        if entry is None:
            _fail("candidate_mismatch")
        if fit.get("hyperparameter_sha256") != entry["hyperparameter_sha256"]:
            _fail("candidate_mismatch")
        if prediction.get("candidate_id") != candidate_id:
            _fail("candidate_mismatch")
        if prediction.get("hyperparameter_sha256") != entry["hyperparameter_sha256"]:
            _fail("candidate_mismatch")
        candidates = {model_id: entry}
    else:
        candidates = {}
    try:
        kind, configuration, parameters_json = _runtime._kernel_for_job(
            fit, specification_hashes, candidates, neural_contract
        )
        prediction_kind, prediction_configuration, _ = _runtime._kernel_for_job(
            prediction, specification_hashes, candidates, neural_contract
        )
    except _runtime.InputError as error:
        _translate(error, "specification_mismatch")
    if prediction_kind != kind:
        _fail("specification_mismatch")
    if prediction_configuration.get("model_id") != configuration.get("model_id"):
        _fail("specification_mismatch")
    if prediction.get("model_spec_sha256") != fit.get("model_spec_sha256"):
        _fail("specification_mismatch")
    return kind, configuration, parameters_json


def _read_substrate(manifest_bytes):
    try:
        return _runtime._read_substrate_metadata(manifest_bytes)
    except _runtime.InputError:
        _fail("metadata_mismatch")


@dataclass(frozen=True, repr=False)
class _FitRecord:
    fit_job_json: str
    prediction_job_json: str
    context_id: str
    unit_id: str
    policy_id: str
    representation_id: str
    model_id: str


class _BoundedCache:
    __slots__ = ("_maximum", "_items", "_order")

    def __init__(self, maximum):
        self._maximum = maximum
        self._items = {}
        self._order = []

    def get(self, key):
        return self._items.get(key)

    def put(self, key, value):
        if key in self._items:
            return
        if len(self._order) >= self._maximum:
            oldest = self._order.pop(0)
            self._items.pop(oldest, None)
        self._items[key] = value
        self._order.append(key)


class SourceFactory:
    """Lazy per-fit kernel-argument container over authenticated source inputs."""

    def __init__(
        self,
        *,
        records,
        roles,
        specification_hashes,
        candidate_index,
        neural_contract,
        actions,
        manifest_index,
        noise_map,
        range_map,
        substrate_map,
        report_json,
    ):
        self._records = records
        self._ids = tuple(sorted(records))
        self._roles = roles
        self._specification_hashes = specification_hashes
        self._candidate_index = candidate_index
        self._neural_contract = neural_contract
        self._actions = actions
        self._manifest_index = manifest_index
        self._noise_map = noise_map
        self._range_map = range_map
        self._substrate_map = substrate_map
        self._observations_cache = {}
        self._role_cache = _BoundedCache(_ROLE_CACHE_MAXIMUM)
        self._report_json = report_json

    def fit_job_ids(self):
        return self._ids

    def public_report(self):
        return json.loads(self._report_json)

    def pair(self, fit_job_id):
        if type(fit_job_id) is not str or fit_job_id not in self._records:
            _fail("fit_unavailable")
        record = self._records[fit_job_id]
        try:
            fit = json.loads(record.fit_job_json)
            prediction = json.loads(record.prediction_job_json)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("source_preparation_failed")
        _kind, configuration, parameters_json = _resolve_kernel(
            fit,
            prediction,
            self._specification_hashes,
            self._candidate_index,
            self._neural_contract,
        )
        key = (record.context_id, record.unit_id)
        source_roles = self._roles[key]
        cache_key = (record.policy_id, record.context_id, record.unit_id)
        prepared_role = self._role_cache.get(cache_key)
        if prepared_role is None:
            try:
                prepared_role = _arrays._build_prepared_role(
                    source_roles,
                    record.policy_id,
                    record.representation_id,
                    self._actions,
                    self._manifest_index,
                    self._noise_map,
                    self._range_map,
                )
            except SourceInputError:
                raise
            except _arrays.ArrayInputError:
                _fail("source_preparation_failed")
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                _fail("source_preparation_failed")
            self._role_cache.put(cache_key, prepared_role)
        observations = self._observations_cache.get(key)
        if observations is None:
            try:
                observations = _runtime._build_observations(source_roles, self._substrate_map)
            except SourceInputError:
                raise
            except _runtime.InputError:
                _fail("source_preparation_failed")
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                _fail("source_preparation_failed")
            self._observations_cache[key] = observations
        fitting_observations, validation_observations = observations
        prepared_pair = _arrays.PreparedPair(
            fit_job_json=record.fit_job_json,
            prediction_job_json=record.prediction_job_json,
            inputs=prepared_role,
        )
        return _runtime.RuntimePair(
            prepared_pair=prepared_pair,
            parameters_json=parameters_json,
            configuration_json=json.dumps(configuration, **_JSON_OPTIONS),
            fitting_observations=fitting_observations,
            validation_observations=validation_observations,
        )


def _build_report(records, roles_by_key, candidate_index):
    fits = list(records.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": _PLAN_SHA256,
        "specification_audit_sha256": _runtime._SPEC_AUDIT_SHA256,
        "candidate_registry_sha256": _runtime._CANDIDATES_SHA256,
        "source_fit_jobs": len(fits),
        "source_prediction_jobs": len(fits),
        "models": len({record.model_id for record in fits}),
        "candidates": len(candidate_index),
        "policies": len({record.policy_id for record in fits}),
        "contexts": len({record.context_id for record in fits}),
        "units": len(roles_by_key),
        "metadata_bytes_verified": True,
        "action_bytes_verified": True,
        "specification_bytes_verified": True,
        "source_roles_bound": True,
        "source_arrays_verified": True,
        "loaded_runtime_code_verified": False,
        "live_controller_verified": False,
        "execution_authorized": False,
        "new_scientific_operations": 0,
    }
    report["report_sha256"] = sha256_value(report)
    return json.dumps(report, **_JSON_OPTIONS)


def _compose(
    *,
    plan,
    metadata_bytes,
    action_bytes,
    specification_audit_bytes,
    specification_source_bytes,
    candidate_registry_bytes,
):
    jobs = _authenticate_plan(plan)
    metadata_copy = _authenticate_metadata(metadata_bytes)
    action_copy = _authenticate_actions(action_bytes)
    specification_hashes, neural_contract, records = _load_specification(
        specification_audit_bytes, specification_source_bytes, candidate_registry_bytes
    )
    candidate_index = _build_candidate_index(records)

    snapshots = _snapshot_jobs(jobs)
    pairs = _pair_jobs(snapshots)
    retrievable = [(fit, prediction) for fit, prediction in pairs if _retrievable(fit)]

    observations, contexts, role_rows = _parse_metadata(metadata_copy)
    roles_by_key = _bind_source_roles(retrievable, contexts, observations, role_rows)

    actions, _manifest_uids, manifest_index, noise_map, range_map = _decode_actions(
        action_copy, metadata_copy["manifest_bytes"]
    )
    substrate_map = _read_substrate(metadata_copy["manifest_bytes"])

    fit_records = {}
    for fit, prediction in retrievable:
        fit_records[fit["job_id"]] = _FitRecord(
            fit_job_json=json.dumps(fit, sort_keys=True, ensure_ascii=True),
            prediction_job_json=json.dumps(prediction, sort_keys=True, ensure_ascii=True),
            context_id=fit["context_id"],
            unit_id=fit["unit_id"],
            policy_id=fit["policy_id"],
            representation_id=fit["representation_id"],
            model_id=fit.get("model_id"),
        )

    report_json = _build_report(fit_records, roles_by_key, candidate_index)
    return SourceFactory(
        records=fit_records,
        roles=roles_by_key,
        specification_hashes=specification_hashes,
        candidate_index=candidate_index,
        neural_contract=neural_contract,
        actions=actions,
        manifest_index=manifest_index,
        noise_map=noise_map,
        range_map=range_map,
        substrate_map=substrate_map,
        report_json=report_json,
    )


def prepare_source_factory(
    *,
    plan,
    metadata_bytes,
    action_bytes,
    specification_audit_bytes,
    specification_source_bytes,
    candidate_registry_bytes,
):
    """Authenticate universal source inputs and return a lazy source factory.

    Ordinary failures collapse to a static allowlisted :class:`SourceInputError`;
    ``KeyboardInterrupt`` and ``SystemExit`` propagate unchanged.
    """

    try:
        return _compose(
            plan=plan,
            metadata_bytes=metadata_bytes,
            action_bytes=action_bytes,
            specification_audit_bytes=specification_audit_bytes,
            specification_source_bytes=specification_source_bytes,
            candidate_registry_bytes=candidate_registry_bytes,
        )
    except SourceInputError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("source_preparation_failed")


def require_scientific_execution(*args, **kwargs):
    """Always deny execution, regardless of forged flags or arguments."""

    _fail("scientific_execution_not_authorized")
