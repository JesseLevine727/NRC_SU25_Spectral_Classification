"""Prepare permitted U0 source arrays from recorded P08 metadata and actions.

This module is a fixed composition boundary.  It

* authenticates three caller-supplied frozen action archive byte strings
  against the module's own fixed pins *before* any ZIP or NumPy parsing,
* asks the inherited P01 metadata binder to authenticate the five metadata
  byte strings,
* reads the ordered source UID list from the authenticated manifest CSV,
* loads exactly the three frozen action arrays with a bounded, reviewed plain
  NPY reader that uses ``allow_pickle=False``,
* reuses :func:`atlas_sers.evaluation.p08_actions._validate_action` to verify
  each action array,
* selects only the U0 fitting/validation rows named by the authenticated
  source roles, and
* preserves the inherited native P05 fit-row noise metadata without
  recomputing quantiles, preprocessing, models or augmentation.

Fixed recorded metadata and array byte identity
-----------------------------------------------
The action pins are fixed module constants and there is no pin-override path
in the API.  The authenticated manifest and the action arrays are treated as
recorded metadata whose byte identity is checked against those constants and
the inherited binder.  This is byte-identity verification only: it does not
establish independent historical provenance, model parity, a live-controller
permit or resource enforcement.

Ownership
---------
The caller must own the input dictionaries for the duration of the snapshot
capture.  After capture the input byte values are plain ``bytes``, but the
capture itself is not an atomic multi-memory ownership proof.

Non-claims
----------
* ``arrays_verified``, ``ordered_source_rows_verified`` and
  ``native_noise_metadata_preserved`` are internal consistency statements
  against the module's fixed recorded metadata and the authenticated inputs
  only.  They are not provenance, external-registry, physical-isolation,
  parity or resource evidence.
* The NPY reader is a fixed reviewed reader for this repository's own writer
  format.  It is not a hostile-file sandbox.
* No fitting, prediction, calibration, quantile computation, augmentation or
  file I/O happens here.
* No UID, role, context, class, row value, noise value or path leaves the
  public report.
* Execution through this module is never authorized.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p08_actions as _actions
from atlas_sers.evaluation import p08_plan as _plan
from atlas_sers.evaluation import p08_u0_inputs as _u0_inputs
from atlas_sers.governance.canonical import sha256_value

SCHEMA_VERSION = "nato-sers-p08-u0-source-arrays-v1"
FEATURES = 1401

_ALL_REPRESENTATIONS = ("R_MIN_400_1800", "R_SG_400_1800", "R_ARPLS_400_1800")
_PREPARED_REPRESENTATIONS = ("R_SG_400_1800", "R_ARPLS_400_1800")
_PREPARED_POLICIES = ("PP-U-SG", "PP-U-ARPLS")

_METADATA_KEYS = frozenset(
    {
        "proposal_bytes",
        "attempt_manifest_bytes",
        "manifest_bytes",
        "contexts_bytes",
        "roles_bytes",
    }
)
_ACTION_KEYS = frozenset(_ALL_REPRESENTATIONS)

_MEMBER_NAMES = frozenset({"axis_cm1.npy", "intensity.npy", "observation_uid.npy"})
_MANIFEST_COLUMNS = (
    "observation_uid",
    "first_difference_noise_mad",
    "intensity_range",
)

_AXIS_SHA256 = "5065a3af2dd37f3dd41780ac57f1da7c1bf24c07398f05f0082932d728cbf2d2"
_ROW_ORDER_SHA256 = "b0d9ef9ae34a87443d951742bf1b295df522dd12674d5ca4a9785953cee6a5a7"

_ACTION_PINS = {
    "R_MIN_400_1800": {
        "file_sha256": "db942996c324b7b9f7f7d73b2b371bacffd71ab8400f230e7e16c1d4fa53c2cf",
        "array_sha256": "9b3d815b618c5cf3a7a29a0b571d3b2004326eae03d79ae4ebcaafb94600727e",
        "axis_sha256": _AXIS_SHA256,
        "row_order_sha256": _ROW_ORDER_SHA256,
    },
    "R_SG_400_1800": {
        "file_sha256": "3af989fa417c887b105e3d04559cc41153412d72186575f11c6f18e3722a60ff",
        "array_sha256": "0f79f3820e4f29df9e5336a78f08d89bbb4c7bc25b5969b6fb0f6b6d076b739d",
        "axis_sha256": _AXIS_SHA256,
        "row_order_sha256": _ROW_ORDER_SHA256,
    },
    "R_ARPLS_400_1800": {
        "file_sha256": "f04aef601a8e9c962864f4d23b23612cf8db21caef2df36bf365f17d79d286f8",
        "array_sha256": "43de38c9648a1650301e3f5ace4aecfabef5adaa5357bc0a46a5f733b7af9b94",
        "axis_sha256": _AXIS_SHA256,
        "row_order_sha256": _ROW_ORDER_SHA256,
    },
}

_MAXIMUM_ARCHIVE_BYTES = 8 * 1024 * 1024
_MAX_HEADER_BYTES = 65536
_UNICODE_ITEMSIZE_MINIMUM = 4
_UNICODE_ITEMSIZE_MAXIMUM = 1024

_REASON_CODES = frozenset(
    {
        "scientific_execution_not_authorized",
        "invalid_arguments",
        "invalid_metadata_bytes",
        "invalid_action_bytes",
        "metadata_binding_failed",
        "invalid_action_keys",
        "action_too_large",
        "action_file_hash_mismatch",
        "invalid_action_archive",
        "action_member_mismatch",
        "action_encrypted",
        "action_compression_unsupported",
        "invalid_action_array",
        "action_array_shape_mismatch",
        "action_array_dtype_mismatch",
        "action_array_length_mismatch",
        "action_array_invalid",
        "invalid_manifest",
        "manifest_rows_mismatch",
        "invalid_noise_metadata",
        "noise_coverage_mismatch",
        "noise_value_invalid",
        "invalid_pair",
        "policy_representation_mismatch",
        "array_pin_mismatch",
        "role_uid_mismatch",
        "policy_count_mismatch",
        "preparation_failed",
        "invalid_input",
        "action_invalid",
        "action_keys",
        "axis_invalid",
        "intensity_invalid",
        "intensity_nonfinite",
        "normalization_failed",
        "uid_invalid",
        "uid_mismatch",
        "axis_sha_mismatch",
        "array_sha_mismatch",
        "row_order_sha_mismatch",
        "unlisted_reason_code",
    }
)

__all__ = [
    "SCHEMA_VERSION",
    "ArrayInputError",
    "PreparedRole",
    "PreparedPair",
    "PreparedBinding",
    "prepare_u0_source_arrays",
    "require_scientific_execution",
]


class ArrayInputError(ValueError):
    """ValueError carrying one static allowlisted reason code."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "unlisted_reason_code"
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(reason_code):
    raise ArrayInputError(reason_code) from None


def _translate(error, fallback):
    """Re-raise a delegated static reason, otherwise a static fallback."""

    code = getattr(error, "reason_code", None)
    if type(code) is str and code in _REASON_CODES:
        _fail(code)
    _fail(fallback)


def _as_real(value, reason_code):
    if isinstance(value, (bool, np.bool_)):
        _fail(reason_code)
    if isinstance(value, (complex, np.complexfloating, bytes, bytearray)):
        _fail(reason_code)
    if isinstance(value, str):
        try:
            number = float(value)
        except (OverflowError, ValueError, TypeError):
            _fail(reason_code)
        if math.isfinite(number):
            return number
        _fail(reason_code)
    if isinstance(value, (int, np.integer, float, np.floating)):
        try:
            number = float(value)
        except (OverflowError, ValueError, TypeError):
            _fail(reason_code)
        if math.isfinite(number):
            return number
    _fail(reason_code)


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
    for representation_id in _ALL_REPRESENTATIONS:
        if representation_id not in action_bytes:
            _fail("invalid_action_keys")
        value = action_bytes[representation_id]
        if type(value) is not bytes or len(value) == 0:
            _fail("invalid_action_bytes")
        if len(value) > _MAXIMUM_ARCHIVE_BYTES:
            _fail("action_too_large")
        copied[representation_id] = value
    return copied


def _authenticate_action_hashes(action_copy):
    for representation_id in _ALL_REPRESENTATIONS:
        observed = hashlib.sha256(action_copy[representation_id]).hexdigest()
        if observed != _ACTION_PINS[representation_id]["file_sha256"]:
            _fail("action_file_hash_mismatch")


def _read_manifest(manifest_bytes):
    try:
        frame = pd.read_csv(
            io.BytesIO(manifest_bytes),
            usecols=list(_MANIFEST_COLUMNS),
            keep_default_na=False,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_manifest")
    try:
        frame = frame.reindex(columns=list(_MANIFEST_COLUMNS))
    except Exception:
        _fail("invalid_manifest")
    if frame.shape[0] == 0:
        _fail("invalid_manifest")
    uids = frame["observation_uid"].tolist()
    for uid in uids:
        if type(uid) is not str or not uid.strip():
            _fail("invalid_manifest")
    if len(set(uids)) != len(uids):
        _fail("invalid_manifest")
    return frame, uids


def _extract_action_members(blob):
    """Return the three bounded raw NPY members of one frozen action archive."""

    if type(blob) is not bytes or len(blob) == 0:
        _fail("invalid_action_archive")
    if len(blob) > _MAXIMUM_ARCHIVE_BYTES:
        _fail("action_too_large")
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob), "r")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_action_archive")

    with archive:
        try:
            infos = archive.infolist()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("invalid_action_archive")
        if len(infos) != len(_MEMBER_NAMES):
            _fail("action_member_mismatch")

        names = []
        expanded_total = 0
        compressed_total = 0
        for info in infos:
            if info.is_dir():
                _fail("action_member_mismatch")
            if info.flag_bits & 0x1:
                _fail("action_encrypted")
            if info.compress_type not in (
                zipfile.ZIP_STORED,
                zipfile.ZIP_DEFLATED,
            ):
                _fail("action_compression_unsupported")
            if type(info.filename) is not str or info.filename not in _MEMBER_NAMES:
                _fail("action_member_mismatch")
            if info.orig_filename != info.filename:
                _fail("action_member_mismatch")
            declared = info.file_size
            if type(declared) is not int or declared < 0:
                _fail("action_too_large")
            compressed = info.compress_size
            if type(compressed) is not int or compressed < 0:
                _fail("action_too_large")
            expanded_total += declared
            compressed_total += compressed
            names.append(info.filename)
        if len(set(names)) != len(_MEMBER_NAMES):
            _fail("action_member_mismatch")
        if expanded_total > _MAXIMUM_ARCHIVE_BYTES:
            _fail("action_too_large")
        if compressed_total > _MAXIMUM_ARCHIVE_BYTES:
            _fail("action_too_large")

        members = {}
        for info in infos:
            declared = info.file_size
            try:
                raw = archive.read(info)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                _fail("invalid_action_archive")
            if type(raw) is not bytes or len(raw) != declared:
                _fail("invalid_action_archive")
            members[info.filename] = raw
    return members


def _read_array_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return version, np.lib.format.read_array_header_1_0(
            stream, max_header_size=_MAX_HEADER_BYTES
        )
    if version == (2, 0):
        return version, np.lib.format.read_array_header_2_0(
            stream, max_header_size=_MAX_HEADER_BYTES
        )
    _fail("invalid_action_array")


def _read_plain_action_array(raw, *, expected_shape, unicode_member):
    """Parse one bounded plain NPY array without honouring object pickles."""

    if type(raw) is not bytes or len(raw) == 0:
        _fail("invalid_action_array")
    stream = io.BytesIO(raw)
    try:
        _version, (shape, fortran_order, dtype) = _read_array_header(stream)
    except ArrayInputError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_action_array")

    if type(shape) is not tuple or tuple(shape) != tuple(expected_shape):
        _fail("action_array_shape_mismatch")
    if not isinstance(dtype, np.dtype):
        _fail("action_array_dtype_mismatch")
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        _fail("action_array_dtype_mismatch")
    if not dtype.isnative:
        _fail("action_array_dtype_mismatch")
    if unicode_member:
        if dtype.kind != "U" or not (
            _UNICODE_ITEMSIZE_MINIMUM <= int(dtype.itemsize) <= _UNICODE_ITEMSIZE_MAXIMUM
        ):
            _fail("action_array_dtype_mismatch")
    else:
        if dtype.kind != "f" or int(dtype.itemsize) != 4:
            _fail("action_array_dtype_mismatch")
    if fortran_order is not True and fortran_order is not False:
        _fail("invalid_action_array")

    count = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            _fail("action_array_shape_mismatch")
        count *= dimension

    header_end = stream.tell()
    if header_end + count * int(dtype.itemsize) != len(raw):
        _fail("action_array_length_mismatch")

    stream.seek(0)
    try:
        array = np.lib.format.read_array(
            stream, allow_pickle=False, max_header_size=_MAX_HEADER_BYTES
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("action_array_invalid")
    if type(array) is not np.ndarray:
        _fail("action_array_invalid")
    if tuple(array.shape) != tuple(expected_shape) or array.dtype != dtype:
        _fail("action_array_invalid")
    return array


def _parse_job(job_json):
    if type(job_json) is not str or not job_json:
        _fail("invalid_pair")
    try:
        job = json.loads(job_json)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_pair")
    if type(job) is not dict:
        _fail("invalid_pair")
    return job


def _resolve_policy_and_representation(job):
    if type(job) is not dict:
        _fail("invalid_pair")
    policy_id = job.get("policy_id")
    representation_id = job.get("representation_id")
    if type(policy_id) is not str or policy_id not in _PREPARED_POLICIES:
        _fail("policy_representation_mismatch")
    expected = _plan.POLICY_REPRESENTATION.get(policy_id)
    if type(expected) is not str or expected not in _PREPARED_REPRESENTATIONS:
        _fail("policy_representation_mismatch")
    if representation_id != expected:
        _fail("policy_representation_mismatch")
    return policy_id, expected


def _role_uids(observations):
    uids = []
    for observation in observations:
        uid = getattr(observation, "observation_uid", None)
        if type(uid) is not str or not uid:
            _fail("role_uid_mismatch")
        uids.append(uid)
    if len(set(uids)) != len(uids):
        _fail("role_uid_mismatch")
    return uids


def _select_payload(full_intensity, uids, manifest_index):
    rows = np.empty((len(uids), FEATURES), dtype=np.float32)
    for position, uid in enumerate(uids):
        index = manifest_index.get(uid)
        if index is None:
            _fail("role_uid_mismatch")
        rows[position, :] = full_intensity[index, :]
    return rows.tobytes(order="C"), len(uids)


def _build_prepared_role(
    source_roles,
    policy_id,
    representation_id,
    actions,
    manifest_index,
    noise_map,
    range_map,
):
    fitting_uids = _role_uids(source_roles.fitting)
    validation_uids = _role_uids(source_roles.validation)
    full_intensity = actions[representation_id]["intensity"]
    fitting_payload, fitting_rows = _select_payload(full_intensity, fitting_uids, manifest_index)
    validation_payload, validation_rows = _select_payload(
        full_intensity, validation_uids, manifest_index
    )

    noise_records = []
    for uid in fitting_uids:
        if uid not in noise_map or uid not in range_map:
            _fail("noise_coverage_mismatch")
        noise = _as_real(noise_map[uid], "noise_value_invalid")
        intensity_range = _as_real(range_map[uid], "noise_value_invalid")
        if noise < 0.0:
            _fail("noise_value_invalid")
        if intensity_range <= 0.0:
            _fail("noise_value_invalid")
        noise_records.append((uid, float(noise), float(intensity_range)))

    return PreparedRole(
        source_roles=source_roles,
        policy_id=policy_id,
        representation_id=representation_id,
        fitting_payload=fitting_payload,
        validation_payload=validation_payload,
        noise_records=tuple(noise_records),
        _fitting_rows=fitting_rows,
        _validation_rows=validation_rows,
    )


@dataclass(frozen=True, repr=False)
class PreparedRole:
    """One prepared policy/context/unit U0 source role."""

    source_roles: Any
    policy_id: str
    representation_id: str
    fitting_payload: bytes
    validation_payload: bytes
    noise_records: tuple
    _fitting_rows: int
    _validation_rows: int

    def fitting_values(self):
        array = np.frombuffer(self.fitting_payload, dtype=np.float32)
        array = array.reshape(self._fitting_rows, FEATURES)
        array.setflags(write=False)
        return array

    def validation_values(self):
        array = np.frombuffer(self.validation_payload, dtype=np.float32)
        array = array.reshape(self._validation_rows, FEATURES)
        array.setflags(write=False)
        return array

    def fitting_noise_frame(self):
        uids = [record[0] for record in self.noise_records]
        noise = [record[1] for record in self.noise_records]
        ranges = [record[2] for record in self.noise_records]
        return pd.DataFrame(
            {
                "observation_uid": uids,
                "first_difference_noise_mad": noise,
                "intensity_range": ranges,
            },
            columns=list(_MANIFEST_COLUMNS),
        )


@dataclass(frozen=True, repr=False)
class PreparedPair:
    """One prepared U0 source pair bound to a prepared role."""

    fit_job_json: str
    prediction_job_json: str
    inputs: PreparedRole


@dataclass(frozen=True, repr=False)
class PreparedBinding:
    """Prepared U0 source pairs plus a privacy-safe public report."""

    pairs: tuple
    report_json: str

    def public_report(self):
        return json.loads(self.report_json)


def _prepare_u0_source_arrays(*, metadata_bytes, action_bytes):
    metadata_copy = _copy_metadata(metadata_bytes)
    action_copy = _copy_action_bytes(action_bytes)
    _authenticate_action_hashes(action_copy)

    try:
        binding = _u0_inputs.bind_u0_source_metadata(**metadata_copy)
    except ArrayInputError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        _translate(error, "metadata_binding_failed")

    frame, manifest_uids = _read_manifest(metadata_copy["manifest_bytes"])
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
    for representation_id in _ALL_REPRESENTATIONS:
        members = _extract_action_members(action_copy[representation_id])
        axis = _read_plain_action_array(
            members["axis_cm1.npy"],
            expected_shape=(FEATURES,),
            unicode_member=False,
        )
        intensity = _read_plain_action_array(
            members["intensity.npy"],
            expected_shape=(len(manifest_uids), FEATURES),
            unicode_member=False,
        )
        uids_array = _read_plain_action_array(
            members["observation_uid.npy"],
            expected_shape=(len(manifest_uids),),
            unicode_member=True,
        )
        action = {
            "axis_cm1": axis,
            "intensity": intensity,
            "observation_uid": uids_array,
        }
        pin = _ACTION_PINS[representation_id]
        row_meta = {
            "axis_sha256": pin["axis_sha256"],
            "array_sha256": pin["array_sha256"],
            "row_order_sha256": pin["row_order_sha256"],
        }
        try:
            _actions._validate_action(action, list(manifest_uids), row_meta, row_order_sha256)
        except _actions.ActionAuditError as error:
            _translate(error, "action_invalid")
        actions[representation_id] = action

    pairs = []
    cache = {}
    policies_seen = set()
    for pair in binding.pairs:
        fit_job = _parse_job(pair.fit_job_json)
        prediction_job = _parse_job(pair.prediction_job_json)
        policy_id, representation_id = _resolve_policy_and_representation(fit_job)
        pred_policy, pred_representation = _resolve_policy_and_representation(prediction_job)
        if pred_policy != policy_id or pred_representation != representation_id:
            _fail("policy_representation_mismatch")

        expected_array_sha256 = _ACTION_PINS[representation_id]["array_sha256"]
        if fit_job.get("array_sha256") != expected_array_sha256:
            _fail("array_pin_mismatch")
        if prediction_job.get("array_sha256") != expected_array_sha256:
            _fail("array_pin_mismatch")

        source_roles = pair.source_roles
        context_id = getattr(source_roles, "context_id", None)
        unit_id = getattr(source_roles, "unit_id", None)
        if type(context_id) is not str or type(unit_id) is not str:
            _fail("invalid_pair")
        key = (policy_id, context_id, unit_id)
        prepared_role = cache.get(key)
        if prepared_role is None:
            prepared_role = _build_prepared_role(
                source_roles,
                policy_id,
                representation_id,
                actions,
                manifest_index,
                noise_map,
                range_map,
            )
            cache[key] = prepared_role
        pairs.append(
            PreparedPair(
                fit_job_json=pair.fit_job_json,
                prediction_job_json=pair.prediction_job_json,
                inputs=prepared_role,
            )
        )
        policies_seen.add(policy_id)

    if len(policies_seen) != len(_PREPARED_POLICIES):
        _fail("policy_count_mismatch")

    binding_report = binding.public_report()
    metadata_binding_report_sha256 = binding_report["report_sha256"]

    report = {
        "schema_version": SCHEMA_VERSION,
        "metadata_binding_report_sha256": metadata_binding_report_sha256,
        "action_pins": {key: dict(value) for key, value in _ACTION_PINS.items()},
        "manifest_rows": len(manifest_uids),
        "features": FEATURES,
        "authenticated_action_count": len(_ALL_REPRESENTATIONS),
        "prepared_policy_count": len(policies_seen),
        "prepared_source_roles": len(cache),
        "source_fit_jobs": len(pairs),
        "source_prediction_jobs": len(pairs),
        "arrays_verified": True,
        "ordered_source_rows_verified": True,
        "native_noise_metadata_preserved": True,
        "noise_quantiles_computed": False,
        "preprocessing_recomputed": False,
        "model_parameters_loaded": False,
        "live_controller_verified": False,
        "execution_authorized": False,
        "new_scientific_operations": 0,
    }
    report["report_sha256"] = sha256_value(report)
    report_json = json.dumps(report, sort_keys=True, separators=(",", ":"))
    return PreparedBinding(pairs=tuple(pairs), report_json=report_json)


def prepare_u0_source_arrays(*, metadata_bytes, action_bytes):
    """Prepare permitted U0 source arrays from authenticated inputs.

    Ordinary failures collapse to a static allowlisted :class:`ArrayInputError`;
    ``KeyboardInterrupt`` and ``SystemExit`` propagate unchanged.
    """

    try:
        return _prepare_u0_source_arrays(metadata_bytes=metadata_bytes, action_bytes=action_bytes)
    except ArrayInputError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("preparation_failed")


def require_scientific_execution(*args, **kwargs):
    """Always deny execution, regardless of forged flags or arguments."""

    _fail("scientific_execution_not_authorized")
