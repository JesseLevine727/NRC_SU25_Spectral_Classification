"""Composition of the accepted P08 neural source-artifact checks (P08-T088).

This module binds three already-accepted in-memory boundaries against the
inherited ``validation_logits.npz`` archive bytes produced by
``p05_pilot._save_logits``:

* :func:`atlas_sers.evaluation.p08_source_artifacts.verify_neural_checkpoint_bytes`
* :func:`atlas_sers.evaluation.p08_training_record.verify_neural_training_record`
* :func:`atlas_sers.evaluation.p08_source_predictions.verify_source_prediction_values`

It adds no new artifact format, no new scientific algorithm, no fit, no
prediction and no execution authority.

Non-claims
----------
* ``bundle_consistency_verified``, ``supplied_file_hashes_verified`` and
  ``training_record_pin_verified`` record only internal agreement against
  caller-supplied pins, bytes and metadata.  They are not evidence of
  provenance, external registry membership, physical-master independence,
  real fitting, prediction parity or resource bounds.
* A same-shaped finite but wrong score matrix is externally indistinguishable
  from a correct one here; this composition does not detect model-output
  parity.
* The NPZ reader is bounded parsing of this repository's own plain-array
  writer format.  It is not a hostile-file sandbox and does not defend against
  arbitrary malicious Torch pickles.
* No observations, labels, role identifiers, recipe identifiers, seeds,
  scores, histories, tensors, paths or blobs leave this interface.
* Execution through this module is never authorized.

The caller must not mutate jobs, the training record or validation sequences
while their snapshots are being captured.  The verifier does not modify those
inputs.  Private copies protect against mutation after capture; they do not
provide an atomic concurrency guarantee.
"""

from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np

from atlas_sers.evaluation import p08_source_artifacts as _artifacts
from atlas_sers.evaluation import p08_source_predictions as _predictions
from atlas_sers.evaluation import p08_training_record as _training
from atlas_sers.evaluation.p08_plan import NEURAL_RECIPES
from atlas_sers.governance.canonical import sha256_value

SCHEMA_VERSION = "nato-sers-p08-neural-source-bundle-v1"

_MAXIMUM_ARCHIVE_BYTES = 2 * 1024 * 1024
_MAX_HEADER_SIZE = 10000
_CLASS_COUNT = 3

_LOGITS_MEMBER = "logits.npy"
_CLASSES_MEMBER = "classes.npy"
_UIDS_MEMBER = "uids.npy"
_MEMBER_NAMES = frozenset({_LOGITS_MEMBER, _CLASSES_MEMBER, _UIDS_MEMBER})

_UNICODE_ITEMSIZE_MINIMUM = 4
_UNICODE_ITEMSIZE_MAXIMUM = 1024

_REASON_CODES = frozenset(
    {
        "scientific_execution_not_authorized",
        "invalid_expected_inputs",
        "invalid_job",
        "job_identity_mismatch",
        "job_pair_mismatch",
        "invalid_identifiers",
        "invalid_input",
        "invalid_record",
        "record_not_complete",
        "identity_mismatch",
        "invalid_metrics",
        "invalid_counters",
        "invalid_history",
        "invalid_digests",
        "invalid_recipe_flags",
        "history_sequence_mismatch",
        "history_stopping_mismatch",
        "checkpoint_link_mismatch",
        "training_record_pin_mismatch",
        "invalid_checkpoint_bytes",
        "checkpoint_too_large",
        "invalid_expected_file_sha256",
        "invalid_expected_state_sha256",
        "invalid_class_count",
        "invalid_recipe_id",
        "checkpoint_file_hash_mismatch",
        "checkpoint_deserialization_failed",
        "checkpoint_verification_failed",
        "invalid_checkpoint_wrapper",
        "invalid_checkpoint_state",
        "checkpoint_state_key_mismatch",
        "checkpoint_state_shape_mismatch",
        "checkpoint_state_dtype_mismatch",
        "checkpoint_state_layout_invalid",
        "checkpoint_state_not_finite",
        "checkpoint_state_hash_mismatch",
        "checkpoint_parameter_count_mismatch",
        "validation_uid_mismatch",
        "class_order_mismatch",
        "invalid_scores",
        "nonfinite_scores",
        "verification_failed",
        "invalid_source_prediction_bytes",
        "source_prediction_too_large",
        "source_prediction_file_hash_mismatch",
        "invalid_source_prediction_archive",
        "source_prediction_member_mismatch",
        "source_prediction_encrypted",
        "source_prediction_compression_unsupported",
        "invalid_source_prediction_array",
        "source_prediction_array_shape_mismatch",
        "source_prediction_array_dtype_mismatch",
        "source_prediction_array_length_mismatch",
        "source_prediction_array_invalid",
        "unlisted_reason_code",
    }
)

__all__ = [
    "SCHEMA_VERSION",
    "BundleError",
    "require_scientific_execution",
    "verify_neural_source_bundle",
]


class BundleError(ValueError):
    """ValueError carrying one static allowlisted reason code."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "unlisted_reason_code"
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(reason_code):
    raise BundleError(reason_code) from None


def _translate(error, fallback):
    """Re-raise a delegated static reason, otherwise a static fallback."""

    code = getattr(error, "reason_code", None)
    if type(code) is str and code in _REASON_CODES:
        _fail(code)
    _fail(fallback)


def _is_job_id(value):
    return (
        type(value) is str
        and len(value) == 71
        and value[:7] == "P08JOB-"
        and _predictions._is_lower_hex64(value[7:])
    )


def _extract_members(blob):
    """Return the three bounded raw NPY members of one inherited archive.

    The archive bounds and member cardinality are checked before any member is
    copied or decompressed; each declared uncompressed size is checked before
    that member is read.
    """

    if type(blob) is not bytes or len(blob) == 0:
        _fail("invalid_source_prediction_bytes")
    if len(blob) > _MAXIMUM_ARCHIVE_BYTES:
        _fail("source_prediction_too_large")
    try:
        archive = zipfile.ZipFile(io.BytesIO(blob), "r")
    except BundleError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_source_prediction_archive")

    with archive:
        try:
            infos = archive.infolist()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("invalid_source_prediction_archive")
        if len(infos) != len(_MEMBER_NAMES):
            _fail("source_prediction_member_mismatch")

        names = []
        total = 0
        for info in infos:
            if info.is_dir():
                _fail("source_prediction_member_mismatch")
            if info.flag_bits & 0x1:
                _fail("source_prediction_encrypted")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                _fail("source_prediction_compression_unsupported")
            if type(info.filename) is not str or info.filename not in _MEMBER_NAMES:
                _fail("source_prediction_member_mismatch")
            if info.orig_filename != info.filename:
                _fail("source_prediction_member_mismatch")
            declared = info.file_size
            if type(declared) is not int or declared < 0 or declared > _MAXIMUM_ARCHIVE_BYTES:
                _fail("source_prediction_too_large")
            total += declared
            names.append(info.filename)
        if len(set(names)) != len(_MEMBER_NAMES):
            _fail("source_prediction_member_mismatch")
        if total > _MAXIMUM_ARCHIVE_BYTES:
            _fail("source_prediction_too_large")

        members = {}
        for info in infos:
            declared = info.file_size
            try:
                raw = archive.read(info)
            except BundleError:
                raise
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                _fail("invalid_source_prediction_archive")
            if type(raw) is not bytes or len(raw) != declared:
                _fail("invalid_source_prediction_archive")
            members[info.filename] = raw
    return members


def _read_plain_array(raw, *, expected_shape, unicode_member):
    """Parse one bounded plain NPY array without honouring object pickles."""

    if type(raw) is not bytes or len(raw) == 0:
        _fail("invalid_source_prediction_array")
    stream = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream, max_header_size=_MAX_HEADER_SIZE
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream, max_header_size=_MAX_HEADER_SIZE
            )
        else:
            _fail("invalid_source_prediction_array")
    except BundleError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_source_prediction_array")

    if type(shape) is not tuple or tuple(shape) != tuple(expected_shape):
        _fail("source_prediction_array_shape_mismatch")
    if not isinstance(dtype, np.dtype):
        _fail("source_prediction_array_dtype_mismatch")
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        _fail("source_prediction_array_dtype_mismatch")
    if unicode_member:
        if dtype.kind != "U" or not (
            _UNICODE_ITEMSIZE_MINIMUM <= int(dtype.itemsize) <= _UNICODE_ITEMSIZE_MAXIMUM
        ):
            _fail("source_prediction_array_dtype_mismatch")
    else:
        if dtype.kind != "f" or int(dtype.itemsize) != 8 or not dtype.isnative:
            _fail("source_prediction_array_dtype_mismatch")
    if fortran_order is not True and fortran_order is not False:
        _fail("source_prediction_array_invalid")

    count = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            _fail("source_prediction_array_shape_mismatch")
        count *= dimension

    header_end = stream.tell()
    if header_end + count * int(dtype.itemsize) != len(raw):
        _fail("source_prediction_array_length_mismatch")

    stream.seek(0)
    try:
        array = np.lib.format.read_array(
            stream, allow_pickle=False, max_header_size=_MAX_HEADER_SIZE
        )
    except BundleError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("source_prediction_array_invalid")
    if type(array) is not np.ndarray:
        _fail("source_prediction_array_invalid")
    if tuple(array.shape) != tuple(expected_shape) or array.dtype != dtype:
        _fail("source_prediction_array_invalid")
    return array


def _verify_neural_source_bundle(
    *,
    fit_job,
    prediction_job,
    expected_fit_job_id,
    expected_prediction_job_id,
    expected_role_id,
    expected_validation_uids,
    expected_classes,
    training_record,
    expected_training_record_sha256,
    best_checkpoint_bytes,
    expected_best_checkpoint_file_sha256,
    terminal_checkpoint_bytes,
    expected_terminal_checkpoint_file_sha256,
    source_prediction_bytes,
    expected_source_prediction_file_sha256,
):
    if type(source_prediction_bytes) is not bytes or len(source_prediction_bytes) == 0:
        _fail("invalid_source_prediction_bytes")
    if len(source_prediction_bytes) > _MAXIMUM_ARCHIVE_BYTES:
        _fail("source_prediction_too_large")

    if not _training._is_identifier(expected_role_id):
        _fail("invalid_expected_inputs")
    for pin in (
        expected_training_record_sha256,
        expected_best_checkpoint_file_sha256,
        expected_terminal_checkpoint_file_sha256,
        expected_source_prediction_file_sha256,
    ):
        if not _predictions._is_lower_hex64(pin):
            _fail("invalid_expected_inputs")
    if not _is_job_id(expected_fit_job_id) or not _is_job_id(expected_prediction_job_id):
        _fail("invalid_expected_inputs")

    try:
        expected_uids = _predictions._validate_validation_uids(expected_validation_uids)
        expected_label_classes = _predictions._validate_classes(expected_classes)
    except _predictions.PredictionError as error:
        _translate(error, "invalid_identifiers")
    if len(expected_label_classes) != _CLASS_COUNT:
        _fail("invalid_expected_inputs")

    try:
        fit = _predictions._validate_job(fit_job)
        prediction = _predictions._validate_job(prediction_job)
        _predictions._check_job_hash(fit)
        _predictions._check_job_hash(prediction)
        _predictions._validate_job_semantics(fit)
        _predictions._validate_job_semantics(prediction)
        _predictions._validate_pair(fit, prediction)
    except _predictions.PredictionError as error:
        _translate(error, "invalid_job")

    if fit["job_id"] != expected_fit_job_id:
        _fail("job_identity_mismatch")
    if prediction["job_id"] != expected_prediction_job_id:
        _fail("job_identity_mismatch")

    model_id = fit["model_id"]
    if model_id not in NEURAL_RECIPES:
        _fail("invalid_job")

    # Caller ownership: this captures an internal copy and never mutates the
    # supplied record, jobs or sequences; copies protect later mutation only.
    try:
        record_snapshot = _training._snapshot_record(training_record)
    except _training.TrainingRecordError as error:
        _translate(error, "invalid_record")

    try:
        record_report = _training.verify_neural_training_record(
            record_snapshot,
            expected_role_id=expected_role_id,
            expected_recipe_id=model_id,
            expected_seed=fit["seed"],
            class_count=_CLASS_COUNT,
            expected_best_state_sha256=record_snapshot["best_state_digest"],
            expected_terminal_state_sha256=record_snapshot["terminal_state_digest"],
        )
    except _training.TrainingRecordError as error:
        _translate(error, "invalid_record")

    if record_report["record_sha256"] != expected_training_record_sha256:
        _fail("training_record_pin_mismatch")

    try:
        best_report = _artifacts.verify_neural_checkpoint_bytes(
            best_checkpoint_bytes,
            expected_file_sha256=expected_best_checkpoint_file_sha256,
            expected_state_sha256=record_snapshot["best_state_digest"],
            class_count=_CLASS_COUNT,
            recipe_id=model_id,
        )
    except _artifacts.SourceArtifactError as error:
        _translate(error, "checkpoint_verification_failed")

    try:
        terminal_report = _artifacts.verify_neural_checkpoint_bytes(
            terminal_checkpoint_bytes,
            expected_file_sha256=expected_terminal_checkpoint_file_sha256,
            expected_state_sha256=record_snapshot["terminal_state_digest"],
            class_count=_CLASS_COUNT,
            recipe_id=model_id,
        )
    except _artifacts.SourceArtifactError as error:
        _translate(error, "checkpoint_verification_failed")

    observed_source_file_sha256 = hashlib.sha256(source_prediction_bytes).hexdigest()
    if observed_source_file_sha256 != expected_source_prediction_file_sha256:
        _fail("source_prediction_file_hash_mismatch")

    members = _extract_members(source_prediction_bytes)

    row_count = len(expected_uids)
    logits = _read_plain_array(
        members[_LOGITS_MEMBER],
        expected_shape=(row_count, _CLASS_COUNT),
        unicode_member=False,
    )
    classes_array = _read_plain_array(
        members[_CLASSES_MEMBER],
        expected_shape=(_CLASS_COUNT,),
        unicode_member=True,
    )
    uids_array = _read_plain_array(
        members[_UIDS_MEMBER],
        expected_shape=(row_count,),
        unicode_member=True,
    )
    observed_classes = classes_array.tolist()
    observed_uids = uids_array.tolist()

    try:
        source_report = _predictions.verify_source_prediction_values(
            logits,
            observed_uids=observed_uids,
            observed_classes=observed_classes,
            expected_validation_uids=expected_uids,
            expected_classes=expected_label_classes,
            fit_job=fit,
            prediction_job=prediction,
            expected_fit_job_id=expected_fit_job_id,
            expected_prediction_job_id=expected_prediction_job_id,
        )
    except _predictions.PredictionError as error:
        _translate(error, "verification_failed")

    report = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "bundle_consistency_verified": True,
        "supplied_file_hashes_verified": True,
        "training_record_pin_verified": True,
        "external_registry_membership_verified": False,
        "physical_role_isolation_verified": False,
        "training_completion_verified": False,
        "prediction_parity_verified": False,
        "live_resources_verified": False,
        "class_count": _CLASS_COUNT,
        "row_count": row_count,
        "epochs_completed": record_report["epochs_completed"],
        "optimizer_steps": record_report["optimizer_steps"],
        "best_epoch": record_report["best_epoch"],
        "parameter_count": record_report["parameter_count"],
        "collapse": record_report["collapse"],
        "fit_job_sha256": fit["job_id"][7:],
        "prediction_job_sha256": prediction["job_id"][7:],
        "training_record_sha256": record_report["record_sha256"],
        "best_checkpoint_file_sha256": best_report["checkpoint_file_sha256"],
        "terminal_checkpoint_file_sha256": terminal_report["checkpoint_file_sha256"],
        "source_prediction_file_sha256": observed_source_file_sha256,
        "best_checkpoint_report_sha256": best_report["report_sha256"],
        "terminal_checkpoint_report_sha256": terminal_report["report_sha256"],
        "training_record_report_sha256": record_report["report_sha256"],
        "source_prediction_report_sha256": source_report["report_sha256"],
    }
    report["report_sha256"] = sha256_value(report)
    return report


def verify_neural_source_bundle(
    *,
    fit_job,
    prediction_job,
    expected_fit_job_id,
    expected_prediction_job_id,
    expected_role_id,
    expected_validation_uids,
    expected_classes,
    training_record,
    expected_training_record_sha256,
    best_checkpoint_bytes,
    expected_best_checkpoint_file_sha256,
    terminal_checkpoint_bytes,
    expected_terminal_checkpoint_file_sha256,
    source_prediction_bytes,
    expected_source_prediction_file_sha256,
):
    """Compose the accepted checkpoint, record and prediction checks.

    Ordinary failures collapse to a static allowlisted
    :class:`BundleError`; ``KeyboardInterrupt`` and ``SystemExit`` propagate
    unchanged as the same object.
    """

    try:
        return _verify_neural_source_bundle(
            fit_job=fit_job,
            prediction_job=prediction_job,
            expected_fit_job_id=expected_fit_job_id,
            expected_prediction_job_id=expected_prediction_job_id,
            expected_role_id=expected_role_id,
            expected_validation_uids=expected_validation_uids,
            expected_classes=expected_classes,
            training_record=training_record,
            expected_training_record_sha256=expected_training_record_sha256,
            best_checkpoint_bytes=best_checkpoint_bytes,
            expected_best_checkpoint_file_sha256=expected_best_checkpoint_file_sha256,
            terminal_checkpoint_bytes=terminal_checkpoint_bytes,
            expected_terminal_checkpoint_file_sha256=expected_terminal_checkpoint_file_sha256,
            source_prediction_bytes=source_prediction_bytes,
            expected_source_prediction_file_sha256=expected_source_prediction_file_sha256,
        )
    except BundleError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("verification_failed")


def require_scientific_execution(*args, **kwargs):
    """Always deny execution, regardless of forged flags or arguments."""

    _fail("scientific_execution_not_authorized")
