"""Inherited U0 source-stage numerical backend (P08-T109, corrected P08-T111).

This module is the thin numerical bridge between one already-authenticated
:class:`atlas_sers.evaluation.p08_u0_runtime_inputs.RuntimePair` and the
inherited P03/P05 fitting kernels:

* :func:`invoke_source_fit` dispatches exactly once to the frozen classical or
  neural kernel and returns the actual inherited result object,
* :func:`prepare_fit_artifacts` serializes that result in memory using only the
  inherited checkpoint / logits / summary / prediction formats and runs the
  reusable structural checks before a complete fit may be labelled succeeded,
* :func:`verify_source_prediction` re-authenticates the saved bytes, replays the
  same structural checks, and only then performs the inherited numerical parity
  and metric comparisons.

It is a numerical backend, not a controller.  It is not a permit, file writer,
journal, budget, resource monitor, stage dispatcher or recovery mechanism, and
it never authorizes scientific execution.

Bounded meaning
---------------
* One ``source_fit`` command includes the inherited within-fit
  source-validation work; the dependent ``source_validation_prediction``
  command verifies from the same saved bytes without a second optimization.
* Numerical parity consumes exactly the caller-supplied saved bytes after they
  are authenticated against the prepared pins; it does not establish that those
  bytes are the ones a controller later persists.
* No loaded source code, live permit, running attempt, filesystem object,
  external registry, live resource state or controller completion is
  authenticated or represented here.
* ``prediction_parity_verified`` records only local agreement between a
  caller-supplied pair, the caller-supplied artifact snapshots and the
  inherited in-memory checks.  It is not provenance, physical-isolation,
  completion or resource evidence.
* ``prepare_fit_artifacts`` never runs a model fit, model inference,
  ``_aligned_scores`` or ``_predict_logits``.  It runs only the reusable
  structural checks.  Numerical parity and metric replay happen exclusively in
  :func:`verify_source_prediction`.
* The caller owns every supplied pair and result.  ``invoke_source_fit`` never
  swallows or converts genuine inherited errors into success.  A caller retains
  its result reference if artifact preparation fails.
* A finite, structurally consistent artifact can still disagree with the
  restored model; the numerical replay is a separate check from structure.
* Classical deadline / device / epoch-callback enforcement is intentionally not
  added to the inherited ``run_candidate_fit`` signature.  Outer enforcement is
  the future controller's responsibility.
* Only scalars, counts and hashes leave the verification interface.  No scores,
  labels, paths, sample identifiers, raw metrics or blobs are returned.
* ``KeyboardInterrupt`` and ``SystemExit`` propagate unchanged.

The eventual controller must authenticate its loaded implementation, durably
record the attempt and hold its leases before invoking this backend, and must
persist, write and authenticate the artifact bytes before recording success.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p05_pilot as _pilot
from atlas_sers.evaluation import p08_neural_bundle as _bundle
from atlas_sers.evaluation import p08_plan as _plan
from atlas_sers.evaluation import p08_source_predictions as _predictions
from atlas_sers.evaluation import p08_training_record as _training
from atlas_sers.evaluation.p08_u0_runtime_inputs import RuntimePair
from atlas_sers.governance.canonical import canonical_json_bytes, sha256_value

SCHEMA_VERSION = "nato-sers-p08-u0-stage-backend-v1"

_SUMMARY = "summary.json"
_PREDICTIONS = "predictions.csv"
_BEST = "best.pt"
_TERMINAL = "terminal.pt"
_LOGITS = "validation_logits.npz"

_NEURAL_ARTIFACT_NAMES = frozenset({_SUMMARY, _BEST, _TERMINAL, _LOGITS})
_CLASSICAL_ARTIFACT_NAMES = frozenset({_SUMMARY, _PREDICTIONS})

_PREDICTION_COLUMN_ORDER = (
    "observation_uid",
    "master_sample_id",
    "instrument",
    "station",
    "true_label",
    "fit_id",
    "predicted_label",
    "class_vocabulary",
    "scores",
    "probabilities",
    "probability_status",
)

_UNPROBABILISTIC_STATUS = "uncalibrated"

_REASON_CODES = frozenset(
    {
        "scientific_execution_not_authorized",
        "invalid_arguments",
        "invalid_pair",
        "wrong_kernel",
        "invalid_result",
        "invalid_device",
        "cuda_not_initialized",
        "artifact_preparation_failed",
        "artifact_set_mismatch",
        "artifact_hash_mismatch",
        "invalid_artifact_bytes",
        "fit_not_succeeded",
        "fit_pair_mismatch",
        "invalid_summary",
        "invalid_checkpoint",
        "invalid_predictions",
        "estimator_missing",
        "validation_uid_mismatch",
        "class_order_mismatch",
        "invalid_scores",
        "nonfinite_scores",
        "job_pair_mismatch",
        "job_identity_mismatch",
        "prediction_mismatch",
        "score_parity_mismatch",
        "metric_parity_mismatch",
        "logits_parity_mismatch",
        "invalid_source_prediction_bytes",
        "verification_failed",
        "unlisted_reason_code",
    }
)

__all__ = [
    "FitArtifacts",
    "SCHEMA_VERSION",
    "StageError",
    "invoke_source_fit",
    "prepare_fit_artifacts",
    "require_scientific_execution",
    "verify_source_prediction",
]


class StageError(ValueError):
    """ValueError carrying one static allowlisted reason code."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "unlisted_reason_code"
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(reason_code):
    raise StageError(reason_code) from None


def _torch():
    return importlib.import_module("torch")


def _development():
    return importlib.import_module("atlas_sers.evaluation.p05_development")


def _p03_runtime():
    return importlib.import_module("atlas_sers.evaluation.p03_runtime")


def _p04_runtime():
    return importlib.import_module("atlas_sers.evaluation.p04_runtime")


def _acquisition():
    return importlib.import_module("atlas_sers.models.acquisition")


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _load_json_object(raw, reason_code):
    if type(raw) is not bytes or len(raw) == 0:
        _fail(reason_code)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail(reason_code)
    if type(value) is not dict:
        _fail(reason_code)
    return value


def _require_pair(pair):
    if type(pair) is not RuntimePair:
        _fail("invalid_arguments")


def _require_artifacts(artifacts):
    if type(artifacts) is not FitArtifacts:
        _fail("invalid_arguments")


def _configuration(pair):
    try:
        configuration = pair.configuration()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_pair")
    if type(configuration) is not dict:
        _fail("invalid_pair")
    return configuration


def _kind_of(pair):
    model_id = _configuration(pair).get("model_id")
    if type(model_id) is not str:
        _fail("invalid_pair")
    if model_id in _plan.CLASSICAL_MODELS:
        return "classical", model_id
    if model_id in _plan.NEURAL_RECIPES:
        return "neural", model_id
    _fail("wrong_kernel")


def _job(pair, attribute):
    raw = getattr(pair.prepared_pair, attribute, None)
    if type(raw) is not str or not raw:
        _fail("invalid_pair")
    try:
        job = json.loads(raw)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_pair")
    if type(job) is not dict:
        _fail("invalid_pair")
    return job


def _job_id(job):
    value = job.get("job_id")
    if type(value) is not str or not value:
        _fail("invalid_pair")
    return value


def _job_digest(job_id):
    if type(job_id) is str and len(job_id) > 7 and job_id[:7] == "P08JOB-":
        return job_id[7:]
    return sha256_value(job_id)


def _source_roles(pair):
    roles = getattr(pair.prepared_pair.inputs, "source_roles", None)
    if roles is None:
        _fail("invalid_pair")
    return roles


def _expected_classes(pair):
    """Return the exact prepared source class vocabulary, unchanged."""

    classes = tuple(str(value) for value in _source_roles(pair).classes)
    if not (2 <= len(classes) <= 3) or len(set(classes)) != len(classes):
        _fail("invalid_pair")
    return classes


def _station_of(pair):
    observations = list(pair.fitting_observations) + list(pair.validation_observations)
    if not observations:
        _fail("invalid_pair")
    station = getattr(observations[0], "station", None)
    if type(station) is not str or not station:
        _fail("invalid_pair")
    return station


def _require_result_class(kind, result):
    """Require the exact inherited result class appropriate to the kernel."""

    expected = (
        getattr(_development(), "DevelopmentFitResult", None)
        if kind == "neural"
        else getattr(_p03_runtime(), "CandidateFitOutcome", None)
    )
    if expected is None or type(result) is not expected:
        _fail("invalid_result")


@dataclass(frozen=True, repr=False)
class FitArtifacts:
    """Private in-memory fit artifacts plus their stage identities.

    The live inherited ``result`` is retained because the classical dependent
    verification needs the fitted estimator without a refit.  This object is a
    private hand-off, not a frozen model or an authority capability.
    """

    result: object
    kind: str
    status: str
    fit_job_id: str
    prediction_job_id: str
    artifact_items: tuple

    def artifact_bytes(self):
        """Return a fresh name -> immutable-bytes mapping."""

        return {name: value for name, value in self.artifact_items}


def invoke_source_fit(pair, *, device, global_deadline, on_epoch):
    """Dispatch one authenticated pair to its inherited fitting kernel once.

    Classical kernels are not extended with ``device``, ``global_deadline`` or
    ``on_epoch``; their outer enforcement belongs to the future controller.
    This helper is not a permit or a launch entry point and never retries.
    """

    _require_pair(pair)
    kind, _model_id = _kind_of(pair)
    if kind == "classical":
        return _p03_runtime().run_candidate_fit(**pair.classical_kwargs())
    # The prepared source buffers backing ``values`` and ``validation_values``
    # are immutable/read-only float32 arrays.  Take the fresh kwargs mapping,
    # then hand the inherited kernel writable C-order float32 copies that hold
    # bit-identical values.  This is data ownership only: no preprocessing,
    # augmentation, reordering or value change occurs, and the once-only
    # kernel dispatch is unchanged.
    kwargs = dict(pair.neural_kwargs())
    for name in ("values", "validation_values"):
        if name in kwargs:
            kwargs[name] = np.array(kwargs[name], dtype=np.float32, copy=True, order="C")
    return _development().train_development_fit(
        **kwargs,
        device=device,
        global_deadline=global_deadline,
        on_epoch=on_epoch,
    )


def _torch_state_bytes(state):
    buffer = io.BytesIO()
    _torch().save({"state_dict": state}, buffer)
    return buffer.getvalue()


def _logits_bytes(result):
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        logits=np.asarray(result.validation_logits, dtype=np.float64),
        classes=np.asarray(list(result.classes), dtype=np.str_),
        uids=np.asarray(list(result.validation_uids), dtype=np.str_),
    )
    return buffer.getvalue()


def _neural_artifacts(pair, result, fit_job):
    roles = _source_roles(pair)
    fit_job_id = _job_id(fit_job)
    model_id = fit_job.get("model_id")
    # The LOGICAL execution_id uses the scheduled fit job seed so a failed
    # diagnostic result whose inherited ``seed`` is missing or ``None`` still
    # serializes.  The actual inherited identity is never substituted or
    # manufactured here; it is preserved verbatim in the summary below, and
    # the structural bundle check still rejects a complete wrong-seed result.
    scheduled_seed = fit_job.get("seed")
    slot = {"slot_id": fit_job_id, "recipe_id": model_id, "seed": scheduled_seed}
    unit = {"station": _station_of(pair)}
    identifier = _pilot.execution_id(unit, slot)

    summary = _pilot._result_private_summary(result)
    # Preserve the actual inherited identity.  Never substitute the expected
    # job seed or recipe for a misbound result: the structural bundle check
    # below compares this preserved identity against the frozen job/roles.
    for name in ("seed", "recipe", "role_id"):
        value = getattr(result, name, None)
        if value is not None:
            summary[name] = value
    summary["execution_id"] = identifier
    summary["slot_id"] = fit_job_id
    summary["recipe_id"] = model_id
    unit_id = getattr(roles, "unit_id", None)
    if type(unit_id) is str:
        summary["unit_id"] = unit_id

    items = [(_SUMMARY, canonical_json_bytes(summary))]
    if getattr(result, "best_state_dict", None) is not None:
        items.append((_BEST, _torch_state_bytes(result.best_state_dict)))
    if getattr(result, "terminal_state_dict", None) is not None:
        items.append((_TERMINAL, _torch_state_bytes(result.terminal_state_dict)))
    if getattr(result, "validation_logits", None) is not None:
        items.append((_LOGITS, _logits_bytes(result)))
    return items


def _classical_artifacts(result):
    summary = result.status_record()
    summary["validation_metrics"] = result.validation_metrics
    estimator = getattr(result, "estimator", None)
    audit = None if estimator is None else getattr(estimator, "fit_audit", None)
    if audit is not None:
        summary["fit_audit"] = dataclasses.asdict(audit)

    items = [(_SUMMARY, canonical_json_bytes(summary))]
    frame = getattr(result, "validation_predictions", None)
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        items.append((_PREDICTIONS, frame.to_csv(index=False).encode("utf-8")))
    return items


def _ordered_validation(pair):
    """Return caller validation rows in their exact prepared order.

    The caller-supplied observation order and metadata must agree exactly with
    the prepared ``source_roles.validation`` sequence; no independent sort and
    no field aliasing is permitted.
    """

    roles = _source_roles(pair)
    source = list(getattr(roles, "validation", ()))
    observed = list(pair.validation_observations)
    if not observed or len(observed) != len(source):
        _fail("validation_uid_mismatch")
    for row, raw in zip(observed, source, strict=True):
        if (
            str(row.uid) != str(getattr(raw, "observation_uid", None))
            or str(row.master) != str(getattr(raw, "master_sample_id", None))
            or str(row.station) != str(getattr(raw, "station", None))
            or str(row.target) != str(getattr(raw, "target_analyte", None))
            or str(row.instrument) != str(getattr(raw, "instrument", None))
        ):
            _fail("validation_uid_mismatch")

    uids = [str(row.uid) for row in observed]
    classes = _expected_classes(pair)
    lookup = {label: index for index, label in enumerate(classes)}
    if any(str(row.target) not in lookup for row in observed):
        _fail("class_order_mismatch")
    labels = np.asarray([lookup[str(row.target)] for row in observed], dtype=np.int64)
    return observed, uids, classes, labels


def _project_record(summary):
    if type(summary) is not dict:
        _fail("invalid_summary")
    record = {}
    for name in _training.RECORD_FIELDS:
        if name == "history":
            history = summary.get("history")
            if type(history) is not list:
                _fail("invalid_summary")
            entries = []
            for entry in history:
                if type(entry) is not dict:
                    _fail("invalid_summary")
                projected = {}
                for field in _training.HISTORY_FIELDS:
                    if field not in entry:
                        _fail("invalid_summary")
                    projected[field] = entry[field]
                entries.append(projected)
            record["history"] = entries
        else:
            if name not in summary:
                _fail("invalid_summary")
            record[name] = summary[name]
    return record


def _load_npz_logits(raw):
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != {"logits", "classes", "uids"}:
                _fail("invalid_source_prediction_bytes")
            logits = np.asarray(archive["logits"], dtype=np.float64)
    except StageError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_source_prediction_bytes")
    if logits.ndim != 2 or not bool(np.isfinite(logits).all()):
        _fail("invalid_source_prediction_bytes")
    return logits


def _load_torch_state(torch, raw):
    try:
        loaded = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_checkpoint")
    if not isinstance(loaded, Mapping):
        _fail("invalid_checkpoint")
    state = loaded.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        _fail("invalid_checkpoint")
    return state


def _numbers_equal(observed, expected):
    if type(observed) is bool or type(expected) is bool:
        return observed == expected
    if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
        return float(observed) == float(expected)
    return observed == expected


def _metrics_equal(observed, expected):
    if type(observed) is not dict or type(expected) is not dict:
        return False
    if set(observed) != set(expected):
        return False
    return all(_numbers_equal(observed[key], expected[key]) for key in observed)


def _torch_device(torch, device):
    try:
        torch_device = torch.device(device)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_device")
    if torch_device.type not in ("cpu", "cuda"):
        _fail("invalid_device")
    return torch_device


def _translate_bundle(error):
    reason = getattr(error, "reason_code", None)
    if type(reason) is str and reason in _REASON_CODES:
        _fail(reason)
    _fail("prediction_mismatch")


def _translate_prediction(error):
    reason = getattr(error, "reason_code", None)
    if type(reason) is str and reason in _REASON_CODES:
        _fail(reason)
    _fail("prediction_mismatch")


def _structural_neural(pair, fit_job, prediction_job, artifacts):
    """Reusable structural check for one neural artifact set.

    Runs the accepted inherited source-bundle verifier against the supplied
    bytes.  It performs no fitting and no model inference.
    """

    for required in sorted(_NEURAL_ARTIFACT_NAMES):
        if required not in artifacts:
            _fail("artifact_set_mismatch")

    summary = _load_json_object(artifacts[_SUMMARY], "invalid_summary")
    record = _project_record(summary)
    _observed, uids, classes, labels = _ordered_validation(pair)
    role_id = getattr(_source_roles(pair), "fitting_role_id", None)
    if type(role_id) is not str or not role_id:
        _fail("invalid_pair")

    fit_id = _job_id(fit_job)
    prediction_id = _job_id(prediction_job)
    try:
        bundle_report = _bundle.verify_neural_source_bundle(
            fit_job=fit_job,
            prediction_job=prediction_job,
            expected_fit_job_id=fit_id,
            expected_prediction_job_id=prediction_id,
            expected_role_id=role_id,
            expected_validation_uids=list(uids),
            expected_classes=list(classes),
            training_record=record,
            expected_training_record_sha256=sha256_value(record),
            best_checkpoint_bytes=artifacts[_BEST],
            expected_best_checkpoint_file_sha256=_sha256(artifacts[_BEST]),
            terminal_checkpoint_bytes=artifacts[_TERMINAL],
            expected_terminal_checkpoint_file_sha256=_sha256(artifacts[_TERMINAL]),
            source_prediction_bytes=artifacts[_LOGITS],
            expected_source_prediction_file_sha256=_sha256(artifacts[_LOGITS]),
        )
    except _bundle.BundleError as error:
        _translate_bundle(error)

    return {
        "summary": summary,
        "report": bundle_report,
        "uids": uids,
        "classes": classes,
        "labels": labels,
    }


def _read_predictions_csv(raw):
    if type(raw) is not bytes or len(raw) == 0:
        _fail("invalid_predictions")
    try:
        frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_predictions")
    if frame.empty or tuple(frame.columns) != _PREDICTION_COLUMN_ORDER:
        _fail("invalid_predictions")
    return frame


def _decode_scores(frame):
    rows = []
    for value in frame["scores"]:
        try:
            parsed = json.loads(value)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("invalid_scores")
        if type(parsed) is not list:
            _fail("invalid_scores")
        rows.append(parsed)
    try:
        scores = np.asarray(rows, dtype=np.float64)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_scores")
    if scores.ndim != 2:
        _fail("invalid_scores")
    return scores


def _structural_classical(pair, fit_job, prediction_job, outcome, artifacts):
    """Reusable structural check for one classical artifact set.

    Validates the exact outcome identity, the full fit audit against the
    independently recomputed source hashes, the saved summary identity, the
    saved prediction CSV structure, and the inherited source-prediction
    structure.  It performs no model inference and no fitting.
    """

    if _SUMMARY not in artifacts or _PREDICTIONS not in artifacts:
        _fail("artifact_set_mismatch")
    if (
        getattr(outcome, "status", None) != "complete"
        or getattr(outcome, "reason_code", None) is not None
    ):
        _fail("fit_not_succeeded")
    estimator = getattr(outcome, "estimator", None)
    audit = None if estimator is None else getattr(estimator, "fit_audit", None)
    if estimator is None or audit is None:
        _fail("estimator_missing")

    fit_id = _job_id(fit_job)
    prediction_id = _job_id(prediction_job)
    model_id = fit_job.get("model_id")
    candidate_id = fit_job.get("candidate_id")
    seed = fit_job.get("seed")

    if str(getattr(outcome, "fit_id", None)) != fit_id:
        _fail("fit_pair_mismatch")
    if str(getattr(outcome, "model_id", None)) != str(model_id):
        _fail("fit_pair_mismatch")
    if str(getattr(outcome, "candidate_id", None)) != str(candidate_id):
        _fail("fit_pair_mismatch")
    if getattr(outcome, "seed", None) != seed:
        _fail("fit_pair_mismatch")

    fit_rows = list(pair.fitting_observations)
    # Validate and use the prepared validation order exactly as the neural
    # path does; never compare the CSV to a potentially reordered observation
    # list without confirming the prepared source-role order first.
    validation_rows, validation_uids, class_vocabulary, _validation_labels = _ordered_validation(
        pair
    )
    if not fit_rows or not validation_rows:
        _fail("invalid_pair")

    fit_uids = [str(row.uid) for row in fit_rows]
    fit_masters = [str(row.master) for row in fit_rows]
    fit_instruments = [str(row.instrument) for row in fit_rows]

    runtime = _p03_runtime()
    fit_uid_sha256 = runtime._uid_hash(fit_uids)
    validation_uid_sha256 = runtime._uid_hash(validation_uids)
    # ``fit_master_sha256`` retains repeated master rows (inherited
    # p03_runtime._uid_hash), whereas the fit audit records the unique-master
    # set.  Both are validated independently and must not be confused.
    fit_master_repeat_sha256 = runtime._uid_hash(fit_masters)
    master_uid_sha256 = sha256_value(sorted(set(fit_masters)))
    domain_uid_sha256 = sha256_value(sorted(set(fit_instruments)))

    if getattr(outcome, "fit_uid_sha256", None) != fit_uid_sha256:
        _fail("fit_pair_mismatch")
    if getattr(outcome, "validation_uid_sha256", None) != validation_uid_sha256:
        _fail("fit_pair_mismatch")
    if getattr(outcome, "fit_master_sha256", None) != fit_master_repeat_sha256:
        _fail("fit_pair_mismatch")
    if fit_job.get("fit_uid_sha256") != fit_uid_sha256:
        _fail("fit_pair_mismatch")
    if fit_job.get("validation_uid_sha256") != validation_uid_sha256:
        _fail("fit_pair_mismatch")

    if audit.observation_uid_sha256 != fit_uid_sha256:
        _fail("fit_pair_mismatch")
    if audit.master_uid_sha256 != master_uid_sha256:
        _fail("fit_pair_mismatch")
    if audit.domain_uid_sha256 != domain_uid_sha256:
        _fail("fit_pair_mismatch")
    if int(audit.observations) != len(fit_rows):
        _fail("fit_pair_mismatch")
    if int(audit.masters) != len(set(fit_masters)):
        _fail("fit_pair_mismatch")

    summary = _load_json_object(artifacts[_SUMMARY], "invalid_summary")
    expected_summary = {
        "fit_id": getattr(outcome, "fit_id", None),
        "model_id": getattr(outcome, "model_id", None),
        "candidate_id": getattr(outcome, "candidate_id", None),
        "seed": getattr(outcome, "seed", None),
        "fit_uid_sha256": getattr(outcome, "fit_uid_sha256", None),
        "validation_uid_sha256": getattr(outcome, "validation_uid_sha256", None),
        "fit_master_sha256": getattr(outcome, "fit_master_sha256", None),
        "status": getattr(outcome, "status", None),
        "reason_code": getattr(outcome, "reason_code", None),
    }
    for key, expected in expected_summary.items():
        if key not in summary or summary[key] != expected:
            _fail("fit_pair_mismatch")

    saved_audit = summary.get("fit_audit")
    expected_audit = dataclasses.asdict(audit)
    source_audit = {
        "observation_uid_sha256": fit_uid_sha256,
        "master_uid_sha256": master_uid_sha256,
        "domain_uid_sha256": domain_uid_sha256,
        "observations": len(fit_rows),
        "masters": len(set(fit_masters)),
    }
    if (
        type(saved_audit) is not dict
        or saved_audit != expected_audit
        or saved_audit != source_audit
    ):
        _fail("fit_pair_mismatch")

    frame = _read_predictions_csv(artifacts[_PREDICTIONS])
    expected_uids = validation_uids
    expected_masters = [str(row.master) for row in validation_rows]
    expected_instruments = [str(row.instrument) for row in validation_rows]
    expected_stations = [str(row.station) for row in validation_rows]
    expected_targets = [str(row.target) for row in validation_rows]

    if list(frame["observation_uid"]) != expected_uids:
        _fail("prediction_mismatch")
    if list(frame["master_sample_id"]) != expected_masters:
        _fail("prediction_mismatch")
    if list(frame["instrument"]) != expected_instruments:
        _fail("prediction_mismatch")
    if list(frame["station"]) != expected_stations:
        _fail("prediction_mismatch")
    if list(frame["true_label"]) != expected_targets:
        _fail("prediction_mismatch")
    if list(frame["fit_id"]) != [fit_id] * len(expected_uids):
        _fail("prediction_mismatch")
    if any(value != _UNPROBABILISTIC_STATUS for value in frame["probability_status"]):
        _fail("prediction_mismatch")
    if any(value != "" for value in frame["probabilities"]):
        _fail("prediction_mismatch")

    for value in frame["class_vocabulary"]:
        try:
            decoded = json.loads(value)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            _fail("invalid_predictions")
        if decoded != list(class_vocabulary):
            _fail("class_order_mismatch")

    scores = _decode_scores(frame)
    try:
        source_report = _predictions.verify_source_prediction_values(
            scores,
            observed_uids=list(expected_uids),
            observed_classes=list(class_vocabulary),
            expected_validation_uids=list(expected_uids),
            expected_classes=list(class_vocabulary),
            fit_job=fit_job,
            prediction_job=prediction_job,
            expected_fit_job_id=fit_id,
            expected_prediction_job_id=prediction_id,
        )
    except _predictions.PredictionError as error:
        _translate_prediction(error)

    predicted = [class_vocabulary[int(index)] for index in np.argmax(scores, axis=1)]
    if list(frame["predicted_label"]) != predicted:
        _fail("prediction_mismatch")

    return {
        "scores": scores,
        "frame": frame,
        "class_vocabulary": class_vocabulary,
        "saved_metrics": summary.get("validation_metrics"),
        "source_report": source_report,
    }


def _prepare_fit_artifacts(pair, result):
    _require_pair(pair)
    kind, _model_id = _kind_of(pair)
    _require_result_class(kind, result)
    if type(getattr(result, "status", None)) is not str:
        _fail("invalid_result")

    fit_job = _job(pair, "fit_job_json")
    prediction_job = _job(pair, "prediction_job_json")
    if kind == "neural":
        items = _neural_artifacts(pair, result, fit_job)
    else:
        items = _classical_artifacts(result)

    artifacts = dict(items)
    status = "succeeded" if result.status == "complete" else "failed"
    if status == "succeeded":
        # A future controller must accept fit semantics before recording
        # success.  Complete results therefore pass the reusable structural
        # checks here; failed results keep their diagnostics and stay failed
        # without requiring success-only artifacts.
        if kind == "neural":
            _structural_neural(pair, fit_job, prediction_job, artifacts)
        else:
            _structural_classical(pair, fit_job, prediction_job, result, artifacts)

    return FitArtifacts(
        result=result,
        kind=kind,
        status=status,
        fit_job_id=_job_id(fit_job),
        prediction_job_id=_job_id(prediction_job),
        artifact_items=tuple(items),
    )


def prepare_fit_artifacts(pair, result):
    """Serialize one inherited fit result into immutable in-memory bytes.

    Complete results are structurally accepted before being labelled
    ``succeeded``; failed results preserve the inherited diagnostics.  No fit
    and no model inference run here.  Ordinary serialization or structural
    failures collapse to a static allowlisted :class:`StageError`; the caller
    still owns the original result object.
    """

    try:
        return _prepare_fit_artifacts(pair, result)
    except StageError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("artifact_preparation_failed")


def _authenticate_artifacts(fit_artifacts, saved):
    if type(saved) is not dict:
        _fail("invalid_artifact_bytes")
    pins = fit_artifacts.artifact_bytes()
    if set(saved) != set(pins):
        _fail("artifact_set_mismatch")
    authenticated = {}
    for name in sorted(pins):
        pinned = pins[name]
        value = saved[name]
        if type(value) is not bytes:
            _fail("invalid_artifact_bytes")
        if len(value) != len(pinned) or _sha256(value) != _sha256(pinned):
            _fail("artifact_hash_mismatch")
        authenticated[name] = value
    return authenticated


def _parity_neural(pair, summary, saved, uids, classes, labels, device):
    configuration = _configuration(pair)
    recipe = configuration.get("recipe")
    if type(recipe) is not dict:
        _fail("invalid_pair")
    use_projection = bool(recipe.get("projection"))

    torch = _torch()
    torch_device = _torch_device(torch, device)
    if torch_device.type == "cuda":
        # Refuse an uninitialized CUDA runtime without initializing it.
        if not torch.cuda.is_initialized():
            _fail("cuda_not_initialized")
        rng_devices = [
            torch_device.index if torch_device.index is not None else torch.cuda.current_device()
        ]
    else:
        rng_devices = []

    state = _load_torch_state(torch, saved[_BEST])
    values = np.asarray(pair.prepared_pair.inputs.validation_values(), dtype=np.float32)
    if values.ndim != 2 or len(values) != len(uids):
        _fail("invalid_pair")
    # ``validation_values()`` may be read-only; torch.from_numpy needs a
    # writable C-order copy with identical float32 values.
    matrix = np.array(values, dtype=np.float32, copy=True, order="C")
    tensor = torch.from_numpy(np.ascontiguousarray(matrix[:, None, :]))

    acquisition = _acquisition()
    development = _development()
    # fork_rng restores the global RNG even when parity raises afterwards.
    with torch.random.fork_rng(devices=rng_devices):
        with torch.device("cpu"):
            model = acquisition.AcquisitionClassifier(
                class_count=len(classes), use_projection=use_projection
            )
        model.load_state_dict(state)
        model.to(torch_device)
        recomputed = development._predict_logits(model, tensor, torch_device)

    logits = _load_npz_logits(saved[_LOGITS])
    if recomputed.shape != logits.shape or not bool(np.array_equal(recomputed, logits)):
        _fail("logits_parity_mismatch")

    metrics = _p04_runtime()._metric_values(labels, recomputed, tuple(classes))
    for metric_key, summary_key in (
        ("balanced_accuracy", "best_validation_balanced_accuracy"),
        ("negative_log_likelihood", "best_validation_nll"),
        ("macro_f1", "best_validation_macro_f1"),
        ("predicted_class_count", "best_validation_predicted_class_count"),
    ):
        if summary_key not in summary:
            _fail("invalid_summary")
        if metric_key not in metrics or not _numbers_equal(
            metrics[metric_key], summary[summary_key]
        ):
            _fail("metric_parity_mismatch")
    return logits


def _parity_classical(pair, outcome, context):
    runtime = _p03_runtime()
    class_vocabulary = context["class_vocabulary"]
    validation_values = np.asarray(pair.prepared_pair.inputs.validation_values(), dtype=np.float64)
    recomputed = runtime._aligned_scores(outcome.estimator, validation_values, class_vocabulary)
    scores = context["scores"]
    if recomputed.shape != scores.shape or not bool(np.array_equal(recomputed, scores)):
        _fail("score_parity_mismatch")
    metrics = runtime._master_metrics(context["frame"], scores, class_vocabulary)
    saved_metrics = context["saved_metrics"]
    if type(saved_metrics) is not dict or not _metrics_equal(metrics, saved_metrics):
        _fail("metric_parity_mismatch")


def _verify_neural(pair, saved, fit_job, prediction_job, device):
    context = _structural_neural(pair, fit_job, prediction_job, saved)
    logits = _parity_neural(
        pair,
        context["summary"],
        saved,
        context["uids"],
        context["classes"],
        context["labels"],
        device,
    )

    fit_id = _job_id(fit_job)
    prediction_id = _job_id(prediction_job)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "execution_authorized": False,
        "prediction_parity_verified": True,
        "kernel": "neural",
        "row_count": int(logits.shape[0]),
        "class_count": int(logits.shape[1]),
        "fit_job_sha256": _job_digest(fit_id),
        "prediction_job_sha256": _job_digest(prediction_id),
        "summary_sha256": _sha256(saved[_SUMMARY]),
        "best_checkpoint_sha256": _sha256(saved[_BEST]),
        "terminal_checkpoint_sha256": _sha256(saved[_TERMINAL]),
        "source_prediction_sha256": _sha256(saved[_LOGITS]),
        "bundle_report_sha256": context["report"]["report_sha256"],
    }
    report["report_sha256"] = sha256_value(report)
    return report


def _verify_classical(pair, fit_artifacts, saved, fit_job, prediction_job):
    context = _structural_classical(pair, fit_job, prediction_job, fit_artifacts.result, saved)
    _parity_classical(pair, fit_artifacts.result, context)

    fit_id = _job_id(fit_job)
    prediction_id = _job_id(prediction_job)
    scores = context["scores"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "execution_authorized": False,
        "prediction_parity_verified": True,
        "kernel": "classical",
        "row_count": int(scores.shape[0]),
        "class_count": int(scores.shape[1]),
        "fit_job_sha256": _job_digest(fit_id),
        "prediction_job_sha256": _job_digest(prediction_id),
        "summary_sha256": _sha256(saved[_SUMMARY]),
        "predictions_sha256": _sha256(saved[_PREDICTIONS]),
        "source_prediction_report_sha256": context["source_report"]["report_sha256"],
    }
    report["report_sha256"] = sha256_value(report)
    return report


def _verify(pair, fit_artifacts, saved_artifact_bytes, device):
    _require_pair(pair)
    _require_artifacts(fit_artifacts)
    kind, _model_id = _kind_of(pair)
    if kind != fit_artifacts.kind:
        _fail("fit_pair_mismatch")
    if fit_artifacts.status != "succeeded":
        _fail("fit_not_succeeded")

    fit_job = _job(pair, "fit_job_json")
    prediction_job = _job(pair, "prediction_job_json")
    if _job_id(fit_job) != fit_artifacts.fit_job_id:
        _fail("fit_pair_mismatch")
    if _job_id(prediction_job) != fit_artifacts.prediction_job_id:
        _fail("fit_pair_mismatch")

    saved = _authenticate_artifacts(fit_artifacts, saved_artifact_bytes)
    if kind == "neural":
        return _verify_neural(pair, saved, fit_job, prediction_job, device)
    return _verify_classical(pair, fit_artifacts, saved, fit_job, prediction_job)


def verify_source_prediction(pair, fit_artifacts, *, saved_artifact_bytes, device):
    """Re-authenticate and replay the inherited source-prediction check.

    The authenticated saved snapshots first pass the same reusable structural
    checks used during preparation, then the inherited numerical parity and
    metric comparisons run against those snapshots without a refit.  A returned
    report is a local consistency statement, not an execution capability.
    Ordinary failures collapse to a static allowlisted :class:`StageError`.
    """

    try:
        return _verify(pair, fit_artifacts, saved_artifact_bytes, device)
    except StageError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("verification_failed")


def require_scientific_execution(*args, **kwargs):
    """Always deny execution, regardless of forged flags or arguments."""

    _fail("scientific_execution_not_authorized")
