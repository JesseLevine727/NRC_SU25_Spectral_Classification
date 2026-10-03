"""P08-T094: bind frozen U0 job pairs to authenticated physical source roles.

Metadata-only boundary.  Five raw byte buffers are authenticated before any
parse; then the declared source-fit / source-prediction job pairs are checked
against the attempt manifest and against a physical role registry.

Non-claims
----------
* No score arrays, model parameters or fitted models are loaded here.
* ``registry_bytes_verified`` and the role flags are metadata checks only, not
  execution authorization, provenance or historical proof.
* Execution through this module is never authorized.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass

from atlas_sers.evaluation import p08_attempt_journal as attempt_journal
from atlas_sers.evaluation import p08_source_predictions as sp
from atlas_sers.evaluation import p08_u0_admission as admission
from atlas_sers.evaluation.p08_plan import CLASSICAL_MODELS, _hash
from atlas_sers.governance.canonical import sha256_value

__all__ = [
    "Binding",
    "BindingError",
    "Observation",
    "SourcePair",
    "SourceRoles",
    "bind_u0_source_metadata",
    "require_scientific_execution",
]

SCHEMA_VERSION = "nato-sers-p08-u0-input-binding-v1"
_PROPOSAL_SCHEMA = "nato-sers-p08-universal-smoke-proposal-v1"

_PROPOSAL_PIN = "3b556d39686e8bad330ac48a4c68c30660ce7ff3a1a4d28488bc8c0bdc4f773f"
_ATTEMPT_MANIFEST_PIN = "edf04d94f377ad143adcc0126d27e5d5dcb185dbc1aae0e8f0017eccfb8a654c"
_MANIFEST_PIN = "db1f298a76aeb9962db004776a9f41d6c9afe5b76c39aa9277a24848108d5f90"
_CONTEXTS_PIN = "12be22701e6c9847301bff7978bd6e4a4cd2f4abade84c4a025f1ed2c24810fb"
_ROLES_PIN = "224891579d5df84c42a1c3c827590b6515ec83a13bfe4f07d48f249432f38d91"
_PARENT_PLAN_SHA256 = "179b95e8011a5f6cc02c65c7fab1acf0f6a6241ba1ef02378aac207b9e19cb03"

_MIB = 1024 * 1024
_BYTE_MINIMUM = 1
_BYTE_PINS = {
    "proposal_bytes": _PROPOSAL_PIN,
    "attempt_manifest_bytes": _ATTEMPT_MANIFEST_PIN,
    "manifest_bytes": _MANIFEST_PIN,
    "contexts_bytes": _CONTEXTS_PIN,
    "roles_bytes": _ROLES_PIN,
}
_BYTE_LIMITS = {
    "proposal_bytes": 1 * _MIB,
    "attempt_manifest_bytes": 1 * _MIB,
    "manifest_bytes": 2 * _MIB,
    "contexts_bytes": 2 * _MIB,
    "roles_bytes": 32 * _MIB,
}

_PROPOSAL_KEYS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "parent_plan_sha256",
        "choice_rule",
        "classical_candidate_rule",
        "choices",
        "classical_candidates",
        "jobs",
        "proposal_sha256",
    }
)

_FIT_STAGE = "source_fit"
_PREDICTION_STAGE = "source_validation_prediction"

_MANIFEST_COLUMNS = (
    "observation_uid",
    "master_sample_id",
    "target_analyte",
    "instrument",
    "station",
)
_CONTEXT_COLUMNS = (
    "context_id",
    "station",
    "held_instrument",
    "selection_mode",
    "phase_gate",
    "outer_fit_uid_sha256",
    "outer_test_uid_sha256",
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

_MANIFEST_MAX_ROWS = 598
_CONTEXT_MAX_ROWS = 1000
_ROLE_MAX_ROWS = 200000

_HELD_GATE = "held_evaluation"
_SELECTION_MODES = ("master_cv", "pseudo_domain")
_OUTER_ROLES = ("outer_fit", "outer_test")
_SELECTION_ROLES = ("selection_fit", "selection_validation")
_KNOWN_ROLES = frozenset(_OUTER_ROLES + _SELECTION_ROLES)

_REASON_CODES = frozenset(
    {
        "scientific_execution_not_authorized",
        "invalid_input",
        "bytes_not_authenticated",
        "invalid_proposal",
        "invalid_jobs",
        "job_pair_mismatch",
        "admission_failed",
        "journal_failed",
        "attempt_manifest_mismatch",
        "invalid_manifest",
        "invalid_contexts",
        "invalid_roles",
        "role_binding_mismatch",
        "verification_failed",
    }
)


class BindingError(ValueError):
    """ValueError carrying one static allowlisted reason code."""

    def __init__(self, reason_code):
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "invalid_input"
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(reason_code):
    raise BindingError(reason_code) from None


@dataclass(frozen=True, repr=False)
class Observation:
    observation_uid: str
    master_sample_id: str
    target_analyte: str
    instrument: str
    station: str


@dataclass(frozen=True, repr=False)
class SourceRoles:
    context_id: str
    unit_id: str
    selection_mode: str
    fitting_role_id: str
    validation_role_id: str
    classes: tuple
    fitting: tuple
    validation: tuple


@dataclass(frozen=True, repr=False)
class SourcePair:
    fit_job_json: str
    prediction_job_json: str
    source_roles: SourceRoles


@dataclass(frozen=True, repr=False)
class Binding:
    pairs: tuple
    report_json: str

    def public_report(self):
        return json.loads(self.report_json)


@dataclass(frozen=True, repr=False)
class _Context:
    context_id: str
    station: str
    held_instrument: str
    selection_mode: str
    phase_gate: str
    outer_fit_sha256: str
    outer_test_sha256: str


def _reject_constant(token):
    raise ValueError("nonfinite")


def _parse_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("nonfinite")
    return value


def _reject_duplicate_keys(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError("duplicate")
        seen.add(key)
    return dict(pairs)


def _parse_json(raw, reason):
    try:
        text = raw.decode("utf-8-sig")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
            parse_float=_parse_float,
        )
    except BindingError:
        raise
    except Exception:
        _fail(reason)


def _authenticate_bytes(values):
    for name in _BYTE_PINS:
        value = values.get(name)
        if type(value) is not bytes:
            _fail("invalid_input")
        if len(value) < _BYTE_MINIMUM or len(value) > _BYTE_LIMITS[name]:
            _fail("invalid_input")
    for name, pin in _BYTE_PINS.items():
        if hashlib.sha256(values[name]).hexdigest() != pin:
            _fail("bytes_not_authenticated")


def _read_csv(raw, columns, limit, reason):
    try:
        text = raw.decode("utf-8-sig")
    except Exception:
        _fail(reason)
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = reader.fieldnames
    if not header:
        _fail(reason)
    if len(header) != len(set(header)):
        _fail(reason)
    for column in columns:
        if column not in header:
            _fail(reason)
    rows = []
    try:
        for row in reader:
            if len(rows) >= limit:
                _fail(reason)
            if None in row:
                _fail(reason)
            for value in row.values():
                if value is None:
                    _fail(reason)
            rows.append({column: row[column] for column in columns})
    except BindingError:
        raise
    except Exception:
        _fail(reason)
    return rows


def _validate_proposal(proposal):
    if type(proposal) is not dict or len(proposal) != len(_PROPOSAL_KEYS):
        _fail("invalid_proposal")
    for key in proposal:
        if type(key) is not str or key not in _PROPOSAL_KEYS:
            _fail("invalid_proposal")
    if proposal["schema_version"] != _PROPOSAL_SCHEMA:
        _fail("invalid_proposal")
    if proposal["execution_authorized"] is not False:
        _fail("invalid_proposal")
    if proposal["parent_plan_sha256"] != _PARENT_PLAN_SHA256:
        _fail("invalid_proposal")
    body = {name: proposal[name] for name in _PROPOSAL_KEYS if name != "proposal_sha256"}
    if proposal["proposal_sha256"] != _hash(body):
        _fail("invalid_proposal")
    if proposal["proposal_sha256"] != admission.U0_PROPOSAL_SHA256:
        _fail("invalid_proposal")

    jobs = proposal["jobs"]
    if type(jobs) is not list or len(jobs) != 156:
        _fail("invalid_jobs")
    snapshots = []
    for job in jobs:
        try:
            snapshot = sp._validate_job(job)
            sp._check_job_hash(snapshot)
            sp._validate_job_semantics(snapshot)
        except sp.PredictionError:
            _fail("invalid_jobs")
        snapshots.append(snapshot)

    identifiers = [job["job_id"] for job in snapshots]
    if len(set(identifiers)) != len(identifiers):
        _fail("invalid_jobs")
    if identifiers != sorted(identifiers):
        _fail("invalid_jobs")

    fits = [job for job in snapshots if job["stage"] == _FIT_STAGE]
    predictions = [job for job in snapshots if job["stage"] == _PREDICTION_STAGE]
    if len(fits) != 78 or len(predictions) != 78:
        _fail("invalid_jobs")

    predictions_by_fit = {}
    for prediction in predictions:
        dependencies = prediction["dependencies"]
        if type(dependencies) is not list or len(dependencies) != 1:
            _fail("invalid_jobs")
        predictions_by_fit.setdefault(dependencies[0], []).append(prediction)

    pairs = []
    for fit in fits:
        matches = predictions_by_fit.pop(fit["job_id"], None)
        if not matches or len(matches) != 1:
            _fail("job_pair_mismatch")
        prediction = matches[0]
        try:
            sp._validate_pair(fit, prediction)
        except sp.PredictionError:
            _fail("job_pair_mismatch")
        pairs.append((fit, prediction))
    if predictions_by_fit:
        _fail("job_pair_mismatch")
    return pairs


def _projection(pairs):
    projection = []
    for fit, prediction in pairs:
        for job in (fit, prediction):
            worker = "cpu" if job["model_id"] in CLASSICAL_MODELS else "gpu"
            projection.append(
                {
                    "job_id": job["job_id"],
                    "stage": job["stage"],
                    "worker": worker,
                    "dependencies": list(job["dependencies"]),
                }
            )
    projection.sort(key=lambda item: item["job_id"])
    return projection


def _admit_attempt(attempt):
    if type(attempt) is not dict:
        _fail("invalid_input")
    try:
        ok = admission._u0_binding_ok(attempt)
    except BindingError:
        raise
    except Exception:
        _fail("admission_failed")
    if not ok:
        _fail("admission_failed")


def _replay_attempt(attempt):
    try:
        attempt_journal.replay_attempt_journal(
            attempt,
            [],
            expected_manifest_sha256=admission.U0_MANIFEST_SHA256,
            expected_head_sha256=admission.U0_MANIFEST_SHA256,
        )
    except attempt_journal.JournalError:
        _fail("journal_failed")
    except BindingError:
        raise
    except Exception:
        _fail("journal_failed")


def _parse_manifest(rows):
    observations = {}
    masters = {}
    for row in rows:
        uid = row["observation_uid"]
        master = row["master_sample_id"]
        target = row["target_analyte"]
        instrument = row["instrument"]
        station = row["station"]
        for value in (uid, master, target, instrument, station):
            if not sp._is_identifier(value):
                _fail("invalid_manifest")
        if uid in observations:
            _fail("invalid_manifest")
        previous = masters.get(master)
        if previous is None:
            masters[master] = (target, station)
        elif previous != (target, station):
            _fail("invalid_manifest")
        observations[uid] = Observation(uid, master, target, instrument, station)
    return observations


def _parse_contexts(rows):
    contexts = {}
    for row in rows:
        context_id = row["context_id"]
        station = row["station"]
        held = row["held_instrument"]
        mode = row["selection_mode"]
        gate = row["phase_gate"]
        for value in (context_id, station, held, mode, gate):
            if not sp._is_identifier(value):
                _fail("invalid_contexts")
        if context_id in contexts:
            _fail("invalid_contexts")
        outer_fit = row["outer_fit_uid_sha256"]
        outer_test = row["outer_test_uid_sha256"]
        if not sp._is_lower_hex64(outer_fit) or not sp._is_lower_hex64(outer_test):
            _fail("invalid_contexts")
        contexts[context_id] = _Context(
            context_id, station, held, mode, gate, outer_fit, outer_test
        )
    return contexts


def _parse_roles(rows):
    roles = []
    for row in rows:
        values = (
            row["context_id"],
            row["role_id"],
            row["role"],
            row["selection_unit_id"],
            row["observation_uid"],
            row["master_sample_id"],
            row["target_analyte"],
            row["instrument"],
        )
        for value in values:
            if not sp._is_identifier(value):
                _fail("invalid_roles")
        roles.append(values)
    return roles


def _masters(observations, uids):
    return {observations[uid].master_sample_id for uid in uids}


def _bind_roles(job_pairs, contexts, observations, role_rows):
    job_by_key = {}
    units_by_context = {}
    for fit, prediction in job_pairs:
        key = (fit["context_id"], fit["unit_id"])
        job_by_key.setdefault(key, []).append((fit, prediction))
        units_by_context.setdefault(fit["context_id"], set()).add(fit["unit_id"])

    selected = {}
    for context_id in units_by_context:
        context = contexts.get(context_id)
        if context is None:
            _fail("role_binding_mismatch")
        if context.phase_gate != _HELD_GATE or context.selection_mode not in _SELECTION_MODES:
            _fail("role_binding_mismatch")
        selected[context_id] = context

    rows_by_context = {}
    for row in role_rows:
        if row[0] in selected:
            rows_by_context.setdefault(row[0], []).append(row)

    roles_units = {}
    for context_id, context in selected.items():
        rows = rows_by_context.get(context_id)
        if not rows:
            _fail("role_binding_mismatch")
        groups = {}
        role_ids = {}
        for row in rows:
            (_cid, role_id, role, unit, uid, master, target, instrument) = row
            if role not in _KNOWN_ROLES:
                _fail("invalid_roles")
            expected_role_id = (
                "P04ROLE-"
                + sha256_value({"context_id": context_id, "role": role, "unit": unit})[:24]
            )
            if role_id != expected_role_id:
                _fail("role_binding_mismatch")
            if role in _OUTER_ROLES and unit != role:
                _fail("role_binding_mismatch")
            observation = observations.get(uid)
            if observation is None:
                _fail("role_binding_mismatch")
            if (
                observation.master_sample_id != master
                or observation.target_analyte != target
                or observation.instrument != instrument
            ):
                _fail("role_binding_mismatch")
            if observation.station != context.station:
                _fail("role_binding_mismatch")
            group = groups.setdefault((role, unit), [])
            if uid in group:
                _fail("role_binding_mismatch")
            group.append(uid)
            role_ids[(role, unit)] = role_id

        outer_fit = groups.get(("outer_fit", "outer_fit"))
        outer_test = groups.get(("outer_test", "outer_test"))
        if not outer_fit or not outer_test:
            _fail("role_binding_mismatch")
        outer_fit_set = set(outer_fit)
        outer_test_set = set(outer_test)

        for uid in outer_fit:
            if observations[uid].instrument == context.held_instrument:
                _fail("role_binding_mismatch")
        for uid in outer_test:
            if observations[uid].instrument != context.held_instrument:
                _fail("role_binding_mismatch")
        if _masters(observations, outer_fit) & _masters(observations, outer_test):
            _fail("role_binding_mismatch")
        if outer_fit_set & outer_test_set:
            _fail("role_binding_mismatch")
        if sha256_value(sorted(outer_fit)) != context.outer_fit_sha256:
            _fail("role_binding_mismatch")
        if sha256_value(sorted(outer_test)) != context.outer_test_sha256:
            _fail("role_binding_mismatch")
        resolved_test = sha256_value(sorted(outer_test))

        for unit in sorted(units_by_context[context_id]):
            fit_uids = groups.get(("selection_fit", unit))
            validation_uids = groups.get(("selection_validation", unit))
            if not fit_uids or not validation_uids:
                _fail("role_binding_mismatch")
            fit_set = set(fit_uids)
            validation_set = set(validation_uids)

            if not fit_set <= outer_fit_set:
                _fail("role_binding_mismatch")
            if not validation_set <= outer_fit_set:
                _fail("role_binding_mismatch")
            if _masters(observations, fit_uids) & _masters(observations, validation_uids):
                _fail("role_binding_mismatch")
            if fit_set & validation_set:
                _fail("role_binding_mismatch")

            if context.selection_mode == "master_cv":
                if not unit.startswith("master_cv:"):
                    _fail("role_binding_mismatch")
                if fit_set | validation_set != outer_fit_set:
                    _fail("role_binding_mismatch")
            else:
                if not unit.startswith("pseudo:"):
                    _fail("role_binding_mismatch")
                pseudo_instrument = unit[len("pseudo:") :]
                if not sp._is_identifier(pseudo_instrument):
                    _fail("role_binding_mismatch")
                for uid in validation_uids:
                    if observations[uid].instrument != pseudo_instrument:
                        _fail("role_binding_mismatch")
                for uid in fit_uids:
                    if observations[uid].instrument == pseudo_instrument:
                        _fail("role_binding_mismatch")

            fit_classes = sorted({observations[uid].target_analyte for uid in fit_uids})
            validation_classes = sorted(
                {observations[uid].target_analyte for uid in validation_uids}
            )
            if len(fit_classes) != 3 or fit_classes != validation_classes:
                _fail("role_binding_mismatch")

            resolved_fit = sha256_value(sorted(fit_uids))
            resolved_validation = sha256_value(sorted(validation_uids))
            for fit_job, _prediction in job_by_key[(context_id, unit)]:
                if fit_job["fit_uid_sha256"] != resolved_fit:
                    _fail("role_binding_mismatch")
                if fit_job["validation_uid_sha256"] != resolved_validation:
                    _fail("role_binding_mismatch")
                if fit_job["test_uid_sha256"] != resolved_test:
                    _fail("role_binding_mismatch")

            roles_units[(context_id, unit)] = SourceRoles(
                context_id=context_id,
                unit_id=unit,
                selection_mode=context.selection_mode,
                fitting_role_id=role_ids[("selection_fit", unit)],
                validation_role_id=role_ids[("selection_validation", unit)],
                classes=tuple(fit_classes),
                fitting=tuple(observations[uid] for uid in sorted(fit_uids)),
                validation=tuple(observations[uid] for uid in sorted(validation_uids)),
            )

    pairs = []
    for fit, prediction in job_pairs:
        key = (fit["context_id"], fit["unit_id"])
        pairs.append(
            SourcePair(
                fit_job_json=json.dumps(fit, sort_keys=True, ensure_ascii=True),
                prediction_job_json=json.dumps(prediction, sort_keys=True, ensure_ascii=True),
                source_roles=roles_units[key],
            )
        )
    return pairs, len(selected)


def _build_report(manifest_rows, selected_contexts, job_pairs):
    fits = [fit for fit, _prediction in job_pairs]
    cpu_fit_jobs = sum(1 for fit in fits if fit["model_id"] in CLASSICAL_MODELS)
    report = {
        "schema_version": SCHEMA_VERSION,
        "byte_pins": {name: pin for name, pin in _BYTE_PINS.items()},
        "proposal_sha256": admission.U0_PROPOSAL_SHA256,
        "manifest_sha256": admission.U0_MANIFEST_SHA256,
        "parent_plan_sha256": _PARENT_PLAN_SHA256,
        "manifest_rows": manifest_rows,
        "selected_contexts": selected_contexts,
        "source_units": len({(fit["context_id"], fit["unit_id"]) for fit in fits}),
        "source_fit_jobs": len(fits),
        "source_prediction_jobs": len(fits),
        "cpu_fit_jobs": cpu_fit_jobs,
        "gpu_fit_jobs": len(fits) - cpu_fit_jobs,
        "registry_bytes_verified": True,
        "physical_role_isolation_verified": True,
        "ordered_source_roles_verified": True,
        "arrays_verified": False,
        "model_parameters_loaded": False,
        "live_controller_verified": False,
        "execution_authorized": False,
        "new_scientific_operations": 0,
    }
    report["report_sha256"] = sha256_value(report)
    return json.dumps(report, sort_keys=True, ensure_ascii=True)


def _bind(
    proposal_bytes,
    attempt_manifest_bytes,
    manifest_bytes,
    contexts_bytes,
    roles_bytes,
):
    values = {
        "proposal_bytes": proposal_bytes,
        "attempt_manifest_bytes": attempt_manifest_bytes,
        "manifest_bytes": manifest_bytes,
        "contexts_bytes": contexts_bytes,
        "roles_bytes": roles_bytes,
    }
    _authenticate_bytes(values)

    proposal = _parse_json(values["proposal_bytes"], "invalid_proposal")
    job_pairs = _validate_proposal(proposal)

    attempt = _parse_json(values["attempt_manifest_bytes"], "invalid_input")
    _admit_attempt(attempt)
    _replay_attempt(attempt)
    if attempt.get("jobs") != _projection(job_pairs):
        _fail("attempt_manifest_mismatch")

    manifest_rows = _read_csv(
        values["manifest_bytes"], _MANIFEST_COLUMNS, _MANIFEST_MAX_ROWS, "invalid_manifest"
    )
    observations = _parse_manifest(manifest_rows)

    context_rows = _read_csv(
        values["contexts_bytes"], _CONTEXT_COLUMNS, _CONTEXT_MAX_ROWS, "invalid_contexts"
    )
    contexts = _parse_contexts(context_rows)

    role_rows = _read_csv(values["roles_bytes"], _ROLE_COLUMNS, _ROLE_MAX_ROWS, "invalid_roles")
    roles = _parse_roles(role_rows)

    pairs, selected_contexts = _bind_roles(job_pairs, contexts, observations, roles)
    report_json = _build_report(len(manifest_rows), selected_contexts, job_pairs)
    return Binding(pairs=tuple(pairs), report_json=report_json)


def bind_u0_source_metadata(
    *,
    proposal_bytes,
    attempt_manifest_bytes,
    manifest_bytes,
    contexts_bytes,
    roles_bytes,
):
    """Authenticate five byte buffers and bind U0 jobs to source roles."""
    try:
        return _bind(
            proposal_bytes,
            attempt_manifest_bytes,
            manifest_bytes,
            contexts_bytes,
            roles_bytes,
        )
    except BindingError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("verification_failed")


def require_scientific_execution(*args, **kwargs):
    """Always deny scientific execution, regardless of forged flags."""
    _fail("scientific_execution_not_authorized")
