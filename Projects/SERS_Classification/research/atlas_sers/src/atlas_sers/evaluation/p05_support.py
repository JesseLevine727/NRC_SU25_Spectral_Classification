"""Outcome-blind P05 inherited-role source-support audit.

This module reads three caller-supplied CSV tables (a sample manifest, a P04
context registry, and a P04 role registry) whose SHA-256 digests are pinned by
the caller. It validates the inherited P04 role boundaries, then reports
positive-pair availability and class/instrument support for the two fitting
roles (``selection_fit`` and ``outer_fit``). It never accesses spectra, never
reads observed outcomes, never fits a model, and never writes an output file.
The CLI prints canonical JSON to stdout and exits 0 only when the metadata
audit passes.

Integrity note: the required digests are a caller-pinned consistency check
against a supervisor-verified prerequisite state. They are not independent
authentication of a forged upstream registry and must not be described as such.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas_sers.evaluation.p05_readiness import (
    ReadinessError,
    build_readiness_report,
)

SCHEMA_VERSION = "nato-sers-p05-support-audit-v1"
AUDIT_STATUS = "pass"
SCOPE = "inherited_role_metadata_only"
AUDIT_ID_VERSION = "p05-support-audit-v1"
PAIR_ID_VERSION = "p05-support-pair-v1"

ROLE_NAMES = ("outer_fit", "outer_test", "selection_fit", "selection_validation")
FITTING_ROLE_ORDER = ("selection_fit", "outer_fit")
OUTER_UNIT_BY_ROLE = {"outer_fit": "outer_fit", "outer_test": "outer_test"}

STATION_TASK = {"cwa": "T1-CWA", "pills": "T1-PILLS", "surfaces": "T1-SURF"}

MANIFEST_COLUMNS = (
    "observation_uid",
    "master_sample_id",
    "station",
    "target_analyte",
    "instrument",
    "sensor_family",
)
CONTEXT_COLUMNS = (
    "context_id",
    "station",
    "task_id",
    "domain",
    "held_instrument",
    "selection_mode",
    "phase_gate",
    "selection_unit_count",
    "outer_fit_rows",
    "outer_fit_masters",
    "outer_test_rows",
    "outer_fit_uid_sha256",
    "outer_test_uid_sha256",
)
ROLE_COLUMNS = (
    "context_id",
    "role_id",
    "role",
    "selection_unit_id",
    "observation_uid",
    "master_sample_id",
    "target_analyte",
    "instrument",
)

SELECTION_MODE_UNIT_PREFIX = {
    "inner_master_cv": "outer_fold_as_inner:",
    "pseudo_domain": "pseudo:",
    "master_cv": "master_cv:",
}
SELECTION_MODES_BY_PHASE = {
    "development": ("inner_master_cv",),
    "held_evaluation": ("pseudo_domain", "master_cv"),
}

CLAIM_BOUNDARY = (
    "Metadata-only audit of inherited P04 source-fitting roles. Pair counts are "
    "availability counts, not sampled training pairs or independent replicates, "
    "and are not a held performance evaluation. P05 training remains unauthorized."
)

INTEGRITY_NOTE = (
    "The required SHA-256 digests are a caller-pinned integrity check against a "
    "supervisor-verified prerequisite state, not independent authentication of a "
    "forged upstream registry. Hashes are reported under logical keys only."
)


class SupportAuditError(ValueError):
    """Raised when inherited-role metadata is missing, malformed, or contradictory."""


class _InvalidArguments(Exception):
    """Raised by the CLI parser when command-line arguments are invalid."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _InvalidArguments(message)


@dataclass(frozen=True)
class SupportInputs:
    """Validated-shape CSV rows plus their caller-pinned digests."""

    manifest: tuple[dict[str, Any], ...]
    contexts: tuple[dict[str, Any], ...]
    roles: tuple[dict[str, Any], ...]
    manifest_sha256: str | None
    contexts_sha256: str | None
    roles_sha256: str | None


# --------------------------------------------------------------------------- #
# Canonical hashing helpers (stdlib-only reimplementation of the governance
# convention: compact sorted-key JSON serialized as UTF-8, then SHA-256).
# --------------------------------------------------------------------------- #


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    """Hash a JSON-compatible value with the repository canonical convention."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def uid_set_hash(uids: Iterable[str]) -> str:
    """Hash a UID set exactly as ``sha256_value(sorted(UID_list))``."""

    return sha256_value(sorted(str(uid) for uid in uids))


def audit_identity(
    *,
    context_id: str,
    role_id: str,
    role: str,
    selection_unit_id: str,
) -> str:
    """Return the distinct ``P05AUDIT-...`` provenance identifier for a fitting role."""

    identity = {
        "version": AUDIT_ID_VERSION,
        "parent_context_id": context_id,
        "parent_role_id": role_id,
        "role": role,
        "selection_unit_id": selection_unit_id,
    }
    return f"P05AUDIT-{sha256_value(identity)[:24]}"


def pair_identity(
    *,
    context_id: str,
    role_id: str,
    uid_a: str,
    uid_b: str,
) -> str:
    """Return the deterministic unordered positive-pair identifier.

    This is an audit proposal only, not a frozen sampler specification. The
    context, role, and both constituent UIDs must be nonempty strings, and an
    identical constituent pair is rejected because a self-pair is not positive.
    """

    for label, value in (
        ("context_id", context_id),
        ("role_id", role_id),
        ("uid_a", uid_a),
        ("uid_b", uid_b),
    ):
        if not isinstance(value, str) or not value or value != value.strip():
            raise SupportAuditError(
                f"Pair identity requires a nonempty, unpadded string {label}."
            )
    if uid_a == uid_b:
        raise SupportAuditError("A positive pair requires two distinct observation UIDs.")
    identity = {
        "version": PAIR_ID_VERSION,
        "parent_context_id": context_id,
        "parent_role_id": role_id,
        "sorted_uids": sorted((uid_a, uid_b)),
    }
    return f"P05PAIR-{sha256_value(identity)[:24]}"


# --------------------------------------------------------------------------- #
# Loading and pure construction
# --------------------------------------------------------------------------- #


def build_support_inputs(
    *,
    manifest: Iterable[Mapping[str, Any]],
    contexts: Iterable[Mapping[str, Any]],
    roles: Iterable[Mapping[str, Any]],
) -> SupportInputs:
    """Pure builder for synthetic tests; performs no file access or hashing."""

    return SupportInputs(
        manifest=tuple(dict(row) for row in manifest),
        contexts=tuple(dict(row) for row in contexts),
        roles=tuple(dict(row) for row in roles),
        manifest_sha256=None,
        contexts_sha256=None,
        roles_sha256=None,
    )


def load_support_inputs(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    contexts_path: Path | str,
    contexts_sha256: str,
    roles_path: Path | str,
    roles_sha256: str,
) -> SupportInputs:
    """Verify all pinned digests before parsing any CSV table."""

    manifest_raw = _read_pinned(manifest_path, manifest_sha256, "manifest")
    contexts_raw = _read_pinned(contexts_path, contexts_sha256, "contexts")
    roles_raw = _read_pinned(roles_path, roles_sha256, "roles")
    return SupportInputs(
        manifest=_parse_table(manifest_raw, "manifest", MANIFEST_COLUMNS),
        contexts=_parse_table(contexts_raw, "contexts", CONTEXT_COLUMNS),
        roles=_parse_table(roles_raw, "roles", ROLE_COLUMNS),
        manifest_sha256=str(manifest_sha256).lower(),
        contexts_sha256=str(contexts_sha256).lower(),
        roles_sha256=str(roles_sha256).lower(),
    )


def _read_pinned(path: Path | str, expected: str, logical_key: str) -> bytes:
    if not isinstance(expected, str) or not _is_sha256(expected):
        raise SupportAuditError(
            f"Expected SHA-256 for '{logical_key}' must be 64 hexadecimal characters."
        )
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise SupportAuditError(f"Input table '{logical_key}' is unreadable.") from error
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected.lower():
        raise SupportAuditError(
            f"Input table '{logical_key}' SHA-256 mismatch; refusing to parse."
        )
    return raw


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value.lower())


def _parse_table(
    raw: bytes,
    logical_key: str,
    required_columns: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SupportAuditError(
            f"Input table '{logical_key}' is not valid UTF-8."
        ) from error
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
        if not header or any(not field for field in header):
            raise SupportAuditError(
                f"Input table '{logical_key}' has an empty header field."
            )
        if len(set(header)) != len(header):
            raise SupportAuditError(
                f"Input table '{logical_key}' has duplicate header columns."
            )
        missing = [column for column in required_columns if column not in header]
        if missing:
            raise SupportAuditError(
                f"Input table '{logical_key}' is missing required columns."
            )
        retain = list(required_columns)
        if "outer_test_masters" not in retain and "outer_test_masters" in header:
            retain.append("outer_test_masters")
        column_index = {column: header.index(column) for column in retain}
        rows: list[dict[str, Any]] = []
        for line_number, values in enumerate(reader, start=2):
            if not values:
                raise SupportAuditError(
                    f"Input table '{logical_key}' has a blank row at line {line_number}."
                )
            if len(values) != len(header):
                raise SupportAuditError(
                    "Input table has a row with the wrong field count at "
                    f"line {line_number}."
                )
            selected = [values[column_index[column]] for column in retain]
            rows.append(dict(zip(retain, selected, strict=True)))
    except StopIteration as error:
        raise SupportAuditError(f"Input table '{logical_key}' is empty.") from error
    except csv.Error as error:
        raise SupportAuditError(
            f"Input table '{logical_key}' is not well-formed CSV."
        ) from error
    if not rows:
        raise SupportAuditError(f"Input table '{logical_key}' has no data rows.")
    return tuple(rows)


# --------------------------------------------------------------------------- #
# Boundary validation
# --------------------------------------------------------------------------- #


def _required_text(row: Mapping[str, Any], key: str, name: str) -> str:
    if key not in row:
        raise SupportAuditError(f"{name} is missing required column '{key}'.")
    value = row[key]
    if not isinstance(value, str) or not value:
        raise SupportAuditError(f"{name} has an empty '{key}' value.")
    if value != value.strip():
        raise SupportAuditError(
            f"{name} has a '{key}' value with surrounding whitespace."
        )
    return value


def _count(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise SupportAuditError(f"{name} must be an integer count.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        if not value or any(not character.isdigit() for character in value):
            raise SupportAuditError(f"{name} must be an integer count.")
        try:
            number = int(value)
        except ValueError as error:
            raise SupportAuditError(f"{name} must be an integer count.") from error
    else:
        raise SupportAuditError(f"{name} must be an integer count.")
    if number < minimum:
        raise SupportAuditError(f"{name} must be at least {minimum}.")
    return number


def _validate_inputs(inputs: SupportInputs) -> dict[str, Any]:
    if not inputs.manifest or not inputs.contexts or not inputs.roles:
        raise SupportAuditError("Manifest, context, and role tables must all be nonempty.")

    manifest_by_uid: dict[str, dict[str, str]] = {}
    master_station: dict[str, str] = {}
    master_chemical: dict[str, str] = {}
    for row in inputs.manifest:
        uid = _required_text(row, "observation_uid", "manifest row")
        master = _required_text(row, "master_sample_id", "manifest row")
        station = _required_text(row, "station", "manifest row")
        chemical = _required_text(row, "target_analyte", "manifest row")
        instrument = _required_text(row, "instrument", "manifest row")
        family = _required_text(row, "sensor_family", "manifest row")
        if uid in manifest_by_uid:
            raise SupportAuditError("Manifest contains duplicate observation_uid values.")
        previous_station = master_station.get(master)
        if previous_station is not None and previous_station != station:
            raise SupportAuditError("A master sample maps to more than one station.")
        previous_chemical = master_chemical.get(master)
        if previous_chemical is not None and previous_chemical != chemical:
            raise SupportAuditError("A master sample maps to more than one target analyte.")
        master_station[master] = station
        master_chemical[master] = chemical
        manifest_by_uid[uid] = {
            "uid": uid,
            "master": master,
            "station": station,
            "chemical": chemical,
            "instrument": instrument,
            "family": family,
        }

    contexts: dict[str, dict[str, Any]] = {}
    for row in inputs.contexts:
        context_id = _required_text(row, "context_id", "context row")
        if context_id in contexts:
            raise SupportAuditError("Context registry contains duplicate context_id values.")
        station = _required_text(row, "station", "context row")
        if station not in STATION_TASK:
            raise SupportAuditError("Context declares an unknown station.")
        task_id = _required_text(row, "task_id", "context row")
        domain = _required_text(row, "domain", "context row")
        held_instrument = _required_text(row, "held_instrument", "context row")
        selection_mode = _required_text(row, "selection_mode", "context row")
        phase_gate = _required_text(row, "phase_gate", "context row")
        if phase_gate not in SELECTION_MODES_BY_PHASE:
            raise SupportAuditError("Context declares an unknown phase_gate.")
        if selection_mode not in SELECTION_MODE_UNIT_PREFIX:
            raise SupportAuditError("Context declares an unknown selection_mode.")
        if selection_mode not in SELECTION_MODES_BY_PHASE[phase_gate]:
            raise SupportAuditError(
                "Context selection_mode is inconsistent with its phase_gate."
            )
        if phase_gate == "development":
            if held_instrument != "not_applicable":
                raise SupportAuditError(
                    "A development context must use the not_applicable held instrument."
                )
            if task_id != STATION_TASK[station]:
                raise SupportAuditError(
                    "A development context task id disagrees with its station."
                )
            if domain != f"{station}:within":
                raise SupportAuditError(
                    "A development context domain disagrees with its station."
                )
        else:
            if held_instrument == "not_applicable":
                raise SupportAuditError(
                    "A held-evaluation context must declare a held instrument."
                )
            if task_id != "T3-ZS":
                raise SupportAuditError(
                    "A held-evaluation context must use the T3 task id."
                )
            if domain != f"{station}:{held_instrument}":
                raise SupportAuditError(
                    "A held-evaluation context domain disagrees with its station and "
                    "held instrument."
                )
        outer_fit_hash = _required_text(row, "outer_fit_uid_sha256", "context row")
        outer_test_hash = _required_text(row, "outer_test_uid_sha256", "context row")
        if not _is_sha256(outer_fit_hash) or not _is_sha256(outer_test_hash):
            raise SupportAuditError(
                "Context outer UID-set hashes must be 64 hexadecimal characters."
            )
        outer_test_masters: int | None = None
        raw_outer_test_masters = row.get("outer_test_masters")
        if raw_outer_test_masters is not None and not (
            isinstance(raw_outer_test_masters, str) and not raw_outer_test_masters
        ):
            outer_test_masters = _count(
                raw_outer_test_masters,
                "context outer_test_masters",
                minimum=1,
            )
        contexts[context_id] = {
            "context_id": context_id,
            "station": station,
            "task_id": task_id,
            "domain": domain,
            "held_instrument": held_instrument,
            "selection_mode": selection_mode,
            "phase_gate": phase_gate,
            "selection_unit_count": _count(
                row.get("selection_unit_count"),
                "context selection_unit_count",
                minimum=1,
            ),
            "outer_fit_rows": _count(
                row.get("outer_fit_rows"),
                "context outer_fit_rows",
                minimum=1,
            ),
            "outer_fit_masters": _count(
                row.get("outer_fit_masters"),
                "context outer_fit_masters",
                minimum=1,
            ),
            "outer_test_rows": _count(
                row.get("outer_test_rows"),
                "context outer_test_rows",
                minimum=1,
            ),
            "outer_test_masters": outer_test_masters,
            "outer_fit_uid_sha256": outer_fit_hash.lower(),
            "outer_test_uid_sha256": outer_test_hash.lower(),
        }

    role_meta: dict[str, tuple[str, str, str]] = {}
    role_rows_by_id: dict[str, list[dict[str, str]]] = {}
    role_index: dict[tuple[str, str, str], str] = {}
    seen_uids: dict[str, set[str]] = {}
    for row in inputs.roles:
        context_id = _required_text(row, "context_id", "role row")
        role_id = _required_text(row, "role_id", "role row")
        role = _required_text(row, "role", "role row")
        unit_id = _required_text(row, "selection_unit_id", "role row")
        uid = _required_text(row, "observation_uid", "role row")
        master = _required_text(row, "master_sample_id", "role row")
        chemical = _required_text(row, "target_analyte", "role row")
        instrument = _required_text(row, "instrument", "role row")
        if role not in ROLE_NAMES:
            raise SupportAuditError("Role row declares an unknown role name.")
        if context_id not in contexts:
            raise SupportAuditError("A role references an unknown context.")
        if uid not in manifest_by_uid:
            raise SupportAuditError("A role references an unknown observation UID.")
        source = manifest_by_uid[uid]
        if (
            master != source["master"]
            or chemical != source["chemical"]
            or instrument != source["instrument"]
        ):
            raise SupportAuditError("Role metadata disagrees with the manifest.")
        if source["station"] != contexts[context_id]["station"]:
            raise SupportAuditError("A role observation disagrees with its context station.")
        expected_outer_unit = OUTER_UNIT_BY_ROLE.get(role)
        if expected_outer_unit is not None:
            if unit_id != expected_outer_unit:
                raise SupportAuditError(
                    "An outer role carries an unexpected selection unit id."
                )
        else:
            prefix = SELECTION_MODE_UNIT_PREFIX[contexts[context_id]["selection_mode"]]
            if not unit_id.startswith(prefix) or len(unit_id) == len(prefix):
                raise SupportAuditError(
                    "A selection unit id is inconsistent with its selection mode."
                )
        key = (context_id, role, unit_id)
        previous_role = role_index.get(key)
        if previous_role is None:
            role_index[key] = role_id
        elif previous_role != role_id:
            raise SupportAuditError("A selection unit maps to more than one role id.")
        previous_identity = role_meta.get(role_id)
        if previous_identity is None:
            role_meta[role_id] = key
        elif previous_identity != key:
            raise SupportAuditError("A role id spans more than one context/type/unit.")
        role_seen = seen_uids.setdefault(role_id, set())
        if uid in role_seen:
            raise SupportAuditError("A role contains a duplicate observation UID.")
        role_seen.add(uid)
        role_rows_by_id.setdefault(role_id, []).append(
            {
                "uid": uid,
                "master": master,
                "chemical": chemical,
                "instrument": instrument,
                "family": source["family"],
            }
        )
    if not role_index:
        raise SupportAuditError("Role registry defines no roles.")

    outer_fit_uids: dict[str, frozenset[str]] = {}
    outer_test_uids: dict[str, frozenset[str]] = {}
    for context_id in sorted(contexts):
        context = contexts[context_id]
        fit_role_id = role_index.get((context_id, "outer_fit", "outer_fit"))
        test_role_id = role_index.get((context_id, "outer_test", "outer_test"))
        if fit_role_id is None or not role_rows_by_id.get(fit_role_id):
            raise SupportAuditError("A context lacks a nonempty outer_fit role.")
        if test_role_id is None or not role_rows_by_id.get(test_role_id):
            raise SupportAuditError("A context lacks a nonempty outer_test role.")
        fit_rows = role_rows_by_id[fit_role_id]
        test_rows = role_rows_by_id[test_role_id]
        fit_set = frozenset(record["uid"] for record in fit_rows)
        test_set = frozenset(record["uid"] for record in test_rows)
        if len(fit_rows) != context["outer_fit_rows"]:
            raise SupportAuditError("outer_fit row count disagrees with the context.")
        if len({record["master"] for record in fit_rows}) != context["outer_fit_masters"]:
            raise SupportAuditError("outer_fit master count disagrees with the context.")
        if uid_set_hash(fit_set) != context["outer_fit_uid_sha256"]:
            raise SupportAuditError("outer_fit UID-set hash disagrees with the context.")
        if len(test_rows) != context["outer_test_rows"]:
            raise SupportAuditError("outer_test row count disagrees with the context.")
        if uid_set_hash(test_set) != context["outer_test_uid_sha256"]:
            raise SupportAuditError("outer_test UID-set hash disagrees with the context.")
        if context["outer_test_masters"] is not None and (
            len({record["master"] for record in test_rows})
            != context["outer_test_masters"]
        ):
            raise SupportAuditError("outer_test master count disagrees with the context.")
        if {record["master"] for record in fit_rows} & {
            record["master"] for record in test_rows
        }:
            raise SupportAuditError("Outer fitting and test masters must be disjoint.")
        outer_fit_uids[context_id] = fit_set
        outer_test_uids[context_id] = test_set

    for context_id in sorted(contexts):
        context = contexts[context_id]
        units = sorted(
            {
                unit
                for (cid, role, unit) in role_index
                if cid == context_id
                and role in ("selection_fit", "selection_validation")
            }
        )
        if len(units) != context["selection_unit_count"]:
            raise SupportAuditError(
                "Declared selection-unit count disagrees with the role registry."
            )
        outer_fit = outer_fit_uids[context_id]
        outer_fit_instruments = {
            manifest_by_uid[uid]["instrument"] for uid in outer_fit
        }
        if context["phase_gate"] == "held_evaluation":
            if context["held_instrument"] in outer_fit_instruments:
                raise SupportAuditError("The held instrument appears in outer_fit.")
            outer_test_instruments = {
                manifest_by_uid[uid]["instrument"] for uid in outer_test_uids[context_id]
            }
            if outer_test_instruments != {context["held_instrument"]}:
                raise SupportAuditError(
                    "outer_test must contain only the held instrument."
                )
        for unit in units:
            fit_role_id = role_index.get((context_id, "selection_fit", unit))
            validation_role_id = role_index.get(
                (context_id, "selection_validation", unit)
            )
            if fit_role_id is None or validation_role_id is None:
                raise SupportAuditError(
                    "A selection unit lacks a fit or validation role id."
                )
            fit_rows = role_rows_by_id.get(fit_role_id, [])
            validation_rows = role_rows_by_id.get(validation_role_id, [])
            if not fit_rows or not validation_rows:
                raise SupportAuditError(
                    "A selection unit lacks a nonempty fit or validation role."
                )
            fit_set = {record["uid"] for record in fit_rows}
            validation_set = {record["uid"] for record in validation_rows}
            if not fit_set <= outer_fit or not validation_set <= outer_fit:
                raise SupportAuditError(
                    "Selection-unit UIDs must be subsets of outer_fit."
                )
            if {record["master"] for record in fit_rows} & {
                record["master"] for record in validation_rows
            }:
                raise SupportAuditError(
                    "Selection-unit fitting and validation masters must be disjoint."
                )
            if context["phase_gate"] == "held_evaluation":
                fit_instruments = {record["instrument"] for record in fit_rows}
                validation_instruments = {
                    record["instrument"] for record in validation_rows
                }
                if (
                    context["held_instrument"] in fit_instruments
                    or context["held_instrument"] in validation_instruments
                ):
                    raise SupportAuditError(
                        "The held instrument appears in an inner selection role."
                    )
                if context["selection_mode"] == "pseudo_domain":
                    pseudo_instrument = unit[len("pseudo:") :]
                    if validation_instruments != {pseudo_instrument}:
                        raise SupportAuditError(
                            "A pseudo-unit validation set must contain only its "
                            "pseudo instrument."
                        )
                    if pseudo_instrument in fit_instruments:
                        raise SupportAuditError(
                            "A pseudo-unit fitting set must exclude its pseudo instrument."
                        )

    return {
        "manifest_by_uid": manifest_by_uid,
        "contexts": contexts,
        "role_meta": role_meta,
        "role_rows_by_id": role_rows_by_id,
        "role_index": role_index,
        "outer_fit_uids": outer_fit_uids,
        "outer_test_uids": outer_test_uids,
    }


# --------------------------------------------------------------------------- #
# Audit metrics
# --------------------------------------------------------------------------- #


def _role_metrics(
    context: Mapping[str, Any],
    role_id: str,
    role: str,
    unit_id: str,
    rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Build availability counts and the streaming positive-pair digest.

    Positive-pair digest convention: observations are sorted lexically by
    observation UID; all unordered pairs are visited in that UID-pair order;
    pairs whose two target analytes differ are skipped. For each same-chemical
    pair the ``P05PAIR-...`` identity is streamed as UTF-8 followed by a newline.
    No pair records are materialized or sorted, so the digest is stable
    regardless of input row order and scales to very large roles.
    """

    ordered = sorted(rows, key=lambda record: record["uid"])
    observations = len(ordered)
    masters = {record["master"] for record in ordered}
    instruments = {record["instrument"] for record in ordered}
    chemicals = {record["chemical"] for record in ordered}

    by_chemical: dict[str, list[Mapping[str, str]]] = {}
    by_master: dict[str, dict[str, set[str]]] = {}
    cells: dict[tuple[str, str], list[Any]] = {}
    for record in ordered:
        by_chemical.setdefault(record["chemical"], []).append(record)
        slot = by_master.setdefault(
            record["master"], {"instruments": set(), "families": set()}
        )
        slot["instruments"].add(record["instrument"])
        slot["families"].add(record["family"])
        cell = cells.setdefault((record["chemical"], record["instrument"]), [0, set()])
        cell[0] += 1
        cell[1].add(record["master"])

    categories = {
        "same_master_same_instrument": 0,
        "same_master_different_instrument": 0,
        "different_master_same_instrument": 0,
        "different_master_different_instrument": 0,
    }
    same_chemical_pairs = 0
    cross_substrate_pairs = 0
    digest = hashlib.sha256()
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left["chemical"] != right["chemical"]:
                continue
            same_chemical_pairs += 1
            if left["master"] == right["master"]:
                if left["instrument"] == right["instrument"]:
                    categories["same_master_same_instrument"] += 1
                else:
                    categories["same_master_different_instrument"] += 1
            else:
                if left["instrument"] == right["instrument"]:
                    categories["different_master_same_instrument"] += 1
                else:
                    categories["different_master_different_instrument"] += 1
            if left["family"] != right["family"]:
                cross_substrate_pairs += 1
            digest.update(
                pair_identity(
                    context_id=str(context["context_id"]),
                    role_id=role_id,
                    uid_a=left["uid"],
                    uid_b=right["uid"],
                ).encode("utf-8")
            )
            digest.update(b"\n")

    total_pairs = observations * (observations - 1) // 2
    master_counts_per_chemical = {
        chemical: len({record["master"] for record in group})
        for chemical, group in sorted(by_chemical.items())
    }
    chemicals_with_two_masters = sum(
        1 for count in master_counts_per_chemical.values() if count >= 2
    )
    zero_positive_anchors = sum(
        1 for group in by_chemical.values() if len(group) == 1
    )
    cell_records = [
        {
            "target_analyte": chemical,
            "instrument": instrument,
            "spectra": payload[0],
            "masters": len(payload[1]),
        }
        for (chemical, instrument), payload in sorted(cells.items())
    ]
    return {
        "audit_id": audit_identity(
            context_id=str(context["context_id"]),
            role_id=role_id,
            role=role,
            selection_unit_id=unit_id,
        ),
        "parent_context_id": context["context_id"],
        "parent_role_id": role_id,
        "role": role,
        "selection_unit_id": unit_id,
        "station": context["station"],
        "domain": context["domain"],
        "phase_gate": context["phase_gate"],
        "observation_count": observations,
        "master_count": len(masters),
        "instrument_count": len(instruments),
        "class_count": len(chemicals),
        "master_counts_per_chemical": master_counts_per_chemical,
        "same_chemical_pair_categories": categories,
        "same_chemical_pair_count": same_chemical_pairs,
        "different_chemical_pair_count": total_pairs - same_chemical_pairs,
        "same_chemical_cross_substrate_pair_count": cross_substrate_pairs,
        "same_chemical_cross_substrate_note": (
            "Overlaps the four same-chemical categories; not additive with them."
        ),
        "positive_pair_digest": digest.hexdigest(),
        "masters_with_multiple_instruments": sum(
            1 for slot in by_master.values() if len(slot["instruments"]) >= 2
        ),
        "masters_with_multiple_substrate_families": sum(
            1 for slot in by_master.values() if len(slot["families"]) >= 2
        ),
        "anchors_without_same_chemical_peer": zero_positive_anchors,
        "chemicals_with_two_or_more_masters": chemicals_with_two_masters,
        "two_chemical_two_master_support": chemicals_with_two_masters >= 2,
        "all_chemicals_have_two_or_more_masters": all(
            count >= 2 for count in master_counts_per_chemical.values()
        ),
        "class_instrument_cells": cell_records,
        "cells_with_two_or_more_spectra": sum(
            1 for cell in cell_records if cell["spectra"] >= 2
        ),
        "cells_with_two_or_more_masters": sum(
            1 for cell in cell_records if cell["masters"] >= 2
        ),
    }


def _build_entries(derived: Mapping[str, Any]) -> list[dict[str, Any]]:
    role_index: dict[tuple[str, str, str], str] = derived["role_index"]
    role_rows_by_id: dict[str, list[dict[str, str]]] = derived["role_rows_by_id"]
    contexts: dict[str, dict[str, Any]] = derived["contexts"]
    ordered = sorted(
        (
            (context_id, role, unit, role_index[(context_id, role, unit)])
            for (context_id, role, unit) in role_index
            if role in FITTING_ROLE_ORDER
        ),
        key=lambda item: (
            item[0],
            FITTING_ROLE_ORDER.index(item[1]),
            item[2],
        ),
    )
    return [
        _role_metrics(contexts[context_id], role_id, role, unit, role_rows_by_id[role_id])
        for context_id, role, unit, role_id in ordered
    ]


def _min_median_max(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        raise SupportAuditError("Summary statistics require a nonempty role group.")
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _summaries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(
            (entry["station"], entry["phase_gate"], entry["role"]), []
        ).append(entry)
    summaries: list[dict[str, Any]] = []
    for key in sorted(groups):
        station, phase_gate, role = key
        items = groups[key]
        same_master_cross = [
            entry["same_chemical_pair_categories"]["same_master_different_instrument"]
            for entry in items
        ]
        summaries.append(
            {
                "station": station,
                "phase_gate": phase_gate,
                "role": role,
                "role_count": len(items),
                "roles_lacking_same_master_cross_instrument_positives": sum(
                    1 for value in same_master_cross if value == 0
                ),
                "roles_lacking_different_master_cross_instrument_positives": sum(
                    1
                    for entry in items
                    if entry["same_chemical_pair_categories"][
                        "different_master_different_instrument"
                    ]
                    == 0
                ),
                "roles_lacking_two_chemical_two_master_support": sum(
                    1 for entry in items if not entry["two_chemical_two_master_support"]
                ),
                "roles_containing_zero_positive_anchors": sum(
                    1
                    for entry in items
                    if entry["anchors_without_same_chemical_peer"] > 0
                ),
                "observations": _min_median_max(
                    [entry["observation_count"] for entry in items]
                ),
                "masters": _min_median_max(
                    [entry["master_count"] for entry in items]
                ),
                "same_master_cross_instrument_pairs": _min_median_max(same_master_cross),
            }
        )
    return summaries


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #


def _readiness_summary(readiness: Mapping[str, Any]) -> dict[str, Any]:
    try:
        crossing = readiness["illustrative_full_crossing"]["fits_per_selection_unit"]
        loss_total = readiness["loss_configuration_total"]
        optimizer_count = readiness["optimizer_candidate_count"]
        seed_count = readiness["training_seed_count"]
        decisions = [
            str(item["decision_id"]) for item in readiness["unresolved_decisions"]
        ]
        input_hashes = dict(readiness["input_hashes"])
    except (KeyError, TypeError) as error:
        raise SupportAuditError(
            "Readiness report lacks the expected public contract fields."
        ) from error
    loss_total = _count(
        loss_total, "readiness loss_configuration_total", minimum=1
    )
    optimizer_count = _count(
        optimizer_count, "readiness optimizer_candidate_count", minimum=1
    )
    seed_count = _count(
        seed_count, "readiness training_seed_count", minimum=1
    )
    crossing = _count(
        crossing, "readiness fits_per_selection_unit", minimum=1
    )
    if crossing != loss_total * optimizer_count * seed_count:
        raise SupportAuditError(
            "Readiness crossing disagrees with its loss/optimizer/seed grids."
        )
    return {
        "loss_configuration_total": loss_total,
        "optimizer_candidate_count": optimizer_count,
        "training_seed_count": seed_count,
        "fits_per_selection_unit": crossing,
        "unresolved_decision_ids": decisions,
        "input_hashes": input_hashes,
    }


def build_support_report(
    inputs: SupportInputs,
    *,
    readiness: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate inherited roles and build the deterministic source-support report.

    ``readiness`` is required and must be the validated public readiness report;
    this builder never substitutes a private fallback crossing.
    """

    derived = _validate_inputs(inputs)
    entries = _build_entries(derived)
    contexts: dict[str, dict[str, Any]] = derived["contexts"]
    role_index: dict[tuple[str, str, str], str] = derived["role_index"]
    readiness_summary = _readiness_summary(readiness)
    crossing = readiness_summary["fits_per_selection_unit"]

    development_contexts = {
        context_id
        for context_id, context in contexts.items()
        if context["phase_gate"] == "development"
    }
    held_contexts = {
        context_id
        for context_id, context in contexts.items()
        if context["phase_gate"] == "held_evaluation"
    }
    development_units = 0
    held_units = 0
    for (context_id, role, _unit) in role_index:
        if role != "selection_fit":
            continue
        if contexts[context_id]["phase_gate"] == "development":
            development_units += 1
        else:
            held_units += 1
    overall_units = development_units + held_units
    overall_contexts = len(development_contexts | held_contexts)

    role_meta: dict[str, tuple[str, str, str]] = derived["role_meta"]
    counts = {
        "manifest_rows": len(inputs.manifest),
        "master_samples": len(
            {record["master"] for record in derived["manifest_by_uid"].values()}
        ),
        "contexts": len(contexts),
        "development_contexts": len(development_contexts),
        "held_evaluation_contexts": len(held_contexts),
        "role_ids": len(role_meta),
        "role_rows": len(inputs.roles),
        "fitting_role_ids": len(entries),
        "selection_units": overall_units,
        "selection_units_development": development_units,
        "selection_units_held_evaluation": held_units,
        "roles_by_name": {
            name: sum(1 for key in role_meta.values() if key[1] == name)
            for name in ROLE_NAMES
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "audit_status": AUDIT_STATUS,
        "scope": SCOPE,
        "scientific_execution_authorized": False,
        "scientific_fits_performed": 0,
        "total_required_fits": None,
        "resource_estimates": None,
        "claim_boundary": CLAIM_BOUNDARY,
        "integrity_note": INTEGRITY_NOTE,
        "counts": counts,
        "input_hashes": {
            "caller_pinned": {
                "manifest": inputs.manifest_sha256,
                "contexts": inputs.contexts_sha256,
                "roles": inputs.roles_sha256,
            },
            "readiness_contracts": readiness_summary["input_hashes"],
        },
        "readiness": readiness_summary,
        "unresolved_decision_ids": readiness_summary["unresolved_decision_ids"],
        "unresolved_decisions_resolved": False,
        "source_role_audit": entries,
        "summaries": _summaries(entries),
        "cost_scenario": {
            "label": "illustrative full-grid Cartesian scenario",
            "assumption": (
                "Illustrative Cartesian expansion of every enumerated loss and "
                "optimizer candidate and neural seed across each inherited source "
                "selection unit."
            ),
            "crossing_per_selection_unit": crossing,
            "crossing_source": "readiness_report",
            "source_contexts": {
                "development": len(development_contexts),
                "held_evaluation": len(held_contexts),
                "overall": overall_contexts,
            },
            "source_selection_units": {
                "development": development_units,
                "held_evaluation": held_units,
                "overall": overall_units,
            },
            "illustrative_fits": {
                "development": development_units * crossing,
                "held_evaluation": held_units * crossing,
                "overall": overall_units * crossing,
            },
            "authorizing": False,
            "exclusions": [
                "final refits",
                "temperature calibration",
                "D0 controls",
                "retries",
            ],
            "note": (
                "Not an approved P05 plan, total required fits, or compute-time "
                "estimate. Selection units are counted from selection_fit roles "
                "only. Training remains unauthorized regardless of the inherited "
                "P04 flags."
            ),
        },
    }


def render_report(report: Mapping[str, Any]) -> str:
    """Render the report as deterministic, byte-equivalent JSON text."""

    return json.dumps(report, indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    return _Parser(
        prog="audit_p05_support.py",
        description=(
            "Metadata-only audit of inherited P04 source-role support. Reads no "
            "spectra, fits no model, and has no train/run/execute option."
        ),
        epilog=(
            "Exit 0 means only that the metadata audit passed. Invalid input exits "
            "1 with a concise generic error and no traceback or record dump. JSON is "
            "printed to stdout; any file redirection is supervisor-controlled. The "
            "required SHA-256 digests are a caller-pinned integrity check, not "
            "independent authentication of a forged upstream registry."
        ),
    )


def _parser_with_arguments() -> argparse.ArgumentParser:
    parser = _build_parser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--contexts-sha256", required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--roles-sha256", required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser_with_arguments()
    try:
        args = parser.parse_args(argv)
    except _InvalidArguments:
        print("error: invalid command-line arguments", file=sys.stderr)
        return 1
    try:
        inputs = load_support_inputs(
            manifest_path=args.manifest,
            manifest_sha256=args.manifest_sha256,
            contexts_path=args.contexts,
            contexts_sha256=args.contexts_sha256,
            roles_path=args.roles,
            roles_sha256=args.roles_sha256,
        )
        readiness = build_readiness_report(project_root=args.project_root)
        report = build_support_report(inputs, readiness=readiness)
    except (SupportAuditError, ReadinessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
