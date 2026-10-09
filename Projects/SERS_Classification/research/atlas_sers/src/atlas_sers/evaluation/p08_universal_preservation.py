"""P08-U1 private preparation slice: F01 spectral display and F07 preservation summaries.

This module is a read-only preparation slice over authenticated frozen P01
preservation evidence. It authenticates the retained P01 state, artifact
manifest, primary manifest, preservation tables and representation archives,
verifies the private catalog membership against the primary manifest, and builds
the F01 master-equal spectral semantic bundle and the F07 domain/action
preservation summaries.

It performs no fitting, no preprocessing, no new metric calculation, no figure
rendering, no output writes and no subprocess execution. Public results contain
approved aggregates only; observation, master and private-example identities and
local filesystem paths never enter public semantics.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p05_comparison import _held_contexts
from atlas_sers.evaluation.p06p11_inputs import PINS as CONTEXT_PINS
from atlas_sers.evaluation.p08_universal_evidence import (
    _authenticated_file,
    _prepare_roots,
    _read_json_file,
    _stream_sha256,
)

__all__ = [
    "PreservationInputError",
    "PreservationSemanticError",
    "load_preservation_inputs",
    "build_preservation_semantics",
]

# --- pinned public inputs (monkeypatchable for synthetic fixtures only) -----
INPUT_SUPPORT_AUDIT_REL = "results/p08_readiness/input_support_audit.json"
INPUT_SUPPORT_AUDIT_SHA256 = (
    "08310a0ef5cd91dacc5a46c129a3043e32cadcd0a06f305f34c568b2e88f3863"
)
PRESERVATION_REPORTING_AUDIT_REL = (
    "results/p08_readiness/preservation_reporting_support_audit.json"
)
PRESERVATION_REPORTING_AUDIT_SHA256 = (
    "afa64083f5cb78a1c6d0f256b40cda8565ce72ed8c86febec59cb4862bc919e0"
)

P01_STATE_FILENAME = "_STATE.json"
P01_ARTIFACT_MANIFEST_FILENAME = "P01_ARTIFACT_HASHES.json"
P01_PRIMARY_MANIFEST_FILENAME = "primary_manifest.csv"
P01_PRESERVATION_METRICS_FILENAME = "preservation_metrics.csv"
P01_PRESERVATION_BY_INSTRUMENT_FILENAME = "preservation_by_instrument.csv"
REPRESENTATIONS_DIRNAME = "representations"
REPRESENTATIONS_SOURCE_REL = "src/atlas_sers/preprocessing/representations.py"

# --- frozen support constants (monkeypatchable for synthetic fixtures only) --
EXPECTED_PRIMARY_SPECTRA = 598
EXPECTED_PRIMARY_MASTERS = 69
EXPECTED_PRIMARY_INSTRUMENTS = 10
EXPECTED_PRIMARY_DOMAINS = 17
EXPECTED_HELD_DOMAINS = 13
EXPECTED_EXPLORATORY_DOMAINS = 4
EXPECTED_PUBLIC_CELLS = 49
EXPECTED_ELIGIBLE_CELLS = 46
EXPECTED_UNAVAILABLE_CELLS = 3
EXPECTED_PUBLIC_CURVES = 138
EXPECTED_HISTORICAL_RECORDS = 4784
EXPECTED_HISTORICAL_REPRESENTATIONS = 8
EXPECTED_PRIMARY_ALIAS_RECORDS = 1794
EXPECTED_DOMAIN_ACTION_GROUPS = 51
EXPECTED_FEATURES = 1401
AXIS_START = 400
AXIS_STOP = 1801

ACTION_ORDER = ("R_MIN_400_1800", "R_SG_400_1800", "R_ARPLS_400_1800")
METRIC_COLUMNS = (
    "baseline_span",
    "candidate_peak_count",
    "changed_point_fraction",
    "clipped_fraction",
    "first_difference_roughness",
    "median_peak_displacement_cm1",
    "rank_correlation",
    "reference_peak_count",
    "shape_correlation",
    "spectral_angle_radians",
    "top_peak_recall_pm5cm1",
)
MANIFEST_COLUMNS = (
    "observation_uid",
    "master_sample_id",
    "station",
    "instrument",
    "sensor_family",
    "target_analyte",
)
PRESERVATION_COLUMNS = (
    "station",
    "instrument",
    "representation_id",
    "metric",
    "n_spectra",
    "n_masters",
    "held_comparison_domain",
    "finite_count",
    "undefined_count",
    "median",
    "q10",
    "q90",
)
SPECTRAL_SCHEMA_VERSION = "nato-sers-p08-f01-spectral-bundle-v1"
SPECTRAL_CAPTION = (
    "P08-F01 public spectral display. The first trace is the MIN representation, "
    "not raw instrument counts. Curves are master-equal descriptive averages of "
    "stored measurements, with each stored observation counted once; they are not "
    "the model's averaged inputs and not individual-master traces. Differences "
    "among representations illustrate whole-pipeline changes and do not establish "
    "chemical peak preservation. Cells in exploratory domains are labelled and do "
    "not enter the primary held comparison; unavailable cells are retained with a "
    "reason. No confidence interval, error bar or significance claim is made."
)
NPZ_KEYS = frozenset({"axis_cm1", "intensity", "observation_uid"})


class PreservationInputError(RuntimeError):
    """Fail-closed loader error carrying a fixed, path-free reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class PreservationSemanticError(ValueError):
    """Fail-closed semantic validation error carrying a reason code."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def _call_check(check: Callable[..., None], stage: str) -> None:
    """Invoke the external gate, tolerating zero-argument callables."""

    if not callable(check):
        raise PreservationInputError("check_required")
    try:
        signature = inspect.signature(check)
    except (TypeError, ValueError):
        check(stage)
        return
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            check(stage)
            return
        if (
            parameter.kind == inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
        ):
            check(stage)
            return
    check()


def _require_hash(input_hashes: dict[str, Any], key: str) -> str:
    value = input_hashes.get(key)
    if not isinstance(value, str) or not value:
        raise PreservationInputError(f"input_hash_missing:{key}")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:
        raise PreservationInputError("csv_read_failure") from None


def _state_files(state: Any) -> dict[str, str]:
    if not isinstance(state, dict):
        raise PreservationInputError("state_not_a_mapping")
    files = state.get("files")
    if not isinstance(files, dict) or not files:
        raise PreservationInputError("state_files_missing")
    resolved: dict[str, str] = {}
    for key, value in files.items():
        if not isinstance(value, str) or not value:
            raise PreservationInputError("state_file_hash_invalid")
        resolved[str(key)] = value
    return resolved


def _artifact_files(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise PreservationInputError("artifact_manifest_not_a_mapping")
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise PreservationInputError("artifact_manifest_files_missing")
    resolved: dict[str, dict[str, Any]] = {}
    for key, value in files.items():
        if not isinstance(value, dict):
            raise PreservationInputError("artifact_manifest_entry_invalid")
        resolved[str(key)] = value
    return resolved


def _lookup_entry(entries: dict[str, Any], name: str) -> Any:
    if name in entries:
        return entries[name]
    for key, value in entries.items():
        if Path(str(key)).name == name:
            return value
    return None


def _verify_state_entry(entries: dict[str, str], name: str, path: Path, source: str) -> None:
    entry = _lookup_entry(entries, name)
    if not isinstance(entry, str) or not entry:
        raise PreservationInputError(f"{source}_entry_missing")
    if entry != _stream_sha256(path):
        raise PreservationInputError(f"{source}_sha256_mismatch")


def _verify_manifest_entry(
    entries: dict[str, Any], name: str, path: Path, source: str
) -> None:
    entry = _lookup_entry(entries, name)
    if not isinstance(entry, dict):
        raise PreservationInputError(f"{source}_entry_missing")
    if entry.get("sha256") != _stream_sha256(path):
        raise PreservationInputError(f"{source}_sha256_mismatch")
    size_value = entry.get("size_bytes")
    if isinstance(size_value, bool):
        raise PreservationInputError(f"{source}_size_missing")
    try:
        size = int(size_value)
    except (TypeError, ValueError):
        raise PreservationInputError(f"{source}_size_missing") from None
    if size != path.stat().st_size:
        raise PreservationInputError(f"{source}_size_mismatch")


def _action_index(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    actions = audit.get("actions")
    if not isinstance(actions, list):
        raise PreservationInputError("input_audit_actions_missing")
    index: dict[str, dict[str, Any]] = {}
    for entry in actions:
        if not isinstance(entry, dict):
            continue
        representation = entry.get("representation_id")
        if representation is None:
            continue
        index[str(representation)] = entry
    return index


def _expected_axis() -> np.ndarray:
    return np.arange(AXIS_START, AXIS_STOP, dtype=np.float32)


def _load_representation_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if frozenset(archive.files) != NPZ_KEYS:
                raise PreservationInputError("npz_keys_invalid")
            if archive["axis_cm1"].dtype != np.dtype("float32"):
                raise PreservationInputError("npz_axis_dtype_invalid")
            if archive["intensity"].dtype != np.dtype("float32"):
                raise PreservationInputError("npz_intensity_dtype_invalid")
            if archive["observation_uid"].dtype.kind != "U":
                raise PreservationInputError("npz_uid_dtype_invalid")
            axis = np.array(archive["axis_cm1"], dtype=np.float32, copy=True)
            intensity = np.array(archive["intensity"], dtype=np.float32, copy=True)
            uid = np.array(archive["observation_uid"], copy=True)
    except PreservationInputError:
        raise
    except Exception:
        raise PreservationInputError("npz_read_failure") from None
    if axis.shape != (EXPECTED_FEATURES,):
        raise PreservationInputError("npz_axis_shape_invalid")
    if intensity.shape != (EXPECTED_PRIMARY_SPECTRA, EXPECTED_FEATURES):
        raise PreservationInputError("npz_intensity_shape_invalid")
    if uid.shape != (EXPECTED_PRIMARY_SPECTRA,):
        raise PreservationInputError("npz_uid_shape_invalid")
    if not np.array_equal(axis, _expected_axis()):
        raise PreservationInputError("npz_axis_values_invalid")
    if not np.isfinite(intensity).all():
        raise PreservationInputError("npz_intensity_nonfinite")
    if (intensity < 0.0).any() or (intensity > 1.0).any():
        raise PreservationInputError("npz_intensity_range_invalid")
    axis.setflags(write=False)
    intensity.setflags(write=False)
    uid.setflags(write=False)
    return {"axis_cm1": axis, "intensity": intensity, "observation_uid": uid}


def load_preservation_inputs(
    *,
    p01_root: Any,
    private_root: Any,
    catalog_path: Any,
    package_root: Any,
    allowed_evidence_roots: Any,
    check: Callable[..., None],
) -> dict[str, Any]:
    """Authenticate and load frozen P01 preservation inputs read-only."""

    if not callable(check):
        raise PreservationInputError("check_required")
    _call_check(check, "start")
    roots = _prepare_roots(allowed_evidence_roots)
    p01_root = Path(p01_root)
    private_root = Path(private_root)
    package_root = Path(package_root)
    catalog_path = Path(catalog_path)

    _call_check(check, "audits")
    input_audit_path = _authenticated_file(
        package_root / INPUT_SUPPORT_AUDIT_REL, roots, INPUT_SUPPORT_AUDIT_SHA256
    )
    input_audit = _read_json_file(input_audit_path, roots)
    preservation_audit_path = _authenticated_file(
        package_root / PRESERVATION_REPORTING_AUDIT_REL,
        roots,
        PRESERVATION_REPORTING_AUDIT_SHA256,
    )
    preservation_audit = _read_json_file(preservation_audit_path, roots)

    input_hashes = preservation_audit.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise PreservationInputError("preservation_audit_hashes_missing")
    if input_hashes.get("input_audit") != INPUT_SUPPORT_AUDIT_SHA256:
        raise PreservationInputError("preservation_audit_input_audit_mismatch")

    _call_check(check, "p01_state")
    state_sha = _require_hash(input_hashes, "P01_state")
    state_path = _authenticated_file(p01_root / P01_STATE_FILENAME, roots, state_sha)
    state = _read_json_file(state_path, roots)
    if state.get("execution_status") != "complete":
        raise PreservationInputError("p01_execution_status_invalid")
    if state.get("scientific_status") != "pass":
        raise PreservationInputError("p01_scientific_status_invalid")

    _call_check(check, "artifact_manifest")
    artifact_sha = _require_hash(input_hashes, "P01_artifact_manifest")
    artifact_path = _authenticated_file(
        p01_root / P01_ARTIFACT_MANIFEST_FILENAME, roots, artifact_sha
    )
    artifact_manifest = _read_json_file(artifact_path, roots)
    state_files = _state_files(state)
    artifact_entries = _artifact_files(artifact_manifest)
    artifact_pin = _lookup_entry(state_files, P01_ARTIFACT_MANIFEST_FILENAME)
    if artifact_pin != artifact_sha:
        raise PreservationInputError("state_artifact_manifest_pin_missing")

    _call_check(check, "primary_manifest")
    manifest_sha = _require_hash(input_hashes, "primary_manifest")
    manifest_path = _authenticated_file(
        p01_root / P01_PRIMARY_MANIFEST_FILENAME, roots, manifest_sha
    )
    manifest = _read_csv(manifest_path)

    _call_check(check, "preservation_metrics")
    metrics_sha = _require_hash(input_hashes, "preservation_metrics")
    metrics_path = _authenticated_file(
        p01_root / P01_PRESERVATION_METRICS_FILENAME, roots, metrics_sha
    )
    metrics = _read_csv(metrics_path)

    _call_check(check, "preservation_by_instrument")
    by_instrument_sha = _require_hash(input_hashes, "preservation_by_instrument")
    by_instrument_path = _authenticated_file(
        p01_root / P01_PRESERVATION_BY_INSTRUMENT_FILENAME, roots, by_instrument_sha
    )
    by_instrument = _read_csv(by_instrument_path)

    _call_check(check, "representations_source")
    source_sha = _require_hash(input_hashes, "preservation_source")
    _authenticated_file(package_root / REPRESENTATIONS_SOURCE_REL, roots, source_sha)

    for name, path in (
        (P01_PRIMARY_MANIFEST_FILENAME, manifest_path),
        (P01_PRESERVATION_METRICS_FILENAME, metrics_path),
        (P01_PRESERVATION_BY_INSTRUMENT_FILENAME, by_instrument_path),
    ):
        _verify_state_entry(state_files, name, path, "state")
        _verify_manifest_entry(artifact_entries, name, path, "artifact_manifest")

    actions = _action_index(input_audit)
    representations: dict[str, dict[str, np.ndarray]] = {}
    for action in ACTION_ORDER:
        entry = actions.get(action)
        if entry is None:
            raise PreservationInputError("input_audit_action_missing")
        archive_sha = entry.get("file_sha256")
        if not isinstance(archive_sha, str) or not archive_sha:
            raise PreservationInputError("input_audit_action_hash_missing")
        _call_check(check, f"npz:{action}")
        npz_path = _authenticated_file(
            p01_root / REPRESENTATIONS_DIRNAME / f"{action}.npz", roots, archive_sha
        )
        representations[action] = _load_representation_npz(npz_path)
        _verify_state_entry(state_files, f"{action}.npz", npz_path, "state")
        _verify_manifest_entry(
            artifact_entries, f"{action}.npz", npz_path, "artifact_manifest"
        )

    _call_check(check, "catalog")
    catalog = _read_json_file(catalog_path, roots)
    canonical = json.dumps(
        catalog, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    catalog_sha = hashlib.sha256(canonical).hexdigest()
    if catalog_sha != preservation_audit.get("private_catalog_sha256"):
        raise PreservationInputError("catalog_canonical_hash_mismatch")

    _call_check(check, "held_contexts")
    context_rel, expected_context_sha = CONTEXT_PINS["contexts"]
    contexts_path = _authenticated_file(
        private_root / context_rel, roots, expected_context_sha
    )
    contexts = pd.read_csv(contexts_path, dtype=str)
    held = _held_contexts(contexts)
    held_domains = sorted(
        {
            (str(station), str(instrument))
            for station, instrument in zip(held["station"], held["held_instrument"], strict=True)
        }
    )
    if len(held_domains) != EXPECTED_HELD_DOMAINS:
        raise PreservationInputError("held_domain_count_mismatch")

    provenance = {
        "input_support_audit_sha256": INPUT_SUPPORT_AUDIT_SHA256,
        "preservation_reporting_support_audit_sha256": PRESERVATION_REPORTING_AUDIT_SHA256,
        "primary_manifest_sha256": manifest_sha,
        "preservation_metrics_sha256": metrics_sha,
        "preservation_by_instrument_sha256": by_instrument_sha,
        "representations_source_sha256": source_sha,
        "private_catalog_sha256": catalog_sha,
        "held_contexts_sha256": expected_context_sha,
        "held_domain_count": len(held_domains),
    }
    return {
        "manifest": manifest,
        "metrics": metrics,
        "preservation_by_instrument": by_instrument,
        "representations": representations,
        "catalog": catalog,
        "held_domains": held_domains,
        "provenance": provenance,
    }


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise PreservationSemanticError(f"catalog_text_invalid:{key}")
    return value


def _require_int(mapping: dict[str, Any], key: str, expected: int) -> None:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise PreservationSemanticError(f"catalog_{key}_invalid")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise PreservationSemanticError(f"catalog_{key}_invalid") from None
    if number != expected:
        raise PreservationSemanticError(f"catalog_{key}_mismatch")


def _validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(manifest, pd.DataFrame):
        raise PreservationSemanticError("manifest_not_a_frame")
    frame = manifest.copy()
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise PreservationSemanticError("manifest_columns_missing")
    for column in MANIFEST_COLUMNS:
        values = frame[column]
        if values.isna().any() or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in values
        ):
            raise PreservationSemanticError("manifest_identity_malformed")
    if len(frame) != EXPECTED_PRIMARY_SPECTRA:
        raise PreservationSemanticError("manifest_row_count")
    if frame.observation_uid.nunique() != EXPECTED_PRIMARY_SPECTRA:
        raise PreservationSemanticError("manifest_observation_duplicate")
    if frame.master_sample_id.nunique() != EXPECTED_PRIMARY_MASTERS:
        raise PreservationSemanticError("manifest_master_count")
    if frame.instrument.nunique() != EXPECTED_PRIMARY_INSTRUMENTS:
        raise PreservationSemanticError("manifest_instrument_count")
    if frame.groupby(["station", "instrument"]).ngroups != EXPECTED_PRIMARY_DOMAINS:
        raise PreservationSemanticError("manifest_domain_count")
    return frame


def _manifest_index(manifest: pd.DataFrame) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in manifest.itertuples(index=False):
        index[str(row.observation_uid)] = {
            "master_sample_id": str(row.master_sample_id),
            "station": str(row.station),
            "instrument": str(row.instrument),
            "sensor_family": str(row.sensor_family),
            "target_analyte": str(row.target_analyte),
        }
    return index


def _validate_catalog_cells(
    catalog: dict[str, Any], manifest_index: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    raw_cells = catalog.get("spectral_figure_cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != EXPECTED_PUBLIC_CELLS:
        raise PreservationSemanticError("catalog_cell_count")
    records: list[dict[str, Any]] = []
    seen_cells: set[tuple[str, str, str]] = set()
    covered_uids: list[str] = []
    for raw in raw_cells:
        if not isinstance(raw, dict):
            raise PreservationSemanticError("catalog_cell_malformed")
        station = _required_text(raw, "station")
        instrument = _required_text(raw, "instrument")
        analyte = _required_text(raw, "analyte")
        cell_key = (station, instrument, analyte)
        if cell_key in seen_cells:
            raise PreservationSemanticError("catalog_cell_key_duplicate")
        seen_cells.add(cell_key)
        groups = raw.get("master_groups")
        if not isinstance(groups, list) or not groups:
            raise PreservationSemanticError("catalog_master_groups_missing")
        seen_masters: set[str] = set()
        master_groups: list[dict[str, Any]] = []
        cell_uids: list[str] = []
        for group in groups:
            if not isinstance(group, dict):
                raise PreservationSemanticError("catalog_master_group_malformed")
            master_id = _required_text(group, "master_id")
            if master_id in seen_masters:
                raise PreservationSemanticError("catalog_master_duplicate")
            seen_masters.add(master_id)
            uids = group.get("observation_uids")
            if not isinstance(uids, list) or not uids:
                raise PreservationSemanticError("catalog_master_uids_missing")
            resolved: list[str] = []
            for value in uids:
                uid = str(value)
                info = manifest_index.get(uid)
                if info is None:
                    raise PreservationSemanticError("catalog_unknown_observation")
                if (
                    info["master_sample_id"] != master_id
                    or info["station"] != station
                    or info["instrument"] != instrument
                    or info["target_analyte"] != analyte
                ):
                    raise PreservationSemanticError("catalog_membership_mismatch")
                resolved.append(uid)
                cell_uids.append(uid)
            if len(set(resolved)) != len(resolved):
                raise PreservationSemanticError("catalog_master_uid_duplicate")
            master_groups.append(
                {"master_id": master_id, "observation_uids": tuple(resolved)}
            )
        if len(set(cell_uids)) != len(cell_uids):
            raise PreservationSemanticError("catalog_cell_uid_duplicate")
        covered_uids.extend(cell_uids)
        n_masters = len(master_groups)
        n_spectra = len(cell_uids)
        _require_int(raw, "n_masters", n_masters)
        _require_int(raw, "n_spectra", n_spectra)
        eligible = raw.get("public_spectral_aggregate_eligible")
        if not isinstance(eligible, bool):
            raise PreservationSemanticError("catalog_eligibility_invalid")
        if eligible != (n_masters >= 2):
            raise PreservationSemanticError("catalog_eligibility_inconsistent")
        records.append(
            {
                "station": station,
                "instrument": instrument,
                "analyte": analyte,
                "master_groups": master_groups,
                "n_masters": n_masters,
                "n_spectra": n_spectra,
                "eligible": eligible,
            }
        )
    if len(set(covered_uids)) != len(covered_uids):
        raise PreservationSemanticError("catalog_cell_uid_duplicate")
    if set(covered_uids) != set(manifest_index):
        raise PreservationSemanticError("catalog_cell_coverage")
    records.sort(key=lambda item: (item["station"], item["instrument"], item["analyte"]))
    return records


def _validate_domain_memberships(
    catalog: dict[str, Any], manifest_index: dict[str, dict[str, str]]
) -> set[str]:
    memberships = catalog.get("domain_memberships")
    if not isinstance(memberships, list) or not memberships:
        raise PreservationSemanticError("catalog_domain_memberships_missing")
    covered: set[str] = set()
    seen_domains: set[tuple[str, str]] = set()
    for membership in memberships:
        if not isinstance(membership, dict):
            raise PreservationSemanticError("catalog_domain_membership_malformed")
        station = _required_text(membership, "station")
        instrument = _required_text(membership, "instrument")
        domain_key = (station, instrument)
        if domain_key in seen_domains:
            raise PreservationSemanticError("catalog_domain_key_duplicate")
        seen_domains.add(domain_key)
        uids = membership.get("observation_uids")
        if not isinstance(uids, list) or not uids:
            raise PreservationSemanticError("catalog_domain_uids_missing")
        for value in uids:
            uid = str(value)
            info = manifest_index.get(uid)
            if (
                info is None
                or info["station"] != station
                or info["instrument"] != instrument
            ):
                raise PreservationSemanticError("catalog_domain_membership_mismatch")
            if uid in covered:
                raise PreservationSemanticError("catalog_domain_uid_duplicate")
            covered.add(uid)
    if len(seen_domains) != EXPECTED_PRIMARY_DOMAINS:
        raise PreservationSemanticError("catalog_domain_count")
    if covered != set(manifest_index):
        raise PreservationSemanticError("catalog_domain_coverage")
    return covered


def _validate_selected_rows(
    catalog: dict[str, Any], metrics: pd.DataFrame, manifest: pd.DataFrame
) -> None:
    selected = catalog.get("selected_preservation_rows")
    if not isinstance(selected, dict) or not selected:
        raise PreservationSemanticError("catalog_selected_rows_missing")
    if set(selected) != set(ACTION_ORDER):
        raise PreservationSemanticError("catalog_selected_rows_coverage")
    order = manifest.observation_uid.astype(str).tolist()
    for uids in selected.values():
        if not isinstance(uids, list):
            raise PreservationSemanticError("catalog_selected_rows_malformed")
        if [str(uid) for uid in uids] != order:
            raise PreservationSemanticError("catalog_selected_rows_order")


def _validate_representations(
    representations: dict[str, Any], manifest: pd.DataFrame
) -> None:
    if set(representations) != set(ACTION_ORDER):
        raise PreservationSemanticError("representation_actions_invalid")
    expected_uids = manifest.observation_uid.astype(str).tolist()
    expected_axis = _expected_axis()
    for action in ACTION_ORDER:
        arrays = representations[action]
        if not isinstance(arrays, dict) or set(arrays) != set(NPZ_KEYS):
            raise PreservationSemanticError("representation_keys_invalid")
        axis = np.asarray(arrays["axis_cm1"])
        intensity = np.asarray(arrays["intensity"])
        uid = np.asarray(arrays["observation_uid"])
        if axis.dtype != np.dtype("float32"):
            raise PreservationSemanticError("representation_axis_dtype_invalid")
        if intensity.dtype != np.dtype("float32"):
            raise PreservationSemanticError("representation_intensity_dtype_invalid")
        if uid.dtype.kind != "U":
            raise PreservationSemanticError("representation_uid_dtype_invalid")
        if axis.shape != (EXPECTED_FEATURES,) or intensity.shape != (
            EXPECTED_PRIMARY_SPECTRA,
            EXPECTED_FEATURES,
        ) or uid.shape != (EXPECTED_PRIMARY_SPECTRA,):
            raise PreservationSemanticError("representation_shape_invalid")
        if not np.array_equal(axis, expected_axis):
            raise PreservationSemanticError("representation_axis_invalid")
        if [str(value) for value in uid] != expected_uids:
            raise PreservationSemanticError("representation_uid_order")
        if not np.isfinite(intensity).all():
            raise PreservationSemanticError("representation_intensity_nonfinite")
        if (intensity < 0.0).any() or (intensity > 1.0).any():
            raise PreservationSemanticError("representation_intensity_range")


def _convert_metric_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    for column in METRIC_COLUMNS:
        series = metrics[column]
        missing = series.isna()
        text = series.astype(str).str.strip()
        missing = missing | text.eq("") | text.str.lower().isin({"nan", "none", "null"})
        try:
            numeric = pd.to_numeric(text.where(~missing), errors="raise")
        except (TypeError, ValueError):
            raise PreservationSemanticError("metric_numeric_conversion") from None
        metrics[column] = numeric.astype(float)
    return metrics


def _validate_historical_metrics(
    metrics: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    if not isinstance(metrics, pd.DataFrame):
        raise PreservationSemanticError("metrics_not_a_frame")
    frame = metrics.copy()
    required = (
        "representation_id",
        "observation_uid",
        "station",
        "instrument",
        "sensor_family",
        *METRIC_COLUMNS,
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise PreservationSemanticError("metrics_columns_missing")
    if len(frame) != EXPECTED_HISTORICAL_RECORDS:
        raise PreservationSemanticError("metrics_row_count")
    if frame.representation_id.nunique() != EXPECTED_HISTORICAL_REPRESENTATIONS:
        raise PreservationSemanticError("metrics_representation_count")
    if frame.duplicated(["representation_id", "observation_uid"]).any():
        raise PreservationSemanticError("metrics_duplicate_record")
    manifest_uids = set(manifest.observation_uid.astype(str))
    if not set(frame.observation_uid.astype(str)) <= manifest_uids:
        raise PreservationSemanticError("metrics_unknown_observation")
    meta = manifest.set_index("observation_uid")
    for _representation, cell in frame.groupby("representation_id", sort=True):
        if set(cell.observation_uid.astype(str)) != manifest_uids:
            raise PreservationSemanticError("metrics_representation_incomplete")
    for column in ("station", "instrument", "sensor_family"):
        expected = frame.observation_uid.map(meta[column]).astype(str)
        if not frame[column].astype(str).eq(expected).all():
            raise PreservationSemanticError("metrics_metadata_mismatch")
    return _convert_metric_columns(frame)


def _cell_curves(
    record: dict[str, Any],
    representations: dict[str, Any],
    uid_positions: dict[str, int],
) -> dict[str, list[float]]:
    curves: dict[str, list[float]] = {}
    for action in ACTION_ORDER:
        intensity = np.asarray(representations[action]["intensity"], dtype=np.float64)
        master_curves = []
        for group in record["master_groups"]:
            positions = [uid_positions[uid] for uid in group["observation_uids"]]
            master_curves.append(intensity[positions].mean(axis=0))
        cell_curve = np.mean(np.vstack(master_curves), axis=0)
        curves[action] = [float(value) for value in cell_curve]
    return curves


def _public_cell(
    record: dict[str, Any],
    index: int,
    held_domains: set[tuple[str, str]],
    curves: dict[str, list[float]],
) -> dict[str, Any]:
    available = bool(record["eligible"])
    if available:
        reason = ""
    elif record["n_masters"] < 2:
        reason = "fewer_than_two_physical_masters"
    else:
        reason = "not_eligible_for_public_spectral_aggregate"
    return {
        "cell_id": f"P08-F01-C{index:03d}",
        "station": record["station"],
        "instrument": record["instrument"],
        "analyte": record["analyte"],
        "n_spectra": record["n_spectra"],
        "n_masters": record["n_masters"],
        "held_comparison_domain": (
            (record["station"], record["instrument"]) in held_domains
        ),
        "available": available,
        "reason": reason,
        "curves": curves,
    }


def _preservation_table(
    manifest: pd.DataFrame,
    primary: pd.DataFrame,
    held_domains: set[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (station, instrument), group in manifest.groupby(
        ["station", "instrument"], sort=True
    ):
        n_masters = int(group.master_sample_id.nunique())
        held = (str(station), str(instrument)) in held_domains
        for action in ACTION_ORDER:
            subset = primary[
                primary.representation_id.eq(action)
                & primary.station.eq(station)
                & primary.instrument.eq(instrument)
            ]
            n_spectra = int(len(subset))
            for metric in METRIC_COLUMNS:
                values = subset[metric].to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                finite_count = int(finite.size)
                undefined_count = int(n_spectra - finite_count)
                if finite.size:
                    median = float(np.median(finite))
                    q10, q90 = (
                        float(value)
                        for value in np.quantile(finite, [0.1, 0.9], method="linear")
                    )
                else:
                    median = q10 = q90 = float("nan")
                rows.append(
                    {
                        "station": str(station),
                        "instrument": str(instrument),
                        "representation_id": action,
                        "metric": metric,
                        "n_spectra": n_spectra,
                        "n_masters": n_masters,
                        "held_comparison_domain": held,
                        "finite_count": finite_count,
                        "undefined_count": undefined_count,
                        "median": median,
                        "q10": q10,
                        "q90": q90,
                    }
                )
    return pd.DataFrame(rows, columns=list(PRESERVATION_COLUMNS))


def build_preservation_semantics(
    *,
    manifest: pd.DataFrame,
    metrics: pd.DataFrame,
    representations: dict[str, Any],
    catalog: dict[str, Any],
    held_domains: Any,
) -> dict[str, Any]:
    """Build F01 spectral and F07 preservation public semantics without I/O."""

    manifest_frame = _validate_manifest(manifest)
    manifest_index = _manifest_index(manifest_frame)
    held_set = {(str(station), str(instrument)) for station, instrument in held_domains}

    domain_count = int(manifest_frame.groupby(["station", "instrument"]).ngroups)
    if len(held_set) != EXPECTED_HELD_DOMAINS:
        raise PreservationSemanticError("held_domain_count")
    domain_keys = {
        (str(station), str(instrument))
        for station, instrument in zip(
            manifest_frame.station, manifest_frame.instrument, strict=True
        )
    }
    if not held_set <= domain_keys:
        raise PreservationSemanticError("held_domain_not_in_manifest")
    if domain_count - len(held_set) != EXPECTED_EXPLORATORY_DOMAINS:
        raise PreservationSemanticError("exploratory_domain_count")

    metrics_frame = _validate_historical_metrics(metrics, manifest_frame)
    _validate_selected_rows(catalog, metrics_frame, manifest_frame)
    covered = _validate_domain_memberships(catalog, manifest_index)
    if covered != set(manifest_frame.observation_uid.astype(str)):
        raise PreservationSemanticError("catalog_domain_coverage")
    records = _validate_catalog_cells(catalog, manifest_index)
    _validate_representations(representations, manifest_frame)

    primary = metrics_frame[metrics_frame.representation_id.isin(ACTION_ORDER)]
    if len(primary) != EXPECTED_PRIMARY_ALIAS_RECORDS:
        raise PreservationSemanticError("primary_alias_record_count")

    eligible_count = sum(1 for record in records if record["eligible"])
    if eligible_count != EXPECTED_ELIGIBLE_CELLS:
        raise PreservationSemanticError("eligible_cell_count")
    if len(records) - eligible_count != EXPECTED_UNAVAILABLE_CELLS:
        raise PreservationSemanticError("unavailable_cell_count")

    uid_positions = {
        uid: index
        for index, uid in enumerate(manifest_frame.observation_uid.astype(str))
    }
    cells: list[dict[str, Any]] = []
    for position, record in enumerate(records, start=1):
        curves = (
            _cell_curves(record, representations, uid_positions)
            if record["eligible"]
            else {}
        )
        cells.append(_public_cell(record, position, held_set, curves))

    public_curves = sum(len(cell["curves"]) for cell in cells)
    if public_curves != EXPECTED_PUBLIC_CURVES:
        raise PreservationSemanticError("public_curve_count")

    spectral = {
        "schema_version": SPECTRAL_SCHEMA_VERSION,
        "figure_id": "P08-F01",
        "axis_cm1": [float(value) for value in _expected_axis()],
        "action_order": list(ACTION_ORDER),
        "cells": cells,
        "caption": SPECTRAL_CAPTION,
    }

    preservation = _preservation_table(manifest_frame, primary, held_set)
    if len(preservation) // len(METRIC_COLUMNS) != EXPECTED_DOMAIN_ACTION_GROUPS:
        raise PreservationSemanticError("domain_action_group_count")

    support = {
        "historical_records": int(len(metrics_frame)),
        "primary_alias_records": int(len(primary)),
        "domain_action_groups": int(len(preservation) // len(METRIC_COLUMNS)),
        "public_cells": len(cells),
        "public_cells_eligible": eligible_count,
        "public_cells_unavailable": len(cells) - eligible_count,
        "public_curves": public_curves,
        "primary_spectra": int(len(manifest_frame)),
        "primary_masters": int(manifest_frame.master_sample_id.nunique()),
        "primary_instruments": int(manifest_frame.instrument.nunique()),
        "primary_domains": domain_count,
        "held_domains": len(held_set),
        "exploratory_domains": domain_count - len(held_set),
    }
    return {
        "spectral": spectral,
        "preservation": preservation,
        "support": support,
    }
