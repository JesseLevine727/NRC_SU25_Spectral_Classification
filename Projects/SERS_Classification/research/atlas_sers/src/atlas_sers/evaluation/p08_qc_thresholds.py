"""P08-T026 source-QC threshold kernel: synthetic leaf.

No private-data or scientific-run authority.
"""

import copy
import hashlib

import numpy as np

from .p08_qc_blocks import canonical_sha256

INGREDIENTS = (
    "first_difference_noise_mad",
    "intensity_range",
    "spike_fraction_proxy",
    "baseline_energy_fraction_proxy",
    "baseline_span_fraction_proxy",
    "negative_fraction",
)
FEATURES = (
    "first_difference_noise_mad_over_intensity_range",
    "spike_fraction_proxy",
    "baseline_energy_fraction_proxy",
    "baseline_span_fraction_proxy",
    "negative_fraction",
)
QUANTILES = (0.5, 0.75, 0.9)
SCHEMA_VERSION = "nato-sers-p08-qc-threshold-state-v1"
_FALLBACK = "invalid_qc_threshold_input"
_REASON_CODES = frozenset(
    {
        _FALLBACK,
        "invalid_qc_matrix",
        "invalid_source_uids",
        "fit_uid_mismatch",
        "nonfinite_source_quantiles",
        "invalid_threshold_state",
        "scientific_execution_not_authorized",
    }
)
_STATE_KEYS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "fit_uid_sha256",
        "qc_input_sha256",
        "source_row_count",
        "valid_source_row_count",
        "features",
        "quantiles",
        "cutpoints",
        "status",
        "state_sha256",
    }
)


class QCThresholdError(ValueError):
    """Fixed, data-free error with a whitelisted ``reason_code``."""

    def __init__(self, reason_code):
        code = (
            reason_code if type(reason_code) is str and reason_code in _REASON_CODES else _FALLBACK
        )
        self.reason_code = code
        super().__init__(code)


def require_scientific_execution(*args, **kwargs):
    """Always refuse scientific execution from this leaf kernel."""
    raise QCThresholdError("scientific_execution_not_authorized")


def _is_lower64(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _normalize_qc(qc):
    if type(qc) is not np.ndarray:
        raise QCThresholdError("invalid_qc_matrix")
    if qc.ndim != 2 or qc.shape[1] != len(INGREDIENTS):
        raise QCThresholdError("invalid_qc_matrix")
    if qc.dtype.kind not in ("i", "u", "f"):
        raise QCThresholdError("invalid_qc_matrix")
    with np.errstate(all="ignore"):
        return np.array(qc, dtype="<f8", order="C", copy=True)


def _derive_from_normalized(normalized):
    n = normalized.shape[0]
    intensity = normalized[:, 1]
    features = np.full((n, len(FEATURES)), np.nan, dtype=np.float64)
    with np.errstate(all="ignore"):
        features[:, 0] = normalized[:, 0] / intensity
    features[:, 1:5] = normalized[:, 2:6]
    ok_ingredients = np.all(np.isfinite(normalized), axis=1)
    ok_features = np.all(np.isfinite(features), axis=1)
    valid = ok_ingredients & (intensity > 0.0) & ok_features
    features[~valid] = np.nan
    return features, valid.astype(bool)


def derive_qc_features(qc):
    try:
        return _derive_from_normalized(_normalize_qc(qc))
    except QCThresholdError:
        raise
    except Exception:
        raise QCThresholdError(_FALLBACK) from None


def _validate_source_uids(source_uids, n):
    if type(source_uids) not in (list, tuple):
        raise QCThresholdError("invalid_source_uids")
    uids, seen = [], set()
    for uid in source_uids:
        if type(uid) is not str or not uid or uid != uid.strip():
            raise QCThresholdError("invalid_source_uids")
        try:
            uid.encode("utf-8")
        except Exception:
            raise QCThresholdError("invalid_source_uids") from None
        if uid in seen:
            raise QCThresholdError("invalid_source_uids")
        seen.add(uid)
        uids.append(uid)
    if len(uids) != n:
        raise QCThresholdError("invalid_source_uids")
    return uids


def _source_cutpoints(valid_rows):
    with np.errstate(all="ignore"):
        cuts = np.quantile(valid_rows, QUANTILES, axis=0, method="linear").T
    if not np.all(np.isfinite(cuts)):
        raise QCThresholdError("nonfinite_source_quantiles")
    for index in range(len(QUANTILES) - 1):
        if np.any(cuts[:, index] > cuts[:, index + 1]):
            raise QCThresholdError("nonfinite_source_quantiles")
    return cuts


def fit_source_thresholds(qc, source_uids, *, expected_fit_uid_sha256):
    try:
        if not _is_lower64(expected_fit_uid_sha256):
            raise QCThresholdError("invalid_threshold_state")
        normalized = _normalize_qc(qc)
        uids = _validate_source_uids(source_uids, normalized.shape[0])
        order = sorted(range(len(uids)), key=lambda i: uids[i])
        sorted_norm = np.ascontiguousarray(normalized[order], dtype="<f8")
        uid_sha = canonical_sha256([uids[i] for i in order])
        if uid_sha != expected_fit_uid_sha256:
            raise QCThresholdError("fit_uid_mismatch")
        values_sha = hashlib.sha256(sorted_norm.tobytes(order="C")).hexdigest()
        qc_sha = canonical_sha256(
            {
                "ingredients": list(INGREDIENTS),
                "source_uid_sha256": uid_sha,
                "shape": [int(sorted_norm.shape[0]), len(INGREDIENTS)],
                "dtype": "<f8",
                "values_sha256": values_sha,
            }
        )
        features, valid = _derive_from_normalized(sorted_norm)
        source_rows = int(sorted_norm.shape[0])
        valid_rows = int(np.count_nonzero(valid))
        if valid_rows == 0:
            cutpoints, status = None, "no_finite_source_qc"
        else:
            cuts = _source_cutpoints(features[valid])
            cutpoints = [[float(v) for v in row] for row in cuts]
            status = "finite_source_qc"
        state = {
            "schema_version": SCHEMA_VERSION,
            "execution_authorized": False,
            "fit_uid_sha256": uid_sha,
            "qc_input_sha256": qc_sha,
            "source_row_count": source_rows,
            "valid_source_row_count": valid_rows,
            "features": list(FEATURES),
            "quantiles": list(QUANTILES),
            "cutpoints": cutpoints,
            "status": status,
        }
        state["state_sha256"] = canonical_sha256(state)
        return state
    except QCThresholdError:
        raise
    except Exception:
        raise QCThresholdError(_FALLBACK) from None


def _validate_cutpoints(cuts):
    if type(cuts) is not list or len(cuts) != len(FEATURES):
        raise QCThresholdError("invalid_threshold_state")
    for row in cuts:
        if type(row) is not list or len(row) != len(QUANTILES):
            raise QCThresholdError("invalid_threshold_state")
        for value in row:
            if type(value) not in (int, float):
                raise QCThresholdError("invalid_threshold_state")
            number = float(value)
            if not np.isfinite(number):
                raise QCThresholdError("invalid_threshold_state")
        for index in range(len(row) - 1):
            if row[index + 1] < row[index]:
                raise QCThresholdError("invalid_threshold_state")


def validate_threshold_state(state, *, expected_fit_uid_sha256=None, expected_qc_input_sha256=None):
    try:
        if type(state) is not dict or set(state.keys()) != _STATE_KEYS:
            raise QCThresholdError("invalid_threshold_state")
        for expected in (expected_fit_uid_sha256, expected_qc_input_sha256):
            if expected is not None and not _is_lower64(expected):
                raise QCThresholdError("invalid_threshold_state")
        sealed = state["state_sha256"]
        if not _is_lower64(sealed):
            raise QCThresholdError("invalid_threshold_state")
        payload = {key: state[key] for key in state if key != "state_sha256"}
        if canonical_sha256(payload) != sealed:
            raise QCThresholdError("invalid_threshold_state")
        if state["schema_version"] != SCHEMA_VERSION:
            raise QCThresholdError("invalid_threshold_state")
        if state["execution_authorized"] is not False:
            raise QCThresholdError("invalid_threshold_state")
        for key in ("fit_uid_sha256", "qc_input_sha256"):
            if not _is_lower64(state[key]):
                raise QCThresholdError("invalid_threshold_state")
        source_rows = state["source_row_count"]
        valid_rows = state["valid_source_row_count"]
        if type(source_rows) is not int or type(valid_rows) is not int:
            raise QCThresholdError("invalid_threshold_state")
        if source_rows < 0 or valid_rows < 0 or valid_rows > source_rows:
            raise QCThresholdError("invalid_threshold_state")
        if state["features"] != list(FEATURES):
            raise QCThresholdError("invalid_threshold_state")
        if state["quantiles"] != list(QUANTILES):
            raise QCThresholdError("invalid_threshold_state")
        status, cutpoints = state["status"], state["cutpoints"]
        if status == "no_finite_source_qc":
            if cutpoints is not None or valid_rows != 0:
                raise QCThresholdError("invalid_threshold_state")
        elif status == "finite_source_qc":
            if valid_rows <= 0:
                raise QCThresholdError("invalid_threshold_state")
            _validate_cutpoints(cutpoints)
        else:
            raise QCThresholdError("invalid_threshold_state")
        for supplied, key in (
            (expected_fit_uid_sha256, "fit_uid_sha256"),
            (expected_qc_input_sha256, "qc_input_sha256"),
        ):
            if supplied is not None and state[key] != supplied:
                raise QCThresholdError("invalid_threshold_state")
        return copy.deepcopy(state)
    except QCThresholdError:
        raise
    except Exception:
        raise QCThresholdError(_FALLBACK) from None
