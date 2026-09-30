"""Synthetic-only contract tests for the P08 single-row QC router.

No real data and no filesystem access. This module has no scientific-run
authority: the synthetic numeric code below executes locally, but the router
must keep ``execution_authorized`` false. Threshold state is built exclusively
from invented arrays through the reviewed threshold module. QCRoutingError is a
ValueError subclass carrying ``reason_code``.
"""

import copy
import hashlib
import json
import math

import numpy as np
import pytest

from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256
from atlas_sers.evaluation.p08_qc_routing import (
    ACTION_IDS,
    QCRoutingError,
    require_scientific_execution,
    route_qc_row,
)
from atlas_sers.evaluation.p08_qc_thresholds import (
    FEATURES,
    QUANTILES,
    fit_source_thresholds,
    validate_threshold_state,
)
from atlas_sers.splits.p02 import build_qc_gate_library

MIN, SG, ARPLS = ACTION_IDS
NOISE = tuple(FEATURES[:2])
BASELINE = tuple(FEATURES[2:])
VALID = {MIN: True, SG: True, ARPLS: True}
OUT_KEYS = (
    "schema_version",
    "execution_authorized",
    "gate_candidate_id",
    "gate_definition_sha256",
    "threshold_state_sha256",
    "qc_row_sha256",
    "features",
    "qc_available",
    "noise_triggered",
    "baseline_triggered",
    "requested_action",
    "selected_action",
    "action_validity",
    "fallback_reason",
    "route_sha256",
)

CONTRACT = {
    "qc_gate_enumeration": {
        "source_quantiles": [0.5, 0.75, 0.9],
        "priority_orders": ["baseline_then_noise", "noise_then_baseline"],
    }
}
GATES = build_qc_gate_library(CONTRACT).to_dict("records")

# Invented baseline [v, 1, v, v, v, v] makes every feature equal v. The
# independently known quantiles [0.5, 0.75, 0.9] of [0, 1, 2, 3] are
# [1.5, 2.25, 2.7]; replicate them literally instead of using np.quantile.
_BASE_VALUES = np.array([0.0, 1.0, 2.0, 3.0])
CUTPOINTS = [[1.5, 2.25, 2.7] for _ in FEATURES]
BASELINE_ROWS = np.array([[v, 1.0, v, v, v, v] for v in _BASE_VALUES], dtype=np.float64)
SOURCE_UIDS = [
    "synthetic-source-a",
    "synthetic-source-b",
    "synthetic-source-c",
    "synthetic-source-d",
]


def _sealed_sha(state):
    return state["state_sha256"]


def _reseal(state):
    resealed = copy.deepcopy(state)
    payload = {key: value for key, value in resealed.items() if key != "state_sha256"}
    resealed["state_sha256"] = canonical_sha256(payload)
    return resealed


STATE = fit_source_thresholds(
    BASELINE_ROWS, SOURCE_UIDS, expected_fit_uid_sha256=canonical_sha256(sorted(SOURCE_UIDS))
)
STATE_SHA = _sealed_sha(STATE)
NO_SOURCE_STATE = fit_source_thresholds(
    np.full((4, 6), np.nan, dtype=np.float64),
    SOURCE_UIDS,
    expected_fit_uid_sha256=canonical_sha256(sorted(SOURCE_UIDS)),
)
NO_SOURCE_SHA = _sealed_sha(NO_SOURCE_STATE)


def _row(overrides=None, base=0.0):
    values = {name: base for name in FEATURES}
    values.update(overrides or {})
    f = [values[name] for name in FEATURES]
    return np.array([f[0], 1.0, f[1], f[2], f[3], f[4]], dtype=np.float64)


def _route(row, gate_id, state=STATE, sha=STATE_SHA, validity=VALID):
    return route_qc_row(row, state, gate_id, validity, expected_state_sha256=sha)


def _expect_reason(reason, func, *args, **kwargs):
    with pytest.raises(QCRoutingError) as excinfo:
        func(*args, **kwargs)
    assert excinfo.value.reason_code == reason


def _feat_values(overrides):
    values = {name: 0.0 for name in FEATURES}
    values.update(overrides or {})
    return [values[name] for name in FEATURES]


def _expected(record, feat_values):
    def active(prefix):
        name = record[prefix + "_feature"]
        if name == "none":
            return False
        i = FEATURES.index(name)
        j = QUANTILES.index(record[prefix + "_quantile"])
        return bool(feat_values[i] > CUTPOINTS[i][j])

    noise, baseline = active("noise"), active("baseline")
    if not noise and not baseline:
        action = record["default_action"]
    elif noise and baseline:
        action = (
            record["noise_action"]
            if record["priority_order"] == "noise_then_baseline"
            else record["baseline_action"]
        )
    elif noise:
        action = record["noise_action"]
    else:
        action = record["baseline_action"]
    return noise, baseline, action


def _single_gate(feature, quantile):
    for gate in GATES:
        if gate["gate_kind"] != "single_trigger":
            continue
        if feature in (gate["noise_feature"], gate["baseline_feature"]) and quantile in (
            gate["noise_quantile"],
            gate["baseline_quantile"],
        ):
            return gate
    raise AssertionError("missing single-trigger gate")


SINGLE_NOISE = next(
    g for g in GATES if g["gate_kind"] == "single_trigger" and g["noise_feature"] != "none"
)
SINGLE_BASELINE = next(
    g for g in GATES if g["gate_kind"] == "single_trigger" and g["baseline_feature"] != "none"
)
POWER_GATE = next(
    g
    for g in GATES
    if g["gate_kind"] == "single_trigger"
    and g["noise_feature"] == FEATURES[0]
    and g["noise_quantile"] == 0.75
)


def _dual_pairs():
    pairs = {}
    for gate in GATES:
        if gate["gate_kind"] == "dual_trigger":
            key = (
                gate["noise_feature"],
                gate["noise_quantile"],
                gate["baseline_feature"],
                gate["baseline_quantile"],
            )
            pairs.setdefault(key, {})[gate["priority_order"]] = gate["gate_candidate_id"]
    return pairs


@pytest.mark.parametrize("record", GATES, ids=[g["gate_candidate_id"] for g in GATES])
def test_all_gates_routable(record):
    scenarios = {"neither": {}}
    if record["noise_feature"] != "none":
        scenarios["noise_only"] = {record["noise_feature"]: 10.0}
    if record["baseline_feature"] != "none":
        scenarios["baseline_only"] = {record["baseline_feature"]: 10.0}
    if record["gate_kind"] == "dual_trigger":
        scenarios["both"] = {record["noise_feature"]: 10.0, record["baseline_feature"]: 10.0}
    for overrides in scenarios.values():
        out = _route(_row(overrides), record["gate_candidate_id"])
        exp_noise, exp_baseline, exp_action = _expected(record, _feat_values(overrides))
        assert out["schema_version"] == "nato-sers-p08-qc-route-v1"
        assert out["execution_authorized"] is False
        assert out["gate_candidate_id"] == record["gate_candidate_id"]
        assert out["gate_definition_sha256"] == canonical_sha256(record)
        assert out["threshold_state_sha256"] == STATE_SHA
        assert out["noise_triggered"] is exp_noise
        assert out["baseline_triggered"] is exp_baseline
        assert out["requested_action"] == exp_action
        assert out["selected_action"] == exp_action
        assert out["fallback_reason"] == "none"


def test_dual_priorities_pick_different_actions():
    for key, ids in _dual_pairs().items():
        row = _row({key[0]: 10.0, key[2]: 10.0})
        baseline_first = _route(row, ids["baseline_then_noise"])
        noise_first = _route(row, ids["noise_then_baseline"])
        assert baseline_first["noise_triggered"] is True
        assert baseline_first["baseline_triggered"] is True
        assert baseline_first["selected_action"] == ARPLS
        assert noise_first["selected_action"] == SG


@pytest.mark.parametrize("feature", FEATURES)
@pytest.mark.parametrize("qi", range(len(QUANTILES)))
def test_cutpoint_boundaries(feature, qi):
    record = _single_gate(feature, QUANTILES[qi])
    cutpoint = CUTPOINTS[FEATURES.index(feature)][qi]
    equal = _route(_row({feature: cutpoint}), record["gate_candidate_id"])
    assert equal["noise_triggered"] is False
    assert equal["baseline_triggered"] is False
    assert equal["selected_action"] == MIN
    above = _route(_row({feature: np.nextafter(cutpoint, np.inf)}), record["gate_candidate_id"])
    expected = (
        record["noise_action"] if record["noise_feature"] == feature else record["baseline_action"]
    )
    assert (above["noise_triggered"] or above["baseline_triggered"]) is True
    assert above["selected_action"] == expected


def _power_state():
    state = copy.deepcopy(STATE)
    state["cutpoints"][0] = [2**53 + 1, 2**53 + 3, 2**53 + 5]
    return _reseal(state)


def test_strict_float_int_cutpoint_comparison():
    state = _power_state()
    sha = _sealed_sha(state)
    gate_id = POWER_GATE["gate_candidate_id"]
    trigger = _route(_row({FEATURES[0]: float(2**53 + 4)}), gate_id, state=state, sha=sha)
    assert trigger["noise_triggered"] is True
    assert trigger["requested_action"] == SG
    assert trigger["selected_action"] == SG
    quiet = _route(_row({FEATURES[0]: float(2**53 + 2)}), gate_id, state=state, sha=sha)
    assert quiet["noise_triggered"] is False
    assert quiet["requested_action"] == MIN
    assert quiet["selected_action"] == MIN


def test_no_finite_source_still_reports_valid_row_features():
    out = _route(_row(), "QC-000-MIN", state=NO_SOURCE_STATE, sha=NO_SOURCE_SHA)
    assert out["qc_available"] is True
    assert isinstance(out["features"], list) and len(out["features"]) == 5
    assert all(math.isfinite(x) for x in out["features"])
    assert out["noise_triggered"] is False
    assert out["baseline_triggered"] is False
    assert out["requested_action"] == MIN
    assert out["selected_action"] == MIN
    assert out["fallback_reason"] == "no_finite_source_qc"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.__setitem__(0, np.nan),
        lambda r: r.__setitem__(1, 0.0),
        lambda r: r.__setitem__(1, -2.0),
        lambda r: (r.__setitem__(0, 1e308), r.__setitem__(1, 1e-308)),
    ],
)
def test_invalid_row_is_unavailable(mutate):
    row = _row()
    mutate(row)
    out = _route(row, "QC-000-MIN")
    assert out["qc_available"] is False
    assert out["features"] is None
    assert out["requested_action"] == MIN
    assert out["selected_action"] == MIN
    assert out["fallback_reason"] == "missing_or_nonfinite_qc"
    assert out["noise_triggered"] is False
    assert out["baseline_triggered"] is False


def test_no_finite_source_reason_precedes_row_unavailable():
    row = _row()
    row[0] = np.nan
    out = _route(row, "QC-000-MIN", state=NO_SOURCE_STATE, sha=NO_SOURCE_SHA)
    assert out["qc_available"] is False
    assert out["features"] is None
    assert out["fallback_reason"] == "no_finite_source_qc"


def test_invalid_selected_action_falls_back_to_min_only():
    row = _row({SINGLE_NOISE["noise_feature"]: 10.0})
    out = _route(
        row, SINGLE_NOISE["gate_candidate_id"], validity={MIN: True, SG: False, ARPLS: True}
    )
    assert out["requested_action"] == SG
    assert out["selected_action"] == MIN
    assert out["fallback_reason"] == "selected_action_invalid_for_row"
    row2 = _row({SINGLE_BASELINE["baseline_feature"]: 10.0})
    out2 = _route(
        row2, SINGLE_BASELINE["gate_candidate_id"], validity={MIN: True, SG: True, ARPLS: False}
    )
    assert out2["requested_action"] == ARPLS
    assert out2["selected_action"] == MIN
    assert out2["fallback_reason"] == "selected_action_invalid_for_row"


def test_dual_invalid_action_never_falls_back_to_other_trigger():
    for key, ids in _dual_pairs().items():
        row = _row({key[0]: 10.0, key[2]: 10.0})
        baseline_first = _route(
            row, ids["baseline_then_noise"], validity={MIN: True, SG: True, ARPLS: False}
        )
        assert baseline_first["noise_triggered"] is True
        assert baseline_first["baseline_triggered"] is True
        assert baseline_first["requested_action"] == ARPLS
        assert baseline_first["selected_action"] == MIN
        assert baseline_first["fallback_reason"] == "selected_action_invalid_for_row"
        noise_first = _route(
            row, ids["noise_then_baseline"], validity={MIN: True, SG: False, ARPLS: True}
        )
        assert noise_first["noise_triggered"] is True
        assert noise_first["baseline_triggered"] is True
        assert noise_first["requested_action"] == SG
        assert noise_first["selected_action"] == MIN
        assert noise_first["fallback_reason"] == "selected_action_invalid_for_row"


def test_min_invalid_is_always_fatal():
    bad = {MIN: False, SG: True, ARPLS: True}
    row = _row({SINGLE_NOISE["noise_feature"]: 10.0})
    _expect_reason(
        "invalid_minimal_action",
        route_qc_row,
        row,
        STATE,
        SINGLE_NOISE["gate_candidate_id"],
        bad,
        expected_state_sha256=STATE_SHA,
    )
    _expect_reason(
        "invalid_minimal_action",
        route_qc_row,
        np.full(6, np.nan),
        NO_SOURCE_STATE,
        SINGLE_NOISE["gate_candidate_id"],
        bad,
        expected_state_sha256=NO_SOURCE_SHA,
    )


def test_inputs_unchanged_and_output_fresh():
    row = _row({FEATURES[0]: 2.0})
    before_row = row.copy()
    before_state = repr(STATE)
    validity = {MIN: True, SG: True, ARPLS: True}
    before_validity = dict(validity)
    out = _route(row, "QC-000-MIN", validity=validity)
    assert np.array_equal(row, before_row)
    assert repr(STATE) == before_state
    assert validity == before_validity
    assert out["action_validity"] == before_validity
    assert out["action_validity"] is not validity
    assert all(out["action_validity"][key] is True for key in ACTION_IDS)
    expected = json.dumps(out, sort_keys=True)
    out["action_validity"][MIN] = False
    out["features"][0] = -999.0
    assert validity == before_validity
    assert np.array_equal(row, before_row)
    fresh = _route(row, "QC-000-MIN", validity=validity)
    assert json.dumps(fresh, sort_keys=True) == expected
    assert fresh["action_validity"] == before_validity
    assert fresh["features"][0] == 2.0
    assert out["features"] is not fresh["features"]


def test_identical_inputs_seal_identically():
    row = _row({FEATURES[1]: 10.0})
    first = _route(row, "QC-000-MIN")
    second = _route(row, "QC-000-MIN")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["route_sha256"] == second["route_sha256"]


def test_no_identity_inputs_api():
    row = _row()
    for key in ("label", "batch", "uid", "station", "instrument", "substrate", "path", "operator"):
        with pytest.raises(TypeError):
            route_qc_row(
                row,
                STATE,
                "QC-000-MIN",
                dict(VALID),
                expected_state_sha256=STATE_SHA,
                **{key: "sentinel"},
            )
    assert not (set(_route(row, "QC-000-MIN")) & {"uid", "label", "batch"})


def test_hashes_keys_and_plain_json():
    row = _row({FEATURES[0]: 2.0})
    out = _route(row, "QC-000-MIN")
    expected_row_sha = hashlib.sha256(np.ascontiguousarray(row, dtype="<f8").tobytes()).hexdigest()
    assert out["qc_row_sha256"] == expected_row_sha
    payload = {key: out[key] for key in OUT_KEYS[:-1]}
    assert out["route_sha256"] == canonical_sha256(payload)
    assert set(out) == set(OUT_KEYS)
    assert json.loads(json.dumps(out)) == out
    assert "synthetic-source-a" not in json.dumps(out)


def test_same_features_different_bytes_change_hashes():
    row_a = _row({FEATURES[0]: 2.0})
    row_b = row_a.copy()
    row_b[0] = 4.0
    row_b[1] = 2.0
    first = _route(row_a, "QC-000-MIN")
    second = _route(row_b, "QC-000-MIN")
    assert first["features"] == second["features"]
    assert first["qc_row_sha256"] != second["qc_row_sha256"]
    assert first["route_sha256"] != second["route_sha256"]


def test_gate_state_and_unused_validity_change_route_hash():
    row = _row({FEATURES[2]: 10.0})
    base = _route(row, SINGLE_BASELINE["gate_candidate_id"])
    other_gate = _route(row, SINGLE_NOISE["gate_candidate_id"])
    assert other_gate["route_sha256"] != base["route_sha256"]
    alt_state = fit_source_thresholds(
        BASELINE_ROWS + 10.0,
        SOURCE_UIDS,
        expected_fit_uid_sha256=canonical_sha256(sorted(SOURCE_UIDS)),
    )
    alt = _route(
        row,
        SINGLE_BASELINE["gate_candidate_id"],
        state=alt_state,
        sha=_sealed_sha(alt_state),
    )
    assert alt["route_sha256"] != base["route_sha256"]
    flap = _route(
        row,
        SINGLE_BASELINE["gate_candidate_id"],
        validity={MIN: True, SG: False, ARPLS: True},
    )
    assert flap["route_sha256"] != base["route_sha256"]


def test_validate_threshold_state_returns_independent_copy():
    copied = validate_threshold_state(STATE)
    assert _sealed_sha(copied) == STATE_SHA
    assert copied is not STATE


BAD_ROWS = [
    np.zeros(5),
    np.zeros((1, 6)),
    [0.0] * 6,
    (0.0,) * 6,
    np.zeros(6, dtype=bool),
    np.zeros(6, dtype=object),
    np.zeros(6, dtype=complex),
    np.array([[0.0] * 6]).view(np.matrix),
    np.ma.array([0.0] * 6),
    np.float64(0.0),
]


@pytest.mark.parametrize("bad_row", BAD_ROWS)
def test_invalid_row_containers(bad_row):
    _expect_reason(
        "invalid_qc_row",
        route_qc_row,
        bad_row,
        STATE,
        "QC-000-MIN",
        dict(VALID),
        expected_state_sha256=STATE_SHA,
    )


@pytest.mark.parametrize("gate", [[], {}, None, 5, "QC-999-NOPE", "qc-000-min", ""])
def test_invalid_gate_candidate(gate):
    _expect_reason(
        "invalid_gate_candidate",
        route_qc_row,
        _row(),
        STATE,
        gate,
        dict(VALID),
        expected_state_sha256=STATE_SHA,
    )


@pytest.mark.parametrize(
    "validity",
    [
        {MIN: True, SG: True},
        {MIN: True, SG: True, ARPLS: True, "extra": True},
        {MIN: 1, SG: True, ARPLS: True},
        {MIN: True, SG: True, ARPLS: None},
        {MIN: True, SG: np.True_, ARPLS: True},
    ],
)
def test_invalid_action_validity(validity):
    _expect_reason(
        "invalid_action_validity",
        route_qc_row,
        _row(),
        STATE,
        "QC-000-MIN",
        validity,
        expected_state_sha256=STATE_SHA,
    )


@pytest.mark.parametrize("digest", ["", "ABC", "0" * 63, "0" * 65, "g" * 64, "A" * 64, 5, None])
def test_malformed_expected_state_digest(digest):
    _expect_reason(
        "invalid_qc_routing_input",
        route_qc_row,
        _row(),
        STATE,
        "QC-000-MIN",
        dict(VALID),
        expected_state_sha256=digest,
    )


def test_mismatched_expected_state_digest():
    _expect_reason(
        "threshold_state_mismatch",
        route_qc_row,
        _row(),
        STATE,
        "QC-000-MIN",
        dict(VALID),
        expected_state_sha256="0" * 64,
    )


def test_tampered_state_rejected():
    tampered = copy.deepcopy(STATE)
    tampered["status"] = "tampered"
    _expect_reason(
        "invalid_threshold_state",
        route_qc_row,
        _row(),
        tampered,
        "QC-000-MIN",
        dict(VALID),
        expected_state_sha256=STATE_SHA,
    )


@pytest.mark.parametrize(
    "malformed",
    [
        [3.0, 2.25, 1.5],
        [True, False, True],
        ["1.5", "2.25", "2.7"],
    ],
)
def test_malformed_cutpoints_resealed_rejected(malformed):
    state = copy.deepcopy(STATE)
    state["cutpoints"][0] = malformed
    state = _reseal(state)
    _expect_reason(
        "invalid_threshold_state",
        route_qc_row,
        _row(),
        state,
        "QC-000-MIN",
        dict(VALID),
        expected_state_sha256=_sealed_sha(state),
    )


def test_nan_cutpoint_without_reseal_rejected():
    state = copy.deepcopy(STATE)
    state["cutpoints"][0][0] = float("nan")
    _expect_reason(
        "invalid_threshold_state",
        route_qc_row,
        _row(),
        state,
        "QC-000-MIN",
        dict(VALID),
        expected_state_sha256=STATE_SHA,
    )


def test_resealed_execution_flag_rejected():
    flagged = _reseal({**copy.deepcopy(STATE), "execution_authorized": True})
    _expect_reason(
        "invalid_threshold_state",
        route_qc_row,
        _row(),
        flagged,
        "QC-000-MIN",
        dict(VALID),
        expected_state_sha256=_sealed_sha(flagged),
    )


def test_errors_are_sanitized():
    assert issubclass(QCRoutingError, ValueError)
    sentinel = "SENTINEL-p08-qc-leak-value"
    err = QCRoutingError(sentinel)
    assert err.reason_code == "invalid_qc_routing_input"
    assert sentinel not in str(err)
    assert repr(sentinel) not in str(err)
    mapping = {sentinel: ["x"]}
    assert QCRoutingError(mapping).reason_code == "invalid_qc_routing_input"
    assert sentinel not in str(QCRoutingError(mapping))
    assert QCRoutingError({}).reason_code == "invalid_qc_routing_input"
    assert QCRoutingError("invalid_qc_row").reason_code == "invalid_qc_row"
    assert QCRoutingError("not_a_whitelisted_reason").reason_code == "invalid_qc_routing_input"
    for bad in ([sentinel], mapping):
        with pytest.raises(QCRoutingError) as excinfo:
            route_qc_row(
                _row(),
                STATE,
                "QC-000-MIN",
                {MIN: True, SG: True, ARPLS: bad},
                expected_state_sha256=STATE_SHA,
            )
        assert excinfo.value.reason_code == "invalid_action_validity"
        assert sentinel not in str(excinfo.value)


def test_strict_errstate_is_locally_ignored_and_does_not_leak():
    before = np.geterr().copy()
    overflow = _row()
    overflow[0] = 1e308
    overflow[1] = 1e-308
    with np.errstate(all="raise"):
        out = _route(overflow, "QC-000-MIN")
    assert out["qc_available"] is False
    assert out["fallback_reason"] == "missing_or_nonfinite_qc"
    assert np.geterr() == before


def test_endian_and_noncontiguous_normalize_identically():
    row = _row({FEATURES[0]: 2.0, FEATURES[2]: 1.0})
    reference = _route(row.copy(), "QC-000-MIN")["qc_row_sha256"]
    big_endian = row.astype(">f8")
    assert _route(big_endian, "QC-000-MIN")["qc_row_sha256"] == reference
    padded = np.zeros(12, dtype=np.float64)
    padded[::2] = row
    view = padded[::2]
    assert not view.flags["C_CONTIGUOUS"]
    assert _route(view, "QC-000-MIN")["qc_row_sha256"] == reference


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((), {}),
        ((True,), {}),
        ((), {"execution_authorized": True}),
        (({"execution_authorized": True},), {}),
        (("QC-000-MIN",), {}),
    ],
)
def test_scientific_execution_always_denied(args, kwargs):
    _expect_reason(
        "scientific_execution_not_authorized",
        require_scientific_execution,
        *args,
        **kwargs,
    )
