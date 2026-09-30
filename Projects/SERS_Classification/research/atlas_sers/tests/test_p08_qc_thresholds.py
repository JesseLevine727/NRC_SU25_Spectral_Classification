"""Synthetic (invented-data) tests for the P08-T027 QC threshold kernel.

Only this test module is added here; the numerical kernel lives in
``atlas_sers.evaluation.p08_qc_thresholds`` and is implemented elsewhere.
No private records, no scientific-run authority, no source-file
authentication claims.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math

import numpy as np
import pytest

from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256
from atlas_sers.evaluation.p08_qc_thresholds import (
    FEATURES,
    INGREDIENTS,
    QUANTILES,
    QCThresholdError,
    derive_qc_features,
    fit_source_thresholds,
    require_scientific_execution,
    validate_threshold_state,
)

SCHEMA_VERSION = "nato-sers-p08-qc-threshold-state-v1"
STATE_KEYS = (
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
)
BASE_ROWS = (
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.1, 0.2, 0.3, 0.4],
    [2.0, 1.0, 0.2, 0.3, 0.4, 0.5],
    [3.0, 1.0, 0.3, 0.4, 0.5, 0.6],
)
BASE_UIDS = ("uid-a", "uid-b", "uid-c", "uid-d")


def _qc(rows):
    return np.array(rows, dtype=np.float64)


def _as_matrix(rows):
    """Return ``rows`` as ``np.matrix`` without a constructor warning."""
    return np.asarray(_qc(rows)).view(np.matrix)


def _fit_hash(uids):
    return canonical_sha256(sorted(uids))


def _qc_input_hash(qc, uids):
    order = sorted(range(len(uids)), key=lambda i: uids[i])
    arr = np.ascontiguousarray(np.asarray(qc)[order].astype("<f8"))
    values = hashlib.sha256(arr.tobytes(order="C")).hexdigest()
    return canonical_sha256(
        {
            "ingredients": list(INGREDIENTS),
            "source_uid_sha256": _fit_hash(uids),
            "shape": [int(arr.shape[0]), int(arr.shape[1])],
            "dtype": "<f8",
            "values_sha256": values,
        }
    )


def _sealed(state):
    return canonical_sha256({k: v for k, v in state.items() if k != "state_sha256"})


def _manual_quantile(values, p):
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = p * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _make_state(rows=BASE_ROWS, uids=BASE_UIDS):
    qc = _qc(rows)
    uids = list(uids)
    return fit_source_thresholds(qc, uids, expected_fit_uid_sha256=_fit_hash(uids))


def _valid_cutpoints():
    return [[float(j + 1), float(j + 2), float(j + 3)] for j in range(5)]


def _resealed(**overrides):
    state = _make_state()
    state.update(overrides)
    state["state_sha256"] = _sealed(state)
    return state


BAD_CUTPOINTS = {
    "row_count": [[1.0, 2.0, 3.0]],
    "col_count": [[1.0], [2.0], [3.0], [4.0], [5.0]],
    "six_rows": _valid_cutpoints() + [[6.0, 7.0, 8.0]],
    "ragged": [
        [1.0, 2.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ],
    "strings": [["a", "b", "c"]] * 5,
    "booleans": [[True, 2.0, 3.0]] + _valid_cutpoints()[1:],
    "nan": [[float("nan"), 2.0, 3.0]] + _valid_cutpoints()[1:],
    "inf": [[float("inf"), 2.0, 3.0]] + _valid_cutpoints()[1:],
    "nonmonotonic": [[3.0, 2.0, 1.0]] + _valid_cutpoints()[1:],
    "not_list": "not-a-cutpoint-table",
}


def test_public_constants_exact():
    assert INGREDIENTS == (
        "first_difference_noise_mad",
        "intensity_range",
        "spike_fraction_proxy",
        "baseline_energy_fraction_proxy",
        "baseline_span_fraction_proxy",
        "negative_fraction",
    )
    assert FEATURES == (
        "first_difference_noise_mad_over_intensity_range",
        "spike_fraction_proxy",
        "baseline_energy_fraction_proxy",
        "baseline_span_fraction_proxy",
        "negative_fraction",
    )
    assert QUANTILES == (0.5, 0.75, 0.9)


def test_derive_hand_computed_ratio_and_validity():
    feats, valid = derive_qc_features(
        _qc([[1.0, 4.0, 0.1, 0.2, 0.3, 0.4], [3.0, 2.0, 0.5, 0.6, 0.7, 0.8]])
    )
    assert feats.shape == (2, 5)
    assert feats.dtype == np.float64 and feats.flags["C_CONTIGUOUS"]
    assert valid.shape == (2,) and valid.dtype == np.bool_
    np.testing.assert_allclose(feats[0], [0.25, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(feats[1], [1.5, 0.5, 0.6, 0.7, 0.8])
    assert valid.tolist() == [True, True]


@pytest.mark.parametrize("col", range(6))
def test_each_nonfinite_ingredient_invalidates_row(col):
    qc = _qc([[1.0, 2.0, 0.1, 0.2, 0.3, 0.4], [0.5, 1.0, 0.6, 0.7, 0.8, 0.9]])
    qc[0, col] = np.nan
    feats, valid = derive_qc_features(qc)
    assert valid.tolist() == [False, True]
    assert np.all(np.isnan(feats[0]))
    np.testing.assert_allclose(feats[1], [0.5, 0.6, 0.7, 0.8, 0.9])


@pytest.mark.parametrize("rng", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_bad_range_invalidates_row(rng):
    feats, valid = derive_qc_features(_qc([[1.0, rng, 0.1, 0.2, 0.3, 0.4]]))
    assert valid.tolist() == [False]
    assert np.all(np.isnan(feats[0]))


def test_ratio_overflow_invalidates_all_features():
    feats, valid = derive_qc_features(_qc([[1e308, 1e-308, 0.1, 0.2, 0.3, 0.4]]))
    assert valid.tolist() == [False]
    assert np.all(np.isnan(feats[0]))


def test_integer_inputs_and_no_input_mutation():
    qc = np.array([[1, 2, 1, 2, 3, 4]], dtype=np.int64)
    before = qc.copy()
    feats, valid = derive_qc_features(qc)
    assert feats[0, 0] == 0.5 and valid.tolist() == [True]
    np.testing.assert_array_equal(qc, before)
    assert qc.dtype == np.int64


def test_noncontiguous_bigendian_f32_and_input_preservation():
    base = _qc([[1.0, 4.0, 0.5, 0.25, 0.125, 0.0625], [3.0, 2.0, 0.5, 0.25, 0.125, 0.0625]])
    wide = np.zeros((2, 12))
    wide[:, ::2] = base
    view = wide[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    ref, _ = derive_qc_features(base)
    for arr in (view, base.astype(">f8"), base.astype(np.float32)):
        snapshot = np.array(arr, copy=True)
        feats, valid = derive_qc_features(arr)
        np.testing.assert_array_equal(feats, ref)
        np.testing.assert_array_equal(valid, [True, True])
        np.testing.assert_array_equal(np.asarray(arr), np.asarray(snapshot))
        assert not feats.dtype.str.startswith(">")


def test_derive_empty_supported():
    feats, valid = derive_qc_features(np.zeros((0, 6)))
    assert feats.shape == (0, 5) and feats.dtype == np.float64
    assert valid.shape == (0,) and valid.dtype == np.bool_


@pytest.mark.parametrize(
    "bad",
    [
        [[1.0, 2.0, 0.1, 0.2, 0.3, 0.4]],
        _as_matrix([[1.0, 2.0, 0.1, 0.2, 0.3, 0.4]]),
        np.ma.masked_array(np.zeros((1, 6))),
        np.zeros((1, 5)),
        np.zeros((1, 7)),
        np.zeros((1, 6, 1)),
        np.zeros((1, 6), dtype=bool),
        np.array([["a"] * 6]),
        np.zeros((1, 6), dtype=object),
        np.zeros((1, 6), dtype=complex),
    ],
)
def test_derive_rejects_unsupported_inputs(bad):
    with pytest.raises(QCThresholdError):
        derive_qc_features(bad)


def test_known_quantiles_and_incomplete_row_contributes_nothing():
    rows = [
        [0.0, 1.0, 10.0, 100.0, 0.0, -1.0],
        [1.0, 1.0, 20.0, 200.0, 0.1, -2.0],
        [2.0, 1.0, 30.0, 300.0, -0.2, -3.0],
        [3.0, 1.0, 40.0, 400.0, -0.3, -4.0],
        [9.0, 0.0, 99.0, 999.0, 0.9, -9.0],
    ]
    uids = ["q0", "q1", "q2", "q3", "q4"]
    qc = _qc(rows)
    state = fit_source_thresholds(qc, uids, expected_fit_uid_sha256=_fit_hash(uids))
    assert state["status"] == "finite_source_qc"
    assert state["source_row_count"] == 5 and state["valid_source_row_count"] == 4
    feats, valid = derive_qc_features(qc)
    for j in range(5):
        expected = [_manual_quantile(feats[valid, j], p) for p in QUANTILES]
        np.testing.assert_allclose(state["cutpoints"][j], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state["cutpoints"][0], [1.5, 2.25, 2.7], rtol=1e-12, atol=1e-12)
    four = fit_source_thresholds(qc[:4], uids[:4], expected_fit_uid_sha256=_fit_hash(uids[:4]))
    assert four["cutpoints"] == state["cutpoints"]
    assert four["valid_source_row_count"] == 4


def test_source_only_invariance_and_exact_input_binding():
    rows = [
        [1.0, 0.0, 0.5, 0.6, 0.7, 0.8],
        [1.0, 1.0, 0.1, 0.2, 0.3, 0.4],
        [2.0, 1.0, 0.2, 0.3, 0.4, 0.5],
        [3.0, 1.0, 0.3, 0.4, 0.5, 0.6],
    ]
    uids = ["a", "b", "c", "d"]
    qa = _qc(rows)
    qb = qa.copy()
    qb[0, 3] = 12345.0  # unused ingredient in the invalid source row
    sa = fit_source_thresholds(qa, uids, expected_fit_uid_sha256=_fit_hash(uids))
    sb = fit_source_thresholds(qb, uids, expected_fit_uid_sha256=_fit_hash(uids))
    assert sa["valid_source_row_count"] == sb["valid_source_row_count"] == 3
    assert sa["cutpoints"] == sb["cutpoints"]
    assert sa["qc_input_sha256"] != sb["qc_input_sha256"]
    assert sa["state_sha256"] != sb["state_sha256"]
    qc2 = qa.copy()
    qc2[1, 2] += 25.0  # a valid cell
    s2 = fit_source_thresholds(qc2, uids, expected_fit_uid_sha256=_fit_hash(uids))
    assert s2["cutpoints"] != sa["cutpoints"]
    assert s2["state_sha256"] != sa["state_sha256"]


def test_fit_signature_takes_no_target_data():
    params = set(inspect.signature(fit_source_thresholds).parameters)
    assert not ({"target", "targets", "held", "heldout", "validation"} & params)


def test_order_invariance_under_paired_row_uid_permutation():
    qc, uids = _qc(BASE_ROWS), list(BASE_UIDS)
    base = fit_source_thresholds(qc, uids, expected_fit_uid_sha256=_fit_hash(uids))
    perm = [2, 0, 3, 1]
    shuffled = fit_source_thresholds(
        qc[perm], [uids[i] for i in perm], expected_fit_uid_sha256=_fit_hash(uids)
    )
    assert shuffled["state_sha256"] == base["state_sha256"]
    assert shuffled["qc_input_sha256"] == base["qc_input_sha256"]
    assert shuffled["fit_uid_sha256"] == base["fit_uid_sha256"]
    assert shuffled["cutpoints"] == base["cutpoints"]


def test_fit_normalizes_f32_endian_and_noncontiguous_identically():
    base = _qc([[1.0, 4.0, 0.5, 0.25, 0.125, 0.0625], [3.0, 2.0, 0.5, 0.25, 0.125, 0.0625]])
    uids = ["x", "y"]
    ref = fit_source_thresholds(base, uids, expected_fit_uid_sha256=_fit_hash(uids))
    wide = np.zeros((2, 12))
    wide[:, ::2] = base
    for variant in (base.astype(np.float32), base.astype(">f8"), wide[:, ::2]):
        state = fit_source_thresholds(variant, uids, expected_fit_uid_sha256=_fit_hash(uids))
        assert state["qc_input_sha256"] == ref["qc_input_sha256"]
        assert state["cutpoints"] == ref["cutpoints"]
        assert state["state_sha256"] == ref["state_sha256"]


def test_utf8_source_uids():
    uids = ["α-β", "日本語", "emoji-😀", "plain"]
    state = _make_state(uids=uids)
    assert state["fit_uid_sha256"] == _fit_hash(uids)
    assert state["valid_source_row_count"] == 4


@pytest.mark.parametrize(
    "bad_uids",
    [
        ("a", "a", "b", "c"),
        ("a", "", "b", "c"),
        ("a", "   ", "b", "c"),
        ("a", "\ud800", "b", "c"),
        ("a", None, "b", "c"),
    ],
)
def test_uid_value_validation_errors(bad_uids):
    qc = _qc([[1.0, 1.0, 0.1, 0.2, 0.3, 0.4]] * len(bad_uids))
    with pytest.raises(QCThresholdError):
        fit_source_thresholds(qc, list(bad_uids), expected_fit_uid_sha256="0" * 64)


@pytest.mark.parametrize("bad", [{"a", "b", "c", "d"}, 123, None])
def test_uid_container_must_be_list_or_tuple(bad):
    with pytest.raises(QCThresholdError):
        fit_source_thresholds(_qc(BASE_ROWS), bad, expected_fit_uid_sha256="0" * 64)


def test_uid_count_must_match_rows():
    with pytest.raises(QCThresholdError):
        fit_source_thresholds(
            _qc(BASE_ROWS),
            ["a", "b", "c"],
            expected_fit_uid_sha256="0" * 64,
        )


@pytest.mark.parametrize("bad", ["", "a" * 63, "0" * 65, "A" * 64, "g" * 64, 1234, None])
def test_expected_fit_hash_must_be_valid_lower64(bad):
    with pytest.raises(QCThresholdError):
        fit_source_thresholds(
            _qc(BASE_ROWS),
            list(BASE_UIDS),
            expected_fit_uid_sha256=bad,
        )


def test_expected_fit_hash_mismatch():
    with pytest.raises(QCThresholdError):
        fit_source_thresholds(
            _qc(BASE_ROWS),
            list(BASE_UIDS),
            expected_fit_uid_sha256="0" * 64,
        )


def test_empty_and_all_invalid_source():
    empty = fit_source_thresholds(np.zeros((0, 6)), [], expected_fit_uid_sha256=_fit_hash([]))
    assert empty["status"] == "no_finite_source_qc" and empty["cutpoints"] is None
    assert empty["source_row_count"] == 0 and empty["valid_source_row_count"] == 0
    assert empty["features"] == list(FEATURES)
    assert empty["quantiles"] == list(QUANTILES)
    rows = [[1.0, 0.0, 0.1, 0.2, 0.3, 0.4], [2.0, -1.0, 0.1, 0.2, 0.3, 0.4]]
    uids = ["m", "n"]
    state = fit_source_thresholds(_qc(rows), uids, expected_fit_uid_sha256=_fit_hash(uids))
    assert state["status"] == "no_finite_source_qc" and state["cutpoints"] is None
    assert state["source_row_count"] == 2 and state["valid_source_row_count"] == 0


def test_repeated_rows_with_distinct_uids_each_contribute():
    row = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
    rows = [row, row, [4.0, 1.0, 0.5, 0.25, 0.125, 0.0625]]
    uids = ["a", "b", "c"]
    state = fit_source_thresholds(_qc(rows), uids, expected_fit_uid_sha256=_fit_hash(uids))
    assert state["valid_source_row_count"] == 3
    assert state["cutpoints"][0][0] == pytest.approx(2.0)


def test_fit_rejects_unsupported_qc():
    with pytest.raises(QCThresholdError):
        fit_source_thresholds(
            [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            ["a"],
            expected_fit_uid_sha256="0" * 64,
        )
    with pytest.raises(QCThresholdError):
        fit_source_thresholds(
            np.zeros((1, 6), dtype=bool),
            ["a"],
            expected_fit_uid_sha256="0" * 64,
        )


def test_state_seals_hashes_and_never_stores_raw_uids():
    uids = ["SENTINEL_UID_A", "SENTINEL_UID_B", "SENTINEL_UID_C", "SENTINEL_UID_D"]
    qc = _qc(BASE_ROWS)
    state = fit_source_thresholds(qc, uids, expected_fit_uid_sha256=_fit_hash(uids))
    assert state["schema_version"] == SCHEMA_VERSION
    assert state["execution_authorized"] is False
    assert set(state) == set(STATE_KEYS)
    assert state["features"] == list(FEATURES)
    assert state["quantiles"] == list(QUANTILES)
    assert state["state_sha256"] == _sealed(state)
    assert state["qc_input_sha256"] == _qc_input_hash(qc, uids)
    assert state["fit_uid_sha256"] == _fit_hash(uids)
    blob = json.dumps(state, sort_keys=True)
    assert "SENTINEL_UID" not in blob
    for uid in uids:
        assert uid not in blob


def test_validate_accepts_fresh_state_and_returns_deep_copy():
    state = _make_state()
    result = validate_threshold_state(state)
    assert result == state and result is not state
    result["cutpoints"][0][0] = 999991.0
    result["features"][0] = "mutated"
    result["quantiles"][0] = 123.0
    assert state["cutpoints"][0][0] != 999991.0
    assert state["features"][0] == FEATURES[0]
    assert state["quantiles"][0] == QUANTILES[0]


def test_validate_rejects_extra_and_missing_keys():
    with pytest.raises(QCThresholdError):
        validate_threshold_state(_resealed(extra_key=1))
    missing = _make_state()
    del missing["features"]
    missing["state_sha256"] = _sealed(missing)
    with pytest.raises(QCThresholdError):
        validate_threshold_state(missing)


@pytest.mark.parametrize("field", ["fit_uid_sha256", "qc_input_sha256"])
def test_validate_rejects_malformed_embedded_hashes(field):
    with pytest.raises(QCThresholdError):
        validate_threshold_state(_resealed(**{field: "not-a-hash"}))


def test_validate_rejects_wrong_state_hash():
    state = _make_state()
    state["state_sha256"] = "0" * 64
    with pytest.raises(QCThresholdError):
        validate_threshold_state(state)


@pytest.mark.parametrize("flag", [True, 0, 1, "False", None])
def test_validate_rejects_non_false_execution_flag(flag):
    with pytest.raises(QCThresholdError):
        validate_threshold_state(_resealed(execution_authorized=flag))


@pytest.mark.parametrize("field", ["source_row_count", "valid_source_row_count"])
@pytest.mark.parametrize("value", [True, 4.0, 3.5, "4", None])
def test_validate_rejects_non_int_counts(field, value):
    with pytest.raises(QCThresholdError):
        validate_threshold_state(_resealed(**{field: value}))


@pytest.mark.parametrize("src,valid", [(4, 5), (3, 4), (-1, 0), (4, -1)])
def test_validate_rejects_invalid_count_ordering(src, valid):
    with pytest.raises(QCThresholdError):
        validate_threshold_state(_resealed(source_row_count=src, valid_source_row_count=valid))


def test_validate_rejects_wrong_features_or_quantiles():
    with pytest.raises(QCThresholdError):
        validate_threshold_state(_resealed(features=list(reversed(FEATURES))))
    with pytest.raises(QCThresholdError):
        validate_threshold_state(_resealed(quantiles=[0.5, 0.75]))


def test_validate_enforces_status_cutpoints_consistency():
    with pytest.raises(QCThresholdError):
        validate_threshold_state(_resealed(status="finite_source_qc", cutpoints=None))
    with pytest.raises(QCThresholdError):
        validate_threshold_state(_resealed(status="no_finite_source_qc"))
    with pytest.raises(QCThresholdError):
        validate_threshold_state(_resealed(status="bogus_status"))


@pytest.mark.parametrize("key", sorted(BAD_CUTPOINTS))
def test_validate_rejects_bad_cutpoints(key):
    with pytest.raises(QCThresholdError):
        if key in ("nan", "inf"):
            # Strict JSON cannot encode nonfinite floats, so mutate a valid
            # sealed state without resealing; the validator must still reject.
            state = _make_state()
            state["cutpoints"] = BAD_CUTPOINTS[key]
            validate_threshold_state(state)
        else:
            validate_threshold_state(_resealed(cutpoints=BAD_CUTPOINTS[key]))


def test_validate_allows_integer_cutpoints_without_normalization():
    ints = [[j + 1, j + 2, j + 3] for j in range(5)]
    state = _resealed(cutpoints=ints)
    result = validate_threshold_state(state)
    assert result["cutpoints"] == ints
    for row in result["cutpoints"]:
        for value in row:
            assert type(value) is int
    assert result["state_sha256"] == state["state_sha256"]


def test_validate_rejects_descending_large_integers_despite_float_rounding():
    big = 2**53
    cuts = [[big + 1, big, big + 2]] + _valid_cutpoints()[1:]
    with pytest.raises(QCThresholdError):
        validate_threshold_state(_resealed(cutpoints=cuts))


def test_validate_preserves_sorted_large_integers_exactly():
    big = 2**53
    cuts = [[big, big + 1, big + 2]] + _valid_cutpoints()[1:]
    result = validate_threshold_state(_resealed(cutpoints=cuts))
    assert result["cutpoints"] == cuts
    assert all(type(value) is int for value in result["cutpoints"][0])


def test_validate_optional_expected_hashes():
    state = _make_state()
    assert (
        validate_threshold_state(
            state,
            expected_fit_uid_sha256=state["fit_uid_sha256"],
            expected_qc_input_sha256=state["qc_input_sha256"],
        )
        == state
    )
    with pytest.raises(QCThresholdError):
        validate_threshold_state(state, expected_fit_uid_sha256="0" * 64)
    with pytest.raises(QCThresholdError):
        validate_threshold_state(state, expected_qc_input_sha256="0" * 64)
    with pytest.raises(QCThresholdError):
        validate_threshold_state(state, expected_fit_uid_sha256="NOTHEX")
    with pytest.raises(QCThresholdError):
        validate_threshold_state(state, expected_qc_input_sha256="ABC")


@pytest.mark.parametrize("bad", [None, [], "state", 7])
def test_validate_rejects_non_mapping_state(bad):
    with pytest.raises(QCThresholdError):
        validate_threshold_state(bad)


def test_require_scientific_execution_always_denies():
    state = _make_state()
    with pytest.raises(QCThresholdError) as excinfo:
        require_scientific_execution(state)
    assert excinfo.value.reason_code == "scientific_execution_not_authorized"
    forged = dict(state)
    forged["execution_authorized"] = True
    with pytest.raises(QCThresholdError) as excinfo2:
        require_scientific_execution(forged)
    assert excinfo2.value.reason_code == "scientific_execution_not_authorized"


def test_errors_do_not_echo_private_sentinel():
    sentinel = "PRIVATE_SENTINEL_12345"
    uids = [sentinel, sentinel, "b", "c"]
    with pytest.raises(QCThresholdError) as excinfo:
        fit_source_thresholds(_qc(BASE_ROWS), uids, expected_fit_uid_sha256="0" * 64)
    rendered = f"{excinfo.value!s}|{excinfo.value!r}|{excinfo.value.reason_code}"
    assert sentinel not in rendered


def test_qc_threshold_error_unknown_code_does_not_echo_sentinel():
    sentinel = "PRIVATE_SENTINEL_UNKNOWN_CODE"
    error = QCThresholdError(sentinel)
    assert sentinel not in f"{error!s}|{error!r}"


def test_numpy_errstate_unchanged_after_numerical_edge_cases():
    before = np.geterr().copy()
    feats, valid = derive_qc_features(_qc([[1e308, 1e-308, 0.0, 0.0, 0.0, 0.0]]))
    assert valid.tolist() == [False]
    assert np.all(np.isnan(feats[0]))
    derive_qc_features(_qc([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    fit_source_thresholds(
        _qc(BASE_ROWS),
        list(BASE_UIDS),
        expected_fit_uid_sha256=_fit_hash(BASE_UIDS),
    )
    assert np.geterr() == before


@pytest.mark.parametrize("bad", [[], {}, ["PRIVATE_SENTINEL_UNHASHABLE_REASON"]])
def test_qc_threshold_error_unhashable_reason_is_static(bad):
    error = QCThresholdError(bad)
    assert error.reason_code == "invalid_qc_threshold_input"
    assert error.args == ("invalid_qc_threshold_input",)
    assert str(error) == "invalid_qc_threshold_input"
    assert "PRIVATE_SENTINEL" not in f"{error!s}|{error!r}"


def test_numerical_blocks_tolerate_global_raise_and_restore_errstate():
    tiny = np.finfo(np.float64).tiny
    rows = [
        [tiny, 1e300, 0.1, 0.2, 0.3, 0.4],
        [2.0 * tiny, 1e300, 0.5, 0.6, 0.7, 0.8],
    ]
    qc = _qc(rows)
    uids = ["u0", "u1"]
    baseline_feats, baseline_valid = derive_qc_features(qc)
    baseline_state = fit_source_thresholds(qc, uids, expected_fit_uid_sha256=_fit_hash(uids))
    before = np.geterr().copy()
    with np.errstate(all="raise"):
        strict_feats, strict_valid = derive_qc_features(qc)
        strict_state = fit_source_thresholds(qc, uids, expected_fit_uid_sha256=_fit_hash(uids))
        assert validate_threshold_state(strict_state) == strict_state
    assert np.geterr() == before
    np.testing.assert_array_equal(strict_feats, baseline_feats)
    np.testing.assert_array_equal(strict_valid, baseline_valid)
    assert strict_state == baseline_state


def test_numerical_blocks_handle_longdouble_range_casts_when_available():
    ld = np.longdouble
    finite_rows = np.array([[1.0, 1.0, 0.1, 0.2, 0.3, 0.4]], dtype=ld)
    before = np.geterr().copy()
    with np.errstate(all="raise"):
        feats, valid = derive_qc_features(finite_rows)
    assert np.geterr() == before
    assert feats.shape == (1, 5)
    assert valid.tolist() == [True]
    if np.finfo(ld).maxexp <= np.finfo(np.float64).maxexp:
        return
    extreme = np.array(
        [
            [1.0, 1.0, 0.1, 0.2, 0.3, 0.4],
            [np.finfo(ld).max, np.finfo(ld).tiny, 0.5, 0.6, 0.7, 0.8],
        ],
        dtype=ld,
    )
    with np.errstate(all="raise"):
        extreme_feats, extreme_valid = derive_qc_features(extreme)
    assert np.geterr() == before
    assert extreme_valid.tolist() == [True, False]
    assert np.all(np.isnan(extreme_feats[1]))


def test_quantile_linear_overflow_reports_nonfinite_source_quantiles():
    rows = [
        [1.0, 1.0, -1e308, 0.0, 0.0, 0.0],
        [2.0, 1.0, 1e308, 0.0, 0.0, 0.0],
    ]
    uids = ["h0", "h1"]
    qc = _qc(rows)
    before = np.geterr().copy()
    with np.errstate(all="raise"):
        with pytest.raises(QCThresholdError) as excinfo:
            fit_source_thresholds(qc, uids, expected_fit_uid_sha256=_fit_hash(uids))
    assert np.geterr() == before
    assert excinfo.value.reason_code == "nonfinite_source_quantiles"
