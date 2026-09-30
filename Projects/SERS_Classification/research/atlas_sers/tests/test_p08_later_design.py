"""P08-T062 design-constraint tests for later-branch behaviour.

These tests exercise only pure, row-local arithmetic already present in
``atlas_sers.preprocessing.representations`` plus one frozen public contract.
They use invented arrays and coordinate bookkeeping. They do not read any
dataset, run any real transform pipeline, or fit any model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from atlas_sers.preprocessing.representations import (
    row_area,
    row_minmax,
    row_snv,
    row_vector,
)

AFFINE_ROWS = np.array(
    [
        [3.0, -1.0, 4.0, 0.5],
        [-2.0, 5.0, -3.0, 1.0],
        [1.5, -4.0, 2.0, -0.5],
    ],
    dtype=float,
)


def _apply(name, matrix, axis):
    """Local dispatch over the four row-local transforms under test."""
    if name == "minmax":
        return row_minmax(matrix)
    if name == "snv":
        return row_snv(matrix)
    if name == "vector":
        return row_vector(matrix)
    if name == "area":
        return row_area(matrix, axis)
    raise ValueError(f"unknown row transform {name!r}")


@pytest.mark.parametrize("transform_name", ["snv", "vector", "area"])
def test_row_minmax_positive_affine_equivalence(transform_name):
    """Positively scaled (and translated) rows keep the same min-max shape.

    This is an arithmetic invariant on invented nonconstant rows. It is not
    evidence about any scientific dataset.
    """
    axis = np.arange(AFFINE_ROWS.shape[1], dtype=float)
    baseline = AFFINE_ROWS.copy()
    transformed, valid, reasons = _apply(transform_name, AFFINE_ROWS, axis)
    assert valid.all()
    assert reasons == ("included", "included", "included")
    normalized, normalized_valid, _ = row_minmax(transformed)
    assert normalized_valid.all()
    reference = row_minmax(baseline)[0]
    assert np.allclose(normalized, reference, rtol=0.0, atol=1e-12)
    assert np.array_equal(baseline, AFFINE_ROWS)


@pytest.mark.parametrize(
    "name,matrix,axis,expected,invariant",
    [
        (
            "snv",
            [[0.0, 1.0, 2.0]],
            None,
            [-np.sqrt(1.5), 0.0, np.sqrt(1.5)],
            "standardized",
        ),
        ("vector", [[3.0, 4.0, 0.0]], None, [0.6, 0.8, 0.0], "unit_norm"),
        ("area", [[1.0, 3.0, 2.0]], [0.0, 1.0, 2.0], [0.0, 0.8, 0.4], "unit_area"),
    ],
)
def test_native_control_invariants(name, matrix, axis, expected, invariant):
    """Native controls keep their own scale; they are not min-max normalized."""
    block = np.asarray(matrix, dtype=float)
    increment = None if axis is None else np.asarray(axis, dtype=float)
    output, valid, _ = _apply(name, block, increment)
    assert valid.all()
    assert np.allclose(output[0], expected, rtol=0.0, atol=1e-12)
    if invariant == "standardized":
        assert np.isclose(float(output[0].mean()), 0.0, atol=1e-12)
        assert np.isclose(float(output[0].std()), 1.0, atol=1e-12)
    elif invariant == "unit_norm":
        assert np.isclose(float(np.linalg.norm(output[0])), 1.0, atol=1e-12)
    elif invariant == "unit_area":
        assert np.isclose(float(np.trapezoid(output[0], x=increment)), 1.0, atol=1e-12)
    else:  # pragma: no cover - guards against silent param drift
        raise AssertionError(f"unexpected invariant {invariant!r}")


@pytest.mark.parametrize(
    "name,reason",
    [
        ("minmax", "nonfinite_or_zero_range"),
        ("snv", "nonfinite_or_zero_scale"),
        ("vector", "nonfinite_or_zero_norm"),
        ("area", "nonfinite_or_zero_area"),
    ],
)
def test_all_zero_rows_are_invalid(name, reason):
    """Degenerate all-zero rows stay invalid; the affine theorem does not apply."""
    matrix = np.zeros((1, 4), dtype=float)
    axis = np.arange(4.0)
    _, valid, reasons = _apply(name, matrix, axis)
    assert not bool(valid[0])
    assert reasons == (reason,)


EVAL_AXIS = np.arange(400.0, 1801.0)  # 400..1800 inclusive, 1 cm step
FULL_SOURCE_AXIS = np.arange(400.0, 1850.0)  # 400..1849 inclusive, 1 cm step
PROCESSED_AXIS = np.arange(400.0, 1801.0)  # 400..1800 inclusive, 1 cm step
SOURCE_AXES = {"full": FULL_SOURCE_AXIS, "processed": PROCESSED_AXIS}


def _unsupported_shift_bins(source_axis, delta):
    """Count evaluation points whose shifted source coordinate is unavailable.

    Convention under test: ``y_shift(v) = y(v - delta)``. No padding,
    wrapping or interpolation is implemented here; this only reports
    coordinate membership.
    """
    required = EVAL_AXIS - float(delta)
    available = set(np.asarray(source_axis, dtype=float).tolist())
    return int(sum(1 for value in required.tolist() if value not in available))


@pytest.mark.parametrize(
    "source_name,delta,expected",
    [
        ("full", 5.0, 5),
        ("full", -5.0, 0),
        ("full", 0.0, 0),
        ("processed", 5.0, 5),
        ("processed", -5.0, 5),
    ],
)
def test_coordinate_support_counts(source_name, delta, expected):
    """Explicit expected counts for shifted-source support on fixed grids."""
    assert _unsupported_shift_bins(SOURCE_AXES[source_name], delta) == expected


def test_coordinate_grids_are_unit_spaced():
    """Both grids are fixed 1 cm grids with the documented endpoints."""
    assert EVAL_AXIS[0] == 400.0
    assert EVAL_AXIS[-1] == 1800.0
    assert np.all(np.diff(EVAL_AXIS) == 1.0)
    assert FULL_SOURCE_AXIS[0] == 400.0
    assert FULL_SOURCE_AXIS[-1] == 1849.0
    assert np.all(np.diff(FULL_SOURCE_AXIS) == 1.0)


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "plan" / "contracts" / "p01_governance_contract.json"
)


def _load_contract():
    with CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _representations(contract):
    return {item["representation_id"]: item for item in contract["representations"]}


@pytest.mark.parametrize(
    "representation_id,terminal_operation",
    [
        ("R_SNV_400_1800", "row_snv"),
        ("R_VECTOR_400_1800", "row_l2_norm"),
        ("R_AREA_400_1800", "integrated_area_norm"),
    ],
)
def test_normalization_controls_are_not_minmax(representation_id, terminal_operation):
    """Normalization controls must not silently end in a min-max step."""
    operations = list(_representations(_load_contract())[representation_id]["operations"])
    assert operations[-1] == terminal_operation
    assert "row_minmax" not in operations


def test_derivative_control_is_derivative_then_snv():
    """R_D1_400_1800 is a Savitzky-Golay derivative then SNV control."""
    spec = _representations(_load_contract())["R_D1_400_1800"]
    operations = list(spec["operations"])
    assert "savgol_derivative_11_3" in operations
    assert operations[-1] == "row_snv"
    assert "row_minmax" not in operations
    assert spec["scope"] == "destructive_control"


def test_primary_representations_end_with_row_minmax():
    """The three candidate representations end with an explicit row_minmax."""
    contract = _load_contract()
    specs = _representations(contract)
    candidate_ids = contract["downstream_preprocessing_policy_composition"][
        "candidate_action_representation_ids"
    ]
    assert candidate_ids == [
        "R_MIN_400_1800",
        "R_SG_400_1800",
        "R_ARPLS_400_1800",
    ]
    for representation_id in candidate_ids:
        assert list(specs[representation_id]["operations"])[-1] == "row_minmax"
    assert specs["R_MIN_400_1800"]["scope"] == "primary"
