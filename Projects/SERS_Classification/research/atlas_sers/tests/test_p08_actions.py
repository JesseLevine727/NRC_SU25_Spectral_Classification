import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation.p08_actions import (
    REQUIRED_ACTIONS,
    ActionAuditError,
    audit_frozen_actions,
)

_N = 2
_SENTINELS = ("SENTINEL_UID_ALPHA_7f3", "SENTINEL_UID_BETA_9c1")


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _base_axis() -> np.ndarray:
    return np.arange(400, 1801, dtype=np.float32)


def _base_intensity() -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, 1401, dtype=np.float32)
    return np.vstack([ramp, ramp[::-1].copy()])


def _base_actions() -> dict:
    axis = _base_axis()
    intensity = _base_intensity()
    uids = np.array(_SENTINELS, dtype="U")
    return {
        representation_id: {
            "axis_cm1": axis.copy(),
            "intensity": intensity.copy(),
            "observation_uid": uids.copy(),
        }
        for representation_id in REQUIRED_ACTIONS
    }


def _base_registry(actions: dict, uids=_SENTINELS) -> pd.DataFrame:
    row_order = hashlib.sha256("\n".join(uids).encode("utf-8")).hexdigest()
    records = [
        {
            "representation_id": representation_id,
            "rows": _N,
            "features": 1401,
            "dtype": "float32",
            "axis_start_cm1": 400,
            "axis_end_cm1": 1800,
            "axis_sha256": _digest(actions[representation_id]["axis_cm1"]),
            "array_sha256": _digest(actions[representation_id]["intensity"]),
            "row_order_sha256": row_order,
            "invalid_rows": 0,
            "invariant_status": "pass",
        }
        for representation_id in REQUIRED_ACTIONS
    ]
    return pd.DataFrame(records)


def _base_row_qc() -> pd.DataFrame:
    records = []
    for representation_id in REQUIRED_ACTIONS:
        for uid in reversed(_SENTINELS):
            records.append(
                {
                    "observation_uid": uid,
                    "representation_id": representation_id,
                    "valid": True,
                    "reason_code": "included",
                    "representation_invariant_status": "pass",
                }
            )
    return pd.DataFrame(records)


def _set(registry: pd.DataFrame, column: str, value) -> None:
    registry[column] = registry[column].astype(object)
    registry.loc[0, column] = value


_UNSET = object()


def _run(actions=_UNSET, registry=_UNSET, manifest=_UNSET, row_qc=_UNSET) -> dict:
    actions = _base_actions() if actions is _UNSET else actions
    registry = _base_registry(actions) if registry is _UNSET else registry
    manifest = list(_SENTINELS) if manifest is _UNSET else manifest
    row_qc = _base_row_qc() if row_qc is _UNSET else row_qc
    return audit_frozen_actions(actions, registry, manifest, row_qc)


def test_valid_result_json_roundtrip_and_no_mutation() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    row_qc = _base_row_qc()
    snapshots = {
        representation_id: {key: value.copy() for key, value in action.items()}
        for representation_id, action in actions.items()
    }
    registry_snapshot = registry.copy(deep=True)
    row_qc_snapshot = row_qc.copy(deep=True)
    result = audit_frozen_actions(actions, registry, list(_SENTINELS), row_qc)
    assert result["status"] == "pass"
    assert result["scope"] == "loaded_frozen_action_integrity_only"
    assert result["action_count"] == 3
    assert result["rows"] == _N
    assert result["features"] == 1401
    assert [record["representation_id"] for record in result["actions"]] == sorted(
        REQUIRED_ACTIONS
    )
    for record in result["actions"]:
        assert record["invalid_rows"] == 0
        assert record["normalization_passed"] is True
    assert json.loads(json.dumps(result)) == result
    for representation_id, action in actions.items():
        for key, value in action.items():
            assert np.array_equal(value, snapshots[representation_id][key])
    pd.testing.assert_frame_equal(registry, registry_snapshot)
    pd.testing.assert_frame_equal(row_qc, row_qc_snapshot)


def test_row_qc_shuffled_and_other_representations_accepted() -> None:
    row_qc = _base_row_qc().iloc[::-1].reset_index(drop=True)
    extra = pd.DataFrame(
        [
            {
                "observation_uid": "OTHER_UID_1",
                "representation_id": "R_OTHER_900_1800",
                "valid": True,
                "reason_code": "included",
                "representation_invariant_status": "pass",
            }
        ]
    )
    row_qc = pd.concat([row_qc, extra], ignore_index=True)
    assert _run(row_qc=row_qc)["status"] == "pass"


def test_registry_other_representation_accepted() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    extra = registry.iloc[[0]].copy()
    extra["representation_id"] = "R_EXTRA_LEGIT_900_1800"
    registry = pd.concat([registry, extra], ignore_index=True)
    assert _run(actions=actions, registry=registry)["action_count"] == 3


def test_missing_action_key_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    del actions["R_MIN_400_1800"]
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "actions_keys"


def test_npz_style_action_key_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    actions["arr_0"] = actions["R_MIN_400_1800"]
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "actions_keys"


def test_action_inner_key_error() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    del actions["R_SG_400_1800"]["intensity"]
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "action_keys"


def test_registry_duplicate_id_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    registry = pd.concat([registry, registry.iloc[[0]]], ignore_index=True)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "registry_duplicate"


def test_registry_missing_id_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    registry = registry.loc[registry["representation_id"] != "R_ARPLS_400_1800"]
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "registry_missing"


def test_registry_missing_column_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions).drop(columns=["array_sha256"])
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "registry_columns"


@pytest.mark.parametrize(
    ("column", "value", "code"),
    [
        ("rows", True, "registry_rows_invalid"),
        ("rows", 2.5, "registry_rows_invalid"),
        ("rows", float("nan"), "registry_rows_invalid"),
        ("rows", "2", "registry_rows_invalid"),
        ("features", True, "registry_features_invalid"),
        ("features", 1401.5, "registry_features_invalid"),
        ("invalid_rows", True, "registry_invalid_rows_invalid"),
        ("invalid_rows", 1, "registry_invalid_rows_invalid"),
        ("axis_start_cm1", True, "registry_bounds_invalid"),
        ("axis_end_cm1", float("inf"), "registry_bounds_invalid"),
        ("dtype", "float64", "registry_dtype_invalid"),
        ("invariant_status", "fail", "registry_invariant_status_invalid"),
    ],
)
def test_invalid_registry_metadata(column: str, value, code: str) -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    _set(registry, column, value)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == code


def test_registry_sha_format_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    registry.loc[1, "array_sha256"] = registry.loc[1, "array_sha256"].upper()
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "registry_sha_invalid"


def test_axis_dtype_and_values_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    actions["R_MIN_400_1800"]["axis_cm1"] = _base_axis().astype(np.float64)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "axis_invalid"
    actions = _base_actions()
    axis = _base_axis()
    axis[0] = np.float32(401.0)
    actions["R_SG_400_1800"]["axis_cm1"] = axis
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "axis_invalid"


def test_intensity_dtype_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    actions["R_ARPLS_400_1800"]["intensity"] = _base_intensity().astype(np.float64)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "intensity_invalid"


def test_array_hash_mismatch_even_if_normalized() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    intensity = actions["R_MIN_400_1800"]["intensity"]
    intensity[0, 1:] = np.float32(0.5)
    intensity[0, 0] = np.float32(0.0)
    intensity[0, -1] = np.float32(1.0)
    assert intensity[0].min() == 0.0
    assert intensity[0].max() == 1.0
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "array_sha_mismatch"


def test_axis_and_row_order_hash_mismatch_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    registry["axis_sha256"] = "0" * 64
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "axis_sha_mismatch"
    registry = _base_registry(actions)
    registry["row_order_sha256"] = "0" * 64
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "row_order_sha_mismatch"


def test_normalization_and_nonfinite_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    actions["R_MIN_400_1800"]["intensity"][0] *= np.float32(2.0)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "normalization_failed"
    actions = _base_actions()
    actions["R_SG_400_1800"]["intensity"][0, 0] = np.float32("nan")
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions)
    assert excinfo.value.reason_code == "intensity_nonfinite"


def test_uid_permutation_duplicate_and_nonstring_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    actions["R_MIN_400_1800"]["observation_uid"] = np.array(
        list(reversed(_SENTINELS)), dtype="U"
    )
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "uid_mismatch"
    actions = _base_actions()
    actions["R_SG_400_1800"]["observation_uid"] = np.array(
        [_SENTINELS[0], _SENTINELS[0]], dtype="U"
    )
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "uid_mismatch"
    actions = _base_actions()
    actions["R_ARPLS_400_1800"]["observation_uid"] = np.array([1, 2])
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "uid_invalid"
    actions = _base_actions()
    actions["R_MIN_400_1800"]["observation_uid"] = np.array(list(_SENTINELS), dtype=object)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "uid_invalid"


@pytest.mark.parametrize(
    "manifest",
    [None, (), [], [1, 2], ["a", "a"], ["a", ""], np.array([b"a", b"b"])],
)
def test_invalid_manifest_rejected(manifest) -> None:
    with pytest.raises(ActionAuditError) as excinfo:
        _run(manifest=manifest)
    assert excinfo.value.reason_code == "manifest_invalid"


def test_numpy_unicode_manifest_accepted() -> None:
    assert _run(manifest=np.array(_SENTINELS, dtype="U"))["status"] == "pass"


def test_row_qc_duplicate_and_missing_rejected() -> None:
    row_qc = _base_row_qc()
    row_qc = pd.concat([row_qc, row_qc.iloc[[0]]], ignore_index=True)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(row_qc=row_qc)
    assert excinfo.value.reason_code == "row_qc_coverage"
    row_qc = _base_row_qc()
    row_qc = row_qc.loc[row_qc["observation_uid"] != _SENTINELS[0]].reset_index(drop=True)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(row_qc=row_qc)
    assert excinfo.value.reason_code == "row_qc_coverage"


def test_row_qc_missing_column_rejected() -> None:
    row_qc = _base_row_qc().drop(columns=["reason_code"])
    with pytest.raises(ActionAuditError) as excinfo:
        _run(row_qc=row_qc)
    assert excinfo.value.reason_code == "row_qc_columns"


@pytest.mark.parametrize("value", [1, 0, 1.0, "True", "included"])
def test_row_qc_non_bool_valid_rejected(value) -> None:
    row_qc = _base_row_qc()
    row_qc["valid"] = row_qc["valid"].astype(object)
    row_qc.loc[0, "valid"] = value
    with pytest.raises(ActionAuditError) as excinfo:
        _run(row_qc=row_qc)
    assert excinfo.value.reason_code == "row_qc_valid"


def test_row_qc_false_and_reason_status_rejected() -> None:
    row_qc = _base_row_qc()
    row_qc["valid"] = np.bool_(False)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(row_qc=row_qc)
    assert excinfo.value.reason_code == "row_qc_valid"
    row_qc = _base_row_qc()
    row_qc.loc[0, "reason_code"] = "excluded"
    with pytest.raises(ActionAuditError) as excinfo:
        _run(row_qc=row_qc)
    assert excinfo.value.reason_code == "row_qc_reason_invalid"
    row_qc = _base_row_qc()
    row_qc.loc[0, "representation_invariant_status"] = "fail"
    with pytest.raises(ActionAuditError) as excinfo:
        _run(row_qc=row_qc)
    assert excinfo.value.reason_code == "row_qc_status_invalid"


def test_invalid_argument_types_rejected() -> None:
    actions = _base_actions()
    with pytest.raises(ActionAuditError) as excinfo:
        audit_frozen_actions(actions, {"bad": "registry"}, list(_SENTINELS), _base_row_qc())
    assert excinfo.value.reason_code == "registry_type"
    with pytest.raises(ActionAuditError) as excinfo:
        audit_frozen_actions(actions, _base_registry(actions), list(_SENTINELS), [1, 2])
    assert excinfo.value.reason_code == "row_qc_type"
    with pytest.raises(ActionAuditError) as excinfo:
        audit_frozen_actions([], _base_registry(actions), list(_SENTINELS), _base_row_qc())
    assert excinfo.value.reason_code == "actions_type"


def test_output_and_errors_do_not_reveal_uids() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    result = audit_frozen_actions(actions, registry, list(_SENTINELS), _base_row_qc())
    serialized = json.dumps(result)
    for sentinel in _SENTINELS:
        assert sentinel not in serialized
    actions["R_MIN_400_1800"]["observation_uid"] = np.array(
        [_SENTINELS[0], _SENTINELS[0]], dtype="U"
    )
    with pytest.raises(ActionAuditError) as excinfo:
        audit_frozen_actions(actions, registry, list(_SENTINELS), _base_row_qc())
    for sentinel in _SENTINELS:
        assert sentinel not in str(excinfo.value)
        assert sentinel not in repr(excinfo.value)
    assert excinfo.value.reason_code == "uid_mismatch"


def test_registry_duplicate_columns_rejected() -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    duplicated_required = pd.concat([registry, registry[["representation_id"]]], axis=1)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=duplicated_required)
    assert excinfo.value.reason_code == "registry_columns"

    registry = _base_registry(actions)
    registry["extra_metric"] = 1
    duplicated_extra = pd.concat([registry, registry[["extra_metric"]]], axis=1)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=duplicated_extra)
    assert excinfo.value.reason_code == "registry_columns"


def test_row_qc_duplicate_columns_rejected() -> None:
    row_qc = _base_row_qc()
    duplicated_required = pd.concat([row_qc, row_qc[["reason_code"]]], axis=1)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(row_qc=duplicated_required)
    assert excinfo.value.reason_code == "row_qc_columns"

    row_qc = _base_row_qc()
    row_qc["extra_metric"] = 1
    duplicated_extra = pd.concat([row_qc, row_qc[["extra_metric"]]], axis=1)
    with pytest.raises(ActionAuditError) as excinfo:
        _run(row_qc=duplicated_extra)
    assert excinfo.value.reason_code == "row_qc_columns"


_MALFORMED_REPRESENTATION_IDS = (None, float("nan"), 1, ["R_LIST"], pd.NA, "", "   ")


@pytest.mark.parametrize("malformed", _MALFORMED_REPRESENTATION_IDS)
def test_registry_requested_representation_id_invalid(malformed) -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    registry["representation_id"] = registry["representation_id"].astype(object)
    registry.at[0, "representation_id"] = malformed
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "registry_representation_id_invalid"


@pytest.mark.parametrize("malformed", _MALFORMED_REPRESENTATION_IDS)
def test_registry_extra_representation_id_invalid(malformed) -> None:
    actions = _base_actions()
    registry = _base_registry(actions)
    extra = registry.iloc[[0]].copy()
    extra["representation_id"] = "R_EXTRA_LEGIT_900_1800"
    registry = pd.concat([registry, extra], ignore_index=True)
    registry["representation_id"] = registry["representation_id"].astype(object)
    registry.at[len(registry) - 1, "representation_id"] = malformed
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions, registry=registry)
    assert excinfo.value.reason_code == "registry_representation_id_invalid"


@pytest.mark.parametrize("malformed", _MALFORMED_REPRESENTATION_IDS)
def test_row_qc_requested_representation_id_invalid(malformed) -> None:
    row_qc = _base_row_qc()
    row_qc["representation_id"] = row_qc["representation_id"].astype(object)
    row_qc.at[0, "representation_id"] = malformed
    with pytest.raises(ActionAuditError) as excinfo:
        _run(row_qc=row_qc)
    assert excinfo.value.reason_code == "row_qc_representation_id_invalid"


@pytest.mark.parametrize("malformed", _MALFORMED_REPRESENTATION_IDS)
def test_row_qc_extra_representation_id_invalid(malformed) -> None:
    row_qc = _base_row_qc()
    extra = row_qc.iloc[[0]].copy()
    extra["representation_id"] = "R_OTHER_900_1800"
    row_qc = pd.concat([row_qc, extra], ignore_index=True)
    row_qc["representation_id"] = row_qc["representation_id"].astype(object)
    row_qc.at[len(row_qc) - 1, "representation_id"] = malformed
    with pytest.raises(ActionAuditError) as excinfo:
        _run(row_qc=row_qc)
    assert excinfo.value.reason_code == "row_qc_representation_id_invalid"


def test_row_qc_extra_representation_metadata_ignored() -> None:
    row_qc = _base_row_qc()
    extra = row_qc.iloc[[0]].copy()
    extra["representation_id"] = "R_OTHER_900_1800"
    extra["valid"] = False
    extra["reason_code"] = "excluded"
    extra["representation_invariant_status"] = "fail"
    row_qc = pd.concat([row_qc, extra], ignore_index=True)
    assert _run(row_qc=row_qc)["status"] == "pass"


def test_registry_bounds_overflow_rejected() -> None:
    actions = _base_actions()
    for column in ("axis_start_cm1", "axis_end_cm1"):
        registry = _base_registry(actions)
        _set(registry, column, 10**400)
        with pytest.raises(ActionAuditError) as excinfo:
            _run(actions=actions, registry=registry)
        assert excinfo.value.reason_code == "registry_bounds_invalid"


@pytest.mark.parametrize(
    "intensity",
    [
        _base_intensity().astype(np.complex64),
        _base_intensity().astype(object),
        _base_intensity().astype(np.int32),
        np.float32(1.0),
    ],
)
def test_intensity_dtype_and_shape_rejected(intensity) -> None:
    actions = _base_actions()
    actions["R_MIN_400_1800"]["intensity"] = intensity
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions)
    assert excinfo.value.reason_code == "intensity_invalid"


@pytest.mark.parametrize(
    "axis",
    [
        np.arange(400, 1800, dtype=np.float32),
        _base_axis().reshape(1, -1),
        np.float32(400.0),
    ],
)
def test_axis_shape_rejected(axis) -> None:
    actions = _base_actions()
    actions["R_SG_400_1800"]["axis_cm1"] = axis
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions)
    assert excinfo.value.reason_code == "axis_invalid"


@pytest.mark.parametrize(
    "uids",
    [
        np.array([_SENTINELS[0]], dtype="U"),
        np.array(_SENTINELS, dtype="U").reshape(1, -1),
        np.array(_SENTINELS * 2, dtype="U"),
    ],
)
def test_uid_shape_rejected(uids) -> None:
    actions = _base_actions()
    actions["R_ARPLS_400_1800"]["observation_uid"] = uids
    with pytest.raises(ActionAuditError) as excinfo:
        _run(actions=actions)
    assert excinfo.value.reason_code == "uid_invalid"
