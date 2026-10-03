"""Synthetic, in-memory tests for the P08-T099 U0 source-array preparation.

Authenticated metadata bytes come from the sibling ``test_p08_u0_inputs``
fixture module; three invented NPZ archives are pinned through
``atlas_sers.evaluation.p08_u0_arrays._ACTION_PINS``.  Nothing here touches
the real dataset, the filesystem or any scientific execution path.
"""

from __future__ import annotations

import builtins
import csv
import hashlib
import io
import json
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.lib import format as npformat

from atlas_sers.evaluation import p08_u0_arrays as subject
from atlas_sers.evaluation import p08_u0_inputs
from tests import test_p08_u0_inputs as metadata_fixture

ARG_NAMES = metadata_fixture.ARG_NAMES
SENTINEL = metadata_fixture.SENTINEL
ACTIONS = ("R_MIN_400_1800", "R_SG_400_1800", "R_ARPLS_400_1800")
POLICY_ACTION = {"PP-U-SG": "R_SG_400_1800", "PP-U-ARPLS": "R_ARPLS_400_1800"}
ACTION_MARKER = {"R_MIN_400_1800": 0.11, "R_SG_400_1800": 0.22, "R_ARPLS_400_1800": 0.33}
QC_COLUMNS = ("first_difference_noise_mad", "intensity_range")
AXIS = np.arange(400, 1801, dtype=np.float32)
MEMBER_AXIS = "axis_cm1.npy"
MEMBER_INTENSITY = "intensity.npy"
MEMBER_UID = "observation_uid.npy"
MODE_ACTIONS = ("master_cv", "pseudo_domain")


def _sha(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def _qc_values(observation_uid):
    digest = int(hashlib.sha256(observation_uid.encode("utf-8")).hexdigest(), 16)
    mad = round(0.0001 + (digest % 9000) / 1e6, 8)
    span = round(0.0100 + ((digest >> 40) % 9000) / 1e6, 8)
    return mad, span


def _augment_manifest(manifest_bytes, qc_hook=None, qc_columns=QC_COLUMNS):
    reader = csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8")))
    fields = list(reader.fieldnames)
    rows = list(reader)
    uids = [row["observation_uid"] for row in rows]
    for row in rows:
        mad, span = _qc_values(row["observation_uid"])
        if qc_hook is not None:
            mad, span = qc_hook(row["observation_uid"], mad, span)
        row["first_difference_noise_mad"] = mad
        row["intensity_range"] = span
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields + list(qc_columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return out.getvalue().encode("utf-8"), uids


def _uid_array(uids):
    width = max(1, max(len(uid) for uid in uids))
    return np.asarray(uids, dtype=f"<U{width}")


def _build_intensity(action, uids):
    axis = AXIS.copy()
    intensity = np.empty((len(uids), axis.size), dtype=np.float32)
    marker = np.float32(ACTION_MARKER[action])
    for index in range(len(uids)):
        row = np.full(axis.size, np.float32(0.2 + (index % 7) * 0.01), dtype=np.float32)
        row[0] = 0.0
        row[-1] = 1.0
        position = 1 + (index * 37) % (axis.size - 2)
        row[position] = np.float32(min(0.99, max(0.01, marker + (index % 3) * 0.01)))
        intensity[index] = row
    return axis, intensity


def _npy_bytes(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _zip_bytes(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members:
            archive.writestr(name, data)
    return buffer.getvalue()


def _savez_bytes(axis, intensity, uid_array):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, axis_cm1=axis, intensity=intensity, observation_uid=uid_array)
    return buffer.getvalue()


def _huge_header_bytes(shape=(10**9, 1401)):
    buffer = io.BytesIO()
    npformat.write_array_header_1_0(
        buffer, {"descr": "<f4", "fortran_order": False, "shape": shape}
    )
    return buffer.getvalue()


def _pins(payload, intensity, axis, uids):
    ordered = [str(uid) for uid in uids]
    return {
        "file_sha256": _sha(payload),
        "array_sha256": _sha(np.ascontiguousarray(intensity).tobytes(order="C")),
        "axis_sha256": _sha(np.ascontiguousarray(axis).tobytes(order="C")),
        "row_order_sha256": _sha("\n".join(ordered).encode("utf-8")),
    }


def _default_assets(uids):
    action_bytes = {}
    pins = {}
    arrays = {}
    for action in ACTIONS:
        axis, intensity = _build_intensity(action, uids)
        uid_array = _uid_array(uids)
        payload = _savez_bytes(axis, intensity, uid_array)
        action_bytes[action] = payload
        pins[action] = _pins(payload, intensity, axis, uids)
        arrays[action] = (axis, intensity, uid_array)
    return action_bytes, pins, arrays


def _assets_with(uids, action, transform):
    action_bytes, pins, arrays = _default_assets(uids)
    axis, intensity, uid_array = arrays[action]
    payload, new_intensity, new_axis, new_uid = transform(
        axis, intensity, uid_array, action_bytes[action]
    )
    action_bytes[action] = payload
    pins[action] = _pins(payload, new_intensity, new_axis, new_uid)
    arrays[action] = (new_axis, new_intensity, new_uid)
    return action_bytes, pins, arrays


def _assets_file_only(uids, action, transform):
    action_bytes, pins, arrays = _default_assets(uids)
    axis, intensity, uid_array = arrays[action]
    payload, new_intensity, new_axis, new_uid = transform(
        axis, intensity, uid_array, action_bytes[action]
    )
    action_bytes[action] = payload
    pins[action] = {**pins[action], "file_sha256": _sha(payload)}
    arrays[action] = (new_axis, new_intensity, new_uid)
    return action_bytes, pins, arrays


def _build_case(
    monkeypatch,
    mode,
    asset_hook=None,
    pair_mutator_extra=None,
    qc_hook=None,
    qc_columns=QC_COLUMNS,
    **case_kwargs,
):
    base = metadata_fixture.make_case(mode, **case_kwargs)
    manifest, uids = _augment_manifest(
        base["manifest_bytes"], qc_hook=qc_hook, qc_columns=qc_columns
    )
    action_bytes, pins, arrays = _default_assets(uids)
    if asset_hook is not None:
        action_bytes, pins, arrays = asset_hook(uids)

    def pair_mutator(common, plan):
        common["array_sha256"] = pins[POLICY_ACTION[common["policy_id"]]]["array_sha256"]
        if pair_mutator_extra is not None:
            pair_mutator_extra(common, plan)

    final = metadata_fixture.make_case(mode, pair_mutator=pair_mutator, **case_kwargs)
    final["manifest_bytes"] = manifest
    metadata_fixture._seal(monkeypatch, final)
    monkeypatch.setattr(subject, "_ACTION_PINS", pins, raising=True)
    return {
        "case": final,
        "metadata_bytes": {name: final[name] for name in ARG_NAMES},
        "action_bytes": action_bytes,
        "pins": pins,
        "arrays": arrays,
        "uids": uids,
        "manifest_bytes": manifest,
    }


def _prepare(monkeypatch, mode, **kwargs):
    ctx = _build_case(monkeypatch, mode, **kwargs)
    binding = subject.prepare_u0_source_arrays(
        metadata_bytes=ctx["metadata_bytes"], action_bytes=ctx["action_bytes"]
    )
    return ctx, binding


def _expect_error(monkeypatch, mode, **kwargs):
    ctx = _build_case(monkeypatch, mode, **kwargs)
    with pytest.raises(subject.ArrayInputError) as info:
        subject.prepare_u0_source_arrays(
            metadata_bytes=ctx["metadata_bytes"], action_bytes=ctx["action_bytes"]
        )
    error = info.value
    assert isinstance(error.reason_code, str) and error.reason_code
    assert SENTINEL not in str(error)
    assert "/" not in str(error)
    for uid in ctx["uids"]:
        assert uid not in str(error)
    return error, ctx


def _expected_rows(ctx, action, uids):
    _, intensity, _ = ctx["arrays"][action]
    lookup = {uid: index for index, uid in enumerate(ctx["uids"])}
    return np.stack([intensity[lookup[uid]] for uid in uids]).astype(np.float32)


def _expected_noise(manifest_bytes, fit_uids):
    frame = (
        pd.read_csv(
            io.BytesIO(manifest_bytes),
            usecols=["observation_uid", *QC_COLUMNS],
            keep_default_na=False,
        )
        .set_index("observation_uid")
        .loc[list(fit_uids)]
        .reset_index()
    )
    frame = frame[["observation_uid", *QC_COLUMNS]].copy()
    for column in QC_COLUMNS:
        frame[column] = frame[column].astype(float)
    return frame


def _uids_by_role(mode):
    base = metadata_fixture.make_case(mode)
    source = [row["observation_uid"] for row in base["rows"] if not row["_is_test"]]
    held = [row["observation_uid"] for row in base["rows"] if row["_is_test"]]
    return source, held


def _install_parse_spies(monkeypatch):
    counters = {"zip": 0, "load": 0, "read": 0}
    real_zip = zipfile.ZipFile

    class CountingZip(real_zip):
        def __init__(self, *args, **kwargs):
            counters["zip"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(zipfile, "ZipFile", CountingZip)
    real_load = np.load

    def counting_load(*args, **kwargs):
        counters["load"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counting_load)
    real_read = npformat.read_array

    def counting_read(*args, **kwargs):
        counters["read"] += 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(npformat, "read_array", counting_read)
    return counters


def test_registered_action_constants():
    assert tuple(POLICY_ACTION.values()) == ("R_SG_400_1800", "R_ARPLS_400_1800")
    assert set(ACTIONS) == set(ACTION_MARKER)
    assert AXIS.shape == (1401,)
    assert AXIS.dtype == np.float32


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
def test_read_plain_action_array_supported_header_versions(version):
    data = np.arange(1401, dtype=np.float32)
    buffer = io.BytesIO()
    npformat.write_array(buffer, data, version=version, allow_pickle=False)

    result = subject._read_plain_action_array(
        buffer.getvalue(), expected_shape=(1401,), unicode_member=False
    )

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, data)
    assert result.dtype == np.float32


def test_read_plain_action_array_rejects_version3():
    data = np.arange(1401, dtype=np.float32)
    buffer = io.BytesIO()
    npformat.write_array(buffer, data, version=(3, 0), allow_pickle=False)

    with pytest.raises(subject.ArrayInputError) as info:
        subject._read_plain_action_array(
            buffer.getvalue(), expected_shape=(1401,), unicode_member=False
        )

    assert info.value.reason_code == "invalid_action_array"


def test_metadata_binder_invoked_once(monkeypatch):
    calls = {"count": 0}
    real = p08_u0_inputs.bind_u0_source_metadata

    def counting(*args, **kwargs):
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(p08_u0_inputs, "bind_u0_source_metadata", counting)
    _, binding = _prepare(monkeypatch, "master_cv")
    assert calls["count"] == 1
    assert len(binding.pairs) == 78


@pytest.mark.parametrize("mode", MODE_ACTIONS)
def test_positive_report_counts_and_flags(monkeypatch, mode):
    ctx, binding = _prepare(monkeypatch, mode)
    report = binding.public_report()
    assert len(binding.pairs) == 78
    assert report["manifest_rows"] == len(ctx["uids"])
    assert report["features"] == 1401
    assert report["authenticated_action_count"] == 3
    assert report["prepared_policy_count"] == 2
    assert report["prepared_source_roles"] == 6
    assert report["source_fit_jobs"] == 78
    assert report["source_prediction_jobs"] == 78
    assert len(report["action_pins"]) == 3
    independent = p08_u0_inputs.bind_u0_source_metadata(**ctx["metadata_bytes"])
    assert report["metadata_binding_report_sha256"] == independent.public_report()["report_sha256"]
    assert report["arrays_verified"] is True
    assert report["ordered_source_rows_verified"] is True
    assert report["native_noise_metadata_preserved"] is True
    assert report["noise_quantiles_computed"] is False
    assert report["preprocessing_recomputed"] is False
    assert report["model_parameters_loaded"] is False
    assert report["live_controller_verified"] is False
    assert report["execution_authorized"] is False
    assert report["new_scientific_operations"] == 0


def test_global_order_differs_from_canonical(monkeypatch):
    ctx, _ = _prepare(monkeypatch, "master_cv")
    assert ctx["uids"] != sorted(ctx["uids"])


@pytest.mark.parametrize("mode", MODE_ACTIONS)
def test_fitting_and_validation_values_are_exact(monkeypatch, mode):
    ctx, binding = _prepare(monkeypatch, mode)
    for pair in binding.pairs:
        roles = pair.inputs.source_roles
        action = pair.inputs.representation_id
        fit_uids = [obs.observation_uid for obs in roles.fitting]
        val_uids = [obs.observation_uid for obs in roles.validation]
        fit_values = pair.inputs.fitting_values()
        val_values = pair.inputs.validation_values()
        assert fit_values.dtype == np.float32
        assert fit_values.flags["C_CONTIGUOUS"] is True
        assert fit_values.shape == (len(fit_uids), 1401)
        assert np.array_equal(fit_values, _expected_rows(ctx, action, fit_uids))
        assert np.array_equal(val_values, _expected_rows(ctx, action, val_uids))


@pytest.mark.parametrize("mode", MODE_ACTIONS)
def test_returned_values_only_reference_source_roles(monkeypatch, mode):
    ctx, binding = _prepare(monkeypatch, mode)
    source = set(_uids_by_role(mode)[0])
    held = set(_uids_by_role(mode)[1])
    seen = set()
    for pair in binding.pairs:
        roles = pair.inputs.source_roles
        for obs in tuple(roles.fitting) + tuple(roles.validation):
            seen.add(obs.observation_uid)
    assert seen <= source
    assert not (seen & held)


@pytest.mark.parametrize("mode", MODE_ACTIONS)
def test_noise_frame_is_exact_source_frame(monkeypatch, mode):
    ctx, binding = _prepare(monkeypatch, mode)
    for pair in binding.pairs:
        roles = pair.inputs.source_roles
        fit_uids = [obs.observation_uid for obs in roles.fitting]
        frame = pair.inputs.fitting_noise_frame()
        expected = _expected_noise(ctx["manifest_bytes"], fit_uids)
        assert list(frame.columns) == ["observation_uid", *QC_COLUMNS]
        assert frame["observation_uid"].tolist() == fit_uids
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected.reset_index(drop=True))


def test_noise_frame_holds_only_fitting_rows(monkeypatch):
    _, binding = _prepare(monkeypatch, "master_cv")
    for pair in binding.pairs[:6]:
        roles = pair.inputs.source_roles
        frame = pair.inputs.fitting_noise_frame()
        assert set(frame["observation_uid"]) == {o.observation_uid for o in roles.fitting}


def test_noise_identical_across_policies_but_values_differ(monkeypatch):
    _, binding = _prepare(monkeypatch, "master_cv")
    sg = next(
        p
        for p in binding.pairs
        if p.inputs.policy_id == "PP-U-SG" and p.inputs.source_roles.unit_id == "master_cv:0"
    )
    ar = next(
        p
        for p in binding.pairs
        if p.inputs.policy_id == "PP-U-ARPLS" and p.inputs.source_roles.unit_id == "master_cv:0"
    )
    assert sg.inputs is not ar.inputs
    pd.testing.assert_frame_equal(sg.inputs.fitting_noise_frame(), ar.inputs.fitting_noise_frame())
    assert not np.array_equal(sg.inputs.fitting_values(), ar.inputs.fitting_values())


def test_noise_frame_is_fresh_and_mutation_isolated(monkeypatch):
    _, binding = _prepare(monkeypatch, "master_cv")
    pair = binding.pairs[0]
    first = pair.inputs.fitting_noise_frame()
    baseline = first.copy(deep=True)
    first.loc[first.index[0], "first_difference_noise_mad"] = -999.0
    second = pair.inputs.fitting_noise_frame()
    pd.testing.assert_frame_equal(second, baseline)


def test_inputs_shared_per_policy_context_unit(monkeypatch):
    _, binding = _prepare(monkeypatch, "master_cv")
    seen = {}
    for pair in binding.pairs:
        roles = pair.inputs.source_roles
        key = (pair.inputs.policy_id, roles.context_id, roles.unit_id)
        seen.setdefault(key, pair.inputs)
        assert pair.inputs is seen[key]
    assert len(seen) == 6


def test_returned_arrays_are_read_only(monkeypatch):
    _, binding = _prepare(monkeypatch, "master_cv")
    pair = binding.pairs[0]
    for values in (pair.inputs.fitting_values(), pair.inputs.validation_values()):
        assert values.flags.writeable is False
        with pytest.raises(ValueError):
            values.setflags(write=True)
        with pytest.raises(ValueError):
            values[0, 0] = np.float32(0.5)


def test_frozen_binding_fields_reject_mutation(monkeypatch):
    _, binding = _prepare(monkeypatch, "master_cv")
    pair = binding.pairs[0]
    roles = pair.inputs.source_roles
    for action in (
        lambda: setattr(binding, "pairs", ()),
        lambda: setattr(pair, "fit_job_json", "x"),
        lambda: setattr(pair, "prediction_job_json", "x"),
        lambda: setattr(pair, "inputs", None),
        lambda: setattr(pair.inputs, "policy_id", "x"),
        lambda: setattr(pair.inputs, "representation_id", "x"),
        lambda: setattr(roles, "unit_id", "x"),
        lambda: setattr(roles, "fitting", ()),
        lambda: setattr(roles, "validation", ()),
    ):
        with pytest.raises(FrozenInstanceError):
            action()


def test_job_json_projects_source_stages_and_array_hash(monkeypatch):
    ctx, binding = _prepare(monkeypatch, "master_cv")
    for pair in binding.pairs:
        fit = json.loads(pair.fit_job_json)
        prediction = json.loads(pair.prediction_job_json)
        assert fit["stage"] == "source_fit"
        assert prediction["stage"] == "source_validation_prediction"
        assert prediction["dependencies"] == [fit["job_id"]]
        action = pair.inputs.representation_id
        assert fit["array_sha256"] == ctx["pins"][action]["array_sha256"]
        assert prediction["array_sha256"] == ctx["pins"][action]["array_sha256"]


def test_input_bytes_unchanged(monkeypatch):
    ctx = _build_case(monkeypatch, "master_cv")
    metadata = dict(ctx["metadata_bytes"])
    actions = dict(ctx["action_bytes"])
    subject.prepare_u0_source_arrays(
        metadata_bytes=ctx["metadata_bytes"], action_bytes=ctx["action_bytes"]
    )
    for name in ARG_NAMES:
        assert ctx["metadata_bytes"][name] == metadata[name]
        assert isinstance(ctx["metadata_bytes"][name], bytes)
    for action in ACTIONS:
        assert ctx["action_bytes"][action] == actions[action]


def test_public_report_is_fresh_and_digest_matches(monkeypatch):
    _, binding = _prepare(monkeypatch, "master_cv")
    first = binding.public_report()
    second = binding.public_report()
    assert first is not second
    assert first == second
    without = {key: value for key, value in first.items() if key != "report_sha256"}
    from atlas_sers.governance.canonical import sha256_value

    assert first["report_sha256"] == sha256_value(without)
    first["tampered"] = True
    assert binding.public_report() == second


def test_public_report_hides_identifiers(monkeypatch):
    ctx, binding = _prepare(monkeypatch, "master_cv")
    rendered = json.dumps(binding.public_report())
    for needle in (SENTINEL, "OBS-", "ctx-", "P04ROLE-"):
        assert needle not in rendered
    for uid in ctx["uids"]:
        assert uid not in rendered
    report = binding.public_report()
    for _key, value in report.items():
        if isinstance(value, str):
            assert "\n" not in value


def test_private_repr_hides_identifiers(monkeypatch):
    ctx, binding = _prepare(monkeypatch, "master_cv")
    rendered = "".join(repr(part) for part in (binding,) + tuple(binding.pairs))
    rendered += "".join(repr(pair.inputs) for pair in binding.pairs)
    for uid in ctx["uids"]:
        assert uid not in rendered
    for row in ctx["case"]["rows"]:
        assert row["master_sample_id"] not in rendered
    assert SENTINEL not in rendered
    assert "P04ROLE-" not in rendered


def test_require_scientific_execution_always_denies():
    calls = (
        lambda: subject.require_scientific_execution(),
        lambda: subject.require_scientific_execution(execution_authorized=True),
    )
    for call in calls:
        with pytest.raises(subject.ArrayInputError) as info:
            call()
        assert info.value.reason_code == "scientific_execution_not_authorized"


@pytest.mark.parametrize("action", ACTIONS)
def test_action_byte_tamper_rejected_before_parse(monkeypatch, action):
    ctx = _build_case(monkeypatch, "master_cv")
    tampered = dict(ctx["action_bytes"])
    data = bytearray(tampered[action])
    data[0] ^= 0xFF
    tampered[action] = bytes(data)
    counters = _install_parse_spies(monkeypatch)
    with pytest.raises(subject.ArrayInputError):
        subject.prepare_u0_source_arrays(
            metadata_bytes=ctx["metadata_bytes"], action_bytes=tampered
        )
    assert counters == {"zip": 0, "load": 0, "read": 0}


def test_missing_action_key_rejected(monkeypatch):
    ctx = _build_case(monkeypatch, "master_cv")
    reduced = {key: value for key, value in ctx["action_bytes"].items() if key != ACTIONS[0]}
    with pytest.raises(subject.ArrayInputError):
        subject.prepare_u0_source_arrays(metadata_bytes=ctx["metadata_bytes"], action_bytes=reduced)


def test_extra_action_key_rejected(monkeypatch):
    ctx = _build_case(monkeypatch, "master_cv")
    extended = {**ctx["action_bytes"], "R_EXTRA": b"junk"}
    with pytest.raises(subject.ArrayInputError):
        subject.prepare_u0_source_arrays(
            metadata_bytes=ctx["metadata_bytes"], action_bytes=extended
        )


def test_wrong_action_payload_rejected(monkeypatch):
    ctx = _build_case(monkeypatch, "master_cv")
    swapped = dict(ctx["action_bytes"])
    swapped[ACTIONS[1]], swapped[ACTIONS[2]] = swapped[ACTIONS[2]], swapped[ACTIONS[1]]
    with pytest.raises(subject.ArrayInputError):
        subject.prepare_u0_source_arrays(metadata_bytes=ctx["metadata_bytes"], action_bytes=swapped)


def test_mutable_action_buffer_rejected(monkeypatch):
    ctx = _build_case(monkeypatch, "master_cv")
    buffered = dict(ctx["action_bytes"])
    buffered[ACTIONS[1]] = bytearray(buffered[ACTIONS[1]])
    with pytest.raises(subject.ArrayInputError):
        subject.prepare_u0_source_arrays(
            metadata_bytes=ctx["metadata_bytes"], action_bytes=buffered
        )


def test_bad_metadata_pin_rejected(monkeypatch):
    ctx = _build_case(monkeypatch, "master_cv")
    tampered = dict(ctx["metadata_bytes"])
    data = bytearray(tampered["manifest_bytes"])
    data[0] ^= 0xFF
    tampered["manifest_bytes"] = bytes(data)
    with pytest.raises(subject.ArrayInputError):
        subject.prepare_u0_source_arrays(metadata_bytes=tampered, action_bytes=ctx["action_bytes"])


def _tfm_wrong_shape(axis, intensity, uid_array, payload):
    trimmed = np.ascontiguousarray(intensity[:, :1400], dtype=np.float32)
    return _savez_bytes(axis, trimmed, uid_array), trimmed, axis, uid_array


def _tfm_wrong_dtype(axis, intensity, uid_array, payload):
    widened = np.ascontiguousarray(intensity, dtype=np.float64)
    return _savez_bytes(axis, widened, uid_array), widened, axis, uid_array


def _tfm_object_dtype(axis, intensity, uid_array, payload):
    boxed = np.empty(intensity.shape, dtype=object)
    boxed[:] = intensity
    return _savez_bytes(axis, boxed, uid_array), boxed, axis, uid_array


def _tfm_nonfinite(axis, intensity, uid_array, payload):
    broken = intensity.copy()
    broken[0, 1] = np.float32(np.nan)
    return _savez_bytes(axis, broken, uid_array), broken, axis, uid_array


def _tfm_out_of_range(axis, intensity, uid_array, payload):
    broken = intensity.copy()
    broken[0, 1] = np.float32(1.5)
    return _savez_bytes(axis, broken, uid_array), broken, axis, uid_array


def _tfm_bad_row_extrema(axis, intensity, uid_array, payload):
    broken = intensity.copy()
    broken[0] = np.float32(0.5)
    return _savez_bytes(axis, broken, uid_array), broken, axis, uid_array


def _tfm_wrong_axis(axis, intensity, uid_array, payload):
    broken = axis.copy()
    broken[0] = np.float32(399.0)
    return _savez_bytes(broken, intensity, uid_array), intensity, broken, uid_array


def _tfm_reordered_uids(axis, intensity, uid_array, payload):
    reordered = uid_array[::-1].copy()
    return _savez_bytes(axis, intensity, reordered), intensity, axis, reordered


def _tfm_content_hash(axis, intensity, uid_array, payload):
    changed = intensity.copy()
    changed[0, 1] = np.float32(0.123)
    return _savez_bytes(axis, changed, uid_array), changed, axis, uid_array


def _tfm_corrupt_zip(axis, intensity, uid_array, payload):
    return b"not-a-zip-archive", intensity, axis, uid_array


def _tfm_extra_member(axis, intensity, uid_array, payload):
    members = [
        (MEMBER_AXIS, _npy_bytes(axis)),
        (MEMBER_INTENSITY, _npy_bytes(intensity)),
        (MEMBER_UID, _npy_bytes(uid_array)),
        ("extra.npy", _npy_bytes(np.zeros(3, dtype=np.float32))),
    ]
    return _zip_bytes(members), intensity, axis, uid_array


def _tfm_missing_member(axis, intensity, uid_array, payload):
    members = [
        (MEMBER_AXIS, _npy_bytes(axis)),
        (MEMBER_INTENSITY, _npy_bytes(intensity)),
    ]
    return _zip_bytes(members), intensity, axis, uid_array


def _tfm_duplicate_member(axis, intensity, uid_array, payload):
    members = [
        (MEMBER_AXIS, _npy_bytes(axis)),
        (MEMBER_INTENSITY, _npy_bytes(intensity)),
        (MEMBER_INTENSITY, _npy_bytes(intensity)),
        (MEMBER_UID, _npy_bytes(uid_array)),
    ]
    return _zip_bytes(members), intensity, axis, uid_array


def _tfm_huge_header(axis, intensity, uid_array, payload):
    members = [
        (MEMBER_AXIS, _npy_bytes(axis)),
        (MEMBER_INTENSITY, _huge_header_bytes()),
        (MEMBER_UID, _npy_bytes(uid_array)),
    ]
    return _zip_bytes(members), intensity, axis, uid_array


MALFORMED_ASSETS = {
    "shape": lambda uids: _assets_with(uids, ACTIONS[1], _tfm_wrong_shape),
    "dtype": lambda uids: _assets_with(uids, ACTIONS[1], _tfm_wrong_dtype),
    "object": lambda uids: _assets_with(uids, ACTIONS[1], _tfm_object_dtype),
    "nonfinite": lambda uids: _assets_with(uids, ACTIONS[1], _tfm_nonfinite),
    "range": lambda uids: _assets_with(uids, ACTIONS[1], _tfm_out_of_range),
    "extrema": lambda uids: _assets_with(uids, ACTIONS[1], _tfm_bad_row_extrema),
    "axis": lambda uids: _assets_with(uids, ACTIONS[1], _tfm_wrong_axis),
    "uids": lambda uids: _assets_with(uids, ACTIONS[1], _tfm_reordered_uids),
    "content": lambda uids: _assets_file_only(uids, ACTIONS[1], _tfm_content_hash),
    "corrupt": lambda uids: _assets_file_only(uids, ACTIONS[1], _tfm_corrupt_zip),
    "extra": lambda uids: _assets_file_only(uids, ACTIONS[1], _tfm_extra_member),
    "missing": lambda uids: _assets_file_only(uids, ACTIONS[1], _tfm_missing_member),
    "duplicate": lambda uids: _assets_file_only(uids, ACTIONS[1], _tfm_duplicate_member),
}


MALFORMED_REASONS = {
    "shape": "action_array_shape_mismatch",
    "dtype": "action_array_dtype_mismatch",
    "object": "action_array_dtype_mismatch",
    "nonfinite": "intensity_nonfinite",
    "range": "normalization_failed",
    "extrema": "normalization_failed",
    "axis": "axis_invalid",
    "uids": "uid_mismatch",
    "content": "array_sha_mismatch",
    "corrupt": "invalid_action_archive",
    "extra": "action_member_mismatch",
    "missing": "action_member_mismatch",
    "duplicate": "action_member_mismatch",
}


@pytest.mark.parametrize("case_id", sorted(MALFORMED_ASSETS))
def test_malformed_action_archives_rejected(monkeypatch, case_id):
    if case_id == "duplicate":
        with pytest.warns(UserWarning):
            error, _ = _expect_error(monkeypatch, "master_cv", asset_hook=MALFORMED_ASSETS[case_id])
    else:
        error, _ = _expect_error(monkeypatch, "master_cv", asset_hook=MALFORMED_ASSETS[case_id])
    assert error.reason_code == MALFORMED_REASONS[case_id]


def MALFORMED_ASSETS_huge():
    return lambda uids: _assets_file_only(uids, ACTIONS[1], _tfm_huge_header)


def test_huge_header_rejected_before_read_array(monkeypatch):
    ctx = _build_case(monkeypatch, "master_cv", asset_hook=MALFORMED_ASSETS_huge())
    forbidden = {"count": 0}
    real_read = npformat.read_array

    def guarded_read(stream, *args, **kwargs):
        position = stream.tell()
        shape = None
        try:
            version = npformat.read_magic(stream)
            if version == (1, 0):
                shape, _fortran_order, _dtype = npformat.read_array_header_1_0(
                    stream, max_header_size=65536
                )
            elif version == (2, 0):
                shape, _fortran_order, _dtype = npformat.read_array_header_2_0(
                    stream, max_header_size=65536
                )
            else:
                raise AssertionError("unexpected fixture header version")
        finally:
            stream.seek(position)
        if shape == (10**9, 1401):
            forbidden["count"] += 1
            raise AssertionError("huge header reached read_array")
        return real_read(stream, *args, **kwargs)

    monkeypatch.setattr(npformat, "read_array", guarded_read)
    try:
        with pytest.raises(subject.ArrayInputError) as info:
            subject.prepare_u0_source_arrays(
                metadata_bytes=ctx["metadata_bytes"], action_bytes=ctx["action_bytes"]
            )
        assert info.value.reason_code == "action_array_shape_mismatch"
    finally:
        assert forbidden["count"] == 0


def test_first_job_array_hash_mismatch_rejected(monkeypatch):
    def extra(common, plan):
        common["array_sha256"] = _sha(b"bogus-array")

    _expect_error(monkeypatch, "master_cv", pair_mutator_extra=extra)


def test_later_job_array_hash_is_validated(monkeypatch):
    bogus = _sha(b"bogus-late-array")
    target = {"plan": None}
    units = metadata_fixture.UNITS_MASTER
    candidates = []
    for policy in POLICY_ACTION:
        for unit in units:
            candidates.append(("C-RBF-SVM", policy, unit, 0))
            for seed_index in range(3):
                for model_id in ("C-RANDOM-FOREST", "C-EXTRA-TREES", "D0-M"):
                    candidates.append((model_id, policy, unit, seed_index))
        first = units[0]
        for seed_index in range(3):
            for model_id in ("D3", "D2", "D1"):
                candidates.append((model_id, policy, first, seed_index))

    def extra(common, plan):
        if plan == target["plan"]:
            common["array_sha256"] = bogus

    chosen = None
    for plan in candidates:
        target["plan"] = plan
        probe = _build_case(monkeypatch, "master_cv", pair_mutator_extra=extra)
        meta = p08_u0_inputs.bind_u0_source_metadata(**probe["metadata_bytes"])
        records = []
        for pair in meta.pairs:
            job = json.loads(pair.fit_job_json)
            records.append(
                (
                    job["policy_id"],
                    job["context_id"],
                    job["unit_id"],
                    job["array_sha256"],
                )
            )
        index = next(
            (position for position, record in enumerate(records) if record[3] == bogus),
            None,
        )
        if index is None:
            continue
        triple = records[index][:3]
        if any(record[:3] == triple for record in records[:index]):
            chosen = plan
            break
    assert chosen is not None

    target["plan"] = chosen
    error, _ = _expect_error(monkeypatch, "master_cv", pair_mutator_extra=extra)
    assert error.reason_code == "array_pin_mismatch"


def _source_noise_hooks():
    source, held = _uids_by_role("master_cv")

    def negative_on_source(uid, mad, span):
        return (-1.0, span) if uid == source[0] else (mad, span)

    def zero_span_on_source(uid, mad, span):
        return (mad, 0.0) if uid == source[1] else (mad, span)

    def nonfinite_on_source(uid, mad, span):
        return (float("inf"), span) if uid == source[2] else (mad, span)

    def bad_on_held(uid, mad, span):
        return (-1.0, span) if uid == held[0] else (mad, span)

    return negative_on_source, zero_span_on_source, nonfinite_on_source, bad_on_held


@pytest.mark.parametrize("index", [0, 1, 2])
def test_invalid_source_noise_rejected(monkeypatch, index):
    hook = _source_noise_hooks()[index]
    _expect_error(monkeypatch, "master_cv", qc_hook=hook)


def test_missing_qc_column_rejected(monkeypatch):
    _expect_error(monkeypatch, "master_cv", qc_columns=("first_difference_noise_mad",))


def test_held_test_bad_noise_is_ignored(monkeypatch):
    hook = _source_noise_hooks()[3]
    _, binding = _prepare(monkeypatch, "master_cv", qc_hook=hook)
    assert len(binding.pairs) == 78


def _held_text_hook(marker):
    _, held = _uids_by_role("master_cv")
    held_set = set(held)

    def hook(uid, mad, span):
        if uid in held_set:
            return marker, marker
        return mad, span

    return hook


@pytest.mark.parametrize("marker", ["", "NA", "not-a-number"])
def test_held_text_qc_is_ignored_and_source_values_convert(monkeypatch, marker):
    ctx, binding = _prepare(monkeypatch, "master_cv", qc_hook=_held_text_hook(marker))
    _, held = _uids_by_role("master_cv")
    held_set = set(held)
    assert len(binding.pairs) == 78
    for pair in binding.pairs:
        roles = pair.inputs.source_roles
        fit_uids = [obs.observation_uid for obs in roles.fitting]
        frame = pair.inputs.fitting_noise_frame()
        assert not (set(frame["observation_uid"]) & held_set)
        for column in QC_COLUMNS:
            values = frame[column].to_numpy(dtype=float)
            assert np.isfinite(values).all()
        expected = _expected_noise(ctx["manifest_bytes"], fit_uids)
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected.reset_index(drop=True))


def test_invalid_source_text_noise_rejected(monkeypatch):
    source, _ = _uids_by_role("master_cv")

    def hook(uid, mad, span):
        return ("not-a-number", span) if uid == source[0] else (mad, span)

    error, _ = _expect_error(monkeypatch, "master_cv", qc_hook=hook)
    assert error.reason_code == "noise_value_invalid"


def test_no_filesystem_or_scientific_operations(monkeypatch):
    from atlas_sers.evaluation import p03_runtime, p04_runtime, p05_development

    ctx = _build_case(monkeypatch, "master_cv")
    counters = {"open": 0, "fit": 0, "train": 0, "noise": 0, "augment": 0, "quantile": 0}

    def deny_open(*args, **kwargs):
        counters["open"] += 1
        raise AssertionError("filesystem access attempted")

    def deny_fit(*args, **kwargs):
        counters["fit"] += 1
        raise AssertionError("candidate fit executed")

    def deny_train(*args, **kwargs):
        counters["train"] += 1
        raise AssertionError("development fit executed")

    def deny_noise(*args, **kwargs):
        counters["noise"] += 1
        raise AssertionError("noise quantiles computed")

    def deny_augment(*args, **kwargs):
        counters["augment"] += 1
        raise AssertionError("preprocessing applied")

    def deny_quantile(*args, **kwargs):
        counters["quantile"] += 1
        raise AssertionError("quantile computed")

    monkeypatch.setattr(builtins, "open", deny_open)
    monkeypatch.setattr(Path, "open", deny_open)
    monkeypatch.setattr(p03_runtime, "run_candidate_fit", deny_fit)
    monkeypatch.setattr(p05_development, "train_development_fit", deny_train)
    monkeypatch.setattr(p04_runtime, "_noise_quantiles", deny_noise)
    monkeypatch.setattr(p04_runtime, "_augment", deny_augment)
    monkeypatch.setattr(np, "quantile", deny_quantile)
    try:
        binding = subject.prepare_u0_source_arrays(
            metadata_bytes=ctx["metadata_bytes"], action_bytes=ctx["action_bytes"]
        )
        assert len(binding.pairs) == 78
    finally:
        assert counters == {
            "open": 0,
            "fit": 0,
            "train": 0,
            "noise": 0,
            "augment": 0,
            "quantile": 0,
        }
