"""Bounded tests for the P08 neural source-bundle composition (P08-T088)."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from atlas_sers.evaluation import p05_pilot
from atlas_sers.evaluation import p08_neural_bundle as module
from atlas_sers.evaluation.p08_plan import SEEDS
from atlas_sers.evaluation.p08_source_artifacts import (
    EXPECTED_PARAMETER_COUNTS,
    PROJECTION_RECIPES,
)
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.models.acquisition import AcquisitionClassifier
from tests.test_p08_source_artifacts import _bundle
from tests.test_p08_source_predictions import make_pair, valid_kwargs
from tests.test_p08_training_record import THREE_CLASSES, build_record

RECIPES = ("D0-M", "D1", "D2", "D3")
POLICIES = ("PP-U-SG", "PP-U-ARPLS")
ROLE_ID = "p08-fitting-role"
DEFAULT_UIDS = ("sample-1", "sample-2", "sample-3")


def _npz_bytes(logits, classes, uids, *, compressed=True, fortran=False):
    buffer = io.BytesIO()
    logits_array = np.asarray(logits)
    if fortran:
        logits_array = np.asfortranarray(logits_array)
    classes_array = np.asarray(list(classes), dtype=np.str_)
    uids_array = np.asarray(list(uids), dtype=np.str_)
    if compressed:
        np.savez_compressed(buffer, logits=logits_array, classes=classes_array, uids=uids_array)
    else:
        np.savez(buffer, logits=logits_array, classes=classes_array, uids=uids_array)
    return buffer.getvalue()


def _member(array, *, version=(1, 0)):
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), version=version)
    return stream.getvalue()


def _archive(members, *, compression=zipfile.ZIP_DEFLATED):
    pairs = members.items() if isinstance(members, dict) else members
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for name, data in pairs:
            archive.writestr(name, data)
    return buffer.getvalue()


def _encrypted_archive(members):
    """Return valid synthetic bytes carrying the encryption flag bit.

    The archive is written normally and the general-purpose bit flag is then
    set in place in both the local file headers and the central directory
    headers, which is exactly where a reader inspects it.  No password
    encryption is claimed and no decrypting reader is invoked.
    """

    blob = bytearray(_archive(members))
    with zipfile.ZipFile(io.BytesIO(bytes(blob)), "r") as archive:
        infos = archive.infolist()

    for info in infos:
        offset = info.header_offset
        if blob[offset : offset + 4] != b"PK\x03\x04":
            raise AssertionError("local header signature mismatch")
        flags = int.from_bytes(blob[offset + 6 : offset + 8], "little") | 0x1
        blob[offset + 6 : offset + 8] = flags.to_bytes(2, "little")

    end = blob.rfind(b"PK\x05\x06")
    if end < 0:
        raise AssertionError("end of central directory missing")
    count = int.from_bytes(blob[end + 10 : end + 12], "little")
    position = int.from_bytes(blob[end + 16 : end + 20], "little")
    for _ in range(count):
        if blob[position : position + 4] != b"PK\x01\x02":
            raise AssertionError("central header signature mismatch")
        flags = int.from_bytes(blob[position + 8 : position + 10], "little") | 0x1
        blob[position + 8 : position + 10] = flags.to_bytes(2, "little")
        name_length = int.from_bytes(blob[position + 28 : position + 30], "little")
        extra_length = int.from_bytes(blob[position + 30 : position + 32], "little")
        comment_length = int.from_bytes(blob[position + 32 : position + 34], "little")
        position += 46 + name_length + extra_length + comment_length
    return bytes(blob)


def _forbid_member_reads(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("member read/decompression must not run")

    monkeypatch.setattr(zipfile.ZipFile, "read", forbidden)
    return calls


def _manual_npy(descr, shape, payload, *, fortran_order=False, version=(1, 0)):
    header = repr({"descr": descr, "fortran_order": fortran_order, "shape": shape}) + "\n"
    raw = header.encode("latin1")
    if version == (1, 0):
        body = b"\x93NUMPY" + bytes((1, 0)) + len(raw).to_bytes(2, "little") + raw
    else:
        body = b"\x93NUMPY" + bytes((2, 0)) + len(raw).to_bytes(4, "little") + raw
    return body + payload


def make_bundle(
    recipe="D0-M",
    policy="PP-U-SG",
    *,
    role_id=ROLE_ID,
    uids=None,
    classes=None,
    best_seed=101,
    terminal_seed=202,
    source_bytes=None,
    npz_kwargs=None,
    expected_training_record_sha256=None,
    best_file_sha=None,
    terminal_file_sha=None,
    source_file_sha=None,
):
    sp = valid_kwargs(
        model_id=recipe,
        policy=policy,
        uids=uids or DEFAULT_UIDS,
        classes=classes or THREE_CLASSES,
    )
    uids_list = list(sp["observed_uids"])
    classes_list = list(sp["observed_classes"])
    use_projection = PROJECTION_RECIPES[recipe]
    best = _bundle(3, use_projection, seed=best_seed)
    terminal = _bundle(3, use_projection, seed=terminal_seed)
    record = build_record(
        recipe=recipe,
        seed=sp["fit_job"]["seed"],
        role_id=role_id,
        best_digest=best["state_sha256"],
        terminal_digest=terminal["state_sha256"],
    )
    if source_bytes is None:
        source_bytes = _npz_bytes(sp["scores"], classes_list, uids_list, **(npz_kwargs or {}))
    return {
        "fit_job": sp["fit_job"],
        "prediction_job": sp["prediction_job"],
        "expected_fit_job_id": sp["fit_job"]["job_id"],
        "expected_prediction_job_id": sp["prediction_job"]["job_id"],
        "expected_role_id": role_id,
        "expected_validation_uids": uids_list,
        "expected_classes": classes_list,
        "training_record": record,
        "expected_training_record_sha256": (
            sha256_value(record)
            if expected_training_record_sha256 is None
            else expected_training_record_sha256
        ),
        "best_checkpoint_bytes": best["blob"],
        "expected_best_checkpoint_file_sha256": best_file_sha or best["file_sha256"],
        "terminal_checkpoint_bytes": terminal["blob"],
        "expected_terminal_checkpoint_file_sha256": terminal_file_sha or terminal["file_sha256"],
        "source_prediction_bytes": source_bytes,
        "expected_source_prediction_file_sha256": (
            hashlib.sha256(source_bytes).hexdigest() if source_file_sha is None else source_file_sha
        ),
    }


def verify(bundle):
    return module.verify_neural_source_bundle(**bundle)


def reason(bundle):
    with pytest.raises(module.BundleError) as info:
        verify(bundle)
    return info.value.reason_code


def _replace_source(bundle, blob):
    bundle["source_prediction_bytes"] = blob
    bundle["expected_source_prediction_file_sha256"] = hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------------- #
# Acceptance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("policy", POLICIES)
def test_all_recipes_and_policies_accept(recipe, policy):
    report = verify(make_bundle(recipe, policy))
    assert report["schema_version"] == module.SCHEMA_VERSION
    assert report["bundle_consistency_verified"] is True
    assert report["supplied_file_hashes_verified"] is True
    assert report["training_record_pin_verified"] is True
    assert report["execution_authorized"] is False
    assert report["external_registry_membership_verified"] is False
    assert report["physical_role_isolation_verified"] is False
    assert report["training_completion_verified"] is False
    assert report["prediction_parity_verified"] is False
    assert report["live_resources_verified"] is False
    assert report["class_count"] == 3
    assert report["row_count"] == 3


def test_report_counts_and_hashes():
    bundle = make_bundle()
    report = verify(bundle)
    assert report["epochs_completed"] == 30
    assert report["optimizer_steps"] == 120
    assert report["best_epoch"] == 1
    assert report["collapse"] is True
    assert report["parameter_count"] == EXPECTED_PARAMETER_COUNTS[(3, PROJECTION_RECIPES["D0-M"])]
    assert report["training_record_sha256"] == bundle["expected_training_record_sha256"]
    assert report["fit_job_sha256"] == bundle["fit_job"]["job_id"][7:]
    assert report["prediction_job_sha256"] == bundle["prediction_job"]["job_id"][7:]
    assert (
        report["source_prediction_file_sha256"] == bundle["expected_source_prediction_file_sha256"]
    )
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == sha256_value(body)


def test_report_does_not_leak_role_or_recipe():
    bundle = make_bundle(recipe="D1", role_id="SECRET-ROLE-9")
    report = verify(bundle)
    assert set(report) == {
        "schema_version",
        "execution_authorized",
        "bundle_consistency_verified",
        "supplied_file_hashes_verified",
        "training_record_pin_verified",
        "external_registry_membership_verified",
        "physical_role_isolation_verified",
        "training_completion_verified",
        "prediction_parity_verified",
        "live_resources_verified",
        "class_count",
        "row_count",
        "epochs_completed",
        "optimizer_steps",
        "best_epoch",
        "parameter_count",
        "collapse",
        "fit_job_sha256",
        "prediction_job_sha256",
        "training_record_sha256",
        "best_checkpoint_file_sha256",
        "terminal_checkpoint_file_sha256",
        "source_prediction_file_sha256",
        "best_checkpoint_report_sha256",
        "terminal_checkpoint_report_sha256",
        "training_record_report_sha256",
        "source_prediction_report_sha256",
        "report_sha256",
    }
    rendered = json.dumps(report)
    assert "SECRET-ROLE-9" not in rendered


def test_stored_and_fortran_variants_accept():
    stored = verify(make_bundle(npz_kwargs={"compressed": False}))
    fortran = verify(make_bundle(npz_kwargs={"fortran": True}))
    assert stored["bundle_consistency_verified"] is True
    assert fortran["bundle_consistency_verified"] is True


def test_npy_two_zero_arrays_accept():
    bundle = make_bundle()
    members = {
        "logits.npy": _member(np.zeros((3, 3), dtype=np.float64), version=(2, 0)),
        "classes.npy": _member(
            np.asarray(bundle["expected_classes"], dtype=np.str_), version=(2, 0)
        ),
        "uids.npy": _member(
            np.asarray(bundle["expected_validation_uids"], dtype=np.str_), version=(2, 0)
        ),
    }
    _replace_source(bundle, _archive(members))
    assert verify(bundle)["bundle_consistency_verified"] is True


def test_unicode_ids_and_classes_accept():
    uids = ("échantillon-1", "サンプル-2", "mẫu-3")
    classes = ("класс-甲", "クラス-乙", "クラス-丙")
    report = verify(make_bundle(uids=uids, classes=classes))
    assert report["row_count"] == 3
    assert report["class_count"] == 3


def test_maximum_metadata_bounds_roundtrip():
    uids = tuple(f"u{i:03d}-" + "α" * 251 for i in range(598))
    classes = ("β" * 255 + "A", "β" * 255 + "B", "β" * 255 + "C")
    assert all(len(uid) == 256 for uid in uids)
    assert all(len(label) == 256 for label in classes)

    bundle = make_bundle(uids=uids, classes=classes)
    report = verify(bundle)
    assert report["row_count"] == 598
    assert report["class_count"] == 3
    assert report["execution_authorized"] is False
    assert report["external_registry_membership_verified"] is False
    assert report["physical_role_isolation_verified"] is False
    assert report["training_completion_verified"] is False
    assert report["prediction_parity_verified"] is False
    assert report["live_resources_verified"] is False
    assert len(bundle["source_prediction_bytes"]) <= 2 * 1024 * 1024

    members = module._extract_members(bundle["source_prediction_bytes"])
    shapes = {}
    itemsizes = {}
    for name in ("logits.npy", "classes.npy", "uids.npy"):
        stream = io.BytesIO(members[name])
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        else:
            assert version == (2, 0)
            shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        shapes[name] = shape
        itemsizes[name] = int(dtype.itemsize)
    assert shapes["logits.npy"] == (598, 3)
    assert shapes["classes.npy"] == (3,)
    assert shapes["uids.npy"] == (598,)
    assert itemsizes["uids.npy"] == 1024
    assert itemsizes["classes.npy"] == 1024
    declared = sum(itemsizes[name] * int(np.prod(shapes[name])) for name in shapes)
    assert declared <= 2 * 1024 * 1024


def test_finite_wrong_scores_accepted_with_parity_false():
    bundle = make_bundle()
    wrong = np.full((3, 3), 1e300, dtype=np.float64)
    _replace_source(
        bundle,
        _npz_bytes(wrong, bundle["expected_classes"], bundle["expected_validation_uids"]),
    )
    report = verify(bundle)
    assert report["bundle_consistency_verified"] is True
    assert report["prediction_parity_verified"] is False


def test_integration_with_p05_save_logits(monkeypatch):
    bundle = make_bundle()
    logits = np.zeros((3, 3), dtype=np.float64)
    result = types.SimpleNamespace(
        validation_logits=logits,
        classes=tuple(bundle["expected_classes"]),
        validation_uids=tuple(bundle["expected_validation_uids"]),
    )
    captured = {}

    def capture(path, content):
        captured["blob"] = content

    monkeypatch.setattr(p05_pilot.core, "_atomic_write", capture)
    p05_pilot._save_logits(Path("ignored.npz"), result)
    assert captured["blob"]
    _replace_source(bundle, captured["blob"])
    assert verify(bundle)["bundle_consistency_verified"] is True


# --------------------------------------------------------------------------- #
# Job, identity and pin validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "field,bad",
    [
        ("expected_role_id", ""),
        ("expected_role_id", " padded "),
        ("expected_role_id", None),
        ("expected_training_record_sha256", "Z" * 64),
        ("expected_training_record_sha256", None),
        ("expected_best_checkpoint_file_sha256", "short"),
        ("expected_terminal_checkpoint_file_sha256", "A" * 64),
        ("expected_source_prediction_file_sha256", b"x" * 64),
        ("expected_fit_job_id", "not-a-job"),
        ("expected_prediction_job_id", 123),
    ],
)
def test_invalid_expected_inputs(field, bad):
    bundle = make_bundle()
    bundle[field] = bad
    assert reason(bundle) == "invalid_expected_inputs"


def test_two_classes_rejected():
    bundle = make_bundle()
    bundle["expected_classes"] = ["class-x", "class-y"]
    assert reason(bundle) == "invalid_expected_inputs"


def test_empty_classes_rejected():
    bundle = make_bundle()
    bundle["expected_classes"] = []
    assert reason(bundle) in ("invalid_input", "invalid_identifiers")


def test_stale_job_pin_rejected():
    bundle = make_bundle()
    other = valid_kwargs(model_id="D0-M", context_id="ctx-other")
    bundle["expected_fit_job_id"] = other["fit_job"]["job_id"]
    assert reason(bundle) == "job_identity_mismatch"


def test_non_neural_model_rejected():
    bundle = make_bundle()
    fit, prediction = make_pair("C-RBF-SVM", uids=DEFAULT_UIDS)
    bundle["fit_job"] = fit
    bundle["prediction_job"] = prediction
    bundle["expected_fit_job_id"] = fit["job_id"]
    bundle["expected_prediction_job_id"] = prediction["job_id"]
    assert reason(bundle) == "invalid_job"


def test_selected_strategy_alias_rejected():
    bundle = make_bundle()
    fit, prediction = make_pair("P05-SELECTED", uids=DEFAULT_UIDS)
    bundle["fit_job"] = fit
    bundle["prediction_job"] = prediction
    bundle["expected_fit_job_id"] = fit["job_id"]
    bundle["expected_prediction_job_id"] = prediction["job_id"]
    assert reason(bundle) == "invalid_job"


def test_training_record_pin_mismatch():
    bundle = make_bundle()
    bundle["expected_training_record_sha256"] = "0" * 64
    assert reason(bundle) == "training_record_pin_mismatch"


def test_record_recipe_mismatch():
    bundle = make_bundle(recipe="D0-M")
    record = build_record(
        recipe="D1",
        seed=SEEDS[0],
        role_id=ROLE_ID,
        best_digest=bundle["training_record"]["best_state_digest"],
        terminal_digest=bundle["training_record"]["terminal_state_digest"],
    )
    bundle["training_record"] = record
    bundle["expected_training_record_sha256"] = sha256_value(record)
    assert reason(bundle) == "identity_mismatch"


def test_record_seed_mismatch():
    bundle = make_bundle(recipe="D0-M")
    record = build_record(
        recipe="D0-M",
        seed=SEEDS[1],
        role_id=ROLE_ID,
        best_digest=bundle["training_record"]["best_state_digest"],
        terminal_digest=bundle["training_record"]["terminal_state_digest"],
    )
    bundle["training_record"] = record
    bundle["expected_training_record_sha256"] = sha256_value(record)
    assert reason(bundle) == "identity_mismatch"


def test_role_mismatch():
    bundle = make_bundle()
    bundle["expected_role_id"] = "role-other"
    assert reason(bundle) == "identity_mismatch"


def test_conflicting_state_digest_rejected():
    bundle = make_bundle()
    other = _bundle(3, PROJECTION_RECIPES["D0-M"], seed=999)
    bundle["training_record"]["best_state_digest"] = other["state_sha256"]
    bundle["expected_training_record_sha256"] = sha256_value(bundle["training_record"])
    assert reason(bundle) == "checkpoint_state_hash_mismatch"


def test_wrong_checkpoint_file_hash():
    bundle = make_bundle()
    bundle["expected_best_checkpoint_file_sha256"] = "0" * 64
    assert reason(bundle) == "checkpoint_file_hash_mismatch"


def test_wrong_source_file_hash():
    bundle = make_bundle()
    bundle["expected_source_prediction_file_sha256"] = "0" * 64
    assert reason(bundle) == "source_prediction_file_hash_mismatch"


def test_checkpoint_file_hash_checked_before_load(monkeypatch):
    bundle = make_bundle()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(torch, "load", forbidden)
    bundle["expected_best_checkpoint_file_sha256"] = "0" * 64
    assert reason(bundle) == "checkpoint_file_hash_mismatch"
    assert calls == []


def test_source_file_hash_checked_before_zip(monkeypatch):
    bundle = make_bundle()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("zip parse must not run")

    monkeypatch.setattr(module, "_extract_members", forbidden)
    bundle["expected_source_prediction_file_sha256"] = "0" * 64
    assert reason(bundle) == "source_prediction_file_hash_mismatch"
    assert calls == []


def test_source_bytes_size_bound_before_hashing(monkeypatch):
    bundle = make_bundle()
    blob = b"\x00" * (2 * 1024 * 1024 + 1)
    _replace_source(bundle, blob)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("must not run")

    monkeypatch.setattr(hashlib, "sha256", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(module, "_extract_members", forbidden)
    assert reason(bundle) == "source_prediction_too_large"
    assert calls == []


# --------------------------------------------------------------------------- #
# Source archive parsing
# --------------------------------------------------------------------------- #


def test_wrong_row_order_rejected():
    bundle = make_bundle()
    uids = list(bundle["expected_validation_uids"])
    permuted = [uids[2], uids[1], uids[0]]
    _replace_source(
        bundle,
        _npz_bytes(np.zeros((3, 3), dtype=np.float64), bundle["expected_classes"], permuted),
    )
    assert reason(bundle) == "validation_uid_mismatch"


def test_wrong_class_order_rejected():
    bundle = make_bundle()
    classes = list(bundle["expected_classes"])
    permuted = [classes[2], classes[1], classes[0]]
    _replace_source(
        bundle,
        _npz_bytes(
            np.zeros((3, 3), dtype=np.float64),
            permuted,
            bundle["expected_validation_uids"],
        ),
    )
    assert reason(bundle) == "class_order_mismatch"


def test_float32_logits_rejected():
    bundle = make_bundle()
    _replace_source(
        bundle,
        _npz_bytes(
            np.zeros((3, 3), dtype=np.float32),
            bundle["expected_classes"],
            bundle["expected_validation_uids"],
        ),
    )
    assert reason(bundle) == "source_prediction_array_dtype_mismatch"


def test_nonfinite_logits_rejected():
    bundle = make_bundle()
    logits = np.zeros((3, 3), dtype=np.float64)
    logits[0, 0] = np.nan
    _replace_source(
        bundle,
        _npz_bytes(logits, bundle["expected_classes"], bundle["expected_validation_uids"]),
    )
    assert reason(bundle) == "nonfinite_scores"


def test_corrupt_archive_rejected():
    bundle = make_bundle()
    _replace_source(bundle, b"not-a-zip-archive")
    assert reason(bundle) == "invalid_source_prediction_archive"


def test_unknown_member_rejected():
    bundle = make_bundle()
    members = {
        "logits.npy": _member(np.zeros((3, 3), dtype=np.float64)),
        "classes.npy": _member(np.asarray(bundle["expected_classes"], dtype=np.str_)),
        "uids.npy": _member(np.asarray(bundle["expected_validation_uids"], dtype=np.str_)),
        "extra.npy": _member(np.zeros(1, dtype=np.float64)),
    }
    _replace_source(bundle, _archive(members))
    assert reason(bundle) == "source_prediction_member_mismatch"


def test_duplicate_member_names_rejected():
    bundle = make_bundle()
    logits_member = _member(np.zeros((3, 3), dtype=np.float64))
    classes_member = _member(np.asarray(bundle["expected_classes"], dtype=np.str_))
    with pytest.warns(UserWarning):
        blob = _archive(
            [
                ("logits.npy", logits_member),
                ("classes.npy", classes_member),
                ("classes.npy", classes_member),
            ]
        )
    _replace_source(bundle, blob)
    assert reason(bundle) == "source_prediction_member_mismatch"


def test_unknown_name_at_full_cardinality_rejected():
    bundle = make_bundle()
    members = {
        "logits.npy": _member(np.zeros((3, 3), dtype=np.float64)),
        "classes.npy": _member(np.asarray(bundle["expected_classes"], dtype=np.str_)),
        "unknown.npy": _member(np.asarray(bundle["expected_validation_uids"], dtype=np.str_)),
    }
    _replace_source(bundle, _archive(members))
    assert reason(bundle) == "source_prediction_member_mismatch"


def test_directory_member_rejected():
    bundle = make_bundle()
    members = {
        "logits.npy": _member(np.zeros((3, 3), dtype=np.float64)),
        "classes.npy": _member(np.asarray(bundle["expected_classes"], dtype=np.str_)),
        "subdir/": b"",
    }
    _replace_source(bundle, _archive(members))
    assert reason(bundle) == "source_prediction_member_mismatch"


def test_raw_header_name_alias_rejected():
    bundle = make_bundle()
    members = {
        "logits.npy0": _member(np.zeros((3, 3), dtype=np.float64)),
        "classes.npy": _member(np.asarray(bundle["expected_classes"], dtype=np.str_)),
        "uids.npy": _member(np.asarray(bundle["expected_validation_uids"], dtype=np.str_)),
    }
    blob = _archive(members).replace(b"logits.npy0", b"logits.npy\x00")
    _replace_source(bundle, blob)
    assert reason(bundle) == "source_prediction_member_mismatch"


def test_missing_member_rejected():
    bundle = make_bundle()
    members = {
        "logits.npy": _member(np.zeros((3, 3), dtype=np.float64)),
        "classes.npy": _member(np.asarray(bundle["expected_classes"], dtype=np.str_)),
    }
    _replace_source(bundle, _archive(members))
    assert reason(bundle) == "source_prediction_member_mismatch"


def test_unsupported_compression_rejected():
    bundle = make_bundle()
    members = {
        "logits.npy": _member(np.zeros((3, 3), dtype=np.float64)),
        "classes.npy": _member(np.asarray(bundle["expected_classes"], dtype=np.str_)),
        "uids.npy": _member(np.asarray(bundle["expected_validation_uids"], dtype=np.str_)),
    }
    _replace_source(bundle, _archive(members, compression=zipfile.ZIP_BZIP2))
    assert reason(bundle) == "source_prediction_compression_unsupported"


def test_encrypted_archive_rejected():
    bundle = make_bundle()
    members = {
        "logits.npy": _member(np.zeros((3, 3), dtype=np.float64)),
        "classes.npy": _member(np.asarray(bundle["expected_classes"], dtype=np.str_)),
        "uids.npy": _member(np.asarray(bundle["expected_validation_uids"], dtype=np.str_)),
    }
    _replace_source(bundle, _encrypted_archive(members))
    assert reason(bundle) == "source_prediction_encrypted"


def test_unsupported_npy_version_rejected():
    bundle = make_bundle()
    logits = bytearray(_member(np.zeros((3, 3), dtype=np.float64)))
    logits[6] = 9
    members = {
        "logits.npy": bytes(logits),
        "classes.npy": _member(np.asarray(bundle["expected_classes"], dtype=np.str_)),
        "uids.npy": _member(np.asarray(bundle["expected_validation_uids"], dtype=np.str_)),
    }
    _replace_source(bundle, _archive(members))
    assert reason(bundle) == "invalid_source_prediction_array"


def test_truncated_member_rejected():
    bundle = make_bundle()
    logits = _member(np.zeros((3, 3), dtype=np.float64))
    members = {
        "logits.npy": logits[:-1],
        "classes.npy": _member(np.asarray(bundle["expected_classes"], dtype=np.str_)),
        "uids.npy": _member(np.asarray(bundle["expected_validation_uids"], dtype=np.str_)),
    }
    _replace_source(bundle, _archive(members))
    assert reason(bundle) == "source_prediction_array_length_mismatch"


def test_trailing_member_payload_rejected():
    bundle = make_bundle()
    logits = _member(np.zeros((3, 3), dtype=np.float64))
    members = {
        "logits.npy": logits + b"\x00\x00",
        "classes.npy": _member(np.asarray(bundle["expected_classes"], dtype=np.str_)),
        "uids.npy": _member(np.asarray(bundle["expected_validation_uids"], dtype=np.str_)),
    }
    _replace_source(bundle, _archive(members))
    assert reason(bundle) == "source_prediction_array_length_mismatch"


def test_object_and_bytestring_labels_rejected():
    bundle = make_bundle()
    classes_member = _member(np.asarray(bundle["expected_classes"], dtype=np.str_))
    uids_member = _member(np.asarray(bundle["expected_validation_uids"], dtype=np.str_))
    for bad in (
        _manual_npy("|O", (3,), b"\x00" * 24),
        _manual_npy("|S4", (3,), b"\x00" * 12),
        _manual_npy("<U300", (3,), b"\x00" * (3 * 1200)),
    ):
        members = {
            "logits.npy": _member(np.zeros((3, 3), dtype=np.float64)),
            "classes.npy": bad,
            "uids.npy": uids_member,
        }
        local = make_bundle()
        _replace_source(local, _archive(members))
        assert reason(local) == "source_prediction_array_dtype_mismatch"
    assert classes_member


def test_oversized_declared_header_rejected_before_read_array(monkeypatch):
    bundle = make_bundle()
    members = {
        "logits.npy": _manual_npy("<f8", (10**12, 3), b"\x00" * 24),
        "classes.npy": _member(np.asarray(bundle["expected_classes"], dtype=np.str_)),
        "uids.npy": _member(np.asarray(bundle["expected_validation_uids"], dtype=np.str_)),
    }
    _replace_source(bundle, _archive(members))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("read_array must not run")

    monkeypatch.setattr(np.lib.format, "read_array", forbidden)
    assert reason(bundle) == "source_prediction_array_shape_mismatch"
    assert calls == []


def test_archive_size_bound_before_expansion(monkeypatch):
    bundle = make_bundle()
    monkeypatch.setattr(module, "_MAXIMUM_ARCHIVE_BYTES", 16)
    assert reason(bundle) == "source_prediction_too_large"


def test_declared_member_size_bound_before_expansion(monkeypatch):
    bundle = make_bundle()
    inflated = b"\x00" * (2 * 1024 * 1024 + 1)
    blob = _archive(
        {
            "logits.npy": inflated,
            "classes.npy": b"\x00" * 8,
            "uids.npy": b"\x00" * 8,
        }
    )
    assert len(blob) <= 2 * 1024 * 1024
    _replace_source(bundle, blob)
    calls = _forbid_member_reads(monkeypatch)
    assert reason(bundle) == "source_prediction_too_large"
    assert calls == []


def test_declared_total_size_bound_before_expansion(monkeypatch):
    bundle = make_bundle()
    chunk = b"\x00" * (700 * 1024)
    blob = _archive(
        {
            "logits.npy": chunk,
            "classes.npy": chunk,
            "uids.npy": chunk,
        }
    )
    assert len(blob) <= 2 * 1024 * 1024
    _replace_source(bundle, blob)
    calls = _forbid_member_reads(monkeypatch)
    assert reason(bundle) == "source_prediction_too_large"
    assert calls == []


# --------------------------------------------------------------------------- #
# Ownership, side effects and failure sanitization
# --------------------------------------------------------------------------- #


def test_inputs_not_mutated():
    bundle = make_bundle()
    before = copy.deepcopy(bundle)
    verify(bundle)
    assert bundle["fit_job"] == before["fit_job"]
    assert bundle["prediction_job"] == before["prediction_job"]
    assert bundle["training_record"] == before["training_record"]
    assert bundle["expected_validation_uids"] == before["expected_validation_uids"]


def test_metadata_snapshot_isolated_from_mutation(monkeypatch):
    bundle = make_bundle()
    baseline = verify(bundle)
    real = module._read_plain_array
    state = {"calls": 0}

    def hooked(*args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            bundle["fit_job"]["context_id"] = "ctx-mutated"
            bundle["training_record"]["role_id"] = "role-mutated"
            bundle["expected_validation_uids"][0] = "mutated"
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "_read_plain_array", hooked)
    assert verify(bundle) == baseline
    assert state["calls"] >= 1


def test_cpu_rng_and_default_dtype_preserved():
    bundle = make_bundle(recipe="D1")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4242)
        before = torch.random.get_rng_state().clone()
        before_dtype = torch.get_default_dtype()
        verify(bundle)
        assert torch.equal(torch.random.get_rng_state(), before)
        assert torch.get_default_dtype() == before_dtype


def test_no_scientific_or_filesystem_calls(monkeypatch):
    from atlas_sers.evaluation import p04_runtime, p05_development

    bundle = make_bundle()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("forbidden call attempted")

    monkeypatch.setattr(AcquisitionClassifier, "forward", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden, raising=False)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden, raising=False)
    monkeypatch.setattr(p05_development, "train_development_fit", forbidden)
    monkeypatch.setattr(p04_runtime, "_metric_values", forbidden)
    monkeypatch.setattr(p05_development, "_metric_values", forbidden)
    monkeypatch.setattr(p05_development, "_predict_logits", forbidden)
    verify(bundle)
    assert calls == []
    assert not hasattr(module, "torch")


def test_unexpected_exception_sanitized(monkeypatch):
    bundle = make_bundle()

    def boom(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_PRIVATE_DETAIL")

    monkeypatch.setattr(module, "_extract_members", boom)
    with pytest.raises(module.BundleError) as info:
        verify(bundle)
    assert info.value.reason_code == "verification_failed"
    assert "SYNTHETIC_PRIVATE_DETAIL" not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True


def test_keyboard_interrupt_propagates_same_object(monkeypatch):
    bundle = make_bundle()
    signal = KeyboardInterrupt("stop")

    def boom(*args, **kwargs):
        raise signal

    monkeypatch.setattr(module, "_extract_members", boom)
    with pytest.raises(KeyboardInterrupt) as info:
        verify(bundle)
    assert info.value is signal


def test_system_exit_propagates_same_object(monkeypatch):
    bundle = make_bundle()
    signal = SystemExit(7)

    def boom(*args, **kwargs):
        raise signal

    monkeypatch.setattr(module, "_extract_members", boom)
    with pytest.raises(SystemExit) as info:
        verify(bundle)
    assert info.value is signal


def test_bundle_error_sanitizes_unknown_codes():
    assert module.BundleError("not-a-code").reason_code == "unlisted_reason_code"
    assert module.BundleError(None).reason_code == "unlisted_reason_code"
    assert module.BundleError(["x"]).reason_code == "unlisted_reason_code"
    assert module.BundleError("invalid_job").reason_code == "invalid_job"
    assert str(module.BundleError("invalid_job")) == "invalid_job"


def test_execution_always_denied():
    for call in (
        lambda: module.require_scientific_execution(),
        lambda: module.require_scientific_execution(execution_authorized=True),
        lambda: module.require_scientific_execution(None, authorized=True, token="x"),
    ):
        with pytest.raises(module.BundleError) as info:
            call()
        assert info.value.reason_code == "scientific_execution_not_authorized"
