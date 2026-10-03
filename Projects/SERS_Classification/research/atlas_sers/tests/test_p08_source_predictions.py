"""Bounded tests for the P08-T083 source-prediction semantic boundary."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from atlas_sers.evaluation import p08_source_predictions as sp
from atlas_sers.evaluation.p08_plan import (
    JOB_FIELDS,
    NEURAL_RECIPES,
    SEEDS,
    _hash,
    _new_job,
)
from atlas_sers.governance.canonical import sha256_value

REPRESENTATION = {
    "PP-U-SG": "R_SG_400_1800",
    "PP-U-ARPLS": "R_ARPLS_400_1800",
}

MODEL_IDS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES", "D0-M", "D1", "D2", "D3")

UID_A = "sample-1"
UID_B = "sample-2"
UID_C = "sample-3"
CLASS_X = "class-x"
CLASS_Y = "class-y"


def _digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rehash(job, dependencies=None, **changes):
    body = {name: job[name] for name in JOB_FIELDS if name != "dependencies"}
    body.update(changes)
    deps = job["dependencies"] if dependencies is None else dependencies
    return _new_job(body, deps)


def make_pair(model_id, policy="PP-U-SG", uids=(UID_A, UID_B, UID_C), context_id="ctx-1"):
    spec = _digest("spec-" + model_id)
    if model_id in NEURAL_RECIPES:
        seed = SEEDS[0]
        candidate_id = "fixed_recipe"
        hyper = spec
    elif model_id == "C-RBF-SVM":
        seed = "deterministic"
        candidate_id = "cand-svm"
        hyper = _digest("hyper-svm")
    else:
        seed = SEEDS[1]
        candidate_id = "cand-" + model_id
        hyper = _digest("hyper-" + model_id)
    common = {
        "policy_id": policy,
        "representation_id": REPRESENTATION[policy],
        "array_sha256": _digest("array-" + policy),
        "context_id": context_id,
        "model_id": model_id,
        "model_spec_sha256": spec,
        "unit_id": "unit-1",
        "seed": seed,
        "candidate_id": candidate_id,
        "hyperparameter_sha256": hyper,
        "fit_uid_sha256": _digest("fit-uid"),
        "validation_uid_sha256": sha256_value(sorted(uids)),
        "test_uid_sha256": _digest("test-uid"),
        "resolution": "fixed_spec",
        "evidence_status": "unapproved_future_job",
    }
    fit = _new_job({**common, "stage": "source_fit"}, [])
    prediction = _new_job({**common, "stage": "source_validation_prediction"}, [fit["job_id"]])
    return fit, prediction


def valid_kwargs(
    model_id="C-RBF-SVM",
    policy="PP-U-SG",
    uids=(UID_A, UID_B, UID_C),
    classes=(CLASS_X, CLASS_Y),
    context_id="ctx-1",
):
    fit, prediction = make_pair(model_id, policy, uids=uids, context_id=context_id)
    scores = np.zeros((len(uids), len(classes)), dtype=np.float64)
    return {
        "scores": scores,
        "observed_uids": list(uids),
        "observed_classes": list(classes),
        "expected_validation_uids": list(uids),
        "expected_classes": list(classes),
        "fit_job": fit,
        "prediction_job": prediction,
        "expected_fit_job_id": fit["job_id"],
        "expected_prediction_job_id": prediction["job_id"],
    }


def _reason(kwargs):
    with pytest.raises(sp.PredictionError) as info:
        sp.verify_source_prediction_values(**kwargs)
    return info.value.reason_code


def _rehashed_pair(**changes):
    fit, prediction = make_pair("C-RBF-SVM")
    fit2 = _rehash(fit, **changes)
    prediction2 = _rehash(prediction, dependencies=[fit2["job_id"]], **changes)
    kwargs = valid_kwargs()
    kwargs["fit_job"] = fit2
    kwargs["prediction_job"] = prediction2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    return kwargs


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("policy", ["PP-U-SG", "PP-U-ARPLS"])
def test_all_model_interfaces_and_policies_accept(model_id, policy):
    report = sp.verify_source_prediction_values(**valid_kwargs(model_id=model_id, policy=policy))
    assert report["declared_job_pair_verified"] is True
    assert report["source_prediction_structure_verified"] is True
    assert report["execution_authorized"] is False


def test_report_flags_and_counts():
    report = sp.verify_source_prediction_values(**valid_kwargs())
    assert report["schema_version"] == sp.SCHEMA_VERSION
    assert report["execution_authorized"] is False
    assert report["external_authentication_verified"] is False
    assert report["physical_role_isolation_verified"] is False
    assert report["training_completion_verified"] is False
    assert report["prediction_parity_verified"] is False
    assert report["row_count"] == 3
    assert report["class_count"] == 2


def test_dict_subclass_job_rejected():
    class Job(dict):
        pass

    kwargs = valid_kwargs()
    kwargs["fit_job"] = Job(kwargs["fit_job"])
    assert _reason(kwargs) == "invalid_job"


def test_missing_field_rejected():
    kwargs = valid_kwargs()
    broken = dict(kwargs["fit_job"])
    broken.pop("unit_id")
    kwargs["fit_job"] = broken
    assert _reason(kwargs) == "invalid_job"


def test_extra_field_rejected():
    kwargs = valid_kwargs()
    broken = dict(kwargs["fit_job"])
    broken["extra"] = "x"
    kwargs["fit_job"] = broken
    assert _reason(kwargs) == "invalid_job"


def test_malformed_hash_field_rejected():
    kwargs = valid_kwargs()
    kwargs["fit_job"] = _rehash(kwargs["fit_job"], array_sha256="zz")
    assert _reason(kwargs) == "invalid_job"


def test_reserved_job_id_format_rejected():
    kwargs = valid_kwargs()
    broken = dict(kwargs["fit_job"])
    broken["job_id"] = "P08JOB-" + "g" * 64
    kwargs["fit_job"] = broken
    assert _reason(kwargs) == "invalid_job"


def test_stale_expected_pin_rejected():
    kwargs = valid_kwargs()
    stale, _ = make_pair("C-RBF-SVM", context_id="ctx-old")
    kwargs["expected_fit_job_id"] = stale["job_id"]
    assert _reason(kwargs) == "job_identity_mismatch"


def test_non_string_expected_pin_rejected():
    kwargs = valid_kwargs()
    kwargs["expected_fit_job_id"] = 12345
    assert _reason(kwargs) == "job_identity_mismatch"


def test_forged_self_consistent_job_with_original_pin_rejected():
    kwargs = valid_kwargs()
    forged_fit, forged_prediction = make_pair("C-RBF-SVM", context_id="ctx-forged")
    kwargs["fit_job"] = forged_fit
    kwargs["prediction_job"] = forged_prediction
    assert _reason(kwargs) == "job_identity_mismatch"


def test_pair_field_mismatch_rehashed_and_pins_updated():
    fit, prediction = make_pair("C-RBF-SVM", context_id="ctx-1")
    fit2 = _rehash(fit, context_id="ctx-2")
    prediction2 = _rehash(prediction, dependencies=[fit2["job_id"]], context_id="ctx-3")
    kwargs = valid_kwargs()
    kwargs["fit_job"] = fit2
    kwargs["prediction_job"] = prediction2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    assert _reason(kwargs) == "job_pair_mismatch"


def test_fit_dependencies_must_be_empty():
    fit, prediction = make_pair("C-RBF-SVM")
    fit2 = _rehash(fit, dependencies=[prediction["job_id"]])
    kwargs = valid_kwargs()
    kwargs["fit_job"] = fit2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    assert _reason(kwargs) == "job_pair_mismatch"


def test_prediction_dependencies_must_point_to_fit():
    _, prediction = make_pair("C-RBF-SVM")
    prediction2 = _rehash(prediction, dependencies=["P08JOB-" + "a" * 64])
    kwargs = valid_kwargs()
    kwargs["prediction_job"] = prediction2
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    assert _reason(kwargs) == "job_pair_mismatch"


def test_invalid_stage_rejected():
    kwargs = _rehashed_pair(stage="held_prediction")
    assert _reason(kwargs) == "invalid_job"


def test_unsupported_policy_rejected():
    kwargs = _rehashed_pair(policy_id="PP-U-MIN")
    assert _reason(kwargs) == "invalid_job"


def test_representation_mismatch_rejected():
    kwargs = _rehashed_pair(representation_id="R_MIN_400_1800")
    assert _reason(kwargs) == "invalid_job"


def test_resolution_mismatch_rejected():
    kwargs = _rehashed_pair(resolution="source_selection_dependent")
    assert _reason(kwargs) == "invalid_job"


def test_evidence_status_mismatch_rejected():
    kwargs = _rehashed_pair(evidence_status="historical_reuse_requires_authentication")
    assert _reason(kwargs) == "invalid_job"


def test_unknown_model_rejected():
    fit, prediction = make_pair("C-RANDOM-FOREST")
    fit2 = _rehash(fit, model_id="C-UNKNOWN")
    prediction2 = _rehash(prediction, dependencies=[fit2["job_id"]], model_id="C-UNKNOWN")
    kwargs = valid_kwargs()
    kwargs["fit_job"] = fit2
    kwargs["prediction_job"] = prediction2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    assert _reason(kwargs) == "invalid_job"


def test_selected_strategy_alias_rejected():
    fit, prediction = make_pair("C-RANDOM-FOREST")
    fit2 = _rehash(fit, model_id="P05-SELECTED")
    prediction2 = _rehash(prediction, dependencies=[fit2["job_id"]], model_id="P05-SELECTED")
    kwargs = valid_kwargs()
    kwargs["fit_job"] = fit2
    kwargs["prediction_job"] = prediction2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    assert _reason(kwargs) == "invalid_job"


def test_svm_seed_must_be_deterministic_string():
    kwargs = _rehashed_pair(seed=SEEDS[0])
    assert _reason(kwargs) == "invalid_job"


def test_neural_seed_rejects_bool():
    fit, prediction = make_pair("D0-M")
    fit2 = _rehash(fit, seed=True)
    prediction2 = _rehash(prediction, dependencies=[fit2["job_id"]], seed=True)
    kwargs = valid_kwargs(model_id="D0-M")
    kwargs["fit_job"] = fit2
    kwargs["prediction_job"] = prediction2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    assert _reason(kwargs) == "invalid_job"


def test_neural_candidate_must_be_fixed_recipe():
    fit, prediction = make_pair("D1")
    fit2 = _rehash(fit, candidate_id="cand-x")
    prediction2 = _rehash(prediction, dependencies=[fit2["job_id"]], candidate_id="cand-x")
    kwargs = valid_kwargs(model_id="D1")
    kwargs["fit_job"] = fit2
    kwargs["prediction_job"] = prediction2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    assert _reason(kwargs) == "invalid_job"


def test_neural_hyperparameter_must_match_model_spec():
    fit, prediction = make_pair("D2")
    other = _digest("other")
    fit2 = _rehash(fit, hyperparameter_sha256=other)
    prediction2 = _rehash(prediction, dependencies=[fit2["job_id"]], hyperparameter_sha256=other)
    kwargs = valid_kwargs(model_id="D2")
    kwargs["fit_job"] = fit2
    kwargs["prediction_job"] = prediction2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    assert _reason(kwargs) == "invalid_job"


def test_classical_candidate_rejects_neural_sentinel():
    fit, prediction = make_pair("C-EXTRA-TREES")
    fit2 = _rehash(fit, candidate_id="fixed_recipe")
    prediction2 = _rehash(prediction, dependencies=[fit2["job_id"]], candidate_id="fixed_recipe")
    kwargs = valid_kwargs(model_id="C-EXTRA-TREES")
    kwargs["fit_job"] = fit2
    kwargs["prediction_job"] = prediction2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    assert _reason(kwargs) == "invalid_job"


def test_duplicate_uids_rejected():
    kwargs = valid_kwargs()
    kwargs["observed_uids"] = [UID_A, UID_A, UID_C]
    kwargs["expected_validation_uids"] = [UID_A, UID_A, UID_C]
    assert _reason(kwargs) == "invalid_identifiers"


def test_empty_uids_rejected():
    kwargs = valid_kwargs()
    kwargs["observed_uids"] = []
    kwargs["expected_validation_uids"] = []
    assert _reason(kwargs) == "invalid_input"


def test_oversize_uid_rejected():
    kwargs = valid_kwargs()
    big = "x" * 257
    kwargs["observed_uids"] = [UID_A, UID_B, big]
    kwargs["expected_validation_uids"] = [UID_A, UID_B, big]
    assert _reason(kwargs) == "invalid_identifiers"


def test_control_character_uid_rejected():
    kwargs = valid_kwargs()
    bad = "abc\x01def"
    kwargs["observed_uids"] = [UID_A, UID_B, bad]
    kwargs["expected_validation_uids"] = [UID_A, UID_B, bad]
    assert _reason(kwargs) == "invalid_identifiers"


def test_whitespace_uid_rejected():
    kwargs = valid_kwargs()
    bad = " spaced "
    kwargs["observed_uids"] = [UID_A, UID_B, bad]
    kwargs["expected_validation_uids"] = [UID_A, UID_B, bad]
    assert _reason(kwargs) == "invalid_identifiers"


def test_non_string_uid_rejected():
    kwargs = valid_kwargs()
    kwargs["observed_uids"] = [UID_A, UID_B, 5]
    assert _reason(kwargs) == "invalid_identifiers"


def test_uid_container_type_rejected():
    kwargs = valid_kwargs()
    kwargs["observed_uids"] = {UID_A, UID_B, UID_C}
    assert _reason(kwargs) == "invalid_input"


def test_expected_sequence_checked_separately():
    kwargs = valid_kwargs()
    kwargs["expected_validation_uids"] = [UID_A, UID_B, "bad\x02id"]
    assert _reason(kwargs) == "invalid_identifiers"


def test_row_permutation_rejected():
    kwargs = valid_kwargs()
    kwargs["observed_uids"] = [UID_C, UID_B, UID_A]
    assert _reason(kwargs) == "validation_uid_mismatch"


def test_column_permutation_rejected():
    kwargs = valid_kwargs()
    kwargs["observed_classes"] = [CLASS_Y, CLASS_X]
    assert _reason(kwargs) == "class_order_mismatch"


def test_wrong_uid_set_hash_rejected():
    kwargs = valid_kwargs()
    alt = [UID_A, UID_B, "other-3"]
    kwargs["observed_uids"] = alt
    kwargs["expected_validation_uids"] = list(alt)
    assert _reason(kwargs) == "validation_uid_mismatch"


def test_class_count_invalid():
    kwargs = valid_kwargs()
    kwargs["observed_classes"] = [CLASS_X]
    kwargs["expected_classes"] = [CLASS_X]
    assert _reason(kwargs) == "invalid_input"


def test_scores_wrong_dtype_rejected():
    kwargs = valid_kwargs()
    kwargs["scores"] = np.zeros((3, 2), dtype=np.float32)
    assert _reason(kwargs) == "invalid_scores"


def test_scores_bool_dtype_rejected():
    kwargs = valid_kwargs()
    kwargs["scores"] = np.zeros((3, 2), dtype=bool)
    assert _reason(kwargs) == "invalid_scores"


def test_scores_ndim_rejected():
    kwargs = valid_kwargs()
    kwargs["scores"] = np.zeros((3, 2, 1), dtype=np.float64)
    assert _reason(kwargs) == "invalid_scores"


def test_scores_shape_rejected():
    kwargs = valid_kwargs()
    kwargs["scores"] = np.zeros((2, 2), dtype=np.float64)
    assert _reason(kwargs) == "invalid_scores"


def test_scores_subclass_rejected():
    class SubArray(np.ndarray):
        pass

    kwargs = valid_kwargs()
    kwargs["scores"] = np.zeros((3, 2), dtype=np.float64).view(SubArray)
    assert _reason(kwargs) == "invalid_scores"


def test_scores_matrix_rejected():
    kwargs = valid_kwargs()
    kwargs["scores"] = np.matrix(np.zeros((3, 2), dtype=np.float64))
    assert _reason(kwargs) == "invalid_scores"


def test_scores_masked_array_rejected():
    kwargs = valid_kwargs()
    kwargs["scores"] = np.ma.masked_array(np.zeros((3, 2), dtype=np.float64))
    assert _reason(kwargs) == "invalid_scores"


def test_scores_not_ndarray_rejected():
    kwargs = valid_kwargs()
    kwargs["scores"] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert _reason(kwargs) == "invalid_scores"


def test_scores_nan_rejected():
    kwargs = valid_kwargs()
    scores = np.zeros((3, 2), dtype=np.float64)
    scores[0, 0] = np.nan
    kwargs["scores"] = scores
    assert _reason(kwargs) == "nonfinite_scores"


def test_scores_inf_rejected():
    kwargs = valid_kwargs()
    scores = np.zeros((3, 2), dtype=np.float64)
    scores[1, 1] = np.inf
    kwargs["scores"] = scores
    assert _reason(kwargs) == "nonfinite_scores"


def test_noncontiguous_scores_accepted():
    kwargs = valid_kwargs()
    base = np.arange(12, dtype=np.float64).reshape(3, 4)
    view = base[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    kwargs["scores"] = view
    report = sp.verify_source_prediction_values(**kwargs)
    assert report["source_prediction_structure_verified"] is True


def test_raw_decision_values_accepted():
    kwargs = valid_kwargs()
    kwargs["scores"] = np.array([[-5.0, 10.0], [0.0, 0.0], [1e308, -1e308]], dtype=np.float64)
    report = sp.verify_source_prediction_values(**kwargs)
    assert report["row_count"] == 3


def test_collapsed_model_accepted():
    kwargs = valid_kwargs()
    kwargs["scores"] = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    report = sp.verify_source_prediction_values(**kwargs)
    assert report["class_count"] == 2


def test_three_classes_accepted():
    classes = (CLASS_X, CLASS_Y, "class-z")
    report = sp.verify_source_prediction_values(**valid_kwargs(classes=classes))
    assert report["class_count"] == 3


def test_content_and_report_digests_independently_recomputed():
    kwargs = valid_kwargs()
    scores = kwargs["scores"].copy()
    report = sp.verify_source_prediction_values(**kwargs)
    expected_content = sha256_value(
        {
            "classes": list(kwargs["observed_classes"]),
            "uids": list(kwargs["observed_uids"]),
            "scores": scores.tolist(),
        }
    )
    assert report["prediction_content_sha256"] == expected_content
    without = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == sha256_value(without)
    assert report["ordered_class_sha256"] == sha256_value(list(kwargs["observed_classes"]))
    assert report["ordered_validation_uid_sha256"] == sha256_value(list(kwargs["observed_uids"]))


def test_inputs_not_mutated():
    kwargs = valid_kwargs()
    original_scores = kwargs["scores"].copy()
    original_uids = list(kwargs["observed_uids"])
    original_classes = list(kwargs["observed_classes"])
    sp.verify_source_prediction_values(**kwargs)
    assert np.array_equal(kwargs["scores"], original_scores)
    assert kwargs["observed_uids"] == original_uids
    assert kwargs["observed_classes"] == original_classes


def test_report_does_not_leak_caller_text():
    secret_uid = "SECRET-UID-9"
    secret_class = "SECRET-CLASS-9"
    secret_context = "SECRET-CONTEXT-9"
    kwargs = valid_kwargs(
        uids=(secret_uid, UID_B, UID_C),
        classes=(secret_class, CLASS_Y),
        context_id=secret_context,
    )
    report = sp.verify_source_prediction_values(**kwargs)
    rendered = json.dumps(report)
    assert secret_uid not in rendered
    assert secret_class not in rendered
    assert secret_context not in rendered


def test_error_messages_do_not_leak_caller_text():
    secret_class = "SECRET-CLASS-9"
    kwargs = valid_kwargs()
    kwargs["observed_classes"] = [secret_class, CLASS_Y]
    with pytest.raises(sp.PredictionError) as info:
        sp.verify_source_prediction_values(**kwargs)
    assert info.value.reason_code == "class_order_mismatch"
    assert "SECRET" not in str(info.value)


def test_unicode_context_uses_ascii_job_identity_and_utf8_content_hash():
    context_id = "ctx-ünïcode-漢字"
    kwargs = valid_kwargs(context_id=context_id)
    report = sp.verify_source_prediction_values(**kwargs)
    body = {name: kwargs["fit_job"][name] for name in JOB_FIELDS}
    assert kwargs["fit_job"]["job_id"] == "P08JOB-" + _hash(body)
    assert report["ordered_class_sha256"] == sha256_value(list(kwargs["observed_classes"]))
    assert _hash({"k": context_id}) != sha256_value({"k": context_id})


def test_ordinary_error_is_sanitized(monkeypatch):
    def boom(_job):
        raise RuntimeError("internal detail")

    monkeypatch.setattr(sp, "_validate_job", boom)
    with pytest.raises(sp.PredictionError) as info:
        sp.verify_source_prediction_values(**valid_kwargs())
    assert info.value.reason_code == "verification_failed"
    assert "internal detail" not in str(info.value)


def test_keyboard_interrupt_propagates_same_object(monkeypatch):
    interrupt = KeyboardInterrupt("stop now")

    def raise_interrupt(_job):
        raise interrupt

    monkeypatch.setattr(sp, "_validate_job", raise_interrupt)
    with pytest.raises(KeyboardInterrupt) as info:
        sp.verify_source_prediction_values(**valid_kwargs())
    assert info.value is interrupt


def test_system_exit_propagates_same_object(monkeypatch):
    exit_signal = SystemExit(3)

    def raise_exit(_job):
        raise exit_signal

    monkeypatch.setattr(sp, "_validate_job", raise_exit)
    with pytest.raises(SystemExit) as info:
        sp.verify_source_prediction_values(**valid_kwargs())
    assert info.value is exit_signal


def test_execution_always_denied():
    for call in (
        lambda: sp.require_scientific_execution(),
        lambda: sp.require_scientific_execution(execution_authorized=True),
        lambda: sp.require_scientific_execution(None, authorized=True, token="x"),
    ):
        with pytest.raises(sp.PredictionError) as info:
            call()
        assert info.value.reason_code == "scientific_execution_not_authorized"


def test_reason_code_fallbacks():
    assert sp.PredictionError("invalid_job").reason_code == "invalid_job"
    assert sp.PredictionError("not-a-code").reason_code == "invalid_input"
    assert sp.PredictionError(["unhashable"]).reason_code == "invalid_input"
    assert sp.PredictionError(None).reason_code == "invalid_input"
    assert sp.PredictionError("unlisted_reason_code").reason_code == "invalid_input"


def test_no_filesystem_io(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("filesystem access attempted")

    monkeypatch.setattr("builtins.open", forbidden)
    sp.verify_source_prediction_values(**valid_kwargs())
    assert calls == []


def test_job_id_hash_hook_cannot_change_accepted_job_ids(monkeypatch):
    kwargs = valid_kwargs()
    fit_job = kwargs["fit_job"]
    prediction_job = kwargs["prediction_job"]
    original_fit_id = fit_job["job_id"]
    original_prediction_id = prediction_job["job_id"]
    real = sp.sha256_value
    state = {"calls": 0}

    def hooked(value):
        state["calls"] += 1
        if state["calls"] == 1:
            fit_job["job_id"] = "P08JOB-" + "0" * 64
            prediction_job["job_id"] = "P08JOB-" + "0" * 64
        return real(value)

    monkeypatch.setattr(sp, "sha256_value", hooked)
    report = sp.verify_source_prediction_values(**kwargs)
    assert report["fit_job_sha256"] == original_fit_id[7:]
    assert report["prediction_job_sha256"] == original_prediction_id[7:]
    assert state["calls"] >= 1


def test_isfinite_hook_cannot_change_accepted_content(monkeypatch):
    kwargs = valid_kwargs()
    scores = kwargs["scores"]
    original_scores = scores.copy()
    expected_content = sha256_value(
        {
            "classes": list(kwargs["observed_classes"]),
            "uids": list(kwargs["observed_uids"]),
            "scores": original_scores.tolist(),
        }
    )
    real = np.isfinite
    state = {"calls": 0}

    def hooked(value):
        state["calls"] += 1
        mask = real(value)
        if state["calls"] == 1:
            scores.shape = (2, 3)
        return mask

    monkeypatch.setattr(sp.np, "isfinite", hooked)
    report = sp.verify_source_prediction_values(**kwargs)
    assert report["prediction_content_sha256"] == expected_content
    assert report["row_count"] == 3
    assert report["class_count"] == 2


def test_maximum_uid_count_accepted():
    uids = [f"uid-{index:04d}" for index in range(598)]
    report = sp.verify_source_prediction_values(**valid_kwargs(uids=uids))
    assert report["row_count"] == 598


def test_over_maximum_uid_count_rejected():
    uids = [f"uid-{index:04d}" for index in range(599)]
    assert _reason(valid_kwargs(uids=uids)) == "invalid_input"


def test_tuple_sequences_accepted():
    kwargs = valid_kwargs()
    kwargs["observed_uids"] = tuple(kwargs["observed_uids"])
    kwargs["expected_validation_uids"] = tuple(kwargs["expected_validation_uids"])
    kwargs["observed_classes"] = tuple(kwargs["observed_classes"])
    kwargs["expected_classes"] = tuple(kwargs["expected_classes"])
    report = sp.verify_source_prediction_values(**kwargs)
    assert report["row_count"] == 3


def test_sequence_subclass_rejected():
    class Sequence(list):
        pass

    kwargs = valid_kwargs()
    kwargs["observed_uids"] = Sequence(kwargs["observed_uids"])
    assert _reason(kwargs) == "invalid_input"


def test_duplicate_classes_rejected():
    kwargs = valid_kwargs()
    kwargs["observed_classes"] = [CLASS_X, CLASS_X]
    kwargs["expected_classes"] = [CLASS_X, CLASS_X]
    assert _reason(kwargs) == "invalid_identifiers"


def test_str_subclass_identifier_rejected():
    class Identifier(str):
        pass

    kwargs = valid_kwargs()
    broken = dict(kwargs["fit_job"])
    broken["context_id"] = Identifier("ctx-2")
    kwargs["fit_job"] = broken
    assert _reason(kwargs) == "invalid_job"


def test_str_subclass_job_key_rejected():
    class JobKey(str):
        pass

    kwargs = valid_kwargs()
    broken = dict(kwargs["fit_job"])
    value = broken.pop("context_id")
    broken[JobKey("context_id")] = value
    kwargs["fit_job"] = broken
    assert _reason(kwargs) == "invalid_job"


def test_non_exact_int_seed_rejected():
    class Seed(int):
        pass

    fit, prediction = make_pair("D0-M")
    seed = Seed(SEEDS[0])
    fit2 = _rehash(fit, seed=seed)
    prediction2 = _rehash(prediction, dependencies=[fit2["job_id"]], seed=seed)
    kwargs = valid_kwargs(model_id="D0-M")
    kwargs["fit_job"] = fit2
    kwargs["prediction_job"] = prediction2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    assert _reason(kwargs) == "invalid_job"


def test_non_exact_str_pin_rejected():
    class Pin(str):
        pass

    kwargs = valid_kwargs()
    kwargs["expected_fit_job_id"] = Pin(kwargs["expected_fit_job_id"])
    assert _reason(kwargs) == "job_identity_mismatch"


def test_float_seed_rejected():
    fit, prediction = make_pair("D0-M")
    fit2 = _rehash(fit, seed=1.0)
    prediction2 = _rehash(prediction, dependencies=[fit2["job_id"]], seed=1.0)
    kwargs = valid_kwargs(model_id="D0-M")
    kwargs["fit_job"] = fit2
    kwargs["prediction_job"] = prediction2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    assert _reason(kwargs) == "invalid_job"


def test_unknown_registered_seed_rejected():
    unknown_seed = max(SEEDS) + 1
    fit, prediction = make_pair("D0-M")
    fit2 = _rehash(fit, seed=unknown_seed)
    prediction2 = _rehash(prediction, dependencies=[fit2["job_id"]], seed=unknown_seed)
    kwargs = valid_kwargs(model_id="D0-M")
    kwargs["fit_job"] = fit2
    kwargs["prediction_job"] = prediction2
    kwargs["expected_fit_job_id"] = fit2["job_id"]
    kwargs["expected_prediction_job_id"] = prediction2["job_id"]
    assert _reason(kwargs) == "invalid_job"


def test_stale_internal_job_hash_rejected():
    kwargs = valid_kwargs()
    broken = dict(kwargs["fit_job"])
    broken["context_id"] = "ctx-changed"
    kwargs["fit_job"] = broken
    assert _reason(kwargs) == "job_identity_mismatch"


def test_negative_infinity_rejected():
    kwargs = valid_kwargs()
    scores = np.zeros((3, 2), dtype=np.float64)
    scores[0, 1] = -np.inf
    kwargs["scores"] = scores
    assert _reason(kwargs) == "nonfinite_scores"


def test_non_ascii_validation_data_accepted():
    uids = ("échantillon-1", "サンプル-2", "mẫu-3")
    classes = ("класс-甲", "クラス-乙")
    report = sp.verify_source_prediction_values(**valid_kwargs(uids=uids, classes=classes))
    assert report["row_count"] == 3
    assert report["class_count"] == 2


def test_noncallable_hash_sanitized(monkeypatch):
    monkeypatch.setattr(sp, "sha256_value", None)
    with pytest.raises(sp.PredictionError) as info:
        sp.verify_source_prediction_values(**valid_kwargs())
    assert info.value.reason_code == "verification_failed"
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True


def test_ordinary_hash_failure_sanitized(monkeypatch):
    def boom(_value):
        raise RuntimeError("hash detail")

    monkeypatch.setattr(sp, "sha256_value", boom)
    with pytest.raises(sp.PredictionError) as info:
        sp.verify_source_prediction_values(**valid_kwargs())
    assert info.value.reason_code == "verification_failed"
    assert "hash detail" not in str(info.value)
    assert info.value.__cause__ is None


def test_late_base_exception_propagates_same_object(monkeypatch):
    signal = KeyboardInterrupt("late stop")
    real = sp.sha256_value
    state = {"calls": 0}

    def hooked(value):
        state["calls"] += 1
        if state["calls"] >= 2:
            raise signal
        return real(value)

    monkeypatch.setattr(sp, "sha256_value", hooked)
    with pytest.raises(KeyboardInterrupt) as info:
        sp.verify_source_prediction_values(**valid_kwargs())
    assert info.value is signal


def test_extra_fit_dependencies_rejected_before_hashing(monkeypatch):
    fit, prediction = make_pair("C-RBF-SVM")
    extra_fit = _rehash(fit, dependencies=[prediction["job_id"]])
    kwargs = valid_kwargs()
    kwargs["fit_job"] = extra_fit
    kwargs["expected_fit_job_id"] = extra_fit["job_id"]
    calls = []
    real = sp._hash

    def counting(body):
        calls.append(1)
        return real(body)

    monkeypatch.setattr(sp, "_hash", counting)
    assert _reason(kwargs) == "job_pair_mismatch"
    assert calls == []


def test_empty_prediction_dependencies_rejected_before_hashing(monkeypatch):
    fit, prediction = make_pair("C-RBF-SVM")
    empty_prediction = _rehash(prediction, dependencies=[])
    kwargs = valid_kwargs()
    kwargs["prediction_job"] = empty_prediction
    kwargs["expected_prediction_job_id"] = empty_prediction["job_id"]
    calls = []
    real = sp._hash

    def counting(body):
        calls.append(1)
        return real(body)

    monkeypatch.setattr(sp, "_hash", counting)
    assert _reason(kwargs) == "job_pair_mismatch"
    assert calls == []
