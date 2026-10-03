"""Synthetic tests for bounded P08 neural-checkpoint content verification."""

from __future__ import annotations

import hashlib
import io
from collections import OrderedDict

import pytest
import torch

from atlas_sers.evaluation import p08_source_artifacts as module
from atlas_sers.evaluation.p04_runtime import _state_hash
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256
from atlas_sers.evaluation.p08_source_artifacts import (
    SCHEMA_VERSION,
    SourceArtifactError,
    require_scientific_execution,
    verify_neural_checkpoint_bytes,
)
from atlas_sers.models.acquisition import AcquisitionClassifier

RECIPES = {"D0-M": False, "D1": True, "D2": False, "D3": True}
EXPECTED_COUNTS = {
    (2, False): 208626,
    (2, True): 212786,
    (3, False): 208691,
    (3, True): 212851,
}
_UNSET = object()


def _make_state(class_count, use_projection, seed=17):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AcquisitionClassifier(class_count, use_projection=use_projection)
    model = model.to(device=torch.device("cpu"), dtype=torch.float32)
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _serialize(state):
    buffer = io.BytesIO()
    torch.save({"state_dict": state}, buffer)
    return buffer.getvalue()


def _bundle(class_count, use_projection, seed=17):
    state = _make_state(class_count, use_projection, seed)
    blob = _serialize(state)
    return {
        "state": state,
        "blob": blob,
        "file_sha256": hashlib.sha256(blob).hexdigest(),
        "state_sha256": _state_hash(state),
    }


def _verify_bundle(bundle, class_count, recipe_id):
    return verify_neural_checkpoint_bytes(
        bundle["blob"],
        expected_file_sha256=bundle["file_sha256"],
        expected_state_sha256=bundle["state_sha256"],
        class_count=class_count,
        recipe_id=recipe_id,
    )


def _expect(
    reason,
    blob,
    *,
    class_count=3,
    recipe_id="D0-M",
    file_sha=_UNSET,
    state_sha=_UNSET,
):
    if file_sha is _UNSET:
        file_sha = hashlib.sha256(blob).hexdigest()
    if state_sha is _UNSET:
        state_sha = "0" * 64
    with pytest.raises(SourceArtifactError) as excinfo:
        verify_neural_checkpoint_bytes(
            blob,
            expected_file_sha256=file_sha,
            expected_state_sha256=state_sha,
            class_count=class_count,
            recipe_id=recipe_id,
        )
    assert excinfo.value.reason_code == reason
    return excinfo.value


def _patch_load(monkeypatch, value):
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: value)


class _IntSubclass(int):
    pass


@pytest.mark.parametrize("class_count", [2, 3])
@pytest.mark.parametrize("recipe_id", sorted(RECIPES))
def test_verifies_every_recipe_and_class_count(class_count, recipe_id):
    use_projection = RECIPES[recipe_id]
    bundle = _bundle(class_count, use_projection)
    report = _verify_bundle(bundle, class_count, recipe_id)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["execution_authorized"] is False
    assert report["checkpoint_content_verified"] is True
    assert report["job_provenance_verified"] is False
    assert report["training_completion_verified"] is False
    assert report["prediction_parity_verified"] is False
    assert report["class_count"] == class_count
    assert report["recipe_id"] == recipe_id
    assert report["use_projection"] == use_projection
    assert report["parameter_count"] == EXPECTED_COUNTS[(class_count, use_projection)]
    assert report["checkpoint_file_sha256"] == bundle["file_sha256"]
    assert report["checkpoint_state_sha256"] == bundle["state_sha256"]


def test_report_hash_is_recomputable_and_deterministic():
    bundle = _bundle(3, True)
    first = _verify_bundle(bundle, 3, "D1")
    second = _verify_bundle(bundle, 3, "D1")
    assert first == second
    body = {key: value for key, value in first.items() if key != "report_sha256"}
    assert first["report_sha256"] == canonical_sha256(body)
    for value in first.values():
        assert isinstance(value, (str, bool, int))


def test_all_zero_state_is_accepted():
    state = _make_state(3, False)
    zeroed = {name: torch.zeros_like(value) for name, value in state.items()}
    blob = _serialize(zeroed)
    report = verify_neural_checkpoint_bytes(
        blob,
        expected_file_sha256=hashlib.sha256(blob).hexdigest(),
        expected_state_sha256=_state_hash(zeroed),
        class_count=3,
        recipe_id="D0-M",
    )
    assert report["checkpoint_content_verified"] is True


def test_accepts_dict_and_ordered_dict_state_containers():
    state = _make_state(2, True)
    for container in (dict(state.items()), OrderedDict(state.items())):
        blob = _serialize(container)
        report = verify_neural_checkpoint_bytes(
            blob,
            expected_file_sha256=hashlib.sha256(blob).hexdigest(),
            expected_state_sha256=_state_hash(container),
            class_count=2,
            recipe_id="D1",
        )
        assert report["checkpoint_content_verified"] is True


def test_cpu_rng_state_and_default_dtype_unchanged():
    with torch.random.fork_rng(devices=[]):
        bundle = _bundle(3, False)
        torch.manual_seed(4242)
        before_rng = torch.random.get_rng_state().clone()
        before_dtype = torch.get_default_dtype()
        _verify_bundle(bundle, 3, "D0-M")
        assert torch.get_default_dtype() == before_dtype
        assert torch.equal(torch.random.get_rng_state(), before_rng)


def test_float64_default_dtype_is_restored_and_checkpoint_verified():
    bundle = _bundle(2, False)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        report = _verify_bundle(bundle, 2, "D0-M")
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)
    assert torch.get_default_dtype() == previous
    assert report["checkpoint_content_verified"] is True


def test_torch_load_receives_locked_weights_only_kwargs(monkeypatch):
    bundle = _bundle(3, False)
    real_load = torch.load
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    _verify_bundle(bundle, 3, "D0-M")
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 1
    assert isinstance(args[0], io.BytesIO)
    assert args[0].getvalue() == bundle["blob"]
    assert kwargs == {"weights_only": True, "map_location": "cpu"}


def test_file_hash_mismatch_stops_load_and_model_construction(monkeypatch):
    bundle = _bundle(3, False)
    load_calls = []
    construct_calls = []

    def forbidden_load(*args, **kwargs):
        load_calls.append(1)
        raise AssertionError("torch.load must not run")

    class ForbiddenModel:
        def __init__(self, *args, **kwargs):
            construct_calls.append(1)
            raise AssertionError("model construction must not run")

    monkeypatch.setattr(torch, "load", forbidden_load)
    monkeypatch.setattr(module, "AcquisitionClassifier", ForbiddenModel)
    _expect(
        "checkpoint_file_hash_mismatch",
        bundle["blob"],
        class_count=3,
        recipe_id="D0-M",
        file_sha="0" * 64,
        state_sha=bundle["state_sha256"],
    )
    assert load_calls == []
    assert construct_calls == []


def test_corrupt_bytes_with_matching_file_hash_fail_deserialization():
    _expect("checkpoint_deserialization_failed", b"not-a-torch-checkpoint")


@pytest.mark.parametrize(
    "bad_blob",
    [bytearray(b"x"), memoryview(b"x"), "x", 123, None, b""],
)
def test_invalid_blob_rejected(bad_blob):
    with pytest.raises(SourceArtifactError) as excinfo:
        verify_neural_checkpoint_bytes(
            bad_blob,
            expected_file_sha256="a" * 64,
            expected_state_sha256="b" * 64,
            class_count=3,
            recipe_id="D0-M",
        )
    assert excinfo.value.reason_code == "invalid_checkpoint_bytes"


def test_oversize_blob_rejected_without_large_allocation(monkeypatch):
    bundle = _bundle(3, False)
    monkeypatch.setattr(module, "MAXIMUM_CHECKPOINT_BYTES", len(bundle["blob"]) - 1)
    _expect(
        "checkpoint_too_large",
        bundle["blob"],
        class_count=3,
        recipe_id="D0-M",
        file_sha=bundle["file_sha256"],
        state_sha=bundle["state_sha256"],
    )


@pytest.mark.parametrize(
    "bad",
    [None, 123, True, b"x" * 64, "A" * 64, "a" * 63, "a" * 65, "g" * 64],
)
def test_invalid_expected_file_sha_rejected(bad):
    _expect("invalid_expected_file_sha256", b"x", file_sha=bad)


@pytest.mark.parametrize("bad", [None, 7, False, b"x" * 64, "Z" * 64, "a" * 62])
def test_invalid_expected_state_sha_rejected(bad):
    _expect(
        "invalid_expected_state_sha256",
        b"x",
        file_sha=hashlib.sha256(b"x").hexdigest(),
        state_sha=bad,
    )


@pytest.mark.parametrize("bad", [True, False, 1, 4, 0, 2.0, 3.0, "3", None, _IntSubclass(3)])
def test_invalid_class_count_rejected(bad):
    _expect("invalid_class_count", b"x", class_count=bad)


@pytest.mark.parametrize("bad", ["D0", "d0-m", "D4", "", "D0-MM", None, 1, b"D0-M"])
def test_invalid_recipe_id_rejected(bad):
    _expect("invalid_recipe_id", b"x", recipe_id=bad)


@pytest.mark.parametrize(
    "obj",
    [["state_dict"], 5, None, {"state_dict": {}, "extra": 1}, {"other": 1}],
)
def test_invalid_wrapper_rejected(monkeypatch, obj):
    _patch_load(monkeypatch, obj)
    _expect("invalid_checkpoint_wrapper", b"x")


@pytest.mark.parametrize("state", [[1, 2], 5, None, "state"])
def test_invalid_state_container_rejected(monkeypatch, state):
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("invalid_checkpoint_state", b"x")


def test_non_tensor_state_value_rejected(monkeypatch):
    state = _make_state(3, False)
    name = next(iter(state))
    state[name] = 5
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("invalid_checkpoint_state", b"x")


def test_parameter_state_value_rejected(monkeypatch):
    state = _make_state(3, False)
    name = next(iter(state))
    state[name] = torch.nn.Parameter(torch.zeros_like(state[name]))
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("invalid_checkpoint_state", b"x")


def test_non_string_state_key_rejected(monkeypatch):
    state = _make_state(3, False)
    name = next(iter(state))
    value = state.pop(name)
    state[123] = value
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("invalid_checkpoint_state", b"x")


def test_missing_state_key_rejected(monkeypatch):
    state = _make_state(3, False)
    state.pop(next(iter(state)))
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("checkpoint_state_key_mismatch", b"x")


def test_extra_state_key_rejected(monkeypatch):
    state = _make_state(3, False)
    state["unexpected_state_key"] = torch.zeros(1, dtype=torch.float32)
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("checkpoint_state_key_mismatch", b"x")


def test_wrong_state_shape_rejected(monkeypatch):
    state = _make_state(3, False)
    name = next(iter(state))
    state[name] = torch.zeros(state[name].numel() + 1, dtype=torch.float32)
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("checkpoint_state_shape_mismatch", b"x")


def test_wrong_state_dtype_rejected(monkeypatch):
    state = _make_state(3, False)
    name = next(iter(state))
    state[name] = state[name].to(torch.float64)
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("checkpoint_state_dtype_mismatch", b"x")


def test_wrong_class_count_shapes_rejected():
    bundle = _bundle(2, False)
    _expect(
        "checkpoint_state_shape_mismatch",
        bundle["blob"],
        class_count=3,
        recipe_id="D0-M",
        file_sha=bundle["file_sha256"],
        state_sha=bundle["state_sha256"],
    )


def test_wrong_projection_keys_rejected():
    bundle = _bundle(3, True)
    _expect(
        "checkpoint_state_key_mismatch",
        bundle["blob"],
        class_count=3,
        recipe_id="D0-M",
        file_sha=bundle["file_sha256"],
        state_sha=bundle["state_sha256"],
    )


def test_sparse_state_tensor_rejected(monkeypatch):
    state = _make_state(3, False)
    name = next(iter(state))
    shape = tuple(state[name].shape)
    indices = torch.zeros((len(shape), 0), dtype=torch.long)
    values = torch.zeros(0, dtype=torch.float32)
    state[name] = torch.sparse_coo_tensor(indices, values, size=shape)
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("checkpoint_state_layout_invalid", b"x")


def test_quantized_state_tensor_rejected(monkeypatch):
    state = _make_state(3, False)
    name = next(iter(state))
    state[name] = torch.quantize_per_tensor(
        state[name].to(torch.float32), scale=0.1, zero_point=0, dtype=torch.quint8
    )
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("checkpoint_state_layout_invalid", b"x")


def test_meta_state_tensor_rejected(monkeypatch):
    state = _make_state(3, False)
    name = next(iter(state))
    state[name] = torch.zeros(tuple(state[name].shape), dtype=torch.float32, device="meta")
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("checkpoint_state_layout_invalid", b"x")


def test_complex_state_tensor_rejected(monkeypatch):
    state = _make_state(3, False)
    name = next(iter(state))
    state[name] = torch.zeros(tuple(state[name].shape), dtype=torch.complex64)
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("checkpoint_state_dtype_mismatch", b"x")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_state_tensor_rejected(monkeypatch, bad):
    state = _make_state(3, False)
    name = next(iter(state))
    state[name] = torch.full_like(state[name], bad)
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("checkpoint_state_not_finite", b"x")


def test_wrong_state_hash_rejected():
    bundle = _bundle(3, False)
    _expect(
        "checkpoint_state_hash_mismatch",
        bundle["blob"],
        class_count=3,
        recipe_id="D0-M",
        file_sha=bundle["file_sha256"],
        state_sha="0" * 64,
    )


def test_serialized_float64_state_rejected():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        model = AcquisitionClassifier(3, use_projection=False).to(torch.float64)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    blob = _serialize(state)
    _expect(
        "checkpoint_state_dtype_mismatch",
        blob,
        class_count=3,
        recipe_id="D0-M",
        file_sha=hashlib.sha256(blob).hexdigest(),
        state_sha=_state_hash(state),
    )


def test_keyboard_interrupt_propagates_unchanged(monkeypatch):
    boom = KeyboardInterrupt("stop")

    def raising(*args, **kwargs):
        raise boom

    monkeypatch.setattr(torch, "load", raising)
    blob = b"interrupt"
    with pytest.raises(KeyboardInterrupt) as excinfo:
        verify_neural_checkpoint_bytes(
            blob,
            expected_file_sha256=hashlib.sha256(blob).hexdigest(),
            expected_state_sha256="0" * 64,
            class_count=3,
            recipe_id="D0-M",
        )
    assert excinfo.value is boom


def test_system_exit_propagates_unchanged(monkeypatch):
    boom = SystemExit(7)

    def raising(*args, **kwargs):
        raise boom

    monkeypatch.setattr(torch, "load", raising)
    blob = b"exit"
    with pytest.raises(SystemExit) as excinfo:
        verify_neural_checkpoint_bytes(
            blob,
            expected_file_sha256=hashlib.sha256(blob).hexdigest(),
            expected_state_sha256="0" * 64,
            class_count=3,
            recipe_id="D0-M",
        )
    assert excinfo.value is boom


def test_require_scientific_execution_always_denies():
    for args, kwargs in [((), {}), ((1, 2), {"a": 3}), ((object(),), {"b": object()})]:
        with pytest.raises(SourceArtifactError) as excinfo:
            require_scientific_execution(*args, **kwargs)
        assert excinfo.value.reason_code == "scientific_execution_not_authorized"


class _DictSubclass(dict):
    pass


class _OrderedDictSubclass(OrderedDict):
    pass


class _StrSubclass(str):
    pass


def _assert_sanitized(error):
    assert error.reason_code == "checkpoint_verification_failed"
    assert "SYNTHETIC_PRIVATE_DETAIL" not in str(error)
    assert error.__cause__ is None
    assert error.__suppress_context__ is True


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {},
        object(),
        5,
        None,
        b"checkpoint_file_hash_mismatch",
        _StrSubclass("checkpoint_file_hash_mismatch"),
        ["SYNTHETIC_PRIVATE_DETAIL"],
    ],
)
def test_source_artifact_error_sanitizes_reason_inputs(bad):
    error = SourceArtifactError(bad)
    assert error.reason_code == "unlisted_reason_code"
    assert str(error) == "unlisted_reason_code"


def test_source_artifact_error_accepts_exact_allowlisted_code():
    error = SourceArtifactError("checkpoint_file_hash_mismatch")
    assert error.reason_code == "checkpoint_file_hash_mismatch"
    assert str(error) == "checkpoint_file_hash_mismatch"


def test_dict_subclass_wrapper_rejected(monkeypatch):
    state = _make_state(3, False)
    _patch_load(monkeypatch, _DictSubclass({"state_dict": state}))
    _expect("invalid_checkpoint_wrapper", b"x")


def test_string_subclass_wrapper_key_rejected(monkeypatch):
    state = _make_state(3, False)
    _patch_load(monkeypatch, {_StrSubclass("state_dict"): state})
    _expect("invalid_checkpoint_wrapper", b"x")


def test_dict_subclass_state_rejected(monkeypatch):
    state = _make_state(3, False)
    _patch_load(monkeypatch, {"state_dict": _DictSubclass(state)})
    _expect("invalid_checkpoint_state", b"x")


def test_ordered_dict_subclass_state_rejected(monkeypatch):
    state = _make_state(3, False)
    _patch_load(monkeypatch, {"state_dict": _OrderedDictSubclass(state)})
    _expect("invalid_checkpoint_state", b"x")


def test_string_subclass_state_key_rejected(monkeypatch):
    state = _make_state(3, False)
    name = next(iter(state))
    value = state.pop(name)
    state[_StrSubclass(name)] = value
    _patch_load(monkeypatch, {"state_dict": state})
    _expect("invalid_checkpoint_state", b"x")


def test_constructor_failure_sanitized(monkeypatch):
    bundle = _bundle(3, False)

    class _BoomModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("SYNTHETIC_PRIVATE_DETAIL")

    monkeypatch.setattr(module, "AcquisitionClassifier", _BoomModel)
    error = _expect(
        "checkpoint_verification_failed",
        bundle["blob"],
        file_sha=bundle["file_sha256"],
        state_sha=bundle["state_sha256"],
    )
    _assert_sanitized(error)


def test_validation_failure_sanitized(monkeypatch):
    bundle = _bundle(3, False)

    def boom(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_PRIVATE_DETAIL")

    monkeypatch.setattr(torch, "isfinite", boom)
    error = _expect(
        "checkpoint_verification_failed",
        bundle["blob"],
        file_sha=bundle["file_sha256"],
        state_sha=bundle["state_sha256"],
    )
    _assert_sanitized(error)


def test_report_hash_failure_sanitized(monkeypatch):
    bundle = _bundle(3, False)

    def boom(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_PRIVATE_DETAIL")

    monkeypatch.setattr(module, "canonical_sha256", boom)
    error = _expect(
        "checkpoint_verification_failed",
        bundle["blob"],
        file_sha=bundle["file_sha256"],
        state_sha=bundle["state_sha256"],
    )
    _assert_sanitized(error)


def test_failed_constructor_restores_cpu_rng(monkeypatch):
    bundle = _bundle(3, False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        before = torch.random.get_rng_state().clone()

        class _BoomModel:
            def __init__(self, *args, **kwargs):
                torch.rand(16)
                raise RuntimeError("SYNTHETIC_PRIVATE_DETAIL")

        monkeypatch.setattr(module, "AcquisitionClassifier", _BoomModel)
        error = _expect(
            "checkpoint_verification_failed",
            bundle["blob"],
            file_sha=bundle["file_sha256"],
            state_sha=bundle["state_sha256"],
        )
        _assert_sanitized(error)
        assert torch.equal(torch.random.get_rng_state(), before)


def test_keyboard_interrupt_from_constructor_same_object(monkeypatch):
    bundle = _bundle(3, False)
    boom = KeyboardInterrupt("stop")

    class _Interrupt:
        def __init__(self, *args, **kwargs):
            raise boom

    monkeypatch.setattr(module, "AcquisitionClassifier", _Interrupt)
    with pytest.raises(KeyboardInterrupt) as excinfo:
        _verify_bundle(bundle, 3, "D0-M")
    assert excinfo.value is boom


def test_system_exit_from_report_hash_same_object(monkeypatch):
    bundle = _bundle(3, False)
    boom = SystemExit(9)

    def raising(*args, **kwargs):
        raise boom

    monkeypatch.setattr(module, "canonical_sha256", raising)
    with pytest.raises(SystemExit) as excinfo:
        _verify_bundle(bundle, 3, "D0-M")
    assert excinfo.value is boom


def test_meta_default_device_verifies_saved_cpu_checkpoint(monkeypatch):
    bundle = _bundle(2, False)
    forward_calls = []
    cuda_calls = []

    def forbidden_forward(self, *args, **kwargs):
        forward_calls.append(1)
        raise AssertionError("model forward must not run")

    def forbidden_cuda(*args, **kwargs):
        cuda_calls.append(1)
        raise AssertionError("CUDA must not be used")

    monkeypatch.setattr(AcquisitionClassifier, "forward", forbidden_forward)
    monkeypatch.setattr(torch.cuda, "init", forbidden_cuda)
    previous_device = torch.get_default_device()
    torch.set_default_device("meta")
    try:
        report = _verify_bundle(bundle, 2, "D0-M")
        assert torch.get_default_device().type == "meta"
    finally:
        torch.set_default_device(previous_device)
    assert forward_calls == []
    assert cuda_calls == []
    assert report["checkpoint_content_verified"] is True


def test_float64_default_dtype_preserves_rng_and_checkpoint():
    bundle = _bundle(2, False)
    previous_dtype = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        before = torch.random.get_rng_state().clone()
        torch.set_default_dtype(torch.float64)
        try:
            report = _verify_bundle(bundle, 2, "D0-M")
            assert torch.get_default_dtype() == torch.float64
            assert torch.equal(torch.random.get_rng_state(), before)
        finally:
            torch.set_default_dtype(previous_dtype)
    assert torch.get_default_dtype() == previous_dtype
    assert report["checkpoint_content_verified"] is True
