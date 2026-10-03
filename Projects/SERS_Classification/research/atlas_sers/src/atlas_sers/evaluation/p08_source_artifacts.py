"""Bounded checkpoint-content verification for the prospective U0 runtime.

This module verifies that a serialized neural checkpoint contains a
``{'state_dict': ...}`` wrapper whose tensors match the in-tree
:class:`~atlas_sers.models.acquisition.AcquisitionClassifier` layout for one
of the four declared source-only recipes (``D0-M``, ``D1``, ``D2``, ``D3``)
and either two or three station-local classes.

Non-claims
----------
* This is a *trusted-artifact content check*, not a hostile-pickle sandbox.
  It deserializes supplied bytes with ``torch.load(..., weights_only=True)``;
  that improves safety relative to permissive loading but is not a general
  security boundary for arbitrary untrusted artifacts.
* ``MAXIMUM_CHECKPOINT_BYTES`` is a cheap rejection guard on the input blob.
  It does not prove a hard bound on peak deserialization memory: structurally
  dense or compressed payloads may expand before tensors are inspected.
* The verifier never modifies the torch serialization allowlist, the caller
  default dtype, the caller default device or the CPU RNG state (beyond a
  restored ``fork_rng`` scope).
* A passing report establishes only that the bytes deserialize into the
  expected architecture with matching file and state hashes.  It does not
  establish job provenance, class-label ordering, epoch accounting, training
  completion, prediction parity or any scientific quality claim.  Future
  independently authenticated job/receipt/spec metadata must bind those
  fields.
"""

from __future__ import annotations

import hashlib
import io
from collections import OrderedDict
from typing import Any

import torch

from atlas_sers.evaluation.p04_runtime import _state_hash
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256
from atlas_sers.models.acquisition import AcquisitionClassifier

SCHEMA_VERSION = "nato-sers-p08-neural-checkpoint-content-v1"
MAXIMUM_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAXIMUM_PARAMETERS_EXCLUSIVE = 250000
PROJECTION_RECIPES = {"D0-M": False, "D1": True, "D2": False, "D3": True}
EXPECTED_PARAMETER_COUNTS = {
    (2, False): 208626,
    (2, True): 212786,
    (3, False): 208691,
    (3, True): 212851,
}
_HEX_CHARS = frozenset("0123456789abcdef")
_REASON_CODES = frozenset(
    {
        "scientific_execution_not_authorized",
        "invalid_checkpoint_bytes",
        "checkpoint_too_large",
        "invalid_expected_file_sha256",
        "invalid_expected_state_sha256",
        "invalid_class_count",
        "invalid_recipe_id",
        "checkpoint_file_hash_mismatch",
        "checkpoint_deserialization_failed",
        "checkpoint_verification_failed",
        "invalid_checkpoint_wrapper",
        "invalid_checkpoint_state",
        "checkpoint_state_key_mismatch",
        "checkpoint_state_shape_mismatch",
        "checkpoint_state_dtype_mismatch",
        "checkpoint_state_layout_invalid",
        "checkpoint_state_not_finite",
        "checkpoint_state_hash_mismatch",
        "checkpoint_parameter_count_mismatch",
        "unlisted_reason_code",
    }
)

__all__ = [
    "EXPECTED_PARAMETER_COUNTS",
    "MAXIMUM_CHECKPOINT_BYTES",
    "MAXIMUM_PARAMETERS_EXCLUSIVE",
    "PROJECTION_RECIPES",
    "SCHEMA_VERSION",
    "SourceArtifactError",
    "require_scientific_execution",
    "verify_neural_checkpoint_bytes",
]


class SourceArtifactError(ValueError):
    """ValueError carrying one static allowlisted reason code."""

    def __init__(self, reason_code: Any) -> None:
        if type(reason_code) is not str or reason_code not in _REASON_CODES:
            reason_code = "unlisted_reason_code"
        super().__init__(reason_code)
        self.reason_code = reason_code


def _fail(reason_code: str) -> None:
    raise SourceArtifactError(reason_code) from None


def _is_lower_hex64(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(ch in _HEX_CHARS for ch in value)


def _deserialize(blob: bytes) -> Any:
    try:
        return torch.load(io.BytesIO(blob), weights_only=True, map_location="cpu")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("checkpoint_deserialization_failed")


def _extract_state(obj: Any) -> dict:
    if type(obj) is not dict:
        _fail("invalid_checkpoint_wrapper")
    keys = list(obj.keys())
    if len(keys) != 1:
        _fail("invalid_checkpoint_wrapper")
    if type(keys[0]) is not str or keys[0] != "state_dict":
        _fail("invalid_checkpoint_wrapper")
    state = obj[keys[0]]
    if type(state) is not dict and type(state) is not OrderedDict:
        _fail("invalid_checkpoint_state")
    for state_key, value in state.items():
        if type(state_key) is not str:
            _fail("invalid_checkpoint_state")
        if type(value) is not torch.Tensor:
            _fail("invalid_checkpoint_state")
    return state


def _reference_model(class_count: int, use_projection: bool) -> AcquisitionClassifier:
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        model = AcquisitionClassifier(class_count, use_projection=use_projection)
        model = model.to(device=torch.device("cpu"), dtype=torch.float32)
    return model


def _validate_state_against(reference: AcquisitionClassifier, state: dict) -> None:
    reference_state = reference.state_dict()
    if set(state.keys()) != set(reference_state.keys()):
        _fail("checkpoint_state_key_mismatch")
    for name, expected in reference_state.items():
        tensor = state[name]
        if tensor.shape != expected.shape:
            _fail("checkpoint_state_shape_mismatch")
        if tensor.device.type != "cpu":
            _fail("checkpoint_state_layout_invalid")
        if tensor.layout != torch.strided:
            _fail("checkpoint_state_layout_invalid")
        if tensor.is_quantized:
            _fail("checkpoint_state_layout_invalid")
        if tensor.is_complex():
            _fail("checkpoint_state_dtype_mismatch")
        if tensor.dtype != torch.float32:
            _fail("checkpoint_state_dtype_mismatch")
        if not bool(torch.isfinite(tensor).all()):
            _fail("checkpoint_state_not_finite")


def _state_digest(state: dict) -> str:
    try:
        return _state_hash(state)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("invalid_checkpoint_state")


def _verify_neural_checkpoint_bytes(
    blob: Any,
    *,
    expected_file_sha256: Any,
    expected_state_sha256: Any,
    class_count: Any,
    recipe_id: Any,
) -> dict:
    """Verify a trusted-artifact checkpoint blob and return a scalar report."""
    if type(blob) is not bytes:
        _fail("invalid_checkpoint_bytes")
    if len(blob) == 0:
        _fail("invalid_checkpoint_bytes")
    if len(blob) > MAXIMUM_CHECKPOINT_BYTES:
        _fail("checkpoint_too_large")
    if not _is_lower_hex64(expected_file_sha256):
        _fail("invalid_expected_file_sha256")
    if not _is_lower_hex64(expected_state_sha256):
        _fail("invalid_expected_state_sha256")
    if type(class_count) is not int or class_count not in (2, 3):
        _fail("invalid_class_count")
    if type(recipe_id) is not str or recipe_id not in PROJECTION_RECIPES:
        _fail("invalid_recipe_id")

    file_sha256 = hashlib.sha256(blob).hexdigest()
    if file_sha256 != expected_file_sha256:
        _fail("checkpoint_file_hash_mismatch")

    use_projection = PROJECTION_RECIPES[recipe_id]
    state = _extract_state(_deserialize(blob))
    reference = _reference_model(class_count, use_projection)
    _validate_state_against(reference, state)

    state_sha256 = _state_digest(state)
    if state_sha256 != expected_state_sha256:
        _fail("checkpoint_state_hash_mismatch")

    parameter_count = int(sum(parameter.numel() for parameter in reference.parameters()))
    state_element_count = int(sum(value.numel() for value in state.values()))
    if state_element_count != parameter_count:
        _fail("checkpoint_parameter_count_mismatch")
    if parameter_count != EXPECTED_PARAMETER_COUNTS[(class_count, use_projection)]:
        _fail("checkpoint_parameter_count_mismatch")
    if parameter_count >= MAXIMUM_PARAMETERS_EXCLUSIVE:
        _fail("checkpoint_parameter_count_mismatch")

    report = {
        "schema_version": SCHEMA_VERSION,
        "execution_authorized": False,
        "checkpoint_content_verified": True,
        "job_provenance_verified": False,
        "training_completion_verified": False,
        "prediction_parity_verified": False,
        "class_count": class_count,
        "recipe_id": recipe_id,
        "use_projection": use_projection,
        "parameter_count": parameter_count,
        "checkpoint_file_sha256": file_sha256,
        "checkpoint_state_sha256": state_sha256,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def verify_neural_checkpoint_bytes(
    blob: Any,
    *,
    expected_file_sha256: Any,
    expected_state_sha256: Any,
    class_count: Any,
    recipe_id: Any,
) -> dict:
    """Verify a trusted-artifact checkpoint blob and return a scalar report.

    Ordinary failures collapse to one static allowlisted reason code so that
    no underlying exception text can leak; explicitly raised
    :class:`SourceArtifactError` reasons pass through, and
    ``KeyboardInterrupt``/``SystemExit`` propagate as the same object.
    """
    try:
        return _verify_neural_checkpoint_bytes(
            blob,
            expected_file_sha256=expected_file_sha256,
            expected_state_sha256=expected_state_sha256,
            class_count=class_count,
            recipe_id=recipe_id,
        )
    except SourceArtifactError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        _fail("checkpoint_verification_failed")


def require_scientific_execution(*args: Any, **kwargs: Any) -> None:
    """Always deny execution, regardless of forged flags or arguments."""
    _fail("scientific_execution_not_authorized")
