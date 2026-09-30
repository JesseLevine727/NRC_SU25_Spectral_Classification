"""Guard tests for the pure P08 proposed-resource snapshot checker.

Only invented numeric inputs are used: no filesystem, device, model, private
data or execution access. These assertions describe the guard arithmetic; they
do not establish real resource availability, journal durability, job
selection or permission.
"""

import copy

import numpy as np
import pytest

from atlas_sers.evaluation import p08_resources as guard
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256

GiB = 1024**3
STAGES = ("U0", "U1", "Q1")

# Independent explicit ceilings (pow2 / integer nanoseconds); not the guard.
LIMITS = {
    "U0": {
        "model_fit_attempts": 78,
        "scalar_calibration_attempts": 0,
        "active_wall_ns": 5_400_000_000_000,
        "new_artifact_bytes": 8 * GiB,
        "process_tree_rss_bytes": 16 * GiB,
        "cuda_allocated_bytes": 8 * GiB,
        "max_cpu_workers": 2,
        "max_gpu_workers": 1,
        "worker_threads": 1,
        "filesystem_reserve_bytes": 30 * GiB,
        "automatic_retries": 0,
    },
    "U1": {
        "model_fit_attempts": 195_202,
        "scalar_calibration_attempts": 3_354,
        "active_wall_ns": 172_800_000_000_000,
        "new_artifact_bytes": 80 * GiB,
        "process_tree_rss_bytes": 24 * GiB,
        "cuda_allocated_bytes": 8 * GiB,
        "max_cpu_workers": 4,
        "max_gpu_workers": 1,
        "worker_threads": 1,
        "filesystem_reserve_bytes": 30 * GiB,
        "automatic_retries": 0,
    },
    "Q1": {
        "model_fit_attempts": 1_630_980,
        "scalar_calibration_attempts": 53_880,
        "active_wall_ns": 864_000_000_000_000,
        "new_artifact_bytes": 400 * GiB,
        "process_tree_rss_bytes": 24 * GiB,
        "cuda_allocated_bytes": 8 * GiB,
        "max_cpu_workers": 4,
        "max_gpu_workers": 1,
        "worker_threads": 1,
        "filesystem_reserve_bytes": 30 * GiB,
        "automatic_retries": 0,
    },
}

USAGE_FIELDS = (
    "model_fit_attempts",
    "scalar_calibration_attempts",
    "active_wall_ns",
    "new_artifact_bytes",
)
RESOURCE_FIELDS = (
    "filesystem_free_bytes",
    "process_tree_rss_bytes",
    "cuda_allocated_bytes",
    "cuda_reserved_bytes",
    "cuda_device_used_bytes",
    "active_cpu_workers",
    "active_gpu_workers",
    "model_threads",
    "blas_threads",
    "torch_threads",
)
RECORD_FIELDS = (
    "schema_version",
    "execution_authorized",
    "stage",
    "limits",
    "usage",
    "resources",
    "remaining_model_fit_attempts",
    "remaining_scalar_calibration_attempts",
    "remaining_active_wall_ns",
    "remaining_artifact_bytes",
    "required_filesystem_free_bytes",
    "within_proposed_limits",
    "breaches",
    "snapshot_sha256",
)
REMAINING_KEY = {
    "model_fit_attempts": "remaining_model_fit_attempts",
    "scalar_calibration_attempts": "remaining_scalar_calibration_attempts",
    "active_wall_ns": "remaining_active_wall_ns",
    "new_artifact_bytes": "remaining_artifact_bytes",
}
RESOURCE_CAP_KEY = {
    "process_tree_rss_bytes": "process_tree_rss_bytes",
    "cuda_allocated_bytes": "cuda_allocated_bytes",
    "active_cpu_workers": "max_cpu_workers",
    "active_gpu_workers": "max_gpu_workers",
}
WHITELIST = {
    "invalid_resource_input",
    "invalid_stage",
    "invalid_usage",
    "invalid_resources",
    "scientific_execution_not_authorized",
}


def make_usage(**overrides):
    base = dict.fromkeys(USAGE_FIELDS, 0)
    base.update(overrides)
    return base


def make_resources(**overrides):
    base = {
        "filesystem_free_bytes": 1024 * GiB,
        "process_tree_rss_bytes": 0,
        "cuda_allocated_bytes": 0,
        "cuda_reserved_bytes": 0,
        "cuda_device_used_bytes": 0,
        "active_cpu_workers": 0,
        "active_gpu_workers": 0,
        "model_threads": 1,
        "blas_threads": 1,
        "torch_threads": 1,
    }
    base.update(overrides)
    return base


def evaluate(stage="U0", usage=None, resources=None):
    return guard.evaluate_resource_snapshot(
        stage,
        make_usage() if usage is None else usage,
        make_resources() if resources is None else resources,
    )


@pytest.mark.parametrize("stage", STAGES)
def test_proposed_limits_exact_values_and_types(stage):
    limits = guard.proposed_limits(stage)
    assert set(limits) == {"stage"} | set(LIMITS[stage])
    assert limits["stage"] == stage
    for key, expected in LIMITS[stage].items():
        assert limits[key] == expected
        assert type(limits[key]) is int


@pytest.mark.parametrize("stage", STAGES)
def test_proposed_limits_returns_fresh_mutable_dict(stage):
    first = guard.proposed_limits(stage)
    first["model_fit_attempts"] = -123
    first["automatic_retries"] = 9
    first["injected"] = True
    second = guard.proposed_limits(stage)
    assert second["model_fit_attempts"] == LIMITS[stage]["model_fit_attempts"]
    assert second["automatic_retries"] == 0
    assert "injected" not in second
    assert second is not first


@pytest.mark.parametrize("stage", [None, [], {}, 0, True, "u0", "U2", "Q0", "stage"])
def test_proposed_limits_rejects_unknown_or_bad_stage(stage):
    with pytest.raises(guard.ResourceGuardError) as excinfo:
        guard.proposed_limits(stage)
    assert excinfo.value.reason_code == "invalid_stage"


@pytest.mark.parametrize("code", sorted(WHITELIST))
def test_error_keeps_known_reason_code(code):
    err = guard.ResourceGuardError(code)
    assert isinstance(err, ValueError)
    assert err.reason_code == code


@pytest.mark.parametrize("bad", [None, 3.5, object(), ["x"], {"y": 1}])
def test_error_maps_unknown_reason_without_echoing(bad):
    err = guard.ResourceGuardError(bad)
    assert err.reason_code == "invalid_resource_input"
    assert err.reason_code in WHITELIST
    text = str(err)
    assert repr(bad) not in text
    assert "sentinel" not in text.lower()
    assert "0x" not in text


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((), {}),
        ((True,), {}),
        ((), {"authorized": True}),
        ((), {"execution_authorized": True}),
        ((1, 2), {"permit": "forged", "approved": True}),
    ],
)
def test_require_scientific_execution_always_denies(args, kwargs):
    with pytest.raises(guard.ResourceGuardError) as excinfo:
        guard.require_scientific_execution(*args, **kwargs)
    assert excinfo.value.reason_code == "scientific_execution_not_authorized"


@pytest.mark.parametrize("stage", STAGES)
def test_record_schema_keys_defaults_and_authority(stage):
    result = evaluate(stage)
    assert set(result) == set(RECORD_FIELDS)
    assert list(evaluate(stage)) == list(result)
    assert result["schema_version"] == "nato-sers-p08-resource-snapshot-v1"
    assert result["execution_authorized"] is False
    assert result["stage"] == stage
    assert result["limits"] == {"stage": stage, **LIMITS[stage]}
    assert result["usage"] == make_usage()
    assert result["resources"] == make_resources()
    assert result["breaches"] == []
    assert result["within_proposed_limits"] is True


@pytest.mark.parametrize("stage", STAGES)
def test_snapshot_hash_is_canonical_and_deterministic(stage):
    result = evaluate(stage)
    digest = result["snapshot_sha256"]
    payload = {k: v for k, v in result.items() if k != "snapshot_sha256"}
    assert digest == canonical_sha256(payload)
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert digest == digest.lower()
    assert evaluate(stage)["snapshot_sha256"] == digest


def test_snapshot_hash_changes_with_inputs():
    assert (
        evaluate("U0")["snapshot_sha256"]
        != evaluate("U0", usage=make_usage(model_fit_attempts=1))["snapshot_sha256"]
    )


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(
    "field,code",
    [
        ("model_fit_attempts", "model_fit_budget_exceeded"),
        ("scalar_calibration_attempts", "scalar_calibration_budget_exceeded"),
    ],
)
def test_usage_count_boundaries(stage, field, code):
    cap = LIMITS[stage][field]
    if cap >= 1:
        below = evaluate(stage, usage=make_usage(**{field: cap - 1}))
        assert code not in below["breaches"]
        assert below["within_proposed_limits"] is True
        assert below[REMAINING_KEY[field]] == 1
    equal = evaluate(stage, usage=make_usage(**{field: cap}))
    assert code not in equal["breaches"]
    assert equal["within_proposed_limits"] is True
    assert equal[REMAINING_KEY[field]] == 0
    above = evaluate(stage, usage=make_usage(**{field: cap + 1}))
    assert code in above["breaches"]
    assert above["within_proposed_limits"] is False
    assert above[REMAINING_KEY[field]] == 0


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(
    "field,code",
    [
        ("active_wall_ns", "active_wall_budget_exhausted"),
        ("new_artifact_bytes", "artifact_budget_exhausted"),
    ],
)
def test_usage_exhaustion_boundaries(stage, field, code):
    cap = LIMITS[stage][field]
    below = evaluate(stage, usage=make_usage(**{field: cap - 1}))
    assert code not in below["breaches"]
    assert below["within_proposed_limits"] is True
    assert below[REMAINING_KEY[field]] == 1
    equal = evaluate(stage, usage=make_usage(**{field: cap}))
    assert code in equal["breaches"]
    assert equal["within_proposed_limits"] is False
    assert equal[REMAINING_KEY[field]] == 0
    above = evaluate(stage, usage=make_usage(**{field: cap + 1}))
    assert code in above["breaches"]
    assert above[REMAINING_KEY[field]] == 0


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize(
    "field,code",
    [
        ("process_tree_rss_bytes", "process_tree_memory_exceeded"),
        ("cuda_allocated_bytes", "cuda_allocated_memory_exceeded"),
        ("active_cpu_workers", "cpu_worker_limit_exceeded"),
        ("active_gpu_workers", "gpu_worker_limit_exceeded"),
    ],
)
def test_resource_boundaries(stage, field, code):
    cap = LIMITS[stage][RESOURCE_CAP_KEY[field]]
    below = evaluate(stage, resources=make_resources(**{field: cap - 1}))
    assert code not in below["breaches"]
    assert below["within_proposed_limits"] is True
    equal = evaluate(stage, resources=make_resources(**{field: cap}))
    assert code not in equal["breaches"]
    assert equal["within_proposed_limits"] is True
    above = evaluate(stage, resources=make_resources(**{field: cap + 1}))
    assert code in above["breaches"]
    assert above["within_proposed_limits"] is False


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("field", ["model_threads", "blas_threads", "torch_threads"])
def test_thread_settings_must_equal_one(stage, field):
    code = "worker_thread_limit_violated"
    ok = evaluate(stage, resources=make_resources(**{field: 1}))
    assert code not in ok["breaches"]
    assert ok["within_proposed_limits"] is True
    assert code in evaluate(stage, resources=make_resources(**{field: 0}))["breaches"]
    assert code in evaluate(stage, resources=make_resources(**{field: 2}))["breaches"]


@pytest.mark.parametrize("stage", STAGES)
def test_multiple_bad_threads_emit_one_code(stage):
    resources = make_resources(model_threads=2, blas_threads=0, torch_threads=3)
    result = evaluate(stage, resources=resources)
    assert result["breaches"].count("worker_thread_limit_violated") == 1


@pytest.mark.parametrize("stage", STAGES)
def test_filesystem_reserve_boundaries(stage):
    code = "filesystem_reserve_insufficient"
    usage = make_usage(new_artifact_bytes=1 * GiB)
    remaining = LIMITS[stage]["new_artifact_bytes"] - 1 * GiB
    required = LIMITS[stage]["filesystem_reserve_bytes"] + remaining
    assert required == 30 * GiB + LIMITS[stage]["new_artifact_bytes"] - 1 * GiB
    below = evaluate(
        stage, usage=usage, resources=make_resources(filesystem_free_bytes=required - 1)
    )
    assert code in below["breaches"]
    equal = evaluate(stage, usage=usage, resources=make_resources(filesystem_free_bytes=required))
    assert code not in equal["breaches"]
    assert equal["required_filesystem_free_bytes"] == required
    above = evaluate(
        stage, usage=usage, resources=make_resources(filesystem_free_bytes=required + 1)
    )
    assert code not in above["breaches"]


@pytest.mark.parametrize("stage", STAGES)
def test_reserved_and_device_used_never_replace_allocated(stage):
    cap = LIMITS[stage]["cuda_allocated_bytes"]
    resources = make_resources(
        cuda_allocated_bytes=cap,
        cuda_reserved_bytes=cap + 5 * GiB,
        cuda_device_used_bytes=cap + 9 * GiB,
    )
    result = evaluate(stage, resources=resources)
    assert "cuda_allocated_memory_exceeded" not in result["breaches"]
    assert result["within_proposed_limits"] is True
    assert result["resources"]["cuda_allocated_bytes"] == cap
    assert result["resources"]["cuda_reserved_bytes"] == cap + 5 * GiB
    assert result["resources"]["cuda_device_used_bytes"] == cap + 9 * GiB


def test_breach_codes_are_ordered_and_complete():
    usage = make_usage(
        model_fit_attempts=LIMITS["U0"]["model_fit_attempts"] + 1,
        scalar_calibration_attempts=LIMITS["U0"]["scalar_calibration_attempts"] + 1,
        active_wall_ns=LIMITS["U0"]["active_wall_ns"],
        new_artifact_bytes=LIMITS["U0"]["new_artifact_bytes"] + 1,
    )
    resources = make_resources(
        process_tree_rss_bytes=LIMITS["U0"]["process_tree_rss_bytes"] + 1,
        cuda_allocated_bytes=LIMITS["U0"]["cuda_allocated_bytes"] + 1,
        active_cpu_workers=LIMITS["U0"]["max_cpu_workers"] + 1,
        active_gpu_workers=LIMITS["U0"]["max_gpu_workers"] + 1,
        model_threads=2,
        filesystem_free_bytes=0,
    )
    result = evaluate("U0", usage=usage, resources=resources)
    assert result["breaches"] == [
        "model_fit_budget_exceeded",
        "scalar_calibration_budget_exceeded",
        "active_wall_budget_exhausted",
        "artifact_budget_exhausted",
        "process_tree_memory_exceeded",
        "cuda_allocated_memory_exceeded",
        "cpu_worker_limit_exceeded",
        "gpu_worker_limit_exceeded",
        "worker_thread_limit_violated",
        "filesystem_reserve_insufficient",
    ]
    assert result["within_proposed_limits"] is False
    assert result["execution_authorized"] is False


@pytest.mark.parametrize("stage", STAGES)
def test_remaining_values_clamped_without_underflow(stage):
    huge = 10**30
    usage = make_usage(
        model_fit_attempts=huge,
        scalar_calibration_attempts=huge,
        active_wall_ns=huge,
        new_artifact_bytes=huge,
    )
    result = evaluate(stage, usage=usage, resources=make_resources(filesystem_free_bytes=huge))
    for key in REMAINING_KEY.values():
        assert result[key] == 0
    assert result["required_filesystem_free_bytes"] == LIMITS[stage]["filesystem_reserve_bytes"]
    assert result["required_filesystem_free_bytes"] >= 0


_BAD_SCALARS = [
    pytest.param(True, id="bool"),
    pytest.param(1.0, id="float"),
    pytest.param(-1, id="negative"),
    pytest.param("1", id="string"),
    pytest.param(np.int64(1), id="numpy-int64"),
]


@pytest.mark.parametrize("field", USAGE_FIELDS)
@pytest.mark.parametrize("bad", _BAD_SCALARS)
def test_usage_rejects_bad_scalar_types(field, bad):
    with pytest.raises(guard.ResourceGuardError) as excinfo:
        guard.evaluate_resource_snapshot("U0", make_usage(**{field: bad}), make_resources())
    assert excinfo.value.reason_code == "invalid_usage"


@pytest.mark.parametrize("field", RESOURCE_FIELDS)
@pytest.mark.parametrize("bad", _BAD_SCALARS)
def test_resources_reject_bad_scalar_types(field, bad):
    with pytest.raises(guard.ResourceGuardError) as excinfo:
        guard.evaluate_resource_snapshot("U0", make_usage(), make_resources(**{field: bad}))
    assert excinfo.value.reason_code == "invalid_resources"


@pytest.mark.parametrize("field", USAGE_FIELDS)
def test_usage_missing_key_rejected(field):
    usage = make_usage()
    del usage[field]
    with pytest.raises(guard.ResourceGuardError) as excinfo:
        guard.evaluate_resource_snapshot("U0", usage, make_resources())
    assert excinfo.value.reason_code == "invalid_usage"


def test_usage_extra_key_rejected():
    usage = make_usage()
    usage["unexpected"] = 0
    with pytest.raises(guard.ResourceGuardError) as excinfo:
        guard.evaluate_resource_snapshot("U0", usage, make_resources())
    assert excinfo.value.reason_code == "invalid_usage"


@pytest.mark.parametrize("field", RESOURCE_FIELDS)
def test_resources_missing_key_rejected(field):
    resources = make_resources()
    del resources[field]
    with pytest.raises(guard.ResourceGuardError) as excinfo:
        guard.evaluate_resource_snapshot("U0", make_usage(), resources)
    assert excinfo.value.reason_code == "invalid_resources"


def test_resources_extra_key_rejected():
    resources = make_resources()
    resources["unexpected"] = 0
    with pytest.raises(guard.ResourceGuardError) as excinfo:
        guard.evaluate_resource_snapshot("U0", make_usage(), resources)
    assert excinfo.value.reason_code == "invalid_resources"


class _DictSubclass(dict):
    pass


@pytest.mark.parametrize("bad", [None, [], (), "usage", 3, set(), object(), _DictSubclass()])
def test_usage_wrong_container_rejected(bad):
    with pytest.raises(guard.ResourceGuardError) as excinfo:
        guard.evaluate_resource_snapshot("U0", bad, make_resources())
    assert excinfo.value.reason_code == "invalid_usage"


@pytest.mark.parametrize("bad", [None, [], (), "resources", 3, set(), object(), _DictSubclass()])
def test_resources_wrong_container_rejected(bad):
    with pytest.raises(guard.ResourceGuardError) as excinfo:
        guard.evaluate_resource_snapshot("U0", make_usage(), bad)
    assert excinfo.value.reason_code == "invalid_resources"


@pytest.mark.parametrize("stage", [None, [], {}, 0, True, "u0", "U2", "Q0"])
def test_evaluate_rejects_bad_stage(stage):
    with pytest.raises(guard.ResourceGuardError) as excinfo:
        guard.evaluate_resource_snapshot(stage, make_usage(), make_resources())
    assert excinfo.value.reason_code == "invalid_stage"


def test_inputs_and_outputs_are_not_aliased():
    stage = "U1"
    usage = make_usage(model_fit_attempts=3)
    resources = make_resources(filesystem_free_bytes=500 * GiB)
    usage_snapshot = copy.deepcopy(usage)
    resources_snapshot = copy.deepcopy(resources)
    result = guard.evaluate_resource_snapshot(stage, usage, resources)
    assert usage == usage_snapshot
    assert resources == resources_snapshot
    assert result["usage"] == usage_snapshot
    assert result["resources"] == resources_snapshot
    assert result["usage"] is not usage
    assert result["resources"] is not resources
    result["limits"]["model_fit_attempts"] = -1
    result["usage"]["model_fit_attempts"] = -1
    result["resources"]["filesystem_free_bytes"] = -1
    again = guard.evaluate_resource_snapshot(stage, usage, resources)
    assert again["limits"]["model_fit_attempts"] == LIMITS[stage]["model_fit_attempts"]
    assert again["usage"]["model_fit_attempts"] == 3
    assert again["resources"]["filesystem_free_bytes"] == 500 * GiB
    assert again["within_proposed_limits"] is True


def test_caller_inputs_are_not_mutated_by_the_call():
    usage = make_usage()
    resources = make_resources()
    first = guard.evaluate_resource_snapshot("U0", usage, resources)
    usage["model_fit_attempts"] = 7
    resources["filesystem_free_bytes"] = 31 * GiB
    assert first["usage"]["model_fit_attempts"] == 0
    assert first["resources"]["filesystem_free_bytes"] == 1024 * GiB
    second = guard.evaluate_resource_snapshot("U0", usage, resources)
    assert second["usage"]["model_fit_attempts"] == 7
    assert second["required_filesystem_free_bytes"] == 38 * GiB
    assert second["within_proposed_limits"] is False
    assert second["breaches"] == ["filesystem_reserve_insufficient"]


class EvilReason:
    def __hash__(self):
        raise AssertionError("PRIVATE_SENTINEL")

    def __eq__(self, other):
        raise AssertionError("PRIVATE_SENTINEL")

    def __str__(self):
        raise AssertionError("PRIVATE_SENTINEL")


class ForgedReason(str):
    pass


def test_resource_guard_error_never_probes_untrusted_reason():
    error = guard.ResourceGuardError(EvilReason())
    assert error.args == ("invalid_resource_input",)


def test_resource_guard_error_rejects_str_subclass_reason():
    error = guard.ResourceGuardError(ForgedReason("invalid_stage"))
    assert error.args == ("invalid_resource_input",)


def test_resource_guard_error_does_not_echo_unknown_reason():
    sentinel = "PRIVATE_SENTINEL_UNKNOWN_REASON"
    for reason in (sentinel, [sentinel], {"reason": sentinel}):
        error = guard.ResourceGuardError(reason)
        assert error.args == ("invalid_resource_input",)
        assert sentinel not in str(error)
        assert sentinel not in repr(error.args)


def test_reversed_key_insertion_order_yields_identical_snapshot():
    usage = make_usage()
    resources = make_resources()
    forward = guard.evaluate_resource_snapshot("U0", usage, resources)
    reversed_usage = dict(reversed(list(usage.items())))
    reversed_resources = dict(reversed(list(resources.items())))
    reversed_snapshot = guard.evaluate_resource_snapshot("U0", reversed_usage, reversed_resources)
    assert reversed_snapshot["snapshot_sha256"] == forward["snapshot_sha256"]


@pytest.mark.parametrize("stage", STAGES)
def test_zero_retries_and_no_authority_even_when_within(stage):
    limits = guard.proposed_limits(stage)
    assert limits["automatic_retries"] == 0
    within = evaluate(stage)
    assert within["execution_authorized"] is False
    assert within["limits"]["automatic_retries"] == 0
    assert within["within_proposed_limits"] is True
    over = evaluate(
        stage, usage=make_usage(model_fit_attempts=LIMITS[stage]["model_fit_attempts"] + 1)
    )
    assert over["execution_authorized"] is False
