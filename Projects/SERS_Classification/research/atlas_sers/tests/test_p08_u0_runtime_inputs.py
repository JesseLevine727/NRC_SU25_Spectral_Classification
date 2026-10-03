"""Synthetic tests for the P08-T105 U0 runtime-input adapter.

The five invented metadata byte strings and three invented NPZ action
archives come from the sibling ``test_p08_u0_arrays`` fixture; the model
specification audit and the eighteen inherited-contract source bytes are the
real public repository files named by that audit.  A small invented candidate
registry is pinned through ``subject._CANDIDATES_SHA256``.  No private data, no
filesystem access at prepare time, no fits, predictions or GPU work.
"""

from __future__ import annotations

import builtins
import csv
import hashlib
import io
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p08_u0_arrays as arrays_binder
from atlas_sers.evaluation import p08_u0_runtime_inputs as subject
from atlas_sers.governance.canonical import sha256_value
from tests import test_p08_u0_arrays as arrays_fixture
from tests import test_p08_u0_inputs as metadata_fixture

ROOT = Path(__file__).resolve().parents[1]

_AUDIT_BYTES = (ROOT / "results/p08_readiness/universal_slot_ledger_audit.json").read_bytes()
_AUDIT = json.loads(_AUDIT_BYTES)
_SOURCE_BYTES = {}
for _specification in _AUDIT["specification_inputs"].values():
    for _relative in _specification["inherited_sources"]:
        if _relative not in _SOURCE_BYTES:
            _SOURCE_BYTES[_relative] = (ROOT / _relative).read_bytes()
_SOURCE_PATHS = tuple(sorted(_SOURCE_BYTES))

MODE_ACTIONS = ("master_cv", "pseudo_domain")
CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
CLASSICAL_PARAMS = {
    "C-RBF-SVM": {"C": 0.01, "gamma": "scale", "class_weight": "balanced"},
    "C-RANDOM-FOREST": {
        "n_estimators": 1000,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": "balanced",
    },
    "C-EXTRA-TREES": {
        "n_estimators": 1000,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "bootstrap": False,
    },
}
CANDIDATE_FIELDS = (
    "candidate_id",
    "model_id",
    "declared_candidate_order",
    "parameters_json",
    "hyperparameter_sha256",
)
NEURAL_RECIPES = {
    "D0-M": (0.0, 0.0, False),
    "D1": (0.3, 0.0, True),
    "D2": (0.0, 0.3, False),
    "D3": (0.3, 0.3, True),
}
SENSOR_FAMILIES = ("polymer-A", "NA", "  glass-B  ", "", "Metal_C", "unknown-alloy")
HELD_SENSOR = "HELD-SENTINEL-FAMILY"


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _csv_bytes(fields, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


CANDIDATE_ROWS = [
    {
        "candidate_id": model_id + "-000",
        "model_id": model_id,
        "declared_candidate_order": str(order),
        "parameters_json": _canonical_json(CLASSICAL_PARAMS[model_id]),
        "hyperparameter_sha256": sha256_value(CLASSICAL_PARAMS[model_id]),
    }
    for order, model_id in enumerate(CLASSICAL_MODELS)
]
CANDIDATE_BYTES = _csv_bytes(CANDIDATE_FIELDS, CANDIDATE_ROWS)
CANDIDATE_SHA256 = hashlib.sha256(CANDIDATE_BYTES).hexdigest()


def _public_bytes():
    return _AUDIT_BYTES, dict(_SOURCE_BYTES)


def _install_candidate_pin(monkeypatch):
    monkeypatch.setattr(subject, "_CANDIDATES_SHA256", CANDIDATE_SHA256, raising=True)


def _install_parse_spies(monkeypatch):
    array_spies = arrays_fixture._install_parse_spies(monkeypatch)
    metadata_spies = metadata_fixture._install_parse_spies(monkeypatch)
    binder_spies = {"binder": 0}
    real_prepare = arrays_binder.prepare_u0_source_arrays

    def counting_prepare(*args, **kwargs):
        binder_spies["binder"] += 1
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(arrays_binder, "prepare_u0_source_arrays", counting_prepare, raising=True)
    return {"array": array_spies, "metadata": metadata_spies, "binder": binder_spies}


def _assert_parse_zero(spies, *, allow_json=False):
    assert spies["array"]["zip"] == 0
    assert spies["array"]["load"] == 0
    assert spies["array"]["read"] == 0
    assert spies["metadata"]["csv"] == 0
    assert spies["binder"]["binder"] == 0
    if not allow_json:
        assert spies["metadata"]["json"] == 0


def _default_model_mutator(common, plan):
    model_id = plan[0]
    spec = _AUDIT["model_specification_sha256"][model_id]
    common["model_spec_sha256"] = spec
    if model_id in CLASSICAL_PARAMS:
        common["candidate_id"] = model_id + "-000"
        common["hyperparameter_sha256"] = sha256_value(CLASSICAL_PARAMS[model_id])
    else:
        common["candidate_id"] = "fixed_recipe"
        common["hyperparameter_sha256"] = spec


def _augment_sensor(monkeypatch, ctx, names, held_family):
    reader = csv.DictReader(io.StringIO(ctx["manifest_bytes"].decode("utf-8")))
    fields = list(reader.fieldnames)
    rows = list(reader)
    is_test = {row["observation_uid"]: row["_is_test"] for row in ctx["case"]["rows"]}
    mapping = {}
    source_index = 0
    for row in rows:
        uid = row["observation_uid"]
        if is_test.get(uid):
            family = held_family
        else:
            family = names[source_index % len(names)]
            source_index += 1
        row["sensor_family"] = family
        mapping[uid] = family
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields + ["sensor_family"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    payload = out.getvalue().encode("utf-8")
    ctx["manifest_bytes"] = payload
    ctx["metadata_bytes"]["manifest_bytes"] = payload
    ctx["case"]["manifest_bytes"] = payload
    ctx["sensor_family"] = mapping
    metadata_fixture._seal(monkeypatch, ctx["case"])
    return ctx


def _build_context(
    monkeypatch,
    mode,
    *,
    pair_mutator_extra=None,
    sensor_names=SENSOR_FAMILIES,
    held_family=HELD_SENSOR,
    include_sensor_family=True,
    qc_hook=None,
):
    def extra(common, plan):
        _default_model_mutator(common, plan)
        if pair_mutator_extra is not None:
            pair_mutator_extra(common, plan)

    arrays_kwargs = {"pair_mutator_extra": extra}
    if qc_hook is not None:
        arrays_kwargs["qc_hook"] = qc_hook
    ctx = arrays_fixture._build_case(monkeypatch, mode, **arrays_kwargs)
    _install_candidate_pin(monkeypatch)
    if include_sensor_family:
        _augment_sensor(monkeypatch, ctx, sensor_names, held_family)
    return ctx


def _call(ctx, audit_bytes, source_bytes, candidate_bytes, **overrides):
    arguments = {
        "metadata_bytes": ctx["metadata_bytes"],
        "action_bytes": ctx["action_bytes"],
        "specification_audit_bytes": audit_bytes,
        "specification_source_bytes": source_bytes,
        "candidate_registry_bytes": candidate_bytes,
    }
    arguments.update(overrides)
    return subject.prepare_u0_runtime_inputs(**arguments)


def _prepare(monkeypatch, mode, **build_kwargs):
    ctx = _build_context(monkeypatch, mode, **build_kwargs)
    audit_bytes, source_bytes = _public_bytes()
    return ctx, _call(ctx, audit_bytes, source_bytes, CANDIDATE_BYTES)


def _expect_error(monkeypatch, mode="master_cv", build_kwargs=None, **overrides):
    ctx = _build_context(monkeypatch, mode, **(build_kwargs or {}))
    audit_bytes, source_bytes = _public_bytes()
    with pytest.raises(subject.InputError) as info:
        _call(ctx, audit_bytes, source_bytes, CANDIDATE_BYTES, **overrides)
    error = info.value
    assert isinstance(error.reason_code, str) and error.reason_code
    assert error.reason_code != "unlisted_reason_code"
    assert metadata_fixture.SENTINEL not in str(error)
    return error, ctx


def _classical_pairs(inputs):
    result = []
    for pair in inputs.pairs:
        job = json.loads(pair.prepared_pair.fit_job_json)
        if job["model_id"] in CLASSICAL_MODELS:
            result.append((pair, job))
    return result


def _neural_pairs(inputs):
    result = []
    for pair in inputs.pairs:
        job = json.loads(pair.prepared_pair.fit_job_json)
        if job["model_id"] not in CLASSICAL_MODELS:
            result.append((pair, job))
    return result


@pytest.mark.parametrize("mode", MODE_ACTIONS)
def test_positive_counts_flags_and_hashes(monkeypatch, mode):
    _, inputs = _prepare(monkeypatch, mode)
    assert len(inputs.pairs) == 78
    report = inputs.public_report()
    assert report["pairs"] == 78
    assert report["classical_pairs"] == 42
    assert report["neural_pairs"] == 36
    assert report["models"] == 7
    assert report["candidates"] == 3
    assert report["source_files"] == 18
    assert report["specification_bytes_verified"] is True
    assert report["kernel_arguments_prepared"] is True
    assert report["substrate_metadata_preserved"] is True
    assert report["loaded_runtime_code_verified"] is False
    assert report["live_controller_verified"] is False
    assert report["execution_authorized"] is False
    assert report["new_scientific_operations"] == 0
    assert report["specification_audit_sha256"] == hashlib.sha256(_AUDIT_BYTES).hexdigest()
    assert report["candidate_registry_sha256"] == CANDIDATE_SHA256
    without = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == sha256_value(without)
    assert len(_classical_pairs(inputs)) == 42
    assert len(_neural_pairs(inputs)) == 36


def test_report_links_prepared_array_report(monkeypatch):
    ctx, inputs = _prepare(monkeypatch, "master_cv")
    prepared = arrays_binder.prepare_u0_source_arrays(
        metadata_bytes=ctx["metadata_bytes"], action_bytes=ctx["action_bytes"]
    )
    assert (
        inputs.public_report()["prepared_report_sha256"]
        == prepared.public_report()["report_sha256"]
    )


def _assert_observation(observation, source, ctx, by_uid):
    uid = source.observation_uid
    row = by_uid[uid]
    assert observation.uid == uid
    assert observation.master == row["master_sample_id"]
    assert observation.station == row["station"]
    assert observation.target == row["target_analyte"]
    assert observation.instrument == row["instrument"]
    assert observation.substrate == ctx["sensor_family"][uid]


@pytest.mark.parametrize("mode", MODE_ACTIONS)
def test_source_observations_preserve_recorded_metadata(monkeypatch, mode):
    ctx, inputs = _prepare(monkeypatch, mode)
    by_uid = {row["observation_uid"]: row for row in ctx["case"]["rows"]}
    role_map = _is_test_map(ctx)
    source_uids = {uid for uid, is_test in role_map.items() if not is_test}
    held_uids = {uid for uid, is_test in role_map.items() if is_test}
    returned = set()
    for pair in inputs.pairs:
        roles = pair.prepared_pair.inputs.source_roles
        for source, observation in zip(roles.fitting, pair.fitting_observations, strict=True):
            _assert_observation(observation, source, ctx, by_uid)
            returned.add(source.observation_uid)
        for source, observation in zip(roles.validation, pair.validation_observations, strict=True):
            _assert_observation(observation, source, ctx, by_uid)
            returned.add(source.observation_uid)
    assert returned <= source_uids
    assert not (returned & held_uids)


def _is_test_map(ctx):
    return {row["observation_uid"]: row["_is_test"] for row in ctx["case"]["rows"]}


def test_sensor_strings_are_preserved_not_normalized(monkeypatch):
    ctx, inputs = _prepare(monkeypatch, "master_cv")
    seen = set()
    for pair in inputs.pairs:
        for observation in pair.fitting_observations + pair.validation_observations:
            seen.add(observation.substrate)
    for family in ctx["sensor_family"].values():
        if family != HELD_SENSOR:
            assert family in seen
    assert "NA" in seen
    assert "  glass-B  " in seen
    assert "" in seen
    assert HELD_SENSOR not in seen


def test_classical_kwargs_exact(monkeypatch):
    from atlas_sers.evaluation.p03_runtime import P03Dataset

    _, inputs = _prepare(monkeypatch, "master_cv")
    expected_keys = {
        "dataset",
        "fit_id",
        "model_id",
        "candidate_id",
        "parameters",
        "seed",
        "fit_uids",
        "validation_uids",
        "class_vocabulary",
        "expected_fit_uid_sha256",
        "expected_validation_uid_sha256",
    }
    seen_models = set()
    for pair, job in _classical_pairs(inputs):
        prepared = pair.prepared_pair
        roles = prepared.inputs.source_roles
        kwargs = pair.classical_kwargs()
        assert set(kwargs) == expected_keys
        assert kwargs["fit_id"] == job["job_id"]
        assert kwargs["model_id"] == job["model_id"]
        assert kwargs["candidate_id"] == job["candidate_id"]
        assert kwargs["parameters"] == CLASSICAL_PARAMS[job["model_id"]]
        assert kwargs["seed"] == job["seed"]
        fit_uids = [observation.observation_uid for observation in roles.fitting]
        val_uids = [observation.observation_uid for observation in roles.validation]
        assert fit_uids and val_uids
        assert list(kwargs["fit_uids"]) == fit_uids
        assert list(kwargs["validation_uids"]) == val_uids
        assert tuple(kwargs["class_vocabulary"]) == tuple(roles.classes)
        assert kwargs["expected_fit_uid_sha256"] == job["fit_uid_sha256"]
        assert kwargs["expected_validation_uid_sha256"] == job["validation_uid_sha256"]

        dataset = kwargs["dataset"]
        assert isinstance(dataset, P03Dataset)
        assert list(dataset.metadata.columns) == [
            "observation_uid",
            "master_sample_id",
            "target_analyte",
            "instrument",
            "station",
        ]
        ordered_uids = fit_uids + val_uids
        assert dataset.metadata["observation_uid"].tolist() == ordered_uids
        assert dataset.metadata["master_sample_id"].tolist() == [
            observation.master_sample_id for observation in roles.fitting
        ] + [observation.master_sample_id for observation in roles.validation]
        assert dataset.metadata["target_analyte"].tolist() == [
            observation.target_analyte for observation in roles.fitting
        ] + [observation.target_analyte for observation in roles.validation]
        assert dataset.metadata["instrument"].tolist() == [
            observation.instrument for observation in roles.fitting
        ] + [observation.instrument for observation in roles.validation]
        assert dataset.metadata["station"].tolist() == [
            observation.station for observation in roles.fitting
        ] + [observation.station for observation in roles.validation]
        assert dataset.uid_to_index == {uid: index for index, uid in enumerate(ordered_uids)}
        expected = np.concatenate(
            [prepared.inputs.fitting_values(), prepared.inputs.validation_values()]
        ).astype(np.float64)
        assert dataset.intensity.dtype == np.float64
        assert dataset.intensity.shape == expected.shape
        np.testing.assert_array_equal(dataset.intensity, expected)
        seen_models.add(job["model_id"])
    assert seen_models == set(CLASSICAL_MODELS)


def test_classical_kwargs_return_fresh_objects(monkeypatch):
    _, inputs = _prepare(monkeypatch, "master_cv")
    pair, job = next(
        item for item in _classical_pairs(inputs) if item[1]["model_id"] == "C-RBF-SVM"
    )
    first = pair.classical_kwargs()
    baseline_intensity = first["dataset"].intensity.copy()
    baseline_metadata = first["dataset"].metadata.copy(deep=True)
    baseline_index = dict(first["dataset"].uid_to_index)

    first["parameters"]["C"] = -123.0
    first["dataset"].intensity[:] = -5.0
    first["dataset"].metadata.loc[:, "station"] = "TAMPERED"
    first["dataset"].uid_to_index.clear()

    second = pair.classical_kwargs()
    assert first["dataset"] is not second["dataset"]
    assert first["parameters"] is not second["parameters"]
    assert second["parameters"] == CLASSICAL_PARAMS[job["model_id"]]
    np.testing.assert_array_equal(second["dataset"].intensity, baseline_intensity)
    assert second["dataset"].uid_to_index == baseline_index
    pd.testing.assert_frame_equal(second["dataset"].metadata, baseline_metadata)


def test_neural_kwargs_exact_and_excluded_controller_keys(monkeypatch):
    from atlas_sers.evaluation import p05_sampling

    ctx, inputs = _prepare(monkeypatch, "master_cv")
    expected_keys = {
        "values",
        "observations",
        "noise_metadata",
        "validation_values",
        "validation_observations",
        "role_id",
        "recipe",
        "seed",
        "maximum_fit_seconds",
        "maximum_cuda_allocated_bytes",
    }
    for pair, job in _neural_pairs(inputs):
        prepared = pair.prepared_pair
        roles = prepared.inputs.source_roles
        kwargs = pair.neural_kwargs()
        assert set(kwargs) == expected_keys
        for forbidden in ("device", "global_deadline", "on_epoch"):
            assert forbidden not in kwargs
        assert kwargs["role_id"] == roles.fitting_role_id
        assert kwargs["recipe"] == job["model_id"]
        assert kwargs["seed"] == job["seed"]
        assert kwargs["maximum_fit_seconds"] == 120.0
        assert kwargs["maximum_cuda_allocated_bytes"] == 4294967296
        assert isinstance(kwargs["values"], np.ndarray)
        assert kwargs["values"].dtype == np.float32
        assert kwargs["values"].flags.writeable is False
        np.testing.assert_array_equal(kwargs["values"], prepared.inputs.fitting_values())
        np.testing.assert_array_equal(
            kwargs["validation_values"], prepared.inputs.validation_values()
        )
        assert len(kwargs["observations"]) == len(roles.fitting)
        for source, observation in zip(roles.fitting, kwargs["observations"], strict=True):
            assert isinstance(observation, p05_sampling.Observation)
            assert observation.uid == source.observation_uid
            assert observation.master == source.master_sample_id
            assert observation.station == source.station
            assert observation.target == source.target_analyte
            assert observation.instrument == source.instrument
            assert observation.substrate == ctx["sensor_family"][source.observation_uid]
        pd.testing.assert_frame_equal(
            kwargs["noise_metadata"], prepared.inputs.fitting_noise_frame()
        )
        assert pair.neural_kwargs()["noise_metadata"] is not kwargs["noise_metadata"]


def test_neural_configuration_per_recipe(monkeypatch):
    _, inputs = _prepare(monkeypatch, "master_cv")
    seen = set()
    for pair, job in _neural_pairs(inputs):
        config = pair.configuration()
        assert set(config) == {
            "model_id",
            "model_spec_sha256",
            "recipe",
            "model",
            "sampler",
            "objective",
            "optimization",
            "stopping",
            "maximum_fit_seconds",
            "maximum_cuda_allocated_bytes",
        }
        assert config["model_id"] == job["model_id"]
        assert config["model_spec_sha256"] == job["model_spec_sha256"]
        recipe = config["recipe"]
        lambda_supcon, lambda_pair, projection = NEURAL_RECIPES[job["model_id"]]
        assert recipe["recipe_id"] == job["model_id"]
        assert recipe["lambda_supcon"] == lambda_supcon
        assert recipe["lambda_pair"] == lambda_pair
        assert recipe["projection"] is projection
        assert config["optimization"]["optimizer"] == "AdamW"
        assert config["optimization"]["learning_rate"] == 0.0003
        assert config["optimization"]["threads"] == 1
        assert config["stopping"] == {
            "minimum_epochs": 30,
            "maximum_epochs": 200,
            "patience": 20,
        }
        assert config["maximum_fit_seconds"] == 120
        assert config["maximum_cuda_allocated_bytes"] == 4294967296
        seen.add(job["model_id"])
    assert seen == set(NEURAL_RECIPES)


def test_classical_configuration(monkeypatch):
    _, inputs = _prepare(monkeypatch, "master_cv")
    for pair, job in _classical_pairs(inputs):
        config = pair.configuration()
        assert set(config) == {
            "model_id",
            "model_spec_sha256",
            "candidate_id",
            "hyperparameter_sha256",
            "parameters",
            "threads",
        }
        assert config["model_id"] == job["model_id"]
        assert config["model_spec_sha256"] == job["model_spec_sha256"]
        assert config["candidate_id"] == job["candidate_id"]
        assert config["hyperparameter_sha256"] == job["hyperparameter_sha256"]
        assert config["parameters"] == CLASSICAL_PARAMS[job["model_id"]]
        assert config["threads"] == 1
        assert pair.configuration() is not config


def test_wrong_kernel_dispatch_refuses(monkeypatch):
    _, inputs = _prepare(monkeypatch, "master_cv")
    classical_pair = _classical_pairs(inputs)[0][0]
    neural_pair = _neural_pairs(inputs)[0][0]
    with pytest.raises(subject.InputError) as info:
        classical_pair.neural_kwargs()
    assert info.value.reason_code == "wrong_kernel"
    with pytest.raises(subject.InputError) as info:
        neural_pair.classical_kwargs()
    assert info.value.reason_code == "wrong_kernel"


def test_frozen_objects_and_private_repr(monkeypatch):
    ctx, inputs = _prepare(monkeypatch, "master_cv")
    pair = inputs.pairs[0]
    for action in (
        lambda: setattr(inputs, "pairs", ()),
        lambda: setattr(pair, "parameters_json", "x"),
        lambda: setattr(pair, "configuration_json", "x"),
        lambda: setattr(pair, "fitting_observations", ()),
        lambda: setattr(pair, "prepared_pair", None),
    ):
        with pytest.raises(FrozenInstanceError):
            action()
    rendered = repr(inputs) + "".join(repr(item) for item in inputs.pairs)
    for uid in ctx["uids"]:
        assert uid not in rendered
    for row in ctx["case"]["rows"]:
        assert row["master_sample_id"] not in rendered
    assert metadata_fixture.SENTINEL not in rendered


def test_public_report_is_fresh_and_excludes_private_values(monkeypatch):
    ctx, inputs = _prepare(monkeypatch, "master_cv")
    first = inputs.public_report()
    second = inputs.public_report()
    assert first is not second
    assert first == second
    rendered = inputs.report_json + json.dumps(first)
    for uid in ctx["uids"]:
        assert uid not in rendered
    for row in ctx["case"]["rows"]:
        assert row["master_sample_id"] not in rendered
    for family in set(ctx["sensor_family"].values()):
        if family:
            assert family not in rendered
    assert metadata_fixture.SENTINEL not in rendered
    assert "plan/contracts" not in rendered
    assert "src/atlas_sers" not in rendered


def test_require_scientific_execution_always_denies():
    calls = (
        lambda: subject.require_scientific_execution(),
        lambda: subject.require_scientific_execution(True),
        lambda: subject.require_scientific_execution(execution_authorized=True),
        lambda: subject.require_scientific_execution(None, authorized=True, token="forged"),
    )
    for call in calls:
        with pytest.raises(subject.InputError) as info:
            call()
        assert info.value.reason_code == "scientific_execution_not_authorized"


def test_each_source_byte_tamper_rejected_before_array_binder(monkeypatch):
    ctx = _build_context(monkeypatch, "master_cv")
    audit_bytes, source_template = _public_bytes()
    spies = _install_parse_spies(monkeypatch)
    try:
        for path in _SOURCE_PATHS:
            source_bytes = dict(source_template)
            data = source_bytes[path]
            source_bytes[path] = bytes([data[0] ^ 0xFF]) + data[1:]
            with pytest.raises(subject.InputError) as info:
                _call(ctx, audit_bytes, source_bytes, CANDIDATE_BYTES)
            assert info.value.reason_code == "specification_source_mismatch"
    finally:
        _assert_parse_zero(spies, allow_json=True)


def test_audit_byte_tamper_rejected_before_any_parse(monkeypatch):
    ctx = _build_context(monkeypatch, "master_cv")
    audit_bytes, source_bytes = _public_bytes()
    tampered = bytearray(audit_bytes)
    tampered[0] ^= 0xFF
    spies = _install_parse_spies(monkeypatch)
    try:
        with pytest.raises(subject.InputError) as info:
            _call(ctx, bytes(tampered), source_bytes, CANDIDATE_BYTES)
        assert info.value.reason_code == "audit_hash_mismatch"
    finally:
        _assert_parse_zero(spies)


def test_candidate_byte_tamper_rejected_before_any_parse(monkeypatch):
    ctx = _build_context(monkeypatch, "master_cv")
    audit_bytes, source_bytes = _public_bytes()
    tampered = bytearray(CANDIDATE_BYTES)
    tampered[0] ^= 0xFF
    spies = _install_parse_spies(monkeypatch)
    try:
        with pytest.raises(subject.InputError) as info:
            _call(ctx, audit_bytes, source_bytes, bytes(tampered))
        assert info.value.reason_code == "candidate_registry_hash_mismatch"
    finally:
        _assert_parse_zero(spies)


def test_source_key_map_must_match_exactly(monkeypatch):
    ctx = _build_context(monkeypatch, "master_cv")
    source = dict(_SOURCE_BYTES)
    missing = dict(source)
    missing.pop(_SOURCE_PATHS[0])
    with pytest.raises(subject.InputError):
        _call(ctx, _AUDIT_BYTES, missing, CANDIDATE_BYTES)
    extra = dict(source)
    extra["src/atlas_sers/not_a_real_module.py"] = b"x"
    with pytest.raises(subject.InputError):
        _call(ctx, _AUDIT_BYTES, extra, CANDIDATE_BYTES)
    mutable = dict(source)
    mutable[_SOURCE_PATHS[0]] = bytearray(source[_SOURCE_PATHS[0]])
    with pytest.raises(subject.InputError):
        _call(ctx, _AUDIT_BYTES, mutable, CANDIDATE_BYTES)


def test_candidate_registry_requires_exact_columns(monkeypatch):
    ctx = _build_context(monkeypatch, "master_cv")
    fields = CANDIDATE_FIELDS[:-1]
    payload = _csv_bytes(fields, [{key: row[key] for key in fields} for row in CANDIDATE_ROWS])
    monkeypatch.setattr(
        subject, "_CANDIDATES_SHA256", hashlib.sha256(payload).hexdigest(), raising=True
    )
    with pytest.raises(subject.InputError):
        _call(ctx, _AUDIT_BYTES, dict(_SOURCE_BYTES), payload)


def test_neural_model_spec_mismatch_rejected(monkeypatch):
    target = ("D0-M", "PP-U-ARPLS", "master_cv:2", 2)

    def mutate(common, plan):
        if plan == target:
            bogus = sha256_value({"bogus-model-spec": True})
            common["model_spec_sha256"] = bogus
            common["hyperparameter_sha256"] = bogus

    ctx = _build_context(monkeypatch, "master_cv", pair_mutator_extra=mutate)
    arrays_binder.prepare_u0_source_arrays(
        metadata_bytes=ctx["metadata_bytes"], action_bytes=ctx["action_bytes"]
    )
    audit_bytes, source_bytes = _public_bytes()
    with pytest.raises(subject.InputError) as info:
        _call(ctx, audit_bytes, source_bytes, CANDIDATE_BYTES)
    assert info.value.reason_code == "model_spec_mismatch"


def test_classical_candidate_mismatch_rejected(monkeypatch):
    target = ("C-RANDOM-FOREST", "PP-U-SG", "master_cv:0", 0)

    def unknown_id(common, plan):
        if plan == target:
            common["candidate_id"] = "C-RANDOM-FOREST-999"

    error, _ = _expect_error(monkeypatch, build_kwargs={"pair_mutator_extra": unknown_id})
    assert error.reason_code == "candidate_mismatch"

    def changed_hash(common, plan):
        if plan == target:
            common["hyperparameter_sha256"] = sha256_value({"bogus": 1})

    error, _ = _expect_error(monkeypatch, build_kwargs={"pair_mutator_extra": changed_hash})
    assert error.reason_code == "candidate_mismatch"


def test_neural_wrong_candidate_rejected_upstream(monkeypatch):
    target = ("D1", "PP-U-SG", "master_cv:0", 0)

    def wrong_candidate(common, plan):
        if plan == target:
            common["candidate_id"] = "not-a-fixed-recipe"

    ctx = _build_context(monkeypatch, "master_cv", pair_mutator_extra=wrong_candidate)
    with pytest.raises(arrays_binder.ArrayInputError):
        arrays_binder.prepare_u0_source_arrays(
            metadata_bytes=ctx["metadata_bytes"], action_bytes=ctx["action_bytes"]
        )
    audit_bytes, source_bytes = _public_bytes()
    with pytest.raises(subject.InputError) as info:
        _call(ctx, audit_bytes, source_bytes, CANDIDATE_BYTES)
    assert info.value.reason_code == "input_preparation_failed"


def test_neural_wrong_hyperparameter_rejected_upstream(monkeypatch):
    target = ("D1", "PP-U-SG", "master_cv:0", 0)

    def wrong_hyper(common, plan):
        if plan == target:
            common["hyperparameter_sha256"] = sha256_value({"bogus": 2})

    ctx = _build_context(monkeypatch, "master_cv", pair_mutator_extra=wrong_hyper)
    with pytest.raises(arrays_binder.ArrayInputError):
        arrays_binder.prepare_u0_source_arrays(
            metadata_bytes=ctx["metadata_bytes"], action_bytes=ctx["action_bytes"]
        )
    audit_bytes, source_bytes = _public_bytes()
    with pytest.raises(subject.InputError) as info:
        _call(ctx, audit_bytes, source_bytes, CANDIDATE_BYTES)
    assert info.value.reason_code == "input_preparation_failed"


def test_missing_sensor_family_column_rejected(monkeypatch):
    ctx = _build_context(monkeypatch, "master_cv", include_sensor_family=False)
    with pytest.raises(subject.InputError) as info:
        _call(ctx, _AUDIT_BYTES, dict(_SOURCE_BYTES), CANDIDATE_BYTES)
    assert info.value.reason_code == "missing_sensor_family"


def test_blank_sensor_family_is_legitimate(monkeypatch):
    ctx = _build_context(monkeypatch, "master_cv", sensor_names=("", "family-a"))
    inputs = _call(ctx, _AUDIT_BYTES, dict(_SOURCE_BYTES), CANDIDATE_BYTES)
    seen = set()
    for pair in inputs.pairs:
        for observation in pair.fitting_observations + pair.validation_observations:
            seen.add(observation.substrate)
    assert "" in seen


def test_no_filesystem_or_scientific_operations(monkeypatch):
    import torch

    from atlas_sers.evaluation import p03_runtime, p04_runtime, p05_development, p05_sampling
    from atlas_sers.models import classical as models_classical

    ctx = _build_context(monkeypatch, "master_cv")
    audit_bytes, source_bytes = _public_bytes()
    labels = (
        "open",
        "fit",
        "train",
        "noise",
        "augment",
        "quantile",
        "classical_estimator",
        "sample",
        "sample_alias",
        "prepare_noise",
        "cuda_init",
        "cuda_lazy_init",
    )
    counters = {label: 0 for label in labels}

    def deny(label):
        def inner(*args, **kwargs):
            counters[label] += 1
            raise AssertionError(label)

        return inner

    monkeypatch.setattr(builtins, "open", deny("open"), raising=True)
    monkeypatch.setattr(Path, "open", deny("open"), raising=True)
    monkeypatch.setattr(p03_runtime, "run_candidate_fit", deny("fit"), raising=True)
    monkeypatch.setattr(
        p03_runtime, "build_classical_estimator", deny("classical_estimator"), raising=True
    )
    monkeypatch.setattr(
        models_classical,
        "build_classical_estimator",
        deny("classical_estimator"),
        raising=True,
    )
    monkeypatch.setattr(p05_development, "train_development_fit", deny("train"), raising=True)
    monkeypatch.setattr(p04_runtime, "_noise_quantiles", deny("noise"), raising=True)
    monkeypatch.setattr(p04_runtime, "_augment", deny("augment"), raising=True)
    monkeypatch.setattr(p05_development, "_prepare_noise", deny("prepare_noise"), raising=True)
    monkeypatch.setattr(p05_sampling, "sample_master_views", deny("sample"), raising=True)
    monkeypatch.setattr(p05_development, "sample_master_views", deny("sample_alias"), raising=True)
    monkeypatch.setattr(np, "quantile", deny("quantile"), raising=True)
    monkeypatch.setattr(torch.cuda, "init", deny("cuda_init"), raising=True)
    monkeypatch.setattr(torch.cuda, "_lazy_init", deny("cuda_lazy_init"), raising=True)
    try:
        inputs = _call(ctx, audit_bytes, source_bytes, CANDIDATE_BYTES)
        assert len(inputs.pairs) == 78
        for pair in inputs.pairs:
            pair.configuration()
            job = json.loads(pair.prepared_pair.fit_job_json)
            if job["model_id"] in CLASSICAL_MODELS:
                pair.classical_kwargs()
            else:
                pair.neural_kwargs()
    finally:
        assert counters == {label: 0 for label in labels}, counters
