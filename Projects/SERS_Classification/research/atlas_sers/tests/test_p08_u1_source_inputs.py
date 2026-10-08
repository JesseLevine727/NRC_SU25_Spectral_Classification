"""Synthetic tests for the P08-U1 lazy source input factory."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from atlas_sers.evaluation import p08_plan as _plan
from atlas_sers.evaluation import p08_source_predictions as _sp
from atlas_sers.evaluation import p08_u0_arrays as _arrays
from atlas_sers.evaluation import p08_u0_inputs as _u0_inputs
from atlas_sers.evaluation import p08_u0_runtime_inputs as _u0_runtime
from atlas_sers.evaluation import p08_u1_source_inputs as u1
from atlas_sers.governance.canonical import sha256_value
from tests import test_p08_u0_runtime_inputs as fixture

_METADATA_KEYS = ("manifest_bytes", "contexts_bytes", "roles_bytes")
_JSON = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": True}


def _body(ctx):
    return {
        "schema_version": _plan.SCHEMA_VERSION,
        "execution_authorized": False,
        "jobs": copy.deepcopy(ctx["case"]["full_jobs"]),
        "aliases": [],
        "summary": {},
    }


def _install_plan(monkeypatch, ctx):
    body = _body(ctx)
    digest = _plan._hash(body)
    plan = dict(body)
    plan["plan_sha256"] = digest
    monkeypatch.setattr(u1, "_PLAN_SHA256", digest)
    return plan


def _metadata(ctx):
    return {key: ctx["metadata_bytes"][key] for key in _METADATA_KEYS}


def _call_factory(ctx, plan, metadata=None, audit=None, source=None, registry=None):
    if audit is None or source is None:
        audit, source = fixture._public_bytes()
    return u1.prepare_source_factory(
        plan=plan,
        metadata_bytes=_metadata(ctx) if metadata is None else metadata,
        action_bytes=ctx["action_bytes"],
        specification_audit_bytes=audit,
        specification_source_bytes=source,
        candidate_registry_bytes=fixture.CANDIDATE_BYTES if registry is None else registry,
    )


def _prepare(monkeypatch, ctx):
    plan = _install_plan(monkeypatch, ctx)
    return plan, _call_factory(ctx, plan)


@pytest.mark.parametrize("mode", ["master_cv", "pseudo_domain"])
def test_prepare_both_modes(monkeypatch, mode):
    ctx = fixture._build_context(monkeypatch, mode)
    _plan_dict, factory = _prepare(monkeypatch, ctx)

    ids = factory.fit_job_ids()
    assert isinstance(ids, tuple)
    assert ids
    assert ids == tuple(sorted(ids))

    report = factory.public_report()
    assert report["schema_version"] == u1.SCHEMA_VERSION
    assert report["execution_authorized"] is False
    assert report["new_scientific_operations"] == 0
    assert report["loaded_runtime_code_verified"] is False
    assert report["source_arrays_verified"] is True
    assert report["policies"] == len(_arrays._PREPARED_POLICIES)

    pair = factory.pair(ids[0])
    assert isinstance(pair, _u0_runtime.RuntimePair)
    configuration = pair.configuration()
    if configuration["model_id"] in _plan.CLASSICAL_MODELS:
        assert "dataset" in pair.classical_kwargs()
    else:
        assert "values" in pair.neural_kwargs()


def _assert_pairs_equivalent(left, right):
    assert left.configuration() == right.configuration()
    assert left.parameters_json == right.parameters_json
    left_fitting = [o.uid for o in left.fitting_observations]
    right_fitting = [o.uid for o in right.fitting_observations]
    assert left_fitting == right_fitting
    assert [o.uid for o in left.validation_observations] == [
        o.uid for o in right.validation_observations
    ]
    np.testing.assert_array_equal(
        left.prepared_pair.inputs.fitting_values(),
        right.prepared_pair.inputs.fitting_values(),
    )
    np.testing.assert_array_equal(
        left.prepared_pair.inputs.validation_values(),
        right.prepared_pair.inputs.validation_values(),
    )
    left_frame = left.prepared_pair.inputs.fitting_noise_frame()
    right_frame = right.prepared_pair.inputs.fitting_noise_frame()
    assert list(left_frame["observation_uid"]) == list(right_frame["observation_uid"])
    np.testing.assert_allclose(
        left_frame["first_difference_noise_mad"], right_frame["first_difference_noise_mad"]
    )
    np.testing.assert_allclose(left_frame["intensity_range"], right_frame["intensity_range"])


@pytest.mark.parametrize("mode", ["master_cv", "pseudo_domain"])
def test_parity_with_existing_u0_adapter(monkeypatch, mode):
    ctx = fixture._build_context(monkeypatch, mode)
    _plan_dict, factory = _prepare(monkeypatch, ctx)
    audit_bytes, source_bytes = fixture._public_bytes()
    legacy = fixture._call(ctx, audit_bytes, source_bytes, fixture.CANDIDATE_BYTES)
    legacy_by_id = {}
    for pair in legacy.pairs:
        legacy_by_id[json.loads(pair.prepared_pair.fit_job_json)["job_id"]] = pair
    compared = 0
    for fit_id in factory.fit_job_ids():
        other = legacy_by_id.get(fit_id)
        if other is None:
            continue
        _assert_pairs_equivalent(factory.pair(fit_id), other)
        compared += 1
    assert compared > 0


def test_kernel_uses_exact_later_candidate():
    model_id = sorted(_plan.CLASSICAL_MODELS)[0]
    parameters = {"alpha": 2}
    digest = sha256_value(parameters)
    entry = {
        "model_id": model_id,
        "candidate_id": "CAND-later",
        "hyperparameter_sha256": digest,
        "parameters_json": json.dumps(parameters, **_JSON),
        "declared_candidate_order": 1,
    }
    fit = {
        "job_id": "FIT",
        "stage": "source_fit",
        "model_id": model_id,
        "candidate_id": "CAND-later",
        "hyperparameter_sha256": digest,
        "model_spec_sha256": "spec",
        "representation_id": _arrays._PREPARED_REPRESENTATIONS[0],
        "policy_id": _arrays._PREPARED_POLICIES[0],
    }
    prediction = dict(
        fit, job_id="PRED", stage="source_validation_prediction", dependencies=["FIT"]
    )
    kind, configuration, parameters_json = u1._resolve_kernel(
        fit, prediction, {model_id: "spec"}, {(model_id, "CAND-later"): entry}, {}
    )
    assert kind == "classical"
    assert configuration["candidate_id"] == "CAND-later"
    assert json.loads(parameters_json) == parameters


def test_candidate_index_keeps_later_declared_order():
    model_id = sorted(_plan.CLASSICAL_MODELS)[0]
    first = {"alpha": 1}
    later = {"alpha": 2}
    records = [
        {
            "candidate_id": "CAND-first",
            "model_id": model_id,
            "declared_candidate_order": "0",
            "parameters_json": json.dumps(first, **_JSON),
            "hyperparameter_sha256": sha256_value(first),
        },
        {
            "candidate_id": "CAND-later",
            "model_id": model_id,
            "declared_candidate_order": "1",
            "parameters_json": json.dumps(later, **_JSON),
            "hyperparameter_sha256": sha256_value(later),
        },
    ]
    index = u1._build_candidate_index(records)
    assert index[(model_id, "CAND-later")]["declared_candidate_order"] == 1
    assert index[(model_id, "CAND-later")]["hyperparameter_sha256"] == sha256_value(later)


def _source_pairs(jobs):
    fits = {}
    predictions = {}
    for job in jobs:
        stage = job.get("stage")
        if stage == _sp._FIT_STAGE:
            fits[job.get("job_id")] = job
        elif stage == _sp._PREDICTION_STAGE:
            dependencies = job.get("dependencies")
            if type(dependencies) is list and len(dependencies) == 1:
                predictions[dependencies[0]] = job
    return fits, predictions


def _first_classical_source_pair(jobs):
    fits, predictions = _source_pairs(jobs)
    for job_id, fit in fits.items():
        if fit.get("policy_id") not in _sp._SUPPORTED_POLICIES:
            continue
        if fit.get("model_id") not in _plan.CLASSICAL_MODELS:
            continue
        if fit.get("candidate_id") == "fixed_recipe":
            continue
        prediction = predictions.get(job_id)
        if prediction is not None:
            return fit, prediction
    return None, None


def _reseal_plan(monkeypatch, jobs):
    body = {
        "schema_version": _plan.SCHEMA_VERSION,
        "execution_authorized": False,
        "jobs": jobs,
        "aliases": [],
        "summary": {},
    }
    plan = dict(body)
    plan["plan_sha256"] = _plan._hash(body)
    monkeypatch.setattr(u1, "_PLAN_SHA256", plan["plan_sha256"])
    return plan


def _rebuild_job(source, dependencies, **changes):
    fields = {
        key: value for key, value in source.items() if key not in ("job_id", "dependencies")
    }
    fields.update(changes)
    return _plan._new_job(fields, list(dependencies))


def _prepare_custom(monkeypatch, ctx, plan):
    audit, source = fixture._public_bytes()
    return u1.prepare_source_factory(
        plan=plan,
        metadata_bytes=_metadata(ctx),
        action_bytes=ctx["action_bytes"],
        specification_audit_bytes=audit,
        specification_source_bytes=source,
        candidate_registry_bytes=fixture.CANDIDATE_BYTES,
    )


def _replace_jobs(jobs, replacements):
    return [replacements.get(job.get("job_id"), job) for job in jobs]


def test_later_candidate_end_to_end_yields_distinct_parameters(monkeypatch):
    ctx = fixture._build_context(monkeypatch, "master_cv")
    jobs = copy.deepcopy(ctx["case"]["full_jobs"])
    fit, prediction = _first_classical_source_pair(jobs)
    assert fit is not None

    parameters = {"later_parameter": 424242}
    digest = sha256_value(parameters)
    new_fit = _rebuild_job(
        fit, [], candidate_id="CAND-later-e2e", hyperparameter_sha256=digest
    )
    new_prediction = _rebuild_job(
        prediction,
        [new_fit["job_id"]],
        candidate_id="CAND-later-e2e",
        hyperparameter_sha256=digest,
    )
    jobs.append(new_fit)
    jobs.append(new_prediction)
    plan = _reseal_plan(monkeypatch, jobs)

    record = {
        "candidate_id": "CAND-later-e2e",
        "model_id": fit["model_id"],
        "declared_candidate_order": "1",
        "parameters_json": json.dumps(parameters, **_JSON),
        "hyperparameter_sha256": digest,
    }
    original_reader = _u0_runtime._read_candidate_registry

    def augmented_reader(registry_bytes):
        return list(original_reader(registry_bytes)) + [record]

    monkeypatch.setattr(_u0_runtime, "_read_candidate_registry", augmented_reader)

    factory = _prepare_custom(monkeypatch, ctx, plan)
    assert new_fit["job_id"] in factory.fit_job_ids()
    pair = factory.pair(new_fit["job_id"])
    assert json.loads(pair.parameters_json) == parameters
    assert pair.configuration()["candidate_id"] == "CAND-later-e2e"


def test_tampered_source_array_rejected(monkeypatch):
    ctx = fixture._build_context(monkeypatch, "master_cv")
    jobs = copy.deepcopy(ctx["case"]["full_jobs"])
    fit, prediction = _first_classical_source_pair(jobs)
    assert fit is not None
    replacement = "0" * 64
    new_fit = _rebuild_job(fit, [], array_sha256=replacement)
    new_prediction = _rebuild_job(
        prediction, [new_fit["job_id"]], array_sha256=replacement
    )
    tampered = _replace_jobs(
        jobs, {fit["job_id"]: new_fit, prediction["job_id"]: new_prediction}
    )
    plan = _reseal_plan(monkeypatch, tampered)
    with pytest.raises(u1.SourceInputError):
        _prepare_custom(monkeypatch, ctx, plan)


def test_tampered_candidate_rejected_on_pair(monkeypatch):
    ctx = fixture._build_context(monkeypatch, "master_cv")
    jobs = copy.deepcopy(ctx["case"]["full_jobs"])
    fit, prediction = _first_classical_source_pair(jobs)
    assert fit is not None
    digest = sha256_value({"missing": True})
    new_fit = _rebuild_job(
        fit, [], candidate_id="CAND-missing", hyperparameter_sha256=digest
    )
    new_prediction = _rebuild_job(
        prediction,
        [new_fit["job_id"]],
        candidate_id="CAND-missing",
        hyperparameter_sha256=digest,
    )
    tampered = _replace_jobs(
        jobs, {fit["job_id"]: new_fit, prediction["job_id"]: new_prediction}
    )
    plan = _reseal_plan(monkeypatch, tampered)
    factory = _prepare_custom(monkeypatch, ctx, plan)
    assert new_fit["job_id"] in factory.fit_job_ids()
    with pytest.raises(u1.SourceInputError):
        factory.pair(new_fit["job_id"])


def test_tampered_spec_rejected_on_pair(monkeypatch):
    ctx = fixture._build_context(monkeypatch, "master_cv")
    jobs = copy.deepcopy(ctx["case"]["full_jobs"])
    fit, prediction = _first_classical_source_pair(jobs)
    assert fit is not None
    replacement = "1" * 64
    new_fit = _rebuild_job(fit, [], model_spec_sha256=replacement)
    new_prediction = _rebuild_job(
        prediction, [new_fit["job_id"]], model_spec_sha256=replacement
    )
    tampered = _replace_jobs(
        jobs, {fit["job_id"]: new_fit, prediction["job_id"]: new_prediction}
    )
    plan = _reseal_plan(monkeypatch, tampered)
    factory = _prepare_custom(monkeypatch, ctx, plan)
    assert new_fit["job_id"] in factory.fit_job_ids()
    with pytest.raises(u1.SourceInputError):
        factory.pair(new_fit["job_id"])


def test_tampered_plan_metadata_and_action_rejected(monkeypatch):
    ctx = fixture._build_context(monkeypatch, "master_cv")
    plan = _install_plan(monkeypatch, ctx)

    tampered_plan = dict(plan)
    jobs = copy.deepcopy(plan["jobs"])
    jobs[0]["job_id"] = "P08FIT-tampered"
    tampered_plan["jobs"] = jobs
    with pytest.raises(u1.SourceInputError):
        _call_factory(ctx, tampered_plan)

    metadata = _metadata(ctx)
    manifest = metadata["manifest_bytes"]
    bad_metadata = dict(metadata)
    bad_metadata["manifest_bytes"] = manifest[:-1] + bytes([manifest[-1] ^ 0x01])
    with pytest.raises(u1.SourceInputError):
        _call_factory(ctx, plan, metadata=bad_metadata)

    audit, source = fixture._public_bytes()
    with pytest.raises(u1.SourceInputError):
        _call_factory(ctx, plan, audit=audit[:-1] + bytes([audit[-1] ^ 0x01]), source=source)

    action_bytes = dict(ctx["action_bytes"])
    first = sorted(action_bytes)[0]
    blob = action_bytes[first]
    action_bytes[first] = blob[:-1] + bytes([blob[-1] ^ 0x01])
    with pytest.raises(u1.SourceInputError):
        u1.prepare_source_factory(
            plan=plan,
            metadata_bytes=metadata,
            action_bytes=action_bytes,
            specification_audit_bytes=audit,
            specification_source_bytes=source,
            candidate_registry_bytes=fixture.CANDIDATE_BYTES,
        )


def test_unknown_fit_id_rejected(monkeypatch):
    ctx = fixture._build_context(monkeypatch, "master_cv")
    _plan_dict, factory = _prepare(monkeypatch, ctx)
    with pytest.raises(u1.SourceInputError):
        factory.pair("P08FIT-unknown")
    with pytest.raises(u1.SourceInputError):
        factory.pair(None)


def _held_uids(metadata):
    rows = _u0_inputs._read_csv(
        metadata["roles_bytes"],
        _u0_inputs._ROLE_COLUMNS,
        _u0_inputs._ROLE_MAX_ROWS,
        "invalid_roles",
    )
    roles = _u0_inputs._parse_roles(rows)
    return {row[4] for row in roles if row[2] == "outer_test"}


def test_held_rows_excluded_and_arrays_are_readonly_float32(monkeypatch):
    ctx = fixture._build_context(monkeypatch, "master_cv")
    _plan_dict, factory = _prepare(monkeypatch, ctx)
    metadata = _metadata(ctx)
    held = _held_uids(metadata)
    substrate = _u0_runtime._read_substrate_metadata(metadata["manifest_bytes"])

    assert factory.fit_job_ids()
    for fit_id in factory.fit_job_ids():
        pair = factory.pair(fit_id)
        for observation in pair.fitting_observations + pair.validation_observations:
            assert observation.uid not in held
            assert observation.substrate == substrate[observation.uid]
        values = pair.prepared_pair.inputs.fitting_values()
        assert values.dtype == np.float32
        assert not values.flags.writeable
        assert values.shape[1] == _arrays.FEATURES


def test_no_mutation_escape(monkeypatch):
    ctx = fixture._build_context(monkeypatch, "master_cv")
    plan, factory = _prepare(monkeypatch, ctx)
    ids = factory.fit_job_ids()
    plan["execution_authorized"] = True
    plan["jobs"] = []
    assert factory.fit_job_ids() == ids
    pair = factory.pair(ids[0])
    assert pair.configuration()
    report = factory.public_report()
    report["execution_authorized"] = True
    report["report_sha256"] = "0" * 64
    assert factory.public_report()["execution_authorized"] is False
    configuration = pair.configuration()
    configuration["injected"] = "marker"
    assert "injected" not in factory.pair(ids[0]).configuration()


def test_non_source_and_minimum_jobs_unavailable(monkeypatch):
    ctx = fixture._build_context(monkeypatch, "master_cv")
    _plan_dict, factory = _prepare(monkeypatch, ctx)
    available = set(factory.fit_job_ids())
    all_ids = {job["job_id"] for job in ctx["case"]["full_jobs"]}
    assert all_ids - available
    for job_id in all_ids - available:
        with pytest.raises(u1.SourceInputError):
            factory.pair(job_id)


def test_role_cache_bounded_and_reconstructs(monkeypatch):
    ctx = fixture._build_context(monkeypatch, "master_cv")
    monkeypatch.setattr(u1, "_ROLE_CACHE_MAXIMUM", 2)
    _plan_dict, factory = _prepare(monkeypatch, ctx)
    ids = factory.fit_job_ids()
    keys = {
        (record.policy_id, record.context_id, record.unit_id)
        for record in factory._records.values()
    }
    assert len(keys) > 2
    first = factory.pair(ids[0])
    for fit_id in ids:
        factory.pair(fit_id)
    assert len(factory._role_cache._items) <= 2
    _assert_pairs_equivalent(factory.pair(ids[0]), first)


def test_require_scientific_execution_denies(monkeypatch):
    with pytest.raises(u1.SourceInputError):
        u1.require_scientific_execution()
    with pytest.raises(u1.SourceInputError):
        u1.require_scientific_execution(execution_authorized=True)
