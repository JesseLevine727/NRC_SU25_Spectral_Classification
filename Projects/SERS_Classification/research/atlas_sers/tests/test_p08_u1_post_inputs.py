"""Authentic interface regressions for post-source input preparation; no fitting."""

from __future__ import annotations

import hashlib
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p03_roles, p03_runtime, p05_refit, p08_plan
from atlas_sers.evaluation import p08_u1_post_inputs as mod
from atlas_sers.governance.canonical import sha256_value


def digest(uids):
    return sha256_value(sorted(uids))


@pytest.fixture
def case(monkeypatch):
    uids = [f"u{i}" for i in range(9)]
    manifest = pd.DataFrame(
        {
            "observation_uid": uids,
            "master_sample_id": [f"00{i}" for i in range(9)],
            "target_analyte": list("ABC") * 3,
            "station": ["S"] * 9,
            "instrument": ["I"] * 3 + ["J"] * 3 + ["H"] * 3,
            "sensor_family": ["NA", " Ag ", "Au"] * 3,
            "first_difference_noise_mad": [0.7] * 9,
            "intensity_range": [17.0] * 9,
        }
    )
    contexts = pd.DataFrame(
        [
            dict(
                context_id="ctx",
                phase_gate="held_evaluation",
                station="S",
                held_instrument="H",
                domain="S:H",
                outer_repeat=1,
                outer_fold=0,
            )
        ]
    )
    roles = pd.DataFrame(
        {
            "context_id": ["ctx"] * 9,
            "role": ["outer_fit"] * 6 + ["outer_test"] * 3,
            "role_id": ["original-fit-role"] * 6 + ["original-test-role"] * 3,
            "observation_uid": uids,
        }
    )
    template = dict(
        experiment_id="EXP-C10-T3",
        stage="calibration_crossfit",
        model_id="C-RBF-SVM",
        domain="S:H",
        outer_repeat=1,
        outer_fold=0,
        selection_unit_id="calibration_master_cv:0",
        fit_uid_sha256=digest(uids[:3]),
        validation_uid_sha256=digest(uids[3:6]),
        test_uid_sha256=digest(uids[6:]),
    )
    parameters = {"C": 1.0}
    candidate = dict(
        candidate_id="candidate",
        model_id="C-RBF-SVM",
        hyperparameter_sha256=sha256_value(parameters),
        parameters_json=json.dumps(parameters),
    )
    factory = SimpleNamespace(
        _actions={
            r: {"intensity": np.tile(np.linspace(0, 1, 1401, dtype=np.float32), (9, 1))}
            for r in ("R_SG_400_1800", "R_ARPLS_400_1800")
        },
        _manifest_index={u: i for i, u in enumerate(uids)},
        _specification_hashes={m: "a" * 64 for m in ("C-RBF-SVM", "D1")},
        _candidate_index={("C-RBF-SVM", "candidate"): candidate},
        _roles={
            ("ctx", "master_cv:0"): SimpleNamespace(
                context_id="ctx",
                fitting=tuple(SimpleNamespace(observation_uid=u) for u in uids[:3]),
                validation=tuple(SimpleNamespace(observation_uid=u) for u in uids[3:6]),
            )
        },
    )
    inputs = mod.PostSourceInputs(factory, manifest, contexts, roles, pd.DataFrame([template]), {})

    def resolve(row, *, manifest, p02_tables):
        assert row.selection_unit_id == "calibration_master_cv:0"
        return p03_roles.ResolvedRoles(manifest.iloc[:3], manifest.iloc[3:6], manifest.iloc[6:])

    monkeypatch.setattr(p03_roles, "resolve_fit_roles", resolve)
    return inputs


def job(stage="final_refit", model="C-RBF-SVM", policy="PP-U-SG", **changes):
    inner = stage in (
        "calibration_model_fit",
        "calibration_validation_prediction",
        "source_fit",
        "source_validation_prediction",
        "calibration_prediction_alias",
    )
    representation = p08_plan.POLICY_REPRESENTATION[policy]
    fields = dict(
        policy_id=policy,
        representation_id=representation,
        array_sha256=mod._ACTION_PINS[representation]["array_sha256"],
        context_id="ctx",
        model_id=model,
        model_spec_sha256="a" * 64,
        stage=stage,
        unit_id="calibration_master_cv:0" if stage.startswith("calibration_") else "master_cv:0",
        seed="deterministic" if model == "C-RBF-SVM" else 20260805,
        candidate_id=p08_plan.SOURCE_SELECTION_DEPENDENT,
        hyperparameter_sha256=p08_plan.NOT_APPLICABLE,
        fit_uid_sha256=digest([f"u{i}" for i in range(3 if inner else 6)]),
        validation_uid_sha256=digest(["u3", "u4", "u5"]) if inner else p08_plan.NOT_APPLICABLE,
        test_uid_sha256=digest(["u6", "u7", "u8"]),
        resolution=p08_plan.SOURCE_SELECTION_DEPENDENT,
        evidence_status=p08_plan.EVIDENCE_FUTURE,
    )
    fields.update(changes)
    return p08_plan._new_job(fields, ["selection"])


def selection():
    return dict(
        selection_job_id="selection",
        policy_id="PP-U-SG",
        context_id="ctx",
        model_id="C-RBF-SVM",
        selected_candidate_id="candidate",
        selected_hyperparameter_sha256=sha256_value({"C": 1.0}),
        selected_parameters={"C": 1.0},
    )


@pytest.mark.parametrize("stage", ["calibration_model_fit", "final_refit"])
def test_classical_binds_actual_kernel(case, stage):
    record = job(stage)
    kwargs = case.classical_fit_kwargs(record, selection())
    kernel = (
        p03_runtime.run_candidate_fit
        if stage == "calibration_model_fit"
        else p03_runtime.run_final_fit
    )
    inspect.signature(kernel).bind(**kwargs)
    assert kwargs["fit_id"] == record["job_id"]
    assert kwargs["seed"] == "deterministic" and kwargs["parameters"] == {"C": 1.0}
    assert not set(kwargs["dataset"].metadata.observation_uid) & {"u6", "u7", "u8"}


@pytest.mark.parametrize("policy", ["PP-U-SG", "PP-U-ARPLS"])
@pytest.mark.parametrize("epochs", [30, 200])
def test_neural_binds_actual_kernel_native_qc_and_labels(case, policy, epochs):
    kwargs = case.neural_refit_kwargs(job(model="D1", policy=policy), epochs)
    inspect.signature(p05_refit.train_refit).bind(**kwargs, device="cpu")
    assert kwargs["role_id"] == "original-fit-role"
    assert kwargs["epochs"] == epochs and kwargs["maximum_fit_seconds"] == 120.0
    assert kwargs["maximum_cuda_allocated_bytes"] == 4 * 2**30
    assert [o.substrate for o in kwargs["observations"][:3]] == ["NA", " Ag ", "Au"]
    assert [o.master for o in kwargs["observations"][:3]] == ["000", "001", "002"]
    np.testing.assert_array_equal(kwargs["noise_metadata"].first_difference_noise_mad, [0.7] * 6)
    np.testing.assert_array_equal(kwargs["noise_metadata"].intensity_range, [17.0] * 6)
    assert kwargs["values"].shape == (6, 1401) and kwargs["values"].flags.writeable
    kwargs["values"][:] = -1
    assert (
        case._source_factory._actions[p08_plan.POLICY_REPRESENTATION[policy]]["intensity"].min()
        == 0
    )


@pytest.mark.parametrize("epochs", [0, 29, 201, True])
def test_bad_epochs(case, epochs):
    with pytest.raises(ValueError):
        case.neural_refit_kwargs(job(model="D1"), epochs)


@pytest.mark.parametrize(
    "changes",
    [
        dict(array_sha256="b" * 64),
        dict(model_spec_sha256="b" * 64),
        dict(seed=20260805),
        dict(fit_uid_sha256="b" * 64),
        dict(test_uid_sha256="b" * 64),
    ],
)
def test_rehashed_bad_job_rejected(case, changes):
    with pytest.raises(ValueError):
        case.classical_fit_kwargs(job(**changes), selection())


@pytest.mark.parametrize(
    "changes",
    [
        dict(selection_job_id="wrong"),
        dict(selected_candidate_id="absent"),
        dict(selected_parameters={"C": 2.0}),
        dict(selected_hyperparameter_sha256="b" * 64),
    ],
)
def test_selection_mismatch_rejected_for_final(case, changes):
    chosen = selection() | changes
    with pytest.raises(ValueError):
        case.classical_fit_kwargs(job(), chosen)


def test_calibration_cache_rechecks_hashes(case):
    case.classical_fit_kwargs(job("calibration_model_fit"), selection())
    with pytest.raises(ValueError):
        case.classical_fit_kwargs(
            job("calibration_model_fit", validation_uid_sha256="b" * 64), selection()
        )


@pytest.mark.parametrize(
    "column,value", [("instrument", "H"), ("station", "wrong"), ("master_sample_id", "006")]
)
def test_isolation_rejects_before_fitting(case, column, value):
    case._manifest.loc[0, column] = value
    with pytest.raises(ValueError):
        case.neural_refit_kwargs(job(model="D1"), 30)


@pytest.mark.parametrize(
    "column,value",
    [
        ("first_difference_noise_mad", float("nan")),
        ("intensity_range", 0.0),
        ("first_difference_noise_mad", -1.0),
    ],
)
def test_invalid_native_noise_rejected(case, column, value):
    case._manifest.loc[0, column] = value
    with pytest.raises(ValueError):
        case.neural_refit_kwargs(job(model="D1"), 30)


def test_held_only_and_role_copy(case):
    record = job("held_prediction", model="D1")
    values, metadata, classes, forbidden = case.held_inputs(record)
    assert metadata.observation_uid.tolist() == ["u6", "u7", "u8"]
    assert classes == ("A", "B", "C") and len(forbidden) == 6
    assert values.shape == (3, 1401)
    metadata.loc[0, "sensor_family"] = "changed"
    assert case.role_metadata(record, "test").sensor_family.iloc[0] == "NA"


def test_source_hashes_and_alias(case):
    record = job("source_validation_prediction")
    assert case.role_metadata(record, "validation").observation_uid.tolist() == ["u3", "u4", "u5"]
    alias = job("calibration_prediction_alias")
    assert len(case.role_metadata(alias, "fit")) == 3
    with pytest.raises(ValueError):
        case.role_metadata(job("source_fit", fit_uid_sha256="b" * 64), "fit")


def test_prepare_preserves_text_and_pins(case, monkeypatch):
    metadata = {
        "manifest_bytes": case._manifest.to_csv(index=False).encode(),
        "contexts_bytes": case._contexts.to_csv(index=False).encode(),
        "roles_bytes": case._roles.to_csv(index=False).encode(),
    }
    fit_bytes = case._fit_manifest.to_csv(index=False).encode()
    tables = {
        name: b"master_sample_id,outer_repeat,outer_fold,inner_fold\n001,1,0,0\n"
        for name in mod._P02_TABLE_KEYS
    }

    def h(b):
        return hashlib.sha256(b).hexdigest()

    monkeypatch.setattr(mod, "_BYTE_PINS", {k: h(v) for k, v in metadata.items()})
    monkeypatch.setattr(mod, "_FIT_MANIFEST_SHA256", h(fit_bytes))
    monkeypatch.setattr(mod, "_P02_PINS", {k: h(v) for k, v in tables.items()})
    args = dict(
        source_factory=case._source_factory,
        metadata_bytes=metadata,
        p03_fit_manifest_bytes=fit_bytes,
        p02_table_bytes=tables,
    )
    result = mod.prepare_post_source_inputs(**args)
    assert result._manifest.master_sample_id.iloc[1] == "001"
    assert result._manifest.sensor_family.iloc[0] == "NA"
    assert result._manifest.sensor_family.iloc[1] == " Ag "
    assert result._p02_tables[mod._P02_TABLE_KEYS[0]].master_sample_id.iloc[0] == "001"
    with pytest.raises(ValueError):
        mod.prepare_post_source_inputs(**(args | {"p03_fit_manifest_bytes": fit_bytes + b"\n"}))
