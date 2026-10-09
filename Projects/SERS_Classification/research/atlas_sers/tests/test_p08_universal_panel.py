"""Integration tests for the private P08 universal panel adapter.

One invented, self-consistent schema fixture is exercised against the pure
in-memory consistency boundary of ``p08_universal_panel.assemble_panel``. No
real data is loaded and no model is run.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p05_comparison, p08_plan
from atlas_sers.evaluation import p08_universal_panel as panel
from atlas_sers.governance.canonical import sha256_value

HELD = "INST-HELD"
SRC = "INST-SRC"
PRIM = "INST-PRIM"
STATION = "S1"
SECOND = "S2"
DOMAIN = "D-A"
LABELS = ("A", "B", "C")
FOLDS = (0, 1, 2, 3)
SOURCES = ("SRA", "SRB", "SRC")
PRIMARIES = ("PRA", "PRB", "PRC")
SELECTED = {0: "D0-M", 1: "D0-M", 2: "D1", 3: "D1"}
TEST_MASTERS = {fold: (f"T{fold}A", f"T{fold}B") for fold in FOLDS}


def _hash(value):
    return sha256_value(value)


def _uid(master):
    return f"{master}-1"


def _ctx(fold):
    return f"CTX-{fold}"


def _test_uids(fold):
    return sorted(_uid(master) for master in TEST_MASTERS[fold])


def _fit_uids():
    return sorted(_uid(master) for master in SOURCES)


def _manifest():
    rows = []
    for fold in FOLDS:
        for master in TEST_MASTERS[fold]:
            rows.append((_uid(master), master, STATION, master[-1], HELD))
    for master in SOURCES:
        rows.append((_uid(master), master, STATION, master[-1], SRC))
    for master in PRIMARIES:
        rows.append((_uid(master), master, SECOND, master[-1], PRIM))
    return pd.DataFrame(rows, columns=list(panel._MANIFEST_COLUMNS))


def _contexts():
    return pd.DataFrame(
        [
            {
                "context_id": _ctx(fold),
                "experiment_id": p05_comparison.SOURCE_HELD_EXPERIMENT,
                "phase_gate": p05_comparison.HELD_PHASE,
                "domain": DOMAIN,
                "station": STATION,
                "held_instrument": HELD,
                "outer_repeat": 1,
                "outer_fold": fold,
                "outer_test_uid_sha256": _hash(_test_uids(fold)),
                "outer_fit_uid_sha256": _hash(_fit_uids()),
            }
            for fold in FOLDS
        ]
    )


def _roles():
    rows = []
    for fold in FOLDS:
        context_id = _ctx(fold)
        for master in TEST_MASTERS[fold]:
            rows.append(
                (
                    context_id,
                    f"T-{fold}-{master}",
                    panel.TEST_ROLE,
                    "test",
                    _uid(master),
                    master,
                    master[-1],
                    HELD,
                )
            )
        for master in SOURCES:
            rows.append(
                (
                    context_id,
                    f"F-{fold}-{master}",
                    panel.SOURCE_ROLE,
                    "fit",
                    _uid(master),
                    master,
                    master[-1],
                    SRC,
                )
            )
    return pd.DataFrame(rows, columns=list(panel._ROLE_COLUMNS))


def _job(policy, representation, context_id, model_id, spec, test_sha, fit_sha):
    fields = {field: p08_plan.NOT_APPLICABLE for field in p08_plan.JOB_FIELDS}
    fields.update(
        {
            "policy_id": policy,
            "representation_id": representation,
            "array_sha256": _hash(representation),
            "context_id": context_id,
            "model_id": model_id,
            "model_spec_sha256": spec,
            "stage": panel.SEED_ENSEMBLE_PREDICTION,
            "test_uid_sha256": test_sha,
            "fit_uid_sha256": fit_sha,
            "dependencies": [],
        }
    )
    return {**fields, "job_id": "P08JOB-" + _hash(fields)}


def _probability_row(style):
    if style == "scalars":
        return {"probability_0": 0.8, "probability_1": 0.1, "probability_2": 0.1}
    if style == "scalar_strings":
        return {"probability_0": "0.8", "probability_1": "0.1", "probability_2": "0.1"}
    if style == "vector_list":
        return {"probabilities": [0.8, 0.1, 0.1]}
    return {"probabilities": json.dumps([0.8, 0.1, 0.1])}


def _frame(context_id, job, prob_style, seed_style):
    fold = int(context_id.split("-")[1])
    model = job["model_id"]
    expected = 1 if model == "C-RBF-SVM" else 3
    seed = (
        expected
        if seed_style == "int"
        else (f"{expected}.0" if seed_style == "csv" else f"{expected}.5")
    )
    status = (
        panel._NEURAL_STATUS
        if model in panel.NEURAL_RECIPES
        else panel._CLASSICAL_STATUS
    )
    model_id = (
        "P05-SELECTED" if model in panel.NEURAL_RECIPES and model != "D0-M" else model
    )
    rows = []
    for master in TEST_MASTERS[fold]:
        row = {
            "context_id": context_id,
            "domain": DOMAIN,
            "held_instrument": HELD,
            "representation_id": p08_plan.POLICY_REPRESENTATION[job["policy_id"]],
            "outer_repeat": 1,
            "outer_fold": fold,
            "observation_uid": _uid(master),
            "master_sample_id": master,
            "instrument": HELD,
            "station": STATION,
            "true_label": master[-1],
            "class_vocabulary": LABELS,
            "predicted_label": "A",
            "model_id": model_id,
            "fit_id": job["job_id"],
            "probability_status": status,
            "technical_seed_count": seed,
        }
        row.update(_probability_row(prob_style))
        rows.append(row)
    return pd.DataFrame(rows)


def _build_fixture(prob_style="scalars", seed_style="int"):
    spec = {model: _hash(model) for model in panel.ENDPOINT_MODELS}
    jobs, aliases, frames = [], [], {}
    for policy in panel.POLICIES:
        representation = p08_plan.POLICY_REPRESENTATION[policy]
        for fold in FOLDS:
            context_id = _ctx(fold)
            models = [*panel.CLASSICAL_MODELS, "D0-M"]
            if SELECTED[fold] != "D0-M":
                models.append(SELECTED[fold])
            by_model = {}
            for model in models:
                job = _job(
                    policy,
                    representation,
                    context_id,
                    model,
                    spec[model],
                    _hash(_test_uids(fold)),
                    _hash(_fit_uids()),
                )
                jobs.append(job)
                by_model[model] = job
                frames[job["job_id"]] = _frame(context_id, job, prob_style, seed_style)
            aliases.append(
                p08_plan._new_alias(
                    policy,
                    context_id,
                    panel.D0_STRATEGY,
                    "D0-M",
                    by_model["D0-M"]["job_id"],
                )
            )
            aliases.append(
                p08_plan._new_alias(
                    policy,
                    context_id,
                    panel.SELECTED_STRATEGY,
                    SELECTED[fold],
                    by_model[SELECTED[fold]]["job_id"],
                )
            )
    return {
        "manifest": _manifest(),
        "contexts": _contexts(),
        "roles": _roles(),
        "jobs": jobs,
        "aliases": aliases,
        "endpoint_frames": frames,
    }


def _assemble(fixture):
    return panel.assemble_panel(**fixture)


def _first_frame_id(fixture):
    return next(iter(fixture["endpoint_frames"]))


def _classical_job(fixture, policy="PP-U-MIN", context_id="CTX-0", model="C-RBF-SVM"):
    return next(
        job
        for job in fixture["jobs"]
        if job["policy_id"] == policy
        and job["context_id"] == context_id
        and job["model_id"] == model
    )


def test_full_assemble_returns_all_cells():
    out = _assemble(_build_fixture())
    assert len(out["coverage"]) == 60
    assert out["coverage"]["complete"].all()
    assert len(out["endpoint_index"]) == 60
    assert set(out["coverage"]["reference_count"]) == {1, 2}


def test_probabilities_copied_without_seed_averaging():
    out = _assemble(_build_fixture())
    matrix = out["predictions"][list(panel.PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    assert matrix.shape[0] == 120
    assert np.allclose(matrix, np.array([0.8, 0.1, 0.1]))


def test_selected_alias_shared_for_d0_and_separate_for_d1():
    coverage = _assemble(_build_fixture())["coverage"]
    shared = coverage[(coverage.context_id == "CTX-0") & (coverage.model_id == "D0-M")]
    assert set(shared.reference_count) == {2}
    d1 = coverage[
        (coverage.context_id == "CTX-2") & (coverage.model_id == "P05-SELECTED")
    ]
    assert set(d1.recipe_id) == {"D1"}
    assert set(d1.reference_count) == {1}
    separate = coverage[
        (coverage.context_id == "CTX-2") & (coverage.model_id == "D0-M")
    ]
    assert set(separate.reference_count) == {1}


def test_non_d0_neural_frame_may_record_selected_model_id():
    out = _assemble(_build_fixture())
    recipe = out["coverage"][out["coverage"].recipe_id == "D1"]
    assert not recipe.empty
    assert set(recipe.model_id) == {"P05-SELECTED"}


def test_global_identity_and_fixed_vocabulary():
    out = _assemble(_build_fixture())
    assert set(out["registered_test_rows"].class_vocabulary) == {LABELS}
    assert {"T0A", "SRA", "PRA"} <= set(out["global_masters"])
    assert PRIM in out["global_instruments"]
    assert out["panels"] is not None


def test_integer_master_ids_accepted():
    fixture = _build_fixture()
    fixture["manifest"].loc[
        fixture["manifest"].observation_uid.eq("T0A-1"), "master_sample_id"
    ] = "1001"
    fixture["roles"]["master_sample_id"] = fixture["roles"]["master_sample_id"].astype(
        object
    )
    fixture["roles"].loc[
        fixture["roles"].observation_uid.eq("T0A-1"), "master_sample_id"
    ] = 1001
    for frame in fixture["endpoint_frames"].values():
        if (frame.observation_uid == "T0A-1").any():
            frame["master_sample_id"] = frame["master_sample_id"].astype(object)
            frame.loc[frame.observation_uid.eq("T0A-1"), "master_sample_id"] = 1001
    assert len(_assemble(fixture)["coverage"]) == 60


@pytest.mark.parametrize(
    "prob_style", ["scalars", "scalar_strings", "vector_list", "vector_json"]
)
def test_probability_representation_styles(prob_style):
    assert len(_assemble(_build_fixture(prob_style=prob_style))["coverage"]) == 60


def test_csv_decimal_seed_count_accepted():
    assert len(_assemble(_build_fixture(seed_style="csv"))["coverage"]) == 60


def test_fractional_seed_count_rejected():
    with pytest.raises(panel.P08PanelError, match="endpoint_seed_count_mismatch"):
        _assemble(_build_fixture(seed_style="bad"))


def test_inputs_are_not_mutated():
    fixture = _build_fixture()
    snapshot = copy.deepcopy(fixture)
    _assemble(fixture)
    pd.testing.assert_frame_equal(fixture["manifest"], snapshot["manifest"])
    pd.testing.assert_frame_equal(fixture["contexts"], snapshot["contexts"])
    pd.testing.assert_frame_equal(fixture["roles"], snapshot["roles"])
    assert fixture["jobs"] == snapshot["jobs"]
    assert fixture["aliases"] == snapshot["aliases"]
    for job_id, frame in fixture["endpoint_frames"].items():
        pd.testing.assert_frame_equal(frame, snapshot["endpoint_frames"][job_id])


def _mut_missing_endpoint(fixture):
    job = _classical_job(fixture)
    fixture["jobs"].remove(job)
    fixture["endpoint_frames"].pop(job["job_id"])


def _mut_extra_uid(fixture):
    job_id = _first_frame_id(fixture)
    frame = fixture["endpoint_frames"][job_id]
    extra = frame.iloc[[0]].copy()
    extra["observation_uid"] = "GHOST-1"
    fixture["endpoint_frames"][job_id] = pd.concat([frame, extra], ignore_index=True)


def _mut_duplicate_uid(fixture):
    job_id = _first_frame_id(fixture)
    frame = fixture["endpoint_frames"][job_id]
    fixture["endpoint_frames"][job_id] = pd.concat(
        [frame, frame.iloc[[0]]], ignore_index=True
    )


def _mut_corrupt_argmax(fixture):
    job_id = _first_frame_id(fixture)
    frame = fixture["endpoint_frames"][job_id].copy()
    frame.loc[0, "predicted_label"] = "B"
    fixture["endpoint_frames"][job_id] = frame


def _mut_corrupt_vocabulary(fixture):
    job_id = _first_frame_id(fixture)
    frame = fixture["endpoint_frames"][job_id].copy()
    frame.at[0, "class_vocabulary"] = ("B", "A", "C")
    fixture["endpoint_frames"][job_id] = frame


def _mut_corrupt_identity(fixture):
    job_id = _first_frame_id(fixture)
    frame = fixture["endpoint_frames"][job_id].copy()
    frame.loc[0, "master_sample_id"] = "WRONG"
    fixture["endpoint_frames"][job_id] = frame


def _mut_source_held_instrument(fixture):
    fixture["manifest"].loc[
        fixture["manifest"].observation_uid.eq("SRA-1"), "instrument"
    ] = HELD
    fixture["roles"].loc[fixture["roles"].observation_uid.eq("SRA-1"), "instrument"] = (
        HELD
    )


def _mut_source_wrong_station(fixture):
    fixture["manifest"].loc[
        fixture["manifest"].observation_uid.eq("SRA-1"), "station"
    ] = SECOND


def _mut_source_master_overlap(fixture):
    held_master = TEST_MASTERS[0][0]
    fixture["manifest"].loc[
        fixture["manifest"].observation_uid.eq("SRA-1"), "master_sample_id"
    ] = held_master
    fixture["roles"].loc[
        fixture["roles"].observation_uid.eq("SRA-1"), "master_sample_id"
    ] = held_master


def _mut_alias_bad_hash(fixture):
    fixture["aliases"][0]["alias_id"] = "P08ALIAS-not-a-real-hash"


def _mut_alias_bad_target(fixture):
    for index, alias in enumerate(fixture["aliases"]):
        if (
            alias["policy_id"] == "PP-U-MIN"
            and alias["context_id"] == "CTX-0"
            and alias["strategy"] == "D0-M"
        ):
            target = _classical_job(fixture)["job_id"]
            fixture["aliases"][index] = p08_plan._new_alias(
                "PP-U-MIN", "CTX-0", "D0-M", "D0-M", target
            )
            return


def _mut_changed_selected_recipe(fixture):
    for index, alias in enumerate(fixture["aliases"]):
        if (
            alias["policy_id"] == "PP-U-ARPLS"
            and alias["context_id"] == "CTX-2"
            and alias["strategy"] == "P05-SELECTED"
        ):
            job = next(
                job
                for job in fixture["jobs"]
                if job["policy_id"] == "PP-U-ARPLS"
                and job["context_id"] == "CTX-2"
                and job["model_id"] == "D0-M"
            )
            fixture["aliases"][index] = p08_plan._new_alias(
                "PP-U-ARPLS", "CTX-2", "P05-SELECTED", "D0-M", job["job_id"]
            )
            return


REJECTIONS = (
    (_mut_missing_endpoint, "endpoint_missing"),
    (_mut_extra_uid, "endpoint_observation_unknown"),
    (_mut_duplicate_uid, "endpoint_observation_duplicate"),
    (_mut_corrupt_argmax, "endpoint_predicted_label_mismatch"),
    (_mut_corrupt_vocabulary, "endpoint_vocabulary_invalid"),
    (_mut_corrupt_identity, "endpoint_identity_mismatch"),
    (_mut_source_held_instrument, "outer_fit_held_instrument_present"),
    (_mut_source_wrong_station, "outer_fit_station_mismatch"),
    (_mut_source_master_overlap, "outer_fit_master_overlap"),
    (_mut_alias_bad_hash, "alias_hash_invalid"),
    (_mut_alias_bad_target, "alias_target_model_mismatch"),
    (_mut_changed_selected_recipe, "selected_recipe_inconsistent"),
)


@pytest.mark.parametrize(
    "mutate,code", REJECTIONS, ids=[code for _mutate, code in REJECTIONS]
)
def test_rejections(mutate, code):
    fixture = _build_fixture()
    mutate(fixture)
    with pytest.raises(panel.P08PanelError, match=code):
        _assemble(fixture)
