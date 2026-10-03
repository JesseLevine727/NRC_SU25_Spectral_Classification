"""Independent synthetic tests for the P08-T096 U0 source-metadata binding.

Five invented byte payloads (proposal JSON, attempt-manifest JSON and three
CSV registries) are authenticated against monkeypatched byte pins and bound
through ``atlas_sers.evaluation.p08_u0_inputs.bind_u0_source_metadata``.
Everything is deterministic, invented and in-memory: no real data, no
filesystem access and never any scientific execution.

The fixtures mirror the corrected production contract:

* the proposal carries the sorted full job list; the attempt manifest carries
  the globally sorted compact projection (``worker`` is added there only);
* roles use the four exact names ``outer_fit``/``outer_test`` (unit equals
  role) and ``selection_fit``/``selection_validation``;
* pseudo units are ``pseudo:INST-V0`` / ``pseudo:INST-V1`` / ``pseudo:INST-V2``
  matching the instrument strings on the invented observations;
* outer source/test membership is defined by ``_is_test`` alone;
* ``subject._BYTE_PINS`` maps the five argument names to SHA256(file bytes)
  while ``admission.U0_PROPOSAL_SHA256`` / ``admission.U0_MANIFEST_SHA256`` are
  the proposal/attempt *content* digests.
"""

from __future__ import annotations

import builtins
import csv
import hashlib
import io
import json
import random
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from atlas_sers.evaluation import p08_u0_admission as admission
from atlas_sers.evaluation import p08_u0_inputs as subject
from atlas_sers.evaluation.p08_plan import NEURAL_RECIPES, SEEDS, _hash, _new_job
from atlas_sers.governance.canonical import sha256_value

PARENT_PLAN_SHA256 = "179b95e8011a5f6cc02c65c7fab1acf0f6a6241ba1ef02378aac207b9e19cb03"
PROPOSAL_SCHEMA = "nato-sers-p08-universal-smoke-proposal-v1"
ATTEMPT_SCHEMA = "nato-sers-p08-attempt-manifest-v1"
EXPECTED_SEEDS = (20260805, 20260817, 20260829)

CLASSES = ("class-a", "class-b", "class-c")
STATION = "STATION-U0"
HELD_INSTRUMENT = "INST-HELD"
SOURCE_INSTRUMENT = "INST-A"
PSEUDO_INSTRUMENTS = ("INST-V0", "INST-V1", "INST-V2")
SENTINEL = "PRIVATE-COLUMN-SENTINEL-2f8a1c"

UNITS_MASTER = ("master_cv:0", "master_cv:1", "master_cv:2")
UNITS_PSEUDO = ("pseudo:INST-V0", "pseudo:INST-V1", "pseudo:INST-V2")

SELECTION_KEYS = {"selection_fit": "fit", "selection_validation": "validation"}
ROLE_STAGE_REASONS = ("role_binding_mismatch", "invalid_roles")

CLASSICAL_MODELS = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES")
REPRESENTATION = {"PP-U-SG": "R_SG_400_1800", "PP-U-ARPLS": "R_ARPLS_400_1800"}
POLICIES = ("PP-U-SG", "PP-U-ARPLS")
MODES = ("master_cv", "pseudo_domain")

ARG_NAMES = (
    "proposal_bytes",
    "attempt_manifest_bytes",
    "manifest_bytes",
    "contexts_bytes",
    "roles_bytes",
)

MANIFEST_FIELDS = (
    "observation_uid",
    "master_sample_id",
    "target_analyte",
    "instrument",
    "station",
    "private_column",
)
CONTEXT_FIELDS = (
    "context_id",
    "station",
    "held_instrument",
    "selection_mode",
    "phase_gate",
    "outer_fit_uid_sha256",
    "outer_test_uid_sha256",
)
ROLE_FIELDS = (
    "context_id",
    "role_id",
    "role",
    "selection_unit_id",
    "observation_uid",
    "master_sample_id",
    "target_analyte",
    "instrument",
)


def _digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha_bytes(data):
    return hashlib.sha256(bytes(data)).hexdigest()


def _units(mode):
    if mode == "master_cv":
        return UNITS_MASTER
    return UNITS_PSEUDO


def _worker(model_id):
    return "cpu" if model_id in CLASSICAL_MODELS else "gpu"


def _master(group, target):
    return f"M-{group}-{target}"


def _observation_rows(mode, repeats=(), extra_val_obs=False):
    rows = []
    for group in range(3):
        instrument = SOURCE_INSTRUMENT if mode == "master_cv" else PSEUDO_INSTRUMENTS[group]
        for target in CLASSES:
            master = _master(group, target)
            count = 2 if (group, target) in set(repeats) else 1
            for rep in range(count):
                rows.append(
                    {
                        "observation_uid": f"OBS-{mode}-{group}-{target}-{rep}",
                        "master_sample_id": master,
                        "target_analyte": target,
                        "instrument": instrument,
                        "station": STATION,
                        "private_column": SENTINEL,
                        "_group": group,
                        "_is_test": False,
                    }
                )
    if extra_val_obs:
        rows.append(
            {
                "observation_uid": f"OBS-{mode}-extra-{CLASSES[0]}",
                "master_sample_id": _master(0, CLASSES[0]),
                "target_analyte": CLASSES[0],
                "instrument": PSEUDO_INSTRUMENTS[1],
                "station": STATION,
                "private_column": SENTINEL,
                "_group": 0,
                "_is_test": False,
            }
        )
    for target in CLASSES:
        rows.append(
            {
                "observation_uid": f"OBS-{mode}-test-{target}",
                "master_sample_id": f"TEST-M-{target}",
                "target_analyte": target,
                "instrument": HELD_INSTRUMENT,
                "station": STATION,
                "private_column": SENTINEL,
                "_group": None,
                "_is_test": True,
            }
        )
    return rows


def _select(rows, mode):
    source = [r for r in rows if not r["_is_test"]]
    sets = {}
    for index, unit in enumerate(_units(mode)):
        if mode == "master_cv":
            validation = [r for r in source if r["_group"] == index]
        else:
            validation = [r for r in source if r["instrument"] == PSEUDO_INSTRUMENTS[index]]
        validation_masters = {r["master_sample_id"] for r in validation}
        fitting = [
            r
            for r in source
            if r["master_sample_id"] not in validation_masters
            and not (mode == "pseudo_domain" and r["instrument"] == PSEUDO_INSTRUMENTS[index])
        ]
        sets[unit] = {
            "fit": sorted(r["observation_uid"] for r in fitting),
            "validation": sorted(r["observation_uid"] for r in validation),
        }
    return sets


def _role_id(context_id, role, unit):
    return "P04ROLE-" + sha256_value({"context_id": context_id, "role": role, "unit": unit})[:24]


def _role_rows(context_id, mode, rows, sets):
    by_uid = {r["observation_uid"]: r for r in rows}
    out = []
    for role in ("outer_fit", "outer_test"):
        role_id = _role_id(context_id, role, role)
        members = sorted(
            (r for r in rows if (role == "outer_fit") != r["_is_test"]),
            key=lambda r: r["observation_uid"],
        )
        for row in members:
            out.append(
                {
                    "context_id": context_id,
                    "role_id": role_id,
                    "role": role,
                    "selection_unit_id": role,
                    "observation_uid": row["observation_uid"],
                    "master_sample_id": row["master_sample_id"],
                    "target_analyte": row["target_analyte"],
                    "instrument": row["instrument"],
                }
            )
    for unit in _units(mode):
        for role, key in (("selection_fit", "fit"), ("selection_validation", "validation")):
            role_id = _role_id(context_id, role, unit)
            for uid in sets[unit][key]:
                row = by_uid[uid]
                out.append(
                    {
                        "context_id": context_id,
                        "role_id": role_id,
                        "role": role,
                        "selection_unit_id": unit,
                        "observation_uid": uid,
                        "master_sample_id": row["master_sample_id"],
                        "target_analyte": row["target_analyte"],
                        "instrument": row["instrument"],
                    }
                )
    return out


def _sets_from_roles(role_rows, units, context_id):
    sets = {unit: {"fit": [], "validation": []} for unit in units}
    for row in role_rows:
        if row["context_id"] != context_id:
            continue
        key = SELECTION_KEYS.get(row["role"])
        if key is None or row["selection_unit_id"] not in sets:
            continue
        sets[row["selection_unit_id"]][key].append(row["observation_uid"])
    for unit in sets:
        sets[unit]["fit"] = sorted(set(sets[unit]["fit"]))
        sets[unit]["validation"] = sorted(set(sets[unit]["validation"]))
    return sets


def _rerole(row, context_id=None, unit=None):
    new = dict(row)
    if context_id is not None:
        new["context_id"] = context_id
    if unit is not None:
        new["selection_unit_id"] = unit
    new["role_id"] = _role_id(new["context_id"], new["role"], new["selection_unit_id"])
    return new


def _context_row(mode, rows, context_id):
    source = [r for r in rows if not r["_is_test"]]
    held = [r for r in rows if r["_is_test"]]
    return {
        "context_id": context_id,
        "station": STATION,
        "held_instrument": HELD_INSTRUMENT,
        "selection_mode": mode,
        "phase_gate": "held_evaluation",
        "outer_fit_uid_sha256": sha256_value(sorted(r["observation_uid"] for r in source)),
        "outer_test_uid_sha256": sha256_value(sorted(r["observation_uid"] for r in held)),
    }


def _add_extra_registry(contexts, role_rows, context_id):
    extra_id = context_id + "-extra"
    extra_context = dict(contexts[0])
    extra_context["context_id"] = extra_id
    cloned = [
        _rerole(row, context_id=extra_id) for row in role_rows if row["context_id"] == context_id
    ]
    extra_unit_roles = [
        _rerole(row, unit="master_cv:unused")
        for row in role_rows
        if row["context_id"] == context_id
        and row["role"] in SELECTION_KEYS
        and row["selection_unit_id"] == "master_cv:0"
    ]
    return contexts + [extra_context], role_rows + cloned + extra_unit_roles


def _model_fields(model_id, seed_index):
    spec = _digest("spec-" + model_id)
    if model_id in NEURAL_RECIPES:
        return SEEDS[seed_index], "fixed_recipe", spec
    if model_id == "C-RBF-SVM":
        return "deterministic", "cand-svm", _digest("hyper-svm")
    return SEEDS[seed_index], "cand-" + model_id, _digest("hyper-" + model_id)


def _build_jobs(context_id, units, sets, test_sha, pair_mutator=None):
    plans = []
    for policy in POLICIES:
        for unit in units:
            plans.append(("C-RBF-SVM", policy, unit, 0))
            for seed_index in range(3):
                plans.append(("C-RANDOM-FOREST", policy, unit, seed_index))
                plans.append(("C-EXTRA-TREES", policy, unit, seed_index))
                plans.append(("D0-M", policy, unit, seed_index))
        first = units[0]
        for seed_index in range(3):
            plans.append(("D1", policy, first, seed_index))
            plans.append(("D2", policy, first, seed_index))
            plans.append(("D3", policy, first, seed_index))
    jobs = []
    for plan in plans:
        model_id, policy, unit, seed_index = plan
        seed, candidate_id, hyper = _model_fields(model_id, seed_index)
        common = {
            "policy_id": policy,
            "representation_id": REPRESENTATION[policy],
            "array_sha256": _digest("array-" + policy),
            "context_id": context_id,
            "model_id": model_id,
            "model_spec_sha256": _digest("spec-" + model_id),
            "unit_id": unit,
            "seed": seed,
            "candidate_id": candidate_id,
            "hyperparameter_sha256": hyper,
            "fit_uid_sha256": sha256_value(sets[unit]["fit"]),
            "validation_uid_sha256": sha256_value(sets[unit]["validation"]),
            "test_uid_sha256": test_sha,
            "resolution": "fixed_spec",
            "evidence_status": "unapproved_future_job",
        }
        if pair_mutator is not None:
            pair_mutator(common, plan)
        fit = _new_job({**common, "stage": "source_fit"}, [])
        prediction = _new_job({**common, "stage": "source_validation_prediction"}, [fit["job_id"]])
        jobs.extend((fit, prediction))
    return jobs


def _compact(job):
    return {
        "job_id": job["job_id"],
        "stage": job["stage"],
        "worker": _worker(job["model_id"]),
        "dependencies": list(job["dependencies"]),
    }


def _json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _csv_bytes(fieldnames, rows, seed=0):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    shuffled = [dict(row) for row in rows]
    random.Random(seed).shuffle(shuffled)
    for row in shuffled:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _assemble(mode, rows, role_rows, sets, contexts, pair_mutator=None, attempt_mutator=None):
    units = _units(mode)
    context_id = contexts[0]["context_id"]
    test_uids = sorted(r["observation_uid"] for r in rows if r["_is_test"])
    full_jobs = _build_jobs(
        context_id, units, sets, sha256_value(test_uids), pair_mutator=pair_mutator
    )
    full_jobs.sort(key=lambda job: job["job_id"])
    compact_jobs = sorted((_compact(job) for job in full_jobs), key=lambda job: job["job_id"])
    proposal = {
        "schema_version": PROPOSAL_SCHEMA,
        "execution_authorized": False,
        "parent_plan_sha256": PARENT_PLAN_SHA256,
        "choice_rule": "u0_source_unit_choice_v1",
        "classical_candidate_rule": "classical_candidate_v1",
        "choices": [],
        "classical_candidates": [],
        "jobs": full_jobs,
    }
    proposal["proposal_sha256"] = _hash(proposal)
    attempt = {
        "schema_version": ATTEMPT_SCHEMA,
        "execution_authorized": False,
        "proposal_sha256": proposal["proposal_sha256"],
        "jobs": compact_jobs,
    }
    if attempt_mutator is not None:
        attempt_mutator(attempt["jobs"])
    attempt["manifest_sha256"] = sha256_value(
        {key: value for key, value in attempt.items() if key != "manifest_sha256"}
    )
    return {
        "mode": mode,
        "rows": rows,
        "role_rows": role_rows,
        "sets": sets,
        "contexts": contexts,
        "full_jobs": full_jobs,
        "compact_jobs": compact_jobs,
        "proposal": proposal,
        "attempt": attempt,
        "proposal_bytes": _json_bytes(proposal),
        "attempt_manifest_bytes": _json_bytes(attempt),
        "manifest_bytes": _csv_bytes(MANIFEST_FIELDS, rows, seed=1),
        "contexts_bytes": _csv_bytes(CONTEXT_FIELDS, contexts, seed=2),
        "roles_bytes": _csv_bytes(ROLE_FIELDS, role_rows, seed=3),
    }


def make_case(
    mode,
    *,
    repeats=(),
    extra_val_obs=False,
    drop_class=None,
    row_mutator=None,
    role_mutator=None,
    pair_mutator=None,
    attempt_mutator=None,
    subset_violation=False,
    extra_registry=False,
):
    rows = _observation_rows(mode, repeats=repeats, extra_val_obs=extra_val_obs)
    if drop_class is not None:
        rows = [r for r in rows if r["target_analyte"] != drop_class]
    if row_mutator is not None:
        rows = row_mutator(rows) or rows
    context_id = "ctx-" + mode
    sets = _select(rows, mode)
    role_rows = _role_rows(context_id, mode, rows, sets)
    if subset_violation:
        held_uid = next(r["observation_uid"] for r in rows if r["_is_test"])
        unit0 = _units(mode)[0]
        sets[unit0]["fit"] = sorted(set(sets[unit0]["fit"]) | {held_uid})
        role_rows = _role_rows(context_id, mode, rows, sets)
    if role_mutator is not None:
        role_mutator(rows, role_rows, sets)
    contexts = [_context_row(mode, rows, context_id)]
    if extra_registry:
        contexts, role_rows = _add_extra_registry(contexts, role_rows, context_id)
    return _assemble(
        mode,
        rows,
        role_rows,
        sets,
        contexts,
        pair_mutator=pair_mutator,
        attempt_mutator=attempt_mutator,
    )


def _seal(monkeypatch, case):
    pins = {name: _sha_bytes(case[name]) for name in ARG_NAMES}
    monkeypatch.setattr(subject, "_BYTE_PINS", pins)
    monkeypatch.setattr(admission, "U0_PROPOSAL_SHA256", case["proposal"]["proposal_sha256"])
    monkeypatch.setattr(admission, "U0_MANIFEST_SHA256", case["attempt"]["manifest_sha256"])
    return pins


def _call(case, **overrides):
    parts = {name: case[name] for name in ARG_NAMES}
    parts.update(overrides)
    return subject.bind_u0_source_metadata(**parts)


def _bind(monkeypatch, case):
    _seal(monkeypatch, case)
    return _call(case)


def _reason(monkeypatch, case, **overrides):
    _seal(monkeypatch, case)
    with pytest.raises(subject.BindingError) as info:
        _call(case, **overrides)
    assert isinstance(info.value.reason_code, str)
    assert SENTINEL not in str(info.value)
    return info.value


def _reject(monkeypatch, case, **overrides):
    return _reason(monkeypatch, case, **overrides)


def _reject_reason(monkeypatch, case, expected, **overrides):
    reason = _reason(monkeypatch, case, **overrides).reason_code
    if isinstance(expected, str):
        expected = (expected,)
    assert reason in expected, reason
    return reason


def _install_parse_spies(monkeypatch):
    calls = {"json": 0, "csv": 0}
    real_loads = json.loads

    def counting_loads(*args, **kwargs):
        calls["json"] += 1
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(json, "loads", counting_loads)

    real_reader = csv.DictReader

    class CountingReader(real_reader):
        def __init__(self, *args, **kwargs):
            calls["csv"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(csv, "DictReader", CountingReader)
    return calls


def _manifest_text(case):
    return case["manifest_bytes"].decode("utf-8")


def _with_manifest_text(case, text):
    return {**case, "manifest_bytes": text.encode("utf-8")}


def test_registered_seed_tuple_matches_reviewed_constants():
    assert tuple(SEEDS) == EXPECTED_SEEDS


@pytest.mark.parametrize("mode", MODES)
def test_binding_positive_counts_flags_and_registry(monkeypatch, mode):
    case = make_case(mode)
    binding = _bind(monkeypatch, case)
    report = binding.public_report()
    assert len(binding.pairs) == 78
    assert report["manifest_rows"] == len(case["rows"])
    assert report["selected_contexts"] == 1
    assert report["source_units"] == 3
    assert report["source_fit_jobs"] == 78
    assert report["source_prediction_jobs"] == 78
    assert report["cpu_fit_jobs"] == 42
    assert report["gpu_fit_jobs"] == 36
    assert report["registry_bytes_verified"] is True
    assert report["physical_role_isolation_verified"] is True
    assert report["ordered_source_roles_verified"] is True
    assert report["arrays_verified"] is False
    assert report["model_parameters_loaded"] is False
    assert report["live_controller_verified"] is False
    assert report["execution_authorized"] is False
    assert report["new_scientific_operations"] == 0
    assert report["proposal_sha256"] == case["proposal"]["proposal_sha256"]
    assert report["manifest_sha256"] == case["attempt"]["manifest_sha256"]


def test_extra_registry_context_and_extra_unit_are_ignored(monkeypatch):
    case = make_case("master_cv", extra_registry=True)
    assert len(case["contexts"]) == 2
    binding = _bind(monkeypatch, case)
    report = binding.public_report()
    assert report["selected_contexts"] == 1
    assert report["source_units"] == 3
    assert len(binding.pairs) == 78
    main_id = case["contexts"][0]["context_id"]
    assert {pair.source_roles.context_id for pair in binding.pairs} == {main_id}
    assert {pair.source_roles.unit_id for pair in binding.pairs} == set(_units("master_cv"))


@pytest.mark.parametrize("mode", MODES)
def test_public_report_is_fresh_and_digest_matches(monkeypatch, mode):
    binding = _bind(monkeypatch, make_case(mode))
    first = binding.public_report()
    second = binding.public_report()
    assert first is not second
    assert first == second
    without = {key: value for key, value in first.items() if key != "report_sha256"}
    assert first["report_sha256"] == sha256_value(without)
    first["tampered"] = True
    assert binding.public_report() == second
    assert isinstance(binding.report_json, str)
    assert json.loads(binding.report_json) == second


@pytest.mark.parametrize("mode", MODES)
def test_public_report_exposes_no_identifiers_or_sentinels(monkeypatch, mode):
    case = make_case(mode)
    binding = _bind(monkeypatch, case)
    rendered = binding.report_json + json.dumps(binding.public_report())
    for needle in (SENTINEL, "ctx-" + mode, "OBS-" + mode, "TEST-M-"):
        assert needle not in rendered
    for row in case["rows"]:
        assert row["observation_uid"] not in rendered
        assert row["master_sample_id"] not in rendered


@pytest.mark.parametrize("mode", MODES)
def test_pairs_are_immutable_and_project_exact_full_jobs(monkeypatch, mode):
    case = make_case(mode)
    binding = _bind(monkeypatch, case)
    assert isinstance(binding.pairs, tuple)
    jobs = {job["job_id"]: job for job in case["full_jobs"]}
    fit_ids = []
    expected = _sets_from_roles(case["role_rows"], _units(mode), case["contexts"][0]["context_id"])
    for pair in binding.pairs:
        fit = json.loads(pair.fit_job_json)
        prediction = json.loads(pair.prediction_job_json)
        assert fit == jobs[fit["job_id"]]
        assert prediction == jobs[prediction["job_id"]]
        assert fit["stage"] == "source_fit"
        assert prediction["stage"] == "source_validation_prediction"
        assert prediction["dependencies"] == [fit["job_id"]]
        fit_ids.append(fit["job_id"])
        with pytest.raises(FrozenInstanceError):
            pair.fit_job_json = "x"
        with pytest.raises(FrozenInstanceError):
            pair.prediction_job_json = "x"
        with pytest.raises(FrozenInstanceError):
            pair.source_roles = None
        roles = pair.source_roles
        with pytest.raises(FrozenInstanceError):
            roles.unit_id = "x"
        with pytest.raises(FrozenInstanceError):
            roles.fitting = ()
        with pytest.raises(FrozenInstanceError):
            roles.validation = ()
        with pytest.raises(FrozenInstanceError):
            roles.classes = ()
        assert tuple(sorted(o.observation_uid for o in roles.fitting)) == tuple(
            expected[roles.unit_id]["fit"]
        )
        assert tuple(sorted(o.observation_uid for o in roles.validation)) == tuple(
            expected[roles.unit_id]["validation"]
        )
        assert roles.classes == tuple(sorted(CLASSES))
        for observation in tuple(roles.fitting) + tuple(roles.validation):
            with pytest.raises(FrozenInstanceError):
                observation.observation_uid = "x"
            with pytest.raises(FrozenInstanceError):
                observation.master_sample_id = "x"
            with pytest.raises(FrozenInstanceError):
                observation.instrument = "x"
    assert fit_ids == sorted(fit_ids)


def test_private_repr_excludes_identifiers(monkeypatch):
    case = make_case("master_cv")
    binding = _bind(monkeypatch, case)
    rendered = repr(binding) + "".join(repr(pair) for pair in binding.pairs)
    for row in case["rows"]:
        assert row["observation_uid"] not in rendered
        assert row["master_sample_id"] not in rendered
    assert SENTINEL not in rendered
    assert "P04ROLE-" not in rendered


def test_role_instances_reused_per_context_and_unit(monkeypatch):
    binding = _bind(monkeypatch, make_case("master_cv"))
    seen = {}
    for pair in binding.pairs:
        roles = pair.source_roles
        key = (roles.context_id, roles.unit_id)
        seen.setdefault(key, roles)
        assert roles is seen[key]
    assert len(seen) == 3


@pytest.mark.parametrize("mode", MODES)
def test_source_role_and_observation_metadata_are_bound(monkeypatch, mode):
    case = make_case(mode)
    binding = _bind(monkeypatch, case)
    by_uid = {row["observation_uid"]: row for row in case["rows"]}
    for pair in binding.pairs:
        roles = pair.source_roles
        assert roles.context_id == case["contexts"][0]["context_id"]
        assert roles.unit_id in _units(mode)
        assert roles.selection_mode == mode
        assert roles.fitting_role_id.startswith("P04ROLE-")
        assert roles.validation_role_id.startswith("P04ROLE-")
        assert set(roles.classes) == set(CLASSES)
        assert len(roles.classes) == 3
        assert roles.fitting and roles.validation
        for observation in tuple(roles.fitting) + tuple(roles.validation):
            row = by_uid[observation.observation_uid]
            assert observation.master_sample_id == row["master_sample_id"]
            assert observation.target_analyte == row["target_analyte"]
            assert observation.instrument == row["instrument"]
            assert observation.station == row["station"]


def test_unused_private_column_is_ignored(monkeypatch):
    case = make_case("master_cv")
    assert SENTINEL.encode("utf-8") in case["manifest_bytes"]
    binding = _bind(monkeypatch, case)
    assert SENTINEL not in binding.report_json


@pytest.mark.parametrize("mode", MODES)
def test_outer_test_items_never_enter_source_roles(monkeypatch, mode):
    case = make_case(mode)
    binding = _bind(monkeypatch, case)
    test_uids = {row["observation_uid"] for row in case["rows"] if row["_is_test"]}
    bound = set()
    for pair in binding.pairs:
        for side in (pair.source_roles.fitting, pair.source_roles.validation):
            bound.update(observation.observation_uid for observation in side)
    assert not (bound & test_uids)


def test_repeated_observations_are_bound(monkeypatch):
    case = make_case("master_cv", repeats=((0, CLASSES[0]),))
    binding = _bind(monkeypatch, case)
    repeated = {
        row["observation_uid"] for row in case["rows"] if row["observation_uid"].endswith("-1")
    }
    assert repeated
    bound = set()
    for pair in binding.pairs:
        bound.update(o.observation_uid for o in pair.source_roles.fitting)
        bound.update(o.observation_uid for o in pair.source_roles.validation)
    assert repeated <= bound


def test_pseudo_domain_extra_validation_observation_not_in_fit(monkeypatch):
    case = make_case("pseudo_domain", extra_val_obs=True)
    binding = _bind(monkeypatch, case)
    extra_uid = f"OBS-pseudo_domain-extra-{CLASSES[0]}"
    assert any(row["observation_uid"] == extra_uid for row in case["rows"])
    pair = next(p for p in binding.pairs if p.source_roles.unit_id == "pseudo:INST-V0")
    masters_fit = {o.master_sample_id for o in pair.source_roles.fitting}
    masters_val = {o.master_sample_id for o in pair.source_roles.validation}
    assert _master(0, CLASSES[0]) in masters_val
    assert _master(0, CLASSES[0]) not in masters_fit
    assert not (masters_fit & masters_val)


def test_binding_does_not_mutate_input_bytes(monkeypatch):
    case = make_case("master_cv")
    originals = {name: case[name] for name in ARG_NAMES}
    _bind(monkeypatch, case)
    for name in ARG_NAMES:
        assert case[name] == originals[name]
        assert isinstance(case[name], bytes)


@pytest.mark.parametrize("name", ARG_NAMES)
def test_tampered_argument_rejected_before_parsing(monkeypatch, name):
    case = make_case("master_cv")
    _seal(monkeypatch, case)
    spies = _install_parse_spies(monkeypatch)
    tampered = bytes([case[name][0] ^ 0xFF]) + case[name][1:]
    try:
        with pytest.raises(subject.BindingError) as info:
            _call(case, **{name: tampered})
        assert info.value.reason_code == "bytes_not_authenticated"
    finally:
        assert spies == {"json": 0, "csv": 0}


def test_mutable_buffer_rejected(monkeypatch):
    case = make_case("master_cv")
    _reject(monkeypatch, case, proposal_bytes=bytearray(case["proposal_bytes"]))


def test_oversize_proposal_rejected(monkeypatch):
    case = make_case("master_cv")
    _seal(monkeypatch, case)
    spies = _install_parse_spies(monkeypatch)
    try:
        with pytest.raises(subject.BindingError) as info:
            _call(case, proposal_bytes=b"x" * ((1 << 20) + 1))
        assert info.value.reason_code == "invalid_input"
    finally:
        assert spies == {"json": 0, "csv": 0}


def test_bytes_subclass_rejected_but_exact_bytes_accepted(monkeypatch):
    class NotQuiteBytes(bytes):
        pass

    case = make_case("master_cv")
    _reject(monkeypatch, case, manifest_bytes=NotQuiteBytes(case["manifest_bytes"]))
    binding = _bind(monkeypatch, make_case("master_cv"))
    assert len(binding.pairs) == 78


@pytest.mark.parametrize("value", ["text", None, 5])
def test_non_bytes_argument_rejected(monkeypatch, value):
    case = make_case("master_cv")
    _reject(monkeypatch, case, proposal_bytes=value)


def test_duplicate_json_keys_rejected(monkeypatch):
    case = make_case("master_cv")
    raw = b'{"schema_version":"a","schema_version":"b"}'
    _reject_reason(monkeypatch, {**case, "proposal_bytes": raw}, "invalid_proposal")


def test_nonfinite_json_rejected(monkeypatch):
    case = make_case("master_cv")
    raw = b'{"schema_version":NaN}'
    _reject_reason(monkeypatch, {**case, "proposal_bytes": raw}, "invalid_proposal")


def test_duplicate_csv_headers_rejected(monkeypatch):
    case = make_case("master_cv")
    lines = _manifest_text(case).splitlines()
    lines[0] = lines[0].replace("master_sample_id", "observation_uid", 1)
    _reject_reason(
        monkeypatch, _with_manifest_text(case, "\n".join(lines) + "\n"), "invalid_manifest"
    )


def test_missing_required_column_rejected(monkeypatch):
    case = make_case("master_cv")
    reader = csv.DictReader(io.StringIO(_manifest_text(case)))
    fields = [field for field in reader.fieldnames if field != "station"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in reader:
        writer.writerow(row)
    _reject_reason(monkeypatch, _with_manifest_text(case, buffer.getvalue()), "invalid_manifest")


def test_missing_csv_cell_rejected(monkeypatch):
    case = make_case("master_cv")
    lines = _manifest_text(case).splitlines()
    lines[1] = ",".join(lines[1].split(",")[:-1])
    _reject_reason(
        monkeypatch, _with_manifest_text(case, "\n".join(lines) + "\n"), "invalid_manifest"
    )


def test_extra_csv_cell_rejected(monkeypatch):
    case = make_case("master_cv")
    lines = _manifest_text(case).splitlines()
    lines[1] = lines[1] + ",EXTRA"
    _reject_reason(
        monkeypatch, _with_manifest_text(case, "\n".join(lines) + "\n"), "invalid_manifest"
    )


def test_duplicate_uid_row_rejected(monkeypatch):
    case = make_case("master_cv")
    lines = _manifest_text(case).splitlines()
    lines.insert(2, lines[1])
    _reject_reason(
        monkeypatch, _with_manifest_text(case, "\n".join(lines) + "\n"), "invalid_manifest"
    )


def test_role_label_mismatch_rejected(monkeypatch):
    def mutate(rows, role_rows, sets):
        for row in role_rows:
            if (
                row["role"] == "selection_fit"
                and row["selection_unit_id"] == _units("master_cv")[0]
            ):
                row["target_analyte"] = "tampered-analyte"
                break

    _reject_reason(monkeypatch, make_case("master_cv", role_mutator=mutate), ROLE_STAGE_REASONS)


def test_missing_role_rejected(monkeypatch):
    def mutate(rows, role_rows, sets):
        unit0 = _units("master_cv")[0]
        role_rows[:] = [
            row
            for row in role_rows
            if not (row["role"] == "selection_validation" and row["selection_unit_id"] == unit0)
        ]

    _reject_reason(monkeypatch, make_case("master_cv", role_mutator=mutate), ROLE_STAGE_REASONS)


def test_same_master_in_fitting_and_validation_rejected(monkeypatch):
    def mutate(rows, role_rows, sets):
        unit0 = _units("master_cv")[0]
        uid = next(
            row["observation_uid"]
            for row in role_rows
            if row["role"] == "selection_fit"
            and row["selection_unit_id"] == unit0
            and row["target_analyte"] == CLASSES[0]
        )
        new_master = _master(0, CLASSES[0])
        for row in rows:
            if row["observation_uid"] == uid:
                row["master_sample_id"] = new_master
        for row in role_rows:
            if row["observation_uid"] == uid:
                row["master_sample_id"] = new_master

    _reject_reason(
        monkeypatch,
        make_case("master_cv", role_mutator=mutate),
        "role_binding_mismatch",
    )


def test_outer_test_master_reused_in_source_rejected(monkeypatch):
    def mutate(rows):
        rows.append(
            {
                "observation_uid": "OBS-test-reuse",
                "master_sample_id": _master(0, CLASSES[0]),
                "target_analyte": CLASSES[0],
                "instrument": HELD_INSTRUMENT,
                "station": STATION,
                "private_column": SENTINEL,
                "_group": None,
                "_is_test": True,
            }
        )
        return rows

    _reject_reason(monkeypatch, make_case("master_cv", row_mutator=mutate), ROLE_STAGE_REASONS)


def test_source_observation_on_held_instrument_rejected(monkeypatch):
    def mutate(rows, role_rows, sets):
        uid = next(r["observation_uid"] for r in rows if not r["_is_test"])
        for row in rows:
            if row["observation_uid"] == uid:
                row["instrument"] = HELD_INSTRUMENT
        for row in role_rows:
            if row["observation_uid"] == uid:
                row["instrument"] = HELD_INSTRUMENT

    _reject_reason(monkeypatch, make_case("master_cv", role_mutator=mutate), ROLE_STAGE_REASONS)


def test_wrong_pseudo_instrument_rejected(monkeypatch):
    def mutate(rows, role_rows, sets):
        unit0 = _units("pseudo_domain")[0]
        uid = next(
            row["observation_uid"]
            for row in role_rows
            if row["role"] == "selection_validation"
            and row["selection_unit_id"] == unit0
            and row["target_analyte"] == CLASSES[0]
        )
        for row in rows:
            if row["observation_uid"] == uid:
                row["instrument"] = PSEUDO_INSTRUMENTS[2]
        for row in role_rows:
            if row["observation_uid"] == uid:
                row["instrument"] = PSEUDO_INSTRUMENTS[2]

    _reject_reason(monkeypatch, make_case("pseudo_domain", role_mutator=mutate), ROLE_STAGE_REASONS)


def test_role_id_mismatch_rejected(monkeypatch):
    def mutate(rows, role_rows, sets):
        for row in role_rows:
            if row["role"] == "selection_fit":
                row["role_id"] = "P04ROLE-" + "0" * 24
                break

    _reject_reason(monkeypatch, make_case("master_cv", role_mutator=mutate), ROLE_STAGE_REASONS)


def test_source_uid_set_hash_mismatch_rejected(monkeypatch):
    target_plan = ("D0-M", POLICIES[1], _units("master_cv")[2], 2)

    def mutate(common, plan):
        if plan == target_plan:
            common["fit_uid_sha256"] = _digest("wrong-uid-set")

    _reject_reason(
        monkeypatch,
        make_case("master_cv", pair_mutator=mutate),
        "role_binding_mismatch",
    )


def test_subset_violation_rejected(monkeypatch):
    _reject_reason(
        monkeypatch,
        make_case("master_cv", subset_violation=True),
        "role_binding_mismatch",
    )


def test_missing_class_rejected(monkeypatch):
    _reject_reason(monkeypatch, make_case("master_cv", drop_class=CLASSES[2]), ROLE_STAGE_REASONS)


def test_attempt_worker_projection_mismatch_rejected(monkeypatch):
    def mutate(jobs):
        jobs[0]["worker"] = "gpu" if jobs[0]["worker"] == "cpu" else "cpu"

    _reject(monkeypatch, make_case("master_cv", attempt_mutator=mutate))


def test_attempt_dependency_projection_mismatch_rejected(monkeypatch):
    def mutate(jobs):
        jobs[0]["dependencies"] = ["P08JOB-" + "0" * 64]

    _reject(monkeypatch, make_case("master_cv", attempt_mutator=mutate))


def test_require_scientific_execution_always_denies():
    calls = (
        lambda: subject.require_scientific_execution(),
        lambda: subject.require_scientific_execution(True),
        lambda: subject.require_scientific_execution(execution_authorized=True),
        lambda: subject.require_scientific_execution(None, authorized=True, token="forged"),
    )
    for call in calls:
        with pytest.raises(subject.BindingError) as info:
            call()
        assert info.value.reason_code == "scientific_execution_not_authorized"


def test_ordinary_error_is_sanitized(monkeypatch):
    def boom(_value):
        raise RuntimeError("internal detail " + SENTINEL)

    monkeypatch.setattr(subject, "sha256_value", boom, raising=False)
    case = make_case("master_cv")
    _seal(monkeypatch, case)
    with pytest.raises(subject.BindingError) as info:
        _call(case)
    assert isinstance(info.value.reason_code, str)
    assert "internal detail" not in str(info.value)
    assert SENTINEL not in str(info.value)


def test_keyboard_interrupt_propagates_same_object(monkeypatch):
    signal = KeyboardInterrupt("stop")

    def boom(_value):
        raise signal

    monkeypatch.setattr(subject, "sha256_value", boom, raising=False)
    case = make_case("master_cv")
    _seal(monkeypatch, case)
    with pytest.raises(KeyboardInterrupt) as info:
        _call(case)
    assert info.value is signal


def test_system_exit_propagates_same_object(monkeypatch):
    signal = SystemExit(7)

    def boom(_value):
        raise signal

    monkeypatch.setattr(subject, "sha256_value", boom, raising=False)
    case = make_case("master_cv")
    _seal(monkeypatch, case)
    with pytest.raises(SystemExit) as info:
        _call(case)
    assert info.value is signal


def test_no_scientific_or_filesystem_operations(monkeypatch):
    from atlas_sers.evaluation import p03_runtime, p05_development

    counters = {"open": 0, "fit": 0, "train": 0}

    def deny_open(*args, **kwargs):
        counters["open"] += 1
        raise AssertionError("filesystem access attempted")

    def deny_fit(*args, **kwargs):
        counters["fit"] += 1
        raise AssertionError("candidate fit executed")

    def deny_train(*args, **kwargs):
        counters["train"] += 1
        raise AssertionError("development fit executed")

    monkeypatch.setattr(builtins, "open", deny_open)
    monkeypatch.setattr(Path, "open", deny_open)
    monkeypatch.setattr(p03_runtime, "run_candidate_fit", deny_fit)
    monkeypatch.setattr(p05_development, "train_development_fit", deny_train)
    try:
        binding = _bind(monkeypatch, make_case("master_cv"))
        assert len(binding.pairs) == 78
    finally:
        assert counters == {"open": 0, "fit": 0, "train": 0}
