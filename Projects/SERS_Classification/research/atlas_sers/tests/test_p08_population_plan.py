"""Invention-only tests for the P08 filtered-population metadata plan builder.

These tests never fit a model, select a recipe or compute a scientific metric.
They exercise the public ``build_population_plan`` contract over synthetic
metadata and independently recompute canonical set/table hashes with
``atlas_sers.governance.canonical.sha256_value``.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from atlas_sers.evaluation import p08_population_plan as p08
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.splits import p02

build_population_plan = p08.build_population_plan

PID = "POP-1"
DIGEST = sha256_value({"population": "fixture-1"})
ALGORITHM = "StratifiedGroupKFold with shuffle true and repeat seed"
FOUR_ROLES = {"source", "test", "fit", "validation"}
TABLE_NAMES = (
    "master_splits",
    "domain_registry",
    "t3_partitions",
    "context_registry",
    "inner_selection_registry",
    "inner_master_split_registry",
    "unit_registry",
    "role_registry",
)
VALID_PSEUDO_SUPPORT = {
    "validation_unit": "one source acquisition instrument",
    "validation_requires_all_station_classes": True,
    "remaining_training_requires_all_station_classes": True,
    "minimum_supported_pseudo_domains": 2,
    "fallback": "three-fold stratified master-grouped inner CV",
}


def _split_contract(*, domains, seeds=(17,), folds=4, classes=3, masters=3, algorithm=ALGORITHM):
    return {
        "canonical_algorithm": algorithm,
        "stratification_label": "target_analyte",
        "group_label": "master_sample_id",
        "outer_repeat_seeds": list(seeds),
        "outer_folds_per_station": folds,
        "primary_domain_eligibility": {
            "requirements": {"test_classes": classes, "pooled_test_masters_minimum": masters},
            "domains": list(domains),
        },
    }


def _p02_contract(**overrides):
    contract = {"inner_master_folds": 3}
    contract.update(overrides)
    return contract


def _default_split_contract():
    return _split_contract(domains=["ST1:H-1"])


def _build(
    manifest,
    *,
    population_id=PID,
    population_sha256=DIGEST,
    split_contract=None,
    p02_contract=None,
):
    return build_population_plan(
        manifest=manifest,
        population_id=population_id,
        population_sha256=population_sha256,
        split_contract=_default_split_contract() if split_contract is None else split_contract,
        p02_contract=_p02_contract() if p02_contract is None else p02_contract,
    )


def _row(uid, mid, station, instrument, target):
    return {
        "observation_uid": uid,
        "master_sample_id": mid,
        "station": station,
        "instrument": instrument,
        "target_analyte": target,
    }


def _manifest(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "observation_uid",
            "master_sample_id",
            "station",
            "instrument",
            "target_analyte",
        ],
    )


def _rows_fallback(*, masters_per_class=12, numeric_ids=False):
    rows, counter = [], 0
    for ci in range(3):
        for mi in range(masters_per_class):
            counter += 1
            mid = str(counter) if numeric_ids else f"M{ci}-{mi:02d}"
            for instrument in ("H-1", "S-1", "S-2"):
                rows.append(_row(f"U-{mid}-{instrument}", mid, "ST1", instrument, f"T{ci}"))
    return rows


def _rows_pseudo(*, masters_per_class=18):
    cohorts = ("S-1", "S-2", "S-3")
    rows = []
    for ci in range(3):
        for mi in range(masters_per_class):
            mid = f"P{ci}-{mi:02d}"
            for instrument in ("H-1", cohorts[mi % 3]):
                rows.append(_row(f"U-{mid}-{instrument}", mid, "ST1", instrument, f"T{ci}"))
    return rows


def _rows_three_held():
    rows = []
    for ci in range(3):
        for mi in range(9):
            mid = f"H{ci}-{mi:02d}"
            for instrument in ("S-1", "S-2"):
                rows.append(_row(f"U-{mid}-{instrument}", mid, "ST1", instrument, f"T{ci}"))
        rows.append(_row(f"U-HELD-{ci}-H-1", f"HELD-{ci}", "ST1", "H-1", f"T{ci}"))
    return rows


def _rows_degenerate():
    rows = []
    for station, held in (("ST3", "H-3"), ("ST4", "H-4"), ("ST5", "H-5")):
        for ci in range(3):
            for mi in range(6):
                mid = f"{held}-{ci}-{mi:02d}"
                rows.append(_row(f"U-{mid}", mid, station, held, f"T{ci}"))
    for mi in range(6):  # ST3 source only carries class T0
        mid = f"ST3-S3-{mi:02d}"
        rows.append(_row(f"U-{mid}", mid, "ST3", "S-3", "T0"))
    for ci in range(3):  # ST4 deliberately has no source instrument
        for mi in range(2):  # ST5 source has too few masters per class
            mid = f"ST5-S5-{ci}-{mi}"
            rows.append(_row(f"U-{mid}", mid, "ST5", "S-5", f"T{ci}"))
    return rows


@pytest.fixture(scope="module")
def fallback_manifest():
    return _manifest(_rows_fallback())


@pytest.fixture(scope="module")
def pseudo_manifest():
    return _manifest(_rows_pseudo())


@pytest.fixture(scope="module")
def parity_manifest():
    return _manifest(_rows_fallback(numeric_ids=True))


def _uid_hash(values):
    return sha256_value(sorted({str(value) for value in values if value is not None}))


def _role_uids(snapshot, role):
    return set(snapshot.loc[snapshot["role"] == role, "observation_uid"])


def test_input_shuffle_is_deterministic_and_inputs_are_not_mutated(fallback_manifest):
    manifest = fallback_manifest.copy(deep=True)
    contract = _split_contract(domains=["ST1:H-1"])
    p02_contract = _p02_contract()
    before_manifest = manifest.copy(deep=True)
    before_contract = copy.deepcopy(contract)
    before_p02 = copy.deepcopy(p02_contract)

    first = _build(manifest, split_contract=contract, p02_contract=p02_contract)
    shuffled = manifest.sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = _build(
        shuffled,
        split_contract=copy.deepcopy(contract),
        p02_contract=copy.deepcopy(p02_contract),
    )

    for name in TABLE_NAMES:
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
    assert first.validation_report["plan_sha256"] == second.validation_report["plan_sha256"]
    assert first.validation_report["table_hashes"] == second.validation_report["table_hashes"]
    pd.testing.assert_frame_equal(manifest, before_manifest)
    assert contract == before_contract and p02_contract == before_p02


def test_row_filtering_keeps_inherited_outer_labels_but_changes_hashes(fallback_manifest):
    full = _build(fallback_manifest)
    suffix = fallback_manifest["master_sample_id"].str[-2:].astype(int)
    mask = (fallback_manifest["instrument"] == "S-2") & (suffix < 6)
    trimmed = _build(fallback_manifest[~mask].reset_index(drop=True))

    inherited = ["outer_repeat", "station", "outer_fold", "master_sample_id"]
    left = full.master_splits[inherited].sort_values(inherited).reset_index(drop=True)
    right = trimmed.master_splits[inherited].sort_values(inherited).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert (
        full.validation_report["metadata_canonical_sha256"]
        != trimmed.validation_report["metadata_canonical_sha256"]
    )
    assert set(full.context_registry["context_id"]) != set(trimmed.context_registry["context_id"])
    assert full.validation_report["plan_sha256"] != trimmed.validation_report["plan_sha256"]


def test_population_id_and_digest_independently_change_binding(fallback_manifest):
    base = _build(fallback_manifest)
    other_digest = _build(
        fallback_manifest, population_sha256=sha256_value({"population": "fixture-2"})
    )
    other_id = _build(fallback_manifest, population_id="POP-2")
    assert base.validation_report["plan_sha256"] != other_digest.validation_report["plan_sha256"]
    assert base.validation_report["plan_sha256"] != other_id.validation_report["plan_sha256"]
    assert set(base.context_registry["context_id"]).isdisjoint(
        other_id.context_registry["context_id"]
    )


def test_added_repeat_seed_changes_plan_binding(fallback_manifest):
    one = _build(fallback_manifest, split_contract=_split_contract(domains=["ST1:H-1"], seeds=[17]))
    two = _build(
        fallback_manifest, split_contract=_split_contract(domains=["ST1:H-1"], seeds=[17, 31])
    )
    assert (
        one.validation_report["split_contract_sha256"]
        != two.validation_report["split_contract_sha256"]
    )
    assert one.validation_report["plan_sha256"] != two.validation_report["plan_sha256"]
    assert len(two.context_registry) == 2 * len(one.context_registry)


def test_numeric_looking_master_ids_sort_lexicographically(parity_manifest):
    plan = _build(parity_manifest)
    assert {"1", "2", "10"} <= set(plan.master_splits["master_sample_id"])
    for _, group in plan.master_splits.groupby(["outer_repeat", "outer_fold"]):
        ids = list(group["master_sample_id"])
        assert ids == sorted(ids)
    assert sha256_value(sorted(["1", "2", "10"])) != sha256_value(sorted(["1", "2", "10"], key=int))


def test_canonical_set_hashing_distinguishes_newline_ids_and_rejects_nan():
    assert p08._hash_set(["a", "b"]) != p08._hash_set(["a\nb"])
    assert p08._hash_set(["b", "a"]) == sha256_value(sorted({"a", "b"}))
    bad_split = _default_split_contract()
    bad_split["nonfinite_identity"] = float("nan")
    with pytest.raises(ValueError):
        _build(_baseline_manifest(), split_contract=bad_split)


def test_role_membership_hashes_and_master_disjointness(fallback_manifest):
    plan = _build(fallback_manifest)
    roles = plan.role_registry
    assert not plan.unit_registry.empty
    assert set(roles["role"]) == FOUR_ROLES
    for row in plan.unit_registry.to_dict(orient="records"):
        snapshot = roles[
            (roles["context_id"] == row["context_id"])
            & (roles["purpose"] == row["purpose"])
            & (roles["unit_id"] == row["unit_id"])
        ]
        fit = snapshot[snapshot["role"] == "fit"]
        validation = snapshot[snapshot["role"] == "validation"]
        assert row["fit_observation_set_sha256"] == _uid_hash(fit["observation_uid"])
        assert row["validation_observation_set_sha256"] == _uid_hash(validation["observation_uid"])
        assert set(fit["master_sample_id"]).isdisjoint(set(validation["master_sample_id"]))

    for context in plan.context_registry.to_dict("records"):
        station, held = context["domain"].split(":")
        outer = roles[
            (roles["context_id"] == context["context_id"]) & (roles["purpose"] == "outer")
        ]
        source = outer[outer["role"] == "source"]
        test = outer[outer["role"] == "test"]
        assert len(source) == context["source_rows"] and len(test) == context["test_rows"]
        assert set(source["master_sample_id"]).isdisjoint(set(test["master_sample_id"]))
        assert held not in set(source["instrument"])
        assert context["source_observation_set_sha256"] == _uid_hash(source["observation_uid"])
        assert context["test_observation_set_sha256"] == _uid_hash(test["observation_uid"])

        split = plan.master_splits[plan.master_splits["outer_repeat"] == context["outer_repeat"]]
        in_fold = set(split.loc[split["outer_fold"] == context["outer_fold"], "master_sample_id"])
        rows = fallback_manifest[fallback_manifest["station"] == station]
        expected_test = set(
            rows.loc[
                rows["master_sample_id"].isin(in_fold) & (rows["instrument"] == held),
                "observation_uid",
            ]
        )
        expected_source = set(
            rows.loc[
                (~rows["master_sample_id"].isin(in_fold)) & (rows["instrument"] != held),
                "observation_uid",
            ]
        )
        assert _role_uids(outer, "test") == expected_test
        assert _role_uids(outer, "source") == expected_source


def test_t3_membership_is_exhaustive_and_context_bound(fallback_manifest):
    plan = _build(fallback_manifest)
    t3 = plan.t3_partitions
    assert not t3.empty
    for column in ("domain", "outer_repeat", "outer_fold", "observation_uid", "context_id"):
        assert column in t3.columns
    duplicated = t3.duplicated(subset=["domain", "outer_repeat", "outer_fold", "observation_uid"])
    assert not duplicated.any()
    assert set(t3["observation_uid"]) == set(fallback_manifest["observation_uid"])
    assert set(t3["context_id"]) == set(plan.context_registry["context_id"])
    for _, group in t3.groupby(["domain", "outer_repeat", "outer_fold"]):
        assert group["context_id"].nunique() == 1
    for context in plan.context_registry.to_dict("records"):
        station, held = context["domain"].split(":")
        split = plan.master_splits[plan.master_splits["outer_repeat"] == context["outer_repeat"]]
        in_fold = set(split.loc[split["outer_fold"] == context["outer_fold"], "master_sample_id"])
        rows = fallback_manifest[fallback_manifest["station"] == station]
        is_test = rows["master_sample_id"].isin(in_fold)
        is_held = rows["instrument"] == held
        expected_by_role = {
            "train_source": set(rows.loc[~is_test & ~is_held, "observation_uid"]),
            "test_target": set(rows.loc[is_test & is_held, "observation_uid"]),
            "excluded_train_target": set(rows.loc[~is_test & is_held, "observation_uid"]),
            "excluded_test_source": set(rows.loc[is_test & ~is_held, "observation_uid"]),
        }
        members = t3[
            (t3["domain"] == context["domain"])
            & (t3["outer_repeat"] == context["outer_repeat"])
            & (t3["outer_fold"] == context["outer_fold"])
        ]
        assert set(members["role"]) <= set(expected_by_role)
        for role, expected_uids in expected_by_role.items():
            role_members = members[members["role"] == role]
            assert set(role_members["observation_uid"]) == expected_uids


def test_absent_held_instrument_and_unknown_contexts_are_retained():
    rows = _rows_fallback(masters_per_class=12)
    for mi in range(6):
        rows.append(_row(f"U-ST2-{mi}", f"ST2-{mi}", "ST2", "S-9", "T0"))
    plan = _build(
        _manifest(rows),
        split_contract=_split_contract(domains=["ST1:H-1", "ST2:H-9"]),
    )
    registry = plan.domain_registry.set_index("domain")
    assert not bool(registry.loc["ST2:H-9", "eligible"])
    assert set(registry.loc["ST2:H-9", "reason_code"].split("|")) == {
        "no_observations",
        "class_support_insufficient",
        "master_support_insufficient",
    }
    st2 = plan.context_registry[plan.context_registry["domain"] == "ST2:H-9"]
    assert len(st2) == 4
    assert st2["selection_mode"].eq("not_applicable").all()
    assert not st2["metadata_ready"].any()
    assert plan.unit_registry[plan.unit_registry["context_id"].isin(st2["context_id"])].empty
    assert len(plan.context_registry) == 8


def test_all_ineligible_population_keeps_outer_roles_only(fallback_manifest):
    reference = _build(fallback_manifest)
    plan = _build(fallback_manifest, split_contract=_split_contract(domains=["ST1:H-1"], classes=5))
    assert not plan.domain_registry["eligible"].any()
    for name in ("inner_selection_registry", "inner_master_split_registry", "unit_registry"):
        empty = getattr(plan, name)
        assert empty.empty
        assert list(empty.columns) == list(getattr(reference, name).columns)
        expected = sha256_value({"columns": list(empty.columns), "records": []})
        assert plan.validation_report["table_hashes"][name] == expected
    roles = plan.role_registry
    assert not roles.empty
    assert set(roles["purpose"]) == {"outer"}
    assert set(roles["role"]) <= {"source", "test"}
    assert not plan.context_registry.empty
    assert plan.validation_report["counts"]["selection_unit_count"] == 0
    assert plan.validation_report["counts"]["calibration_unit_count"] == 0
    assert not plan.context_registry["selection_supported"].any()
    assert not plan.context_registry["calibration_supported"].any()


def test_pseudo_mode_uses_physically_disjoint_source_cohorts(pseudo_manifest):
    plan = _build(pseudo_manifest)
    assert "pseudo_domain" in set(plan.context_registry["selection_mode"])
    supported = plan.inner_selection_registry[plan.inner_selection_registry["supported"]]
    assert len(supported) >= 2
    assert supported["master_disjoint"].all()
    for context in plan.context_registry.to_dict("records"):
        subset = plan.unit_registry[plan.unit_registry["context_id"] == context["context_id"]]
        calibration = subset[subset["purpose"] == "calibration"]
        selection = subset[subset["purpose"] == "selection"]
        assert len(calibration) == (3 if context["calibration_supported"] else 0)
        assert len(selection) == (3 if context["selection_supported"] else 0)
    cohorts = pseudo_manifest[pseudo_manifest["instrument"].str.startswith("S-")]
    assert cohorts.groupby("master_sample_id")["instrument"].nunique().max() == 1


def test_overlapping_source_views_fall_back_to_master_cv(fallback_manifest):
    plan = _build(fallback_manifest)
    eligible = plan.context_registry[plan.context_registry["domain_eligible"]]
    assert set(eligible["selection_mode"]) == {"master_cv"}
    assert eligible["calibration_supported"].all()
    selection = plan.unit_registry[plan.unit_registry["purpose"] == "selection"]
    calibration = plan.unit_registry[plan.unit_registry["purpose"] == "calibration"]
    assert set(selection["unit_id"]) == set(calibration["unit_id"])
    merged = selection.merge(calibration, on=["context_id", "unit_id"], suffixes=("_s", "_c"))
    assert (merged["fit_observation_set_sha256_s"] == merged["fit_observation_set_sha256_c"]).all()
    same_validation = (
        merged["validation_observation_set_sha256_s"]
        == merged["validation_observation_set_sha256_c"]
    )
    assert same_validation.all()
    for context in plan.context_registry.to_dict("records"):
        subset = plan.unit_registry[plan.unit_registry["context_id"] == context["context_id"]]
        assert (subset["purpose"] == "selection").sum() == 3
        assert (subset["purpose"] == "calibration").sum() == 3


def test_three_held_masters_are_pooled_eligible_with_empty_and_sparse_folds():
    plan = _build(
        _manifest(_rows_three_held()),
        split_contract=_split_contract(domains=["ST1:H-1"], classes=3, masters=3),
    )
    domain = plan.domain_registry.iloc[0]
    assert bool(domain["eligible"]) and int(domain["observed_masters"]) == 3
    contexts = plan.context_registry
    assert len(contexts) == 4
    assert sorted(contexts["outer_fold"].tolist()) == [0, 1, 2, 3]
    assert contexts["empty_test_fold"].sum() >= 1
    assert contexts["sparse_test_fold"].sum() >= 1
    sparse = contexts[contexts["sparse_test_fold"] & ~contexts["empty_test_fold"]]
    reliable = sparse[
        sparse["source_has_all_station_classes"]
        & sparse["selection_supported"]
        & sparse["calibration_supported"]
    ]
    assert not reliable.empty
    assert reliable["metadata_ready"].all()


def test_source_empty_missing_class_and_low_inner_support_disable_metadata():
    plan = _build(
        _manifest(_rows_degenerate()),
        split_contract=_split_contract(
            domains=["ST3:H-3", "ST4:H-4", "ST5:H-5"], classes=3, masters=3
        ),
    )
    contexts = plan.context_registry
    by_domain = {domain: group for domain, group in contexts.groupby("domain")}

    empty = by_domain["ST4:H-4"]
    assert empty["source_rows"].eq(0).all()
    assert empty["reason_code"].str.contains("empty_source").all()
    empty_units = plan.unit_registry[plan.unit_registry["context_id"].isin(empty["context_id"])]
    assert empty_units.empty

    missing = by_domain["ST3:H-3"]
    assert not missing["source_has_all_station_classes"].any()
    assert missing["reason_code"].str.contains("source_missing_station_class").all()
    missing_units = plan.unit_registry
    supported = missing_units[
        missing_units["context_id"].isin(missing["context_id"]) & missing_units["support"]
    ]
    assert supported.empty

    low = by_domain["ST5:H-5"]
    assert low["selection_mode"].eq("unsupported").all()
    low_inner = plan.inner_master_split_registry
    assert low_inner[low_inner["context_id"].isin(low["context_id"])].empty
    low_units = plan.unit_registry[plan.unit_registry["context_id"].isin(low["context_id"])]
    assert low_units.empty
    assert not contexts["metadata_ready"].any()


def test_mocked_inner_splitter_retains_unsupported_calibration_unit(monkeypatch, fallback_manifest):
    class _SingleClassSplitter:
        def __init__(self, n_splits, shuffle, random_state):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def split(self, X, y, groups):
            labels = list(y)
            for fold in range(self.n_splits):
                test = [
                    index
                    for index, label in zip(X.index, labels, strict=True)
                    if label == f"T{fold}"
                ]
                yield [index for index in X.index if index not in set(test)], test

    monkeypatch.setattr(p08, "StratifiedGroupKFold", _SingleClassSplitter)
    plan = _build(fallback_manifest)
    calibration = plan.unit_registry[plan.unit_registry["purpose"] == "calibration"]
    assert not calibration.empty
    assert not calibration["support"].any()
    assert plan.context_registry["calibration_supported"].eq(False).all()
    assert plan.validation_report["counts"]["calibration_supported_context_count"] == 0
    assert plan.master_splits["outer_fold"].nunique() == 4


def _duplicate_column_manifest():
    frame = _manifest(_rows_fallback())
    return pd.concat([frame, frame[["station"]]], axis=1)


def _missing_column_manifest():
    return _manifest(_rows_fallback()).drop(columns=["target_analyte"])


def _missing_station_column_manifest():
    return _manifest(_rows_fallback()).drop(columns=["station"])


def _whitespace_column_manifest():
    return _manifest(_rows_fallback()).rename(columns={"instrument": " instrument"})


def _whitespace_value_manifest():
    frame = _manifest(_rows_fallback())
    frame.loc[0, "station"] = "   "
    return frame


def _null_metadata_manifest():
    frame = _manifest(_rows_fallback())
    frame.loc[0, "station"] = None
    return frame


def _blank_metadata_manifest():
    frame = _manifest(_rows_fallback())
    frame.loc[0, "master_sample_id"] = ""
    return frame


def _nan_metadata_manifest():
    frame = _manifest(_rows_fallback())
    frame["target_analyte"] = frame["target_analyte"].astype(object)
    frame.loc[0, "target_analyte"] = float("nan")
    return frame


def _nonstring_metadata_manifest():
    frame = _manifest(_rows_fallback())
    frame["target_analyte"] = frame["target_analyte"].astype(object)
    frame.loc[0, "target_analyte"] = 3
    return frame


def _duplicate_uid_manifest():
    frame = _manifest(_rows_fallback())
    frame.loc[1, "observation_uid"] = frame.loc[0, "observation_uid"]
    return frame


def _conflicting_master_manifest():
    frame = _manifest(_rows_fallback())
    frame.loc[0, "station"] = "ST2"
    return frame


def _conflicting_target_manifest():
    frame = _manifest(_rows_fallback())
    frame.loc[0, "target_analyte"] = "T1"
    return frame


def _baseline_manifest():
    return _manifest(_rows_fallback())


def _split_kwargs(**overrides):
    return {
        "manifest": _baseline_manifest(),
        "split_contract": _split_contract(domains=["ST1:H-1"], **overrides),
    }


def _p02_kwargs(**overrides):
    return {"manifest": _baseline_manifest(), "p02_contract": _p02_contract(**overrides)}


def _manifest_kwargs(factory):
    return {"manifest": factory()}


def _pseudo_support(**overrides):
    block = dict(VALID_PSEUDO_SUPPORT)
    block.update(overrides)
    return block


def _pseudo_missing_key():
    block = _pseudo_support()
    del block["validation_unit"]
    return _p02_kwargs(pseudo_domain_support=block)


_BAD_INPUT_CASES = {
    "bad-digest": lambda: {"manifest": _baseline_manifest(), "population_sha256": "not-a-digest"},
    "blank-token": lambda: {"manifest": _baseline_manifest(), "population_id": "bad token"},
    "non-dict-contract": lambda: {"manifest": _baseline_manifest(), "split_contract": []},
    "missing-column": lambda: _manifest_kwargs(_missing_column_manifest),
    "missing-station-column": lambda: _manifest_kwargs(_missing_station_column_manifest),
    "duplicate-column": lambda: _manifest_kwargs(_duplicate_column_manifest),
    "whitespace-column": lambda: _manifest_kwargs(_whitespace_column_manifest),
    "whitespace-value": lambda: _manifest_kwargs(_whitespace_value_manifest),
    "null-metadata": lambda: _manifest_kwargs(_null_metadata_manifest),
    "blank-metadata": lambda: _manifest_kwargs(_blank_metadata_manifest),
    "nan-metadata": lambda: _manifest_kwargs(_nan_metadata_manifest),
    "non-string-metadata": lambda: _manifest_kwargs(_nonstring_metadata_manifest),
    "duplicate-uid": lambda: _manifest_kwargs(_duplicate_uid_manifest),
    "conflicting-master": lambda: _manifest_kwargs(_conflicting_master_manifest),
    "conflicting-target": lambda: _manifest_kwargs(_conflicting_target_manifest),
    "bad-algorithm": lambda: _split_kwargs(algorithm="random"),
    "bool-folds": lambda: _split_kwargs(folds=True),
    "zero-folds": lambda: _split_kwargs(folds=0),
    "repeated-seeds": lambda: _split_kwargs(seeds=[17, 17]),
    "non-int-seed": lambda: _split_kwargs(seeds=[17, "x"]),
    "empty-domains": lambda: {
        "manifest": _baseline_manifest(),
        "split_contract": _split_contract(domains=[]),
    },
    "malformed-domain": lambda: {
        "manifest": _baseline_manifest(),
        "split_contract": _split_contract(domains=["ST1"]),
    },
    "duplicate-domain": lambda: {
        "manifest": _baseline_manifest(),
        "split_contract": _split_contract(domains=["ST1:H-1", "ST1:H-1"]),
    },
    "missing-requirements": lambda: {
        "manifest": _baseline_manifest(),
        "split_contract": {
            **_split_contract(domains=["ST1:H-1"]),
            "primary_domain_eligibility": {},
        },
    },
    "bad-strat-label": lambda: {
        "manifest": _baseline_manifest(),
        "split_contract": {
            **_split_contract(domains=["ST1:H-1"]),
            "stratification_label": "wrong",
        },
    },
    "bool-class-count": lambda: _split_kwargs(classes=True),
    "bool-master-count": lambda: _split_kwargs(masters=True),
    "bool-inner-folds": lambda: _p02_kwargs(inner_master_folds=True),
    "low-inner-folds": lambda: _p02_kwargs(inner_master_folds=1),
    "psc-validation-bool-false": lambda: _p02_kwargs(
        pseudo_domain_support=_pseudo_support(validation_requires_all_station_classes=False)
    ),
    "psc-remaining-bool-false": lambda: _p02_kwargs(
        pseudo_domain_support=_pseudo_support(remaining_training_requires_all_station_classes=False)
    ),
    "psc-validation-string": lambda: _p02_kwargs(
        pseudo_domain_support=_pseudo_support(validation_requires_all_station_classes="yes")
    ),
    "psc-remaining-float": lambda: _p02_kwargs(
        pseudo_domain_support=_pseudo_support(remaining_training_requires_all_station_classes=1.0)
    ),
    "psc-min-bool": lambda: _p02_kwargs(
        pseudo_domain_support=_pseudo_support(minimum_supported_pseudo_domains=True)
    ),
    "psc-min-float": lambda: _p02_kwargs(
        pseudo_domain_support=_pseudo_support(minimum_supported_pseudo_domains=2.0)
    ),
    "psc-wrong-fallback": lambda: _p02_kwargs(
        pseudo_domain_support=_pseudo_support(fallback="wrong fallback")
    ),
    "psc-wrong-unit": lambda: _p02_kwargs(
        pseudo_domain_support=_pseudo_support(validation_unit="wrong unit")
    ),
    "psc-unknown-key": lambda: _p02_kwargs(
        pseudo_domain_support=_pseudo_support(unexpected_key="x")
    ),
    "psc-missing-key": _pseudo_missing_key,
}


@pytest.mark.parametrize(
    "factory",
    list(_BAD_INPUT_CASES.values()),
    ids=list(_BAD_INPUT_CASES),
)
def test_malformed_inputs_are_rejected(factory):
    kwargs = factory()
    with pytest.raises((TypeError, ValueError)):
        _build(**kwargs)


def test_valid_pseudo_domain_support_is_accepted(pseudo_manifest):
    plan = _build(
        pseudo_manifest,
        p02_contract=_p02_contract(pseudo_domain_support=_pseudo_support()),
    )
    assert plan.validation_report["status"] == "metadata_plan_only"
    assert plan.validation_report["plan_sha256"]


def test_absent_pseudo_domain_support_block_is_allowed(fallback_manifest):
    plan = _build(fallback_manifest)
    assert plan.validation_report["plan_sha256"]


def test_domain_registry_reports_all_failed_eligibility_reasons():
    plan = _build(
        _baseline_manifest(),
        split_contract=_split_contract(domains=["ST1:H-1"], classes=5, masters=99),
    )
    reason = plan.domain_registry.iloc[0]["reason_code"]
    assert "class" in reason and "master" in reason


def test_validation_report_denies_scientific_operations(fallback_manifest):
    report = _build(fallback_manifest).validation_report
    assert report["status"] == "metadata_plan_only"
    assert report["scientific_operations"] == 0
    assert report["execution_authorized"] is False
    assert report["exact_fit_counts_enumerated"] is False
    assert {"predictions", "metrics", "scores", "fitted_model", "selected_recipe"}.isdisjoint(
        report
    )
    assert "model_fitting" in report["denied_operations"]


def test_table_hashes_bind_tables_and_role_identities_are_unique(fallback_manifest):
    plan = _build(fallback_manifest)
    for name in TABLE_NAMES:
        frame = getattr(plan, name)
        expected = sha256_value(
            {"columns": list(frame.columns), "records": frame.to_dict("records")}
        )
        assert plan.validation_report["table_hashes"][name] == expected
    assert not plan.role_registry.duplicated(
        subset=["context_id", "purpose", "unit_id", "role", "observation_uid"]
    ).any()


def test_extra_qc_and_sensor_columns_are_ignored(fallback_manifest):
    extra = fallback_manifest.copy()
    extra["qc_flag"] = "pass"
    extra["sensor_gain"] = [float(i) for i in range(len(extra))]
    base = _build(fallback_manifest)
    other = _build(extra)
    assert (
        base.validation_report["metadata_canonical_sha256"]
        == other.validation_report["metadata_canonical_sha256"]
    )
    pd.testing.assert_frame_equal(base.master_splits, other.master_splits)


def test_master_splits_adds_provenance_over_inherited_columns(fallback_manifest):
    plan = _build(fallback_manifest)
    columns = set(plan.master_splits.columns)
    assert {"population_id", "population_sha256"} <= columns
    assert {
        "outer_repeat",
        "outer_seed",
        "outer_fold",
        "station",
        "master_sample_id",
        "target_analyte",
    } <= columns
    assert plan.master_splits["population_id"].eq(PID).all()
    assert plan.master_splits["population_sha256"].eq(DIGEST).all()
    inherited_only = plan.master_splits.drop(columns=["population_id", "population_sha256"])
    assert not inherited_only.duplicated(
        subset=["outer_repeat", "station", "master_sample_id"]
    ).any()


_INNER_SUPPORT_COLUMNS = (
    "supported",
    "reason_code",
    "fit_rows",
    "validation_rows",
    "fit_observation_set_sha256",
    "validation_observation_set_sha256",
)
_INNER_ASSIGNMENT_COLUMNS = (
    "domain",
    "outer_repeat",
    "outer_fold",
    "inner_fold",
    "master_sample_id",
    "target_analyte",
    "selection_mode",
)


def _assert_inherited_inner_parity(plan):
    selection, assignments = p02.build_inner_selection(
        plan.t3_partitions, {"inner_master_folds": 3}
    )
    assert not selection.empty
    assert not assignments.empty

    pseudo_columns = [
        "domain",
        "outer_repeat",
        "outer_fold",
        "pseudo_instrument",
        "supported",
        "reason_code",
        "fit_rows",
        "validation_rows",
        "fit_observation_set_sha256",
        "validation_observation_set_sha256",
    ]
    pseudo_sort = ["domain", "outer_repeat", "outer_fold", "pseudo_instrument"]
    registry = plan.inner_selection_registry
    assert not registry.empty
    left = registry[pseudo_columns].sort_values(pseudo_sort).reset_index(drop=True)
    right = selection[pseudo_columns].sort_values(pseudo_sort).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)

    assignment_sort = [
        "domain",
        "outer_repeat",
        "outer_fold",
        "inner_fold",
        "master_sample_id",
    ]
    assignment_registry = plan.inner_master_split_registry
    assert not assignment_registry.empty
    left_assign = (
        assignment_registry[list(_INNER_ASSIGNMENT_COLUMNS)]
        .sort_values(assignment_sort)
        .reset_index(drop=True)
    )
    right_assign = (
        assignments[list(_INNER_ASSIGNMENT_COLUMNS)]
        .sort_values(assignment_sort)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(left_assign, right_assign, check_dtype=False)


def test_inherited_inner_selection_parity_fallback(fallback_manifest):
    assert callable(getattr(p02, "build_inner_selection", None))
    _assert_inherited_inner_parity(_build(fallback_manifest))


def test_inherited_inner_selection_parity_pseudo(pseudo_manifest):
    assert callable(getattr(p02, "build_inner_selection", None))
    _assert_inherited_inner_parity(_build(pseudo_manifest))


def _tamper_t3_role(frame):
    frame = frame.copy()
    assert "role" in frame.columns
    frame.loc[frame.index[0], "role"] = "unknown_role"
    return frame


def _tamper_t3_context(frame):
    frame = frame.copy()
    assert "partition_id" in frame.columns
    assert not frame.empty
    partition = frame["partition_id"].iloc[0]
    remaining = frame[frame["partition_id"] != partition].reset_index(drop=True)
    assert not remaining.empty
    assert set(remaining["partition_id"]) != set(frame["partition_id"])
    return remaining


def _tamper_t3_excluded(frame):
    frame = frame.copy()
    assert "role" in frame.columns
    excluded = frame.index[frame["role"].isin(["excluded_train_target", "excluded_test_source"])]
    assert len(excluded)
    return frame.drop(excluded[0]).reset_index(drop=True)


@pytest.mark.parametrize(
    "mutate",
    [_tamper_t3_role, _tamper_t3_context, _tamper_t3_excluded],
    ids=["unknown-role", "dropped-context", "dropped-excluded-row"],
)
def test_tampered_t3_partitions_are_rejected(monkeypatch, fallback_manifest, mutate):
    assert callable(getattr(p02, "build_t3_partitions", None))
    assert not _build(fallback_manifest).t3_partitions.empty
    real = p02.build_t3_partitions

    def wrapper(*args, **kwargs):
        return mutate(real(*args, **kwargs))

    monkeypatch.setattr(p08.p02, "build_t3_partitions", wrapper)
    with pytest.raises((TypeError, ValueError)):
        _build(fallback_manifest)


def test_absent_domain_station_is_rejected():
    rows = _rows_fallback(masters_per_class=6)
    split = _split_contract(domains=["ABSENT:H-1"])
    with pytest.raises(ValueError):
        _build(_manifest(rows), split_contract=split)
