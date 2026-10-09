"""Tests for p08_f07_data (invented, deterministic, no I/O)."""

import copy

import pandas as pd
import pytest

from atlas_sers.visualization import p08_f07_data as mod


def _preservation():
    rows = []
    for i in range(1, 18):
        st, ins, held = f"S{i:02d}", f"I{i:02d}", i <= 13
        for rep in mod.ACTION_BY_REPRESENTATION:
            for metric in mod.METRIC_ORDER:
                n_spectra, n_masters = 20 + i, 2 + (i % 5)
                finite, undefined, med, q10, q90 = n_spectra, 0, 0.5, 0.4, 0.6
                if i == 1 and rep == "R_MIN_400_1800" and metric == mod.DISPLACEMENT_METRIC:
                    finite, undefined = 0, n_spectra
                    med = q10 = q90 = float("nan")
                rows.append(
                    {
                        "station": st,
                        "instrument": ins,
                        "representation_id": rep,
                        "metric": metric,
                        "n_spectra": n_spectra,
                        "n_masters": n_masters,
                        "held_comparison_domain": held,
                        "finite_count": finite,
                        "undefined_count": undefined,
                        "median": med,
                        "q10": q10,
                        "q90": q90,
                    }
                )
    return pd.DataFrame(rows)


def _pairs():
    rows = []
    for i in range(1, 14):
        for estimand in mod.ESTIMAND_ORDER:
            for endpoint in mod.ENDPOINT_ORDER:
                for model in mod.MODEL_ORDER:
                    for policy in mod.POLICY_ORDER:
                        x, effect = 0.5 + i * 0.001, 0.05
                        if i == 1 and model == "D0-M":
                            effect = 0.0
                        elif i == 1 and model == "C-RBF-SVM":
                            effect = 0.2
                        elif i == 1 and model == "P05-SELECTED":
                            effect = -0.1
                        rows.append(
                            {
                                "estimand": estimand,
                                "contrast_id": "C",
                                "family_id": "F",
                                "endpoint": endpoint,
                                "model_id": model,
                                "policy_id": policy,
                                "domain": f"D{i:02d}",
                                "station": f"S{i:02d}",
                                "instrument": f"I{i:02d}",
                                "contexts": i + 1,
                                "unit_appearances": i + 2,
                                "physical_masters": i + 3,
                                "distinct_units": i + 4,
                                "x_balanced_accuracy": x,
                                "y_balanced_accuracy": x + effect,
                                "effect": effect,
                            }
                        )
    return rows


def _bundle(pairs=None):
    semantic = {"f02_pairs": _pairs() if pairs is None else pairs}
    return {
        "semantic": semantic,
        "semantic_sha256": mod._canonical_sha256(semantic),
        "manifest": {},
    }


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def test_prepare():
    bundle = _bundle()
    out = mod.prepare_f07(_preservation(), bundle)
    sem, manifest = out["semantic"], out["manifest"]
    assert manifest["counts"] == {
        "summaries": 561,
        "points": 1020,
        "held_points": 780,
        "exploratory_points": 240,
        "domains": 17,
        "held_domains": 13,
        "exploratory_domains": 4,
    }
    assert manifest["status"] == "prepared"
    assert manifest["reviewed"] is False and manifest["published"] is False
    assert len(sem["summaries"]) == 561 and len(sem["points"]) == 1020
    assert all(set(r) == set(mod.SUMMARY_FIELDS) for r in sem["summaries"])
    assert all(set(p) == set(mod.POINT_FIELDS) for p in sem["points"])
    assert sem["source_model_semantic_sha256"] == bundle["semantic_sha256"]
    held = [p for p in sem["points"] if p["held_comparison_domain"]]
    expl = [p for p in sem["points"] if not p["held_comparison_domain"]]
    assert len(held) == 780 and len(expl) == 240
    assert all(p["available"] is True and p["reason"] is None for p in held)
    assert all(
        p["available"] is False
        and p["balanced_accuracy"] is None
        and p["model_domain"] is None
        and p["reason"] == "outside_held_comparison"
        for p in expl
    )
    mins = [p for p in held if p["policy_id"] == "PP-U-MIN"]
    assert all(p["policy_minus_min"] == 0.0 for p in mins)
    assert all(p["balanced_accuracy"] == p["min_balanced_accuracy"] for p in mins)
    sg = {
        (p["station"], p["endpoint"], p["estimand"], p["model_id"]): p
        for p in held
        if p["policy_id"] == "PP-U-SG"
    }
    ar = {
        (p["station"], p["endpoint"], p["estimand"], p["model_id"]): p
        for p in held
        if p["policy_id"] == "PP-U-ARPLS"
    }
    assert all(
        abs(sg[k]["min_balanced_accuracy"] - ar[k]["min_balanced_accuracy"]) <= 1e-12 for k in sg
    )
    assert all(
        abs(
            ar[k]["policy_minus_min"]
            - (ar[k]["balanced_accuracy"] - ar[k]["min_balanced_accuracy"])
        )
        <= 1e-12
        for k in ar
    )
    d1 = [p for p in sem["points"] if p["station"] == "S01" and p["policy_id"] == "PP-U-MIN"]
    assert all(
        p["peak_displacement_median_cm1"] is None
        and p["peak_displacement_finite_count"] == 0
        and p["peak_displacement_undefined_count"] == 21
        for p in d1
    )
    assert all(p["peak_recall_median"] == 0.5 for p in d1)
    assert held[0]["original_n_spectra"] != held[0]["model_contexts"]
    assert out["semantic_sha256"] == mod._canonical_sha256(sem)
    assert mod.prepare_f07(_preservation(), bundle)["semantic_sha256"] == out["semantic_sha256"]


def test_refusals():
    preservation, bundle = _preservation(), _bundle()
    bad_hash = copy.deepcopy(bundle)
    bad_hash["semantic_sha256"] = "0" * 64
    assert _raises(lambda: mod.prepare_f07(preservation, bad_hash))
    assert _raises(lambda: mod.prepare_f07(preservation, _bundle(_pairs()[:-1])))
    assert _raises(
        lambda: mod.prepare_f07(preservation, _bundle(_pairs() + [copy.deepcopy(_pairs()[0])]))
    )
    unknown = _pairs()
    unknown[0]["model_id"] = "NOPE"
    assert _raises(lambda: mod.prepare_f07(preservation, _bundle(unknown)))
    mismatch = _pairs()
    for row in mismatch:
        if row["model_id"] == "D0-M" and row["policy_id"] == "PP-U-ARPLS":
            row["contexts"] += 1
    assert _raises(lambda: mod.prepare_f07(preservation, _bundle(mismatch)))
    contradictory = preservation.copy()
    contradictory.loc[0, "finite_count"] = 1
    contradictory.loc[0, "undefined_count"] = 20
    contradictory.loc[0, "median"] = float("nan")
    assert _raises(lambda: mod.prepare_f07(contradictory, bundle))
    bad_metric = preservation.copy()
    bad_metric.loc[0, "metric"] = "nope"
    assert _raises(lambda: mod.prepare_f07(bad_metric, bundle))


@pytest.mark.parametrize(
    "problem",
    [
        "missing_summary",
        "duplicate_summary",
        "infinity",
        "count_mismatch",
        "bad_order",
        "wrong_label",
        "reference_mismatch",
        "held_mismatch",
    ],
)
def test_additional_contract_refusals(problem):
    frame, pairs = _preservation(), _pairs()
    if problem == "missing_summary":
        frame = frame.iloc[:-1]
    elif problem == "duplicate_summary":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif problem == "infinity":
        frame.loc[0, "median"] = float("inf")
    elif problem == "count_mismatch":
        frame.loc[0, "n_spectra"] += 1
    elif problem == "bad_order":
        frame.loc[0, "q10"] = 1.0
    elif problem == "wrong_label":
        pairs[0]["station"] = {"unapproved": "nested"}
    elif problem == "reference_mismatch":
        pairs[0]["x_balanced_accuracy"] += 0.01
        pairs[0]["effect"] -= 0.01
    else:
        for row in pairs:
            if row["station"] == "S01":
                row["station"] = "S99"
    with pytest.raises(ValueError):
        mod.prepare_f07(frame, _bundle(pairs))


def test_shuffle_and_mutation_leave_frozen_semantics():
    frame, bundle = _preservation(), _bundle()
    out = mod.prepare_f07(frame, bundle)
    shuffled = mod.prepare_f07(
        frame.sample(frac=1, random_state=8), _bundle(list(reversed(_pairs())))
    )
    # Source hash changes with input order, but all scientific rows are identical.
    assert out["semantic"]["summaries"] == shuffled["semantic"]["summaries"]
    assert out["semantic"]["points"] == shuffled["semantic"]["points"]
    bundle["semantic"]["f02_pairs"][0]["y_balanced_accuracy"] = 0.0
    assert out["semantic_sha256"] == mod._canonical_sha256(out["semantic"])
    assert out["manifest"]["semantic_sha256"] == out["semantic_sha256"]


if __name__ == "__main__":
    test_prepare()
    test_refusals()
    print("ok")
