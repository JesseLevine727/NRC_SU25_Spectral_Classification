"""CPU synthetic tests for the pure P05 aggregation module."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p05_results as p05
from atlas_sers.evaluation.p04_results import endpoint_metrics, ensemble_seed_predictions

SEEDS = p05.SEEDS
CLASSES = ("A", "B", "C")
PERMIT = "a" * 64
DEV_BASE = {
    "d1": (0.6, 0.3, 0.1),
    "d2": (0.2, 0.7, 0.1),
    "d3": (0.2, 0.7, 0.1),
    "d4": (0.9, 0.05, 0.05),
    "d5": (0.1, 0.2, 0.7),
}
HELD_BASE = {"h1": (0.7, 0.2, 0.1), "h2": (0.1, 0.8, 0.1), "h3": (0.1, 0.8, 0.1)}
CONTEXT_META = {
    "ctx-dev": {"phase": "development", "held": "", "experiment_id": "P04-CORE-DEV"},
    "ctx-held": {"phase": "held_evaluation", "held": "INST-H", "experiment_id": "P04-CORE-T3"},
}
MANIFEST_ROWS = (
    ("d1", "m1", "INST-A", "A"),
    ("d2", "m2", "INST-A", "B"),
    ("d3", "m2", "INST-A", "B"),
    ("d4", "m2", "INST-B", "B"),
    ("d5", "m3", "INST-B", "C"),
    ("h1", "m4", "INST-H", "A"),
    ("h2", "m5", "INST-H", "B"),
    ("h3", "m5", "INST-H", "B"),
)


def _hash(value):
    payload = json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _probs(context, recipe, seed, uid):
    if context == "ctx-dev" and recipe == "D0-M" and uid == "d1":
        return {
            SEEDS[0]: (0.51, 0.49, 0.0),
            SEEDS[1]: (0.51, 0.49, 0.0),
            SEEDS[2]: (0.01, 0.99, 0.0),
        }[seed]
    base = DEV_BASE[uid] if context == "ctx-dev" else HELD_BASE[uid]
    return tuple(reversed(base)) if recipe == "D3" else base


def _refit_spec(context, recipe, seed, fitting_uids, calibration_slot_ids):
    identity = {
        "context_id": context,
        "fitting_role_id": f"{context}:{recipe}:{seed}",
        "source_uid_set_sha256": _hash(fitting_uids),
        "recipe_id": recipe,
        "seed": seed,
        "epochs": 30,
        "calibration_slot_ids": calibration_slot_ids,
        "permit_sha256": PERMIT,
    }
    refit_id = _hash(identity)
    return refit_id, {
        **identity,
        "refit_id": refit_id,
        "fitting_uids": fitting_uids,
        "classes": list(CLASSES),
    }


def _base(held_classes=("A", "B")):
    manifest = [
        {
            "observation_uid": u,
            "master_sample_id": m,
            "instrument": i,
            "target_analyte": t,
            "station": "STN1",
        }
        for u, m, i, t in MANIFEST_ROWS
    ]
    by_uid = {row["observation_uid"]: row for row in manifest}
    test_uids = {
        "ctx-dev": ["d1", "d2", "d3", "d4", "d5"],
        "ctx-held": [u for u in ("h1", "h2", "h3") if by_uid[u]["target_analyte"] in held_classes],
    }
    fitting = {"ctx-dev": ["ctx-dev-fit-1", "ctx-dev-fit-2"], "ctx-held": ["ctx-held-fit-1"]}
    calib = {"ctx-dev": ["ctx-dev-cal-1"], "ctx-held": ["ctx-held-cal-1"]}
    unique_refits, refit_ids = {}, {}
    for context in CONTEXT_META:
        for recipe in ("D0-M", "D3"):
            for seed in SEEDS:
                rid, spec = _refit_spec(context, recipe, seed, fitting[context], calib[context])
                unique_refits[rid] = spec
                refit_ids[(context, recipe, seed)] = rid
    endpoints, contexts = [], []
    for context, uids in test_uids.items():
        meta = CONTEXT_META[context]
        endpoints.append(
            {
                "context_id": context,
                "test_uids": list(uids),
                "test_masters": sorted({by_uid[u]["master_sample_id"] for u in uids}),
                "test_classes": sorted({by_uid[u]["target_analyte"] for u in uids}),
                "station": "STN1",
                "phase_gate": meta["phase"],
                "selection_mode": "nested",
                "held_instrument": meta["held"],
            }
        )
        contexts.append(
            {
                "context_id": context,
                "experiment_id": meta["experiment_id"],
                "domain": "DOM",
                "station": "STN1",
                "held_instrument": meta["held"],
                "outer_repeat": "0",
                "outer_fold": "0",
                "selection_mode": "nested",
                "phase_gate": meta["phase"],
                "outer_test_uid_sha256": _hash(sorted(uids)),
            }
        )
    aliases = []
    for context in CONTEXT_META:
        for strategy in p05.STRATEGIES:
            recipe = p05.STRATEGY_RECIPE.get(strategy, "D0-M")
            for seed in SEEDS:
                aliases.append(
                    {
                        "context_id": context,
                        "strategy": strategy,
                        "seed": seed,
                        "refit_id": refit_ids[(context, recipe, seed)],
                    }
                )
    alias_count = len(p05.STRATEGIES) * len(SEEDS) * len(endpoints)
    plan = {
        "schema_version": p05.PLAN_SCHEMA_VERSION,
        "protocol_version": p05.PLAN_PROTOCOL_VERSION,
        "permit_sha256": PERMIT,
        "unique_refits": unique_refits,
        "strategy_aliases": aliases,
        "endpoints": endpoints,
        "decisions": [{"context_id": c, "selected_recipe_id": "D0-M"} for c in CONTEXT_META],
        "counts": {
            "context_count": len(endpoints),
            "strategy_count": len(p05.STRATEGIES),
            "seed_count": len(SEEDS),
            "strategy_alias_count": len(aliases),
            "unique_refit_count": len(unique_refits),
            "endpoint_count": len(endpoints),
            "expected_strategy_alias_count": alias_count,
            "maximum_strategy_alias_count": p05.MAXIMUM_STRATEGY_ALIAS_COUNT,
        },
    }
    plan["plan_id"] = _hash({k: v for k, v in plan.items() if k != "plan_id"})
    predictions = {}
    for (context, recipe, seed), rid in refit_ids.items():
        rows = []
        for uid in test_uids[context]:
            p = _probs(context, recipe, seed, uid)
            rows.append(
                {
                    "observation_uid": uid,
                    "probability_0": p[0],
                    "probability_1": p[1],
                    "probability_2": p[2],
                }
            )
        predictions[rid] = pd.DataFrame(rows)
    return {
        "plan": plan,
        "contexts": contexts,
        "manifest": manifest,
        "predictions": predictions,
        "refit_ids": refit_ids,
    }


def _run(fx):
    return p05.aggregate_predictions(
        plan=fx["plan"],
        contexts=fx["contexts"],
        manifest=fx["manifest"],
        predictions=fx["predictions"],
    )


def _rehash(plan):
    plan["plan_id"] = _hash({k: v for k, v in plan.items() if k != "plan_id"})


def _find_refit(plan, context, recipe, seed):
    for rid, spec in plan["unique_refits"].items():
        if (spec["context_id"], spec["recipe_id"], spec["seed"]) == (context, recipe, seed):
            return rid
    raise KeyError((context, recipe, seed))


def _find_alias(plan, context, strategy, seed):
    for alias in plan["strategy_aliases"]:
        if (alias["context_id"], alias["strategy"], alias["seed"]) == (context, strategy, seed):
            return alias
    raise KeyError((context, strategy, seed))


def _touch_plan(fx, **changes):
    fx["plan"].update(changes)
    _rehash(fx["plan"])


def _touch_refit(fx, context, recipe, original_seed, **changes):
    fx["plan"]["unique_refits"][_find_refit(fx["plan"], context, recipe, original_seed)].update(
        changes
    )
    _rehash(fx["plan"])


def _touch_alias(fx, context, original_strategy, original_seed, **changes):
    _find_alias(fx["plan"], context, original_strategy, original_seed).update(changes)
    _rehash(fx["plan"])


def _touch_endpoint(fx, index, **changes):
    fx["plan"]["endpoints"][index].update(changes)
    _rehash(fx["plan"])


def _c_endpoint_dup(fx):
    fx["plan"]["endpoints"].append(copy.deepcopy(fx["plan"]["endpoints"][0]))
    _rehash(fx["plan"])


def _c_decision_context(fx):
    fx["plan"]["decisions"].pop()
    _rehash(fx["plan"])


def _c_alias_duplicate(fx):
    fx["plan"]["strategy_aliases"].append(copy.deepcopy(fx["plan"]["strategy_aliases"][0]))
    _rehash(fx["plan"])


def _c_alias_missing(fx):
    fx["plan"]["strategy_aliases"].pop()
    _rehash(fx["plan"])


def _c_missing_counts(fx):
    fx["plan"].pop("counts")
    _rehash(fx["plan"])


def _c_decision_recipe(fx):
    fx["plan"]["decisions"][0]["selected_recipe_id"] = "D9"
    _rehash(fx["plan"])


def _c_count_bool(fx):
    fx["plan"]["counts"]["context_count"] = True
    _rehash(fx["plan"])


def _c_refit_reference(fx):
    rid, spec = _refit_spec("ctx-dev", "D1", SEEDS[0], ["ctx-dev-fit-1"], ["ctx-dev-cal-1"])
    fx["plan"]["unique_refits"][rid] = spec
    _rehash(fx["plan"])


def _c_manifest_conflict(fx):
    fx["manifest"][1]["master_sample_id"] = "m1"
    fx["manifest"][1]["target_analyte"] = "C"


def _c_endpoint_uid_dup(fx):
    uid = fx["plan"]["endpoints"][0]["test_uids"][0]
    _touch_endpoint(fx, 0, test_uids=[uid, uid])


def _c_context_phase(fx):
    fx["contexts"][0]["phase_gate"] = "bogus"
    _touch_endpoint(fx, 0, phase_gate="bogus")


def _c_context_held(fx):
    fx["contexts"][0]["held_instrument"] = 1
    _touch_endpoint(fx, 0, held_instrument=1)


def _c_context_unsupported(fx):
    for spec in fx["plan"]["unique_refits"].values():
        if spec["context_id"] == "ctx-held":
            spec["classes"] = ["A", "X", "Y"]
    _rehash(fx["plan"])


def _c_pred_key(fx, extra):
    if extra:
        fx["predictions"]["extra"] = next(iter(fx["predictions"].values())).copy()
    else:
        fx["predictions"].pop(next(iter(fx["predictions"])))


def _c_pred_frame(fx, drop):
    key = next(iter(fx["predictions"]))
    frame = fx["predictions"][key]
    if drop == "type":
        fx["predictions"][key] = "nope"
    elif drop == "column":
        fx["predictions"][key] = frame.drop(columns=["probability_2"])
    elif drop == "unknown":
        frame.loc[frame.index[0], "observation_uid"] = "zzz"
    elif drop == "dup":
        frame.loc[frame.index[1], "observation_uid"] = frame.loc[frame.index[0], "observation_uid"]
    else:
        fx["predictions"][key] = frame.iloc[:-1].copy()


def _c_pred_value(fx, kind):
    frame = fx["predictions"][next(iter(fx["predictions"]))]
    if kind == "negative":
        frame.loc[0, "probability_0"] = -0.1
    elif kind == "nonfinite":
        frame.loc[0, "probability_0"] = np.inf
    else:
        frame.loc[0, ["probability_0", "probability_1", "probability_2"]] = 0.5


CASES = [
    ("schema", lambda fx: _touch_plan(fx, schema_version="bogus")),
    ("protocol", lambda fx: _touch_plan(fx, protocol_version="bogus")),
    ("plan_id", lambda fx: fx["plan"].__setitem__("plan_id", "0" * 64)),
    ("permit_malformed", lambda fx: _touch_plan(fx, permit_sha256="xyz")),
    ("plan_field_missing", _c_missing_counts),
    ("plan_count_bool", _c_count_bool),
    ("refits_empty", lambda fx: _touch_plan(fx, unique_refits={})),
    ("refit_id", lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], refit_id="0" * 64)),
    (
        "recipe_unregistered",
        lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], recipe_id="D9"),
    ),
    ("seed_unregistered", lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], seed=123)),
    ("epoch_29", lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], epochs=29)),
    ("epoch_bool", lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], epochs=True)),
    (
        "calib_dups",
        lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], calibration_slot_ids=["c", "c"]),
    ),
    (
        "source_ids_not_str",
        lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], fitting_uids=[1]),
    ),
    (
        "source_uid_mismatch",
        lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], fitting_uids=["zzz"]),
    ),
    (
        "identity_tamper",
        lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], fitting_role_id="tampered"),
    ),
    (
        "refit_permit_mismatch",
        lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], permit_sha256="b" * 64),
    ),
    (
        "class_order_diff",
        lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], classes=["A", "B", "D"]),
    ),
    (
        "class_vocab_bad",
        lambda fx: _touch_refit(fx, "ctx-dev", "D0-M", SEEDS[0], classes=["B", "A", "C"]),
    ),
    ("endpoint_duplicate", _c_endpoint_dup),
    ("decision_recipe", _c_decision_recipe),
    ("decision_context", _c_decision_context),
    ("alias_strategy", lambda fx: _touch_alias(fx, "ctx-dev", "D0-M", SEEDS[0], strategy="bogus")),
    ("alias_seed", lambda fx: _touch_alias(fx, "ctx-dev", "D0-M", SEEDS[0], seed=123)),
    ("alias_duplicate", _c_alias_duplicate),
    ("alias_missing", _c_alias_missing),
    (
        "alias_recipe_d0",
        lambda fx: _touch_alias(
            fx, "ctx-dev", "D0-M", SEEDS[0], refit_id=fx["refit_ids"][("ctx-dev", "D3", SEEDS[0])]
        ),
    ),
    (
        "alias_recipe_selected",
        lambda fx: _touch_alias(
            fx,
            "ctx-dev",
            "P05-SELECTED",
            SEEDS[0],
            refit_id=fx["refit_ids"][("ctx-dev", "D3", SEEDS[0])],
        ),
    ),
    (
        "alias_refit",
        lambda fx: _touch_alias(
            fx,
            "ctx-dev",
            "D0-M",
            SEEDS[0],
            refit_id=fx["refit_ids"][("ctx-held", "D0-M", SEEDS[0])],
        ),
    ),
    ("alias_context", lambda fx: _touch_alias(fx, "ctx-dev", "D0-M", SEEDS[0], context_id="nope")),
    ("refit_reference", _c_refit_reference),
    (
        "plan_count",
        lambda fx: _touch_plan(
            fx,
            **{
                "counts": {
                    **fx["plan"]["counts"],
                    "unique_refit_count": fx["plan"]["counts"]["unique_refit_count"] + 1,
                }
            },
        ),
    ),
    ("manifest_dup", lambda fx: fx["manifest"].append(copy.deepcopy(fx["manifest"][0]))),
    ("manifest_field", lambda fx: fx["manifest"][0].pop("station")),
    ("manifest_conflict", _c_manifest_conflict),
    ("manifest_empty", lambda fx: fx["manifest"].clear()),
    ("endpoint_uids_empty", lambda fx: _touch_endpoint(fx, 0, test_uids=[])),
    ("endpoint_uid_dup", _c_endpoint_uid_dup),
    ("endpoint_uid_missing", lambda fx: _touch_endpoint(fx, 0, test_uids=["zzz"])),
    ("endpoint_masters", lambda fx: _touch_endpoint(fx, 0, test_masters=["zzz"])),
    ("endpoint_classes", lambda fx: _touch_endpoint(fx, 0, test_classes=["X", "Y", "Z"])),
    ("context_dup", lambda fx: fx["contexts"].append(copy.deepcopy(fx["contexts"][0]))),
    ("context_field", lambda fx: fx["contexts"][0].pop("domain")),
    ("context_missing", lambda fx: fx["contexts"].pop()),
    ("context_endpoint_field", lambda fx: fx["contexts"][0].__setitem__("station", "OTHER")),
    ("context_phase", _c_context_phase),
    ("context_held_type", _c_context_held),
    ("context_hash", lambda fx: fx["contexts"][0].__setitem__("outer_test_uid_sha256", "0" * 64)),
    ("context_repeat_bool", lambda fx: fx["contexts"][0].__setitem__("outer_repeat", True)),
    ("context_fold_float", lambda fx: fx["contexts"][0].__setitem__("outer_fold", 1.5)),
    ("context_fold_nan", lambda fx: fx["contexts"][0].__setitem__("outer_fold", float("nan"))),
    ("context_unsupported", _c_context_unsupported),
    ("pred_extra", lambda fx: _c_pred_key(fx, True)),
    ("pred_missing", lambda fx: _c_pred_key(fx, False)),
    ("pred_type", lambda fx: _c_pred_frame(fx, "type")),
    ("pred_column", lambda fx: _c_pred_frame(fx, "column")),
    ("pred_uid_unknown", lambda fx: _c_pred_frame(fx, "unknown")),
    ("pred_uid_dup", lambda fx: _c_pred_frame(fx, "dup")),
    ("pred_uid_missing", lambda fx: _c_pred_frame(fx, "short")),
    ("pred_negative", lambda fx: _c_pred_value(fx, "negative")),
    ("pred_nonfinite", lambda fx: _c_pred_value(fx, "nonfinite")),
    ("pred_not_normalized", lambda fx: _c_pred_value(fx, "flat")),
    ("pred_station", lambda fx: fx["manifest"][0].__setitem__("station", "OTHER")),
    ("pred_held", lambda fx: fx["manifest"][5].__setitem__("instrument", "INST-X")),
]


@pytest.mark.parametrize("mutate", [case for _, case in CASES], ids=[name for name, _ in CASES])
def test_rejections(mutate):
    fx = _base()
    mutate(fx)
    with pytest.raises(p05.P05ResultsError):
        _run(fx)


def test_valid_fixture_outputs_and_selected_alias():
    out = _run(_base())
    assert set(out) == {
        "seed_predictions",
        "ensemble_predictions",
        "spectrum_metrics",
        "master_metrics",
        "coverage",
    }
    for table in out.values():
        assert set(table["experiment_id"]) == {"P05-CORE-DEV", "P05-CORE-T3"}
        assert (table["protocol_version"] == p05.RESULTS_PROTOCOL_VERSION).all()
        assert "source_context_experiment_id" in table.columns
    seed = out["seed_predictions"]
    cols = [c for c in seed.columns if c != "model_id"]
    order = ["context_id", "observation_uid"]
    d0 = seed[seed["model_id"].eq("D0-M")].sort_values(order)[cols].reset_index(drop=True)
    selected = (
        seed[seed["model_id"].eq("P05-SELECTED")].sort_values(order)[cols].reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(d0, selected)
    cov = out["coverage"]
    assert set(cov["status"]) == {"pass"}
    assert set(cov["strategy"]) == set(p05.STRATEGIES)


def test_ensemble_uses_probability_mean_not_majority_vote():
    out = _run(_base())
    ens = out["ensemble_predictions"]
    row = ens[ens["model_id"].eq("D0-M") & ens["observation_uid"].eq("d1")].iloc[0]
    assert row["probability_0"] == pytest.approx(1.03 / 3)
    assert row["probability_1"] == pytest.approx(1.97 / 3)
    assert row["probability_2"] == pytest.approx(0.0)
    assert row["probability_1"] > row["probability_0"]
    votes = out["seed_predictions"]
    votes = votes[votes["model_id"].eq("D0-M") & votes["observation_uid"].eq("d1")]
    winners = votes[list(p05.PROBABILITY_COLUMNS)].to_numpy().argmax(axis=1)
    assert np.bincount(winners, minlength=3).argmax() == 0
    subset = out["seed_predictions"][out["seed_predictions"]["model_id"].eq("D0-M")].reset_index(
        drop=True
    )
    helper = ensemble_seed_predictions(subset)
    keys = ["context_id", "observation_uid"]
    merged = helper[keys + list(p05.PROBABILITY_COLUMNS)].merge(
        ens[ens["model_id"].eq("D0-M")][keys + list(p05.PROBABILITY_COLUMNS)],
        on=keys,
        suffixes=("_h", "_p"),
    )
    for column in p05.PROBABILITY_COLUMNS:
        assert np.allclose(merged[f"{column}_h"], merged[f"{column}_p"])


@pytest.mark.parametrize(
    ("held_classes", "macro_f1"), [(("A", "B"), 2.0 / 3.0), (("A",), 1.0 / 3.0)]
)
def test_absent_class_support_preserved(held_classes, macro_f1):
    out = _run(_base(held_classes))
    table = out["spectrum_metrics"]
    row = table[table["model_id"].eq("D0-M") & table["context_id"].eq("ctx-held")].iloc[0]
    assert row["balanced_accuracy"] == pytest.approx(1.0)
    assert row["macro_f1"] == pytest.approx(macro_f1)
    assert json.loads(row["per_class_support"]) == {
        "A": 1,
        "B": 2 if "B" in held_classes else 0,
        "C": 0,
    }
    assert row["observed_class_count"] == len(held_classes)
    assert json.loads(row["missing_classes"]) == [c for c in CLASSES if c not in held_classes]
    master = out["master_metrics"]
    master_row = master[master["model_id"].eq("D0-M") & master["context_id"].eq("ctx-held")].iloc[0]
    assert json.loads(master_row["per_class_support"]) == {
        "A": 1,
        "B": int("B" in held_classes),
        "C": 0,
    }


def test_master_metrics_differ_from_row_average():
    out = _run(_base())
    spectrum, master = out["spectrum_metrics"], out["master_metrics"]
    row = spectrum[spectrum["model_id"].eq("D0-M") & spectrum["context_id"].eq("ctx-dev")].iloc[0]
    cell = master[master["model_id"].eq("D0-M") & master["context_id"].eq("ctx-dev")].iloc[0]
    row_acc = row["balanced_accuracy"]
    master_acc = cell["balanced_accuracy"]
    assert row_acc == pytest.approx(5.0 / 9.0)
    assert master_acc == pytest.approx(1.0 / 3.0)
    assert row_acc != master_acc
    # Master m2: equal instruments give P(B)=(0.7+0.05)/2=0.375.
    # Averaging its three rows instead gives (0.7+0.7+0.05)/3, a different decision.
    expected_nll = -np.log([1.03 / 3, 0.375, 0.7]).mean()
    assert cell["negative_log_likelihood"] == pytest.approx(expected_nll)


def test_metrics_match_endpoint_metrics_helper():
    out = _run(_base())
    d0 = out["ensemble_predictions"]
    d0 = d0[d0["model_id"].eq("D0-M")].reset_index(drop=True)
    helper_spectrum, helper_master = endpoint_metrics(d0)
    for helper, produced in (
        (helper_spectrum, out["spectrum_metrics"]),
        (helper_master, out["master_metrics"]),
    ):
        produced = produced[produced["model_id"].eq("D0-M")]
        keys = [c for c in p05.GROUP_COLUMNS if c in helper.columns and c in produced.columns]
        metrics = [
            c
            for c in helper.columns
            if c not in keys and c in produced.columns and pd.api.types.is_numeric_dtype(helper[c])
        ]
        merged = helper[keys + metrics].merge(
            produced[keys + metrics], on=keys, suffixes=("_helper", "_produced")
        )
        for column in metrics:
            assert np.allclose(
                merged[f"{column}_helper"], merged[f"{column}_produced"], equal_nan=True
            )
