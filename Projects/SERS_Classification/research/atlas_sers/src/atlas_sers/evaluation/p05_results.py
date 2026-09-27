"""Pure in-memory aggregation of calibrated P05 refit predictions.

This module never reads a file, touches torch, fits, infers or selects. The
output ``experiment_id`` names the P05 evaluation family
(``P05-CORE-DEV`` or ``P05-CORE-T3``); the parent split-registry coordinate is
retained as ``source_context_experiment_id``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p04_results import endpoint_metrics, ensemble_seed_predictions

__all__ = ["P05ResultsError", "aggregate_predictions"]

RESULTS_PROTOCOL_VERSION = "nato-sers-p05-core-results-v1"
PLAN_SCHEMA_VERSION = "nato-sers-p05-refit-plan-v1"
PLAN_PROTOCOL_VERSION = "nato-sers-p05-core-20260925-v1"
SEEDS = (20260805, 20260817, 20260829)
STRATEGIES = ("D0-M", "P05-SELECTED", "D3")
STRATEGY_RECIPE = {"D0-M": "D0-M", "D3": "D3"}
REGISTERED_RECIPES = frozenset({"D0-M", "D1", "D2", "D3"})
PHASE_GATE_EXPERIMENT = {"development": "P05-CORE-DEV", "held_evaluation": "P05-CORE-T3"}
HELD_INSTRUMENT_SENTINELS = frozenset({"", "not_applicable"})
PROBABILITY_COLUMNS = ("probability_0", "probability_1", "probability_2")
GROUP_COLUMNS = (
    "context_id",
    "experiment_id",
    "domain",
    "held_instrument",
    "outer_repeat",
    "outer_fold",
    "station",
    "candidate_id",
)
CONTEXT_FIELDS = (
    "context_id",
    "experiment_id",
    "domain",
    "station",
    "held_instrument",
    "outer_repeat",
    "outer_fold",
    "selection_mode",
    "phase_gate",
    "outer_test_uid_sha256",
)
MANIFEST_FIELDS = (
    "observation_uid",
    "master_sample_id",
    "instrument",
    "target_analyte",
    "station",
)
HEX = frozenset("0123456789abcdef")
MAXIMUM_STRATEGY_ALIAS_COUNT = 2880


class P05ResultsError(ValueError):
    """Raised when the plan, contexts, manifest or predictions are invalid."""


def _hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _seq(value: Any, code: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise P05ResultsError(code)
    return list(value)


def _map(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P05ResultsError(code)
    return value


def _text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise P05ResultsError(code)
    return value


def _int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise P05ResultsError(code)
    return value


def _count(value: Any, code: str) -> int:
    if isinstance(value, bool):
        raise P05ResultsError(code)
    if isinstance(value, int):
        if value < 0:
            raise P05ResultsError(code)
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        if len(value) > 1 and value.startswith("0"):
            raise P05ResultsError(code)
        return int(value)
    raise P05ResultsError(code)


def _text_list(value: Any, code: str) -> list[str]:
    return [_text(item, code) for item in _seq(value, code)]


def _classes(value: Any, code: str = "class_vocabulary_invalid") -> tuple[str, ...]:
    parsed = json.loads(value) if isinstance(value, str) else value
    classes = tuple(_text(item, code) for item in _seq(parsed, code))
    if len(classes) != 3 or len(set(classes)) != 3 or classes != tuple(sorted(classes)):
        raise P05ResultsError(code)
    return classes


def _validate_plan(plan: Any) -> Mapping[str, Any]:
    plan = _map(plan, "plan_malformed")
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise P05ResultsError("plan_schema_mismatch")
    if plan.get("protocol_version") != PLAN_PROTOCOL_VERSION:
        raise P05ResultsError("plan_protocol_mismatch")
    content = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan.get("plan_id") != _hash(content):
        raise P05ResultsError("plan_id_mismatch")
    permit = _text(plan.get("permit_sha256"), "permit_sha256_malformed")
    if len(permit) != 64 or not set(permit) <= HEX:
        raise P05ResultsError("permit_sha256_malformed")
    for field in ("unique_refits", "strategy_aliases", "endpoints", "decisions", "counts"):
        if field not in plan:
            raise P05ResultsError("plan_field_missing")
    return plan


def _refits(plan: Mapping[str, Any], permit: str):
    refits = _map(plan["unique_refits"], "unique_refits_malformed")
    if not refits:
        raise P05ResultsError("unique_refits_empty")
    indexed: dict[str, dict[str, Any]] = {}
    context_classes: dict[str, tuple[str, ...]] = {}
    for refit_id, raw in refits.items():
        spec = _map(raw, "refit_malformed")
        stored_id = _text(spec.get("refit_id"), "refit_id_malformed")
        if _text(refit_id, "refit_id_malformed") != stored_id:
            raise P05ResultsError("refit_id_mismatch")
        context_id = _text(spec.get("context_id"), "refit_context_malformed")
        recipe = _text(spec.get("recipe_id"), "refit_recipe_malformed")
        if recipe not in REGISTERED_RECIPES:
            raise P05ResultsError("refit_recipe_unregistered")
        seed = _int(spec.get("seed"), "refit_seed_malformed")
        if seed not in SEEDS:
            raise P05ResultsError("refit_seed_unregistered")
        epochs = _int(spec.get("epochs"), "refit_epochs_malformed")
        if not 30 <= epochs <= 200:
            raise P05ResultsError("refit_epochs_malformed")
        if spec.get("permit_sha256") != permit:
            raise P05ResultsError("refit_permit_mismatch")
        calibration_slot_ids = _text_list(
            spec.get("calibration_slot_ids"), "refit_calibration_malformed"
        )
        if not calibration_slot_ids or calibration_slot_ids != sorted(set(calibration_slot_ids)):
            raise P05ResultsError("refit_calibration_malformed")
        fitting_uids = _text_list(spec.get("fitting_uids"), "refit_fitting_uids_malformed")
        if not fitting_uids or fitting_uids != sorted(set(fitting_uids)):
            raise P05ResultsError("refit_fitting_uids_malformed")
        if _hash(fitting_uids) != spec.get("source_uid_set_sha256"):
            raise P05ResultsError("refit_source_uid_mismatch")
        identity = {
            "context_id": context_id,
            "fitting_role_id": _text(spec.get("fitting_role_id"), "refit_role_malformed"),
            "source_uid_set_sha256": _text(
                spec.get("source_uid_set_sha256"), "refit_source_uid_malformed"
            ),
            "recipe_id": recipe,
            "seed": seed,
            "epochs": epochs,
            "calibration_slot_ids": calibration_slot_ids,
            "permit_sha256": permit,
        }
        if _hash(identity) != refit_id:
            raise P05ResultsError("refit_identity_mismatch")
        classes = _classes(spec.get("classes"), "refit_class_vocabulary_invalid")
        previous = context_classes.get(context_id)
        if previous is None:
            context_classes[context_id] = classes
        elif previous != classes:
            raise P05ResultsError("context_class_order_mismatch")
        indexed[refit_id] = {
            "context_id": context_id,
            "recipe_id": recipe,
            "seed": seed,
            "classes": classes,
        }
    return indexed, context_classes


def _index(plan: Mapping[str, Any], refits: Mapping[str, Any]):
    endpoints: dict[str, Any] = {}
    for raw in _seq(plan["endpoints"], "endpoints_malformed"):
        endpoint = _map(raw, "endpoint_malformed")
        context_id = _text(endpoint.get("context_id"), "endpoint_context_malformed")
        if context_id in endpoints:
            raise P05ResultsError("endpoint_duplicate")
        endpoints[context_id] = endpoint
    if not endpoints:
        raise P05ResultsError("endpoints_empty")
    decisions: dict[str, str] = {}
    for raw in _seq(plan["decisions"], "decisions_malformed"):
        decision = _map(raw, "decision_malformed")
        context_id = _text(decision.get("context_id"), "decision_context_malformed")
        selected = _text(decision.get("selected_recipe_id"), "decision_recipe_malformed")
        if selected not in REGISTERED_RECIPES:
            raise P05ResultsError("decision_recipe_unregistered")
        if context_id in decisions:
            raise P05ResultsError("decision_duplicate")
        decisions[context_id] = selected
    if set(decisions) != set(endpoints):
        raise P05ResultsError("decision_context_mismatch")
    aliases: dict[tuple[str, str, int], str] = {}
    for raw in _seq(plan["strategy_aliases"], "aliases_malformed"):
        alias = _map(raw, "alias_malformed")
        context_id = _text(alias.get("context_id"), "alias_context_malformed")
        strategy = _text(alias.get("strategy"), "alias_strategy_malformed")
        seed = _int(alias.get("seed"), "alias_seed_malformed")
        refit_id = _text(alias.get("refit_id"), "alias_refit_malformed")
        if strategy not in STRATEGIES or seed not in SEEDS:
            raise P05ResultsError("alias_invalid")
        if context_id not in endpoints:
            raise P05ResultsError("alias_context_unknown")
        key = (context_id, strategy, seed)
        if key in aliases:
            raise P05ResultsError("alias_duplicate")
        spec = refits.get(refit_id)
        if spec is None or spec["context_id"] != context_id or spec["seed"] != seed:
            raise P05ResultsError("alias_refit_mismatch")
        if spec["recipe_id"] != STRATEGY_RECIPE.get(strategy, decisions[context_id]):
            raise P05ResultsError("alias_recipe_mismatch")
        aliases[key] = refit_id
    expected_aliases = len(STRATEGIES) * len(SEEDS) * len(endpoints)
    if len(aliases) != expected_aliases or len(aliases) > MAXIMUM_STRATEGY_ALIAS_COUNT:
        raise P05ResultsError("alias_count_mismatch")
    for context_id in endpoints:
        for strategy in STRATEGIES:
            for seed in SEEDS:
                if (context_id, strategy, seed) not in aliases:
                    raise P05ResultsError("alias_missing")
    if set(aliases.values()) != set(refits):
        raise P05ResultsError("refit_reference_mismatch")
    counts = _map(plan["counts"], "counts_malformed")
    expected_counts = {
        "context_count": len(endpoints),
        "strategy_count": len(STRATEGIES),
        "seed_count": len(SEEDS),
        "strategy_alias_count": len(aliases),
        "unique_refit_count": len(refits),
        "endpoint_count": len(endpoints),
        "expected_strategy_alias_count": expected_aliases,
        "maximum_strategy_alias_count": MAXIMUM_STRATEGY_ALIAS_COUNT,
    }
    for key, value in expected_counts.items():
        if _int(counts.get(key), "plan_count_malformed") != value:
            raise P05ResultsError("plan_count_mismatch")
    return endpoints, aliases


def _manifest(manifest: Any) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    master_targets: dict[str, str] = {}
    for raw in _seq(manifest, "manifest_malformed"):
        record = _map(raw, "manifest_record_malformed")
        for field in MANIFEST_FIELDS:
            if field not in record:
                raise P05ResultsError("manifest_field_missing")
        uid = _text(record["observation_uid"], "manifest_uid_malformed")
        if uid in index:
            raise P05ResultsError("manifest_uid_duplicate")
        entry = {
            "master_sample_id": _text(record["master_sample_id"], "manifest_master_malformed"),
            "instrument": _text(record["instrument"], "manifest_instrument_malformed"),
            "target_analyte": _text(record["target_analyte"], "manifest_target_malformed"),
            "station": _text(record["station"], "manifest_station_malformed"),
        }
        previous = master_targets.get(entry["master_sample_id"])
        if previous is None:
            master_targets[entry["master_sample_id"]] = entry["target_analyte"]
        elif previous != entry["target_analyte"]:
            raise P05ResultsError("manifest_master_label_conflict")
        index[uid] = entry
    if not index:
        raise P05ResultsError("manifest_empty")
    return index


def _endpoints(
    endpoints: Mapping[str, Any], manifest: Mapping[str, Mapping[str, str]]
) -> dict[str, list[str]]:
    test_uids: dict[str, list[str]] = {}
    for context_id, endpoint in endpoints.items():
        raw = _seq(endpoint.get("test_uids"), "endpoint_test_uids_malformed")
        if not raw:
            raise P05ResultsError("endpoint_test_uids_empty")
        texts = _text_list(raw, "endpoint_test_uid_malformed")
        if len(set(texts)) != len(texts):
            raise P05ResultsError("endpoint_test_uid_duplicate")
        if any(uid not in manifest for uid in texts):
            raise P05ResultsError("endpoint_uid_missing_from_manifest")
        masters = sorted({manifest[uid]["master_sample_id"] for uid in texts})
        if _text_list(endpoint.get("test_masters"), "endpoint_masters_malformed") != masters:
            raise P05ResultsError("endpoint_test_masters_mismatch")
        classes = sorted({manifest[uid]["target_analyte"] for uid in texts})
        if _text_list(endpoint.get("test_classes"), "endpoint_classes_malformed") != classes:
            raise P05ResultsError("endpoint_test_classes_mismatch")
        test_uids[context_id] = texts
    return test_uids


def _contexts(contexts: Any, endpoints, test_uids, context_classes):
    raw_index: dict[str, Any] = {}
    for raw in _seq(contexts, "contexts_malformed"):
        record = _map(raw, "context_malformed")
        for field in CONTEXT_FIELDS:
            if field not in record:
                raise P05ResultsError("context_field_missing")
        context_id = _text(record["context_id"], "context_id_malformed")
        if context_id in raw_index:
            raise P05ResultsError("context_duplicate")
        raw_index[context_id] = record
    if set(raw_index) != set(endpoints):
        raise P05ResultsError("context_endpoint_mismatch")
    prepared: dict[str, Any] = {}
    for context_id, record in raw_index.items():
        endpoint = endpoints[context_id]
        for field in ("station", "phase_gate", "selection_mode", "held_instrument"):
            if record[field] != endpoint.get(field):
                raise P05ResultsError("context_endpoint_field_mismatch")
        phase_gate = _text(record["phase_gate"], "context_phase_gate_malformed")
        output_experiment = PHASE_GATE_EXPERIMENT.get(phase_gate)
        if output_experiment is None:
            raise P05ResultsError("context_phase_gate_unknown")
        held = record["held_instrument"]
        if not isinstance(held, str):
            raise P05ResultsError("context_held_instrument_malformed")
        fit_classes = context_classes.get(context_id)
        if fit_classes is None:
            raise P05ResultsError("context_refits_missing")
        test_classes = set(_text_list(endpoint.get("test_classes"), "endpoint_classes_malformed"))
        if not test_classes <= set(fit_classes):
            raise P05ResultsError("context_test_class_unsupported")
        if record["outer_test_uid_sha256"] != _hash(sorted(test_uids[context_id])):
            raise P05ResultsError("context_test_uid_sha256_mismatch")
        prepared[context_id] = {
            "context_id": context_id,
            "experiment_id": output_experiment,
            "source_context_experiment_id": _text(
                record["experiment_id"], "context_experiment_malformed"
            ),
            "domain": _text(record["domain"], "context_domain_malformed"),
            "station": _text(record["station"], "context_station_malformed"),
            "held_instrument": held,
            "outer_repeat": _count(record["outer_repeat"], "context_outer_repeat_malformed"),
            "outer_fold": _count(record["outer_fold"], "context_outer_fold_malformed"),
            "selection_mode": _text(record["selection_mode"], "context_selection_malformed"),
            "phase_gate": phase_gate,
        }
    return prepared


def _validated_refits(refits, predictions, test_uids, contexts, manifest):
    if not isinstance(predictions, Mapping) or set(predictions) != set(refits):
        raise P05ResultsError("prediction_keys_mismatch")
    validated: dict[str, Any] = {}
    for refit_id, spec in refits.items():
        frame = predictions[refit_id]
        if not isinstance(frame, pd.DataFrame):
            raise P05ResultsError("prediction_frame_malformed")
        if not {"observation_uid", *PROBABILITY_COLUMNS} <= set(frame.columns):
            raise P05ResultsError("prediction_frame_malformed")
        context = contexts[spec["context_id"]]
        classes = spec["classes"]
        uids = [_text(value, "prediction_uid_malformed") for value in frame["observation_uid"]]
        if len(set(uids)) != len(uids) or set(uids) != set(test_uids[spec["context_id"]]):
            raise P05ResultsError("prediction_uid_mismatch")
        values = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
        if values.shape != (len(uids), len(PROBABILITY_COLUMNS)):
            raise P05ResultsError("prediction_probability_shape")
        if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
            raise P05ResultsError("prediction_probability_invalid")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
            raise P05ResultsError("prediction_probability_not_normalized")
        labels, masters, instruments = [], [], []
        held = context["held_instrument"]
        for uid in uids:
            metadata = manifest[uid]
            if metadata["target_analyte"] not in classes:
                raise P05ResultsError("prediction_label_outside_class_vocabulary")
            if metadata["station"] != context["station"]:
                raise P05ResultsError("context_station_mismatch")
            if held not in HELD_INSTRUMENT_SENTINELS and metadata["instrument"] != held:
                raise P05ResultsError("context_held_instrument_mismatch")
            labels.append(metadata["target_analyte"])
            masters.append(metadata["master_sample_id"])
            instruments.append(metadata["instrument"])
        validated[refit_id] = {
            "context_id": spec["context_id"],
            "recipe_id": spec["recipe_id"],
            "seed": spec["seed"],
            "classes": classes,
            "uids": uids,
            "labels": labels,
            "masters": masters,
            "instruments": instruments,
            "values": values,
        }
    return validated


def _seed_frame(aliases, validated, contexts) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key in sorted(aliases):
        context_id, strategy, seed = key
        refit_id = aliases[key]
        data = validated[refit_id]
        context = contexts[context_id]
        vocabulary = json.dumps(list(data["classes"]), separators=(",", ":"))
        values = data["values"]
        for position, uid in enumerate(data["uids"]):
            rows.append(
                {
                    "context_id": context_id,
                    "experiment_id": context["experiment_id"],
                    "source_context_experiment_id": context["source_context_experiment_id"],
                    "protocol_version": RESULTS_PROTOCOL_VERSION,
                    "domain": context["domain"],
                    "station": context["station"],
                    "held_instrument": context["held_instrument"],
                    "outer_repeat": context["outer_repeat"],
                    "outer_fold": context["outer_fold"],
                    "model_id": strategy,
                    "candidate_id": data["recipe_id"],
                    "refit_id": refit_id,
                    "seed": seed,
                    "observation_uid": uid,
                    "master_sample_id": data["masters"][position],
                    "instrument": data["instruments"][position],
                    "true_label": data["labels"][position],
                    "class_vocabulary": vocabulary,
                    "probability_0": float(values[position, 0]),
                    "probability_1": float(values[position, 1]),
                    "probability_2": float(values[position, 2]),
                }
            )
    return pd.DataFrame(rows)


def _context_lookup(contexts) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "context_id": context_id,
                "source_context_experiment_id": context["source_context_experiment_id"],
            }
            for context_id, context in contexts.items()
        ]
    )


def _attach_context(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    frame = frame.drop(columns=["source_context_experiment_id"], errors="ignore")
    frame = frame.merge(lookup, on="context_id", how="left", validate="many_to_one")
    frame["protocol_version"] = RESULTS_PROTOCOL_VERSION
    return frame


def _enrich(metrics: pd.DataFrame, ensemble: pd.DataFrame, *, dedup_master: bool) -> pd.DataFrame:
    records = []
    for keys, cell in ensemble.groupby(list(GROUP_COLUMNS), dropna=False, sort=False):
        classes = _classes(cell["class_vocabulary"].iloc[0])
        truth = cell.drop_duplicates("master_sample_id") if dedup_master else cell
        labels = truth["true_label"].astype(str).to_numpy()
        support = {label: int(np.sum(labels == label)) for label in classes}
        records.append(
            {
                **dict(zip(GROUP_COLUMNS, keys, strict=True)),
                "class_vocabulary": json.dumps(list(classes), separators=(",", ":")),
                "per_class_support": json.dumps(support, sort_keys=True, separators=(",", ":")),
                "observed_class_count": int(sum(value > 0 for value in support.values())),
                "missing_classes": json.dumps(
                    [label for label in classes if support[label] == 0], separators=(",", ":")
                ),
            }
        )
    return metrics.merge(
        pd.DataFrame(records), on=list(GROUP_COLUMNS), how="left", validate="one_to_one"
    )


def _strategy_tables(seed_frame: pd.DataFrame, lookup: pd.DataFrame):
    ensembles, spectra, masters = [], [], []
    for strategy in STRATEGIES:
        subset = seed_frame[seed_frame["model_id"].eq(strategy)].reset_index(drop=True)
        ensemble = _attach_context(ensemble_seed_predictions(subset), lookup)
        ensemble["model_id"] = strategy
        ensembles.append(ensemble)
        spectrum, master = endpoint_metrics(ensemble)
        spectrum = _attach_context(spectrum, lookup)
        master = _attach_context(master, lookup)
        spectrum["model_id"] = strategy
        master["model_id"] = strategy
        spectra.append(_enrich(spectrum, ensemble, dedup_master=False))
        masters.append(_enrich(master, ensemble, dedup_master=True))
    return ensembles, spectra, masters


def _coverage(seed_frame, endpoints, contexts, test_uids, aliases, validated) -> pd.DataFrame:
    rows = []
    for context_id in sorted(endpoints):
        context = contexts[context_id]
        expected = sorted(test_uids[context_id])
        expected_hash = _hash(expected)
        for strategy in STRATEGIES:
            refit_id = aliases[(context_id, strategy, SEEDS[0])]
            cell = seed_frame[
                seed_frame["context_id"].eq(context_id) & seed_frame["model_id"].eq(strategy)
            ]
            observed = sorted({str(value) for value in cell["observation_uid"].tolist()})
            seed_count = int(cell["seed"].nunique())
            passed = (
                _hash(observed) == expected_hash
                and len(observed) == len(expected)
                and seed_count == len(SEEDS)
            )
            rows.append(
                {
                    "context_id": context_id,
                    "experiment_id": context["experiment_id"],
                    "source_context_experiment_id": context["source_context_experiment_id"],
                    "protocol_version": RESULTS_PROTOCOL_VERSION,
                    "strategy": strategy,
                    "candidate_id": validated[refit_id]["recipe_id"],
                    "expected_test_rows": len(expected),
                    "observed_test_rows": len(observed),
                    "expected_seed_count": len(SEEDS),
                    "observed_seed_count": seed_count,
                    "expected_test_uid_sha256": expected_hash,
                    "observed_test_uid_sha256": _hash(observed),
                    "status": "pass" if passed else "fail",
                }
            )
    return pd.DataFrame(rows)


def aggregate_predictions(*, plan, contexts, manifest, predictions) -> dict[str, pd.DataFrame]:
    """Aggregate calibrated P05 predictions into the frozen result tables."""
    plan = _validate_plan(plan)
    refits, context_classes = _refits(plan, plan["permit_sha256"])
    endpoints, aliases = _index(plan, refits)
    manifest_index = _manifest(manifest)
    test_uids = _endpoints(endpoints, manifest_index)
    context_index = _contexts(contexts, endpoints, test_uids, context_classes)
    validated = _validated_refits(refits, predictions, test_uids, context_index, manifest_index)
    seed_frame = _seed_frame(aliases, validated, context_index)
    lookup = _context_lookup(context_index)
    ensembles, spectra, masters = _strategy_tables(seed_frame, lookup)
    coverage = _coverage(seed_frame, endpoints, context_index, test_uids, aliases, validated)
    return {
        "seed_predictions": seed_frame,
        "ensemble_predictions": pd.concat(ensembles, ignore_index=True),
        "spectrum_metrics": pd.concat(spectra, ignore_index=True),
        "master_metrics": pd.concat(masters, ignore_index=True),
        "coverage": coverage,
    }
