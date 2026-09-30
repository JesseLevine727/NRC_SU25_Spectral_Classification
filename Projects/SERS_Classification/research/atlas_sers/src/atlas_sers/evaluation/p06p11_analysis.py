"""Pure in-memory P06/P11 frozen-prediction analysis orchestration (T017).

No I/O, tools, git or real data live here.  The caller supplies an in-memory
panel and the frozen paired metrics; every random draw is produced by the
frozen ``p06p11_*`` helpers.  This module only checks, orchestrates and
aggregates, and never mutates its inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from atlas_sers.evaluation import p05_comparison as p05c
from atlas_sers.evaluation import p06p11_diagnostics as diag
from atlas_sers.evaluation import p06p11_hierarchy as hier
from atlas_sers.evaluation import p06p11_inference as inf
from atlas_sers.evaluation import p06p11_predictions as preds

WEIGHT_METHODS = (
    ("crossed_weight", "crossed"),
    ("master_weight", "master"),
    ("instrument_weight", "instrument"),
)
BCA_WEIGHT_REASON = "crossed_cluster_acceleration_not_specified"
BCA_HIERARCHY_REASON = "hierarchical_acceleration_not_specified"
_INTERVAL_FIELDS = (
    "planned_draws",
    "defined_draws",
    "undefined_draws",
    "lower",
    "upper",
    "reason_code",
)


class P06P11AnalysisError(RuntimeError):
    """Fatal, reasoned refusal to continue the frozen analysis."""

    def __init__(self, reason_code):
        super().__init__(reason_code)
        self.reason_code = reason_code


def _names(design, *attributes):
    for attribute in attributes:
        value = getattr(design, attribute, None)
        if value is not None:
            return tuple(value)
    raise P06P11AnalysisError("design_identity_missing")


def _design_masters(design):
    return _names(design, "masters", "master_ids", "master_order")


def _design_instruments(design):
    return _names(design, "instruments", "instrument_ids", "instrument_order")


def _design_domains(design):
    return _names(design, "domains", "domain_ids", "domain_order")


def _subset(frame, model_id, reference_model_id, aggregation_id):
    return frame[
        frame["model_id"].eq(model_id)
        & frame["reference_model_id"].eq(reference_model_id)
        & frame["aggregation_id"].eq(aggregation_id)
    ]


def _frozen_domains(frame, model_id, reference_model_id, aggregation_id):
    subset = _subset(frame, model_id, reference_model_id, aggregation_id)
    if "complete_contexts" in subset.columns:
        subset = subset[subset["complete_contexts"] > 0]
    return {str(row.domain): float(row.delta) for row in subset.itertuples(index=False)}


def _frozen_means(frame, model_id, reference_model_id, aggregation_id):
    subset = _subset(frame, model_id, reference_model_id, aggregation_id)
    if subset.empty:
        raise P06P11AnalysisError("frozen_support_mismatch")
    return [float(value) for value in subset["mean_delta"].tolist()]


def _bind(mapping, key, value):
    previous = mapping.get(key)
    if previous is None:
        mapping[key] = value
    elif previous != value:
        raise P06P11AnalysisError("inconsistent_global_identity")


def _record_identities(rows, master_station, master_class, domain_instrument):
    for record in rows.itertuples(index=False):
        _bind(master_station, str(record.master_sample_id), str(record.station))
        _bind(master_class, str(record.master_sample_id), str(record.true_label))
        _bind(domain_instrument, str(record.domain), str(record.instrument))


def _interval_row(contrast, method, point_estimate, summary, bca_reason):
    row = {
        "model_id": contrast["model_id"],
        "reference_model_id": contrast["reference_model_id"],
        "aggregation_id": contrast["aggregation_id"],
        "method": method,
        "point_estimate": point_estimate,
    }
    for field in _INTERVAL_FIELDS:
        row[field] = summary.get(field)
    row["bca_lower"] = None
    row["bca_upper"] = None
    row["bca_reason_code"] = bca_reason
    return row


def _mean(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if values.size else 0.0


def _max(values):
    values = np.asarray(values)
    return int(values.max()) if values.size else 0


def analyze_panel(
    panel,
    paired_metrics,
    *,
    draws=10000,
    master_seed=2026092901,
    instrument_seed=2026092902,
    hierarchy_seed=2026092903,
    check=None,
):
    """Run the frozen-support inference orchestration over an in-memory panel."""
    if check is not None:
        check()
    diagnostics = diag.domain_diagnostics(paired_metrics)
    tables = {
        name: diagnostics[name].copy()
        for name in ("domain_metrics", "summary", "leave_one_out", "sign_flip")
    }
    frozen_domains = tables["domain_metrics"]
    frozen_summary = tables["summary"]

    contrasts = []
    master_station = {}
    master_class = {}
    domain_instrument = {}
    for index, (model_id, reference_model_id) in enumerate(p05c.PAIRS):
        for aggregation_id in p05c.AGGREGATIONS:
            if check is not None:
                check()
            rows = preds.pair_units(
                panel,
                model_id=model_id,
                reference_model_id=reference_model_id,
                aggregation_id=aggregation_id,
            )
            design = inf.compile_pair(rows)
            masters = _design_masters(design)
            instruments = _design_instruments(design)
            domains = _design_domains(design)
            _record_identities(rows, master_station, master_class, domain_instrument)
            ones_m = np.ones((1, len(masters)), dtype=float)
            ones_i = np.ones((1, len(instruments)), dtype=float)
            overall, per_domain = inf.score_weights(design, ones_m, ones_i)
            overall = np.asarray(overall, dtype=float).reshape(-1)
            per_domain = np.asarray(per_domain, dtype=float).reshape(1, -1)
            expected = _frozen_domains(frozen_domains, model_id, reference_model_id, aggregation_id)
            if set(map(str, domains)) != set(expected):
                raise P06P11AnalysisError("frozen_support_mismatch")
            for position, domain in enumerate(domains):
                if not math.isclose(
                    float(per_domain[0, position]),
                    expected[str(domain)],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise P06P11AnalysisError("frozen_point_estimate_mismatch")
            for value in _frozen_means(
                frozen_summary, model_id, reference_model_id, aggregation_id
            ):
                if not math.isclose(float(overall[0]), value, rel_tol=0.0, abs_tol=1e-12):
                    raise P06P11AnalysisError("frozen_point_estimate_mismatch")
            contrasts.append(
                {
                    "key": f"c{index:03d}_{aggregation_id}",
                    "model_id": model_id,
                    "reference_model_id": reference_model_id,
                    "aggregation_id": aggregation_id,
                    "design": design,
                    "masters": tuple(map(str, masters)),
                    "instruments": tuple(map(str, instruments)),
                    "domains": tuple(map(str, domains)),
                    "point_estimate": float(overall[0]),
                }
            )
            if check is not None:
                check()

    global_masters = sorted({master for contrast in contrasts for master in contrast["masters"]})
    global_instruments = sorted(
        {instrument for contrast in contrasts for instrument in contrast["instruments"]}
    )
    for frame in panel.values():
        if "master_sample_id" in frame.columns:
            if set(map(str, frame["master_sample_id"])) - set(global_masters):
                raise P06P11AnalysisError("unrepresented_identity")
        if "instrument" in frame.columns:
            if set(map(str, frame["instrument"])) - set(global_instruments):
                raise P06P11AnalysisError("unrepresented_identity")

    master_weight, instrument_weight = inf.positive_weights(
        tuple(global_masters),
        tuple(global_instruments),
        draws=draws,
        master_seed=master_seed,
        instrument_seed=instrument_seed,
    )
    master_weight = np.asarray(master_weight, dtype=float)
    instrument_weight = np.asarray(instrument_weight, dtype=float)
    batch_draws = int(master_weight.shape[0])
    master_position = {name: index for index, name in enumerate(global_masters)}
    instrument_position = {name: index for index, name in enumerate(global_instruments)}

    arrays = {
        "global.master_weights": master_weight,
        "global.instrument_weights": instrument_weight,
    }
    registry_contrasts = {}
    interval_rows = []
    feasibility_rows = []
    for contrast in contrasts:
        if check is not None:
            check()
        key = contrast["key"]
        design = contrast["design"]
        local_master = master_weight[:, [master_position[name] for name in contrast["masters"]]]
        local_instrument = instrument_weight[
            :, [instrument_position[name] for name in contrast["instruments"]]
        ]
        method_inputs = {
            "crossed": (local_master, local_instrument),
            "master": (local_master, np.ones_like(local_instrument)),
            "instrument": (np.ones_like(local_master), local_instrument),
        }
        method_scores = {}
        for method, suffix in WEIGHT_METHODS:
            if check is not None:
                check()
            factor_master, factor_instrument = method_inputs[suffix]
            overall, per_domain = inf.score_weights(design, factor_master, factor_instrument)
            overall = np.asarray(overall, dtype=float).reshape(-1)
            arrays[f"{key}.{suffix}.scores"] = overall
            arrays[f"{key}.{suffix}.domain_scores"] = np.asarray(per_domain, dtype=float)
            method_scores[method] = f"{key}.{suffix}.scores"
            interval_rows.append(
                _interval_row(
                    contrast,
                    method,
                    contrast["point_estimate"],
                    inf.summarize_draws(overall),
                    BCA_WEIGHT_REASON,
                )
            )
            if check is not None:
                check()

        if check is not None:
            check()
        hierarchical = hier.hierarchical_draws(design, draws=draws, seed=hierarchy_seed)
        scores = np.asarray(hierarchical["draws"], dtype=float)
        empty_cells = np.asarray(hierarchical["empty_cells"])
        undefined_occurrences = np.asarray(hierarchical["undefined_domain_occurrences"])
        affected_domains = np.asarray(hierarchical["affected_domains"])
        sampled_domains = np.asarray(hierarchical["sampled_domains"])
        arrays[f"{key}.hierarchical.scores"] = scores
        arrays[f"{key}.hierarchical.empty_cells"] = empty_cells
        arrays[f"{key}.hierarchical.undefined_domain_occurrences"] = undefined_occurrences
        arrays[f"{key}.hierarchical.affected_domains"] = affected_domains
        arrays[f"{key}.hierarchical.sampled_domains"] = sampled_domains
        interval_rows.append(
            _interval_row(
                contrast,
                "hierarchical",
                contrast["point_estimate"],
                inf.summarize_draws(scores),
                BCA_HIERARCHY_REASON,
            )
        )
        if check is not None:
            check()
        undefined_draws = int(np.count_nonzero(empty_cells > 0))
        feasibility_rows.append(
            {
                "model_id": contrast["model_id"],
                "reference_model_id": contrast["reference_model_id"],
                "aggregation_id": contrast["aggregation_id"],
                "planned_draws": int(draws),
                "defined_draws": batch_draws - undefined_draws,
                "undefined_draws": undefined_draws,
                "mean_empty_cells": _mean(empty_cells),
                "max_empty_cells": _max(empty_cells),
                "mean_undefined_domain_occurrences": _mean(undefined_occurrences),
                "max_undefined_domain_occurrences": _max(undefined_occurrences),
                "mean_affected_domains": _mean(affected_domains),
                "max_affected_domains": _max(affected_domains),
            }
        )
        registry_contrasts[key] = {
            "model_id": contrast["model_id"],
            "reference_model_id": contrast["reference_model_id"],
            "aggregation_id": contrast["aggregation_id"],
            "masters": contrast["masters"],
            "instruments": contrast["instruments"],
            "domains": contrast["domains"],
            "method_scores": method_scores,
            "hierarchical_scores": f"{key}.hierarchical.scores",
            "hierarchical_sampled_domains": f"{key}.hierarchical.sampled_domains",
        }
        if check is not None:
            check()

    if check is not None:
        check()
    tables["intervals"] = pd.DataFrame(interval_rows)
    tables["feasibility"] = pd.DataFrame(feasibility_rows)
    registry = {
        "masters": tuple(global_masters),
        "instruments": tuple(global_instruments),
        "master_weight_array": "global.master_weights",
        "instrument_weight_array": "global.instrument_weights",
        "contrasts": registry_contrasts,
        "master_station": dict(master_station),
        "master_class": dict(master_class),
        "domain_instrument": dict(domain_instrument),
    }
    return {"tables": tables, "arrays": arrays, "registry": registry}
