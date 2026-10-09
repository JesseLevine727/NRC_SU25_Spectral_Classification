"""P08-F07 bounded preservation/accuracy semantic join (no I/O)."""

from __future__ import annotations

import hashlib
import json
import math
import numbers

import numpy as np
import pandas as pd

FIGURE_ID = "P08-F07"
RESEARCH_QUESTION_ID = "RQ-S05"
SCHEMA_VERSION = "nato-sers-p08-f07-data-v1"

PRESERVATION_COLUMNS = (
    "station",
    "instrument",
    "representation_id",
    "metric",
    "n_spectra",
    "n_masters",
    "held_comparison_domain",
    "finite_count",
    "undefined_count",
    "median",
    "q10",
    "q90",
)

ACTION_BY_REPRESENTATION = {
    "R_MIN_400_1800": "PP-U-MIN",
    "R_SG_400_1800": "PP-U-SG",
    "R_ARPLS_400_1800": "PP-U-ARPLS",
}
REPRESENTATION_BY_ACTION = {v: k for k, v in ACTION_BY_REPRESENTATION.items()}
ACTION_ORDER = ("PP-U-MIN", "PP-U-SG", "PP-U-ARPLS")

METRIC_ORDER = (
    "baseline_span",
    "candidate_peak_count",
    "changed_point_fraction",
    "clipped_fraction",
    "first_difference_roughness",
    "median_peak_displacement_cm1",
    "rank_correlation",
    "reference_peak_count",
    "shape_correlation",
    "spectral_angle_radians",
    "top_peak_recall_pm5cm1",
)
DISPLACEMENT_METRIC = "median_peak_displacement_cm1"
RECALL_METRIC = "top_peak_recall_pm5cm1"

ESTIMAND_ORDER = ("equal_context", "pooled_four_fold")
ENDPOINT_ORDER = ("M01", "M06")
MODEL_ORDER = ("C-RBF-SVM", "C-RANDOM-FOREST", "C-EXTRA-TREES", "D0-M", "P05-SELECTED")
POLICY_ORDER = ("PP-U-SG", "PP-U-ARPLS")

F02_KEYS = frozenset(
    (
        "estimand",
        "contrast_id",
        "family_id",
        "endpoint",
        "model_id",
        "policy_id",
        "domain",
        "station",
        "instrument",
        "contexts",
        "unit_appearances",
        "physical_masters",
        "distinct_units",
        "x_balanced_accuracy",
        "y_balanced_accuracy",
        "effect",
    )
)

SUMMARY_FIELDS = (
    "station",
    "instrument",
    "action",
    "representation_id",
    "metric",
    "held_comparison_domain",
    "n_spectra",
    "n_masters",
    "finite_count",
    "undefined_count",
    "median",
    "q10",
    "q90",
)

POINT_FIELDS = (
    "station",
    "instrument",
    "held_comparison_domain",
    "policy_id",
    "representation_id",
    "model_id",
    "endpoint",
    "estimand",
    "original_n_spectra",
    "original_n_masters",
    "model_domain",
    "model_contexts",
    "model_unit_appearances",
    "model_physical_masters",
    "model_distinct_units",
    "balanced_accuracy",
    "min_balanced_accuracy",
    "policy_minus_min",
    "peak_displacement_median_cm1",
    "peak_displacement_q10_cm1",
    "peak_displacement_q90_cm1",
    "peak_displacement_finite_count",
    "peak_displacement_undefined_count",
    "peak_recall_median",
    "peak_recall_q10",
    "peak_recall_q90",
    "peak_recall_finite_count",
    "peak_recall_undefined_count",
    "available",
    "reason",
)

CAPTION = (
    "P08-F07 joins accepted F02 balanced-accuracy pairs to accepted preservation "
    "summaries. The source study comprised 598 original spectra from 69 physical "
    "masters across 10 instruments, partitioned into 17 station-instrument domains "
    "(13 held comparison domains, 4 exploratory domains). Endpoint M01 uses "
    "individual predictions and M06 uses mean model probabilities; neither is a mean "
    "spectrum. Model selection and calibration are source-only, and the "
    "selected CNN identity is fixed across preprocessing. Peak diagnostics are "
    "descriptive against the frozen observed-spectrum reference and are not "
    "clean-chemistry truth. No causal distinction, winner selection, or "
    "new-instrument guarantee is implied. Summaries are stored-observation-weighted "
    "medians and 10th/90th percentiles (linear interpolation), not confidence "
    "intervals or master-equal chemical estimates. Original preservation counts "
    "and held-prediction support counts are kept separate. The baseline-span "
    "proxy is not measured fluorescence, and clipped fraction does not prove "
    "physical detector saturation. M06 averages model probabilities within each "
    "sample/instrument and then equally across instruments, not input spectra "
    "or hard labels."
)


def _canonical_sha256(payload):
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strict_int(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _strict_bool(value):
    return isinstance(value, (bool, np.bool_))


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("expected a numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("expected a finite value")
    return result


def _finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("summary value must not be boolean")
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.floating) and np.isnan(value):
        return None
    if not isinstance(value, numbers.Real):
        raise ValueError("summary value must be numeric or missing")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("summary value must be finite")
    return result


def _bounded_accuracy(value):
    result = _finite_number(value)
    if result < 0.0 or result > 1.0:
        raise ValueError("balanced accuracy out of range")
    return result


def _prepare_preservation(preservation):
    if not isinstance(preservation, pd.DataFrame):
        raise ValueError("preservation must be a pandas DataFrame")
    if set(preservation.columns) != set(PRESERVATION_COLUMNS) or len(preservation.columns) != len(
        PRESERVATION_COLUMNS
    ):
        raise ValueError("preservation columns mismatch")
    summaries = []
    domain_meta = {}
    seen = set()
    for rec in preservation.to_dict("records"):
        station, instrument = rec["station"], rec["instrument"]
        if any(not isinstance(v, str) or not v for v in (station, instrument)):
            raise ValueError("station/instrument labels must be nonempty strings")
        rep, metric = rec["representation_id"], rec["metric"]
        if rep not in ACTION_BY_REPRESENTATION:
            raise ValueError("unknown representation_id")
        if metric not in METRIC_ORDER:
            raise ValueError("unknown metric")
        for value in (
            rec["n_spectra"],
            rec["n_masters"],
            rec["finite_count"],
            rec["undefined_count"],
        ):
            if not _strict_int(value):
                raise ValueError("counts must be strict integers")
        n_spectra, n_masters = int(rec["n_spectra"]), int(rec["n_masters"])
        finite, undefined = int(rec["finite_count"]), int(rec["undefined_count"])
        if n_spectra <= 0 or n_masters <= 0 or n_masters > n_spectra:
            raise ValueError("invalid spectra/masters counts")
        if finite < 0 or undefined < 0 or finite + undefined != n_spectra:
            raise ValueError("finite+undefined must equal n_spectra")
        if not _strict_bool(rec["held_comparison_domain"]):
            raise ValueError("held_comparison_domain must be boolean")
        held = bool(rec["held_comparison_domain"])
        median = _finite_or_none(rec["median"])
        q10 = _finite_or_none(rec["q10"])
        q90 = _finite_or_none(rec["q90"])
        if finite == 0:
            if median is not None or q10 is not None or q90 is not None:
                raise ValueError("missing summaries required when finite_count is 0")
        elif median is None or q10 is None or q90 is None:
            raise ValueError("summaries required when finite data present")
        elif q10 > median or median > q90:
            raise ValueError("summary ordering violated")
        key = (station, instrument, rep, metric)
        if key in seen:
            raise ValueError("duplicate domain/action/metric row")
        seen.add(key)
        meta = domain_meta.get((station, instrument))
        expected = [n_spectra, n_masters, held]
        if meta is None:
            domain_meta[(station, instrument)] = expected
        elif meta != expected:
            raise ValueError("domain metadata inconsistent")
        summaries.append(
            {
                "station": station,
                "instrument": instrument,
                "action": ACTION_BY_REPRESENTATION[rep],
                "representation_id": rep,
                "metric": metric,
                "held_comparison_domain": held,
                "n_spectra": n_spectra,
                "n_masters": n_masters,
                "finite_count": finite,
                "undefined_count": undefined,
                "median": median,
                "q10": q10,
                "q90": q90,
            }
        )
    if len(seen) != 561 or len(domain_meta) != 17:
        raise ValueError("unexpected preservation shape")
    for station, instrument in domain_meta:
        reps = {k[2] for k in seen if k[0] == station and k[1] == instrument}
        if reps != set(ACTION_BY_REPRESENTATION):
            raise ValueError("domain is missing actions")
        for rep in reps:
            metrics = {k[3] for k in seen if k[0] == station and k[1] == instrument and k[2] == rep}
            if metrics != set(METRIC_ORDER):
                raise ValueError("action is missing metrics")
    held_domains = {d for d, m in domain_meta.items() if m[2]}
    if len(held_domains) != 13 or len(domain_meta) - len(held_domains) != 4:
        raise ValueError("unexpected held/exploratory domain counts")
    summaries.sort(
        key=lambda r: (
            r["station"],
            r["instrument"],
            ACTION_ORDER.index(r["action"]),
            METRIC_ORDER.index(r["metric"]),
        )
    )
    lookup = {
        (r["station"], r["instrument"], r["representation_id"], r["metric"]): r for r in summaries
    }
    return summaries, domain_meta, held_domains, lookup


def _index_pairs(pairs):
    if not isinstance(pairs, list):
        raise ValueError("f02_pairs must be a list")
    index, domain_map = {}, {}
    for row in pairs:
        if not isinstance(row, dict) or set(row.keys()) != set(F02_KEYS):
            raise ValueError("unexpected f02_pairs keys")
        if any(
            not isinstance(row[k], str) or not row[k]
            for k in ("station", "instrument", "domain", "contrast_id", "family_id")
        ):
            raise ValueError("f02 labels must be nonempty strings")
        estimand, endpoint = row["estimand"], row["endpoint"]
        model_id, policy = row["model_id"], row["policy_id"]
        if estimand not in ESTIMAND_ORDER or endpoint not in ENDPOINT_ORDER:
            raise ValueError("unknown estimand or endpoint")
        if model_id not in MODEL_ORDER or policy not in POLICY_ORDER:
            raise ValueError("unknown model or policy")
        for field in ("contexts", "unit_appearances", "physical_masters", "distinct_units"):
            if not _strict_int(row[field]) or int(row[field]) <= 0:
                raise ValueError("support metadata must be positive integers")
        x = _bounded_accuracy(row["x_balanced_accuracy"])
        y = _bounded_accuracy(row["y_balanced_accuracy"])
        effect = _finite_number(row["effect"])
        if abs(effect - (y - x)) > 1e-12:
            raise ValueError("effect must equal y minus x")
        domain = row["domain"]
        key = (estimand, endpoint, model_id, policy, domain)
        if key in index:
            raise ValueError("duplicate f02 grid cell")
        index[key] = row
        pair = (row["station"], row["instrument"])
        if domain_map.get(domain, pair) != pair:
            raise ValueError("domain<->station/instrument inconsistent")
        domain_map[domain] = pair
    if len(index) != 520 or len(domain_map) != 13:
        raise ValueError("unexpected f02 grid size")
    for estimand in ESTIMAND_ORDER:
        for endpoint in ENDPOINT_ORDER:
            for model_id in MODEL_ORDER:
                for policy in POLICY_ORDER:
                    for domain in domain_map:
                        if (estimand, endpoint, model_id, policy, domain) not in index:
                            raise ValueError("incomplete f02 grid")
    for estimand in ESTIMAND_ORDER:
        for endpoint in ENDPOINT_ORDER:
            for model_id in MODEL_ORDER:
                for domain in domain_map:
                    sg = index[(estimand, endpoint, model_id, "PP-U-SG", domain)]
                    ar = index[(estimand, endpoint, model_id, "PP-U-ARPLS", domain)]
                    for field in (
                        "contexts",
                        "unit_appearances",
                        "physical_masters",
                        "distinct_units",
                    ):
                        if sg[field] != ar[field]:
                            raise ValueError("supports inconsistent across policies")
                    if (
                        abs(float(sg["x_balanced_accuracy"]) - float(ar["x_balanced_accuracy"]))
                        > 1e-12
                    ):
                        raise ValueError("MIN x values inconsistent across policies")
    return index, domain_map


def _peak_fields(displacement, recall):
    return {
        "peak_displacement_median_cm1": None if displacement is None else displacement["median"],
        "peak_displacement_q10_cm1": None if displacement is None else displacement["q10"],
        "peak_displacement_q90_cm1": None if displacement is None else displacement["q90"],
        "peak_displacement_finite_count": None
        if displacement is None
        else displacement["finite_count"],
        "peak_displacement_undefined_count": None
        if displacement is None
        else displacement["undefined_count"],
        "peak_recall_median": None if recall is None else recall["median"],
        "peak_recall_q10": None if recall is None else recall["q10"],
        "peak_recall_q90": None if recall is None else recall["q90"],
        "peak_recall_finite_count": None if recall is None else recall["finite_count"],
        "peak_recall_undefined_count": None if recall is None else recall["undefined_count"],
    }


def _build_points(domain_meta, index, domain_map, lookup):
    held_domain_by_pair = {pair: dom for dom, pair in domain_map.items()}
    points = []
    for station, instrument in sorted(domain_meta):
        n_spectra, n_masters, held = domain_meta[(station, instrument)]
        held = bool(held)
        for policy in ACTION_ORDER:
            rep = REPRESENTATION_BY_ACTION[policy]
            displacement = lookup.get((station, instrument, rep, DISPLACEMENT_METRIC))
            recall = lookup.get((station, instrument, rep, RECALL_METRIC))
            for endpoint in ENDPOINT_ORDER:
                for estimand in ESTIMAND_ORDER:
                    for model_id in MODEL_ORDER:
                        point = {
                            "station": station,
                            "instrument": instrument,
                            "held_comparison_domain": held,
                            "policy_id": policy,
                            "representation_id": rep,
                            "model_id": model_id,
                            "endpoint": endpoint,
                            "estimand": estimand,
                            "original_n_spectra": int(n_spectra),
                            "original_n_masters": int(n_masters),
                        }
                        if held:
                            domain = held_domain_by_pair[(station, instrument)]
                            ref = policy if policy in POLICY_ORDER else "PP-U-SG"
                            row = index[(estimand, endpoint, model_id, ref, domain)]
                            x = float(row["x_balanced_accuracy"])
                            if policy == "PP-U-MIN":
                                balanced, delta = x, 0.0
                            else:
                                balanced = float(row["y_balanced_accuracy"])
                                delta = float(row["effect"])
                            point.update(
                                {
                                    "model_domain": domain,
                                    "model_contexts": int(row["contexts"]),
                                    "model_unit_appearances": int(row["unit_appearances"]),
                                    "model_physical_masters": int(row["physical_masters"]),
                                    "model_distinct_units": int(row["distinct_units"]),
                                    "balanced_accuracy": balanced,
                                    "min_balanced_accuracy": x,
                                    "policy_minus_min": delta,
                                    "available": True,
                                    "reason": None,
                                }
                            )
                        else:
                            point.update(
                                {
                                    "model_domain": None,
                                    "model_contexts": None,
                                    "model_unit_appearances": None,
                                    "model_physical_masters": None,
                                    "model_distinct_units": None,
                                    "balanced_accuracy": None,
                                    "min_balanced_accuracy": None,
                                    "policy_minus_min": None,
                                    "available": False,
                                    "reason": "outside_held_comparison",
                                }
                            )
                        point.update(_peak_fields(displacement, recall))
                        if set(point) != set(POINT_FIELDS):
                            raise ValueError("point field whitelist mismatch")
                        points.append(point)
    return points


def _validate_model_bundle(model_bundle):
    if not isinstance(model_bundle, dict):
        raise ValueError("model_bundle must be a mapping")
    for key in ("semantic", "semantic_sha256", "manifest"):
        if key not in model_bundle:
            raise ValueError("model_bundle is missing a required key")
    if not isinstance(model_bundle["semantic"], dict):
        raise ValueError("model_bundle semantic must be a mapping")
    if not isinstance(model_bundle["semantic_sha256"], str):
        raise ValueError("model semantic_sha256 must be a string")
    try:
        expected = _canonical_sha256(model_bundle["semantic"])
    except (TypeError, ValueError):
        raise ValueError("model semantic is not canonical-JSON serializable") from None
    if expected != model_bundle["semantic_sha256"]:
        raise ValueError("model semantic SHA256 mismatch")
    return model_bundle["semantic"]


def prepare_f07(preservation, model_bundle):
    """Join accepted preservation summaries with accepted F02 pairs (pure)."""
    model_semantic = _validate_model_bundle(model_bundle)
    summaries, domain_meta, held_domains, lookup = _prepare_preservation(preservation)
    index, domain_map = _index_pairs(model_semantic.get("f02_pairs"))
    if set(domain_map.values()) != held_domains:
        raise ValueError("f02 domains do not match held preservation domains")
    points = _build_points(domain_meta, index, domain_map, lookup)
    semantic = {
        "figure_id": FIGURE_ID,
        "research_question_id": RESEARCH_QUESTION_ID,
        "schema_version": SCHEMA_VERSION,
        "source_model_semantic_sha256": model_bundle["semantic_sha256"],
        "caption": CAPTION,
        "summaries": summaries,
        "points": points,
    }
    semantic_sha256 = _canonical_sha256(semantic)
    held_points = sum(1 for point in points if point["held_comparison_domain"])
    manifest = {
        "figure_id": FIGURE_ID,
        "status": "prepared",
        "semantic_sha256": semantic_sha256,
        "reviewed": False,
        "published": False,
        "counts": {
            "summaries": len(summaries),
            "points": len(points),
            "held_points": held_points,
            "exploratory_points": len(points) - held_points,
            "domains": len(domain_meta),
            "held_domains": len(held_domains),
            "exploratory_domains": len(domain_meta) - len(held_domains),
        },
    }
    return {"semantic": semantic, "semantic_sha256": semantic_sha256, "manifest": manifest}
