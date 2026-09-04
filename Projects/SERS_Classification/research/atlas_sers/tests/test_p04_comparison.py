from __future__ import annotations

import json

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p04_comparison import (
    CLASSICAL_MODELS,
    _prepare_domain_delta,
    _weighted_domain_delta,
    compare_endpoint_metrics,
    master_clustered_paired_bootstrap,
)
from atlas_sers.governance.canonical import sha256_value


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = ["a", "b", "c"]
    d0_rows = []
    classical_rows = []
    for index, label in enumerate(labels):
        common = {
            "context_id": "ctx",
            "experiment_id": "EXP-N00-T3",
            "domain": "station:instrument",
            "held_instrument": "instrument",
            "outer_repeat": 0,
            "outer_fold": 0,
            "observation_uid": f"uid-{index}",
            "master_sample_id": f"master-{index}",
            "instrument": "instrument",
            "station": "station",
            "true_label": label,
            "class_vocabulary": json.dumps(labels),
        }
        probability = [0.05, 0.05, 0.05]
        probability[index] = 0.9
        d0_rows.append(
            {
                **common,
                "candidate_id": "D0-candidate",
                "predicted_label": label,
                "seed_count": 3,
                **{f"probability_{column}": probability[column] for column in range(3)},
            }
        )
        for model in CLASSICAL_MODELS:
            classical_rows.append(
                {
                    **common,
                    "candidate_id": model,
                    "comparison_model_id": model,
                    "predicted_label": label,
                    **{f"probability_{column}": probability[column] for column in range(3)},
                }
            )
    expected = pd.DataFrame(
        [
            {
                "context_id": "ctx",
                "experiment_id": "EXP-N00-T3",
                "domain": "station:instrument",
                "station": "station",
                "held_instrument": "instrument",
                "outer_repeat": 0,
                "outer_fold": 0,
                "expected_test_rows": 3,
                "expected_test_uid_sha256": sha256_value(["uid-0", "uid-1", "uid-2"]),
            }
        ]
    )
    return pd.DataFrame(d0_rows), pd.DataFrame(classical_rows), expected


def test_comparison_requires_exact_common_uid_endpoints() -> None:
    d0, classical, expected = _frames()
    paired, summary, support = compare_endpoint_metrics(
        d0_ensemble=d0, classical=classical, expected=expected
    )

    assert len(paired) == len(CLASSICAL_MODELS) * 2
    assert support.common_complete.all()
    assert set(summary.aggregation_id) == {"M01", "M06"}
    assert summary.mean_paired_delta_d0_minus_classical.eq(0).all()


def test_paired_bootstrap_resamples_physical_masters_not_spectra() -> None:
    d0, classical, expected = _frames()
    _, _, support = compare_endpoint_metrics(d0_ensemble=d0, classical=classical, expected=expected)
    bootstrap = master_clustered_paired_bootstrap(
        d0_ensemble=d0,
        classical=classical,
        support=support,
        draws=100,
        seed=7,
    )

    assert len(bootstrap) == len(CLASSICAL_MODELS) * 2 * 2
    assert bootstrap.bootstrap_draws.eq(100).all()
    assert bootstrap.independent_physical_masters.eq(3).all()
    assert bootstrap.estimate_d0_minus_classical_ba.eq(0).all()


def test_missing_classical_endpoint_remains_in_failure_sensitive_denominator() -> None:
    d0, classical, expected = _frames()
    extra_d0 = d0.copy()
    extra_d0["context_id"] = "ctx-missing-classical"
    extra_d0["outer_fold"] = 1
    extra_d0["observation_uid"] += "-extra"
    extra_d0["master_sample_id"] += "-extra"
    extra_expected = expected.copy()
    extra_expected["context_id"] = "ctx-missing-classical"
    extra_expected["outer_fold"] = 1
    extra_expected["expected_test_uid_sha256"] = sha256_value(
        sorted(extra_d0.observation_uid)
    )
    _, summary, support = compare_endpoint_metrics(
        d0_ensemble=pd.concat([d0, extra_d0], ignore_index=True),
        classical=classical,
        expected=pd.concat([expected, extra_expected], ignore_index=True),
    )
    assert summary.planned_endpoints.eq(2).all()
    assert summary.common_coverage.eq(0.5).all()
    assert summary.d0_mean_ba_common.eq(1).all()
    assert summary.classical_mean_ba_common.eq(1).all()
    assert summary.d0_failure_sensitive_mean_ba.eq(1).all()
    assert summary.classical_failure_sensitive_mean_ba.eq(0.5).all()
    assert not support.loc[
        support.context_id.eq("ctx-missing-classical"), "common_complete"
    ].any()


def test_bootstrap_reports_domain_specific_master_support() -> None:
    d0, classical, expected = _frames()
    d0_extra, classical_extra = [], []
    for copy in range(2):
        for source, destination in ((d0, d0_extra), (classical, classical_extra)):
            additional = source.copy()
            additional["context_id"] = "ctx-2"
            additional["domain"] = "station:other-instrument"
            additional["held_instrument"] = "other-instrument"
            additional["instrument"] = "other-instrument"
            additional["observation_uid"] += f"-extra-{copy}"
            additional["master_sample_id"] += f"-extra-{copy}"
            destination.append(additional)
    d0 = pd.concat([d0, *d0_extra], ignore_index=True)
    classical = pd.concat([classical, *classical_extra], ignore_index=True)
    second = expected.copy()
    second["context_id"] = "ctx-2"
    second["domain"] = "station:other-instrument"
    second["held_instrument"] = "other-instrument"
    second["expected_test_rows"] = 6
    second["expected_test_uid_sha256"] = sha256_value(
        sorted(d0.loc[d0.context_id.eq("ctx-2"), "observation_uid"])
    )
    expected = pd.concat([expected, second], ignore_index=True)
    _, _, support = compare_endpoint_metrics(d0_ensemble=d0, classical=classical, expected=expected)
    bootstrap = master_clustered_paired_bootstrap(
        d0_ensemble=d0, classical=classical, support=support, draws=10, seed=7
    )
    for domain, count in (
        ("station:instrument", 3),
        ("station:other-instrument", 6),
        ("__overall__", 9),
    ):
        assert (
            bootstrap.loc[bootstrap.domain.eq(domain), "independent_physical_masters"]
            .eq(count)
            .all()
        )


def test_cached_bootstrap_statistic_matches_reference_with_repeated_views() -> None:
    rng = np.random.default_rng(11)
    records = []
    for domain in ("instrument-a", "instrument-b"):
        for label in ("a", "b", "c"):
            for master in range(4):
                for _ in range(master + 1):
                    records.append(
                        {
                            "domain": domain,
                            "true_label": label,
                            "master_sample_id": f"{label}-{master}",
                            "base_weight": float(rng.uniform(0.2, 2)),
                            "d0_correct": float(rng.integers(0, 6)) / 5,
                            "classical_correct": float(rng.integers(0, 6)) / 5,
                        }
                    )
    units = pd.DataFrame(records)
    evaluate = _prepare_domain_delta(units)
    for _ in range(30):
        weights = {
            master: int(rng.integers(0, 4)) for master in units.master_sample_id.unique()
        }
        expected = _weighted_domain_delta(units, weights)
        observed = evaluate(weights)
        assert observed.keys() == expected.keys()
        assert np.allclose(list(observed.values()), list(expected.values()), atol=1e-12)
    missing_class = {
        master: int(not master.startswith("c-")) for master in units.master_sample_id.unique()
    }
    assert evaluate(missing_class) == _weighted_domain_delta(units, missing_class) == {}
