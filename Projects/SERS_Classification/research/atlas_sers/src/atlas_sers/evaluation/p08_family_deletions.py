"""Preplanned descriptive leave-one-known-platform-family sensitivity.

Computed from the aggregate paired-domain table only.  This module is pure and
deterministic: it performs no fitting, resampling, file I/O, subprocess call or
input mutation.  Instrument families are derived through the frozen
``atlas_sers.splits.p02.instrument_family`` mapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas_sers.splits.p02 import instrument_family

__all__ = ["prepare_family_deletions"]

_SUPPORTED_ESTIMANDS = ("equal_context", "pooled_four_fold")
_UNKNOWN_FAMILY = "unknown"
_REQUIRED_COLUMNS = ("estimand", "contrast_id", "domain", "instrument", "effect")
_OUTPUT_COLUMNS = (
    "estimand",
    "contrast_id",
    "excluded_family",
    "excluded_domain_count",
    "retained_domain_count",
    "effect",
    "reason",
)


def _empty_output() -> pd.DataFrame:
    """Return an explicitly typed empty result frame."""
    return pd.DataFrame(
        {
            "estimand": pd.Series(dtype="object"),
            "contrast_id": pd.Series(dtype="object"),
            "excluded_family": pd.Series(dtype="object"),
            "excluded_domain_count": pd.Series(dtype="int64"),
            "retained_domain_count": pd.Series(dtype="int64"),
            "effect": pd.Series(dtype="object"),
            "reason": pd.Series(dtype="object"),
        }
    )


def _validate_names(series: pd.Series, label: str) -> None:
    valid = series.map(lambda value: isinstance(value, str) and bool(value.strip()))
    if not bool(valid.all()):
        raise ValueError(f"{label} must contain only nonempty strings")


def _validate_effects(effect: pd.Series) -> None:
    for value in effect.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("effect must not be boolean")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("effect must be a finite real number")
        if not np.isfinite(float(value)):
            raise ValueError("effect must be finite")


def prepare_family_deletions(paired_domains: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-known-family sensitivity from the paired-domain table.

    ``paired_domains`` must provide ``estimand``, ``contrast_id``, ``domain``,
    ``instrument`` and ``effect`` columns (extra metadata columns are allowed).
    For every estimand/contrast and every known instrument family, all domains
    whose instrument belongs to that family are removed and the equally
    weighted mean of the retained domain effects is reported.  Domains are
    never weighted by sample counts.
    """
    if not isinstance(paired_domains, pd.DataFrame):
        raise TypeError("paired_domains must be a pandas DataFrame")
    if not paired_domains.columns.is_unique:
        raise ValueError("column names must be unique")

    missing = [column for column in _REQUIRED_COLUMNS if column not in paired_domains.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}")

    work = paired_domains.loc[:, list(_REQUIRED_COLUMNS)].copy()

    _validate_names(work["estimand"], "estimand")
    _validate_names(work["contrast_id"], "contrast_id")
    _validate_names(work["domain"], "domain")
    _validate_names(work["instrument"], "instrument")
    _validate_effects(work["effect"])

    unsupported = sorted(set(work["estimand"].tolist()) - set(_SUPPORTED_ESTIMANDS))
    if unsupported:
        raise ValueError(f"unsupported estimand(s): {unsupported}")

    if work.duplicated(subset=["estimand", "contrast_id", "domain"]).any():
        raise ValueError("duplicate (estimand, contrast_id, domain) rows are not allowed")

    domain_instrument = work.loc[:, ["domain", "instrument"]].drop_duplicates()
    instrument_counts = domain_instrument.groupby("domain")["instrument"].nunique()
    if bool((instrument_counts > 1).any()):
        raise ValueError("each domain must map to exactly one instrument")

    support = work.loc[:, ["estimand", "contrast_id", "domain", "instrument"]].drop_duplicates()
    for estimand, group in support.groupby("estimand"):
        signatures = (
            group.groupby("contrast_id")[["domain", "instrument"]]
            .apply(lambda frame: frozenset(map(tuple, frame.to_numpy())))
        )
        if signatures.nunique() > 1:
            raise ValueError(
                f"contrasts within estimand {estimand!r} must share identical support"
            )

    if work.empty:
        return _empty_output()

    work = work.assign(effect=work["effect"].astype(float)).sort_values(
        ["estimand", "contrast_id", "domain"], kind="stable"
    )

    family_map = {
        name: instrument_family(name) for name in sorted(set(work["instrument"].tolist()))
    }
    work = work.assign(family=work["instrument"].map(family_map))

    records = []
    for (estimand, contrast_id), group in work.groupby(["estimand", "contrast_id"], sort=True):
        known_families = sorted(
            family for family in set(group["family"]) if family.lower() != _UNKNOWN_FAMILY
        )
        for family in known_families:
            excluded_domains = set(group.loc[group["family"] == family, "domain"].tolist())
            retained = group.loc[~group["domain"].isin(excluded_domains)]
            excluded_count = len(excluded_domains)
            retained_count = int(len(retained))
            if retained_count == 0:
                value = None
                reason = "no_retained_domains"
            else:
                value = float(retained["effect"].mean())
                reason = ""
            records.append(
                {
                    "estimand": estimand,
                    "contrast_id": contrast_id,
                    "excluded_family": family,
                    "excluded_domain_count": excluded_count,
                    "retained_domain_count": retained_count,
                    "effect": value,
                    "reason": reason,
                }
            )

    if not records:
        return _empty_output()
    output = pd.DataFrame(
        {
            "estimand": pd.Series([row["estimand"] for row in records], dtype="object"),
            "contrast_id": pd.Series([row["contrast_id"] for row in records], dtype="object"),
            "excluded_family": pd.Series(
                [row["excluded_family"] for row in records], dtype="object"
            ),
            "excluded_domain_count": pd.Series(
                [row["excluded_domain_count"] for row in records], dtype="int64"
            ),
            "retained_domain_count": pd.Series(
                [row["retained_domain_count"] for row in records], dtype="int64"
            ),
            "effect": pd.Series([row["effect"] for row in records], dtype="object"),
            "reason": pd.Series([row["reason"] for row in records], dtype="object"),
        }
    )
    output = output.sort_values(
        by=["estimand", "contrast_id", "excluded_family"], kind="mergesort"
    ).reset_index(drop=True)
    return output.loc[:, list(_OUTPUT_COLUMNS)]
