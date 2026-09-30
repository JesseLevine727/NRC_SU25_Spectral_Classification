"""Outcome-blind structural support audit for grouped uncertainty specification.

This module exposes :func:`audit_support`, a deterministic, read-only audit of
the support structure of one model/endpoint's spectrum table before a grouped
uncertainty procedure is specified. It is explicitly not an uncertainty
estimator, not a statistical significance analysis and not a resampling scheme.

The audit is outcome blind. Only the identifier and label columns listed in
``_REQUIRED_COLUMNS`` are read. Score, probability, loss, prediction or other
outcome columns may be present in the input but have no influence on the result
and are never exported. The exported tables also never contain the values of
``context_id``, ``observation_uid`` or ``master_sample_id``.

Missing cells and the frozen estimator
--------------------------------------
The support tables describe only the class cells that are actually represented.
A ``class-cell`` is a pair of an evaluation context and a class that occurs in
that context; it is not a slot in a presumed fixed-class universe. The frozen
equal-context/equal-class score uses only the true classes that were originally
observed in each evaluation context. A class that never occurs in an original
test context is simply outside that context's frozen score, so its absence does
not by itself make any published balanced accuracy undefined. The structural
exposure quantified here is a resampling effect instead: bootstrap resampling of
a context that did originally represent a class can empty that cell, leaving a
required cell with no members to score in the resample. This audit quantifies
that structural exposure only. It does not authorize changing the estimator's
estimand, does not select weights, does not simulate resampling, does not
prescribe discarding empty cells and selects no missing-cell policy.

The ``empty_class_risk`` table reports, per domain, the exact expected number of
empty class cells under one domain-local class-stratified master resample. For a
represented context/class cell with ``k`` distinct masters in a class pool of
``N`` distinct masters for that domain, the probability that the cell receives
no master in ``N`` independent uniform draws with replacement is
``(1 - k / N) ** N``. ``expected_empty_class_cells`` is the sum of those per-cell
probabilities, so it is an expected count of empty cells, not the probability
that at least one cell is empty. The minimum and maximum columns describe the
individual cell probabilities.
"""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import pandas as pd

__all__ = ["SupportAuditError", "audit_support"]

_REQUIRED_COLUMNS = (
    "context_id",
    "domain",
    "station",
    "instrument",
    "master_sample_id",
    "observation_uid",
    "true_label",
)

_DOMAIN_KEYS = ("domain", "station", "instrument")
_CONTEXT_IDENTITY = ("domain", "station", "instrument")
_OBSERVATION_IDENTITY = ("master_sample_id", "station", "instrument", "true_label")
_MASTER_IDENTITY = ("station", "true_label")

_RISK_COLUMNS = (
    "domain",
    "station",
    "instrument",
    "class_cells",
    "expected_empty_class_cells",
    "minimum_cell_empty_probability",
    "maximum_cell_empty_probability",
)


class SupportAuditError(ValueError):
    """Raised when a support table violates a structural requirement."""


def _raise(reason: str) -> NoReturn:
    raise SupportAuditError(reason)


def _validate(rows: pd.DataFrame) -> None:
    if not isinstance(rows, pd.DataFrame):
        _raise("input_not_dataframe")
    if rows.columns.duplicated().any():
        _raise("duplicate_columns")
    if any(column not in rows.columns for column in _REQUIRED_COLUMNS):
        _raise("missing_columns")
    if rows.shape[0] == 0:
        _raise("empty_frame")

    for column in _REQUIRED_COLUMNS:
        series = rows[column]
        if series.isna().any():
            _raise("missing_value")
        if not series.map(lambda value: isinstance(value, str)).all():
            _raise("non_string_value")
        trimmed = series.map(str.strip)
        if (trimmed == "").any():
            _raise("empty_value")
        if (trimmed != series).any():
            _raise("untrimmed_value")

    if rows.duplicated(subset=["context_id", "observation_uid"]).any():
        _raise("duplicate_observation_rows")

    observation_identity = rows.groupby("observation_uid")[
        list(_OBSERVATION_IDENTITY)
    ].nunique()
    if (observation_identity > 1).to_numpy().any():
        _raise("contradictory_observation_identity")

    master_identity = rows.groupby("master_sample_id")[list(_MASTER_IDENTITY)].nunique()
    if (master_identity > 1).to_numpy().any():
        _raise("contradictory_master_identity")

    context_identity = rows.groupby("context_id")[list(_CONTEXT_IDENTITY)].nunique()
    if (context_identity > 1).to_numpy().any():
        _raise("contradictory_context_identity")

    domain_identity = rows.groupby("domain")[["station", "instrument"]].nunique()
    if (domain_identity > 1).to_numpy().any():
        _raise("contradictory_domain_identity")


def audit_support(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Audit the support structure of a spectrum table and return five tables.

    Parameters
    ----------
    rows:
        A :class:`pandas.DataFrame` with the columns ``context_id``, ``domain``,
        ``station``, ``instrument``, ``master_sample_id``, ``observation_uid``
        and ``true_label``. Every required value must be a nonempty, already
        trimmed Python string. Extra columns are ignored and the input is never
        mutated.

    Returns
    -------
    dict of str to pandas.DataFrame
        The deterministic tables ``domains``, ``class_cell_histogram``,
        ``master_domain_histogram``, ``instrument_domains`` and
        ``empty_class_risk``, ordered by sorted public grouping keys.

    Raises
    ------
    SupportAuditError
        If the input violates any structural requirement. The exception message
        is a stable reason code and never embeds private input values.

    Notes
    -----
    See the module docstring for the meaning of missing cells and why this audit
    does not authorize changing the frozen equal-context/class estimand.
    """
    _validate(rows)
    work = rows.loc[:, list(_REQUIRED_COLUMNS)].copy()
    domain_keys = list(_DOMAIN_KEYS)

    context_meta = (
        work.loc[:, ["context_id", *domain_keys]]
        .drop_duplicates(subset=["context_id"])
        .reset_index(drop=True)
    )

    cells = (
        work.groupby(["context_id", "true_label"], as_index=False)["master_sample_id"]
        .nunique()
        .rename(columns={"master_sample_id": "masters_in_cell"})
    )
    cells = cells.merge(context_meta, on="context_id", how="left")

    domains = work.groupby(domain_keys, as_index=False).agg(
        contexts=("context_id", "nunique"),
        unique_spectra=("observation_uid", "nunique"),
        unique_masters=("master_sample_id", "nunique"),
        classes=("true_label", "nunique"),
    )
    singleton = (cells["masters_in_cell"] == 1).astype("int64")
    cell_summary = (
        cells.assign(_singleton=singleton)
        .groupby(domain_keys, as_index=False)
        .agg(
            class_cells=("masters_in_cell", "size"),
            singleton_class_cells=("_singleton", "sum"),
        )
    )
    domains = domains.merge(cell_summary, on=domain_keys, how="left")
    domains["class_cells"] = domains["class_cells"].fillna(0).astype("int64")
    domains["singleton_class_cells"] = (
        domains["singleton_class_cells"].fillna(0).astype("int64")
    )
    domains = domains.sort_values(domain_keys).reset_index(drop=True)

    class_cell_histogram = (
        cells.groupby("masters_in_cell", as_index=False)
        .size()
        .rename(columns={"size": "cells"})
        .sort_values("masters_in_cell")
        .reset_index(drop=True)
    )

    master_domains = work.groupby("master_sample_id", as_index=False)["domain"].nunique()
    master_domain_histogram = (
        master_domains.rename(columns={"domain": "domains_per_master"})
        .groupby("domains_per_master", as_index=False)
        .size()
        .rename(columns={"size": "masters"})
        .sort_values("domains_per_master")
        .reset_index(drop=True)
    )

    instrument_domains = (
        work.groupby("instrument", as_index=False)
        .agg(domains=("domain", "nunique"), stations=("station", "nunique"))
        .sort_values("instrument")
        .reset_index(drop=True)
    )

    class_pool = (
        work.groupby(["domain", "true_label"], as_index=False)["master_sample_id"]
        .nunique()
        .rename(columns={"master_sample_id": "pool_masters"})
    )
    cell_members = work.groupby(
        ["domain", "context_id", "true_label"], as_index=False
    )["master_sample_id"].nunique()
    cell_members = cell_members.rename(columns={"master_sample_id": "cell_masters"})
    cell_members = cell_members.merge(class_pool, on=["domain", "true_label"], how="left")

    empty_probability = (
        1.0 - cell_members["cell_masters"] / cell_members["pool_masters"]
    ) ** cell_members["pool_masters"]
    probability_values = empty_probability.to_numpy(dtype="float64")
    if not np.isfinite(probability_values).all() or not (
        (probability_values >= 0.0) & (probability_values <= 1.0)
    ).all():
        _raise("non_finite_probability")
    cell_members = cell_members.assign(empty_probability=empty_probability)

    empty_class_risk = cell_members.groupby("domain", as_index=False).agg(
        class_cells=("empty_probability", "size"),
        expected_empty_class_cells=("empty_probability", "sum"),
        minimum_cell_empty_probability=("empty_probability", "min"),
        maximum_cell_empty_probability=("empty_probability", "max"),
    )
    domain_meta = (
        context_meta.loc[:, domain_keys]
        .drop_duplicates(subset=["domain"])
        .reset_index(drop=True)
    )
    empty_class_risk = empty_class_risk.merge(domain_meta, on="domain", how="left")
    empty_class_risk = (
        empty_class_risk.loc[:, list(_RISK_COLUMNS)]
        .sort_values(domain_keys)
        .reset_index(drop=True)
    )

    return {
        "domains": domains,
        "class_cell_histogram": class_cell_histogram,
        "master_domain_histogram": master_domain_histogram,
        "instrument_domains": instrument_domains,
        "empty_class_risk": empty_class_risk,
    }
