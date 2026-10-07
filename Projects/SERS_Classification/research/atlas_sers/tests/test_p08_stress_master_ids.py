"""P08-T243 tests: recorded master-ID type is preserved exactly.

The metadata-only stress score-support adapter must accept a recorded master ID
that is either an exact ``int`` (never ``bool``) or a nonempty stripped ``str``,
preserve the concrete value/type everywhere, and reject every other type or a
mixed-type catalog.  These tests take the public text fixture, invent integer
master IDs (``1000 + i``) and check that the integer type survives
normalisation, pooling and hashing without mutating the supplied data.

The fixture is reached through the public helper contract: ``support_for(None)``
returns ``(Scenario, SCORE_SUPPORT_CATALOG)``, ``Scenario.catalog`` is the
prediction catalog and ``build_memberships(Scenario)`` returns the raw
membership catalog.  No fallback guessing is performed.
"""

from __future__ import annotations

import copy

import pytest

from atlas_sers.evaluation.p08_perturbation_score_support import (
    INVALID,
    build_stress_score_support,
    validate_stress_score_support,
)
from atlas_sers.evaluation.p08_perturbation_scores import iter_stress_score_records
from tests.test_p08_perturbation_scores import build_memberships, support_for

builder = build_stress_score_support


def _default_pair():
    sc, _ = support_for(None)
    return sc.catalog, build_memberships(sc)


def _integer_memberships(memberships):
    masters = sorted({row["master_id"] for row in memberships["test_rows"]})
    mapping = {master: 1000 + index for index, master in enumerate(masters)}
    integer = copy.deepcopy(memberships)
    for row in integer["test_rows"]:
        row["master_id"] = mapping[row["master_id"]]
    return integer, mapping


def _observation_context_key(row):
    """Everything in a row except the recorded master ID and its label.

    Two rows sharing a master ID are only a genuine label conflict when they
    describe distinct observation/context keys rather than being an exact
    duplicate record.
    """

    return tuple(
        sorted((key, value) for key, value in row.items() if key not in ("master_id", "label"))
    )


def _find_cross_fold_pair(rows):
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            first, second = rows[i], rows[j]
            if (
                first["domain"] == second["domain"]
                and first["outer_repeat"] == second["outer_repeat"]
                and first["outer_fold"] != second["outer_fold"]
                and first["label"] == second["label"]
            ):
                return first, second
    return None


def test_integer_master_ids_preserved():
    catalog, memberships = _default_pair()
    integer, mapping = _integer_memberships(memberships)

    result = builder(prediction_catalog=copy.deepcopy(catalog), memberships=integer)

    assert result["global_master_ids"] == sorted(mapping.values())
    assert all(type(value) is int for value in result["global_master_ids"])
    assert all(type(value) is not bool for value in result["global_master_ids"])

    assert len(result["memberships"]["test_rows"]) == len(integer["test_rows"])
    for row in result["memberships"]["test_rows"]:
        assert type(row["master_id"]) is int

    for record in result["context_records"]:
        for unit in record["master_units"]:
            assert type(unit["master_id"]) is int


def test_integer_build_is_deterministic():
    catalog, memberships = _default_pair()
    integer, _ = _integer_memberships(memberships)

    first = builder(prediction_catalog=copy.deepcopy(catalog), memberships=copy.deepcopy(integer))
    second = builder(prediction_catalog=copy.deepcopy(catalog), memberships=copy.deepcopy(integer))

    assert first["catalog_sha256"] == second["catalog_sha256"]


def test_build_does_not_mutate_inputs():
    catalog, memberships = _default_pair()
    integer, _ = _integer_memberships(memberships)
    catalog_arg = copy.deepcopy(catalog)
    memberships_arg = copy.deepcopy(integer)

    builder(prediction_catalog=catalog_arg, memberships=memberships_arg)

    assert catalog_arg == catalog
    assert memberships_arg == integer


def test_integer_summary_matches_text_fixture():
    catalog, memberships = _default_pair()
    text = builder(
        prediction_catalog=copy.deepcopy(catalog), memberships=copy.deepcopy(memberships)
    )
    integer, _ = _integer_memberships(memberships)

    result = builder(prediction_catalog=copy.deepcopy(catalog), memberships=integer)

    assert result["summary"] == text["summary"]


def test_integer_ids_change_digest():
    catalog, memberships = _default_pair()
    text = builder(
        prediction_catalog=copy.deepcopy(catalog), memberships=copy.deepcopy(memberships)
    )
    integer, _ = _integer_memberships(memberships)

    result = builder(prediction_catalog=copy.deepcopy(catalog), memberships=integer)

    assert result["catalog_sha256"] != text["catalog_sha256"]


def test_integer_ids_sorted_numerically_not_lexicographically():
    # No weights are generated here: this is the raw membership catalog, and the
    # homogeneous integer IDs must keep their numeric ordering (2 < 10 < 100),
    # distinct from a future lexicographic bootstrap identity mapping.
    catalog, memberships = _default_pair()
    masters = sorted({row["master_id"] for row in memberships["test_rows"]})
    assert len(masters) >= 3

    chosen = [2, 10, 100] + [1000 + index for index in range(len(masters) - 3)]
    mapping = dict(zip(masters, chosen, strict=True))
    integer = copy.deepcopy(memberships)
    for row in integer["test_rows"]:
        row["master_id"] = mapping[row["master_id"]]

    result = builder(prediction_catalog=copy.deepcopy(catalog), memberships=integer)

    assert result["global_master_ids"] == sorted(chosen)
    assert result["global_master_ids"][:3] == [2, 10, 100]
    assert result["global_master_ids"] != sorted(chosen, key=str)
    assert all(type(value) is int for value in result["global_master_ids"])


def test_validate_and_lazy_iterator_accept_integer_catalog():
    catalog, memberships = _default_pair()
    integer, _ = _integer_memberships(memberships)
    result = builder(prediction_catalog=copy.deepcopy(catalog), memberships=integer)

    # The score-support iterator must accept the integer catalog.  Do not
    # consume the iterator: it must stay lazy and master-ID aware.
    iterator = iter_stress_score_records(result)
    assert iterator is not None

    snapshot = validate_stress_score_support(copy.deepcopy(result))
    assert snapshot["global_master_ids"] == result["global_master_ids"]
    assert all(type(value) is int for value in snapshot["global_master_ids"])


def test_reject_mixed_master_id_types():
    catalog, memberships = _default_pair()
    integer, _ = _integer_memberships(memberships)
    integer["test_rows"][0]["master_id"] = str(integer["test_rows"][0]["master_id"])

    with pytest.raises(ValueError) as excinfo:
        builder(prediction_catalog=copy.deepcopy(catalog), memberships=integer)

    assert str(excinfo.value) == INVALID


@pytest.mark.parametrize("bad_value", [True, False, 1.0, 2.5, None, "", "   "])
def test_reject_non_int_non_text_master_ids(bad_value):
    catalog, memberships = _default_pair()
    tampered = copy.deepcopy(memberships)
    tampered["test_rows"][0]["master_id"] = bad_value

    with pytest.raises(ValueError) as excinfo:
        builder(prediction_catalog=copy.deepcopy(catalog), memberships=tampered)

    assert str(excinfo.value) == INVALID
    assert repr(bad_value) not in str(excinfo.value)


def test_reject_label_conflict_for_same_integer_master():
    catalog, memberships = _default_pair()
    integer, _ = _integer_memberships(memberships)
    rows = integer["test_rows"]
    pair = next(
        (
            (rows[i], rows[j])
            for i in range(len(rows))
            for j in range(i + 1, len(rows))
            if rows[i]["context_id"] == rows[j]["context_id"]
            and rows[i]["observation_uid"] != rows[j]["observation_uid"]
        ),
        None,
    )
    assert pair is not None
    first, second = pair
    assert first["observation_uid"] != second["observation_uid"]
    assert first["context_id"] == second["context_id"]
    assert _observation_context_key(first) != _observation_context_key(second)
    second["master_id"] = first["master_id"]
    second["label"] = first["label"] + "_flip"

    with pytest.raises(ValueError) as excinfo:
        builder(prediction_catalog=copy.deepcopy(catalog), memberships=integer)

    assert str(excinfo.value) == INVALID
    assert "_flip" not in str(excinfo.value)


def test_reject_same_integer_master_across_folds():
    catalog, memberships = _default_pair()
    integer, _ = _integer_memberships(memberships)
    pair = _find_cross_fold_pair(integer["test_rows"])
    assert pair is not None

    first, second = pair
    second["master_id"] = first["master_id"]

    with pytest.raises(ValueError) as excinfo:
        builder(prediction_catalog=copy.deepcopy(catalog), memberships=integer)

    assert str(excinfo.value) == INVALID
