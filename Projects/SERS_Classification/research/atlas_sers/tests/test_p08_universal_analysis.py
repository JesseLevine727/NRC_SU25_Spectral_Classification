"""Compact invented-fixture tests for the P08 universal analysis driver.

No field data, no files, no network, CPU only.  The full-panel draw count is
monkeypatched to a small value; all identities, panels, probabilities and
hierarchy supports are synthetic.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from atlas_sers.evaluation import p08_universal_analysis as analysis
from atlas_sers.evaluation import p08_universal_units as units
from atlas_sers.evaluation.p06p11_hierarchy import hierarchical_draws
from atlas_sers.evaluation.p06p11_inference import compile_pair

_VOCABULARY = ("A", "B", "C")

_REGISTERED_SPECS = (
    ("c1", 1, "m1", "A", "o1"),
    ("c1", 1, "m2", "B", "o2"),
    ("c2", 2, "m3", "A", "o3"),
    ("c2", 2, "m4", "B", "o4"),
    ("c3", 3, "m5", "A", "o5"),
    ("c3", 3, "m6", "B", "o6"),
    ("c4", 4, "m7", "A", "o7"),
    ("c4", 4, "m8", "B", "o8"),
)

GLOBAL_MASTERS = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]
GLOBAL_INSTRUMENTS = ["i1"]

_POLICY_SHIFT = {"PP-U-MIN": 0, "PP-U-SG": 1, "PP-U-ARPLS": 2}
# D0-M and P05-SELECTED deliberately coincide; the registry must keep both.
_MODEL_SHIFT = {
    "C-RBF-SVM": 0,
    "C-RANDOM-FOREST": 1,
    "C-EXTRA-TREES": 2,
    "D0-M": 0,
    "P05-SELECTED": 0,
}


def _registered_frame():
    rows = []
    for context_id, fold, master, label, observation in _REGISTERED_SPECS:
        rows.append(
            {
                "context_id": context_id,
                "domain": "d1",
                "station": "s1",
                "instrument": "i1",
                "outer_repeat": 1,
                "outer_fold": fold,
                "observation_uid": observation,
                "master_sample_id": master,
                "true_label": label,
                "class_vocabulary": _VOCABULARY,
            }
        )
    return pd.DataFrame(rows)


def _predictions_frame(registered):
    rows = []
    for record in registered.itertuples(index=False):
        vocabulary = tuple(record.class_vocabulary)
        true_index = vocabulary.index(record.true_label)
        for policy_id in units.POLICIES:
            for model_id in units.MODELS:
                choice = (
                    true_index + _POLICY_SHIFT[policy_id] + _MODEL_SHIFT[model_id]
                ) % 3
                probabilities = [0.0, 0.0, 0.0]
                probabilities[choice] = 0.8
                others = [index for index in range(3) if index != choice]
                probabilities[others[0]] = 0.1
                probabilities[others[1]] = 0.1
                rows.append(
                    {
                        "context_id": record.context_id,
                        "policy_id": policy_id,
                        "model_id": model_id,
                        "observation_uid": record.observation_uid,
                        "master_sample_id": record.master_sample_id,
                        "instrument": record.instrument,
                        "station": record.station,
                        "true_label": record.true_label,
                        "class_vocabulary": vocabulary,
                        "probability_0": probabilities[0],
                        "probability_1": probabilities[1],
                        "probability_2": probabilities[2],
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panels_fixture():
    registered = _registered_frame()
    predictions = _predictions_frame(registered)
    return units.build_units(predictions, registered)


def _compile_design(rows):
    frame = pd.DataFrame(
        rows,
        columns=list(analysis._ID_COLUMNS) + ["correct_model", "correct_reference"],
    )
    return compile_pair(frame)


def _row(context, domain, station, instrument, master, label, unit):
    return {
        "context_id": context,
        "domain": domain,
        "station": station,
        "instrument": instrument,
        "master_sample_id": master,
        "unit_id": unit,
        "true_label": label,
        "correct_model": 0,
        "correct_reference": 0,
    }


def test_registry_is_finite_deterministic_and_complete():
    registry = analysis.contrast_registry()
    assert isinstance(registry, list)
    assert len(registry) == 52
    assert registry == analysis.contrast_registry()

    families = {}
    for entry in registry:
        assert isinstance(entry["contrast_id"], str) and entry["contrast_id"]
        families[entry["family_id"]] = families.get(entry["family_id"], 0) + 1
    assert families == {"universal_policy": 20, "policy_model_interaction": 32}

    available = [entry for entry in registry if entry["available"]]
    unavailable = [entry for entry in registry if not entry["available"]]
    assert len(available) == 44
    assert len(unavailable) == 8
    assert all(entry["reason"] == analysis.QC_REASON for entry in unavailable)
    assert all(entry["policy_id"] == "PP-QC-SRC" for entry in unavailable)
    assert analysis.QC_POLICY == "PP-QC-SRC"

    effects = [entry for entry in registry if entry["family_id"] == "universal_policy"]
    assert all(entry["coefficients"] == (1.0, -1.0) for entry in effects)
    assert all(entry["procedure_labels"][1][0] == "PP-U-MIN" for entry in effects)

    interactions = [
        entry for entry in registry if entry["family_id"] == "policy_model_interaction"
    ]
    assert all(
        entry["coefficients"] == (1.0, -1.0, -1.0, 1.0) for entry in interactions
    )

    hand = next(
        entry
        for entry in interactions
        if entry["available"]
        and entry["policy_id"] == "PP-U-SG"
        and entry["deep_model_id"] == "D0-M"
        and entry["comparator_model_id"] == "C-RBF-SVM"
        and entry["endpoint"] == "M01"
    )
    assert hand["procedure_labels"] == (
        ("PP-U-SG", "D0-M", "M01"),
        ("PP-U-MIN", "D0-M", "M01"),
        ("PP-U-SG", "C-RBF-SVM", "M01"),
        ("PP-U-MIN", "C-RBF-SVM", "M01"),
    )

    deeps = {entry["deep_model_id"] for entry in interactions}
    assert deeps == {"D0-M", "P05-SELECTED"}
    assert len([e for e in interactions if e["deep_model_id"] == "P05-SELECTED"]) == 16
    assert (
        len([e for e in interactions if e["comparator_model_id"] == "C-EXTRA-TREES"])
        == 8
    )
    assert not any(
        "C-EXTRA-TREES" in {e["model_id"], e["deep_model_id"], e["comparator_model_id"]}
        for e in unavailable
    )


def test_analyze_panel_integrated(panels_fixture, monkeypatch):
    panels = copy.deepcopy(panels_fixture)
    monkeypatch.setattr(analysis, "DRAW_COUNT", 8)

    real = analysis.analyze_contrast
    calls = []

    def spy(rows, columns, coefficients, **kwargs):
        calls.append(
            {
                "columns": tuple(columns),
                "coefficients": tuple(coefficients),
                "master_weights": kwargs["master_weights"],
                "instrument_weights": kwargs["instrument_weights"],
                "global_masters": kwargs["global_masters"],
                "global_instruments": kwargs["global_instruments"],
                "hierarchical_draws": kwargs["hierarchical_draws"],
                "hierarchical_seed": kwargs["hierarchical_seed"],
            }
        )
        return real(rows, columns, coefficients, **kwargs)

    monkeypatch.setattr(analysis, "analyze_contrast", spy)

    result = analysis.analyze_panel(
        panels,
        global_masters=GLOBAL_MASTERS,
        global_instruments=GLOBAL_INSTRUMENTS,
        domain_families={"d1": "unknown"},
    )

    assert len(calls) == 88
    first = calls[0]
    assert all(call["master_weights"] is first["master_weights"] for call in calls)
    assert all(
        call["instrument_weights"] is first["instrument_weights"] for call in calls
    )
    assert all(
        np.array_equal(call["master_weights"], first["master_weights"])
        for call in calls
    )
    assert all(
        np.array_equal(call["instrument_weights"], first["instrument_weights"])
        for call in calls
    )
    assert all(call["hierarchical_seed"] == 2026093003 for call in calls)
    assert all(call["hierarchical_draws"] == 8 for call in calls)
    assert all(call["global_masters"] == list(GLOBAL_MASTERS) for call in calls)
    assert all(call["global_instruments"] == list(GLOBAL_INSTRUMENTS) for call in calls)

    # All 15 policy/model panels are summarized for both endpoints/estimands.
    assert set(result["metrics"]) == {"equal_context", "pooled_four_fold"}
    for estimand in result["metrics"]:
        assert set(result["metrics"][estimand]) == set(units.POLICIES)
        for policy_id in result["metrics"][estimand]:
            assert set(result["metrics"][estimand][policy_id]) == {"M01", "M06"}

    # Distinct estimator keys, never merged.
    assert set(result["contrasts"]) == {"equal_context", "pooled_four_fold"}
    assert (
        result["contrasts"]["equal_context"]
        is not result["contrasts"]["pooled_four_fold"]
    )

    # Unknown families are delegated to the engine and never pooled.
    available_id = next(
        entry["contrast_id"]
        for entry in result["registry"]
        if entry["available"] and entry["endpoint"] == "M01"
    )
    assert (
        result["contrasts"]["equal_context"][available_id]["descriptive"][
            "leave_one_family"
        ]
        == {}
    )

    summary = result["contrast_summary"]
    assert len(summary) == 104
    assert set(summary["estimand"]) == {"equal_context", "pooled_four_fold"}
    assert (summary["family_id"] == "universal_policy").sum() == 40
    assert (summary["family_id"] == "policy_model_interaction").sum() == 64
    assert summary["available"].eq(False).sum() == 16
    assert (summary["family_size"] == 20).sum() == 40
    assert (summary["family_size"] == 32).sum() == 64

    # Known-answer numerics on the invented fixture: MIN D0-M BA=1, SG D0-M
    # BA=0, MIN Extra Trees BA=0, SG Extra Trees BA=1, so the SG D0-vs-ET
    # interaction is -2 and the SG D0-M policy effect is -1.
    for estimand in ("equal_context", "pooled_four_fold"):
        nested = result["contrasts"][estimand]
        assert nested["universal_policy::PP-U-SG::D0-M::M01"]["point"][
            "overall"
        ] == pytest.approx(-1.0)
        assert nested["universal_policy::PP-U-SG::P05-SELECTED::M01"]["point"][
            "overall"
        ] == pytest.approx(-1.0)
        assert nested["policy_model_interaction::PP-U-SG::D0-M::C-EXTRA-TREES::M01"][
            "point"
        ]["overall"] == pytest.approx(-2.0)

    # The coincident D0-M/P05-SELECTED strategy names keep separate slots.
    universal_ids = {
        entry["contrast_id"]
        for entry in result["registry"]
        if entry["family_id"] == "universal_policy"
    }
    assert "universal_policy::PP-U-SG::D0-M::M01" in universal_ids
    assert "universal_policy::PP-U-SG::P05-SELECTED::M01" in universal_ids

    # Every future-QC slot per estimand keeps missing raw and adjusted p.
    unavailable_ids = [
        entry["contrast_id"] for entry in result["registry"] if not entry["available"]
    ]
    assert len(unavailable_ids) == 8
    for estimand in ("equal_context", "pooled_four_fold"):
        for contrast_id in unavailable_ids:
            adjustment = result["contrasts"][estimand][contrast_id][
                "sensitivity_adjustment"
            ]
            assert adjustment["domain_raw_p"] is None
            assert adjustment["domain_adjusted_p"] is None
            assert adjustment["instrument_raw_p"] is None
            assert adjustment["instrument_adjusted_p"] is None

    # Independently recompute Holm and confirm every family slot remains.
    for estimand in ("equal_context", "pooled_four_fold"):
        for family_id, expected_size in (
            ("universal_policy", 20),
            ("policy_model_interaction", 32),
        ):
            family = [
                entry for entry in result["registry"] if entry["family_id"] == family_id
            ]
            assert len(family) == expected_size
            raw = []
            for entry in family:
                nested = result["contrasts"][estimand][entry["contrast_id"]]
                sign = nested.get("sign_sensitivity")
                if entry["available"] and sign:
                    raw.append(sign["domain"]["p_descriptive"])
                else:
                    raw.append(None)
            expected = analysis.holm_fixed_family(raw, family_size=expected_size)
            assert len(expected) == expected_size
            for entry, adjusted in zip(family, expected, strict=True):
                slot = result["contrasts"][estimand][entry["contrast_id"]][
                    "sensitivity_adjustment"
                ]
                assert slot["family_size"] == expected_size
                if adjusted is None:
                    assert slot["domain_adjusted_p"] is None
                else:
                    assert slot["domain_adjusted_p"] == pytest.approx(adjusted)

    weights = result["weights"]
    expected_master = np.random.Generator(np.random.PCG64(2026093001)).exponential(
        1.0, size=(8, len(GLOBAL_MASTERS))
    )
    expected_instrument = np.random.Generator(np.random.PCG64(2026093002)).exponential(
        1.0, size=(8, len(GLOBAL_INSTRUMENTS))
    )
    assert np.array_equal(weights["master_weights"], expected_master)
    assert np.array_equal(weights["instrument_weights"], expected_instrument)
    assert weights["number_draws"] == 8
    assert weights["global_masters"] == list(GLOBAL_MASTERS)
    assert weights["global_instruments"] == list(GLOBAL_INSTRUMENTS)

    for label in analysis._BOUNDARY_LABELS:
        assert result["boundary"][label] is True
        assert result[label] is True

    assert set(result["hierarchy_support"]) == {"equal_context", "pooled_four_fold"}

    # Inputs are unmodified.
    assert panels["equal_context"]["PP-U-MIN"]["M01"].equals(
        panels_fixture["equal_context"]["PP-U-MIN"]["M01"]
    )


def test_exact_support_mismatch_refused_before_engine(panels_fixture, monkeypatch):
    panels = copy.deepcopy(panels_fixture)
    frame = panels["equal_context"]["PP-U-SG"]["M01"]
    target = frame.index[frame["model_id"] == "C-RBF-SVM"][0]
    frame.loc[target, "unit_id"] = "corrupted-unit"

    monkeypatch.setattr(analysis, "DRAW_COUNT", 8)
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        raise AssertionError("analyze_contrast must not run")

    monkeypatch.setattr(analysis, "analyze_contrast", spy)
    with pytest.raises(ValueError):
        analysis.analyze_panel(
            panels,
            global_masters=GLOBAL_MASTERS,
            global_instruments=GLOBAL_INSTRUMENTS,
        )
    assert calls == []


def test_corrupt_later_estimand_frame_refused_before_engine(
    panels_fixture, monkeypatch
):
    panels = copy.deepcopy(panels_fixture)
    frame = panels["pooled_four_fold"]["PP-U-ARPLS"]["M06"]
    target = frame.index[frame["model_id"] == "C-RBF-SVM"][0]
    frame.loc[target, "unit_id"] = "corrupted-unit"

    monkeypatch.setattr(analysis, "DRAW_COUNT", 8)
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        raise AssertionError("analyze_contrast must not run")

    monkeypatch.setattr(analysis, "analyze_contrast", spy)
    with pytest.raises(ValueError):
        analysis.analyze_panel(
            panels,
            global_masters=GLOBAL_MASTERS,
            global_instruments=GLOBAL_INSTRUMENTS,
        )
    assert calls == []


def test_hierarchy_support_matches_inherited_sparse_design():
    rows = [
        _row("c1", "d1", "s1", "i1", "m1", "A", "u1"),
        _row("c1", "d1", "s1", "i1", "m3", "B", "u3"),
        _row("c2", "d1", "s1", "i1", "m2", "A", "u2"),
        _row("c2", "d1", "s1", "i1", "m4", "B", "u4"),
        _row("c3", "d2", "s2", "i2", "m5", "A", "u5"),
        _row("c3", "d2", "s2", "i2", "m6", "B", "u6"),
    ]
    design = _compile_design(rows)
    replay = analysis.hierarchy_support(design, draws=256, seed=2026093003)
    inherited = hierarchical_draws(design, draws=256, seed=2026093003)

    assert np.array_equal(replay["sampled_domains"], inherited["sampled_domains"])
    assert np.array_equal(replay["empty_cells"], inherited["empty_cells"])
    assert np.array_equal(
        replay["undefined_domain_occurrences"],
        inherited["undefined_domain_occurrences"],
    )
    assert np.array_equal(replay["affected_domains"], inherited["affected_domains"])
    assert replay["affected_domain_flags"].shape == (256, 2)

    # A represented context/class cell genuinely becomes empty, so the replay
    # must observe undefined domain occurrences.
    assert replay["undefined_draws"] >= 1
    assert replay["affected_domain_names"] == ["d1"]
    assert replay["affected_draws_by_domain"]["d1"] >= 1
    assert replay["affected_draws_by_domain"]["d2"] == 0
    assert replay["undefined_occurrences_by_domain"]["d2"] == 0


def test_hierarchy_support_complete_fixture_has_no_undefined_draws():
    rows = [
        _row("c1", "d1", "s1", "i1", "m1", "A", "u1"),
        _row("c2", "d1", "s1", "i1", "m2", "B", "u2"),
        _row("c3", "d2", "s2", "i2", "m3", "A", "u3"),
        _row("c4", "d2", "s2", "i2", "m4", "B", "u4"),
    ]
    design = _compile_design(rows)
    replay = analysis.hierarchy_support(design, draws=64, seed=2026093003)
    inherited = hierarchical_draws(design, draws=64, seed=2026093003)

    assert np.array_equal(replay["sampled_domains"], inherited["sampled_domains"])
    assert np.array_equal(replay["empty_cells"], inherited["empty_cells"])
    assert np.array_equal(
        replay["undefined_domain_occurrences"],
        inherited["undefined_domain_occurrences"],
    )
    assert np.array_equal(replay["affected_domains"], inherited["affected_domains"])
    assert replay["undefined_draws"] == 0
    assert replay["affected_domain_names"] == []
