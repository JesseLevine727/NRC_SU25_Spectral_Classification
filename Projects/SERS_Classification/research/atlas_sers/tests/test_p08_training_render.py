"""Tests for the P08 U1 compact training-diagnostic renderer."""

from __future__ import annotations

import copy
import csv as csvmod
import json
import re
from io import StringIO

import pytest

from atlas_sers.visualization.p08_f02_render import _canonical_sha
from atlas_sers.visualization.p08_training_figure_data import prepare_training_diagnostics
from atlas_sers.visualization.p08_training_render import prepare_training_render
from tests import test_p08_training_figure_data as fixture


def _prepared():
    jobs, monitors = fixture._valid()
    return prepare_training_diagnostics(jobs, monitors)


def _rendered():
    return prepare_training_render(_prepared())


def _rows(panel):
    return list(csvmod.DictReader(StringIO(panel["csv"])))


def test_24_panels_fixed_order_and_types():
    out = _rendered()
    panels = out["panels"]
    assert len(panels) == 24
    keys = [(p["policy_id"], p["recipe"], p["stage"]) for p in panels]
    assert len(set(keys)) == 24
    assert keys[0] == ("PP-U-SG", "D0-M", "source_fit")
    assert keys[-1] == ("PP-U-ARPLS", "D3", "final_refit")
    for panel in panels:
        assert set(panel) == {
            "slug",
            "policy_id",
            "recipe",
            "stage",
            "semantic_sha256",
            "tex",
            "html",
            "csv",
        }
        assert panel["semantic_sha256"] == out["semantic_sha256"]
        assert panel["tex"].startswith("\\documentclass")
        assert "<svg" in panel["html"]
        assert panel["csv"].splitlines()[0].startswith("policy_id,")
        assert panel["semantic_sha256"] in panel["tex"]
        assert panel["semantic_sha256"] in panel["html"]


def test_attrition_n_runs_two_to_one_preserved():
    found = False
    for panel in _rendered()["panels"]:
        values = [
            int(r["n_runs_at_epoch"])
            for r in _rows(panel)
            if r["metric"] == "chemical_ce" and r["n_runs_at_epoch"]
        ]
        if 2 in values and 1 in values:
            assert values[-1] == 1
            found = True
            break
    assert found


def test_refit_validation_not_recorded_none_not_zero():
    seen = False
    for panel in _rendered()["panels"]:
        if panel["stage"] == "source_fit":
            continue
        rows = [r for r in _rows(panel) if r["metric"] in ("train_nll", "validation_nll")]
        if not rows:
            continue
        seen = True
        for row in rows:
            assert row["median"] == "" and row["q10"] == "" and row["q90"] == ""
            assert row["reason"] == "metric_not_recorded"
    assert seen


def test_17g_coordinate_and_csv_parity():
    for panel in _rendered()["panels"]:
        for row in _rows(panel):
            if row["q10"]:
                assert row["q10"] == format(float(row["q10"]), ".17g")
                assert row["q10"] in panel["tex"]
                return
    raise AssertionError("no finite q10 found")


def test_html_offline_handlers_and_no_cdn():
    for panel in _rendered()["panels"]:
        stripped = panel["html"].replace("http://www.w3.org/2000/svg", "")
        assert "http://" not in stripped and "https://" not in stripped
        assert "cdn" not in panel["html"].lower()
        assert "addEventListener" in panel["html"]
        if any(row["median"] for row in _rows(panel)):
            assert 'type="checkbox"' in panel["html"]


def test_json_decodes_and_matches_csv():
    for panel in _rendered()["panels"]:
        match = re.search(r'<script type="application/json">(.*?)</script>', panel["html"], re.S)
        assert match
        data = json.loads(match.group(1))
        assert data["csv"] == panel["csv"]
        assert data["semantic_sha256"] == panel["semantic_sha256"]
        assert data["group"]["stage"] == panel["stage"]


def _resealed(mutate):
    prepared = _prepared()
    mutate(prepared["semantic"])
    prepared["semantic_sha256"] = _canonical_sha(prepared["semantic"])
    return prepared


def test_tampered_hash_refused():
    prepared = _prepared()
    prepared["semantic_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        prepare_training_render(prepared)


def test_missing_group_refused():
    with pytest.raises(ValueError):
        prepare_training_render(_resealed(lambda s: s["groups"].pop()))


def test_duplicate_curve_row_refused():
    with pytest.raises(ValueError):
        prepare_training_render(_resealed(lambda s: s["curves"].append(dict(s["curves"][0]))))


def test_bad_finite_and_quantile_refused():
    def bad_finite(s):
        s["curves"][0]["median"] = None

    def bad_quantile(s):
        for c in s["curves"]:
            if c["q10"] is not None:
                if c["q10"] == c["q90"]:
                    c["q10"] = c["q90"] + 1.0
                else:
                    c["q10"], c["q90"] = c["q90"], c["q10"]
                return
        raise AssertionError("fixture has no finite quantiles to reverse")

    with pytest.raises(ValueError):
        prepare_training_render(_resealed(bad_finite))
    with pytest.raises(ValueError):
        prepare_training_render(_resealed(bad_quantile))


def test_extra_field_refused_and_no_input_mutation():
    prepared = _prepared()
    snapshot = copy.deepcopy(prepared["semantic"])
    prepare_training_render(prepared)
    assert prepared["semantic"] == snapshot

    def add_field(s):
        s["extra_field"] = True

    with pytest.raises(ValueError):
        prepare_training_render(_resealed(add_field))


def test_population_must_be_598_69_10():
    def tamper(s):
        s["population"]["masters"] = 70

    with pytest.raises(ValueError):
        prepare_training_render(_resealed(tamper))


def test_non_source_nll_finite_rejected():
    def tamper(s):
        for c in s["curves"]:
            if c["stage"] == "final_refit" and c["metric"] == "validation_nll":
                c["median"] = 0.5
                c["q10"] = 0.4
                c["q90"] = 0.6
                c["finite_count"] = c["n_runs_at_epoch"]
                c["undefined_count"] = 0
                c["reason"] = None
                return
        raise AssertionError("no final_refit validation_nll row to corrupt")

    with pytest.raises(ValueError):
        prepare_training_render(_resealed(tamper))


def test_common_axes_and_integer_run_ticks():
    out = _rendered()
    axes = out["semantic"]["definitions"]["axis"]
    for panel in out["panels"]:
        axis = axes[f"{panel['recipe']}|{panel['stage']}"]
        for sub in ("A", "B", "C", "D"):
            assert axis[sub]["yticks"]
            for tick in axis[sub]["yticks"]:
                if sub == "D":
                    assert float(tick).is_integer()
                if any(row["median"] for row in _rows(panel)):
                    assert format(float(tick), ".17g") in panel["tex"]


def test_native_html_okabe_ito_dashes_and_shapes():
    out = _rendered()
    hexes = {"blue": "0072B2", "vermillion": "D55E00", "green": "009E73", "purple": "CC79A7"}
    all_html = "".join(p["html"] for p in out["panels"])
    for value in hexes.values():
        assert f"#{value}" in all_html
    for panel in out["panels"]:
        for value in hexes.values():
            assert "{" + value + "}" in panel["tex"]
        if any(row["median"] for row in _rows(panel)):
            assert "stroke-dasharray" in panel["html"]
            assert "dash" in panel["tex"].lower()
    if 'data-series="total_loss"' in all_html:
        assert "<rect" in all_html
    if 'data-series="supcon_loss"' in all_html:
        assert "<polygon" in all_html


def test_csv_download_button_and_json_object():
    for panel in _rendered()["panels"]:
        html = panel["html"]
        assert 'class="p08-download"' in html
        assert "text/csv" in html
        match = re.search(r'<script type="application/json">(.*?)</script>', html, re.S)
        assert match
        data = json.loads(match.group(1))
        assert isinstance(data, dict)
        assert data["csv"] == panel["csv"]


def test_disabled_auxiliary_and_run_attrition_refusals():
    def disabled(s):
        for c in s["curves"]:
            if c["recipe"] == "D2" and c["metric"] == "supcon_loss":
                c.update(
                    median=1.0,
                    q10=1.0,
                    q90=1.0,
                    reason=None,
                    finite_count=c["n_runs_at_epoch"],
                    undefined_count=0,
                )

    with pytest.raises(ValueError, match="disabled auxiliary"):
        prepare_training_render(_resealed(disabled))

    def decreasing_then_increasing(s):
        for c in s["curves"]:
            if c["recipe"] == "D1" and c["epoch"] == 2:
                c.update(
                    n_runs_at_epoch=1,
                    finite_count=1 if c["median"] is not None else 0,
                    undefined_count=0 if c["median"] is not None else 1,
                )

    with pytest.raises(ValueError, match="nonincreasing"):
        prepare_training_render(_resealed(decreasing_then_increasing))


def test_scope_black_text_and_plot_labels():
    out = _rendered()
    for panel in out["panels"]:
        assert "RQ-S01 / Scope E" in panel["tex"]
        assert '<span style="color:' not in panel["html"]
        if any(row["median"] for row in _rows(panel)):
            assert ">epoch</text>" in panel["html"]
            assert 'transform="rotate(-90' in panel["html"]
            assert 'aria-hidden="true"' in panel["html"]
    assert "caption" in out["semantic"]["definitions"]
