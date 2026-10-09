import csv
import io
import json
import re

import pytest

from atlas_sers.visualization.p08_f03_render import (
    DOMAIN_KEYS,
    EFFECT_KEYS,
    ENDPOINTS,
    ESTIMANDS,
    MODELS,
    POLICIES,
    RESEARCH_QUESTION_ID,
    _canonical_sha,
    _is_primary,
    _panel_data,
    prepare_f03,
)

DOMAINS = [f"D{i:02d}" for i in range(1, 14)]
MODES = ("crossed", "master_only", "instrument_only")


def _contrast(endpoint, model, policy):
    return f"universal_policy::{policy}::{model}::{endpoint}"


def _effect(e, en, m, p):
    return {
        "estimand": e,
        "contrast_id": _contrast(en, m, p),
        "family_id": "F-" + e,
        "endpoint": en,
        "model_id": m,
        "policy_id": p,
        "available": True,
        "reason": None,
        "family_size": 20,
        "adjustment": "holm",
        "point_effect": 0.2,
        "domain_raw_p": 0.01,
        "domain_adjusted_p": 0.05,
        "instrument_raw_p": None,
        "instrument_adjusted_p": 0.5,
        "hierarchy_planned": 13,
        "hierarchy_defined": 10,
        "hierarchy_undefined": 3,
        "hierarchy_lower": 0.0,
        "hierarchy_upper": 0.4,
        "hierarchy_reason": None,
        "crossed_lower": 0.1,
        "crossed_upper": 0.3,
        "crossed_reason": None,
        "master_only_lower": None,
        "master_only_upper": None,
        "master_only_reason": "undefined weighted interval",
        "instrument_only_lower": 0.05,
        "instrument_only_upper": 0.35,
        "instrument_only_reason": None,
    }


def _domain(e, en, m, p, k, name):
    return {
        "estimand": e,
        "contrast_id": _contrast(en, m, p),
        "family_id": "F-" + e,
        "endpoint": en,
        "model_id": m,
        "policy_id": p,
        "domain": name,
        "station": f"S{k}",
        "instrument": f"I{k}",
        "point_effect": -0.5 + 0.05 * k,
        "contexts": 1,
        "unit_appearances": 2,
        "physical_masters": 1,
        "distinct_units": 1,
    }


def _fixture():
    eff, dom = [], []
    for e in ESTIMANDS:
        for en in ENDPOINTS:
            for m in MODELS:
                for p in POLICIES:
                    eff.append(_effect(e, en, m, p))
                    for k, name in enumerate(DOMAINS):
                        dom.append(_domain(e, en, m, p, k, name))
    return eff, dom


def _meta():
    return {
        "population": {
            "primary_spectra": 598,
            "held_spectra": 557,
            "masters": 69,
            "instruments": 10,
            "held_domains": 13,
            "contexts": 260,
        },
        "independent_unit": "physical_master",
        "selection": "source-only",
        "interval_caption": "95%",
        "interaction_caption": "none",
        "counts_reference": "counts",
        "display": "concise",
    }


def _prepared(eff, dom, meta=None):
    semantic = {"f03_effects": eff, "f03_domains": dom, "metadata": meta or _meta()}
    return {"semantic": semantic, "semantic_sha256": _canonical_sha(semantic), "manifest": {}}


def test_full_grid_and_panels():
    out = prepare_f03(_prepared(*_fixture()))
    sem = out["semantic"]
    assert len(sem["f03_effects"]) == 40 and len(sem["f03_domains"]) == 520
    assert len(out["panels"]) == 12
    assert len({p["slug"] for p in out["panels"]}) == 12
    for e in ESTIMANDS:
        for en in ENDPOINTS:
            for mode in MODES:
                panel = _panel_data(sem, e, en, mode)
                assert len(panel["rows"]) == 10
                assert all(len(r["domains"]) == 13 for r in panel["rows"])


def test_rectangular_csv_payload():
    out = prepare_f03(_prepared(*_fixture()))
    for panel in out["panels"]:
        rows = list(csv.reader(io.StringIO(panel["csv"])))
        header = rows[0]
        assert header[:3] == ["rowtype", "semantic_sha256", "interval_mode"]
        assert len(header) == len(set(header))
        assert all(len(r) == len(header) for r in rows)
        overall = [r for r in rows[1:] if r[0] == "overall"]
        domain_rows = [r for r in rows[1:] if r[0] == "domain"]
        assert len(overall) == 10
        assert len(domain_rows) == 130
        assert all(r[1] == out["semantic_sha256"] for r in rows[1:])
        assert all(r[2] == panel["slug"].rsplit("-", 1)[-1] for r in rows[1:])


def test_manifest_and_formats():
    prep = _prepared(*_fixture())
    out = prepare_f03(prep)
    sem = out["semantic"]
    assert sem["figure_id"] == "P08-F03"
    assert sem["research_question_id"] == RESEARCH_QUESTION_ID
    assert sem["schema_version"]
    assert sem["source_semantic_sha256"] == _canonical_sha(prep["semantic"])
    manifest = out["manifest"]
    assert manifest["status"] == "prepared"
    assert manifest["figure_id"] == "P08-F03"
    assert manifest["research_question_id"] == RESEARCH_QUESTION_ID
    assert manifest["source_semantic_sha256"] == _canonical_sha(prep["semantic"])
    assert manifest["semantic_sha256"] == out["semantic_sha256"]
    assert manifest["reviewed"] is False and manifest["published"] is False
    for panel in out["panels"]:
        assert panel["semantic_sha256"] == out["semantic_sha256"]
        assert "***" not in panel["tex"] and "***" not in panel["html"]
        assert "\\includegraphics" not in panel["tex"]
        assert "http://www.w3.org/2000/svg" in panel["html"]
        stripped = panel["html"].replace("http://www.w3.org/2000/svg", "")
        assert "http://" not in stripped and "https://" not in stripped
        assert "<svg" in panel["html"] and "<table" in panel["html"]
        assert "font-size:16px" in panel["html"]
        assert "font-size:14px" in panel["html"]
        assert 'type="application/json"' in panel["html"]
        assert "panel-data-" in panel["html"]
        assert "\\begin{document}" in panel["tex"]
        assert "\\documentclass[border=0pt]{standalone}" in panel["tex"]
        assert "181.86mm" in panel["tex"]
        assert "\\fontsize{6}" not in panel["tex"]
        assert "\\fontsize{8}{10}" in panel["tex"]
        if panel["slug"].endswith("master_only"):
            assert "interval NA" in panel["tex"] and "interval NA" in panel["html"]
        assert "data-tip=" in panel["html"] and 'tabindex="0"' in panel["html"]
        assert "universal" in panel["html"]


def test_visible_scope_and_payload_escaping():
    eff, dom = _fixture()
    eff[0]["reason"] = "bad </script> tag"
    out = prepare_f03(_prepared(eff, dom))
    html = out["panels"][0]["html"]
    assert "<strong>Endpoint:</strong>" in html
    assert "<strong>Estimand:</strong>" in html
    assert "<strong>Interval mode:</strong>" in html
    assert "<strong>Scope:</strong>" in html
    assert html.count("</script>") == 2
    block = re.search(r'<script type="application/json"[^>]*>(.*?)</script>', html)
    assert block and "</script>" not in block.group(1)
    assert json.loads(block.group(1))["rows"][0]["effect"]["reason"] == "bad </script> tag"


def test_primary_tag_logic():
    assert _is_primary("equal_context", "crossed") is True
    assert _is_primary("equal_context", "master_only") is False
    assert _is_primary("other_estimand", "crossed") is False


def test_actual_contrast_ids():
    out = prepare_f03(_prepared(*_fixture()))
    effects = out["semantic"]["f03_effects"]
    for row in effects:
        assert row["contrast_id"] == (
            f"universal_policy::{row['policy_id']}::{row['model_id']}::{row['endpoint']}"
        )
    for e in ESTIMANDS:
        ids = {r["contrast_id"] for r in effects if r["estimand"] == e}
        assert len(ids) == 20
    grouped = {}
    for row in effects:
        grouped.setdefault((row["endpoint"], row["model_id"], row["policy_id"]), set()).add(
            row["contrast_id"]
        )
    for ids in grouped.values():
        assert len(ids) == 1


def test_na_and_outside_point_retained():
    eff, dom = _fixture()
    eff[0]["point_effect"] = 0.95
    eff[0]["crossed_lower"], eff[0]["crossed_upper"] = 0.0, 0.1
    out = prepare_f03(_prepared(eff, dom))
    row = out["semantic"]["f03_effects"][0]
    assert row["point_effect"] == 0.95
    panel = _panel_data(out["semantic"], row["estimand"], row["endpoint"], "master_only")
    match = [
        r
        for r in panel["rows"]
        if r["model_id"] == row["model_id"] and r["policy_id"] == row["policy_id"]
    ][0]
    assert match["lower"] is None and match["upper"] is None and match["reason"]


def test_na_reason_kept_out_of_native_header():
    out = prepare_f03(_prepared(*_fixture()))
    panel = out["panels"][-1]
    assert "undefined weighted interval" not in panel["tex"]
    assert "undefined weighted interval" in panel["csv"]
    assert "undefined weighted interval" in panel["html"]


def test_refuses_duplicate_and_missing_grid():
    eff, dom = _fixture()
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff + [dict(eff[0])], dom))
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff[:-1], dom))
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff, dom[:-1]))


def test_refuses_wrong_or_fake_contrast():
    eff, dom = _fixture()
    eff[0]["contrast_id"] = "fake::endpoint-only"
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff, dom))


def test_refuses_domain_identity_and_overall_mismatch():
    eff, dom = _fixture()
    target = next(
        r
        for r in dom
        if r["domain"] == DOMAINS[0]
        and r["estimand"] == ESTIMANDS[0]
        and r["endpoint"] == ENDPOINTS[0]
        and r["model_id"] == MODELS[0]
        and r["policy_id"] == POLICIES[0]
    )
    target["station"] = "OTHER"
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff, dom))
    eff, dom = _fixture()
    dom[0]["family_id"] = "WRONG"
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff, dom))


def test_refuses_bad_scalar_types():
    eff, dom = _fixture()
    eff[0]["available"] = 1
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff, dom))
    eff, dom = _fixture()
    eff[0]["available"] = False
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff, dom))
    eff, dom = _fixture()
    eff[0]["family_size"] = 20.0
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff, dom))
    eff, dom = _fixture()
    eff[0]["crossed_reason"] = {"nested": 1}
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff, dom))


def test_refuses_bad_hash_and_projects_metadata():
    eff, dom = _fixture()
    bad = _prepared(eff, dom)
    bad["semantic_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        prepare_f03(bad)
    meta = _meta()
    meta["private"] = "secret"
    meta["population"]["private"] = 1
    out = prepare_f03(_prepared(eff, dom, meta=meta))
    assert "private" not in out["semantic"]["metadata"]
    assert "private" not in out["semantic"]["metadata"]["population"]
    assert EFFECT_KEYS[0] == "estimand" and DOMAIN_KEYS[0] == "estimand"


def test_refuses_wrong_metadata_identity():
    eff, dom = _fixture()
    meta = _meta()
    meta["population"]["primary_spectra"] = 597
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff, dom, meta=meta))
    meta = _meta()
    meta["independent_unit"] = "spectrum"
    with pytest.raises(ValueError):
        prepare_f03(_prepared(eff, dom, meta=meta))
