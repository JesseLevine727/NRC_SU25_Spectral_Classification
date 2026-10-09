"""Synthetic tests for the pure P08-F01 renderer.

No field data, no fitting, no compilation and no rendering is performed here.
"""
# Literal-format assertions match the renderer's brace-preserving templates.
# ruff: noqa: UP031
import copy
import csv
import hashlib
import io
import json
import math
import re

import pytest

from atlas_sers.visualization.p08_f01_render import (
    ACTION_ORDER,
    AXIS_N,
    CAPTION,
    FIGURE_ID,
    RESEARCH_QUESTION_ID,
    STYLES,
    UNAVAILABLE_REASON,
    prepare_f01,
)


def _curve(offset=0.0):
    return [min(1.0, max(0.0, 0.5 + 0.25 * math.sin(i / 40.0) + offset))
            for i in range(AXIS_N)]


def _cell(cell_id, station, instrument, analyte, n_spectra, n_masters, held, available):
    if available:
        curves = {action: _curve(k * 0.001) for k, action in enumerate(ACTION_ORDER)}
        reason = ""
    else:
        curves = {}
        reason = UNAVAILABLE_REASON
    return {"cell_id": cell_id, "station": station, "instrument": instrument,
            "analyte": analyte, "n_spectra": n_spectra, "n_masters": n_masters,
            "held_comparison_domain": held, "available": available,
            "reason": reason, "curves": curves}


def _bundle():
    cells = [
        _cell("P08-F01-C001", "Alpha & Co", "Instr_#1", "Benzene_<x>", 7, 3, True, True),
        _cell("P08-F01-C002", "Alpha & Co", "Instr_#1", "Toluene", 5, 2, True, True),
        _cell("P08-F01-C003", "Beta", "Instr-2", "Pyridine </script> & <b>", 4, 1, False, False),
    ]
    return {"schema_version": "nato-sers-p08-f01-spectral-bundle-v1",
            "figure_id": "P08-F01",
            "axis_cm1": [400.0 + i for i in range(AXIS_N)],
            "action_order": list(ACTION_ORDER),
            "cells": cells,
            "caption": "Synthetic bounded caption."}


def _single_domain_bundle():
    cells = [
        _cell("P08-F01-C101", "Gamma", "Instr-9", "A1", 9, 4, True, True),
        _cell("P08-F01-C102", "Gamma", "Instr-9", "A2", 7, 3, True, True),
        _cell("P08-F01-C103", "Gamma", "Instr-9", "A3", 2, 1, True, False),
    ]
    return {"schema_version": "nato-sers-p08-f01-spectral-bundle-v1",
            "figure_id": "P08-F01",
            "axis_cm1": [400.0 + i for i in range(AXIS_N)],
            "action_order": list(ACTION_ORDER),
            "cells": cells,
            "caption": "Synthetic bounded caption."}


def _sha(semantic):
    payload = json.dumps(semantic, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rows(csv_text):
    return list(csv.DictReader(io.StringIO(csv_text)))


def test_stable_hash_no_mutation_and_domains():
    bundle = _bundle()
    snapshot = copy.deepcopy(bundle)
    first = prepare_f01(bundle)
    second = prepare_f01(bundle)
    assert bundle == snapshot
    assert first["semantic_sha256"] == second["semantic_sha256"]
    assert _sha(first["semantic"]) == first["semantic_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", first["semantic_sha256"])
    assert len(first["panels"]) == 2
    assert [p["slug"] for p in first["panels"]] == ["P08-F01-D01", "P08-F01-D02"]
    assert first["semantic"]["research_question_id"] == RESEARCH_QUESTION_ID
    assert first["semantic"]["figure_id"] == FIGURE_ID
    assert first["semantic"]["independent_unit"] == "physical master"
    assert first["panels"][0]["exploratory"] is False
    assert first["panels"][1]["exploratory"] is True
    assert first["manifest"]["reviewed"] is False
    assert first["manifest"]["published"] is False
    assert first["manifest"]["status"] == "prepared"


def test_all_points_styles_and_formats_agree():
    result = prepare_f01(_bundle())
    semantic = result["semantic"]
    assert len(semantic["axis_cm1"]) == AXIS_N
    assert semantic["action_order"] == list(ACTION_ORDER)
    for cell in semantic["cells"]:
        if cell["available"]:
            for action in ACTION_ORDER:
                values = cell["curves"][action]
                assert len(values) == AXIS_N
                assert all(0.0 <= v <= 1.0 for v in values)
    sha = result["semantic_sha256"]
    for panel in result["panels"]:
        assert set(panel) >= {"slug", "tex", "html", "csv"}
        assert sha in panel["tex"] and sha in panel["html"]
        assert panel["slug"] in panel["tex"] and panel["slug"] in panel["html"]
        assert "Raman shift (cm$^{-1}$)" in panel["tex"]
        assert "Raman shift (cm\u207b\u00b9)" in panel["html"]
        assert "Mean scaled intensity" in panel["tex"]
        assert "Mean scaled intensity" in panel["html"]
        assert CAPTION[:40] in panel["tex"] and CAPTION[:40] in panel["html"]
        assert "RQ-S01" in panel["tex"] and "RQ-S01" in panel["html"]
    tex = result["panels"][0]["tex"]
    assert tex.count("\\addplot") == 6
    assert "dashed" in tex and "dash pattern" in tex
    html_text = result["panels"][0]["html"]
    for action in ACTION_ORDER:
        assert 'data-action="%s"' % action in html_text
    for color in ("#000000", "#0072B2", "#D55E00"):
        assert color in html_text


def test_csv_support_and_na_row():
    result = prepare_f01(_bundle())
    domain1 = _rows(result["panels"][0]["csv"])
    assert len(domain1) == 2 * 3 * AXIS_N
    assert set(r["representation_id"] for r in domain1) == set(ACTION_ORDER)
    assert all(r["available"] == "True" for r in domain1)
    domain2 = _rows(result["panels"][1]["csv"])
    assert len(domain2) == 1
    na = domain2[0]
    assert na["available"] == "False"
    assert na["reason"] == UNAVAILABLE_REASON
    assert na["representation_id"] == ""
    assert na["raman_shift_cm1"] == "" and na["intensity"] == ""
    header = result["panels"][0]["csv"].splitlines()[0]
    assert header == ("cell_id,station,instrument,analyte,n_spectra,n_masters,"
                      "held_comparison_domain,available,reason,representation_id,"
                      "raman_shift_cm1,intensity")


def test_no_external_assets_or_includegraphics():
    result = prepare_f01(_bundle())
    for panel in result["panels"]:
        assert "\\includegraphics" not in panel["tex"]
        assert "cdn.plot.ly" not in panel["html"]
        assert not re.search(r"<script[^>]*\ssrc=", panel["html"])
        assert not re.search(r'href="https?:', panel["html"])
        assert panel["html"].rstrip().endswith("</html>")
        assert panel["html"].count("</script>") == 2
    assert "data:text/csv;base64," in result["panels"][0]["html"]
    assert result["panels"][0]["html"].count('checked="checked"') == 3


def test_privacy_extra_fields_refused():
    bundle = _bundle()
    bundle["master_ids"] = ["m1", "m2"]
    with pytest.raises(ValueError):
        prepare_f01(bundle)
    bundle = _bundle()
    bundle["cells"][0]["operator"] = "alice"
    with pytest.raises(ValueError):
        prepare_f01(bundle)
    bundle = _bundle()
    bundle["cells"][0]["curves"]["R_MIN_400_1800"][0] = 0.0
    assert bundle["cells"][0]["curves"]["R_MIN_400_1800"][0] == 0.0


def test_malformed_shapes_and_values_refused():
    cases = []
    bundle = _bundle()
    bundle["schema_version"] = "other"
    cases.append(bundle)
    bundle = _bundle()
    bundle["figure_id"] = "P08-F02"
    cases.append(bundle)
    bundle = _bundle()
    bundle["action_order"] = list(ACTION_ORDER[:2])
    cases.append(bundle)
    bundle = _bundle()
    bundle["axis_cm1"] = bundle["axis_cm1"][:10]
    cases.append(bundle)
    bundle = _bundle()
    bundle["axis_cm1"][3] = 999.0
    cases.append(bundle)
    bundle = _bundle()
    bundle["cells"][0]["n_masters"] = True
    cases.append(bundle)
    bundle = _bundle()
    bundle["cells"][0]["n_spectra"] = 0
    cases.append(bundle)
    bundle = _bundle()
    bundle["cells"][0]["available"] = 1
    cases.append(bundle)
    bundle = _bundle()
    bundle["cells"][0]["curves"][ACTION_ORDER[0]][0] = float("nan")
    cases.append(bundle)
    bundle = _bundle()
    bundle["cells"][0]["curves"][ACTION_ORDER[0]][5] = 1.5
    cases.append(bundle)
    bundle = _bundle()
    bundle["cells"][0]["curves"][ACTION_ORDER[0]] = [0.0] * 10
    cases.append(bundle)
    for bundle in cases:
        with pytest.raises(ValueError):
            prepare_f01(bundle)


def test_eligibility_and_duplicates_refused():
    bundle = _bundle()
    bundle["cells"][0]["reason"] = "why"
    with pytest.raises(ValueError):
        prepare_f01(bundle)
    bundle = _bundle()
    del bundle["cells"][0]["curves"][ACTION_ORDER[1]]
    with pytest.raises(ValueError):
        prepare_f01(bundle)
    bundle = _bundle()
    bundle["cells"][2]["n_masters"] = 2
    with pytest.raises(ValueError):
        prepare_f01(bundle)
    bundle = _bundle()
    bundle["cells"][2]["curves"] = {ACTION_ORDER[0]: _curve()}
    with pytest.raises(ValueError):
        prepare_f01(bundle)
    bundle = _bundle()
    bundle["cells"][0]["n_masters"] = 1
    with pytest.raises(ValueError):
        prepare_f01(bundle)
    bundle = _bundle()
    bundle["cells"][1]["cell_id"] = "P08-F01-C001"
    with pytest.raises(ValueError):
        prepare_f01(bundle)
    bundle = _bundle()
    bundle["cells"][1]["analyte"] = "Benzene_<x>"
    with pytest.raises(ValueError):
        prepare_f01(bundle)
    bundle = _bundle()
    bundle["cells"] = [bundle["cells"][1], bundle["cells"][0], bundle["cells"][2]]
    with pytest.raises(ValueError):
        prepare_f01(bundle)


def test_tex_and_html_injection_escaping():
    result = prepare_f01(_bundle())
    tex = result["panels"][0]["tex"]
    assert "Instr_#1" not in tex
    assert "Instr\\_\\#1" in tex
    assert "Benzene_<x>" not in tex
    assert "Benzene\\_<x>" in tex
    html_text = result["panels"][1]["html"]
    assert "</script> & <b>" not in html_text
    assert "&lt;/script&gt; &amp; &lt;b&gt;" in html_text
    assert "\\u003c/script\\u003e" in html_text


def test_single_domain_three_cell_layout():
    result = prepare_f01(_single_domain_bundle())
    assert len(result["panels"]) == 1
    panel = result["panels"][0]
    assert panel["exploratory"] is False
    tex = panel["tex"]
    assert "\\begin{minipage}{181.86mm}" in tex
    assert "\\vspace*{6pt}\\noindent\\hspace*{3pt}" in tex
    assert "\\end{tikzpicture}\\par\\vspace*{6pt}" in tex
    assert "group size=1 by 3" in tex
    assert "vertical sep=1.65cm" in tex
    assert "height=4.6cm" in tex
    assert "legend to name=" not in tex
    assert "\\ref{f01legend}" not in tex
    assert tex.count("xlabel={Raman shift (cm$^{-1}$)}") == 1
    assert tex.count("\\nextgroupplot") == 3
    assert "\\bfseries\\fontsize{10}{12}\\selectfont" in tex
    assert "Unavailable: fewer than two physical samples" in tex
    assert "\\end{minipage}" in tex


def test_metadata_and_hash_rendered_in_both_formats():
    result = prepare_f01(_single_domain_bundle())
    sha = result["semantic_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    panel = result["panels"][0]
    for token in ("RQ-S01", "Gamma", "Instr-9", "physical master",
                  "fixed universal preprocessing",
                  "descriptive primary-population display"):
        assert token in panel["tex"], token
        assert token in panel["html"], token
    assert ("\\texttt{%s}" % sha) in panel["tex"]
    assert sha in panel["html"]


def test_svg_ticks_titles_and_hover_semantics():
    result = prepare_f01(_single_domain_bundle())
    html = result["panels"][0]["html"]
    assert "#cccccc" not in html
    assert 'stroke="#000000" stroke-width="0.8"' in html
    assert "physical masters: 4; stored spectra: 9" in html
    assert "| Gamma | Instr-9 | A1 |" in html
    assert "cb.checked" in html
    assert "physical masters: ' + cell.n_masters" in html


def test_frozen_styles_marker_none_and_parity():
    result = prepare_f01(_bundle())
    actions = result["semantic"]["style"]["actions"]
    for action in ACTION_ORDER:
        assert actions[action]["marker"] == "none"
        assert actions[action]["color"] == STYLES[action]["color"]
        assert actions[action]["line_style"] == STYLES[action]["line_style"]
        assert STYLES[action]["marker"] == "none"
    html = result["panels"][0]["html"]
    for action in ACTION_ORDER:
        assert STYLES[action]["color"] in html


def test_reject_master_count_exceeding_spectra():
    bundle = _bundle()
    bundle["cells"][0]["n_masters"] = 8
    with pytest.raises(ValueError):
        prepare_f01(bundle)
    bundle = _bundle()
    bundle["cells"][0]["n_spectra"] = 2
    with pytest.raises(ValueError):
        prepare_f01(bundle)


def test_reject_mixed_held_flags_within_domain():
    bundle = _bundle()
    bundle["cells"][1]["held_comparison_domain"] = False
    with pytest.raises(ValueError):
        prepare_f01(bundle)


def test_reject_more_than_three_cells_per_domain():
    cells = [
        _cell("P08-F01-C001", "Alpha", "Instr-1", "A", 6, 2, True, True),
        _cell("P08-F01-C002", "Alpha", "Instr-1", "B", 6, 2, True, True),
        _cell("P08-F01-C003", "Alpha", "Instr-1", "C", 6, 2, True, True),
        _cell("P08-F01-C004", "Alpha", "Instr-1", "D", 6, 2, True, True),
    ]
    bundle = {"schema_version": "nato-sers-p08-f01-spectral-bundle-v1",
              "figure_id": "P08-F01",
              "axis_cm1": [400.0 + i for i in range(AXIS_N)],
              "action_order": list(ACTION_ORDER),
              "cells": cells,
              "caption": ""}
    with pytest.raises(ValueError):
        prepare_f01(bundle)


def test_exact_axis_values_required():
    bundle = _bundle()
    bundle["axis_cm1"][7] = 407.000000001
    with pytest.raises(ValueError):
        prepare_f01(bundle)
    bundle = _bundle()
    bundle["axis_cm1"] = [float(400 + i) for i in range(AXIS_N)]
    prepare_f01(bundle)


def test_tex_contains_every_axis_point():
    result = prepare_f01(_single_domain_bundle())
    tex = result["panels"][0]["tex"]
    assert tex.count("(") >= 6 * AXIS_N


def test_reviewed_layout_clearance_and_readable_html():
    panel = prepare_f01(_single_domain_bundle())["panels"][0]
    assert "[yshift=22mm]group c1r1.north west" in panel["tex"]
    assert "[yshift=12mm]group c1r1.north west" in panel["tex"]
    assert "[yshift=-14mm]group c1r3.south west" in panel["tex"]
    assert ".meta{font-size:16px}" in panel["html"]
    assert ".tick{font-size:16px}" in panel["html"]
    assert "font-size:11px" not in panel["html"]
