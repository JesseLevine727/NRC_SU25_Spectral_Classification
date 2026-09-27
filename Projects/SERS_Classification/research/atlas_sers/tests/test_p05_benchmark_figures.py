"""Synthetic tests for the P05 benchmark paired-figure module.

All inputs are aggregate domain-mean DataFrames assembled in memory. No
spectra, training runs, torch, or private records are touched. The real
LaTeX/PDF/PNG toolchain is exercised only by the guarded integration test.
"""

from __future__ import annotations

import copy
import hashlib
import re
import shutil
from html.parser import HTMLParser

import pandas as pd
import pytest

from atlas_sers.visualization import p05_benchmark_figures as mod


def _reference() -> str:
    for reference in mod.REGISTERED_REFERENCES:
        if all((model, reference) in mod.PAIR_SET for model in mod.MODEL_ORDER):
            return reference
    return mod.REGISTERED_REFERENCES[0]


def _models(reference: str) -> list[str]:
    return [model for model in mod.MODEL_ORDER if (model, reference) in mod.PAIR_SET]


def _row(
    station,
    domain,
    instrument,
    model,
    reference,
    aggregation,
    planned,
    common,
    model_ba,
    reference_ba,
):
    if common > 0:
        delta = model_ba - reference_ba
    else:
        model_ba = reference_ba = delta = float("nan")
    return {
        "station": station,
        "domain": domain,
        "held_instrument": instrument,
        "model_id": model,
        "reference_model_id": reference,
        "aggregation_id": aggregation,
        "planned_contexts": planned,
        "common_contexts": common,
        "common_coverage": common / planned,
        "mean_model_balanced_accuracy": model_ba,
        "mean_reference_balanced_accuracy": reference_ba,
        "mean_delta_balanced_accuracy": delta,
    }


def _base_records() -> list[dict]:
    reference = _reference()
    aggregation = mod.AGGREGATION_ORDER[0]
    records = []
    for s_index, station in enumerate(mod.STATION_ORDER):
        for d_index in range(2):
            domain = f"{station}-d{d_index}"
            instrument = f"inst-{s_index}-{d_index}"
            for m_index, model in enumerate(_models(reference)):
                records.append(
                    _row(
                        station,
                        domain,
                        instrument,
                        model,
                        reference,
                        aggregation,
                        planned=10,
                        common=5,
                        model_ba=0.6 + 0.05 * m_index,
                        reference_ba=0.55,
                    )
                )
    return records


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(_base_records())


def _semantic() -> pd.DataFrame:
    return mod._build_semantic(_base_frame())


def _first_spec(frame: pd.DataFrame) -> dict:
    return mod._figure_specs(frame)[0]


@pytest.fixture()
def stub_compile(monkeypatch):
    def _compile(tex_path, pdf_path, png_path, log_path):
        pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        log_path.write_text("stub\n", encoding="utf-8")

    monkeypatch.setattr(mod, "_compile", _compile)


def test_valid_semantic_columns_and_model_order():
    frame = _semantic()
    assert tuple(frame.columns) == mod.SEMANTIC_COLUMNS
    assert len(frame) == len(_base_records())
    order = {value: index for index, value in enumerate(mod.MODEL_ORDER)}
    for _, cell in frame.groupby("domain", sort=False):
        assert list(cell.model_id) == sorted(cell.model_id, key=order.__getitem__)


def test_private_columns_dropped_and_never_leak(tmp_path, stub_compile):
    frame = _base_frame()
    frame["private_uid"] = "SENTINEL-UID"
    frame["raw_logits"] = "SENTINEL-LOGITS"
    root = tmp_path / "private"
    manifest = mod.generate_pair_figures(frame, root)
    semantic = (root / mod.SEMANTIC_NAME).read_text(encoding="utf-8")
    assert "SENTINEL" not in semantic
    assert "private_uid" not in semantic
    for record in manifest["figures"]:
        stem = mod._figure_stem(record["reference_model_id"], record["aggregation_id"])
        for suffix in (".tex", ".html"):
            assert "SENTINEL" not in (root / f"{stem}{suffix}").read_text(encoding="utf-8")


@pytest.mark.parametrize("bad", [None, "text", b"bytes", pd.DataFrame()])
def test_non_frame_or_empty_rejected(bad):
    with pytest.raises(mod.BenchmarkFigureError):
        mod._build_semantic(bad)


def test_missing_columns_rejected():
    frame = _base_frame().drop(columns=["mean_delta_balanced_accuracy"])
    with pytest.raises(mod.BenchmarkFigureError):
        mod._build_semantic(frame)


_INVALID_ROWS = [
    ("station", lambda r: r.__setitem__("station", "unknown")),
    ("model", lambda r: r.__setitem__("model_id", "unknown")),
    ("reference", lambda r: r.__setitem__("reference_model_id", "unknown")),
    ("aggregation", lambda r: r.__setitem__("aggregation_id", "unknown")),
    ("negative_planned", lambda r: r.__setitem__("planned_contexts", -1)),
    ("zero_planned", lambda r: r.__setitem__("planned_contexts", 0)),
    ("negative_common", lambda r: r.__setitem__("common_contexts", -1)),
    ("common_over_planned", lambda r: r.__setitem__("common_contexts", 99)),
    ("float_planned", lambda r: r.__setitem__("planned_contexts", 10.5)),
    ("float_common", lambda r: r.__setitem__("common_contexts", 3.5)),
    ("bool_planned", lambda r: r.__setitem__("planned_contexts", True)),
    ("bool_common", lambda r: r.__setitem__("common_contexts", False)),
    ("coverage_wrong", lambda r: r.__setitem__("common_coverage", 0.9)),
    ("delta_wrong", lambda r: r.__setitem__("mean_delta_balanced_accuracy", 0.9)),
    ("nan_model_ba", lambda r: r.__setitem__("mean_model_balanced_accuracy", float("nan"))),
    ("inf_reference_ba", lambda r: r.__setitem__("mean_reference_balanced_accuracy", float("inf"))),
    ("out_of_unit_model_ba", lambda r: r.__setitem__("mean_model_balanced_accuracy", 1.5)),
    ("blank_domain", lambda r: r.__setitem__("domain", "  ")),
    ("untrimmed_instrument", lambda r: r.__setitem__("held_instrument", " inst ")),
]


@pytest.mark.parametrize(
    "mutate",
    [mutate for _, mutate in _INVALID_ROWS],
    ids=[name for name, _ in _INVALID_ROWS],
)
def test_invalid_row_rejected(mutate):
    records = _base_records()
    mutate(records[0])
    with pytest.raises(mod.BenchmarkFigureError):
        mod._build_semantic(pd.DataFrame(records))


def test_unregistered_pair_rejected():
    candidates = [
        (model, reference)
        for reference in mod.REGISTERED_REFERENCES
        for model in mod.MODEL_ORDER
        if (model, reference) not in mod.PAIR_SET
    ]
    if not candidates:
        pytest.skip("every model/reference pair is registered")
    model, reference = candidates[0]
    records = _base_records()
    records[0]["model_id"] = model
    records[0]["reference_model_id"] = reference
    with pytest.raises(mod.BenchmarkFigureError):
        mod._build_semantic(pd.DataFrame(records))


def test_duplicate_keys_rejected():
    records = _base_records()
    records.append(copy.deepcopy(records[0]))
    with pytest.raises(mod.BenchmarkFigureError):
        mod._build_semantic(pd.DataFrame(records))


def _shared_domain(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["domain"], []).append(record)
    return next((group for group in grouped.values() if len(group) >= 2), None)


def test_domain_station_conflict_rejected():
    records = _base_records()
    shared = _shared_domain(records)
    if shared is None:
        pytest.skip("no domain shared by multiple models")
    other = next(s for s in mod.STATION_ORDER if s != shared[0]["station"])
    shared[1]["station"] = other
    with pytest.raises(mod.BenchmarkFigureError):
        mod._build_semantic(pd.DataFrame(records))


def test_domain_instrument_conflict_rejected():
    records = _base_records()
    shared = _shared_domain(records)
    if shared is None:
        pytest.skip("no domain shared by multiple models")
    shared[1]["held_instrument"] = "different-instrument"
    with pytest.raises(mod.BenchmarkFigureError):
        mod._build_semantic(pd.DataFrame(records))


def test_nan_means_with_common_rejected():
    records = _base_records()
    records[0]["mean_model_balanced_accuracy"] = float("nan")
    with pytest.raises(mod.BenchmarkFigureError):
        mod._build_semantic(pd.DataFrame(records))


def test_finite_means_with_zero_common_rejected():
    records = _base_records()
    records[0]["common_contexts"] = 0
    records[0]["common_coverage"] = 0.0
    with pytest.raises(mod.BenchmarkFigureError):
        mod._build_semantic(pd.DataFrame(records))


def test_zero_common_retained_and_not_plotted():
    records = _base_records()
    row = records[0]
    row["common_contexts"] = 0
    row["common_coverage"] = 0.0
    row["mean_model_balanced_accuracy"] = float("nan")
    row["mean_reference_balanced_accuracy"] = float("nan")
    row["mean_delta_balanced_accuracy"] = float("nan")
    frame = mod._build_semantic(pd.DataFrame(records))
    assert len(frame) == len(records)
    stored = frame[(frame.domain == row["domain"]) & (frame.model_id == row["model_id"])].iloc[0]
    assert pd.isna(stored["mean_model_balanced_accuracy"])
    assert pd.isna(stored["mean_delta_balanced_accuracy"])
    spec = _first_spec(frame)
    figure = mod._plotly_figure(spec["rows"], spec)
    plotted = sum(len(trace.x) for trace in figure.data if trace.mode == "markers")
    assert plotted == int(frame.common_contexts.gt(0).sum())


def test_empty_panel_annotations():
    records = _base_records()
    station = mod.STATION_ORDER[0]
    for row in records:
        if row["station"] == station:
            row["common_contexts"] = 0
            row["common_coverage"] = 0.0
            row["mean_model_balanced_accuracy"] = float("nan")
            row["mean_reference_balanced_accuracy"] = float("nan")
            row["mean_delta_balanced_accuracy"] = float("nan")
    frame = mod._build_semantic(pd.DataFrame(records))
    spec = _first_spec(frame)
    figure = mod._plotly_figure(spec["rows"], spec)
    assert any("No common" in (annotation.text or "") for annotation in figure.layout.annotations)
    assert "No common" in mod._tikz_source(spec["rows"], spec, "deadbeef")


def test_ordering_invariance_and_no_input_mutation():
    records = _base_records()
    snapshot = copy.deepcopy(records)
    frame = pd.DataFrame(records)
    forward = mod._build_semantic(frame)
    shuffled = frame.sample(frac=1.0, random_state=11).reset_index(drop=True)
    backward = mod._build_semantic(shuffled)
    pd.testing.assert_frame_equal(forward, backward)
    assert records == snapshot


def test_tikz_and_plotly_same_coordinates_and_precision():
    frame = _semantic()
    spec = _first_spec(frame)
    tex = mod._tikz_source(spec["rows"], spec, "deadbeef")
    figure = mod._plotly_figure(spec["rows"], spec)
    plotted = sum(len(trace.x) for trace in figure.data if trace.mode == "markers")
    coordinates = []
    for line in tex.splitlines():
        if not line.startswith(r"\addplot[only marks"):
            continue
        body = line.split("coordinates {", 1)[1].rsplit("}", 1)[0]
        coordinates.extend(re.findall(r"\(([^,]+),([^)]+)\)", body))
    assert len(coordinates) == plotted
    for row in spec["rows"].itertuples(index=False):
        if row.common_contexts <= 0:
            continue
        token = (
            f"({float(row.mean_reference_balanced_accuracy):.17g},"
            f"{float(row.mean_model_balanced_accuracy):.17g})"
        )
        assert token in tex


def test_reference_identity_and_unit_scales():
    frame = _semantic()
    spec = _first_spec(frame)
    tex = mod._tikz_source(spec["rows"], spec, "deadbeef")
    assert r"\addplot[black, dotted, line width=0.8pt] coordinates {(0,0) (1,1)};" in tex
    assert "xmin=0, xmax=1" in tex
    assert "ymin=0, ymax=1" in tex
    assert "xtick={0,0.5,1}" in tex
    figure = mod._plotly_figure(spec["rows"], spec)
    identity = [trace for trace in figure.data if trace.mode == "lines"]
    assert identity
    assert all(list(trace.x) == [0.0, 1.0] and list(trace.y) == [0.0, 1.0] for trace in identity)
    assert tuple(figure.layout.xaxis.range) == (0.0, 1.0)
    assert tuple(figure.layout.yaxis.range) == (0.0, 1.0)
    assert figure.layout.yaxis.scaleanchor == "x"


def test_black_serif_typography():
    frame = _semantic()
    spec = _first_spec(frame)
    figure = mod._plotly_figure(spec["rows"], spec)
    assert "Times New Roman" in figure.layout.font.family
    assert "serif" in figure.layout.font.family
    assert figure.layout.font.color == "black"
    assert figure.layout.paper_bgcolor == "white"
    assert figure.layout.plot_bgcolor == "white"
    tex = mod._tikz_source(spec["rows"], spec, "deadbeef")
    assert tex.count("color=black") >= 4
    assert "\\selectfont" in tex


def test_shape_redundancy_open_markers():
    assert len(set(mod.MODEL_HTML_SYMBOL.values())) == len(mod.MODEL_ORDER)
    assert len(set(mod.MODEL_TEX_MARK.values())) == len(mod.MODEL_ORDER)
    assert any(symbol.endswith("-open") for symbol in mod.MODEL_HTML_SYMBOL.values())
    frame = _semantic()
    spec = _first_spec(frame)
    figure = mod._plotly_figure(spec["rows"], spec)
    present = _models(_reference())
    symbols = {trace.marker.symbol for trace in figure.data if trace.mode == "markers"}
    assert symbols == {mod.MODEL_HTML_SYMBOL[model] for model in present}
    tex = mod._tikz_source(spec["rows"], spec, "deadbeef")
    if len(present) > 1:
        assert "fill=none" in tex
    if "D0-M" in present:
        assert "mark=*" in tex


def test_tex_literal_escaping():
    assert mod._tex_literal("{x}") == r"\{x\}"
    assert mod._tex_literal("$v$") == r"\$v\$"
    assert mod._tex_literal("a&b#c%d_e") == r"a\&b\#c\%d\_e"
    assert mod._tex_literal("~^\\") == r"\textasciitilde{}\textasciicircum{}\textbackslash{}"
    assert mod._tex_literal("<>") == "<>"


def test_html_literal_escaping():
    escaped = mod._html_literal("<script>a&b\"c'</script>")
    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped
    assert "&amp;" in escaped
    assert "&quot;" in escaped
    assert "&#x27;" in escaped


def test_malicious_metadata_escaped_not_raw():
    records = _base_records()
    records[0]["domain"] = "<script>alert('x')</script>"
    records[0]["held_instrument"] = "</script><img src=x>"
    frame = mod._build_semantic(pd.DataFrame(records))
    spec = _first_spec(frame)
    figure = mod._plotly_figure(spec["rows"], spec)
    flattened = []
    for trace in figure.data:
        for entry in trace.customdata or []:
            flattened.extend(str(value) for value in entry)
    joined = " ".join(flattened)
    assert "<script>" not in joined
    assert "<img" not in joined
    assert "&lt;script&gt;" in joined


def test_caption_three_seed_is_neural_only():
    caption = " ".join(mod.CAPTION_LINES)
    assert "three-seed" in caption
    line = next(line for line in mod.CAPTION_LINES if "three-seed" in line)
    assert "neural" in line.lower()
    assert "classical" not in line.lower()


def test_generate_bundle_digest_and_file_hashes(tmp_path, stub_compile):
    root = tmp_path / "bundle"
    manifest = mod.generate_pair_figures(_base_frame(), root)
    digest = hashlib.sha256((root / mod.SEMANTIC_NAME).read_bytes()).hexdigest()
    assert manifest["data_sha256"] == digest
    assert manifest["semantic_path"] == mod.SEMANTIC_NAME
    for record in manifest["figures"]:
        stem = mod._figure_stem(record["reference_model_id"], record["aggregation_id"])
        assert digest in (root / f"{stem}.tex").read_text(encoding="utf-8")
        assert digest in (root / f"{stem}.html").read_text(encoding="utf-8")
        for suffix in ("tex", "pdf", "png", "html"):
            path = root / f"{stem}.{suffix}"
            assert record[f"{suffix}_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for entry in manifest["files"]:
        path = root / entry["path"]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    names = {entry["path"] for entry in manifest["files"]}
    assert mod.SEMANTIC_NAME in names
    assert mod.MANIFEST_NAME not in names


def test_html_embeds_plotly_without_cdn(tmp_path, stub_compile):
    class ResourceParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.external_resources = []

        def handle_starttag(self, tag, attrs):
            # Inspect actual resource elements, not dormant URL strings inside
            # Plotly's bundled JavaScript (which includes unused map modules).
            for key, value in attrs:
                resource = key == "src" or (tag == "link" and key == "href")
                if resource and value and value.startswith(("http:", "https:", "//")):
                    self.external_resources.append((tag, key, value))

    root = tmp_path / "offline"
    manifest = mod.generate_pair_figures(_base_frame(), root)
    for record in manifest["figures"]:
        stem = mod._figure_stem(record["reference_model_id"], record["aggregation_id"])
        html = (root / f"{stem}.html").read_text(encoding="utf-8")
        assert "cdn.plot.ly" not in html
        parser = ResourceParser()
        parser.feed(html)
        assert parser.external_resources == []
        assert "plotly" in html.lower()


def test_generate_refuses_overwrite(tmp_path, stub_compile):
    root = tmp_path / "once"
    mod.generate_pair_figures(_base_frame(), root)
    with pytest.raises(FileExistsError):
        mod.generate_pair_figures(_base_frame(), root)


def test_manifest_files_skip_symlinks_and_disallowed(tmp_path):
    root = tmp_path / "scan"
    root.mkdir()
    (root / "a.csv").write_text("x", encoding="utf-8")
    (root / "b.tex").write_text("y", encoding="utf-8")
    (root / "junk.txt").write_text("z", encoding="utf-8")
    target = tmp_path / "target.csv"
    target.write_text("t", encoding="utf-8")
    (root / "link.csv").symlink_to(target)
    (root / mod.MANIFEST_NAME).write_text("{}", encoding="utf-8")
    names = {entry["path"] for entry in mod._manifest_files(root)}
    assert names == {"a.csv", "b.tex"}


def test_invalid_input_writes_no_output(tmp_path):
    records = _base_records()
    records[0]["station"] = "unknown"
    root = tmp_path / "absent"
    with pytest.raises(mod.BenchmarkFigureError):
        mod.generate_pair_figures(pd.DataFrame(records), root)
    assert not root.exists()


@pytest.mark.skipif(
    shutil.which("pdflatex") is None or shutil.which("pdftocairo") is None,
    reason="TeX toolchain unavailable",
)
def test_real_compile_m06_three_panels(tmp_path):
    reference = "C-RANDOM-FOREST"
    models = _models(reference)
    assert len(models) == len(mod.MODEL_ORDER)
    aggregation = "M06"
    records = []
    for s_index, station in enumerate(mod.STATION_ORDER):
        for d_index in range(2):
            domain = f"{station}-d{d_index}"
            for m_index, model in enumerate(models):
                records.append(
                    _row(
                        station,
                        domain,
                        f"inst-{s_index}-{d_index}",
                        model,
                        reference,
                        aggregation,
                        planned=12,
                        common=6,
                        model_ba=0.6 + 0.05 * m_index - 0.1 * s_index + 0.07 * d_index,
                        reference_ba=0.55 + 0.12 * d_index + 0.03 * s_index,
                    )
                )
    root = tmp_path / "m06_compiled"
    manifest = mod.generate_pair_figures(pd.DataFrame(records), root)
    stem = mod._figure_stem(reference, aggregation)
    assert (root / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
    assert (root / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
    assert manifest["figures"][0]["plotted_rows"] == len(records)
