"""Synthetic tests for the P05 aggregate pilot figures module."""

from __future__ import annotations

import copy
import hashlib
import shutil

import pandas as pd
import pytest

from atlas_sers.visualization import p05_pilot_figures as mod

SEEDS = (20260805, 20260817, 20260829)


def _epoch_count(station: str, recipe: str, seed: int) -> int:
    base = 30 + (mod.SEED_INDEX[seed] * 5) + (mod.RECIPE_IDS.index(recipe) * 3)
    return min(base + mod.STATIONS.index(station), mod.MAXIMUM_EPOCHS)


def _history(epochs: int) -> list[dict]:
    entries = []
    for epoch in range(1, epochs + 1):
        frac = epoch / (epochs + 1)
        entries.append(
            {
                "epoch": epoch,
                "chemical_ce": round(2.0 / (epoch + 1), 6),
                "train_balanced_accuracy": round(min(0.99, 0.5 + 0.4 * frac), 6),
                "validation_balanced_accuracy": round(min(0.99, 0.45 + 0.4 * frac), 6),
                "validation_nll": round(1.5 - frac, 6),
                "validation_macro_f1": round(min(0.99, 0.4 + 0.45 * frac), 6),
                "epoch_optimizer_steps": mod.BATCHES_PER_EPOCH,
                "total_optimizer_steps": epoch * mod.BATCHES_PER_EPOCH,
            }
        )
    return entries


def _record(station: str, recipe: str, seed: int) -> dict:
    epochs = _epoch_count(station, recipe, seed)
    best = 7 if (station, recipe, seed) == (mod.STATIONS[0], mod.RECIPE_IDS[0], SEEDS[0]) else 15
    best = min(best, epochs)
    return {
        "station": station,
        "recipe_id": recipe,
        "seed": seed,
        "status": "complete",
        "best_epoch": best,
        "history": _history(epochs),
    }


@pytest.fixture()
def records() -> list[dict]:
    return [
        _record(station, recipe, seed)
        for station in mod.STATIONS
        for recipe in mod.RECIPE_IDS
        for seed in SEEDS
    ]


def test_exact_population_and_seeds(records):
    frame = mod.build_semantic_data(records)
    assert frame["seed_index"].nunique() == 3
    keys = set(zip(frame["station"], frame["recipe_id"], frame["seed_index"], strict=True))
    assert len(keys) == mod.EXPECTED_FITS == 36
    assert int(frame["is_best"].sum()) == mod.EXPECTED_FITS


def test_variable_stops_and_pre30_best(records):
    frame = mod.build_semantic_data(records)
    maxima = frame.groupby(["station", "recipe_id", "seed_index"])["epoch"].max()
    assert maxima.nunique() > 1
    assert bool(((frame["is_best"]) & (frame["epoch"] < 30)).any())


def test_ordering_invariance_and_no_mutation(records):
    snapshot = copy.deepcopy(records)
    forward = mod.build_semantic_data(records)
    reverse = mod.build_semantic_data(list(reversed(records)))
    pd.testing.assert_frame_equal(forward, reverse)
    assert records == snapshot


def test_semantic_privacy_allowlist(records):
    frame = mod.build_semantic_data(records)
    assert tuple(frame.columns) == mod.SEMANTIC_COLUMNS
    joined = " ".join(frame.columns).lower()
    for forbidden in ("uid", "master", "path", "observation", "spectra", "instrument", "logit"):
        assert forbidden not in joined


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda r: r.pop(), ValueError),
        (lambda r: r.append(copy.deepcopy(r[0])), ValueError),
        (lambda r: r[0].__setitem__("seed", float(r[0]["seed"])), ValueError),
        (lambda r: r[0]["history"][0].__setitem__("epoch", 1.0), ValueError),
        (lambda r: r[0]["history"][0].__setitem__("chemical_ce", float("nan")), ValueError),
        (lambda r: r[0]["history"][0].__setitem__("validation_balanced_accuracy", 1.5), ValueError),
        (lambda r: r[0].pop("history"), ValueError),
        (lambda r: r[0]["history"][0].__setitem__("epoch_optimizer_steps", 3), ValueError),
        (lambda r: r[0]["history"][0].__setitem__("total_optimizer_steps", 5), ValueError),
        (lambda r: r[0].__setitem__("station", "unknown"), ValueError),
    ],
)
def test_invalid_records_rejected(records, mutate, error):
    mutate(records)
    with pytest.raises(error):
        mod.build_semantic_data(records)


def test_native_coordinate_full_precision(records):
    frame = mod.build_semantic_data(records)
    frame = frame.copy()
    frame.loc[0, "chemical_ce"] = 0.12345678901234567
    tex = mod._learning_curves_tikz(frame, "deadbeef")
    point = frame.iloc[0]
    coordinate = f"({int(point['epoch'])},{float(point['chemical_ce']):.17g})"
    assert coordinate in tex
    value = float(point["chemical_ce"])
    assert float(f"{value:.17g}") == value
    assert float(f"{value:.6g}") != value
    assert "\\includegraphics" not in tex


def _stub_compile(monkeypatch):
    def _compile(tex_path, pdf_path, png_path, log_path):
        pdf_path.write_bytes(b"%PDF-1.4\n")
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n")
        log_path.write_text("stub\n", encoding="utf-8")

    monkeypatch.setattr(mod, "_compile", _compile)


def test_html_and_tikz_share_digest(records, tmp_path, monkeypatch):
    _stub_compile(monkeypatch)
    root = tmp_path / "pilot"
    manifest = mod.generate_pilot_figures(records, root)
    digest = hashlib.sha256((root / mod.SEMANTIC_NAME).read_bytes()).hexdigest()
    assert manifest["data_sha256"] == digest
    for stem in ("P05P01_learning_curves", "P05P02_best_checkpoints"):
        assert digest in (root / f"{stem}.tex").read_text(encoding="utf-8")
        assert digest in (root / f"{stem}.html").read_text(encoding="utf-8")


def test_manifest_and_exclusivity(records, tmp_path, monkeypatch):
    _stub_compile(monkeypatch)
    root = tmp_path / "pilot"
    manifest = mod.generate_pilot_figures(records, root)
    names = {entry["path"] for entry in manifest["files"]}
    for stem in ("P05P01_learning_curves", "P05P02_best_checkpoints"):
        assert {f"{stem}.tex", f"{stem}.html", f"{stem}.pdf", f"{stem}.png"} <= names
    assert mod.SEMANTIC_NAME in names
    with pytest.raises(FileExistsError):
        mod.generate_pilot_figures(records, root)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex unavailable")
def test_real_compile_smoke(records, tmp_path):
    root = tmp_path / "compiled"
    mod.generate_pilot_figures(records, root)
    assert (root / "P05P01_learning_curves.pdf").read_bytes().startswith(b"%PDF")
    assert (root / "P05P02_best_checkpoints.pdf").read_bytes().startswith(b"%PDF")
