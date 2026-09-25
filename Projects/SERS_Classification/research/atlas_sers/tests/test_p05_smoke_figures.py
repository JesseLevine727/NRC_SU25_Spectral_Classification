# ruff: noqa: E501
"""Synthetic tests for the P05 training-only smoke figure module."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_sers.governance.canonical import sha256_file
from atlas_sers.visualization import p05_smoke_figures as p05


def test_actual_role_taxonomy() -> None:
    assert p05.ROLE_LABELS == ("cwa_dense", "pills_dense", "surfaces_dense", "surfaces_sparse")
    assert p05.ROLE_SUPPORT == {
        "cwa_dense": 13,
        "pills_dense": 9,
        "surfaces_dense": 13,
        "surfaces_sparse": 4,
    }
    assert p05.SPARSE_ROLE == "surfaces_sparse"
    assert tuple(p05.DISPLAY_LABELS[role] for role in p05.ROLE_LABELS) == (
        "CWA",
        "Pills",
        "Surfaces",
        "Surfaces (sparse)",
    )
    assert p05.ROLE_LABELS != ("surface", "colloid", "film", "solution")


def _primary(role: str, recipe: str, seed: int) -> dict:
    return {
        "execution_kind": p05.PRIMARY_KIND,
        "role_label": role,
        "recipe_id": recipe,
        "seed": seed,
        "status": "complete",
        "history": [
            {
                "epoch": epoch,
                "chemical_ce": 1.0 + 0.01 * epoch,
                "gradient_norm_mean": 0.5 + 0.02 * epoch,
            }
            for epoch in p05.EPOCHS
        ],
        "uid": "UID-SHOULD-NOT-LEAK",
        "master_path": "/private/run/master.csv",
        "prediction_rows": [{"label": "secret"}],
    }


def _replay(role: str, recipe: str, seed: int) -> dict:
    record = _primary(role, recipe, seed)
    record["execution_kind"] = p05.REPLAY_KIND
    record["uid"] = "REPLAY-SHOULD-NOT-LEAK"
    record["history"] = [{"epoch": 1, "chemical_ce": 99.0, "gradient_norm_mean": 99.0}]
    return record


def _records() -> list[dict]:
    records = []
    for role in p05.ROLE_LABELS:
        for recipe in p05.RECIPE_IDS:
            for seed in p05.SEEDS:
                records.append(_primary(role, recipe, seed))
    records.append(_replay(p05.ROLE_LABELS[0], p05.RECIPE_IDS[0], p05.SEEDS[0]))
    records.append(_replay(p05.ROLE_LABELS[1], p05.RECIPE_IDS[1], p05.SEEDS[1]))
    assert len(records) == 34
    return records


def _fake_compile(tex_path, pdf_path, png_path, log_path) -> None:
    pdf_path.write_bytes(b"%PDF-1.4\n")
    png_path.write_bytes(b"\x89PNG\r\n")
    log_path.write_text("synthetic log\n", encoding="utf-8")
    log_path.unlink()


def test_full_synthetic_fixture_shape() -> None:
    frame = p05.build_semantic_data(_records())
    assert len(frame) == 256
    assert list(frame.columns) == list(p05.SEMANTIC_COLUMNS)


def test_replays_are_filtered() -> None:
    frame = p05.build_semantic_data(_records())
    assert len(frame) == 256
    assert not (frame["chemical_ce"] == 99.0).any()
    assert not (frame["gradient_norm_mean"] == 99.0).any()


def test_schema_and_privacy() -> None:
    frame = p05.build_semantic_data(_records())
    assert list(frame.columns) == list(p05.SEMANTIC_COLUMNS)
    text = frame.to_csv(index=False)
    for token in (
        "UID-SHOULD-NOT-LEAK",
        "REPLAY-SHOULD-NOT-LEAK",
        "/private/",
        "master_path",
        "prediction_rows",
    ):
        assert token not in text
    assert set(frame["role_label"]) == set(p05.ROLE_LABELS)
    assert set(frame["recipe_id"]) == set(p05.RECIPE_IDS)
    assert set(frame["seed_index"]) == {1, 2}
    for epoch in p05.EPOCHS:
        assert (frame["epoch"] == epoch).sum() == 32


def test_deterministic_row_ordering() -> None:
    forward = p05.build_semantic_data(_records())
    backward = p05.build_semantic_data(list(reversed(_records())))
    assert forward.equals(backward)
    assert list(forward["role_label"]) == [role for role in p05.ROLE_LABELS for _ in range(64)]
    assert list(forward["epoch"]) == list(p05.EPOCHS) * 32


def test_duplicate_primary_rejected() -> None:
    records = _records()
    records.append(_primary(p05.ROLE_LABELS[0], p05.RECIPE_IDS[0], p05.SEEDS[0]))
    with pytest.raises(ValueError):
        p05.build_semantic_data(records)


def test_missing_primary_rejected() -> None:
    records = [
        record
        for record in _records()
        if not (
            record.get("execution_kind") == p05.PRIMARY_KIND
            and record.get("role_label") == p05.ROLE_LABELS[0]
            and record.get("recipe_id") == p05.RECIPE_IDS[0]
            and record.get("seed") == p05.SEEDS[0]
        )
    ]
    with pytest.raises(ValueError):
        p05.build_semantic_data(records)


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf"), -1.0, None, "0.5", True],
)
def test_invalid_metric_rejected(bad: object) -> None:
    records = _records()
    target = next(r for r in records if r.get("execution_kind") == p05.PRIMARY_KIND)
    target["history"][0]["chemical_ce"] = bad
    with pytest.raises((TypeError, ValueError)):
        p05.build_semantic_data(records)


def test_truncated_history_rejected() -> None:
    records = _records()
    target = next(r for r in records if r.get("execution_kind") == p05.PRIMARY_KIND)
    target["history"] = target["history"][:7]
    with pytest.raises(ValueError):
        p05.build_semantic_data(records)


def test_duplicate_epoch_rejected() -> None:
    records = _records()
    target = next(r for r in records if r.get("execution_kind") == p05.PRIMARY_KIND)
    target["history"][1] = dict(target["history"][0])
    with pytest.raises(ValueError):
        p05.build_semantic_data(records)


@pytest.mark.parametrize(
    "bad",
    ["not-a-list", b"bytes", 42, {"execution_kind": "primary"}, [1, 2, 3]],
)
def test_malformed_records_rejected(bad: object) -> None:
    with pytest.raises(TypeError):
        p05.build_semantic_data(bad)


def test_unknown_execution_kind_rejected() -> None:
    records = _records()
    records[0] = {**records[0], "execution_kind": "mystery"}
    with pytest.raises(ValueError):
        p05.build_semantic_data(records)


def test_unknown_role_rejected() -> None:
    records = _records()
    records[0] = {**records[0], "role_label": "unknown"}
    with pytest.raises(ValueError):
        p05.build_semantic_data(records)


def test_generate_smoke_figures(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(p05, "_compile", _fake_compile)
    root = tmp_path / "fresh" / "out"
    manifest = p05.generate_smoke_figures(_records(), root)
    assert root.is_dir()
    digest = sha256_file(root / p05.SEMANTIC_NAME)
    assert manifest["data_sha256"] == digest
    for entry in manifest["files"]:
        assert not Path(entry["path"]).is_absolute()
        assert sha256_file(root / entry["path"]) == entry["sha256"]
    for stem in ("P05S01_training_ce", "P05S02_gradient_norm"):
        tex = (root / f"{stem}.tex").read_text(encoding="utf-8")
        html = (root / f"{stem}.html").read_text(encoding="utf-8")
        html_text = html.replace("\\u002f", "/")
        assert (root / f"{stem}.pdf").is_file()
        assert (root / f"{stem}.png").is_file()
        assert not (root / f"{stem}.log").exists()
        assert "pgfplots" in tex
        assert r"\begin{groupplot}" in tex
        assert r"\addplot" in tex
        assert "coordinates" in tex
        assert f"data_sha256={digest}" in tex
        assert f"data_sha256={digest}" in html
        for token in (
            "training-only numerical check",
            "no unseen-data comparison",
            p05.SPARSE_DISCLAIMER,
            "overlap is not evidence of learned invariance",
            "ordinary classification",
            "chemical similarity",
            "matched-sample consistency",
        ):
            assert token in tex
            assert token in html_text
        assert "UID-SHOULD-NOT-LEAK" not in html
        assert "REPLAY-SHOULD-NOT-LEAK" not in html
    for path in root.iterdir():
        assert not path.is_symlink()


def test_generate_refuses_existing_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(p05, "_compile", _fake_compile)
    root = tmp_path / "out"
    root.mkdir()
    with pytest.raises(FileExistsError):
        p05.generate_smoke_figures(_records(), root)


def test_plotly_annotations_preserve_subplot_titles() -> None:
    frame = p05.build_semantic_data(_records())
    figure = p05._plotly_figure(frame, p05.FIGURE_SPECS[0])
    texts = [annotation.text for annotation in figure.layout.annotations]
    assert len(texts) == 5
    for role in p05.ROLE_LABELS:
        expected = f"{p05.DISPLAY_LABELS[role]} ({p05.ROLE_SUPPORT[role]} masters)"
        assert expected in texts
    footers = [text for text in texts if p05.SPARSE_DISCLAIMER in text]
    assert len(footers) == 1
    caption = footers[0]
    assert p05.TRAINING_CLAIM not in caption
    for recipe in p05.RECIPE_IDS:
        assert p05.RECIPE_DESCRIPTIONS[recipe] in caption
    assert "seed1" in caption and "seed2" in caption


def test_generate_refuses_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        p05.generate_smoke_figures(_records(), link)


def test_generate_refuses_ancestor_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    link = real / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError):
        p05.generate_smoke_figures(_records(), link / "out")


def test_failure_preserves_evidence(monkeypatch, tmp_path: Path) -> None:
    def boom(tex_path, pdf_path, png_path, log_path) -> None:
        raise RuntimeError("synthetic compile failure")

    monkeypatch.setattr(p05, "_compile", boom)
    root = tmp_path / "out"
    with pytest.raises(RuntimeError):
        p05.generate_smoke_figures(_records(), root)
    assert root.is_dir()
    assert (root / p05.SEMANTIC_NAME).is_file()
    assert (root / "P05S01_training_ce.tex").is_file()
