from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import atlas_sers.evaluation.p06p11_inputs as mod


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _synth(
    tmp_path: Path,
    *,
    csv_text: str = "a,b\n1,2\n",
    json_text: str = '{"ok": true}',
) -> dict[str, list[str]]:
    (tmp_path / "sub").mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "sub" / "t.csv"
    json_path = tmp_path / "sub" / "t.json"
    csv_path.write_text(csv_text)
    json_path.write_text(json_text)
    return {
        "table": ["sub/t.csv", _sha256(csv_path)],
        "meta": ["sub/t.json", _sha256(json_path)],
    }


def _boom(*_args, **_kwargs):
    raise RuntimeError("boom")


def test_success_returns_frames_dicts_and_hashes(tmp_path, monkeypatch):
    pins = _synth(tmp_path, csv_text="a,b\n1,\n")
    monkeypatch.setattr(mod, "PINS", pins)

    out = mod.load_inputs(tmp_path)

    assert set(out) == {"table", "meta", "hashes"}
    assert out["table"].iloc[0, 0] == "1"
    assert out["table"].iloc[0, 1] == ""
    assert out["meta"] == {"ok": True}
    assert out["hashes"] == {"table": pins["table"][1], "meta": pins["meta"][1]}
    assert mod.verify_inputs(tmp_path) == out["hashes"]


def test_tamper_rejected_before_parquet_read(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import pandas as pd

    parquet_path = tmp_path / "p.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(parquet_path)
    monkeypatch.setattr(mod, "PINS", {"preds": ["p.parquet", "0" * 64]})
    monkeypatch.setattr(mod.pd, "read_parquet", _boom)

    with pytest.raises(mod.InputIntegrityError) as excinfo:
        mod.load_inputs(tmp_path)

    assert excinfo.value.reason_code == "hash_mismatch"


def test_post_read_mutation_rejected(tmp_path, monkeypatch):
    pins = _synth(tmp_path)
    monkeypatch.setattr(mod, "PINS", pins)
    original = mod.pd.read_csv

    def _mutating_read(path, *args, **kwargs):
        frame = original(path, *args, **kwargs)
        (tmp_path / "sub" / "t.json").write_text('{"changed": 1}')
        return frame

    monkeypatch.setattr(mod.pd, "read_csv", _mutating_read)

    with pytest.raises(mod.InputIntegrityError) as excinfo:
        mod.load_inputs(tmp_path)

    assert excinfo.value.reason_code == "hash_mismatch"


def test_file_symlink_rejected(tmp_path, monkeypatch):
    real = tmp_path / "real.csv"
    real.write_text("a\n1\n")
    link = tmp_path / "link.csv"
    link.symlink_to(real)
    monkeypatch.setattr(mod, "PINS", {"t": ["link.csv", _sha256(real)]})

    with pytest.raises(mod.InputIntegrityError) as excinfo:
        mod.verify_inputs(tmp_path)

    assert excinfo.value.reason_code == "symlink_rejected"


def test_parent_symlink_rejected(tmp_path, monkeypatch):
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    (real_dir / "t.csv").write_text("a\n1\n")
    link_dir = tmp_path / "link_dir"
    link_dir.symlink_to(real_dir, target_is_directory=True)
    monkeypatch.setattr(mod, "PINS", {"t": ["link_dir/t.csv", _sha256(real_dir / "t.csv")]})

    with pytest.raises(mod.InputIntegrityError) as excinfo:
        mod.verify_inputs(tmp_path)

    assert excinfo.value.reason_code == "symlink_rejected"


def test_unsafe_and_missing_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PINS", {"t": ["a/../b.csv", "0" * 64]})
    with pytest.raises(mod.InputIntegrityError) as unsafe:
        mod.verify_inputs(tmp_path)
    assert unsafe.value.reason_code == "unsafe_relative_path"

    monkeypatch.setattr(mod, "PINS", {"t": ["nope.csv", "0" * 64]})
    with pytest.raises(mod.InputIntegrityError) as missing:
        mod.verify_inputs(tmp_path)
    assert missing.value.reason_code == "not_regular_file"


def test_root_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PINS", {})
    with pytest.raises(mod.InputIntegrityError) as missing:
        mod.verify_inputs(tmp_path / "absent")
    assert missing.value.reason_code == "missing_root"

    plain_file = tmp_path / "file"
    plain_file.write_text("x")
    with pytest.raises(mod.InputIntegrityError) as not_dir:
        mod.verify_inputs(plain_file)
    assert not_dir.value.reason_code == "not_directory"


def test_bad_json_and_read_failure(tmp_path, monkeypatch):
    json_path = tmp_path / "t.json"
    json_path.write_text("{ not json")
    monkeypatch.setattr(mod, "PINS", {"meta": ["t.json", _sha256(json_path)]})
    with pytest.raises(mod.InputIntegrityError) as malformed:
        mod.load_inputs(tmp_path)
    assert malformed.value.reason_code == "read_failure"

    json_path.write_text("[1, 2]")
    monkeypatch.setattr(mod, "PINS", {"meta": ["t.json", _sha256(json_path)]})
    with pytest.raises(mod.InputIntegrityError) as not_object:
        mod.load_inputs(tmp_path)
    assert not_object.value.reason_code == "json_not_object"

    pins = _synth(tmp_path)
    monkeypatch.setattr(mod, "PINS", pins)
    monkeypatch.setattr(mod.pd, "read_csv", _boom)
    with pytest.raises(mod.InputIntegrityError) as failure:
        mod.load_inputs(tmp_path)
    assert failure.value.reason_code == "read_failure"


def test_errors_do_not_leak_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PINS", {"t": ["missing.csv", "0" * 64]})
    with pytest.raises(mod.InputIntegrityError) as excinfo:
        mod.load_inputs(tmp_path)
    message = str(excinfo.value)
    assert str(tmp_path) not in message
    assert "/" not in message
    assert "\\" not in message


def test_root_symlink_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PINS", {})
    real_root = tmp_path / "real_root"
    real_root.mkdir()
    link_root = tmp_path / "link_root"
    link_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(mod.InputIntegrityError) as excinfo:
        mod.verify_inputs(link_root)

    assert excinfo.value.reason_code == "symlink_rejected"


def test_ancestor_of_root_symlink_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PINS", {})
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    (real_dir / "child").mkdir()
    link_dir = tmp_path / "link_dir"
    link_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(mod.InputIntegrityError) as excinfo:
        mod.verify_inputs(link_dir / "child")

    assert excinfo.value.reason_code == "symlink_rejected"


def test_hash_read_failure_code_only(tmp_path, monkeypatch):
    target = tmp_path / "t.csv"
    target.write_text("a\n1\n")
    monkeypatch.setattr(mod, "PINS", {"t": ["t.csv", "0" * 64]})
    real_open = Path.open

    def _failing_open(self, *args, **kwargs):
        if self.name == "t.csv":
            raise OSError("/private/secret/path")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _failing_open)

    with pytest.raises(mod.InputIntegrityError) as excinfo:
        mod.verify_inputs(tmp_path)

    assert excinfo.value.reason_code == "hash_read_failure"
    assert str(excinfo.value) == "hash_read_failure"
    assert "secret" not in str(excinfo.value)
    assert "/" not in str(excinfo.value)


def test_success_parquet_read(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import pandas as pd

    parquet_path = tmp_path / "p.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(parquet_path)
    monkeypatch.setattr(mod, "PINS", {"preds": ["p.parquet", _sha256(parquet_path)]})

    out = mod.load_inputs(tmp_path)

    assert list(out["preds"]["x"]) == [1, 2]
