from __future__ import annotations

import pytest

from atlas_sers.paths import PrivateRootNotConfigured, private_data_root


def test_private_root_must_be_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLAS_PRIVATE_ROOT", raising=False)
    with pytest.raises(PrivateRootNotConfigured):
        private_data_root()


def test_private_root_must_exist(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("ATLAS_PRIVATE_ROOT", str(missing))
    with pytest.raises(PrivateRootNotConfigured):
        private_data_root()
