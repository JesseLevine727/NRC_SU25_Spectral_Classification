from __future__ import annotations

import numpy as np

from atlas_sers.data.native import (
    index_native_sources,
    load_native_spectrum,
    spectrum_diagnostics,
)


def test_two_column_native_source_indexes_loads_and_hashes(tmp_path) -> None:
    directory = tmp_path / "Mira" / "Mira 1"
    directory.mkdir(parents=True)
    axis = np.arange(400, 410, dtype=float)
    intensity = np.linspace(1, 3, len(axis)) ** 2
    np.savetxt(directory / "Scan 7.csv", np.column_stack([axis, intensity]), delimiter=",")
    index = index_native_sources(tmp_path)
    record = {**index[("Mira-1", 7)], "source_scan_id": 7}
    loaded_axis, loaded_intensity = load_native_spectrum(tmp_path, record)
    assert np.array_equal(loaded_axis, axis)
    assert np.allclose(loaded_intensity, intensity)
    diagnostics = spectrum_diagnostics(loaded_axis, loaded_intensity)
    assert diagnostics["numeric_qc_status"] == "pass"
    assert diagnostics["axis_strictly_increasing"] is True
    assert len(diagnostics["axis_sha256"]) == 64
    assert len(diagnostics["intensity_sha256"]) == 64


def test_non_increasing_axis_is_rejected_by_numeric_qc() -> None:
    diagnostics = spectrum_diagnostics(
        np.asarray([400.0, 402.0, 401.0]), np.asarray([1.0, 2.0, 3.0])
    )
    assert diagnostics["numeric_qc_status"] == "invalid_axis"
    assert diagnostics["axis_strictly_increasing"] is False
