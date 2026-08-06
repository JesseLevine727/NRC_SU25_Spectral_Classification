from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from atlas_sers.governance.canonical import sha256_file
from atlas_sers.preprocessing.representations import (
    arpls_correct,
    despike,
    row_area,
    row_minmax,
    row_snv,
    row_vector,
)

SOURCE_PROJECT = Path(__file__).resolve().parents[1]


def _git(repository: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True)


def _candidate_arrays(axis: np.ndarray, raw: np.ndarray, parameters: dict) -> dict:
    clean = despike(
        raw,
        window=int(parameters["despike_window"]),
        threshold=float(parameters["despike_mad_threshold"]),
    )
    smooth = savgol_filter(clean, 11, 3, axis=1, mode="interp")
    corrected = arpls_correct(
        clean,
        lam=float(parameters["arpls_lambda"]),
        iterations=int(parameters["arpls_iterations"]),
        relative_tolerance=float(parameters["arpls_relative_weight_tolerance"]),
        logistic_clip=float(parameters["arpls_logistic_clip"]),
    )
    derivative = savgol_filter(clean, 11, 3, deriv=1, delta=1.0, axis=1, mode="interp")
    return {
        "axis_cm1": axis,
        "minimal_minmax": row_minmax(raw)[0].astype(np.float32),
        "minimal_snv": row_snv(raw)[0].astype(np.float32),
        "minimal_vector": row_vector(raw)[0].astype(np.float32),
        "minimal_area": row_area(raw, axis)[0].astype(np.float32),
        "despike_savgol_minmax": row_minmax(smooth)[0].astype(np.float32),
        "arpls_minmax": row_minmax(corrected)[0].astype(np.float32),
        "savgol_derivative1_snv": row_snv(derivative)[0].astype(np.float32),
    }


def _private_fixture(private: Path, native: Path, p01_contract: dict) -> None:
    (private / "tables").mkdir(parents=True)
    (private / "arrays").mkdir()
    (private / "preprocessing").mkdir()
    native_directory = native / "Mira" / "Mira 1"
    native_directory.mkdir(parents=True)
    (native / "Notes.txt").write_text(
        "Types of SERS sensors\nTotal of 4 different types\nna (no SERS sensor, Raman spectra)\n"
    )
    axis = np.arange(400, 1850, dtype=float)
    rows: list[dict] = []
    raw_rows: list[np.ndarray] = []
    source_uids: list[str] = []
    for index in range(30):
        scan = index + 1
        intensity = (
            3
            + 0.001 * axis
            + np.sin(axis / (9 + index % 5))
            + 0.15 * np.cos(axis / (5 + index % 3))
            + 0.01 * index
        )
        path = native_directory / f"Scan {scan}.csv"
        np.savetxt(path, np.column_stack([axis, intensity]), delimiter=",", fmt="%.18e")
        loaded = np.loadtxt(path, delimiter=",")
        loaded_axis = loaded[:, 0].astype(np.float64)
        loaded_intensity = loaded[:, 1].astype(np.float64)
        source_uid = f"source-{scan:03d}"
        source_uids.append(source_uid)
        raw_rows.append(loaded_intensity.astype(np.float32))
        master = index // 2
        rows.append(
            {
                "observation_uid": source_uid,
                "recording_uid": f"recording-{scan:03d}",
                "session": 1,
                "excel_row": scan,
                "recording_subindex": 0,
                "master_sample_id": master,
                "target_analyte": f"target-{master % 3}",
                "station": f"station-{master % 2}",
                "instrument": "Mira-1",
                "sensor_family": "sensor-family",
                "sensor_variant": "sensor-variant",
                "source_scan_id": scan,
                "source_format": "two_column_csv",
                "source_reference_count": 1,
                "sensor_flag_conflict": False,
                "manual_severe_qc_flag": index >= 24,
                "manual_low_signal_or_noise_flag": False,
                "system_suitability": "pass",
                "tier_all_parseable_sers": True,
                "tier_attributable_sers": True,
                "tier_unique_attributable_sers": True,
                "tier_notes_clear_sers": index < 24,
                "tier_reason_codes": "included" if index < 24 else "severe_quality_note",
                "is_named_sers": True,
                "axis_sha256": hashlib.sha256(loaded_axis.tobytes()).hexdigest(),
                "intensity_sha256": hashlib.sha256(loaded_intensity.tobytes()).hexdigest(),
                "axis_strictly_increasing": True,
                "source_text_path": str(path),
                "comments": "synthetic fixture",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(private / "tables" / "tier_unique_attributable_sers.csv", index=False)
    manifest.to_csv(private / "tables" / "recordings_manifest_fresh.csv", index=False)
    raw = np.asarray(raw_rows, dtype=np.float32)
    np.savez_compressed(
        private / "arrays" / "unique_attributable_common_support_raw.npz",
        axis_cm1=axis.astype(np.float32),
        intensity=raw,
        observation_uid=np.asarray(source_uids),
    )
    np.savez_compressed(
        private / "preprocessing" / "preprocessing_candidates.npz",
        **_candidate_arrays(axis, raw.astype(float), p01_contract["transformation_parameters"]),
    )
    (private / "FINAL_VALIDATION_REPORT.json").write_text('{"status":"pass"}\n')


def _project_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repository = tmp_path / "repository"
    project = repository / "research" / "atlas_sers"
    private = tmp_path / "derived_inputs"
    native = tmp_path / "native_inputs"
    artifacts = tmp_path / "artifacts"
    shutil.copytree(
        SOURCE_PROJECT,
        project,
        ignore=shutil.ignore_patterns(".pytest_cache", ".ruff_cache", "__pycache__", "*.egg-info"),
    )
    p01_path = project / "plan" / "contracts" / "p01_governance_contract.json"
    p01 = json.loads(p01_path.read_text())
    p01["population_contract"].update(
        {
            "primary_rows": 30,
            "notes_clear_rows": 24,
            "mira1_excluded_rows": 0,
            "physical_masters": 15,
            "targets": 3,
            "instruments": 1,
            "sensor_families": 1,
            "stations": 2,
        }
    )
    p01_path.write_text(json.dumps(p01, indent=2, sort_keys=True) + "\n")
    _private_fixture(private, native, p01)

    research_path = project / "plan" / "contracts" / "research_contract.json"
    research = json.loads(research_path.read_text())
    declarations = [
        {
            "path": "${ATLAS_PRIVATE_ROOT}/tables/tier_unique_attributable_sers.csv",
            "sha256": sha256_file(private / "tables" / "tier_unique_attributable_sers.csv"),
            "expected_rows": 30,
        },
        {
            "path": "${ATLAS_PRIVATE_ROOT}/arrays/unique_attributable_common_support_raw.npz",
            "sha256": sha256_file(
                private / "arrays" / "unique_attributable_common_support_raw.npz"
            ),
            "expected_shape": [30, 1450],
        },
        {
            "path": "${ATLAS_PRIVATE_ROOT}/preprocessing/preprocessing_candidates.npz",
            "sha256": sha256_file(private / "preprocessing" / "preprocessing_candidates.npz"),
        },
        {
            "path": "${ATLAS_PRIVATE_ROOT}/FINAL_VALIDATION_REPORT.json",
            "sha256": sha256_file(private / "FINAL_VALIDATION_REPORT.json"),
            "required_status": "pass",
        },
    ]
    research["authoritative_inputs"] = declarations
    research_path.write_text(json.dumps(research, indent=2, sort_keys=True) + "\n")
    phase_path = project / "plan" / "registries" / "phase_registry.csv"
    phase = pd.read_csv(phase_path)
    phase.loc[phase.phase_id.isin(["P00", "P01"]), "execution_status"] = "complete"
    phase.to_csv(phase_path, index=False)
    repository.mkdir(exist_ok=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "ATLAS Test")
    _git(repository, "config", "user.email", "atlas-test@example.invalid")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "Synthetic P01 fixture")
    return project, private, native, artifacts


def _run(project: Path, private: Path, native: Path, artifacts: Path, phase: str, command: str):
    environment = dict(os.environ)
    environment.update(
        {
            "ATLAS_PRIVATE_ROOT": str(private),
            "ATLAS_NATIVE_ROOT": str(native),
            "ATLAS_ARTIFACT_ROOT": str(artifacts),
            "PYTHONPATH": str(project / "src"),
        }
    )
    result = subprocess.run(
        [sys.executable, str(project / "scripts" / f"run_{phase}.py"), command],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_synthetic_p00_to_p01_build_validates_and_verified_skips(tmp_path: Path) -> None:
    project, private, native, artifacts = _project_fixture(tmp_path)
    p00 = _run(project, private, native, artifacts, "p00", "dry-run")
    assert p00["status"] == "pass"
    first = _run(project, private, native, artifacts, "p01", "build")
    assert first["status"] == "pass"
    assert first["action"] == "new"
    repeated = _run(project, private, native, artifacts, "p01", "build")
    assert repeated == {**first, "action": "verified_skip"}
    validation = _run(project, private, native, artifacts, "p01", "validate")
    assert validation["status"] == "pass"
    run_dir = artifacts / "p01" / "runs" / first["run_id"]
    report = json.loads((run_dir / "P01_VALIDATION_REPORT.json").read_text())
    assert report["status"] == "pass"
    assert report["fit_invocations"] == 0
    assert report["split_invocations"] == 0
    assert len(report["representations"]) == 8
    assert len(report["figures"]) == 8
    assert all(report["checks"].values())
    f08_tikz = (run_dir / "figures" / "tikz" / "F08_nonlinear_embeddings.tex").read_text()
    f08_html = (run_dir / "figures" / "html" / "F08_nonlinear_embeddings.html").read_text()
    assert f08_tikz.count(r"\nextgroupplot") == 3
    assert all(label in f08_tikz for label in ("PCA", "PC1", "PC2", "UMAP", "t-SNE"))
    assert all(label in f08_html for label in ("PCA", "PC1", "PC2", "UMAP", "t-SNE"))
