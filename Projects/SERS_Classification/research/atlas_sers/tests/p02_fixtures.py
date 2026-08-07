from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from atlas_sers.governance.canonical import canonical_json_bytes, sha256_file, sha256_value


def synthetic_manifest(rows: int = 598) -> pd.DataFrame:
    station_design = {
        "cwa": {
            "targets": [("4_nitrophenol", 8), ("ethanol", 9), ("ethyl_paraoxon", 7)],
            "instruments": ["Agilent-3", "Mira-2", "Pendar-2", "Pendar-3", "RMX-1"],
        },
        "pills": {
            "targets": [("4_ANPP", 7), ("benzyl_fentanyl", 5), ("blank", 8)],
            "instruments": [
                "Agilent-1",
                "Agilent-3",
                "Mira-3",
                "Pendar-1",
                "Pendar-2",
                "Pendar-3",
            ],
        },
        "surfaces": {
            "targets": [("4_ANPP", 4), ("acetaminophen", 11), ("benzyl_fentanyl", 10)],
            "instruments": [
                "Agilent-3",
                "Mira-1",
                "Mira-2",
                "Pendar-2",
                "Pendar-3",
                "RMX-2",
            ],
        },
    }
    records: list[dict[str, object]] = []
    master_index = 0
    observation_index = 0
    for station, design in station_design.items():
        for target, count in design["targets"]:
            for _ in range(count):
                master = f"master-{master_index:03d}"
                master_index += 1
                for instrument in design["instruments"]:
                    records.append(
                        _observation(observation_index, master, station, target, instrument)
                    )
                    observation_index += 1
    base = list(records)
    cursor = 0
    while len(records) < rows:
        template = dict(base[cursor % len(base)])
        template["observation_uid"] = f"observation-{observation_index:04d}"
        template["first_difference_noise_mad"] = 0.02 + 0.0001 * observation_index
        records.append(template)
        observation_index += 1
        cursor += 1
    if len(records) != rows:
        raise ValueError("Synthetic P02 fixture requested fewer rows than its base design.")
    return pd.DataFrame(records)


def _observation(
    index: int, master: str, station: str, target: str, instrument: str
) -> dict[str, object]:
    return {
        "observation_uid": f"observation-{index:04d}",
        "master_sample_id": master,
        "station": station,
        "instrument": instrument,
        "sensor_family": f"sensor-{index % 4}",
        "target_analyte": target,
        "first_difference_noise_mad": 0.02 + 0.0001 * index,
        "intensity_range": 2.0 + 0.001 * index,
        "spike_fraction_proxy": 0.001 * (index % 7),
        "baseline_energy_fraction_proxy": 0.1 + 0.001 * (index % 11),
        "baseline_span_fraction_proxy": 0.2 + 0.001 * (index % 13),
        "negative_fraction": 0.0,
    }


def install_synthetic_p01(artifact_root: Path, manifest: pd.DataFrame) -> str:
    run_id = "P01-synthetic-prerequisite"
    run_dir = artifact_root / "p01" / "runs" / run_id
    run_dir.mkdir(parents=True)
    manifest.to_csv(run_dir / "primary_manifest.csv", index=False, lineterminator="\n")
    protected = {"schema_version": "synthetic-p01-protected-v1", "fixture": True}
    protected_hash = sha256_value(protected)
    (run_dir / "protected_state.json").write_bytes(
        canonical_json_bytes(protected, pretty=True)
    )
    report = {
        "status": "pass",
        "run_id": run_id,
        "protected_state_sha256": protected_hash,
        "checks": {"synthetic_prerequisite": True},
    }
    (run_dir / "P01_VALIDATION_REPORT.json").write_bytes(
        canonical_json_bytes(report, pretty=True)
    )
    artifact_hashes = {"schema_version": "synthetic-p01-hashes-v1", "files": {}}
    (run_dir / "P01_ARTIFACT_HASHES.json").write_bytes(
        canonical_json_bytes(artifact_hashes, pretty=True)
    )
    files = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file()
    }
    state = {
        "schema_version": "atlas-artifact-state-v1",
        "phase": "P01",
        "run_id": run_id,
        "protected_state_sha256": protected_hash,
        "execution_status": "complete",
        "scientific_status": "pass",
        "files": files,
    }
    (run_dir / "_STATE.json").write_bytes(canonical_json_bytes(state, pretty=True))
    latest = {
        "schema_version": "p01-latest-pointer-v1",
        "run_id": run_id,
        "status": "pass",
        "protected_state_sha256": protected_hash,
        "report_sha256": sha256_file(run_dir / "P01_VALIDATION_REPORT.json"),
        "artifact_manifest_sha256": sha256_file(run_dir / "P01_ARTIFACT_HASHES.json"),
    }
    (artifact_root / "p01" / "LATEST.json").write_bytes(
        canonical_json_bytes(latest, pretty=True)
    )
    return hashlib.sha256(manifest.to_csv(index=False).encode()).hexdigest()
