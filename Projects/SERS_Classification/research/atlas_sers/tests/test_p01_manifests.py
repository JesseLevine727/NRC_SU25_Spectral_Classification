from __future__ import annotations

import hashlib

import pandas as pd

from atlas_sers.data.manifests import build_population_manifests


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(6):
        rows.append(
            {
                "observation_uid": f"source-{index}",
                "master_sample_id": index // 2,
                "target_analyte": f"target-{index // 2}",
                "station": f"station-{index // 4}",
                "instrument": f"instrument-{index % 2}",
                "sensor_family": "sensor",
                "source_scan_id": index,
                "axis_sha256": hashlib.sha256(f"axis-{index}".encode()).hexdigest(),
                "intensity_sha256": hashlib.sha256(f"intensity-{index}".encode()).hexdigest(),
                "tier_notes_clear_sers": index < 4,
                "source_text_path": "/private/source",
                "comments": "private free text",
            }
        )
    return pd.DataFrame(rows)


def _contract() -> dict[str, object]:
    return {
        "primary_rows": 6,
        "notes_clear_rows": 4,
        "mira1_excluded_rows": 6,
        "physical_masters": 3,
        "targets": 3,
        "instruments": 2,
        "sensor_families": 1,
        "stations": 2,
        "unique_source_key": ["instrument", "source_scan_id"],
    }


def test_population_manifests_are_exact_sanitized_and_deterministic() -> None:
    first, report = build_population_manifests(_frame(), _contract())
    second, repeated = build_population_manifests(_frame(), _contract())
    assert report["status"] == "pass"
    assert repeated == report
    assert len(first["primary_manifest.csv"]) == 6
    assert len(first["notes_clear_manifest.csv"]) == 4
    assert len(first["mira1_excluded_manifest.csv"]) == 6
    assert first["primary_manifest.csv"].observation_uid.is_unique
    assert first["primary_manifest.csv"].equals(second["primary_manifest.csv"])
    assert "source_text_path" not in first["primary_manifest.csv"]
    assert "comments" not in first["primary_manifest.csv"]


def test_mapping_conflict_fails_closed() -> None:
    frame = _frame()
    frame.loc[1, "target_analyte"] = "conflict"
    _, report = build_population_manifests(frame, _contract())
    assert report["status"] == "fail"
    assert report["checks"]["mapping_consistent"] is False
