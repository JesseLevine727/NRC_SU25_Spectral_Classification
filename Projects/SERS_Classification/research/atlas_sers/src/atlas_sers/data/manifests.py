"""Canonical P01 population manifests and deterministic observation identifiers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from atlas_sers.governance.canonical import sha256_value

PATH_COLUMNS = {
    "source_text_path",
    "source_spc_path",
    "source_pdf_path",
    "source_prb_path",
    "source_dark_spc_path",
}
FREE_TEXT_COLUMNS = {
    "comments",
    "instrument_result",
    "logged_file_name",
    "master_description",
    "operator_initials",
    "rmx_peaks",
    "rmx_replacelist",
    "sample_raw",
}


def observation_uid(row: pd.Series) -> str:
    identity = {
        "instrument": str(row["instrument"]),
        "source_scan_id": int(row["source_scan_id"]),
        "source_axis_sha256": str(row["axis_sha256"]),
        "source_intensity_sha256": str(row["intensity_sha256"]),
    }
    return "OBS-" + sha256_value(identity)[:20]


def _mapping_conflicts(frame: pd.DataFrame) -> dict[str, int]:
    source = frame.groupby(["instrument", "source_scan_id"], dropna=False).agg(
        target_count=("target_analyte", "nunique"),
        master_count=("master_sample_id", "nunique"),
    )
    master = frame.groupby("master_sample_id", dropna=False).agg(
        target_count=("target_analyte", "nunique"),
        station_count=("station", "nunique"),
    )
    return {
        "source_target_conflicts": int((source.target_count != 1).sum()),
        "source_master_conflicts": int((source.master_count != 1).sum()),
        "master_target_conflicts": int((master.target_count != 1).sum()),
        "master_station_conflicts": int((master.station_count != 1).sum()),
    }


def build_population_manifests(
    authoritative: pd.DataFrame,
    population_contract: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Sanitize and validate the primary and two immutable sensitivity populations."""

    source = authoritative.copy()
    source.insert(0, "source_observation_uid", source.pop("observation_uid"))
    source.insert(0, "observation_uid", source.apply(observation_uid, axis=1))
    source = source.drop(columns=sorted((PATH_COLUMNS | FREE_TEXT_COLUMNS) & set(source.columns)))
    primary = source.reset_index(drop=True)
    notes_clear = primary[primary["tier_notes_clear_sers"].astype(bool)].reset_index(drop=True)
    mira_excluded = primary[primary["instrument"] != "Mira-1"].reset_index(drop=True)
    populations = {
        "primary_manifest.csv": primary,
        "notes_clear_manifest.csv": notes_clear,
        "mira1_excluded_manifest.csv": mira_excluded,
    }
    counts = {
        "primary_rows": len(primary),
        "notes_clear_rows": len(notes_clear),
        "mira1_excluded_rows": len(mira_excluded),
        "physical_masters": int(primary["master_sample_id"].nunique()),
        "targets": int(primary["target_analyte"].nunique()),
        "instruments": int(primary["instrument"].nunique()),
        "sensor_families": int(primary["sensor_family"].nunique()),
        "stations": int(primary["station"].nunique()),
    }
    duplicate_source_keys = int(
        primary.duplicated(list(population_contract["unique_source_key"])).sum()
    )
    mapping_conflicts = _mapping_conflicts(primary)
    checks = {
        "declared_population_counts_match": all(
            counts[key] == int(population_contract[key]) for key in counts
        ),
        "observation_uids_unique": primary["observation_uid"].nunique() == len(primary),
        "source_observation_uids_unique": primary["source_observation_uid"].nunique()
        == len(primary),
        "source_keys_unique": duplicate_source_keys == 0,
        "mapping_consistent": all(value == 0 for value in mapping_conflicts.values()),
        "sensitivity_populations_nested": set(notes_clear.observation_uid).issubset(
            set(primary.observation_uid)
        )
        and set(mira_excluded.observation_uid).issubset(set(primary.observation_uid)),
        "resolved_paths_removed": not PATH_COLUMNS.intersection(primary.columns),
        "free_text_removed": not FREE_TEXT_COLUMNS.intersection(primary.columns),
    }
    return populations, {
        "counts": counts,
        "duplicate_source_keys": duplicate_source_keys,
        "mapping_conflicts": mapping_conflicts,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
