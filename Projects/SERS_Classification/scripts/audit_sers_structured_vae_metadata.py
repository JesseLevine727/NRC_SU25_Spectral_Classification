#!/usr/bin/env python3
"""Audit metadata, support, pairing, and identifiability for structured VAE v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


PROTOCOL = "sers-structured-vae-v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cramers_v(first: Iterable[Any], second: Iterable[Any]) -> float:
    table = pd.crosstab(
        pd.Series(first, dtype="string"), pd.Series(second, dtype="string")
    )
    values = table.to_numpy(dtype=float)
    total = float(values.sum())
    if total <= 0 or min(values.shape) <= 1:
        return np.nan
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / total
    valid = expected > 0
    chi_squared = float(
        np.sum(((values - expected) ** 2)[valid] / expected[valid])
    )
    phi_squared = chi_squared / total
    rows, columns = values.shape
    corrected = max(
        0.0,
        phi_squared
        - ((columns - 1.0) * (rows - 1.0)) / max(total - 1.0, 1.0),
    )
    corrected_rows = rows - ((rows - 1.0) ** 2) / max(total - 1.0, 1.0)
    corrected_columns = (
        columns - ((columns - 1.0) ** 2) / max(total - 1.0, 1.0)
    )
    denominator = min(corrected_columns - 1.0, corrected_rows - 1.0)
    return float(np.sqrt(corrected / denominator)) if denominator > 0 else np.nan


def association_rows(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    pairs = [
        ("target_analyte", "instrument"),
        ("target_analyte", "sensor_family"),
        ("instrument", "sensor_family"),
        ("target_analyte", "station"),
        ("target_analyte", "session"),
        ("instrument", "source_format"),
        ("sensor_family", "sample_matrix"),
    ]
    rows: list[dict[str, Any]] = []
    for first, second in pairs:
        selection = manifest[first].notna() & manifest[second].notna()
        a = manifest.loc[selection, first].astype(str)
        b = manifest.loc[selection, second].astype(str)
        rows.append(
            {
                "first": first,
                "second": second,
                "n": int(selection.sum()),
                "first_levels": int(a.nunique()),
                "second_levels": int(b.nunique()),
                "nonzero_cells": int(
                    (pd.crosstab(a, b).to_numpy() > 0).sum()
                ),
                "possible_cells": int(a.nunique() * b.nunique()),
                "support_fraction": float(
                    (pd.crosstab(a, b).to_numpy() > 0).sum()
                    / max(a.nunique() * b.nunique(), 1)
                ),
                "cramers_v_bias_corrected": cramers_v(a, b),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(a, b)
                ),
                "mutual_information_nats": float(mutual_info_score(a, b)),
            }
        )
    return rows


def long_crosstab(
    manifest: pd.DataFrame, row: str, column: str
) -> pd.DataFrame:
    table = pd.crosstab(manifest[row], manifest[column])
    return (
        table.rename_axis(index=row, columns=column)
        .stack(future_stack=True)
        .rename("observations")
        .reset_index()
        .assign(supported=lambda frame: frame["observations"] > 0)
    )


def pair_summary(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    master_rows: list[dict[str, Any]] = []
    for master_id, group in manifest.groupby("master_sample_id", sort=True):
        group = group.reset_index(drop=True)
        target_count = int(group["target_analyte"].nunique())
        if target_count != 1:
            raise ValueError(f"Master sample {master_id} has multiple targets")
        cross_instrument = 0
        cross_sensor = 0
        cross_instrument_same_sensor = 0
        cross_instrument_cross_sensor = 0
        for left, right in combinations(range(len(group)), 2):
            first = group.iloc[left]
            second = group.iloc[right]
            instrument_differs = first["instrument"] != second["instrument"]
            sensor_differs = first["sensor_family"] != second["sensor_family"]
            if not instrument_differs and not sensor_differs:
                continue
            pair_rows.append(
                {
                    "master_sample_id": master_id,
                    "target_analyte": first["target_analyte"],
                    "left_observation_uid": first["observation_uid"],
                    "right_observation_uid": second["observation_uid"],
                    "left_instrument": first["instrument"],
                    "right_instrument": second["instrument"],
                    "left_sensor_family": first["sensor_family"],
                    "right_sensor_family": second["sensor_family"],
                    "cross_instrument": bool(instrument_differs),
                    "cross_sensor_family": bool(sensor_differs),
                    "cross_instrument_same_sensor": bool(
                        instrument_differs and not sensor_differs
                    ),
                    "cross_instrument_cross_sensor": bool(
                        instrument_differs and sensor_differs
                    ),
                }
            )
            cross_instrument += int(instrument_differs)
            cross_sensor += int(sensor_differs)
            cross_instrument_same_sensor += int(
                instrument_differs and not sensor_differs
            )
            cross_instrument_cross_sensor += int(
                instrument_differs and sensor_differs
            )
        master_rows.append(
            {
                "master_sample_id": master_id,
                "target_analyte": group["target_analyte"].iloc[0],
                "observations": len(group),
                "instruments": int(group["instrument"].nunique()),
                "sensor_families": int(group["sensor_family"].nunique()),
                "cross_instrument_pairs": cross_instrument,
                "cross_sensor_pairs": cross_sensor,
                "cross_instrument_same_sensor_pairs": (
                    cross_instrument_same_sensor
                ),
                "cross_instrument_cross_sensor_pairs": (
                    cross_instrument_cross_sensor
                ),
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(master_rows)


def fold_pair_count(frame: pd.DataFrame) -> dict[str, int]:
    cross_instrument_pairs = 0
    cross_sensor_pairs = 0
    paired_masters = 0
    cross_sensor_masters = 0
    for _, group in frame.groupby("master_sample_id"):
        instruments = group["instrument"].nunique()
        sensors = group["sensor_family"].nunique()
        paired_masters += int(instruments >= 2)
        cross_sensor_masters += int(sensors >= 2)
        for left, right in combinations(range(len(group)), 2):
            first = group.iloc[left]
            second = group.iloc[right]
            cross_instrument_pairs += int(
                first["instrument"] != second["instrument"]
            )
            cross_sensor_pairs += int(
                first["sensor_family"] != second["sensor_family"]
            )
    return {
        "cross_instrument_pairs": cross_instrument_pairs,
        "cross_sensor_pairs": cross_sensor_pairs,
        "masters_with_multiple_instruments": paired_masters,
        "masters_with_multiple_sensors": cross_sensor_masters,
    }


def nested_fold_audit(
    manifest: pd.DataFrame, nested: pd.DataFrame
) -> pd.DataFrame:
    indexed = manifest.set_index("observation_uid", drop=False)
    rows: list[dict[str, Any]] = []
    for outer_fold in sorted(nested["outer_fold"].unique()):
        outer = nested[nested["outer_fold"].eq(outer_fold)]
        validation_folds = sorted(
            int(value)
            for value in outer.loc[
                outer["outer_partition"].eq("development"),
                "inner_validation_fold",
            ].dropna().unique()
        )
        for inner_fold in validation_folds:
            validation_uids = outer.loc[
                outer["outer_partition"].eq("development")
                & outer["inner_validation_fold"].eq(inner_fold),
                "observation_uid",
            ]
            training_uids = outer.loc[
                outer["outer_partition"].eq("development")
                & ~outer["inner_validation_fold"].eq(inner_fold),
                "observation_uid",
            ]
            for partition, uids in (
                ("train", training_uids),
                ("validation", validation_uids),
            ):
                frame = indexed.loc[uids].reset_index(drop=True)
                pairs = fold_pair_count(frame)
                target_instrument = pd.crosstab(
                    frame["target_analyte"], frame["instrument"]
                )
                target_sensor = pd.crosstab(
                    frame["target_analyte"], frame["sensor_family"]
                )
                rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(inner_fold),
                        "partition": partition,
                        "observations": len(frame),
                        "master_samples": int(
                            frame["master_sample_id"].nunique()
                        ),
                        "targets": int(frame["target_analyte"].nunique()),
                        "instruments": int(frame["instrument"].nunique()),
                        "sensor_families": int(
                            frame["sensor_family"].nunique()
                        ),
                        "target_instrument_supported_cells": int(
                            (target_instrument.to_numpy() > 0).sum()
                        ),
                        "target_instrument_possible_cells": int(
                            target_instrument.shape[0]
                            * target_instrument.shape[1]
                        ),
                        "target_sensor_supported_cells": int(
                            (target_sensor.to_numpy() > 0).sum()
                        ),
                        "target_sensor_possible_cells": int(
                            target_sensor.shape[0] * target_sensor.shape[1]
                        ),
                        **pairs,
                    }
                )
    return pd.DataFrame(rows)


def metadata_fields(manifest: pd.DataFrame) -> pd.DataFrame:
    semantic_roles = {
        "target_analyte": "chemical target",
        "instrument": "primary nuisance/domain",
        "sensor_family": "primary nuisance/domain",
        "sensor_variant": "nested sensor nuisance",
        "master_sample_id": "physical-sample grouping/pair identity",
        "team": "operator/team nuisance candidate",
        "operator_initials": "operator nuisance candidate",
        "station": "matrix/scenario factor; chemically confounded",
        "sample_matrix": "matrix factor; chemically confounded",
        "carrier_geometry": "partial surface geometry",
        "nominal_concentration": "partial chemical covariate",
        "session": "trial-session nuisance candidate",
        "paper_sheet": "worksheet/proxy only",
        "scenario": "scenario/proxy only",
        "source_format": "instrument-derived file format",
        "instrument_serial": "partial instrument identity",
        "firmware_package": "partial instrument-specific acquisition",
        "integration_time": "partial acquisition covariate",
        "averages": "partial acquisition covariate",
        "laser_power": "constant where observed",
        "smart_tip_type": "partial instrument-specific acquisition",
        "system_suitability": "partial instrument-specific QC",
        "software_version": "constant where observed",
        "measurement_duration": "partial acquisition covariate",
        "instrument_start_date": "partial date/batch proxy",
        "rmx_ccdbias": "RMX-only acquisition covariate",
        "rmx_ccdgain": "RMX-only acquisition covariate",
        "rmx_laserwavenum": "RMX-only acquisition covariate",
        "rmx_scancount": "RMX-only acquisition covariate",
        "rmx_totalexposurems": "RMX-only acquisition covariate",
        "rmx_singleexposurems": "RMX-only acquisition covariate",
        "rmx_scanmode": "constant for RMX",
    }
    rows = []
    for field, role in semantic_roles.items():
        if field not in manifest:
            continue
        values = manifest[field]
        counts = values.astype("string").value_counts(dropna=True)
        rows.append(
            {
                "field": field,
                "semantic_role": role,
                "nonmissing": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "coverage_fraction": float(values.notna().mean()),
                "unique_nonmissing": int(values.nunique(dropna=True)),
                "largest_level_count": (
                    int(counts.iloc[0]) if len(counts) else 0
                ),
                "smallest_level_count": (
                    int(counts.iloc[-1]) if len(counts) else 0
                ),
                "top_levels": "; ".join(
                    f"{key}:{int(value)}"
                    for key, value in counts.head(8).items()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    manifest: pd.DataFrame,
    fields: pd.DataFrame,
    associations: pd.DataFrame,
    pairs: pd.DataFrame,
    masters: pd.DataFrame,
    folds: pd.DataFrame,
) -> str:
    target_instrument = pd.crosstab(
        manifest["target_analyte"], manifest["instrument"]
    )
    target_sensor = pd.crosstab(
        manifest["target_analyte"], manifest["sensor_family"]
    )
    instruments_per_master = manifest.groupby(
        "master_sample_id"
    )["instrument"].nunique()
    sensors_per_master = manifest.groupby(
        "master_sample_id"
    )["sensor_family"].nunique()
    acquisition_fields = fields[
        fields["semantic_role"].str.contains(
            "acquisition|instrument-specific", regex=True
        )
    ]
    return f"""# Structured/disentangled VAE metadata and identifiability audit

Protocol family: `{PROTOCOL}`  
Status: completed before structured-model preregistration or execution.

## Frozen population

- Spectra: {len(manifest)}
- Master samples: {manifest['master_sample_id'].nunique()}
- Analytes: {manifest['target_analyte'].nunique()}
- Instruments: {manifest['instrument'].nunique()}
- Sensor families: {manifest['sensor_family'].nunique()}
- Every master sample has exactly one analyte: {bool((manifest.groupby('master_sample_id')['target_analyte'].nunique() == 1).all())}

## Pair structure

- Masters measured on at least two instruments: {(instruments_per_master >= 2).sum()}/{len(instruments_per_master)}
- Masters spanning at least two sensor families: {(sensors_per_master >= 2).sum()}/{len(sensors_per_master)}
- Cross-instrument observation pairs: {int(pairs['cross_instrument'].sum())}
- Cross-sensor observation pairs: {int(pairs['cross_sensor_family'].sum())}
- Cross-instrument pairs using the same sensor family: {int(pairs['cross_instrument_same_sensor'].sum())}
- Cross-instrument pairs also changing sensor family: {int(pairs['cross_instrument_cross_sensor'].sum())}
- Median instruments per master: {instruments_per_master.median():.1f}

Real pairs are sufficient for same-master consistency and carefully defined
cross-reconstruction. They are not randomized interventions: instrument,
sensor, matrix, scenario, and acquisition choices remain observational.

## Support and confounding

- Target×instrument supported cells:
  {(target_instrument.to_numpy() > 0).sum()}/{target_instrument.size}
  ({(target_instrument.to_numpy() > 0).mean():.3f})
- Target×sensor supported cells:
  {(target_sensor.to_numpy() > 0).sum()}/{target_sensor.size}
  ({(target_sensor.to_numpy() > 0).mean():.3f})
- Training-fold cross-instrument pairs range:
  {folds.loc[folds['partition'].eq('train'), 'cross_instrument_pairs'].min()}–
  {folds.loc[folds['partition'].eq('train'), 'cross_instrument_pairs'].max()}
- Validation-fold cross-instrument pairs range:
  {folds.loc[folds['partition'].eq('validation'), 'cross_instrument_pairs'].min()}–
  {folds.loc[folds['partition'].eq('validation'), 'cross_instrument_pairs'].max()}

Analyte, instrument, and sensor are substantially confounded. Consequently:

1. unconditional instrument/sensor adversaries are scientifically unsafe;
2. a low nuisance-probe score is not evidence of disentanglement if analyte
   information also falls;
3. objectives and probes must be target-adjusted and cell-balanced;
4. unsupported analyte-domain cells cannot be inferred from training results;
5. held-out-domain results must retain supported/unsupported flags.

## Metadata usability

- Instrument and sensor-family labels are complete and usable as primary
  nuisance variables.
- `master_sample_id` is complete and usable for grouping and real-pair
  consistency, but it is not an independent preparation/batch identifier.
- Acquisition metadata are fragmented by instrument family. The
  {len(acquisition_fields)} audited acquisition fields have coverage ranging
  from {acquisition_fields['coverage_fraction'].min():.3f} to
  {acquisition_fields['coverage_fraction'].max():.3f}. They may be used for
  descriptive probes within supported instrument families, not as a universal
  nuisance target.
- Nominal concentration is recorded for
  {int(manifest['nominal_concentration'].notna().sum())}/{len(manifest)}
  spectra and has only
  {manifest['nominal_concentration'].nunique(dropna=True)} levels. It is a
  partial chemical covariate, not a global supervised factor.
- No defensible independent preparation ID is available. Session, paper sheet,
  scenario, team, and date can be reported as proxies but cannot establish
  preparation invariance.

## Mechanisms justified by this audit

1. A fixed-capacity chemical/nuisance partition.
2. Target-adjusted, cell-balanced instrument adversarial suppression.
3. Sensor-family adversarial suppression only as a secondary, strongly
   confounded objective.
4. Instrument- and/or sensor-conditioned decoding.
5. Same-master cross-instrument consistency using only pairs contained within
   the current grouped training partition.
6. Cross-reconstruction or latent swapping only where source and reference
   share `master_sample_id`; never manufacture a chemical ground truth across
   unrelated samples.
7. Dependence penalties as diagnostics/regularizers, not independent proof of
   semantic identifiability.

## Required negative controls

- Structural-loss weights set to zero must reproduce the frozen standard VAE.
- Permuted nuisance labels within analyte strata must remove any genuine
  adversarial/conditioning advantage.
- Permuted pair assignments within analyte strata must destroy real
  same-master consistency gains.
- Chemical-label permutation must drive supported-class performance toward
  chance.
- Every nuisance-suppression result must be paired with chemical retention,
  partition activity, and reconstruction/peak checks.

## Identifiability conclusion

The data can support a conservative claim of a **structured** or
**nuisance-suppressed chemical representation** if all registered evidence is
consistent. The observational, confounded design cannot by itself establish
unique causal factor recovery. The term **disentangled** must be reserved for
convergent evidence from partition-specific probes, real-pair behavior,
dependence diagnostics, negative controls, spectral preservation, and unseen
domain confirmation.
"""


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nato-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_structured_vae"
        / "structured_vae_v1"
        / "audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    bundle = args.nato_bundle.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(
        bundle / "core_preprocessing_manifest.csv", low_memory=False
    )
    nested = pd.read_csv(bundle / "nested_group_cv_assignments.csv")
    fields = metadata_fields(manifest)
    associations = pd.DataFrame(association_rows(manifest))
    pairs, masters = pair_summary(manifest)
    folds = nested_fold_audit(manifest, nested)
    target_instrument = long_crosstab(
        manifest, "target_analyte", "instrument"
    )
    target_sensor = long_crosstab(
        manifest, "target_analyte", "sensor_family"
    )
    instrument_sensor = long_crosstab(
        manifest, "instrument", "sensor_family"
    )

    outputs = {
        "metadata_field_audit.csv": fields,
        "categorical_association_audit.csv": associations,
        "real_pair_registry.csv": pairs,
        "master_sample_pair_summary.csv": masters,
        "nested_fold_support_audit.csv": folds,
        "target_instrument_support.csv": target_instrument,
        "target_sensor_support.csv": target_sensor,
        "instrument_sensor_support.csv": instrument_sensor,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    report = build_report(
        manifest, fields, associations, pairs, masters, folds
    )
    (output_dir / "METADATA_IDENTIFIABILITY_AUDIT.md").write_text(report)
    write_json(
        output_dir / "audit_summary.json",
        {
            "protocol_family": PROTOCOL,
            "status": "complete_before_structured_model_preregistration",
            "source_manifest": str(
                bundle / "core_preprocessing_manifest.csv"
            ),
            "source_manifest_sha256": sha256_file(
                bundle / "core_preprocessing_manifest.csv"
            ),
            "nested_assignments_sha256": sha256_file(
                bundle / "nested_group_cv_assignments.csv"
            ),
            "spectra": len(manifest),
            "master_samples": int(manifest["master_sample_id"].nunique()),
            "targets": int(manifest["target_analyte"].nunique()),
            "instruments": int(manifest["instrument"].nunique()),
            "sensor_families": int(
                manifest["sensor_family"].nunique()
            ),
            "cross_instrument_pairs": int(
                pairs["cross_instrument"].sum()
            ),
            "cross_sensor_pairs": int(
                pairs["cross_sensor_family"].sum()
            ),
            "target_instrument_support_fraction": float(
                target_instrument["supported"].mean()
            ),
            "target_sensor_support_fraction": float(
                target_sensor["supported"].mean()
            ),
            "identifiability_claim_ceiling": (
                "structured_or_nuisance_suppressed_without_convergent_"
                "multi_evidence_support"
            ),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "cross_instrument_pairs": int(
                    pairs["cross_instrument"].sum()
                ),
                "cross_sensor_pairs": int(
                    pairs["cross_sensor_family"].sum()
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
