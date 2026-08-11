"""Reconcile terminal selection/outer evidence for the complete P03 run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from atlas_sers.evaluation.p03_collect import collect_selection_evidence
from atlas_sers.evaluation.p03_plan import SELECTION_FIT_STAGES
from atlas_sers.evaluation.p03_runtime import TERMINAL_STATUSES
from atlas_sers.governance.canonical import sha256_value
from atlas_sers.governance.p03_store import P03ShardStore


@dataclass(frozen=True)
class P03ReconciledEvidence:
    fit_status: pd.DataFrame
    final_predictions: pd.DataFrame
    calibration_records: pd.DataFrame
    outer_validation: pd.DataFrame


def _canonicalize_terminal_fit_ledger(
    terminal_fit_ledger: pd.DataFrame,
    fit_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Restore manifest-declared scalar types after heterogeneous CSV reads."""

    ledger = terminal_fit_ledger.copy()
    if "seed" not in fit_manifest:
        return ledger
    if "seed" not in ledger:
        raise RuntimeError("P03 terminal fit ledger omits the planned seed field.")
    planned_seed = fit_manifest.set_index("fit_id").seed.astype("string")
    expected_seed = ledger.fit_id.map(planned_seed).astype("string")
    observed_seed = ledger.seed.astype("string")
    mismatch = observed_seed.ne(expected_seed).fillna(True)
    if mismatch.any():
        raise RuntimeError("P03 terminal fit seed differs from the fit manifest.")
    ledger["seed"] = expected_seed
    return ledger


def collect_outer_evidence(
    *,
    p03_run_root: Path,
    fit_manifest: pd.DataFrame,
    expected_run_registry: pd.DataFrame,
    selection_shard_manifest: pd.DataFrame,
    protected_state_sha256: str,
    shard_target_fits: int,
) -> P03ReconciledEvidence:
    """Prove complete terminal coverage and collect only validated outer artifacts."""

    selection = collect_selection_evidence(
        selection_run_root=p03_run_root / "selection",
        fit_manifest=fit_manifest,
        selection_shard_manifest=selection_shard_manifest,
        protected_state_sha256=protected_state_sha256,
        shard_target_fits=shard_target_fits,
    )
    executable = expected_run_registry[
        ~expected_run_registry.execution_status.astype(str).eq(
            "manifest_only_exploratory"
        )
    ].reset_index(drop=True)
    outer_store = P03ShardStore(run_root=p03_run_root / "outer")
    expected = {index: protected_state_sha256 for index in range(len(executable))}
    outer_validation = pd.DataFrame(outer_store.validation_table(expected))
    if len(outer_validation) != len(executable) or not outer_validation.valid.all():
        invalid = outer_validation.loc[
            ~outer_validation.valid, "shard_id"
        ].astype(int).tolist()
        raise RuntimeError(f"P03 outer shards are incomplete or corrupt: {invalid}")
    status_frames: list[pd.DataFrame] = [selection.fit_status]
    prediction_frames: list[pd.DataFrame] = []
    calibrations: list[dict[str, object]] = []
    for outer_index, run in executable.iterrows():
        shard = outer_store.shards / outer_store._name(  # noqa: SLF001
            int(outer_index)
        )
        descriptor = json.loads((shard / "outer_descriptor.json").read_text())
        if (
            int(descriptor["outer_index"]) != outer_index
            or str(descriptor["outer_run_id"]) != str(run.outer_run_id)
            or str(descriptor["experiment_id"]) != str(run.experiment_id)
        ):
            raise RuntimeError(f"Outer shard {outer_index} identity differs from plan.")
        statuses = pd.read_csv(shard / "fit_status.csv", low_memory=False)
        if (
            len(statuses) != int(descriptor["terminal_fit_count"])
            or sha256_value(sorted(statuses.fit_id.astype(str)))
            != str(descriptor["terminal_fit_id_sha256"])
            or not statuses.status.astype(str).isin(TERMINAL_STATUSES).all()
        ):
            raise RuntimeError(f"Outer shard {outer_index} terminal ledger differs.")
        status_frames.append(statuses)
        prediction_path = shard / "final_predictions.parquet"
        if prediction_path.is_file():
            predictions = pd.read_parquet(prediction_path)
            if len(predictions) != int(descriptor["final_prediction_rows"]):
                raise RuntimeError(
                    f"Outer shard {outer_index} prediction count differs."
                )
            prediction_frames.append(predictions)
        elif int(descriptor["final_prediction_rows"]) != 0:
            raise RuntimeError(f"Outer shard {outer_index} lost final predictions.")
        calibration_path = shard / "calibration.json"
        if calibration_path.is_file():
            calibration = json.loads(calibration_path.read_text())
            calibration.update(
                {
                    "outer_index": outer_index,
                    "outer_run_id": str(run.outer_run_id),
                    "experiment_id": str(run.experiment_id),
                }
            )
            calibrations.append(calibration)
    fit_status = pd.concat(status_frames, ignore_index=True)
    if not fit_status.fit_id.astype(str).is_unique:
        raise RuntimeError("P03 terminal ledger contains duplicate fit IDs.")
    planned_fit_ids = set(fit_manifest.fit_id.astype(str))
    observed_fit_ids = set(fit_status.fit_id.astype(str))
    if planned_fit_ids != observed_fit_ids:
        raise RuntimeError(
            "P03 terminal fit coverage differs: "
            f"missing={len(planned_fit_ids - observed_fit_ids)} "
            f"extra={len(observed_fit_ids - planned_fit_ids)}"
        )
    fit_status = _canonicalize_terminal_fit_ledger(fit_status, fit_manifest)
    planned_selection_ids = set(
        fit_manifest.loc[
            fit_manifest.stage.astype(str).isin(SELECTION_FIT_STAGES), "fit_id"
        ].astype(str)
    )
    selection_ids = set(selection.fit_status.fit_id.astype(str))
    if planned_selection_ids != selection_ids:
        raise RuntimeError("P03 selection/outer ledger boundary is inconsistent.")
    final_predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    if not final_predictions.empty:
        required = {"outer_run_id", "procedure_id", "observation_uid"}
        if not required <= set(final_predictions):
            raise RuntimeError("P03 final predictions miss endpoint identity fields.")
        if final_predictions.duplicated(
            ["outer_run_id", "procedure_id", "observation_uid"]
        ).any():
            raise RuntimeError("P03 final predictions repeat an endpoint observation.")
        planned_test_hash = (
            fit_manifest.groupby("outer_run_id").test_uid_sha256.first().to_dict()
        )
        for outer_run_id, group in final_predictions.groupby(
            "outer_run_id", sort=True
        ):
            for _, procedure in group.groupby("procedure_id", sort=True):
                observed_hash = sha256_value(
                    sorted(procedure.observation_uid.astype(str))
                )
                if observed_hash != str(planned_test_hash[str(outer_run_id)]):
                    raise RuntimeError(
                        f"Final prediction UIDs differ for outer run {outer_run_id}."
                    )
    return P03ReconciledEvidence(
        fit_status=fit_status,
        final_predictions=final_predictions,
        calibration_records=pd.DataFrame(calibrations),
        outer_validation=outer_validation,
    )
