#!/usr/bin/env python3
"""Run the exact frozen-VAE identity gate for structured VAE v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

import sers_baseline_common as baseline
import sers_vae_adequacy_common as adequacy


PROTOCOL = "sers-structured-vae-v1"
HISTORY_TOLERANCE = 1.0e-12


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def recursive_tensor_exact(first: Any, second: Any) -> bool:
    if torch.is_tensor(first) and torch.is_tensor(second):
        first_cpu = first.detach().cpu()
        second_cpu = second.detach().cpu()
        return (
            first_cpu.dtype == second_cpu.dtype
            and first_cpu.shape == second_cpu.shape
            and torch.equal(first_cpu, second_cpu)
        )
    if isinstance(first, dict) and isinstance(second, dict):
        return set(first) == set(second) and all(
            recursive_tensor_exact(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(
            recursive_tensor_exact(a, b) for a, b in zip(first, second)
        )
    return first == second


def state_exact(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> tuple[bool, str | None]:
    if set(first) != set(second):
        return False, "state_keys"
    for key in sorted(first):
        if not recursive_tensor_exact(first[key], second[key]):
            return False, key
    return True, None


def history_difference(
    first: pd.DataFrame, second: pd.DataFrame
) -> tuple[float, bool]:
    if list(first.columns) != list(second.columns) or len(first) != len(second):
        return np.inf, False
    maximum = 0.0
    for column in first:
        if pd.api.types.is_numeric_dtype(first[column]):
            left = first[column].to_numpy(dtype=float)
            right = second[column].to_numpy(dtype=float)
            finite_equal = np.array_equal(np.isfinite(left), np.isfinite(right))
            if not finite_equal:
                return np.inf, False
            finite = np.isfinite(left)
            if finite.any():
                maximum = max(
                    maximum,
                    float(np.max(np.abs(left[finite] - right[finite]))),
                )
        elif not first[column].equals(second[column]):
            return np.inf, False
    return maximum, maximum <= HISTORY_TOLERANCE


def parse_fold(path: Path) -> tuple[int, int]:
    parts = path.stem.split("__")
    outer = int(next(part[1:] for part in parts if part.startswith("o")))
    inner = int(next(part[1:] for part in parts if part.startswith("i")))
    return outer, inner


def initialize(
    output_dir: Path,
    protocol_path: Path,
    protocol_doc: Path,
    audit_dir: Path,
    nato_bundle: Path,
    adequacy_bundle: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "predeclared_protocol.json": protocol_path,
        "PREDECLARED_PROTOCOL.md": protocol_doc,
    }
    for name, source in copies.items():
        target = output_dir / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Existing {name} differs from preregistration")
        if not target.exists():
            shutil.copyfile(source, target)
    target_audit = output_dir / "audit"
    target_audit.mkdir(exist_ok=True)
    for source in sorted(audit_dir.iterdir()):
        if not source.is_file():
            continue
        target = target_audit / source.name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Existing audit differs: {source.name}")
        if not target.exists():
            shutil.copyfile(source, target)
    input_paths = {
        "protocol": protocol_path,
        "protocol_document": protocol_doc,
        "nato_artifact_catalog": nato_bundle / "artifact_hashes.json",
        "adequacy_artifact_catalog": adequacy_bundle / "artifact_hashes.json",
        "adequacy_selected_configuration": (
            adequacy_bundle / "selected_configuration.json"
        ),
        "metadata_audit_summary": audit_dir / "audit_summary.json",
    }
    write_json(
        output_dir / "input_hashes.json",
        {
            key: {"path": str(path.resolve()), "sha256": baseline.sha256_file(path)}
            for key, path in input_paths.items()
        },
    )
    write_json(
        output_dir / "environment.json",
        {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "deterministic_algorithms_required": True,
            "canonical_evaluation_device": "cpu",
        },
    )


def execution_fingerprint(protocol_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        protocol_path,
        Path(__file__),
        Path(adequacy.__file__),
        Path(baseline.__file__),
    ):
        digest.update(str(path.resolve()).encode())
        digest.update(baseline.sha256_file(path).encode())
    digest.update(b"structured-identity-cache-v1")
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=repository / "configs" / "sers_structured_vae_v1.json",
    )
    parser.add_argument(
        "--protocol-doc",
        type=Path,
        default=repository / "docs" / "SERS_STRUCTURED_VAE_PROTOCOL_V1.md",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_structured_vae"
        / "structured_vae_v1"
        / "audit",
    )
    parser.add_argument(
        "--nato-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "nato_sers_field_trial"
        / "preprocessing_v2",
    )
    parser.add_argument(
        "--adequacy-bundle",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_vae_adequacy"
        / "adequacy_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository
        / "Workspace"
        / "sers_structured_vae"
        / "structured_vae_v1",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    protocol = json.loads(args.protocol.read_text())
    if (
        protocol.get("protocol_version") != PROTOCOL
        or protocol.get("status_before_structured_model_execution")
        != "predeclared"
    ):
        raise ValueError("Structured-VAE protocol is not preregistered")
    baseline.verify_hash_catalog(args.nato_bundle)
    if baseline.sha256_file(
        args.nato_bundle / "artifact_hashes.json"
    ) != protocol["immutable_inputs"]["nato_artifact_catalog_sha256"]:
        raise ValueError("Frozen NATO bundle changed")
    if baseline.sha256_file(
        args.adequacy_bundle / "artifact_hashes.json"
    ) != protocol["immutable_inputs"]["adequacy_artifact_catalog_sha256"]:
        raise ValueError("Frozen adequacy bundle changed")
    initialize(
        args.output_dir,
        args.protocol,
        args.protocol_doc,
        args.audit_dir,
        args.nato_bundle,
        args.adequacy_bundle,
    )
    fingerprint = execution_fingerprint(args.protocol)
    dataset = baseline.load_nato_dataset(args.nato_bundle)
    folds = dataset.manifest["grouped_sample_fold_5"].to_numpy(dtype=int)
    reference_paths = sorted(
        (
            args.adequacy_bundle
            / "selection_cache"
            / "stage_2_beta"
        ).glob("*beta0p25*.pt")
    )
    if len(reference_paths) != 20:
        raise ValueError(
            f"Expected 20 frozen beta-0.25 caches, found {len(reference_paths)}"
        )
    records: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    cache_dir = args.output_dir / "identity_control"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for reference_path in reference_paths:
        outer_fold, inner_fold = parse_fold(reference_path)
        reference = torch.load(
            reference_path, map_location="cpu", weights_only=False
        )
        config_values = {
            key: value
            for key, value in reference["config"].items()
            if key != "identifier"
        }
        config = adequacy.AdequacyConfig(**config_values)
        if (
            config.beta_target != 0.25
            or config.maximum_epoch != 500
            or config.architecture != "base_maxpool"
        ):
            raise ValueError("Reference identity configuration changed")
        train_mask = (folds != outer_fold) & (folds != inner_fold)
        validation_mask = folds == inner_fold
        cache_path = cache_dir / reference_path.name
        if cache_path.exists():
            candidate = torch.load(
                cache_path, map_location="cpu", weights_only=False
            )
            if candidate.get("execution_fingerprint") != fingerprint:
                raise ValueError(f"Stale identity cache: {cache_path.name}")
            history = candidate["history"]
            states = candidate["states"]
            optimizer_states = candidate["optimizer_states"]
        else:
            history, states, optimizer_states = (
                adequacy.train_registered_checkpoints(
                    dataset.representations["arpls_minmax"][train_mask],
                    dataset.observation_uid[train_mask],
                    dataset.representations["arpls_minmax"][
                        validation_mask
                    ],
                    dataset.observation_uid[validation_mask],
                    config,
                    int(reference["run_seed"]),
                    [500],
                    torch.device(args.device),
                )
            )
            torch.save(
                {
                    "execution_fingerprint": fingerprint,
                    "config": config.record(),
                    "run_seed": int(reference["run_seed"]),
                    "history": history,
                    "states": states,
                    "optimizer_states": optimizer_states,
                },
                cache_path,
            )
        maximum_difference, history_ok = history_difference(
            history, reference["history"]
        )
        state_ok, bad_state_key = state_exact(
            states[500], reference["states"][500]
        )
        optimizer_ok = recursive_tensor_exact(
            optimizer_states[500], reference["optimizer_states"][500]
        )
        records.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "run_seed": int(reference["run_seed"]),
                "history_maximum_absolute_difference": maximum_difference,
                "history_within_tolerance": history_ok,
                "checkpoint_tensors_exact": state_ok,
                "first_bad_state_key": bad_state_key,
                "optimizer_state_exact": optimizer_ok,
                "identity_pass": history_ok and state_ok and optimizer_ok,
            }
        )
        annotated = history.copy()
        annotated.insert(0, "outer_fold", outer_fold)
        annotated.insert(1, "inner_fold", inner_fold)
        history_frames.append(annotated)
        print(
            json.dumps(
                {
                    "outer": outer_fold,
                    "inner": inner_fold,
                    "history_max_difference": maximum_difference,
                    "state_exact": state_ok,
                    "optimizer_exact": optimizer_ok,
                }
            )
        )
    result = pd.DataFrame(records).sort_values(
        ["outer_fold", "inner_fold"]
    )
    result.to_csv(args.output_dir / "identity_control_metrics.csv", index=False)
    pd.concat(history_frames, ignore_index=True).to_csv(
        args.output_dir / "identity_control_histories.csv", index=False
    )
    passed = bool(result["identity_pass"].all())
    summary = {
        "protocol": PROTOCOL,
        "candidate": "mixed_z64_zero_structure",
        "reference": "frozen adequacy stage_2_beta beta0p25",
        "fold_count": len(result),
        "history_tolerance": HISTORY_TOLERANCE,
        "maximum_history_absolute_difference": float(
            result["history_maximum_absolute_difference"].max()
        ),
        "all_checkpoint_tensors_exact": bool(
            result["checkpoint_tensors_exact"].all()
        ),
        "all_optimizer_states_exact": bool(
            result["optimizer_state_exact"].all()
        ),
        "identity_gate_passed": passed,
        "structured_selection_permitted": passed,
    }
    write_json(args.output_dir / "identity_control_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
