"""Contract-only P05 readiness inventory.

This module enumerates the declared P05 loss grids from public contracts and
reports the design decisions that still block acquisition-aware training. It
imports only the standard library, reads three public contract files, and never
accesses scientific data, a numeric runtime, or a training loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "nato-sers-p05-readiness-v1"
BLOCKED_STATUS = "blocked_pending_design_decisions"

CONTRACT_RELATIVE_PATHS = {
    "hyperparameter_registry": "plan/contracts/hyperparameter_registry.json",
    "p04_execution_contract": "plan/contracts/p04_execution_contract.json",
    "compute_budget": "plan/contracts/compute_budget.json",
}

LOSS_PARAMETER_ORDER = (
    "supcon_temperature",
    "lambda_supcon",
    "lambda_pair",
    "lambda_coral",
    "lambda_domain",
)

CANDIDATE_ORDER = ("D1", "D2", "D3", "D4", "D5")

HISTORICAL_STAGE = "D1_to_D5_source_development"

UNRESOLVED_DECISIONS: tuple[dict[str, str], ...] = (
    {
        "decision_id": "P05-U01",
        "topic": "loss_optimizer_nesting",
        "description": (
            "Whether the declared loss grids nest inside the inherited optimizer "
            "grid, run as a separate outer loop, or are resolved in stages; the "
            "nesting determines total cost and is not fixed by this inventory."
        ),
    },
    {
        "decision_id": "P05-U02",
        "topic": "exact_fit_budget",
        "description": (
            "Exact source-context, selection-unit, conditional-ladder, and refit "
            "multiplicities; total required fits and resource estimates remain "
            "unknown and are not asserted here."
        ),
    },
    {
        "decision_id": "P05-U03",
        "topic": "auxiliary_head_architecture",
        "description": (
            "Layers, activation, and normalization of the supervised-contrastive "
            "projection head. The auxiliary projection dimension of 64 is already "
            "declared, but the head's internal layers, activation, and "
            "normalization remain unspecified."
        ),
    },
    {
        "decision_id": "P05-U04",
        "topic": "auxiliary_parameter_accounting",
        "description": (
            "Whether auxiliary-head parameters count against the frozen 250,000 "
            "trainable-parameter ceiling and how they are reported."
        ),
    },
    {
        "decision_id": "P05-U05",
        "topic": "loss_normalization",
        "description": (
            "Exact normalization of the contrastive and consistency losses, "
            "including per-anchor versus per-positive aggregation."
        ),
    },
    {
        "decision_id": "P05-U06",
        "topic": "paired_kl_cosine_combination",
        "description": (
            "Relative weighting of symmetric KL divergence and cosine distance, "
            "the probability stabilization epsilon, and whether either term is "
            "optional."
        ),
    },
    {
        "decision_id": "P05-U07",
        "topic": "master_pair_sampling",
        "description": (
            "Master-balanced sampling formula and positive-weight normalization; "
            "no exact weighting is frozen."
        ),
    },
    {
        "decision_id": "P05-U08",
        "topic": "uid_level_pair_identity",
        "description": (
            "A UID-level pair identity that distinguishes constituent observation "
            "UIDs and fitting-role or context provenance when a master/instrument "
            "view contains multiple spectra."
        ),
    },
    {
        "decision_id": "P05-U09",
        "topic": "batch_fallback",
        "description": (
            "Deterministic handling of infeasible batches and zero-positive "
            "anchors, including whether such an anchor is omitted with accounting "
            "or the batch is rejected."
        ),
    },
    {
        "decision_id": "P05-U10",
        "topic": "conditional_alignment",
        "description": (
            "Memory-bank construction, size, update rule, and class/domain cell "
            "handling for conditional CORAL."
        ),
    },
    {
        "decision_id": "P05-U11",
        "topic": "adversarial_details",
        "description": (
            "Gradient-reversal placement, the detached or conditioned input "
            "combination, the source-instrument vocabulary, and the exact sigmoid "
            "schedule endpoints and length."
        ),
    },
    {
        "decision_id": "P05-U12",
        "topic": "d0_comparison",
        "description": (
            "Whether the G3 D0 comparator reuses frozen P04 source-only evidence "
            "or requires matched source-only refits."
        ),
    },
    {
        "decision_id": "P05-U13",
        "topic": "g3_support_aggregation",
        "description": (
            "The pseudo-domain and within-source denominators, common versus "
            "failure-sensitive handling, and aggregation of the G3 advancement "
            "thresholds."
        ),
    },
    {
        "decision_id": "P05-U14",
        "topic": "calibration_epoch_inheritance",
        "description": (
            "How P04 temperature calibration and final-refit epoch rules map onto "
            "the P05 conditional ladder, if at all."
        ),
    },
)


class ReadinessError(RuntimeError):
    """Raised when a public contract is missing, malformed, or inconsistent."""


class _InvalidArguments(Exception):
    """Raised by the CLI parser when command-line arguments are invalid."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _InvalidArguments(message)


def default_project_root() -> Path:
    """Return the package root that contains the public ``plan/contracts`` tree."""
    return Path(__file__).resolve().parents[3]


def render_report(report: dict[str, Any]) -> str:
    """Render the report as deterministic, byte-equivalent JSON text."""
    return json.dumps(report, indent=2, sort_keys=True)


def build_readiness_report(*, project_root: Path | None = None) -> dict[str, Any]:
    """Build the deterministic, JSON-serializable P05 readiness report."""
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else default_project_root()
    )
    documents: dict[str, dict[str, Any]] = {}
    input_hashes: dict[str, str] = {}
    for name, relative in CONTRACT_RELATIVE_PATHS.items():
        document, raw = _load_contract(root / relative, name)
        documents[name] = document
        input_hashes[relative] = hashlib.sha256(raw).hexdigest()

    registry = documents["hyperparameter_registry"]
    p04 = documents["p04_execution_contract"]
    compute = documents["compute_budget"]

    _require_nonempty_string(
        _require_key(registry, "protocol_version", "hyperparameter_registry"),
        "hyperparameter_registry.protocol_version",
    )
    _require_nonempty_string(
        _require_key(p04, "protocol_version", "p04_execution_contract"),
        "p04_execution_contract.protocol_version",
    )
    _require_nonempty_string(
        _require_key(compute, "protocol_version", "compute_budget"),
        "compute_budget.protocol_version",
    )

    registry_optimization = _require_mapping(
        _require_key(registry, "deep_optimization", "hyperparameter_registry"),
        "hyperparameter_registry.deep_optimization",
    )
    p04_optimization = _require_mapping(
        _require_key(p04, "optimization", "p04_execution_contract"),
        "p04_execution_contract.optimization",
    )
    losses = _require_mapping(
        _require_key(registry, "losses", "hyperparameter_registry"),
        "hyperparameter_registry.losses",
    )

    registry_optimizer = _require_nonempty_string(
        _require_key(
            registry_optimization,
            "optimizer",
            "hyperparameter_registry.deep_optimization",
        ),
        "hyperparameter_registry.deep_optimization.optimizer",
    )
    p04_optimizer = _require_nonempty_string(
        _require_key(
            p04_optimization,
            "optimizer",
            "p04_execution_contract.optimization",
        ),
        "p04_execution_contract.optimization.optimizer",
    )
    if p04_optimizer != registry_optimizer:
        raise ReadinessError("P04 and registry optimizer identifiers disagree.")

    registry_learning_rates = _numeric_grid(
        _require_key(
            registry_optimization,
            "learning_rate",
            "hyperparameter_registry.deep_optimization",
        ),
        "hyperparameter_registry.deep_optimization.learning_rate",
    )
    registry_weight_decays = _numeric_grid(
        _require_key(
            registry_optimization,
            "weight_decay",
            "hyperparameter_registry.deep_optimization",
        ),
        "hyperparameter_registry.deep_optimization.weight_decay",
    )
    learning_rates = _numeric_grid(
        _require_key(
            p04_optimization,
            "learning_rates",
            "p04_execution_contract.optimization",
        ),
        "p04_execution_contract.optimization.learning_rates",
    )
    weight_decays = _numeric_grid(
        _require_key(
            p04_optimization,
            "weight_decays",
            "p04_execution_contract.optimization",
        ),
        "p04_execution_contract.optimization.weight_decays",
    )
    if learning_rates != registry_learning_rates:
        raise ReadinessError("P04 and registry learning-rate grids disagree.")
    if weight_decays != registry_weight_decays:
        raise ReadinessError("P04 and registry weight-decay grids disagree.")

    seeds = _integer_seeds(
        _require_key(
            p04_optimization,
            "training_seeds",
            "p04_execution_contract.optimization",
        ),
        "p04_execution_contract.optimization.training_seeds",
    )
    optimizer_count = len(learning_rates) * len(weight_decays)

    candidate_order = p04_optimization.get("candidate_order")
    if candidate_order is not None:
        _validate_candidate_order(candidate_order, optimizer_count)

    temperature = _numeric_grid(
        _require_key(losses, "supcon_temperature", "hyperparameter_registry.losses"),
        "hyperparameter_registry.losses.supcon_temperature",
    )
    lambda_supcon = _numeric_grid(
        _require_key(losses, "lambda_supcon", "hyperparameter_registry.losses"),
        "hyperparameter_registry.losses.lambda_supcon",
    )
    lambda_pair = _numeric_grid(
        _require_key(losses, "lambda_pair", "hyperparameter_registry.losses"),
        "hyperparameter_registry.losses.lambda_pair",
    )
    lambda_coral = _numeric_grid(
        _require_key(losses, "lambda_coral", "hyperparameter_registry.losses"),
        "hyperparameter_registry.losses.lambda_coral",
    )
    lambda_domain = _numeric_grid(
        _require_key(losses, "lambda_domain", "hyperparameter_registry.losses"),
        "hyperparameter_registry.losses.lambda_domain",
    )

    records = _enumerate_loss_configurations(
        temperature=temperature,
        lambda_supcon=lambda_supcon,
        lambda_pair=lambda_pair,
        lambda_coral=lambda_coral,
        lambda_domain=lambda_domain,
    )
    counts = _counts(records)
    total = len(records)
    illustrative = {
        "assumption": (
            "Illustrative Cartesian expansion only: every enumerated loss "
            "configuration crossed with every inherited P04 optimizer candidate "
            "and neural seed inside a single source-selection unit."
        ),
        "loss_configuration_count": total,
        "optimizer_candidate_count": optimizer_count,
        "training_seed_count": len(seeds),
        "fits_per_selection_unit": total * optimizer_count * len(seeds),
        "authorizing": False,
        "note": (
            "Not an authorized P05 budget. It excludes role and context "
            "multiplicities, the conditional candidate ladder, and final refits."
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": BLOCKED_STATUS,
        "report_generated": True,
        "scientific_execution_authorized": False,
        "scientific_fits_performed": 0,
        "contract_protocol_versions": {
            "hyperparameter_registry": registry.get("protocol_version"),
            "p04_execution_contract": p04.get("protocol_version"),
            "compute_budget": compute.get("protocol_version"),
        },
        "input_hashes": input_hashes,
        "loss_configuration_counts": counts,
        "loss_configuration_total": total,
        "loss_configurations": records,
        "optimizer_candidate_count": optimizer_count,
        "training_seed_count": len(seeds),
        "training_seeds": seeds,
        "declared_projection_dimension": _declared_projection_dimension(registry),
        "illustrative_full_crossing": illustrative,
        "total_required_fits": None,
        "resource_estimates": None,
        "historical_nonauthorizing_estimate": _historical_estimate(compute),
        "unresolved_decisions": [dict(item) for item in UNRESOLVED_DECISIONS],
    }


def _enumerate_loss_configurations(
    *,
    temperature: list[float],
    lambda_supcon: list[float],
    lambda_pair: list[float],
    lambda_coral: list[float],
    lambda_domain: list[float],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for supcon_temperature in temperature:
        for supcon in lambda_supcon:
            records.append(
                _loss_record(
                    "D1",
                    {
                        "supcon_temperature": supcon_temperature,
                        "lambda_supcon": supcon,
                    },
                )
            )
    for pair in lambda_pair:
        records.append(_loss_record("D2", {"lambda_pair": pair}))
    for supcon_temperature in temperature:
        for supcon in lambda_supcon:
            for pair in lambda_pair:
                records.append(
                    _loss_record(
                        "D3",
                        {
                            "supcon_temperature": supcon_temperature,
                            "lambda_supcon": supcon,
                            "lambda_pair": pair,
                        },
                    )
                )
    for supcon_temperature in temperature:
        for supcon in lambda_supcon:
            for pair in lambda_pair:
                for coral in lambda_coral:
                    records.append(
                        _loss_record(
                            "D4",
                            {
                                "supcon_temperature": supcon_temperature,
                                "lambda_supcon": supcon,
                                "lambda_pair": pair,
                                "lambda_coral": coral,
                            },
                        )
                    )
    for supcon_temperature in temperature:
        for supcon in lambda_supcon:
            for pair in lambda_pair:
                for domain in lambda_domain:
                    records.append(
                        _loss_record(
                            "D5",
                            {
                                "supcon_temperature": supcon_temperature,
                                "lambda_supcon": supcon,
                                "lambda_pair": pair,
                                "lambda_domain": domain,
                            },
                        )
                    )
    return records


def _loss_record(candidate_id: str, parameters: dict[str, float]) -> dict[str, Any]:
    ordered = [
        (key, parameters[key]) for key in LOSS_PARAMETER_ORDER if key in parameters
    ]
    suffix = "|".join(f"{key}={value!r}" for key, value in ordered)
    return {
        "loss_id": f"{candidate_id}:{suffix}",
        "candidate_id": candidate_id,
        "parameters": {key: value for key, value in ordered},
    }


def _counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {candidate: 0 for candidate in CANDIDATE_ORDER}
    for record in records:
        counts[str(record["candidate_id"])] += 1
    return counts


def _historical_estimate(compute: dict[str, Any]) -> dict[str, Any]:
    stages = _require_key(compute, "stages", "compute_budget")
    if not isinstance(stages, list):
        raise ReadinessError("compute_budget.stages must be a list.")
    matches = [
        stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("stage") == HISTORICAL_STAGE
    ]
    if not matches:
        raise ReadinessError(
            "compute_budget lacks the historical P05 development stage."
        )
    if len(matches) != 1:
        raise ReadinessError(
            "compute_budget must declare exactly one historical P05 stage."
        )
    stage = matches[0]
    low = _nonnegative_int(
        stage.get("fit_estimate_low"),
        "compute_budget historical fit_estimate_low",
    )
    high = _nonnegative_int(
        stage.get("fit_estimate_high"),
        "compute_budget historical fit_estimate_high",
    )
    if low > high:
        raise ReadinessError(
            "compute_budget historical P05 bounds must satisfy low <= high."
        )
    return {
        "stage": HISTORICAL_STAGE,
        "gate": stage.get("gate"),
        "fit_estimate_low": low,
        "fit_estimate_high": high,
        "authorizing": False,
        "note": (
            "Historical nonauthorizing reference retained for context only; "
            "it is not a cap and does not authorize execution."
        ),
    }


def _declared_projection_dimension(registry: dict[str, Any]) -> int:
    architecture = _require_mapping(
        _require_key(registry, "deep_architecture", "hyperparameter_registry"),
        "hyperparameter_registry.deep_architecture",
    )
    return _positive_int(
        _require_key(
            architecture,
            "projection_dimension",
            "hyperparameter_registry.deep_architecture",
        ),
        "hyperparameter_registry.deep_architecture.projection_dimension",
    )


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReadinessError(f"{name} must be a JSON object.")
    return value


def _require_key(mapping: dict[str, Any], key: str, name: str) -> Any:
    if key not in mapping:
        raise ReadinessError(f"{name} is missing required field '{key}'.")
    return mapping[key]


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReadinessError(f"{name} must be a nonempty string.")
    return value


def _numeric_grid(values: Any, name: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ReadinessError(f"{name} must be a nonempty list of numbers.")
    numbers: list[float] = []
    seen: set[float] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReadinessError(f"{name} must contain only numbers.")
        try:
            number = float(value)
        except (OverflowError, ValueError) as error:
            raise ReadinessError(
                f"{name} entries must be representable finite numbers."
            ) from error
        if not math.isfinite(number) or number <= 0:
            raise ReadinessError(
                f"{name} entries must be finite and strictly positive."
            )
        if number in seen:
            raise ReadinessError(f"{name} must not contain duplicate entries.")
        seen.add(number)
        numbers.append(number)
    return numbers


def _integer_seeds(values: Any, name: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ReadinessError(f"{name} must be a nonempty list of integers.")
    seeds: list[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ReadinessError(f"{name} must contain only integers.")
        if value in seen:
            raise ReadinessError(f"{name} must not contain duplicate seeds.")
        seen.add(value)
        seeds.append(int(value))
    return seeds


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReadinessError(f"{name} must be a strictly positive integer.")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReadinessError(f"{name} must be a nonnegative integer.")
    return int(value)


def _validate_candidate_order(values: Any, expected: int) -> None:
    if not isinstance(values, list) or not values:
        raise ReadinessError("candidate_order must be a nonempty list.")
    if any(not isinstance(value, str) or not value for value in values):
        raise ReadinessError("candidate_order entries must be nonempty strings.")
    if len(values) != expected:
        raise ReadinessError("candidate_order length disagrees with the optimizer grid.")
    if len(set(values)) != len(values):
        raise ReadinessError("candidate_order must not contain duplicate entries.")


def _load_contract(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReadinessError(f"{name} contract file is unreadable.") from error
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReadinessError(f"{name} contract file is not valid JSON.") from error
    if not isinstance(document, dict):
        raise ReadinessError(f"{name} contract must be a JSON object.")
    return document, raw


def _build_parser() -> argparse.ArgumentParser:
    return _Parser(
        prog="run_p05.py",
        description=(
            "P05 contract-only readiness inventory; no training and no data access."
        ),
        epilog=(
            "'readiness' prints the report and exits 0 when report generation "
            "succeeds. 'check' prints the same report and exits 2 because P05 "
            "design decisions remain unresolved. Exit 2 is expected and never "
            "authorizes training."
        ),
    )


def _parser_with_arguments() -> argparse.ArgumentParser:
    parser = _build_parser()
    parser.add_argument("command", choices=("readiness", "check"))
    parser.add_argument("--project-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser_with_arguments()
    try:
        args = parser.parse_args(argv)
    except _InvalidArguments as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        report = build_readiness_report(project_root=args.project_root)
    except ReadinessError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(render_report(report))
    return 0 if args.command == "readiness" else 2


if __name__ == "__main__":
    raise SystemExit(main())
