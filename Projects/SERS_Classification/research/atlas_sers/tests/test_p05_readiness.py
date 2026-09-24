from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from atlas_sers.evaluation.p05_readiness import (
    CONTRACT_RELATIVE_PATHS,
    ReadinessError,
    _integer_seeds,
    _numeric_grid,
    build_readiness_report,
    main,
    render_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_IMPORT_ROOTS = {"torch", "numpy", "pandas", "sklearn", "scipy"}


def _base_contracts() -> tuple[dict, dict, dict]:
    registry = {
        "protocol_version": "test-hyperparameters-v1",
        "deep_architecture": {"projection_dimension": 64},
        "deep_optimization": {
            "optimizer": "AdamW",
            "learning_rate": [0.0003, 0.001],
            "weight_decay": [0.00001, 0.0001, 0.001],
        },
        "losses": {
            "supcon_temperature": [0.05, 0.1, 0.2],
            "lambda_supcon": [0.1, 0.3, 1.0],
            "lambda_pair": [0.1, 0.3, 1.0],
            "lambda_coral": [0.01, 0.1],
            "lambda_domain": [0.01, 0.1],
        },
    }
    p04 = {
        "protocol_version": "test-p04-v1",
        "optimization": {
            "optimizer": "AdamW",
            "learning_rates": [0.0003, 0.001],
            "weight_decays": [0.00001, 0.0001, 0.001],
            "candidate_order": [
                "LR0.0003_WD0.00001",
                "LR0.0003_WD0.0001",
                "LR0.0003_WD0.001",
                "LR0.001_WD0.00001",
                "LR0.001_WD0.0001",
                "LR0.001_WD0.001",
            ],
            "training_seeds": [20260805, 20260817, 20260829],
        },
    }
    compute = {
        "protocol_version": "test-compute-v1",
        "stages": [
            {
                "stage": "D1_to_D5_source_development",
                "fit_estimate_low": 300,
                "fit_estimate_high": 700,
                "gate": "G3",
            }
        ],
    }
    return registry, p04, compute


def _install(
    root: Path,
    *,
    registry: dict | None = None,
    p04: dict | None = None,
    compute: dict | None = None,
) -> Path:
    default_registry, default_p04, default_compute = _base_contracts()
    contracts = root / "plan" / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    payloads = {
        "hyperparameter_registry.json": (
            default_registry if registry is None else registry
        ),
        "p04_execution_contract.json": default_p04 if p04 is None else p04,
        "compute_budget.json": default_compute if compute is None else compute,
    }
    for name, payload in payloads.items():
        (contracts / name).write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_current_contract_counts_and_illustrative_assumption() -> None:
    report = build_readiness_report(project_root=PROJECT_ROOT)

    assert report["loss_configuration_counts"] == {
        "D1": 9,
        "D2": 3,
        "D3": 27,
        "D4": 54,
        "D5": 54,
    }
    assert report["loss_configuration_total"] == 147
    assert len(report["loss_configurations"]) == 147
    assert report["optimizer_candidate_count"] == 6
    assert report["training_seed_count"] == 3
    assert report["illustrative_full_crossing"]["fits_per_selection_unit"] == 2646
    assert report["illustrative_full_crossing"]["authorizing"] is False
    assert report["total_required_fits"] is None
    assert report["resource_estimates"] is None
    assert report["status"] == "blocked_pending_design_decisions"
    assert report["scientific_execution_authorized"] is False
    assert report["scientific_fits_performed"] == 0
    assert report["report_generated"] is True
    assert report["declared_projection_dimension"] == 64


def test_recipes_are_conditional_and_complete() -> None:
    report = build_readiness_report(project_root=PROJECT_ROOT)
    by_candidate: dict[str, list[dict]] = {}
    for record in report["loss_configurations"]:
        by_candidate.setdefault(record["candidate_id"], []).append(record)

    expected = {
        "D1": {"supcon_temperature", "lambda_supcon"},
        "D2": {"lambda_pair"},
        "D3": {"supcon_temperature", "lambda_supcon", "lambda_pair"},
        "D4": {"supcon_temperature", "lambda_supcon", "lambda_pair", "lambda_coral"},
        "D5": {"supcon_temperature", "lambda_supcon", "lambda_pair", "lambda_domain"},
    }
    for candidate, keys in expected.items():
        assert by_candidate[candidate]
        assert all(
            set(record["parameters"]) == keys for record in by_candidate[candidate]
        )


def test_recipes_are_unique_and_render_repeatably() -> None:
    first = build_readiness_report(project_root=PROJECT_ROOT)
    second = build_readiness_report(project_root=PROJECT_ROOT)
    identifiers = [record["loss_id"] for record in first["loss_configurations"]]

    assert len(identifiers) == len(set(identifiers))
    assert render_report(first) == render_report(second)


def test_input_hashes_are_relative_and_root_is_not_disclosed() -> None:
    report = build_readiness_report(project_root=PROJECT_ROOT)

    assert set(report["input_hashes"]) == set(CONTRACT_RELATIVE_PATHS.values())
    assert all(len(value) == 64 for value in report["input_hashes"].values())
    assert str(PROJECT_ROOT) not in render_report(report)


def test_existing_contract_files_are_unchanged() -> None:
    paths = [PROJECT_ROOT / relative for relative in CONTRACT_RELATIVE_PATHS.values()]
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }

    build_readiness_report(project_root=PROJECT_ROOT)

    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert before == after


@pytest.mark.parametrize(
    "grid",
    [
        [],
        [0.1, 0.1],
        [0.1, float("inf")],
        [0.1, float("nan")],
        [0.1, True],
        [0.1, "0.2"],
        [0.1, -0.2],
        [0.0],
        [10**400],
        "0.1",
        None,
    ],
)
def test_numeric_grid_rejections(grid: object) -> None:
    with pytest.raises(ReadinessError):
        _numeric_grid(grid, "test_grid")


@pytest.mark.parametrize(
    "seeds",
    [[], [1, 1], [1, True], [1, 1.5], ["1"], [1, None], 1],
)
def test_seed_rejections(seeds: object) -> None:
    with pytest.raises(ReadinessError):
        _integer_seeds(seeds, "test_seeds")


def test_missing_loss_grid_rejected(tmp_path: Path) -> None:
    registry, _, _ = _base_contracts()
    del registry["losses"]
    root = _install(tmp_path, registry=registry)

    with pytest.raises(ReadinessError):
        build_readiness_report(project_root=root)


def test_optimizer_grid_disagreement_rejected(tmp_path: Path) -> None:
    registry, _, _ = _base_contracts()
    registry["deep_optimization"]["learning_rate"] = [0.0003, 0.002]
    root = _install(tmp_path, registry=registry)

    with pytest.raises(ReadinessError):
        build_readiness_report(project_root=root)


def test_optimizer_identifier_mismatch_rejected(tmp_path: Path) -> None:
    registry, _, _ = _base_contracts()
    registry["deep_optimization"]["optimizer"] = "SGD"
    root = _install(tmp_path, registry=registry)

    with pytest.raises(ReadinessError):
        build_readiness_report(project_root=root)


def test_missing_registry_optimizer_rejected(tmp_path: Path) -> None:
    registry, _, _ = _base_contracts()
    del registry["deep_optimization"]["optimizer"]
    root = _install(tmp_path, registry=registry)

    with pytest.raises(ReadinessError):
        build_readiness_report(project_root=root)


def test_missing_p04_optimizer_rejected(tmp_path: Path) -> None:
    registry, p04, _ = _base_contracts()
    del p04["optimization"]["optimizer"]
    root = _install(tmp_path, registry=registry, p04=p04)

    with pytest.raises(ReadinessError):
        build_readiness_report(project_root=root)


@pytest.mark.parametrize(
    "name",
    ["hyperparameter_registry", "p04_execution_contract", "compute_budget"],
)
def test_missing_protocol_version_rejected(tmp_path: Path, name: str) -> None:
    registry, p04, compute = _base_contracts()
    documents = {
        "hyperparameter_registry": registry,
        "p04_execution_contract": p04,
        "compute_budget": compute,
    }
    del documents[name]["protocol_version"]
    root = _install(tmp_path, registry=registry, p04=p04, compute=compute)

    with pytest.raises(ReadinessError):
        build_readiness_report(project_root=root)


def test_missing_projection_dimension_rejected(tmp_path: Path) -> None:
    registry, _, _ = _base_contracts()
    del registry["deep_architecture"]["projection_dimension"]
    root = _install(tmp_path, registry=registry)

    with pytest.raises(ReadinessError):
        build_readiness_report(project_root=root)


def test_close_float_recipes_get_distinct_ids(tmp_path: Path) -> None:
    registry, _, _ = _base_contracts()
    registry["losses"]["lambda_pair"] = [0.10000001, 0.10000002]
    root = _install(tmp_path, registry=registry)

    report = build_readiness_report(project_root=root)
    identifiers = [record["loss_id"] for record in report["loss_configurations"]]
    d2_values = {
        record["parameters"]["lambda_pair"]
        for record in report["loss_configurations"]
        if record["candidate_id"] == "D2"
    }

    assert len(identifiers) == len(set(identifiers))
    assert d2_values == {0.10000001, 0.10000002}


def test_duplicate_historical_stage_rejected(tmp_path: Path) -> None:
    registry, p04, compute = _base_contracts()
    compute["stages"].append(dict(compute["stages"][0]))
    root = _install(tmp_path, registry=registry, p04=p04, compute=compute)

    with pytest.raises(ReadinessError):
        build_readiness_report(project_root=root)


@pytest.mark.parametrize(
    ("low", "high"),
    [(-1, 700), (300, -1), (700, 300), (True, 700), (300, 700.5)],
)
def test_invalid_historical_bounds_rejected(
    tmp_path: Path, low: object, high: object
) -> None:
    registry, p04, compute = _base_contracts()
    compute["stages"][0]["fit_estimate_low"] = low
    compute["stages"][0]["fit_estimate_high"] = high
    root = _install(tmp_path, registry=registry, p04=p04, compute=compute)

    with pytest.raises(ReadinessError):
        build_readiness_report(project_root=root)


def test_cli_unrepresentable_number_returns_one(tmp_path: Path, capsys) -> None:
    registry, _, _ = _base_contracts()
    registry["losses"]["lambda_pair"] = [10**400]
    root = _install(tmp_path, registry=registry)

    assert main(["readiness", "--project-root", str(root)]) == 1
    assert capsys.readouterr().err.strip()


def test_malformed_contract_rejected(tmp_path: Path) -> None:
    root = _install(tmp_path)
    (root / "plan" / "contracts" / "compute_budget.json").write_text(
        "{ not json", encoding="utf-8"
    )

    with pytest.raises(ReadinessError):
        build_readiness_report(project_root=root)


def test_cli_readiness_and_check_exit_codes(tmp_path: Path, capsys) -> None:
    root = _install(tmp_path)

    assert main(["readiness", "--project-root", str(root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked_pending_design_decisions"

    assert main(["check", "--project-root", str(root)]) == 2
    json.loads(capsys.readouterr().out)


def test_cli_help_exits_zero_and_documents_check(capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])

    assert info.value.code == 0
    captured = capsys.readouterr().out
    assert "check" in captured
    assert "exits 2" in captured


def test_cli_invalid_input_returns_one(tmp_path: Path, capsys) -> None:
    root = _install(tmp_path)
    (root / "plan" / "contracts" / "p04_execution_contract.json").write_text(
        "[]", encoding="utf-8"
    )

    assert main(["readiness", "--project-root", str(root)]) == 1
    assert capsys.readouterr().err.strip()

    assert main(["unknown", "--project-root", str(root)]) == 1
    assert capsys.readouterr().err.strip()


def test_cli_does_not_write_artifacts(tmp_path: Path) -> None:
    root = _install(tmp_path)
    before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))

    assert main(["readiness", "--project-root", str(root)]) == 0

    after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    assert before == after


def _import_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_training_or_numeric_runtime_imports() -> None:
    module_source = (
        PROJECT_ROOT / "src" / "atlas_sers" / "evaluation" / "p05_readiness.py"
    ).read_text(encoding="utf-8")
    wrapper_source = (PROJECT_ROOT / "scripts" / "run_p05.py").read_text(
        encoding="utf-8"
    )

    assert FORBIDDEN_IMPORT_ROOTS.isdisjoint(_import_roots(module_source))
    assert FORBIDDEN_IMPORT_ROOTS.isdisjoint(_import_roots(wrapper_source))
    assert "atlas_sers.evaluation.p05_readiness" in wrapper_source
