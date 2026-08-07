"""Load and cross-validate the P00 governance registries and contracts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SCOPES = {"P", "S", "E", "X", "primary", "secondary", "exploratory"}
ALLOWED_PHASE_STATUSES = {"planned", "executing", "complete"}
ALLOWED_MODEL_STATUSES = {"planned", "frozen", "retired"}
ALLOWED_ARTIFACT_PRIVACY = {"private", "public_after_review", "public"}
ALLOWED_POLICY_STATUSES = {"planned", "prohibited"}


@dataclass(frozen=True)
class RegistrySpec:
    filename: str
    id_field: str
    allow_empty: bool = False


@dataclass
class GovernanceBundle:
    plan_root: Path
    registries: dict[str, list[dict[str, str]]]
    headers: dict[str, list[str]]
    contracts: dict[str, object]

    def rows(self, filename: str) -> list[dict[str, str]]:
        return self.registries[filename]


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def load_governance(plan_root: Path) -> GovernanceBundle:
    contract_path = plan_root / "contracts" / "p00_governance_contract.json"
    p00 = json.loads(contract_path.read_text())
    specs = p00["required_registries"]
    registries: dict[str, list[dict[str, str]]] = {}
    headers: dict[str, list[str]] = {}
    for filename in specs:
        header, rows = _read_csv(plan_root / "registries" / filename)
        headers[filename] = header
        registries[filename] = rows
    contracts = {
        path.name: json.loads(path.read_text())
        for path in sorted((plan_root / "contracts").glob("*.json"))
    }
    return GovernanceBundle(plan_root, registries, headers, contracts)


def _references(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def _ids(rows: list[dict[str, str]], field: str) -> set[str]:
    return {row[field] for row in rows if row.get(field)}


def _append_missing(errors: list[str], *, registry: str, row_id: str, fields: list[str]) -> None:
    if fields:
        errors.append(f"{registry}:{row_id} has empty required fields {','.join(fields)}")


def validate_governance(bundle: GovernanceBundle) -> dict[str, object]:
    errors: list[str] = []
    checks: dict[str, bool] = {}
    p00 = bundle.contracts["p00_governance_contract.json"]
    required = p00["required_registries"]

    identifiers_unique = True
    required_fields_complete = True
    scopes_valid = True
    for filename, id_field in required.items():
        rows = bundle.rows(filename)
        header = bundle.headers[filename]
        if id_field not in header:
            errors.append(f"{filename} is missing ID column {id_field}")
            identifiers_unique = False
            continue
        if not rows and filename != "deviations.csv":
            errors.append(f"{filename} has no rows")
            identifiers_unique = False
        values = [row.get(id_field, "") for row in rows]
        if any(not value for value in values) or len(values) != len(set(values)):
            errors.append(f"{filename} IDs are empty or duplicated")
            identifiers_unique = False
        for row in rows:
            missing = [
                field for field in header if field != "approved_by" and not row.get(field, "")
            ]
            if missing:
                _append_missing(
                    errors, registry=filename, row_id=row.get(id_field, "?"), fields=missing
                )
                required_fields_complete = False
            if "scope" in row and row["scope"] not in ALLOWED_SCOPES:
                errors.append(f"{filename}:{row.get(id_field)} has invalid scope {row['scope']}")
                scopes_valid = False
    checks["registry_ids_unique_and_nonempty"] = identifiers_unique
    checks["registry_required_fields_complete"] = required_fields_complete
    checks["registry_scopes_valid"] = scopes_valid

    expected_deviation_fields = p00["deviation_fields"]
    checks["deviation_schema_exact"] = bundle.headers["deviations.csv"] == expected_deviation_fields
    if not checks["deviation_schema_exact"]:
        errors.append("deviations.csv header does not match the P00 contract")

    phases = bundle.rows("phase_registry.csv")
    tasks = bundle.rows("task_registry.csv")
    questions = bundle.rows("research_question_registry.csv")
    policies = bundle.rows("preprocessing_policy_registry.csv")
    experiments = bundle.rows("experiment_registry.csv")
    metrics = bundle.rows("metric_registry.csv")
    models = bundle.rows("model_registry.csv")
    figures = bundle.rows("figure_registry.csv")
    gates = bundle.rows("decision_gate_registry.csv")
    artifacts = bundle.rows("artifact_registry.csv")
    phase_ids = _ids(phases, "phase_id")
    task_ids = _ids(tasks, "task_id")
    question_ids = _ids(questions, "research_question_id")
    policy_ids = _ids(policies, "policy_id")
    experiment_ids = _ids(experiments, "experiment_id")
    metric_ids = _ids(metrics, "metric_id")
    model_ids = _ids(models, "model_id")
    gate_ids = _ids(gates, "gate_id")
    artifact_ids = _ids(artifacts, "artifact_id")
    figure_ids = _ids(figures, "figure_id")

    phase_refs_valid = True
    for row in phases:
        for dependency in _references(row["depends_on"]):
            if dependency != "completed_restart" and dependency not in phase_ids:
                errors.append(f"{row['phase_id']} references unknown phase dependency {dependency}")
                phase_refs_valid = False
        for gate in _references(row["completion_gate"]):
            if gate not in gate_ids:
                errors.append(f"{row['phase_id']} references unknown completion gate {gate}")
                phase_refs_valid = False
        if row["execution_status"] not in ALLOWED_PHASE_STATUSES:
            errors.append(f"{row['phase_id']} has invalid execution status")
            phase_refs_valid = False
    checks["phase_dependencies_and_gates_valid"] = phase_refs_valid

    task_refs_valid = True
    model_by_id = {row["model_id"]: row for row in models}
    artifact_by_id = {row["artifact_id"]: row for row in artifacts}
    for row in experiments:
        if row["phase"] not in phase_ids:
            errors.append(f"{row['experiment_id']} references unknown phase {row['phase']}")
            task_refs_valid = False
        for task in _references(row["task_id"]):
            if task not in task_ids and task not in {"ALL", "T1"}:
                errors.append(f"{row['experiment_id']} references unknown task {task}")
                task_refs_valid = False
        for question in _references(row["research_question_ids"]):
            if question != "ALL" and question not in question_ids:
                errors.append(
                    f"{row['experiment_id']} references unknown research question {question}"
                )
                task_refs_valid = False
        for policy in _references(row["preprocessing_policy_ids"]):
            if policy != "ALL" and policy not in policy_ids:
                errors.append(
                    f"{row['experiment_id']} references unknown preprocessing policy {policy}"
                )
                task_refs_valid = False
        if row["model_id"] not in model_ids:
            errors.append(f"{row['experiment_id']} references unknown model {row['model_id']}")
            task_refs_valid = False
        elif model_by_id[row["model_id"]]["title"] != row["model_or_method"]:
            errors.append(f"{row['experiment_id']} model title does not match model registry")
            task_refs_valid = False
        for artifact in _references(row["artifact_ids"]):
            if artifact not in artifact_ids:
                errors.append(f"{row['experiment_id']} references unknown artifact {artifact}")
                task_refs_valid = False
            elif row["experiment_id"] not in _references(
                artifact_by_id[artifact]["producer_experiments"]
            ):
                errors.append(f"{artifact} does not declare producer {row['experiment_id']}")
                task_refs_valid = False
    checks["experiment_phase_task_model_artifact_references_valid"] = task_refs_valid

    question_refs_valid = True
    for row in questions:
        for policy in _references(row["preprocessing_policy_ids"]):
            if policy not in policy_ids:
                errors.append(
                    f"{row['research_question_id']} references unknown preprocessing policy "
                    f"{policy}"
                )
                question_refs_valid = False
        for model in _references(row["model_ids"]):
            if model not in model_ids:
                errors.append(f"{row['research_question_id']} references unknown model {model}")
                question_refs_valid = False
        for metric in _references(row["metric_ids"]):
            if metric not in metric_ids:
                errors.append(f"{row['research_question_id']} references unknown metric {metric}")
                question_refs_valid = False
        for figure in _references(row["figure_ids"]):
            if figure not in figure_ids:
                errors.append(f"{row['research_question_id']} references unknown figure {figure}")
                question_refs_valid = False
    checks["research_question_references_valid"] = question_refs_valid

    p01 = bundle.contracts["p01_governance_contract.json"]
    representation_ids = {
        item["representation_id"] for item in p01["representations"]
    }
    policy_refs_valid = True
    for row in policies:
        if row["status"] not in ALLOWED_POLICY_STATUSES:
            errors.append(f"{row['policy_id']} has invalid policy status {row['status']}")
            policy_refs_valid = False
        if row["status"] == "prohibited":
            if row["policy_id"] != "PP-POSTTEST-HYBRID":
                errors.append(f"{row['policy_id']} is an unrecognized prohibited policy")
                policy_refs_valid = False
            continue
        for action in _references(row["candidate_action_ids"]):
            if action not in representation_ids:
                errors.append(f"{row['policy_id']} references unknown action {action}")
                policy_refs_valid = False
        if row["fallback_policy_id"] not in policy_ids:
            errors.append(
                f"{row['policy_id']} references unknown fallback {row['fallback_policy_id']}"
            )
            policy_refs_valid = False
    checks["preprocessing_policy_references_and_statuses_valid"] = policy_refs_valid

    allowed_preprocessing_regimes = set(
        bundle.contracts["split_contract.json"]["preprocessing_information_regimes"]
    ) | {"not_applicable", "all_registered"}
    preprocessing_regimes_valid = True
    for row in experiments:
        for regime in _references(row["preprocessing_information_regime"]):
            if regime not in allowed_preprocessing_regimes:
                errors.append(
                    f"{row['experiment_id']} has unknown preprocessing regime {regime}"
                )
                preprocessing_regimes_valid = False
    checks["preprocessing_information_regimes_valid"] = preprocessing_regimes_valid

    model_refs_valid = True
    for row in models:
        if row["phase_introduced"] not in phase_ids:
            errors.append(f"{row['model_id']} references unknown phase")
            model_refs_valid = False
        if row["status"] not in ALLOWED_MODEL_STATUSES:
            errors.append(f"{row['model_id']} has invalid status")
            model_refs_valid = False
    checks["model_registry_references_valid"] = model_refs_valid

    figure_refs_valid = all(row["phase"] in phase_ids for row in figures)
    if not figure_refs_valid:
        errors.append("figure registry contains an unknown phase reference")
    checks["figure_phase_references_valid"] = figure_refs_valid

    artifact_refs_valid = True
    for row in artifacts:
        if row["phase"] not in phase_ids:
            errors.append(f"{row['artifact_id']} references unknown phase")
            artifact_refs_valid = False
        producers = _references(row["producer_experiments"])
        if any(producer != "P00" and producer not in experiment_ids for producer in producers):
            errors.append(f"{row['artifact_id']} references an unknown producer")
            artifact_refs_valid = False
        if row["privacy"] not in ALLOWED_ARTIFACT_PRIVACY:
            errors.append(f"{row['artifact_id']} has invalid privacy")
            artifact_refs_valid = False
        if not row["logical_path"].startswith("${ATLAS_ARTIFACT_ROOT}/"):
            errors.append(f"{row['artifact_id']} is not artifact-root relative")
            artifact_refs_valid = False
        if "ATLAS_PRIVATE_ROOT" in row["logical_path"]:
            errors.append(f"{row['artifact_id']} points into the input namespace")
            artifact_refs_valid = False
        if row["phase"] == "P00" and row["privacy"] != "private":
            errors.append(f"{row['artifact_id']} must remain private")
            artifact_refs_valid = False
    checks["artifact_paths_privacy_and_producers_valid"] = artifact_refs_valid

    contracts_valid = True
    for name, contract in bundle.contracts.items():
        if not isinstance(contract, dict):
            errors.append(f"{name} is not a JSON object")
            contracts_valid = False
            continue
        version = str(contract.get("protocol_version", contract.get("$id", "")))
        if not version.startswith("atlas-sers-"):
            errors.append(f"{name} does not use the ATLAS namespace")
            contracts_valid = False
    checks["contracts_parse_and_use_atlas_namespace"] = contracts_valid

    deep_budget_valid = True
    hyperparameters = bundle.contracts["hyperparameter_registry.json"]
    parameter_max = int(hyperparameters["deep_architecture"]["maximum_parameters"])
    epoch_max = int(hyperparameters["deep_optimization"]["maximum_epochs"])
    for row in models:
        if row["kind"] != "deep":
            continue
        if int(row["parameter_budget"]) > parameter_max or int(row["epoch_budget"]) > epoch_max:
            errors.append(f"{row['model_id']} exceeds the frozen deep budget")
            deep_budget_valid = False
    checks["deep_models_obey_parameter_and_epoch_budgets"] = deep_budget_valid

    checks["all_governance_checks_pass"] = all(checks.values())
    return {
        "schema_version": "p00-registry-validation-v1",
        "status": "pass" if checks["all_governance_checks_pass"] else "fail",
        "checks": checks,
        "errors": sorted(set(errors)),
        "counts": {filename: len(rows) for filename, rows in bundle.registries.items()},
    }
