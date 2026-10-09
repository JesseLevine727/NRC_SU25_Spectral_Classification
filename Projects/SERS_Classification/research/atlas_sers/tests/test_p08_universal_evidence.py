"""T311 synthetic tests for the corrected T310 read-only evidence loader.

Only temporary directories and miniature synthetic records are used.  No field
path, no fitting and no resampling is exercised.  The boundary test builds a
real miniature ledger plus a real ArtifactStore so receipt hash semantics and
artifact verification are exercised, not re-implemented.
"""

import hashlib
import sqlite3

import pandas as pd
import pytest

from atlas_sers.evaluation import p08_u1_standing_accounting as standing_module
from atlas_sers.evaluation import p08_u1_store as store_module
from atlas_sers.evaluation import p08_universal_evidence as m
from atlas_sers.evaluation.p08_plan import JOB_FIELDS
from atlas_sers.evaluation.p08_u1_artifacts import ArtifactStore
from atlas_sers.governance.canonical import sha256_value


def test_pinned_manifest_derives_frozen_platform_families(tmp_path, monkeypatch):
    """The real P01 input omits this derived P02 field; retain it for deletions."""
    source = tmp_path / "manifest.csv"
    source.write_text("instrument\nAgilent-1\nAgilent-3\nMira-2\nPendar-1\nRMX-2\n")
    before = source.read_bytes()
    monkeypatch.setitem(m.PINS, "manifest", ("manifest.csv", hashlib.sha256(before).hexdigest()))
    loaded = m._pinned_frame(tmp_path, "manifest", m._prepare_roots([tmp_path]))
    assert loaded.instrument_family.tolist() == ["Agilent", "Agilent", "Mira", "Pendar", "RMX"]
    assert source.read_bytes() == before


def test_pinned_manifest_rejects_conflicting_platform_family(tmp_path, monkeypatch):
    source = tmp_path / "manifest.csv"
    source.write_text("instrument,instrument_family\nAgilent-1,Mira\n")
    monkeypatch.setitem(
        m.PINS, "manifest", ("manifest.csv", hashlib.sha256(source.read_bytes()).hexdigest())
    )
    with pytest.raises(m.UniversalEvidenceError, match="platform_family_conflict"):
        m._pinned_frame(tmp_path, "manifest", m._prepare_roots([tmp_path]))


def test_incomplete_run_refused_before_hashing(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m.load_evidence(
            private_root=tmp_path,
            graph_path=tmp_path / "missing-graph.gz",
            bridge_path=tmp_path / "missing-bridge.gz",
            completed_run_root=run,
            allowed_evidence_roots=[tmp_path],
        )
    assert excinfo.value.reason_code == "run_incomplete"


def test_secure_path_rejects_escape(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    roots = m._prepare_roots([allowed])
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._secure_file(outside, roots)
    assert excinfo.value.reason_code == "path_escape"


def test_secure_path_rejects_symlink(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    real = allowed / "real.txt"
    real.write_text("x")
    link = allowed / "link.txt"
    link.symlink_to(real)
    roots = m._prepare_roots([allowed])
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._secure_file(link, roots)
    assert excinfo.value.reason_code == "symlink_rejected"


def test_prepare_roots_rejects_symlink_ancestor(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "inner").mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._prepare_roots([link / "inner"])
    assert excinfo.value.reason_code == "allowed_root_symlink"


def test_authenticated_file_rejects_tampered_hash(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "data.bin"
    target.write_bytes(b"payload")
    roots = m._prepare_roots([allowed])
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._authenticated_file(target, roots, "0" * 64)
    assert excinfo.value.reason_code == "hash_mismatch"


def test_strict_loads_rejects_duplicates_and_overflow():
    assert m._strict_loads('{"a": 1}') == {"a": 1}
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._strict_loads('{"a": 1, "a": 2}')
    assert excinfo.value.reason_code == "json_invalid"
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._strict_loads('{"a": 1e999}')
    assert excinfo.value.reason_code == "json_invalid"
    with pytest.raises(m.UniversalEvidenceError):
        m._strict_loads('{"a": NaN}')


def test_selector_parsing_filters_context_and_strategy():
    kind, value = m.validate_selector(
        {"context_id": "ctx", "model_id": "P05-SELECTED"},
        model_id="D1",
        context_id="ctx",
    )
    assert (kind, value) == ("neural", ("ctx", "P05-SELECTED"))
    kind, value = m.validate_selector(
        {"context_id": "ctx", "model_id": "D0-M"}, model_id="D0-M", context_id="ctx"
    )
    assert (kind, value) == ("neural", ("ctx", "D0-M"))
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m.validate_selector(
            {"context_id": "ctx", "model_id": "D0-M"}, model_id="D2", context_id="ctx"
        )
    assert excinfo.value.reason_code == "selector_recipe_mismatch"
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m.validate_selector(
            {"context_id": "other", "model_id": "P05-SELECTED"},
            model_id="D1",
            context_id="ctx",
        )
    assert excinfo.value.reason_code == "selector_context_mismatch"
    kind, value = m.validate_selector(
        {"outer_run_id": "r1"}, model_id="C-RBF-SVM", context_id=None
    )
    assert (kind, value) == ("classical", "r1")
    with pytest.raises(m.UniversalEvidenceError):
        m.validate_selector({"outer_run_id": "r1"}, model_id="D1", context_id=None)


def test_apply_selector_filters_context_and_strategy():
    frame = pd.DataFrame(
        {
            "context_id": ["ctx", "ctx", "other", "ctx"],
            "model_id": ["D0-M", "P05-SELECTED", "D0-M", "D0-M"],
            "prob": ["0.1", "0.2", "0.3", "0.4"],
        }
    )
    out = m.apply_selector(frame, kind="neural", value=("ctx", "D0-M"))
    assert list(out["prob"]) == ["0.1", "0.4"]
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m.apply_selector(frame, kind="neural", value=("ctx", "missing"))
    assert excinfo.value.reason_code == "selector_no_match"
    classical = pd.DataFrame({"outer_run_id": ["r1", "r2"], "prob": ["a", "b"]})
    assert list(m.apply_selector(classical, kind="classical", value="r1")["prob"]) == [
        "a"
    ]


def test_validate_pointer_shape():
    pointer = {
        "file": "/abs/path.parquet",
        "sha256": "a" * 64,
        "selector": {"outer_run_id": "r1"},
    }
    assert m.validate_pointer(pointer) is pointer
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m.validate_pointer({"file": "/abs/path.parquet"})
    assert excinfo.value.reason_code == "pointer_invalid"


def test_audit_operation_coverage_and_dependencies():
    m.audit_operation_coverage({"a", "b"}, {"a"}, {"b"})
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m.audit_operation_coverage({"a", "b"}, {"a"}, set())
    assert excinfo.value.reason_code == "operation_coverage"
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m.audit_operation_coverage({"a"}, {"a"}, {"a"})
    assert excinfo.value.reason_code == "operation_overlap"
    graph = {"A": {"dependencies": ["B"]}, "B": {"dependencies": []}}
    m._audit_new_dependencies({"A", "B"}, {"A", "B"}, set(), graph)
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._audit_new_dependencies({"A"}, {"A"}, set(), graph)
    assert excinfo.value.reason_code == "dependency_outside_new_set"
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._audit_new_dependencies({"A", "B"}, {"A"}, set(), graph)
    assert excinfo.value.reason_code == "dependency_incomplete"


def _mini_record():
    record = {
        "stage": "source_fit",
        "model_id": "D0-M",
        "policy_id": "PP-U-SG",
        "dependencies": [],
        "context_id": "ctx",
        "seed": 7,
    }
    record["job_id"] = "P08JOB-" + m.job_sha256(record)
    return record


def _mini_binding(record, digest):
    binding = {
        "job_id": record["job_id"],
        "context_id": "ctx",
        "model_id": "D0-M",
        "stage": "source_fit",
        "seed": 7,
        "evidence_status": "complete_saved_pipeline_endpoint",
        "resolved_source_values": {},
        "evidence": [{"file": "/abs/path.parquet", "sha256": digest}],
        "scientific_execution_authorized": False,
    }
    binding["binding_sha256"] = m._sha256_json(binding)
    return binding


def test_binding_adaptation_minimal_success_and_refusal():
    record = _mini_record()
    digest = "a" * 64
    binding = _mini_binding(record, digest)
    verified = {"/abs/path.parquet": digest}
    assert m.normalize_operation_binding(binding, record, verified) is binding
    tampered = dict(binding, job_id="P08JOB-" + "b" * 64)
    tampered["binding_sha256"] = m._sha256_json(
        {key: value for key, value in tampered.items() if key != "binding_sha256"}
    )
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m.normalize_operation_binding(tampered, record, verified)
    assert excinfo.value.reason_code == "min_binding_job_mismatch"
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m.normalize_operation_binding(binding, record, {"/abs/path.parquet": "c" * 64})
    assert excinfo.value.reason_code == "min_evidence_unlisted"


def _completion():
    return {
        "summary": {"x": 1},
        "counters": {
            "completed_new_fits": m.R2_NEW_FITS,
            "completed_new_calibrations": m.R2_NEW_CAL,
            "completed_new_operations": m.R2_NEW_OPERATIONS,
            "reused_fits": m.R2_REUSED_FITS,
            "reused_predictions": m.R2_REUSED_PREDICTIONS,
            "reused_epoch_selections": m.R2_REUSED_EPOCH,
        },
        "ledger_summary": {},
        "event_verification": {},
    }


def test_check_completion_exact_counters():
    m._check_completion(_completion())
    bad = dict(_completion(), counters={"newfits": m.R2_NEW_FITS})
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._check_completion(bad)
    assert excinfo.value.reason_code == "completion_counters_invalid"
    wrong_type = _completion()
    wrong_type["counters"]["completed_new_fits"] = True
    with pytest.raises(m.UniversalEvidenceError):
        m._check_completion(wrong_type)


def test_check_close_exact_structure():
    m._check_close(
        {
            "store": {"already_closed": False, "clean": True},
            "worker_shutdown_errors": [],
        }
    )
    with pytest.raises(m.UniversalEvidenceError):
        m._check_close({"store": {"clean": True}})
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._check_close(
            {
                "store": {"already_closed": False, "clean": False},
                "worker_shutdown_errors": [],
            }
        )
    assert excinfo.value.reason_code == "close_not_clean"


def test_authenticate_permit_pins_and_destination(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    recovery = {"schema_version": m.R2_ACCOUNTING_SCHEMA}
    permit_body = {
        "reviewed": True,
        "graph_plan_sha256": m.GRAPH_PLAN_SHA256,
        "graph_archive_sha256": m.GRAPH_SHA256,
        "destination": str(run_root),
        "recovery_accounting": recovery,
    }
    permit = dict(permit_body, permit_sha256=m._sha256_json(permit_body))
    monkeypatch.setattr(m, "REVIEWED_PERMIT_SHA256", permit["permit_sha256"])
    binding = {
        "schema_version": "nato-sers-p08-u1-binding-v1",
        "permit_sha256": permit["permit_sha256"],
        "graph_plan_sha256": m.GRAPH_PLAN_SHA256,
        "destination": str(run_root),
        "global_deadline": "2026-01-01T00:00:00+00:00",
        "recovery_accounting": recovery,
    }
    m._authenticate_permit(permit, binding, run_root)
    wrong = dict(binding, destination=str(tmp_path / "elsewhere"))
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._authenticate_permit(permit, wrong, run_root)
    assert excinfo.value.reason_code == "destination_mismatch"
    bad_plan = dict(binding, graph_plan_sha256="0" * 64)
    with pytest.raises(m.UniversalEvidenceError) as excinfo_plan:
        m._authenticate_permit(permit, bad_plan, run_root)
    assert excinfo_plan.value.reason_code == "binding_graph_plan_mismatch"


def _mini_job(**overrides):
    record = {name: "z" for name in JOB_FIELDS}
    record.update(overrides)
    if not isinstance(record.get("dependencies"), (list, tuple)):
        record["dependencies"] = []
    record["dependencies"] = sorted(record["dependencies"])
    record["job_id"] = "P08JOB-" + sha256_value(
        {name: record[name] for name in JOB_FIELDS}
    )
    return record


def _build_mini_run(tmp_path, monkeypatch):
    # Patch only fixed counts/expected pins, never the verification logic.
    for name, value in (
        ("R2_NEW_FITS", 0),
        ("R2_NEW_CAL", 0),
        ("R2_NEW_OPERATIONS", 1),
        ("R2_REUSED_FITS", 1),
        ("R2_REUSED_PREDICTIONS", 1),
        ("R2_REUSED_EPOCH", 1),
        ("R2_HISTORICAL_OVERHEAD_ATTEMPTS", 14),
        ("R2_MAX_FIT_TOTAL", 15),
        ("MAX_UNIQUE_FIT_JOBS", 1),
        ("MAX_SCALAR_ATTEMPTS", 0),
    ):
        monkeypatch.setattr(m, name, value)

    # The ledger primitive validates the sealed R2 profile against its own
    # module constants, so shrink those to the one-pair synthetic graph.
    monkeypatch.setattr(store_module, "R2_REPLAY_FIT_COUNT", 0)
    monkeypatch.setattr(store_module, "R2_ADDITIONAL_FIT_COUNT", 1)
    monkeypatch.setattr(store_module, "R2_EXPECTED_REUSE_PAIRS", 1)
    monkeypatch.setattr(store_module, "R2_SELECTOR_COUNT", 1)

    model_id = "D0-M"
    context_id = "ctx"
    fit_job = _mini_job(
        policy_id="PP-U-SG",
        model_id=model_id,
        stage="source_fit",
        context_id=context_id,
    )
    pred_job = _mini_job(
        policy_id="PP-U-SG",
        model_id=model_id,
        stage="source_validation_prediction",
        context_id=context_id,
        dependencies=[fit_job["job_id"]],
    )
    selector_job = _mini_job(
        policy_id="PP-U-SG",
        model_id=model_id,
        stage="select_refit_epochs",
        context_id=context_id,
        dependencies=[pred_job["job_id"]],
    )
    endpoint_job = _mini_job(
        policy_id="PP-U-SG",
        model_id=model_id,
        stage="seed_ensemble_prediction",
        context_id=context_id,
        dependencies=[selector_job["job_id"]],
    )

    run_root = tmp_path / "run"
    recovery_accounting = {
        "schema_version": store_module.R2_ACCOUNTING_SCHEMA,
        "parent_binding_sha256": "a" * 64,
        "parent_inventory_sha256": "b" * 64,
        "replay_fit_job_ids": [],
        "additional_reuse_fit_job_ids": [fit_job["job_id"]],
        "reused_epoch_selection_job_ids": [selector_job["job_id"]],
        "baseline_active_seconds": 4000,
        "baseline_artifact_bytes": 4400000000,
    }
    binding = {
        "schema_version": "nato-sers-p08-u1-binding-v1",
        "permit_sha256": "c" * 64,
        "graph_plan_sha256": m.GRAPH_PLAN_SHA256,
        "destination": str(run_root),
        "global_deadline": "2026-01-01T00:00:00+00:00",
        "recovery_accounting": recovery_accounting,
    }
    binding_json = m._canonical(binding)
    binding_sha = hashlib.sha256(binding_json.encode("utf-8")).hexdigest()

    # ``create`` owns ledger-directory creation and expects it to be absent.
    store = store_module.P08U1Store.create(
        run_root / "ledger",
        binding,
        [fit_job, pred_job, selector_job, endpoint_job],
    )

    artifacts_root = run_root / "artifacts"
    artifacts_root.mkdir()
    artifacts = ArtifactStore(artifacts_root, binding_sha256=binding_sha)

    for job, name, blob in (
        (fit_job, "model.bin", b"model"),
        (pred_job, "preds.npy", b"pred"),
        (selector_job, "epochs.json", b"{}"),
    ):
        receipt = artifacts.write(job, {name: blob}, status="complete")
        store.record_reuse(
            job["job_id"],
            {
                "job_id": job["job_id"],
                "job_sha256": store_module.job_sha256(job),
                "current_receipt": receipt,
            },
        )
    store.seal_reuse()

    endpoint_receipt = artifacts.write(
        endpoint_job,
        {"predictions.csv": b"context_id,model_id,prob\nctx,D0-M,0.5\n"},
        status="complete",
    )
    store.start(
        endpoint_job["job_id"],
        "CPU",
        {
            "active_seconds": 4001,
            "artifact_bytes": 4400000100,
            "rss_bytes": 1024,
            "gpu_bytes": 0,
            "free_disk_bytes": 200 * 2**30,
            "active_workers": 0,
        },
    )
    store.finish(endpoint_job["job_id"], "complete", endpoint_receipt)
    store.close()

    graph_by_id = {
        fit_job["job_id"]: fit_job,
        pred_job["job_id"]: pred_job,
        selector_job["job_id"]: selector_job,
        endpoint_job["job_id"]: endpoint_job,
    }
    return {
        "run_root": run_root,
        "roots": m._prepare_roots([tmp_path]),
        "binding": binding,
        "graph_by_id": graph_by_id,
        "new_ids": set(graph_by_id),
        "fit_job": fit_job,
        "pred_job": pred_job,
        "selector_job": selector_job,
        "endpoint_job": endpoint_job,
    }


def test_audit_ledger_verifies_receipts_and_endpoint_frame(tmp_path, monkeypatch):
    fixture = _build_mini_run(tmp_path, monkeypatch)
    frames = m._audit_ledger(
        fixture["run_root"],
        fixture["roots"],
        fixture["binding"],
        fixture["graph_by_id"],
        fixture["new_ids"],
    )
    frame = frames[fixture["endpoint_job"]["job_id"]]
    assert list(frame["prob"]) == ["0.5"]
    # Corrupt the reused blob: artifact verification must now refuse.
    fit_job = fixture["fit_job"]
    blob = (
        fixture["run_root"]
        / "artifacts"
        / "jobs"
        / fit_job["job_id"][7:9]
        / fit_job["job_id"]
        / "model.bin"
    )
    blob.unlink()
    with pytest.raises(m.UniversalEvidenceError):
        m._audit_ledger(
            fixture["run_root"],
            fixture["roots"],
            fixture["binding"],
            fixture["graph_by_id"],
            fixture["new_ids"],
        )


def test_authenticate_graph_dependency_closure(monkeypatch):
    monkeypatch.setattr(m, "EXPECTED_GRAPH_JOBS", 2)
    monkeypatch.setattr(m, "EXPECTED_ALIASES", 1)

    fit_job = _mini_job(policy_id="PP-U-SG", model_id="D0-M", stage="source_fit")
    pred_job = _mini_job(
        policy_id="PP-U-SG",
        model_id="D0-M",
        stage="source_validation_prediction",
        dependencies=[fit_job["job_id"]],
    )
    by_id, _, _, _, _ = m._authenticate_graph(
        {
            "plan_sha256": m.GRAPH_PLAN_SHA256,
            "jobs": [fit_job, pred_job],
            "aliases": ["a"],
        }
    )
    assert set(by_id) == {fit_job["job_id"], pred_job["job_id"]}

    missing = _mini_job(
        policy_id="PP-U-SG",
        model_id="D0-M",
        stage="source_validation_prediction",
        dependencies=["P08JOB-" + "0" * 64],
    )
    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._authenticate_graph(
            {
                "plan_sha256": m.GRAPH_PLAN_SHA256,
                "jobs": [fit_job, missing],
                "aliases": ["a"],
            }
        )
    assert excinfo.value.reason_code == "graph_job_dependency_missing"

    cross = _mini_job(
        policy_id="PP-U-ARPLS",
        model_id="D0-M",
        stage="source_validation_prediction",
        dependencies=[fit_job["job_id"]],
    )
    with pytest.raises(m.UniversalEvidenceError) as excinfo_cross:
        m._authenticate_graph(
            {
                "plan_sha256": m.GRAPH_PLAN_SHA256,
                "jobs": [fit_job, cross],
                "aliases": ["a"],
            }
        )
    assert excinfo_cross.value.reason_code == "graph_job_dependency_policy_mismatch"


def _standing_profile(completed):
    return {
        "schema_version": standing_module.STANDING_ACCOUNTING_SCHEMA,
        "authority_sha256": "a" * 64,
        "parent_audit_sha256": "b" * 64,
        "parent_binding_sha256": "c" * 64,
        "parent_inventory_sha256": "d" * 64,
        "recovery_generation": 1,
        "replay_job_ids": [],
        "replay_counts_by_stage": {stage: {} for stage in standing_module.KNOWN_STAGES},
        "completed_job_ids_by_stage": {
            stage: sorted(completed.get(stage, ()))
            for stage in standing_module.KNOWN_STAGES
        },
        "baseline_active_seconds": 1.0,
        "baseline_artifact_bytes": 1,
    }


def _shrink_standing_constants(monkeypatch):
    for name, value in (
        ("STANDING_EXPECTED_PILOT_FITS", 1),
        ("STANDING_MIN_ACTIVE_SECONDS", 0),
        ("STANDING_MIN_ARTIFACT_BYTES", 0),
        ("STANDING_HISTORICAL_OVERHEAD", 14),
        ("STANDING_MAX_UNIQUE_FITS", 1),
        ("STANDING_MAX_SCALAR_ATTEMPTS", 1),
        ("STANDING_MAX_FIT_TOTAL", 15),
    ):
        monkeypatch.setattr(standing_module, name, value)
    for name, value in (
        ("MAX_UNIQUE_FIT_JOBS", 1),
        ("MAX_SCALAR_ATTEMPTS", 1),
        ("EXPECTED_SG_ARPLS_JOBS", 5),
    ):
        monkeypatch.setattr(m, name, value, raising=False)
    for name, value in (
        ("MAX_UNIQUE_FIT_JOBS", 1),
        ("MAX_SCALAR_ATTEMPTS", 1),
    ):
        monkeypatch.setattr(store_module, name, value, raising=False)


def _build_mini_standing_run(tmp_path, monkeypatch):
    _shrink_standing_constants(monkeypatch)
    fit = _mini_job(policy_id="PP-U-SG", model_id="D0-M", stage="source_fit", context_id="ctx")
    pred = _mini_job(
        policy_id="PP-U-SG", model_id="D0-M", stage="source_validation_prediction",
        context_id="ctx", dependencies=[fit["job_id"]],
    )
    selector = _mini_job(
        policy_id="PP-U-SG", model_id="D0-M", stage="select_refit_epochs",
        context_id="ctx", dependencies=[pred["job_id"]],
    )
    scalar = _mini_job(
        policy_id="PP-U-SG", model_id="D0-M", stage="scalar_calibration",
        context_id="ctx", dependencies=[pred["job_id"]],
    )
    endpoint = _mini_job(
        policy_id="PP-U-SG", model_id="D0-M", stage="seed_ensemble_prediction",
        context_id="ctx", dependencies=[selector["job_id"], scalar["job_id"]],
    )
    completed = {
        "source_fit": [fit["job_id"]],
        "source_validation_prediction": [pred["job_id"]],
        "select_refit_epochs": [selector["job_id"]],
        "scalar_calibration": [scalar["job_id"]],
    }
    run_root = tmp_path / "run"
    binding = {
        "schema_version": "nato-sers-p08-u1-binding-v1",
        "permit_sha256": "c" * 64,
        "graph_plan_sha256": m.GRAPH_PLAN_SHA256,
        "destination": str(run_root),
        "global_deadline": "2026-01-01T00:00:00+00:00",
        "recovery_accounting": _standing_profile(completed),
    }
    binding_json = m._canonical(binding)
    binding_sha = hashlib.sha256(binding_json.encode("utf-8")).hexdigest()
    store = store_module.P08U1Store.create(
        run_root / "ledger", binding, [fit, pred, selector, scalar, endpoint]
    )
    artifacts_root = run_root / "artifacts"
    artifacts_root.mkdir()
    artifacts = ArtifactStore(artifacts_root, binding_sha256=binding_sha)
    for job, name, blob in (
        (fit, "model.bin", b"model"),
        (pred, "preds.npy", b"pred"),
        (selector, "epochs.json", b"{}"),
        (scalar, "scalar.json", b"{}"),
    ):
        receipt = artifacts.write(job, {name: blob}, status="complete")
        store.record_reuse(
            job["job_id"],
            {
                "job_id": job["job_id"],
                "job_sha256": store_module.job_sha256(job),
                "current_receipt": receipt,
            },
        )
    store.seal_reuse()
    endpoint_receipt = artifacts.write(
        endpoint,
        {"predictions.csv": b"context_id,model_id,prob\nctx,D0-M,0.5\n"},
        status="complete",
    )
    store.start(
        endpoint["job_id"],
        "CPU",
        {
            "active_seconds": 2,
            "artifact_bytes": 2,
            "rss_bytes": 1024,
            "gpu_bytes": 0,
            "free_disk_bytes": 200 * 2**30,
            "active_workers": 0,
        },
    )
    store.finish(endpoint["job_id"], "complete", endpoint_receipt)
    ledger_summary = store.public_summary()
    event_verification = store.verify_events()
    store.close()
    graph_by_id = {job["job_id"]: job for job in (fit, pred, selector, scalar, endpoint)}
    return {
        "run_root": run_root,
        "roots": m._prepare_roots([tmp_path]),
        "binding": binding,
        "graph_by_id": graph_by_id,
        "new_ids": set(graph_by_id),
        "ledger_summary": ledger_summary,
        "event_verification": event_verification,
        "scalar_job": scalar,
        "endpoint_job": endpoint,
    }


def _controller_summary(total):
    # Exactly the keys produced by the accepted controller ``_summary``:
    # there is no ``total`` key.
    return {
        "state": "complete",
        "complete": total,
        "completed": total,
        "remaining": 0,
        "running": 0,
        "stage": "complete",
        "stages": {},
        "elapsed_seconds": 1.0,
    }


def test_standing_audit_ledger_and_completion(tmp_path, monkeypatch):
    fixture = _build_mini_standing_run(tmp_path, monkeypatch)
    frames = m._audit_ledger(
        fixture["run_root"],
        fixture["roots"],
        fixture["binding"],
        fixture["graph_by_id"],
        fixture["new_ids"],
    )
    frame = frames[fixture["endpoint_job"]["job_id"]]
    assert list(frame["prob"]) == ["0.5"]

    accounting = m._check_accounting_profile(
        m.execution_accounting(fixture["binding"])
    )
    assert accounting["schema_version"] == standing_module.STANDING_ACCOUNTING_SCHEMA
    counters = m._standing_completion_counters(accounting)
    assert counters == {
        "completed_new_fits": 0,
        "completed_new_calibrations": 0,
        "completed_new_operations": 1,
        "reused_fits": 1,
        "reused_predictions": 1,
        "reused_selectors": 1,
        "reused_scalar_calibrations": 1,
        "reused_operations": 4,
    }
    # The ledger's generic operation count is not the total reuse counter.
    assert fixture["ledger_summary"]["reuse_operations"] == 0
    completion = {
        "summary": _controller_summary(5),
        "counters": counters,
        "ledger_summary": fixture["ledger_summary"],
        "event_verification": fixture["event_verification"],
    }
    m._check_completion(completion, accounting)

    # Generic-vs-total reuse confusion must be refused.
    confused = dict(
        fixture["ledger_summary"],
        reuse_operations=counters["reused_operations"],
    )
    with pytest.raises(m.UniversalEvidenceError) as excinfo_ops:
        m._check_completion(dict(completion, ledger_summary=confused), accounting)
    assert excinfo_ops.value.reason_code == "completion_ledger_reuse_operations_mismatch"

    # Missing ledger keys are not silently skipped.
    missing = {
        key: value
        for key, value in fixture["ledger_summary"].items()
        if key != "reuse_fits"
    }
    with pytest.raises(m.UniversalEvidenceError) as excinfo_missing:
        m._check_completion(dict(completion, ledger_summary=missing), accounting)
    assert excinfo_missing.value.reason_code == "completion_ledger_reuse_fits_missing"

    # Exact counter set and values.
    with pytest.raises(m.UniversalEvidenceError):
        m._check_completion(
            dict(completion, counters=dict(counters, reused_scalar_calibrations=2)),
            accounting,
        )
    with pytest.raises(m.UniversalEvidenceError):
        m._check_completion(
            dict(
                completion,
                counters={k: v for k, v in counters.items() if k != "reused_fits"},
            ),
            accounting,
        )

    # A controller-shaped summary that disagrees with the accepted job total
    # is refused even though it contains no ``total`` key.
    with pytest.raises(m.UniversalEvidenceError) as excinfo_total:
        m._check_completion(dict(completion, summary=_controller_summary(4)), accounting)
    assert excinfo_total.value.reason_code == "completion_summary_complete"

    # The pre-close event count/head must match the ledger summary exactly.
    with pytest.raises(m.UniversalEvidenceError) as excinfo_head:
        m._check_completion(
            dict(
                completion,
                event_verification=dict(
                    fixture["event_verification"], event_head="0" * 64
                ),
            ),
            accounting,
        )
    assert excinfo_head.value.reason_code == "completion_event_head_mismatch"

    with pytest.raises(m.UniversalEvidenceError) as excinfo:
        m._check_accounting_profile({"schema_version": "nato-sers-p08-u1-unknown-v1"})
    assert excinfo.value.reason_code == "accounting_profile_unsupported"


def test_standing_counters_require_meta_and_match_tables(tmp_path, monkeypatch):
    fixture = _build_mini_standing_run(tmp_path, monkeypatch)
    accounting = m._check_accounting_profile(
        m.execution_accounting(fixture["binding"])
    )
    ledger_path = fixture["run_root"] / "ledger" / "ledger.sqlite3"
    conn = sqlite3.connect(str(ledger_path))
    conn.row_factory = sqlite3.Row
    try:
        m._check_counters_standing(conn, accounting)

        conn.execute("DELETE FROM meta WHERE key='reuse_calibration_count'")
        conn.commit()
        with pytest.raises(m.UniversalEvidenceError) as excinfo_missing:
            m._check_counters_standing(conn, accounting)
        assert excinfo_missing.value.reason_code == "ledger_meta_missing"

        conn.execute(
            "INSERT INTO meta(key,value) VALUES('reuse_calibration_count','0')"
        )
        conn.commit()
        with pytest.raises(m.UniversalEvidenceError) as excinfo_table:
            m._check_counters_standing(conn, accounting)
        assert excinfo_table.value.reason_code == "reuse_count_table_mismatch"

        conn.execute("UPDATE meta SET value='abc' WHERE key='reuse_calibration_count'")
        conn.commit()
        with pytest.raises(m.UniversalEvidenceError) as excinfo_malformed:
            m._check_counters_standing(conn, accounting)
        assert excinfo_malformed.value.reason_code == "ledger_meta_invalid"

        conn.execute("UPDATE meta SET value='1' WHERE key='reuse_calibration_count'")
        conn.commit()
        m._check_counters_standing(conn, accounting)
    finally:
        conn.close()


def test_standing_audit_ledger_refuses_mutated_scalar_blob(tmp_path, monkeypatch):
    fixture = _build_mini_standing_run(tmp_path, monkeypatch)
    scalar = fixture["scalar_job"]
    blob = (
        fixture["run_root"]
        / "artifacts"
        / "jobs"
        / scalar["job_id"][7:9]
        / scalar["job_id"]
        / "scalar.json"
    )
    blob.write_bytes(b"tampered")
    with pytest.raises(m.UniversalEvidenceError):
        m._audit_ledger(
            fixture["run_root"],
            fixture["roots"],
            fixture["binding"],
            fixture["graph_by_id"],
            fixture["new_ids"],
        )
