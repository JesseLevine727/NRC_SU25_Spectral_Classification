"""Focused tests for p08_training_evidence binding and sink integration."""

from __future__ import annotations

import copy
import hashlib
import json
import os

import pytest

from atlas_sers.evaluation import p08_plan, p08_universal_evidence
from atlas_sers.visualization import p08_live_monitor, p08_training_evidence


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


_RECIPE_FLAGS = {
    "D0-M": (False, False),
    "D1": (True, False),
    "D2": (False, True),
    "D3": (True, True),
}


def _make_job(*, model_id, policy_id, seed, stage, counter):
    fields = {
        "policy_id": policy_id,
        "representation_id": (
            "R_SG_400_1800"
            if policy_id == "PP-U-SG"
            else "R_ARPLS_400_1800"
        ),
        "array_sha256": "a" * 64,
        "context_id": f"ctx-{counter}",
        "model_id": model_id,
        "model_spec_sha256": "b" * 64,
        "stage": stage,
        "unit_id": f"unit-{counter}",
        "seed": seed,
        "candidate_id": f"cand-{counter}",
        "hyperparameter_sha256": "c" * 64,
        "fit_uid_sha256": "d" * 64,
        "validation_uid_sha256": "e" * 64,
        "test_uid_sha256": "f" * 64,
        "dependencies": [],
        "resolution": "fixed_spec",
        "evidence_status": "unapproved_future_job",
    }
    job = dict(fields)
    job["job_id"] = "P08JOB-" + p08_plan._hash(fields)
    return job


def _record(model_id, index, stage):
    supcon_enabled, paired_enabled = _RECIPE_FLAGS[model_id]
    record = {
        "epoch": index,
        "chemical_ce": 1.0 + 0.01 * index,
        "total_loss": 2.0 + 0.02 * index,
        "supcon_enabled": supcon_enabled,
        "paired_enabled": paired_enabled,
        "supcon_loss": (0.5 + 0.01 * index) if supcon_enabled else 0.0,
        "paired_loss": (0.7 + 0.01 * index) if paired_enabled else 0.0,
    }
    if stage == "source_fit":
        record.update(
            {
                "train_nll": 0.9 + 0.01 * index,
                "validation_nll": 1.1 + 0.01 * index,
                "train_balanced_accuracy": 0.5,
                "validation_balanced_accuracy": 0.4,
                "best_epoch": 1,
                "nonimproving_epochs": index - 1,
            }
        )
    return record


def _records(model_id, stage, count=30):
    return [_record(model_id, index, stage) for index in range(1, count + 1)]


def _source_summary(job, records):
    return {
        "status": "complete",
        "slot_id": job["job_id"],
        "recipe_id": job["model_id"],
        "recipe": job["model_id"],
        "seed": job["seed"],
        "epochs_completed": len(records),
        "history": copy.deepcopy(records),
    }


def _final_summary(job, records):
    return {
        "status": "complete",
        "fit_job_id": job["job_id"],
        "recipe": job["model_id"],
        "seed": job["seed"],
        "epochs": len(records),
        "epochs_completed": len(records),
        "history": copy.deepcopy(records),
    }


def _entry(job, summary):
    return {
        "job": copy.deepcopy(job),
        "summary": summary,
        "summary_sha256": "0" * 64,
        "receipt_sha256": "1" * 64,
    }


def _evidence(jobs, summaries):
    return {
        "jobs": jobs,
        "training_fit_summaries": summaries,
        "diagnostics": {
            "graph_sha256": p08_universal_evidence.GRAPH_SHA256,
            "outer_reverification_required": True,
        },
    }


class _NullStream:
    def write(self, text):
        return len(text)

    def flush(self):
        return None


def _build_monitor(
    root,
    job,
    records,
    *,
    epoch_budget,
    stop_reason,
    validation_available,
):
    root.mkdir(parents=True, exist_ok=True)
    output_dir = os.path.join(str(root), job["job_id"])
    monitor = p08_live_monitor.EpochMonitor(
        output_dir,
        job_id=job["job_id"][len("P08JOB-"):],
        model_id=job["model_id"],
        policy_id=job["policy_id"],
        seed=job["seed"],
        stage=job["stage"],
        validation_available=validation_available,
        epoch_budget=epoch_budget,
        refresh_seconds=1000,
        stream=_NullStream(),
    )
    for record in records:
        monitor(record)
    monitor.finish("complete", stop_reason=stop_reason)
    return output_dir


def _bind(evidence, tmp_path, *, roots, check=lambda: None, allowed=None):
    return p08_training_evidence.bind_training_histories(
        evidence,
        monitor_roots=roots,
        allowed_evidence_roots=(
            allowed if allowed is not None else [str(tmp_path)]
        ),
        check=check,
    )


# ---------------------------------------------------------------------------
# Retained universal-evidence sink tests
# ---------------------------------------------------------------------------


def test_collect_training_summary_records_and_filters():
    record = {
        "job_id": "P08JOB-" + "a" * 64,
        "model_id": "D1",
        "policy_id": "PP-U-SG",
        "stage": "source_fit",
    }
    raw = json.dumps({"status": "complete"}).encode("utf-8")
    receipt = {"sha256": "b" * 64}
    sink = {}
    p08_universal_evidence._collect_training_summary(
        record, receipt, {"summary.json": raw}, sink
    )
    assert set(sink) == {record["job_id"]}
    stored = sink[record["job_id"]]
    assert stored["summary"] == {"status": "complete"}
    assert stored["summary_sha256"] == hashlib.sha256(raw).hexdigest()
    assert stored["receipt_sha256"] == "b" * 64
    assert stored["job"] == record
    assert stored["job"] is not record

    classical = {}
    p08_universal_evidence._collect_training_summary(
        {
            "job_id": "x",
            "model_id": "C-RBF-SVM",
            "policy_id": "PP-U-SG",
            "stage": "source_fit",
        },
        receipt,
        {},
        classical,
    )
    assert classical == {}
    p08_universal_evidence._collect_training_summary(record, receipt, {}, None)


def test_collect_training_summary_rejects_bad_inputs():
    record = {
        "job_id": "P08JOB-" + "a" * 64,
        "model_id": "D2",
        "policy_id": "PP-U-ARPLS",
        "stage": "final_refit",
    }
    receipt = {"sha256": "b" * 64}
    raw = json.dumps({"status": "complete"}).encode("utf-8")
    sink = {}
    p08_universal_evidence._collect_training_summary(
        record, receipt, {"summary.json": raw}, sink
    )
    with pytest.raises(p08_universal_evidence.UniversalEvidenceError):
        p08_universal_evidence._collect_training_summary(
            record, receipt, {"summary.json": raw}, sink
        )
    with pytest.raises(p08_universal_evidence.UniversalEvidenceError):
        p08_universal_evidence._collect_training_summary(
            record, receipt, {}, {}
        )
    with pytest.raises(p08_universal_evidence.UniversalEvidenceError):
        p08_universal_evidence._collect_training_summary(
            record, receipt, {"summary.json": b"{not json"}, {}
        )


# ---------------------------------------------------------------------------
# Job identity / fixture integrity
# ---------------------------------------------------------------------------


def test_job_fixture_uses_plan_hash_exactly():
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=1,
    )
    fields = {name: job[name] for name in p08_plan.JOB_FIELDS}
    assert set(job.keys()) == set(p08_plan.JOB_FIELDS) | {"job_id"}
    assert job["job_id"] == "P08JOB-" + p08_plan._hash(fields)


def test_bind_rejects_tampered_job_identity(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=17,
    )
    tampered = copy.deepcopy(job)
    tampered["seed"] = 20260817
    records = _records("D1", "source_fit")
    root = tmp_path / "m"
    _build_monitor(
        root,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    evidence = _evidence(
        [tampered],
        {tampered["job_id"]: _entry(tampered, _source_summary(job, records))},
    )
    with pytest.raises(p08_training_evidence.TrainingEvidenceError):
        _bind(
            evidence,
            tmp_path,
            roots=[{"label": "m", "root": str(root)}],
        )


# ---------------------------------------------------------------------------
# End-to-end source/final with the real aggregator
# ---------------------------------------------------------------------------


def test_bind_source_and_final_end_to_end_real_aggregator(tmp_path):
    source_jobs = [
        _make_job(
            model_id=recipe,
            policy_id=policy,
            seed=seed,
            stage="source_fit",
            counter=index,
        )
        for index, (recipe, policy, seed) in enumerate(
            (
                ("D0-M", "PP-U-SG", 20260805),
                ("D1", "PP-U-SG", 20260817),
                ("D2", "PP-U-ARPLS", 20260829),
                ("D3", "PP-U-ARPLS", 20260805),
            ),
            start=1,
        )
    ]
    final_job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="final_refit",
        counter=50,
    )
    missing_job = _make_job(
        model_id="D2",
        policy_id="PP-U-SG",
        seed=20260817,
        stage="source_fit",
        counter=60,
    )

    root = tmp_path / "monitors"
    summaries = {}
    for job in source_jobs:
        records = _records(job["model_id"], "source_fit")
        _build_monitor(
            root,
            job,
            records,
            epoch_budget=200,
            stop_reason="patience",
            validation_available=True,
        )
        summaries[job["job_id"]] = _entry(job, _source_summary(job, records))

    final_records = _records("D1", "final_refit")
    _build_monitor(
        root,
        final_job,
        final_records,
        epoch_budget=30,
        stop_reason="fixed_duration",
        validation_available=False,
    )
    summaries[final_job["job_id"]] = _entry(
        final_job, _final_summary(final_job, final_records)
    )

    missing_records = _records("D2", "source_fit")
    summaries[missing_job["job_id"]] = _entry(
        missing_job, _source_summary(missing_job, missing_records)
    )

    jobs = source_jobs + [final_job, missing_job]
    evidence = _evidence(jobs, summaries)
    before = _canonical(evidence)

    result = _bind(
        evidence,
        tmp_path,
        roots=[{"label": "monitors", "root": str(root)}],
    )
    prepared = result["prepared"]
    receipt = result["private_receipt"]

    assert prepared["manifest"]["counts"]["total_expected"] == 6
    assert prepared["manifest"]["counts"]["monitored"] == 5
    assert prepared["manifest"]["counts"]["missing_history"] == 1
    assert receipt["semantic_sha256"] == prepared["semantic_sha256"]
    assert receipt["total_expected"] == 6
    assert receipt["monitored"] == 5
    assert receipt["missing_history"] == 1
    assert receipt["runtime_external_authentication_verified"] is False
    assert receipt["no_publication"] is True
    assert receipt["schema"] == p08_training_evidence.TRAINING_EVIDENCE_SCHEMA
    assert _canonical(evidence) == before

    missing_group = next(
        group
        for group in prepared["semantic"]["groups"]
        if group["policy_id"] == "PP-U-SG"
        and group["recipe"] == "D2"
        and group["stage"] == "source_fit"
    )
    assert missing_group["status"] == "missing_histories"
    assert missing_group["monitored_jobs"] == 0


def test_source_summary_actual_schema_has_no_budget_or_reason(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=2,
    )
    records = _records("D1", "source_fit")
    summary = _source_summary(job, records)
    assert set(summary) == {
        "status",
        "slot_id",
        "recipe_id",
        "recipe",
        "seed",
        "epochs_completed",
        "history",
    }
    assert "epoch_budget" not in summary
    assert "budget" not in summary
    assert "stop_reason" not in summary
    root = tmp_path / "m"
    _build_monitor(
        root,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    evidence = _evidence([job], {job["job_id"]: _entry(job, summary)})
    result = _bind(evidence, tmp_path, roots=[{"label": "m", "root": str(root)}])
    assert result["private_receipt"]["monitored"] == 1


def test_final_summary_schema_and_absence_of_validation(tmp_path):
    job = _make_job(
        model_id="D3",
        policy_id="PP-U-ARPLS",
        seed=20260829,
        stage="final_refit",
        counter=3,
    )
    records = _records("D3", "final_refit")
    summary = _final_summary(job, records)
    assert set(summary) == {
        "status",
        "fit_job_id",
        "recipe",
        "seed",
        "epochs",
        "epochs_completed",
        "history",
    }
    assert "stop_reason" not in summary
    assert all("train_nll" not in record for record in records)
    root = tmp_path / "m"
    _build_monitor(
        root,
        job,
        records,
        epoch_budget=30,
        stop_reason="fixed_duration",
        validation_available=False,
    )
    evidence = _evidence([job], {job["job_id"]: _entry(job, summary)})
    result = _bind(evidence, tmp_path, roots=[{"label": "m", "root": str(root)}])
    assert result["private_receipt"]["monitored"] == 1


# ---------------------------------------------------------------------------
# Candidate streaming / duplicates / corruption
# ---------------------------------------------------------------------------


def test_bind_duplicate_history_with_different_elapsed(tmp_path):
    job = _make_job(
        model_id="D0-M",
        policy_id="PP-U-ARPLS",
        seed=20260805,
        stage="source_fit",
        counter=7,
    )
    records = _records("D0-M", "source_fit")
    first = tmp_path / "a"
    second = tmp_path / "b"
    first_dir = _build_monitor(
        first,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    second_dir = _build_monitor(
        second,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    with open(os.path.join(first_dir, "semantic.json"), encoding="utf-8") as handle:
        first_elapsed = json.load(handle)["elapsed_seconds"]
    with open(
        os.path.join(second_dir, "semantic.json"), encoding="utf-8"
    ) as handle:
        second_elapsed = json.load(handle)["elapsed_seconds"]
    assert first_elapsed != second_elapsed

    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, records))}
    )
    result = _bind(
        evidence,
        tmp_path,
        roots=[
            {"label": "a", "root": str(first)},
            {"label": "b", "root": str(second)},
        ],
    )
    receipt = result["private_receipt"]
    assert receipt["monitored"] == 1
    selections = [
        item["selection"]
        for item in receipt["examined_monitors"]
        if item["job_id"] == job["job_id"]
    ]
    assert selections == ["selected", "exact_duplicate_numerical_history"]
    assert {item["label"] for item in receipt["examined_monitors"]} == {
        "a",
        "b",
    }


def test_bind_duplicated_corrupted_monitor_is_refused(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=8,
    )
    records = _records("D1", "source_fit")
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    _build_monitor(
        good,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    corrupted = copy.deepcopy(records)
    corrupted[0]["chemical_ce"] += 1.0
    _build_monitor(
        bad,
        job,
        corrupted,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, records))}
    )
    with pytest.raises(p08_training_evidence.TrainingEvidenceError):
        _bind(
            evidence,
            tmp_path,
            roots=[
                {"label": "good", "root": str(good)},
                {"label": "bad", "root": str(bad)},
            ],
        )


def test_bind_excludes_known_noncomplete_status(tmp_path):
    job = _make_job(
        model_id="D3",
        policy_id="PP-U-SG",
        seed=20260829,
        stage="source_fit",
        counter=16,
    )
    records = _records("D3", "source_fit")
    root = tmp_path / "m"
    root.mkdir(parents=True, exist_ok=True)
    output_dir = os.path.join(str(root), job["job_id"])
    monitor = p08_live_monitor.EpochMonitor(
        output_dir,
        job_id=job["job_id"][len("P08JOB-"):],
        model_id=job["model_id"],
        policy_id=job["policy_id"],
        seed=job["seed"],
        stage=job["stage"],
        validation_available=True,
        epoch_budget=200,
        refresh_seconds=1000,
        stream=_NullStream(),
    )
    for record in records:
        monitor(record)
    monitor.finish("interrupted", stop_reason="interrupted")

    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, records))}
    )
    result = _bind(evidence, tmp_path, roots=[{"label": "m", "root": str(root)}])
    receipt = result["private_receipt"]
    assert receipt["monitored"] == 0
    assert receipt["missing_history"] == 1
    selections = [
        item["selection"]
        for item in receipt["examined_monitors"]
        if item["job_id"] == job["job_id"]
    ]
    assert selections == ["excluded_noncomplete"]


def test_bind_rejects_unknown_monitor_status(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=15,
    )
    records = _records("D1", "source_fit")
    root = tmp_path / "m"
    output_dir = _build_monitor(
        root,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    semantic_path = os.path.join(output_dir, "semantic.json")
    with open(semantic_path, encoding="utf-8") as handle:
        document = json.load(handle)
    document["status"] = "mystery"
    with open(semantic_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)

    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, records))}
    )
    with pytest.raises(p08_training_evidence.TrainingEvidenceError):
        _bind(evidence, tmp_path, roots=[{"label": "m", "root": str(root)}])


# ---------------------------------------------------------------------------
# Missing / partial / tampered summaries
# ---------------------------------------------------------------------------


def test_bind_missing_history_is_not_fabricated(tmp_path):
    job = _make_job(
        model_id="D2",
        policy_id="PP-U-SG",
        seed=20260817,
        stage="source_fit",
        counter=9,
    )
    records = _records("D2", "source_fit")
    root = tmp_path / "empty"
    root.mkdir()
    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, records))}
    )
    result = _bind(evidence, tmp_path, roots=[{"label": "empty", "root": str(root)}])
    receipt = result["private_receipt"]
    assert receipt["monitored"] == 0
    assert receipt["missing_history"] == 1
    manifest = result["prepared"]["manifest"]
    assert manifest["counts"]["monitored"] == 0
    assert manifest["counts"]["missing_history"] == 1
    group = next(
        item
        for item in result["prepared"]["semantic"]["groups"]
        if item["policy_id"] == "PP-U-SG"
        and item["recipe"] == "D2"
        and item["stage"] == "source_fit"
    )
    assert group["status"] == "missing_histories"


def test_bind_rejects_partial_monitor_history(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=10,
    )
    full_records = _records("D1", "source_fit")
    short_records = full_records[:10]
    root = tmp_path / "m"
    _build_monitor(
        root,
        job,
        short_records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, full_records))}
    )
    with pytest.raises(p08_training_evidence.TrainingEvidenceError):
        _bind(evidence, tmp_path, roots=[{"label": "m", "root": str(root)}])


def test_bind_rejects_changed_source_history(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=21,
    )
    records = _records("D1", "source_fit")
    root = tmp_path / "m"
    _build_monitor(
        root,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    summary = _source_summary(job, records)
    summary["history"][0]["chemical_ce"] += 1.0
    evidence = _evidence([job], {job["job_id"]: _entry(job, summary)})
    with pytest.raises(p08_training_evidence.TrainingEvidenceError):
        _bind(evidence, tmp_path, roots=[{"label": "m", "root": str(root)}])


def test_bind_rejects_identity_mismatch(tmp_path):
    owner = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=31,
    )
    other = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=32,
    )
    records = _records("D1", "source_fit")
    root = tmp_path / "mon"
    root.mkdir(parents=True, exist_ok=True)
    output_dir = os.path.join(str(root), other["job_id"])
    monitor = p08_live_monitor.EpochMonitor(
        output_dir,
        job_id=owner["job_id"][len("P08JOB-"):],
        model_id=other["model_id"],
        policy_id=other["policy_id"],
        seed=other["seed"],
        stage=other["stage"],
        validation_available=True,
        epoch_budget=200,
        refresh_seconds=1000,
        stream=_NullStream(),
    )
    for record in records:
        monitor(record)
    monitor.finish("complete", stop_reason="patience")
    evidence = _evidence(
        [other],
        {other["job_id"]: _entry(other, _source_summary(other, records))},
    )
    with pytest.raises(p08_training_evidence.TrainingEvidenceError):
        _bind(evidence, tmp_path, roots=[{"label": "mon", "root": str(root)}])


def test_bind_rejects_malformed_summary_digest(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=41,
    )
    records = _records("D1", "source_fit")
    root = tmp_path / "mon"
    _build_monitor(
        root,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    entry = _entry(job, _source_summary(job, records))
    entry["summary_sha256"] = "not-a-digest"
    evidence = _evidence([job], {job["job_id"]: entry})
    with pytest.raises(p08_training_evidence.TrainingEvidenceError):
        _bind(evidence, tmp_path, roots=[{"label": "mon", "root": str(root)}])


# ---------------------------------------------------------------------------
# Roots, checks and non-mutation
# ---------------------------------------------------------------------------


def test_bind_rejects_monitor_root_outside_allowed(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=51,
    )
    records = _records("D1", "source_fit")
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    _build_monitor(
        outside,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, records))}
    )
    with pytest.raises(
        (
            p08_universal_evidence.UniversalEvidenceError,
            p08_training_evidence.TrainingEvidenceError,
            OSError,
            ValueError,
        )
    ):
        _bind(
            evidence,
            tmp_path,
            roots=[{"label": "outside", "root": str(outside)}],
            allowed=[str(allowed)],
        )


def test_bind_dangling_job_ancestor_is_refused(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=13,
    )
    records = _records("D1", "source_fit")
    root = tmp_path / "root"
    root.mkdir()
    os.symlink(
        str(tmp_path / "does-not-exist"),
        os.path.join(str(root), job["job_id"]),
    )
    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, records))}
    )
    with pytest.raises(
        (
            p08_training_evidence.TrainingEvidenceError,
            p08_universal_evidence.UniversalEvidenceError,
            OSError,
        )
    ):
        _bind(evidence, tmp_path, roots=[{"label": "root", "root": str(root)}])


def test_bind_check_is_mandatory(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=71,
    )
    records = _records("D1", "source_fit")
    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, records))}
    )
    with pytest.raises(TypeError):
        p08_training_evidence.bind_training_histories(
            evidence,
            monitor_roots=[{"label": "m", "root": str(tmp_path)}],
            allowed_evidence_roots=[str(tmp_path)],
        )
    with pytest.raises(TypeError):
        p08_training_evidence.bind_training_histories(
            evidence,
            monitor_roots=[{"label": "m", "root": str(tmp_path)}],
            allowed_evidence_roots=[str(tmp_path)],
            check=None,
        )


def test_bind_check_refusal(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=72,
    )
    records = _records("D1", "source_fit")
    root = tmp_path / "mon"
    _build_monitor(
        root,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, records))}
    )
    calls = []

    def check():
        calls.append(True)
        return False

    with pytest.raises(p08_training_evidence.TrainingEvidenceError):
        _bind(
            evidence,
            tmp_path,
            roots=[{"label": "mon", "root": str(root)}],
            check=check,
        )
    assert calls


def test_bind_does_not_mutate_inputs(tmp_path):
    job = _make_job(
        model_id="D2",
        policy_id="PP-U-ARPLS",
        seed=20260817,
        stage="source_fit",
        counter=61,
    )
    records = _records("D2", "source_fit")
    root = tmp_path / "mon"
    _build_monitor(
        root,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, records))}
    )
    before = _canonical(evidence)
    _bind(evidence, tmp_path, roots=[{"label": "mon", "root": str(root)}])
    assert _canonical(evidence) == before


def test_bind_public_prepared_carries_no_private_paths(tmp_path):
    job = _make_job(
        model_id="D1",
        policy_id="PP-U-SG",
        seed=20260805,
        stage="source_fit",
        counter=81,
    )
    records = _records("D1", "source_fit")
    root = tmp_path / "mon"
    _build_monitor(
        root,
        job,
        records,
        epoch_budget=200,
        stop_reason="patience",
        validation_available=True,
    )
    evidence = _evidence(
        [job], {job["job_id"]: _entry(job, _source_summary(job, records))}
    )
    result = _bind(evidence, tmp_path, roots=[{"label": "mon", "root": str(root)}])
    prepared_text = json.dumps(result["prepared"])
    assert str(tmp_path) not in prepared_text
    assert job["job_id"] not in prepared_text
    assert (
        result["private_receipt"]["semantic_sha256"]
        == result["prepared"]["semantic_sha256"]
    )
