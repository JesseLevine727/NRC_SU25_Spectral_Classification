"""Synthetic, in-memory tests for the P08-T039 pure attempt-journal kernel.

No filesystem, clock, device or model access; every identifier is invented.
These checks do not prove a durable journal, filesystem persistence, an
authenticated U0 mapping or a complete official smoke run. Small fixtures
(<= 78 fits) are permitted by the bounded manifest schema.
"""

import copy

import pytest

from atlas_sers.evaluation.p08_attempt_journal import (
    JournalError,
    replay_attempt_journal,
    require_scientific_execution,
)
from atlas_sers.evaluation.p08_qc_blocks import canonical_sha256

MANIFEST_KEYS = {
    "schema_version",
    "execution_authorized",
    "proposal_sha256",
    "jobs",
    "manifest_sha256",
}
JOB_KEYS = {"job_id", "stage", "worker", "dependencies"}
EVENT_KEYS = {
    "seq",
    "previous_sha256",
    "session_id",
    "event_type",
    "elapsed_ns",
    "artifact_bytes",
    "job_id",
    "status",
    "receipt_sha256",
    "event_sha256",
}
ATTEMPT_KEYS = {"stage", "worker", "session_id", "status", "receipt_sha256"}
SUMMARY_KEYS = {
    "schema_version",
    "execution_authorized",
    "manifest_sha256",
    "head_sha256",
    "session_count",
    "active_session_id",
    "closed_session_wall_ns",
    "active_session_elapsed_ns",
    "active_wall_ns",
    "new_artifact_bytes",
    "model_fit_attempts",
    "source_prediction_attempts",
    "active_cpu_workers",
    "active_gpu_workers",
    "attempts",
    "in_flight_job_ids",
    "succeeded_job_ids",
    "failed_job_ids",
    "interrupted_job_ids",
    "restart_allowed",
    "journal_state",
    "summary_sha256",
}
REASONS = {
    "invalid_journal_input",
    "invalid_manifest",
    "manifest_hash_mismatch",
    "invalid_event",
    "event_hash_mismatch",
    "journal_head_mismatch",
    "invalid_transition",
    "scientific_execution_not_authorized",
}
R1, R2, R3, R4 = "1" * 64, "2" * 64, "3" * 64, "4" * 64
ZERO = "0" * 64


def digest(payload, self_key):
    return canonical_sha256({k: v for k, v in payload.items() if k != self_key})


def job(jid, stage, worker, deps=()):
    return {"job_id": jid, "stage": stage, "worker": worker, "dependencies": list(deps)}


def jobs_for(n_fits, n_preds=None):
    n_preds = n_fits if n_preds is None else n_preds
    fits = [
        job(f"fit{i:03d}", "source_fit", "cpu" if i % 2 else "gpu") for i in range(1, n_fits + 1)
    ]
    preds = [
        job(
            f"pred{i:03d}",
            "source_validation_prediction",
            "cpu" if i % 2 else "gpu",
            [f"fit{i:03d}"],
        )
        for i in range(1, n_preds + 1)
    ]
    return fits + preds


def manifest(jobs):
    m = {
        "schema_version": "nato-sers-p08-attempt-manifest-v1",
        "execution_authorized": False,
        "proposal_sha256": "a" * 64,
        "jobs": jobs,
        "manifest_sha256": "",
    }
    return reseal_manifest(m)


def build(n_fits):
    return manifest(jobs_for(n_fits))


def reseal_manifest(m):
    m["manifest_sha256"] = digest(m, "manifest_sha256")
    return m


def spec(sid, etype, elapsed, artifact, jid=None, status=None, receipt=None):
    return {
        "session_id": sid,
        "event_type": etype,
        "elapsed_ns": elapsed,
        "artifact_bytes": artifact,
        "job_id": jid,
        "status": status,
        "receipt_sha256": receipt,
    }


def sopen(sid=1, elapsed=0, artifact=0):
    return spec(sid, "session_open", elapsed, artifact)


def progress(sid, elapsed, artifact):
    return spec(sid, "progress", elapsed, artifact)


def start(sid, jid, elapsed, artifact):
    return spec(sid, "attempt_start", elapsed, artifact, jid=jid)


def finish(sid, jid, status, receipt, elapsed, artifact):
    return spec(sid, "attempt_finish", elapsed, artifact, jid=jid, status=status, receipt=receipt)


def close(sid, elapsed, artifact):
    return spec(sid, "session_close", elapsed, artifact)


def link(m, specs):
    events, prev = [], m["manifest_sha256"]
    for seq, sp in enumerate(specs, 1):
        ev = {"seq": seq, "previous_sha256": prev, **sp, "event_sha256": ""}
        ev["event_sha256"] = digest(ev, "event_sha256")
        prev = ev["event_sha256"]
        events.append(ev)
    return events


def reseal(m, events):
    prev = m["manifest_sha256"]
    for ev in events:
        ev["previous_sha256"] = prev
        ev["event_sha256"] = digest(ev, "event_sha256")
        prev = ev["event_sha256"]
    return events


def run(m, events):
    head = events[-1]["event_sha256"] if events else m["manifest_sha256"]
    return replay_attempt_journal(
        m,
        events,
        expected_manifest_sha256=m["manifest_sha256"],
        expected_head_sha256=head,
    )


def fails(m, events):
    with pytest.raises(JournalError) as ei:
        run(m, events)
    err = ei.value
    assert err.reason_code in REASONS
    text = str(err)
    if isinstance(m, dict):
        for value in (m.get("proposal_sha256"), m.get("manifest_sha256")):
            if isinstance(value, str) and len(value) >= 8:
                assert value not in text
        for j in m.get("jobs", []):
            if isinstance(j, dict) and isinstance(j.get("job_id"), str):
                if len(j["job_id"]) >= 4:
                    assert j["job_id"] not in text
    return err


def neg(specs, n_fits=2):
    m = build(n_fits)
    return m, link(m, specs)


def seq_success():
    return [
        sopen(1),
        start(1, "fit001", 10, 0),
        finish(1, "fit001", "succeeded", R1, 20, 100),
        start(1, "fit002", 30, 100),
        finish(1, "fit002", "succeeded", R2, 40, 250),
        start(1, "pred001", 50, 250),
        finish(1, "pred001", "succeeded", R3, 60, 300),
        start(1, "pred002", 70, 300),
        finish(1, "pred002", "succeeded", R4, 80, 400),
        close(1, 90, 400),
    ]


def success2():
    m = build(2)
    return m, link(m, seq_success())


def test_manifest_keys_and_pairing():
    m = build(2)
    assert set(m) == MANIFEST_KEYS
    ids = [j["job_id"] for j in m["jobs"]]
    assert ids == sorted(ids) and len(set(ids)) == 4
    assert all(set(j) == JOB_KEYS for j in m["jobs"])
    assert {j["worker"] for j in m["jobs"]} == {"cpu", "gpu"}
    fits = [j for j in m["jobs"] if j["stage"] == "source_fit"]
    preds = [j for j in m["jobs"] if j["stage"] == "source_validation_prediction"]
    assert all(j["dependencies"] == [] for j in fits)
    assert {tuple(j["dependencies"]) for j in preds} == {("fit001",), ("fit002",)}
    for p in preds:
        paired = next(f for f in fits if f["job_id"] == p["dependencies"][0])
        assert paired["worker"] == p["worker"]
    assert m["manifest_sha256"] == digest(m, "manifest_sha256")


def test_event_link_seq_and_previous_head():
    m = build(2)
    ev = link(m, [sopen(), progress(1, 5, 0), close(1, 5, 0)])
    assert list(range(1, 4)) == [e["seq"] for e in ev]
    assert set(ev[0]) == EVENT_KEYS
    assert ev[0]["previous_sha256"] == m["manifest_sha256"]
    for a, b in zip(ev, ev[1:], strict=False):
        assert b["previous_sha256"] == a["event_sha256"]
    for e in ev:
        assert e["event_sha256"] == digest(e, "event_sha256")
    assert run(m, ev)["head_sha256"] == ev[-1]["event_sha256"]


def test_hash_excludes_self_key():
    m = build(2)
    ev = link(m, [sopen()])[0]
    assert m["manifest_sha256"] == canonical_sha256(
        {k: v for k, v in m.items() if k != "manifest_sha256"}
    )
    assert ev["event_sha256"] == canonical_sha256(
        {k: v for k, v in ev.items() if k != "event_sha256"}
    )
    s = run(m, link(m, [sopen(), close(1, 1, 0)]))
    assert s["summary_sha256"] == canonical_sha256(
        {k: v for k, v in s.items() if k != "summary_sha256"}
    )


def test_success_summary_exact():
    m, ev = success2()
    s = run(m, ev)
    assert set(s) == SUMMARY_KEYS
    assert s["schema_version"] == "nato-sers-p08-attempt-summary-v1"
    assert s["execution_authorized"] is False
    assert s["manifest_sha256"] == m["manifest_sha256"]
    assert s["head_sha256"] == ev[-1]["event_sha256"]
    assert s["session_count"] == 1 and s["active_session_id"] is None
    assert s["closed_session_wall_ns"] == 90 and s["active_session_elapsed_ns"] == 0
    assert s["active_wall_ns"] == 90 and s["new_artifact_bytes"] == 400
    assert s["model_fit_attempts"] == 2 and s["source_prediction_attempts"] == 2
    assert s["active_cpu_workers"] == 0 and s["active_gpu_workers"] == 0
    assert s["in_flight_job_ids"] == []
    assert s["succeeded_job_ids"] == ["fit001", "fit002", "pred001", "pred002"]
    assert s["failed_job_ids"] == [] and s["interrupted_job_ids"] == []
    assert s["restart_allowed"] is True and s["journal_state"] == "closed"
    assert list(s["attempts"]) == ["fit001", "fit002", "pred001", "pred002"]
    for rec in s["attempts"].values():
        assert set(rec) == ATTEMPT_KEYS
        assert rec["status"] == "succeeded" and rec["session_id"] == 1
    assert s["attempts"]["fit001"]["worker"] == "cpu"
    assert s["attempts"]["fit002"]["worker"] == "gpu"
    assert s["attempts"]["pred001"]["stage"] == "source_validation_prediction"
    assert "expected_manifest_sha256" not in s
    assert "expected_head_sha256" not in s


def test_full_78_bound_and_counts():
    m = build(78)
    assert len(m["jobs"]) == 156
    assert run(m, [])["journal_state"] == "not_started"
    for n_fits, n_preds in [(79, 79), (0, 0), (2, 1), (2, 3), (1, 0)]:
        fails(manifest(jobs_for(n_fits, n_preds)), [])


def test_empty_journal():
    m = build(2)
    s = run(m, [])
    assert s["journal_state"] == "not_started"
    assert s["session_count"] == 0 and s["active_session_id"] is None
    assert s["active_session_elapsed_ns"] == 0 and s["active_wall_ns"] == 0
    assert s["closed_session_wall_ns"] == 0 and s["new_artifact_bytes"] == 0
    assert s["restart_allowed"] is True and s["attempts"] == {}
    assert s["manifest_sha256"] == s["head_sha256"] == m["manifest_sha256"]


def test_open_inflight_consumed_and_blocks_restart():
    m = build(2)
    s = run(m, link(m, [sopen(), start(1, "fit001", 7, 50)]))
    assert s["journal_state"] == "open" and s["active_session_id"] == 1
    assert s["session_count"] == 1 and s["in_flight_job_ids"] == ["fit001"]
    assert s["model_fit_attempts"] == 1
    assert s["active_cpu_workers"] == 1 and s["active_gpu_workers"] == 0
    assert s["active_session_elapsed_ns"] == 7 and s["active_wall_ns"] == 7
    assert s["new_artifact_bytes"] == 50
    assert s["restart_allowed"] is False
    rec = s["attempts"]["fit001"]
    assert set(rec) == ATTEMPT_KEYS
    assert rec["status"] == "running" and rec["receipt_sha256"] is None


def test_open_without_busy_jobs_still_blocks_restart():
    m = build(2)
    s = run(m, link(m, [sopen(), progress(1, 30, 10)]))
    assert s["journal_state"] == "open" and s["in_flight_job_ids"] == []
    assert s["active_wall_ns"] == 30 and s["restart_allowed"] is False


def test_no_elapsed_double_count_across_sessions():
    m = build(2)
    ev = link(
        m,
        [
            sopen(1),
            start(1, "fit001", 10, 0),
            finish(1, "fit001", "succeeded", R1, 20, 80),
            close(1, 100, 80),
            sopen(2, artifact=80),
            progress(2, 10, 80),
            start(2, "pred001", 20, 80),
            progress(2, 30, 80),
        ],
    )
    s = run(m, ev)
    assert s["session_count"] == 2 and s["active_session_id"] == 2
    assert s["closed_session_wall_ns"] == 100
    assert s["active_session_elapsed_ns"] == 30 and s["active_wall_ns"] == 130
    assert s["new_artifact_bytes"] == 80
    assert s["model_fit_attempts"] == 1 and s["source_prediction_attempts"] == 1
    assert s["restart_allowed"] is False
    assert s["attempts"]["fit001"]["status"] == "succeeded"
    assert s["attempts"]["fit001"]["receipt_sha256"] == R1


def test_counters_charge_running_failed_interrupted():
    m = build(3)
    ev = link(
        m,
        [
            sopen(1),
            start(1, "fit001", 5, 0),
            start(1, "fit002", 6, 0),
            finish(1, "fit002", "failed", R1, 7, 0),
            start(1, "fit003", 8, 0),
            finish(1, "fit003", "interrupted", R2, 9, 0),
        ],
    )
    s = run(m, ev)
    assert s["model_fit_attempts"] == 3
    assert s["in_flight_job_ids"] == ["fit001"] and s["active_cpu_workers"] == 1
    assert s["failed_job_ids"] == ["fit002"]
    assert s["interrupted_job_ids"] == ["fit003"]
    assert s["succeeded_job_ids"] == []
    assert s["attempts"]["fit001"]["status"] == "running"
    assert s["attempts"]["fit001"]["receipt_sha256"] is None
    assert s["attempts"]["fit003"]["receipt_sha256"] == R2


def test_replay_deterministic_and_non_mutating():
    m, ev = success2()
    m0, ev0 = copy.deepcopy(m), copy.deepcopy(ev)
    s1 = run(m, ev)
    s2 = run(m, ev)
    assert s1 == s2 and s1["summary_sha256"] == s2["summary_sha256"]
    assert m == m0 and ev == ev0


def test_expected_hashes_keyword_only_required():
    m = build(2)
    with pytest.raises(TypeError):
        replay_attempt_journal(m, [])
    with pytest.raises(TypeError):
        replay_attempt_journal(m, [], m["manifest_sha256"], m["manifest_sha256"])


def test_negative_missing_open_first_event():
    for first in [
        progress(1, 1, 0),
        start(1, "fit001", 1, 0),
        finish(1, "fit001", "succeeded", R1, 1, 0),
        close(1, 1, 0),
    ]:
        m, ev = neg([first])
        fails(m, ev)


def test_negative_double_open_and_bad_session_number():
    for specs in ([sopen(1), sopen(1)], [sopen(1), sopen(2)], [sopen(2)]):
        m, ev = neg(specs)
        fails(m, ev)


def test_negative_skip_seq():
    m = build(2)
    ev = link(m, [sopen(), progress(1, 5, 0), close(1, 5, 0)])
    ev[1]["seq"] = 7
    reseal(m, ev)
    fails(m, ev)


def test_negative_unmatched_and_wrong_session_finish():
    m, ev = neg([sopen(), finish(1, "fit001", "succeeded", R1, 5, 0)])
    fails(m, ev)
    m, ev = neg([sopen(), start(1, "fit001", 5, 0), finish(2, "fit001", "succeeded", R1, 6, 0)])
    fails(m, ev)


def test_negative_close_with_pending_and_double_close():
    m, ev = neg([sopen(), start(1, "fit001", 5, 0), close(1, 6, 0)])
    fails(m, ev)
    m, ev = neg([sopen(), close(1, 5, 0), close(1, 6, 0)])
    fails(m, ev)


def test_negative_duplicate_and_retry_attempts():
    m, ev = neg([sopen(), start(1, "fit001", 5, 0), start(1, "fit001", 6, 0)])
    fails(m, ev)
    for status in ("failed", "interrupted", "succeeded"):
        m, ev = neg(
            [
                sopen(),
                start(1, "fit001", 5, 0),
                finish(1, "fit001", status, R1, 6, 0),
                start(1, "fit001", 7, 0),
            ]
        )
        fails(m, ev)


def test_negative_prediction_dependencies():
    m, ev = neg([sopen(), start(1, "pred001", 5, 0)])
    fails(m, ev)
    for status in ("failed", "interrupted"):
        m, ev = neg(
            [
                sopen(),
                start(1, "fit001", 5, 0),
                finish(1, "fit001", status, R1, 6, 0),
                start(1, "pred001", 7, 0),
            ]
        )
        fails(m, ev)


def test_negative_unknown_job_and_backward_counters():
    m, ev = neg([sopen(), start(1, "nope", 5, 0)])
    fails(m, ev)
    m, ev = neg([sopen(), progress(1, 10, 0), progress(1, 5, 0)])
    fails(m, ev)
    m, ev = neg([sopen(), progress(1, 10, 50), progress(1, 20, 10)])
    fails(m, ev)
    m, ev = neg([sopen(1, 5, 0)])
    fails(m, ev)


def test_digest_tamper_rejected():
    m, ev = success2()
    bad = copy.deepcopy(ev)
    bad[1]["elapsed_ns"] += 1
    fails(m, bad)
    bad = copy.deepcopy(ev)
    bad[1]["previous_sha256"] = ZERO
    fails(m, bad)
    bad = copy.deepcopy(ev)
    bad[0]["event_type"] = "progress"
    fails(m, bad)
    with pytest.raises(JournalError) as ei:
        replay_attempt_journal(
            m, ev, expected_manifest_sha256=ZERO, expected_head_sha256=ev[-1]["event_sha256"]
        )
    assert ei.value.reason_code in REASONS
    with pytest.raises(JournalError) as ei:
        replay_attempt_journal(
            m, ev, expected_manifest_sha256=m["manifest_sha256"], expected_head_sha256=ZERO
        )
    assert ei.value.reason_code in REASONS


def test_manifest_shape_and_flag_rejected():
    m = build(2)
    m2 = dict(m)
    m2["extra"] = 1
    fails(reseal_manifest(m2), [])
    m2 = {k: v for k, v in build(2).items() if k != "proposal_sha256"}
    fails(reseal_manifest(m2), [])
    for flag in (True, 1, None, "x"):
        m2 = dict(build(2))
        m2["execution_authorized"] = flag
        fails(reseal_manifest(m2), [])


def test_event_shape_extra_missing_rejected():
    m = build(2)
    for mode in ("extra", "missing"):
        ev = link(m, [sopen(), progress(1, 5, 0), close(1, 5, 0)])
        if mode == "extra":
            ev[1]["extra"] = 1
        else:
            del ev[1]["job_id"]
        fails(m, reseal(m, ev))


@pytest.mark.parametrize(
    "index,key,value",
    [
        (1, "elapsed_ns", True),
        (1, "elapsed_ns", 1.5),
        (1, "elapsed_ns", -1),
        (1, "artifact_bytes", True),
        (1, "artifact_bytes", -1),
        (0, "session_id", True),
        (0, "session_id", 1.5),
        (0, "session_id", 0),
        (0, "session_id", -1),
    ],
)
def test_event_numeric_types_rejected(index, key, value):
    m = build(2)
    ev = link(m, [sopen(), progress(1, 5, 0), close(1, 5, 0)])
    ev[index][key] = value
    fails(m, reseal(m, ev))


def test_invalid_receipt_and_declared_hashes_rejected():
    m = build(2)
    fails(
        m,
        link(
            m, [sopen(), start(1, "fit001", 5, 0), finish(1, "fit001", "succeeded", "short", 6, 0)]
        ),
    )
    m2 = dict(m)
    m2["manifest_sha256"] = "XYZ"
    fails(m2, [])
    m3 = build(2)
    m3["proposal_sha256"] = "A" * 64
    fails(reseal_manifest(m3), [])
    ev = link(m, [sopen(), close(1, 5, 0)])
    ev2 = copy.deepcopy(ev)
    ev2[-1]["event_sha256"] = "zz"
    with pytest.raises(JournalError):
        replay_attempt_journal(
            m, ev2, expected_manifest_sha256=m["manifest_sha256"], expected_head_sha256="zz"
        )
    with pytest.raises(JournalError):
        replay_attempt_journal(
            m, [], expected_manifest_sha256="nope", expected_head_sha256=m["manifest_sha256"]
        )
    with pytest.raises(JournalError):
        replay_attempt_journal(
            m, [], expected_manifest_sha256=m["manifest_sha256"], expected_head_sha256="nope"
        )


def test_unknown_stage_worker_status_event_type_rejected():
    for field, value in (("stage", "other"), ("worker", "tpu")):
        m = build(2)
        m["jobs"][0] = dict(m["jobs"][0])
        m["jobs"][0][field] = value
        fails(reseal_manifest(m), [])
    m = build(2)
    fails(m, link(m, [sopen(), start(1, "fit001", 5, 0), finish(1, "fit001", "ok", R1, 6, 0)]))
    fails(m, link(m, [sopen(), spec(1, "heartbeat", 5, 0), close(1, 5, 0)]))


def test_manifest_dependency_and_identity_negatives():
    base = jobs_for(2)
    cases = []
    j = copy.deepcopy(base)
    j[0]["dependencies"] = ["fit002"]
    cases.append(j)
    j = copy.deepcopy(base)
    j[1]["job_id"] = j[0]["job_id"]
    cases.append(j)
    j = copy.deepcopy(base)
    j[0], j[1] = j[1], j[0]
    cases.append(j)
    cases.append(jobs_for(2, 1))
    cases.append(jobs_for(1, 2))
    j = copy.deepcopy(base)
    j[2]["worker"] = "gpu" if j[2]["worker"] == "cpu" else "cpu"
    cases.append(j)
    j = copy.deepcopy(base)
    j[2]["dependencies"] = ["fit999"]
    cases.append(j)
    j = copy.deepcopy(base)
    j[2]["dependencies"] = ["fit001", "fit002"]
    cases.append(j)
    j = copy.deepcopy(base)
    j[2]["dependencies"] = ["pred005"]
    cases.append(j)
    j = copy.deepcopy(base)
    j[0]["job_id"] = ""
    cases.append(j)
    j = copy.deepcopy(base)
    j[2]["dependencies"] = ["fit001", "fit001"]
    cases.append(j)
    for js in cases:
        fails(manifest(js), [])


def test_artifact_highwater_global_monotonic():
    m = build(2)
    ev = link(
        m,
        [
            sopen(1),
            progress(1, 10, 100),
            close(1, 10, 100),
            sopen(2, artifact=100),
            progress(2, 5, 100),
        ],
    )
    assert run(m, ev)["new_artifact_bytes"] == 100
    m2, ev2 = neg([sopen(1), progress(1, 10, 100), close(1, 10, 100), sopen(2, 0, 80)])
    fails(m2, ev2)


def test_long_sequence_counters_persist_across_pauses():
    m = build(3)
    ev = link(
        m,
        [
            sopen(1),
            start(1, "fit001", 10, 0),
            finish(1, "fit001", "succeeded", R1, 20, 100),
            close(1, 20, 100),
            sopen(2, artifact=100),
            start(2, "fit002", 10, 100),
            finish(2, "fit002", "failed", R2, 15, 150),
            close(2, 15, 150),
            sopen(3, artifact=150),
            progress(3, 5, 150),
            start(3, "fit003", 6, 150),
        ],
    )
    s = run(m, ev)
    assert s["session_count"] == 3
    assert s["closed_session_wall_ns"] == 35
    assert s["active_session_elapsed_ns"] == 6 and s["active_wall_ns"] == 41
    assert s["new_artifact_bytes"] == 150
    assert s["model_fit_attempts"] == 3
    assert s["succeeded_job_ids"] == ["fit001"]
    assert s["failed_job_ids"] == ["fit002"]
    assert s["in_flight_job_ids"] == ["fit003"]
    assert s["restart_allowed"] is False


def test_failed_attempt_cannot_be_retried_later():
    m = build(3)
    ev = link(
        m,
        [
            sopen(1),
            start(1, "fit002", 5, 0),
            finish(1, "fit002", "failed", R1, 6, 0),
            close(1, 6, 0),
            sopen(2),
            start(2, "fit002", 5, 0),
        ],
    )
    fails(m, ev)


def test_require_scientific_execution_always_denies():
    m = build(2)
    ev = link(m, seq_success())
    run(m, ev)
    for kwargs in (
        {},
        {"execution_authorized": True},
        {"flags": [True, True], "authority": "root"},
    ):
        with pytest.raises(JournalError) as ei:
            require_scientific_execution(m, ev, **kwargs)
        assert ei.value.reason_code == "scientific_execution_not_authorized"


class _Poison:
    def __hash__(self):
        raise AssertionError("hash invoked")

    def __repr__(self):
        raise AssertionError("repr invoked")

    def __eq__(self, other):
        raise AssertionError("eq invoked")


def test_unknown_reason_and_custom_objects_static():
    assert JournalError("not_a_real_code").reason_code == "invalid_journal_input"
    assert JournalError("invalid_event").reason_code == "invalid_event"
    assert JournalError(["bad"]).reason_code == "invalid_journal_input"
    for bad_manifest in (_Poison(), {"x": _Poison()}, ["bad"]):
        with pytest.raises(JournalError) as ei:
            replay_attempt_journal(
                bad_manifest,
                [],
                expected_manifest_sha256="b" * 64,
                expected_head_sha256="b" * 64,
            )
        assert ei.value.reason_code in REASONS
    m = build(2)
    with pytest.raises(JournalError) as ei:
        replay_attempt_journal(
            m,
            [_Poison()],
            expected_manifest_sha256=m["manifest_sha256"],
            expected_head_sha256=m["manifest_sha256"],
        )
    assert ei.value.reason_code in REASONS
    with pytest.raises(JournalError):
        replay_attempt_journal(
            m,
            _Poison(),
            expected_manifest_sha256=m["manifest_sha256"],
            expected_head_sha256=m["manifest_sha256"],
        )


def test_replay_result_isolated_from_events_and_reruns_identically():
    m, events = success2()
    m_before = copy.deepcopy(m)
    events_before = copy.deepcopy(events)
    baseline = run(m, events)
    baseline_summary = copy.deepcopy(baseline)

    baseline["attempts"]["fit001"]["status"] = "tampered"
    baseline["succeeded_job_ids"].append("tampered-job")

    assert m == m_before
    assert events == events_before
    fresh = run(m, events)
    assert fresh == baseline_summary


def test_public_replay_wraps_private_value_error(monkeypatch):
    from atlas_sers.evaluation import p08_attempt_journal as journal

    def _boom(*args, **kwargs):
        raise ValueError("PRIVATE_SENTINEL")

    monkeypatch.setattr(journal, "_replay_attempt_journal", _boom)
    m, events = success2()

    with pytest.raises(JournalError) as excinfo:
        replay_attempt_journal(
            m,
            events,
            expected_manifest_sha256=m["manifest_sha256"],
            expected_head_sha256=events[-1]["event_sha256"],
        )

    error = excinfo.value
    assert error.reason_code == "invalid_journal_input"
    assert error.__suppress_context__ is True
    assert "PRIVATE_SENTINEL" not in str(error)


def test_public_replay_propagates_keyboard_interrupt(monkeypatch):
    from atlas_sers.evaluation import p08_attempt_journal as journal

    def _interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(journal, "_replay_attempt_journal", _interrupt)
    m, events = success2()

    with pytest.raises(KeyboardInterrupt):
        replay_attempt_journal(
            m,
            events,
            expected_manifest_sha256=m["manifest_sha256"],
            expected_head_sha256=events[-1]["event_sha256"],
        )


def test_journal_error_normalizes_hostile_and_subclassed_reasons():
    error = JournalError(_Poison())
    assert error.reason_code == "invalid_journal_input"
    assert type(error.reason_code) is str

    class _Reason(str):
        pass

    subclassed = JournalError(_Reason("invalid_event"))
    assert subclassed.reason_code == "invalid_journal_input"
    assert type(subclassed.reason_code) is str


def test_concurrent_jobs_share_one_open_session():
    m = build(2)
    partial_specs = [
        sopen(1),
        start(1, "fit001", 1, 0),
        start(1, "fit002", 2, 0),
    ]

    partial = run(m, link(m, partial_specs))
    summary = partial
    assert summary["active_cpu_workers"] == 1
    assert summary["active_gpu_workers"] == 1
    assert summary["model_fit_attempts"] == 2
    assert summary["restart_allowed"] is False

    full_specs = partial_specs + [
        finish(1, "fit001", "succeeded", R1, 3, 0),
        start(1, "pred001", 4, 0),
    ]

    full = run(m, link(m, full_specs))
    summary = full
    assert summary["active_cpu_workers"] == 1
    assert summary["active_gpu_workers"] == 1
    assert summary["model_fit_attempts"] == 2
    assert summary["source_prediction_attempts"] == 1
    assert sorted(summary["in_flight_job_ids"]) == ["fit002", "pred001"]


@pytest.mark.parametrize(
    ("index", "key", "value"),
    [
        (0, "seq", True),
        (0, "seq", 1.0),
        (1, "artifact_bytes", 0.0),
        (1, "artifact_bytes", "0"),
        (1, "elapsed_ns", "5"),
    ],
)
def test_replay_rejects_non_strict_numeric_event_fields(index, key, value):
    m = build(2)
    events = link(m, [sopen(), progress(1, 5, 0), close(1, 5, 0)])
    events[index][key] = value
    reseal(m, events)
    fails(m, events)


def test_execution_authorized_zero_is_rejected_despite_bool_equality():
    m = build(2)
    m["execution_authorized"] = 0
    reseal_manifest(m)
    fails(m, [])
