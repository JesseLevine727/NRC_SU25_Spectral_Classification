"""Focused tests for p05_outer_inputs.prepare_outer_inputs (outer-test arrays)."""

from __future__ import annotations

import types

import numpy as np
import pytest

from atlas_sers.evaluation import p05_core_run as core
from atlas_sers.evaluation import p05_outer_inputs as outer
from atlas_sers.evaluation import p05_refit_io as refit_io

STATION, HELD, SRC_INSTR = "cwa", "H", "S"
ROWS, FEATURES, CTX = 5, 1401, "ctx1"
LABELS = ["tC", "fA", "tA", "fB", "fC"]


def _context():
    return {
        "context_id": CTX,
        "experiment_id": "e1",
        "phase_gate": "development",
        "outer_repeat": "0",
        "outer_fold": "0",
        "station": STATION,
        "domain": "d",
        "held_instrument": HELD,
        "partition_id": "p1",
    }


def _make_support(fit, test, *, stations=None):
    stations = stations or {}
    roles, manifest = [], []
    for role_name, role_id, rows in (
        (outer.OUTER_FIT_ROLE, "fit", fit),
        (outer.OUTER_TEST_ROLE, "test", test),
    ):
        for uid, master, label, instr in rows:
            roles.append(
                {
                    "context_id": CTX,
                    "role_id": role_id,
                    "role": role_name,
                    "observation_uid": uid,
                    "master_sample_id": master,
                    "instrument": instr,
                    "target_analyte": label,
                }
            )
            manifest.append(
                {
                    "observation_uid": uid,
                    "master_sample_id": master,
                    "target_analyte": label,
                    "instrument": instr,
                    "station": stations.get(uid, STATION),
                    "sensor_family": "synthetic-substrate",
                }
            )
    return types.SimpleNamespace(
        contexts=(_context(),), roles=tuple(roles), manifest=tuple(manifest)
    )


FIT = (("fA", "fA", "A", SRC_INSTR), ("fB", "fB", "B", SRC_INSTR), ("fC", "fC", "C", SRC_INSTR))
TEST = (("tA", "tA", "A", HELD), ("tC", "tC", "C", HELD))


def _endpoint(support, *, role_id=None, uids=None, drop=()):
    rows = [r for r in support.roles if r["role"] == outer.OUTER_TEST_ROLE]
    endpoint = {
        "context_id": CTX,
        "outer_test_role_id": role_id or refit_io._role_id_for(support, CTX, outer.OUTER_TEST_ROLE),
        "test_uids": uids or refit_io._sorted_uids(rows),
        "test_masters": sorted({r["master_sample_id"] for r in rows}),
        "test_classes": sorted({r["target_analyte"] for r in rows}),
    }
    for key in drop:
        endpoint.pop(key, None)
    return endpoint


def _bundle(support, p01):
    return {
        "support": support,
        "artifact_root": p01,
        "p01_path": p01,
        "contract": {
            "population": {"rows": ROWS, "features": FEATURES},
            "input_pins": {"representation_sha256": "0" * 64, "p04plan_run_id": "run1"},
        },
    }


@pytest.fixture
def loader(monkeypatch):
    calls = []

    def fake(path, sha, uids, rows):
        calls.append(1)
        intensity = np.empty((ROWS, FEATURES), dtype="float32")
        for index in range(ROWS):
            intensity[index, :] = np.float32(index + 1)
        return intensity, list(LABELS)

    monkeypatch.setattr(core, "_load_representation", fake)
    return calls


def test_success_orders_rows_by_sorted_test_uids(loader, tmp_path):
    support = _make_support(FIT, TEST)
    result = outer.prepare_outer_inputs(_bundle(support, tmp_path), _endpoint(support))
    assert loader == [1]
    assert result["observation_uids"] == ("tA", "tC")
    assert result["classes"] == ("A", "B", "C")
    assert result["values"].dtype == np.float32
    assert result["values"].shape == (2, FEATURES)
    assert np.all(result["values"][0] == np.float32(3))
    assert np.all(result["values"][1] == np.float32(1))


@pytest.mark.parametrize("labels", [("A", "A"), ("A", "C")])
def test_one_and_two_class_test_support(loader, tmp_path, labels):
    test = tuple((uid, uid, label, HELD) for uid, label in zip(("tA", "tC"), labels, strict=True))
    support = _make_support(FIT, test)
    result = outer.prepare_outer_inputs(_bundle(support, tmp_path), _endpoint(support))
    assert loader == [1] and result["values"].shape == (2, FEATURES)


FAILURES = [
    ("wrong_role", FIT, TEST, None, {"role_id": "wrong"}, "outer_test_role_mismatch"),
    ("missing_uids", FIT, TEST, None, {"drop": ("test_uids",)}, "outer_test_uids_mismatch"),
    (
        "missing_masters",
        FIT,
        TEST,
        None,
        {"drop": ("test_masters",)},
        "outer_test_masters_mismatch",
    ),
    (
        "missing_classes",
        FIT,
        TEST,
        None,
        {"drop": ("test_classes",)},
        "outer_test_classes_mismatch",
    ),
    ("uid_order", FIT, TEST, None, {"uids": ["tC", "tA"]}, "outer_test_uids_mismatch"),
    (
        "source_held",
        tuple((uid, master, label, HELD) for uid, master, label, _ in FIT),
        TEST,
        None,
        {},
        "held_instrument_in_source",
    ),
    (
        "wrong_test_instr",
        FIT,
        tuple((uid, master, label, "X") for uid, master, label, _ in TEST),
        None,
        {},
        "held_instrument_test_mismatch",
    ),
    (
        "master_overlap",
        FIT,
        (("tA", "fA", "A", HELD), ("tC", "tC", "C", HELD)),
        None,
        {},
        "outer_test_master_overlap",
    ),
    ("station", FIT, TEST, {"tA": "other"}, {}, "outer_station_mismatch"),
    ("source_classes", FIT[:2], TEST, None, {}, "source_class_count_mismatch"),
    (
        "label_outside",
        FIT,
        (("tA", "tA", "A", HELD), ("tC", "tC", "D", HELD)),
        None,
        {},
        "outer_test_class_unknown",
    ),
]


@pytest.mark.parametrize("name,fit,test,stations,ep_kwargs,code", FAILURES)
def test_rejections_before_loader(loader, tmp_path, name, fit, test, stations, ep_kwargs, code):
    support = _make_support(fit, test, stations=stations)
    with pytest.raises(outer.P05OuterInputsError) as exc:
        outer.prepare_outer_inputs(_bundle(support, tmp_path), _endpoint(support, **ep_kwargs))
    assert code in " ".join(str(a) for a in exc.value.args) + str(exc.value)
    assert loader == []
