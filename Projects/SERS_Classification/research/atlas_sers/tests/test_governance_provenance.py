"""Permanent regressions for order-independent BLAS provenance capture."""

from __future__ import annotations

import copy
import itertools
import subprocess
import sys
import types
from collections import Counter

import pytest

from atlas_sers.governance import provenance
from atlas_sers.governance.artifacts import ArtifactStore
from atlas_sers.governance.canonical import canonical_json_bytes

_BLAS_RECORDS = [
    {
        "internal_api": "openblas",
        "version": "0.3.27",
        "num_threads": 8,
        "prefix": "libopenblas",
        "threading_layer": "pthreads",
        "architecture": "x86_64",
        "user_api": "blas",
        "filepath": "/private/libopenblas.so",
    },
    {
        "internal_api": "blis",
        "version": "1.0.0",
        "num_threads": 8,
        "prefix": "libblis",
        "threading_layer": "pthreads",
        "architecture": "x86_64",
        "user_api": "blas",
        "filepath": "/private/libblis.so",
    },
    {
        "internal_api": "openmp",
        "version": "5.0",
        "num_threads": 8,
        "prefix": "libgomp",
        "threading_layer": "openmp",
        "architecture": "x86_64",
        "user_api": "blas",
        "filepath": "/private/libgomp.so",
    },
]


def _install_inventory(monkeypatch: pytest.MonkeyPatch, records) -> None:
    module = types.ModuleType("threadpoolctl")
    module.threadpool_info = lambda: records
    monkeypatch.setitem(sys.modules, "threadpoolctl", module)


def _inventory() -> list:
    return provenance._blas_inventory()


def test_inventory_order_independent_for_all_permutations(monkeypatch):
    results = []
    for permutation in itertools.permutations(_BLAS_RECORDS):
        _install_inventory(monkeypatch, permutation)
        results.append(_inventory())
    assert all(result == results[0] for result in results)
    assert len({canonical_json_bytes(result) for result in results}) == 1


def test_inventory_keeps_duplicates_without_mutation(monkeypatch):
    records = [_BLAS_RECORDS[0], copy.deepcopy(_BLAS_RECORDS[0]), _BLAS_RECORDS[1]]
    snapshot = copy.deepcopy(records)
    _install_inventory(monkeypatch, records)
    result = _inventory()
    assert len(result) == 3
    assert records == snapshot
    sanitized = {
        key: _BLAS_RECORDS[0][key]
        for key in sorted(_BLAS_RECORDS[0])
        if key != "filepath"
    }
    counts = Counter(canonical_json_bytes(record) for record in result)
    assert counts[canonical_json_bytes(sanitized)] == 2
    assert len(counts) == 2


def test_inventory_excludes_private_filepath(monkeypatch):
    _install_inventory(monkeypatch, _BLAS_RECORDS)
    result = _inventory()
    assert all("filepath" not in record for record in result)
    assert b"/private/" not in canonical_json_bytes(result)


def test_inventory_key_and_optional_fields_are_stable(monkeypatch):
    versioned_a = {"internal_api": "openblas", "num_threads": 8, "version": "1"}
    versioned_b = {"version": "1", "num_threads": 8, "internal_api": "openblas"}
    nullable = {"internal_api": "openblas", "num_threads": 8, "version": None}
    absent = {"num_threads": 8, "internal_api": "openblas"}

    expected = {
        canonical_json_bytes(versioned_a),
        canonical_json_bytes(nullable),
        canonical_json_bytes(absent),
    }
    assert canonical_json_bytes(versioned_a) == canonical_json_bytes(versioned_b)
    assert canonical_json_bytes(nullable) != canonical_json_bytes(absent)

    records = [versioned_a, versioned_b, nullable, absent]
    for permutation in itertools.permutations(records):
        _install_inventory(monkeypatch, list(permutation))
        result = _inventory()
        assert len(result) == 4
        assert {canonical_json_bytes(record) for record in result} == expected
        counts = Counter(canonical_json_bytes(record) for record in result)
        assert counts[canonical_json_bytes(versioned_a)] == 2


@pytest.fixture
def governance_roots(tmp_path):
    repository = tmp_path / "repository"
    project = repository / "research" / "atlas_sers"
    project.mkdir(parents=True)
    (project / "marker.txt").write_text("fixture\n")
    derived = tmp_path / "derived"
    derived.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    return repository, project, derived, artifacts


def _patch_environment(monkeypatch: pytest.MonkeyPatch, records) -> None:
    _install_inventory(monkeypatch, records)
    monkeypatch.setattr(provenance, "_gpu_inventory", lambda root: [])
    monkeypatch.setattr(provenance, "_cuda_compiler", lambda root: None)
    monkeypatch.setattr(provenance, "_cpu_model", lambda: "fixture-cpu")
    monkeypatch.setattr(provenance, "_dependency_lock", lambda: {"numpy": "2.1.0"})
    monkeypatch.setattr(provenance, "_command", lambda args, *, cwd: "0" * 40)
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, b"", b""),
    )


def _protected_hash(repository, project, artifacts) -> str:
    captured = provenance.capture_provenance(
        repository_root=repository, project_root=project, artifact_root=artifacts
    )
    return captured["protected_environment_sha256"]


def _variant(index: int, **updates):
    records = copy.deepcopy(_BLAS_RECORDS)
    records[index].update(updates)
    return records


def test_protected_hash_ignores_blas_record_order(monkeypatch, governance_roots):
    repository, project, _derived, artifacts = governance_roots
    hashes = set()
    for permutation in itertools.permutations(_BLAS_RECORDS):
        _patch_environment(monkeypatch, list(permutation))
        hashes.add(_protected_hash(repository, project, artifacts))
    assert len(hashes) == 1


def test_protected_hash_tracks_true_blas_changes(monkeypatch, governance_roots):
    repository, project, _derived, artifacts = governance_roots
    _patch_environment(monkeypatch, _BLAS_RECORDS)
    baseline = _protected_hash(repository, project, artifacts)

    variants = [
        _variant(0, version="9.9.9"),
        _variant(0, num_threads=1),
        _variant(0, architecture="aarch64"),
        _variant(0, internal_api="mkl"),
        _variant(0, user_api="mkl"),
        _variant(0, threading_layer="sequential"),
        [*_BLAS_RECORDS, copy.deepcopy(_BLAS_RECORDS[0])],
        _BLAS_RECORDS[:2],
        [*_BLAS_RECORDS, {"internal_api": "mkl", "version": "2024.0"}],
    ]
    for records in variants:
        _patch_environment(monkeypatch, records)
        assert _protected_hash(repository, project, artifacts) != baseline


def test_artifact_store_verifies_skip_for_permuted_blas(monkeypatch, governance_roots):
    repository, project, derived, artifacts = governance_roots
    _patch_environment(monkeypatch, _BLAS_RECORDS)
    store = ArtifactStore(
        artifact_root=artifacts, input_root=derived, project_root=repository
    )
    run_id = "P00-" + "a" * 24
    lease = store.begin(
        run_id=run_id,
        protected_state_sha256=_protected_hash(repository, project, artifacts),
    )
    assert lease.action == "new"
    assert lease.work_dir is not None
    (lease.work_dir / "result.json").write_text('{"status":"pass"}\n')
    store.commit(lease, scientific_status="pass")

    _patch_environment(monkeypatch, list(reversed(_BLAS_RECORDS)))
    repeated = store.begin(
        run_id=run_id,
        protected_state_sha256=_protected_hash(repository, project, artifacts),
    )
    assert repeated.action == "verified_skip"


def test_artifact_store_quarantines_true_blas_change(monkeypatch, governance_roots):
    repository, project, derived, artifacts = governance_roots
    _patch_environment(monkeypatch, _BLAS_RECORDS)
    store = ArtifactStore(
        artifact_root=artifacts, input_root=derived, project_root=repository
    )
    run_id = "P00-" + "b" * 24
    lease = store.begin(
        run_id=run_id,
        protected_state_sha256=_protected_hash(repository, project, artifacts),
    )
    assert lease.work_dir is not None
    (lease.work_dir / "result.json").write_text('{"status":"pass"}\n')
    store.commit(lease, scientific_status="pass")

    _patch_environment(monkeypatch, _variant(0, num_threads=1))
    changed = _protected_hash(repository, project, artifacts)
    again = store.begin(run_id=run_id, protected_state_sha256=changed)
    assert again.action == "new"
    assert list(store.quarantine.glob(f"{run_id}--*"))
