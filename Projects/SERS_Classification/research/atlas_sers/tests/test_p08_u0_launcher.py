"""P08-T152 cohesive invented-data integration tests for the U0 outer launcher.

Test-only, invented-data integration tests.  They exercise the real launcher
entry point (``launch`` -> ``_run`` -> ``authenticated_runtime`` ->
``prepare_u0_runtime_inputs`` -> ``_SourceSession`` -> the actual numerical CPU
backend -> artifact persistence/readback -> close) inside fresh isolated
subprocesses (``sys.executable -I -B``) with no GPU and a single thread.

The parent fixture modules provide invented raw metadata/action buffers and
test-only pins.  The real public specification audit plus the eighteen
inherited public source files are reused unchanged; nothing here reads real
SERS metadata or arrays, and no test ever produces a scientific permit.

The public runtime catalog is always regenerated from the exact current
project source bytes inside a test-local package copy, independent of whether
the current ``src`` has been published.  The production scripts, constants and
files are never modified: the child patches only its in-memory copy of the
launcher constants and installs test-only pins into the already authenticated
project modules.

The lazy-import check below asserts only the authenticated catalog loader
identity for ``p03_metrics``; it does not require that module to be newly
imported on every run and is paired with the dedicated lazy-import tests.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests import test_p08_u0_inputs as metadata_fixture
from tests import test_p08_u0_runtime_inputs as runtime_fixture
from tests.p08_store_fixtures import resources

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "atlas_sers"
SCRIPTS = ROOT / "scripts"
CONTRACTS = ROOT / "plan" / "contracts"

BOOTSTRAP_NAME = "check_p08_u0_runtime.py"
CATALOG_NAME = "p08_runtime_source_catalog.json"
CONTRACT_FILES = (
    "hyperparameter_registry.json",
    "p03_governance_contract.json",
    "p04_execution_contract.json",
    "p05_core_contract.json",
)
TEST_REVISION = "a" * 40
GIB = 1024**3


# ---------------------------------------------------------------------------
# Child driver (written verbatim into a fresh process)
# ---------------------------------------------------------------------------


_DRIVER = '''#!/usr/bin/env python3
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import sys
import time
import types


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    config_path = sys.argv[1]
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    scenario = config["scenario"]
    summary = {
        "scenario": scenario,
        "runtime_entered": False,
        "setup_cuda_called": False,
        "sample_calls": 0,
        "loader_ok": None,
        "fit_calls": 0,
        "verify_calls": 0,
        "candidate_fit_calls": 0,
        "neural_fit_calls": 0,
        "error_type": None,
        "error_reason": None,
        "error_is_sentinel": False,
        "opens": 0,
        "mutation_events": 0,
        "process_events": 0,
        "network_events": 0,
        "imported_modules": [],
        "atlas_modules_loaded": [],
        "torch_imported": False,
        "cuda_observed": False,
        "fit_job_id": None,
        "snapshot_model_fit_attempts": None,
        "snapshot_source_prediction_attempts": None,
        "pair_report_completed_pair_count": None,
        "pair_report_incomplete": None,
        "close_report_json": None,
        "postclose_bytes": None,
        "initial_marker_elapsed_ns": None,
        "session_started_ns": None,
        "io_started_ns": None,
        "io_control_sha256": None,
        "io_control_bytes": None,
        "launch_present_before_fit": None,
        "launch_sha_before_fit": None,
        "injection_applied": False,
        "cuda_initialized": False,
        "reached_success": False,
        "cublas_workspace_config": None,
    }

    launcher = _load_module(config["launcher_path"], "p08_u0_launcher_under_test")

    output_root = config["output_root"]

    counters = {"open": 0, "mutation": 0, "process": 0, "network": 0}
    imported_after_hook = set()

    _MUTATION_EVENTS = frozenset(
        {
            "os.rename",
            "os.remove",
            "os.unlink",
            "os.rmdir",
            "os.mkdir",
            "os.makedirs",
            "os.link",
            "os.symlink",
            "os.chmod",
            "os.chown",
            "os.truncate",
            "os.utime",
            "os.replace",
            "shutil.copyfile",
            "shutil.copytree",
            "shutil.rmtree",
            "shutil.move",
        }
    )
    _PROCESS_EVENTS = frozenset(
        {
            "subprocess.Popen",
            "os.system",
            "os.exec",
            "os.fork",
            "os.spawn",
            "os.posix_spawn",
        }
    )
    _NETWORK_EVENTS = frozenset(
        {
            "socket.__new__",
            "socket.bind",
            "socket.connect",
            "socket.getaddrinfo",
        }
    )

    def _audit(event, args):
        if event in ("open", "os.open"):
            counters["open"] += 1
        elif event in _MUTATION_EVENTS:
            counters["mutation"] += 1
        elif event in _PROCESS_EVENTS:
            counters["process"] += 1
        elif event in _NETWORK_EVENTS:
            counters["network"] += 1
        elif event == "import" and args:
            imported_after_hook.add(args[0])

    os.fstatvfs = lambda descriptor: types.SimpleNamespace(
        f_bavail=64 * 1024 ** 3, f_frsize=1
    )

    if scenario == "default_denied":
        # Explicitly exercise the unset-deployment branch of the real pin.
        launcher.APPROVED_PERMIT_SHA256 = None
    else:
        launcher.BOOTSTRAP_SHA256 = config["bootstrap_sha256"]
        launcher.PROPOSAL_SHA256 = config["proposal_sha256"]
        launcher.MANIFEST_SHA256 = config["manifest_sha256"]
        launcher.APPROVED_PERMIT_SHA256 = config["permit_sha256"]

    sentinel = KeyboardInterrupt("p08-t156-sentinel")
    holder = {"runtime": None, "owner": None, "sentinel": sentinel}

    def _fake_setup_cuda(torch_module):
        # Test-only CUDA override: the real numerical CPU backend still runs.
        summary["setup_cuda_called"] = True
        return torch_module

    if scenario != "default_denied":
        launcher._setup_cuda = _fake_setup_cuda

    def _install(runtime):
        from atlas_sers.evaluation import p03_runtime
        from atlas_sers.evaluation import p05_development
        from atlas_sers.evaluation import p08_u0_inputs as inputs_module
        from atlas_sers.evaluation import p08_u0_arrays as arrays_module
        from atlas_sers.evaluation import p08_u0_runtime_inputs as runtime_inputs
        from atlas_sers.evaluation import p08_u0_admission as admission
        from atlas_sers.evaluation import p08_serial_resources as serial
        from atlas_sers.evaluation import p08_u0_session as session_module

        summary["runtime_entered"] = True
        holder["runtime"] = runtime
        holder["session_module"] = session_module

        inputs_module._BYTE_PINS = dict(config["metadata_pins"])
        arrays_module._ACTION_PINS = config["action_pins"]
        runtime_inputs._CANDIDATES_SHA256 = config["candidates_sha256"]
        admission.U0_PROPOSAL_SHA256 = config["proposal_sha256"]
        admission.U0_MANIFEST_SHA256 = config["manifest_sha256"]

        resources_snapshot = config["resource_record"]
        if scenario == "resource_limit":
            resources_snapshot = config["overlimit_record"]
        cuda_packet = config["cuda_packet"]

        def _fake_sample(output_directory, torch_module, model_threads):
            summary["sample_calls"] += 1
            started_ns = time.monotonic_ns()
            captured = dict(resources_snapshot)
            finished_ns = time.monotonic_ns()
            return started_ns, finished_ns, captured, cuda_packet

        serial._sample = _fake_sample

        positive = scenario == "interrupt_after_first"

        # The real numerical stage backend and the numerical entry points are
        # spied in every scenario.  Only the positive scenario may reach them;
        # any other scenario counts then raises instead of running dozens of
        # fits, so a stray call is a genuine test failure.
        real_invoke = session_module.stage_backend.invoke_source_fit
        real_verify = session_module.stage_backend.verify_source_prediction

        if positive:

            def _counting_invoke(*args, **kwargs):
                summary["fit_calls"] += 1
                lazy = importlib.import_module("atlas_sers.evaluation.p03_metrics")
                summary["loader_ok"] = lazy.__spec__.loader is runtime._finder
                return real_invoke(*args, **kwargs)

            def _counting_verify(*args, **kwargs):
                summary["verify_calls"] += 1
                return real_verify(*args, **kwargs)

        else:

            def _counting_invoke(*args, **kwargs):
                summary["fit_calls"] += 1
                raise AssertionError(
                    "invoke_source_fit called in scenario " + repr(scenario)
                )

            def _counting_verify(*args, **kwargs):
                summary["verify_calls"] += 1
                raise AssertionError(
                    "verify_source_prediction called in scenario " + repr(scenario)
                )

        session_module.stage_backend.invoke_source_fit = _counting_invoke
        session_module.stage_backend.verify_source_prediction = _counting_verify

        real_candidate_fit = p03_runtime.run_candidate_fit

        if positive:

            def _counting_candidate_fit(*args, **kwargs):
                summary["candidate_fit_calls"] += 1
                return real_candidate_fit(*args, **kwargs)

        else:

            def _counting_candidate_fit(*args, **kwargs):
                summary["candidate_fit_calls"] += 1
                raise AssertionError(
                    "run_candidate_fit called in scenario " + repr(scenario)
                )

        def _forbidden_neural_fit(*args, **kwargs):
            summary["neural_fit_calls"] += 1
            raise AssertionError("train_development_fit is forbidden")

        p03_runtime.run_candidate_fit = _counting_candidate_fit
        p05_development.train_development_fit = _forbidden_neural_fit

        if positive:
            store_module = importlib.import_module("atlas_sers.evaluation.p08_u0_store")
            real_create = store_module.create_store

            def _capture_create(*args, **kwargs):
                owner = real_create(*args, **kwargs)
                holder["owner"] = owner
                return owner

            store_module.create_store = _capture_create

            real_run_pair = session_module._SourceSession.run_pair
            real_close = session_module._SourceSession.close

            def _capture_io_provenance(self):
                summary["session_started_ns"] = self._started_ns
                summary["io_started_ns"] = self._io._started_ns
                summary["io_control_sha256"] = self._io._control_sha256
                summary["io_control_bytes"] = self._io._control_bytes
                if summary["launch_present_before_fit"] is None:
                    launch_path = (
                        pathlib.Path(output_root) / "control" / "launch.json"
                    )
                    summary["launch_present_before_fit"] = launch_path.is_file()
                    if launch_path.is_file():
                        summary["launch_sha_before_fit"] = hashlib.sha256(
                            launch_path.read_bytes()
                        ).hexdigest()

            def _postclose_bytes():
                total = 0
                root = pathlib.Path(output_root)
                if root.is_dir():
                    for path in root.rglob("*"):
                        if path.is_file() and path.name != "terminal.json":
                            total += path.stat().st_size
                return total

            def _wrapped_run_pair(self, fit_id, *args, **kwargs):
                summary["fit_job_id"] = fit_id
                _capture_io_provenance(self)
                events = holder["owner"].snapshot()["events"]
                assert events[0]["event_type"] == "session_open", events[0]
                summary["initial_marker_elapsed_ns"] = events[0]["elapsed_ns"]
                real_run_pair(self, fit_id, *args, **kwargs)
                snapshot = holder["owner"].snapshot()
                inner = snapshot.get("summary", {})
                summary["snapshot_model_fit_attempts"] = inner.get(
                    "model_fit_attempts"
                )
                summary["snapshot_source_prediction_attempts"] = inner.get(
                    "source_prediction_attempts"
                )
                report = self.report()
                summary["pair_report_completed_pair_count"] = report.get(
                    "completed_pair_count"
                )
                summary["pair_report_incomplete"] = report.get("incomplete")
                raise holder["sentinel"]

            def _wrapped_close(self, *args, **kwargs):
                _capture_io_provenance(self)
                report = real_close(self, *args, **kwargs)
                summary["postclose_bytes"] = _postclose_bytes()
                summary["close_report_json"] = json.dumps(
                    report, sort_keys=True, default=str
                )
                return report

            session_module._SourceSession.run_pair = _wrapped_run_pair
            session_module._SourceSession.close = _wrapped_close

    real_load_bootstrap = launcher._load_bootstrap

    def _bootstrap_wrapper(package_root):
        module = real_load_bootstrap(package_root)
        real_ctx = module.authenticated_runtime

        @contextlib.contextmanager
        def _wrapped_ctx(root):
            with real_ctx(root) as runtime:
                if scenario != "default_denied":
                    _install(runtime)
                yield runtime

        module.authenticated_runtime = _wrapped_ctx
        return module

    if scenario != "default_denied":
        launcher._load_bootstrap = _bootstrap_wrapper

    if scenario == "finder_replacement":
        real_prepare = launcher._prepare_runtime_inputs

        def _finder_prepare(runtime, *args, **kwargs):
            result = real_prepare(runtime, *args, **kwargs)
            owned = runtime._owned_meta_path
            owned[:] = [entry for entry in owned if entry is not runtime._finder]
            summary["injection_applied"] = True
            return result

        launcher._prepare_runtime_inputs = _finder_prepare

    if scenario == "module_replacement":
        real_prepare = launcher._prepare_runtime_inputs

        def _module_prepare(runtime, *args, **kwargs):
            result = real_prepare(runtime, *args, **kwargs)
            sys.modules["atlas_sers.evaluation"] = types.ModuleType(
                "atlas_sers.evaluation"
            )
            summary["injection_applied"] = True
            return result

        launcher._prepare_runtime_inputs = _module_prepare

    sys.addaudithook(_audit)

    try:
        launcher.launch(config["package_root"], config["permit_path"])
        summary["reached_success"] = True
    except BaseException as exc:  # noqa: BLE001
        summary["error_type"] = type(exc).__name__
        reason = getattr(exc, "reason_code", None)
        summary["error_reason"] = reason if type(reason) is str else None
        summary["error_is_sentinel"] = exc is holder["sentinel"]

    torch_module = sys.modules.get("torch")
    summary["torch_imported"] = torch_module is not None
    summary["cuda_observed"] = torch_module is not None
    summary["cuda_initialized"] = bool(
        torch_module is not None and torch_module.cuda.is_initialized()
    )
    summary["atlas_modules_loaded"] = sorted(
        name
        for name in sys.modules
        if name == "atlas_sers" or name.startswith("atlas_sers.")
    )
    summary["imported_modules"] = sorted(imported_after_hook)
    summary["opens"] = counters["open"]
    summary["mutation_events"] = counters["mutation"]
    summary["process_events"] = counters["process"]
    summary["network_events"] = counters["network"]
    summary["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


# ---------------------------------------------------------------------------
# Discovery and package construction
# ---------------------------------------------------------------------------


def _find_launcher_path():
    path = SCRIPTS / "run_p08_u0_smoke.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_launcher(path):
    spec = importlib.util.spec_from_file_location("p08_u0_launcher_loaded", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_catalog_bytes(pkg_root, existing, revision):
    src_dir = pkg_root / "src" / "atlas_sers"
    entries = {}
    for path in sorted(src_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(pkg_root).as_posix()
        parts = relative.split("/")
        is_package = parts[-1] == "__init__.py"
        stems = parts[2:-1]
        if is_package:
            module = "atlas_sers" if not stems else "atlas_sers." + ".".join(stems)
        else:
            leaf = parts[-1][:-3]
            module = "atlas_sers." + ".".join(stems + [leaf])
        payload = path.read_bytes()
        entries[module] = {
            "is_package": is_package,
            "module": module,
            "relative_path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    catalog = {
        "$id": existing["$id"],
        "execution_authorized": False,
        "import_roots": list(existing["import_roots"]),
        "namespace": existing["namespace"],
        "schema_version": existing["schema_version"],
        "source_revision": revision,
        "sources": [entries[module] for module in sorted(entries)],
    }
    return json.dumps(
        catalog, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _build_bootstrap_bytes(catalog_sha256):
    text = (SCRIPTS / BOOTSTRAP_NAME).read_text(encoding="utf-8")
    marker = 'EXPECTED_CATALOG_SHA256 = "'
    start = text.index(marker) + len(marker)
    end = text.index('"', start)
    return (text[:start] + catalog_sha256 + text[end:]).encode("utf-8")


@pytest.fixture(scope="module")
def package_template(tmp_path_factory):
    base = tmp_path_factory.mktemp("p08-launcher-template")
    pkg = base / "pkg"
    (pkg / "src").mkdir(parents=True)
    shutil.copytree(
        SRC,
        pkg / "src" / "atlas_sers",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    contracts_dir = pkg / "plan" / "contracts"
    contracts_dir.mkdir(parents=True)
    for name in CONTRACT_FILES:
        shutil.copy2(CONTRACTS / name, contracts_dir / name)
    (pkg / "scripts").mkdir()
    existing = json.loads((CONTRACTS / CATALOG_NAME).read_text(encoding="utf-8"))
    catalog_bytes = _build_catalog_bytes(pkg, existing, TEST_REVISION)
    catalog_sha = hashlib.sha256(catalog_bytes).hexdigest()
    (contracts_dir / CATALOG_NAME).write_bytes(catalog_bytes)
    bootstrap_bytes = _build_bootstrap_bytes(catalog_sha)
    (pkg / "scripts" / BOOTSTRAP_NAME).write_bytes(bootstrap_bytes)
    bootstrap_sha = hashlib.sha256(bootstrap_bytes).hexdigest()
    return {
        "pkg": pkg,
        "catalog_sha": catalog_sha,
        "revision": TEST_REVISION,
        "bootstrap_sha": bootstrap_sha,
    }


def _materialize(monkeypatch, tmp_path, template):
    pkg = tmp_path / "pkg"
    shutil.copytree(template["pkg"], pkg)
    launcher_path = _find_launcher_path()
    launcher = _load_launcher(launcher_path)

    ctx, inputs = runtime_fixture._prepare(monkeypatch, "master_cv")
    first_fit = json.loads(inputs.pairs[0].prepared_pair.fit_job_json)
    first_pred = json.loads(inputs.pairs[0].prepared_pair.prediction_job_json)
    assert first_fit["model_id"] == "C-RANDOM-FOREST"

    data = tmp_path / "data"
    data.mkdir()
    metadata_paths = {}
    metadata_pins = {}
    for name in metadata_fixture.ARG_NAMES:
        payload = ctx["metadata_bytes"][name]
        path = data / (name + ".bin")
        path.write_bytes(payload)
        metadata_paths[name] = str(path)
        metadata_pins[name] = hashlib.sha256(payload).hexdigest()
    action_paths = {}
    for action, payload in ctx["action_bytes"].items():
        path = data / (action + ".npz")
        path.write_bytes(payload)
        action_paths[action] = str(path)
    audit_path = data / "specification_audit.json"
    audit_path.write_bytes(runtime_fixture._AUDIT_BYTES)
    candidate_path = data / "candidate_registry.csv"
    candidate_path.write_bytes(runtime_fixture.CANDIDATE_BYTES)
    candidates_sha = hashlib.sha256(runtime_fixture.CANDIDATE_BYTES).hexdigest()

    permit = {
        "schema_version": launcher.PERMIT_SCHEMA,
        "stage": "U0",
        "execution_authorized": True,
        "proposal_sha256": ctx["case"]["proposal"]["proposal_sha256"],
        "manifest_sha256": ctx["case"]["attempt"]["manifest_sha256"],
        "catalog_sha256": template["catalog_sha"],
        "source_revision": template["revision"],
        "limits": dict(launcher._EXPECTED_LIMITS),
        "output_root": str(tmp_path / "out"),
        "metadata_paths": metadata_paths,
        "action_paths": action_paths,
        "specification_audit_path": str(audit_path),
        "candidate_registry_path": str(candidate_path),
    }
    return {
        "base": tmp_path,
        "pkg": pkg,
        "launcher": launcher,
        "launcher_path": launcher_path,
        "permit": permit,
        "permit_path": data / "permit.json",
        "metadata_pins": metadata_pins,
        "action_pins": ctx["pins"],
        "candidates_sha": candidates_sha,
        "proposal_sha": ctx["case"]["proposal"]["proposal_sha256"],
        "manifest_sha": ctx["case"]["attempt"]["manifest_sha256"],
        "bootstrap_sha": template["bootstrap_sha"],
        "resource_record": resources(filesystem_free_bytes=64 * GIB),
        "overlimit_record": resources(process_tree_rss_bytes=10**18),
        "cuda_packet": {
            "observed": False,
            "initialized": False,
            "allocated": 0,
            "reserved": 0,
            "device_used": 0,
            "peak": 0,
        },
        "ctx": ctx,
        "inputs": inputs,
        "first_fit_id": first_fit["job_id"],
        "first_pred_id": first_pred["job_id"],
        "action_keys": list(launcher._ACTION_KEYS),
    }


# ---------------------------------------------------------------------------
# Child-asset writing and subprocess helpers
# ---------------------------------------------------------------------------


def _write_child_assets(env, scenario):
    permit_bytes = json.dumps(
        env["permit"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    env["permit_path"].write_bytes(permit_bytes)
    permit_sha = hashlib.sha256(permit_bytes).hexdigest()
    config = {
        "scenario": scenario,
        "launcher_path": str(env["launcher_path"]),
        "package_root": str(env["pkg"]),
        "permit_path": str(env["permit_path"]),
        "output_root": env["permit"]["output_root"],
        "bootstrap_sha256": env["bootstrap_sha"],
        "proposal_sha256": env["proposal_sha"],
        "manifest_sha256": env["manifest_sha"],
        "permit_sha256": permit_sha,
        "metadata_pins": env["metadata_pins"],
        "action_pins": env["action_pins"],
        "candidates_sha256": env["candidates_sha"],
        "resource_record": env["resource_record"],
        "overlimit_record": env["overlimit_record"],
        "cuda_packet": env["cuda_packet"],
    }
    config_path = env["base"] / ("config-" + scenario + ".json")
    config_path.write_text(
        json.dumps(config, sort_keys=True, ensure_ascii=True), encoding="utf-8"
    )
    driver_path = env["base"] / ("driver-" + scenario + ".py")
    driver_path.write_text(_DRIVER, encoding="utf-8")
    return config_path, driver_path


def _run_child(driver_path, config_path, cwd, *, cublas_workspace_config=":4096:8"):
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    environment.pop("PYTHONPATH", None)
    # Deterministically pin the child's CUDA workspace setting, overriding any
    # inherited host value; ``None`` means the variable must be absent.
    if cublas_workspace_config is None:
        environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        environment["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
    return subprocess.run(
        [sys.executable, "-I", "-B", str(driver_path), str(config_path)],
        capture_output=True,
        text=True,
        env=environment,
        cwd=str(cwd),
        timeout=120,
        check=False,
    )


def _summary_from(result):
    stderr = result.stderr[-4000:]
    assert result.returncode == 0, (
        f"child exited rc={result.returncode}; stderr={stderr}"
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines, f"child produced no summary on stdout; stderr={stderr}"
    return json.loads(lines[-1])


def _readtree(root):
    base = Path(root)
    result = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_file():
            result[str(path.relative_to(base))] = path.read_bytes()
    return result


# ---------------------------------------------------------------------------
# Positive cohesive path and relaunch refusal
# ---------------------------------------------------------------------------


def test_interrupt_after_first_pair_and_relaunch_refuses(
    monkeypatch, tmp_path, package_template
):
    env = _materialize(monkeypatch, tmp_path, package_template)
    config_path, driver_path = _write_child_assets(env, "interrupt_after_first")
    result = _run_child(driver_path, config_path, env["base"])
    summary = _summary_from(result)

    assert summary["scenario"] == "interrupt_after_first", summary
    assert summary["reached_success"] is False, summary
    assert summary["runtime_entered"] is True, summary
    assert summary["setup_cuda_called"] is True, summary
    assert summary["sample_calls"] >= 1, summary
    assert summary["fit_calls"] == 1, summary
    assert summary["verify_calls"] == 1, summary
    assert summary["candidate_fit_calls"] == 1, summary
    assert summary["neural_fit_calls"] == 0, summary
    assert summary["error_type"] == "KeyboardInterrupt", summary
    assert summary["error_is_sentinel"] is True, summary
    assert summary["loader_ok"] is True, summary
    assert summary["cuda_initialized"] is False, summary
    assert summary["fit_job_id"] == env["first_fit_id"], summary
    assert summary["snapshot_model_fit_attempts"] == 1, summary
    assert summary["snapshot_source_prediction_attempts"] == 1, summary
    assert summary["pair_report_completed_pair_count"] == 1, summary
    assert summary["pair_report_incomplete"] is True, summary

    output_root = Path(env["permit"]["output_root"])
    control = output_root / "control"
    launch_bytes = (control / "launch.json").read_bytes()
    terminal_bytes = (control / "terminal.json").read_bytes()
    launch = json.loads(launch_bytes)
    terminal = json.loads(terminal_bytes)
    assert terminal["status"] == "failed", terminal
    assert terminal["reason_code"] == "interrupted", terminal
    assert terminal["completed_pair_count"] is None, terminal
    assert terminal["counters_known"] is False, terminal

    close_text = summary["close_report_json"]
    assert close_text, "session close report was not captured"
    close_report = json.loads(close_text)

    launch_sha = hashlib.sha256(launch_bytes).hexdigest()
    assert summary["launch_present_before_fit"] is True, summary
    assert summary["launch_sha_before_fit"] == launch_sha, summary
    assert summary["io_control_sha256"] == launch_sha, summary
    assert type(summary["io_control_bytes"]) is int, summary
    assert summary["io_control_bytes"] == len(launch_bytes), summary
    assert summary["io_started_ns"] == launch["outer_start_monotonic_ns"], summary
    assert summary["session_started_ns"] == launch["outer_start_monotonic_ns"], summary
    assert summary["initial_marker_elapsed_ns"] == 0, summary

    # The real close report never invents launch provenance fields.
    assert "launch_sha256" not in close_report
    assert "outer_start_monotonic_ns" not in close_report
    assert close_report["closed"] is True, close_report
    assert close_report["incomplete"] is True, close_report
    assert close_report["completed_pair_count"] == 1, close_report
    assert close_report["fit_attempt_count"] == 1, close_report
    assert close_report["prediction_attempt_count"] == 1, close_report
    assert close_report["head_sha256"], close_report
    assert close_report["manifest_sha256"], close_report
    assert close_report["journal_state"], close_report

    artifacts = output_root / "artifacts"
    fit_id = env["first_fit_id"]
    pred_id = env["first_pred_id"]
    names = {path.name for path in artifacts.iterdir() if path.is_file()}
    for required in (
        fit_id + "-summary.json",
        fit_id + "-predictions.csv",
        fit_id + "-receipt.json",
        pred_id + "-verification.json",
        pred_id + "-receipt.json",
    ):
        assert required in names
    for receipt_name in (fit_id + "-receipt.json", pred_id + "-receipt.json"):
        receipt = json.loads((artifacts / receipt_name).read_bytes())
        for item in receipt["artifacts"]:
            data = (artifacts / item["name"]).read_bytes()
            assert len(data) == item["size_bytes"]
            assert hashlib.sha256(data).hexdigest() == item["sha256"]

    artifact_bytes = sum(
        path.stat().st_size for path in artifacts.iterdir() if path.is_file()
    )
    final_tree = _readtree(output_root)
    final_total = sum(
        len(payload)
        for name, payload in final_tree.items()
        if name != "control/terminal.json"
    )
    assert close_report["observed_logical_bytes"] == final_total, (
        close_report,
        final_total,
    )
    assert summary["postclose_bytes"] == final_total, (
        summary["postclose_bytes"],
        final_total,
    )
    assert close_report["observed_logical_bytes"] >= artifact_bytes, close_report
    assert close_report["recorded_artifact_bytes"] >= close_report[
        "observed_logical_bytes"
    ], close_report
    assert close_report["recorded_active_wall_ns"] >= 0, close_report
    assert (
        close_report["measured_elapsed_ns"] >= close_report["recorded_active_wall_ns"]
    ), close_report
    assert close_report["measured_elapsed_ns"] == (
        close_report["recorded_active_wall_ns"] + close_report["finalization_tail_ns"]
    ), close_report

    held = {
        row["observation_uid"]
        for row in env["ctx"]["case"]["rows"]
        if row["_is_test"]
    }
    bound = set()
    for pair in env["inputs"].pairs:
        for observation in tuple(pair.fitting_observations) + tuple(
            pair.validation_observations
        ):
            bound.add(observation.uid)
    assert bound
    assert not (bound & held)

    before = _readtree(output_root)
    assert before

    relaunch_config, relaunch_driver = _write_child_assets(env, "relaunch")
    relaunch = _run_child(relaunch_driver, relaunch_config, env["base"])
    relaunch_summary = _summary_from(relaunch)
    assert relaunch_summary["error_reason"] == "output_exists"
    assert relaunch_summary["runtime_entered"] is False
    assert relaunch_summary["setup_cuda_called"] is False
    assert relaunch_summary["fit_calls"] == 0
    assert relaunch_summary["verify_calls"] == 0
    assert relaunch_summary["sample_calls"] == 0
    assert relaunch_summary["candidate_fit_calls"] == 0
    assert relaunch_summary["neural_fit_calls"] == 0
    assert _readtree(output_root) == before


# ---------------------------------------------------------------------------
# Independent zero-fit refusal tests, each in a fresh child
# ---------------------------------------------------------------------------


def test_unset_pin_denies_before_any_open(monkeypatch, tmp_path, package_template):
    env = _materialize(monkeypatch, tmp_path, package_template)
    config_path, driver_path = _write_child_assets(env, "default_denied")
    result = _run_child(driver_path, config_path, env["base"])
    summary = _summary_from(result)
    assert summary["scenario"] == "default_denied"
    assert summary["error_reason"] == "scientific_execution_not_authorized"
    assert summary["opens"] == 0
    assert summary["mutation_events"] == 0
    assert summary["process_events"] == 0
    assert summary["network_events"] == 0
    assert summary["atlas_modules_loaded"] == []
    assert summary["torch_imported"] is False
    assert summary["runtime_entered"] is False
    assert summary["setup_cuda_called"] is False
    assert summary["fit_calls"] == 0
    assert summary["verify_calls"] == 0
    assert summary["candidate_fit_calls"] == 0
    assert summary["neural_fit_calls"] == 0
    assert not Path(env["permit"]["output_root"]).exists()


def test_deployed_pin_rejects_unapproved_permit(monkeypatch, tmp_path):
    """The released pin refuses an unapproved permit before any output claim."""
    launcher = _load_launcher(_find_launcher_path())
    assert launcher.APPROVED_PERMIT_SHA256 == (
        "b845de4ab7a340cd5c217b557c19affd34e17b6051f1bab7d4cc0dd208ee70d0"
    )
    # The in-process pytest interpreter is not ``-I``; relax only that
    # positional environment guard for this synthetic refusal test.
    monkeypatch.setattr(launcher, "_require_environment", lambda: None)

    def _forbidden(*args, **kwargs):
        raise AssertionError("scientific stage reached before the permit hash check")

    monkeypatch.setattr(launcher, "_claim_output", _forbidden)
    monkeypatch.setattr(launcher, "_load_bootstrap", _forbidden)
    monkeypatch.setattr(launcher, "_run", _forbidden)

    package_root = tmp_path / "package"
    package_root.mkdir()
    permit_path = tmp_path / "permit.json"
    permit_path.write_bytes(b"{}")

    with pytest.raises(launcher.LaunchError) as excinfo:
        launcher.launch(str(package_root), str(permit_path))

    assert excinfo.value.reason_code == "permit_hash_mismatch"
    assert permit_path.read_bytes() == b"{}"
    assert launcher.APPROVED_PERMIT_SHA256 != hashlib.sha256(b"{}").hexdigest()


# ---------------------------------------------------------------------------
# CUDA workspace preflight (real entry, isolated subprocess, never training)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "workspace_config",
    [None, "", ":16:8", "0:0:0"],
    ids=["missing", "empty", "wrong", "invalid"],
)
def test_cublas_workspace_config_required(
    monkeypatch, tmp_path, package_template, workspace_config
):
    """An absent or incorrect workspace setting denies before any work."""
    env = _materialize(monkeypatch, tmp_path, package_template)
    config_path, driver_path = _write_child_assets(env, "cublas_workspace_invalid")
    result = _run_child(
        driver_path,
        config_path,
        env["base"],
        cublas_workspace_config=workspace_config,
    )
    summary = _summary_from(result)
    assert summary["error_reason"] == "cublas_workspace_config_invalid", summary
    assert summary["error_type"] == "LaunchError", summary
    assert summary["reached_success"] is False, summary
    assert summary["runtime_entered"] is False, summary
    assert summary["setup_cuda_called"] is False, summary
    assert summary["torch_imported"] is False, summary
    assert summary["cuda_initialized"] is False, summary
    assert summary["atlas_modules_loaded"] == [], summary
    assert summary["mutation_events"] == 0, summary
    assert summary["process_events"] == 0, summary
    assert summary["network_events"] == 0, summary
    assert summary["sample_calls"] == 0, summary
    assert summary["fit_calls"] == 0, summary
    assert summary["verify_calls"] == 0, summary
    assert summary["candidate_fit_calls"] == 0, summary
    assert summary["neural_fit_calls"] == 0, summary
    assert not Path(env["permit"]["output_root"]).exists()


def test_valid_cublas_workspace_config_reaches_denied_authority_gate(
    monkeypatch, tmp_path, package_template
):
    """A valid setting is preserved and still hits the unset-pin gate."""
    env = _materialize(monkeypatch, tmp_path, package_template)
    config_path, driver_path = _write_child_assets(env, "default_denied")
    result = _run_child(
        driver_path,
        config_path,
        env["base"],
        cublas_workspace_config=":4096:8",
    )
    summary = _summary_from(result)
    assert summary["error_reason"] == "scientific_execution_not_authorized", summary
    assert summary["cublas_workspace_config"] == ":4096:8", summary
    assert summary["runtime_entered"] is False, summary
    assert summary["setup_cuda_called"] is False, summary
    assert summary["torch_imported"] is False, summary
    assert summary["atlas_modules_loaded"] == [], summary
    assert summary["sample_calls"] == 0, summary
    assert summary["fit_calls"] == 0, summary
    assert summary["verify_calls"] == 0, summary
    assert summary["candidate_fit_calls"] == 0, summary
    assert summary["neural_fit_calls"] == 0, summary
    assert not Path(env["permit"]["output_root"]).exists()


def test_changed_permit_bytes_rejected(monkeypatch, tmp_path, package_template):
    env = _materialize(monkeypatch, tmp_path, package_template)
    config_path, driver_path = _write_child_assets(env, "permit_hash_mismatch")
    data = bytearray(env["permit_path"].read_bytes())
    data[0] ^= 0xFF
    env["permit_path"].write_bytes(bytes(data))
    result = _run_child(driver_path, config_path, env["base"])
    summary = _summary_from(result)
    assert summary["error_reason"] == "permit_hash_mismatch"
    assert summary["runtime_entered"] is False
    assert summary["setup_cuda_called"] is False
    assert summary["fit_calls"] == 0
    assert summary["verify_calls"] == 0
    assert summary["candidate_fit_calls"] == 0
    assert summary["neural_fit_calls"] == 0
    assert not Path(env["permit"]["output_root"]).exists()


def test_changed_catalog_after_pinning_rejected(monkeypatch, tmp_path, package_template):
    env = _materialize(monkeypatch, tmp_path, package_template)
    config_path, driver_path = _write_child_assets(env, "catalog_hash_mismatch")
    catalog_path = env["pkg"] / "plan" / "contracts" / CATALOG_NAME
    data = bytearray(catalog_path.read_bytes())
    data[0] ^= 0xFF
    catalog_path.write_bytes(bytes(data))
    result = _run_child(driver_path, config_path, env["base"])
    summary = _summary_from(result)
    assert summary["error_reason"] == "catalog_hash_mismatch"
    assert summary["runtime_entered"] is False
    assert summary["setup_cuda_called"] is False
    assert summary["fit_calls"] == 0
    assert summary["verify_calls"] == 0
    assert summary["candidate_fit_calls"] == 0
    assert summary["neural_fit_calls"] == 0
    control = Path(env["permit"]["output_root"]) / "control"
    assert (control / "launch.json").is_file()
    terminal = json.loads((control / "terminal.json").read_bytes())
    assert terminal["status"] == "failed"


def test_changed_action_archive_rejected(monkeypatch, tmp_path, package_template):
    env = _materialize(monkeypatch, tmp_path, package_template)
    config_path, driver_path = _write_child_assets(env, "input_preparation_failed")
    action = env["action_keys"][1]
    path = Path(env["permit"]["action_paths"][action])
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))
    result = _run_child(driver_path, config_path, env["base"])
    summary = _summary_from(result)
    assert summary["error_reason"] == "input_preparation_failed"
    assert summary["setup_cuda_called"] is False
    assert summary["sample_calls"] == 0
    assert summary["fit_calls"] == 0
    assert summary["verify_calls"] == 0
    assert summary["candidate_fit_calls"] == 0
    assert summary["neural_fit_calls"] == 0
    assert not (Path(env["permit"]["output_root"]) / "artifacts").exists()


def test_finder_replacement_rejected(monkeypatch, tmp_path, package_template):
    env = _materialize(monkeypatch, tmp_path, package_template)
    config_path, driver_path = _write_child_assets(env, "finder_replacement")
    result = _run_child(driver_path, config_path, env["base"])
    summary = _summary_from(result)
    assert summary["error_reason"] == "scope_finder_changed"
    assert summary["injection_applied"] is True
    assert summary["runtime_entered"] is True
    assert summary["setup_cuda_called"] is True
    assert summary["fit_calls"] == 0
    assert summary["verify_calls"] == 0
    assert summary["candidate_fit_calls"] == 0
    assert summary["neural_fit_calls"] == 0


def test_module_object_replacement_rejected(monkeypatch, tmp_path, package_template):
    env = _materialize(monkeypatch, tmp_path, package_template)
    config_path, driver_path = _write_child_assets(env, "module_replacement")
    result = _run_child(driver_path, config_path, env["base"])
    summary = _summary_from(result)
    assert summary["error_reason"] == "module_object_mismatch"
    assert summary["injection_applied"] is True
    assert summary["runtime_entered"] is True
    assert summary["fit_calls"] == 0
    assert summary["verify_calls"] == 0
    assert summary["candidate_fit_calls"] == 0
    assert summary["neural_fit_calls"] == 0


def test_resource_overlimit_rejected(monkeypatch, tmp_path, package_template):
    env = _materialize(monkeypatch, tmp_path, package_template)
    config_path, driver_path = _write_child_assets(env, "resource_limit")
    result = _run_child(driver_path, config_path, env["base"])
    summary = _summary_from(result)
    assert summary["error_reason"] == "resource_limit_exceeded"
    assert summary["runtime_entered"] is True
    assert summary["setup_cuda_called"] is True
    assert summary["sample_calls"] >= 1
    assert summary["fit_calls"] == 0
    assert summary["verify_calls"] == 0
    assert summary["candidate_fit_calls"] == 0
    assert summary["neural_fit_calls"] == 0
