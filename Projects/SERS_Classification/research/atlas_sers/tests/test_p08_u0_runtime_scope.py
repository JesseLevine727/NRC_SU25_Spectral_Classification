"""Integration tests for the p08 authenticated runtime ownership scope.

Every test spawns an isolated ``-I -B`` child through the existing public
helpers in ``test_p08_u0_runtime_import``.  The real atlas_sers package is
never imported by this test module (other test modules in the same pytest
process may import the real package) and no GPU or scientific kernel is
ever executed.
"""

from pathlib import Path

import pytest

from tests.test_p08_u0_runtime_import import (
    _payload,
    _pin,
    _run_child,
    _standard_specs,
    _write_fixture,
)

LAZY_NAME = "atlas_sers.models.lazy_fixture"
LAZY_REL = "src/atlas_sers/models/lazy_fixture.py"
LAZY_SRC = "VALUE = 'captured-lazy'\n"
SCOPE_SCHEMA = "nato-sers-p08-authenticated-runtime-scope-v1"
ROGUE_NAME = "atlas_sers.models.rogue_lazy"

REPORT_KEYS = set(
    "schema_version execution_authorized external_dependencies_authenticated "
    "project_sources_verified project_modules_loaded_from_authenticated_bytes "
    "catalog_sha256 source_revision verified_source_count "
    "loaded_project_module_count required_import_count scope_active".split()
)


_PRELUDE = """
audit.EXPECTED_CATALOG_SHA256 = __PIN__
import importlib, types, threading
ORIG_META = sys.meta_path
RESULT = {}
def _reason(exc):
    return getattr(exc, "reason_code", None) or type(exc).__name__
def _exit(cm):
    try:
        cm.__exit__(None, None, None)
        return None
    except BaseException as exc:
        return _reason(exc)
def _closed(handle):
    try:
        handle.verify_loaded()
        return None
    except BaseException as exc:
        return _reason(exc)
"""


def _render(template, **values):
    text = _PRELUDE + _dedent(template)
    for key, value in values.items():
        text = text.replace(f"__{key}__", str(value))
    return text


def _dedent(text):
    import textwrap

    return textwrap.dedent(text)


def _lazy_specs():
    specs = dict(_standard_specs())
    specs[LAZY_NAME] = (LAZY_REL, False, LAZY_SRC)
    return specs


def _leaf(specs):
    for name, (rel, is_pkg, _src) in specs.items():
        if not is_pkg:
            return name, rel
    raise AssertionError("no leaf runtime module in specs")


def _run(tmp_path, template, specs, **values):
    body = _render(template, **values)
    fixture = _write_fixture(tmp_path.resolve(), specs)
    proc = _run_child(_pin(body, fixture["catalog_sha256"]), fixture)
    return _payload(proc), fixture


_POSITIVE = """
with audit.authenticated_runtime(PKG_ROOT) as runtime:
    finder = sys.meta_path[0]
    report = runtime.verify_loaded()
    RESULT["keys"] = sorted(report)
    RESULT["schema"] = report["schema_version"]
    RESULT["auth"] = (report["execution_authorized"], report["external_dependencies_authenticated"])
    RESULT["verify"] = (
        report["project_sources_verified"],
        report["project_modules_loaded_from_authenticated_bytes"],
    )
    RESULT["loaded0"] = report["loaded_project_module_count"]
    RESULT["verified"] = report["verified_source_count"]
    RESULT["unknown_keys"] = [
        k for k in ("scientific_execution_performed", "cuda_initialized") if k in report
    ]
    report["loaded_project_module_count"] = -1
    RESULT["report_isolated"] = (
        runtime.verify_loaded()["loaded_project_module_count"] == RESULT["loaded0"]
    )
    data = runtime.source_bytes(__REL__)
    RESULT["src_type"] = type(data).__name__
    RESULT["src_text"] = bytes(data).decode("utf-8")
    RESULT["head_mid"] = sys.meta_path[0] is finder
    mod = importlib.import_module(__NAME__)
    RESULT["lazy_value"] = mod.VALUE
    RESULT["loaded1"] = runtime.verify_loaded()["loaded_project_module_count"]
    other = sys.modules[__OTHER__]
    RESULT["loader_identity"] = mod.__loader__ is finder and other.__loader__ is finder
    RESULT["head_end"] = sys.meta_path[0] is finder
    called = []
    for _n, _m in list(sys.modules.items()):
        if _n.startswith("atlas_sers"):
            called.extend(getattr(_m, "CALLED", []) or [])
    RESULT["called"] = called
    text = repr(runtime)
    RESULT["repr_path"] = PKG_ROOT in text
    RESULT["repr_src"] = "captured-lazy" in text
RESULT["meta_restored"] = sys.meta_path is ORIG_META
RESULT["closed_verify"] = _closed(runtime)
try:
    runtime.source_bytes(__REL__)
    RESULT["closed_source"] = None
except BaseException as exc:
    RESULT["closed_source"] = _reason(exc)
_emit(RESULT)
"""


_CAPTURED_FAULT = """
lazy_abs = os.path.join(PKG_ROOT, "src", "atlas_sers", "models", "lazy_fixture.py")
with audit.authenticated_runtime(PKG_ROOT) as runtime:
    before = bytes(runtime.source_bytes(__REL__))
    with open(lazy_abs, "w", encoding="utf-8") as fh:
        fh.write("VALUE = 'poison'\\n")
    RESULT["fault_injected"] = os.path.exists(lazy_abs)
    RESULT["captured_same"] = bytes(runtime.source_bytes(__REL__)) == before
    RESULT["lazy_value"] = importlib.import_module(__NAME__).VALUE
RESULT["meta_restored"] = sys.meta_path is ORIG_META
_emit(RESULT)
"""


_UNKNOWN_ON_DISK = """
with audit.authenticated_runtime(PKG_ROOT) as runtime:
    try:
        importlib.import_module("atlas_sers.models.rogue_lazy")
        RESULT["raised"] = False
        RESULT["exc_type"] = None
    except ImportError:
        RESULT["raised"] = True
        RESULT["exc_type"] = "ImportError"
    except BaseException as exc:
        RESULT["raised"] = False
        RESULT["exc_type"] = type(exc).__name__
    RESULT["in_sys"] = "atlas_sers.models.rogue_lazy" in sys.modules
RESULT["meta_restored"] = sys.meta_path is ORIG_META
_emit(RESULT)
"""


_INVALID_NAMES = """
cm = audit.authenticated_runtime(PKG_ROOT)
handle = cm.__enter__()
real_reader = audit._read_verified_file
calls = [0]
def _deny(*args, **kwargs):
    calls[0] += 1
    raise AssertionError("disk reread")
try:
    audit._read_verified_file = _deny
    try:
        handle.source_bytes(__BAD__)
        RESULT["reason"] = None
    except BaseException as exc:
        RESULT["reason"] = _reason(exc)
    try:
        RESULT["known_ok"] = len(bytes(handle.source_bytes(__REL__))) > 0
    except BaseException as exc:
        RESULT["known_ok"] = False
        RESULT["known_err"] = _reason(exc)
finally:
    audit._read_verified_file = real_reader
RESULT["calls"] = calls[0]
RESULT["exit_reason"] = _exit(cm)
RESULT["meta_restored"] = sys.meta_path is ORIG_META
_emit(RESULT)
"""


_OWNERSHIP = """
with audit.authenticated_runtime(PKG_ROOT) as runtime:
    owner = os.getpid()
    outcomes = []
    real_getpid = audit.os.getpid
    audit.os.getpid = lambda: owner + 424242
    try:
        for meth, args in (("verify_loaded", ()), ("source_bytes", (__REL__,))):
            try:
                getattr(runtime, meth)(*args)
                outcomes.append(("pid", meth, None))
            except BaseException as exc:
                outcomes.append(("pid", meth, _reason(exc)))
    finally:
        audit.os.getpid = real_getpid
    def _worker():
        for meth, args in (("verify_loaded", ()), ("source_bytes", (__REL__,))):
            try:
                getattr(runtime, meth)(*args)
                outcomes.append(("thread", meth, None))
            except BaseException as exc:
                outcomes.append(("thread", meth, _reason(exc)))
    worker = threading.Thread(target=_worker)
    worker.start()
    worker.join()
    RESULT["outcomes"] = outcomes
    RESULT["owner_ok"] = runtime.verify_loaded()["scope_active"]
RESULT["meta_restored"] = sys.meta_path is ORIG_META
_emit(RESULT)
"""


_SUBSTITUTION = """
cm = audit.authenticated_runtime(PKG_ROOT)
handle = cm.__enter__()
try:
    original = sys.modules[__TARGET__]
    clone = types.ModuleType(__TARGET__)
    clone.__dict__.update(original.__dict__)
    sys.modules[__TARGET__] = clone
    try:
        handle.source_bytes(__PATH__)
        RESULT["method_reason"] = None
    except BaseException as exc:
        RESULT["method_reason"] = _reason(exc)
finally:
    RESULT["exit_reason"] = _exit(cm)
RESULT["meta_restored"] = sys.meta_path is ORIG_META
RESULT["closed"] = _closed(handle)
_emit(RESULT)
"""


_FINDER_CHANGE = """
cm = audit.authenticated_runtime(PKG_ROOT)
handle = cm.__enter__()
try:
    sys.meta_path = list(sys.meta_path)
    try:
        handle.verify_loaded()
        RESULT["method_reason"] = None
    except BaseException as exc:
        RESULT["method_reason"] = _reason(exc)
finally:
    RESULT["exit_reason"] = _exit(cm)
RESULT["meta_restored"] = sys.meta_path is ORIG_META
RESULT["closed"] = _closed(handle)
_emit(RESULT)
"""


_CONSUMER = """
sentinel = __EXC__("consumer failure")
handle = None
caught = None
try:
    with audit.authenticated_runtime(PKG_ROOT) as handle:
        sys.meta_path = list(sys.meta_path)
        raise sentinel
except BaseException as exc:
    caught = exc
RESULT["same"] = caught is sentinel
RESULT["type"] = type(caught).__name__ if caught is not None else None
RESULT["meta_restored"] = sys.meta_path is ORIG_META
RESULT["closed"] = _closed(handle)
_emit(RESULT)
"""


_CUDA = """
os.environ["P08_FAKE_CUDA_INIT"] = "1"
error = None
entered = False
try:
    with audit.authenticated_runtime(PKG_ROOT):
        entered = True
except BaseException as exc:
    error = exc
RESULT["entered"] = entered
RESULT["error"] = type(error).__name__ if error is not None else None
RESULT["reason_code"] = getattr(error, "reason_code", None) if error is not None else None
RESULT["meta_restored"] = sys.meta_path is ORIG_META
_emit(RESULT)
"""


def test_positive_owned_scope_lifecycle(tmp_path):
    specs = _lazy_specs()
    other_name, _ = _leaf(specs)
    payload, _ = _run(
        tmp_path,
        _POSITIVE,
        specs,
        REL=repr(LAZY_REL),
        NAME=repr(LAZY_NAME),
        OTHER=repr(other_name),
    )
    assert payload["keys"] and set(payload["keys"]) == REPORT_KEYS
    assert len(payload["keys"]) == 11
    assert payload["schema"] == SCOPE_SCHEMA
    assert payload["auth"] == [False, False]
    assert payload["verify"] == [True, True]
    assert payload["unknown_keys"] == []
    assert payload["loaded0"] == 11
    assert payload["verified"] == 12
    assert payload["report_isolated"] is True
    assert payload["src_type"] == "bytes"
    assert payload["src_text"] == LAZY_SRC
    assert payload["head_mid"] is True
    assert payload["head_end"] is True
    assert payload["lazy_value"] == "captured-lazy"
    assert payload["loaded1"] == 12
    assert payload["loader_identity"] is True
    assert payload["called"] == []
    assert payload["repr_path"] is False
    assert payload["repr_src"] is False
    assert payload["meta_restored"] is True
    assert payload["closed_verify"] == "scope_closed"
    assert payload["closed_source"] == "scope_closed"


def test_source_bytes_survives_disk_fault(tmp_path):
    payload, _ = _run(
        tmp_path,
        _CAPTURED_FAULT,
        _lazy_specs(),
        REL=repr(LAZY_REL),
        NAME=repr(LAZY_NAME),
    )
    assert payload["fault_injected"] is True
    assert payload["captured_same"] is True
    assert payload["lazy_value"] == "captured-lazy"
    assert payload["meta_restored"] is True


def test_uncatalogued_lazy_module_has_no_fallback(tmp_path):
    fixture = _write_fixture(tmp_path.resolve(), dict(_standard_specs()))
    sentinel = tmp_path / "rogue_executed.txt"
    rogue_abs = Path(fixture["pkg_root"]) / "src" / "atlas_sers" / "models" / "rogue_lazy.py"
    rogue_abs.write_text(
        "import os\n"
        f"if not os.path.exists({str(sentinel)!r}):\n"
        f"    open({str(sentinel)!r}, 'w').close()\n"
        "VALUE = 'rogue'\n",
        encoding="utf-8",
    )
    proc = _run_child(_pin(_render(_UNKNOWN_ON_DISK), fixture["catalog_sha256"]), fixture)
    payload = _payload(proc)
    assert payload["raised"] is True
    assert payload["exc_type"] == "ImportError"
    assert payload["in_sys"] is False
    assert payload["meta_restored"] is True
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "bad",
    [
        "/tmp/absolute_escape.py",
        "../escape.py",
        "src/atlas_sers/models/not_catalogued.py",
        b"src/atlas_sers/models/lazy_fixture.py",
    ],
)
def test_invalid_source_bytes_names_do_not_read_disk(tmp_path, bad):
    payload, _ = _run(
        tmp_path,
        _INVALID_NAMES,
        _lazy_specs(),
        BAD=repr(bad),
        REL=repr(LAZY_REL),
    )
    assert payload["reason"] == "unknown_source"
    assert payload["known_ok"] is True
    assert payload["calls"] == 0
    assert payload["exit_reason"] is None
    assert payload["meta_restored"] is True


def test_scope_ownership_rejects_other_pid_and_thread(tmp_path):
    payload, _ = _run(tmp_path, _OWNERSHIP, _lazy_specs(), REL=repr(LAZY_REL))
    assert payload["outcomes"] == [
        ["pid", "verify_loaded", "scope_owner_changed"],
        ["pid", "source_bytes", "scope_owner_changed"],
        ["thread", "verify_loaded", "scope_owner_changed"],
        ["thread", "source_bytes", "scope_owner_changed"],
    ]
    assert payload["owner_ok"] is True
    assert payload["meta_restored"] is True


def test_module_object_substitution_is_detected(tmp_path):
    specs = dict(_standard_specs())
    target, path = _leaf(specs)
    payload, _ = _run(
        tmp_path,
        _SUBSTITUTION,
        specs,
        TARGET=repr(target),
        PATH=repr(path),
    )
    assert payload["method_reason"] == "module_object_mismatch"
    assert payload["exit_reason"] is not None
    assert payload["meta_restored"] is True
    assert payload["closed"] == "scope_closed"


def test_finder_change_detected_on_normal_exit(tmp_path):
    payload, _ = _run(tmp_path, _FINDER_CHANGE, dict(_standard_specs()))
    assert payload["method_reason"] == "scope_finder_changed"
    assert payload["exit_reason"] == "scope_finder_changed"
    assert payload["meta_restored"] is True
    assert payload["closed"] == "scope_closed"


@pytest.mark.parametrize("exc_name", ["ValueError", "KeyboardInterrupt", "SystemExit"])
def test_consumer_exception_object_is_preserved(tmp_path, exc_name):
    payload, _ = _run(tmp_path, _CONSUMER, dict(_standard_specs()), EXC=exc_name)
    assert payload["same"] is True
    assert payload["type"] == exc_name
    assert payload["meta_restored"] is True
    assert payload["closed"] == "scope_closed"


def test_scope_entry_rejects_initialized_fake_cuda(tmp_path):
    payload, _ = _run(tmp_path, _CUDA, dict(_standard_specs()))
    assert payload["entered"] is False
    assert payload["error"] is not None
    assert payload["reason_code"] == "cuda_initialized"
    assert payload["meta_restored"] is True
