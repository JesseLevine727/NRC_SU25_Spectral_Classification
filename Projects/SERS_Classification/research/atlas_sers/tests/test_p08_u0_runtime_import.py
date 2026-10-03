"""T132: fresh-process runtime import regression tests for the U0 audit.

These tests exercise the real public boundary of
``research/atlas_sers/scripts/check_p08_u0_runtime.py``
(``RuntimeAuditError.reason_code``, ``inspect_runtime``, ``main``) inside a
child interpreter launched with ``sys.executable -I -B``.  The parent never
clears or mutates any ``atlas_sers`` modules; the standalone audit script is
loaded by the child through ``importlib.util.spec_from_file_location``.

All fixtures are synthetic and local to ``tmp_path``.  No real SERS data,
metadata, arrays, fitting, projection or CUDA work is performed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda catalog: catalog.pop("$id"), id="missing-id"),
        pytest.param(
            lambda catalog: catalog.__setitem__("$id", "nato-sers-p08-runtime-source-catalog-v0"),
            id="wrong-id",
        ),
    ],
)
def test_catalog_identifier_invalid(tmp_path, mutate):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    pin = _repin_catalog(fixture, mutate)
    proc = _run_child(_failure_body(pin), fixture)
    payload = _payload(proc)
    assert payload["error"] == "catalog_identifier_invalid"
    assert payload["project_modules"] == 0


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_p08_u0_runtime.py"
CATALOG_REL = "plan/contracts/p08_runtime_source_catalog.json"
REPORT_SCHEMA = "nato-sers-p08-u0-runtime-import-audit-v1"

REQUIRED_IMPORTS = (
    "atlas_sers.evaluation.p08_u0_runtime_inputs",
    "atlas_sers.evaluation.p08_u0_session",
    "atlas_sers.evaluation.p03_runtime",
    "atlas_sers.evaluation.p05_development",
    "atlas_sers.evaluation.p04_runtime",
    "atlas_sers.models.classical",
    "atlas_sers.models.deep",
    "atlas_sers.models.acquisition",
)

REPORT_KEYS = frozenset(
    {
        "schema_version",
        "execution_authorized",
        "live_runtime_accepted",
        "scientific_execution_performed",
        "external_dependencies_authenticated",
        "project_sources_verified",
        "project_modules_loaded_from_authenticated_bytes",
        "catalog_sha256",
        "source_revision",
        "verified_source_count",
        "loaded_project_module_count",
        "required_import_count",
        "cuda_initialized",
        "report_sha256",
    }
)

_CALLABLE_SPECS = {
    "atlas_sers.evaluation.p03_runtime": ("run_candidate_fit", "function"),
    "atlas_sers.evaluation.p05_development": ("train_development_fit", "function"),
    "atlas_sers.evaluation.p08_u0_session": ("_SourceSession", "class"),
    "atlas_sers.evaluation.p08_u0_runtime_inputs": ("prepare_u0_runtime_inputs", "function"),
}


# --------------------------------------------------------------------------
# Synthetic fixture construction (parent only; writes are normal test setup)
# --------------------------------------------------------------------------


def _root_source():
    return (
        textwrap.dedent(
            """
        import os
        import sys
        import types

        _INITIALIZED = os.environ.get("P08_FAKE_CUDA_INIT") == "1"

        def _forbidden(name):
            def _raise(*args, **kwargs):
                raise AssertionError("forbidden CUDA probe: " + name)
            return _raise

        if "torch" not in sys.modules:
            _cuda = types.SimpleNamespace(
                is_initialized=lambda: _INITIALIZED,
                device_count=_forbidden("device_count"),
                current_device=_forbidden("current_device"),
                get_device_properties=_forbidden("get_device_properties"),
            )
            sys.modules["torch"] = types.SimpleNamespace(cuda=_cuda)

        import torch  # noqa: F401
        VALUE = "atlas_sers"
        """
        ).strip()
        + "\n"
    )


def _package_source(name):
    return f"VALUE = {name!r}\n"


def _leaf_source(module):
    return f"VALUE = {module!r}\n"


def _callable_source(module, name, kind):
    if kind == "class":
        return (
            "VALUE = {module!r}\n"
            "CALLED = []\n"
            "class {name}:\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        CALLED.append({name!r})\n"
            "        raise AssertionError({msg!r})\n"
        ).format(module=module, name=name, msg=name + " must not be constructed")
    return (
        "VALUE = {module!r}\n"
        "CALLED = []\n"
        "def {name}(*args, **kwargs):\n"
        "    CALLED.append({name!r})\n"
        "    raise AssertionError({msg!r})\n"
    ).format(module=module, name=name, msg=name + " must not be called")


def _package_rel(module):
    return "src/" + module.replace(".", "/") + "/__init__.py"


def _module_rel(module):
    return "src/" + module.replace(".", "/") + ".py"


def _standard_specs():
    specs = {
        "atlas_sers": (_package_rel("atlas_sers"), True, _root_source()),
        "atlas_sers.evaluation": (
            _package_rel("atlas_sers.evaluation"),
            True,
            _package_source("atlas_sers.evaluation"),
        ),
        "atlas_sers.models": (
            _package_rel("atlas_sers.models"),
            True,
            _package_source("atlas_sers.models"),
        ),
    }
    for module, (name, kind) in _CALLABLE_SPECS.items():
        specs[module] = (_module_rel(module), False, _callable_source(module, name, kind))
    for module in (
        "atlas_sers.evaluation.p04_runtime",
        "atlas_sers.models.acquisition",
        "atlas_sers.models.classical",
        "atlas_sers.models.deep",
    ):
        specs[module] = (_module_rel(module), False, _leaf_source(module))
    return specs


def _variant_specs(module, source):
    specs = _standard_specs()
    rel, is_pkg, _ = specs[module]
    specs[module] = (rel, is_pkg, source)
    return specs


def _write_fixture(base, specs):
    pkg_root = base / "pkgroot"
    written = {}
    for module in sorted(specs):
        rel, is_pkg, source = specs[module]
        path = pkg_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        written[module] = (rel, is_pkg, path)
    sources = []
    for module in sorted(specs):
        rel, is_pkg, path = written[module]
        data = path.read_bytes()
        sources.append(
            {
                "module": module,
                "relative_path": rel,
                "is_package": is_pkg,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    catalog = {
        "schema_version": "nato-sers-p08-runtime-source-catalog-v1",
        "$id": "nato-sers-p08-runtime-source-catalog-v1",
        "execution_authorized": False,
        "source_revision": "a" * 40,
        "namespace": "atlas_sers",
        "import_roots": list(REQUIRED_IMPORTS),
        "sources": sources,
    }
    raw = json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode("utf-8")
    catalog_path = pkg_root / CATALOG_REL
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_bytes(raw)
    return {
        "pkg_root": pkg_root,
        "catalog_path": catalog_path,
        "catalog_bytes": raw,
        "catalog_sha256": hashlib.sha256(raw).hexdigest(),
        "sources": sources,
        "source_paths": sorted(str(path) for path in written_module_paths(written)),
        "module_count": len(sources),
    }


def written_module_paths(written):
    return [entry[2] for entry in written.values()]


def _repin_catalog(fixture, mutate):
    catalog = json.loads(fixture["catalog_bytes"].decode("utf-8"))
    mutate(catalog)
    raw = json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fixture["catalog_path"].write_bytes(raw)
    fixture["catalog_bytes"] = raw
    return hashlib.sha256(raw).hexdigest()


def _mutate_unknown_catalog_key(catalog):
    catalog["unexpected_key"] = True


def _mutate_unknown_entry_key(catalog):
    catalog["sources"][0]["unexpected_key"] = True


def _mutate_wrong_roots(catalog):
    catalog["import_roots"] = ["atlas_sers.models.acquisition"]


def _mutate_boolean_size(catalog):
    catalog["sources"][0]["size_bytes"] = True


def _mutate_oversize_source(catalog):
    catalog["sources"][0]["size_bytes"] = 2 * 1024 * 1024 + 1


def _mutate_too_many_entries(catalog):
    template = dict(catalog["sources"][0])
    extra = 0
    while len(catalog["sources"]) <= 256:
        entry = dict(template)
        entry["module"] = f"atlas_sers.extra_{extra}"
        entry["relative_path"] = f"src/atlas_sers/extra_{extra}.py"
        entry["is_package"] = False
        entry["size_bytes"] = 0
        entry["sha256"] = hashlib.sha256(b"").hexdigest()
        catalog["sources"].append(entry)
        extra += 1


def _mutate_invalid_relative_path(catalog):
    catalog["sources"][0]["relative_path"] = "../escape.py"


# --------------------------------------------------------------------------
# Child bootstrap and subprocess plumbing (no parent-side atlas imports)
# --------------------------------------------------------------------------

_CHILD_PREAMBLE = textwrap.dedent(
    r"""
    import importlib.util
    import json
    import os
    import sys

    AUDIT_PATH = sys.argv[1]
    PKG_ROOT = sys.argv[2]
    ALLOWED_SOURCES = frozenset(json.loads(sys.argv[3]))

    _saved_dwb = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    _spec = importlib.util.spec_from_file_location("p08_audit_under_test", AUDIT_PATH)
    audit = importlib.util.module_from_spec(_spec)
    sys.modules["p08_audit_under_test"] = audit
    _spec.loader.exec_module(audit)
    sys.dont_write_bytecode = _saved_dwb

    def _emit(payload):
        sys.stdout.write(
            "__P08__" + json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )

    def _reason(exc):
        code = getattr(exc, "reason_code", None)
        return code if isinstance(code, str) else None
    """
).strip()


def _pin(template, pin):
    return textwrap.dedent(template).strip().replace("__PIN__", repr(pin))


def _run_child(body, fixture, *, env=None, isolated=True, no_bytecode=True, timeout=40):
    code = _CHILD_PREAMBLE + "\n" + body + "\n"
    argv = [sys.executable]
    if isolated:
        argv.append("-I")
    if no_bytecode:
        argv.append("-B")
    argv += [
        "-c",
        code,
        str(SCRIPT_PATH),
        str(fixture["pkg_root"]),
        json.dumps(fixture["source_paths"]),
    ]
    child_env = dict(os.environ)
    child_env.pop("PYTHONDONTWRITEBYTECODE", None)
    if env:
        child_env.update(env)
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=child_env,
    )


def _payload(proc):
    assert proc.returncode == 0, proc.stderr
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__P08__"):
            return json.loads(line[len("__P08__") :])
    raise AssertionError(
        f"child produced no payload rc={proc.returncode} "
        f"stderr={proc.stderr[-2000:]!r} stdout={proc.stdout[-2000:]!r}"
    )


_FAILURE_BODY = r"""
audit.EXPECTED_CATALOG_SHA256 = __PIN__
try:
    audit.inspect_runtime(PKG_ROOT)
except BaseException as exc:
    _emit({
        "error": _reason(exc),
        "type": type(exc).__name__,
        "project_modules": len([
            n for n in sys.modules if n == "atlas_sers" or n.startswith("atlas_sers.")
        ]),
    })
else:
    _emit({"error": None, "type": None, "project_modules": -1})
"""


def _failure_body(pin):
    return _pin(_FAILURE_BODY, pin)


_POSITIVE_BODY = r"""
_orig_meta = sys.meta_path
_opened = set()
_visited = set()
_violations = []
_catalog = os.path.join(
    os.path.normpath(PKG_ROOT), "plan", "contracts", "p08_runtime_source_catalog.json"
)
_write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_write_events = frozenset({
    "os.remove", "os.rename", "os.link", "os.symlink", "os.mkdir", "os.rmdir",
    "os.truncate", "os.chmod", "os.chown", "os.utime", "os.replace", "os.unlink",
    "shutil.copyfile", "shutil.copymode", "shutil.copystat", "shutil.move",
    "shutil.rmtree", "tempfile.mkstemp",
})
_proc_events = frozenset({
    "subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
})
_src_prefix = os.path.join(os.path.normpath(PKG_ROOT), "src") + os.sep

def _record(event):
    _violations.append(event)
    raise RuntimeError("forbidden audit event: " + event)


def _hook(event, args):
    if event in _proc_events or event.startswith("socket."):
        _record(event)
    if event in _write_events:
        _record(event)
    if event == "open":
        path = args[0] if args else None
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if isinstance(mode, str) and any(ch in mode for ch in "wax+"):
            _record("open mode " + mode)
        if isinstance(flags, int) and (flags & _write_flags):
            _record("open flags")
        if isinstance(path, (str, bytes, os.PathLike)):
            normalized = os.path.normpath(os.fsdecode(path))
            if normalized.startswith(_src_prefix):
                if normalized not in ALLOWED_SOURCES:
                    _record("unlisted project source open: " + normalized)
                _opened.add(normalized)


sys.addaudithook(_hook)

_original_read_verified_file = audit._read_verified_file


def _read_verified_file(path, max_bytes, reason_prefix):
    normalized = os.path.normpath(os.fsdecode(path))
    if normalized != _catalog and normalized not in ALLOWED_SOURCES:
        raise RuntimeError("unexpected verified read: " + normalized)
    if normalized in ALLOWED_SOURCES:
        _visited.add(normalized)
    return _original_read_verified_file(path, max_bytes, reason_prefix)


audit._read_verified_file = _read_verified_file
audit.EXPECTED_CATALOG_SHA256 = __PIN__
report = audit.inspect_runtime(PKG_ROOT)
_names = [
    "atlas_sers",
    "atlas_sers.evaluation",
    "atlas_sers.evaluation.p03_runtime",
    "atlas_sers.evaluation.p04_runtime",
    "atlas_sers.evaluation.p05_development",
    "atlas_sers.evaluation.p08_u0_runtime_inputs",
    "atlas_sers.evaluation.p08_u0_session",
    "atlas_sers.models",
    "atlas_sers.models.acquisition",
    "atlas_sers.models.classical",
    "atlas_sers.models.deep",
]
values = {name: getattr(sys.modules[name], "VALUE", None) for name in _names}
_callable_modules = [
    "atlas_sers.evaluation.p03_runtime",
    "atlas_sers.evaluation.p05_development",
    "atlas_sers.evaluation.p08_u0_session",
    "atlas_sers.evaluation.p08_u0_runtime_inputs",
]
calls = {name: list(sys.modules[name].CALLED) for name in _callable_modules}
_emit({
    "report": report,
    "meta_same": sys.meta_path is _orig_meta,
    "opened": sorted(_opened),
    "visited": sorted(_visited),
    "violations": list(_violations),
    "calls": calls,
    "values": values,
    "cuda_is_initialized": sys.modules["torch"].cuda.is_initialized(),
})
"""


# --------------------------------------------------------------------------
# Positive boundary test
# --------------------------------------------------------------------------


def test_positive_runtime_inspection_authenticated(tmp_path):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    proc = _run_child(_pin(_POSITIVE_BODY, fixture["catalog_sha256"]), fixture)
    assert proc.returncode == 0, proc.stderr
    payload = _payload(proc)
    report = payload["report"]

    assert set(report) == REPORT_KEYS
    assert report["schema_version"] == REPORT_SCHEMA
    assert report["execution_authorized"] is False
    assert report["live_runtime_accepted"] is False
    assert report["scientific_execution_performed"] is False
    assert report["external_dependencies_authenticated"] is False
    assert report["project_sources_verified"] is True
    assert report["project_modules_loaded_from_authenticated_bytes"] is True
    assert report["cuda_initialized"] is False
    assert report["catalog_sha256"] == fixture["catalog_sha256"]
    assert report["source_revision"] == "a" * 40
    assert report["verified_source_count"] == fixture["module_count"]
    assert report["loaded_project_module_count"] == fixture["module_count"]
    assert report["required_import_count"] == len(REQUIRED_IMPORTS)

    unhashed = {key: value for key, value in report.items() if key != "report_sha256"}
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert report["report_sha256"] == hashlib.sha256(canonical).hexdigest()

    assert payload["meta_same"] is True
    assert payload["cuda_is_initialized"] is False
    assert payload["opened"] == []
    assert payload["violations"] == []
    assert sorted(payload["visited"]) == fixture["source_paths"]
    for module, calls in payload["calls"].items():
        assert calls == [], module
    for module, value in payload["values"].items():
        assert value == module


# --------------------------------------------------------------------------
# Catalog and source authentication failures
# --------------------------------------------------------------------------


def test_catalog_hash_mismatch_precedes_parse_and_import(tmp_path):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    fixture["catalog_path"].write_bytes(
        b'{"schema_version": "nato-sers-p08-runtime-source-catalog-v1"'
    )
    proc = _run_child(_failure_body("0" * 64), fixture)
    payload = _payload(proc)
    assert payload["error"] == "catalog_hash_mismatch"
    assert payload["project_modules"] == 0


@pytest.mark.parametrize(
    "tamper, reason",
    [
        ("length", "source_size_mismatch"),
        ("same_size", "source_hash_mismatch"),
    ],
)
def test_later_source_tamper_blocks_first_project_import(tmp_path, tamper, reason):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    last = fixture["sources"][-1]
    target = fixture["pkg_root"] / last["relative_path"]
    original = target.read_bytes()
    if tamper == "length":
        target.write_bytes(original + b"\n")
    else:
        poisoned = b"#" + original[1:]
        assert len(poisoned) == len(original)
        target.write_bytes(poisoned)
    proc = _run_child(_failure_body(fixture["catalog_sha256"]), fixture)
    payload = _payload(proc)
    assert payload["error"] == reason
    assert payload["project_modules"] == 0


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_unknown_catalog_key,
        _mutate_unknown_entry_key,
        _mutate_wrong_roots,
        _mutate_boolean_size,
        _mutate_oversize_source,
        _mutate_too_many_entries,
        _mutate_invalid_relative_path,
    ],
    ids=[
        "unknown-catalog-key",
        "unknown-entry-key",
        "wrong-roots",
        "boolean-size",
        "oversize-source",
        "too-many-entries",
        "invalid-relative-path",
    ],
)
def test_catalog_schema_and_resource_bounds_rejected(tmp_path, mutate):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    pin = _repin_catalog(fixture, mutate)
    proc = _run_child(_failure_body(pin), fixture)
    payload = _payload(proc)
    assert payload["type"] == "RuntimeAuditError"
    assert payload["error"] is not None
    assert payload["project_modules"] == 0


def test_unlisted_project_module_has_no_discovery_fallback(tmp_path):
    specs = _variant_specs(
        "atlas_sers",
        _standard_specs()["atlas_sers"][2] + "from . import unlisted\n",
    )
    fixture = _write_fixture(tmp_path.resolve(), specs)
    unlisted = fixture["pkg_root"] / "src/atlas_sers/unlisted.py"
    unlisted.write_text(
        "import builtins\nbuiltins.__P08_UNLISTED__ = True\nVALUE = 'unlisted'\n",
        encoding="utf-8",
    )
    body = _pin(
        r"""
        import builtins
        audit.EXPECTED_CATALOG_SHA256 = __PIN__
        _orig_meta = sys.meta_path
        try:
            audit.inspect_runtime(PKG_ROOT)
        except BaseException as exc:
            _emit({
                "error": _reason(exc),
                "type": type(exc).__name__,
                "unlisted_loaded": "atlas_sers.unlisted" in sys.modules,
                "marker": hasattr(builtins, "__P08_UNLISTED__"),
                "meta_same": sys.meta_path is _orig_meta,
            })
        else:
            _emit({"error": None, "type": None})
        """,
        fixture["catalog_sha256"],
    )
    proc = _run_child(body, fixture)
    payload = _payload(proc)
    assert payload["type"] == "ImportError"
    assert payload["unlisted_loaded"] is False
    assert payload["marker"] is False
    assert payload["meta_same"] is True


# --------------------------------------------------------------------------
# Interpreter-state guards
# --------------------------------------------------------------------------


def test_preexisting_project_modules_rejected_and_preserved(tmp_path):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    body = _pin(
        r"""
        import types
        audit.EXPECTED_CATALOG_SHA256 = __PIN__
        root = types.ModuleType("atlas_sers")
        sys.modules["atlas_sers"] = root
        try:
            audit.inspect_runtime(PKG_ROOT)
        except BaseException as exc:
            first = {"error": _reason(exc), "type": type(exc).__name__,
                     "same": sys.modules.get("atlas_sers") is root}
        else:
            first = {"error": None}
        del sys.modules["atlas_sers"]
        sub = types.ModuleType("atlas_sers.evaluation.p03_runtime")
        sys.modules["atlas_sers.evaluation.p03_runtime"] = sub
        try:
            audit.inspect_runtime(PKG_ROOT)
        except BaseException as exc:
            second = {"error": _reason(exc), "type": type(exc).__name__,
                      "same": sys.modules.get("atlas_sers.evaluation.p03_runtime") is sub}
        else:
            second = {"error": None}
        _emit({"first": first, "second": second})
        """,
        fixture["catalog_sha256"],
    )
    proc = _run_child(body, fixture)
    payload = _payload(proc)
    assert payload["first"]["error"] == "preexisting_project_modules"
    assert payload["first"]["same"] is True
    assert payload["second"]["error"] == "preexisting_project_modules"
    assert payload["second"]["same"] is True


def test_not_isolated_when_dash_i_missing(tmp_path):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    proc = _run_child(_failure_body(fixture["catalog_sha256"]), fixture, isolated=False)
    payload = _payload(proc)
    assert payload["error"] == "not_isolated"


def test_bytecode_writes_enabled_when_dash_b_missing(tmp_path):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    proc = _run_child(_failure_body(fixture["catalog_sha256"]), fixture, no_bytecode=False)
    payload = _payload(proc)
    assert payload["error"] == "bytecode_writes_enabled"


def test_cuda_initialized_refused_without_other_probes(tmp_path):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    proc = _run_child(
        _failure_body(fixture["catalog_sha256"]),
        fixture,
        env={"P08_FAKE_CUDA_INIT": "1"},
    )
    payload = _payload(proc)
    assert payload["error"] == "cuda_initialized"


# --------------------------------------------------------------------------
# Descriptor-walk link refusals
# --------------------------------------------------------------------------


def test_source_symlink_parent_dir_symlink_and_hardlink_refused(tmp_path):
    for mode in ("final_symlink", "parent_dir_symlink", "hardlink"):
        base = tmp_path.resolve() / mode
        base.mkdir()
        fixture = _write_fixture(base, _standard_specs())
        leaf = fixture["pkg_root"] / "src/atlas_sers/models/classical.py"
        if mode == "final_symlink":
            real = leaf.parent / "classical_real.py"
            real.write_bytes(leaf.read_bytes())
            leaf.unlink()
            leaf.symlink_to(real)
        elif mode == "parent_dir_symlink":
            models = leaf.parent
            real_models = models.parent / "models_real"
            models.rename(real_models)
            models.symlink_to(real_models, target_is_directory=True)
        else:
            os.link(str(leaf), str(leaf.parent / "classical_hard.py"))
        proc = _run_child(_failure_body(fixture["catalog_sha256"]), fixture)
        payload = _payload(proc)
        assert payload["error"] is not None, mode
        assert payload["project_modules"] == 0, mode


def test_catalog_symlink_refused(tmp_path):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    catalog = fixture["catalog_path"]
    real = catalog.with_name("real_catalog.json")
    real.write_bytes(catalog.read_bytes())
    catalog.unlink()
    catalog.symlink_to(real)
    proc = _run_child(_failure_body("0" * 64), fixture)
    payload = _payload(proc)
    assert payload["error"] is not None
    assert payload["project_modules"] == 0


# --------------------------------------------------------------------------
# Cache, meta_path and signal/error boundaries
# --------------------------------------------------------------------------


def test_stale_bytecode_cache_is_not_used(tmp_path):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    leaf = fixture["pkg_root"] / "src/atlas_sers/models/classical.py"
    authentic = leaf.read_bytes()
    info = leaf.stat()
    prefix = b'VALUE = "POISON"\n'
    assert len(authentic) >= len(prefix)
    poisoned = prefix + b"#" * (len(authentic) - len(prefix))
    assert len(poisoned) == len(authentic)
    leaf.write_bytes(poisoned)
    os.utime(leaf, ns=(info.st_atime_ns, info.st_mtime_ns))
    pyc = Path(importlib.util.cache_from_source(str(leaf)))
    import py_compile

    py_compile.compile(str(leaf), cfile=str(pyc), doraise=True)
    assert pyc.exists()
    leaf.write_bytes(authentic)
    os.utime(leaf, ns=(info.st_atime_ns, info.st_mtime_ns))

    proc = _run_child(_pin(_POSITIVE_BODY, fixture["catalog_sha256"]), fixture)
    assert proc.returncode == 0, proc.stderr
    payload = _payload(proc)
    assert payload["values"]["atlas_sers.models.classical"] == "atlas_sers.models.classical"


def test_meta_path_restored_after_import_failure(tmp_path):
    specs = _variant_specs(
        "atlas_sers.evaluation.p03_runtime",
        'raise RuntimeError("fixture import failure")\n',
    )
    fixture = _write_fixture(tmp_path.resolve(), specs)
    body = _pin(
        r"""
        audit.EXPECTED_CATALOG_SHA256 = __PIN__
        _orig_meta = sys.meta_path
        try:
            audit.inspect_runtime(PKG_ROOT)
        except BaseException as exc:
            _emit({"error": _reason(exc), "type": type(exc).__name__,
                   "meta_same": sys.meta_path is _orig_meta})
        else:
            _emit({"error": None, "type": None})
        """,
        fixture["catalog_sha256"],
    )
    proc = _run_child(body, fixture)
    payload = _payload(proc)
    assert payload["type"] == "RuntimeError"
    assert payload["meta_same"] is True
    assert "report" not in payload


def test_loaded_module_object_replacement_rejected(tmp_path):
    replacement = (
        textwrap.dedent(
            """
        import sys
        import types

        VALUE = "atlas_sers.models.acquisition"

        _replacement = types.ModuleType(__name__)
        _replacement.__dict__.update(globals())
        sys.modules[__name__] = _replacement
        """
        ).strip()
        + "\n"
    )
    specs = _variant_specs("atlas_sers.models.acquisition", replacement)
    fixture = _write_fixture(tmp_path.resolve(), specs)
    proc = _run_child(_failure_body(fixture["catalog_sha256"]), fixture)
    payload = _payload(proc)
    assert payload["type"] == "RuntimeAuditError"
    assert payload["error"] == "module_object_mismatch"


def test_keyboard_interrupt_propagates_and_restores_meta_path(tmp_path):
    specs = _variant_specs(
        "atlas_sers.evaluation.p08_u0_runtime_inputs",
        'raise KeyboardInterrupt("fixture interrupt")\n',
    )
    fixture = _write_fixture(tmp_path.resolve(), specs)
    body = _pin(
        r"""
        audit.EXPECTED_CATALOG_SHA256 = __PIN__
        _orig_meta = sys.meta_path
        try:
            audit.inspect_runtime(PKG_ROOT)
        except KeyboardInterrupt:
            _emit({"propagated": True,
                   "meta_same": sys.meta_path is _orig_meta,
                   "report": None})
        except BaseException as exc:
            _emit({"propagated": False, "type": type(exc).__name__})
        else:
            _emit({"propagated": False, "type": "returned"})
        """,
        fixture["catalog_sha256"],
    )
    proc = _run_child(body, fixture)
    payload = _payload(proc)
    assert payload["propagated"] is True
    assert payload["meta_same"] is True
    assert payload["report"] is None


# --------------------------------------------------------------------------
# Static parser errors and main() boundary secrecy
# --------------------------------------------------------------------------


def test_duplicate_keys_and_non_finite_catalog_rejected_statically(tmp_path):
    base = tmp_path.resolve() / "duplicate"
    base.mkdir()
    fixture = _write_fixture(base, _standard_specs())
    raw = fixture["catalog_bytes"]
    index = raw.index(b'"namespace":')
    duplicate = raw[:index] + b'"namespace":"zzz",' + raw[index:]
    fixture["catalog_path"].write_bytes(duplicate)
    pin = hashlib.sha256(duplicate).hexdigest()
    proc = _run_child(_failure_body(pin), fixture)
    payload = _payload(proc)
    assert payload["error"] == "catalog_duplicate_key"
    assert payload["project_modules"] == 0

    base = tmp_path.resolve() / "nonfinite"
    base.mkdir()
    fixture = _write_fixture(base, _standard_specs())
    raw = fixture["catalog_bytes"]
    nonfinite = raw.replace(b'"execution_authorized":false', b'"execution_authorized":NaN')
    assert nonfinite != raw
    fixture["catalog_path"].write_bytes(nonfinite)
    pin = hashlib.sha256(nonfinite).hexdigest()
    proc = _run_child(_failure_body(pin), fixture)
    payload = _payload(proc)
    assert payload["error"] == "catalog_non_finite"
    assert payload["project_modules"] == 0


def test_main_does_not_leak_external_reason_marker(tmp_path):
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    body = _pin(
        r"""
        import io
        from contextlib import redirect_stdout

        class _Weird(Exception):
            reason_code = "private-marker"

        def _boom(_root):
            raise _Weird("boom")

        audit.inspect_runtime = _boom
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = audit.main(["--package-root", PKG_ROOT])
        lines = buffer.getvalue().strip().splitlines()
        emitted = json.loads(lines[-1]) if lines else {}
        _emit({"rc": rc, "emitted": emitted, "raw": buffer.getvalue()})
        """,
        fixture["catalog_sha256"],
    )
    proc = _run_child(body, fixture)
    payload = _payload(proc)
    assert payload["rc"] == 2
    assert payload["emitted"]["error"] == "runtime_audit_failed"
    assert payload["emitted"]["execution_authorized"] is False
    assert "private-marker" not in json.dumps(payload["emitted"])
    assert "private-marker" not in payload["raw"]


def test_captured_buffers_survive_fixture_disk_fault_injection(tmp_path):
    # Explicit test-only fault injection: poison a synthetic fixture source on
    # disk after capture.  No real SERS source, metadata or data is touched and
    # no production write path is exercised.
    fixture = _write_fixture(tmp_path.resolve(), _standard_specs())
    body = _pin(
        r"""
        audit.EXPECTED_CATALOG_SHA256 = __PIN__
        _target = os.path.join(
            PKG_ROOT, "src", "atlas_sers", "models", "acquisition.py"
        )
        _original = audit._load_verified_sources
        _captured = {}

        def _wrapper(root, bindings):
            verified = _original(root, bindings)
            data, origin, is_pkg = verified["atlas_sers.models.acquisition"]
            _captured["bytes"] = data
            with open(_target, "w", encoding="utf-8") as handle:
                handle.write('VALUE = "POISONED-ON-DISK"\n')
            return verified

        audit._load_verified_sources = _wrapper
        report = audit.inspect_runtime(PKG_ROOT)
        with open(_target, encoding="utf-8") as handle:
            disk = handle.read()
        _emit({
            "report": report,
            "captured": _captured["bytes"].decode("utf-8"),
            "disk": disk,
            "acquisition_value": sys.modules["atlas_sers.models.acquisition"].VALUE,
        })
        """,
        fixture["catalog_sha256"],
    )
    proc = _run_child(body, fixture)
    payload = _payload(proc)
    report = payload["report"]
    assert payload["captured"] == "VALUE = 'atlas_sers.models.acquisition'\n"
    assert payload["acquisition_value"] == "atlas_sers.models.acquisition"
    assert payload["disk"] == 'VALUE = "POISONED-ON-DISK"\n'
    for key in (
        "execution_authorized",
        "live_runtime_accepted",
        "scientific_execution_performed",
        "external_dependencies_authenticated",
        "cuda_initialized",
    ):
        assert report[key] is False, key
    assert report["project_sources_verified"] is True
    assert report["project_modules_loaded_from_authenticated_bytes"] is True
