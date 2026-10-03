#!/usr/bin/env python3
"""P08-T131 fixed-source fresh-process U0 runtime import audit.

Connects reviewer-authenticated catalog bytes to the project modules that a
fresh, isolated interpreter actually imports.  The authenticated ``src``
buffers are compiled and executed through a private finder/loader; no project
bytecode cache, discovery, or fallback path is consulted.  The audit never
reads SERS metadata/arrays, never calls fit/predict/Session construction,
never initializes CUDA, never spawns processes, and never writes files.  It
produces no scientific permit: a future launcher must re-run the fresh guard
in its own process and cannot inherit authority from this report.

Trust limits: trusted interpreter, stdlib, and external dependencies are
assumed.  This is not a malicious-interpreter proof.  Project modules are
executed as ordinary reviewed imports, not as arbitrary code execution, and
every project buffer is verified before the first project import.

Usage:
    python -I -B path/to/scripts/check_p08_u0_runtime.py \
        --package-root /absolute/path/to/research/atlas_sers
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import keyword
import os
import stat
import sys
import types

EXPECTED_CATALOG_SHA256 = "6b7a04bcef77b012b07a5757df2e76e3cca336601e96caf73244ad85fbc18f1f"
CATALOG_RELATIVE_PATH = "plan/contracts/p08_runtime_source_catalog.json"
EXPECTED_CATALOG_SCHEMA = "nato-sers-p08-runtime-source-catalog-v1"
REPORT_SCHEMA = "nato-sers-p08-u0-runtime-import-audit-v1"
NAMESPACE = "atlas_sers"
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
EXPECTED_CALLABLE_SURFACE = {
    "atlas_sers.evaluation.p03_runtime": {"run_candidate_fit": "function"},
    "atlas_sers.evaluation.p05_development": {"train_development_fit": "function"},
    "atlas_sers.evaluation.p08_u0_session": {"_SourceSession": "type"},
    "atlas_sers.evaluation.p08_u0_runtime_inputs": {"prepare_u0_runtime_inputs": "function"},
}
MAX_CATALOG_BYTES = 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 32 * 1024 * 1024
MAX_SINGLE_SOURCE_BYTES = 2 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_CATALOG_KEYS = frozenset(
    {
        "schema_version",
        "$id",
        "execution_authorized",
        "namespace",
        "source_revision",
        "import_roots",
        "sources",
    }
)
_SOURCE_ENTRY_KEYS = frozenset({"module", "relative_path", "is_package", "sha256", "size_bytes"})


class RuntimeAuditError(ValueError):
    """Static, path-free audit failure carrying only a reason code."""

    def __init__(self, reason_code):
        super().__init__(reason_code)
        self.reason_code = reason_code


def _is_project_module(name):
    return name == NAMESPACE or name.startswith(NAMESPACE + ".")


def _reject_constant(name):
    raise RuntimeAuditError("catalog_non_finite")


def _no_duplicate_keys(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise RuntimeAuditError("catalog_duplicate_key")
        seen[key] = value
    return seen


def _require_directory_walk_support():
    """Reject missing filesystem flags statically before any read."""
    for flag_name in (
        "O_RDONLY",
        "O_NOFOLLOW",
        "O_DIRECTORY",
        "O_CLOEXEC",
        "O_NONBLOCK",
    ):
        if not hasattr(os, flag_name):
            raise RuntimeAuditError("filesystem_flags_unsupported")
    supports = getattr(os, "supports_dir_fd", None)
    if supports is None or os.open not in supports:
        raise RuntimeAuditError("filesystem_flags_unsupported")


def _open_no_symlink_absolute(path, reason_prefix):
    """Descriptor-walk an absolute path, rejecting any symlinked component."""
    if not isinstance(path, str) or not path or path[0] != "/":
        raise RuntimeAuditError(reason_prefix + "_path_invalid")
    parts = [part for part in path.split("/") if part]
    if not parts or any(part in (".", "..") for part in parts):
        raise RuntimeAuditError(reason_prefix + "_path_invalid")
    try:
        current = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        raise RuntimeAuditError(reason_prefix + "_open_failed") from None
    try:
        for index, part in enumerate(parts):
            if index == len(parts) - 1:
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
            else:
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            try:
                next_descriptor = os.open(part, flags, dir_fd=current)
            except OSError:
                raise RuntimeAuditError(reason_prefix + "_open_failed") from None
            previous = current
            current = next_descriptor
            try:
                os.close(previous)
            except OSError:
                raise RuntimeAuditError(reason_prefix + "_open_failed") from None
    except BaseException:
        os.close(current)
        raise
    return current


def _read_verified_file(path, max_bytes, reason_prefix):
    """Read one regular, single-link file with a bounded, stable fd read."""
    descriptor = _open_no_symlink_absolute(path, reason_prefix)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeAuditError(reason_prefix + "_not_regular")
        if before.st_nlink != 1:
            raise RuntimeAuditError(reason_prefix + "_link_count")
        if before.st_size > max_bytes:
            raise RuntimeAuditError(reason_prefix + "_too_large")
        chunks = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        if remaining <= 0:
            raise RuntimeAuditError(reason_prefix + "_too_large")
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size, after.st_nlink) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_nlink,
        ):
            raise RuntimeAuditError(reason_prefix + "_unstable")
        if len(data) != before.st_size:
            raise RuntimeAuditError(reason_prefix + "_length_mismatch")
    finally:
        os.close(descriptor)
    return data


def _canonical_package_root(package_root):
    if not isinstance(package_root, str) or not package_root:
        raise RuntimeAuditError("package_root_invalid")
    if not os.path.isabs(package_root):
        raise RuntimeAuditError("package_root_not_absolute")
    normalized = os.path.normpath(package_root)
    if normalized != package_root:
        raise RuntimeAuditError("package_root_not_canonical")
    if os.path.realpath(package_root) != normalized:
        raise RuntimeAuditError("package_root_symlink")
    try:
        info = os.stat(normalized)
    except OSError:
        raise RuntimeAuditError("package_root_unavailable") from None
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeAuditError("package_root_not_directory")
    return normalized


def _load_catalog(package_root):
    catalog_path = os.path.join(package_root, *CATALOG_RELATIVE_PATH.split("/"))
    if os.path.realpath(catalog_path) != os.path.normpath(catalog_path):
        raise RuntimeAuditError("catalog_path_symlink")
    raw = _read_verified_file(catalog_path, MAX_CATALOG_BYTES, "catalog")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_CATALOG_SHA256:
        raise RuntimeAuditError("catalog_hash_mismatch")
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeAuditError("catalog_parse_failed") from None
    return document, digest


def _validate_catalog(document):
    if not isinstance(document, dict):
        raise RuntimeAuditError("catalog_schema_invalid")
    if not set(document).issubset(_CATALOG_KEYS):
        raise RuntimeAuditError("catalog_unknown_key")
    if document.get("schema_version") != EXPECTED_CATALOG_SCHEMA:
        raise RuntimeAuditError("catalog_schema_version")
    if document.get("$id") != EXPECTED_CATALOG_SCHEMA:
        raise RuntimeAuditError("catalog_identifier_invalid")
    if document.get("execution_authorized") is not False:
        raise RuntimeAuditError("catalog_execution_authorized")
    if document.get("namespace") != NAMESPACE:
        raise RuntimeAuditError("catalog_namespace_invalid")
    revision = document.get("source_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(ch not in _HEX for ch in revision)
    ):
        raise RuntimeAuditError("catalog_revision_invalid")
    roots = document.get("import_roots")
    if not isinstance(roots, list) or roots != list(REQUIRED_IMPORTS):
        raise RuntimeAuditError("catalog_import_roots_invalid")
    sources = document.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 256:
        raise RuntimeAuditError("catalog_source_count")
    return revision, sources


def _module_from_relative_path(relative_path):
    if not isinstance(relative_path, str) or not relative_path:
        raise RuntimeAuditError("source_path_invalid")
    if relative_path.startswith("/") or "\\" in relative_path:
        raise RuntimeAuditError("source_path_invalid")
    parts = relative_path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise RuntimeAuditError("source_path_invalid")
    if len(parts) < 3 or parts[0] != "src" or parts[1] != NAMESPACE:
        raise RuntimeAuditError("source_path_invalid")
    if not parts[-1].endswith(".py"):
        raise RuntimeAuditError("source_path_invalid")
    is_package = parts[-1] == "__init__.py"
    stems = parts[2:-1]
    leaf = "" if is_package else parts[-1][:-3]
    identifiers = list(stems)
    if not is_package:
        identifiers.append(leaf)
    for identifier in identifiers:
        if (
            not isinstance(identifier, str)
            or not identifier.isidentifier()
            or keyword.iskeyword(identifier)
        ):
            raise RuntimeAuditError("source_path_invalid")
    if is_package:
        module = NAMESPACE if not stems else NAMESPACE + "." + ".".join(stems)
    else:
        module = NAMESPACE + "." + ".".join(stems + [leaf])
    return module, is_package


def _validate_sources(sources):
    bindings = {}
    total = 0
    for entry in sources:
        if not isinstance(entry, dict):
            raise RuntimeAuditError("source_entry_invalid")
        if not set(entry).issubset(_SOURCE_ENTRY_KEYS):
            raise RuntimeAuditError("source_entry_unknown_key")
        module = entry.get("module")
        relative_path = entry.get("relative_path")
        is_package = entry.get("is_package")
        sha = entry.get("sha256")
        size = entry.get("size_bytes")
        if not isinstance(module, str) or (
            module != NAMESPACE and not module.startswith(NAMESPACE + ".")
        ):
            raise RuntimeAuditError("source_module_invalid")
        if module in bindings:
            raise RuntimeAuditError("source_module_duplicate")
        if not isinstance(is_package, bool):
            raise RuntimeAuditError("source_package_flag_invalid")
        expected_module, path_is_package = _module_from_relative_path(relative_path)
        if expected_module != module or path_is_package != is_package:
            raise RuntimeAuditError("source_binding_mismatch")
        if not isinstance(sha, str) or len(sha) != 64 or any(ch not in _HEX for ch in sha):
            raise RuntimeAuditError("source_hash_invalid")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > MAX_SINGLE_SOURCE_BYTES
        ):
            raise RuntimeAuditError("source_size_invalid")
        total += size
        if total > MAX_TOTAL_SOURCE_BYTES:
            raise RuntimeAuditError("source_total_too_large")
        bindings[module] = (relative_path, is_package, sha, size)
    if list(bindings) != sorted(bindings):
        raise RuntimeAuditError("source_module_order")
    for module in bindings:
        if module == NAMESPACE:
            continue
        parent = module.rsplit(".", 1)[0]
        if parent != NAMESPACE and parent not in bindings:
            raise RuntimeAuditError("source_parent_missing")
    return bindings


def _load_verified_sources(package_root, bindings):
    verified = {}
    for module in sorted(bindings):
        relative_path, is_package, sha, size = bindings[module]
        absolute = os.path.join(package_root, *relative_path.split("/"))
        if os.path.realpath(absolute) != os.path.normpath(absolute):
            raise RuntimeAuditError("source_path_symlink")
        data = _read_verified_file(absolute, MAX_SINGLE_SOURCE_BYTES, "source")
        if len(data) != size:
            raise RuntimeAuditError("source_size_mismatch")
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeAuditError("source_hash_mismatch")
        verified[module] = (data, absolute, is_package)
    return verified


class _RuntimeFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Meta path finder and loader over authenticated in-memory buffers."""

    def __init__(self, verified):
        self._verified = verified
        self._attempted = set()
        self._executed = {}

    def find_spec(self, fullname, path=None, target=None):
        if not _is_project_module(fullname):
            return None
        entry = self._verified.get(fullname)
        if entry is None:
            raise ImportError("unauthenticated project module")
        _, origin, is_package = entry
        spec = importlib.util.spec_from_loader(fullname, self, origin=origin, is_package=is_package)
        if spec is None:
            raise ImportError("spec construction failed")
        spec.has_location = True
        if is_package:
            spec.submodule_search_locations = [os.path.dirname(origin)]
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        if name in self._attempted:
            raise ImportError("duplicate execution")
        entry = self._verified.get(name)
        if entry is None:
            raise ImportError("unverified module")
        data, origin, _ = entry
        spec = getattr(module, "__spec__", None)
        if spec is None or spec.origin != origin:
            raise ImportError("spec origin mismatch")
        self._attempted.add(name)
        code = compile(data, origin, "exec", dont_inherit=True)
        exec(code, module.__dict__)
        self._executed[name] = module


def _verify_loaded_modules(finder, verified):
    present = {name for name in sys.modules if _is_project_module(name)}
    if present != set(finder._executed):
        raise RuntimeAuditError("module_namespace_mismatch")
    for name, module in finder._executed.items():
        if sys.modules.get(name) is not module:
            raise RuntimeAuditError("module_object_mismatch")
        spec = getattr(module, "__spec__", None)
        if (
            spec is None
            or spec.loader is not finder
            or getattr(module, "__loader__", None) is not finder
        ):
            raise RuntimeAuditError("module_loader_mismatch")
        entry = verified.get(name)
        if entry is None:
            raise RuntimeAuditError("module_unverified")
        _, expected_origin, is_package = entry
        if spec.origin != expected_origin:
            raise RuntimeAuditError("module_origin_mismatch")
        if getattr(module, "__file__", None) != expected_origin:
            raise RuntimeAuditError("module_file_mismatch")
        if is_package:
            locations = getattr(spec, "submodule_search_locations", None)
            expected_dir = os.path.dirname(expected_origin)
            if locations is None or list(locations) != [expected_dir]:
                raise RuntimeAuditError("module_package_path_mismatch")
    return len(finder._executed)


def _verify_callable_surface():
    for module_name, required in EXPECTED_CALLABLE_SURFACE.items():
        module = sys.modules.get(module_name)
        namespace = getattr(module, "__dict__", None) if module is not None else None
        if not isinstance(namespace, dict):
            raise RuntimeAuditError("callable_surface_incomplete")
        for attribute, kind in required.items():
            value = namespace.get(attribute)
            if kind == "function":
                if not isinstance(value, types.FunctionType):
                    raise RuntimeAuditError("callable_surface_incomplete")
            elif kind == "type":
                if not isinstance(value, type):
                    raise RuntimeAuditError("callable_surface_incomplete")
            else:
                raise RuntimeAuditError("callable_surface_incomplete")


def _cuda_initialized_flag():
    torch = sys.modules.get("torch")
    if torch is None:
        raise RuntimeAuditError("torch_module_missing")
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        raise RuntimeAuditError("torch_cuda_missing")
    probe = getattr(cuda, "is_initialized", None)
    if not callable(probe):
        raise RuntimeAuditError("cuda_probe_unavailable")
    if bool(probe()):
        raise RuntimeAuditError("cuda_initialized")
    return False


def inspect_runtime(package_root):
    """Verify all catalog bytes, then import the fixed roots in-process."""
    if sys.flags.isolated != 1:
        raise RuntimeAuditError("not_isolated")
    if not sys.dont_write_bytecode:
        raise RuntimeAuditError("bytecode_writes_enabled")
    for name in list(sys.modules):
        if _is_project_module(name):
            raise RuntimeAuditError("preexisting_project_modules")
    _require_directory_walk_support()
    root = _canonical_package_root(package_root)
    document, catalog_digest = _load_catalog(root)
    revision, sources = _validate_catalog(document)
    bindings = _validate_sources(sources)
    verified = _load_verified_sources(root, bindings)
    finder = _RuntimeFinder(verified)
    original_meta_path = sys.meta_path
    sys.meta_path = [finder] + list(original_meta_path)
    try:
        for name in REQUIRED_IMPORTS:
            importlib.import_module(name)
        loaded_count = _verify_loaded_modules(finder, verified)
        _verify_callable_surface()
    finally:
        sys.meta_path = original_meta_path
    report = {
        "schema_version": REPORT_SCHEMA,
        "execution_authorized": False,
        "live_runtime_accepted": False,
        "scientific_execution_performed": False,
        "external_dependencies_authenticated": False,
        "project_sources_verified": True,
        "project_modules_loaded_from_authenticated_bytes": True,
        "catalog_sha256": catalog_digest,
        "source_revision": revision,
        "verified_source_count": len(verified),
        "loaded_project_module_count": loaded_count,
        "required_import_count": len(REQUIRED_IMPORTS),
        "cuda_initialized": _cuda_initialized_flag(),
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def _emit(payload):
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--package-root", required=True)
    args = parser.parse_args(argv)
    try:
        if sys.flags.isolated != 1:
            raise RuntimeAuditError("not_isolated")
        if not sys.dont_write_bytecode:
            raise RuntimeAuditError("bytecode_writes_enabled")
        report = inspect_runtime(args.package_root)
    except RuntimeAuditError as exc:
        _emit(
            {
                "schema_version": REPORT_SCHEMA,
                "execution_authorized": False,
                "error": exc.reason_code,
            }
        )
        return 2
    except Exception:  # noqa: BLE001 - static reason code only
        _emit(
            {
                "schema_version": REPORT_SCHEMA,
                "execution_authorized": False,
                "error": "runtime_audit_failed",
            }
        )
        return 2
    _emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
