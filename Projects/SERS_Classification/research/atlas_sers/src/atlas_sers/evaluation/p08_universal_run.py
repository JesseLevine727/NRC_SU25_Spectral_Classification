"""T323 compact PRIVATE final analysis child and faithful result bundle.

This child performs no fitting, calibration, model/policy selection,
preprocessing, GPU work, networking, subprocess execution or plotting.  It
orchestrates three already-authenticated universal modules, validates the
declared production dimensions of the already-computed results, and stores a
faithful serialization of those results.

The serialization codec is intentionally small and fixed.  It never uses
pickle, never deserializes arbitrary Python objects and never imports code
whose name appears inside a payload.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import platform
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from atlas_sers.evaluation.p08_universal_analysis import analyze_panel
from atlas_sers.evaluation.p08_universal_evidence import load_evidence
from atlas_sers.evaluation.p08_universal_panel import assemble_panel

RUN_VERSION = "T323-1"
BUNDLE_VERSION = "T323-bundle-1"
BUNDLE_KIND = "p08_universal_bundle"

EXPECTED_REPORT_CELLS = 3900
EXPECTED_DISTINCT_ENDPOINTS = 3237
EXPECTED_GLOBAL_MASTERS = 69
EXPECTED_GLOBAL_INSTRUMENTS = 10
EXPECTED_DRAW_COUNT = 10000
EXPECTED_MASTER_SEED = 2026093001
EXPECTED_INSTRUMENT_SEED = 2026093002
EXPECTED_HIERARCHY_SEED = 2026093003

BOUNDARY_FLAGS = (
    "conditional_on_saved_fits_observed_support",
    "descriptive_sign_symmetry_not_randomized",
    "no_G4_decision",
    "no_model_or_policy_selection",
)

CHECK_INITIAL = "guard:initial"
CHECK_STAGE_LOAD = "guard:stage:load"
CHECK_STAGE_ASSEMBLE = "guard:stage:assemble"
CHECK_STAGE_ANALYZE = "guard:stage:analyze"
CHECK_STAGE_ANALYZE_DONE = "guard:stage:analyze:done"
CHECK_BUNDLE_DONE = "guard:bundle:done"
CHECK_PAYLOAD_PREFIX = "guard:payload:"
CHECK_MANIFEST = "guard:manifest"
CHECK_RECEIPT = "guard:receipt"
CHECK_BUNDLE_INIT = "guard:bundle:init"

# Specific protection roots only.  Broad read allowances such as
# allowed_evidence_roots / private_root are deliberately NOT rejected.
COMPLETED_RUN_ROOT = None
PROTECTED_GRAPH_PATHS = ()
PROTECTED_PUBLIC_PACKAGE_ROOT = None
MODULE_SOURCE_DIR = str(Path(__file__).resolve().parent)

RECEIPT_NAME = "receipt.json"
ERROR_NAME = "errorcode.json"
BUNDLE_DIRNAME = "bundle"
MANIFEST_NAME = "manifest.json"

_SAFE_NUMPY_KINDS = frozenset("fiubUS")
_SAFE_EXTENSION_DTYPES = frozenset(
    {
        "Int64",
        "Int32",
        "Int16",
        "Int8",
        "UInt64",
        "UInt32",
        "UInt16",
        "UInt8",
        "Float64",
        "Float32",
        "boolean",
        "string",
    }
)
_SOFTWARE_MODULES = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("pyarrow", "pyarrow"),
)


class _BundleError(Exception):
    """Internal refusal; the code is stable and path-free."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


# ---------------------------------------------------------------------------
# Guard and destination safety
# ---------------------------------------------------------------------------


def _require_check(check):
    if not callable(check):
        raise TypeError("check must be a callable external guard")
    return check


def _canonical(path):
    path = Path(path)
    try:
        if path.exists() or path.is_symlink():
            return path.resolve()
    except OSError:
        pass
    return path.parent.resolve() / path.name


def _is_within(candidate, root):
    candidate = Path(candidate)
    root = Path(root)
    return candidate == root or root in candidate.parents


def _assert_no_symlink_ancestry(path):
    current = Path(path).absolute()
    parts = []
    while True:
        parts.append(current)
        if current.parent == current:
            break
        current = current.parent
    for part in reversed(parts):
        if part.is_symlink():
            raise _BundleError("symlink_ancestry")


def _protected_public_package_root():
    if PROTECTED_PUBLIC_PACKAGE_ROOT is not None:
        return Path(PROTECTED_PUBLIC_PACKAGE_ROOT)
    try:
        import atlas_sers
    except Exception:
        return None
    module_file = getattr(atlas_sers, "__file__", None)
    if not module_file:
        return None
    return Path(module_file).resolve().parents[2]


def _completed_run_root(evidence_kwargs):
    if COMPLETED_RUN_ROOT is not None:
        return Path(COMPLETED_RUN_ROOT)
    if isinstance(evidence_kwargs, Mapping):
        for key in ("run_root", "completed_run_root", "root"):
            value = evidence_kwargs.get(key)
            if isinstance(value, (str, os.PathLike)):
                return Path(value)
    return None


def _protected_roots(evidence_kwargs):
    roots = []
    run_root = _completed_run_root(evidence_kwargs)
    if run_root is not None:
        roots.append(run_root)
    for item in PROTECTED_GRAPH_PATHS:
        roots.append(Path(item))
    package = _protected_public_package_root()
    if package is not None:
        roots.append(package)
    roots.append(Path(MODULE_SOURCE_DIR))
    return roots


def _refuse_destination(output_path, evidence_kwargs):
    candidate = _canonical(output_path)
    for root in _protected_roots(evidence_kwargs):
        if root is None:
            continue
        try:
            root_canonical = _canonical(root)
        except OSError:
            continue
        if _is_within(candidate, root_canonical):
            raise _BundleError("destination_protected")


def _make_private_dir(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise _BundleError("output_exists")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise _BundleError("output_parent_invalid")
    _assert_no_symlink_ancestry(parent)
    path.mkdir(mode=0o700, exist_ok=False)
    os.chmod(path, 0o700)


def _ensure_private_dir(path):
    path = Path(path)
    if path.is_symlink():
        raise _BundleError("symlink_ancestry")
    if path.exists():
        if not path.is_dir():
            raise _BundleError("output_parent_invalid")
        return
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise _BundleError("output_parent_invalid")
    _assert_no_symlink_ancestry(parent)
    path.mkdir(mode=0o700, exist_ok=False)
    os.chmod(path, 0o700)


def _exclusive_write_bytes(path, data, mode=0o600):
    path = Path(path)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    except FileExistsError as exc:
        raise _BundleError("output_exists") from exc
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
    finally:
        if fd != -1:
            os.close(fd)


# ---------------------------------------------------------------------------
# Small fixed recursive codec
# ---------------------------------------------------------------------------


def _encode_float(value):
    if math.isnan(value):
        return {"t": "nan"}
    if math.isinf(value):
        return {"t": "pinf" if value > 0 else "ninf"}
    return {"t": "float", "v": repr(value)}


def _is_extension(dtype):
    return isinstance(dtype, pd.api.extensions.ExtensionDtype)


def _extension_allowed(dtype):
    if isinstance(dtype, pd.StringDtype):
        return True
    return str(dtype) in _SAFE_EXTENSION_DTYPES


def _dtype_descriptor(dtype):
    """Return a small whitelist descriptor for a supported dtype."""
    if isinstance(dtype, pd.StringDtype):
        na_value = getattr(dtype, "na_value", pd.NA)
        storage = getattr(dtype, "storage", None)
        return {
            "t": "string_dtype",
            "na": "pd.NA" if na_value is pd.NA else "nan",
            "storage": None if storage is None else str(storage),
        }
    return str(dtype)


def _is_mapping_key(key):
    if isinstance(key, str):
        return True
    if isinstance(key, tuple):
        return all(isinstance(part, str) for part in key)
    return False


def _string_dtype(na_value, storage):
    return pd.StringDtype(storage=storage, na_value=na_value)


def _safe_dtype(spec):
    if isinstance(spec, Mapping):
        if (
            set(spec) != {"t", "na", "storage"}
            or spec.get("t") != "string_dtype"
            or spec.get("na") not in {"pd.NA", "nan"}
            or spec.get("storage") not in {None, "python", "pyarrow"}
        ):
            raise _BundleError("unsupported_dtype")
        na_value = pd.NA if spec.get("na") == "pd.NA" else np.nan
        return _string_dtype(na_value, spec.get("storage"))
    if spec == "object":
        return "object"
    if spec == "string":
        return _string_dtype(pd.NA, None)
    if spec == "str":
        return _string_dtype(np.nan, None)
    if spec in _SAFE_EXTENSION_DTYPES:
        return spec
    raise _BundleError("unsupported_dtype")


def _validate_relpath(rel):
    if not isinstance(rel, str) or not rel:
        raise _BundleError("bad_relpath")
    if "\\" in rel:
        raise _BundleError("bad_relpath")
    pure = PurePosixPath(rel)
    if pure.is_absolute():
        raise _BundleError("bad_relpath")
    if any(part in ("", ".", "..") for part in pure.parts):
        raise _BundleError("bad_relpath")


class _Writer:
    def __init__(self, root, check):
        self.root = Path(root)
        self.check = check
        self.counter = 0
        self.files = {}

    def encode(self, value):
        return self._encode(value)

    def _store(self, array):
        array = np.asarray(array)
        if array.dtype.kind == "O" or array.dtype.hasobject:
            raise _BundleError("object_array_rejected")
        if array.dtype.kind not in _SAFE_NUMPY_KINDS:
            raise _BundleError("unsupported_array_dtype")
        self.counter += 1
        rel = f"payload/{self.counter:06d}.npy"
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        data = buffer.getvalue()
        self.check(CHECK_PAYLOAD_PREFIX + rel)
        target = self.root / rel
        _ensure_private_dir(target.parent)
        _exclusive_write_bytes(target, data)
        self.files[rel] = {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        nonfinite = 0
        if array.dtype.kind in ("f", "c"):
            nonfinite = int(np.count_nonzero(~np.isfinite(array)))
        return {
            "t": "ndarray",
            "file": rel,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "nonfinite": nonfinite,
        }

    def _encode(self, value):
        if value is None:
            return {"t": "none"}
        if value is pd.NA:
            return {"t": "pd.NA"}
        if value is pd.NaT:
            return {"t": "pd.NaT"}
        if isinstance(value, (bool, np.bool_)):
            return {"t": "bool", "v": bool(value)}
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return {"t": "int", "v": str(int(value))}
        if isinstance(value, (float, np.floating)):
            return _encode_float(float(value))
        if isinstance(value, (str, np.str_)):
            return {"t": "str", "v": str(value)}
        if isinstance(value, np.ndarray):
            return self._store(value)
        if isinstance(value, tuple):
            return {"t": "tuple", "items": [self._encode(v) for v in value]}
        if isinstance(value, list):
            return {"t": "list", "items": [self._encode(v) for v in value]}
        if isinstance(value, Mapping):
            keys = list(value.keys())
            if all(isinstance(key, str) for key in keys):
                return {
                    "t": "dict",
                    "items": {key: self._encode(item) for key, item in value.items()},
                }
            if all(_is_mapping_key(key) for key in keys):
                return {
                    "t": "dict",
                    "items": [
                        [self._encode(key), self._encode(item)] for key, item in value.items()
                    ],
                }
            raise _BundleError("dict_nonstring_key")
        if isinstance(value, pd.DataFrame):
            return self._encode_frame(value)
        if isinstance(value, pd.Series):
            return self._encode_series(value)
        raise _BundleError("unsupported_type:" + type(value).__name__)

    def _encode_frame(self, frame):
        if isinstance(frame.index, pd.MultiIndex):
            raise _BundleError("multiindex_rejected")
        columns = [self._encode(column) for column in frame.columns]
        data = []
        dtypes = []
        for position in range(frame.shape[1]):
            series = frame.iloc[:, position]
            data.append(self._encode_column(series))
            dtypes.append(_dtype_descriptor(series.dtype))
        return {
            "t": "dataframe",
            "columns": columns,
            "index": self._encode_index(frame.index),
            "index_name": self._encode(frame.index.name),
            "columns_name": self._encode(frame.columns.name),
            "dtypes": dtypes,
            "data": data,
            "shape": list(frame.shape),
        }

    def _encode_series(self, series):
        return {
            "t": "series",
            "name": self._encode(series.name),
            "index": self._encode_index(series.index),
            "dtype": _dtype_descriptor(series.dtype),
            "data": self._encode_column(series),
        }

    def _encode_index(self, index):
        if isinstance(index, pd.RangeIndex):
            return {
                "t": "range_index",
                "start": int(index.start),
                "stop": int(index.stop),
                "step": int(index.step),
                "name": self._encode(index.name),
            }
        if isinstance(index, pd.MultiIndex):
            raise _BundleError("multiindex_rejected")
        dtype = index.dtype
        if not _is_extension(dtype) and dtype.kind in _SAFE_NUMPY_KINDS:
            return {
                "t": "index",
                "name": self._encode(index.name),
                "dtype": _dtype_descriptor(dtype),
                "data": self._store(np.asarray(index.to_numpy())),
            }
        return {
            "t": "index",
            "name": self._encode(index.name),
            "dtype": _dtype_descriptor(dtype),
            "data": {"t": "list", "items": [self._encode(v) for v in index.tolist()]},
        }

    def _encode_column(self, series):
        dtype = series.dtype
        if _is_extension(dtype):
            if not _extension_allowed(dtype):
                raise _BundleError("unsupported_extension_dtype")
            return {"t": "list", "items": [self._encode(v) for v in series.tolist()]}
        if dtype.kind == "O":
            return {"t": "list", "items": [self._encode(v) for v in series.tolist()]}
        if dtype.kind in _SAFE_NUMPY_KINDS:
            return self._store(np.asarray(series.to_numpy()))
        raise _BundleError("unsupported_column_dtype")


def _safe_payload_path(root, rel):
    _validate_relpath(rel)
    base = Path(root).resolve()
    current = Path(root)
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            raise _BundleError("symlink_payload")
    path = base / PurePosixPath(rel)
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise _BundleError("payload_unreadable") from exc
    if resolved != base and base not in resolved.parents:
        raise _BundleError("path_traversal")
    if not path.is_file():
        raise _BundleError("payload_missing")
    return path


class _Reader:
    def __init__(self, root, listed):
        self.root = Path(root)
        self.listed = frozenset(listed)

    def decode(self, node):
        if not isinstance(node, Mapping):
            raise _BundleError("bad_node")
        kind = node.get("t")
        if kind == "none":
            return None
        if kind == "pd.NA":
            return pd.NA
        if kind == "pd.NaT":
            return pd.NaT
        if kind == "bool":
            return bool(node["v"])
        if kind == "int":
            return int(node["v"])
        if kind == "float":
            return float(node["v"])
        if kind == "nan":
            return float("nan")
        if kind == "pinf":
            return float("inf")
        if kind == "ninf":
            return float("-inf")
        if kind == "str":
            return str(node["v"])
        if kind == "list":
            return [self.decode(item) for item in node["items"]]
        if kind == "tuple":
            return tuple(self.decode(item) for item in node["items"])
        if kind == "dict":
            items = node["items"]
            if isinstance(items, Mapping):
                return {str(key): self.decode(item) for key, item in items.items()}
            if not isinstance(items, list):
                raise _BundleError("bad_dict")
            result = {}
            for pair in items:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise _BundleError("bad_dict")
                key = self.decode(pair[0])
                if not _is_mapping_key(key):
                    raise _BundleError("dict_bad_key")
                if key in result:
                    raise _BundleError("dict_duplicate_key")
                result[key] = self.decode(pair[1])
            return result
        if kind == "ndarray":
            return self._load_ndarray(node)
        if kind == "dataframe":
            return self._decode_frame(node)
        if kind == "series":
            return self._decode_series(node)
        if kind == "range_index":
            return pd.RangeIndex(
                int(node["start"]),
                int(node["stop"]),
                int(node["step"]),
                name=self.decode(node["name"]),
            )
        if kind == "index":
            return self._decode_index(node)
        raise _BundleError("unknown_node")

    def _load_ndarray(self, node):
        rel = node.get("file")
        if not isinstance(rel, str) or not rel.endswith(".npy") or rel not in self.listed:
            raise _BundleError("unlisted_payload")
        path = _safe_payload_path(self.root, rel)
        array = np.load(path, allow_pickle=False)
        if array.dtype.kind == "O" or array.dtype.hasobject:
            raise _BundleError("object_array_rejected")
        if str(array.dtype) != str(node["dtype"]):
            raise _BundleError("ndarray_dtype_mismatch")
        if list(array.shape) != list(node["shape"]):
            raise _BundleError("ndarray_shape_mismatch")
        if array.dtype.kind in ("f", "c"):
            nonfinite = int(np.count_nonzero(~np.isfinite(array)))
            if nonfinite != int(node["nonfinite"]):
                raise _BundleError("ndarray_nonfinite_mismatch")
        return array

    def _decode_index(self, node):
        kind = node["t"]
        name = self.decode(node.get("name"))
        if kind == "range_index":
            return pd.RangeIndex(
                int(node["start"]), int(node["stop"]), int(node["step"]), name=name
            )
        if kind == "index":
            data = node["data"]
            if data["t"] == "ndarray":
                return pd.Index(self.decode(data), name=name)
            values = [self.decode(item) for item in data["items"]]
            dtype = _safe_dtype(node["dtype"])
            if dtype == "object":
                return pd.Index(values, dtype=object, name=name, tupleize_cols=False)
            return pd.Index(pd.array(values, dtype=dtype), name=name)
        raise _BundleError("bad_index")

    def _decode_column(self, node, dtype_name):
        if node["t"] == "ndarray":
            return self.decode(node)
        if node["t"] == "list":
            values = [self.decode(item) for item in node["items"]]
            dtype = _safe_dtype(dtype_name)
            if dtype == "object":
                column = np.empty(len(values), dtype=object)
                for position, value in enumerate(values):
                    column[position] = value
                return column
            return pd.array(values, dtype=dtype)
        raise _BundleError("bad_column")

    def _decode_frame(self, node):
        index = self._decode_index(node["index"])
        columns = [self.decode(column) for column in node["columns"]]
        frame = pd.DataFrame(index=index)
        for position, item in enumerate(node["data"]):
            values = self._decode_column(item, node["dtypes"][position])
            frame.insert(position, columns[position], values, allow_duplicates=True)
        frame.index.name = self.decode(node.get("index_name"))
        frame.columns.name = self.decode(node.get("columns_name"))
        return frame

    def _decode_series(self, node):
        index = self._decode_index(node["index"])
        values = self._decode_column(node["data"], node["dtype"])
        return pd.Series(values, index=index, name=self.decode(node.get("name")))


def _canonical_json_bytes(value):
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ValueError as exc:
        raise _BundleError("nonfinite_json") from exc
    return text.encode("utf-8")


def _strict_json_loads(raw):
    def _reject(_constant):
        raise _BundleError("nonfinite_json")

    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject)
    except _BundleError:
        raise
    except Exception as exc:
        raise _BundleError("manifest_decode") from exc


def write_bundle(root, payload, check):
    check = _require_check(check)
    check(CHECK_BUNDLE_INIT)
    root = Path(root)
    _make_private_dir(root)
    writer = _Writer(root, check)
    encoded = writer.encode(payload)
    manifest = {
        "kind": BUNDLE_KIND,
        "version": BUNDLE_VERSION,
        "payload": encoded,
        "files": writer.files,
    }
    blob = _canonical_json_bytes(manifest)
    check(CHECK_MANIFEST)
    _exclusive_write_bytes(root / MANIFEST_NAME, blob)
    check(CHECK_BUNDLE_DONE)
    return {
        "manifest_sha256": hashlib.sha256(blob).hexdigest(),
        "files": dict(writer.files),
        "counts": {"num_files": len(writer.files)},
    }


def read_bundle(root, *, expected_manifest_sha256, check=None):
    if check is not None:
        _require_check(check)
    if not isinstance(expected_manifest_sha256, str) or not expected_manifest_sha256:
        raise _BundleError("manifest_pin_required")
    if check is not None:
        check(CHECK_MANIFEST)
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise _BundleError("bundle_root_invalid")
    _assert_no_symlink_ancestry(root)
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise _BundleError("manifest_missing")
    raw = manifest_path.read_bytes()
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_manifest_sha256):
        raise _BundleError("manifest_pin_mismatch")
    manifest = _strict_json_loads(raw)
    if not isinstance(manifest, Mapping):
        raise _BundleError("manifest_not_mapping")
    if manifest.get("kind") != BUNDLE_KIND or manifest.get("version") != BUNDLE_VERSION:
        raise _BundleError("manifest_identity")
    files = manifest.get("files")
    descriptor = manifest.get("payload")
    if not isinstance(files, Mapping) or not isinstance(descriptor, Mapping):
        raise _BundleError("manifest_shape")
    listed = set()
    for rel, meta in files.items():
        _validate_relpath(rel)
        if not rel.endswith(".npy"):
            raise _BundleError("manifest_file_entry")
        if not isinstance(meta, Mapping):
            raise _BundleError("manifest_file_entry")
        if not isinstance(meta.get("size"), int) or not isinstance(meta.get("sha256"), str):
            raise _BundleError("manifest_file_entry")
        listed.add(rel)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise _BundleError("symlink_inventory")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != listed | {MANIFEST_NAME}:
        raise _BundleError("inventory_mismatch")
    for rel, meta in files.items():
        if check is not None:
            check(CHECK_PAYLOAD_PREFIX + rel)
        path = _safe_payload_path(root, rel)
        data = path.read_bytes()
        if len(data) != meta["size"]:
            raise _BundleError("size_mismatch")
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise _BundleError("hash_mismatch")
    if check is not None:
        check("guard:bundle:decode")
    reader = _Reader(root, listed)
    return reader.decode(descriptor)


# ---------------------------------------------------------------------------
# Validation of declared production dimensions
# ---------------------------------------------------------------------------


def _extract_diagnostics(evidence):
    if not isinstance(evidence, Mapping):
        raise _BundleError("evidence_not_mapping")
    diagnostics = evidence.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise _BundleError("diagnostics_missing")
    return diagnostics


def _validate_diagnostics(diagnostics):
    if int(diagnostics.get("report_cells", -1)) != EXPECTED_REPORT_CELLS:
        raise _BundleError("diagnostics_report_cells")
    if int(diagnostics.get("distinct_endpoints", -1)) != EXPECTED_DISTINCT_ENDPOINTS:
        raise _BundleError("diagnostics_distinct_endpoints")
    if diagnostics.get("outer_reverification_required") is not True:
        raise _BundleError("diagnostics_outer_reverification")
    for key in ("graph_sha256", "graph_plan_sha256", "min_bridge_sha256"):
        if not isinstance(diagnostics.get(key), str) or not diagnostics.get(key):
            raise _BundleError("diagnostics_hash_pin")


def _validate_panel(assembled):
    if not isinstance(assembled, Mapping):
        raise _BundleError("panel_not_mapping")
    coverage = assembled.get("coverage")
    if not isinstance(coverage, pd.DataFrame):
        raise _BundleError("coverage_missing")
    if len(coverage) != EXPECTED_REPORT_CELLS:
        raise _BundleError("coverage_rows")
    for column in (
        "expected_n_spectra",
        "actual_n_spectra",
        "expected_n_masters",
        "actual_n_masters",
        "complete",
    ):
        if column not in coverage.columns:
            raise _BundleError("coverage_column:" + column)
    if not bool(coverage["complete"].all()):
        raise _BundleError("coverage_incomplete")
    if not bool((coverage["expected_n_spectra"] == coverage["actual_n_spectra"]).all()):
        raise _BundleError("coverage_spectra_mismatch")
    if not bool((coverage["expected_n_masters"] == coverage["actual_n_masters"]).all()):
        raise _BundleError("coverage_masters_mismatch")
    endpoint_index = assembled.get("endpoint_index")
    if not isinstance(endpoint_index, pd.DataFrame):
        raise _BundleError("endpoint_index_missing")
    if len(endpoint_index) != EXPECTED_REPORT_CELLS:
        raise _BundleError("endpoint_index_rows")
    if "endpoint_job_id" not in endpoint_index.columns:
        raise _BundleError("endpoint_index_column")
    if int(endpoint_index["endpoint_job_id"].nunique()) != EXPECTED_DISTINCT_ENDPOINTS:
        raise _BundleError("endpoint_index_distinct")
    if len(assembled.get("global_masters", [])) != EXPECTED_GLOBAL_MASTERS:
        raise _BundleError("global_masters")
    if len(assembled.get("global_instruments", [])) != EXPECTED_GLOBAL_INSTRUMENTS:
        raise _BundleError("global_instruments")
    for key in ("panels", "domain_families"):
        if key not in assembled:
            raise _BundleError("panel_key:" + key)


def _validate_analysis(analysis):
    if not isinstance(analysis, Mapping):
        raise _BundleError("analysis_not_mapping")
    weights = analysis.get("weights")
    if not isinstance(weights, Mapping):
        raise _BundleError("analysis_weights")
    if int(weights.get("number_draws", -1)) != EXPECTED_DRAW_COUNT:
        raise _BundleError("analysis_draws")
    if int(weights.get("master_seed", -1)) != EXPECTED_MASTER_SEED:
        raise _BundleError("analysis_master_seed")
    if int(weights.get("instrument_seed", -1)) != EXPECTED_INSTRUMENT_SEED:
        raise _BundleError("analysis_instrument_seed")
    if int(weights.get("hierarchy_seed", -1)) != EXPECTED_HIERARCHY_SEED:
        raise _BundleError("analysis_hierarchy_seed")
    if len(weights.get("global_masters", [])) != EXPECTED_GLOBAL_MASTERS:
        raise _BundleError("analysis_global_masters")
    if len(weights.get("global_instruments", [])) != EXPECTED_GLOBAL_INSTRUMENTS:
        raise _BundleError("analysis_global_instruments")
    master_weights = weights.get("master_weights")
    if not isinstance(master_weights, np.ndarray):
        raise _BundleError("analysis_master_weights")
    if master_weights.shape != (EXPECTED_DRAW_COUNT, EXPECTED_GLOBAL_MASTERS):
        raise _BundleError("analysis_master_weights_shape")
    instrument_weights = weights.get("instrument_weights")
    if not isinstance(instrument_weights, np.ndarray):
        raise _BundleError("analysis_instrument_weights")
    if instrument_weights.shape != (EXPECTED_DRAW_COUNT, EXPECTED_GLOBAL_INSTRUMENTS):
        raise _BundleError("analysis_instrument_weights_shape")
    nested = analysis.get("boundary")
    if not isinstance(nested, Mapping):
        raise _BundleError("analysis_boundary")
    for label in BOUNDARY_FLAGS:
        if nested.get(label) is not True:
            raise _BundleError("analysis_boundary:" + label)
        if analysis.get(label) is not True:
            raise _BundleError("analysis_root_boundary:" + label)


# ---------------------------------------------------------------------------
# Receipt and failure surface
# ---------------------------------------------------------------------------


def _software_versions():
    versions = {"python": platform.python_version()}
    for label, module_name in _SOFTWARE_MODULES:
        try:
            module = __import__(module_name)
        except Exception:
            versions[label] = "unavailable"
            continue
        version = getattr(module, "__version__", None)
        versions[label] = str(version) if version else "unavailable"
    return versions


def _write_json(path, value):
    _exclusive_write_bytes(path, _canonical_json_bytes(value))


def _build_receipt(*, diagnostics, analysis, bundle_info, elapsed):
    weights = analysis.get("weights", {}) if isinstance(analysis, Mapping) else {}
    boundary_source = analysis.get("boundary", {}) if isinstance(analysis, Mapping) else {}
    return {
        "version": RUN_VERSION,
        "status": "success",
        "elapsed_seconds": float(elapsed),
        "hash_pins": {
            "graph_sha256": diagnostics.get("graph_sha256"),
            "graph_plan_sha256": diagnostics.get("graph_plan_sha256"),
            "min_bridge_sha256": diagnostics.get("min_bridge_sha256"),
        },
        "software_versions": _software_versions(),
        "bundle": {
            "manifest_sha256": bundle_info["manifest_sha256"],
            "num_files": bundle_info["counts"]["num_files"],
        },
        "counts": {
            "report_cells": int(diagnostics.get("report_cells", 0)),
            "distinct_endpoints": int(diagnostics.get("distinct_endpoints", 0)),
            "global_masters": len(weights.get("global_masters", []) or []),
            "global_instruments": len(weights.get("global_instruments", []) or []),
            "draws": int(weights.get("number_draws", 0) or 0),
        },
        "boundary_flags": {
            label: bool(boundary_source.get(label, False)) for label in BOUNDARY_FLAGS
        },
        "outer_reverification_required": bool(
            diagnostics.get("outer_reverification_required", False)
        ),
        "no_automatic_superiority_claim": True,
    }


def _write_errorcode(output_path, exc):
    message = str(exc)
    try:
        message = message.replace(str(output_path), "<output>")
    except Exception:
        pass
    payload = {
        "status": "error",
        "version": RUN_VERSION,
        "error_type": type(exc).__name__,
        "error_code": getattr(exc, "code", None),
        "message": message,
    }
    try:
        _write_json(Path(output_path) / ERROR_NAME, payload)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_TRAINING_SNAPSHOT_SCHEMA = "nato-sers-p08-training-evidence-snapshot-v1"
_TRAINING_ENTRY_KEYS = frozenset({"job", "summary", "summary_sha256", "receipt_sha256"})
_TRAINING_POLICIES = frozenset({"PP-U-SG", "PP-U-ARPLS"})
_TRAINING_MODELS = frozenset({"D0-M", "D1", "D2", "D3"})
_TRAINING_STAGES = frozenset({"source_fit", "final_refit"})
_TRAINING_JOB_PREFIX = "P08JOB-"
_HEX64_DIGITS = frozenset("0123456789abcdef")


def _is_lower_hex64(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(char in _HEX64_DIGITS for char in value)
    )


def _is_training_job_key(value):
    return (
        isinstance(value, str)
        and value.startswith(_TRAINING_JOB_PREFIX)
        and _is_lower_hex64(value[len(_TRAINING_JOB_PREFIX) :])
    )


def _validate_training_fit_summaries(summaries):
    if not isinstance(summaries, dict):
        raise _BundleError("training_evidence_bindings_not_mapping")
    if not all(_is_training_job_key(key) for key in summaries):
        raise _BundleError("training_evidence_job_key_invalid")
    jobs = []
    for job_key in sorted(summaries):
        entry = summaries[job_key]
        if not isinstance(entry, dict) or set(entry) != _TRAINING_ENTRY_KEYS:
            raise _BundleError("training_evidence_entry_shape_invalid")
        if not _is_training_job_key(job_key):
            raise _BundleError("training_evidence_job_key_invalid")
        job = entry["job"]
        if not isinstance(job, dict) or job.get("job_id") != job_key:
            raise _BundleError("training_evidence_job_id_mismatch")
        if not isinstance(entry["summary"], dict):
            raise _BundleError("training_evidence_summary_invalid")
        if not _is_lower_hex64(entry["summary_sha256"]):
            raise _BundleError("training_evidence_summary_sha256_invalid")
        if not _is_lower_hex64(entry["receipt_sha256"]):
            raise _BundleError("training_evidence_receipt_sha256_invalid")
        policy = job.get("policy_id")
        model = job.get("model_id")
        stage = job.get("stage")
        if policy not in _TRAINING_POLICIES:
            raise _BundleError("training_evidence_policy_forbidden")
        if model not in _TRAINING_MODELS:
            raise _BundleError("training_evidence_model_forbidden")
        if stage not in _TRAINING_STAGES:
            raise _BundleError("training_evidence_stage_forbidden")
        jobs.append(job)
    return jobs


def _snapshot_training_evidence(evidence, diagnostics):
    if "training_fit_summaries" not in evidence:
        raise _BundleError("training_evidence_bindings_missing")
    summaries = evidence["training_fit_summaries"]
    bound_jobs = _validate_training_fit_summaries(summaries)
    graph_jobs = evidence.get("jobs")
    if not isinstance(graph_jobs, (list, tuple)):
        raise _BundleError("training_evidence_graph_jobs_invalid")
    expected = []
    for job in graph_jobs:
        if not isinstance(job, dict):
            raise _BundleError("training_evidence_graph_job_invalid")
        if (
            job.get("policy_id") in _TRAINING_POLICIES
            and job.get("model_id") in _TRAINING_MODELS
            and job.get("stage") in _TRAINING_STAGES
        ):
            if not _is_training_job_key(job.get("job_id")):
                raise _BundleError("training_evidence_graph_job_id_invalid")
            expected.append(job)
    expected.sort(key=lambda job: job["job_id"])
    if _canonical_json_bytes(expected) != _canonical_json_bytes(bound_jobs):
        raise _BundleError("training_evidence_graph_coverage_mismatch")
    return {
        "schema_version": _TRAINING_SNAPSHOT_SCHEMA,
        "jobs": expected,
        "training_fit_summaries": summaries,
        "diagnostics": diagnostics,
        "private_only": True,
        "no_publication": True,
    }


def run_analysis(*, evidence_kwargs, output, check):
    """Run the exact three-stage universal workflow and store a result bundle."""
    check = _require_check(check)
    check(CHECK_INITIAL)

    output_path = Path(output)
    _refuse_destination(output_path, evidence_kwargs)
    _make_private_dir(output_path)
    created = True
    started = time.monotonic()
    try:
        check(CHECK_STAGE_LOAD)
        evidence = load_evidence(**dict(evidence_kwargs))
        diagnostics = _extract_diagnostics(evidence)
        _validate_diagnostics(diagnostics)
        training_evidence = _snapshot_training_evidence(evidence, diagnostics)

        check(CHECK_STAGE_ASSEMBLE)
        assembled = assemble_panel(
            manifest=evidence["manifest"],
            contexts=evidence["contexts"],
            roles=evidence["roles"],
            jobs=evidence["jobs"],
            aliases=evidence["aliases"],
            endpoint_frames=evidence["endpoint_frames"],
        )
        del evidence
        _validate_panel(assembled)

        check(CHECK_STAGE_ANALYZE)
        analysis = analyze_panel(
            assembled["panels"],
            global_masters=assembled["global_masters"],
            global_instruments=assembled["global_instruments"],
            domain_families=assembled["domain_families"],
        )
        check(CHECK_STAGE_ANALYZE_DONE)
        _validate_analysis(analysis)

        bundle_root = output_path / BUNDLE_DIRNAME
        bundle_info = write_bundle(
            bundle_root,
            {
                "evidence_diagnostics": diagnostics,
                "panel": assembled,
                "analysis": analysis,
                "training_evidence": training_evidence,
            },
            check,
        )

        check(CHECK_RECEIPT)
        receipt = _build_receipt(
            diagnostics=diagnostics,
            analysis=analysis,
            bundle_info=bundle_info,
            elapsed=time.monotonic() - started,
        )
        _write_json(output_path / RECEIPT_NAME, receipt)
        return receipt
    except Exception as exc:
        if created:
            _write_errorcode(output_path, exc)
        raise
