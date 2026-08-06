"""Resolve private roots without serializing workstation-specific paths."""

from __future__ import annotations

import os
from pathlib import Path


class PrivateRootNotConfigured(RuntimeError):
    """Raised when a private root is required but not configured."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configured_root(variable: str, *, must_exist: bool) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise PrivateRootNotConfigured(f"Set {variable} outside public configuration.")
    root = Path(value).expanduser().resolve()
    if must_exist and not root.is_dir():
        raise PrivateRootNotConfigured(f"Configured {variable} directory does not exist.")
    return root


def private_data_root() -> Path:
    """Return the configured immutable input root.

    The environment variable is intentionally required. Falling back to a path
    inside the checkout risks accidentally publishing restricted artifacts.
    """

    return _configured_root("ATLAS_PRIVATE_ROOT", must_exist=True)


def native_data_root() -> Path:
    """Return the immutable native instrument-export root required by P01."""

    return _configured_root("ATLAS_NATIVE_ROOT", must_exist=True)


def artifact_root() -> Path:
    """Return the configured private output root without creating it."""

    return _configured_root("ATLAS_ARTIFACT_ROOT", must_exist=False)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_private_roots(
    *, input_root: Path, output_root: Path, public_project_root: Path | None = None
) -> None:
    """Reject output locations that can overwrite inputs or enter the public tree."""

    public_root = (public_project_root or project_root()).resolve()
    inputs = input_root.resolve()
    outputs = output_root.resolve()
    if outputs == inputs or is_relative_to(outputs, inputs) or is_relative_to(inputs, outputs):
        raise ValueError("ATLAS input and artifact roots must not overlap.")
    if outputs == public_root or is_relative_to(outputs, public_root):
        raise ValueError("ATLAS_ARTIFACT_ROOT must be outside the public project tree.")
