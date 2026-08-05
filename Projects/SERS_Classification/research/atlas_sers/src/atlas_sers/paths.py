"""Resolve private inputs without serializing workstation-specific paths."""

from __future__ import annotations

import os
from pathlib import Path


class PrivateRootNotConfigured(RuntimeError):
    """Raised when a private root is required but not configured."""


def private_data_root() -> Path:
    """Return the configured immutable input root.

    The environment variable is intentionally required. Falling back to a path
    inside the checkout risks accidentally publishing restricted artifacts.
    """

    value = os.environ.get("ATLAS_PRIVATE_ROOT")
    if not value:
        raise PrivateRootNotConfigured("Set ATLAS_PRIVATE_ROOT outside the Git checkout.")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise PrivateRootNotConfigured(f"Configured private root is not a directory: {root}")
    return root


def artifact_root() -> Path:
    """Return the configured private output root without creating it."""

    value = os.environ.get("ATLAS_ARTIFACT_ROOT")
    if not value:
        raise PrivateRootNotConfigured("Set ATLAS_ARTIFACT_ROOT outside the Git checkout.")
    return Path(value).expanduser().resolve()
