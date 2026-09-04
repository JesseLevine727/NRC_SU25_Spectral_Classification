#!/usr/bin/env python3
"""Publish aggregate-only P04 evidence after final comparison passes."""

from __future__ import annotations

import json

from atlas_sers.governance.p04_publication import publish_p04
from atlas_sers.paths import artifact_root, project_root

if __name__ == "__main__":
    print(
        json.dumps(
            publish_p04(artifact_root=artifact_root(), project_root=project_root()),
            indent=2,
            sort_keys=True,
        )
    )
