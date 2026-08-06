# Local artifacts

This directory is an ignored convenience mount for local runs. Row-level
predictions, partitions, embeddings, trained weights, histories, and generated
spectral arrays remain private.

Only disclosure-reviewed aggregate tables and their paired TikZ/HTML figures
may be promoted into a future public release.

P00 writes to `${ATLAS_ARTIFACT_ROOT}/p00/runs/<run_id>/` using an atomic
temporary directory. The sibling `p00/quarantine/` preserves invalid prior
states with reasons, and `p00/LATEST.json` identifies the last checked run
without exposing a local path. The artifact root must be outside the public
project and must not overlap `ATLAS_PRIVATE_ROOT`.
