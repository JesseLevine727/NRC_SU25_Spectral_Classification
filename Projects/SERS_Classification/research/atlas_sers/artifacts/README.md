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

P01 uses the same protected-run protocol beneath `p01/`. Its private payload
contains row-level manifests, native provenance, eight representation bundles,
QC and preservation evidence, descriptive embeddings/clusters, and F02–F09 in
aggregate CSV, native TikZ, vector PDF, and standalone HTML forms. Exact files
and validation behavior are documented in `../plan/P01_EXECUTION.md`. Neither
P00 nor P01 output is approved for publication merely because a run passes.
