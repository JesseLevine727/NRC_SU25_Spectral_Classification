# Contributing

## Working sequence

1. Read `PUBLICATION_POLICY.md` and the relevant plan phase.
2. Work on the branch authorized for the current repository workflow.
3. Use the frozen compatibility variables `ATLAS_PRIVATE_ROOT`,
   `ATLAS_NATIVE_ROOT`, and `ATLAS_ARTIFACT_ROOT` for governed executions.
   Do not hard-code workstation paths.
4. Add or update tests with each implementation slice.
5. Run the public scaffold validator and the relevant scientific validation.
6. Inspect the staged file list and diff before committing.
7. Open a draft pull request and record the experiment/decision identifiers
   affected by the change.

## Implementation conventions

- Use the package under `src/atlas_sers`; do not add new top-level analysis
  scripts without a thin CLI wrapper and a tested package implementation.
- Pass paths through configuration. Never hard-code a local data location.
- Treat a physical master, rather than a spectrum row, as the resampling and
  split unit unless a registered task explicitly says otherwise.
- Record every executed run with protocol, code, configuration, input, split,
  and hyperparameter hashes.
- Keep exploratory and confirmatory outputs in distinct artifact namespaces.
- A failed run is a result: record its status and reason.
- Do not edit a completed run directory. Let the artifact store verify/skip it
  or quarantine a stale, corrupt, incomplete, or conflicting state.

## P00 checks

Before implementing P01 or later code, run from this directory:

```bash
python3 scripts/validate_public_scaffold.py
python3 scripts/run_p00.py audit
pytest -q
python3 scripts/run_p00.py dry-run
python3 scripts/run_p00.py dry-run
```

The first dry run must report `pass`; the second must report
`verified_skip`. Exit status `1` means failed governance or input validation,
and `2` means an authoritative input is unavailable. Unset/unsafe roots stop
execution before a definitive report. Record any post-lock change in
`plan/registries/deviations.csv` before using outcomes to justify it.

For P01, additionally run:

```bash
python3 scripts/run_p01.py audit
python3 scripts/run_p01.py dry-run
python3 scripts/run_p01.py build
python3 scripts/run_p01.py build
python3 scripts/run_p01.py validate
```

The first definitive build must pass, the identical build must report
`verified_skip`, and validation must rehash the latest run. See
`plan/P01_EXECUTION.md`. P01 authorizes no split or predictive model code.

## Figures

Every registered figure requires a frozen aggregate data table, native TikZ or
PGFPlots source, standalone HTML, and semantic-parity verification. Raster
wrappers are not valid native figure sources.

## Pull-request checklist

- [ ] Public-facing text identifies the study as NATO SERS; legacy `ATLAS`
      implementation identifiers are changed only through a versioned migration.
- [ ] Every raw, row-level, or model artifact in the staged set is intentionally
      approved for public release rather than swept in from a generated workspace.
- [ ] No absolute source paths or workstation identifiers are present.
- [ ] Split and preprocessing fitting roles remain leakage-safe.
- [ ] Research-question, preprocessing-policy, actual-action, access-regime,
      platform-family, fallback, and policy-hash fields are complete.
- [ ] No held-test outcome or forbidden target-batch statistic selects a policy.
- [ ] Tests and validators pass.
- [ ] P00 passes and repeats as `verified_skip` before any P01 work.
- [ ] P01 passes, repeats as `verified_skip`, and validates before any P02 work.
- [ ] Figure sources and HTML use the same frozen aggregate data.
- [ ] Claim scope is labelled primary, secondary, exploratory, or prohibited.
