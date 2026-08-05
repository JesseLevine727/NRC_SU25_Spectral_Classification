# Contributing

## Working sequence

1. Read `PUBLICATION_POLICY.md` and the relevant plan phase.
2. Create a narrowly scoped branch.
3. Keep private inputs outside the checkout and access them through
   `ATLAS_PRIVATE_ROOT`.
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

## Figures

Every registered figure requires a frozen aggregate data table, native TikZ or
PGFPlots source, standalone HTML, and semantic-parity verification. Raster
wrappers are not valid native figure sources.

## Pull-request checklist

- [ ] Only ATLAS is used as the public project identifier.
- [ ] No raw or row-level derived data is present.
- [ ] No absolute source paths or workstation identifiers are present.
- [ ] Split and preprocessing fitting roles remain leakage-safe.
- [ ] Tests and validators pass.
- [ ] Figure sources and HTML use the same frozen aggregate data.
- [ ] Claim scope is labelled primary, secondary, exploratory, or prohibited.
