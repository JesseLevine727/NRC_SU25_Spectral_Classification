# Private data mount point

No data belongs in this directory. Set `ATLAS_PRIVATE_ROOT` to an immutable
directory outside the Git checkout. The private root supplies the logical
inputs declared by `../plan/contracts/research_contract.json`.

P00 resolves only `${ATLAS_PRIVATE_ROOT}` declarations, verifies each file by
streaming SHA-256 plus its declared row count, array shape, or status, and fails
closed. Missing inputs are `blocked`; mismatches are `fail`. Neither condition
is converted into a definitive pass.

Public tests must use small synthetic fixtures generated in memory. Do not add
spectra, row-level manifests, labels, recording notes, instrument exports, or
derived arrays here.
