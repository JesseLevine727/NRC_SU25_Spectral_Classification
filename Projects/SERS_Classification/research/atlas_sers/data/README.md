# Private data mount point

No data belongs in this directory. Set `ATLAS_PRIVATE_ROOT` to an immutable
derived-input directory and `ATLAS_NATIVE_ROOT` to the immutable native
vendor-export directory, both outside the Git checkout. The derived root
supplies the logical inputs declared by
`../plan/contracts/research_contract.json`; the native root supplies only the
source-reversibility records required by P01.

P00 resolves only `${ATLAS_PRIVATE_ROOT}` declarations, verifies each file by
streaming SHA-256 plus its declared row count, array shape, or status, and fails
closed. Missing inputs are `blocked`; mismatches are `fail`. Neither condition
is converted into a definitive pass.

Public tests must use small synthetic fixtures generated in memory. Do not add
spectra, row-level manifests, labels, recording notes, instrument exports, or
derived arrays here.

P01 re-reads native coordinates and intensities, verifies hashes and common-grid
reconstruction, and serializes only sanitized logical source identities to its
private artifact root. Native filenames and resolved paths are not outputs.
