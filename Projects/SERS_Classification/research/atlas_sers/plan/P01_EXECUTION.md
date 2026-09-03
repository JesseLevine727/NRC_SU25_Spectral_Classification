# P01 data and representation freeze

P01 converts the verified NATO SERS source archive into a source-reversible,
analysis-ready private dataset. It freezes row identity, population tiers,
native-source provenance, eight row-local spectral representations,
preservation evidence, descriptive structure analyses, and figures F02–F09.
P01 performs no predictive model fit and constructs no train/test split.
The later family-aware and QC-adaptive policies compose three of these frozen
arrays; they do not alter P01 transforms or its protected evidence.

## Preconditions

- P00 has a current, schema-valid private report with status `pass` and a
  matching `verified_skip` restart.
- `phase_registry.csv` records P00 and P01 as `complete`; the latter means the
  P01 implementation and contracts are frozen for definitive execution, not
  that P02 has begun.
- `ATLAS_PRIVATE_ROOT` contains the four immutable derived inputs declared in
  `research_contract.json` and the supplementary recording manifest.
- `ATLAS_NATIVE_ROOT` contains the immutable vendor exports used to reconstruct
  native coordinates and intensity values.
- `ATLAS_ARTIFACT_ROOT` is a separate private directory outside the checkout
  and outside both input roots.
- A TeX engine with PGFPlots is available so every native TikZ figure can be
  compiled to vector PDF.

The implementation fails closed if a root is missing or unsafe, P00 is stale,
an authoritative hash or shape differs, a source cannot be reconstructed, or
any required artifact or figure form is incomplete.

## Frozen populations

P01 creates three nested manifests from the same authoritative row order:

| Population | Rows | Role |
|---|---:|---|
| primary | 598 | definitive population |
| notes-clear | 500 | recording-note sensitivity |
| Mira-1-excluded | 575 | acquisition-system sensitivity |

The primary manifest must contain 69 physical masters, seven targets, ten
instruments, four sensor families, and three stations. Every primary row has a
deterministic sanitized observation ID and exactly one unique
`(instrument, source_scan_id)` key. Duplicate source keys, inconsistent
master/target/source mappings, non-nested tiers, or changed counts are fatal.
Free-text notes, source paths, and native filenames are never serialized.

## Source reversibility

The native registry re-reads the four supported vendor layouts. For every
selected spectrum it records a sanitized logical source identity, parser and
format, units, native point count, finite/increasing-axis checks, native and
effective support, and SHA-256 hashes of the numeric file, native coordinate
array, and native intensity array.

The authoritative common-grid input is independently reconstructed from the
native arrays. All 598 rows must reproduce byte-for-byte after float32 linear
interpolation on 400–1,849 cm⁻¹. Interpolation outside measured native or
effective support is prohibited. A seeded audit then independently rebuilds
random rows, instrument boundary rows, and tier-boundary rows and checks every
registered representation against its stored bundle.

## Frozen representations

Every operation below is row-local and therefore does not use population or
future split information:

| ID | Frozen operation |
|---|---|
| `R_MIN_400_1800` | linear interpolation; per-row min–max scaling |
| `R_MIN_400_1849` | full-common-range interpolation; per-row min–max scaling |
| `R_SG_400_1800` | isolated impulse replacement; SG(11,3); min–max |
| `R_ARPLS_400_1800` | isolated impulse replacement; arPLS; min–max |
| `R_SNV_400_1800` | interpolation; standard-normal-variate scaling |
| `R_VECTOR_400_1800` | interpolation; L2 normalization |
| `R_AREA_400_1800` | nonnegative shift; integrated-area normalization |
| `R_D1_400_1800` | isolated impulse replacement; first SG derivative; SNV |

The impulse detector uses a rolling-median window of 5, MAD threshold of 10,
and isolated neighborhood of 3. Savitzky–Golay uses window 11, polynomial 3,
and `interp` edge handling. arPLS uses lambda 100,000, at most 12 iterations,
relative-weight tolerance 0.001, and logistic clipping at 60. These values were
recovered from the immutable preprocessing-candidate archive and are verified
numerically rather than re-estimated from P01 outcomes.

The downstream action set is exactly `R_MIN_400_1800`, `R_SG_400_1800`, and
`R_ARPLS_400_1800`. Every action has 1,401 coordinates on 400–1,800 cm⁻¹ and a
final per-row `[0,1]` scale. `PP-FAMILY-SRC` selects one existing action for a
supported acquisition-platform family; `PP-QC-SRC` selects one existing action
per row using source-frozen QC rules. Any new transform, combined SG+arPLS
array, or changed parameter requires a versioned deviation and complete P01
rebuild.

Nonfinite, insufficient-support, flat, zero-range, zero-norm, and zero-area
rows receive explicit reason codes. No invalid row is silently repaired. Each
bundle hashes the axis, shape, observation order, source bundle, operations,
parameters, code, configuration, and run identity.

`R_MIN_400_1800` is the primary representation because it puts every spectrum
on the same row-wise scale while retaining measured peak shape and avoiding the
terminal common-range edge. The other representations are prespecified
sensitivities. `R_D1_400_1800` is a destructive control, not a candidate to be
promoted by attractive clustering.

## Preservation and descriptive structure

P01 quantifies what each transform changes for every row and instrument:

- value range, mean, standard deviation, L2 norm, and integrated area;
- Pearson and Spearman shape agreement and spectral angle;
- peak count and peak-location retention;
- high-frequency/noise and background summaries;
- replaced impulses, clipping, and invalid-row reasons.

These diagnostics characterize preservation; they do not prove denoising or
clean-spectrum recovery.

The permitted future QC-gate inputs are limited to normalized first-difference
noise, spike fraction, baseline energy/span fractions, and negative fraction
already reconstructable from the native registry. Instrument, platform family,
SERS sensor, station, master, label, model confidence, and target-batch
statistics are forbidden QC-gate features.

Raw arrays and all eight representations are analyzed at spectrum and
physical-master levels with frozen settings:

- PCA variance and components required for 95% variance;
- K-means for k=2…12, 50 initializations, and five seeds;
- a 3×3 HDBSCAN grid over minimum cluster sizes 8/12/20 and minimum samples
  3/5/10;
- UMAP with cosine distance, 15 neighbors, minimum distance 0.1, and three
  seeds;
- t-SNE with perplexity 30, PCA initialization, automatic learning rate, and
  three seeds;
- cluster stability, trustworthiness, collapse/noise rates, and NMI/ARI
  association with target, station, instrument, and sensor metadata.

These are descriptive fits. Visual separation and metadata association cannot
be interpreted as predictive generalization or used to tune a later test set.

## Figures F02–F09

Each figure is generated from one frozen aggregate CSV table in four matched
forms:

1. aggregate CSV under `figures/data/`;
2. native TikZ/PGFPlots under `figures/tikz/`;
3. compiled vector PDF under `figures/pdf/`;
4. standalone self-contained Plotly HTML under `figures/html/`.

The aggregate table SHA-256 is embedded in both TikZ and HTML. The figure
manifest verifies semantic parity, successful compilation, native TikZ (no
raster wrapper), and standalone HTML. F02–F09 cover population flow, factor
support, native-axis coverage, instrument spectra, preservation, PCA diagnostics,
matched PCA/UMAP/t-SNE master views, and clustering association/stability.

## Commands

Run from `research/atlas_sers` after exporting all three private roots:

```bash
python3 scripts/validate_public_scaffold.py
python3 scripts/run_p01.py audit
pytest -q
python3 scripts/run_p01.py dry-run
python3 scripts/run_p00.py dry-run
python3 scripts/run_p00.py dry-run
python3 scripts/run_p01.py build
python3 scripts/run_p01.py build
python3 scripts/run_p01.py validate
```

`audit` cross-checks the public governance contract. `dry-run` reports the
exact populations, representations, figures, and storage estimate without
reading private inputs or constructing data. The first `build` performs the
definitive private transaction and reports action `new`; the second must
rehash the completed transaction and report `verified_skip`. `validate`
rehashes the latest completed run and its pointer.

The P00 rerun is intentionally after final public edits. P01 refuses a P00
lock whose code, configuration, commit, dirty-state hash, dependencies, or
artifact evidence is stale.

## Private output contract

The completed directory is
`${ATLAS_ARTIFACT_ROOT}/p01/runs/<run_id>/`. Its required top-level payloads
are listed exactly in `p01_governance_contract.json`. They include:

- three population manifests and the native-source registry;
- eight deterministic NPZ bundles under `representations/` plus their
  registry and row QC;
- candidate-reproduction and reversibility evidence;
- row- and instrument-level preservation tables;
- PCA, embedding, clustering, association, and stability tables;
- F02–F09 aggregate/TikZ/PDF/HTML outputs and compilation logs;
- `P01_VALIDATION_REPORT.json` and `P01_ARTIFACT_HASHES.json`.

The artifact store adds `_STATE.json`, updates `p01/LATEST.json`, and preserves
stale, corrupt, incomplete, conflicting, or failed attempts beneath
`p01/quarantine/`. Completed evidence is never overwritten.

## Status and interpretation contract

- `pass` / exit 0: every required scientific, provenance, privacy, figure,
  schema, and hash check passed.
- `fail` / exit 1: an input, invariant, reproduction, privacy, figure, schema,
  artifact, or current-P00 check failed.
- `blocked` / exit 2: required input/evidence is unavailable.

P01 authorizes zero predictive fits and zero split invocations. It cannot
establish denoising, clean-spectrum recovery, causal chemical/nuisance
disentanglement, or classification generalization. It establishes that later
experiments use a complete, consistent, reversible, explicitly transformed
dataset.

## Historical boundary to P02

P02 remained forbidden until the P01 report was schema-valid and `pass`, every
artifact rehashes, the identical build returns `verified_skip`, the latest-run
validator passes, and the G0 evidence is reviewed. Only P02 may freeze
physical-master-grouped partitions and information regimes. P01 results cannot
be used to move representation thresholds or choose a representation based on
future classifier outcomes. P02 may only enumerate source roles, metadata-only
family support, and the registered QC quantile procedure. It may not invent a
new preprocessing array or use target outcomes to choose one. Those gates
passed before the separate P02 freeze began; its evidence is documented in
`P02_EXECUTION.md`.
