# Repository architecture

## Design objective

The codebase separates private data handling, scientific transformation,
evaluation, and publication. This prevents exploratory convenience code from
quietly changing the confirmatory protocol and makes leakage boundaries
testable.

## Artifact flow

```text
private immutable inputs
        │
        ▼
P00 hash verification + sanitized provenance + governance dry run
        │
        ▼
data inventory and parser validation
        │
        ▼
row-level private manifest + native/effective axis registry
        │
        ├──────────────► exploratory geometry and clustering
        │
        ▼
frozen representations + master-grouped split registry
        │
        ├──────────────► classical selection
        ├──────────────► deep development
        └──────────────► registered diagnostics
                               │
                               ▼
private row-level predictions and run records
                               │
                               ▼
disclosure-reviewed aggregate tables
                         ┌─────┴─────┐
                         ▼           ▼
                    native TikZ   standalone HTML
```

No arrow from test observations may feed preprocessing fitting, candidate
selection, calibration, thresholding, or early stopping in a source-only task.

### `atlas_sers.governance`

Owns P00/P01 registry validation, canonical serialization, deterministic run
identities, streaming authoritative-input verification, sanitized provenance,
the no-training dry-run registry, and atomic artifact commits. A matching
successful protected state is verified and skipped. Incomplete, corrupt,
stale, or conflicting run directories are moved into a private quarantine with
a structured reason; completed evidence is never overwritten.

P00 may inspect metadata declared in contracts, but it may not construct
representations, splits, predictions, embeddings, or fitted objects.
P01 adds immutable-manifest, native-source, representation, descriptive-fit,
figure, and restart validation while continuing to prohibit predictive models
and split construction.

## Package boundaries

### `atlas_sers.data`

Owns instrument adapters, metadata canonicalization, provenance, observation
identifiers, master attribution, measured-support checks, and numeric QC. It
must not perform model fitting.

### `atlas_sers.preprocessing`

Owns interpolation within measured support, per-spectrum scaling, optional
smoothing/baseline/alignment candidates, representation identifiers, and
preservation diagnostics. Any learned transformation exposes explicit `fit`
and `transform` roles so leakage tests can audit it.

### `atlas_sers.exploration`

Owns PCA, K-means, HDBSCAN, UMAP, t-SNE, cluster stability, metadata
association, and support visualizations. Exploratory outputs cannot select a
confirmatory test result after test labels are revealed.

### `atlas_sers.splits`

Owns deterministic physical-master-grouped outer partitions, inner selection
partitions, held-instrument roles, held-chemical roles, adaptation/calibration
roles, and fatal leakage assertions. Split records are immutable experiment
inputs.

### `atlas_sers.models`

Contains two peer families:

- `classical`: PCA-LDA, logistic regression, SVM, PLS-DA, random forest, and
  Extra Trees pipelines selected only by registered inner objectives;
- `deep`: compact residual encoders, acquisition-aware supervised contrastive
  objectives, paired-view consistency, and prespecified domain controls.

VAE-family models belong here only as registered representation diagnostics or
clearly scoped baselines. They are not presumed denoisers.

### `atlas_sers.evaluation`

Owns row- and master-level metrics, domain-balanced aggregation, calibration,
selective prediction, open-set metrics, hierarchical bootstrap intervals,
robustness perturbations, nuisance probes, failure accounting, and decision
gate evaluation. It consumes frozen predictions; it does not retrain models.

### `atlas_sers.visualization`

Owns the colorblind-safe style, aggregate figure tables, TikZ/PGFPlots source,
standalone HTML, semantic-parity hashes, and figure manifests. It may consume
only publication-approved aggregates.

## Configuration hierarchy

Machine-readable contracts in `plan/contracts` define immutable study-wide
rules. Future executable configs should be layered without duplicating those
rules:

```text
contracts/                 immutable scientific contract
configs/representations/   preprocessing candidates
configs/tasks/             task and information regime
configs/models/            model family and search space
configs/runs/              seed, fold, compute target, output location
```

A run record stores the resolved configuration and hashes it. Local paths are
resolved at runtime and never serialized into public artifacts.

## Test layers

1. **Unit:** parsers, transforms, metrics, and deterministic seeds.
2. **Contract:** schemas and registries match the master plan.
3. **Leakage:** master and held-domain roles are disjoint; fitted transforms see
   only permitted rows.
4. **Integration:** synthetic spectra traverse ingestion through prediction.
5. **Reproducibility:** a frozen synthetic run reproduces hashes and metrics.
6. **Publication:** restricted formats/paths are absent and figure pairs agree.

Real observations are never test fixtures in the public repository.

P00 adds synthetic regression tests for every run-identity field, fail-closed
hash/shape/status checks, private-path containment, Git tracking detection,
byte-stable dry runs, atomic commits, quarantine, and verified restart skips.
P01 adds synthetic vendor layouts, UID/mapping and support failures, transform
invariants, independent recomputation, deterministic arrays, structure-analysis
reproducibility, figure semantic parity, full P00→P01 integration, corruption
handling, and verified restart skips.

## Implementation order

1. Pass P00 governance, private input verification, and the no-training dry run.
2. Rebuild the private inventory through package adapters, validate native
   hashes/reversibility, and freeze representations and descriptive evidence.
3. Freeze master-grouped split registries only after P01 passes.
4. Implement the common estimator/evaluation interface.
5. Run classical selection and lock the endpoint baseline.
6. Develop the compact deep endpoint on source-development partitions only.
7. Freeze the deep candidate before primary held-domain evaluation.
8. Execute primary comparison, calibration, robustness, and representation
   diagnostics.
9. Evaluate secondary adaptation/open-set questions without broadening claims.
10. Generate paired figures and evaluate the publication decision gates.

The phase registry remains authoritative if this summary and the master plan
ever diverge.
