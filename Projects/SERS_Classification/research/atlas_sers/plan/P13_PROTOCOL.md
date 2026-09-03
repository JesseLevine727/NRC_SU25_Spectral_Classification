# P13 NATO field-trial substrate portability protocol

**Amendment version:** `nato-sers-p13-v1-draft`

**Timing:** drafted after P03 outcomes were available. P13 is therefore a
versioned secondary study and does not retroactively alter the P00–P03 primary
analysis, hashes, selections, or claims.

## Scientific question

The field trial was intended to assess whether the SERS substrates recovered
the required analyte signal independently of the acquisition instrument. P13
translates that purpose into a bounded predictive question:

> Within each support-qualified substrate family, station, and analyte setting,
> is analyte-discriminative signal recoverable on an instrument excluded from
> all fitting without a practically important loss relative to source
> instruments?

This is evidence of predictive portability over tested conditions. It is not
proof of universal instrument independence, a physical SERS mechanism, or a
global ranking of substrates.

## Frozen starting population

- 598 stored spectra.
- 69 independent physical-master samples.
- Four normalized substrate families.
- Ten acquisition instruments.
- 2,760 possible master × substrate × instrument cells.
- 374 observed and 2,386 unobserved cells.
- 67 masters measured on at least two instruments.
- 39 masters measured on at least two substrate families.
- 32 masters supporting at least one complete two-substrate × two-instrument
  crossover.

Missing matrix cells mean the combination was not observed; they do not encode
a failed detection. F44 is the authoritative design visualization.

## Evidence tracks

### Track A — substrate-conditioned held-instrument recoverability

For every metadata-qualified station × analyte × substrate-family × held-
instrument cell:

1. exclude the held instrument from transformation fitting, estimator fitting,
   model selection, calibration, stopping, and threshold selection;
2. group every split and uncertainty calculation by physical master;
3. fit the frozen classical panel under `PP-U-MIN`;
4. evaluate the held instrument at spectrum, instrument-view, and physical-
   master levels without conflating those units;
5. report each cell before any equal-cell or support-weighted summary; and
6. estimate minimum held-instrument performance and the matched source-to-held
   loss with physical-master resampling.

### Track B — same-master crossover evidence

Use support-qualified crossover blocks to compare the same physical sample
across at least two substrates and at least two instruments. Estimate substrate,
instrument, and substrate × instrument effects with master blocking or a
hierarchical model. Outcomes include correctness, true-class probability,
classification margin, and representation distance. Direct crossover evidence
must remain distinct from model-based extrapolation.

### Track C — recorded field-trial outcome corroboration

Analyze recorded `Y`/`N` target-detection outcomes separately from analyte
classification. Report completeness and missingness by station, analyte,
substrate, and instrument. Exclude ambiguous `M` values from the definite
binary endpoint. Do not train the primary analyte classifier on this field-log
outcome, and do not interpret `N` as proof that the spectrum contains no
analyte information.

### Track D — compact deep-learning comparison

After the classical P13 analysis is frozen, evaluate a compact 1D model on the
identical eligible cells, master splits, preprocessing arrays, and test UIDs.
Use source-only early stopping, multiple seeds, regularization, and collapse
checks. Repeated spectra provide structured views but never increase the count
of independent chemical samples beyond 69.

## Preprocessing policy

`PP-U-MIN` remains primary: row-local interpolation over measured
400–1,800 cm⁻¹ support followed by per-spectrum min–max scaling. This preserves
the acquisition-shift challenge. Universal Savitzky–Golay and arPLS policies
are paired sensitivities on identical UIDs. An instrument-aware preprocessing
rule is a separate information regime and must be selected from source data or
declared as target-instrument calibration; it cannot be optimized using held
labels or performance.

## Models

The primary classical panel contains RBF SVM, PCA–LDA, PLS-DA, elastic-net
logistic regression, Random Forest, Extra Trees, and the frozen source-only
selection procedure where supported. Fixed-family results and source-selected
results are reported separately. A model cannot be promoted retrospectively
because it performed well on P03 held instruments.

The deep comparison must use the P04 compact architecture contract and the same
P13 support cells. Architecture and epoch choices cannot use P03 or P13 held-
instrument outcomes.

## Decisions that must be frozen before P13 fitting

The machine-readable decision and support registries deliberately mark the
following items pending:

- minimum scientifically useful recoverability threshold `tau`;
- maximum acceptable source-to-held loss `delta`;
- minimum physical-master support per class in source and held roles;
- treatment of substrate variants;
- multiplicity family and interval method; and
- handling of the eight unavailable P03 `C-SELECTED` endpoints in later paired
  classical/deep comparisons.

Li-Lin and the project owner must approve `tau` and `delta` before outcome
calculation. A nonsignificant instrument coefficient is not evidence of
equivalence.

## Required results

- F44: sample × substrate × instrument coverage, complete.
- F45: held-instrument recoverability by substrate and eligible cell.
- F46: paired same-master substrate × instrument crossover effects.
- F47: recorded detection completeness and agreement with model evidence.
- Machine-readable support, split, prediction, metric, interval, and bounded-
  claim tables.
- Native TikZ, vector PDF, PNG review copy, and standalone HTML generated from
  one semantic table for every quantitative figure.

## Completion states

Every evaluated cell must end as one of:

- `supports_portability`;
- `inferior_portability`;
- `inconclusive`;
- `unsupported_by_design`; or
- `unavailable_terminal_failure`.

No unsupported or failed cell may disappear from its declared denominator.
