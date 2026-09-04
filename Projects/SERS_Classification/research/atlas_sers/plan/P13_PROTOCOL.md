# P13 NATO field-trial substrate portability protocol

**Amendment version:** `nato-sers-p13-v1-locked`

**Approved and locked:** 2026-09-04 by the project owner

**Outcome timing:** P03 outcomes were known before P13 was proposed. No P13
predictive, crossover-effect, or field-log outcome was calculated or used to
choose this protocol.

P13 is a versioned secondary study. It does not retroactively alter the
P00–P03 analysis, hashes, selections, or claims. The machine-readable decision,
support, split, experiment, metric, and figure registries are the execution
authority; [P13_FREEZE_MEMO.md](P13_FREEZE_MEMO.md) is the concise freeze record.

## Scientific question

The field trial was intended to assess whether the SERS substrates recovered
the required analyte signal independently of the acquisition instrument. P13
translates that purpose into a bounded predictive question:

> Within each support-qualified station and substrate family, is three-class
> analyte-discriminative signal recoverable on an instrument excluded from all
> fitting without a practically important loss relative to source instruments?

The evaluation domain is **station × substrate family × held instrument**. A
class-support cell is that domain × analyte. Analyte is not part of the
evaluation-domain key because three-class balanced accuracy requires all three
analytes in the same domain.

This study can provide evidence of predictive portability over tested
conditions. It cannot prove universal instrument independence, a physical SERS
mechanism, successful acquisition by arbitrary instruments, or a global
substrate ranking.

## Frozen starting population and support

- 598 stored spectra and 69 independent physical masters.
- Four normalized substrate families and ten acquisition instruments.
- 2,760 possible master × substrate × instrument cells: 374 observed and 2,386
  unobserved.
- 67 masters measured on at least two instruments, 39 on at least two substrate
  families, and 32 supporting at least one complete two-substrate ×
  two-instrument crossover.
- 34 observed station × substrate-family × held-instrument domains: 13
  confirmatory, three exploratory low-support, and 18 unsupported by design.
- 34 observed analyte-specific crossover blocks: eight confirmatory, seven
  exploratory low-support, and 19 descriptive singletons.

The support audit uses metadata and split roles only. Missing matrix cells mean
the combination was not observed; they are not failed measurements. F44 is the
authoritative design visualization. The generated support registries retain
every observed domain and crossover block with its reason code.

## Units, predictions, and splits

The independent unit is the physical master. The primary prediction unit is a
master–substrate–instrument view: average the predicted class probabilities of
technical repeats within that view before scoring. Spectrum-level results may
be reported as secondary diagnostics but cannot replace the master-level
primary endpoint.

Reuse the five repeated four-fold P02 physical-master splits. For each domain:

1. exclude the held instrument from transformation fitting, estimator fitting,
   model selection, calibration, stopping, and threshold selection;
2. use only non-held-instrument spectra from P02 training masters for fitting;
3. use only non-held-instrument spectra from P02 validation masters for
   selection and calibration; and
4. evaluate the held instrument only on P02 outer-test masters.

The matched source reference uses outer-test masters that have both the held
view and at least one eligible source-instrument view. This keeps the
source-to-held loss paired by physical sample.

## Locked support tiers

A confirmatory domain requires, for every analyte:

- at least three held-instrument masters;
- at least three source training masters in every outer split;
- at least two source instruments in every outer split; and
- at least three masters with both held- and source-instrument views.

An exploratory domain requires at least two held masters and two source
training masters per analyte in every outer split. Source-instrument diversity
and pairing limitations remain reported but are not exploratory eligibility
gates. Everything else is `unsupported_by_design` and remains visible.

For analyte-specific two-substrate × two-instrument crossover blocks, at least
three physical masters is confirmatory, exactly two is exploratory, and one is
descriptive with no interval claim.

## Models and preprocessing

`C-SELECTED`, selected strictly from source data, is the primary classical
procedure. A fixed RBF SVM is the main estimator sensitivity. PCA–LDA, PLS-DA,
elastic-net logistic regression, Random Forest, and Extra Trees are secondary.
The compact deep comparison waits for the P04 architecture contract and must
use identical eligible cells, preprocessing arrays, split roles, and test UIDs.

`PP-U-MIN` is primary: row-local interpolation over measured 400–1,800 cm⁻¹
support followed by per-spectrum min–max scaling. Universal Savitzky–Golay and
arPLS policies are paired sensitivities on identical UIDs. An instrument-aware
or substrate-aware rule is a separate information regime and must be selected
from source data or declared as target-instrument calibration; it cannot be
optimized using held labels or held performance.

## Primary endpoint and bounded decision

The primary metric is three-class balanced accuracy at the averaged
master–substrate–instrument-view level. Define source-to-held loss as matched
source balanced accuracy minus held-instrument balanced accuracy.

For a confirmatory domain to support portability, both conditions must hold:

1. the lower 95% confidence bound for held balanced accuracy is at least
   `tau = 0.60`; and
2. the upper 95% confidence bound for source-to-held balanced-accuracy loss is
   at most `delta = 0.10`.

Use a 95% physical-master-clustered hierarchical bootstrap with 10,000
resamples. Use BCa intervals where stable and percentile intervals otherwise.
The substrate-family claim is an intersection-union decision: every
confirmatory domain for that substrate must pass both bounds. Holm correction
applies to individual secondary cell claims.

Each observed domain must end as exactly one of:

- `supports_portability`;
- `inferior_portability`;
- `inconclusive`;
- `unsupported_by_design`; or
- `unavailable_terminal_failure`.

An inferiority conclusion requires interval evidence of a threshold violation;
an estimate that misses a threshold without decisive interval evidence is
inconclusive. A substrate with no confirmatory domains cannot support a
confirmatory portability claim.

## Incomplete endpoints and field-log corroboration

P03 terminal failures remain in the declared denominator. Model comparisons
use both the common successfully evaluated endpoints and a chance-performance
sensitivity for unavailable endpoints. A positive comparison claim is
prohibited if that sensitivity reverses it; no hidden estimator substitution is
allowed.

Analyze recorded field-trial outcomes separately from analyte classification:

- nonblank `Y` is a recorded detection endpoint;
- blank `N` is a recorded-specificity endpoint;
- ambiguous `M` is excluded from the definite binary endpoint;
- missing stays missing; and
- report complete-case estimates plus best- and worst-case missingness bounds.

The recorded outcome cannot train the primary classifier, and `N` is not proof
that a stored spectrum contains no analyte information.

## Required evidence

- F44: sample × substrate × instrument coverage, complete.
- F45: held-instrument recoverability by substrate and eligible domain.
- F46: paired same-master substrate × instrument crossover effects.
- F47: field-log completeness and agreement with model evidence.
- Machine-readable support, split, prediction, metric, interval, failure, and
  bounded-claim tables.
- Native TikZ, vector PDF, PNG review copy, and standalone HTML generated from
  one semantic table for each quantitative figure.

The next authorized work is the deterministic no-fit expansion and execution
of classical experiments `EXP-P13-C01` through `EXP-P13-C04`. The locked
thresholds or support tiers may not be changed after P13 outcomes are accessed;
any necessary correction requires a dated protocol version and deviation log.
