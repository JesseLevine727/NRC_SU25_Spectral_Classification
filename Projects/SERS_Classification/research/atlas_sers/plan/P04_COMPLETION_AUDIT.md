# P04 compact D0 completion audit

**Protocol:** `nato-sers-p04-v1-locked`

**No-fit plan:** `P04PLAN-ef3155b83c14ab242a47067a`

**Execution:** `P04-e845290bb15d37882f29da6f`

**Status:** P04 scientific execution, reconciliation, comparison, and public figure release complete. Repository delivery is recorded by the publication commit and its GitHub Actions run.

This record distinguishes training validity, predictive performance, and phase
delivery. Completion of P04 does not complete the acquisition-aware comparison
or establish that a model has separated chemical signal from nuisance.

## Frozen design and training validity

| Requirement | Evidence | State |
| --- | --- | --- |
| Population and tensor | 598 spectra, 69 physical masters, PP-U-MIN, 1,401 channels over 400–1,800 cm⁻¹; manifest/axis/order/min–max checks | pass |
| Physical-master isolation | exact P02 five-repeat/four-fold roles; held instruments and outer-test masters excluded from T3 fitting, validation, calibration, and stopping | pass |
| Architecture | 208,691 trainable parameters; 64-dimensional embedding; three-class station head; GroupNorm; deterministic ordered pooling | pass |
| Exact no-fit expansion | 320 contexts; 15,498 inner fits; 960 final refits; 16,458 total fits; 132,392 UID-role rows | pass |
| Optimizer and training rule | six AdamW LR/weight-decay combinations; three registered neural seeds; batch 48; max 200/min 30/patience 20; source-only checkpoint and final-epoch selection | frozen |
| G2 development gate | 60/60 development contexts; 3,240/3,240 finite inner fits; one selection per context | pass |
| Gate diagnostics | 2,824 development inner fits with no diagnostic; 416 with the overfit diagnostic; none with collapse or numerical failure | retained |
| Source-only separation | no prior P03/P13 outcome table enters the fit/selection path; investigators already knew those results | procedural separation; not analyst blinding |
| Pre-outcome implementation failures | two deterministic cuBLAS environment failures and one adaptive-pooling backward failure, each retained in an immutable smoke-test run | recorded in deviations |

The development freeze SHA-256 is
`0f99f80feac9e6bafee84203bfe64b5ca57789f7b9a6af93e6d3c592bb682f6d`.
The execution protected-state SHA-256 is
`e845290bb15d37882f29da6f843620846ae61c75bfca2231c6e6c7686d748d19`.

## Final reconciliation

All 320 immutable context shards passed validation. The exact fit-ID set has
16,458 records with no duplicates: 15,498 inner-selection fits and 960 final
refits. All fits completed; no endpoint is missing. The audit reconciled 960
final checkpoints, 810,890 epoch-history rows, 537,012 source-validation
prediction rows, 17,325 seed test predictions, and 5,775 ensemble test
predictions. Both prediction schemas and all 640 M01/M06 endpoint records
passed. Duplicate seed rows and invalid probability vectors are rejected
before averaging. A finite fit with collapsed or overfit predictions remains
in the denominator.

The final aggregation state is
`85280d95620696993af0fc93afb61f86d66ce1f7bfb6e1205139dc2b80bf9793`.

The first aggregation attempt was quarantined because the P01 manifest lacks
the derived `instrument_family` field required by the export schema. The
reporting adapter now uses the existing P02 acquisition-family mapping, with
regression tests for both seed and ensemble records. Exact comparisons against
the quarantined output confirmed that all original scalar prediction fields
and all 640 metric records were unchanged. No model, split, spectrum,
probability, or checkpoint was changed. The four protected training-file hashes
remain identical to the execution freeze.

When unfinished tail work was reassigned, redundant original workers sometimes
encountered an already-owned shard lock and exited. The active owner continued;
no duplicate model fit ran and no existing result was replaced. These worker
lease exits are not model-fit failures.

| Training quantity | Development | T3 source-only selection |
| --- | ---: | ---: |
| Inner fits | 3,240 | 12,258 |
| Median executed inner epochs | 51 | 42 |
| Maximum executed inner epochs | 159 | 165 |
| Median selected-candidate checkpoint epoch | 32 | 23 |
| Median final-refit epochs | 30 | 30 |
| Maximum final-refit epochs | 83 | 94 |
| Overfit diagnostic | 416 | 3,927 |
| Collapse diagnostic | 0 | 176 |
| Underfit diagnostic | 0 | 30 |
| Instability diagnostic | 0 | 33 |

| Selected learning rate / weight decay | Development contexts | T3 contexts |
| --- | ---: | ---: |
| 0.0003 / 0.00001 | 5 | 28 |
| 0.0003 / 0.0001 | 3 | 19 |
| 0.0003 / 0.001 | 2 | 26 |
| 0.001 / 0.00001 | 19 | 54 |
| 0.001 / 0.0001 | 17 | 72 |
| 0.001 / 0.001 | 14 | 61 |

Each context selected independently using its authorized source roles; these
frequencies do not select a new global configuration after held evaluation.

No inner fit reached the 200-epoch ceiling. The final-refit minimum of 30 epochs
and source-only median-checkpoint rule were applied as frozen; these results
do not justify retrospectively choosing different durations. Summed fit time
was 92,201.42 seconds (25.61 fit-hours), not exclusive GPU time or end-to-end
wall time. Final checkpoints occupied 814,176,960 bytes; the immutable context
store contained 3,520 files occupying 879,804,405 bytes.

## Comparison and scientific boundaries

P04 compares D0 with frozen C-SELECTED, RBF SVM, Random Forest, and Extra Trees
on exact common P02 test rows, retaining missing classical endpoints in coverage
and failure-sensitive summaries. M01 scores spectra; M06 gives physical masters
equal weight after instrument-view probability averaging.

The 5,000-draw paired master bootstrap is conditional on the observed domain
set. It uses pooled out-of-fold class recalls, whereas the endpoint table
averages fold-level balanced accuracies. Their point estimates need not be
identical. P11's final 10,000-draw hierarchical domain/master inference remains
outstanding, and no P04 baseline result alone passes G4 or proves equivalence.
The inherited fold-level BA averages recall over classes present in that test
cell; 40 of the 260 T3 cells contain two classes and one contains a single
class, while all 60 development cells contain all three. The pooled bootstrap
requires all three station-local classes within each domain. Neither the
frozen training metric nor the historical P03 predictions was changed.

### Frozen results

| Endpoint | Mean domain BA | Worst domain BA | Coverage |
| --- | ---: | ---: | ---: |
| Development, spectra (M01) | 0.780880 | 0.632492 | 60/60 |
| Development, masters (M06) | 0.915741 | 0.791667 | 60/60 |
| Held instrument, spectra (M01) | 0.711197 | 0.378558 | 260/260 |
| Held instrument, masters (M06) | 0.765491 | 0.366667 | 260/260 |

| Comparator | Common cells | Pooled M01 D0-minus-classical BA | Conditional 95% interval |
| --- | ---: | ---: | ---: |
| C-SELECTED | 252/260 | +0.050316 | [+0.022216, +0.078152] |
| Fixed RBF-SVM | 260/260 | +0.039410 | [+0.017874, +0.062467] |
| Fixed Random Forest | 260/260 | +0.022365 | [−0.016715, +0.057329] |
| Fixed Extra Trees | 260/260 | +0.005426 | [−0.023058, +0.032121] |

This supports a bounded advantage over C-SELECTED and RBF-SVM, not superiority
over all classical methods. On the fold-mean estimand, Extra Trees scored
0.715726 versus D0 0.711197. For C-SELECTED, the common-success means were
0.659086 versus D0 0.710406; retaining its eight unavailable cells with score
zero gave 0.638806 versus D0 0.711197 over all 260 planned cells. The fixed
classical comparators and D0 had full endpoint coverage.

Held-instrument spectrum NLL was 1.634888, Brier score 0.445802, and ECE
0.265709. Uniform three-class probabilities give NLL 1.098612 and Brier
0.666667. Thus squared probability error improved over uniform probabilities,
but log loss exposed costly confident mistakes. D0's common-cell mean NLL was
worse than all four classical comparators; probability calibration under shift
is not solved.

The comparison state is
`7cbbf38486c0a90c6911e477f4339e6f6f722e28b42c8a6edec32c6dd7d03ca2`.
Its persisted report passed every check. An initial console-print failure on a
NumPy boolean occurred after this successful commit; the CLI now uses the
project's canonical JSON serializer, with a regression test. This changed no
frozen comparison statistic or artifact.

The P13 extension scores the frozen D0 predictions on 15 eligible PP-U-MIN
substrate domains, including all 13 confirmatory domains. Exploratory
P13-DOM-005 is outside P04's frozen domain set. This extension supplies the
held-recovery endpoint and its P13 10,000-draw interval only. The matched-source
loss, deep SG/arPLS sensitivities, and full P13 D01/D02 portability experiments
remain outstanding. No P13 support tier or practical margin is changed.
P04 D0 fitting pools source substrates within a station, whereas the P13
classical fits are substrate restricted. The reused prediction comparison
therefore establishes shared test views, not matching training support; its
differences cannot be attributed solely to the learner. Exact P13 deep refits
are needed for that controlled comparison.
Seven of the 13 confirmatory views passed held recovery alone. P04/P13 test
subset parity passed in all 260 P04 contexts, while the union of eligible P13
substrate views equalled the full P04 test set in 139 contexts. The difference
is retained rather than treated as universal population equality.

## Figures, reproducibility, and delivery

F19 (architecture), F20 (source-only learning curves), and F48 (paired held-domain
effects) use a single semantic table each for native TikZ, vector PDF, 300-DPI
PNG, and self-contained HTML. Final native figures and browser-rendered HTML
were visually reviewed. `pdfimages -list` found no raster objects in any of the
three PDFs. Republished outputs were byte-identical across all 34 result and
figure artifacts. The public release validator passed its 19-file result
package and the linked three-figure hash manifest; the public scaffold audit
passed 364 files, and the P13 release still passed unchanged.

Public release manifest SHA-256:
`33b1705a5b1e4577a4aabd8336f51d3c39bc191544f03ce74a5e5f58f9fd03ea`.

Settled-source lint passed and the full test suite passed **190 tests with no
skips**. Aggregate and comparison CLI replays both passed and retained exactly
the same frozen-state hashes and modification times, confirming verified
skips. GitHub Actions attached to the scoped `main` publication commit is the
repository delivery record. No unrelated worktree
changes, row predictions, fold identities, or checkpoints belong to this
public commit.

## Next phase

G2 makes D0 a valid control for P05 source-only supervised-contrastive and
paired-consistency development. P05 must first expand its exact roles and fit
budget, verify pair support and sampling, and freeze its advancement procedure.
It cannot use the newly observed P04/P13 held outcomes to choose losses,
augmentation, preprocessing, or training policy. P04 has not trained D1–D5.
