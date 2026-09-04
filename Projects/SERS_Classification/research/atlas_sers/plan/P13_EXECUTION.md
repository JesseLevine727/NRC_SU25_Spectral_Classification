# P13 classical substrate-portability execution

**Protocol:** `nato-sers-p13-v1-locked`

**Outcome-blind execution plan:** `P13PLAN-c0d28caeacb8088107fc09bc`

This document records how the locked P13 design is converted into executable
roles. It supplements, and does not rewrite, [P13_PROTOCOL.md](P13_PROTOCOL.md)
or [P13_FREEZE_MEMO.md](P13_FREEZE_MEMO.md).

## No-fit gate

Before any P13 estimator was fitted, the execution expansion reconstructed and
validated:

- 960 domain × preprocessing × repeat × fold contexts;
- 3,200 unique fit, calibration, held-test, and matched-source role records;
- 6,720 context-specific procedure records;
- 42,360 individual fit invocations, including every declared technical tree
  seed and source-only calibration fold;
- 1,680 domain × preprocessing × procedure × repeat endpoints; and
- 240 independently resumable domain × preprocessing × repeat shards.

The gate checks the immutable 598-spectrum/69-master population, the five
repeated four-fold P02 physical-master splits, representation UID order and
shape, source-only candidate provenance, held-instrument exclusion, outer and
calibration master disjointness, endpoint accounting, and exact support-tier
cardinality. All checks passed with zero model-fit invocations.

## Candidate resolution

For the 13 confirmatory station–instrument combinations and the two pills
exploratory combinations represented in P03, `C-SELECTED` reuses the exact P03
`EXP-C09-T3` source-only selected candidate for the same repeat and fold. Each
fixed family reuses the lexicographic source-only winner within that family.
The candidate remains fixed across PP-U-MIN, PP-U-SG, and PP-U-ARPLS so the
preprocessing sensitivity does not also change the model.

The exploratory CWA/H-SERS/Agilent-3 domain had no P03 source-only selection.
Its `C-SELECTED` endpoint is therefore retained as unavailable. The six named
fixed families use their first declared family candidate there and remain
exploratory. No held-out outcome is used to borrow, replace, or select a
candidate.

## Fit and prediction sequence

For every context and procedure:

1. fit the frozen candidate on each source-only calibration-fit role;
2. predict its disjoint source-only calibration-validation role;
3. average the three declared technical-seed score vectors for Random Forest
   and Extra Trees;
4. fit one scalar temperature to the combined cross-fitted source scores;
5. refit every declared seed on the complete outer source-fit role;
6. predict the held-instrument role and the matched source role only after
   fitting and calibration are complete; and
7. average technical-seed outputs before applying the frozen temperature.

Every planned fit receives one terminal status. A failed calibration or final
seed makes that fold endpoint unavailable; no replacement candidate is fitted.
An outer fold containing no held view is recorded as an empty fold by design,
not as a model failure. All other folds and all declared denominators remain.

## Aggregation and inference

For each stored spectrum, calibrated out-of-fold probabilities are averaged
over the five outer repeats. Technical repeats are then averaged within the
master–substrate–instrument view. Source instruments are equally weighted when
forming a matched source prediction for a physical master.

The domain endpoint is three-class held balanced accuracy. Matched loss is
source balanced accuracy minus held balanced accuracy on exactly those outer
test masters having both views. The 10,000-resample bootstrap is stratified by
analyte and clustered at physical master. BCa intervals are used when their
jackknife acceleration is stable; otherwise percentile intervals are retained
with the method named. The locked `tau = 0.60`, `delta = 0.10`,
intersection–union, Holm, terminal-failure, common-endpoint, and chance
sensitivity rules are then applied.

## Secondary analyses

The crossover analysis uses the canonical A/B ordering in the frozen support
registry. Its predictive contrast is

`[(B − A) at instrument B] − [(B − A) at instrument A]`

for correctness and calibrated true-class probability. It also reports the
difference between substrates in their PP-U-MIN cross-instrument cosine
distance. Predictive contrasts require all four held-view cells; unsupported
blocks remain visible, and singleton blocks have descriptive points without
interval claims.

The field-log analysis uses the master–substrate–instrument view. Agreeing
technical-repeat log values define the view value; conflicts are ambiguous.
Nonblank `Y` is detection success, blank `N` is specificity success, `M` and
conflicts are excluded from the definite endpoint, and truly missing logs
generate worst/best missingness bounds. Model–field agreement compares this
recorded success indicator with correct/incorrect analyte classification on the
same held view.

## Reproduction

With the three private roots configured outside the repository:

```bash
python scripts/run_p13.py plan
python scripts/run_p13.py validate-plan
python scripts/run_p13.py execute-batch --worker-index 0 --worker-count 1
python scripts/run_p13.py aggregate
python scripts/publish_p13_results.py
```

Multiple workers may execute disjoint modulo partitions by assigning unique
`--worker-index` values under the same `--worker-count`. Each shard uses an
atomic commit, validates hashes on restart, and returns `verified_skip` when it
is already complete.

## Execution closeout

Execution `P13-3d21aa17c7d6cd750ca9d286` completed on 2026-09-04. All 240
shards and 42,360 fit records reconciled, aggregation passed every registered
check, and the aggregate release and F45–F47 were generated. This section is a
post-outcome status note; it does not alter the no-fit contract above. See
[P13_COMPLETION_AUDIT.md](P13_COMPLETION_AUDIT.md) for the evidence matrix and
[P13_RESULTS.md](../results/p13_portability/P13_RESULTS.md) for the bounded
scientific interpretation.
