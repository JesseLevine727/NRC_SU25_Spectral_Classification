# Frozen-prediction uncertainty specification

**Version:** 1. **Locked:** 2026-09-29, after benchmark outcomes and before resampling.
**Authority:** owner-approved documented amendment; see [readiness](P06P11_READINESS.md).

This is a post-benchmark analysis specification, not a new preregistration. It adds a support-preserving analysis without replacing the original hierarchical-bootstrap criterion. Models, predictions, splits, preprocessing, selection rules, reported scores and practical margins remain frozen. No fitting, calibration or prediction generation is authorized.

## 1. Evidence and contrasts

Authenticate the P05 reporting receipt `f09f066bbc46b7c1369b340b69ebed4758cf2587723392d80b665085ca0ecb9e` and its upstream bindings. The seed ensemble, P03/P04 reference shards, roles and context registry must match their pinned hashes. Reproduce saved endpoint, paired, coverage and summary tables before inference. Reconcile supported predictions against the roles, not a convenient row intersection.

The primary contrast is `P05-SELECTED minus C-SELECTED`, M01 balanced accuracy, on the 252 complete paired contexts across the same 13 domains. The eight incomplete contexts stay missing. All 17 existing `p05_comparison.PAIRS` contrasts are retained at M01 and M06, giving 34 contrast–endpoint combinations. The other 33 are secondary/exploratory. Contrasts use their own declared complete common contexts; no pooled cross-contrast support intersection is substituted. No best-performing model is selected from these results.

The eight existing methods remain the core panel. Additional saved PCA-LDA/prior-control or T1/T2 evidence may enter descriptive coverage/metric tables only after role/procedure authentication. They do not extend this resampling family. Missing selected-classical T1 evidence or neural T2/exploratory evidence remains explicit; no post-test choice of a fixed model fills the gap.

## 2. Fixed estimator

First average each neural model's probabilities over the three registered seeds, exactly as saved. For M01 each stored spectrum is a prediction unit. For M06 first average probabilities within each master/instrument, then equally across that master's instruments in the context, using the existing endpoint helper. Argmax follows the stored sorted class vocabulary. M06 averages predictions, not spectra. Neither endpoint pools predictions across outer contexts.

In each context, average prediction correctness within each originally represented true class, then average those recalls equally. Average context balanced accuracies equally within each domain, then domains equally. Paired differences use identical support and weighting for both methods. Originally absent classes stay absent; no assumed three-class denominator changes the point score. Unit resampling weights must match saved context and domain scores to absolute tolerance `1e-12`.

## 3. Approved positive-weight addition

Generate exactly 10,000 draws for each of three labelled modes: crossed master-and-instrument weights, master-only weights, and instrument-only weights. Each draw samples independent `Exponential(mean=1)` weights for the fixed global lists of 69 physical masters and ten instrument identities. A master weight is shared across all of its spectra, contexts, repeats, domains and methods. An instrument weight is shared across its station domains. The unused factor in either one-factor sensitivity is set to one.

For context c and original class k, weighted recall is the sum of master-weighted correct prediction units divided by the sum of master weights over all prediction units in that cell. Repeated spectra contribute their own units to M01; M06 contains one unit per master/context. Average recalls/classes and contexts as in Section 2. Average domain scores with their instrument weights, normalized over the fixed supported domain set. This weights shared acquisition identities without pretending that repeated contexts are new independent samples.

Use NumPy `Generator(PCG64(seed))` and lexicographically sorted string identity lists. The master stream uses seed `2026092901`; the instrument stream uses `2026092902`. Generate the two full weight arrays independently and reuse them across endpoints, contrasts and one-factor sensitivities. Do not derive seeds from Python hash, input row order or analysis outcomes. Reject nonfinite/nonpositive generated weights as an execution failure; do not redraw. Compute in batches of at most 128 draws; batching may not change weights or scores.

Report the 0.025 and 0.975 empirical quantiles using NumPy's `method="linear"`. These are approximate percentile uncertainty intervals conditional on saved fits and the observed design. They do not include retraining uncertainty, identify unobserved factor combinations, or establish finite-sample 95% coverage. Singleton cells retain their observed correctness; weighting cannot invent within-cell information.

BCa is unavailable for this addition (`crossed_cluster_acceleration_not_specified`). Do not substitute an observation-level jackknife for crossed master/instrument deletion. If all draws are identical, report the point interval with `degenerate_distribution` and explicitly state that this is not evidence of certainty.

## 4. Original hierarchical sensitivity and feasibility

Use a separate `Generator(PCG64(2026092903))`. For each contrast–endpoint combination, reset this stream; sort domains, classes and master IDs lexicographically. Generate a 10,000-by-D array of domain indices with replacement, where D is the fixed supported domain count. For each selected occurrence of a domain, independently draw N masters with replacement from each of that domain's original true-class pools of N distinct masters. Carry every occurrence of each sampled master across that domain's spectra and repeated contexts. Repeated selections of the same domain receive independent within-domain draws. This is the originally planned nested hierarchy; unlike the addition, it does not preserve shared masters across domains.

Implement the draw order as follows: draw the complete domain-index array first; visit source domains in sorted order; visit their selected occurrences in row-major order; visit classes in sorted order, drawing all selected occurrences of one class with `rng.multinomial(N, [1/N]*N, size=occurrences)`. This specifies the stochastic ordering independently of score-computation batches.

Recompute the fixed estimator. If any originally represented context–class cell in a selected domain occurrence has zero sampled weight, mark that occurrence and the overall draw undefined (`resampled_original_class_empty`). Do not drop the cell, insert zero, change its denominator, omit its context, or retry the draw. Report planned/defined/undefined draw counts, empty-cell counts and affected-domain counts.

Only if all 10,000 draws are defined may an unconditional percentile interval be reported. Otherwise its bounds remain missing (`hierarchical_fixed_support_undefined`); do not present quantiles conditional on the surviving draws as the original interval. No BCa interval is issued without a matching justified multilevel acceleration definition (`hierarchical_acceleration_not_specified`). This is a declared feasibility boundary, not a claim that software supplied a valid interval.

## 5. Stability and descriptive tests

For each contrast/endpoint, publish all paired domain scores/effects, their mean, median, quartiles (`method="linear"`), minimum and maximum, positive/negative/tied counts, and both methods' worst-domain scores on common support. Publish leave-one-domain-out effects and leave-one-instrument-identity-out effects, recomputing the equal mean over retained domains. These are deletions, not refits or new test partitions.

Enumerate all 2^13 domain sign assignments for full-domain contrasts and all 2^10 instrument sign assignments. Instrument signs are shared across all domains of that identity. Use the absolute equal-domain mean difference as the two-sided statistic; count ties using tolerance `1e-12`. Exact descriptive p is the fraction at least as extreme, including the observed assignment; no Monte Carlo correction is needed. Shared physical samples mean neither scheme establishes independent exchangeable domains or randomized treatment. Label both as symmetry-based sensitivity tests, not confirmatory p-values.

The sole primary contrast receives no adjustment. For transparency, apply Holm to the other 33 descriptive p-values as one explicitly post-benchmark exploratory family, separately for each sign scheme. Do not relabel these as the previously registered T1, preprocessing, robustness, adaptation or open-set families; those experiments are outside this analysis. No bootstrap tail proportion is presented as a null p-value.

## 6. Other metrics and decision gate

Report supported frozen-context macro-F1, NLL, Brier, confusion and class sensitivity. Reliability plots use fixed ten equal-width confidence bins with inclusive final bin; distinguish pooled repeated predictions from equal-context ECE. Any pooled counts are repeated appearances, not independent samples. Display support denominators and common-support versus full-support differences. No new calibration is fitted.

Generate a G4 checklist from evidence: M01 mean difference at least `0.03`; original hierarchical lower bound greater than zero; at least eight of 13 domains strictly improving; selected neural worst-domain BA minus classical worst-domain BA at least `-0.03`; authenticated T1 chemistry difference at least `-0.02`; unchanged primary input preservation. Each criterion is supported, failed or unassessable. An unavailable original interval or missing T1 comparator is unassessable, not passed. Report the added weighted interval beside the original criterion without using it to promote the method. Promotion requires every criterion supported. No result here establishes chemical/nuisance disentanglement or causal substrate independence.

## 7. Execution and publication guards

DeepSeek V4.1 Flash implements against public source/synthetic examples. The supervisor independently reviews pairing, identity, exact point reproduction, empty cells, random streams, resource use and test evidence. Preserve original artifacts and unrelated working-tree changes.

Before real analysis, run a bounded 100-draw synthetic dry run (separate seed `2026092904`, at most 60 seconds). This only estimates resources and tests execution; it does not inspect real inferential results. The full frozen-data analysis ceiling is 30 minutes of CPU wall time, 2 GiB resident memory and 1 GiB additional private artifacts, with one CPU thread and no CUDA. A guard failure stops execution and preserves diagnostics; no automatic scientific retry or expanded draw count. Separate figure rendering may use at most ten minutes, 2 GiB RAM and 1 GiB outputs; no scientific resampling during rendering. The supervisor records exact code/input/protocol hashes and software versions before launch.

Store prediction rows, identifiers, random arrays and draw-level outputs privately. Publish reviewed aggregates, reason codes, source hashes, source code, this specification and its review record. Native TikZ, offline HTML, vector PDF and PNG figures must share semantic aggregate tables. Scientific results remain unpublished until independent numerical and visual review, public-boundary checks, regression, exact-path main push and remote CI verification.
