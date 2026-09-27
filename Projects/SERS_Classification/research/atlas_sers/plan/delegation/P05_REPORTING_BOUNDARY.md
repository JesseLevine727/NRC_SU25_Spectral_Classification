# P05 reporting boundary

## Scope and timing

This document specifies descriptive reporting while comprehensive source training is still active. No new outer prediction or comparison with the historical prediction files has occurred. It implements the reporting section of [the comprehensive execution plan](../P05_COMPREHENSIVE_EXECUTION.md); it does not change the permit, numerical settings, source selector or statistical endpoints.

The report must distinguish implementation tests, source-selection evidence and held-out scientific results. Synthetic figure previews establish rendering behavior only. They must not appear as benchmark findings.

## Inputs and publication gate

Reporting follows completed comparison authentication. The authentication chain must establish complete source evidence, frozen context-local decisions, completed unique refits and source-only temperatures, frozen new predictions, reproducible aggregation and pinned historical comparisons. It must check the producer's actual receipts, manifests, tables and cumulative-time bounds. A retained completion receipt is insufficient if its stage subsequently failed.

Reporting performs no training, temperature fitting, prediction, recipe selection or preprocessing search. Scientific reporting and persistence inherit the cumulative execution clock and storage ceiling. Public artifacts contain allowlisted aggregates and anonymous plot indices, not observation or physical-master identifiers, private context/slot identifiers, source paths, labels, row-level probabilities, logits or checkpoints. Figures are rendered from the same allowlisted semantic tables used for numerical review.

## Fixed descriptive displays

1. **Paired performance.** Show all registered new-strategy/reference pairs for both M01 and M06. A paired scatter point is a station–instrument domain mean over common complete outer contexts. Use the same zero-to-one scales on both axes, an equality line and distinguishable markers for the three strategies. Domain-level paired-change displays use the same common contexts and retain missing-reference denominators. Missing data are not zero performance.
2. **Source versus outer performance.** Match each strategy to its already-frozen recipe within the same context. The source coordinate is the mean per-spectrum best-validation balanced accuracy across the three seeds within each inherited selection unit, then equally across units. Guard units are excluded from this coordinate. The outer coordinate is the registered M01 or M06 metric after three-seed probability averaging; its aggregation level is labelled. These are different estimators: the source score was used for selection, and the outer score comes from the refitted ensemble. Their difference is descriptive, not an unbiased estimate of a single transfer gap. Label pseudo-instrument validation and master-only validation separately; the latter does not establish instrument generalization.
3. **Selection and learning diagnostics.** Report recipe-selection counts, forced master-CV fallbacks and pseudo-domain fallbacks separately. Report source-fit counts, the fraction predicting fewer than two classes, best-epoch distributions and inherited refit epochs. A best epoch may precede epoch 30 even though the run must train for at least 30 epochs. Refit epochs remain the frozen clipped median; plotting an epoch distribution does not authorize a new training schedule. Strategy aliases count as strategy contributions, not additional neural executions.
4. **Reliability.** Show confidence against observed correctness separately by station, phase and endpoint for all three new strategies. Use up to ten equal-mass confidence bins, consistent with the existing ECE convention. The pooled diagram counts context–spectrum appearances for M01 or context–master appearances after instrument-balanced probability averaging for M06. Repeated appearances are not independent samples. Report its descriptive pooled ECE separately from the registered mean per-context ECE; pooling and averaging context-level ECE need not give the same value. No additional temperature is fitted.
5. **Coverage and computation.** Retain one-, two- and three-class outer-test support, missing historical references, source failures/collapse and actual fit/update counts. Read scientific time and execution counts from authenticated receipts. Distinguish reused pilot fits, new source fits, unique refits, calibration operations and aliased strategy endpoints. Do not infer elapsed time from a fit count or label all strategy endpoints as separate trained models.

Both development and held-instrument results remain visible and separately labelled. Report equal-context and equal-domain summaries explicitly. Worst-domain performance means the lowest domain mean, not the lowest individual split. Repeated splits and seeds do not increase the number of independent physical samples.

## Figure and prose acceptance

Every scientific figure has native TikZ, offline HTML and PDF/PNG previews with a shared semantic-data digest. Use black text, a conventional serif font, redundant colour/shape coding and readable support labels. Inspect compiled figures for clipped text, overlapping points, axis consistency and caption accuracy. HTML hover content must obey the same publication boundary as its static counterpart.

Each P05 figure-compilation child process is limited to 120 seconds or the remaining cumulative execution time, whichever is smaller; compilation does not retry. Reporting rechecks stored tables, numeric costs, figure hashes and the exact public-file inventory before stage closure. Unexpected files, directories or symbolic links cannot enter an accepted public subtree. This subtree remains inside the private artifact root until publication review.

The report must explain M01 as predictions on individual spectra and M06 as averaging model probabilities within each instrument, then equally across instruments for a physical sample. Neither endpoint averages the raw spectra before classification. Historical D0-ERM and the new matched D0-M control remain distinct.

State descriptive differences with their denominators. Do not add outcome-selected confidence intervals, hypothesis tests, a new winning-model rule or a claim of definitive superiority. Formal uncertainty remains P11. These experiments test classification robustness under the fixed minimal preprocessing policy; they do not establish chemical–nuisance disentanglement or resolve the deferred preprocessing and substrate questions.

Publication requires independent numerical and visual review and a verified push to main. Implementation acceptance, a completed source stage or a synthetic figure is not completion of the scientific benchmark.

## Sealed-publication handoff

The read-only publication gate requires the exact reporting-receipt SHA-256 accepted by the supervisor. It resolves only the fixed comprehensive run, checks the completed receipt and summary, verifies the stage manifest and upstream reporting bindings, and authenticates the exact public subtree. It returns relative public-file paths and hashes, not a recommendation to publish. It neither copies files nor recalculates model scores.

The returned table cells are lexical CSV strings. Numerical review must convert metric and count columns explicitly while retaining text identifiers and empty missing-value cells. For an overall equal-domain result, average the domain rows directly; averaging station means would assign the wrong weights when stations contain different numbers of domains. Paired differences use common complete contexts and retain their contributing-domain and context denominators. No missing reference is converted into a valid zero score.

Review must reconcile the 14 table inventories, both figure groups, source and refit update counts, reused pilot fits, unique refits and calibrations, strategy aliases and cumulative time. The cost table's time bound ends at comparison; the reporting receipt additionally includes reporting. The publication gate receives a remaining audit deadline and returns the sealed reporting bound without creating a new execution allowance. Numerical and visual review remain separate from implementation acceptance.

After review, copy only the authenticated public files into an unoccupied release directory and verify their destination hashes. Preserve the reporting receipt digest and the relative file/hash inventory in the public release record; exclude private receipts, manifests and provenance files. Do not overwrite an existing result release. The concise result narrative and figure index must cite these accepted aggregate files. Run public-boundary validation on the exact release snapshot, then push the reviewed files to `main` and verify GitHub CI.
