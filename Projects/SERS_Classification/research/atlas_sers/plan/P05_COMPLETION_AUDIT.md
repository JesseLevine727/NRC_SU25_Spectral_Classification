# P05 core benchmark: completion and publication audit

## Scope

This audit covers the approved D0-M/D1/D2/D3 core benchmark, not the entire original P05/P06/P11 programme. Scientific execution completed on 2026-09-29. The [comprehensive protocol](P05_COMPREHENSIVE_EXECUTION.md), [recovery amendment](P05_COMPREHENSIVE_RECOVERY.md), frozen numerical contract and nested source-only selection rules remain unchanged.

DeepSeek V4.1 Flash authored the implementation and final narrative/navigation drafts. The supervisor independently reviewed code, controlled each launch, authenticated predecessor evidence, checked results and figures, and corrected the prose. All scientific stages ran from unchanged commit `fc0ca7b2ad732e0fedf466ba2eadc8b1aac1aa53`, with automatic restarts disabled.

## Scientific acceptance

| Boundary | Accepted evidence |
|---|---|
| Source recovery | 8,720 completed original fits reused; 6,184 recovery successes (one exact replay plus 6,183 unstarted fits); no recovery failure. Original interruption retained. |
| Source completion | 14,904 new successes plus 36 reused pilot fits = 14,940 evidence fits. The 14,905 new source attempts include the interruption. |
| Nested selection | All 320 decisions frozen before new outer predictions; 281 D0-M, 14 D1, 11 D2, 14 D3. There were 192 mandatory master-only fallbacks and 39 promotions among 128 transfer-selection-eligible contexts. |
| Final fitting | 1,995 distinct fixed-epoch refits and 1,995 source-only scalar calibrations; zero failures; 280,256 exact updates. The 2,880 strategy/seed aliases are not distinct trained models. |
| Prediction | 1,995 unique prediction executions, 35,409 private seed-level rows, all 320 contexts complete. These rows are not independent samples. |
| Aggregation | Three-seed probability averaging, M01 spectrum and M06 instrument-balanced physical-master endpoints; all 960 context/strategy coverage checks passed. |
| Historical comparison | 260 held contexts, 17 registered pairs, two endpoints. All new strategies and four fixed historical references have 260/260 coverage. C-SELECTED has 252/260; two CWA and six surfaces references are incomplete/missing. |
| Reporting | 14 allowlisted aggregate tables, cost JSON, 12 paired and 23 diagnostic figures. No new fitting, calibration, prediction or optimizer update. |
| Publication authentication | Independent read-only consumer passed actual receipt, source bindings, table schemas/counts, cost reconciliation, figure bytes and exact inventory. All 159 copied public files match their authenticated source hashes. |

The [public release record](../results/p05_comprehensive/release_manifest.json) retains the reporting receipt digest and relative file hashes. It does not expose private manifests or provenance. Model states, raw spectra, row-level probabilities, physical-master/observation identities and operator/source paths remain private.

## Results and interpretation checks

The [report](../results/p05_comprehensive/P05_RESULTS.md) derives overall held performance by averaging the 13 domain rows directly, not equally weighting three station means. Paired comparisons use common complete contexts. Missing references remain missing; no historical model was refitted.

Fixed D3 improves mean balanced accuracy over D0-M by 1.68 percentage points at M01 and 1.32 at M06. Source-selected improvements are 0.17 and 0.04 points. D3's M01 domain effects are six positive, six negative and one tie. Its higher mean does not establish general superiority or authorize outcome-based promotion. Tree ensembles remain strong, particularly at M06.

Held class support is 220 three-class, 39 two-class and one one-class context; balanced accuracy covers observed true classes. Repeated contexts, seeds and appearances are not independent chemical samples. Source-only promotion is not held-out benefit. M06 averages predictions, never raw spectra. Pooled reliability ECE and mean per-context ECE remain distinct. No new confidence interval, significance test or causal disentanglement claim was introduced.

## Computation and interruption accounting

The benchmark completed 16,899 new neural fits in 16,900 attempts. The one original interrupted attempt has an uncertain final update count; its observed history is a lower bound, not an exact terminal count. New successful updates total 3,149,052. All-attempt observed and charged bounds are 3,149,120 and 3,149,852; the reused pilot's 6,880 updates are excluded from these new-work counts.

The cumulative scientific bound through reporting is 73,686.14479080099 seconds; independent publication authentication adds 35.06945816401276 seconds, giving 73,721.214248965 seconds (20.48 hours) within 48 hours. This includes conservative earlier charges and is not measured GPU-hours. The public cost table ends at comparison; the release record additionally includes reporting and publication authentication. The 100-GiB governed storage checks passed at stage closure. The approved outer GPU ceiling is 8 GiB; the frozen inner guard remains 4 GiB.

## Numerical, visual and prose review

The immutable execution release had passed 2,472 regression tests with four CUDA-dependent skips before launch. The public follow-up revision had the same passing full CI; its changes did not alter scientific code. These software checks are separate from the actual stage authentication above.

The supervisor reviewed all 35 rendered figures in contact sheets and inspected full-size representatives of paired performance, domain changes, source-versus-held results, epoch distributions and reliability. Actual offline HTML was opened in headless Chrome for paired and six-panel source-versus-held figures. Native PDFs use embedded Computer Modern fonts; text is black, and marker shapes supplement colour. Small epoch frequencies require zooming in HTML; no data-dependent axis change was made to sealed figures.

Lenarizer review corrected ambiguous probability-averaging language, control naming, endpoint units, loss availability, time/ceiling accounting and over-broad comparison wording. The preservation check flagged removal of duplicated numbers and addition of the explicit 0.01-point rounding explanation/figure count; every affected value was manually reconciled to the accepted tables. No scientific value or evidentiary strength was changed.

Release validation passed against the clean staged snapshot: 784 public files checked, with all 159 new scientific files matching the authenticated hashes. All 140 figure-format links are indexed and resolve; the seven-model, two-endpoint headline table matches the direct 13-domain aggregates. A focused regression of publication, public metrics, source diagnostics, reliability and both figure renderers passed 254 tests (146 expected sparse-class warnings) in 34.49 seconds. No production code, tests, contracts or registries changed in this release.

Only the explicit release paths and reviewed documents enter the commit. Unrelated legacy workspace files and ignored older LaTeX build logs are excluded. The owner-worktree scaffold check flagged those older build by-products; the exact clean release snapshot passed without changing or deleting them. Remote main and CI must still be verified after this pre-push acceptance record.

## Remaining boundary

The completed core supplies descriptive held evidence. Remaining P06 synthesis and P11 grouped/domain-aware uncertainty are needed before a definitive superiority claim. P08 preprocessing, D4/D5, adaptation, P14 RBF/SOM work and controlled P13 substrate-restricted deep refits remain outside this execution. No new fits follow automatically. Remote main and CI verification are the final repository handoff gate.
