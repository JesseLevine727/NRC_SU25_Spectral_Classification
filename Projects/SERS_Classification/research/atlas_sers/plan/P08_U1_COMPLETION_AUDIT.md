# P08-U1 universal benchmark completion audit

**Evidence date:** 2026-10-09. **Scope:** universal MIN versus frozen SG/arPLS
only. This is not completion of every P08 branch.

| Requirement | Current evidence |
| --- | --- |
| Frozen data, grid and preprocessing | Authenticated 598-spectrum primary population; 400–1800 cm⁻¹, 1,401 channels, inherited operation order, final [0,1] scaling; no combined SG/arPLS arm. |
| Full fixed model panel and support | RBF-SVM, Random Forest, Extra Trees, ordinary CNN and frozen context-local source-selected CNN; 260 contexts, 13 held domains, both endpoints. |
| No sample/instrument leakage | Completed reader verifies full role membership, held-instrument exclusion and master separation; all selection, calibration and stopping depend on source data only. |
| Actual training, not just a smoke test | 195,202 unique fit slots, 3,354 scalar calibrations and 404,814 operations complete; zero remaining/running; clean store and no shutdown errors. |
| Finite budget and retained recovery evidence | Latest approved 12 CPU/1 GPU, 44-GiB process-tree RAM, 8-GiB allocated GPU; unchanged inner 120-second/4-GiB neural guard; 48-hour/80-GiB cumulative caps. Final release accounting is in the release manifest. Historical attempts are retained, not counter-reset. |
| Epoch/loss monitoring | Private per-epoch monitor retained; public aggregate training figures cover 6,366/6,402 expected histories, with 36 missing historical pilot histories explicitly disclosed. |
| Numerical endpoints and source-only choices | 3,900 complete report cells, 3,237 distinct endpoints; selected/D0 aliases retained; probabilities combined in M06, not spectra. |
| Statistical and sensitivity audit | 10,000 shared weighted draws; original hierarchy feasibility and sign/Holm sensitivities retained; both estimands; all registered available contrasts and future-QC missing slots. |
| Probability, weakest-domain and preservation reporting | Nine aggregate tables plus diagnostic JSON, 51 preservation groups, 49 spectral cells and 352 separately labelled family-deletion rows. |
| Required visual deliverables | F01–F04/F07 plus training diagnostics: 117 panels each in native TikZ, vector PDF, PNG and offline HTML, from shared semantic data. |
| Review and privacy | All-panel hash/bounds/browser checks; direct representative inspection across all figure types; two-master spectral disclosure rule and aggregate-only exports. Details in the release review. |
| Concise report | [NATO SERS universal-preprocessing report](../reports/NATO_SERS_UNIVERSAL_PREPROCESSING_REPORT.md), with two score tables, plain-language interpretation and figure links. |
| Reviewed implementation | 47 promoted Python files; final targeted 422 tests + 40 subtests; full-suite environmental failures individually covered by corrected reruns. |
| Main publication and CI | Final transport gate: verify the release commit is on `origin/main` and the NATO SERS research workflow succeeds before declaring the persistent goal complete. |

## Authoritative public entry points

- [Results and interpretation](../reports/NATO_SERS_UNIVERSAL_PREPROCESSING_REPORT.md)
- [Figure browser](../results/p08_universal/release/figures/index.html)
- [Release review](../results/p08_universal/release/RELEASE_REVIEW.md)
- [Artifact/code hashes and resource accounting](../results/p08_universal/release/release_manifest.json)
- [Execution authority and recovery amendments](P08_U1_EXECUTION.md)
- [Implementation history](delegation/P08_U1_REVIEW.md)

No superiority gate is passed. Universal baseline correction improves the
observed mean for every tested strategy, but not every domain. These results
do not establish clean-spectrum recovery, instrument-independent substrate
chemistry, an optimal per-instrument rule, or performance on arbitrary new
instruments. QC-adaptive/family policies, disturbances, range, normalization,
population sensitivities and new architectures were not run by this goal.
