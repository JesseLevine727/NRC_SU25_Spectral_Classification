# P08 diagnostic reuse and figure-operation accounting

**Status:** prospective, no-fit reporting specification. Numerical reporting, figure rendering and scientific execution remain unauthorized. This inventory specifies primary diagnostic reuse and registered figure delivery; it does not claim that every experimental branch's numerical implementation has passed review.

The [machine-readable declaration](contracts/p08_reporting_accounting.json) binds the public evidence hashes, finite blocks, delivery dependencies and stage ownership. Its public-metadata tests verify these declarations without reading private rows or producing numerical outputs.

## 1. Reuse boundary

The [identity audit](../results/p08_readiness/preservation_reporting_support_audit.json) authenticates the frozen P01 preservation table against its retained state and artifact manifest, and the primary membership against the existing P08 input pin. It verifies all **4,784** historical diagnostic records across eight representations. The three primary actions contain **1,794 records**: one per action and each of the **598** original observations. Every record matches its observation, instrument, station and substrate-family metadata. No diagnostic value was evaluated, summarized or recomputed in this audit, and no spectral array was loaded.

These are reusable observed-spectrum diagnostics, not clean-chemical references. Their definitions remain those in `preprocessing/representations.py`: shape/rank correlation, spectral angle, first-difference roughness, baseline span, changed-point and boundary-value fractions, peak counts, peak recall and displacement. The boundary-value fraction is not proof of physical detector saturation; baseline span is a proxy, not an independently measured fluorescence contribution. Missing peak metrics remain missing. No numerical chemistry-preservation threshold is introduced.

The retained instrument/action summary has **80 rows**, including **30** for the three primary actions. It combines an instrument's stations and is not silently relabelled as a station–instrument-domain summary. The future domain/action view has **51 groups** across all **17** recorded domains and the three actions. It can summarize the existing diagnostic records without recalculating their spectral metrics. The four exploratory domains remain visible and distinguished from the 13 held-comparison domains.

Each domain/action summary retains finite and undefined counts, the median and the inherited 0.10/0.90 quantiles using linear interpolation for all eleven stored numeric columns. These summaries describe recorded row-level diagnostics; they are not master-equal chemical estimates. Report observation and physical-master denominators beside them. Master-equal averaging applies to the spectral display curves below, not retrospectively to the saved diagnostic definitions.

## 2. Probability and weakest-domain diagnostics

The accepted [case-scoring ledger](P08_PERTURBATION_PROTOCOL.md#13-sample-membership-pooled-scoring-and-curve-dependencies) already contains the scoring operations needed for balanced accuracy and the inherited probability diagnostics. Its **655,296 context/case** and **174,912 pooled/case** jobs are **830,208 existing jobs**, not an additional ledger of fits or predictions. Their numerical outputs must include supported macro-F1, NLL, Brier score, confusion, class recall and the fixed ten-bin reliability components under [the statistical protocol](P08_STATISTICAL_PROTOCOL.md#7-weakest-domains-probability-quality-and-preservation). Pooled jobs reconstruct prediction units from four disjoint folds before these calculations; they do not average fold confusion matrices or calibration scores without their required denominators.

The contrast point and weighted-batch operations carry the registered lowest-domain diagnostics: each required procedure's lowest domain accuracy, the corresponding policy-pair difference, and the minimum paired domain effect. These are different quantities and retain every tie. Interactions keep their four procedures rather than being mislabelled as a two-pipeline comparison. Shared-weight versions recompute minima within each draw; no extra significance family or worst-case guarantee follows.

The six robustness families remain separate. Directional curves, normalized loss areas, source-noise amplitudes and stochastic-repetition spread accompany absolute clean/stressed accuracy. Stochastic repetitions are averaged as scores, not as a new probability ensemble. Existing scoring and inference descriptors own these numerical calculations; figure renderers do not repeat them.

## 3. Finite reporting units and dependencies

Each reporting block below has a fixed index set, input identity and stop boundary. A block is an operation family, not a claim of completed work. Failed prerequisites produce explicit unavailable records; they do not reduce the registered index set or authorize retries.

| Block | Allocated units | Input and required output |
|---|---:|---|
| Preservation evidence authentication | 1 | Authenticate the retained table, aggregate, source definition, primary manifest and their frozen state bindings before any reuse. |
| Historical preservation-row references | 1,794 aliases, no new calculation | Action and original manifest-row order select exactly one authenticated P01 record. These are not independent samples or new metric jobs. |
| Domain/action preservation summaries | 51 | Use the same stored records and original domain membership; retain distributions, lower tails, undefined metrics and denominators. |
| Public spectral-cell preparation | 49 | One station–instrument/analyte cell, containing matched MIN/SG/arPLS membership and master-equal mean curves or an unavailable-display reason. |
| Private example preparation | Up to 49, conditional on approved private-example reporting | One preselected observation per cell, the same observation across all three actions. No selection from new model outcomes. |
| Fixed-route QC case summaries | 260 × 96 = 24,960 | Registered context/case route and action-validity records. Report original route, any invalid-action MIN-input fallback and complete-MIN-model fallback separately. |
| Source-noise display summaries | 13 | Group the authenticated context-level source-reference amplitudes by original held domain; retain all four quantile labels, context counts, median and range, and unavailable references. No held-noise fitting. |
| Registered public figure delivery | 11 × 8 = 88 | The fixed eleven figure bundles and the eight delivery stages below. Panel availability follows scientific support, not appearance. |
| Conditional private-example delivery | At most 49 × 8 = 392 | Private subviews of P08-F01, using the same delivery checks and no new hypothesis or public figure ID. |

The non-alias reporting ceiling is **25,603 operation descriptors**, including the conditional private-example preparation and delivery allowance. This is operation accounting, not a timing forecast. The existing **830,208** case-scoring jobs are referenced, not added again. The representation-integrity and clean-parity operations remain in their existing input/prediction graphs, also without duplicate accounting.

All membership selections are fixed by the private catalog whose digest is published in the identity audit. Domain/action keys follow sorted station–instrument domains and the fixed three action IDs. Spectral cells follow sorted station, instrument and analyte; their masters and observations use lexical recorded identities. Case summaries follow the existing context and 96-case registries. Public summaries contain approved aggregates only. A public count or hash does not authorize disclosure of the underlying row identities.

The reporting allowance belongs to its producing experimental stage: universal preservation and P08-F01–F04/F07 to U1; adaptive support/routing and P08-F05–F06 to Q1; robustness summaries and P08-F08 to S1; the remaining figures to their separately proposed range, normalization and population stages. Existing active-wall and artifact ceilings include reporting, failed attempts and logs. No new independent allowance is created here, and a successful stage does not launch another.

The QC case summary is shared across its four evaluation methods because one gate is frozen per outer context. It does not multiply the routing count by classifier. Source-noise display labels remain **0.50, 0.75, 0.90 and 0.95**, as registered in the disturbance protocol; no additional quantile is introduced for a plot. Unconditional reporting accounts for **25,162** descriptors; **441** private-example preparation/delivery descriptors remain conditional. Historical row references and existing scoring jobs remain aliases, not additional work in that sum.

## 4. Spectral display support

The primary manifest has **49 domain/analyte cells**. Of these, **46** contain at least two physical masters and can support the planned public aggregate; **three** must be labelled unavailable for that spectral display. They are not failed sensor measurements and remain in scientific analyses. The display therefore has at most **138 public mean curves** across the three actions, with unavailable cells retained in its support table.

Within each eligible cell, first average stored measurements within each physical master, then average those master curves equally. Do not renormalize the aggregate or present it as the model's input. Classification still uses individual spectra; the combined-prediction endpoint averages model probabilities, not these plotted spectra.

The audit fixes **49 private example observations** by the existing lexical selection rule, before new classifier outcomes. Those examples are illustrative, not statistically representative. Any approved private-example run uses the same chosen observation for all actions and does not substitute a more favorable trace. Their semantic tables, TikZ coordinates and embedded HTML payloads stay private.

## 5. Eight delivery stages per figure bundle

1. Prepare one frozen semantic-data bundle from the authenticated scientific outputs and fixed support definitions. A bundle may contain multiple specified panels; it is not one newly fitted model or one scientific score.
2. Review disclosure, units, independent-sample counts, missingness and caption claims. Public spectral cells retain the two-master minimum; private example bundles remain private.
3. Generate native TikZ/PGFPlots from the reviewed semantic bundle, with black standard-LaTeX or Times-compatible text and no embedded raster plots.
4. Generate self-contained offline HTML from that same bundle, with no remote scripts/fonts and no additional private hover fields.
5. Compile the native source to vector PDF; preserve build identity and private logs.
6. Produce a PNG review copy from that PDF.
7. Review TikZ, PDF/PNG and HTML for agreement in data, axes, categories, estimates, intervals, unavailable cells and claims. Compilation alone is insufficient.
8. Seal the figure manifest with semantic/source/output hashes, actual fonts, provenance and review status. Publication requires a separate accepted disclosure and visual review.

The source [figure manifest](P08_FIGURE_PLAN.csv) remains authoritative for the eleven questions and bundle IDs. No new numerical result or figure is produced by this readiness inventory. Full readiness still requires reconciliation with the statistical-operation ledger, final requirement-wide review and separate execution permission.
