# Publication strategy for the RBF/SOM extension

**Date:** 2026-09-24.

**Status:** literature-informed publication plan; no P14 results, novelty claim, journal acceptance, or completed external validation are implied.

**Protocol:** [P14 extension](P14_RBF_SOM_EXTENSION.md).

**Current project evidence:** [P03 classical](../results/p03_classical/P03_CLASSICAL_RESULTS.md), [P04 compact neural](../results/p04_deep/P04_RESULTS.md), [P13 portability](../results/p13_portability/P13_RESULTS.md).

## 1. Recommended scientific story

The strongest initial paper is an integrated study of chemical identification and reliability under field acquisition shift, using the repeated physical samples to explain why methods succeed or fail. RBF networks add a controlled local-similarity model; SOMs add inspectable spectral prototypes and a map of acquisition dependence. The experiment should determine whether these additions change a chemical measurement conclusion, not simply enlarge an accuracy leaderboard.

A working umbrella question is:

> How do preprocessing, prototype structure, and acquisition-aware learning affect analyte identification and reliability when a field SERS instrument and its test samples are excluded from training?

This umbrella organizes a manuscript; it does not replace the already locked RQ-P01 primary hypothesis. The RBF/SOM questions are explicitly dated post-outcome extensions. Main-text claims must distinguish the original comparison from these additions.

The publishable contribution could be a combination of: a reproducible multi-instrument benchmark with physical-master isolation; direct evidence from repeated views; a measured tradeoff between simple prototypes and larger learned representations; and an operational account of preprocessing, confidence, and supported substrate conditions. The field data are small and incompletely crossed. Claims should therefore be limited to this observed acquisition setting unless an independent validation study supports more.

RBF networks and SOMs are established algorithms. Merely being the first implementation in this repository is not methodological novelty. A useful manuscript must explain what is learned about spectroscopy, transfer, or reliable measurement that the existing papers did not establish.

## 2. Literature basis and its limits

This is a targeted primary-source scan, not an exhaustive systematic review. Source access and publisher scopes were checked for this planning update on 2026-09-24; some publisher full-text pages were unavailable. The notes below rely on accessible papers, abstracts, and institutional versions. No published accuracy is used as a performance target for this dataset.

| Source | Evidence relevant to this plan | Design consequence and limitation |
|---|---|---|
| West, Stepney and Hancock, 2025, *Unsupervised self-organising map classification of Raman spectra from prostate cell lines uncovers substratified prostate cancer disease states*, Scientific Reports | SOM analysis of live-cell Raman spectra identified biochemical subgroups using minimal preprocessing | Supports spectral prototype exploration; it does not establish transfer across our instruments or prove that visible clusters are chemically pure. [Paper](https://www.nature.com/articles/s41598-024-83708-6) |
| Banbury et al., 2019, *Development of the Self Optimising Kohonen Index Network (SKiNET) for Raman Spectroscopy Based Detection of Anatomical Eye Tissue* | Combined mapping, feature interpretation, and supervised refinement for Raman tissue classification | SOM classification is established; retain ordinary SOM first and label any later supervised refinement separately. Its tissue evaluation differs from master-grouped instrument exclusion. [Institutional publication record and paper](https://research.birmingham.ac.uk/en/publications/development-of-the-self-optimising-kohonen-index-network-skinet-f/) |
| *One class self-organizing maps with Monte Carlo permutation for gemstone classification using Raman spectroscopy*, 2025, npj Heritage Science | Investigated a one-class Raman SOM approach | Motivates a later acceptance/rejection analysis, not a claim that prototype distance distinguishes unfamiliar chemistry from field-quality shift. [Paper](https://www.nature.com/articles/s40494-025-02070-6) |
| Bolanča et al., 2012, *Development of Artificial Neural Network Model for Diesel Fuel Properties Prediction using Vibrational Spectroscopy* | Used feedforward and radial-basis networks with IR/Raman variables for fuel-property prediction | RBF spectroscopy is not new; this regression application does not settle small-sample SERS classification under instrument exclusion. [Paper](https://acta-arhiv.chem-soc.si/59/59-2-249.pdf) |
| Wurzberger and Schwenker, 2024, *Learning in Deep Radial Basis Function Networks*, Entropy 26, 368 | Studied initialization and training of deep RBF architectures, with image and speech experiments | Supports a possible later neural extension, but provides no direct SERS transfer evidence. Start with a shallow finite-centre model so depth is not confounded with prototype construction. [Paper](https://doi.org/10.3390/e26050368) |
| Wang et al., Analyst 2025, *Functional regression for SERS spectrum transformation across diverse instruments*; first published online December 2024 | Evaluated mapping between standard and target instrument spectra, using training analytes measured on both systems | Confirms that instrument transfer is a substantive SERS problem. Its access to target-instrument training spectra differs from the current zero-shot question; any analogous mapping belongs in the separate target-access branch. [Paper](https://pubs.rsc.org/en/content/articlepdf/2025/an/d4an01177e) |

The inferred opportunity is a controlled study connecting prototype geometry, repeat consistency, preprocessing, and held-instrument prediction on a sparse field design. This scan does not justify the phrase “first ever,” nor does it justify treating an RBF-SVM as an RBF neural-network precedent without checking the actual architecture.

Before manuscript submission, refresh the search for combinations of Raman/SERS, self-organizing/Kohonen/SKiNET/LVQ, RBF/prototype/codebook classifiers, grouped validation, calibration transfer, and acquisition shift. Record which papers used independent samples, excluded instruments, target calibration, and explicit failure accounting. Review methods and supplements before making novelty comparisons; a title or abstract is insufficient for claiming a validation gap.

## 3. Minimum complete study versus optional extensions

The P14 minimum package is EXP-P14-00–05 and synthesis EXP-P14-10: verified roles/support; ordinary SOM and K-means geometry; RBF/SOM/linear and established comparator results; repeated-view diagnostics; universal MIN/SG/arPLS effects; and physical-master learning curves. Show both gains and losses. A negative new-model result does not make this package incomplete.

P14-06 error detection can strengthen operational relevance if independent source calibration is supported. P14-07 SOM-centred RBF and P14-08 encoder-head comparisons are later mechanism tests. P14-09 substrate-restricted refits answer the field-trial purpose but must obey the existing P13 design. Broader adaptive preprocessing, unknown chemistry, or new deep RBF layers are not necessary to finish the first extension.

The whole project still needs the planned P05/P06/P11 work before a final claim about its original acquisition-aware deep hypothesis. If a manuscript is submitted before that comparison, its scope must be explicitly the completed benchmark/prototype study, and no result of the unfinished primary hypothesis is implied. Keep a documented completion/deferred table instead of continually adding models to avoid writing.

## 4. Evidence-to-publication routes

| Evidence outcome | Defensible paper emphasis | What remains necessary |
|---|---|---|
| Original acquisition-aware model passes the existing G4 rule | Methods paper explaining repeated-view learning under acquisition shift, with RBF/SOM as strong controls and interpretation | P05/P06/P11 completed; original primary effect and uncertainty; comparison with forests; full limitations |
| RBF adds useful performance, cost, or reliability, while the original deep promotion is not established | Prototype-supported chemometric benchmark or a clearly labelled secondary method finding | Named comparator effects, identical test roles, capacity/sample-efficiency evidence, calibration, and disclosed post-outcome timing |
| Forests remain strongest and RBF/SOM clarify failure conditions or preprocessing tradeoffs | Analytical benchmark with interpretable acquisition dependence and a negative neural result | Concrete chemical/measurement insight beyond a ranking; stable grouped results, paired spectra, and uncertainty |
| Apparent geometric improvements coincide with lost chemistry or fragile held performance | Measurement-limit and failure-analysis study | Demonstrate the failure rather than infer it from a pretty/poor map; specify which acquisition cells lack support |
| A hybrid shows a reproducible mechanism-specific advantage | Possible later methods-focused paper | Equal-budget ablations and preferably independent dataset/instrument replication; a new architecture label alone is insufficient |

No route guarantees acceptance. Lack of statistical separation is inconclusive, not proof of equality. A result selected as the best of many added models is not the original primary result. Do not manufacture a positive novelty claim by switching the headline endpoint from individual spectra to combined-sample predictions.

Recommend one coherent manuscript initially. A second paper is reasonable only if a later extension has a distinct question, enough independent evidence, and a clear relationship to the first study. One paper per algorithm would fragment the shared evidence and exaggerate novelty.

## 5. Main-text figures and supplementary allocation

Choose the type of evidence now; fill it with all declared results later. Panel allocation may change for readability, but the scientific contrasts and missingness cannot disappear during writing.

| Main-text figure | Proposed panels | Scientific purpose |
|---|---|---|
| 1. Dataset and evaluation | Existing source flow/F44 support matrix plus held-instrument/held-master schematic | Establish what is independent and which factor combinations were observed |
| 2. What spectral prototypes represent | F-P14-01 map recolouring and F-P14-02 measured/prototype spectra; PCA PC1–PC2 reference | Explain whether organisation follows chemistry or acquisition |
| 3. Which methods transfer | Existing primary results plus F-P14-04 paired domain dots/intervals, including forests and failures | Answer the predictive question without hiding weak instruments |
| 4. What preprocessing changes | F-P14-05 paired effects and spectral examples; supported F-P14-03 repeat connections | Test whether reduced acquisition structure preserves useful chemistry |
| 5. Data and model requirements | F-P14-06 master-count curves and F-P14-07 prototype/cost tradeoffs | Quantify how much support and model complexity the observed task needs |
| 6. Reliability or bounded portability | F-P14-09 if completed; otherwise existing calibrated-confidence or P13 recovery/loss evidence | State when predictions are unreliable and how far the conclusions extend |

F-P14-08 stability, every seed/map configuration, complete learning subsets, confusion matrices, per-analyte recall, all failed/unsupported runs, full search budgets, and F-P14-10 optional ablations belong in the supplement. The existing P13 results must be contextualized if included; they are not new P14 discoveries. Cite exact source roles if comparing substrate views, because pooled-source D0 reuse and substrate-restricted refits answer different questions.

Prioritize spectra, connected scatter plots, paired effects, and learning curves. A bar chart is used only when it communicates a specific relationship better. Native TikZ and offline HTML are mandatory counterparts; retain common plot data, units, axis limits, uncertainty, and denominators. Figures intended only for internal discussion must not silently become public row-level releases.

## 6. Manuscript structure and possible titles

1. **Introduction:** field SERS instrument variability; why ordinary within-instrument accuracy is insufficient; repeated observations as an opportunity; exact primary and extension questions.
2. **Dataset and experimental design:** curation, 69-master unit, station-conditioned tasks, incomplete substrate matrix, existing outcome exposure, role freezing and information access.
3. **Methods:** classical/D0/acquisition-aware comparators as completed; shallow RBF and ordinary SOM; source-only fitting; MIN/SG/arPLS factorial; paired and master-count experiments; calibration and inference.
4. **Results:** supported chemical signal and acquisition structure; matched predictive comparisons; preprocessing and repeats; capacity and reliability; unavailable outcomes.
5. **Discussion:** chemical interpretation tied to spectra; relationship to field-trial substrate goals; when additional neural complexity helps or fails; transfer limitations and what future acquisitions would resolve.
6. **Reproducibility and conclusion:** versioned code, data access, complete aggregate evidence, exact claim boundary, and future work separated from completed results.

Result-neutral working titles:

- “Chemical identification across instruments in field-trial SERS: preprocessing, spectral prototypes, and predictive reliability.”
- “Spectral prototypes and acquisition dependence in a multi-instrument SERS field dataset.”

Use a stronger title about improved transfer only if the corresponding named contrast supports it. Avoid “instrument-independent SERS,” “chemical disentanglement,” or “denoised chemical spectra” as descriptions of these experiments.

## 7. Journal strategy

Venue choice follows the evidence and analytical contribution. These are fit recommendations, not acceptance predictions; do not use impact factors or publication speed as a substitute for scientific fit.

- **Journal of Chemometrics:** a natural candidate if the principal contribution is rigorous method comparison, sample efficiency, prototype interpretation, or reliable validation on difficult chemical data. Its official scope includes fundamental and applied chemometrics and data science in chemistry. [Publisher scope](https://analyticalsciencejournals.onlinelibrary.wiley.com/hub/journal/1099128x/homepage/productinformation.html)
- **Analyst:** suitable to consider if the completed study produces a clear new insight or method for analytical measurement under acquisition shift. Its scope explicitly includes spectroscopy and ML/data processing, but excludes routine applications of existing methods without significant novelty. The manuscript therefore needs measurement insight beyond trying RBF/SOM on another dataset. [Publisher scope](https://www.rsc.org/publishing/journals/analyst)
- **Analytical Methods:** consider if the strongest outcome is a practical validated workflow or a specific operational improvement. Its author guidance accommodates full papers and technical notes with definite practical advantages. [Publisher guidance](https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/analytical-methods)

Chemometrics and Intelligent Laboratory Systems is an additional venue to review at manuscript time; its official scope page was inaccessible during this update, so no unverified editorial requirements are asserted here. The initial recommendation is a chemometrics/analytical-science audience, with a stronger methods venue considered if novelty and validation warrant it.

## 8. Evidence gates before writing claims

- Every chemical-generalization claim counts physical masters and uses the correct held role.
- All-data descriptive maps are visibly separated from source-trained predictive maps.
- Each headline comparison uses the same aggregation, source/test roles, and denominator; historical headline scores with different pooling are not subtracted.
- Chemical associations are evaluated within station and supported conditions; substrate labels do not substitute for chemical evidence.
- Low prototype distance, lower instrument predictability, better map topology, or high classification confidence alone is not a successful chemical result.
- Model size includes preprocessing and prototype state as well as supervised weights. Lower cost is measured on the same hardware and evaluation scope.
- New-chemistry rejection is claimed only after its independent held-chemical experiment; ordinary error detection cannot substitute for it.
- The manuscript discloses earlier test exposure and the date at which this extension was designed. Existing-domain bootstrap intervals are not evidence about arbitrary unmeasured instruments.
- Strong new-method claims should seek an independent replication dataset or later untouched acquisitions where feasible. This is an evidential strengthening step, not a claim that new data currently exist or an authorization to collect them.

The next concrete deliverable is the P14 no-fit audit and implementation specification. Its first scientific outputs should make the data geometry and model comparison inspectable; the manuscript direction is then decided from the complete core evidence, including negative findings.
