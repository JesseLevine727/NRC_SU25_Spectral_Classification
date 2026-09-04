# NATO SERS P13 substrate-portability results

**Protocol:** `nato-sers-p13-v1-locked`

**Execution run:** `P13-3d21aa17c7d6cd750ca9d286`

**Outcome access:** after the P13 design, support, split, model-role, threshold,
inference, and failure-handling registries were locked on 2026-09-04

**Scope:** classical experiments `EXP-P13-C01` through `EXP-P13-C04`

## Bottom line

The locked classical study does **not** establish that any substrate family is
instrument-independent. Under the primary source-selected procedure and
minimal preprocessing, no confirmatory evaluation domain satisfied both
portability bounds. The pSERS Metrohm silver family has direct evidence of
inferior portability in two tested domains; the H-SERS H-Kit result is
unresolved because one domain was inconclusive and two primary endpoints were
unavailable after predeclared terminal failures. The other two substrate
families had no confirmatory domains.

This is not a finding that the substrates contain no chemical signal. Several
individual analyte cells and several instrument domains showed strong
held-instrument recovery. Rather, performance was heterogeneous across
instrument, substrate, station, and analyte, and the evidence did not meet the
much stronger requirement that every confirmatory domain for a substrate pass
both a recovery and a matched-loss bound.

The fixed RBF-SVM sensitivity completed all 13 confirmatory domains and reached
the portability state in two pSERS pill domains, but it also showed three
inferior domains and eight inconclusive domains. It therefore reinforces the
primary conclusion: portability is condition-specific, not universal over the
tested instruments.

## What was evaluated

- Population: 598 spectra from 69 independent physical masters.
- Design: five repeated four-fold physical-master splits inherited from P02.
- Evaluation domain: station × normalized substrate family × held instrument.
- Primary unit: average technical-repeat probabilities within each
  master–substrate–instrument view.
- Information regime: the held instrument was excluded from transformation
  fitting, estimator fitting, source-only selection, calibration, and stopping.
- Primary model procedure: source-only `C-SELECTED` from the P03 freeze.
- Main model sensitivity: fixed `C-RBF-SVM`.
- Primary representation: `PP-U-MIN`, measured-support interpolation over
  400–1,800 cm⁻¹ followed by per-spectrum min–max scaling.
- Preprocessing sensitivities: universal Savitzky–Golay (`PP-U-SG`) and arPLS
  (`PP-U-ARPLS`) on identical eligible views.
- Inference: 10,000-resample physical-master-clustered hierarchical bootstrap;
  95% BCa intervals where stable and percentile intervals otherwise.

Portability required both

`LCB95(held balanced accuracy) >= 0.60`

and

`UCB95(matched source balanced accuracy − held balanced accuracy) <= 0.10`.

The substrate-family decision was intersection–union: every confirmatory
domain for that family had to pass both conditions.

## Execution and accounting

| Item | Count |
| --- | ---: |
| Deterministic execution contexts | 960 |
| Registered procedure/fold endpoints | 6,720 |
| Registered fits | 42,360 |
| Complete fits | 41,917 |
| Failed or protocol-excluded fits | 443 |
| Observation-level out-of-fold prediction rows | 217,016 |
| Averaged master-view prediction rows | 25,738 |
| Domain × preprocessing × procedure result rows | 336 |
| Crossover result rows | 238 |
| Field-log result rows | 35 |
| Resumable execution shards | 240/240 valid |

The 443 non-complete fit records comprise 189 rank failures, 91 strict
convergence failures, and 163 protocol exclusions propagated from a failed
calibration dependency. At the endpoint level, 6,437 of 6,720 fold endpoints
completed, 54 were empty outer folds by design, and 229 were unavailable.
Sixty unavailable exploratory endpoints had no frozen P03 source-only
selection for C-SELECTED. Another 163 inherited a calibration fit failure and
six inherited an outer-final fit failure. No failed candidate was silently
replaced and every declared endpoint remained in its denominator.

All 240 shards passed hash and accounting reconciliation. Complete endpoints
had five out-of-fold repeat predictions per view; physical-master isolation,
held-instrument exclusion, preprocessing-view parity, all-domain retention,
and private/public separation passed.

## C01 — primary held-instrument recoverability

Of the 34 observed domains, 13 were confirmatory, three exploratory, and 18
unsupported by design. The primary C-SELECTED result across the 13
confirmatory domains was:

| State | Domains |
| --- | ---: |
| supports portability | 0 |
| inconclusive | 5 |
| inferior portability | 2 |
| unavailable terminal failure | 6 |

| Domain | Held instrument | Held BA (95% CI) | Source−held loss (95% CI) | State |
| --- | --- | ---: | ---: | --- |
| CWA / H-SERS H-Kit | Mira-2 | 0.709 (0.529, 0.852) | 0.059 (−0.158, 0.249) | inconclusive |
| CWA / H-SERS H-Kit | Pendar-2 | — | — | unavailable |
| CWA / H-SERS H-Kit | RMX-1 | — | — | unavailable |
| pills / pSERS silver | Agilent-1 | 0.958 (0.792, 1.000) | 0.042 (0.000, 0.125) | inconclusive |
| pills / pSERS silver | Agilent-3 | 0.958 (0.751, 1.000) | 0.042 (0.000, 0.125) | inconclusive |
| pills / pSERS silver | Mira-3 | 0.333 (0.333, 0.333) | 0.667 (0.667, 0.667) | inferior |
| pills / pSERS silver | Pendar-1 | 0.933 (0.667, 1.000) | 0.067 (0.000, 0.329) | inconclusive |
| pills / pSERS silver | Pendar-3 | 0.952 (0.716, 1.000) | 0.048 (0.000, 0.143) | inconclusive |
| surfaces / pSERS silver | Agilent-3 | — | — | unavailable |
| surfaces / pSERS silver | Mira-1 | 0.571 (0.381, 0.619) | 0.381 (0.190, 0.476) | inferior |
| surfaces / pSERS silver | Pendar-2 | — | — | unavailable |
| surfaces / pSERS silver | Pendar-3 | — | — | unavailable |
| surfaces / pSERS silver | RMX-2 | — | — | unavailable |

The two Agilent and two Pendar pill estimates were high, but their matched-loss
upper intervals exceeded the locked 0.10 margin. They are therefore
inconclusive rather than successful. The Mira-3 pill domain was at three-class
chance performance; the Mira-1 surface domain also violated the recovery/loss
bounds. This heterogeneity is visible in F45.

### Substrate-level primary decisions

| Substrate family | Confirmatory domains | Supporting | Inferior | Inconclusive | Unavailable | Family decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| H-SERS H-Kit | 3 | 0 | 0 | 1 | 2 | unavailable terminal failure |
| pSERS Metrohm silver | 10 | 0 | 2 | 4 | 4 | inferior portability |
| GaN polymer | 0 | 0 | 0 | 0 | 0 | no confirmatory domains |
| NRC Canadian SERS | 0 | 0 | 0 | 0 | 0 | no confirmatory domains |

### Fixed RBF-SVM sensitivity

The fixed RBF-SVM completed all 13 confirmatory domains: two supported
portability, three were inferior, and eight were inconclusive. It supported the
two pSERS pill/Agilent domains at BA = 1.000 with zero matched loss. It was
inferior for pSERS pills/Mira-3 (BA = 0.333, loss = 0.667), pSERS
surfaces/Agilent-3 (BA = 0.492, loss = 0.508), and pSERS surfaces/Mira-1
(BA = 0.667, loss = 0.333). Because the remaining domains did not all pass,
this sensitivity also cannot support substrate-wide portability.

### Per-analyte secondary cells

The primary analysis retained all 102 domain × analyte cells. Among the 39
confirmatory cells, 11 passed their recovery and matched-loss bounds after Holm
correction, four were inferior, six were inconclusive, and 18 belonged to an
unavailable domain. These cell-level successes show that analyte signal can be
recoverable in specific conditions, but they cannot override the failed
three-class domain or substrate-family intersection–union decision.

## Classical procedure comparison

The comparison below averages the 13 declared confirmatory domains. “Common”
uses the five domains completed by every procedure. “Chance-imputed” assigns
BA = 1/3 to an unavailable endpoint. A positive comparison is allowed only
when its direction relative to C-SELECTED agrees under both views.

| Procedure | Successful domains | Mean BA, successful | Mean BA, common | Mean BA, chance-imputed | Common Δ vs selected | Chance Δ vs selected | Stable positive comparison? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Extra Trees | 13 | 0.771 | 0.853 | 0.771 | +0.026 | +0.201 | yes |
| Elastic-net logistic | 5 | 0.853 | 0.853 | 0.533 | +0.026 | −0.037 | no |
| PCA–LDA | 11 | 0.756 | 0.883 | 0.691 | +0.056 | +0.120 | yes |
| PLS-DA | 13 | 0.739 | 0.822 | 0.739 | −0.005 | +0.169 | no |
| Random Forest | 13 | 0.776 | 0.853 | 0.776 | +0.026 | +0.206 | yes |
| fixed RBF-SVM | 13 | 0.760 | 0.844 | 0.760 | +0.017 | +0.189 | yes |
| C-SELECTED | 7 | 0.774 | 0.827 | 0.570 | reference | reference | no positive self-comparison |

The practical lesson is that a fixed, robust estimator is preferable for the
next comparison: source-only selection was fragile because the frozen selected
set sometimes included a candidate that hit a strict convergence or rank
failure. The forest procedures and fixed RBF-SVM remained evaluable in all 13
confirmatory domains. This is a secondary model-comparison result, not a new
post-outcome license to replace the registered primary analysis.

## C02 — same-master substrate × instrument crossover

All 34 crossover blocks were retained for each of seven procedures (238 rows).
All blocks yielded the declared PP-U-MIN representation-distance contrast.
Predictive four-cell completeness was much more limited: only 13
procedure/block rows were complete—nine confirmatory and four exploratory—on
blocks X-021, X-024, and X-026. The other 225 rows remain explicitly
`unavailable_by_design`.

X-021 and X-024 compare NRC Canadian SERS (substrate A) with pSERS silver
(substrate B), Agilent-1 (instrument A) with Mira-3 (instrument B), for 4-ANPP
and benzyl fentanyl respectively. The locked interaction is the substrate-B
effect at instrument B minus the substrate-B effect at instrument A. Thus a
negative value means the relative pSERS advantage deteriorated on Mira-3.

- X-021, 4-ANPP, three masters: C-SELECTED correctness interaction −1.000
  (95% CI −2.000, −0.333) and true-class-probability interaction −0.848
  (−0.998, −0.767). The fixed RBF-SVM was also −1.000 (−1.000, −1.000) and
  −0.761 (−0.784, −0.739).
- X-024, benzyl fentanyl, three masters: fixed RBF-SVM correctness interaction
  −1.000 (−1.000, −1.000) and probability interaction −0.628
  (−0.687, −0.585). C-SELECTED was unavailable.
- X-026, blank, two masters: exploratory. Fixed RBF correctness interaction
  was 0.000 and the probability interaction was 0.156 (0.149, 0.162).

The two confirmatory blocks directly contradict a simple additive
“substrate works the same regardless of instrument” story for those analytes
and instruments. They do not identify whether the cause is substrate physics,
sample preparation, the optical system, preprocessing, or another acquisition
factor. Most planned blocks lacked four-cell predictive support, so this is
focused evidence rather than a comprehensive causal decomposition.

## C03 — recorded field-log corroboration

The recorded Y/N/M field log is a separate acquisition record, not the analyte
classifier label. Complete-case success and missingness bounds were computed at
the master–substrate–instrument-view level. Primary C-SELECTED results were:

| Substrate / endpoint | Definite / eligible views | Complete-case | Worst–best missing bound | Model–field agreement |
| --- | ---: | ---: | ---: | ---: |
| H-SERS H-Kit / nonblank detection | 69 / 74 | 0.232 | 0.219–0.274 | 0.451 over 51 predicted definite views |
| NRC Canadian SERS / nonblank detection | 5 / 15 | 0.000 | 0.000–0.667 | 0.750 over 4 views |
| NRC Canadian SERS / blank specificity | 3 / 10 | 1.000 | 0.300–1.000 | 1.000 over 2 views |
| pSERS silver / nonblank detection | 100 / 154 | 0.750 | 0.564–0.812 | 0.608 over 74 views |
| pSERS silver / blank specificity | 9 / 39 | 0.889 | 0.205–0.974 | 0.778 over 9 views |

H-SERS recorded nonblank detection was low. pSERS had substantially higher
recorded nonblank detection, but 33 logs were missing and 21 were ambiguous or
conflicting. The NRC family had too few definite records for a narrow statement.
Model–field agreement was not uniformly high, emphasizing that operator
detection and multiclass analyte identification are different endpoints.

## C04 — preprocessing sensitivity

There were 181 valid paired endpoint comparisons on identical master views.
Savitzky–Golay usually changed little: for C-SELECTED the median held-BA change
was 0.000 (mean +0.008), and for fixed RBF-SVM it was 0.000 (mean +0.014).
arPLS was more consequential: C-SELECTED median held-BA change was +0.054
(mean +0.103), and RBF-SVM median change was +0.004 (mean +0.071), with large
domain-specific ranges (up to +0.488 and +0.524 respectively).

However, arPLS did not merely improve every spectrum. Across all procedures it
moved some endpoints from inferior to inconclusive or supporting, while moving
other endpoints from inconclusive/supporting to a weaker state. For the primary
procedure, arPLS changed the pSERS pills/Mira-3 domain from inferior to
inconclusive but did not rescue the pSERS surfaces/Mira-1 domain. It changed
three high-performing pSERS pill domains from inconclusive to supporting.
Savitzky–Golay left most point estimates unchanged and also changed some
borderline interval decisions.

Therefore the minimal pipeline remains the frozen primary result, while arPLS
is evidence that background treatment can materially affect portability in
specific acquisition domains. The data do not support choosing arPLS after
seeing held-instrument performance. A future instrument- or substrate-aware
policy must be selected source-only or evaluated in a separately declared
calibration regime.

## Interpretation and next decision

The most defensible scientific statement is:

> In this incomplete 69-master field-trial design, analyte information was
> recoverable across instruments in several specific cells, but classical
> zero-shot evidence did not support substrate-wide instrument independence.
> Performance and the benefit of baseline correction were strongly
> acquisition-condition dependent.

The result supports continuing to compact deep learning only as a stringent
secondary test, not because classical ML failed overall. P04 must freeze a
small 1-D architecture, parameter budget, optimizer, epoch/early-stopping rule,
and source-only selection without using P13 held outcomes. It can then be
applied to the identical P13 domains and test views. The principal comparison
should include fixed RBF-SVM and the robust forest procedures, with
C-SELECTED retained as the registered primary reference. The publishable
question is whether structured repeated-view learning improves worst-domain
recovery or calibration beyond robust classical models under acquisition
shift—not whether a large network wins on 598 correlated spectra.

## Evidence map

- F45: `plan/figures/{tikz,pdf,png,html}/F45_substrate_recoverability.*`
- F46: `plan/figures/{tikz,pdf,png,html}/F46_substrate_instrument_crossover.*`
- F47: `plan/figures/{tikz,pdf,png,html}/F47_recorded_detection_agreement.*`
- Public aggregate tables: `results/p13_portability/tables/`
- Semantic figure tables: `results/p13_portability/semantic/`
- Figure and release hashes: `p13_figure_manifest.csv` and
  `release_manifest.json`
- Private row predictions, fit ledgers, calibration records, and shard state:
  outside the public repository under the protected artifact root.

Claims are limited to the tested station, substrate, analyte, and instrument
conditions. These results do not establish universal instrument independence,
a causal SERS mechanism, or a global substrate ranking.
