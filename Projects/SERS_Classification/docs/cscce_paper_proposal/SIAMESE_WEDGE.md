# Where is the wedge for a Siamese / one-shot Raman or SERS paper?

6 November 2026. Internal scoping. New measurements are allowed, normal Raman is acceptable, and this is a standalone paper (separate from the Technical Note).

## The rule every candidate had to pass
Library search is already one-shot identification: one reference per chemical, matched by a fixed similarity score. A learned (Siamese) similarity earns a paper only if it beats the best fixed score on chemicals never seen in training, or answers a question library search cannot.

## What five research passes found
- **Same kind of spectrum on both sides: no wedge.**
  - Fixed matching is at the ceiling. Sun et al. 2027 (*Spectrochim. Acta A* 368, 129057; read in full) is a Siamese network on 151 real substances measured on 2 instruments: cosine 99.29% top-1 vs learned 99.57%.
  - Our own results agree.
  - Only 8 of 126 Raman/SERS papers compare against library search at all, and none against a strong one (derivative + lag tolerance).
- **Learned similarity clearly wins only when the query and reference come from different kinds of measurement that preprocessing cannot reconcile.**
  - IR: experimental vs DFT (IRSA, 48% vs 29% top-1).
  - Mass spectrometry: positive vs negative ionisation.
  - NMR: spectrum vs structure.
  - It also needs thousands of training compounds.
- **The bottleneck for Raman is data.** Open, licence-clean experimental organic Raman with structures covers only about 150–180 compounds. The large sets are restricted (SDBS, about 2,800 compounds, no bulk use; KnowItAll, AI-training rights reserved).

## The candidates after the stress test

| | Idea | Verdict | Why |
|---|---|---|---|
| **T1** | Identify compounds with no measured reference: match experimental Raman (later SERS) against a DFT-calculated library | **Weak but open; best mechanism** | Nobody has published top-k numbers for experimental organic Raman against a DFT library, so the benchmark alone is new. IRSA's margin is probably inflated: its fixed baselines had no shift tolerance and its split was random. Expected margin with about 150–200 compounds: under +10 points; about 20–30% chance of clearing +10. A warning: on measured vs simulated X-ray patterns (RealBind 2026), fixed correlation beat a contrastive model 82.8% vs 5.3%. Becomes strong if AIST permits SDBS use. |
| **T2** | Analog search: when the compound is not in the library, return its closest structural relatives | **Weak; highest upside** | Koshute 2023 (read in full): max-Pearson library matching flagged only 64% of unseen fentanyl analogs at 1% false alarm, vs 91% for a trained model. But that baseline ignored the 248 negatives and had no derivative or lag. In MS the learned gain shrinks to 4.6% of the headroom under clean splits (Strobel 2025). In IR, plain correlation already gave the most structure-informative hits (Varmuza 2003). |
| **T4** | Calibrated same/different verification | **Weak as a method** | Any score can be calibrated, including the hit-quality index (HQI). Non-learned calibrated verification already ships (TruScan p-value). The finding that the 0.95 HQI rule is not calibrated is a measurement result, not a method. |
| **T5** | Benchmark: learned vs fixed similarity on unseen chemicals | **Strong, but not a "Siamese wins" paper** | Only 3 experimental-Raman papers test unseen compounds against a library baseline, and none uses a strong fixed method. |

Closed earlier: SERS-to-SERS one-shot, cross-substrate Siamese, the Raman→SERS learned metric (S1), mixtures, ExtendingSiamese. Bacteria/cells are crowded.

## Recommendation
**Lead with T1.** It is the one setting where every other spectroscopy shows learning winning. Design the paper to publish either way: if the Siamese wins, the paper reports a method; if not, it reports a benchmark showing when it doesn't. Run T2's first test alongside it, because it costs almost nothing.

1. **Zero-lab pre-gates (about 1 week, open data only).**
   - **T1:** match the about 26 Flanagan & Glavin 2025 liquids (CC BY, 785 nm) against QM9S and ViBench calculated libraries.
     - Fixed baselines: cosine, Pearson, HQI, derivative + lag, peak assignment, CaPSim; each with global and three-range frequency scaling.
     - Kill if the best fixed method already gets ≥70% top-1 on same-formula isomer pools (no room) or <10% top-5 (DFT too poor).
   - **T2 Stage 0 (no training):** on about 150 open experimental compounds plus 5,000 calculated spectra, measure how much of the achievable structural similarity (Tanimoto) the best fixed metric already captures.
     - Kill if it captures ≥80% at top-5, or if less than 0.08 Tanimoto of headroom remains.
2. **Emails to send now, in parallel (the people they go to have to answer):**
   - AIST, for written permission to use SDBS Raman. This is the single thing that would make T1 strong.
   - The Calvo-Castro group (Hertfordshire), for their 53 + 21 NPS Raman spectra.
   - The Carleton library, for whether a KnowItAll licence covers ML training.
3. **If a pre-gate passes: about 30 instrument-hours (T1 gate).**
   - Measure about 120 small stockroom compounds (CHONF, ≤9 heavy atoms, checked for QM9 membership first) at 785 and 632.8 nm, for n ≈ 150–200 with the Flanagan set.
   - Five-fold cross-validation grouped by scaffold.
   - Two-tower contrastive model with isomer hard negatives, plus a gradient-boosted re-ranker over the fixed scores as a mandatory control.
   - GO: at least +10 points top-1 over the best fixed method with a CI excluding 0, at least +5 over the re-ranker, and the same sign on every candidate pool and level of theory.
4. **SERS arm (exploratory).** Our thiols on Au/Ag paper against the Molecular Vibration Explorer library of thiols bound to Au. It has benzenethiol and 4-MBA; we would compute the missing thiols at its level. About 35 queries, so it can't decide anything on its own.

**Effort:** about 3–5 months for the full paper.
**Venues:** *Analytical Chemistry* if positive; *Digital Discovery*, *JCIM* or *Analyst* as a benchmark.
**Hardware:** 32 cores, 60 GB RAM, RTX 5080. Enough for the model and for DFT on a few dozen thiols.

## Sources
The scratchpad copies are session-local and may be cleaned up.
- **Full reports:** `A_raman_sers_lit.md`, `B_other_spectroscopies.md`, `C_datasets.md`, `D1_redteam_T1_dft.md`, `D2_redteam_T2_T4_T5.md`.
- **Full-text checks:** `full_text_checks.md` (Sun 2027, Koshute 2023, Ju 2025, Chen 2025).
- **Location:** `/tmp/claude-1000/-home-elfo-Documents-NRC-NRC-SU25-Spectral-Classification-Projects-SERS-Classification/e5b064dc-da05-42ac-b179-ebe8f64bcd49/scratchpad/wedge/`.
