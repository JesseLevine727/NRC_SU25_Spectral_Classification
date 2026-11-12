# Turning the CSCCE poster work into a paper with the sensors we have

2 November 2026. Internal planning note (not for Li-Lin as written).

## 1. Every SERS data set on disk

Searched the whole NRC folder and the home directory. Duplicates were removed by file hash. Acquisition metadata comes from the LabSpec project files and the `.spc` headers.

| Set | Instrument, laser | Chemicals × substrates | Independent sensors | Spectra | Date | Used so far |
|---|---|---|---|---|---|---|
| **A. Poster maps** | LabRAM, 632.8 nm | 4-NP on Ag, PICO, P-SERS; benzenethiol and pyridine on Ag, Au, PICO, P-SERS | 11 (1 per cell) | 275 | 11 Jul 2025 | poster, pilot block A |
| **B. Thiol/acid maps** | LabRAM, 632.8 nm | 4-MBA, 4-MBN, 2,6-PDCA on Ag, Au | 6 (1 per cell) | 293 | 30 Jun 2025 | pilot block B only |
| **C. Same chemicals** | XploRA, 784 nm (10 mW) | 4-MBA, 4-MBN, 2,6-PDCA on Ag, Au | 6 (1 per cell) | 484 | 30 Jun 2025 | pilot block B only |
| **D. Handheld preparations** | SciApps ReporteR, 785 nm | benzenethiol: Ag ×2, Au ×2, Au plasma-cleaned ×1, P-SERS ×1; 4-NP: Ag ×2, P-SERS ×1; pyridine ×6 (substrate unknown); fentanyl ×7 | 22 | 115 scans (5 per sensor) | Oct 2020 – Jun 2025 | **essentially unused** (4-NP check only) |
| E. Single-substrate | LabRAM / XploRA | DMF on Au (228 + 47); 11-mercaptoundecanoic acid (1 map, substrate unknown); Au blank at 632 and 785 nm | 5 | ~400 | 2025 | not used for transfer |
| F. References | LabRAM / XploRA | powders (4-MBA, 2,6-PDCA) at 632/785; pure Raman of benzenethiol, pyridine, 4-NP, DMF and others | – | – | 2025 | pilot E1 |
| G. NATO field trial | 10 handhelds | 69 physical samples, 4 substrate families | 69 | 598 | Mar 2024 | **Paper 1, do not reuse** |
| Not SERS | LabRAM | `Feb26_Spectra`, `Jesse_Dataset_PT2`: solvents and mixtures | – | – | 2026 | mixture project |

- **Not new data:** `Mixture_Classification/Data/Legacy/SERS_txt` holds text copies of set D, and the SERS "blind unknowns" are byte-identical copies of set D scans.
- **Total outside NATO:** about 50 SERS sensors over three instruments.
- **The constraint:** with one exception, no instrument has more than one sensor per chemical × substrate cell. The exception is set D (handheld): benzenethiol on Ag ×2 and Au ×2, and 4-NP on Ag ×2. Set D is therefore the only place where a held-out sensor of the *same* substrate can be tested.

## 2. Three ways to turn the work into a paper

### Option A: ML methods paper, no new sensors (recommended if sensors are scarce)

**Working title.** *Count sensors, not spectra: sizing and validating neural networks for small SERS data sets.*

**Question.** How should neural networks be sized and validated when SERS data come from only a few sensors? Does any network beat simple derivative matching when the substrate changes?

**Why this is a real ML paper.** It is about model capacity, validation design and generalisation under domain shift, which are core ML questions, studied on spectroscopy data. The contribution is a method that transfers to other spectroscopy labs.

**Contributions** (all already prototyped on set A, 2 Nov 2026):
1. **The independent unit is the sensor.**
   - 5 or 25 spots per map give the same accuracy: nearest centroid 2.3–2.7%, CNN 11–13% error.
   - Training on three substrate types instead of two cuts the nearest-centroid error from 7.4% to 2.7%.
   - Random spot splits inflate accuracy (98.8% on the poster).
2. **Formal network sizing** (Zhang and Gupta procedure, as in `SIZING_METHOD.md`).
   - CNNs from 684 to 340,704 weights and MLPs from 27 to 10,760 weights were all tested.
   - None reaches 5% held-out-substrate error: training error goes to 0 while held-out error stays at 10–20%.
3. **Memorisation versus domain shift.**
   - An MLP with only 84 weights (fewer weights than training spectra) shows the same gap, so the gap is substrate shift, not over-capacity.
   - The textbook over-learning test misreads it when the validation set is a new domain.
   - Prescription: report within-domain validation (held-out sensor, same substrate) and cross-domain validation (held-out substrate) separately. Set D allows the within-domain check.
4. **Simple derivative matching beats every network.** Held-out-substrate error: nearest centroid 2.7%, best MLP 7.3%, CNN 13.8%, PCA-input MLPs 13.8–16.3%, PCA + LDA 11.8%.
5. **A short checklist** for SERS ML studies: unit, splits, sizing, validation type, baselines.

**Data used.**
- Set A: substrate shift.
- Sets B and C: Ag↔Au, and whether the result holds on a second instrument and laser.
- Set D: multi-sensor, multi-year; the within-domain check.
- NATO data not used, to keep Paper 1 separate.

**What still has to be done** (code for most of it exists in the scratchpad `overtrain_check/`):
1. A targeted novelty check (Section 4).
2. Freeze sets A–D with metadata. Exclude 4-NP from the main analysis (does not bind Au; probably reacts on Ag) and report it as a boundary case.
3. Pre-register: models (nearest centroid, kNN, PLS-DA, PCA + LDA, linear SVM; CNN and MLP width sweeps; the poster Siamese network as a reference), splits, target error and selection rule.
4. Run on A, B and C. Run the within-domain versus cross-domain check on D. Add learning curves.
5. Statistics with the sensor as the unit (exact intervals, permutation tests); 5 seeds per network.
6. Code and data release (journal requirement; needs NRC approval).

**Venues.**
- *Chemometrics and Intelligent Laboratory Systems* (Elsevier).
- *Journal of Chemometrics* (Wiley).
- *Analyst* (RSC).
- *Digital Discovery* (RSC, gold open access £2,200; wants open data and code).

Full paper, about 8–10 pages. None of these were verified for this framing yet.

**Prior art to position against** (verified earlier):
- Guo et al. 2017, common mistakes in cross-validation (*Anal. Methods* 9, 4410).
- Beleites et al. 2013, sample-size planning with learning curves (*Anal. Chim. Acta* 760, 25).
- Guo, Popp and Bocklitz 2021, chemometrics protocol (*Nat. Protoc.* 16, 5426).
- Guo and Bocklitz 2026, Siamese networks and replicate variability (*Talanta* 302, 129422).
- Mahmud et al. 2027 review: 6.2% of 129 SERS–ML studies validated externally (*Talanta* 311, 130241).

Our novelty must be: substrate as the held-out domain, network sizing by independent units, and the memorisation-versus-shift diagnostic.

**Risks.**
- Validation-practice papers exist. The sizing and domain-shift angle has to be clearly new (novelty check first).
- Reviewers may want a second, public data set to show generality. Optional addition.
- The paper reports openly that the poster's network was oversized. Li-Lin will see this in the manuscript; frame it as "the poster model was the starting point, and this is how to size it".
- Overlap with Paper 1's "classical ≈ deep" message. Keep the focus on sizing and units, and cite Paper 1.

### Option B: Technical Note on substrate transfer (needs some new sensors)

The *Analytical Methods* plan in `CSCCE_PAPER_PROPOSAL.tex`, scaled to the sensors available:

| Tier | New sensors | What it gives |
|---|---|---|
| 0 | none | benzenethiol + pyridine on 4 substrates (1 sensor per cell); B as a second-day Ag↔Au set; D as a stress test. Weak: reviewers will ask about single sensors |
| 1 | **in-house only**: 2nd sensor of benzenethiol, pyridine, 4-MBA, 4-MBN, 2,6-PDCA on Ag and Au + 2 blanks = **12** | two sensors per cell on printed substrates; Ag↔Au transfer with replicates. No new commercial sensors |
| 2 | Tier 1 + 2nd benzenethiol and pyridine sensor on PICO and P-SERS = **16** | replicates on every substrate for two chemicals |
| 3 | full plan = **24** | complete 4 × 4 matrix with two sensors per cell |

The in-house Ag and Au sensors are printed at NRC and may be easy to make. PICO and P-SERS are bought and may be the scarce ones. **This is the question for Li-Lin.**

### Option C: fold the poster into Paper 2 as its pilot

Use it if neither A nor B is wanted. No separate paper.

## 3. Recommendation

- **If sensors are scarce:** write Option A from the existing data. It is genuinely ML, the main experiments are already prototyped, and it does not depend on the lab.
- **If Tier 1 is possible** (printing is cheap): add the 12 sensors to Option A as the within-domain test set. That strengthens contribution 3.
- **Option B** stays available as a later short note if commercial sensors become available.

## 4. Next steps

1. Ask Li-Lin which sensors are scarce (commercial vs printed) and whether Tier 1 is possible.
2. Run a targeted novelty check for Option A: network sizing / independent units / domain shift in Raman and SERS machine learning.
3. Move the scratchpad experiments (`overtrain_check/`) into the repo, e.g. `research/substrate_transfer_note/`, before they are lost.
4. Freeze data sets A–D and pre-register the analysis.
