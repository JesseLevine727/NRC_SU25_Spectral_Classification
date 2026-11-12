# Is there a Siamese paper in our work?

3 November 2026. Internal scoping, not for Li-Lin as written. Data-set letters follow `PAPER_SCOPING.md`.

## Bottom line

- **No positive Siamese result exists anywhere in our work.** In every head-to-head comparison, a fixed metric (library search) matched or beat the Siamese network (section 1).
- Retraining will not change that on the same tasks, because each one is already solved by a fixed transform:
  - substrate background: first derivative;
  - wavenumber-axis offset between devices: lag-tolerant correlation;
  - unknown chemicals: a distance threshold.
- **One task gives a learned metric real room: identifying SERS spectra against a *normal-Raman* library.**
  - A fixed metric gets only 11 of 23 of our SERS maps right there.
  - No published learned method does this task. CaPSim does it with a hand-designed metric. DeepRaman, a learned model, matched SERS against a SERS library only (full text checked).
- This was the only Siamese paper with a real chance (option S1). **Its frozen feasibility test was a no-go:** 4.3/23 maps, close to chance and below every fixed metric (8–13/23). See S1.
- What remains is a negative-result paper (S3), or folding these comparisons into the Technical Note.

## 1. Every piece of Siamese work we have

| Work | Data | Validation | Siamese | Fixed metric / classical |
|---|---|---|---|---|
| Summer 2025 one-shot pair model | old consolidated SERS | one random 20/80 spot split | 98.8% | not compared (split leaks) |
| CSCCE poster (triplet, 340,704 weights) | set A | leave-one-substrate-type-out (LOSTO), selected on the same folds | BA 0.975 (reruns 0.962/0.972) | 2nd-derivative nearest centroid 0.987 |
| K-shot test | set A | LOSTO, 5 seeds | 0.84–0.93; one fold 0.000 at K=3 | n/a ("not robust") |
| Representation baselines | set A, NATO | predeclared | poster 0.858–0.928; NATO strict 0.632 | NC 0.943; PCA-logistic 0.725 |
| Supervised contrastive successor | NATO | nested, 3 seeds, gates | 0.701 (Siamese 0.632) | 0.685; promotion not supported |
| Open-world study | NATO | preregistered, leave-one-chemical-out | unknown-chemical AUROC 0.635 | classical 0.658 |
| ATLAS P05 | NATO | 13 held station–instrument domains | 71.9 / 76.2% | CNN 70.2 / 74.9%, Random Forest 70.1 / 76.7% |
| Sizing sweep (this session) | set A | inner substrate validation | sized CNN 13.8% error | NC 2.7% |
| Standard-Raman extension (`Standard_Raman_Classification/ExtendingSiamese`) | 12 old + 6 new-device liquids | 4 new chemicals enrolled with 1 spectrum each; known-only thresholds | 765/765, 96/96 new; new-device controls rejected until recalibration or fine-tuning | **same 765/765 and 96/96** (ALS + L2, no training); with ±13 cm⁻¹ lag tolerance, 94% of controls accepted and 100% of unknowns rejected at the known-only threshold |

- The "device shift" in the standard-Raman extension is a wavenumber-axis offset of about 7–9 cm⁻¹ on the February 2026 instrument:
  - benzene 985.8 vs 992.4 cm⁻¹;
  - pyridine 983.6 / 1023.4 vs 992.4 / 1032.3 cm⁻¹.
- Siamese mixtures (standard Raman): exact match 0.283 vs 0.984 for the best classical model.
- Scripts for the two new checks are in `scratchpad/siamese_scope/`: `classical_extend.py`, `lag_extend.py`, `raman_to_sers.py`.

## 2. Literature verdicts

Full write-up with a source-verification column: `scratchpad/siamese_novelty.md`.

| Angle | Verdict | Closest |
|---|---|---|
| **0. Learned metric, SERS query vs normal-Raman library, one-shot enrolment from Raman** | **Open as a learned method.** The task itself is published only in fixed form | Ju et al. 2023 *ACS Nano* 17, 21251 (CaPSim). DeepRaman (Fan et al. 2023 *Anal. Chem.* 95, 4863) matched SERS against a *SERS* library only, confirmed in its full text. None of CaPSim's 61 citing papers trains a learned Raman-to-SERS metric |
| 1. One-shot identification of new chemicals | Already published | Liu 2019 CILS; Park 2021 Analyst; Li 2022 Analyst; Wang 2025 Appl. Spectrosc. |
| 2. Cross-substrate Siamese (LOSTO) | Partly published: cross-instrument done, cross-substrate-type open | Bao 2024 SAA; Guo & Bocklitz 2026 Talanta; Park 2023 Analyst |
| 3. Physics-based augmentation as contrastive views | Partly published: substrate-background swap is open | Bjerrum 2017; Blazhko 2021; Mozaffari & Tay 2021; Liu 2025 ACS Sens. |
| 4. Open-set rejection | Partly published | Skvortsova 2022; Li 2022 (conformal); Balytskyi 2022 |
| 6. Siamese vs library search, grouped validation (negative result) | Partly published: our negative result is not published | Guo & Bocklitz 2026 (no library-search comparator); Contreras 2024 |
| 7. Mixtures | Already published | DeepCID, DeepRaman, Tan 2025 Angew. |

## 3. Options

### S1. One-shot SERS identification from normal-Raman references with a learned metric: feasibility NO-GO (3 Nov 2026)

**Result of the frozen feasibility test.** Protocol, code and results are in `scratchpad/s1_feasibility/` (`PROTOCOL.md`, `RESULTS.md`).

| Method | Maps correct (/23) |
|---|---|
| Siamese, strong SERS-physics augmentation (primary) | 4.3 mean over 3 seeds; ensemble 4 |
| Siamese, weak augmentation (control) | 3.3 |
| Siamese, pretrained on RRUFF (1,379 minerals) + in-house (Arm C) | 8.7 mean over 3 seeds; ensemble 9 |
| Chance | 3.8 |
| CaPSim, paper settings / repo settings | 8 / 12 |
| Derivative ±10 cm⁻¹ lags | 11 |
| Fixed combination | 13 |

- The encoder was trained on our 16 in-house normal-Raman chemicals.
- It learned to separate those chemicals, not a general similarity: synthetic validation peaked at the earliest checkpoints and stayed at fixed-cosine level.
- Real SERS differs from Raman in ways the augmentation did not cover:
  - pyridine's ν1 band shifts +18–22 cm⁻¹ (992 → 1010–1014 cm⁻¹);
  - substrate bands near 1096, 1124, 1380 and 1595 cm⁻¹ appear in every map and the Au blank.
- **The rescue was tried and failed.** Pretraining on 5,106 RRUFF spectra from 1,379 minerals (downloaded manually through the user's browser) raised synthetic validation to 0.88–0.90 and the test to 9/23. That is still below every fixed comparator but CaPSim's paper settings (8/23), and far from 17. The remaining gap is synthetic-to-real: the augmentation does not reproduce real adsorption shifts and substrate bands.

**Use in the Technical Note instead.**
- On these maps, a normal-Raman library identifies at most 13/23, including CaPSim, the published method for exactly this problem.
- A SERS library from the other substrate types identifies nearly all of them (2.7% error on set A).
- This answers Ju et al.'s question for paper and chip substrates, and covers the journal's "compared to other available methods" requirement.

The original plan follows.

**Question.** Can a Siamese network trained only on normal Raman spectra identify a chemical's SERS spectrum when the library holds only that chemical's normal Raman spectrum?

**Why the task is real.** SERS and normal Raman spectra of the same chemical differ: adsorption shifts bands, surface selection rules change relative intensities, thiols lose the S–H band. A SERS library per chemical (and per substrate) is costly. A normal-Raman spectrum of a new chemical is easy to measure.

**Headroom in our data** (fixed metrics, 23 SERS maps: 6 chemicals × 4 substrate types × 2 instruments; 6-chemical normal-Raman library):

| Fixed metric | Maps correct |
|---|---|
| ALS baseline + correlation | 7/23 |
| First derivative + correlation | 7/23 |
| First derivative + correlation, ±10 cm⁻¹ lags (best) | 11/23 (95% CI 0.27–0.69) |
| CaPSim, paper settings (peaks by intensity; K_smooth 5, N_peak 10, w_max 36 cm⁻¹) | 8/23 |
| CaPSim, our repo implementation's settings (peaks by count; K 3, N 12, w ≈ 30 cm⁻¹) | 12/23 |
| CaPSim, 24 settings (chosen on the test maps; descriptive only) | 5–12/23 |

- The result is the same with the window at 400–2300 or 600–1700 cm⁻¹.
- CaPSim does not solve our data either, and it fails on different maps from derivative correlation:
  - benzenethiol: CaPSim (paper settings) 0/4, derivative 4/4;
  - 4-MBA: CaPSim 4/4, derivative 0/4.
  - A reviewer will ask whether simply combining fixed metrics closes the gap, so that is a required control.
- Sources on disk:
  - the CaPSim paper, its SI, and Halas's computational-chromatography paper (Bajomo et al. 2022 *PNAS*, origin of CaPE): `Docs/Papers/Library/`;
  - an implementation note: `Docs/Reference_Material/CaPE_CaPSIM_Implementation.pdf`;
  - code: `Projects/Standard_Raman_Classification/Scripts/CaPSim.py`, `CaPSim_kNN.py`.
  - The repo code picks characteristic peaks by count (as in the 2022 CaPE paper); the 2023 paper picks them by intensity.
- Failures:
  - pyridine on all 4 substrates;
  - 4-MBA called 4-MBN on all 4 maps;
  - 2,6-PDCA on 3 of 4 maps;
  - 4-NP on Ag (surface chemistry).
- CaPSim reports the same picture on its own data: best fixed metric 50–70%, CaPSim 75–82%.

**Design.**
- **Train only on normal Raman spectra.** No SERS is used for training or model selection.
  - In-house: about 20 chemicals (12 pure-Raman library, 5 solvents from 2026, aniline, plus the powder/neat references).
  - Plus a public Raman library for class diversity (to be chosen).
  - The 6 test chemicals are excluded from training.
- **SERS-physics augmentation:**
  - per-band shifts of a few cm⁻¹;
  - per-band intensity scaling;
  - broadening;
  - S–H band removal for thiols;
  - measured substrate backgrounds (98 background spectra and the Au blanks);
  - noise.
- **Size the network by the Zhang–Gupta procedure on synthetic validation only.**
- **Sealed test:** the 23 SERS maps, scored per map. Add the handheld set D maps wherever a Raman reference exists (benzenethiol, 4-NP, pyridine).
- **Comparators:**
  - fixed metrics, including squared first-difference cosine;
  - CaPSim with the paper's settings (our implementation; 8/23 on the pilot) and the repo's settings (12/23), with settings fixed before the test;
  - the best combination of fixed metrics;
  - DeepRaman's released model;
  - an augmentation-only control: the same augmentation with a fixed metric, to show the network adds something.
- **Open-set rejection:** a threshold for chemicals outside the library. CaPSim names this as its own open limitation.

**Go / no-go (2–3 days).** Go if the network gets at least about 17/23 maps and beats both CaPSim and the best fixed-metric combination on the same maps. 17/23 vs 11/23 with no reversals gives exact McNemar p = 0.031. Otherwise stop.

**Risks.**
- Only 6 test chemicals and 23 maps, so the intervals are wide. More maps would help.
- Pyridine may not be identifiable at all. Check the pyridine SERS maps for the adsorbed-pyridine bands near 1008/1035 cm⁻¹ first.
- 4-MBA vs 4-MBN differ mainly in the COO⁻ and C≡N bands. Augmentation must not erase them.
- **Overlap with Paper 2:** it uses the same blocks A/B and the pilot's E1 comparison with Raman references. Agree the boundary with Li-Lin first.
- The training library's size and diversity may limit generalisation.

**Why it suits the poster story.** It keeps a Siamese network and one-shot enrolment at the centre, as the poster did. What changes is that enrolment now uses a chemical's normal Raman spectrum instead of its SERS spectrum. That is where a learned metric has something to learn.

**Venues** (unrestricted; article types not verified):
- *Analytical Chemistry* (DeepRaman), *Analyst*, *Anal. Chim. Acta*, *ACS Sensors*, *J. Raman Spectrosc.*, *Applied Spectroscopy*, *Digital Discovery*, *CILS*.
- Short formats to check: *Analyst* communication, *Chem. Commun.*, *Anal. Methods* communication.
- The CaPSim group publishes in *ACS Nano* and *PNAS*, which would be a reach.

**Effort.** Feasibility 2–3 days. Paper about 6–8 weeks if go.

### S2. One full paper combining the Technical Note and S1

"Identifying SERS spectra with libraries built elsewhere":
- Library from other substrate types: a first derivative is enough (the Technical Note result).
- Library of normal Raman spectra: a fixed metric fails and a learned metric is needed (S1).

It is one coherent story and a stronger full paper, but only if S1 is a go.

### S3. Negative result: Siamese vs library search under grouped validation

- **Data exist:** poster LOSTO, standard-Raman extension, sizing sweep; NATO only with Li-Lin's agreement (Paper 1 data).
- **Message:** each advantage claimed for Siamese networks (one-shot enrolment, substrate or device invariance, unknown rejection) is matched by library search with the matching physical fix.
- **Not published as such.** Guo & Bocklitz 2026 has no library-search comparator.
- **Risks:**
  - It contradicts the poster directly.
  - The user asked not to raise the poster network's problems with Li-Lin.
  - Li-Lin co-authored Mozaffari & Tay 2022 on overfitting CNNs, a related critical stance.
- **Venues:** *J. Raman Spectrosc.*, *Analyst*, *CILS*, *Digital Discovery*. About 3–4 weeks.

### Not viable

- One-shot new-chemical Siamese, SERS to SERS: already published, and library search does the same natively.
- Cross-substrate Siamese on set A: derivative nearest centroid is already at 2.7% error, so at best a tie. Use background swap only as part of S1's augmentation.
- Siamese for mixtures: already published, and our result was 0.283 vs 0.984.
- Standard-Raman extension as a paper: lag-tolerant library search matches it.
- A pro-Siamese NATO paper: the preregistered results go the other way.

## 4. Needed before S1 can go ahead

1. The feasibility test (no lab time, no Li-Lin input).
2. Li-Lin: the Paper 2 boundary; whether a few more SERS test maps are possible; Raman references for the handheld chemicals.
3. A public Raman library choice for training (licence and size).
