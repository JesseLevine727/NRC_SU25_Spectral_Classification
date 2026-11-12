# Option B with existing data only: making the Technical Note as robust as possible

2 November 2026. Internal plan. Data-set letters follow `PAPER_SCOPING.md`.

## 0. The claim, sized to the evidence

**Will claim.**
- For analytes that bind every tested substrate, a library of first-derivative SERS spectra identifies the chemical on a substrate type absent from the library.
- This holds for **X of Y held-out sensors across three instruments**, each analysed within its own instrument.
- Random spot-level testing overstates performance.
- Where transfer fails, the substrate changes the spectrum (4-NP).

**Will not claim.**
- Universal substrate independence.
- That one method beats another (too few sensors to rank them).
- Sensor-to-sensor reproducibility on the LabRAM (no replicate sensors there).
- Quantitation.
- Transfer across instruments (that is Paper 1).

## 1. The design that makes it robust: one development set, three sealed test sets

The derivative input was originally chosen while looking at set A's test results, so A cannot also be the proof.

| Role | Data set | Instrument | Chemicals × substrates | Held-out sensor tests |
|---|---|---|---|---|
| **Development** (fix the method here) | A | LabRAM 632.8 nm, 11 Jul 2025 | benzenethiol, pyridine × Ag, Au, PICO, P-SERS (4-NP separately) | 8 (+3 for 4-NP) |
| **Sealed test 1**: other chemicals, other day | B | LabRAM 632.8 nm, 30 Jun 2025 | 4-MBA, 4-MBN, 2,6-PDCA × Ag, Au | 6 |
| **Sealed test 2**: second instrument and laser | C | XploRA 784 nm, 30 Jun 2025 | same as B | 6 |
| **Sealed test 3**: handheld, sensors from 2020–2025 | D | ReporteR 785 nm | benzenethiol (Ag ×2, Au ×2–3, P-SERS ×1), 4-NP (Ag ×2, P-SERS ×1), fentanyl (Au ×3–5, P-SERS ×2) if NRC allows; pyridine ×6 if its substrate is known | 9–22 |

- **Freeze the method on A before running anything on B, C or D:** preprocessing, window, matching rule, and the per-map decision rule (average of spot predictions or majority vote).
- **Then score B, C and D once each.** Each is analysed within its own instrument, so no train/test pair crosses instruments.
- **Pooled sensor-level evidence:** about 26–36 held-out sensors over three instruments. If all are correct, the 95% lower bound on per-sensor accuracy is about 0.89–0.90 (Clopper–Pearson). One error still gives about 0.85. A single data set of 8–11 sensors cannot give this.
- **Combining A and B is a trap.** The chemicals in A and B were measured on different days, so day would be confounded with chemical. Keep B separate.

## 2. Answering the reviewer objections we can answer with existing data

| Objection | Answer from existing data |
|---|---|
| "One lucky sensor per cell" | Set D has replicate sensors of the same chemical on the same substrate, from different days and years (benzenethiol Ag ×2, Au 2020 vs 2025; 4-NP Ag ×2; fentanyl Au ×3). Compare sensor-to-sensor distance with substrate-to-substrate and chemical-to-chemical distance (Fig. 5 or ESI). The pilot 4-NP add-on already found tiny sensor-to-sensor differences within a session |
| "You chose the method on the test data" | A is development only; B, C and D are sealed tests (Section 1). The protocol is frozen and dated before they are run |
| "It only works on one instrument" | Replicated within instrument on the LabRAM, XploRA and ReporteR |
| "Three chemicals is too easy" | 5 chemicals on the LabRAM (A + B), each fold tested against its own library. Per-sensor uncertainty is reported, not just accuracy |
| "Two of the four substrates are both silver paper" | Report per held-out substrate. PICO, the only non-paper chip, is the hard test and is discussed explicitly |
| "Compare with standard practice" | Include correlation library search (hit-quality index) on raw and on derivative spectra, plus PCA + LDA, PLS-DA and kNN. Results are descriptive, with no ranking claims |
| "What about neural networks?" | ESI only: a CNN and an MLP, sized on A's inner folds. No claim beyond "did not improve on matching here" |
| "Why does it work?" | Mechanism figure: raw vs derivative spectra; variance shared between substrate and chemical; Au blank; band table against Raman references (powders and pure liquids exist for most analytes) |
| "Spot-level results are inflated" | Show random-spot vs held-out-substrate on the same data. Spots-per-map learning curve: 5 spots give the same accuracy as 25 |
| "Data/code availability" | JCAMP-DX or repository deposit; code release (needs NRC approval) |
| "Is the 4-NP failure the model's fault?" | Boundary-case paragraph: no binding on Au (chemists' knowledge); in-house Ag map dominated by substrate bands; the handheld 4-NP Ag sensors look like normal 4-NP. Claim no mechanism beyond that |

## 3. Paper layout (about 6–8 journal pages, Technical Note)

- **Fig. 1:** design. Substrate photos (paper fibres vs the PICO chip) and the matrix of all four data sets with sensor counts and instruments.
- **Fig. 2:** why derivatives transfer. Raw vs derivative spectra of benzenethiol and pyridine on the four substrates, plus variance shares.
- **Fig. 3:** main result. Per held-out sensor, correct or not with spot-level accuracy, for A, B, C and D. Pooled sensor-level confidence interval. Random-split vs held-out comparison inset.
- **Fig. 4:** where it fails. 4-NP across substrates and instruments; benzenethiol on PICO.
- **Fig. 5 (or ESI):** sensor-to-sensor vs substrate-to-substrate distances from set D.
- **Table 1:** data sets and acquisition (instrument, laser, power, objective, exposure, map, dates).
- **Table 2:** results per data set and held-out substrate, with comparators.
- **ESI:** all confusion matrices; preprocessing sensitivity; spots-per-map curve; network sizing results; protocol freeze record; band assignments.

## 4. Work plan

1. **Freeze the data** (about 2 days). Rebuild sets A–D from the raw files, with a metadata table taken from the `.ngc` and `.spc` headers. Fix the 4-NP status and the fentanyl decision.
2. **Develop on A** (about 3 days). Nested leave-one-substrate-out over a small declared menu: raw / baseline-corrected / 1st and 2nd derivative × nearest centroid / correlation search / PCA + LDA / PLS-DA / kNN. Choose one primary method by inner folds only. Write and date the protocol freeze.
3. **Run the sealed tests** on B, C and D once each (about 2 days).
4. **Controls and mechanism** (about 3 days): label-permutation null at sensor level; sensitivity to window and derivative settings; spots-per-map curve; variance decomposition; band table vs references; sensor-to-sensor distances from D.
5. **Figures and tables** (about 1 week), then **writing** (1–2 weeks), then co-author and NRC review.

Rough total: 4–6 weeks of work before NRC review. This is an estimate.

## 5. Still needed from Li-Lin (no lab time)

1. What PICO is (maker, material).
2. Concentrations and solvents for sets A–D; LabRAM laser power at the sample.
3. In-house batches and print passes; what "SC", "PL" (pyridine S1PL–S6PL) and "post-O2" mean; the substrate of the handheld pyridine sensors.
4. Whether fentanyl spectra may be published.
5. Author list. Whether NRC can release the spectra.

## 6. What stays weak, honestly

- Still one sensor per cell on the LabRAM. Set D addresses sensor variation only on the handheld.
- Four substrate types, two of them silver paper. One concentration. Pure analytes.
- The editor's "complex samples" criterion is still the main desk-rejection risk.
- Tier 1 (12 printed Ag/Au sensors) would close the first gap if it ever becomes possible.

## 7. Decision points for Jesse

- Include fentanyl? It would make set D a three-chemical test, but needs NRC clearance.
- Network results in the ESI? Li-Lin co-authored a 2022 paper on 1-D CNNs for Raman identification (Mozaffari & Tay, *Spectrochim. Acta A*, 10.1016/j.saa.2022.120961). Keep the network part neutral and small, or leave it out.
- Freeze the protocol before touching B, C or D. Nothing has been run on them with the derivative/nearest-centroid method yet; keep it that way until step 2 is done.
