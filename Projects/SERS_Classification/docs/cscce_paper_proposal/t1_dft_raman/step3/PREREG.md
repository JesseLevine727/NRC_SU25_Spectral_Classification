# T1 step 3: predeclared confirmation test (written 2026-11-08, before any step 3 DFT result exists)

## Why this test
Step 2's predeclared result was a tie (Siamese 34.2% vs fitted calibration 35.8%, trained on 29
compounds). Its post hoc pooled analysis suggested the Siamese pulls ahead once it is trained on
about 110 compounds, and only on compounds without O-H/N-H groups (69% vs 51% on 45 such
compounds). That came from data already looked at, so it needs fresh compounds.

## Question
Trained on all 110 step 2 compounds, does the fine-tuned Siamese beat the strongest calibration at
identifying new compounds without O-H/N-H groups, against the full DFT library?

## Test set (fixed now: test_compounds.csv, sha256 4a9884c990cf18ac207bfaca34484e7e3cb59ab66b80d6c0b9d720b6dd8d6ce3)
- VibraCLIP experimental spectra (Zenodo 10.5281/zenodo.15348392, `experimental_dataset.pkl`;
  Thermo OMNIC library, probably FT-Raman 1064 nm; commercial origin, test use only) of compounds
  that are not in QM9S, not among the 110 training compounds, contain only C/H/N/O/F, have no O-H
  or N-H (SMARTS `[#7,#8;!H0]`), and have at most 30 atoms including H. That gives 59 compounds
  (18 with <=20 atoms, 16 with 21-25, 25 with 26-30).
- Every VibraCLIP spectrum of these compounds is a test query. No test spectrum is used for any
  fitting, tuning or model choice.
- The only allowed exclusions are technical DFT failures (SCF or optimisation not converging, or an
  imaginary frequency below -5 cm^-1 left after 3 displacement-and-reoptimise rounds). Each one is
  reported by name. Scoring happens once, after every listed compound has finished or failed.

## Library
- QM9S (129,817) + our 8 step 1 compounds + one new entry per test compound, all B3LYP/TZVP.
- New entries: same level of theory as step 1 (see qm9s-level-of-theory). Polarizability
  derivatives come from finite-field analytic gradients (dpol_field.py) instead of 6n displaced
  polarizabilities. Validated before this test: Raman activities agree with the displacement route
  to within 0.25% (max) on acetonitrile, THF and 2,2-dimethoxypropane.
- Library spectra are processed exactly as before: 785 nm cross-section, three-range scaling.

## Training data
All 127 step 2 spectra of 110 compounds (our 37 spectra, the 81 VibraCLIP test compounds of step 2
and the 9 VibraCLIP overlap spectra).

## Fixed preprocessing (as step 2)
Window 500 to 1800 cm^-1 at 2 cm^-1; VibraCLIP queries multiplied by
((1e7/785 - nu) / (1e7/1064 - nu))^4; clip at 0, sqrt intensity, unit norm.

## Methods
- Fixed: sqrt-intensity cosine, Lorentzian FWHM 16 cm^-1.
- **Calibration (comparator):** the 6-parameter fitted calibration fitted on our 37 spectra, as in
  the step 2 primary test. Chosen now from step 2 evidence only: it was the strongest calibration
  variant on step 2's VibraCLIP compounds (35.8% top-1, against 25.9% for a single fit on all 110
  compounds and 24.7% for per-instrument parameters).
- **Siamese:** the step 2 pretrained d2w32 checkpoint (500-1800 window, all step 2 query compounds
  removed from pretraining; the test compounds are not in QM9S, so they never appear in
  pretraining). Measured tower fine-tuned on all 127 training spectra with the step 2 recipe (300
  steps, lr 1e-4, AdamW), seeds 0, 1, 2. Library entries of the test compounds are excluded from
  the fine-tuning negatives, as in step 2.

## Primary metric and decision rule (same as step 2)
- Top-1 against the full library, per test compound (mean over its spectra).
- Siamese score per compound = mean top-1 indicator over the 3 seeds.
- Delta = Siamese - calibration. 95% CI by paired bootstrap over compounds (2000 resamples, seed 0).
- **Siamese wins** if Delta >= 0.10 and the CI lower bound > 0.
- **Calibration wins** if Delta <= -0.10 and the CI upper bound < 0.
- Otherwise **tie**.

## Controls and secondary analyses (reported, not part of the decision)
- Label-shuffle Siamese (seeds 1-3), trained on all 127 spectra; flagged if it exceeds fixed by
  more than 5 points.
- Calibration fitted on all 127 training spectra (single parameter set), and per instrument.
- Top-5, top-20, median rank; results by size tier (<=20, 21-25, 26-30 atoms) and chemical class.

## Known limitations, stated in advance
- Power: resampling step 2's non-donor results gives about 80% chance of a Siamese win at 59
  compounds. This is optimistic, because the effect size comes from the data that suggested it.
- Size: the test compounds have 10-18 heavy atoms, while QM9S has at most 9. They sit among few
  library entries of their size, which may make retrieval easier for every method. The comparison
  is paired, so this affects absolute numbers more than Delta.
- O-H/N-H compounds are left out by design. Step 2 showed every method fails on them with
  gas-phase monomer DFT, so this test speaks only to compounds without hydrogen-bond donors.

## Amendment 1 (2026-11-09, before any scoring; no test result has been computed or seen)
- The finite-field step required the SCF energy change to fall below 1e-12 Ha, which is at the
  numerical noise floor for the larger molecules. 9 compounds (vc12, vc13, vc16 to vc22) failed at
  that step for this reason alone, after a successful optimisation and Hessian. This is a pipeline
  bug, not a property of the compounds, so they are rerun instead of excluded.
- Fix: energy test 1e-10 Ha; the orbital gradient test, which sets the density and gradient
  precision, stays at 1e-8 (one fallback to 5e-8), up to 150 cycles. Revalidated on
  2,2-dimethoxypropane: Raman activities within 0.23% of the displacement route, as before. The
  step now also completes on 3-heptanone (vc22), which failed overnight.
- The optimised geometry and Hessian are saved before the last step, so a failure there no longer
  discards them.
- Two molecules run at once, 16 threads each. This changes no computed value.
- Everything else in this document is unchanged.
