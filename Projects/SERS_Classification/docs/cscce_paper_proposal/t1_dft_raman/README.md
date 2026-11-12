# T1: experimental Raman vs a DFT library (Siamese vs calibration)

Working copy of the T1 wedge pilot (see `../SIAMESE_WEDGE.md`). Scripts were developed in a Claude
Code session scratchpad and copied here by `scripts/sync_to_project.sh`.

## Steps
- **Pre-gate and step 1** (2026-11-07/08): 37 measured spectra of 29 liquids (Flanagan 785 nm +
  NRC) against QM9S plus our own B3LYP/TZVP calculations for 8 missing compounds.
  Summary: `results/summaries/siamese_ext_summary.txt`.
- **Step 2** (2026-11-08): predeclared transfer test on 81 VibraCLIP compounds
  (`step2/PREREG.md`). Result: tie. Summary: `results/summaries/siamese_v2_summary.txt`.
- **Step 3** (started 2026-11-08): predeclared confirmation on 59 fresh compounds without O-H/N-H,
  training on all 110 step 2 compounds (`step3/PREREG.md`, `step3/test_compounds.csv`). DFT for the
  test compounds runs in `scripts/run_dft_step3.sh`; `scripts/chain_step3.sh` scores once when it
  finishes and writes `results/summaries/siamese_v3_summary.txt`.

## DFT
B3LYP/TZVP (Gaussian/Ahlrichs TZVP, as QM9S) in PySCF: `scripts/dft_raman.py`, `tzvp_basis.py`,
`fix_imag.py`. Polarizability derivatives from finite-field analytic gradients
(`dpol_field.py`, default) or 6n displaced CPKS polarizabilities (`DPOL_ROUTE=displace`, step 1).
Our outputs are in `results/dft/` (step 1) and `results/dft_step3/` (step 3), in QM9S units.

## Not included here
- QM9S (figshare) and the Flanagan spectra: public downloads.
- VibraCLIP spectra (Zenodo 10.5281/zenodo.15348392): Thermo OMNIC library spectra of
  commercial origin. Used as test data only; do not republish or commit them. The data files
  (`siamese_data_*.npz`) contain them and stay outside the repository.
