# Paper 2 pilot: phases 1 and 2 (baselines only)

## Context

Paper 2 will test whether a model can separate a SERS spectrum into a chemical (analyte) part and
substrate / acquisition parts. Before any neural network, this pilot checks the data and measures
how well simple chemometric baselines already do. It is a feasibility check, not evidence for a
paper: each cell below is usually one prepared substrate mapped at many spots.

## Hard rules

- Work only inside this folder (`research/paper2_pilot/`). Read raw data in place; never modify,
  move or copy raw data files.
- Python: `/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/.venv/bin/python`
  (numpy, pandas, scipy, scikit-learn, matplotlib available). No pip installs, no network access,
  no git commands.
- No neural networks. CPU only. Fixed random seeds. Each script should finish in a few minutes.
- **The independent unit is one map** (one prepared substrate). Spots within a map are repeat
  views. Never put spots from the same map in both training and test data.
- Fit any scaling, PCA or model only on training data.

## Data (all paths relative to `/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Data/`)

File format: `.txt`, two tab-separated columns (Raman shift in cm-1, intensity). Filenames encode
map and grid position, e.g. `bt - ag - 632nm - map1_Y4_X2.txt` (map 1, row 4, column 2). Ignore
`.ngc` / `.ngv` files and any file whose name indicates an average (`avg`, `average`).

**Block A (632 nm, one instrument):** `Jesse_Dataset_Update/SERs/`
- analyte folders: `4-NP - 632nm` (4-nitrophenol), `BT - 632nm` (benzenethiol), `Pyridine - 632nm`
- substrate subfolders are spelled inconsistently; normalize to four substrates:
  `Ag`/`AgNP` -> Ag, `Au`/`AuNP` -> Au, `PICO` -> PICO, `pSERS`/`pSERs`/`pSERSs` -> pSERS
  (Ag and AgNP are the same substrate; Au and AuNP are the same). 4-nitrophenol has no Au cell.

**Block B (two lasers):** `Jesse_Dataset_v2/SERS/`
- analytes: `4-mercaptobenzoic acid`, `2,6-pyridinecarboxilic acid`, `4-mercaptobenzonitrile`
- SERS subfolders: `AgNP - 632nm`, `AgNP - 785nm`, `AuNP - 632nm`, `AuNP - 785nm`
  -> substrate (Ag, Au) x laser (632, 785)
- `powder - non-SERs` subfolders are neat (non-SERS) reference spectra, not SERS data.

**Substrate-only blanks:** `Jesse_Dataset_v2/SERS/Backgrounds/AuNP/`

**Neat Raman references:** `Jesse_Dataset_v2/Pure_Raman_txt/` (e.g. benzenethiol, pyridine) and
`Jesse_Dataset_Update/Raman/` (4-nitrophenol, 4-mercaptobenzonitrile), plus the Block B powder
folders. Record the laser for each reference if it can be determined from names or folders.

## Phase 1: inventory and loader

1. `load.py` builds:
   - `outputs/inventory.csv`: one row per spectrum with block, analyte, substrate, laser, map_id,
     y, x, path, n_points, shift_min, shift_max.
   - `outputs/spectra_blockA.npz`, `outputs/spectra_blockB.npz`: raw intensities linearly
     interpolated onto a common 1 cm-1 axis per block, covering only the range every file in that
     block measures (no extrapolation), with matching label arrays. Do the same for references
     (`outputs/references.npz`) and blanks (`outputs/backgrounds.npz`).
2. `figures/cell_means_blockA.png` and `figures/cell_means_blockB.png`: mean spectrum per cell
   (min-max scaled for display only), with the analyte's reference overlaid where available.
3. `PHASE1_SUMMARY.md` (one page max): table of maps and spectra per cell for each block; the
   common axis range of each block; which references and blanks were found and matched; any
   problems (missing cells, unreadable or odd files, unequal map sizes).

Check your own counts against the raw folders before continuing.

## Phase 2: baselines (no neural networks)

Preprocessing: per-spectrum min-max scaling (primary). Sensitivity: arPLS baseline correction then
min-max, using the project's own code. Import `despike` and `arpls_correct` from
`../atlas_sers/src/atlas_sers/preprocessing/representations.py` (add `../atlas_sers/src` to
`sys.path`) with despike window 5, threshold 10.0; arPLS lam 100000, iterations 12,
relative_tolerance 0.001, logistic_clip 60.0.

Write `baselines.py` and run it to produce:

- **E1 Reference recovery.** For each block, fit NMF (scikit-learn) and a simple MCR-ALS
  (alternating non-negative least squares, e.g. with `scipy.optimize.nnls`, at most 50 iterations).
  Number of components k = number of analytes + number of substrates; also report k-1 and k+1.
  For each analyte with a reference: best correlation between the reference and any recovered
  component, and the fraction of the reference's 8 tallest peaks matched within +/-10 cm-1.
  Do-nothing baseline: correlation between the reference and that analyte's mean SERS spectrum.
- **E2 Variance decomposition** (how separable the factors are). Using map-mean spectra, report
  the fraction of variance explained by: analyte alone, substrate alone, laser alone (Block B),
  an additive model (analyte + substrate [+ laser]) and the full cell means. The gap between the
  additive model and full cell means is the interaction size.
- **E3 Unseen combinations.** For each cell whose analyte, substrate (and laser) all appear in
  other cells: hold out the whole cell and predict its mean spectrum with (a) an additive model
  fitted on the remaining cell means (analyte effect + substrate effect [+ laser effect]),
  (b) the same analyte's mean on other substrates, (c) the same substrate's mean over other
  analytes. Report correlation and RMSE against the real held-out cell mean.
- **E4 Unseen substrate.** Classify the analyte with leave-one-substrate-out (Block A: 4 folds;
  Block B: leave-one-substrate-out and leave-one-laser-out). Models: Random Forest (500 trees),
  Extra Trees (500 trees), PCA + LDA. Report balanced accuracy per spectrum and per map (average
  the predicted probabilities over a map's spots). Chance is 33%.

Write `PHASE2_RESULTS.md` (one page max): one small table per experiment for both preprocessing
versions, one or two plain sentences per table on what it shows, figure paths, and a limitations
section (spots are not independent; mostly one map per cell; this is feasibility only).

## Stop

Stop after Phase 2. Do not build or train any neural network. Finish by printing the paths of
`PHASE1_SUMMARY.md` and `PHASE2_RESULTS.md`.
