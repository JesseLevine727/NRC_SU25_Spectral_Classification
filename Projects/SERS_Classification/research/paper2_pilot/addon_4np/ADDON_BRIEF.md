# Paper 2 pilot add-on: chip-to-chip versus substrate variation (4-nitrophenol)

## Question

In blocks A and B every chemical x substrate cell is one prepared substrate, so substrate effects
cannot be told apart from preparation luck. 4-nitrophenol (4-NP) on silver was prepared more than
once. Measure how much spectra differ **between preparations of the same thing** and compare that
with how much they differ **between substrates** and **between chemicals**.

We do not know whether the repeat preparations were made the same way, so treat the
between-preparation number as an upper bound on chip-to-chip variation.

## Hard rules

- Work only inside this folder (`research/paper2_pilot/addon_4np/`). Another job is running in
  the parent folder: do not create, edit or delete anything outside `addon_4np/`. You may read
  `../outputs/*.npz`, `../outputs/inventory.csv` and raw data, never modify them.
- Python: `/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/.venv/bin/python`.
  CPU only. No pip installs, no network, no git commands.

## Data

1. **Repeat 4-NP-on-silver preparations** (binary Galactic `.spc` files):
   `/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Data/Jesse_Dataset_v2/SERS/4-Nitrophenol/`
   - three samples, each with five scans `R1`-`R5`: `Ag-23pp-SC-4NP`, `S01-Ag-psers-4NP`,
     `S03_23PP_SC_4NP`. Ignore the `avg` files.
   - the names suggest two `23pp-SC` silver preparations and one pSERS silver preparation.
2. **Block A maps** (already loaded): `../outputs/spectra_blockA.npz` with `../outputs/inventory.csv`.
   Use the 4-NP maps on Ag, PICO and pSERS, and the benzenethiol and pyridine maps on Ag.

## Steps

1. **SPC reader.** No SPC library is installed: write a minimal reader for the Galactic SPC
   new-format (little-endian) layout: 512-byte main header (flags byte, version byte, exponent byte,
   int32 point count, float64 first and last x, int32 subfile count), optional x-value array when
   the flag says so, then 32-byte subfile header and y values (float32 when the exponent byte is
   0x80, otherwise scaled int32). Validate it: report point count and x range for each file, check
   the five scans of a sample look alike, and plot one parsed spectrum. If a file cannot be parsed
   unambiguously, stop and report that instead of guessing.
2. Put all spectra on a common 1 cm-1 axis (intersection only, no extrapolation). Report whether
   the `.spc` x range matches block A's (a mismatch suggests a different instrument or session).
3. Preprocessing: per-spectrum min-max (primary); arPLS + min-max as in phase 2 (sensitivity).
4. Distance between two mean spectra = 1 - Pearson correlation (also report RMSE). Compute:
   - **within a preparation:** average distance of each scan/spot to its own preparation mean;
   - **between preparations, same chemical and substrate type:** pairwise distances among the three
     `.spc` sample means and the block A 4-NP/Ag map mean (four preparations of 4-NP on silver).
     Report the two `23pp-SC` samples' pair separately;
   - **between substrates, same chemical:** block A 4-NP on Ag vs PICO vs pSERS;
   - **between chemicals, same substrate:** block A 4-NP vs benzenethiol vs pyridine on Ag.
5. Ratios: between-substrate / between-preparation, and between-chemical / between-preparation.

## Deliverables

`check_4np.py`, `figure_4np.png` (mean spectra of the preparations and of the substrates, overlaid)
and `RESULTS_4NP.md` (half a page): the distance table, the two ratios, two or three plain
sentences on whether substrate differences are clearly larger than chip-to-chip differences, and
the caveats (preparation methods unknown; possible instrument/session differences; one chemical).
Print the path of `RESULTS_4NP.md` when done.
