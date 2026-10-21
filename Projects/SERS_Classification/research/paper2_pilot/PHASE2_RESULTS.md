# Phase 2 results — chemometric baselines

Primary: per-spectrum min–max. Sensitivity: despike (5, 10) + arPLS (λ = 1e5, 12 iterations,
rtol 1e-3, clip 60) + min–max (`atlas_sers`), applied identically to SERS and reference spectra.
Seeds fixed, CPU only, no neural networks. r = Pearson; peaks = reference’s 8 tallest maxima
matched within ±10 cm⁻¹ by the best positive component. Code: `baselines.py`; full values:
`outputs/e1_reference_recovery.csv`, `e2_variance.csv`, `e3_unseen.csv`, `e4_classification.csv`.

## E1 — reference recovery (Base = reference vs analyte mean; Block A k = 7, Block B k = 5; k ± 1 in text)

| Block | Analyte | Ref | Base mm | NMF mm | MCR mm | Base ar | NMF ar | MCR ar |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | 4NP | 632 | 0.43 | 0.61 (4/8) | 0.61 (4/8) | 0.57 | 0.59 (4/8) | 0.59 (4/8) |
| A | BT | unknown | 0.01 | 0.29 (7/8) | 0.29 (7/8) | 0.39 | 0.44 (7/8) | 0.44 (7/8) |
| A | Py | unknown | 0.16 | 0.21 (4/8) | 0.21 (3/8) | 0.18 | 0.25 (6/8) | 0.25 (6/8) |
| B | MBA | 632 | 0.73 | 0.72 (5/8) | 0.72 (5/8) | 0.33 | 0.44 (1/8) | 0.44 (1/8) |
| B | MBA | 785 | 0.67 | 0.94 (5/8) | 0.94 (6/8) | 0.40 | 0.43 (5/8) | 0.43 (5/8) |
| B | PDCA | 632 | 0.03 | 0.19 (3/8) | 0.19 (3/8) | 0.28 | 0.31 (4/8) | 0.31 (4/8) |
| B | PDCA | 785 | 0.35 | 0.31 (3/8) | 0.31 (3/8) | 0.22 | 0.33 (4/8) | 0.33 (4/8) |
| B | MBN | 632 | 0.77 | 0.66 (0/8) | 0.66 (0/8) | 0.61 | 0.64 (4/8) | 0.64 (4/8) |
| B | MBN | 785 | 0.61 | 0.72 (4/8) | 0.72 (4/8) | 0.73 | 0.66 (6/8) | 0.66 (6/8) |

In Block A NMF/MCR beat the analyte mean for all analytes but r stays low (0.21–0.61); in Block B
only MBA 785 is strong (0.94), MBA 632 and MBN 632 are no better than the mean, and arPLS lowers
MBA (0.94 → 0.43) while making the advantage over the mean more consistent (5/6). k±1 changed mean
r by ≤0.04 (A 0.40/0.37/0.37 min–max, 0.41/0.43/0.42 arPLS; B 0.59/0.59/0.60, 0.45/0.47/0.49);
B k=7 gave 0.59/0.49, and MCR matched NMF, so they are not independent evidence.

## E2 — variance decomposition (map-mean spectra)

| Block | Prep | Analyte | Substrate | Laser | Additive | Full cells | Interaction |
|---|---|---:|---:|---:|---:|---:|---:|
| A | min–max | 0.08 | 0.80 | — | 0.86 | 1.00 | 0.14 |
| A | arPLS | 0.18 | 0.46 | — | 0.65 | 1.00 | 0.35 |
| B | min–max | 0.74 | 0.01 | 0.17 | 0.93 | 1.00 | 0.07 |
| B | arPLS | 0.62 | 0.09 | 0.06 | 0.78 | 1.00 | 0.22 |

Block A is substrate-dominated (0.80; analyte 0.08), while Block B is analyte-dominated (0.74).
The additive model leaves a 7–35% interaction gap; “full cells” is 1.00 by construction (one map
per cell).

## E3 — unseen combinations (held-out whole cell; mean r (RMSE) over cells)

| Block | Prep | Additive | Other substrate | Other analyte |
|---|---|---:|---:|---:|
| A | min–max | 0.78 (0.188) | 0.44 (0.312) | 0.81 (0.138) |
| A | arPLS | 0.61 (0.147) | 0.46 (0.145) | 0.54 (0.110) |
| B | min–max | 0.96 (0.068) | 0.95 (0.080) | 0.87 (0.201) |
| B | arPLS | 0.86 (0.060) | 0.74 (0.074) | 0.65 (0.103) |

In Block B the additive model predicts held-out cells very well, but a same-substrate template
from a *different* analyte is nearly as good under min–max (0.87), so much of the predictability
is substrate/acquisition shape, not chemistry; in Block A the additive model (0.78) does not beat
that template (0.81) under min–max, though it becomes best after arPLS (0.61 vs 0.54).

## E4 — unseen substrate / laser (fold-mean balanced accuracy: per spectrum / per map)

| Block / scheme | RF mm | ET mm | LDA mm | RF ar | ET ar | LDA ar |
|---|---:|---:|---:|---:|---:|---:|
| A, leave-substrate-out (4 folds) | 0.52 / 0.58 | 0.41 / 0.33 | 0.73 / 0.75 | 0.66 / 0.67 | 0.69 / 0.67 | 0.79 / 0.83 |
| B, leave-substrate-out (2 folds) | 0.87 / 0.92 | 0.92 / 1.00 | 0.99 / 1.00 | 0.85 / 0.92 | 0.90 / 0.92 | 0.79 / 0.83 |
| B, leave-laser-out (2 folds) | 0.67 / 0.75 | 0.75 / 0.75 | 0.98 / 1.00 | 0.95 / 1.00 | 0.98 / 1.00 | 0.94 / 1.00 |

Block B is far above 33% chance in every scheme, while Block A is at or near chance for the pSERS
fold and mostly for Ag, with only PCA+LDA consistently well above chance; the Au fold tests only BT
versus Py (no 4NP/Au cell) and is not strictly comparable.

Figures: `figures/e1_reference_corr.png`, `e1_best_components.png`, `e2_variance.png`,
`e3_unseen.png`, `e4_classification.png`.

## Limitations

- **Spots are repeat views, not replicates.** Maps are the independent unit and all splits respect
  that, but per-spectrum scores are optimistic and map-level n is only 11 (A) or 12 (B).
- **Mostly one map per cell.** E2’s full-cell model is saturated at 1.00 and its “interaction” is
  the additive residual; E4’s map-level metric averages spots within one map per cell. This is a
  feasibility check, not a generalization claim.
- **Reference and axis limits.** BT/Py references have no laser in their names (632-class axis,
  recorded as unknown); Block B is scored against both 632 and 785 references; the common
  reference axis (365–2445 cm⁻¹) misses the full 785 window, and the Block B axis truncates 632
  below 366 and 785 above 2453 cm⁻¹.
- **Method limits.** Min–max retains background that inflates r; arPLS removes it but can remove
  signal. NMF/MCR have rotational ambiguity and MCR hit its iteration cap. Peak matching is lenient
  (BT: 7/8 at r = 0.29). The do-nothing baseline averages over substrates/lasers; E3’s Block B
  donors were matched on laser; E4 uses fixed 20-component PCA+LDA and untuned 500-tree forests.
