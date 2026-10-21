# Phase 3 results — VAE model ladder

Primary min–max; arPLS + min–max for E1/E4 only. Seeds 0/1/2, fixed config (1-D conv encoder, Adam 1e-3 / wd 1e-4, batch 64, ≤300 epochs, patience 30, KL warm-up 50 epochs, MSE). One map per cell: indicative only. Maps are independent; test maps never entered training, validation or early stopping (10% of training-map spots were held out for early stopping). E1 target (b) is the phase-2 additive analyte effect, i.e. baseline-derived. Code: `vae_models.py`, `run_phase3.py`; all values: `outputs/phase3_*.csv`.

Weights (physics templates in brackets): VAE/β-VAE 31,865; CVAE 31,900 (A), 32,412 (B); Physics 42,992 (A; 27,109 + 15,883) and 37,429 (B; 26,989 + 10,440); under 50k excluding templates.

## E1 — chemical part vs Raman reference and phase-2 SERS signature

Pearson r, mean over seeds (B reference averaged over 632/785). “Base” = phase-2 NMF at k = 7 (A) / 5 (B); for the signature, best NMF component vs that signature. CVAE = decoded analyte mean chemical block with substrate (and B laser) blocks at their overall means; Physics = A_a. VAE and β-VAE have no chemical part (n/a). For B the phase-2 signature was computed within substrate×laser cells.

| Block | Ref Base | Ref CVAE | Ref Phys | Sig Base | Sig CVAE | Sig Phys |
|---|---:|---:|---:|---:|---:|---:|
| A (mean of 4NP/BT/Py) | 0.37 | 0.08 | 0.20 | 0.88 | 0.11 | 0.34 |
| B (mean of PDCA/MBA/MBN) | 0.59 | 0.52 | 0.55 | 0.37 | −0.19 | −0.07 |

Per analyte: A Physics 4NP 0.43, BT 0.00, Py 0.16; B Physics 0.19 (PDCA), 0.80 (MBA), 0.67 (MBN). CVAE signatures: A 0.53/−0.56/0.35 (4NP/BT/Py); B 0.88/−0.75/−0.71 (PDCA/MBA/MBN). Neither neural chemical part beat NMF on average. Physics recovered B MBA/MBN references but not PDCA or A BT/Py; against the signature it was positive for 4NP/Py and all B arPLS, strongly negative for MBA/MBN min–max. arPLS means (ref / signature): A NMF 0.43/0.75, CVAE 0.31/0.02, Physics 0.38/0.54; B 0.47/0.55, 0.40/0.19, 0.45/0.46. Peak matching is lenient (CVAE A min–max: 0.99 of peaks at r = 0.08).

## E2 — separation (map-grouped 5-fold probes; pooled BA per spectrum)

“Var” = phase-2 map-mean R²; “Raw” = PCA(20)+logistic regression on real spectra under the same folds. CVAE uses its chemical/substrate block, Physics p/q; own-label probes are shown, cross-probes are in `phase3_e2_probes.csv`. MIG = mean over folds (8 bins).

| Block | Label | Var | Raw | VAE | β-VAE | CVAE | Physics | MIG VAE / β |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | analyte | 0.08 | 0.54 | 0.12 | 0.10 | 0.17 | 0.08 | 0.08 / 0.00 |
| A | substrate | 0.80 | 0.71 | 0.29 | 0.52 | 0.62 | 0.56 | 0.13 / 0.01 |
| B | analyte | 0.74 | 1.00 | 0.17 | 0.26 | 0.60 | 0.76 | 0.00 / 0.00 |
| B | substrate | 0.01 | 0.74 | 0.29 | 0.27 | 0.39 | 0.32 | 0.00 / 0.00 |

Chance is 0.33/0.25 (A analyte/substrate) and 0.33/0.50 (B). Only Physics in B approached the raw probe (0.76 vs 1.00); no latent block beat raw, and CVAE’s substrate block was below chance in B.

**Swap test** (chemical block of i + substrate block of j, different substrate; 500-tree RF on real training spectra; mean over seeds; chance in brackets):

| Block | Model | Analyte i | Substrate j | i on j |
|---|---|---:|---:|---:|
| A | CVAE | 0.23 | 0.23 | 0.07 (0.08) |
| A | Physics | 0.28 | 0.60 | 0.17 (0.08) |
| B | CVAE | 0.44 | 0.55 | 0.27 (0.17) |
| B | Physics | 0.58 | 0.78 | 0.53 (0.17) |

Physics is the only model that transferred both parts in B; CVAE transferred substrate but lost much of the analyte.

## E3 — unseen combinations (held-out whole cell; mean over cells and seeds)

| Block | Phase-2 additive | Phase-2 other substrate | Phase-2 other analyte | CVAE | Physics |
|---|---:|---:|---:|---:|---:|
| A | 0.78 (0.188) | 0.44 (0.312) | 0.81 (0.138) | 0.47 (0.247) | 0.78 (0.193) |
| B | 0.96 (0.068) | 0.95 (0.080) | 0.87 (0.201) | 0.85 (0.114) | 0.92 (0.119) |

Values are mean r (RMSE). Physics matched the additive model in A but not B; CVAE was worse in both. In B, physics beat the other-analyte template (0.92 vs 0.87) but not the additive/other-substrate baselines.

## E4 — unseen substrate / laser (BA per map / per spectrum)

| Block / scheme | Prep | RF | ET | PCA+LDA | VAE | β-VAE | CVAE | Physics |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A, leave-substrate-out | min–max | 0.58/0.52 | 0.33/0.41 | 0.75/0.73 | 0.38/0.38 | 0.32/0.37 | 0.32/0.34 | 0.33/0.34 |
| A, leave-substrate-out | arPLS | 0.67/0.66 | 0.67/0.69 | 0.83/0.79 | 0.29/0.32 | 0.38/0.35 | 0.44/0.44 | 0.33/0.35 |
| B, leave-substrate-out | min–max | 0.92/0.87 | 1.00/0.92 | 1.00/0.99 | 0.33/0.33 | 0.47/0.44 | 0.67/0.62 | 0.81/0.71 |
| B, leave-substrate-out | arPLS | 0.92/0.85 | 0.92/0.90 | 0.83/0.79 | 0.33/0.33 | 0.42/0.37 | 0.56/0.49 | 0.94/0.86 |
| B, leave-laser-out | min–max | 0.75/0.67 | 0.75/0.75 | 1.00/0.98 | 0.33/0.33 | 0.36/0.40 | 0.47/0.49 | 0.42/0.47 |
| B, leave-laser-out | arPLS | 1.00/0.95 | 1.00/0.98 | 1.00/0.94 | 0.33/0.33 | 0.56/0.50 | 0.72/0.71 | 0.86/0.83 |

All ladder models were at/near chance in A. In B, Physics was the best ladder model (up to 0.94 map BA after arPLS), but below PCA+LDA and mostly below RF/ET. The A/Au fold has only BT vs Py (no 4NP/Au) and is not strictly comparable.

## Failures, collapse and runtime

474 fits on an RTX 5080 in 7.5 min (E1 24 s, E2 114 s, E3 106 s, E4 176 s); all 3 seeds kept. No NaNs or crashed fits (`phase3_failures.csv` empty). Plain VAE and β-VAE collapsed in every E2/E4 fold (0 of 8 active units, latent variance ~0), so probes are at chance and MIG ≈ 0. CVAE blocks collapsed in several fits (E2 3/15 A, 4/15 B; E4 7/12 A min–max); its chemical probe was below chance in A (0.17) but useful in B (0.60), its laser block was near-perfect in B (0.99), and its substrate block was below chance in B. Physics p classified B analytes well (0.76) but was weak in A (0.08); q was weak in both. The interaction was nearly constant (G = 0.66–0.82, range ≤0.14), so the analyte×substrate term was effectively unlearned and the good B reconstruction/E3 prediction comes mainly from A_a + B_s.

## Bottom line

Only the physics-structured rung separated chemical from substrate in Block B: p gave 0.76 analyte BA under map-grouped CV, the swap test read analyte i on substrate j 0.53 of the time (chance 0.17), and held-out-cell prediction reached r = 0.92. Plain VAE/β-VAE collapsed, and the CVAE chemical block worked in B (0.60) but not A (0.17), so the useful ingredients were supervised p/q plus the additive A/B decoder, not unsupervised latents. No rung beat the phase-2 baselines on unseen combinations or unseen substrate/laser (E3 physics 0.92 vs additive 0.96; E4 physics ≤0.94 map BA vs up to 1.00 for PCA+LDA), so this pilot does not show that a VAE separates chemistry better than the additive baseline; the physics decoder is the only ladder component that remains interpretable and transferable in Block B.

Figures: `figures/phase3_e1_physics_spectra.png`, `figures/phase3_e2_swap_example.png`, `figures/phase3_e4_comparison.png`. CSVs: `outputs/phase3_e1.csv`, `phase3_e2_summary.csv`, `phase3_e2_mig.csv`, `phase3_e2_swap.csv`, `phase3_e3.csv`, `phase3_e4.csv`, `phase3_params.csv`, `phase3_runtime.csv`, `phase3_diagnostics.csv`.
