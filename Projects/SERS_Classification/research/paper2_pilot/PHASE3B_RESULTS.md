# Phase 3b results — corrected reconstruction loss

Phase 3b repeats phase 3 with one change: reconstruction is the **sum** (not mean) of squared errors over spectral points, KL stays summed over latent dimensions and batch-averaged, cross-entropy is weighted by the number of points (2,269 A / 2,088 B) to preserve the phase-3 head weight, beta = 1 (VAE, CVAE) / 4 (β-VAE), physics has no KL, and the 50-epoch warm-up is unchanged. Everything else is identical; one map per cell: indicative only. Run: `run_phase3.py --recon sum --tag phase3b`; phase 3 outputs untouched.

## E1 — chemical part (Pearson r, seed mean; min–max)

“P2” = phase-2 NMF (k = 7 A / 5 B); “Sig” = phase-2 additive SERS signature (baseline-derived).

| Block / target | P2 NMF | P3 CVAE | P3b CVAE | P3 Phys | P3b Phys |
|---|---:|---:|---:|---:|---:|
| A, ref | 0.37 | 0.08 | **0.29** | 0.20 | 0.19 |
| A, sig | 0.88 | 0.11 | **0.59** | 0.34 | 0.33 |
| B, ref | 0.59 | 0.52 | 0.52 | 0.56 | 0.54 |
| B, sig | 0.37 | −0.19 | 0.01 | −0.07 | −0.01 |

“Ref” = Raman reference; “sig” = SERS signature.

arPLS (A ref/sig, B ref/sig): P2 0.43/0.75, 0.47/0.55; P3b CVAE 0.39/0.68, 0.43/0.51; P3b Phys 0.38/0.54, 0.43/0.49. B signature means hide opposing analytes (CVAE PDCA 0.91, MBA −0.42, MBN −0.47); CVAE improved every A signature analyte (0.81/0.34/0.62), but no neural part beats NMF on references.

## E2 — separation (map-grouped 5-fold probes; pooled BA per spectrum; min–max)

“Var” = phase-2 map-mean R²; “Raw” = PCA(20)+logistic regression, same folds. Chance: A 0.33 analyte / 0.25 substrate; B 0.33 / 0.50.

| Block / target | P2 Var | P2 Raw | P3 VAE | P3b VAE | P3 β | P3b β | P3 CVAE | P3b CVAE | P3 Phys | P3b Phys |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A, analyte | 0.08 | 0.54 | 0.12 | 0.10 | 0.10 | 0.10 | 0.17 | 0.14 | 0.08 | 0.06 |
| A, substrate | 0.80 | 0.71 | 0.29 | **0.61** | 0.52 | **0.62** | 0.62 | 0.65 | 0.56 | 0.54 |
| B, analyte | 0.74 | 1.00 | 0.17 | 0.80 | 0.26 | 0.73 | 0.60 | **0.96** | 0.76 | 0.65 |
| B, substrate | 0.01 | 0.74 | 0.29 | 0.46 | 0.27 | 0.37 | 0.39 | 0.49 | 0.32 | 0.22 |

MIG stayed low (0.03–0.06). Swap joint i-on-j: CVAE 0.18/0.85 (A/B), physics 0.17/0.53; CVAE substrate-j 0.71/0.97. CVAE now transfers both factors in B; physics is unchanged.

## E3 — unseen combinations (held-out whole cell; mean r (RMSE) over cells/seeds)

| Block | P2 additive | P2 other substrate | P2 other analyte | P3 CVAE | P3b CVAE | P3 Phys | P3b Phys |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.78 (0.188) | 0.44 (0.312) | 0.81 (0.138) | 0.47 (0.247) | 0.50 (0.229) | 0.78 (0.193) | 0.78 (0.191) |
| B | 0.96 (0.068) | 0.95 (0.080) | 0.87 (0.201) | 0.85 (0.114) | **0.95 (0.075)** | 0.92 (0.119) | 0.92 (0.113) |

Min–max throughout (phase-2 columns corrected on review: an earlier draft averaged them over both preprocessings). In B the CVAE now ties the best phase-2 baselines (0.95 vs additive 0.96 and other-substrate 0.95); in A it stays below them. Physics is essentially identical to phase 3.

## E4 — unseen substrate / laser (balanced accuracy per map; mean over folds/seeds)

| Block / scheme / prep | P2 best | P3 best (model) | P3b best (model) |
|---|---:|---|---|
| A, LOSO, min–max | 0.75 | 0.38 (VAE) | 0.43 (CVAE) |
| A, LOSO, arPLS | 0.83 | 0.44 (CVAE) | **0.71 (CVAE)** |
| B, LOSO, min–max | 1.00 | 0.81 (Phys) | 0.92 (CVAE) |
| B, LOSO, arPLS | 0.92 | 0.94 (Phys) | **1.00 (CVAE)** |
| B, LOLO, min–max | 1.00 | 0.47 (CVAE) | 0.81 (CVAE) |
| B, LOLO, arPLS | 1.00 | 0.86 (Phys) | 0.92 (CVAE/β) |

“P2 best” = best mean of Random Forest / Extra Trees / PCA+LDA; LOSO/LOLO = leave-substrate-out / leave-laser-out; per-model and per-spectrum BA in `phase3b_e4.csv`. CVAE matches or nearly matches the non-neural baselines in B and is the best rung in A (arPLS), where phase 3 was at chance.

## Active units, failures and runtime

Active units = latent dimensions whose posterior mean varies over training-map spectra with variance > 0.01 (training maps only).

| Model | Dims | Phase 3 active | Phase 3b active | Phase 3b collapsed fits |
|---|---:|---|---|---:|
| VAE | 8 | 0 every fit (78/78) | 4.7 mean (2–8) | 0/78 |
| β-VAE | 8 | 0 every fit (78/78) | 3.4 mean (1–8) | 0/78 |
| CVAE | 8 (A), 10 (B) | blocks collapsed in 64/159 fits | 8.0/8 (A), 9.6/10 (B); chem 4/4 always | 0/159 |
| Physics | n/a | no Gaussian latents | no Gaussian latents (p/q categorical) | 13/159 (12/12 B leave-substrate-out; 1 A arPLS) |

474 fits on an RTX 5080 in 15.1 min (E1 51 s, E2 244 s, E3 295 s, E4 282 s), 0 NaNs or crashed fits. Collapse disappeared for VAE/β-VAE/CVAE; physics still sometimes predicts a single p/q class when a whole substrate is held out.

## Bottom line

The fixed loss changes which rung separates chemical from substrate in B: the CVAE’s supervised chemical block is now strongest (analyte BA 0.96, swap i-on-j 0.85, held-out-cell r 0.95), overtaking the physics decoder, which is essentially unchanged (0.65/0.53/0.92). Plain VAE/β-VAE no longer collapse (4.7 and 3.4 of 8 active units) and rise above chance, but remain weaker than the supervised CVAE and below the raw/additive baselines. No rung separates chemistry in A (CVAE analyte BA 0.14 vs chance 0.33), so the loss fix mainly rescues the latent ladder rather than changing the A conclusion.

Figures: `figures/phase3b_*.png`; CSVs: `outputs/phase3b_*.csv` (E1–E4, probes, MIG, swap, active units, diagnostics, params, runtime).
