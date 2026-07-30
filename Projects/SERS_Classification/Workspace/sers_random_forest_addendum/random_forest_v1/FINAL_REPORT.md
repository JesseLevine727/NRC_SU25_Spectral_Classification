# NATO SERS random-forest addendum v1 — final report

## Headline

A random forest was selected inside each master-group outer fold. Its locked balanced accuracy was 0.690 ± 0.041 on all 598 spectra, 0.742 ± 0.045 on the 500 quality-pass spectra, and 0.387 ± 0.153 on the 98 field-quality-stress spectra. Intervals use five physical-master folds after averaging the three declared forest seeds.

## Locked comparison

| Model | Strict BA | Quality BA | Stress BA |
|---|---:|---:|---:|
| Classical | 0.685 | 0.745 | 0.478 |
| Siamese | 0.632 | 0.677 | 0.370 |
| Contrastive successor | 0.701 | 0.731 | 0.430 |
| Random forest | 0.690 | 0.742 | 0.387 |

These are not row-random train/test splits: no physical master sample crosses a fold. Seed repeats are not counted as independent samples.

## What was selected

- quality_pass: 1/5 folds selected derivative_1, max_features=0.1, max_depth=None, min_leaf=4, inverse_master_frequency.
- quality_pass: 1/5 folds selected derivative_1, max_features=0.1, max_depth=None, min_leaf=4, uniform_rows.
- quality_pass: 1/5 folds selected derivative_1, max_features=sqrt, max_depth=12, min_leaf=2, inverse_master_frequency.
- quality_pass: 1/5 folds selected derivative_1, max_features=sqrt, max_depth=12, min_leaf=2, uniform_rows.
- quality_pass: 1/5 folds selected derivative_1, max_features=sqrt, max_depth=None, min_leaf=1, uniform_rows.
- strict_core: 2/5 folds selected derivative_1, max_features=0.1, max_depth=None, min_leaf=2, inverse_master_frequency.
- strict_core: 1/5 folds selected derivative_1, max_features=0.1, max_depth=12, min_leaf=2, uniform_rows.
- strict_core: 1/5 folds selected derivative_1, max_features=0.1, max_depth=12, min_leaf=4, uniform_rows.
- strict_core: 1/5 folds selected derivative_1, max_features=sqrt, max_depth=None, min_leaf=4, inverse_master_frequency.

## Field shift and uncertainty

- Confidence-only quality-versus-stress detection: mean AUROC 0.847 over fold/seed runs. This tests whether low maximum probability flags bad field spectra; it does not prove denoising.
- Master-label permutation control: mean balanced accuracy 0.167, maximum 0.370; seven-class chance is 0.143.

## Held-domain results

| Training subset | Protocol | Held domain | BA mean | 95% half-width | Domains |
|---|---|---|---:|---:|---:|
| quality_pass | domain_and_sample | instrument | 0.585 | 0.165 | 10 |
| quality_pass | domain_only | instrument | 0.620 | 0.112 | 10 |
| quality_pass | domain_and_sample | sensor_family | 0.479 | 0.548 | 2 |
| quality_pass | domain_only | sensor_family | 0.496 | 0.259 | 4 |
| strict_core | domain_and_sample | instrument | 0.562 | 0.154 | 10 |
| strict_core | domain_only | instrument | 0.608 | 0.108 | 10 |
| strict_core | domain_and_sample | sensor_family | 0.611 | 0.458 | 3 |
| strict_core | domain_only | sensor_family | 0.508 | 0.218 | 4 |

Instrument/sensor means average the declared forest seeds within each held domain first. Wide intervals reflect very few domains and incomplete analyte-domain support.

## Predictive spectral regions

- 980–1000 cm⁻¹: mean held-fold BA drop 0.041.
- 1100–1120 cm⁻¹: mean held-fold BA drop 0.027.
- 1000–1020 cm⁻¹: mean held-fold BA drop 0.023.
- 1340–1360 cm⁻¹: mean held-fold BA drop 0.021.
- 740–760 cm⁻¹: mean held-fold BA drop 0.014.
- 1420–1440 cm⁻¹: mean held-fold BA drop 0.008.
- 520–540 cm⁻¹: mean held-fold BA drop 0.006.
- 880–900 cm⁻¹: mean held-fold BA drop 0.005.
- 1020–1040 cm⁻¹: mean held-fold BA drop 0.004.
- 860–880 cm⁻¹: mean held-fold BA drop 0.004.

Band permutation measures model reliance, not causal chemical assignment. Adjacent Raman variables are correlated, so importance can be divided among neighboring bands.

## Interpretation

The forest is a nonlinear classifier over preprocessed intensity variables. It does not learn a cleaned spectrum, reconstruct a chemical-only signal, or establish chemical/nuisance factorization. Its value is as a rigorous small-data baseline and as evidence for where performance fails under real field and domain shift.
