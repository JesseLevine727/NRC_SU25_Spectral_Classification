# T1 step 2: predeclared test (written 2026-11-08, before any step 2 model is fitted or run)

## Question
Does a fine-tuned Siamese matcher beat the fitted calibration at identifying unseen compounds,
measured on a different instrument, against the full DFT library?

## Data
- Library: QM9S (B3LYP/TZVP) + our 8 B3LYP/TZVP compounds = 129,825 calculated spectra, 785 nm
  cross-section, three-range scaling, as in step 1.
- Training queries: our existing 37 spectra of 29 compounds (Flanagan 785 nm + NRC).
- **Primary test set:** the 81 VibraCLIP experimental compounds (Thermo OMNIC library spectra,
  probably FT-Raman 1064 nm) that are in QM9S and not among our 29. All 81, H-bond donors included.
  No test spectrum is used for any fitting, tuning or model choice.
- Secondary only: 9 VibraCLIP spectra of compounds we already have (same compound, new instrument).

## Fixed preprocessing (both sides, all methods)
- Window 500 to 1800 cm^-1 at 2 cm^-1 (VibraCLIP spectra start at 500).
- VibraCLIP queries multiplied by ((1e7/785 - nu) / (1e7/1064 - nu))^4 to map 1064 nm onto the
  785 nm library convention.
- Clip at 0, sqrt intensity, unit norm.

## Methods
- Fixed: sqrt-intensity cosine, Lorentzian FWHM 16 cm^-1 (FWHM 10 also reported).
- Fitted calibration: per-region scale (3) + order-4 Legendre intensity response, fitted on the 37
  training spectra with the same contrastive objective as before (300 Adam steps).
- Siamese: d2w32 two-tower with fixed-cosine skip, re-pretrained once on pseudo-measured QM9S
  spectra at the 500-1800 window (6000 steps, same recipe and seed as the step 1 sweep), with all
  query compounds removed from the pretraining pool. Measured tower fine-tuned on the 37 training
  spectra (300 steps, lr 1e-4), seeds 0, 1, 2.

## Primary metric and decision rule
- Top-1 against the full 129,825-entry library, per test compound.
- Siamese score per compound = mean top-1 indicator over the 3 seeds.
- Delta = Siamese - fitted. 95% CI by paired bootstrap over compounds (2000 resamples, seed 0).
- **Siamese wins** if Delta >= 0.10 and the CI lower bound > 0.
- **Calibration wins** if Delta <= -0.10 and the CI upper bound < 0.
- Otherwise **tie**.

## Controls and secondary analyses (reported, not part of the decision)
- Label-shuffle Siamese (seeds 1-3); flagged if it exceeds fixed by more than 5 points.
- No 1064 nm correction (sensitivity).
- Split by H-bond donor (O-H/N-H) vs none.
- Pooled 110 compounds (old 29 + 81): leave-one-compound-out with learning curve, and
  leave-one-class-out, for fixed / fitted / Siamese.
