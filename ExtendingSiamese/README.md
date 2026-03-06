# ExtendingSiamese

Cleaned working area for the standard Raman Siamese extension and open-set experiments.

## Layout

- [`data/`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/data): fixed inputs and raw Feb26 spectra
- [`models/`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/models): pretrained and fine-tuned `.pth` weights
- [`scripts/`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/scripts): reproducible experiment and plotting scripts
- [`results/`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/results): CSV/JSON outputs
- [`figures/`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/figures): plots
- [`notebooks/`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/notebooks): exploratory notebook

## Main Scripts

- [`scripts/evaluate_feb26_extension.py`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/scripts/evaluate_feb26_extension.py): align Feb26 spectra and test the frozen encoder after adding the new classes into the reference
- [`scripts/evaluate_open_set_feb26.py`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/scripts/evaluate_open_set_feb26.py): frozen-encoder open-set analysis and threshold sweep
- [`scripts/train_cross_device_knowns.py`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/scripts/train_cross_device_knowns.py): fine-tune on overlapping known classes from both devices
- [`scripts/evaluate_open_set_retrained.py`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/scripts/evaluate_open_set_retrained.py): open-set evaluation after cross-device fine-tuning
- [`scripts/plot_chemical_views.py`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/scripts/plot_chemical_views.py): spectra galleries, centroid heatmaps, and centroid UMAP plots

## Open-Set Summary

1. Included new Feb26 chemicals in the reference
   Result: the frozen encoder classified everything correctly once one aligned Feb26 exemplar per class was added to the reference.
   Metrics: overall `765/765`, new chemicals `96/96`, both `100%` top-1.
   Outputs: [`results/evaluation_plus_feb26.json`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/results/evaluation_plus_feb26.json)

2. Open set with the frozen encoder and known-only reference
   Result: true unknowns were all rejected, but Feb26 `benzene` and `pyridine` controls were also rejected because device shift pushed their distances up.
   Metrics: unknown reject `100%`; controls still had the correct nearest class, but failed the global threshold.
   Outputs: [`results/open_set_summary.json`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/results/open_set_summary.json)

3. Recalibrated threshold using known spectra from both devices
   Result: this looked more like a threshold-calibration problem than a complete embedding failure.
   Metrics: threshold moved from about `0.14665` to `0.23911`; unknown reject stayed `100%`; Feb26 controls were mostly recovered (`benzene` reject `0%`, `pyridine` reject `4%`).
   Outputs: [`results/open_set_threshold_sweep_recalibrated.csv`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/results/open_set_threshold_sweep_recalibrated.csv)

4. Fine-tuned on overlapping known classes from both devices, then reran open set
   Result: cross-device knowns moved much closer while the held-out unknowns stayed separable.
   Metrics: best threshold about `0.12122`; known queries accepted `100%`; held-out controls accepted `100%`; held-out unknowns rejected `100%`.
   Outputs: [`results/open_set_summary_retrained.json`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/results/open_set_summary_retrained.json)

## Visual Inspection

- Spectra gallery: [`figures/chemical_spectra_gallery.png`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/figures/chemical_spectra_gallery.png)
- Device-shift stacks: [`figures/device_shift_stacked_spectra.png`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/figures/device_shift_stacked_spectra.png)
- Centroid similarity panels: [`figures/centroid_similarity_panels.png`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/figures/centroid_similarity_panels.png)
- Centroid UMAP panels: [`figures/centroid_umap_panels.png`](/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/ExtendingSiamese/figures/centroid_umap_panels.png)
