# Open-Set Experiments

1. Include Feb26 chemicals in the reference
   Setup: aligned Feb26 spectra to the existing axis, added one Feb26 exemplar per class to the reference, kept the frozen encoder.
   Result: everything classified correctly.
   Metrics: overall `765/765` top-1, true new chemicals `96/96` top-1.

2. Open set with frozen encoder and known-only reference
   Setup: kept `aniline`, `dcm`, `diethylamine`, and `n-hexane` out of the reference and rejected by nearest-distance threshold.
   Result: true unknowns were fully rejected, but Feb26 `benzene` and `pyridine` controls were also rejected because of device shift.
   Metrics: unknown reject `100%`; best threshold `0.14665`; control nearest classes were still correct.

3. Recalibrated threshold with known spectra from both devices
   Setup: threshold selection used original known queries plus Feb26 `benzene` and `pyridine` controls.
   Result: unknowns stayed fully rejected and most known controls were recovered.
   Metrics: threshold `0.23911`; unknown reject `100%`; `benzene` reject `0%`; `pyridine` reject `4%`.

4. Fine-tuned on overlapping known classes from both devices
   Setup: fine-tuned the encoder with original training data plus Feb26 `benzene` and `pyridine`, while keeping the true unknown chemicals held out.
   Result: cross-device knowns moved close enough to be accepted while unknowns stayed rejectable.
   Metrics: threshold `0.12122`; known accept `100%`; held-out controls accept `100%`; held-out unknown reject `100%`.
