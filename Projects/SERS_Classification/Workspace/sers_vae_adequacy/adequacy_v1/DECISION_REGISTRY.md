# SERS VAE adequacy decision registry

1. **Leakage control:** all epochs, architecture, loss, latent-width, and beta decisions used grouped inner NATO data only.
2. **Exact continuation:** the reproduced first 100 epochs agree with the original trajectories within the registered `1e-12` tolerance.
3. **Convergence:** epoch 300 remained insufficient; constant learning rate met the registered convergence definition at epoch 500.
4. **Architecture:** residual/multiscale and single-pool candidates did not give converged, consistent gains; the base two-pool backbone remains frozen.
5. **Loss:** the peak/multiscale objective improved correlation but not repeatable-peak recall; spectral-composite remains frozen.
6. **Latent width:** z32 and z128 did not outperform converged z64 consistently; z64 remains frozen.
7. **KL strength:** beta 0.25 best preserved chemistry and peaks, while beta 4 reduced instrument leakage at unacceptable spectral/chemical cost.
8. **Preprocessing:** arPLS+min-max remains the primary separability view; minimal+min-max remains mandatory because it preserves spectra and peaks better.
9. **Outer confirmation:** 500 epochs improved quality and field-stress performance over the original VAE but did not beat frozen PCA/logistic on ordinary grouped folds.
10. **Domain confirmation:** unseen-instrument transfer improved, but unseen sensor-family, field-stress, and same-master invariance remain unresolved.
11. **Poster:** minimal-view transfer improved; arPLS transfer fell, especially for held-out Ag and PICO, demonstrating a preservation/invariance trade-off.
12. **Adequacy:** retain the selected model as a converged mixed-latent comparator/backbone, not as an invariant or disentangled representation.
13. **Next boundary:** the next goal may partition chemical and nuisance latents, but must not reopen this frozen backbone using locked outcomes.

Failed inner gates: `gate_instrument_probe, gate_same_master_distance`.
