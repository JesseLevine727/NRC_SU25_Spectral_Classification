# Figure captions

**Figure 1 — Primary performance.** Master-level unseen-chemical AUROC on
quality-pass and field-stress spectra, acquisition-quality AUROC, and
supported-class balanced accuracy. Small points in chemical-novelty panels
are six held chemicals; error bars are 95% bootstrap intervals after averaging
seeds and outer models. Acquisition-shift small points are leave-one-master-out
jackknife estimates across 69 physical masters, with 95% normal jackknife
intervals. The dotted line marks chance AUROC. Quality and stress classification
markers are horizontally offset.

**Figure 2 — CNN development selection and promotion.** Counts of the
development-only surrogate-selected objective and novelty score across 30
held-chemical/outer-fold tasks, binary preregistered gate decisions, and the
chemical-level comparison between the selected CNN and globally strongest
frozen primary comparator. No true held chemical or stress row selected a
candidate.

**Figure 3 — Source-only domain generalization.** Seed-averaged
`domain_and_sample` quality-pass balanced accuracy for each held instrument,
mean and worst performance across instruments, the gap between `domain_only`
and unseen-master transfer for derivative ERM, and supported/unsupported test
rows. Unsupported classes remain in predictions but not supported-class
accuracy.

**Figure 4 — Negative and sensitivity controls.** Master-label permutation,
fixed preprocessing/augmentation ERM ablations, quality-open AUROC for every
unselected score, and MC-dropout 10/20/50-pass sensitivity. Dots are held
chemicals after technical averaging; MC error bars are SD across chemicals.

**Figure 5 — Fixed-threshold selective behavior.** Accepted-known master
balanced accuracy and unseen-chemical rejection versus thresholds fit only to
known development data. Lines average six held chemicals after seeds and
outer models; light bands are ±1 SD across held chemicals. Requested retention
is not recalibrated on test unknowns or stress spectra.

**Figure 6 — Residual-CNN confusion matrices.** Row-normalized known-class
confusion for quality and stress partitions. Counts pool technical model
repeats only for visualization; inferential summaries remain grouped by held
chemical/master.
