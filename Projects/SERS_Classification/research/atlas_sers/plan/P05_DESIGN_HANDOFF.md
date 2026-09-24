# P05 design handoff: support before acquisition-aware training

**Date:** 2026-09-24. **Status:** proposals for the next design gate, not an execution contract.

The next scientific question is whether the completed ordinary CNN can benefit from the repeated-master structure through supervised contrastive learning and paired consistency. Primary preprocessing remains frozen minimal min–max. P08 tests preprocessing separately after the primary method comparison; P14 retains the later RBF-network/SOM extension.

## What the metadata audit can decide

The [T003 assignment](delegation/P05_TASK_003_SOURCE_SUPPORT.md) checks available fitting-role support after the original master and instrument exclusions. It does not infer support from the full 598-row dataset and then assume that support survives inside every training split. Held-test metadata is inspected only for boundary validation; held spectra and predictive outcomes are not used.

Counts of spectrum pairs are not counts of independent specimens. Multiple pairs can share a spectrum or physical master, and the repeated outer splits reuse the same finite dataset. Likewise, two spectra in a chemical/instrument cell do not necessarily represent two distinct specimens.

## Decisions required before runtime implementation

1. **Define behavior when a loss has no support.** A fitting role with only one instrument cannot provide cross-instrument paired consistency, cross-instrument CORAL, or a nontrivial instrument-classification adversary. Do not weaken leakage exclusions to obtain pairs. A proposed rule is to omit an unavailable auxiliary term with explicit accounting while retaining the chemical classifier and any supported terms. Whether this role contributes to a particular mechanism's advancement denominator must be specified separately. No such fallback is frozen by this memo.
2. **Specify master-aware sampling and exact loss equations.** State the distribution over classes, masters, instruments, and observations; the treatment of repeated same-master/same-instrument spectra; zero-positive-anchor accounting; weighting normalization; and KL/cosine combination. Distinguish role-level pair availability from minibatch feasibility. A sampler must be tested on the sparse roles, not merely on dense synthetic examples.
3. **Keep an honest D0 control.** The existing P04 result remains immutable. A changed sampler or optimization schedule is a changed procedure: any additional matched D0 control must be named, costed, and reported alongside the historical baseline, not substituted silently.
4. **Replace the rough estimate with a finite staged budget proposal.** Enumerate loss candidates, optimizer nesting, seeds, eligible source-selection units, optional D4/D5 controls, final refits, calibration, retries/failures, and expected resource bounds. The full Cartesian arithmetic is a warning about an interpretation of the grids, not a recommended or authorized sweep. A staged plan that fixes source-selected optimizer settings before selected loss comparisons is worth evaluating, but requires a dated amendment if it changes the approved search procedure.
5. **Resolve the source-only G3 selection boundary.** Master-grouped validation is not necessarily pseudo-instrument validation. The frozen roles contain both. Define which scores and supported pseudo-domains enter each G3 denominator; never relabel ordinary master-CV evidence as instrument-generalization evidence. A global choice aggregated over tasks must not introduce the current test instrument or test masters through another task's fitting/selection results. Specify the information boundary before generating any new scores.
6. **Freeze runtime details and accounting.** Specify the auxiliary head and total parameter count, conditional alignment/adversary details, epoch inheritance, checkpoint selection, calibration, numerical failures, and immutable run identities. The backbone's parameter count alone is not the full acquisition-aware model's parameter count.

The readiness IDs P05-U01 through P05-U14 remain the tracking vocabulary. A descriptive support audit informs these choices; it does not close them automatically or grant training authorization.

## Next bounded implementation sequence

- Review a versioned finite execution proposal and identify which changes need project-owner approval.
- Have the implementation worker add synthetic-tested loss functions and the master-aware sampler under an explicit file and compute allowlist. Include single-instrument, singleton-class/master, repeated-spectrum, zero-positive, and numerical-gradient cases.
- Construct the actual P05 role, pair, candidate, and fit registries from the approved design; reconcile every fit and control in the budget. These are distinct from audit identities inherited from P04.
- Only then authorize a small source-only numerical smoke, inspect its gradients/loss contributions/collapse checks, and decide whether the finite development ladder may run.

No unavailable paired signal should be concealed, no raw baseline should be removed opportunistically to rescue a loss, and no improvement is claimed before predictive experiments. A valid negative result remains scientifically useful.
