# NATO SERS preprocessing-v2 decision registry

## Immutable controls

- The validated preprocessing-v1 bundle was hash-verified before execution.
- `minimal_minmax`, `arpls_minmax`, and `derivative_1` remain mandatory controls.
- Outer-test and the 98-spectrum flagged-quality cohort were not used for selection.

## Predeclared smoothing decision

Selected final representations: `minimal_minmax`, `arpls_minmax`, `derivative_1`.

| Candidate | Inner clean target | Inner noise target | Repeatable peak recall | Eligible smoother | Final |
|---|---:|---:|---:|---|---|
| `minimal_minmax` | 0.697 | 0.699 | 1.000 | no | yes |
| `minimal_sg7_minmax` | 0.695 | 0.696 | 0.971 | no | no |
| `minimal_sg11_minmax` | 0.693 | 0.697 | 0.903 | no | no |
| `minimal_sg15_minmax` | 0.691 | 0.694 | 0.811 | no | no |
| `arpls_minmax` | 0.707 | 0.699 | 1.000 | no | yes |
| `arpls_sg7_minmax` | 0.705 | 0.696 | 0.961 | no | no |
| `arpls_sg11_minmax` | 0.709 | 0.700 | 0.863 | no | no |
| `arpls_sg15_minmax` | 0.707 | 0.690 | 0.758 | no | no |
| `derivative_1` | 0.701 | 0.682 | nan | no | yes |

Noise-gated policy trigger: `True`.
Noise-gated policy decision: Considered and rejected. Uniform smoothing reduced the target-blind high-frequency score but did not meet the predeclared synthetic-noise accuracy or prediction-agreement benefits and failed repeatable-peak preservation. A fold-fitted gate would add an unsupported threshold, make model inputs split-dependent, and could select instrument resolution rather than field noise.
- No uniform smoother met the repeatable-peak preservation gate.
- No uniform smoother met either classifier-based synthetic-noise benefit gate.
- The flagged-quality cohort is confirmatory and cannot justify a threshold.
- A threshold learned inside each fold would prevent one immutable downstream input archive.
- A threshold fitted on all 598 spectra would expose outer-test distribution information.

## Alignment decision

Accepted: `False`.
Final operation: `none`.

- Named calibration/standard spectra do not cover all ten field instruments.
- Standard-covered instruments: ['Mira-1', 'Mira-2', 'Mira-3', 'Pendar-2', 'Pendar-3']; missing: ['Agilent-1', 'Agilent-3', 'Pendar-1', 'RMX-1', 'RMX-2'].
- A correction for only the covered systems would be system-aware and partial.
- Same-master lags are retained as diagnostics but cannot replace a shared calibration standard because chemistry, sensor, and instrument response are confounded.
- Flexible or target-informed per-spectrum warping remains prohibited.

## Closed operations

- Common axis: 400--1800 cm^-1 at 1 cm^-1.
- Candidate spike detection/derived interpolation: retained from v1.
- Saturation masks: retained from v1; no automatic repair.
- Per-spectrum min--max: retained for reconstructive intensity inputs.
- arPLS parameters: retained from v1.
- First derivative: retained unchanged from the poster/Siamese control.
- Flexible or target-informed alignment: rejected.
- Per-instrument headline preprocessing: rejected.

The full machine-readable gates are copied from `configs/nato_sers_preprocessing_v2.json`.
