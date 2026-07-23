# NATO SERS field-trial data audit and VAE roadmap

Audit date: 2026-07-22  
Canonical source: `NRC_SU25_Spectral_Classification/2026July21/NATO SERS Data`  
Scope: SERS measurements only; the sibling `Cell Imaging Data` directory is unrelated and excluded.

## Executive conclusion

The archive is usable, but the files cannot safely be treated as one ready-made matrix of spectra. It mixes normal Raman and SERS observations, duplicate conversions, several vendor export conventions, calibration and unlogged scans, missing instrument archives, inconsistent log entries, and highly unequal baselines and intensity scales.

An auditable reconstruction produces:

- 1,172 expanded recording-log observations;
- 721 observations with an explicitly named SERS sensor;
- 626 of those with a matching readable spectrum;
- 598 strict-core spectra after resolving ground truth and removing contradictory source references;
- 500 conservative quality-pass spectra after additionally applying the field notes' severe/low-signal flags;
- 7 target classes, 4 normalized SERS sensor families, 10 Raman instruments, and 69 represented master samples;
- a common 400--1800 cm⁻¹ grid with 1,401 points.

The most important scientific limitation is design confounding. Sensor, Raman instrument, station/matrix, target class, and physical sample are not fully crossed. A random spectrum split will therefore exaggerate generalization, and even a leave-one-sensor-out split can be impossible to interpret. In the strictest p-SERS test, holding out both p-SERS and its physical specimens leaves **zero supported p-SERS target classes** in training.

The original two-latent SERS-VAE idea is a useful starting hypothesis, but this archive requires at least separate treatment of chemical identity, SERS sensor, Raman instrument/vendor processing, and matrix/acquisition effects. A β-VAE alone cannot identify those factors from this confounded design. The strongest available supervision is the set of repeated measurements sharing a `master_sample_id` across instruments or sensors.

## 1. What is in `2026July21`

The outer directory contains two independent collections:

| Directory | Size | Relevance |
|---|---:|---|
| `Cell Imaging Data` | 95 MB | HeLa imaging files; not SERS and excluded |
| `NATO SERS Data` | 1.4 GB | March 2024 NATO Raman/SERS field-trial archive |

There is another complete copy at `/home/elfo/Documents/NRC/2026July21/NATO SERS Data`. All 7,148 files were compared with the canonical copy and are byte-identical. The exception-like cases in `Z Converted` are not scientific differences: five Mira text files use a different delimiter but contain the same numeric values.

### Archive inventory

| Extension | Files | Meaning |
|---|---:|---|
| `.spc` | 2,516 | Binary spectra, including duplicates in `Z Converted` and RMX dark spectra |
| `.txt` | 1,420 | RMX exports, converted spectra, and notes |
| `.pdf` | 1,341 | Mira/Pendar reports and duplicates |
| `.csv` | 934 | Readable Agilent/Mira/Pendar spectra |
| `.prb` | 933 | Password-protected Pendar bundles and duplicates |
| `.xlsx` | 2 | Recording log and master sample list |
| `.cez` | 2 | Agilent Resolve containers |

The scientific source tree is organized as Agilent 1/3, Mira 1/2/3, Pendar 1/2/3, and RMX 1/2. The log also names Agilent 2 and PMCDS, but their spectra are absent.

`Z Converted` contains 2,882 convenience conversions of source files. Of these, 2,877 are byte-for-byte duplicates. The remaining five are delimiter-only Mira conversions with identical numeric arrays. They must not be added as new observations.

## 2. Notes and workbook semantics

### `Notes.txt`

The note defines four SERS sensor types and explicitly states:

> `na (no SERS sensor, Raman spectra)`

Consequently, literal `na` is an exclusion rule for the SERS-only dataset, not a missing SERS label. The following aliases were normalized while retaining every original string in `sensor_raw`:

| Normalized family | Recorded aliases and variants |
|---|---|
| `pSERS_Metrohm_silver` | p-SERS, P-SERS Ag, Ag P-SERS, Metrohm silver, MR AG SRS, MAG/MAg, Silver SERS |
| `NRC_Canadian_SERS` | NRC-SRS, NRC SERS, Canadian/Can/CAN SERS, CAN-H2, NRC Sensor ANH2, NRC SERS CAN H2, KI and Au variants |
| `H_SERS_H_Kit` | H-Kit, H-SERS Ag, H-Ag SERS |
| `GaN_polymer` | AgGaN, AuGaN, AgPol, AuPol |

There are 20 rows where a named SERS sensor coexists with `Normal Raman = Y`. The explicit sensor name is treated as authoritative, but `sensor_flag_conflict=True` preserves the inconsistency for sensitivity analysis. Three rows say `Normal Raman = N` but omit the sensor; these are marked ambiguous and excluded from the strict SERS set:

- session 1 row 274: C151, Mira-2, scan 53;
- session 2 row 76: pill 13, Mira-3, scan 71;
- session 2 row 77: pill 13, Pendar-3, scan M0353.

### Master sample list

The master workbook contains 73 samples: pills 1--24, surfaces 25--49, and CWA 50--73. It is the only appropriate source for the chemical ground-truth label.

| Target | Master samples | Strict-core spectra | Strict-core master samples |
|---|---:|---:|---:|
| Benzyl fentanyl | 18 | 120 | 15 |
| 4-ANPP | 11 | 102 | 11 |
| Blank | 9 | 87 | 8 |
| Acetaminophen | 11 | 81 | 11 |
| Ethyl paraoxon | 7 | 54 | 7 |
| 4-nitrophenol | 8 | 92 | 8 |
| Ethanol | 9 | 62 | 9 |

The labels `Results` and `Target Det (Y/N)` in the recording workbook are instrument/library or operator outcomes. They are **not** ground truth and must never be used as the target label. They can later be retained as auxiliary performance annotations.

The master description is also parsed into `sample_matrix`, `carrier_geometry`, and `nominal_concentration`. Surface entries distinguish Hexa-L and Square-P coupons; CWA entries preserve 3 or 4 mM when present. A master-list `Blank` is a valid negative-class specimen. It is different from untracked `BG`, `-`, or standalone background rows.

### Recording log

The three recording sheets contain 1,171 non-empty rows. One row records two Mira scans (`4214-4213`), so it expands to 1,172 observations. Compact CWA identifiers such as `C150`, `C1S0`, and `C2S4` are resolved to master IDs. The known Agilent-1 typo `1798` is corrected to source scan `2798`, with both values retained.

## 3. Spectral formats and source selection

Only the readable source exports are used for the model matrix. SPC, PDF, PRB, CEZ, and dark-spectrum paths remain in the manifest for provenance.

| System | Indexed readable field scans | Native axis | Canonical signal and caveats |
|---|---:|---|---|
| Agilent-1 | 54 | CSV 200--2000; SPC 350--2000 | Aggregate CSV `SORS` column; below 350 is padded and excluded |
| Agilent-3 | 157 | same | Same aggregate convention |
| Mira-1 | 74 | 400--2300, 1 cm⁻¹ | Two-column CSV; calibration/suitability exports excluded from the scan index |
| Mira-2 | 166 | same | Two-column CSV |
| Mira-3 | 157 | same | Two-column CSV |
| Pendar-1 | 115 | 275--1849, 1 cm⁻¹ | Two-column CSV |
| Pendar-2 | 249 | same | Two-column CSV |
| Pendar-3 | 160 | same | Two-column CSV |
| RMX-1 | 115 | scan-dependent, approximately -92 to 2915 | Main `spectrum` block; processed output plus a separate `spectrumdark` block |
| RMX-2 | 126 | scan-dependent, approximately -93 to 2907 | Same; interpolate each scan using its own axis |

These indexes contain 1,373 field-like source scans, but only 1,036 unique scans are referenced by the recording log. The other 337 files include pre/post, calibration, suitability, or otherwise unlogged measurements and are not eligible merely because they occur in a vendor directory.

The readable signals were checked against representative binary SPC files:

- Mira and Pendar CSV values match SPC values to export rounding (maximum intensity differences around 0.005 in checked files);
- RMX main and dark TXT blocks match their respective SPC files to roughly 10⁻⁶;
- Agilent `SORS` columns match the corresponding SPC curves to rounding (median checked RMSE about 0.00027).

This supports using the readable CSV/TXT exports without requiring a proprietary binary reader.

### Common spectral region

The literal intersection is approximately 400--1849 cm⁻¹. The generated dataset uses **400--1800 cm⁻¹ at 1 cm⁻¹ spacing**, because that is safely supported by every system and avoids edge behavior. Interpolation must be performed separately for each spectrum, especially RMX, whose calibrated axes move slightly between scans.

### Containers and reports

- Mira/Pendar PDFs were readable and used to extract serial, software/firmware, exposure or integration metadata, averages, laser setting, smart-tip type, suitability, and duration when available.
- All 23 strict-core Mira-1 spectra carry a failed system-suitability state. Mira-2 and Mira-3 strict-core spectra carry passing states.
- Pendar report timestamps are not reliable identifiers: some clocks report 2018, 2023, March 2024, or April 5. The log scan ID is authoritative.
- Pendar `.prb` files are password-protected 7z archives. Their contents are unnecessary because CSV/SPC agreement was verified.
- Agilent `RES90090.cez` is an encrypted ZIP containing a large database. `RES90152.cez` lacks a usable central directory and appears incomplete or corrupt. The per-scan SPC and aggregate CSV exports remain complete enough for this dataset.
- The RMX main spectrum is already a vendor output. Do **not** subtract the separately stored dark spectrum again without a controlled validation showing that the main block is uncorrected.

## 4. The strict SERS dataset

### Inclusion funnel

| Sequential rule | Observations remaining |
|---|---:|
| Explicitly named SERS sensor | 721 |
| Readable source spectrum matched | 626 |
| Master sample and target resolved | 615 |
| No contradictory reuse of the same source scan | 599 |
| One primary observation per consistent repeated reference | 598 |
| Numeric spectrum valid | 598 |

The 95 unmatched named-SERS observations comprise 56 PMCDS measurements, one Agilent-2 measurement, one record with no recognizable system, 31 RMX-1 rows without a logged filename, four RMX-2 rows without a filename, and two Mira-3 rows without a filename.

The archive contains 38 manifest rows involved in contradictory source references; 19 of those rows are named SERS observations. Examples include Mira-2 scan 26 assigned to two sensor families, Pendar-2 scan 403 assigned to ethanol and 4-nitrophenol, Pendar-3 scan 396 assigned to acetaminophen and benzyl fentanyl, and RMX-1 scan 56 assigned both to a background and a sample. All members of a contradictory source group are excluded rather than guessing.

After source-reference deduplication, the 598 common-grid arrays contain no byte-identical duplicate spectra.

### Strict-core balance

| Instrument | Spectra | Sensor-family composition summary |
|---|---:|---|
| Agilent-1 | 29 | 21 p-SERS, 8 NRC |
| Agilent-3 | 73 | 52 p-SERS, 12 H-Kit, 6 GaN/polymer, 3 NRC |
| Mira-1 | 23 | 20 p-SERS, 3 GaN/polymer |
| Mira-2 | 75 | 54 H-Kit, 16 NRC, 4 GaN/polymer, 1 p-SERS |
| Mira-3 | 72 | 52 p-SERS, 20 NRC |
| Pendar-1 | 46 | 32 p-SERS, 14 NRC |
| Pendar-2 | 98 | 49 H-Kit, 41 p-SERS, 8 GaN/polymer |
| Pendar-3 | 70 | 47 p-SERS, 17 NRC, 6 H-Kit |
| RMX-1 | 47 | 32 H-Kit, 13 NRC, 2 GaN/polymer |
| RMX-2 | 65 | 58 p-SERS, 7 NRC |

Sensor-family totals are 324 p-SERS/Metrohm silver, 153 H-SERS/H-Kit, 98 NRC/Canadian SERS, and 23 GaN/polymer.

### Design confounding

Sensor and station are not independent:

| Station | p-SERS | NRC | H-Kit | GaN/polymer |
|---|---:|---:|---:|---:|
| Pills | 166 | 42 | 0 | 0 |
| Surfaces | 158 | 18 | 1 | 5 |
| CWA | 0 | 38 | 152 | 18 |

Target and sensor are also not fully crossed:

| Target | p-SERS | NRC | H-Kit | GaN/polymer |
|---|---:|---:|---:|---:|
| Benzyl fentanyl | 100 | 15 | 0 | 5 |
| 4-ANPP | 85 | 17 | 0 | 0 |
| Blank | 67 | 20 | 0 | 0 |
| Acetaminophen | 72 | 8 | 1 | 0 |
| Ethyl paraoxon | 0 | 14 | 40 | 0 |
| 4-nitrophenol | 0 | 13 | 61 | 18 |
| Ethanol | 0 | 11 | 51 | 0 |

Thus, a network can learn target shortcuts from station, substrate, or instrument identity. Missing cells cannot be repaired by class weights.

### Quality annotations

The strict core retains questionable spectra rather than silently deleting them. It contains 88 rows with severe field-note flags and 23 with low-signal/noise flags; 13 overlap, so the conservative quality-pass set contains 500 spectra.

| Instrument | Core | Severe | Low/noise | Quality-pass |
|---|---:|---:|---:|---:|
| Agilent-1 | 29 | 0 | 0 | 29 |
| Agilent-3 | 73 | 1 | 0 | 72 |
| Mira-1 | 23 | 0 | 2 | 21 |
| Mira-2 | 75 | 0 | 1 | 74 |
| Mira-3 | 72 | 0 | 1 | 71 |
| Pendar-1 | 46 | 6 | 0 | 40 |
| Pendar-2 | 98 | 52 | 11 | 42 |
| Pendar-3 | 70 | 28 | 6 | 42 |
| RMX-1 | 47 | 1 | 1 | 45 |
| RMX-2 | 65 | 0 | 1 | 64 |

Primary results should be reported on the strict core and repeated on the quality-pass subset. Training may use quality weights, but test-set filtering must be declared in advance.

## 5. Baseline and vendor-processing differences

The instrument outputs are not on a common intensity scale. On the 400--1800 grid, the median spectrum level ranges from roughly 0.09 for RMX, 12--50 for Pendar, 138--142 for Agilent, and 10,000--19,000 for Mira. Pooling raw intensities would make device identity trivial.

Two baseline summaries deliberately tell different parts of the story:

| Instrument | Rolling lower-envelope span / spectral span | Candidate AsLS span / spectral span |
|---|---:|---:|
| Agilent-1 | 0.106 | 0.006 |
| Agilent-3 | 0.120 | 0.008 |
| Mira-1 | 0.899 | 0.705 |
| Mira-2 | 0.708 | 0.462 |
| Mira-3 | 0.887 | 0.691 |
| Pendar-1 | 0.055 | 0.033 |
| Pendar-2 | 0.098 | 0.026 |
| Pendar-3 | 0.061 | 0.033 |
| RMX-1 | 0.015 | 0.367 |
| RMX-2 | 0.003 | 0.551 |

The rolling proxy treats Agilent/Pendar and especially RMX lower envelopes as nearly flat, while the candidate AsLS fit interprets broad RMX structure as baseline. Mira is consistently dominated by a strong curved fluorescence/background component under both measures. This disagreement is important: there is no ground-truth baseline in the archive, and broad chemical structure can be removed by an overly flexible correction.

Uniform AsLS (`λ=10⁶`, asymmetry `p=0.001`, 12 iterations) followed by SNV produced only a modest grouped linear target-classification change, from 0.693 to 0.704 mean balanced accuracy. It did not make the systems invariant: grouped instrument classification was 0.857 from raw-SNV and **0.892 after AsLS-SNV**. The latter increase means the correction can expose or amplify vendor-specific features.

Therefore, “apply baseline correction” is a model-selection question, not a settled cleanup step.

## 6. Recommended preprocessing protocol

Preserve all stages rather than overwriting spectra.

1. **Select observations from the log.** Never train on every file in a vendor folder; many are calibration, pre/post, normal Raman, repeated conversions, or unlogged scans.
2. **Use the readable canonical signal.** Agilent `SORS`, Mira/Pendar CSV intensity, and the RMX main `spectrum` block. Keep source and report paths.
3. **Apply the 400--1800 cm⁻¹ grid.** Sort/check each native axis, reject non-finite or constant data, then interpolate once per spectrum.
4. **Flag spikes and saturation.** Detect isolated extreme second differences and long plateaus. Keep the unmodified input and a spike mask. Do not automatically erase broad or repeatable peaks.
5. **Compare at least two declared input branches.**
   - vendor-output spectrum + per-spectrum SNV;
   - one uniform baseline algorithm + SNV.
6. **Retain background information.** Save the estimated baseline separately. A two-channel input (`corrected spectrum`, `estimated baseline`) or a decoder conditioned on baseline coefficients is safer than destroying this variation irreversibly.
7. **Tune preprocessing only within training folds.** Compare AsLS/airPLS parameter grids by reconstruction, peak preservation, analyte performance, and domain leakage. Never select parameters from the held-out instrument/sensor result.
8. **Start without smoothing or derivatives.** Add a mild Savitzky-Golay or derivative branch only if grouped validation shows a benefit. Derivatives can amplify the noisy Pendar measurements and make VAE reconstruction harder.
9. **Use SNV or robust vector scaling as the initial scale correction.** Avoid global min-max scaling and raw intensity pooling. MSC is not the first choice because the archive has no single defensible reference spectrum spanning all systems.
10. **Do not blindly divide by integration/exposure.** Acquisition metadata and vendor processing differ; some systems use auto modes and processed units. Preserve exposure, averages, laser setting, gain, duration, suitability, and smart-tip type as covariates.
11. **Treat wavenumber alignment conservatively.** Estimate one small instrument-level correction from standards if the calibration scans support it. Avoid flexible per-sample warping, which can move peaks toward class templates and leak labels.
12. **Run both core and quality-pass analyses.** Report whether conclusions depend on removing the field failures.

The three generated numeric arrays—vendor output, raw-SNV, and candidate AsLS-corrected-SNV—exist to support this comparison. The AsLS branch is explicitly a candidate, not the frozen production representation.

## 7. Dataset schema and generated artifacts

Every row has a stable `observation_uid`. Important field groups are:

- source identity: session, Excel row, logged/corrected scan ID, instrument, source paths, format, reference conflicts;
- ground truth: master sample ID, scenario, station, master description, target analyte, matrix, carrier geometry, nominal concentration;
- SERS metadata: original sensor text, normalized family, normalized variant, normal-Raman flag conflicts;
- acquisition metadata: serial/version, integration/exposure, averages, laser, tip, suitability, RMX gain/bias/mode and dark path;
- quality: numeric checks, native-axis range, intensity summaries, negative fraction, baseline proxies, field-note flags;
- inclusion: named/ambiguous SERS, strict core, and conservative quality-pass.

Generated under `Workspace/nato_sers_field_trial`:

| Artifact | Purpose |
|---|---|
| `recordings_manifest.csv` | All 1,172 observations with provenance and flags |
| `sers_core_manifest.csv` | 598 strict SERS observations |
| `sers_qc_pass_manifest.csv` | 500 conservative quality-pass observations |
| `sers_core_spectra_raw_common_grid.npz` | 598 × 1,401 canonical vendor-output matrix |
| `sers_core_spectra_preprocessing_candidates.npz` | Vendor, SNV, baseline, corrected, and corrected-SNV arrays |
| `audit_summary.json` | Inclusion, source, sensor, instrument, and target counts |
| `preprocessing_diagnostics.json` | Background, linear baselines, instrument leakage, and domain transfer |
| `grouped_sample_cv_assignments.csv` | Five balanced folds that keep each master sample intact |
| `domain_evaluation_partitions.csv` | Sensor/instrument domain-only and domain+sample partitions |
| `split_summary.json` | Counts, supported classes, and exclusions for every domain protocol |
| `figures/instrument_background_examples.png` | Representative spectra and candidate baselines |
| `figures/dataset_balance_heatmaps.png` | Target × sensor/instrument missing-cell audit |

Rebuild with:

```bash
.venv/bin/python scripts/build_nato_sers_field_trial.py
.venv/bin/python scripts/analyze_nato_sers_preprocessing.py
.venv/bin/python scripts/make_nato_sers_splits.py
```

The scripts read but never modify the source archive.

## 8. Leakage-safe validation

### Ordinary model development

Use `master_sample_id` as the grouping key. All spectra of one physical master sample must remain in the same fold, regardless of instrument, sensor, session, or scan number. The emitted five folds contain 117--124 spectra each and all seven target classes.

A random spectrum split may still be reported only as a diagnostic demonstrating leakage sensitivity. It is not a headline result.

### Domain-transfer protocols

Report two distinct questions:

1. **Domain-only transfer:** hold out one sensor or instrument, but permit other-domain spectra from the same master samples in training. This isolates spectral domain change and supports paired analysis, but it is not unseen-specimen deployment.
2. **Domain-and-sample transfer:** hold out one sensor or instrument and remove every test master sample from training. Score only target classes represented in training, and list unsupported classes explicitly.

The second protocol exposes the archive's limits:

- GaN/polymer: 23 supported test spectra across two classes;
- NRC/Canadian SERS: 78 supported test spectra across six classes; blank becomes unsupported;
- H-Kit: only one supported test spectrum remains; three of four test classes become unsupported;
- p-SERS: no supported test class remains after the same specimens are removed.

Consequently, this dataset cannot support one pooled “unseen substrate accuracy” number. Results must be per held-out domain, restricted to supported classes, with macro recall/balanced accuracy, macro F1, per-class recall, confusion matrices, and uncertainty intervals.

### Stronger future design

To make substrate-agnostic claims identifiable, collect a deliberately crossed extension:

- every analyte and blank on every sensor family;
- multiple independently prepared physical specimens per analyte × sensor cell;
- measurements of each cell on multiple Raman systems;
- randomized acquisition order and balanced sessions/operators;
- shared calibration standards and dark/reference measurements;
- enough independent specimens to hold out both specimen and sensor without eliminating a class.

## 9. Applying `SERS-VAE.md` to this archive

### What the original outline proposes

The outline contains four phases:

1. a standard VAE trained on normalized spectra, evaluated by reconstruction and latent visualization;
2. a two-bucket encoder producing `z_mol` and `z_surf`, with a concatenated decoder and either molecule supervision or β-VAE pressure;
3. a swap test combining the molecule latent from one spectrum with the surface latent from another;
4. a classifier using only `z_mol`, tested on an unseen surface against a raw-spectrum CNN.

It also recommends restricting spectral windows to informative regions and tuning β between weak disentanglement and damaged reconstruction.

### What should remain

- Train a small standard VAE first; it establishes reconstruction and capacity baselines.
- Compare raw-SNV and uniformly corrected-SNV inputs.
- Use a molecule-focused latent for downstream classification.
- Keep the swap test, but only where an actual held-out counterpart exists.
- Compare against a raw 1D CNN and simple chemometric/linear baselines.
- Treat β as a grouped-validation hyperparameter rather than an assumed improvement.

### What must change

`z_mol + z_surf` is too coarse for this archive. “Surface” is entangled with at least four distinct factors:

- SERS sensor family and sensor variant;
- Raman instrument and vendor processing pipeline;
- station/sample matrix and carrier geometry;
- acquisition/session/operator and quality state.

A practical first structured model is:

```text
encoder(x) -> z_chem, z_nuisance
decoder(z_chem, z_nuisance, observed instrument, observed sensor) -> spectrum
```

If 598 spectra support it without overfitting, split `z_nuisance` into `z_sensor` and `z_instrument`. Do not begin with a large multi-branch architecture; the dataset is small.

Useful losses are:

- reconstruction plus KL terms;
- supervised target classification from `z_chem`;
- supervised contrastive or metric loss pulling together spectra with the same master sample/analyte across domains;
- sensor/instrument prediction from nuisance latents;
- carefully weighted adversarial probes discouraging sensor/instrument prediction from `z_chem`.

The adversarial term is dangerous under the current missing-cell design: because target and sensor are correlated, forcing all sensor information out can also erase genuine target information. Apply it only on balanced/crossed subsets or condition it on target, and compare against a no-adversary ablation.

Unsupervised β-VAE independence does not assign semantic meaning to latent dimensions and is not identifiable under this confounding. It can be an ablation, not the main justification for molecule/surface separation.

### Validating the latent representation

Reconstruction MSE alone is insufficient. Report spectral angle/correlation and peak-region error in addition to normalized MSE. Then run explicit leakage probes:

- target prediction from `z_chem` should be high on grouped, supported-class tests;
- instrument and sensor prediction from `z_chem` should fall relative to a standard VAE;
- target prediction from nuisance latents should fall, conditional on what the design makes possible;
- instrument/sensor prediction from their nuisance representation should remain high enough to reconstruct domain style.

The diagnostic linear models already show why this matters: instrument identity remains around 0.86--0.89 grouped balanced accuracy after normalization/correction.

For swap validation, use measured pairs wherever possible: encode analyte/sample A in domain 1, combine its chemical latent with a nuisance/domain code from domain 2, and compare the decoded spectrum with the **actual measurement of sample A in domain 2**. A visually plausible synthetic spectrum in an unobserved target × sensor cell is not proof.

### Recommended model sequence

1. PCA/logistic regression, PLS-DA or SVM, and a small 1D CNN on both preprocessing branches.
2. Small 1D-convolutional standard VAE; latent size roughly 8--32, chosen by grouped validation.
3. Semi-supervised VAE with target classification from `z_chem`.
4. Add same-master/analyte cross-domain contrastive supervision.
5. Add observed sensor/instrument conditioning and nuisance leakage probes.
6. Only then test β, adversarial invariance, latent swaps, or extra nuisance partitions.
7. Repeat all conclusions on core versus quality-pass data and with leave-one-instrument/sensor partitions.

With only 598 core spectra, architecture and hyperparameter searches must be small, nested within the grouped folds, and reported with multiple seeds. Augmentation—small peak shifts, intensity changes, noise, or baseline slopes—must be fitted to training-domain ranges and never substitute for missing target × sensor measurements.

## 10. Decisions and missing information to resolve

Before treating the dataset as final, the field team should answer:

1. Can the 56 PMCDS and one Agilent-2 SERS spectra be recovered?
2. What sensors belong to the three `Normal Raman = N` rows with missing sensor names?
3. Are the 20 named-sensor/`Normal Raman = Y` rows truly SERS measurements?
4. Can the owners resolve the contradictory repeated scan IDs listed in the manifest?
5. What processing did each vendor apply before export, especially to RMX main spectra and Agilent `SORS`?
6. Should the scientific task be one seven-class model across pills, coupons, and CWA, or separate station-specific tasks?
7. Is `blank` a seventh class, or should the deployment problem be detection first and identification second?
8. Is the intended claim transfer to a new Raman instrument, a new SERS sensor, a new specimen on a known sensor, or all three? These are different experiments.

Until those are answered, the 598-row core manifest is the defensible dataset, while the 500-row subset is the defensible quality sensitivity analysis.
