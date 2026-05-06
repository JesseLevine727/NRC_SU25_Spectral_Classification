# SERS Substrate-Agnostic Audit

## Conclusion

The prior SERS classification work did not establish substrate-agnostic chemical detection. The strongest saved notebook results classify chemical-substrate pairs, not chemicals independent of substrate.

Evidence:

- `Workspace/Siamese_Network_OneShot*.ipynb` creates `Class = Label + "__" + Substrate`.
- The saved Siamese reports score classes such as `4np__AgNP`, `pyridine__PICO`, and `bt__pSERS`.
- `Workspace/PCA_Centroid_Correlation_Notebook.ipynb` also constructs the same chemical-substrate `Class` for SERS PCA analysis.
- `Workspace/data/processed/consolidated_SERS.csv` has separate `Label` and `Substrate` columns, but the previous supervised Siamese target used their concatenation.

## Label Canonicalization

`bt` and `benzenethiol` refer to the same chemical identity in this dataset. Current Siamese runs canonicalize:

| Original label | Canonical label |
|---|---|
| `bt` | `benzenethiol` |

The source CSV is not modified; canonicalization happens in the training/evaluation pipeline.

## Data Coverage

`Workspace/data/processed/consolidated_SERS.csv` contains 503 SERS spectra over 5 chemical labels and 6 substrate labels.

| Chemical | Ag | AgNP | Au | AuNP | PICO | pSERS |
|---|---:|---:|---:|---:|---:|---:|
| 4np | 0 | 25 | 0 | 0 | 25 | 25 |
| benzenethiol | 25 | 0 | 25 | 0 | 0 | 0 |
| bt | 0 | 0 | 0 | 0 | 25 | 25 |
| n,n-dimethylformamide | 0 | 0 | 0 | 228 | 0 | 0 |
| pyridine | 0 | 25 | 0 | 25 | 25 | 25 |

Implications:

- `n,n-dimethylformamide` cannot be evaluated substrate-agnostically with this dataset because it appears on only `AuNP`.
- Valid substrate-held-out evaluation is possible for `4np`, `benzenethiol`, `bt`, and `pyridine`.
- Coverage is sparse and imbalanced, so substrate-held-out metrics are more meaningful than random train/test splits.
- The six labels above are raw dataset labels. The corrected scientific interpretation groups `AgNP` with `Ag` and `AuNP` with `Au`.

After canonicalizing `bt -> benzenethiol`, the old six-substrate-label matrix was:

| Chemical | Ag | AgNP | Au | AuNP | PICO | pSERS | Current total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `4np` | 0 | 25 | 0 | 0 | 25 | 25 | 75 |
| `benzenethiol` | 25 | 0 | 25 | 0 | 25 | 25 | 100 |
| `pyridine` | 0 | 25 | 0 | 25 | 25 | 25 | 100 |

That matrix is retained for traceability, but it should not be the primary scientific matrix if `Ag/AgNP` and `Au/AuNP` are substrate-family duplicates.

The corrected grouped-substrate-family matrix is:

| Chemical | Ag | Au | PICO | pSERS | Current total | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| `4np` | 25 | 0 | 25 | 25 | 75 | `4np` does not respond on Au, so Au should be treated as N/A rather than a missing target cell. |
| `benzenethiol` | 25 | 25 | 25 | 25 | 100 | Complete across four substrate families. |
| `pyridine` | 25 | 25 | 25 | 25 | 100 | Complete across four substrate families. |
| `n,n-dimethylformamide` | 0 | 228 | 0 | 0 | 228 | Present only on Au; useful only if expanded to additional substrate families. |

The corrected minimum target matrix is:

| Chemical | Ag | Au | PICO | pSERS | Target total | Priority |
|---|---:|---:|---:|---:|---:|---|
| `4np` | 25+ | N/A | 25+ | 25+ | 75+ | Already covers its valid responding substrate families. |
| `benzenethiol` | 25+ | 25+ | 25+ | 25+ | 100+ | Already complete; prioritize independent repeats only if time allows. |
| `pyridine` | 25+ | 25+ | 25+ | 25+ | 100+ | Already complete; prioritize independent repeats only if time allows. |
| additional chemical, e.g. `n,n-dimethylformamide` if responsive | 25+ | 25+ | 25+ | 25+ | 100+ | Highest-value expansion for a stronger claim. |

The old ungrouped target below is superseded by the corrected grouped matrix:

| Chemical | Ag | AgNP | Au | AuNP | PICO | pSERS | Target total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `4np` | 25+ | 25+ | 25+ | 25+ | 25+ | 25+ | 150+ |
| `benzenethiol` | 25+ | 25+ | 25+ | 25+ | 25+ | 25+ | 150+ |
| `pyridine` | 25+ | 25+ | 25+ | 25+ | 25+ | 25+ | 150+ |

Preferred collection target, if time allows: `2-3` independent preparations/maps per chemical-substrate pair with `25` spectra per preparation. This is more useful for substrate-agnostic detection than collecting many additional correlated spectra from a single existing map.

## Corrected Grouped-Substrate Results

Run:

```bash
./.venv/bin/python scripts/sers_substrate_agnostic_detection.py --group-metal-substrates --out Workspace/substrate_agnostic/grouped_metal_substrates/classical_baselines/results.csv --confusions-dir Workspace/substrate_agnostic/grouped_metal_substrates/classical_baselines/confusions
./.venv/bin/python scripts/run_siamese_sers_sweep.py --group-metal-substrates --epochs 100 --seeds 42 --out-dir Workspace/substrate_agnostic/grouped_metal_substrates/sweeps/siamese_feature_loss_sweep
./.venv/bin/python scripts/analyze_sers_geometry.py --group-metal-substrates --prototype-mode row_mean --out-dir Workspace/substrate_agnostic/grouped_metal_substrates/diagnostics/geometry_analysis
```

The corrected grouped folds are `Ag`, `Au`, `PICO`, and `pSERS`. `Ag` includes the original `Ag` and `AgNP` rows. `Au` includes the original `Au` and `AuNP` rows.

Best grouped Siamese result selected from the grouped sweep:

- Feature/model: `derivative_1` + Siamese Conv1D encoder + nearest chemical prototype
- Loss: triplet loss with margin `0.2`
- Prototype mode: `row_mean`
- Device: CUDA/GPU
- Canonical labels: `bt -> benzenethiol`
- Grouped substrates: `AgNP -> Ag`, `AuNP -> Au`
- Mean held-out-substrate-family accuracy: `0.975`
- Mean held-out-substrate-family balanced accuracy: `0.975`
- Mean held-out-substrate-family macro F1: `0.895`

Per-fold accuracy for the corrected grouped best Siamese run:

| Held-out substrate family | Test labels | Accuracy |
|---|---|---:|
| Ag | `4np,benzenethiol,pyridine` | 0.920 |
| Au | `benzenethiol,pyridine` | 0.980 |
| PICO | `4np,benzenethiol,pyridine` | 1.000 |
| pSERS | `4np,benzenethiol,pyridine` | 1.000 |

Grouped best Siamese confusion summary:

| Held-out substrate family | Main errors |
|---|---|
| Ag | 6/25 `4np` predicted as `benzenethiol`; `benzenethiol` and `pyridine` are 25/25 correct. |
| Au | 1/25 `benzenethiol` predicted as `4np`; `pyridine` is 25/25 correct. |
| PICO | 75/75 correct. |
| pSERS | 75/75 correct. |

Best grouped classical baseline:

- Feature/model: `derivative_2` + nearest centroid
- Mean held-out-substrate-family accuracy: `0.987`
- Mean held-out-substrate-family balanced accuracy: `0.987`
- Mean held-out-substrate-family macro F1: `0.987`

Grouped raw-spectrum Siamese comparison:

- Feature/model: raw cropped spectra + Siamese Conv1D encoder
- Loss/prototype mode: triplet loss + `row_mean`
- Mean held-out-substrate-family accuracy: `0.440`
- Mean held-out-substrate-family balanced accuracy: `0.440`
- Mean held-out-substrate-family macro F1: `0.399`

This confirms that the derivative preprocessing remains important even after substrate regrouping.

Grouped geometry analysis:

| Space | Mean chemical-label silhouette | Mean substrate-family silhouette | Label minus substrate |
|---|---:|---:|---:|
| derivative input | 0.304 | 0.101 | 0.203 |
| Siamese embedding | 0.797 | -0.040 | 0.837 |

The corrected interpretation is that the model is generally learning chemical organization across substrate families. The remaining weakness is no longer a distinct `AgNP` substrate failure. It is a smaller `4np`-on-silver-family ambiguity against `benzenethiol`.

## Historical Six-Substrate-Label Baselines

Run:

```bash
./.venv/bin/python scripts/sers_substrate_agnostic_detection.py
./.venv/bin/python scripts/sers_siamese_substrate_agnostic.py --feature derivative_1 --loss triplet --prototype-mode substrate_balanced --margin 0.2
./.venv/bin/python scripts/sers_siamese_substrate_agnostic.py --feature derivative_1 --loss batch_hard_triplet --out Workspace/substrate_agnostic/archive/comparison_runs/batch_hard_derivative1_results.csv
./.venv/bin/python scripts/run_siamese_sers_sweep.py --epochs 100 --seeds 42
```

Both scripts evaluate chemical-label prediction with leave-one-substrate-out folds. Substrate is used only as the held-out group, not as a target. The Siamese script keeps the original notebook architecture but uses chemical `Label` for contrastive pairs instead of `Label__Substrate`.

The report identifies an instrumental artifact around `300 cm^-1`; these scripts default to retaining only spectra from `330 cm^-1` upward.

Historical best ungrouped canonical triplet-Siamese result:

- Feature/model: `derivative_1` + Siamese Conv1D encoder + nearest chemical prototype
- Loss: substrate-aware triplet loss with margin `0.2`
- Prototype mode: `substrate_balanced`
- Device: CUDA/GPU required by default
- Canonical labels: `bt -> benzenethiol`
- Mean held-out-substrate accuracy: `0.854`
- Mean held-out-substrate balanced accuracy: `0.854`
- Mean held-out-substrate macro F1: `0.686`

Per-fold accuracy for this best run:

| Held-out substrate | Accuracy |
|---|---:|
| Ag | 0.920 |
| AgNP | 0.380 |
| Au | 0.960 |
| AuNP | 1.000 |
| PICO | 0.867 |
| pSERS | 1.000 |

The dominant ungrouped failure was `AgNP`: canonical `4np` on `AgNP` was predicted as `benzenethiol` for all 25 spectra, and 6/25 `pyridine` on `AgNP` spectra were also predicted as `benzenethiol`. This is now considered a superseded diagnostic because `Ag` and `AgNP` should be evaluated as one silver substrate family.

Best contrastive-Siamese comparison:

- Feature/model: `derivative_1` + Siamese Conv1D encoder + nearest chemical prototype
- Loss: contrastive loss
- Mean held-out-substrate accuracy: `0.806`
- Mean held-out-substrate balanced accuracy: `0.806`
- Mean held-out-substrate macro F1: `0.568`

Batch-hard triplet comparison:

- Feature/model: `derivative_1` + balanced-batch hard triplet mining
- Loss: hardest same-label positive and hardest different-label negative mined within each batch
- Mean held-out-substrate accuracy: `0.768`
- Mean held-out-substrate balanced accuracy: `0.768`
- Mean held-out-substrate macro F1: `0.480`

Historical best ungrouped classical comparison:

- Feature/model: `peak_emphasis` + cosine kNN
- Mean held-out-substrate accuracy: `0.761`
- Mean held-out-substrate balanced accuracy: `0.761`
- Mean held-out-substrate macro F1: `0.535`

Cleaned output layout:

- `Workspace/substrate_agnostic/current/best_siamese_triplet/results.csv`
- `Workspace/substrate_agnostic/current/best_siamese_triplet/confusions/*.csv`
- `Workspace/substrate_agnostic/diagnostics/agnp_failure/*.csv`
- `Workspace/substrate_agnostic/diagnostics/agnp_failure/*.png`
- `Workspace/substrate_agnostic/diagnostics/geometry_analysis/*.csv`
- `Workspace/substrate_agnostic/diagnostics/geometry_analysis/geometry_analysis.md`
- `Workspace/substrate_agnostic/sweeps/siamese_feature_loss_sweep/summary.csv`
- `Workspace/substrate_agnostic/classical_baselines/results.csv`
- `Workspace/substrate_agnostic/classical_baselines/confusions/*.csv`
- `Workspace/substrate_agnostic/archive/comparison_runs/`

## Historical AgNP Failure Deep Dive

The old six-substrate-label bottleneck was held-out `AgNP`, not held-out `pSERS`. After canonicalizing `bt -> benzenethiol`, `pSERS` reached 75/75 correct spectra in the best ungrouped Siamese run.

The AgNP confusion is concentrated in `4np`:

| True | Pred 4np | Pred benzenethiol | Pred pyridine |
|---|---:|---:|---:|
| 4np | 0 | 25 | 0 |
| pyridine | 0 | 3 | 22 |

The deep-dive diagnostics in `Workspace/substrate_agnostic/diagnostics/agnp_failure/` show:

- The raw `4np` AgNP files are present: 25 map spectra, each 1024 rows by 2 columns.
- Average spectra and the raw-file audit show the AgNP `4np` map is heterogeneous, with many spectra peaking near `1094-1096 cm^-1` and others near `1570-1585 cm^-1`.
- In first-derivative input PCA, AgNP `4np` remains closer to `4np` than to `benzenethiol`.
- In the trained AgNP-held-out Siamese embedding, AgNP `4np` moves closer to the `benzenethiol` prototype than the `4np` prototype.
- UMAP and t-SNE diagnostics were added for the derivative input space and the Siamese embedding spaces. They are useful qualitative figures, but the primary evidence remains the leave-one-substrate-out confusion matrices and prototype distances.
- All-class geometry analysis shows the derivative input has no negative class-level prototype margins, while the Siamese embedding has exactly one negative margin: held-out `AgNP` `4np` collapsing toward `benzenethiol`. PCA, UMAP, and t-SNE projection tables were generated for every held-out substrate and show the same unique negative embedding case.
- Silhouette visualizations in `Workspace/substrate_agnostic/diagnostics/geometry_analysis/` show the embedding increases mean chemical-label silhouette from `0.304` to `0.791` while reducing substrate silhouette from `0.054` to `-0.246`.

This remains useful as a historical diagnostic, but it is no longer the primary scientific interpretation after substrate-family correction. In the corrected grouped analysis, `AgNP` is not a separate held-out substrate. The remaining weakness is smaller: held-out silver-family `4np` has some ambiguity against `benzenethiol`, while the overall grouped Siamese accuracy rises to `0.975`.

## Practical Next Steps

- Treat old 92.8-98.8% Siamese accuracy as pair-classification accuracy, not substrate-agnostic performance.
- Use `scripts/sers_substrate_agnostic_detection.py --group-metal-substrates` as the baseline gate for future work.
- Add one or more additional chemicals measured on all four substrate families. `n,n-dimethylformamide` is a candidate only if it responds reliably beyond Au.
- Optimize on leave-one-substrate-out validation only; random row splits are not adequate for this goal.
- Prefer independent preparations/maps over many additional spectra from the same existing map.
