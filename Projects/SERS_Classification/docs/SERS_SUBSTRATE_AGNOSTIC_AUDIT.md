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

After canonicalizing `bt -> benzenethiol`, the current three-chemical substrate-agnostic matrix is:

| Chemical | Ag | AgNP | Au | AuNP | PICO | pSERS | Current total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `4np` | 0 | 25 | 0 | 0 | 25 | 25 | 75 |
| `benzenethiol` | 25 | 0 | 25 | 0 | 25 | 25 | 100 |
| `pyridine` | 0 | 25 | 0 | 25 | 25 | 25 | 100 |

The minimum target matrix should fill every missing chemical-substrate pair before adding many more spectra to already-covered cells:

| Chemical | Ag | AgNP | Au | AuNP | PICO | pSERS | Target total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `4np` | 25+ | 25+ | 25+ | 25+ | 25+ | 25+ | 150+ |
| `benzenethiol` | 25+ | 25+ | 25+ | 25+ | 25+ | 25+ | 150+ |
| `pyridine` | 25+ | 25+ | 25+ | 25+ | 25+ | 25+ | 150+ |

Preferred collection target, if time allows: `2-3` independent preparations/maps per chemical-substrate pair with `25` spectra per preparation. This is more useful for substrate-agnostic detection than collecting many additional correlated spectra from a single existing map.

## Current Substrate-Agnostic Baselines

Run:

```bash
./.venv/bin/python scripts/sers_substrate_agnostic_detection.py
./.venv/bin/python scripts/sers_siamese_substrate_agnostic.py --feature derivative_1 --loss triplet --prototype-mode substrate_balanced --margin 0.2
./.venv/bin/python scripts/sers_siamese_substrate_agnostic.py --feature derivative_1 --loss batch_hard_triplet --out Workspace/substrate_agnostic/archive/comparison_runs/batch_hard_derivative1_results.csv
./.venv/bin/python scripts/run_siamese_sers_sweep.py --epochs 100 --seeds 42
```

Both scripts evaluate chemical-label prediction with leave-one-substrate-out folds. Substrate is used only as the held-out group, not as a target. The Siamese script keeps the original notebook architecture but uses chemical `Label` for contrastive pairs instead of `Label__Substrate`.

The report identifies an instrumental artifact around `300 cm^-1`; these scripts default to retaining only spectra from `330 cm^-1` upward.

Current best canonical triplet-Siamese result:

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

The remaining dominant failure is `AgNP`: canonical `4np` on `AgNP` is predicted as `benzenethiol` for all 25 spectra, and 6/25 `pyridine` on `AgNP` spectra are also predicted as `benzenethiol`.

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

Current best classical comparison:

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

## AgNP Failure Deep Dive

The current bottleneck is held-out `AgNP`, not held-out `pSERS`. After canonicalizing `bt -> benzenethiol`, `pSERS` reaches 75/75 correct spectra in the best Siamese run.

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

This means the failure is not simply "not enough data" in a generic sense. The specific issue is that the learned embedding does not have enough substrate coverage to preserve `4np` identity when AgNP is absent from training for that chemical. The derivative representation still contains useful chemical structure, but the Siamese embedding collapses AgNP `4np` toward `benzenethiol`.

## Practical Next Steps

- Treat old 92.8-98.8% Siamese accuracy as pair-classification accuracy, not substrate-agnostic performance.
- Use `scripts/sers_substrate_agnostic_detection.py` as the baseline gate for future work.
- Add more chemicals measured on more substrates, especially missing chemical-substrate combinations and non-singleton chemicals like `n,n-dimethylformamide`.
- Optimize on leave-one-substrate-out validation only; random row splits are not adequate for this goal.
- Try hybrid inference or an auxiliary geometry-preserving loss before increasing model capacity, because the AgNP failure is introduced by the learned embedding rather than by the first-derivative input representation.
