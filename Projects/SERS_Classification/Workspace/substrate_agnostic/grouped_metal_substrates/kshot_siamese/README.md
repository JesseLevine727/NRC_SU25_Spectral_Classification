# K-Shot Substrate-Agnostic Siamese Evaluation

This run tests whether the current substrate-agnostic result is still a formal few-shot result.
`K` is sampled per `chemical_substrate` unit from the held-in substrate families only.
Each fold then tests all known chemical labels in the held-out substrate family.

## Protocol

- Feature: `derivative_1`
- Loss: `triplet`
- Prototype mode: `row_mean`
- Epochs: `100`
- Device requirement: CUDA unless `--allow-cpu` is explicitly passed
- Grouped metal substrates: `True`
- K values: `1,3,5,10,25`
- Seeds: `42,43,44,45,46`

## Summary By K

|      k |   mean_accuracy |   std_accuracy |   mean_balanced_accuracy |   std_balanced_accuracy |   mean_macro_f1 |   mean_true_label_macro_f1 |   std_true_label_macro_f1 |   min_fold_accuracy |   min_fold_true_label_macro_f1 |   mean_support_rows |   n_fold_runs |
|-------:|----------------:|---------------:|-------------------------:|------------------------:|----------------:|---------------------------:|--------------------------:|--------------------:|-------------------------------:|--------------------:|--------------:|
|  1.000 |           0.913 |          0.107 |                    0.913 |                   0.107 |           0.825 |                      0.905 |                     0.140 |               0.653 |                          0.542 |               8.250 |        20.000 |
|  3.000 |           0.840 |          0.239 |                    0.840 |                   0.239 |           0.770 |                      0.819 |                     0.260 |               0.000 |                          0.000 |              24.750 |        20.000 |
|  5.000 |           0.873 |          0.173 |                    0.873 |                   0.173 |           0.801 |                      0.850 |                     0.218 |               0.333 |                          0.167 |              41.250 |        20.000 |
| 10.000 |           0.874 |          0.165 |                    0.874 |                   0.165 |           0.804 |                      0.853 |                     0.200 |               0.560 |                          0.471 |              82.500 |        20.000 |
| 25.000 |           0.930 |          0.114 |                    0.930 |                   0.114 |           0.842 |                      0.924 |                     0.138 |               0.600 |                          0.522 |             206.250 |        20.000 |

## Interpretation

This is a formal few-shot test, but it is not yet a robust few-shot substrate-agnostic result. The model can perform well with very few support spectra, but the variance is large across seeds and held-out substrate families. `K=25` is closest to the full-data setting and reaches the best mean true-label macro F1, but it still underperforms the full-data grouped Siamese run (`0.975` accuracy) and has weak seeded folds.

For poster wording: the project began as few-shot chemical-substrate pair learning and has now moved to substrate-agnostic transfer. The current substrate-agnostic result should be described as Siamese metric learning on a small dataset unless the K-shot protocol is explicitly reported with its variance.

## Fold Means By K And Held-Out Family

|   k | held_out_substrate   |   accuracy |   balanced_accuracy |   macro_f1 |   true_label_macro_f1 |   n_support |
|----:|:---------------------|-----------:|--------------------:|-----------:|----------------------:|------------:|
|   1 | Ag                   |      0.805 |               0.805 |      0.760 |                 0.760 |       8.000 |
|   1 | Au                   |      0.944 |               0.944 |      0.644 |                 0.967 |       9.000 |
|   1 | PICO                 |      0.955 |               0.955 |      0.954 |                 0.954 |       8.000 |
|   1 | pSERS                |      0.949 |               0.949 |      0.941 |                 0.941 |       8.000 |
|   3 | Ag                   |      0.728 |               0.728 |      0.659 |                 0.659 |      24.000 |
|   3 | Au                   |      0.780 |               0.780 |      0.593 |                 0.790 |      27.000 |
|   3 | PICO                 |      0.853 |               0.853 |      0.828 |                 0.828 |      24.000 |
|   3 | pSERS                |      1.000 |               1.000 |      1.000 |                 1.000 |      24.000 |
|   5 | Ag                   |      0.755 |               0.755 |      0.703 |                 0.703 |      40.000 |
|   5 | Au                   |      0.988 |               0.988 |      0.796 |                 0.994 |      45.000 |
|   5 | PICO                 |      0.885 |               0.885 |      0.874 |                 0.874 |      40.000 |
|   5 | pSERS                |      0.864 |               0.864 |      0.831 |                 0.831 |      40.000 |
|  10 | Ag                   |      0.661 |               0.661 |      0.586 |                 0.586 |      80.000 |
|  10 | Au                   |      0.988 |               0.988 |      0.796 |                 0.994 |      90.000 |
|  10 | PICO                 |      0.851 |               0.851 |      0.836 |                 0.836 |      80.000 |
|  10 | pSERS                |      0.997 |               0.997 |      0.997 |                 0.997 |      80.000 |
|  25 | Ag                   |      0.787 |               0.787 |      0.754 |                 0.754 |     200.000 |
|  25 | Au                   |      0.980 |               0.980 |      0.660 |                 0.990 |     225.000 |
|  25 | PICO                 |      0.955 |               0.955 |      0.953 |                 0.953 |     200.000 |
|  25 | pSERS                |      1.000 |               1.000 |      1.000 |                 1.000 |     200.000 |

## Files

- `detailed_results.csv`
- `summary_by_k.csv`
- `confusions/k*/seed*/*.csv`
- `diagnostics/geometry_k*/geometry_analysis.md`
- `diagnostics/geometry_k*/projections/*.png`
- `diagnostics/geometry_k*/silhouette_*.png`
