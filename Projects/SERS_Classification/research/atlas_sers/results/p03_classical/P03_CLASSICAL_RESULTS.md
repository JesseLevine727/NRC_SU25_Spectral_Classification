# NATO SERS P03 classical benchmark report

**Public aggregate release approved by the project owner on 2026-09-02.**

This report contains aggregate performance, support, failure, calibration, and
cost summaries. It intentionally excludes row-level predictions, fold
assignments, model binaries, fit caches, and the full terminal ledger.

## Frozen identity

- Execution run: `P03-513a0f9686c37cbc0d682645`
- No-fit plan: `P03PLAN-e5d8a0af054928b455b93238`
- Protected state: `513a0f9686c37cbc0d682645d62c9ef1e5268f6853a792cf66f18076ae47c501`
- Fit-manifest rows: 260,356
- Terminal fit-ledger rows: 260,356
- Primary population/policy: 598 spectra, 69 physical masters, `PP-U-MIN`.
- Primary representation: `R_MIN_400_1800` (400–1,800 cm⁻¹, row min–max).

## Terminal accounting

| terminal_status | fit_records |
| --- | --- |
| complete | 219664 |
| convergence_failure | 2860 |
| excluded_by_protocol | 17645 |
| rank_failure | 1578 |
| unsupported_candidate | 18609 |

Every planned fit ID has exactly one terminal status. Unsupported, failed, inactive, and dependency-excluded records remain in the ledger and endpoint denominator.

## Endpoint coverage

| experiment_id | aggregation_level | planned_endpoint_count | complete_endpoint_count | unavailable_endpoint_count | completion_fraction |
| --- | --- | --- | --- | --- | --- |
| EXP-C00-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C00-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C00-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C00-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C00-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C00-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C00-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C00-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C00-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C00-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C00-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C00-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C01-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C01-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C01-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C01-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C01-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C01-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C02-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C02-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C02-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C02-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C02-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C02-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C03-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C03-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C03-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C03-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C03-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C03-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C04-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C04-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C04-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C04-T1 | spectrum | 5 | 5 | 0 | 1.0000 |
| EXP-C04-T1 | instrument_balanced_master | 5 | 5 | 0 | 1.0000 |
| EXP-C04-T1 | spectrum | 5 | 5 | 0 | 1.0000 |

_Table truncated to 36 of 238 rows._

## Within-station classical results

| station | procedure_id | aggregation_level | mean_balanced_accuracy | minimum_repeat_balanced_accuracy | maximum_repeat_balanced_accuracy | repeat_count |
| --- | --- | --- | --- | --- | --- | --- |
| cwa | C-EXTRA-TREES | instrument_balanced_master | 0.8575 | 0.7890 | 0.9048 | 5 |
| cwa | C-EXTRA-TREES | spectrum | 0.6183 | 0.5881 | 0.6343 | 5 |
| cwa | C-LOGREG-EN | instrument_balanced_master | 0.8726 | 0.8155 | 0.9107 | 5 |
| cwa | C-LOGREG-EN | spectrum | 0.6516 | 0.6288 | 0.6739 | 5 |
| cwa | C-NEAREST-CENTROID | instrument_balanced_master | 0.7075 | 0.6462 | 0.7679 | 5 |
| cwa | C-NEAREST-CENTROID | spectrum | 0.5781 | 0.5377 | 0.6160 | 5 |
| cwa | C-PCA-LDA | instrument_balanced_master | 0.8525 | 0.8261 | 0.9107 | 5 |
| cwa | C-PCA-LDA | spectrum | 0.6617 | 0.6420 | 0.6794 | 5 |
| cwa | C-PLS-DA | instrument_balanced_master | 0.7765 | 0.6567 | 0.8366 | 5 |
| cwa | C-PLS-DA | spectrum | 0.6267 | 0.5964 | 0.6552 | 5 |
| cwa | C-PRIOR:C-PRIOR-000 | instrument_balanced_master | 0.3333 | 0.3333 | 0.3333 | 5 |
| cwa | C-PRIOR:C-PRIOR-000 | spectrum | 0.3333 | 0.3333 | 0.3333 | 5 |
| cwa | C-PRIOR:C-PRIOR-001 | instrument_balanced_master | 0.3333 | 0.3333 | 0.3333 | 5 |
| cwa | C-PRIOR:C-PRIOR-001 | spectrum | 0.3333 | 0.3333 | 0.3333 | 5 |
| cwa | C-RANDOM-FOREST | instrument_balanced_master | 0.8631 | 0.8631 | 0.8631 | 5 |
| cwa | C-RANDOM-FOREST | spectrum | 0.6236 | 0.5971 | 0.6408 | 5 |
| cwa | C-RBF-SVM | instrument_balanced_master | 0.8126 | 0.7738 | 0.8737 | 5 |
| cwa | C-RBF-SVM | spectrum | 0.6138 | 0.6051 | 0.6183 | 5 |
| cwa | C-SPECTRAL-MATCH | instrument_balanced_master | 0.8988 | 0.8571 | 0.9524 | 5 |
| cwa | C-SPECTRAL-MATCH | spectrum | 0.5803 | 0.5398 | 0.6090 | 5 |
| pills | C-EXTRA-TREES | instrument_balanced_master | 1.0000 | 1.0000 | 1.0000 | 5 |
| pills | C-EXTRA-TREES | spectrum | 0.8217 | 0.8030 | 0.8349 | 5 |
| pills | C-LOGREG-EN | instrument_balanced_master | 1.0000 | 1.0000 | 1.0000 | 5 |
| pills | C-LOGREG-EN | spectrum | 0.8113 | 0.7996 | 0.8156 | 5 |
| pills | C-NEAREST-CENTROID | instrument_balanced_master | 0.9905 | 0.9524 | 1.0000 | 5 |
| pills | C-NEAREST-CENTROID | spectrum | 0.6837 | 0.6542 | 0.7377 | 5 |
| pills | C-PCA-LDA | instrument_balanced_master | 1.0000 | 1.0000 | 1.0000 | 5 |
| pills | C-PCA-LDA | spectrum | 0.7712 | 0.7584 | 0.7809 | 5 |
| pills | C-PLS-DA | instrument_balanced_master | 1.0000 | 1.0000 | 1.0000 | 5 |
| pills | C-PLS-DA | spectrum | 0.6889 | 0.6818 | 0.6976 | 5 |
| pills | C-PRIOR:C-PRIOR-000 | instrument_balanced_master | 0.3262 | 0.2976 | 0.3333 | 5 |
| pills | C-PRIOR:C-PRIOR-000 | spectrum | 0.3221 | 0.2770 | 0.3333 | 5 |
| pills | C-PRIOR:C-PRIOR-001 | instrument_balanced_master | 0.3333 | 0.3333 | 0.3333 | 5 |
| pills | C-PRIOR:C-PRIOR-001 | spectrum | 0.3333 | 0.3333 | 0.3333 | 5 |
| pills | C-RANDOM-FOREST | instrument_balanced_master | 1.0000 | 1.0000 | 1.0000 | 5 |
| pills | C-RANDOM-FOREST | spectrum | 0.8266 | 0.8139 | 0.8471 | 5 |
| pills | C-RBF-SVM | instrument_balanced_master | 1.0000 | 1.0000 | 1.0000 | 5 |
| pills | C-RBF-SVM | spectrum | 0.7497 | 0.7406 | 0.7665 | 5 |
| pills | C-SPECTRAL-MATCH | instrument_balanced_master | 0.8495 | 0.7429 | 0.9048 | 5 |
| pills | C-SPECTRAL-MATCH | spectrum | 0.6481 | 0.6176 | 0.6781 | 5 |
| surfaces | C-EXTRA-TREES | instrument_balanced_master | 0.9697 | 0.9697 | 0.9697 | 5 |
| surfaces | C-EXTRA-TREES | spectrum | 0.7316 | 0.7063 | 0.7597 | 5 |
| surfaces | C-LOGREG-EN | instrument_balanced_master | 0.9636 | 0.9394 | 0.9697 | 5 |
| surfaces | C-LOGREG-EN | spectrum | 0.7736 | 0.7472 | 0.7894 | 5 |
| surfaces | C-NEAREST-CENTROID | instrument_balanced_master | 0.7055 | 0.6227 | 0.7697 | 5 |
| surfaces | C-NEAREST-CENTROID | spectrum | 0.5947 | 0.5850 | 0.6156 | 5 |
| surfaces | C-PCA-LDA | instrument_balanced_master | 0.9697 | 0.9697 | 0.9697 | 5 |
| surfaces | C-PCA-LDA | spectrum | 0.8018 | 0.7812 | 0.8246 | 5 |
| surfaces | C-PLS-DA | instrument_balanced_master | 0.9515 | 0.9091 | 0.9697 | 5 |
| surfaces | C-PLS-DA | spectrum | 0.8023 | 0.7748 | 0.8375 | 5 |
| surfaces | C-PRIOR:C-PRIOR-000 | instrument_balanced_master | 0.3158 | 0.2848 | 0.3424 | 5 |
| surfaces | C-PRIOR:C-PRIOR-000 | spectrum | 0.2625 | 0.1592 | 0.3333 | 5 |
| surfaces | C-PRIOR:C-PRIOR-001 | instrument_balanced_master | 0.3333 | 0.3333 | 0.3333 | 5 |
| surfaces | C-PRIOR:C-PRIOR-001 | spectrum | 0.3333 | 0.3333 | 0.3333 | 5 |
| surfaces | C-RANDOM-FOREST | instrument_balanced_master | 0.9697 | 0.9697 | 0.9697 | 5 |
| surfaces | C-RANDOM-FOREST | spectrum | 0.7504 | 0.7408 | 0.7678 | 5 |
| surfaces | C-RBF-SVM | instrument_balanced_master | 0.9564 | 0.9364 | 0.9697 | 5 |
| surfaces | C-RBF-SVM | spectrum | 0.7357 | 0.7209 | 0.7561 | 5 |
| surfaces | C-SPECTRAL-MATCH | instrument_balanced_master | 0.8012 | 0.7758 | 0.8394 | 5 |
| surfaces | C-SPECTRAL-MATCH | spectrum | 0.6301 | 0.6006 | 0.6681 | 5 |

Values summarize five technical split repeats; ranges are not confidence intervals.

## Primary unseen-instrument classical comparator

| procedure_id | aggregation_level | mean_domain_balanced_accuracy | minimum_repeat_mean | maximum_repeat_mean | worst_observed_domain | complete_repeat_count |
| --- | --- | --- | --- | --- | --- | --- |
| C-SELECTED | instrument_balanced_master | 0.7088 | 0.6926 | 0.7296 | 0.3333 | 0 |
| C-SELECTED | spectrum | 0.6402 | 0.6230 | 0.6592 | 0.3333 | 0 |

Domain means are unweighted across the 13 eligible domains. The worst-domain value is descriptive. P03 alone makes no classical-versus-deep claim.

## Source-only candidate selection

| station | selection_outcome_model | selection_count | selection_denominator | selection_fraction |
| --- | --- | --- | --- | --- |
| cwa | C-EXTRA-TREES | 1 | 60 | 0.0167 |
| cwa | C-EXTRA-TREES | 1 | 60 | 0.0167 |
| cwa | C-LOGREG-EN | 3 | 60 | 0.0500 |
| cwa | C-LOGREG-EN | 3 | 60 | 0.0500 |
| cwa | C-LOGREG-EN | 1 | 60 | 0.0167 |
| cwa | C-LOGREG-EN | 1 | 60 | 0.0167 |
| cwa | C-LOGREG-EN | 3 | 60 | 0.0500 |
| cwa | C-LOGREG-EN | 2 | 60 | 0.0333 |
| cwa | C-LOGREG-EN | 2 | 60 | 0.0333 |
| cwa | C-NEAREST-CENTROID | 3 | 60 | 0.0500 |
| cwa | C-NEAREST-CENTROID | 3 | 60 | 0.0500 |
| cwa | C-PCA-LDA | 1 | 60 | 0.0167 |
| cwa | C-PCA-LDA | 3 | 60 | 0.0500 |
| cwa | C-PCA-LDA | 4 | 60 | 0.0667 |
| cwa | C-PCA-LDA | 1 | 60 | 0.0167 |
| cwa | C-PCA-LDA | 2 | 60 | 0.0333 |
| cwa | C-PCA-LDA | 1 | 60 | 0.0167 |
| cwa | C-PLS-DA | 1 | 60 | 0.0167 |
| cwa | C-PLS-DA | 1 | 60 | 0.0167 |
| cwa | C-PLS-DA | 3 | 60 | 0.0500 |
| cwa | C-RANDOM-FOREST | 1 | 60 | 0.0167 |
| cwa | C-RANDOM-FOREST | 1 | 60 | 0.0167 |
| cwa | C-RANDOM-FOREST | 1 | 60 | 0.0167 |
| cwa | C-RBF-SVM | 1 | 60 | 0.0167 |
| cwa | C-RBF-SVM | 1 | 60 | 0.0167 |
| cwa | C-RBF-SVM | 2 | 60 | 0.0333 |
| cwa | C-SPECTRAL-MATCH | 5 | 60 | 0.0833 |
| cwa | C-SPECTRAL-MATCH | 4 | 60 | 0.0667 |
| cwa | C-SPECTRAL-MATCH | 4 | 60 | 0.0667 |
| pills | C-LOGREG-EN | 4 | 100 | 0.0400 |
| pills | C-NEAREST-CENTROID | 4 | 100 | 0.0400 |
| pills | C-PLS-DA | 65 | 100 | 0.6500 |
| pills | C-SPECTRAL-MATCH | 27 | 100 | 0.2700 |
| surfaces | C-EXTRA-TREES | 1 | 100 | 0.0100 |
| surfaces | C-EXTRA-TREES | 1 | 100 | 0.0100 |
| surfaces | C-LOGREG-EN | 7 | 100 | 0.0700 |
| surfaces | C-LOGREG-EN | 1 | 100 | 0.0100 |
| surfaces | C-LOGREG-EN | 1 | 100 | 0.0100 |
| surfaces | C-LOGREG-EN | 2 | 100 | 0.0200 |
| surfaces | C-LOGREG-EN | 5 | 100 | 0.0500 |

_Table truncated to 40 of 70 rows._

| station | median_modal_fraction | minimum_modal_fraction | median_normalized_entropy | decision_count |
| --- | --- | --- | --- | --- |
| cwa | 0.2000 | 0.2000 | 1.0000 | 12 |
| pills | 0.8000 | 0.4000 | 0.7219 | 20 |
| surfaces | 0.4000 | 0.2000 | 0.9610 | 20 |

Selection used only P02 source pseudo-domains or the exact source-master-CV fallback. No held-target spectrum, statistic, label, QC summary, or outcome selected a candidate.

## Negative and confounding controls

| experiment_id | control_kind | procedure_id | aggregation_level | mean_domain_balanced_accuracy | minimum_repeat_balanced_accuracy | maximum_repeat_balanced_accuracy | worst_domain_balanced_accuracy | repeat_count | minimum_complete_domain_count | expected_domain_count | summary_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-C09-CONTROL-META | acquisition metadata only | C-METADATA-LOGREG | instrument_balanced_master | 0.3160 | 0.3039 | 0.3272 | 0.1667 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-META | acquisition metadata only | C-METADATA-LOGREG | spectrum | 0.3224 | 0.2996 | 0.3390 | 0.1734 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-00 | instrument_balanced_master | 0.3807 | 0.3361 | 0.4028 | 0.1667 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-00 | spectrum | 0.3659 | 0.3317 | 0.3883 | 0.1212 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-01 | instrument_balanced_master | 0.3822 | 0.3513 | 0.4245 | 0.1806 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-01 | spectrum | 0.3811 | 0.3533 | 0.4370 | 0.2169 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-02 | instrument_balanced_master | 0.3100 | 0.2801 | 0.3499 | 0.0833 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-02 | spectrum | 0.3113 | 0.2865 | 0.3588 | 0.1204 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-03 | instrument_balanced_master | 0.3256 | 0.2782 | 0.3584 | 0.1250 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-03 | spectrum | 0.3251 | 0.2821 | 0.3641 | 0.1204 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-04 | instrument_balanced_master | 0.3283 | 0.2780 | 0.3559 | 0.1587 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-04 | spectrum | 0.3299 | 0.2794 | 0.3544 | 0.1042 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-05 | instrument_balanced_master | 0.3250 | 0.2540 | 0.3784 | 0.0833 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-05 | spectrum | 0.3315 | 0.2924 | 0.3605 | 0.1515 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-06 | instrument_balanced_master | 0.3187 | 0.2588 | 0.3492 | 0.1136 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-06 | spectrum | 0.3117 | 0.2496 | 0.3690 | 0.1083 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-07 | instrument_balanced_master | 0.3487 | 0.3023 | 0.3961 | 0.1310 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-07 | spectrum | 0.3430 | 0.2966 | 0.3931 | 0.1217 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-08 | instrument_balanced_master | 0.3641 | 0.3152 | 0.4152 | 0.1389 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-08 | spectrum | 0.3496 | 0.3029 | 0.3956 | 0.1879 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-09 | instrument_balanced_master | 0.3108 | 0.2375 | 0.3450 | 0.0000 | 5 | 12 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-09 | spectrum | 0.3222 | 0.2714 | 0.3422 | 0.1515 | 5 | 12 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-10 | instrument_balanced_master | 0.3701 | 0.3261 | 0.4128 | 0.1667 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-10 | spectrum | 0.3753 | 0.3445 | 0.4070 | 0.1571 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-11 | instrument_balanced_master | 0.3295 | 0.2756 | 0.3754 | 0.0952 | 5 | 12 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-11 | spectrum | 0.3369 | 0.2837 | 0.3919 | 0.0952 | 5 | 12 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-12 | instrument_balanced_master | 0.3748 | 0.3297 | 0.4103 | 0.1429 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-12 | spectrum | 0.3774 | 0.3281 | 0.4120 | 0.1429 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-13 | instrument_balanced_master | 0.3510 | 0.3297 | 0.3711 | 0.1429 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-13 | spectrum | 0.3417 | 0.3255 | 0.3588 | 0.1217 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-14 | instrument_balanced_master | 0.3146 | 0.2640 | 0.3697 | 0.1576 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-14 | spectrum | 0.3200 | 0.2664 | 0.3745 | 0.1481 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-15 | instrument_balanced_master | 0.2514 | 0.2025 | 0.2776 | 0.0000 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-15 | spectrum | 0.2562 | 0.1988 | 0.2915 | 0.0000 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-16 | instrument_balanced_master | 0.3448 | 0.2922 | 0.3938 | 0.1217 | 5 | 9 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-16 | spectrum | 0.3489 | 0.3130 | 0.3874 | 0.1481 | 5 | 9 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-17 | instrument_balanced_master | 0.3393 | 0.2597 | 0.4040 | 0.0893 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-17 | spectrum | 0.3169 | 0.2549 | 0.3686 | 0.1429 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-18 | instrument_balanced_master | 0.3372 | 0.2844 | 0.3678 | 0.1310 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-18 | spectrum | 0.3326 | 0.3038 | 0.3588 | 0.1310 | 5 | 11 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-19 | instrument_balanced_master | 0.3509 | 0.2839 | 0.4248 | 0.0476 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PERM | permuted master labels | C-PERMUTED-SELECTED:CTRL-PERM-19 | spectrum | 0.3445 | 0.2971 | 0.4010 | 0.0673 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-CONTROL-PRIOR | source prior | C-PRIOR:CTRL-PRIOR-EMPIRICAL | instrument_balanced_master | 0.3111 | 0.2713 | 0.3333 | 0.1429 | 5 | 13 | 13 | complete |
| EXP-C09-CONTROL-PRIOR | source prior | C-PRIOR:CTRL-PRIOR-EMPIRICAL | spectrum | 0.3035 | 0.2536 | 0.3301 | 0.1044 | 5 | 13 | 13 | complete |
| EXP-C09-CONTROL-PRIOR | source prior | C-PRIOR:CTRL-PRIOR-UNIFORM | instrument_balanced_master | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 5 | 13 | 13 | complete |
| EXP-C09-CONTROL-PRIOR | source prior | C-PRIOR:CTRL-PRIOR-UNIFORM | spectrum | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 5 | 13 | 13 | complete |
| EXP-C09-T3 | real spectra | C-SELECTED | instrument_balanced_master | 0.7088 | 0.6926 | 0.7296 | 0.3333 | 5 | 10 | 13 | incomplete_terminal_cells |
| EXP-C09-T3 | real spectra | C-SELECTED | spectrum | 0.6402 | 0.6230 | 0.6592 | 0.3333 | 5 | 10 | 13 | incomplete_terminal_cells |

Permutation results are leakage/chance diagnostics, not a formal permutation-test p-value. Metadata and prior controls cannot select or promote a model.

## Compute evidence (M23–M25)

| experiment_id | task_id | effective_model_id | stage_group | planned_fit_records | complete_fit_records | terminal_failure_records | timed_fit_records | total_training_seconds | median_training_seconds | p95_training_seconds | timed_inference_records | inference_prediction_rows | median_milliseconds_per_prediction | p95_milliseconds_per_prediction | sized_model_records | median_serialized_model_bytes | maximum_serialized_model_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-C00-T1 | T1-CWA | C-PRIOR | final_refit_and_prediction | 40 | 40 | 0 | 40 | 0.0715 | 0.0017 | 0.0039 | 40 | 2080 | 0.1368 | 0.4181 | 40 | 879.0000 | 880.0000 |
| EXP-C00-T1 | T1-PILLS | C-PRIOR | final_refit_and_prediction | 40 | 40 | 0 | 40 | 0.0784 | 0.0016 | 0.0043 | 40 | 2080 | 0.1243 | 0.3373 | 40 | 871.0000 | 872.0000 |
| EXP-C00-T1 | T1-SURF | C-PRIOR | final_refit_and_prediction | 40 | 40 | 0 | 40 | 0.0836 | 0.0015 | 0.0045 | 40 | 1820 | 0.1533 | 0.6277 | 40 | 879.0000 | 880.0000 |
| EXP-C01-T1 | T1-CWA | C-SPECTRAL-MATCH | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.1012 | 0.0043 | 0.0086 | 20 | 1040 | 0.2076 | 0.4305 | 20 | 34483.0000 | 34490.0000 |
| EXP-C01-T1 | T1-CWA | C-SPECTRAL-MATCH | selection | 180 | 180 | 0 | 180 | 2.4396 | 0.0135 | 0.0146 | 180 | 9360 | 0.0039 | 0.0064 | 180 | 34483.0000 | 34490.0000 |
| EXP-C01-T1 | T1-PILLS | C-SPECTRAL-MATCH | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.0784 | 0.0035 | 0.0089 | 20 | 1040 | 0.1726 | 0.3786 | 20 | 34474.0000 | 34482.0000 |
| EXP-C01-T1 | T1-PILLS | C-SPECTRAL-MATCH | selection | 180 | 180 | 0 | 180 | 2.4477 | 0.0135 | 0.0147 | 180 | 9360 | 0.0038 | 0.0056 | 180 | 34475.0000 | 34482.0000 |
| EXP-C01-T1 | T1-SURF | C-SPECTRAL-MATCH | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.0829 | 0.0038 | 0.0064 | 20 | 910 | 0.2650 | 0.6382 | 20 | 34483.0000 | 34490.0000 |
| EXP-C01-T1 | T1-SURF | C-SPECTRAL-MATCH | selection | 180 | 180 | 0 | 180 | 2.4076 | 0.0133 | 0.0145 | 180 | 8190 | 0.0041 | 0.0061 | 180 | 34483.0000 | 34490.0000 |
| EXP-C02-T1 | T1-CWA | C-NEAREST-CENTROID | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.1299 | 0.0043 | 0.0184 | 20 | 1040 | 0.2125 | 0.4568 | 20 | 34479.0000 | 79601.0000 |
| EXP-C02-T1 | T1-CWA | C-NEAREST-CENTROID | selection | 480 | 480 | 0 | 480 | 6.9180 | 0.0145 | 0.0165 | 480 | 24960 | 0.0114 | 0.0396 | 480 | 57040.0000 | 79601.0000 |
| EXP-C02-T1 | T1-PILLS | C-NEAREST-CENTROID | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.2784 | 0.0116 | 0.0267 | 20 | 1040 | 0.2668 | 0.6768 | 20 | 79585.0000 | 79593.0000 |
| EXP-C02-T1 | T1-PILLS | C-NEAREST-CENTROID | selection | 480 | 480 | 0 | 480 | 5.2955 | 0.0109 | 0.0150 | 480 | 24960 | 0.0098 | 0.0236 | 480 | 57032.0000 | 79593.0000 |
| EXP-C02-T1 | T1-SURF | C-NEAREST-CENTROID | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.2219 | 0.0107 | 0.0210 | 20 | 910 | 0.3125 | 0.8009 | 20 | 79593.0000 | 79601.0000 |
| EXP-C02-T1 | T1-SURF | C-NEAREST-CENTROID | selection | 480 | 480 | 0 | 480 | 5.2193 | 0.0107 | 0.0153 | 480 | 21840 | 0.0099 | 0.0277 | 480 | 57040.0000 | 79601.0000 |
| EXP-C03-T1 | T1-CWA | C-PCA-LDA | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.5333 | 0.0247 | 0.0429 | 20 | 1040 | 0.2578 | 0.5604 | 20 | 465467.0000 | 477237.0000 |
| EXP-C03-T1 | T1-CWA | C-PCA-LDA | selection | 600 | 600 | 0 | 600 | 10.0628 | 0.0161 | 0.0221 | 600 | 31200 | 0.0074 | 0.0170 | 600 | 239387.0000 | 477237.0000 |
| EXP-C03-T1 | T1-PILLS | C-PCA-LDA | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.3507 | 0.0163 | 0.0305 | 20 | 1040 | 0.1637 | 0.4835 | 20 | 239399.0000 | 239399.0000 |
| EXP-C03-T1 | T1-PILLS | C-PCA-LDA | selection | 600 | 600 | 0 | 600 | 10.8642 | 0.0177 | 0.0240 | 600 | 31200 | 0.0079 | 0.0113 | 600 | 216804.0000 | 477249.0000 |
| EXP-C03-T1 | T1-SURF | C-PCA-LDA | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.3997 | 0.0170 | 0.0370 | 20 | 910 | 0.1939 | 0.4558 | 20 | 465479.0000 | 477249.0000 |
| EXP-C03-T1 | T1-SURF | C-PCA-LDA | selection | 600 | 600 | 0 | 600 | 12.2025 | 0.0203 | 0.0238 | 600 | 27300 | 0.0099 | 0.0143 | 600 | 194196.0000 | 477249.0000 |
| EXP-C04-T1 | T1-CWA | C-PLS-DA | final_refit_and_prediction | 20 | 20 | 0 | 20 | 1.1106 | 0.0518 | 0.1179 | 20 | 1040 | 0.1883 | 0.5333 | 20 | 490903.0000 | 646143.0000 |
| EXP-C04-T1 | T1-CWA | C-PLS-DA | selection | 300 | 300 | 0 | 300 | 10.4044 | 0.0333 | 0.0586 | 300 | 15600 | 0.0078 | 0.0184 | 300 | 341295.0000 | 635391.0000 |
| EXP-C04-T1 | T1-PILLS | C-PLS-DA | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.4155 | 0.0209 | 0.0376 | 20 | 1040 | 0.2142 | 0.6949 | 20 | 130585.0000 | 131193.0000 |
| EXP-C04-T1 | T1-PILLS | C-PLS-DA | selection | 300 | 300 | 0 | 300 | 8.7841 | 0.0278 | 0.0473 | 300 | 15600 | 0.0068 | 0.0094 | 300 | 341287.0000 | 628983.0000 |
| EXP-C04-T1 | T1-SURF | C-PLS-DA | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.7550 | 0.0297 | 0.0893 | 20 | 910 | 0.1948 | 0.6896 | 20 | 344687.0000 | 628735.0000 |
| EXP-C04-T1 | T1-SURF | C-PLS-DA | selection | 300 | 300 | 0 | 300 | 7.8320 | 0.0246 | 0.0387 | 300 | 13650 | 0.0072 | 0.0103 | 300 | 339631.0000 | 627967.0000 |
| EXP-C05-T1 | T1-CWA | C-LOGREG-EN | final_refit_and_prediction | 20 | 20 | 0 | 20 | 65.6715 | 1.5980 | 9.9896 | 20 | 1040 | 0.1424 | 0.4226 | 20 | 68885.0000 | 68885.0000 |
| EXP-C05-T1 | T1-CWA | C-LOGREG-EN | selection | 1800 | 1774 | 26 | 1800 | 6226.4965 | 2.8770 | 9.8981 | 1774 | 93600 | 0.0099 | 0.0227 | 1774 | 68885.0000 | 68885.0000 |
| EXP-C05-T1 | T1-PILLS | C-LOGREG-EN | final_refit_and_prediction | 20 | 20 | 0 | 20 | 3.6048 | 0.1236 | 0.6357 | 20 | 1040 | 0.1822 | 0.3617 | 20 | 68877.0000 | 68877.0000 |
| EXP-C05-T1 | T1-PILLS | C-LOGREG-EN | selection | 1800 | 1753 | 47 | 1800 | 7584.4814 | 3.9499 | 10.5990 | 1753 | 93600 | 0.0110 | 0.0152 | 1753 | 68877.0000 | 68877.0000 |
| EXP-C05-T1 | T1-SURF | C-LOGREG-EN | final_refit_and_prediction | 20 | 20 | 0 | 20 | 18.6930 | 0.3243 | 2.1643 | 20 | 910 | 0.1635 | 0.3602 | 20 | 68885.0000 | 68885.0000 |
| EXP-C05-T1 | T1-SURF | C-LOGREG-EN | selection | 1800 | 1770 | 30 | 1800 | 5181.4247 | 2.4202 | 7.9523 | 1770 | 81900 | 0.0120 | 0.0178 | 1770 | 68885.0000 | 68885.0000 |
| EXP-C06-T1 | T1-CWA | C-RBF-SVM | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.2260 | 0.0116 | 0.0157 | 20 | 1040 | 0.2551 | 0.3711 | 20 | 1366279.0000 | 1764882.0000 |
| EXP-C06-T1 | T1-CWA | C-RBF-SVM | selection | 2160 | 2160 | 0 | 2160 | 37.4326 | 0.0171 | 0.0218 | 2160 | 112320 | 0.0472 | 0.0805 | 2160 | 1158547.0000 | 1686286.0000 |
| EXP-C06-T1 | T1-PILLS | C-RBF-SVM | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.2140 | 0.0104 | 0.0156 | 20 | 1040 | 0.2166 | 0.3139 | 20 | 1489788.0000 | 1686278.0000 |
| EXP-C06-T1 | T1-PILLS | C-RBF-SVM | selection | 2160 | 2160 | 0 | 2160 | 39.7783 | 0.0192 | 0.0223 | 2160 | 112320 | 0.0507 | 0.0679 | 2160 | 1124855.0000 | 1405578.0000 |
| EXP-C06-T1 | T1-SURF | C-RBF-SVM | final_refit_and_prediction | 20 | 20 | 0 | 20 | 0.1924 | 0.0092 | 0.0139 | 20 | 910 | 0.2415 | 0.3367 | 20 | 1321376.0000 | 1506638.0000 |
| EXP-C06-T1 | T1-SURF | C-RBF-SVM | selection | 2160 | 2160 | 0 | 2160 | 37.4036 | 0.0184 | 0.0212 | 2160 | 98280 | 0.0467 | 0.0646 | 2160 | 1012597.0000 | 1360674.0000 |
| EXP-C07-T1 | T1-CWA | C-RANDOM-FOREST | final_refit_and_prediction | 60 | 60 | 0 | 60 | 254.7740 | 2.5839 | 9.8121 | 60 | 3120 | 0.6965 | 1.9935 | 60 | 3282270.0000 | 5336356.0000 |
| EXP-C07-T1 | T1-CWA | C-RANDOM-FOREST | selection | 2880 | 2880 | 0 | 2880 | 10215.0382 | 2.6017 | 8.5404 | 2880 | 149760 | 0.5893 | 1.6746 | 2880 | 1996629.0000 | 4474911.0000 |
| EXP-C07-T1 | T1-PILLS | C-RANDOM-FOREST | final_refit_and_prediction | 60 | 60 | 0 | 60 | 89.5120 | 1.4841 | 1.8493 | 60 | 3120 | 0.7527 | 1.2119 | 60 | 3264594.0000 | 3697276.0000 |
| EXP-C07-T1 | T1-PILLS | C-RANDOM-FOREST | selection | 2880 | 2880 | 0 | 2880 | 9845.8408 | 2.6527 | 7.7233 | 2880 | 149760 | 0.6225 | 0.9107 | 2880 | 1725600.0000 | 2976393.0000 |
| EXP-C07-T1 | T1-SURF | C-RANDOM-FOREST | final_refit_and_prediction | 60 | 60 | 0 | 60 | 81.9659 | 1.3331 | 2.0775 | 60 | 2730 | 0.7787 | 1.2289 | 60 | 2971906.5000 | 3467921.0000 |
| EXP-C07-T1 | T1-SURF | C-RANDOM-FOREST | selection | 2880 | 2880 | 0 | 2880 | 7856.3086 | 2.1219 | 5.9862 | 2880 | 131040 | 0.6624 | 1.1108 | 2880 | 1558778.0000 | 2965985.0000 |
| EXP-C08-T1 | T1-CWA | C-EXTRA-TREES | final_refit_and_prediction | 60 | 60 | 0 | 60 | 128.5204 | 1.4382 | 4.1916 | 60 | 3120 | 0.7387 | 1.3446 | 60 | 5583026.5000 | 12338846.0000 |
| EXP-C08-T1 | T1-CWA | C-EXTRA-TREES | selection | 2880 | 2880 | 0 | 2880 | 4665.3008 | 1.3148 | 3.5939 | 2880 | 149760 | 0.5677 | 1.7150 | 2880 | 3586390.0000 | 10599616.0000 |
| EXP-C08-T1 | T1-PILLS | C-EXTRA-TREES | final_refit_and_prediction | 60 | 60 | 0 | 60 | 47.0615 | 0.8230 | 0.9302 | 60 | 3120 | 0.7653 | 0.9853 | 60 | 7463837.0000 | 8651076.0000 |
| EXP-C08-T1 | T1-PILLS | C-EXTRA-TREES | selection | 2880 | 2880 | 0 | 2880 | 4506.5186 | 1.3044 | 3.1996 | 2880 | 149760 | 0.6368 | 0.8945 | 2880 | 3123499.0000 | 6679898.0000 |
| EXP-C08-T1 | T1-SURF | C-EXTRA-TREES | final_refit_and_prediction | 60 | 60 | 0 | 60 | 48.4928 | 0.7757 | 1.1210 | 60 | 2730 | 0.8968 | 1.4234 | 60 | 4751961.5000 | 7994172.0000 |

_Table truncated to 50 of 127 rows._

Training time and per-prediction inference latency are observed wall-clock diagnostics on this workstation. They are not hardware-independent complexity measures.

## Figures

| figure_id | title | data_sha256 | vector_only | png_dpi |
| --- | --- | --- | --- | --- |
| F12 | Classical source-only selection frequency and stability | 20afc4b16cdb73db33caea51b6c3172f613959933df82d7164ef8e9fe7775f2d | True | 300 |
| F13 | Within-station classical performance | 111961c37f2c4337c66bbe694743de691c3c285b05d1904fac6d1ec261ec333b | True | 300 |
| F38 | Classical unseen-instrument domain performance and support | 73081c5a2dd38a11f10eeaabae2b1cf27c0f93bff967c8e620bc0ffdc281803b | True | 300 |
| F39 | Within-station versus unseen-instrument classical performance | 3f062a572214f2c48ed95719e9ef6abac39af444b1142aadfe95c5de7e2d2c53 | True | 300 |
| F40 | Spectrum versus instrument-balanced master performance | 949803bbf84ddfb965db8282260d16a71432b340a43874a1a92e44746690e8c3 | True | 300 |
| F41 | Classical unseen-instrument confusion matrices | 9c8bd012684fd017f5ff1e96f2d5ab5822148827a84341b7a731485ee370688d | True | 300 |
| F42 | Classical unseen-instrument calibration reliability | 9b814b0234b849393cbd192f71d73b56e0554964a4e14d851044bdd40d4b2ce1 | True | 300 |
| F43 | Classical negative controls and acquisition confounding | 71d8195a47fd1ffaaa9e9dec39468251abe2263c2303a2320d9cf8d2f1b00c6c | True | 300 |

Each figure is generated from one frozen CSV as native TikZ/PGFPlots, vector PDF, 300-DPI PNG, and standalone self-contained HTML. Figure tables and HTML remain private until disclosure review.

## Interpretation limits

- Spectra, folds, and technical repeats are not independent chemical samples.
- The endpoint concerns the tested 13 station/instrument domains and three-class station tasks; it does not establish arbitrary instrument or chemistry generalization.
- `PP-U-MIN` is the fixed primary policy. This phase does not establish universal superiority over smoothing, baseline correction, or adaptive preprocessing.
- C12 is a secondary source-to-source covariance augmentation control, not target adaptation and not evidence of causal nuisance disentanglement.
- Probability metrics are reported only where source-development cross-fitted calibration is valid; permutation controls intentionally omit them.
- P03 freezes a classical comparator for P04/P06. It does not compare against a deep model or justify a deep-learning claim by itself.
