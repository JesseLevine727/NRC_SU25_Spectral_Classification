# Preprocessing pilot: recovery results

**2026-10-08. Decision: GO for the separately authorized full universal-preprocessing benchmark (U1).**

The single approved recovery replay completed **78 source fits and 78 saved-output verification operations**, with no failures or unstarted jobs. The measured launcher duration, including cleanup, was **757.695 seconds (12 minutes 38 seconds)**. The pilot establishes that the frozen preprocessing/model combinations can execute and produce internally consistent outputs. It does not establish which preprocessing method is best or how well models generalize to a held-out instrument.

## What ran

The replay used the existing smoothing representation (`R_SG_400_1800`) and baseline-corrected representation (`R_ARPLS_400_1800`), with **39 fits per representation**. Minimal-preprocessing fits were not rerun. Coverage comprised **five preselected source-validation units across three stations**, not the complete benchmark. Each fitting set contained **7–12 physical samples**, with **5–12 separate physical samples** in its validation set.

| Model or CNN recipe | Successful fits | Completed epochs |
|---|---:|---:|
| Extra Trees | 18 | Not applicable |
| Random Forest | 18 | Not applicable |
| RBF-SVM | 6 | Not applicable |
| Ordinary matched CNN (D0-M) | 18 | 30–78 |
| CNN recipe D1 | 6 | 30–35 |
| CNN recipe D2 | 6 | 30–45 |
| CNN recipe D3 | 6 | 30–43 |

All **36 CNN fits** performed optimizer updates and completed within the inherited 120-second per-fit guard. None reached the 200-epoch ceiling; none raised the inherited collapse flag. D0-M/D2 contained **208,691 trainable parameters**, and D1/D3 contained **212,851**. These are execution diagnostics, not evidence of predictive superiority or sufficient training data.

## Independent checks and resource use

Review replayed **3,048 journal events**, verified **156 terminal receipts** and **534 referenced artifact entries**, and checked all **78 saved prediction-parity records**. All **72 neural checkpoints** loaded with finite tensors; all **36 logit archives** and **42 classical prediction tables** contained finite numerical outputs. Reconstructing all 78 frozen input pairs found **zero fitting/validation physical-sample overlaps** and **zero held-instrument rows in the consumed pairs**. No held-test predictions were made.

The replay retained **64,739,703 bytes**. Including the original failed attempt, retained run artifacts totaled **64,846,625 bytes (61.84 MiB)**. The largest recorded per-fit allocated-CUDA peak was **141.01 MiB**, below the 4-GiB inner guard. A live observation after 66 completed pairs recorded a **2.05-GiB main-process resident high-water mark**, with no child processes observed. This is not a complete final process-tree memory trace. Outer resource checks passed; their point-in-time observations do not establish hard operating-system isolation or a complete resource history.

With the conservative ten-minute reserve for the prior attempt and finalization, charged execution time was **1,357.695 seconds**, below the combined 90-minute ceiling. Fit timings in the [aggregate review](../results/p08_u0/recovery_review.json) exclude outer checking, persistence and saved-output verification; they should not be treated as a complete full-benchmark runtime estimate.

## What changed, and what did not

DeepSeek V4.1 Flash authored the CUDA preflight and recovery-permit patch; the supervisor reviewed it. The launcher now refuses a missing or incorrect `CUBLAS_WORKSPACE_CONFIG` before claiming an output directory or importing CUDA. It requires `:4096:8` and never silently replaces a conflicting value. Four regression cases failed against the old launcher; after the fix, **85 focused tests passed in 9.67 seconds**. A separate non-training deterministic CUDA check passed on the actual GPU. No code-correction round or further scientific retry was required.

The original [failed-attempt report](P08_U0_PILOT_REPORT.md) remains unchanged. Across both attempts there were **83 fit attempts: 82 successes and the original one pre-training failure**, plus **82 successful validation operations**. Four successful slots were repeated during recovery; they are not four additional independent experiments. Scientific kernels, data, representations, roles, candidates, seeds and stopping rules were unchanged.

## Next step

The operational pilot gate passes. The next scientific phase is the full universal-preprocessing benchmark under its existing plan and a separately approved execution budget. No preprocessing winner was selected from this pilot, no extra model experiment was added, and no full-benchmark training was launched. The [approval record](../plan/P08_U0_EXECUTION.md) and aggregate review define the scope. Lenarizer guided the report's separation of measured results, resource observations and untested claims.
