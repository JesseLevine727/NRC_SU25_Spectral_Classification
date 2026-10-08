# P08-U1 supervised implementation review

## Authority and current status

The owner approved the [U1 execution limits](../P08_U1_EXECUTION.md) on 2026-10-08, conditional on implementation review. The accepted U0 recovery is unchanged. **No new U1 scientific fit, scalar calibration or held prediction has started at this checkpoint.**

DeepSeek V4.1 Flash is the implementation worker through the existing tool-disabled OpenCode patch-author wrapper. Only selected public source snapshots and synthetic fixture instructions are supplied. The supervisor applies patches, runs checks and controls scientific execution and publication. Worker statements are not acceptance evidence.

## T271–T273: per-fit live epoch monitor

Accepted as an observational component, not as a full-run controller or scientific launch gate.

- Added `visualization/p08_live_monitor.py` and focused synthetic tests. The module uses the standard library only and does not import training libraries or perform fitting/inference.
- Completed epochs are appended to private JSONL with flush/fsync; offline HTML refreshes after accepted epochs. Final semantic JSON and native pgfplots/TikZ use identical coordinates and axis definitions. Console/HTML identify an opaque job token, preprocessing, recipe, seed and stage.
- Training objective/components are distinguished from clean-evaluation training NLL and source-validation NLL/accuracy. Source validation is not held-test performance. Fixed-duration refits have no invented validation curve or best-checkpoint claim.
- Invalid records do not alter accepted evidence. I/O/render/stream failures close the monitor to further epochs and propagate. Clock observations must be finite and nondecreasing; final elapsed time is frozen. Persistence is not claimed to be an atomic multi-file transaction, and partial files remain evidence.
- Supervisor review rejected the initial HTML/TikZ axis mismatch, hidden write errors, missing visible metadata and incomplete timing checks. Two bounded worker correction rounds addressed these; the supervisor then corrected the synthetic default recipe label to D3 because that fixture includes both auxiliary losses. No scientific kernel changed.
- Focused result: **70 passed**, including real `pdflatex` compilation for source-fitting and refit plots. Ruff passed. The supervisor inspected rendered native PDF/PNG and offline browser HTML for readable legends, labels and agreement. These displays contain invented test values, not benchmark results.

Command from the package directory:

```text
../../.venv/bin/python -m pytest tests/test_p08_live_monitor.py -q --tb=short
../../.venv/bin/ruff check src/atlas_sers/visualization/p08_live_monitor.py tests/test_p08_live_monitor.py
```

Remaining integration: connect the callback to the governed worker, expose a campaign-level index/count view, and include monitoring overhead in cumulative resource observations. Passing these component tests does not establish live GPU resource enforcement or full-benchmark completion.

## T274/T277/T280: all-candidate source inputs

Accepted as an input adapter, not as execution authority. The factory authenticates the full universal graph, metadata, three representation archives, candidate registry and inherited specification sources. It retains source pairs only and uses bounded role-array caching. Kernel arguments preserve the existing physical-master roles, instrument exclusion, candidate parameters, class ordering, QC values, substrate strings and neural RNG role IDs.

The first worker response was truncated and was not applied. Review of its completed replacement found an important integration defect: a source-only validator was being applied to non-source and MIN graph jobs. The final bounded correction validates global graph identities before filtering supported source pairs. Other corrections covered metadata pins, exact pilot-container access, candidate coverage, cache eviction and mutation tests. Supervisor changes were limited to patch-format normalization and formatting.

Focused source-input tests: **17 passed**. A separate read-only check authenticated the actual 607,221-job graph and resolved all **184,392 SG/arPLS source-fit configurations**, across 260 contexts and 681 source units. All **78 pilot input pairs** matched the existing U0 adapter exactly, including arrays, observations, parameters and QC. This check took 27.69 seconds and peaked at 2.02 GiB RSS; it performed **zero fits or predictions**. Its aggregate factory-report SHA-256 is `6c1549fe8692ee5832d8a4ccfe3cd93ea0a2d038d44f95baf459fd5009df94e5`.

## T276/T279: source-only selection

Accepted as a pure selection adapter. Classical settings use the unchanged inherited lexicographic objective, complete candidate/unit/seed coverage and declared tie-breaking order. Neural refit duration uses the inherited per-seed median-best-epoch rule, rounded and clipped to 30–200 epochs; recipe identities are not reselected.

Review corrected job-key handling and neural specification identity. Two final synthetic fixture corrections made the intended tests meaningful: equal complexity for a declared-order tie and a deterministic SVM registry row. Neither changes scientific selection. Focused result: **41 passed**; Ruff passed. No real-data selection has been performed.

## T275/T278/T284: durable execution ledger

The reviewed core records immutable job identities, dependency/attempt accounting, pilot reuse and cumulative resource observations in a single-controller SQLite ledger. It does not measure hardware or authorize numerical kernels. Initial corrections removed per-start full-table budget scans, strengthened journal/table consistency checks, preserved failed-create evidence, enforced CPU/GPU lanes and retained breached-resource observations. The worker's repeated/truncated output tail was not applied; only its complete, nonduplicated patch sections were used.

Accepted as a ledger primitive after the final bounded correction. Stored job payloads are now bound to their content-addressed IDs on reopen, and database-write failures latch the instance against further starts. Focused result: **30 passed**, with Ruff passing. The supervisor normalized two worker patch paths to the explicitly assigned files; no scientific logic was changed. Full controller acceptance has not been granted.

## Pilot reuse and next integration

A fresh read-only inventory of the accepted recovery directory confirmed all **3,515 files**, **64,739,703 bytes**, and inventory SHA-256 `f0ed157a0d928a7114344e2520adbbef5daddaed39ce4f4b0e8fd41a74dabd4f`. No retained evidence was modified. Together with exact source-input parity, this supports reuse; importing the pilot's terminal receipts into the new controller still requires review.

T282 integrates post-source physical roles and fitting/held-only input separation. T283 integrates the frozen scalar-calibration and seed-aggregation order. Both are bounded implementation tasks with synthetic tests, not scientific execution. Full worker/controller integration, live resource measurement, loaded-runtime binding and end-to-end admission remain required before launch. No later preprocessing branch is included.

## Reviewed component release

The four accepted components above pass **158 focused tests** together. A clean tracked-package export with only those reviewed additions passed the public scaffold validator (**1,063 files checked**) and package-wide Ruff. The export excludes unrelated user changes, local LaTeX build files and pending implementation drafts. This is an implementation milestone, not a benchmark-result release; remote CI is checked separately after the main-branch push.

The next integration drafts remain outside this release. T282 produced no code before its worker output limit; T286 is its bounded completion attempt with a smaller source context. T283 initially passed 18 synthetic tests, but metadata/seed validation and audit-count corrections were requested in T287. T285 composes the existing calibration-fit/final-refit/held-prediction kernels without changing them. None of these drafts grants execution permission or changes the frozen experiment.

Calibration wording must distinguish physical-master-separated cross-fit roles from equal-master weighting. The inherited classical helper fits temperature to its cross-fitted observation rows after tree-seed aggregation; the neural helper first averages logits by physical master. U1 preserves both inherited numerical procedures. Neither a new weighting rule nor a calibration experiment is introduced by this adapter work.
