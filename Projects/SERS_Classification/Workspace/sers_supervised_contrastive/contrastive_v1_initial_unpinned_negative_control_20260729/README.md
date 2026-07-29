# Archived unpinned negative-control evaluation

This directory preserves the one canonical artifact that failed the first
clean-rebuild comparison on 2026-07-29, together with that mismatch report.

The trained state hashes, training histories, seeds, balanced accuracies,
macro-F1 values, and accuracies were identical. The mismatch was confined to
derived probability/embedding diagnostics from a later cached regeneration
performed outside the thread-pinned rebuild shell, plus the output order of
the encoder/classifier parameter columns.

The canonical `negative_control_metrics.csv` was subsequently regenerated
from the same cached model states with the deterministic environment used by
`scripts/rebuild_sers_contrastive.sh`:

- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `PYTHONHASHSEED=0`

No model was retrained and no locked performance or promotion outcome changed.
