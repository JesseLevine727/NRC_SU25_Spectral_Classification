# Archived unpinned negative-control evaluation

This directory preserves the one canonical artifact involved in the first
clean-rebuild mismatch on 2026-07-29, together with that mismatch report. The
directory name records the initial hypothesis; subsequent diagnosis showed
that thread pinning was not the deciding cause.

The trained state hashes, training histories, seeds, balanced accuracies,
macro-F1 values, and accuracies were identical. The mismatch was confined to
derived probability/embedding diagnostics and the output order of the
encoder/classifier parameter columns.

Element-wise comparison proved that the canonical and rebuild checkpoint
tensors and their metadata were exactly identical. The actual distinction was
evaluation from the just-trained in-memory CUDA module versus evaluation after
serializing and reloading that same state. The canonical table reflected the
checkpoint-reloaded evaluation; the first clean-rebuild table reflected the
in-memory evaluation.

`scripts/rebuild_sers_contrastive.sh` now performs a cached negative-control
evaluation after checkpoint reload. The serialized state is therefore the
explicit reproducibility boundary.

No model was retrained, no checkpoint changed, and no locked balanced
accuracy, macro-F1, accuracy, or promotion outcome changed.
