# P05-T006 — core sampler, objectives, and model adapter

**Date:** 2026-09-25. **Author:** OpenCode Go / DeepSeek v4.1 Flash. No scientific fitting or raw-data access.

Read only this task, `plan/P05_CORE_PROTOCOL.md`, `plan/contracts/p05_core_contract.json`, and `src/atlas_sers/models/deep.py`, at most four reads. Do not list directories or use unrelated searches. Return scoped `apply_patch` additions; supervisor applies and tests. No shell, direct edits, Git, external agents, installed dependencies or dataset access. No execution claims.

Allowed additions:

- `src/atlas_sers/evaluation/p05_sampling.py` (standard library only);
- `src/atlas_sers/evaluation/p05_objectives.py` (torch objectives);
- `src/atlas_sers/models/acquisition.py` (adapter around unchanged P04 model);
- `tests/test_p05_sampling.py`;
- `tests/test_p05_objectives.py`.

## Required API and behavior

`p05_sampling.Observation`: frozen dataclass fields `uid, master, station, target, instrument, substrate`, all strings. `validate_rows(rows)` rejects duplicate/blank/trim-altered UIDs, master/class contradictions, empty identifying fields and mixed stations. Unknown substrate strings are allowed but do not imply different known families. Functions must not mutate caller inputs.

`sample_master_views(rows, *, role_id, seed, epoch, batch_ordinal, max_batch_size=48)` returns a frozen `MasterBatch` with `indices: tuple[int,...]`, `weights: tuple[float,...]`, `draw_sha256: str`. Include every master; select up to two distinct instruments/master uniformly and one UID/instrument uniformly. Sort identities before RNG use; return indices ordered by UID. Weights follow the protocol and sum to one. RNG is SHA-256 of `[sampler_version, role_id, seed, epoch, batch_ordinal]`, converted to an integer for a local `random.Random`, never global RNG. Reject impossible capacity before sampling and invalid epoch/ordinal/seed types. Capacity is sum over masters of min(2, number of instruments), not blindly twice all masters. Draw digest binds version, role, seed, epoch, ordinal and selected UIDs. Input row permutation must preserve selected UID sequence and weights.

`weighted_cross_entropy(logits, labels, weights)` returns weighted unreduced CE with validated finite positive row weights normalized to sum one. `supervised_contrastive(projections, rows, weights, *, temperature=0.1, epsilon=1e-8)` and `paired_consistency(logits, embeddings, rows, *, epsilon=1e-8)` return `LossResult` with tensor `loss`, bool `available`, optional string `reason`, and integer `counts` dictionary. Validate shapes, finite values, unique rows, one station and legal same-master/instrument multiplicity. Use stable log-sum-exp / log-softmax, no CPU detaching of differentiable terms. Follow all weighting and fallback equations exactly; mask absent positives without 0×infinity NaNs. No-positive/no-negative/no-pair cases must return differentiable zeros and correct counters, not exceptions for valid sparse roles. Invalid schemas/nonfinite values must raise. SupCon counts include eligible/zero-positive anchors, ordered positive pairs and negative pairs; paired counts include unordered pairs and eligible masters. Known-family comparison is casefolded; `'',na,n/a,none,unknown,not_applicable,unspecified` are unknown.

`AcquisitionClassifier(class_count=3, *, use_projection=False)` owns `backbone=CompactSERSClassifier(class_count)` and optional `projection=nn.Linear(64,64,bias=True)`. `forward(values)` returns `(logits, embedding, normalized_projection_or_None)`. Classification uses only the backbone embedding. The runtime will reset shared RNG after head creation. Enforce/read out no batch normalization and count every parameter; do not edit the historical model. The protocol's three-class totals must match exactly.

## Tests and acceptance

Use synthetic observations only. Cover deterministic seed replay, changed draws across seeds/steps, shuffled input invariance, all-master inclusion, no duplicate UID, at most two instruments, weights/equal class-master contributions even with many repeats, singleton classes, all-single-instrument roles, capacity failure, invalid metadata and no input mutation. Check loss values against independent manual CE/KL/cosine and weighted log-softmax examples, permutation invariance, unknown-family behavior, zero-positive anchors, no negatives, no pairs, both-view gradients, differentiable fallback zeros, extreme finite logits, finite-difference/autograd checks, exact parameter counts and auxiliary-head gradient flow. Include expected rejection of same-master/same-instrument duplicates in objective inputs. Torch tests may importorskip when torch is absent, but supervisor runs them with torch installed. Avoid vacuous assertions or tests that merely compare the implementation to itself.

Deliver a compact correct implementation in separate module and test patches if needed. Stop after patch delivery. Full runtime, registries, data extraction, figures and actual smoke are separate assignments.
