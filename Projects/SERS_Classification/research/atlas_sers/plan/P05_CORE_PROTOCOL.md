# P05 core protocol and bounded numerical smoke

**Locked:** 2026-09-25, before P05 outcome-bearing fitting. **Owner instruction:** lock the sparse-split, sampling/loss and budget rules, delegate implementation to DeepSeek, and run a small reviewed smoke. The owner separately approved nested source-only selection per outer test split. This is a dated P05 amendment, not a rewrite of P00–P04/P13.

## 1. Scientific scope and information boundary

The question is whether the compact CNN can use repeated physical samples to improve chemical classification across acquisition conditions. Inputs remain `PP-U-MIN` / `R_MIN_400_1800`, 400–1800 cm−1, 1,401 channels. Neither smoke outcomes nor subsequent acquisition-aware results select preprocessing. P08 and P14 remain separate.

The core ladder is **D0-M, D1, D2, D3**: matched ordinary cross-entropy; cross-entropy plus supervised contrastive loss; cross-entropy plus paired consistency; and both auxiliary terms. D0-M is a new sampler/objective-matched control, not a replacement for the immutable historical P04 D0 result. D4/CORAL and D5/adversarial training receive **zero fits** in this version and need their own mechanism specification and budget amendment before implementation/execution.

There is no global outcome-selected winner. For later evaluation, each outer context selects using only its own authorized source roles. No score or selected setting from another context enters this decision. Historical held results are known; this is a prospective procedural freeze, not a claim that the whole research program was preregistered or never inspected.

The current authorization ends at the numerical smoke. It authorizes neither the full core ladder nor P06 outer predictions. The smoke uses only four explicitly registered `selection_fit` roles, fixed recipes, and training diagnostics. No validation/test metric selects a model, checkpoint, weight, temperature, or budget. The frozen NPZ container must be opened for extraction, but only allowlisted fitting rows enter any fit or diagnostic calculation. No held-test probabilities or metrics are produced.

## 2. Exact sampling and equal-master objective

Every batch includes **every physical master in its fitting role**, not a spectrum-frequency draw. Within each master, uniformly select two distinct recorded instruments without replacement where available, otherwise its single instrument. Uniformly select one stored observation from each selected master/instrument view. Repeats rotate across draws; the same UID is never duplicated within a batch. Sort input identities before random draws and return rows in canonical UID order.

The audit found at most 19 masters in any inherited fitting role, so this produces at most 38 rows and stays below the 48-row cap. If a future role exceeds capacity, fail closed before fitting; do not discard masters or silently change the sampling scheme. All available classes and masters are included, so two masters per class and cross-instrument views are present whenever the role actually supports them. A single-instrument master contributes one row, not two copies of itself.

For C classes, M_c fitting masters of class c, and V_m sampled views of master m, give each sampled row weight **1/(C M_c V_m)**. These sum to one. Cross-entropy is the weighted sum of unreduced per-row losses. This is equal class → equal physical master → equal sampled instrument-view weighting; it is not P04's spectrum-frequency class weighting. All four core recipes use the same batches, augmentation draws, backbone initialization and training RNG schedule for a role/seed. Recipe identity must not enter those shared random streams.

Sampling draws use SHA-256-derived seeds over sampler version, inherited role ID, training seed, epoch and batch ordinal. Within a master/instrument, having more repeats increases which measurement can be sampled, not that master's total objective weight. No interpolation between chemicals, peak deletion, or new denoising is introduced.

## 3. Model and exact auxiliary losses

Reuse `CompactSERSClassifier` unchanged: its station-local three-class model has **208,691** trainable parameters. D1/D3 add one bias-enabled `Linear(64,64)` projection and L2 normalization with epsilon **1e-8**; no hidden layer, extra dropout, or batch normalization. The auxiliary head adds **4,160** parameters, giving **212,851** total. D0-M/D2 have no unused projection head. All trainable auxiliary parameters count against the strict 250,000 ceiling. Classification always uses the pre-projection embedding.

### Supervised contrastive loss

For each anchor, positives are other unique sampled UIDs from the same station and chemical. Other chemicals in that station are negatives; cross-station batches are rejected. The denominator contains every other sampled row, excluding the anchor. Similarity is dot product of L2-normalized projections divided by temperature **0.1**, evaluated through stable log-sum-exp.

Positive weights are 1.0 for different-master/same-instrument; 1.5 for different-master/different-instrument; and 2.0 for same-master/different-instrument. Multiply by 1.25 for different **known normalized substrate families**, capped at 2.5. Unknown/missing/NA family labels do not establish different-substrate support. Same-master/same-instrument positives cannot occur under the sampler and are rejected if a caller supplies them.

For each eligible anchor, normalize its positive weights to sum to one and take their weighted negative log-probability. Average anchor losses using the equal-class/master/view weights, renormalized over eligible anchors only. Record eligible and zero-positive anchors separately. If there are no eligible positives or no different-chemical negatives, return a differentiable zero and `available=false` with a reason; never present absence of supervision as successful disentanglement.

### Paired consistency

Use unordered pairs of distinct instruments measuring the **same physical master**. For pair i,j define symmetric KL as 0.5[KL(p_i||p_j)+KL(p_j||p_i)], using log-softmax probabilities without detached targets. The pair loss is **0.5 × symmetric KL + 0.5 × (1 − cosine(h_i,h_j))**, with pre-projection embeddings normalized using epsilon 1e-8. Average pairs within a master, then equally over eligible masters. Under the two-view sampler there is one pair per eligible master. Gradients flow into both views.

If no cross-instrument master pair exists, return a differentiable zero, `available=false`, and zero pair/master counts. Do not drop the fitting role, borrow held rows, or weaken master exclusion. D2 then reduces exactly to D0-M; D3 reduces exactly to D1, under the matched random streams. This equivalence is a required smoke check for the sparse role.

Total loss is CE + lambda_supcon × SupCon + lambda_pair × paired. Fixed recipes: D0-M=(0,0), D1=(0.3,0), D2=(0,0.3), D3=(0.3,0.3). These values lie in the previously proposed grids but are locked centrally before P05 results; this version does **not** search the full grid. Missing terms do not trigger redistribution or renormalization of the remaining lambda values.

## 4. Frozen optimization and numerical smoke

All recipes use AdamW, learning rate **0.0003**, weight decay **0.0001**, gradient clipping **5.0**, deterministic torch operations, and one CPU thread. The same P04 source-only augmentation implementation and noise quantiles are reused without editing it. Initialization, dropout, batch and augmentation streams are recipe-independent. Reset the training RNG after optional head construction so adding the head cannot shift backbone/dropout random draws.

Smoke roles are selected by metadata only: the lexicographically first supported within-station `selection_fit` role for each of CWA, pills and surfaces; plus the surface single-instrument selection role minimizing (master count, role ID). These have 13,9,13,4 masters respectively; the sparse role has four spectra and two zero-positive anchors. Exact private role IDs, source hashes and UID sets must be frozen by the generated no-fit registry before arrays enter a fit.

Run four recipes × four roles × two seeds (**20260805, 20260817**) = **32 primary fits**. Each fit runs exactly **8 epochs × 4 batches = 32 optimizer steps**. There is no early stopping or best-checkpoint selection; retain the terminal checkpoint. Reserve **two additional exact numerical replays**, dense-CWA D3 and sparse-surface D2 at the first seed, for a maximum of **34 fit executions / 1,088 optimizer steps**. Replays are identified as replay executions of the same numerical fit specification, never independent evidence or replacement successes. No automatic retry is permitted.

Use a single CUDA process if resource checks pass; otherwise record a CPU execution decision before fitting, not a silent mid-run fallback. Limit the complete smoke to **900 wall-clock seconds**, each fit to **120 seconds**, and allocated CUDA memory to **4 GiB**. External watchdogs may use a short termination grace beyond these limits. All failures/partial histories remain recorded. A budget exceedance stops work; it does not grant more budget.

Required records include initial/final backbone and head hashes, exact parameter counts, batch/augmentation/pair digests, CE and each auxiliary component, applicability and support counts, finite/nonzero gradient checks, unclipped gradient norms and clipping fractions, train-only chemical metrics, embedding variance/norms, per-epoch optimizer steps, elapsed time and memory, terminal status, input/contract/code/environment hashes. Public outputs exclude identities, paths, spectra, row-level predictions and checkpoints.

Numerical acceptance requires all registered primary fits finite and complete; expected auxiliary branches exercised on dense roles; absence explicitly recorded on the sparse role; actual core-weight updates; correct head gradients where enabled; both planned replays agreeing in semantic histories/checkpoint hashes; and sparse D0-M/D2 and D1/D3 equivalence. Training accuracy or eight-epoch class collapse is diagnostic, **not** an advancement claim or a reason to choose a recipe. A small smoke cannot establish generalization, optimal epochs, or superiority.

Figures will show epoch-wise loss/support and gradient trajectories as scatter/line plots, with identical semantic data in native TikZ and offline HTML, plus compiled PDF/PNG previews. Label every plot as training-only numerical smoke; no generalization confidence interval or chemical-disentanglement claim is appropriate.

## 5. Finite later-core budget and nested selection (not authorized to execute)

The fixed four-recipe ladder replaces the unaffordable Cartesian loss × optimizer crossing. Full development retains the three P04 seeds, 30–200 epochs, patience 20 and source-only checkpoint ordering. The new all-master sampler uses four draws per epoch for every recipe; the original P04 D0 remains historical and D0-M is the matched comparator.

Inherited inner units total **861**. There are **128** T3 contexts with real pseudo-domain selection and **132** with master-CV fallback (including all 100 pills contexts). For the 128 pseudo-domain contexts, separately register three source-master-CV guard units per context, adding **384** unit slots. Construct them using only that context's outer-source masters, class-stratified deterministic hash ordering and round-robin allocation; require all three classes in fit and validation. If that support fails, record the guard unavailable and disallow G3 advancement for that context. Do not relabel master-CV as pseudo-instrument validation.

Thus the exact **planned inner fit-slot ceiling is (861+384) × 4 recipes × 3 seeds = 14,940**. Unsupported guard slots are reason-coded exclusions, not silently removed or filled with different data. The later executable ledger must reconcile all slots before any full development launch. D4/D5 and all additional hyperparameter trials have zero slots.

For each pseudo-domain context, compare every candidate against D0-M using only that context's complete three-seed source predictions: mean pseudo-domain BA gain ≥0.02; worst pseudo-domain BA change ≥−0.02; mean guard-unit BA change ≥−0.02; strictly positive paired BA improvement in at least 60% of pseudo-domains; collapse fraction ≤5%. Seeds are averaged within unit before unit summaries. All scheduled pseudo-units, including roles with unavailable auxiliary terms, remain in denominators. A missing/failed fit prevents that candidate's advancement. Candidates passing all criteria are ordered by mean pseudo-domain BA, worst pseudo-domain BA, mean macro-F1, then D1,D2,D3 complexity order. If none passes, or the context has only master-CV fallback, the adaptive procedure returns D0-M; D3 remains a fixed mechanistic control. There is no cross-context winner selection or pill-specific transfer claim from master-CV.

P06 may later refit D0-M, selected recipe, and fixed D3 across 320 contexts and three seeds: **at most 2,880 neural refit slots**. Identical selected/control specifications are aliases, not duplicate fits. Epoch inheritance uses the rounded median selected-recipe inner best epoch, clipped to [30,200]; calibration uses only the context's inherited source-validation logits with equal physical-master weight, not guard predictions or outer-test rows. At most **2,880 scalar calibrations** are separately accounted; they are not neural fits.

Overall neural ceiling: **17,820 later fits + 34 current smoke executions = 17,854**, before any separately approved retry/amendment. This is a fit-count ceiling, not a GPU-hour estimate or authorization. Throughput from the smoke is only a lower-information pilot for a later cost estimate; it cannot trigger an unattended full run. Reserve at most 100 GiB for later private artifacts, subject to a separate pre-launch disk audit. No duration for the full ladder is authorized now.

## 6. Decision closure and audit trail

U01/U02: fixed recipes/optimizer and explicit finite slot ledger; no Cartesian grid. U03/U04: one 64→64 head and total-parameter accounting. U05/U06: exact objective equations above. U07/U08: all-master/two-view sampler; pair identities include role, sorted observation UIDs, contract/version and immutable input provenance. U09: unavailable auxiliary terms omitted with explicit counters, no split relaxation. U10/U11: D4/D5 explicitly deferred with zero budget, not claimed resolved scientifically. U12: new matched D0-M. U13: owner-approved nested source-only G3 with separate guard units and honest unsupported-context fallback. U14: inherited source-only epoch/calibration rules for later development; neither selection nor calibration occurs in the smoke.

The old contract-only `run_p05.py readiness/check` remains the historical grid audit and cannot authorize this new core scope. The core plan/smoke entrypoint must validate this new contract explicitly. Neither an old P04 authorization nor a successful metadata audit authorizes P05 execution by itself.
