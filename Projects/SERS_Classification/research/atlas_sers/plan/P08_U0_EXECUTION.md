# P08 U0: bounded preprocessing pilot

**Approved:** 2026-10-08. **Scope:** one source-only pilot attempt and an independently reviewed go/no-go report; no full benchmark.

The owner requested the next pilot goal after review of the completed P08 readiness package and explicitly approved a **100,000-token** agent-work ceiling. The pilot reuses the existing frozen proposal, inputs, physical-sample-separated roles, candidates, recipes and seeds. Approval covers at most **78 source fits** and their **78 dependent source-validation operations**, executed serially. These comprise **42 classical** and **36 neural** fit–prediction pairs and are already included in the prospective U1 budget.

| Limit | Approved ceiling |
|---|---:|
| Cumulative active execution, including setup and finalization | 90 minutes |
| New private run artifacts | 8 GiB |
| Process-tree RAM | 16 GiB |
| Allocated GPU memory | 8 GiB |
| Free-space reserve beyond the remaining artifact allowance | 30 GiB |
| Automatic scientific retries | 0 |

The inherited neural settings remain **30–200 epochs**, **20-epoch patience**, a **120-second** per-fit guard and a **4-GiB** inner allocated-GPU guard. Source validation supports the inherited stopping and saved-output verification procedures. Held-test prediction, winner selection, calibration, final refitting, adaptive routing, resampling, perturbations and preprocessing-array reconstruction remain excluded.

The private permit's byte SHA-256 is `a0ddf4adbbdad560de83941e5ca8f5330b98e000928f887ac88a69afdb04947d`. It binds the existing proposal, attempt manifest, source catalog, input locations, limits and one exclusive private destination. The launcher accepts only that permit; no command-line replacement hash or alternate destination is permitted. Historical no-fit readiness records remain evidence of their original scope, not this separate execution approval.

DeepSeek V4.1 Flash authored the deployment pin and synthetic refusal test (T268), followed by a prose-only clarification (T269). The supervisor reviews the patch, verifies the consumed source/input identities and controls the sole launch. The deployment does not alter scientific kernels, models, splits or numerical settings. Four later-stage families—adaptive preprocessing, robustness, normalization/range controls and population sensitivities—are not prerequisites for this pilot.

If preflight refuses execution or the attempt fails, preserve the partial evidence and issue a no-go report. No automatic retry, fresh output destination or full-benchmark launch follows. Permit binding, relevant checks, one attempt and its report are the entire scope. At most two corrective implementation rounds may address the same defect; a required redesign becomes a separately proposed task. The agent budget has no automatic extension.

Completion requires reviewed accounting of attempted, completed and unstarted jobs, artifact integrity, source-only numerical outputs and observed resource use, followed by a concise public report and reviewed publication to `main`. A low source-validation score alone is not an implementation failure. Pilot scores do not establish the best preprocessing method or held-instrument generalization. Resource measurements are observations under the existing cooperative guards, not proof of hard operating-system isolation.
