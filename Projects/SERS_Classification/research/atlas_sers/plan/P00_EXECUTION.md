# P00 governance execution

P00 establishes the state from which ATLAS can begin data engineering. It is a
definitive governance and input-integrity check, not a model experiment. It
must import no training package, invoke zero fits, create no representation or
split, and make no scientific interpretation.

## Preconditions

- `ATLAS_PRIVATE_ROOT` resolves the immutable files declared in
  `contracts/research_contract.json`.
- `ATLAS_ARTIFACT_ROOT` is a different private directory outside this public
  project.
- The authoritative inputs are untracked by Git.
- Contracts and all nine registries parse and cross-reference successfully.

## Commands

Run from `research/atlas_sers`:

```bash
python3 -m pip install -e '.[dev]'
python3 scripts/validate_public_scaffold.py
python3 scripts/run_p00.py audit
pytest -q
python3 scripts/run_p00.py dry-run
python3 scripts/run_p00.py dry-run
```

`audit` is public and structural; it does not resolve private inputs. `dry-run`
performs the authoritative private verification and writes private evidence.
The first unchanged execution reports action `new`; the mandatory repeat must
report `verified_skip` with the same deterministic run ID.

## Required private outputs

The run directory `${ATLAS_ARTIFACT_ROOT}/p00/runs/<run_id>/` contains exactly:

1. `environment_lock.json`
2. `input_verification.json`
3. `protected_state.json`
4. `deviations.csv`
5. `expected_run_registry.csv`
6. `fit_count_by_phase_model_task.csv`
7. `estimated_gpu_hours.csv`
8. `estimated_cpu_hours.csv`
9. `estimated_disk_bytes.json`
10. `shard_manifest.csv`
11. `P00_VALIDATION_REPORT.json`
12. `P00_ARTIFACT_HASHES.json`

The artifact store adds `_STATE.json` as its atomic commit record and maintains
`p00/LATEST.json`. The dry-run rows are provisional: every registered
experiment is enumerated, but fields that depend on P01/P02 are explicitly
unresolved and every `fit_authorized` value is `false`.

## Status and failure contract

- `pass` / exit `0`: all checks passed and P00 evidence is complete.
- `fail` / exit `1`: governance, hash, shape, declared status, privacy, or
  reproducibility did not pass.
- `blocked` / exit `2`: one or more authoritative inputs are missing.
- An unset root, unsafe root overlap, or non-contained declared path stops
  execution rather than writing a false report.

A successful matching run is rehashed and skipped. An incomplete, corrupt,
stale, or conflicting run is moved to private quarantine with a structured
reason before a new atomic attempt. No completed evidence is overwritten.

## Boundary to P01

P01 is authorized only after all of the following are true:

- the public scaffold validator, structural audit, and synthetic tests pass;
- `P00_VALIDATION_REPORT.json` conforms to its schema and reports `pass`;
- the authoritative input checks and artifact hash manifest are complete;
- a second invocation returns `verified_skip`;
- `phase_registry.csv` records P00 as `complete`.

Only then may P01 build the canonical observation manifest, parse instrument
metadata, determine measured support, or materialize preprocessing candidates.
P00 never authorizes P02 split construction or any classical/deep model fit.
