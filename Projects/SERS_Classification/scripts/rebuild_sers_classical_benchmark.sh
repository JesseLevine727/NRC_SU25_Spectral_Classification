#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

rebuild_dir="Workspace/sers_classical_benchmark/classical_benchmark_v2_rebuild"

if [[ -e "$rebuild_dir" ]]; then
  echo "Refusing to reuse an existing rebuild directory: $rebuild_dir" >&2
  echo "Move it to an archive location before requesting a clean rebuild." >&2
  exit 2
fi

OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python scripts/run_sers_classical_benchmark.py \
    --output-dir "$rebuild_dir" \
    --stage all \
    --jobs 32

.venv/bin/python scripts/finalize_sers_classical_benchmark.py \
  --output-dir "$rebuild_dir"

.venv/bin/python scripts/validate_sers_classical_benchmark.py \
  --output-dir "$rebuild_dir"

.venv/bin/python scripts/compare_sers_classical_benchmark_rebuild.py

.venv/bin/python scripts/validate_sers_classical_benchmark.py \
  --require-clean-rebuild
