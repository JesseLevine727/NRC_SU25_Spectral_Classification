#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

rebuild_dir="Workspace/sers_supervised_contrastive/contrastive_v1_rebuild"

if [[ -e "$rebuild_dir" ]]; then
  echo "Refusing to reuse an existing rebuild directory: $rebuild_dir" >&2
  echo "Move it to an archive location before requesting a clean rebuild." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

.venv/bin/python scripts/run_sers_contrastive_selection.py \
  --output-dir "$rebuild_dir" \
  --stage all

.venv/bin/python scripts/run_sers_contrastive_final.py \
  --output-dir "$rebuild_dir" \
  --stage all

.venv/bin/python scripts/finalize_sers_contrastive.py \
  --output-dir "$rebuild_dir"

.venv/bin/python scripts/validate_sers_contrastive.py \
  --output-dir "$rebuild_dir"

.venv/bin/python scripts/compare_sers_contrastive_rebuild.py

.venv/bin/python scripts/validate_sers_contrastive.py \
  --require-clean-rebuild
