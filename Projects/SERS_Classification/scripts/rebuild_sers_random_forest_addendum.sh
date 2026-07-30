#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

OUTPUT_DIR="${1:-Workspace/sers_random_forest_addendum/random_forest_v1_rebuild}"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
  exit 1
fi

.venv/bin/python scripts/run_sers_random_forest_addendum.py \
  --output-dir "$OUTPUT_DIR" --stage all --jobs 4
.venv/bin/python scripts/finalize_sers_random_forest_addendum.py \
  --output-dir "$OUTPUT_DIR"
.venv/bin/python scripts/validate_sers_random_forest_addendum.py \
  --output-dir "$OUTPUT_DIR"
