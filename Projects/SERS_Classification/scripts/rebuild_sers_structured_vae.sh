#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi

REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPOSITORY/.venv/bin/python"
OUTPUT_DIR="$(realpath -m "$1")"
WORKSPACE_DIR="$REPOSITORY/Workspace"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment is unavailable: $PYTHON" >&2
  exit 2
fi

case "$OUTPUT_DIR" in
  /|"$REPOSITORY"|"$WORKSPACE_DIR"|"$REPOSITORY/Workspace/sers_structured_vae")
    echo "Refusing unsafe or overly broad output directory: $OUTPUT_DIR" >&2
    exit 2
    ;;
esac

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Clean rebuild requires a nonexistent output directory: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON" "$REPOSITORY/scripts/audit_sers_structured_vae_metadata.py" \
  --output-dir "$OUTPUT_DIR/audit"
"$PYTHON" "$REPOSITORY/scripts/run_sers_structured_vae_identity.py" \
  --audit-dir "$OUTPUT_DIR/audit" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda

for STAGE in controls instrument_adversary pair dependence; do
  "$PYTHON" "$REPOSITORY/scripts/run_sers_structured_vae_selection.py" \
    --stage "$STAGE" \
    --output-dir "$OUTPUT_DIR" \
    --training-device cuda \
    --metric-device cpu
done

"$PYTHON" "$REPOSITORY/scripts/finalize_sers_structured_vae_inner.py" \
  --output-dir "$OUTPUT_DIR" \
  --device cpu

for STAGE in sensitivity outer domain poster; do
  "$PYTHON" "$REPOSITORY/scripts/run_sers_structured_vae_confirmation.py" \
    --stage "$STAGE" \
    --output-dir "$OUTPUT_DIR" \
    --training-device cuda \
    --metric-device cpu
done

"$PYTHON" "$REPOSITORY/scripts/export_sers_structured_vae_swaps.py" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda
"$PYTHON" "$REPOSITORY/scripts/finalize_sers_structured_vae.py" \
  --output-dir "$OUTPUT_DIR"
"$PYTHON" "$REPOSITORY/scripts/validate_sers_structured_vae.py" \
  --output-dir "$OUTPUT_DIR"

echo "Clean rebuild complete: $OUTPUT_DIR"
