#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python scripts/audit_sers_structured_vae_metadata.py
.venv/bin/python scripts/run_sers_structured_vae_identity.py --device cuda
.venv/bin/python scripts/run_sers_structured_vae_selection.py --stage controls --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_selection.py --stage instrument_adversary --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_selection.py --stage pair --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_selection.py --stage dependence --training-device cuda --metric-device cpu
.venv/bin/python scripts/finalize_sers_structured_vae_inner.py --device cpu
.venv/bin/python scripts/run_sers_structured_vae_confirmation.py --stage sensitivity --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_confirmation.py --stage outer --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_confirmation.py --stage domain --training-device cuda --metric-device cpu
.venv/bin/python scripts/run_sers_structured_vae_confirmation.py --stage poster --training-device cuda --metric-device cpu
.venv/bin/python scripts/export_sers_structured_vae_swaps.py --device cuda
.venv/bin/python scripts/finalize_sers_structured_vae.py
.venv/bin/python scripts/validate_sers_structured_vae.py

# Independent clean rebuild (the destination must not already exist):
# scripts/rebuild_sers_structured_vae.sh Workspace/sers_structured_vae/structured_vae_v1_rebuild
