#!/usr/bin/env bash
set -euo pipefail
.venv/bin/python scripts/run_sers_standard_vae_selection.py --device cuda
.venv/bin/python scripts/run_sers_standard_vae_final.py --stage all --training-device cuda --evaluation-device cpu
.venv/bin/python scripts/finalize_sers_standard_vae.py
.venv/bin/python scripts/run_sers_standard_vae_selection.py --device cuda --output-dir Workspace/sers_standard_vae/standard_vae_v1_rebuild
.venv/bin/python scripts/run_sers_standard_vae_final.py --stage all --training-device cuda --evaluation-device cpu --output-dir Workspace/sers_standard_vae/standard_vae_v1_rebuild
.venv/bin/python scripts/finalize_sers_standard_vae.py --output-dir Workspace/sers_standard_vae/standard_vae_v1_rebuild
.venv/bin/python scripts/compare_sers_standard_vae_rebuild.py
.venv/bin/python scripts/validate_sers_standard_vae.py --require-clean-rebuild
