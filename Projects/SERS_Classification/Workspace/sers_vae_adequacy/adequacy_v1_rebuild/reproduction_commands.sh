#!/usr/bin/env bash
set -euo pipefail
.venv/bin/python scripts/run_sers_vae_adequacy_selection.py --device cuda
.venv/bin/python scripts/run_sers_vae_adequacy_ablation.py --device cuda
.venv/bin/python scripts/run_sers_vae_adequacy_final.py --stage all --training-device cuda --evaluation-device cpu
.venv/bin/python scripts/finalize_sers_vae_adequacy.py
.venv/bin/python scripts/validate_sers_vae_adequacy.py
.venv/bin/python scripts/run_sers_vae_adequacy_selection.py --device cuda --output-dir Workspace/sers_vae_adequacy/adequacy_v1_rebuild
.venv/bin/python scripts/run_sers_vae_adequacy_ablation.py --device cuda --output-dir Workspace/sers_vae_adequacy/adequacy_v1_rebuild
.venv/bin/python scripts/run_sers_vae_adequacy_final.py --stage all --training-device cuda --evaluation-device cpu --output-dir Workspace/sers_vae_adequacy/adequacy_v1_rebuild
.venv/bin/python scripts/finalize_sers_vae_adequacy.py --output-dir Workspace/sers_vae_adequacy/adequacy_v1_rebuild
.venv/bin/python scripts/compare_sers_vae_adequacy_rebuild.py
.venv/bin/python scripts/validate_sers_vae_adequacy.py --require-clean-rebuild
