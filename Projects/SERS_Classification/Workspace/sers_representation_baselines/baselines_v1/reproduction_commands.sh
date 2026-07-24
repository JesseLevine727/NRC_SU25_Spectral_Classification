#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
.venv/bin/python scripts/run_sers_representation_baselines.py \
  --stage selection --device cuda --output-dir /home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/Workspace/sers_representation_baselines/baselines_v1
.venv/bin/python scripts/run_sers_baseline_final.py \
  --stage all --device cuda --output-dir /home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/Workspace/sers_representation_baselines/baselines_v1
# Canonical, bitwise-stable inference replay from the CUDA-trained checkpoints.
.venv/bin/python scripts/run_sers_baseline_final.py \
  --stage all --device cpu --output-dir /home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/Workspace/sers_representation_baselines/baselines_v1
.venv/bin/python scripts/finalize_sers_baseline_bundle.py \
  --output-dir /home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/Workspace/sers_representation_baselines/baselines_v1
.venv/bin/python scripts/validate_sers_baseline_bundle.py \
  --output-dir /home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/Workspace/sers_representation_baselines/baselines_v1
