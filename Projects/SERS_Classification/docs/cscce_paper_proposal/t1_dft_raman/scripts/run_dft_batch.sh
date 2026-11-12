#!/bin/bash
# Run the B3LYP/TZVP Raman pipeline for the measured compounds missing from QM9S (smallest first).
S="$1"
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore DFT_DF=0
cd "$S/results/dft"
while IFS=, read -r tag smi; do
  [ -f "$S/results/dft/new_${tag}.npz" ] && continue
  echo "=== $tag $smi $(date +%T)"
  "$S/venv/bin/python" -I "$S/scripts/dft_raman.py" smiles "$smi" "$S/results/dft/new_${tag}.npz" 2>&1 | grep -v -i warn
done <<LIST
val_acetonitrile,CC#N
formic_acid,OC=O
acetic_acid,CC(=O)O
acetone,CC(C)=O
ethylene_glycol,OCCO
tetrahydrofuran,C1CCOC1
diethylamine,CCNCC
isobutylamine,CC(C)CN
dimethoxypropane,COC(C)(C)OC
LIST
echo "=== done $(date +%T)"
