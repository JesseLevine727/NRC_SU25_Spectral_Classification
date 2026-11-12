#!/bin/bash
# Step 3 DFT batch: B3LYP/TZVP Raman for the fixed test list (step3/test_compounds.csv), cheapest first.
# Run two copies at once (16 threads each); a lock directory per compound keeps them from overlapping.
# Each result is checked for imaginary modes and repaired (fix_imag.py). A failure is logged; the
# optimised geometry and Hessian are kept in <tag>_hess.npz, so a rerun redoes only the last step.
S="$1"; L="$2"; O="$S/results/dft_step3"
mkdir -p "$O"
export OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore DFT_DF=0 DPOL_ROUTE=field DPOL_NPROC=5 DPOL_THREADS=3 FIX_NPROC=4
cd "$O"
tail -n +2 "$L" | while IFS=, read -r tag smi rest; do
  f="$O/$tag.npz"
  [ -f "$f" ] && continue
  mkdir "$O/$tag.lock" 2>/dev/null || continue
  echo "=== $tag $smi $(date '+%F %T')"
  "$S/venv/bin/python" -I "$S/scripts/dft_raman.py" smiles "$smi" "$f" --nproc 4 --threads 4 2>&1 | grep -a -v -i warn | grep -a -E "SMILES|resumed|optimised|hessian|polariz|Error|Traceback|assert"
  if [ -f "$f" ]; then
    "$S/venv/bin/python" -I "$S/scripts/fix_imag.py" "$f" 2>&1 | grep -a -E "imaginary|lowest|Error|Traceback"
  else
    echo "FAILED $tag"
  fi
done
echo "=== stream done $(date '+%F %T')"
