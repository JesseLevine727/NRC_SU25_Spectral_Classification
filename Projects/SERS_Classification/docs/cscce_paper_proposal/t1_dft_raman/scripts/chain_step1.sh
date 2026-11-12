#!/bin/bash
# After the DFT batch: repair saddle points, extend the library, rerun the comparison, summarise.
set -e
S=/tmp/claude-1000/-home-elfo-Documents-NRC-NRC-SU25-Spectral-Classification-Projects-SERS-Classification/e5b064dc-da05-42ac-b179-ebe8f64bcd49/scratchpad/t1pilot
until ! pgrep -f "[r]un_dft_batch" >/dev/null; do sleep 30; done
echo "batch done $(date +%T)"
OMP_NUM_THREADS=4 PYTHONWARNINGS=ignore $S/venv/bin/python -I $S/scripts/fix_imag.py $S/results/dft/new_acetone.npz $S/results/dft/new_dimethoxypropane.npz 2>&1 | grep -a -E "imaginary|lowest|Traceback|Error"
echo "fix done $(date +%T)"
$S/venv/bin/python -I $S/scripts/extend_library.py $S/results/siamese_data.npz $S/results/pregate/qm9s_rdkit.csv $S/results/queries $S/results/dft $S/results/siamese_data_ext.npz
$S/scripts/run_ext.sh $S > $S/results/run_ext.log 2>&1
echo "rerun done $(date +%T)"
/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/.venv/bin/python -I $S/scripts/analyze_ext.py $S/results/siamese_ext $S/results/siamese_data_ext.npz | tee $S/results/siamese_ext/summary.txt
