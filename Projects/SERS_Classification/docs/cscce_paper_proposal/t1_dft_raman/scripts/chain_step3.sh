#!/bin/bash
# After the step 3 DFT batch (two streams, logs dft_step3_batch_A/B.log): build, score once
# (run_v3.sh), copy everything into the project.
SP="$1"; S=$SP/t1pilot
until grep -q "^=== stream done" $S/results/dft_step3_batch_A.log 2>/dev/null && \
      grep -q "^=== stream done" $S/results/dft_step3_batch_B.log 2>/dev/null; do sleep 60; done
echo "DFT batch done $(date '+%F %T')"
$S/scripts/run_v3.sh $S $SP > $S/results/run_v3.log 2>&1 && echo "scoring done $(date '+%F %T')" || echo "scoring FAILED $(date '+%F %T')"
$S/scripts/sync_to_project.sh $SP
cp $S/results/run_v3.log $S/results/chain_step3.log /home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/docs/cscce_paper_proposal/t1_dft_raman/results/summaries/ 2>/dev/null
