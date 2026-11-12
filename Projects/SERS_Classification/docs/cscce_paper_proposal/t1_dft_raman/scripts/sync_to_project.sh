#!/bin/bash
# Copy the T1 scripts, predeclarations, our own DFT results and text summaries into the project.
# Not copied: QM9S and Flanagan downloads, data files (.npz with spectra), and anything holding
# VibraCLIP spectra (commercial origin, test use only).
SP="$1"; S=$SP/t1pilot
D=/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/docs/cscce_paper_proposal/t1_dft_raman
mkdir -p $D/scripts $D/step2 $D/step3 $D/results/dft $D/results/dft_step3 $D/results/summaries
rsync -a --exclude __pycache__ --exclude '*_backup.py' $S/scripts/ $D/scripts/
rsync -a $SP/step2/scripts/ --exclude __pycache__ $D/step2/scripts/
cp $SP/step2/PREREG.md $SP/step2/PREREG.sha256 $SP/step2/out/vibraclip_triage.csv $D/step2/
cp $SP/step3/PREREG.md $SP/step3/PREREG.sha256 $SP/step3/test_compounds.csv $D/step3/
rsync -a $S/results/dft/ $D/results/dft/
rsync -a $S/results/dft_step3/ $D/results/dft_step3/
for f in siamese_ext/summary.txt siamese_v2/summary_secondary.txt siamese_v2/summary.txt siamese_v3/summary.txt run_v2.log run_ext.log dft_step3_batch_run1.log dft_step3_batch_A.log dft_step3_batch_B.log capsim_step2/summary.txt capsim_step2/diagnostics.txt capsim_check.log; do
  [ -f $S/results/$f ] && cp $S/results/$f $D/results/summaries/$(echo $f | tr / _)
done
echo "synced to $D"
