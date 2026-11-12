#!/bin/bash
# Step 3 (see step3/PREREG.md): build the data file and score the fresh test set once.
set -e
S="$1"; B="$2"; P=/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/.venv/bin/python
export T1_LO=500 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VIB=$B/step2/dl/vibraclip/x/vibraclip_supplementary_data/datasets/pickle_files/experimental_dataset.pkl
X=${X:-$S/results/siamese_data_v3.npz}; O=${O:-$S/results/siamese_v3}
mkdir -p $O
echo "== build $(date '+%F %T')"
$S/venv/bin/python -I $S/scripts/build_v3.py $S/results/siamese_data_v2.npz $B/step3/test_compounds.csv $S/results/dft_step3 $VIB $X
if [ "$DRYRUN" = 1 ]; then  # pipeline check only: test spectra replaced by noise, so no result can be seen early
  $P -I -c "import numpy as np,sys; d=dict(np.load(sys.argv[1])); m=d['q_split']==3; r=np.abs(np.random.default_rng(0).normal(size=(m.sum(),d['q_x'].shape[1]))); d['q_x'][m]=r/np.linalg.norm(r,axis=1,keepdims=True); np.savez(sys.argv[1],**d)" $X
fi
ln -sf $S/results/siamese_v2/ckpt_d2_w32_s1.pt $O/
SIA="$P -I $S/scripts/siamese.py"
FT="--depth 2 --width 32 --steps 300 --lr 1e-4"
echo "== fixed and calibration $(date '+%F %T')"
$P -I $S/scripts/fixed_eval.py $X $O/fixed_baseline.json
FIT_SPLITS=transfer $P -I $S/scripts/fitted_control.py $X $O/fitted_transfer.json
FIT_SPLITS=transfer FIT_TRAIN_SPLITS=0,1,2 $P -I $S/scripts/fitted_control.py $X $O/fitted_transfer_all.json
INST_SPLITS=transfer $P -I $S/scripts/fitted_control_inst.py $X $O/fitted_inst.json
echo "== Siamese $(date '+%F %T')"
for sd in 0 1 2; do TRAIN_SPLITS=0,1,2 $SIA transfer --data $X --out $O $FT --seed $sd; done
for sd in 1 2 3; do TRAIN_SPLITS=0,1,2 $SIA transfer --data $X --out $O $FT --seed $sd --shuffle 1; done
echo "== analysis $(date '+%F %T')"
$P -I $S/scripts/analyze_v3.py $O $X | tee $O/summary.txt
