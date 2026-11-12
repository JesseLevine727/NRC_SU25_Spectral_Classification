#!/bin/bash
# Step 2 (see step2/PREREG.md): primary held-out transfer test on 81 VibraCLIP compounds, then
# sensitivity (no 1064 nm correction) and the pooled 110-compound secondary analyses.
set -e
S="$1"; P=/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/.venv/bin/python
export T1_LO=500 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
X=$S/results/siamese_data_v2.npz; XN=$S/results/siamese_data_v2_nocorr.npz; O=$S/results/siamese_v2
mkdir -p $O $O/nocorr $O/pooled $O/pooled/byclass
SIA="$P -I $S/scripts/siamese.py"
FT="--depth 2 --width 32 --steps 300 --lr 1e-4"
echo "== pretrain d2w32 at 500-1800 $(date +%T)"
$SIA sweep --data $X --out $O --depths 2 --widths 32 --steps 6000 --eval_every 1000
ln -sf $O/ckpt_d2_w32_s1.pt $O/nocorr/; ln -sf $O/ckpt_d2_w32_s1.pt $O/pooled/; ln -sf $O/ckpt_d2_w32_s1.pt $O/pooled/byclass/
echo "== primary $(date +%T)"
$P -I $S/scripts/fixed_eval.py $X $O/fixed_baseline.json
FIT_SPLITS=transfer $P -I $S/scripts/fitted_control.py $X $O/fitted_transfer.json
for sd in 0 1 2; do $SIA transfer --data $X --out $O $FT --seed $sd; done
for sd in 1 2 3; do $SIA transfer --data $X --out $O $FT --seed $sd --shuffle 1; done
echo "== sensitivity, no laser correction $(date +%T)"
$P -I $S/scripts/fixed_eval.py $XN $O/nocorr/fixed_baseline.json
FIT_SPLITS=transfer $P -I $S/scripts/fitted_control.py $XN $O/nocorr/fitted_transfer.json
for sd in 0 1 2; do $SIA transfer --data $XN --out $O/nocorr $FT --seed $sd; done
echo "== pooled 110 compounds $(date +%T)"
FIT_SPLITS=compound,class $P -I $S/scripts/fitted_control.py $X $O/pooled/fitted_control.json
$SIA finetune --data $X --out $O/pooled $FT --ntrain 5 10 20 40 80 109 --repeats 3 --seed 0
for sd in 0 1 2; do $SIA finetune --data $X --out $O/pooled/byclass $FT --ntrain 999 --by_class 1 --seed $sd; done
$SIA finetune --data $X --out $O/pooled $FT --ntrain 999 --shuffle 1 --seed 1
echo "== done $(date +%T)"
