#!/bin/bash
# Rerun fixed baseline, fitted calibration and Siamese fine-tuning on the extended (29-compound) data.
set -e
S="$1"; P=/home/elfo/Documents/NRC/NRC_SU25_Spectral_Classification/Projects/SERS_Classification/.venv/bin/python
X=$S/results/siamese_data_ext.npz; O=$S/results/siamese_ext
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $O
$P -I $S/scripts/fixed_eval.py $X $O/fixed_baseline.json
$P -I $S/scripts/fitted_control.py $X $O/fitted_control.json
FT="$P -I $S/scripts/siamese.py finetune --data $X --depth 2 --width 32 --steps 300 --lr 1e-4"
run() { d=$O/$1; mkdir -p $d; ln -sf $S/results/siamese/ckpt_d2_w32_s1.pt $d/; shift; echo "== $d $*"; $FT --out $d "$@"; }
run seed0 --ntrain 5 10 15 20 28 --repeats 3 --seed 0
run seed1 --ntrain 99 --seed 1
run seed2 --ntrain 99 --seed 2
for sd in 0 1 2; do run byclass --ntrain 99 --by_class 1 --seed $sd; done
for sd in 1 2 3; do run shuffle --ntrain 99 --shuffle 1 --seed $sd; done
run shuffle --ntrain 99 --by_class 1 --shuffle 1 --seed 1
echo "== done"
