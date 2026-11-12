#!/bin/bash
# Pause the two batch streams (SIGSTOP, nothing lost), run speed_test.py on vc15 at 32 threads, resume.
S="$1"; shift
PG="$*"
for g in $PG; do kill -STOP -- -$g; done
trap 'for g in $PG; do kill -CONT -- -$g; done; echo "streams resumed $(date +%T)"' EXIT
echo "streams paused $(date +%T)"
cd $S/scripts && OMP_NUM_THREADS=32 PYTHONWARNINGS=ignore $S/venv/bin/python -I speed_test.py "CC1=CC=C(C=C1)CC#N" $S/results/dft_step3/vc15.npz 432 1250 2>&1 | grep -a -v -i warn | grep -a -v "^  *$" | grep -a -E "stage|optimisation|hessian|whole|distance|frequencies|activities|lowest|Error|Traceback|assert"
