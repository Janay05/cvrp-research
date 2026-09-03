#!/bin/bash
set -e
cd /mnt/c/internship/iitm/cvrp
for s in 1 2 3 4 5; do
    echo "--- seed $s ---"
    python3 src/verifier.py data/instances/I/Valle-D-Aosta.vrp results/bench/eject2_vda_5seed/sol_${s}.txt 2>&1 | tail -4
done
