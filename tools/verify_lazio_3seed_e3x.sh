#!/bin/bash
set -e
cd /mnt/c/internship/iitm/cvrp
for s in 1 2 3; do
    echo "--- seed $s ---"
    python3 src/verifier.py data/instances/I/Lazio.vrp results/bench/lazio_3seed_with_e3x/sol_${s}.txt 2>&1 | tail -4
done
