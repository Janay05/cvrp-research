#!/bin/bash
set -e
cd /mnt/c/internship/iitm/cvrp
for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "--- seed $s ---"
    python3 src/verifier.py data/instances/I/Lazio.vrp results/bench/lazio_final_10seed/sol_${s}.txt 2>&1 | tail -4
done
