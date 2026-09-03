#!/bin/bash
set -e
cd /mnt/c/internship/iitm/cvrp
for s in 1 2 3; do
    echo "--- eject2 seed $s ---"
    python3 src/verifier.py data/instances/I/Lazio.vrp results/bench/eject2_lazio_3seed/sol_${s}.txt 2>&1 | tail -4
done
for s in 1 2 3; do
    echo "--- baseline seed $s ---"
    python3 src/verifier.py data/instances/I/Lazio.vrp results/bench/eject2_lazio_baseline_3seed/sol_${s}.txt 2>&1 | tail -4
done
