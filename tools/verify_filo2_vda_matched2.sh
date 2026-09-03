#!/bin/bash
set -e
cd /mnt/c/internship/iitm/cvrp
declare -A reported
reported[1]=21742280
reported[2]=21738205
reported[3]=21772546
reported[4]=21741231
reported[5]=21731006
for s in 1 2 3 4 5; do
    echo "--- FILO2 VDA seed $s ---"
    python3 src/verify_filo2.py data/instances/I/Valle-D-Aosta.vrp \
        "results/bench/filo2_vda_matched2/Valle-D-Aosta.vrp_seed-${s}.vrp.sol" "${reported[$s]}"
done
