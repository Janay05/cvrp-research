#!/bin/bash
set -e
cd /mnt/c/internship/iitm/cvrp
declare -A reported
reported[1]=3166192457
reported[2]=3162957143
reported[3]=3164549287
reported[4]=3163674562
reported[5]=3165085253
reported[6]=3164147177
reported[7]=3164994602
reported[8]=3164105668
reported[9]=3165686350
reported[10]=3165235632
for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "--- FILO2 seed $s ---"
    python3 src/verify_filo2.py data/instances/I/Lazio.vrp \
        "results/bench/filo2_lazio_final_10seed/Lazio.vrp_seed-${s}.vrp.sol" "${reported[$s]}"
done
