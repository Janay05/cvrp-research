#!/bin/bash
set -e
cd /mnt/c/internship/iitm/cvrp
declare -A reported
reported[1]=21744223
reported[2]=21736423
reported[3]=21757348
reported[4]=21735977
reported[5]=21733333
for s in 1 2 3 4 5; do
    echo "--- FILO2 VDA seed $s ---"
    python3 src/verify_filo2.py data/instances/I/Valle-D-Aosta.vrp \
        "results/bench/filo2_vda_matched/Valle-D-Aosta.vrp_seed-${s}.vrp.sol" "${reported[$s]}"
done
