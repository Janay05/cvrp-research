#!/bin/bash
set -e
cd /mnt/c/internship/iitm/cvrp
for s in 6 7 8 9 10 11 12 13 14 15; do
    echo -n "seed $s: "
    python3 src/verifier.py data/instances/I/Valle-D-Aosta.vrp results/bench/routemin_tuned_vda_5seed/sol_${s}.txt 2>&1 | grep -E "SUCCESS|ERROR|matches reported Final Cost" | tr '\n' ' '
    echo
done
