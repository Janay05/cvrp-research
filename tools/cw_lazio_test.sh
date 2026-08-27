#!/bin/bash
# Does Clarke & Wright hold up at Lazio (~1M customers)?
# Construction-only (stage budgets ~0) to isolate CW quality from the search that follows.
#
# Memory is the constraint, as it was for ROUTEMIN: the wide list costs n*k*4 bytes, so the
# k=1000 that made CW beat FILO2's construction at VDA would be ~4 GB here on top of a
# ~3.5 GB baseline -- infeasible. k=300 already measured at 7.11 GB peak / 177 s setup, i.e.
# 94% of this machine's WSL ceiling.
#
# Reference: our MST 3,208,434,488 / 40,513 routes; FILO2 CW 3,177,770,000 / 40,252.
set -e
cd /mnt/c/internship/iitm/cvrp
K=${1:-300}
/usr/bin/time -v ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp \
    --seed 1 -p 4 --construction cw --routemin-k "$K" --cw-neighbors 100 \
    --stage2-ms 1 --stage3-ms 1 --stage5-ms 1 \
    --out "/tmp/cwlaz_${K}.txt" --log "/tmp/cwlaz_${K}.log" 2>&1 \
    | grep -iE "Cost BEFORE Stage 3|Setup|Total time|Maximum resident"
head -2 "/tmp/cwlaz_${K}.txt" | tail -1
