#!/bin/bash
# Sweep Stage 4's route-dissolution threshold at VDA, on top of the re-tuned ROUTEMIN config.
# Target: our consistent 801 routes vs FILO2's 800.
# Reference (frac=0.2, the historical default), seed 1: cost 21,726,943, routes 801.
# Usage: dissolve_sweep_vda.sh <frac...>
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/dissolve_sweep_vda
mkdir -p "$OUT"
for F in "$@"; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed 1 -p 2 --construction cw --routemin-k 1000 --cw-neighbors 100 \
        --routemin-iters 6000 --stage2-ms 38000 --stage3-ms 1000 --stage5-ms 24000 \
        --ruin-mult 1.5 --stage4-dissolve-frac "$F" \
        --out "$OUT/sol_${F}.txt" --log "$OUT/log_${F}.txt" > "$OUT/stdout_${F}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${F}.txt" | awk '{print $3}')
    r=$(grep -m1 "^Num Routes:" "$OUT/sol_${F}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${F}.txt" | awk '{print $3}')
    echo "dissolve-frac=$F: cost=$c routes=$r wall_ms=$t"
done
