#!/bin/bash
# Sweep the ruin-walk multiplier at VDA. Motivation: report 010 SS2.2 found a quality
# ceiling (converged, not time-starved) while report 009 measured FILO2 ruining ~23
# customers/iteration against our ~9-13 -- a ceiling plus an undersized ruin is the
# signature of too little diversification. Uses the better 46/31 Stage2/Stage5 split found
# by the split sweep (baseline at mult=1.0: 21,761,069).
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/ruin_sweep_vda
mkdir -p "$OUT"
for M in "$@"; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed 1 -p 2 --construction cw --routemin-k 1000 --cw-neighbors 100 \
        --routemin-iters 2000 --stage2-ms 46000 --stage3-ms 1000 --stage5-ms 31000 \
        --ruin-mult "$M" \
        --out "$OUT/sol_m${M}.txt" --log "$OUT/log_m${M}.txt" > "$OUT/stdout_m${M}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_m${M}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_m${M}.txt" | awk '{print $3}')
    echo "mult=$M: cost=$c wall_ms=$t"
done
