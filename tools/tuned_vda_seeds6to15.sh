#!/bin/bash
# Extend the tuned-config VDA comparison to seeds 6-15. The 5-seed result (mean favours us
# by 0.0255%) had t = -0.78 with sign flips across seeds -- a tie leaning our way, not an
# established win. More seeds is the only way to tell which it actually is.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/routemin_tuned_vda_5seed
mkdir -p "$OUT"
for s in 6 7 8 9 10 11 12 13 14 15; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed "$s" -p 2 --construction cw --routemin-k 1000 --cw-neighbors 100 \
        --routemin-iters 6000 --stage2-ms 38000 --stage3-ms 1000 --stage5-ms 24000 \
        --ruin-mult 1.5 \
        --out "$OUT/sol_${s}.txt" --log "$OUT/log_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "seed $s: cost=$c wall_ms=$t"
done
