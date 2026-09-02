#!/bin/bash
# Extends SS0.14's 3-seed Lazio check to seeds 4-10, same config, to see whether the
# all-seeds-beat-FILO2 pattern (SS0.14: -0.0104% to -0.0163%, 3-for-3) holds on a bigger
# sample or washes out into noise (established seed spread ~0.04-0.065%).
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/lazio_more_seeds
mkdir -p "$OUT"
for s in 4 5 6 7 8 9 10; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp --seed "$s" -p 4 \
        --routemin-k 500 --routemin-iters 12000 --stage2-ms 45000 --stage3-ms 12000 --stage5-ms 45000 \
        --out "$OUT/sol_${s}.txt" --log "$OUT/log_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "seed $s: cost=$c wall_ms=$t"
done
