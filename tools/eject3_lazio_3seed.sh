#!/bin/bash
# 3-seed Lazio check of the depth-3 ejection chain extension.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/eject3_lazio_3seed
mkdir -p "$OUT"
for s in 1 2 3; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp --seed "$s" -p 4 \
        --routemin-k 500 --routemin-iters 12000 --stage2-ms 45000 --stage3-ms 12000 --stage5-ms 45000 \
        --out "$OUT/sol_${s}.txt" --log "$OUT/log_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "seed $s: cost=$c wall_ms=$t"
done
