#!/bin/bash
# Clean, contemporaneous no-eject2 baseline for the exact same 3 seeds, run immediately
# before/after the eject2 version to isolate its true marginal effect from construction-time
# variance (established to swing ~85-105s+ run to run at Lazio scale, unrelated to eject2).
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/eject2_lazio_baseline_3seed
mkdir -p "$OUT"
for s in 1 2 3; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp --seed "$s" -p 4 \
        --routemin-k 500 --routemin-iters 12000 --stage2-ms 45000 --stage3-ms 12000 --stage5-ms 45000 \
        --out "$OUT/sol_${s}.txt" --log "$OUT/log_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "seed $s: cost=$c wall_ms=$t"
done
