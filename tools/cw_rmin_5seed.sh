#!/bin/bash
# 5-seed check of Clarke & Wright + ROUTEMIN, the combination that on seed 1 gave
# 21,779,452 in 92.3 s -- better AND faster than every prior configuration.
# Run sequentially, not concurrently: these are time-budgeted stages, so co-running 5
# instances steals wall clock from each and distorts both cost and the timing figure.
# Baselines at ~102 s, 5-seed: MST 22,000,544; MST+ROUTEMIN 21,895,496; FILO2 21,740,517.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/cw_rmin_5seed
mkdir -p "$OUT"
echo "seed,cost,routes,total_ms"
total=0
for s in 1 2 3 4 5; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed "$s" -p 2 --construction cw --routemin-k 1000 --cw-neighbors 100 \
        --routemin-iters 2000 --stage2-ms 31000 --stage3-ms 1000 --stage5-ms 46000 \
        --out "$OUT/sol_${s}.txt" --log "$OUT/log_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    r=$(grep -m1 "^Num Routes:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "$s,$c,$r,$t"
    total=$((total + c))
done
echo "mean cost: $((total / 5))"
