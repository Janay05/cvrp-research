#!/bin/bash
# 5-seed equal-wall-clock check of the fixed ROUTEMIN path, so the improvement is judged on
# a mean rather than the single seed used while debugging.
# Budgets trimmed (stage2 32s / stage5 46s vs the usual 40s/60s) to absorb ROUTEMIN + the
# wide-neighbour-list build, landing total wall clock near the ~102 s every other
# Valle-D-Aosta comparison uses.
# Baseline to beat: results/bench/stage1_vda_p2.csv, mean 22,000,544 over seeds 1-5.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/routemin_5seed
mkdir -p "$OUT"
pids=()
for s in 1 2 3 4 5; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed "$s" -p 2 --routemin-iters 2000 --routemin-k 1000 \
        --stage2-ms 32000 --stage3-ms 1000 --stage5-ms 46000 \
        --out "$OUT/sol_${s}.txt" --log "$OUT/log_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "seed,cost,routes,total_ms"
total=0
for s in 1 2 3 4 5; do
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    r=$(grep -m1 "^Num Routes:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "$s,$c,$r,$t"
    total=$((total + c))
done
echo "mean cost: $((total / 5))"
