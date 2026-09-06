#!/bin/bash
# 5-seed VDA at the re-tuned config: harder ROUTEMIN (6000 iters vs the old 2000), with the
# extra routemin time bought back out of Stage 2/Stage 5 so wall clock stays at the ~87s
# reference, plus the 46/31->38/24 rebalance and --ruin-mult 1.5.
# Baseline to beat: eject2_vda_5seed mean 21,762,742 @ 88.6s; FILO2 (89s) mean 21,745,054.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/routemin_tuned_vda_5seed
mkdir -p "$OUT"
total=0
for s in 1 2 3 4 5; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed "$s" -p 2 --construction cw --routemin-k 1000 --cw-neighbors 100 \
        --routemin-iters 6000 --stage2-ms 38000 --stage3-ms 1000 --stage5-ms 24000 \
        --ruin-mult 1.5 \
        --out "$OUT/sol_${s}.txt" --log "$OUT/log_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    r=$(grep -m1 "^Num Routes:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "seed $s: cost=$c routes=$r wall_ms=$t"
    total=$((total + c))
done
echo "mean cost: $((total / 5))"
