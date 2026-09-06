#!/bin/bash
# ROUTEMIN is the right lever at Lombardia (50k iters: routes 12,770->12,737, cost -0.269%)
# but at --routemin-k 500 (copied from Lazio) each iteration costs ~15.4ms, so 50k iters
# blows the wall-clock budget 3x over. Narrowing the routemin candidate width should make
# each iteration far cheaper, buying many more of them inside the same ~331s budget.
# Usage: lombardia_tune2.sh <routemin_k> <routemin_iters> <stage2_ms> <stage5_ms> <tag>
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/lombardia_tune
mkdir -p "$OUT"
K="$1"; IT="$2"; S2="$3"; S5="$4"; TAG="$5"
./src/build_wsl/cvrp_parallel data/instances/I/Lombardia.vrp \
    --seed 1 -p 4 --routemin-k "$K" --routemin-iters "$IT" \
    --stage2-ms "$S2" --stage3-ms 12000 --stage5-ms "$S5" --ruin-mult 1.5 \
    --out "$OUT/sol_${TAG}.txt" --log "$OUT/log_${TAG}.txt" > "$OUT/stdout_${TAG}.txt" 2>&1
c=$(grep -m1 "^Final Cost:" "$OUT/sol_${TAG}.txt" | awk '{print $3}')
r=$(grep -m1 "^Num Routes:" "$OUT/sol_${TAG}.txt" | awk '{print $3}')
t=$(grep -m1 "Total time:" "$OUT/stdout_${TAG}.txt" | awk '{print $3}')
s1=$(grep -m1 -oE 'Stage 1: [0-9.]+ ms' "$OUT/stdout_${TAG}.txt" | awk '{print $3}')
echo "k=$K iters=$IT s2=$S2 s5=$S5: cost=$c routes=$r wall_ms=$t stage1_ms=$s1"
