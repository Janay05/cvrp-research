#!/bin/bash
# Lombardia has never been tuned -- the first run used Lazio's flags verbatim, and Lombardia
# is a different instance (Q=150 vs 50, ~75 customers/route vs ~25, 12.7k routes vs 40k).
# Leading hypothesis: route count. We land on 12,770 routes against FILO2's 12,720; at
# ~106k cost per route those 50 extra routes are the same order as the whole 1.44M gap.
# So: much harder ROUTEMIN, with the time bought back out of Stage 2/Stage 5 to hold wall
# clock near the 331s reference.
# Baseline: routemin 12000 / s2 45000 / s5 45000 -> 1,350,876,414, 12,770 routes, 331.3s.
# Usage: lombardia_tune.sh <routemin_iters> <stage2_ms> <stage5_ms> <ruin_mult> <tag>
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/lombardia_tune
mkdir -p "$OUT"
IT="$1"; S2="$2"; S5="$3"; M="$4"; TAG="$5"
./src/build_wsl/cvrp_parallel data/instances/I/Lombardia.vrp \
    --seed 1 -p 4 --routemin-k 500 --routemin-iters "$IT" \
    --stage2-ms "$S2" --stage3-ms 12000 --stage5-ms "$S5" --ruin-mult "$M" \
    --out "$OUT/sol_${TAG}.txt" --log "$OUT/log_${TAG}.txt" > "$OUT/stdout_${TAG}.txt" 2>&1
c=$(grep -m1 "^Final Cost:" "$OUT/sol_${TAG}.txt" | awk '{print $3}')
r=$(grep -m1 "^Num Routes:" "$OUT/sol_${TAG}.txt" | awk '{print $3}')
t=$(grep -m1 "Total time:" "$OUT/stdout_${TAG}.txt" | awk '{print $3}')
echo "iters=$IT s2=$S2 s5=$S5 mult=$M: cost=$c routes=$r wall_ms=$t"
