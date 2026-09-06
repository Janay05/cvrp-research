#!/bin/bash
# Harder ROUTEMIN gave a large cost gain at VDA (0.14%) but cost ~28s of wall clock. This
# buys that time back out of the Stage 2/Stage 5 budgets so total wall clock stays at the
# ~87s reference, making it a valid equal-time comparison. The question this answers: is
# routemin time worth more per second than local-search time?
# Usage: routemin_equaltime_vda.sh <routemin_iters> <stage2_ms> <stage5_ms> <tag>
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/routemin_equaltime_vda
mkdir -p "$OUT"
IT="$1"; S2="$2"; S5="$3"; TAG="$4"
./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
    --seed 1 -p 2 --construction cw --routemin-k 1000 --cw-neighbors 100 \
    --routemin-iters "$IT" --stage2-ms "$S2" --stage3-ms 1000 --stage5-ms "$S5" \
    --ruin-mult 1.5 \
    --out "$OUT/sol_${TAG}.txt" --log "$OUT/log_${TAG}.txt" > "$OUT/stdout_${TAG}.txt" 2>&1
c=$(grep -m1 "^Final Cost:" "$OUT/sol_${TAG}.txt" | awk '{print $3}')
r=$(grep -m1 "^Num Routes:" "$OUT/sol_${TAG}.txt" | awk '{print $3}')
t=$(grep -m1 "Total time:" "$OUT/stdout_${TAG}.txt" | awk '{print $3}')
echo "iters=$IT s2=$S2 s5=$S5: cost=$c routes=$r wall_ms=$t"
