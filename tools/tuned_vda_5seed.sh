#!/bin/bash
# 5-seed VDA run of the tuned config (Stage2/Stage5 split 46/31 instead of 31/46, plus
# --ruin-mult), against the eject2_vda_5seed baseline (31/46, mult 1.0, mean 21,762,742).
# Multi-seed because the single-seed sweeps that suggested these values had a spread
# comparable to VDA's own per-seed noise band -- not trustworthy on one seed.
# Usage: tuned_vda_5seed.sh <ruin_mult> <outdir_suffix>
set -e
cd /mnt/c/internship/iitm/cvrp
M="$1"
OUT="results/bench/tuned_vda_5seed_$2"
mkdir -p "$OUT"
total=0
for s in 1 2 3 4 5; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed "$s" -p 2 --construction cw --routemin-k 1000 --cw-neighbors 100 \
        --routemin-iters 2000 --stage2-ms 46000 --stage3-ms 1000 --stage5-ms 31000 \
        --ruin-mult "$M" \
        --out "$OUT/sol_${s}.txt" --log "$OUT/log_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "seed $s: cost=$c wall_ms=$t"
    total=$((total + c))
done
echo "mean cost (mult=$M): $((total / 5))"
