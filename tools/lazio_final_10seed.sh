#!/bin/bash
# Definitive 10-seed Lazio comparison on the FINAL build (Stage5 fix + Stage3 fix + E21/E22/
# E31/E32/E33 + their Rev variants). Supersedes SS0.14's 3-seed check (pre-Rev) and
# lazio_more_seeds/ (pre-Rev). Same config throughout this report's Lazio comparisons.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/lazio_final_10seed
mkdir -p "$OUT"
for s in 1 2 3 4 5 6 7 8 9 10; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp --seed "$s" -p 4 \
        --routemin-k 500 --routemin-iters 12000 --stage2-ms 45000 --stage3-ms 12000 --stage5-ms 45000 \
        --out "$OUT/sol_${s}.txt" --log "$OUT/log_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "seed $s: cost=$c wall_ms=$t"
done
