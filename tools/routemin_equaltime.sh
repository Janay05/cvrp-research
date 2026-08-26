#!/bin/bash
# Equal-wall-clock test of the fixed ROUTEMIN path.
# ROUTEMIN + the wide neighbour list cost ~23 s (k=1000) / ~14 s (k=300) on top of the
# stage budgets, so the budgets are trimmed to land the TOTAL near the ~102 s all previous
# Valle-D-Aosta comparisons used. Anything else would not be an apples-to-apples number.
# Reference at ~102 s: our all-time best 21,923,585 (no routemin); FILO2 21,738,409.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/routemin_equaltime
mkdir -p "$OUT"
run() {
    local tag=$1 k=$2 s2=$3 s5=$4
    echo "=== $tag (k=$k, stage2=${s2}ms, stage5=${s5}ms) ==="
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed 1 -p 2 --routemin-iters 2000 --routemin-k "$k" \
        --stage2-ms "$s2" --stage3-ms 1000 --stage5-ms "$s5" \
        --out "$OUT/sol_${tag}.txt" --log "$OUT/log_${tag}.txt" 2>&1 \
        | grep -iE "^Final cost|^Total time"
    grep -i routemin "$OUT/log_${tag}.txt" || true
    head -2 "$OUT/sol_${tag}.txt" | tail -1
    echo
}
run k1000 1000 32000 46000
run k300  300  35000 52000
