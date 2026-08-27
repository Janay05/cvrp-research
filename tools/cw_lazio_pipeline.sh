#!/bin/bash
# Full Lazio pipeline with Clarke & Wright, now that parallelising symmetrize dropped the
# wide-list cost from 188 s to 33 s of setup (less than the 38 s the OLD baseline spent with
# no wide list at all).
# Budgets sized so total wall clock lands near the ~315 s the Lazio baseline used.
# Baselines: ours (MST) 3,182,981,663 / 40,431 routes @315 s; FILO2 3,159,235,192 / 40,111.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/cw_lazio_pipeline
mkdir -p "$OUT"
run() {
    local tag=$1; shift
    echo "=== $tag ==="
    /usr/bin/time -v ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp --seed 1 -p 4 \
        "$@" --out "$OUT/sol_${tag}.txt" --log "$OUT/log_${tag}.txt" 2>&1 \
        | grep -iE "^Final cost|Total time|Setup|Maximum resident"
    head -2 "$OUT/sol_${tag}.txt" | tail -1
    grep -i routemin "$OUT/log_${tag}.txt" || true
    echo
}
run cw_only --construction cw --routemin-k 300 --cw-neighbors 100 \
    --stage2-ms 105000 --stage3-ms 40000 --stage5-ms 130000
run cw_rmin --construction cw --routemin-k 300 --cw-neighbors 100 --routemin-iters 2000 \
    --stage2-ms 95000 --stage3-ms 35000 --stage5-ms 115000
