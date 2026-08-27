#!/bin/bash
# Does Clarke & Wright's better starting point survive to the FINAL answer?
# Construction-only, CW k=1000 gives 22,185,233 vs MST's 22,621,853 at VDA -- but our search
# grinds ~2.8% off whatever it starts from, so a better start does not automatically mean a
# better finish (it may just converge to the same attractor).
#
# Budgets are trimmed to absorb the wide-list build (~11 s) and ROUTEMIN (~11 s) so total
# wall clock lands near the ~102 s every other VDA comparison uses.
# Baselines at ~102 s, 5-seed: MST 22,000,544; MST+ROUTEMIN 21,895,496; FILO2 21,740,517.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/cw_pipeline
mkdir -p "$OUT"
run() {
    local tag=$1; shift
    echo "=== $tag ==="
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp --seed 1 -p 2 \
        "$@" --out "$OUT/sol_${tag}.txt" --log "$OUT/log_${tag}.txt" 2>&1 \
        | grep -iE "^Final cost|Total time"
    head -2 "$OUT/sol_${tag}.txt" | tail -1
    grep -i routemin "$OUT/log_${tag}.txt" || true
    echo
}
run cw_only --construction cw --routemin-k 1000 --cw-neighbors 100 \
    --stage2-ms 35000 --stage3-ms 1000 --stage5-ms 53000
run cw_plus_rmin --construction cw --routemin-k 1000 --cw-neighbors 100 \
    --routemin-iters 2000 --stage2-ms 31000 --stage3-ms 1000 --stage5-ms 46000
