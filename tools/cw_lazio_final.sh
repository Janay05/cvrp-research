#!/bin/bash
# Equal-time Lazio comparison, done properly.
#
# The previous attempt overran (466 s vs a 315 s target) because I sized budgets as if they
# controlled total runtime. They do not: at Lazio, Stage 3 and Stage 4/5 carry large
# UNBUDGETED overheads -- the MST baseline run shows stage3 budgeted 30 s taking 90 s, and
# stage5 budgeted 60 s taking 118 s. Stage 1 (CW construction) is only 8.3 s, so it is not
# the problem.
#
# So the honest equal-time test is to hold the budgets IDENTICAL to the baseline run and
# change only --construction. Baseline: -p 4 --stage2-ms 60000 --stage3-ms 30000
# --stage5-ms 60000 -> 3,182,981,663 / 40,431 routes in 314.5 s.
# FILO2 at equal wall clock: 3,159,235,192 / 40,111 routes in 315 s.
#
# cw_rmin trims budgets slightly to pay for ROUTEMIN (~22 s of extra Stage 1 at Lazio).
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/cw_lazio_final
mkdir -p "$OUT"
run() {
    local tag=$1; shift
    echo "=== $tag ==="
    /usr/bin/time -v ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp --seed 1 -p 4 \
        "$@" --out "$OUT/sol_${tag}.txt" --log "$OUT/log_${tag}.txt" 2>&1 \
        | grep -iE "^Final cost|^Setup|^Stage|Total time|Maximum resident"
    head -2 "$OUT/sol_${tag}.txt" | tail -1
    grep -i routemin "$OUT/log_${tag}.txt" || true
    echo
}
# Identical budgets to the published MST baseline; only --construction differs.
run cw_only --construction cw --routemin-k 300 --cw-neighbors 100 \
    --stage2-ms 60000 --stage3-ms 30000 --stage5-ms 60000
# Same, minus ~22 s of budget to pay for ROUTEMIN.
run cw_rmin --construction cw --routemin-k 300 --cw-neighbors 100 --routemin-iters 2000 \
    --stage2-ms 55000 --stage3-ms 25000 --stage5-ms 55000
