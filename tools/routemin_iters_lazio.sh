#!/bin/bash
# Is ROUTEMIN simply under-converged at Lazio?
#
# The iteration count is a flat 2000 per chunk regardless of instance size, but each
# iteration destroys exactly TWO routes -- so what matters is iterations per route:
#   VDA   2000 iters / ~405 routes per chunk  = ~4.9 passes per route
#   Lazio 2000 iters / ~10,100 routes per chunk = ~0.20 passes per route  (25x less)
# So Lazio's ROUTEMIN may be stopping long before it has even looked at most routes once,
# which would explain why we sit 123 routes above FILO2 while our tour quality is 31% better.
#
# Construction + ROUTEMIN only (stage budgets ~0) to isolate the route-count response.
# At k=500 with 2000 iters the observed effect was 40,300 -> 40,215 live routes.
set -e
cd /mnt/c/internship/iitm/cvrp
ITERS=${1:-10000}
OUT=results/bench/routemin_iters_lazio
mkdir -p "$OUT"
/usr/bin/time -v ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp --seed 1 -p 4 \
    --routemin-k 500 --routemin-iters "$ITERS" \
    --stage2-ms 1 --stage3-ms 1 --stage5-ms 1 \
    --out "$OUT/sol_${ITERS}.txt" --log "$OUT/log_${ITERS}.txt" 2>&1 \
    | grep -iE "Cost BEFORE Stage 3|^Setup|Total time|Maximum resident"
grep -i routemin "$OUT/log_${ITERS}.txt"
head -2 "$OUT/sol_${ITERS}.txt" | tail -1
