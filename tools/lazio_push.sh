#!/bin/bash
# The 8k-iteration run landed at 3,159,961,835 in 269.4 s -- 0.023% behind FILO2's
# 3,159,235,192 but 45.6 s faster. That leaves ~45 s of headroom under FILO2's 315 s.
#
# Spending it on ROUTEMIN rather than stage time: each surplus route costs ~219,323 in depot
# legs at Lazio, and our tour quality is already 31% better than FILO2's, so route count is
# where the remaining value is. 8k -> 12k iterations costs roughly +35 s of Stage 1.
set -e
cd /mnt/c/internship/iitm/cvrp
ITERS=${1:-12000}
OUT=results/bench/lazio_push
mkdir -p "$OUT"
/usr/bin/time -v ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp --seed 1 -p 4 \
    --routemin-k 500 --routemin-iters "$ITERS" \
    --stage2-ms 45000 --stage3-ms 12000 --stage5-ms 45000 \
    --out "$OUT/sol_${ITERS}.txt" --log "$OUT/log_${ITERS}.txt" 2>&1 \
    | grep -iE "^Final cost|^Setup|^Stage|Total time|Maximum resident"
grep -i routemin "$OUT/log_${ITERS}.txt"
head -2 "$OUT/sol_${ITERS}.txt" | tail -1
