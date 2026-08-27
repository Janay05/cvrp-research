#!/bin/bash
# With the WSL ceiling raised 7.6 GB -> 9.7 GB (+4 GB swap), retry Lazio with a wider
# candidate list. k was pinned at 300 purely by memory: at VDA, widening 300 -> 1000 was
# worth a further ~1% on construction quality, so there is likely headroom-limited gain
# left here too.
#
# Budgets identical to the committed cw_rmin run so this is comparable:
#   k=300 result: 3,171,997,628 / 40,267 routes / 284.9 s / 7.43 GB peak.
# Swap now exists specifically so an overrun degrades into paging rather than the abrupt
# VM loss seen twice earlier -- but watch the peak RSS line regardless.
set -e
cd /mnt/c/internship/iitm/cvrp
K=${1:-500}
OUT=results/bench/cw_lazio_widek
mkdir -p "$OUT"
/usr/bin/time -v ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp --seed 1 -p 4 \
    --construction cw --routemin-k "$K" --cw-neighbors 100 --routemin-iters 2000 \
    --stage2-ms 55000 --stage3-ms 25000 --stage5-ms 55000 \
    --out "$OUT/sol_k${K}.txt" --log "$OUT/log_k${K}.txt" 2>&1 \
    | grep -iE "^Final cost|^Setup|Total time|Maximum resident"
head -2 "$OUT/sol_k${K}.txt" | tail -1
grep -i routemin "$OUT/log_k${K}.txt" || true
