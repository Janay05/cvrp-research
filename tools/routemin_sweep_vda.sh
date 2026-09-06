#!/bin/bash
# Does harder route minimization close the VDA gap? We land on 801-803 routes, FILO2 on
# exactly 800; at ~27k cost per route, 1-3 extra depot round-trips is the same order as our
# entire remaining ~14k gap. Report 009's T3 found ROUTEMIN net-negative but explicitly
# flagged it as possibly dependent on local-search richness, which has improved a lot since.
# Usage: routemin_sweep_vda.sh <iters...>
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/routemin_sweep_vda
mkdir -p "$OUT"
for IT in "$@"; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed 1 -p 2 --construction cw --routemin-k 1000 --cw-neighbors 100 \
        --routemin-iters "$IT" --stage2-ms 46000 --stage3-ms 1000 --stage5-ms 31000 \
        --ruin-mult 1.5 \
        --out "$OUT/sol_${IT}.txt" --log "$OUT/log_${IT}.txt" > "$OUT/stdout_${IT}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${IT}.txt" | awk '{print $3}')
    r=$(grep -m1 "^Num Routes:" "$OUT/sol_${IT}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${IT}.txt" | awk '{print $3}')
    echo "routemin-iters=$IT: cost=$c routes=$r wall_ms=$t"
done
