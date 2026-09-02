#!/bin/bash
# 5-seed VDA check of E31/E32/E33 (3-segment exchange family, report 009's "remaining for a
# fuller T5.2"), same config as tools/cw_rmin_5seed.sh (the current CW+ROUTEMIN baseline,
# mean 21,791,054 @ 94.6s, report 010 SS0.6) so the comparison is apples-to-apples.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/e31_32_33_vda_5seed
mkdir -p "$OUT"
echo "seed,cost,routes,total_ms"
total=0
for s in 1 2 3 4 5; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed "$s" -p 2 --construction cw --routemin-k 1000 --cw-neighbors 100 \
        --routemin-iters 2000 --stage2-ms 31000 --stage3-ms 1000 --stage5-ms 46000 \
        --out "$OUT/sol_${s}.txt" --log "$OUT/log_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    r=$(grep -m1 "^Num Routes:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "$s,$c,$r,$t"
    total=$((total + c))
done
echo "mean cost: $((total / 5))"
