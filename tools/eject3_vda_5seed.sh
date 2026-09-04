#!/bin/bash
# 5-seed VDA check of the depth-3 ejection chain extension, same config as every other VDA
# 5-seed benchmark this session, isolated against the "eject2 only" baseline
# (eject2_vda_5seed, mean 21,762,742).
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/eject3_vda_5seed
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
