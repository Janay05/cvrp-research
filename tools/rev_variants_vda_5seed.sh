#!/bin/bash
# 5-seed VDA check of the E21_rev/E22_rev/E31_rev/E32_rev/E33_rev operators, same config as
# tools/e31_32_33_vda_5seed.sh (mean 21,776,503, the "without Rev" baseline) so this isolates
# the Rev variants' own marginal effect directly -- no need to re-isolate via git stash since
# that baseline was already measured cleanly.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/rev_variants_vda_5seed
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
