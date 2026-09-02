#!/bin/bash
# Clean current-build baseline (Stage5+Stage3 fixes, no E31/E32/E33) at VDA, same config as
# tools/cw_rmin_5seed.sh, for an apples-to-apples comparison with e31_32_33_vda_5seed.sh --
# report 010 SS0.6's published 94.6s baseline predates the Stage5 fix, so it isn't valid for
# isolating E31/E32/E33's wall-clock effect.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/baseline_current_vda_5seed
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
