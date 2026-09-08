#!/bin/bash
# Regression check: the swapstar_cap change applies to ROUTEMIN's local_search at EVERY
# instance, not just Lombardia (where it was motivated). VDA also runs ROUTEMIN, so this
# confirms no regression there. Same tuned config as routemin_tuned_vda_5seed.
# Baseline to match or beat (unlimited SWAP* in routemin, seeds 1-5): 21,739,508.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/swapstarcap_vda_5seed
mkdir -p "$OUT"
total=0
for s in 1 2 3 4 5; do
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed "$s" -p 2 --construction cw --routemin-k 1000 --cw-neighbors 100 \
        --routemin-iters 6000 --stage2-ms 38000 --stage3-ms 1000 --stage5-ms 24000 \
        --ruin-mult 1.5 \
        --out "$OUT/sol_${s}.txt" --log "$OUT/log_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    r=$(grep -m1 "^Num Routes:" "$OUT/sol_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "seed $s: cost=$c routes=$r wall_ms=$t"
    total=$((total + c))
done
echo "mean cost: $((total / 5))"
