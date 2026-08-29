#!/bin/bash
# Multi-seed confirmation of the Lazio result, for BOTH solvers at equal wall clock.
#
# Necessary, not optional: seed 1 gave ours 3,158,865,362 @291.7s vs FILO2 3,159,235,192
# @315s -- a 0.012% margin. Report 008 measured Lazio seed spread at ~0.04% (ours) and
# ~0.065% (FILO2), so a single-seed margin that small is inside the noise and establishes
# nothing on its own.
#
# Runs sequentially: our config peaks at ~9.4 GB and FILO2 at ~7 GB, so they cannot share
# this 9.7 GB VM.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/lazio_multiseed_final
mkdir -p "$OUT/filo2"

echo "=== ours (CW + ROUTEMIN 12k, k=500) ==="
for s in 2 3; do
    /usr/bin/time -f "%e s  %M KB" ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp \
        --seed "$s" -p 4 --routemin-k 500 --routemin-iters 12000 \
        --stage2-ms 45000 --stage3-ms 12000 --stage5-ms 45000 \
        --out "$OUT/ours_${s}.txt" --log "$OUT/ourslog_${s}.txt" > "$OUT/stdout_${s}.txt" 2>&1
    c=$(grep -m1 "^Final Cost:" "$OUT/ours_${s}.txt" | awk '{print $3}')
    r=$(grep -m1 "^Num Routes:" "$OUT/ours_${s}.txt" | awk '{print $3}')
    t=$(grep -m1 "Total time:" "$OUT/stdout_${s}.txt" | awk '{print $3}')
    echo "  seed $s: cost=$c routes=$r ms=$t"
done

echo "=== FILO2 at our wall clock (--optimization-seconds 292) ==="
for s in 2 3; do
    ./baselines/filo2/build_wsl_tl/filo2 data/instances/I/Lazio.vrp \
        --seed "$s" --optimization-seconds 292 --outpath "$OUT/filo2/" \
        > "$OUT/filo2/stdout_${s}.txt" 2>&1
    f=$(ls "$OUT"/filo2/*seed-${s}.out 2>/dev/null | head -1)
    echo "  seed $s: $(cat "$f")"
done
