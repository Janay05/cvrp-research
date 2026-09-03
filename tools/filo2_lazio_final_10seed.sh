#!/bin/bash
# FILO2 at Lazio, seeds 1-10, matched to our current wall clock. Companion to
# lazio_final_10seed.sh -- run only after that script's cvrp_parallel process has fully
# exited (this VM cannot hold both processes' peak memory at once).
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/filo2_lazio_final_10seed
mkdir -p "$OUT"
for s in 1 2 3 4 5 6 7 8 9 10; do
    ./baselines/filo2/build_wsl_tl/filo2 data/instances/I/Lazio.vrp \
        --seed "$s" --optimization-seconds 220 --outpath "$OUT/" \
        > "$OUT/stdout_${s}.txt" 2>&1
    f=$(ls "$OUT"/*seed-${s}.out 2>/dev/null | head -1)
    echo "seed $s: $(cat "$f")"
done
