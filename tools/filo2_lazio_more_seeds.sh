#!/bin/bash
# FILO2 at Lazio, seeds 4-10, matched to our CURRENT actual wall clock (SS0.14 mean 219.1s,
# rounded to 219s) -- a stricter equal-time comparison than seeds 1-3 got (those used
# 292-315s, matching our OLD pre-fix wall clock, so FILO2 there had MORE time than we now
# take and still lost -- this makes the seeds 4-10 comparison the fairer, tighter test).
# Run only after our own solver's seeds 4-10 have fully exited -- this VM cannot hold both
# processes' peak memory at once (ours ~9.4GB, FILO2 ~7GB, VM total 9.7GB).
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/filo2_lazio_more_seeds
mkdir -p "$OUT"
for s in 4 5 6 7 8 9 10; do
    ./baselines/filo2/build_wsl_tl/filo2 data/instances/I/Lazio.vrp \
        --seed "$s" --optimization-seconds 219 --outpath "$OUT/" \
        > "$OUT/stdout_${s}.txt" 2>&1
    f=$(ls "$OUT"/*seed-${s}.out 2>/dev/null | head -1)
    echo "seed $s: $(cat "$f")"
done
