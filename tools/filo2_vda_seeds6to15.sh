#!/bin/bash
# FILO2 on VDA seeds 6-15 at 86s, matching our tuned config's actual mean wall clock
# (86.4s). Seeds 1-5 reuse the existing 89s runs (filo2_vda_matched2) -- FILO2's VDA
# time-sensitivity over this range is negligible (SS0.17: 86s vs 102s barely differed),
# and giving it the extra 2.6s there is conservative, i.e. favourable to FILO2.
set -e
cd /mnt/c/internship/iitm/cvrp
EXE=./baselines/filo2/build_wsl_tl/filo2
INST=data/instances/I/Valle-D-Aosta.vrp
OUTDIR=results/bench/filo2_vda_seeds6to15
mkdir -p "$OUTDIR"
for s in 6 7 8 9 10 11 12 13 14 15; do
    "$EXE" "$INST" --seed "$s" --optimization-seconds 86 --outpath "$OUTDIR/" \
        > "$OUTDIR/stdout_${s}.txt" 2>&1
    f=$(ls "$OUTDIR"/*seed-${s}.out 2>/dev/null | head -1)
    echo "seed $s: $(cat "$f")"
done
