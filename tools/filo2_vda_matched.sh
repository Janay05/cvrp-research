#!/bin/bash
# Corrects the same stale-budget issue found and fixed at Lazio (report 010 SS0.16), now for
# VDA: the published FILO2 VDA baseline (21,740,517) was run at --optimization-seconds 102,
# but our current build (E31/E32/E33 + Rev variants) runs VDA in ~86s, not 102s. Re-running
# FILO2 at our actual current wall clock for a genuine apples-to-apples comparison, same
# invocation pattern as tools/filo2_multistart.sh, sequential (matches how our own 5-seed VDA
# benchmarks were run).
set -e
cd /mnt/c/internship/iitm/cvrp
EXE=./baselines/filo2/build_wsl_tl/filo2
INST=data/instances/I/Valle-D-Aosta.vrp
OUTDIR=results/bench/filo2_vda_matched
mkdir -p "$OUTDIR"
SECS=86
for s in 1 2 3 4 5; do
    "$EXE" "$INST" --seed "$s" --optimization-seconds "$SECS" --outpath "$OUTDIR/" \
        > "$OUTDIR/stdout_${s}.txt" 2>&1
    f=$(ls "$OUTDIR"/*seed-${s}.out 2>/dev/null | head -1)
    echo "seed $s: $(cat "$f")"
done
