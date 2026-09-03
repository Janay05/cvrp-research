#!/bin/bash
# Re-matched to our new wall clock after adding the depth-2 ejection chain operator
# (~88.6s mean, up from ~86s in SS0.17). Same invocation pattern as filo2_vda_matched.sh.
set -e
cd /mnt/c/internship/iitm/cvrp
EXE=./baselines/filo2/build_wsl_tl/filo2
INST=data/instances/I/Valle-D-Aosta.vrp
OUTDIR=results/bench/filo2_vda_matched2
mkdir -p "$OUTDIR"
SECS=89
for s in 1 2 3 4 5; do
    "$EXE" "$INST" --seed "$s" --optimization-seconds "$SECS" --outpath "$OUTDIR/" \
        > "$OUTDIR/stdout_${s}.txt" 2>&1
    f=$(ls "$OUTDIR"/*seed-${s}.out 2>/dev/null | head -1)
    echo "seed $s: $(cat "$f")"
done
