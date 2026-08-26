#!/bin/bash
# Research: does the multi-start (best-of-N) trick help FILO2 too?
# Stage 6-C gave OUR solver best-of-12 at no wall-clock cost. But FILO2 is also a
# randomized (seeded) solver and is single-threaded, so on the same 28-core machine
# it could equally be run 12x concurrently. If best-of-12 helps FILO2 by a similar
# margin, then Stage 6-C is NOT a competitive advantage -- it's a technique both
# solvers get, and comparing our-best-of-12 vs their-single-run is not apples-to-apples.
set -e
cd /mnt/c/internship/iitm/cvrp
EXE=./baselines/filo2/build_wsl_tl/filo2
INST=data/instances/I/Valle-D-Aosta.vrp
OUTDIR=results/bench/filo2_multistart_vda
mkdir -p "$OUTDIR"
N=12
SECS=102

START=$(date +%s.%N)
pids=()
for i in $(seq 1 $N); do
    "$EXE" "$INST" --seed "$i" --optimization-seconds "$SECS" \
        --outpath "$OUTDIR/s${i}_" > "$OUTDIR/stdout_${i}.txt" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
END=$(date +%s.%N)
WALL=$(echo "$END - $START" | bc)

echo "=== FILO2 best-of-${N} on VDA, ${SECS}s each, ${WALL}s wall ==="
best=""
for i in $(seq 1 $N); do
    f=$(ls "$OUTDIR"/s${i}_*.out 2>/dev/null | head -1)
    cost=$(awk '{print $1}' "$f")
    echo "  seed $i: $cost"
    if [ -z "$best" ] || [ "$cost" -lt "$best" ]; then best="$cost"; fi
done
echo "--- FILO2 best-of-${N}: $best ---"
