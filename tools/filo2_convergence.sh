#!/bin/bash
# Research: FILO2's cost-vs-time convergence curve on VDA.
# The decisive question is not "who wins at 102s" but "how long does FILO2 need
# to reach the cost WE reach at 102s on 2 cores". Runs several time limits
# concurrently (FILO2 is single-threaded, so N processes = N cores).
set -e
cd /mnt/c/internship/iitm/cvrp
EXE=./baselines/filo2/build_wsl_tl/filo2
INST=data/instances/I/Valle-D-Aosta.vrp
OUTDIR=results/bench/filo2_convergence_vda
mkdir -p "$OUTDIR"

pids=()
for secs in 3 5 10 20 40 102; do
    "$EXE" "$INST" --seed 1 --optimization-seconds "$secs" \
        --outpath "$OUTDIR/t${secs}_" > "$OUTDIR/stdout_${secs}.txt" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "=== FILO2 convergence on VDA (seed 1) ==="
for secs in 3 5 10 20 40 102; do
    f=$(ls "$OUTDIR"/t${secs}_*.out 2>/dev/null | head -1)
    if [ -n "$f" ]; then
        cost=$(awk '{print $1}' "$f")
        echo "  ${secs}s: $cost"
    else
        echo "  ${secs}s: (no output)"
    fi
done
