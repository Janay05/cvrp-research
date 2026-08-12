#!/usr/bin/env bash
# Runs the WSL-built filo2 binary over a set of seeds against one instance, timing
# each run externally (bash's TIMEFORMAT, wall clock) and reading filo2's own
# reported cost/elapsed-seconds from its <name>_seed-<n>.out file. Used for
# docs/reports/008_verified_linux_benchmarking.md's verified-flags Linux comparison.
#
# Usage: run_filo2_wsl.sh <instance.vrp> <outpath> <seed1> [seed2 ...]
set -euo pipefail

INSTANCE="$1"
OUTPATH="$2"
shift 2
SEEDS=("$@")

FILO2_BIN="/mnt/c/internship/iitm/cvrp/baselines/filo2/build_wsl/filo2"
mkdir -p "$OUTPATH"

CSV="$OUTPATH/timing.csv"
echo "seed,wall_s,filo2_cost,filo2_elapsed_s" > "$CSV"

for SEED in "${SEEDS[@]}"; do
    echo "=== filo2 seed=$SEED ==="
    START=$(date +%s.%N)
    "$FILO2_BIN" "$INSTANCE" --seed "$SEED" --outpath "$OUTPATH/" > "$OUTPATH/seed${SEED}.stdout.log" 2>&1
    END=$(date +%s.%N)
    WALL=$(echo "$END - $START" | bc)

    BASENAME=$(basename "$INSTANCE")
    OUTFILE="$OUTPATH/${BASENAME}_seed-${SEED}.out"
    if [ -f "$OUTFILE" ]; then
        COST=$(cut -f1 "$OUTFILE")
        ELAPSED=$(cut -f2 "$OUTFILE")
    else
        COST="ERROR"
        ELAPSED="ERROR"
    fi
    echo "$SEED,$WALL,$COST,$ELAPSED" >> "$CSV"
    echo "wall=${WALL}s cost=${COST} filo2_elapsed=${ELAPSED}s"
done

echo "Done. Summary: $CSV"
cat "$CSV"
