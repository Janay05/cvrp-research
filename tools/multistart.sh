#!/bin/bash
# Stage 6-C (docs/reports/009_plan_beating_filo2.md): parallel multi-start portfolio.
# Launches N independent solver instances concurrently (each its own OS process, each with a
# different seed and its own --out/--log so they don't race on the same files), waits for all
# of them, then reports the best (lowest cost) final solution. "Embarrassingly parallel" per
# the plan -- no shared state between starts, no code changes to the solver's search itself.
#
# Usage: multistart.sh <instance.vrp> <num_starts> <p_per_start> <stage2_ms> <stage3_ms> <stage5_ms> <tag> [base_seed]
set -e
cd /mnt/c/internship/iitm/cvrp

INSTANCE="$1"
N="$2"
P_PER_START="$3"
STAGE2_MS="$4"
STAGE3_MS="$5"
STAGE5_MS="$6"
TAG="$7"
BASE_SEED="${8:-1}"

OUTDIR="results/multistart_${TAG}"
mkdir -p "$OUTDIR"

EXE=./src/build_wsl/cvrp_parallel

START_WALL=$(date +%s.%N)

pids=()
for i in $(seq 0 $((N - 1))); do
    seed=$((BASE_SEED + i))
    "$EXE" "$INSTANCE" --seed "$seed" -p "$P_PER_START" \
        --stage2-ms "$STAGE2_MS" --stage3-ms "$STAGE3_MS" --stage5-ms "$STAGE5_MS" \
        --out "$OUTDIR/sol_${i}.txt" --log "$OUTDIR/log_${i}.txt" \
        > "$OUTDIR/stdout_${i}.txt" 2>&1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

END_WALL=$(date +%s.%N)
WALL=$(echo "$END_WALL - $START_WALL" | bc)

echo "=== Multistart results ($N starts, p=$P_PER_START each, ${WALL}s wall) ==="
best_cost=""
best_i=""
for i in $(seq 0 $((N - 1))); do
    seed=$((BASE_SEED + i))
    cost=$(grep "^Final Cost:" "$OUTDIR/sol_${i}.txt" | awk '{print $3}')
    echo "  start $i (seed $seed): cost=$cost"
    if [ -z "$best_cost" ] || [ "$cost" -lt "$best_cost" ]; then
        best_cost="$cost"
        best_i="$i"
    fi
done

echo "--- best: start $best_i, cost=$best_cost ---"
cp "$OUTDIR/sol_${best_i}.txt" "$OUTDIR/best.txt"
echo "wall_seconds,best_cost,num_starts,p_per_start" > "$OUTDIR/summary.csv"
echo "$WALL,$best_cost,$N,$P_PER_START" >> "$OUTDIR/summary.csv"
