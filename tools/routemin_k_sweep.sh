#!/bin/bash
# Does ROUTEMIN's candidate-list width explain why route count rises instead of falling?
# FILO2 runs ROUTEMIN at gamma=1.0 over ~1500 neighbours; our port used k=30.
# P=1 so there is no per-chunk bin-packing additivity penalty confounding the route count.
# Reference points on Valle-D-Aosta: FILO2 goes 810 -> 801 live routes; ours (k=30) 810 -> 814.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/routemin_k_sweep
mkdir -p "$OUT"
for k in 30 100 300; do
    echo "=== k=$k ==="
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed 1 -p 1 --routemin-iters 2000 --routemin-k "$k" \
        --stage2-ms 40000 --stage3-ms 1000 --stage5-ms 60000 \
        --out "$OUT/sol_k${k}.txt" --log "$OUT/log_k${k}.txt" 2>&1 | grep -iE "^Final cost"
    grep -iE "routemin" "$OUT/log_k${k}.txt" || true
    head -2 "$OUT/sol_k${k}.txt" | tail -1
    echo
done
