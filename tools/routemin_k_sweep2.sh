#!/bin/bash
# Follow-up to routemin_k_sweep.sh, which showed route count flipping from rising to
# falling as ROUTEMIN's candidate width grew (k=30 -> +4 routes, k=300 -> -5 routes).
# Two questions: (a) does going wider still (toward FILO2's 1500) keep helping,
# (b) does the benefit survive in the production P=2 config?
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/routemin_k_sweep2
mkdir -p "$OUT"
run() {
    local p=$1 k=$2
    echo "=== P=$p k=$k ==="
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp \
        --seed 1 -p "$p" --routemin-iters 2000 --routemin-k "$k" \
        --stage2-ms 40000 --stage3-ms 1000 --stage5-ms 60000 \
        --out "$OUT/sol_p${p}_k${k}.txt" --log "$OUT/log_p${p}_k${k}.txt" 2>&1 \
        | grep -iE "^Final cost|^Total time"
    grep -iE "routemin" "$OUT/log_p${p}_k${k}.txt" || true
    head -2 "$OUT/sol_p${p}_k${k}.txt" | tail -1
    echo
}
run 1 1000
run 2 300
run 2 1000
