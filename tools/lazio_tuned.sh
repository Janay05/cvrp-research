#!/bin/bash
# Full Lazio pipeline with ROUTEMIN iterations raised, tuned to land near FILO2's 315 s.
#
# 20,000 iters gets construction+ROUTEMIN alone to 3,159,827,051 (within 0.019% of FILO2's
# CONVERGED 3,159,235,192) and 40,132 routes, but costs ~188 s of Stage 1, which with a 73 s
# setup leaves almost nothing for the search stages. So this trades some ROUTEMIN convergence
# back for stage time and checks where the balance actually lands.
#
# Stage 3/4 carry large unbudgeted overheads at Lazio (stage3 budgeted 25 s ran 73 s), so
# budgets are set well under the nominal remaining time.
# Target to beat: FILO2 3,159,235,192 / 40,111 routes / 315 s.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/lazio_tuned
mkdir -p "$OUT"
run() {
    local tag=$1; shift
    echo "=== $tag ==="
    /usr/bin/time -v ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp --seed 1 -p 4 \
        "$@" --out "$OUT/sol_${tag}.txt" --log "$OUT/log_${tag}.txt" 2>&1 \
        | grep -iE "^Final cost|^Setup|^Stage|Total time|Maximum resident"
    grep -i routemin "$OUT/log_${tag}.txt"
    head -2 "$OUT/sol_${tag}.txt" | tail -1
    echo
}
run rm8k  --routemin-k 500 --routemin-iters 8000 \
    --stage2-ms 45000 --stage3-ms 12000 --stage5-ms 45000
