#!/usr/bin/env bash
# Chunk-count (P) sweep on Valle-D-Aosta: holds the settled per-stage time budgets fixed
# and varies only -p, to see how cost and wall time trade off as chunking increases.
# Usage: sweep_chunks.sh <p1> <p2> ...
set -uo pipefail
cd /mnt/c/internship/iitm/cvrp

SEEDS="1,2,3"
BUDGETS="--stage2-ms 40000 --stage3-ms 1000 --stage5-ms 60000"

for P in "$@"; do
    TAG="011_sweep_p${P}"
    echo "=== P=${P} ==="
    python3 tools/bench.py --instances data/instances/I/Valle-D-Aosta.vrp --seeds "$SEEDS" \
        --extra-args "-p ${P} ${BUDGETS}" --tag "$TAG" \
        --exe src/build_wsl/cvrp_parallel --timeout 300
done

echo "=== SWEEP DONE ==="
