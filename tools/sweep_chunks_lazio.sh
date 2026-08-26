#!/usr/bin/env bash
# Light chunk-count (P) check on Lazio: holds the settled per-stage time budgets fixed
# and varies only -p, to see whether the VDA sweep's "smaller P can be better" pattern
# holds at ~50x the scale. Fewer seeds than the VDA sweep since each Lazio run is
# ~4-5 minutes vs VDA's ~100s.
# Usage: sweep_chunks_lazio.sh <p1> <p2> ...
set -uo pipefail
cd /mnt/c/internship/iitm/cvrp

SEEDS="1,2"
BUDGETS="--stage2-ms 200000 --stage3-ms 2000 --stage5-ms 20000"

for P in "$@"; do
    TAG="012_sweep_lazio_p${P}"
    echo "=== P=${P} ==="
    python3 tools/bench.py --instances data/instances/I/Lazio.vrp --seeds "$SEEDS" \
        --extra-args "-p ${P} ${BUDGETS}" --tag "$TAG" \
        --exe src/build_wsl/cvrp_parallel --timeout 600
done

echo "=== LAZIO SWEEP DONE ==="
