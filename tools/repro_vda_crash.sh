#!/usr/bin/env bash
# Reproduces the intermittent VDA crash by re-running each given seed with a clean,
# uniquely-named log file and a reliably-captured exit code.
set -uo pipefail
cd /mnt/c/internship/iitm/cvrp

for SEED in "$@"; do
    LOG="/tmp/vda_repro_seed${SEED}.log"
    echo "=== seed ${SEED} ==="
    timeout -s KILL 200 ./src/build_wsl/cvrp_parallel -f data/instances/I/Valle-D-Aosta.vrp \
        -p 4 --stage2-ms 40000 --stage3-ms 1000 --stage5-ms 60000 --seed "$SEED" \
        > "$LOG" 2>&1
    CODE=$?
    echo "seed ${SEED} exit=${CODE} log=${LOG}"
    if [ "$CODE" -ne 0 ]; then
        echo "--- last 40 lines of ${LOG} ---"
        tail -40 "$LOG"
    fi
done
