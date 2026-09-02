#!/bin/bash
# Re-verification of Stage 6-C multi-start with the current (post Stage5-fix +
# Stage3-fix, CW+ROUTEMIN-default) build. Report 009's original stage6c_multistart_vda
# result predates both time-budget fixes and used the old MST-default single-start
# baseline (22,000,544). This uses today's best single-start VDA config
# (tools/cw_rmin_5seed.sh's flags) so the multistart lift is measured against the
# current baseline (21,791,054 @ 94.6s, report 010 SS0.6), not a stale one.
set -e
cd /mnt/c/internship/iitm/cvrp
OUT=results/bench/multistart_vda_current
mkdir -p "$OUT"
pids=()
for i in $(seq 1 12); do
    ./src/build_wsl/cvrp_parallel data/instances/I/Valle-D-Aosta.vrp --seed "$i" -p 2 \
        --construction cw --routemin-k 1000 --cw-neighbors 100 --routemin-iters 2000 \
        --stage2-ms 31000 --stage3-ms 1000 --stage5-ms 46000 \
        --out "$OUT/sol_${i}.txt" --log "$OUT/log_${i}.txt" \
        > "$OUT/stdout_${i}.txt" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
echo ALL_DONE
