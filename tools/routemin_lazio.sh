#!/bin/bash
# Does the ROUTEMIN fix hold at Lazio (~1M customers)?
#
# Memory is the binding constraint here, which is why k starts small: the wide neighbour
# list costs n*k*4 bytes, so k=1000 (which was best at VDA) would be ~4 GB on top of our
# ~3.5 GB baseline -- i.e. the same 7.6 GB WSL ceiling that crashed the VM twice earlier in
# this program. k=100 is ~400 MB, the same width stage5_neighborLists already uses.
#
# Also watching: greedy_ffd_kmin is O(chunkSize x used_bins), which is ~0.1 s at VDA's 10k
# chunk but could be seconds at Lazio's 250k chunk -- worth seeing in the Stage 1 timing.
#
# Baselines at ~315 s: ours 3,182,981,663 / 40,431 routes; FILO2 3,159,235,192 / 40,111.
set -e
cd /mnt/c/internship/iitm/cvrp
K=${1:-100}
OUT=results/bench/routemin_lazio_k${K}
mkdir -p "$OUT"
/usr/bin/time -v ./src/build_wsl/cvrp_parallel data/instances/I/Lazio.vrp \
    --seed 1 -p 4 --routemin-iters 2000 --routemin-k "$K" \
    --stage2-ms 60000 --stage3-ms 30000 --stage5-ms 60000 \
    --out "$OUT/sol.txt" --log "$OUT/log.txt" 2>&1 \
    | grep -iE "^Final cost|Total time|Setup|Stage 1 &|Maximum resident"
echo "--- routemin ---"
grep -i routemin "$OUT/log.txt" || true
grep -iE "Stage 1:" "$OUT/log.txt" || true
head -2 "$OUT/sol.txt" | tail -1
