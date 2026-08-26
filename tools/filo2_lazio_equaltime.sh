#!/bin/bash
# Research: equal-wall-clock FILO2 vs our solver at Lazio (~1M customers) scale.
# Our Lazio gap (+0.77%, report 008) is smaller than VDA's, and that comparison used
# FILO2's own default iteration budget (~340s) vs our ~263-315s -- i.e. it was NOT
# equal-time. This is the strongest remaining "maybe we win at scale" counterargument,
# so it needs the same equal-time treatment VDA got.
# Run alone: FILO2 at 1M customers is memory-heavy and this machine has crashed before.
set -e
cd /mnt/c/internship/iitm/cvrp
OUTDIR=results/bench/filo2_lazio_equaltime
mkdir -p "$OUTDIR"
/usr/bin/time -v ./baselines/filo2/build_wsl_tl/filo2 \
    data/instances/I/Lazio.vrp --seed 1 --optimization-seconds 315 \
    --outpath "$OUTDIR/" 2>&1 | grep -iE "obj =|Run completed|Maximum resident|Elapsed"
echo "--- .out file ---"
cat "$OUTDIR"/*.out 2>/dev/null
