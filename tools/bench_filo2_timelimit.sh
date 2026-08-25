#!/bin/bash
set -e
cd /mnt/c/internship/iitm/cvrp
EXE=./baselines/filo2/build_wsl_tl/filo2
INST=data/instances/I/Valle-D-Aosta.vrp
OUTDIR=results/bench/filo2_tl_vda
mkdir -p "$OUTDIR"
for seed in 1 2 3 4 5; do
  echo "=== seed $seed ===" >&2
  "$EXE" "$INST" --seed $seed --optimization-seconds 102 --outpath "$OUTDIR/" 2>&1 | tail -6
done
