#!/bin/bash
cd /mnt/c/internship/iitm/cvrp
python3 tools/bench.py \
  --exe ./src/build_wsl/cvrp_parallel \
  --instances data/instances/I/Valle-D-Aosta.vrp \
  --seeds 1,2,3,4,5 \
  --extra-args "-p 2 --stage2-ms 40000 --stage3-ms 1000 --stage5-ms 60000" \
  --tag stage1_vda_p2
