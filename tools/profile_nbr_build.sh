#!/bin/bash
# Where does the wide neighbour-list build time actually go at Lazio scale?
# k=300 adds ~150 s to Stage 0 (38 s -> 188 s) and raising the build thread count from 4 to
# 28 only recovered ~6 s, so the cost is NOT in the parallel kNN queries. Candidates are
# KDTree::build (serial) and symmetrize (serial, and its presence check is a linear scan of
# nbr[j] for every neighbour of every node).
set -e
cd /mnt/c/internship/iitm/cvrp
mkdir -p /tmp/nbrprof && cd /tmp/nbrprof
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-DPROFILE_NBR_BUILD \
      /mnt/c/internship/iitm/cvrp/src > /dev/null 2>&1
make cvrp_parallel -j > /dev/null 2>&1
cd /mnt/c/internship/iitm/cvrp
/tmp/nbrprof/cvrp_parallel data/instances/I/Lazio.vrp --seed 1 -p 4 \
    --construction cw --routemin-k 300 --cw-neighbors 100 \
    --stage2-ms 1 --stage3-ms 1 --stage5-ms 1 \
    --out /tmp/np.txt --log /tmp/np.log 2>&1 | grep -E "^\[nbr|Setup"
