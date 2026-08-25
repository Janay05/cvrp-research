#!/bin/bash
set -e
cd /mnt/c/internship/iitm/cvrp/baselines/filo2
mkdir -p build_wsl_tl
cd build_wsl_tl
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_TIMELIMIT=ON -DENABLE_VERBOSE=ON -DENABLE_GUI=OFF ..
make -j
