#!/bin/bash
# Collect the 15-seed VDA comparison: our tuned config vs FILO2 at matched wall clock.
set -e
cd /mnt/c/internship/iitm/cvrp
echo "seed,ours,ours_wall_ms,filo2"
for s in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    c=$(grep -m1 "^Final Cost:" results/bench/routemin_tuned_vda_5seed/sol_${s}.txt | awk '{print $3}')
    t=$(grep -m1 "Total time:" results/bench/routemin_tuned_vda_5seed/stdout_${s}.txt | awk '{print $3}')
    if [ "$s" -le 5 ]; then
        f=$(awk '{print $1}' results/bench/filo2_vda_matched2/Valle-D-Aosta.vrp_seed-${s}.out)
    else
        f=$(awk '{print $1}' results/bench/filo2_vda_seeds6to15/Valle-D-Aosta.vrp_seed-${s}.out)
    fi
    echo "$s,$c,$t,$f"
done
