#!/bin/bash


for problem_instance in $(ls problems/tsp)
do
  python3 framework/src/main.py --problem TSP --problem_file $problem_instance --time_limit 120 --algorithm ["GABE1,GABE2,GAAdaptiveMut,SD_RLS,GAStatic,GADynamic,GAStandard"] --SD_RLS_R $(wc -l problems/tsp/$problem_instance | head -n1 | cut -d " " -f1) --sample_size 5
done
