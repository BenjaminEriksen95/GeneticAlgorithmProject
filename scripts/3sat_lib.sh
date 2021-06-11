#!/bin/bash

for problem_instance in $(ls problems/3sat)
do
    python3 framework/src/main.py --problem ThreeSAT --problem_file $problem_instance --time_limit 60  --sample_size 2 --algorithm ["GABE1,GABE2,GAAdaptiveMut,SD_RLS,GAStatic,GADynamic,GAStandard"] --SD_RLS_R $(wc -l problems/3sat/$problem_instance | head -n1 | cut -d " " -f1)
done
