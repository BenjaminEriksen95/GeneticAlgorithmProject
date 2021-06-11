#!/bin/bash

for n in 50 100 200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000
do
  python3 framework/src/main.py --problem TSP --size $n --time_limit 60 --sample_size 5 --algorithm ["GABE1,GABE2,GAAdaptiveMut,SD_RLS,GAStatic,GADynamic,GAStandard"] --SD_RLS_R $((n+1))
done
