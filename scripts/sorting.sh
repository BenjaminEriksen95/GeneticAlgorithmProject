#!/bin/bash

for n in 2 4 6 8 10 12 14 16 18 20 25 30 35 40
do
  python3 framework/src/main.py --problem Sorting --size $n --goal $n --SD_RLS_R $((n+1)) --time_limit 15 --algorithm ["GABE1,GABE2,GAAdaptiveMut,SD_RLS,GAStatic,GADynamic,GAStandard"] --sample_size 15
done
