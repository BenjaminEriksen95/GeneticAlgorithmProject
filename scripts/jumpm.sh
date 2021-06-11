#!/bin/bash



for n in 10 20 30 40 50 70 90 100
do
for m in 2 3 4 5
do
  python3 framework/src/main.py --problem JumpM --size $n --m $m --goal $((n+m)) --SD_RLS_R $((n+1)) --time_limit 30 --algorithm ["GABE1,GABE2,GAAdaptiveMut,SD_RLS,GAStatic,GADynamic,GAStandard"] --sample_size 15
done
done
