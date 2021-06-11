  #!/bin/bash

for n in 10 30 50 70 90 100 120 140 160 180 200 220 240 250
do
  python3 framework/src/main.py --problem LeadingOnes --size $n --goal $n --time_limit 15 --sample_size 15 --algorithm ["GABE1,GABE2,GAAdaptiveMut,SD_RLS,GAStatic,GADynamic,GAStandard"] --SD_RLS_R $((n+1))
done
