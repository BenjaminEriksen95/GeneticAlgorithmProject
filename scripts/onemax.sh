#!/bin/bash

for n in 10 30 50 70 90 100 150 200 250 300 350 450 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000
do
  python3 framework/src/main.py --problem OneMax --size $n --goal $n --SD_RLS_R $((n+1)) --time_limit 30 --algorithm ["GABE1,GABE2,SD_RLS, GAStandard"] --sample_size 15
done
