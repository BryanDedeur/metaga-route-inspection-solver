#!/bin/bash

# python main.py -i benchmarks/gdb/gdb1.dat -k 0,0 -s 8115,3520,8647,9420,3116,6377,6207,4187,3641,8591,3580,8524,2650,2811,9963,7537,3472,3714,8158,7284,6948,6119,5253,5134,7350,2652,9968,3914,6899,4715 -j MMMR

# Define a list of seeds
SEEDS="8115,3520,8647,9420,3116,6377,6207,4187,3641,8591,3580,8524,2650,2811,9963,7537,3472,3714,8158,7284,6948,6119,5253,5134,7350,2652,9968,3914,6899,4715"

# Use GNU Parallel to run the Python script with each seed on a separate core
parallel -j 16 "python main.py -i benchmarks/bmcv/C05.dat -k 0,0 -s {} -j MMMR" ::: $(echo "$SEEDS" | tr ',' '\n')