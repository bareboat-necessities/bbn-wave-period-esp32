#!/bin/bash -e

make clean
make -j4 all

# run each wave height parallel
seq 0 4 | xargs -n1 -P5 ./waves_sim
./tunings

