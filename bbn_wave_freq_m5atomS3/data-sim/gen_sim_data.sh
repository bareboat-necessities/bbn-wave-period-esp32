#!/bin/bash -e

make clean
make -j4 all
./waves_sim
