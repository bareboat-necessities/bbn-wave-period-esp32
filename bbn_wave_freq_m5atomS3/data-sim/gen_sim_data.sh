#!/bin/bash -e

make clean
make -j4 all
./gen_sim_data
