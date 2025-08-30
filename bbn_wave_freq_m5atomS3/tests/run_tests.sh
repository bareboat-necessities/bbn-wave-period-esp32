#!/bin/bash

make clean; make all
./tests_sea
./tests_freq
./tests > results.csv

