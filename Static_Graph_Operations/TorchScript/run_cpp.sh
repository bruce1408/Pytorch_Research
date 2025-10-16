#!/bin/bash

set -e

mkdir build
cd build 
cmake -DCMAKE_PREFIX_PATH=/home/jiang/Downloads/libtorch ..
make -j