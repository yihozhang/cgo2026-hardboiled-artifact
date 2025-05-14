#!/bin/bash

cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=../../halide-install \
  -DCMAKE_BUILD_TYPE=Release