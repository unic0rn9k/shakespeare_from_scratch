#!/usr/bin/sh
# Run unit tests with julia and precompilation disabled
mkdir -p './test/artifacts/'
rm -rf './test/artifacts/*'
cd test
for p in py/* 
do
  python3 $p
done
TEST='true' julia 'test.jl'