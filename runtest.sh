#!/usr/bin/sh
# Run unit tests with julia and precompilation disabled
mkdir -p './test/artifacts/'
rm -rf './test/artifacts/*'
TEST='true' julia --compiled-modules=no --code-coverage=user --project=. -e 'using Pkg; Pkg.test()'
