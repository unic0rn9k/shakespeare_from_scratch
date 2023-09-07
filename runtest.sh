#!/usr/bin/sh
# Run unit tests with julia and precompilation disabled
julia --compiled-modules=no --code-coverage=user --project=. -e 'using Pkg; Pkg.test("shakespeare")'
