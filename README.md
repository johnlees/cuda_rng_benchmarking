# curand vs dust

## Overview

At baseline, curand is about 3x faster (1.45ms for 100k threads, 10k draws; vs 4.88ms) than dust for uniform/normal draws.
dust is about 2x faster than curand for Poisson draws.

Replacing the divide with a multiplication by a reciprocal, and adding a cast of RNG state from
uint64_t to uint32_t (lowest 32 bits) roughly doubles the speed of the dust unif draw (2.67ms).

Replacing the 64-bit xoshiro256** generator used in dust with the 32-bit xoshiro128+ generator
roughly doubles speed again (1.28ms), making it slightly faster than curand.
(see the `xoshiro128` branch).

## Install

With `nvcc` installed and on the path, run `make`.

## Run

Programs can be run with `<method>_<dist> threads draws`, where `method` is `curand` or `dust`,
and `dist` is one of `unif`, `norm` or `poisson`.

`threads` sets the total number of threads run.
`draws` is the number of draws each thread does (for a total of `threads * draws` calls to the RNG).

The program will output the time taken for the kernel run.

## Profiling

Remove the programs, and then rebuild them by running `CXXFLAGS="-pg -O2 --generate-line-info" make`

Profiles can be run with a command such as:
```
ncu -o dust_unif --set full dust_unif 100000 10000
```

Note that curand runs its seeding for all the threads in a kernel, whereas dust does this
on the host and copies it over. So when profiling you want to measure the draw kernel time
(i.e. ignore the first kernel with curand). The times output to stdout do this for you.
