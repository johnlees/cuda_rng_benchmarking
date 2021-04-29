CXXFLAGS+=-Wall -Wextra -O3 -std=c++14
CPPFLAGS+=-I"."
CUDA_LDLIBS=-lcudadevrt -lcudart_static

CUDA_LDFLAGS =-L/usr/local/cuda-11.1/lib64 -L${CUDA_HOME}/targets/x86_64-linux/lib/stubs -L${CUDA_HOME}/targets/x86_64-linux/lib
CUDAFLAGS +=-Xptxas -dlcm=ca --cudart static -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75

all: curand_poisson curand_norm curand_unif dust_poisson dust_norm dust_unif

curand_poisson:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) curand_poisson.cu -o $@
