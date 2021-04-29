// See https://docs.nvidia.com/cuda/curand/device-api-overview.html#poisson-api-example

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_draws;
    i += blockDim.x * gridDim.x) {
    /* Each thread gets same seed, a different sequence
        number, no offset */
      curand_init(1234, i, 0, &state[i]);
    }
}

__global__ void simple_device_API_kernel(curandState *state,
                    float *draws, const long n_draws, const int draw_per_thread) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_draws; i += blockDim.x * gridDim.x) {
    curandState localState = state[i];
    /* Copy state to local memory for efficiency */
    float draw = 0;
    for (int j = 0; j < draw_per_thread; ++j) {
        draw += curand_poisson(&localState, j);
    }
    draws[i] = draw;
    /* Copy state back to global memory */
    state[i] = localState;
  }
}

int main() {
  curandState *devStates;

  const long total_draws = 1 << 20;
  const int draw_per_thread = 128;

  float* draws;
  CUDA_CALL(cudaMalloc((void**)&draws, total_draws * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&devStates, total_draws *
              sizeof(curandState)));

  const size_t blockSize = 64;
  const size_t blockCount = (total_draws + setup_blockSize - 1) / setup_blockSize;
  setup_kernel<<<blockCount, blockSize>>>(devStates);

  simple_device_API_kernel<<<blockSize, blockCount>>>(devStates, draws, total_draws, draw_per_thread);

  CUDA_CALL(cudaFree(draws));
  CUDA_CALL(cudaFree(devResults));
}