// See https://docs.nvidia.com/cuda/curand/device-api-overview.html#poisson-api-example

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandState *state, const long n_draws) {
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
        float new_draw = curand_uniform(&localState);
        draw += new_draw;
        //printf("%d %d %f %f\n", i, j, new_draw, draw);
        __syncwarp();
    }
    draws[i] = draw;
    /* Copy state back to global memory */
    state[i] = localState;
  }
}

int main(int argc, char *argv[]) {
  using namespace std::chrono;

  curandState *devStates;

  const long total_draws = std::stoi(argv[1]);
  const int draw_per_thread = std::stoi(argv[2]);

  float* draws;
  CUDA_CALL(cudaMalloc((void**)&draws, total_draws * sizeof(float)));
  CUDA_CALL(cudaMalloc((void **)&devStates, total_draws *
              sizeof(curandState)));

  const size_t blockSize = 128;
  const size_t blockCount = (total_draws + blockSize - 1) / blockSize;
  setup_kernel<<<blockCount, blockSize>>>(devStates, total_draws);
  CUDA_CALL(cudaDeviceSynchronize());

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  simple_device_API_kernel<<<blockCount, blockSize>>>(devStates, draws, total_draws, draw_per_thread);
  //simple_device_API_kernel<<<1, 1>>>(devStates, draws, total_draws, draw_per_thread);
  CUDA_CALL(cudaDeviceSynchronize());
  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

  std::cout << total_draws << " threads each drawing " << draw_per_thread << " ~Unif()" << std::endl;
  std::cout << time_span.count() << " s" << std::endl;

  /*
  std::vector<float> h_draws(total_draws);
  CUDA_CALL(cudaMemcpy(h_draws.data(), draws, total_draws * sizeof(float), cudaMemcpyDefault));
  */

  CUDA_CALL(cudaFree(draws));
  CUDA_CALL(cudaFree(devStates));
}
