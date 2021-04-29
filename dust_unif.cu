// See https://docs.nvidia.com/cuda/curand/device-api-overview.html#poisson-api-example

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include <cuda.cuh>
#include <dust_rng.cuh>

template <typename real_t>
__global__ void unif_kernel(uint64_t * rng_state,
                               real_t *draws, const long n_draws, const int draw_per_thread) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_draws; i += blockDim.x * gridDim.x) {
    interleaved<uint64_t> p_rng(rng_state, i, n_draws);
    rng_state_t<real_t> rng_block = get_rng_state<real_t>(p_rng);
    float draw = 0;
    for (int j = 0; j < draw_per_thread; ++j) {
        float new_draw = unif_rand<real_t>(rng_block);
        draw += new_draw;
        //printf("%d %d %f %f\n", i, j, new_draw, draw);
        __syncwarp();
    }
    draws[i] = draw;
    /* Copy state back to global memory */
    put_rng_state(rng_block, p_rng);
  }
}

int main(int argc, char *argv[]) {
  typedef float real_t;
  using namespace std::chrono;

  const long total_draws = std::stoi(argv[1]);
  const int draw_per_thread = std::stoi(argv[2]);

  device_array<uint64_t> rng_state = load_rng<real_t>(total_draws);

  real_t* draws;
  CUDA_CALL(cudaMalloc((void**)&draws, total_draws * sizeof(real_t)));

  const size_t blockSize = 128;
  const size_t blockCount = (total_draws + blockSize - 1) / blockSize;

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  unif_kernel<real_t><<<blockCount, blockSize>>>(rng_state.data(), draws, total_draws, draw_per_thread);
  // unif_kernel<real_t><<<1, 1>>>(rng_state.data(), draws, total_draws, draw_per_thread);
  CUDA_CALL(cudaDeviceSynchronize());
  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

  std::cout << total_draws << " threads each drawing " << draw_per_thread << " ~Unif()" << std::endl;
  std::cout << time_span.count() << " s" << std::endl;

  /*
  std::vector<real_t> h_draws(total_draws);
  CUDA_CALL(cudaMemcpy(h_draws.data(), draws, total_draws * sizeof(real_t), cudaMemcpyDefault));
  */

  CUDA_CALL(cudaFree(draws));
}
