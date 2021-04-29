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
__device__ inline real_t box_muller(rng_state_t<real_t>& rng_state) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  const real_t epsilon = epsilon<real_t>();
  const real_t two_pi = 2 * M_PI;

  real_t u1, u2;
  do {
    u1 = unif_rand(rng_state);
    u2 = unif_rand(rng_state);
  } while (u1 <= epsilon);

  return std::sqrt(-2 * std::log(u1)) * std::cos(two_pi * u2);
}

// The type declarations for mean and sd are ugly but prevent the
// compiler complaining about conflicting inferred types for real_t
__nv_exec_check_disable__
template <typename real_t>
__device__ real_t rnorm(rng_state_t<real_t>& rng_state,
                        typename rng_state_t<real_t>::real_t mean,
                        typename rng_state_t<real_t>::real_t sd) {
  real_t z = box_muller<real_t>(rng_state);
  return z * sd + mean;
}

template <typename real_t>
__global__ void normal_kernel(uint64_t * rng_state,
                               real_t *draws, const long n_draws, const int draw_per_thread) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_draws; i += blockDim.x * gridDim.x) {
    interleaved<uint64_t> p_rng(rng_state, i, n_draws);
    rng_state_t<real_t> rng_block = get_rng_state<real_t>(p_rng);
    float draw = 0;
    for (int j = 0; j < draw_per_thread; ++j) {
        float new_draw = rnorm(rng_block, 0, 1);
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
  normal_kernel<real_t><<<blockCount, blockSize>>>(rng_state.data(), draws, total_draws, draw_per_thread);
  // normal_kernel<real_t><<<1, 1>>>(rng_state.data(), draws, total_draws, draw_per_thread);
  CUDA_CALL(cudaDeviceSynchronize());
  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

  std::cout << total_draws << " threads each drawing " << draw_per_thread << " ~Norm()" << std::endl;
  std::cout << time_span.count() << " s" << std::endl;

  /*
  std::vector<real_t> h_draws(total_draws);
  CUDA_CALL(cudaMemcpy(h_draws.data(), draws, total_draws * sizeof(real_t), cudaMemcpyDefault));
  */

  CUDA_CALL(cudaFree(draws));
}
