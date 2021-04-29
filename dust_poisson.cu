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
__device__ int rpois(rng_state_t<real_t>& rng_state,
                     typename rng_state_t<real_t>::real_t lambda) {
  int x = 0;
  if (lambda == 0) {
    // do nothing, but leave this branch in to help the GPU
  } else if (lambda < 10) {
    // Knuth's algorithm for generating Poisson random variates.
    // Given a Poisson process, the time between events is exponentially
    // distributed. If we have a Poisson process with rate lambda, then,
    // the time between events is distributed Exp(lambda). If X ~
    // Uniform(0, 1), then Y ~ Exp(lambda), where Y = -log(X) / lambda.
    // Thus to simulate a Poisson draw, we can draw X_i ~ Exp(lambda),
    // and N ~ Poisson(lambda), where N is the least number such that
    // \sum_i^N X_i > 1.
    const real_t exp_neg_rate = std::exp(-lambda);

    real_t prod = 1;

    // Keep trying until we surpass e^(-rate). This will take
    // expected time proportional to rate.
    while (true) {
      real_t u = dust::unif_rand<real_t>(rng_state);
      prod = prod * u;
      if (prod <= exp_neg_rate && x <= integer_max()) {
        break;
      }
      x++;
    }
  } else {
    // Transformed rejection due to Hormann.
    //
    // Given a CDF F(x), and G(x), a dominating distribution chosen such
    // that it is close to the inverse CDF F^-1(x), compute the following
    // steps:
    //
    // 1) Generate U and V, two independent random variates. Set U = U - 0.5
    // (this step isn't strictly necessary, but is done to make some
    // calculations symmetric and convenient. Henceforth, G is defined on
    // [-0.5, 0.5]).
    //
    // 2) If V <= alpha * F'(G(U)) * G'(U), return floor(G(U)), else return
    // to step 1. alpha is the acceptance probability of the rejection
    // algorithm.
    //
    // For more details on transformed rejection, see:
    // http://citeseer.ist.psu.edu/viewdoc/citations;jsessionid=1BEB35946CC807879F55D42512E5490C?doi=10.1.1.48.3054.
    //
    // The dominating distribution in this case:
    //
    // G(u) = (2 * a / (2 - |u|) + b) * u + c

    const real_t log_rate = std::log(lambda);

    // Constants used to define the dominating distribution. Names taken
    // from Hormann's paper. Constants were chosen to define the tightest
    // G(u) for the inverse Poisson CDF.
    const real_t b = 0.931 + 2.53 * std::sqrt(lambda);
    const real_t a = -0.059 + 0.02483 * b;

    // This is the inverse acceptance rate. At a minimum (when rate = 10),
    // this corresponds to ~75% acceptance. As the rate becomes larger, this
    // approaches ~89%.
    const real_t inv_alpha = 1.1239 + 1.1328 / (b - 3.4);

    while (true) {
      real_t u = unif_rand<real_t>(rng_state);
      u -= 0.5;
      real_t v = unif_rand<real_t>(rng_state);

      real_t u_shifted = 0.5 - std::fabs(u);
      int k = floor((2 * a / u_shifted + b) * u + lambda + 0.43);

      if (k > integer_max()) {
        // retry in case of overflow.
        continue; // # nocov
      }

      // When alpha * f(G(U)) * G'(U) is close to 1, it is possible to
      // find a rectangle (-u_r, u_r) x (0, v_r) under the curve, such
      // that if v <= v_r and |u| <= u_r, then we can accept.
      // Here v_r = 0.9227 - 3.6224 / (b - 2) and u_r = 0.43.
      if (u_shifted >= 0.07 &&
          v <= 0.9277 - 3.6224 / (b - 2)) {
        x = k;
        break;
      }

      if (k < 0 || (u_shifted < 0.013 && v > u_shifted)) {
        continue;
      }

      // The expression below is equivalent to the computation of step 2)
      // in transformed rejection (v <= alpha * F'(G(u)) * G'(u)).
      real_t s = std::log(v * inv_alpha / (a / (u_shifted * u_shifted) + b));
      real_t t = -lambda + k * log_rate -
        lgamma(static_cast<real_t>(k + 1));
      if (s <= t) {
        x = k;
        break;
      }
    }
  }
  return x;
}

__global__ void poisson_kernel(uint64_t * rng_state,
                               float *draws, const long n_draws, const int draw_per_thread) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_draws; i += blockDim.x * gridDim.x) {
    interleaved<uint64_t> p_rng(rng_state, i, n_draws);
    rng_state_t<real_t> rng_block = dust::get_rng_state<real_t>(p_rng);
    float draw = 0;
    for (int j = 0; j < draw_per_thread; ++j) {
        float new_draw = rpois(rng_block, j);
        draw += new_draw;
        //printf("%d %d %f %f\n", i, j, new_draw, draw);
        __syncwarp();
    }
    draws[i] = draw;
    /* Copy state back to global memory */
    dust::put_rng_state(rng_block, p_rng);
  }
}

int main(int argc, char *argv[]) {
  using namespace std::chrono;

  const long total_draws = std::stoi(argv[1]);
  const int draw_per_thread = std::stoi(argv[2]);

  device_array<uint64_t> rng_state = load_rng(total_draws);

  float* draws;
  CUDA_CALL(cudaMalloc((void**)&draws, total_draws * sizeof(float)));

  const size_t blockSize = 128;
  const size_t blockCount = (total_draws + blockSize - 1) / blockSize;

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  poisson_kernel<<<blockCount, blockSize>>>(rng_state.data(), draws, total_draws, draw_per_thread);
  //simple_device_API_kernel<<<1, 1>>>(devStates, draws, total_draws, draw_per_thread);
  CUDA_CALL(cudaDeviceSynchronize());
  high_resolution_clock::time_point t2 = high_resolution_clock::now();

  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

  std::cout << total_draws << " threads each drawing " << draw_per_thread << " ~Pois()" << std::endl;
  std::cout << time_span.count() << " s" << std::endl;

  /*
  std::vector<float> h_draws(total_draws);
  CUDA_CALL(cudaMemcpy(h_draws.data(), draws, total_draws * sizeof(float), cudaMemcpyDefault));
  */

  CUDA_CALL(cudaFree(draws));
}
