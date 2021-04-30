#pragma once

#include <algorithm>
#include <cfloat>
#include <containers.cuh>

template <typename T>
struct rng_state_t {
  typedef T real_t;
  static __host__ __device__ size_t size() {
    return 4;
  }
  uint32_t state[4];
  __host__ __device__ uint32_t& operator[](size_t i) {
    return state[i];
  }
};

static inline uint32_t rotl(const uint32_t x, int k) {
	return (x << k) | (x >> (32 - k));
}

// This is the core generator (next() in the original C code)
template <typename T>
inline __host__ __device__ uint32_t xoshiro_next(rng_state_t<T>& state) {
	const uint32_t result = s[0] + s[3];

	const uint32_t t = s[1] << 9;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 11);

	return result;
}

// We don't really need to use HOST, but I put them explicitly in all
// the functions in this file as the RNG was trickiest to get right on
// both CPU and GPU, and this seemed to make it clear which you are
// allowed to use in kernels
inline uint64_t splitmix64(uint64_t seed) {
  uint64_t z = (seed += 0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

template <typename T>
inline std::vector<uint32_t> xoshiro_initial_seed(uint64_t seed) {
  // normal brain: for i in 1:4
  // advanced brain: -funroll-loops
  // galaxy brain:
  std::vector<uint64_t> state(rng_state_t<T>::size());
  state[0] = splitmix64(seed);
  state[1] = splitmix64(state[0]);
  state[2] = splitmix64(state[1]);
  state[3] = splitmix64(state[2]);
  std::vector<uint32_t> state32(rng_state_t<T>::size());
  state32[0] = static_cast<uint32_t>(state[0]);
  state32[1] = static_cast<uint32_t>(state[1]);
  state32[2] = static_cast<uint32_t>(state[2]);
  state32[3] = static_cast<uint32_t>(state[3]);
  return state32;
}

/* This is the jump function for the generator. It is equivalent
    to 2^128 calls to next(); it can be used to generate 2^128
    non-overlapping subsequences for parallel computations. */
template <typename T>
inline void xoshiro_jump(rng_state_t<T>& state) {
	static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

	uint32_t s0 = 0;
	uint32_t s1 = 0;
	uint32_t s2 = 0;
	uint32_t s3 = 0;
	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for(int b = 0; b < 32; b++) {
			if (JUMP[i] & UINT32_C(1) << b) {
				s0 ^= s[0];
				s1 ^= s[1];
				s2 ^= s[2];
				s3 ^= s[3];
			}
			next();
		}

	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
}

/* This is the long-jump function for the generator. It is equivalent to
    2^192 calls to next(); it can be used to generate 2^64 starting points,
    from each of which jump() will generate 2^64 non-overlapping
    subsequences for parallel distributed computations. */
template <typename T>
inline void xoshiro_long_jump(rng_state_t<T>& state) {
	static const uint32_t LONG_JUMP[] = { 0xb523952e, 0x0b6f099f, 0xccf5a0ef, 0x1c580662 };

	uint32_t s0 = 0;
	uint32_t s1 = 0;
	uint32_t s2 = 0;
	uint32_t s3 = 0;
	for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++)
		for(int b = 0; b < 32; b++) {
			if (LONG_JUMP[i] & UINT32_C(1) << b) {
				s0 ^= s[0];
				s1 ^= s[1];
				s2 ^= s[2];
				s3 ^= s[3];
			}
			next();
		}

	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
}

template <typename T, typename U = T>
HOST U unif_rand(rng_state_t<T>& state) {
  const uint32_t value = xoshiro_next(state);
  return U(value) / U(std::numeric_limits<uint32_t>::max());
}

template <>
inline __device__ double unif_rand(rng_state_t<double>& state) {
  const uint32_t value = xoshiro_next(state);
#ifdef __CUDA_ARCH__
  // 18446744073709551616.0 == __ull2double_rn(UINT64_MAX)
  double rand = (__ddiv_rn(__ull2double_rn(value), 18446744073709551616.0));
#else
  double rand = double(value) / double(std::numeric_limits<uint32_t>::max());
#endif
  return rand;
}

#define CURAND_2POW32_INV (2.3283064e-10f)

template <>
inline __device__ float unif_rand(rng_state_t<float>& state) {
  const uint32_t value = xoshiro_next(state);
#ifdef __CUDA_ARCH__
  float rand = value * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f);
//  float rand = (__fdiv_rn(__ull2float_rn(value), 18446744073709551616.0f));
#else
  float rand = float(value) / float(std::numeric_limits<uint32_t>::max());
#endif
  return rand;
}

inline HOSTDEVICE int integer_max() {
  #ifdef __CUDA_ARCH__
    return INT_MAX;
  #else
    return std::numeric_limits<int>::max();
  #endif
}

// We need this for the lgamma in rpois to work
#ifdef __NVCC__
template <typename real_t>
real_t lgamma_nvcc(real_t x);

template <>
inline DEVICE float lgamma_nvcc(float x) {
  return ::lgammaf(x);
}

template <>
inline DEVICE double lgamma_nvcc(double x) {
  return ::lgamma(x);
}
#endif

template <typename real_t>
HOSTDEVICE real_t lgamma(real_t x) {
#ifdef __CUDA_ARCH__
  return lgamma_nvcc(x);
#else
  return std::lgamma(x);
#endif
}

template <typename T>
DEVICE T epsilon_nvcc();

template <>
inline DEVICE float epsilon_nvcc() {
  return FLT_EPSILON;
}

template <>
inline DEVICE double epsilon_nvcc() {
  return DBL_EPSILON;
}

template <typename T>
class pRNG { // # nocov
public:
  pRNG(const size_t n, const std::vector<uint64_t>& seed) {
    rng_state_t<T> s;
    auto len = rng_state_t<T>::size();
    auto n_seed = seed.size() / len;
    for (size_t i = 0; i < n; ++i) {
      if (i < n_seed) {
        std::copy_n(seed.begin() + i * len, len, std::begin(s.state));
      } else {
        xoshiro_jump(s);
      }
      state_.push_back(s);
    }
  }

  size_t size() const {
    return state_.size();
  }

  void jump() {
    for (size_t i = 0; i < state_.size(); ++i) {
      xoshiro_jump(state_[i]);
    }
  }

  void long_jump() {
    for (size_t i = 0; i < state_.size(); ++i) {
      xoshiro_long_jump(state_[i]);
    }
  }

  rng_state_t<T>& state(size_t i) {
    return state_[i];
  }

  std::vector<uint32_t> export_state() {
    std::vector<uint32_t> state;
    export_state(state);
    return state;
  }

  void export_state(std::vector<uint32_t>& state) {
    const size_t n = rng_state_t<T>::size();
    state.resize(size() * n);
    for (size_t i = 0, k = 0; i < size(); ++i) {
      for (size_t j = 0; j < n; ++j, ++k) {
        state[k] = state_[i][j];
      }
    }
  }

  void import_state(const std::vector<uint32_t>& state, const size_t len) {
    auto it = state.begin();
    const size_t n = rng_state_t<T>::size();
    for (size_t i = 0; i < len; ++i) {
      for (size_t j = 0; j < n; ++j) {
        state_[i][j] = *it;
        ++it;
      }
    }
  }

  void import_state(const std::vector<uint32_t>& state) {
    import_state(state, size());
  }

private:
  std::vector<rng_state_t<T>> state_;
};

template <typename T>
__device__ rng_state_t<T> get_rng_state(const interleaved<uint32_t>& full_rng_state) {
  rng_state_t<T> rng_state;
  for (size_t i = 0; i < rng_state.size(); i++) {
    rng_state.state[i] = full_rng_state[i];
  }
  return rng_state;
}

// Write state into global memory
template <typename T>
__device__ void put_rng_state(rng_state_t<T>& rng_state,
                   interleaved<uint32_t>& full_rng_state) {
  for (size_t i = 0; i < rng_state.size(); i++) {
    full_rng_state[i] = rng_state.state[i];
  }
}

template <typename T, typename U>
size_t stride_copy(T dest, U src, size_t at, size_t stride) {
  static_assert(!std::is_reference<T>::value,
                "stride_copy should only be used with reference types");
  dest[at] = src;
  return at + stride;
}

template <typename real_t>
device_array<uint32_t> load_rng(const size_t n_state) {
  pRNG<real_t> rng_state(n_state, xoshiro_initial_seed<real_t>(static_cast<uint64_t>(1)));
  const size_t rng_len = rng_state_t<real_t>::size();
  std::vector<uint32_t> rng_i(n_state * rng_len); // Interleaved RNG state
  for (size_t i = 0; i < n_state; ++i) {
    // Interleave RNG state
    rng_state_t<real_t> p_rng = rng_state.state(i);
    size_t rng_offset = i;
    for (size_t j = 0; j < rng_len; ++j) {
      rng_offset = stride_copy(rng_i.data(), p_rng[j],
                               rng_offset, n_state);
    }
  }
  // H -> D copies
  device_array<uint32_t> d_rng(n_state * rng_len);
  d_rng.set_array(rng_i);
  return d_rng;
}