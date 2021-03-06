#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring> // memcpy
#include <new>
#include <stdexcept>
#include <sstream>
#include <vector>

#include <cuda.cuh>

template <typename T>
class device_array {
public:
  // Default constructor
  device_array() : data_(nullptr), size_(0) {
  }

  // Constructor to allocate empty memory
  device_array(const size_t size) : size_(size) {
#ifdef __NVCC__
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemset(data_, 0, size_ * sizeof(T)));
#else
    data_ = new T[size_];
    std::memset(data_, 0, size_ * sizeof(T));
#endif
  }

  // Constructor from vector
  device_array(const std::vector<T>& data) : size_(data.size()) {
#ifdef __NVCC__
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemcpy(data_, data.data(), size_ * sizeof(T),
                         cudaMemcpyDefault));
#else
    data_ = new T[size_];
    std::memcpy(data_, data.data(), size_ * sizeof(T));
#endif
  }

  // Copy
  device_array(const device_array& other) : size_(other.size_) {
#ifdef __NVCC__
    CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
    CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                         cudaMemcpyDefault));
#else
    data_ = new T[size_];
    std::memcpy(data_, other.data_, size_ * sizeof(T));
#endif
  }

  // Copy assign
  device_array& operator=(const device_array& other) {
    if (this != &other) {
      size_ = other.size_;
#ifdef __NVCC__
      CUDA_CALL(cudaFree(data_));
      CUDA_CALL(cudaMalloc((void**)&data_, size_ * sizeof(T)));
      CUDA_CALL(cudaMemcpy(data_, other.data_, size_ * sizeof(T),
                           cudaMemcpyDefault));
#else
      delete[] data_;
      data_ = new T[size_];
      std::memcpy(data_, other.data_, size_ * sizeof(T));
#endif
    }
    return *this;
  }

  // Move
  device_array(device_array&& other) : data_(nullptr), size_(0) {
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  // Move assign
  device_array& operator=(device_array&& other) {
    if (this != &other) {
#ifdef __NVCC__
      CUDA_CALL(cudaFree(data_));
#else
      delete[] data_;
#endif
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  ~device_array() {
#ifdef __NVCC__
    CUDA_CALL_NOTHROW(cudaFree(data_));
#else
    delete[] data_;
#endif
  }

  void get_array(std::vector<T>& dst, const bool async = false) const {
#ifdef __NVCC__
    if (async) {
      CUDA_CALL(cudaMemcpyAsync(dst.data(), data_, dst.size() * sizeof(T),
                          cudaMemcpyDefault));
    } else {
      CUDA_CALL(cudaMemcpy(dst.data(), data_, dst.size() * sizeof(T),
                          cudaMemcpyDefault));
    }
#else
    std::memcpy(dst.data(), data_, dst.size() * sizeof(T));
#endif
  }

  // General method to set the device array, allowing src to be written
  // into the device data_ array starting at dst_offset
  void set_array(const T* src, const size_t src_size,
                 const size_t dst_offset, const bool async = false) {
#ifdef __NVCC__
    if (async) {
      CUDA_CALL(cudaMemcpyAsync(data_ + dst_offset, src,
                          src_size * sizeof(T), cudaMemcpyDefault));
    } else {
      CUDA_CALL(cudaMemcpy(data_ + dst_offset, src,
                          src_size * sizeof(T), cudaMemcpyDefault));
    }
#else
    std::memcpy(data_ + dst_offset, src, src_size * sizeof(T));
#endif
  }

  // Specialised form to set the device array, writing all of src into
  // the device data_
  void set_array(const std::vector<T>& src, const bool async = false) {
    size_ = src.size();
#ifdef __NVCC__
    if (async) {
      CUDA_CALL(cudaMemcpyAsync(data_, src.data(), size_ * sizeof(T),
                          cudaMemcpyDefault));
    } else {
      CUDA_CALL(cudaMemcpy(data_, src.data(), size_ * sizeof(T),
                          cudaMemcpyDefault));
    }
#else
    std::memcpy(data_, src.data(), size_ * sizeof(T));
#endif
  }

  T* data() {
    return data_;
  }

  size_t size() const {
    return size_;
  }

private:
  T* data_;
  size_t size_;
};

// Specialisation of the above for void* memory needed by some cub functions
// Construct once and use set_size() to modify
// Still using malloc/free instead of new and delete, as void type problematic
template <>
class device_array<void> {
public:
  // Default constructor
  device_array() : data_(nullptr), size_(0) {}
  // Constructor to allocate empty memory
  device_array(const size_t size) : size_(size) {
    if (size_ > 0) {
#ifdef __NVCC__
      CUDA_CALL(cudaMalloc((void**)&data_, size_));
#else
      data_ = (void*) std::malloc(size_);
      if (!data_) {
        throw std::bad_alloc();
      }
#endif
    }
  }
  ~device_array() {
#ifdef __NVCC__
    CUDA_CALL_NOTHROW(cudaFree(data_));
#else
    std::free(data_);
#endif
  }
  void set_size(size_t size) {
    size_ = size;
#ifdef __NVCC__
    CUDA_CALL(cudaFree(data_));
    if (size_ > 0) {
      CUDA_CALL(cudaMalloc((void**)&data_, size_));
    } else {
      data_ = nullptr;
    }
#else
    std::free(data_);
    if (size_ > 0) {
      data_ = (void*) std::malloc(size_);
      if (!data_) {
        throw std::bad_alloc();
      }
    } else {
      data_ = nullptr;
    }
#endif
  }
  void* data() {
    return data_;
  }
  size_t size() const {
    return size_;
  }

private:
  device_array ( const device_array<void> & ) = delete;
  device_array ( device_array<void> && ) = delete;

  void* data_;
  size_t size_;
};

// The class from before, which is a light wrapper around a pointer
// This can be used within a kernel with copying memory. There is no
// way of telling if the pointer has been freed or not, so this must
// have a lifecycle that is shorter than the calling function.
template <typename T>
class interleaved {
public:
  DEVICE interleaved(T* data, size_t offset, size_t stride) :
    data_(data + offset),
    stride_(stride) {
  }

  template <typename Container>
  DEVICE interleaved(Container& data, size_t offset, size_t stride) :
    interleaved(data.data(), offset, stride) {
  }

  DEVICE T& operator[](size_t i) {
    return data_[i * stride_];
  }

  DEVICE const T& operator[](size_t i) const {
    return data_[i * stride_];
  }

  DEVICE interleaved<T> operator+(size_t by) {
    return interleaved(data_ + by * stride_, 0, stride_);
  }

  DEVICE const interleaved<T> operator+(size_t by) const {
    return interleaved(data_ + by * stride_, 0, stride_);
  }

private:
  // TODO: these can be set as const.
  T* data_;
  size_t stride_;
};

