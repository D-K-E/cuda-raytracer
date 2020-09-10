#ifndef CBUFFER_HPP
#define CBUFFER_HPP

#include <nextweek/debug.hpp>
#include <nextweek/external.hpp>

template <typename T> void allocate_on_device(T *d_ptr) {
  CUDA_CONTROL(cudaMalloc((void **)&d_ptr, sizeof(T)));
}
template <typename T>
void allocate_on_device(T *d_ptr, int size) {
  CUDA_CONTROL(
      cudaMalloc((void **)&d_ptr, sizeof(T) * size));
}
template <typename T>
cudaError_t copy_to_device(const T *h_ptr, T *d_ptr) {
  return cudaMemcpy(d_ptr, (void *)h_ptr, sizeof(T),
                    cudaMemcpyHostToDevice);
}
template <typename T>
cudaError_t copy_to_device(std::vector<T> hdata, T *d_ptr) {

  return (cudaMemcpy(d_ptr, (const T *)hdata.data(),
                     sizeof(T) * hdata.size(),
                     cudaMemcpyHostToDevice));
}
template <typename T> cudaError_t to_device(T *data) {
  return cudaMallocManaged(&data, sizeof(T));
}
template <typename T>
cudaError_t to_device(std::vector<T> data, T *d_ptr) {
  d_ptr = data.data();
  return cudaMallocManaged(&d_ptr, sizeof(T) * data.size());
}

template <class T> class KernelArg {
public:
  T *h_ptr = nullptr;
  std::size_t size = 0;
  T *d_ptr = nullptr;
  KernelArg() : h_ptr(nullptr), size(0), d_ptr(nullptr) {}
  KernelArg(std::vector<T> vdata) {
    alloc_host(vdata);
    alloc_device();
    to_device();
  }
  KernelArg(T vdata) {
    std::vector<T> vec;
    vec.push_back(vdata);
    alloc_host(vec);
    alloc_device();
    to_device();
  }
  ~KernelArg() {
    cudaFree(d_ptr);
    delete h_ptr;
  }
  void alloc_host(std::vector<T> vdata) {
    size = vdata.size() * sizeof(T);
    h_ptr = vdata.data();
  }
  void alloc_device() {
    if (size == 0) {
      throw std::runtime_error("size is 0. Did you "
                               "allocate data in host ? "
                               "Can not allocate device "
                               "without allocating host "
                               " ");
    }
    CUDA_CONTROL(cudaMalloc((void **)&d_ptr, size));
  }
  void to_device() {
    if (h_ptr == nullptr)
      return;
    CUDA_CONTROL(cudaMemcpy(d_ptr, h_ptr, size,
                            cudaMemcpyHostToDevice));
  }
};

#endif
