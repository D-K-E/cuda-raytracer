#ifndef CBUFFER_HPP
#define CBUFFER_HPP

#include <nextweek/debug.hpp>
#include <nextweek/external.hpp>

template <typename T>
void upload_to_device(thrust::device_ptr<T> &d_ptr,
                      T *h_ptr, int size) {
  d_ptr = thrust::device_malloc<T>(size);
  for (int i = 0; i < size; i++) {
    d_ptr[i] = h_ptr[i];
  }
}
template <typename T>
void upload_to_device(thrust::device_ptr<T> &d_ptr,
                      std::vector<T> &h_ptr) {
  d_ptr = thrust::device_ptr<T>(h_ptr.data());
}
template <typename T>
void download_to_host(T *d_ptr, T *h_ptr, int size) {
  CUDA_CONTROL(
      cudaMemcpy((void *)h_ptr, (const void *)d_ptr,
                 sizeof(T) * size, cudaMemcpyDeviceToHost));
}

template <typename T>
thrust::device_vector<T>
to_device_vec(thrust::host_vector<T> hvec) {
  thrust::device_vector<T> dvec = hvec;
  return dvec;
}
template <typename T>
thrust::device_vector<T> to_device_vec(T &h) {
  thrust::host_vector<T> hvec;
  hvec.push_back(h);
  return to_device_vec(hvec);
}
template <typename T>
thrust::device_vector<T> to_device_vec(T *hs, int size) {
  thrust::host_vector<T> hvec;
  for (int i = 0; i < size; i++) {
    hvec.push_back(hs[i]);
  }
  return to_device_vec(hvec);
}

template <typename T> struct KernelArg {
  // based on the code
  // https://codeyarns.com/2011/04/09/how-to-pass-thrust-device-vector-to-kernel/
  T *arg;
  int length;
  __host__ void free() { cudaFree(arg); }
};
template <typename T>
KernelArg<T> mkKernelArg(thrust::device_vector<T> &dvec) {
  KernelArg<T> karg;
  karg.arg = thrust::raw_pointer_cast(&dvec[0]);
  karg.length = (int)dvec.size();
  return karg;
}
template <typename T>
KernelArg<T> mkKernelArg(thrust::host_vector<T> hvec) {
  return mkKernelArg<T>(to_device_vec<T>(hvec));
}

template <typename T>
KernelArg<T> mkKernelArg(T *hs, int size) {
  return mkKernelArg<T>(to_device_vec<T>(hs, size));
}

template <typename T> KernelArg<T> mkKernelArg(T &h) {
  return mkKernelArg<T>(to_device_vec<T>(h));
}

#endif
