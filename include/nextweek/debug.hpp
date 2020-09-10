#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <nextweek/external.hpp>

void cuda_control(cudaError_t res, const char *const fn,
                  const char *const f, const int l) {
  if (res != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA ERROR :: " << static_cast<unsigned int>(res)
       << std::endl
       << cudaGetErrorName(res) << " file: " << f
       << std::endl
       << " line: " << l << std::endl
       << " function: " << fn << std::endl;
    cudaDeviceReset();
    std::string s = ss.str();
    throw std::runtime_error(s.c_str());
    exit(99);
  }
}

#define CUDA_CONTROL(v)                                    \
  cuda_control((v), #v, __FILE__, __LINE__)

#endif
