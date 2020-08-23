#ifndef DEBUG_HPP
#define DEBUG_HPP
#include <oneweekend/external.hpp>
void cuda_control(cudaError_t res, const char *const fn, const char *const f,
                  const int l) {
  if (res != cudaSuccess) {
    std::cerr << "CUDA ERROR :: " << static_cast<unsigned int>(res) << " "
              << cudaGetErrorName(res) << " file: " << f << " line: " << l
              << " function: " << fn << std::endl;
    cudaDeviceReset();
    exit(99);
  }
}

#define CUDA_CONTROL(v) cuda_control((v), #v, __FILE__, __LINE__)

#endif
