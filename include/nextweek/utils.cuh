#ifndef UTILS_CUH
#define UTILS_CUH

// some utility functions

const float PI = 3.141592653589f;

__device__ float dfmin(float f1, float f2) {
  return f1 < f2 ? f1 : f2;
}
__device__ float dfmax(float f1, float f2) {
  return f1 > f2 ? f1 : f2;
}

__host__ __device__ float clamp(float v, float mn,
                                float mx) {
  if (v < mn)
    return mn;
  if (v > mx)
    return mx;
  return v;
}

#endif
