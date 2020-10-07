#ifndef RAY_CUH
#define RAY_CUH
#include <rest/vec3.cuh>

class Ray {
public:
  Vec3 dir;
  Vec3 orig;
  float tm;

public:
  __host__ __device__ Ray() {}
  __host__ __device__ Ray(const Point3 &p1, const Vec3 &d1,
                 float time = 0.0f)
      : orig(p1), dir(d1), tm(time) {}
  __host__ __device__ Point3 origin() const { return orig; }
  __host__ __device__ Vec3 direction() const { return dir; }
  __host__ __device__ float time() const { return tm; }
  __host__ __device__ Point3 at(float t) const {
    return orig + dir * t;
  }
};

#endif
