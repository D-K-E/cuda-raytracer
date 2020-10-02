#ifndef RAY_CUH
#define RAY_CUH
#include <nextweek/vec3.cuh>

class Ray {
public:
  Vec3 dir;
  Vec3 orig;
  double tm;

public:
  __host__ __device__ Ray() {}
  __host__ __device__ Ray(const Point3 &p1, const Vec3 &d1,
                 double time = 0.0f)
      : orig(p1), dir(d1), tm(time) {}
  __host__ __device__ Point3 origin() const { return orig; }
  __host__ __device__ Vec3 direction() const { return dir; }
  __host__ __device__ double time() const { return tm; }
  __host__ __device__ Point3 at(double t) const {
    return orig + dir * t;
  }
};

#endif
