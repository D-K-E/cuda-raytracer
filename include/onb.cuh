#ifndef ONB_CUH
#define ONB_CUH

#include <rest/external.cuh>
#include <rest/ray.cuh>
#include <rest/vec3.cuh>

class Onb {
public:
  __host__ __device__ Onb() {}

  __host__ __device__ inline Vec3 operator[](int i) const {
    return axis[i];
  }

  __host__ __device__ Vec3 u() const { return axis[0]; }
  __host__ __device__ Vec3 v() const { return axis[1]; }
  __host__ __device__ Vec3 w() const { return axis[2]; }

  __host__ __device__ Vec3 local(float a, float b,
                                 float c) const {
    return a * u() + b * v() + c * w();
  }

  __host__ __device__ Vec3 local(const Vec3 &a) const {
    return a.x() * u() + a.y() * v() + a.z() * w();
  }

  __host__ __device__ void build_from_w(const Vec3 &);

public:
  Vec3 axis[3];
};

__host__ __device__ void Onb::build_from_w(const Vec3 &n) {
  axis[2] = to_unit(n);
  Vec3 a =
      (fabs(w().x()) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
  axis[1] = to_unit(cross(w(), a));
  axis[0] = cross(w(), v());
}

#endif
