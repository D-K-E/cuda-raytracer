// vec3.hpp for cuda
#ifndef VEC3_CUH
#define VEC3_CUH

#include <oneweekend/external.hpp>

class Vec3 {
public:
  double e[3];

  __host__ __device__ Vec3() {}
  __host__ __device__ Vec3(double e1, double e2, double e3) {
    e[0] = e1;
    e[1] = e2;
    e[2] = e3;
  }
  __host__ __device__ Vec3(double e1) {
    e[0] = e1;
    e[1] = e1;
    e[2] = e1;
  }
  __host__ __device__ Vec3(double es[3]) {
    e[0] = es[0];
    e[1] = es[1];
    e[2] = e[2];
  }
  __host__ __device__ inline double x() const {
    return e[0];
  }
  __host__ __device__ inline double y() const {
    return e[1];
  }
  __host__ __device__ inline double z() const {
    return e[2];
  }
  __host__ __device__ inline double r() const { return x(); }
  __host__ __device__ inline double g() const { return y(); }
  __host__ __device__ inline double b() const { return z(); }

  __host__ __device__ inline Vec3 operator-() const {
    return Vec3(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ inline double operator[](int i) const {
    return e[i];
  }

  __host__ __device__ inline Vec3 &
  operator+=(const Vec3 &v);
  __host__ __device__ inline Vec3 &
  operator-=(const Vec3 &v);
  __host__ __device__ inline Vec3 &
  operator*=(const Vec3 &v);
  __host__ __device__ inline Vec3 &
  operator/=(const Vec3 &v);
  __host__ __device__ inline Vec3 &
  operator*=(const double t);
  __host__ __device__ inline Vec3 &
  operator/=(const double t);
  __host__ __device__ inline Vec3 &
  operator+=(const double t);
  __host__ __device__ inline Vec3 &
  operator-=(const double t);

  __host__ __device__ inline double squared_length() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }
  __host__ __device__ inline double length() const {
    return sqrt(squared_length());
  }
  __host__ __device__ inline Vec3 to_unit() const;
  __host__ __device__ inline void unit_vector() const;
};

inline std::ostream &operator<<(std::ostream &os,
                                const Vec3 &t) {
  os << t.x() << " " << t.y() << " " << t.z();
  return os;
}

__host__ __device__ inline Vec3 operator+(const Vec3 &v1,
                                          const Vec3 &v2) {
  return Vec3(v1.x() + v2.x(), v1.y() + v2.y(),
              v1.z() + v2.z());
}

__host__ __device__ inline Vec3 operator-(const Vec3 &v1,
                                          const Vec3 &v2) {
  return Vec3(v1.x() - v2.x(), v1.y() - v2.y(),
              v1.z() - v2.z());
}
__host__ __device__ inline Vec3 operator*(const Vec3 &v1,
                                          const Vec3 &v2) {
  return Vec3(v1.x() * v2.x(), v1.y() * v2.y(),
              v1.z() * v2.z());
}
__host__ __device__ inline Vec3 operator/(const Vec3 &v1,
                                          const Vec3 &v2) {
  return Vec3(v1.x() / v2.x(), v1.y() / v2.y(),
              v1.z() / v2.z());
}
__host__ __device__ inline Vec3 operator*(const Vec3 &v1,
                                          double t) {
  return Vec3(v1.x() * t, v1.y() * t, v1.z() * t);
}
__host__ __device__ inline Vec3 operator/(const Vec3 &v1,
                                          double t) {
  return Vec3(v1.x() / t, v1.y() / t, v1.z() / t);
}
__host__ __device__ inline Vec3 operator+(const Vec3 &v1,
                                          double t) {
  return Vec3(v1.x() + t, v1.y() + t, v1.z() + t);
}
__host__ __device__ inline Vec3 operator-(const Vec3 &v1,
                                          double t) {
  return Vec3(v1.x() - t, v1.y() - t, v1.z() - t);
}

__host__ __device__ inline double dot(const Vec3 &v1,
                                     const Vec3 &v2) {
  return v1.x() * v2.x() + v1.y() * v2.y() +
         v1.z() * v2.z();
}

__host__ __device__ inline Vec3 cross(const Vec3 &v1,
                                      const Vec3 &v2) {
  return Vec3((v1.y() * v2.z() - v1.z() * v2.y()),
              (-(v1.x() * v2.z() - v1.z() * v2.x())),
              (v1.x() * v2.y() - v1.y() * v2.x()));
}

__host__ __device__ inline Vec3 &Vec3::
operator+=(const Vec3 &v) {
  e[0] += v.x();
  e[1] += v.y();
  e[2] += v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator*=(const Vec3 &v) {
  e[0] *= v.x();
  e[1] *= v.y();
  e[2] *= v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator/=(const Vec3 &v) {
  e[0] /= v.x();
  e[1] /= v.y();
  e[2] /= v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator-=(const Vec3 &v) {
  e[0] -= v.x();
  e[1] -= v.y();
  e[2] -= v.z();
  return *this;
}

__host__ __device__ inline Vec3 &Vec3::operator+=(double v) {
  e[0] += v;
  e[1] += v;
  e[2] += v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator-=(double v) {
  e[0] -= v;
  e[1] -= v;
  e[2] -= v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator*=(double v) {
  e[0] *= v;
  e[1] *= v;
  e[2] *= v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator/=(double v) {
  e[0] /= v;
  e[1] /= v;
  e[2] /= v;
  return *this;
}
__host__ __device__ inline Vec3 to_unit(Vec3 v) {
  return v / v.length();
}

#define RND (curand_uniform(&local_rand_state))

__device__ double random_double(curandState *loc, double min,
                              double max) {
  return min + (max - min) * curand_uniform(loc);
}

__device__ Vec3
random_double(curandState *local_rand_state) {
  return Vec3(curand_uniform(local_rand_state),
              curand_uniform(local_rand_state),
              curand_uniform(local_rand_state));
}
__device__ Vec3
random_in_unit_sphere(curandState *local_rand_state) {
  Vec3 p =
      2.0f * random_double(local_rand_state) - Vec3(1.0f);

  while (p.squared_length() >= 1.0f) {
    p = 2.0f * random_double(local_rand_state) - Vec3(1.0f);
  }
  return p;
}
__device__ Vec3 random_in_unit_disk(curandState *lo) {
  Vec3 p = 2.0 * Vec3(curand_uniform(lo),
                      curand_uniform(lo), 0) -
           Vec3(1, 1, 0);

  while (dot(p, p) >= 1.0) {
    p = 2.0 * Vec3(curand_uniform(lo), curand_uniform(lo),
                   0) -
        Vec3(1, 1, 0);
  }
  return p;
}

using Point3 = Vec3;
using Color = Vec3;

#endif
