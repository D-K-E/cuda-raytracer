// vec3.hpp for cuda
#ifndef VEC3_CUH
#define VEC3_CUH

#include <nextweek/external.hpp>
#include <nextweek/utils.cuh>

class Vec3 {
public:
  double e[3];

  __host__ __device__ Vec3() {}
  __host__ __device__ Vec3(double e1, double e2,
                           double e3) {
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
  __host__ __device__ inline double r() const {
    return x();
  }
  __host__ __device__ inline double g() const {
    return y();
  }
  __host__ __device__ inline double b() const {
    return z();
  }

  __host__ __device__ inline Vec3 operator-() const {
    return Vec3(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ inline double
  operator[](int i) const {
    return e[i];
  }
  __host__ __device__ inline double &operator[](int i) {
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
  __host__ std::vector<double> to_v() const {
    std::vector<double> v(3);
    v[0] = x();
    v[1] = y();
    v[2] = z();
    return v;
  }
  __host__ __device__ inline static Vec3
  random(unsigned int seed) {
    return Vec3(randf(seed), randf(seed), randf(seed));
  }
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

__host__ __device__ inline Vec3 &Vec3::
operator+=(double v) {
  e[0] += v;
  e[1] += v;
  e[2] += v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator-=(double v) {
  e[0] -= v;
  e[1] -= v;
  e[2] -= v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator*=(double v) {
  e[0] *= v;
  e[1] *= v;
  e[2] *= v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::
operator/=(double v) {
  e[0] /= v;
  e[1] /= v;
  e[2] /= v;
  return *this;
}
__host__ __device__ inline Vec3 to_unit(Vec3 v) {
  return v / v.length();
}
__host__ __device__ inline double distance(Vec3 v1,
                                           Vec3 v2) {
  return (v1 - v2).length();
}

__host__ __device__ Vec3 min_vec(const Vec3 &v1,
                                 const Vec3 &v2) {
  double xmin = fmin(v1.x(), v2.x());
  double ymin = fmin(v1.y(), v2.y());
  double zmin = fmin(v1.z(), v2.z());
  return Vec3(xmin, ymin, zmin);
}
__host__ __device__ Vec3 max_vec(const Vec3 v1,
                                 const Vec3 v2) {
  double xmax = fmax(v1.x(), v2.x());
  double ymax = fmax(v1.y(), v2.y());
  double zmax = fmax(v1.z(), v2.z());
  return Vec3(xmax, ymax, zmax);
}

#define RND (curand_uniform(&local_rand_state))

__device__ double random_double(curandState *loc,
                                double min, double max) {
  return min + (max - min) * curand_uniform_double(loc);
}
__device__ int random_int(curandState *loc) {
  return (int)curand_uniform_double(loc);
}
__device__ int random_int(curandState *loc, int mn,
                          int mx) {
  return (int)random_double(loc, (double)mn, (double)mx);
}

__device__ Vec3 random_vec(curandState *local_rand_state) {
  return Vec3(curand_uniform_double(local_rand_state),
              curand_uniform_double(local_rand_state),
              curand_uniform_double(local_rand_state));
}
__device__ Vec3 random_vec(curandState *local_rand_state,
                           double mn, double mx) {
  return Vec3(random_double(local_rand_state, mn, mx),
              random_double(local_rand_state, mn, mx),
              random_double(local_rand_state, mn, mx));
}
__device__ Vec3 random_vec(curandState *local_rand_state,
                           double mx, double my,
                           double mz) {
  return Vec3(random_double(local_rand_state, 0, mx),
              random_double(local_rand_state, 0, my),
              random_double(local_rand_state, 0, mz));
}

__device__ Vec3
random_in_unit_sphere(curandState *local_rand_state) {
  while (true) {
    Vec3 p =
        2.0f * random_vec(local_rand_state, -1.0f, 1.0f) -
        Vec3(1.0f);
    if (p.squared_length() < 1.0f)
      return p;
  }
}
__device__ Vec3 random_in_unit_disk(curandState *lo) {
  while (true) {
    Vec3 p = 2.0 * Vec3(random_double(lo, -1.0f, 1.0f),
                        random_double(lo, -1.0f, 1.0f), 0) -
             Vec3(1, 1, 0);
    if (p.squared_length() < 1.0f)
      return p;
  }
}

__device__ Vec3 random_in_hemisphere(curandState *lo,
                                     Vec3 normal) {
  Vec3 in_unit_sphere = random_in_unit_sphere(lo);
  if (dot(in_unit_sphere, normal) > 0.0)
    return in_unit_sphere;
  else
    return -in_unit_sphere;
}

using Point3 = Vec3;
using Color = Vec3;

#endif
