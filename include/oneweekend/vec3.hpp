// vec3.hpp for cuda
#ifndef VEC3_HPP
#define VEC3_HPP

#include <iostream>
#include <math.h>
#include <stdlib.h>

class Vec3 {
public:
  float e[3];

  __host__ __device__ Vec3() {}
  __host__ __device__ Vec3(float e1, float e2, float e3) {
    e[0] = e1;
    e[1] = e2;
    e[2] = e3;
  }
  __host__ __device__ Vec3(float e1) {
    e[0] = e1;
    e[1] = e1;
    e[2] = e1;
  }
  __host__ __device__ Vec3(float es[3]) {
    e[0] = es[0];
    e[1] = es[1];
    e[2] = e[2];
  }
  __host__ __device__ inline float x() const { return e[0]; }
  __host__ __device__ inline float y() const { return e[1]; }
  __host__ __device__ inline float z() const { return e[2]; }
  __host__ __device__ inline float r() const { return x(); }
  __host__ __device__ inline float g() const { return y(); }
  __host__ __device__ inline float b() const { return z(); }

  __host__ __device__ inline Vec3 operator-() const {
    return Vec3(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ inline float operator[](int i) const { return e[i]; }

  __host__ __device__ inline Vec3 &operator+=(const Vec3 &v);
  __host__ __device__ inline Vec3 &operator-=(const Vec3 &v);
  __host__ __device__ inline Vec3 &operator*=(const Vec3 &v);
  __host__ __device__ inline Vec3 &operator/=(const Vec3 &v);
  __host__ __device__ inline Vec3 &operator*=(const float t);
  __host__ __device__ inline Vec3 &operator/=(const float t);
  __host__ __device__ inline Vec3 &operator+=(const float t);
  __host__ __device__ inline Vec3 &operator-=(const float t);

  __host__ __device__ inline float squared_length() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }
  __host__ __device__ inline float length() const {
    return sqrt(squared_length());
  }
  __host__ __device__ inline Vec3 to_unit() const;
  __host__ __device__ inline void unit_vector() const;
};

inline std::ostream &operator<<(std::ostream &os, const Vec3 &t) {
  os << t.x() << " " << t.y() << " " << t.z();
  return os;
}

__host__ __device__ inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z());
}

__host__ __device__ inline Vec3 operator-(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z());
}
__host__ __device__ inline Vec3 operator*(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.x() * v2.x(), v1.y() * v2.y(), v1.z() * v2.z());
}
__host__ __device__ inline Vec3 operator/(const Vec3 &v1, const Vec3 &v2) {
  return Vec3(v1.x() / v2.x(), v1.y() / v2.y(), v1.z() / v2.z());
}
__host__ __device__ inline Vec3 operator*(const Vec3 &v1, float t) {
  return Vec3(v1.x() * t, v1.y() * t, v1.z() * t);
}
__host__ __device__ inline Vec3 operator/(const Vec3 &v1, float t) {
  return Vec3(v1.x() / t, v1.y() / t, v1.z() / t);
}
__host__ __device__ inline Vec3 operator+(const Vec3 &v1, float t) {
  return Vec3(v1.x() + t, v1.y() + t, v1.z() + t);
}
__host__ __device__ inline Vec3 operator-(const Vec3 &v1, float t) {
  return Vec3(v1.x() - t, v1.y() - t, v1.z() - t);
}

__host__ __device__ inline float dot(const Vec3 &v1, const Vec3 &v2) {
  return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

__host__ __device__ inline Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
  return Vec3((v1.y() * v2.z() - v1.z() * v2.y()),
              (-(v1.x() * v2.z() - v1.z() * v2.x())),
              (v1.x() * v2.y() - v1.y() * v2.x()));
}

__host__ __device__ inline Vec3 &Vec3::operator+=(const Vec3 &v) {
  e[0] += v.x();
  e[1] += v.y();
  e[2] += v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator*=(const Vec3 &v) {
  e[0] *= v.x();
  e[1] *= v.y();
  e[2] *= v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator/=(const Vec3 &v) {
  e[0] /= v.x();
  e[1] /= v.y();
  e[2] /= v.z();
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator-=(const Vec3 &v) {
  e[0] -= v.x();
  e[1] -= v.y();
  e[2] -= v.z();
  return *this;
}

__host__ __device__ inline Vec3 &Vec3::operator+=(float v) {
  e[0] += v;
  e[1] += v;
  e[2] += v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator-=(float v) {
  e[0] -= v;
  e[1] -= v;
  e[2] -= v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator*=(float v) {
  e[0] *= v;
  e[1] *= v;
  e[2] *= v;
  return *this;
}
__host__ __device__ inline Vec3 &Vec3::operator/=(float v) {
  e[0] /= v;
  e[1] /= v;
  e[2] /= v;
  return *this;
}
__host__ __device__ inline Vec3 to_unit(Vec3 v) { return v / v.length(); }

using Point3 = Vec3;
using Color = Vec3;

#endif
