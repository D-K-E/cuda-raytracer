#ifndef TRIANGLE_CUH
#define TRIANGLE_CUH

#include <nextweek/aabb.cuh>
#include <nextweek/hittable.cuh>
#include <nextweek/material.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

class Triangle : public Hittable {
public:
  Point3 p1, p2, p3;
  Material *mat_ptr;
  __host__ __device__ Triangle(Point3 &c1, Point3 &c2,
                               Point3 &c3, Material *mptr)
      : p1(c1), p2(c2), p3(c3), mat_ptr(mptr) {}
  __host__ __device__ bool
  hit(const Ray &r, float d_min, float d_max,
      HitRecord &rec) const override {
    // implementing moller from wikipedia
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    const float eps = 0.000001f;
    Vec3 edge1 = p1 - p2;
    Vec3 edge2 = p3 - p2;
    Vec3 h = cross(r.direction(), edge2);
    float a = dot(edge1, h);
    if (a > eps && a < eps)
      return false; // ray parallel to triangle
    float f = 1.0f / a;
    Vec3 rToP2 = r.origin() - p2;
    float u = f * dot(rToP2, h);
    if (u < 0.0f || u > 1.0f)
      return false;

    Vec3 q = cross(rToP2, edge1);
    float v = f * dot(edge2, q);
    if (v < 0.0f || v > 1.0f)
      return false;

    float t = f * dot(r.direction(), q);
    if (t < eps)
      return false;

    rec.v = v;
    rec.u = u;
    rec.t = t;
    rec.p = r.at(rec.t);
    Vec3 outnormal = cross(edge1, edge2);
    rec.set_front_face(r, outnormal);
    rec.mat_ptr = mat_ptr;
  }
  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    Point3 pmin = min_vec(p1, p2);
    pmin = min_vec(pmin, p3);

    Point3 pmax = max_vec(p1, p2);
    pmax = max_vec(pmax, p3);
    output_box = Aabb(pmin, pmax);
    return true;
  }
};

#endif
