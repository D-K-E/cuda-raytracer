#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include <nextweek/aabb.cuh>
#include <nextweek/ray.cuh>

class Material;

struct HitRecord {
  float t;
  Point3 p;
  Vec3 normal;
  Material *mat_ptr;
  float u, v;
  bool front_face;
  __device__ void set_front_face(const Ray &r,
                                 const Vec3 &norm) {
    front_face = dot(r.direction(), norm) < 0.0f;
    normal = front_face ? norm : -norm;
  }
};

class Hittable {
public:
  Material *mat_ptr;
  Point3 center;

public:
  __device__ virtual bool hit(const Ray &r, float d_min,
                              float d_max,
                              HitRecord &rec) const = 0;
  __device__ virtual bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const = 0;
};

#endif
