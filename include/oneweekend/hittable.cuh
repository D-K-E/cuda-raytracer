#ifndef HITTABLE_CUH
#define HITTABLE_CUH

#include <oneweekend/ray.cuh>

class Material;

struct HitRecord {
  double t;
  Point3 p;
  Vec3 normal;
  Material *mat_ptr;
};

class Hittable {
public:
  Material *mat_ptr;

public:
  __device__ virtual bool hit(const Ray &r, double d_min,
                              double d_max,
                              HitRecord &rec) const = 0;
};

#endif
