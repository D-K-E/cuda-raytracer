#ifndef SPHERE_CUH
#define SPHERE_CUH

#include <oneweekend/hittable.cuh>

class Sphere : public Hittable {
public:
  __device__ Sphere() {}
  __device__ Sphere(Vec3 cen, float r) : center(cen), radius(r){};
  __device__ bool hit(const Ray &r, float d_min, float d_max,
                      HitRecord &rec) const override {
    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
      float temp = (-b - sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;
        rec.mat_ptr = mat_ptr;
        return true;
      }
      temp = (-b + sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;
        rec.mat_ptr = mat_ptr;
        return true;
      }
    }
    return false;
  }

public:
  Vec3 center;
  float radius;
  Material * mat_ptr;
};

#endif
