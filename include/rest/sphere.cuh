#ifndef SPHERE_CUH
#define SPHERE_CUH

#include <rest/aabb.cuh>
#include <rest/hittable.cuh>
#include <rest/material.cuh>
#include <rest/utils.cuh>

__host__ __device__ void get_sphere_uv(const Vec3 &p,
                                       float &u, float &v) {
  auto phi = atan2(p.z(), p.x());
  auto theta = asin(p.y());
  u = 1 - (phi + M_PI) / (2 * M_PI);
  v = (theta + M_PI / 2) / M_PI;
}

class Sphere : public Hittable {
public:
  __host__ __device__ Sphere() {}
  __host__ __device__ Sphere(Point3 cen, float r,
                             Material *mat_ptr_)
      : center(cen), radius(r), mat_ptr(mat_ptr_){};
  __host__ __device__ ~Sphere() { delete mat_ptr; }
  __device__ bool hit(const Ray &r, float d_min,
                      float d_max,
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
        Vec3 normal = (rec.p - center) / radius;
        rec.set_front_face(r, normal);
        get_sphere_uv(normal, rec.u, rec.v);
        rec.mat_ptr = mat_ptr;
        return true;
      }
      temp = (-b + sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        Vec3 normal = (rec.p - center) / radius;
        rec.set_front_face(r, normal);
        get_sphere_uv(normal, rec.u, rec.v);
        rec.mat_ptr = mat_ptr;
        return true;
      }
    }
    return false;
  }
  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    output_box =
        Aabb(center - Vec3(radius), center + Vec3(radius));
    return true;
  }

public:
  Vec3 center;
  float radius;
  Material *mat_ptr;
};

class MovingSphere : public Hittable {
public:
  Point3 center1, center2;
  float time0, time1, radius;
  Material *mat_ptr;

public:
  __host__ __device__ MovingSphere();
  __host__ __device__ ~MovingSphere() { delete mat_ptr; };
  __host__ __device__ MovingSphere(Point3 c1, Point3 c2,
                                   float t0, float t1,
                                   float rad, Material *mat)
      : center1(c1), center2(c2), time0(t0), time1(t1),
        radius(rad), mat_ptr(mat) {}
  __host__ __device__ Point3 center(float time) const {
    return center1 +
           ((time - time0) / (time1 - time0)) *
               (center2 - center1);
  }
  __device__ bool hit(const Ray &r, float d_min,
                      float d_max,
                      HitRecord &rec) const override {
    Point3 scenter = center(r.time());
    Vec3 oc = r.origin() - scenter;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
      float temp = (-b - sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - scenter) / radius;
        rec.mat_ptr = mat_ptr;
        return true;
      }
      temp = (-b + sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - scenter) / radius;
        rec.mat_ptr = mat_ptr;
        return true;
      }
    }
    return false;
  }

  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    // NOT CORRECT
    output_box = Aabb(center1 - Vec3(radius),
                      center1 + Vec3(radius));
    return true;
  }
};

#endif
