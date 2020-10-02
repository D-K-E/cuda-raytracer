#ifndef SPHERE_CUH
#define SPHERE_CUH

#include <oneweekend/hittable.cuh>
#include <oneweekend/material.cuh>

class Sphere : public Hittable {
public:
  __device__ Sphere() {}
  __device__ Sphere(Point3 cen, double r, Material *mat_ptr_)
      : center(cen), radius(r), mat_ptr(mat_ptr_){};
  __device__ ~Sphere() { delete mat_ptr; }
  __device__ bool hit(const Ray &r, double d_min,
                      double d_max,
                      HitRecord &rec) const override {
    Vec3 oc = r.origin() - center;
    double a = dot(r.direction(), r.direction());
    double b = dot(oc, r.direction());
    double c = dot(oc, oc) - radius * radius;
    double discriminant = b * b - a * c;
    if (discriminant > 0) {
      double temp = (-b - sqrt(discriminant)) / a;
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
  double radius;
  Material *mat_ptr;
};

class MovingSphere : public Hittable {
public:
  Point3 center1, center2;
  double time0, time1, radius;
  Material *mat_ptr;

public:
  __device__ MovingSphere();
  __device__ ~MovingSphere() { delete mat_ptr; };
  __device__ MovingSphere(Point3 c1, Point3 c2, double t0,
                          double t1, double rad,
                          Material *mat)
      : center1(c1), center2(c2), time0(t0), time1(t1),
        radius(rad), mat_ptr(mat) {}
  __device__ Point3 center(double time) const {
    return center1 +
           ((time - time0) / (time1 - time0)) *
               (center2 - center1);
  }
  __device__ bool hit(const Ray &r, double d_min,
                      double d_max,
                      HitRecord &rec) const override {
    Point3 scenter = center(r.time());
    Vec3 oc = r.origin() - scenter;
    double a = dot(r.direction(), r.direction());
    double b = dot(oc, r.direction());
    double c = dot(oc, oc) - radius * radius;
    double discriminant = b * b - a * c;
    if (discriminant > 0) {
      double temp = (-b - sqrt(discriminant)) / a;
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
};

#endif
