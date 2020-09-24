#ifndef MEDIUMC_CUH
#define MEDIUMC_CUH

#include <nextweek/external.hpp>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

#include <nextweek/hittable.cuh>
#include <nextweek/material.cuh>
#include <nextweek/texture.cuh>

class ConstantMedium : public Hittable {
public:
  __host__ __device__ ConstantMedium(Hittable **&b, float d,
                                     Texture *a, int s,
                                     int e)
      : boundary(b), neg_inv_density(-1 / d),
        phase_function(new Isotropic(a)), start_index(s),
        end_index(e) {}

  __device__ ConstantMedium(Hittable **&b, float d,
                            Texture *a, curandState *s,
                            int si, int ei)
      : boundary(b), neg_inv_density(-1 / d),
        phase_function(new Isotropic(a)), rState(s),
        start_index(si), end_index(ei) {}

  __host__ __device__ ConstantMedium(Hittable **&b, float d,
                                     Color c, int s, int e)
      : boundary(b), neg_inv_density(-1 / d),
        phase_function(new Isotropic(c)), start_index(s),
        end_index(e) {}

  __device__ ConstantMedium(Hittable **b, float d, Color c,
                            curandState *s, int si, int ei)
      : boundary(b), neg_inv_density(-1 / d),
        phase_function(new Isotropic(c)), rState(s),
        start_index(si), end_index(ei) {}

  __device__ bool hit(const Ray &r, float t_min,
                      float t_max,
                      HitRecord &rec) const override {
    //
    // Print occasional samples when debugging. To enable,
    // set enableDebug true.
    const bool enableDebug = false;

    const bool debugging =
        enableDebug && curand_uniform(rState) < 0.00001;

    HitRecord rec1, rec2;

    if (!bound_hit(r, -FLT_MAX, FLT_MAX, rec1))
      return false;

    if (!bound_hit(r, rec1.t + 0.0001f, FLT_MAX, rec2))
      return false;

    if (debugging) {
      printf("\nt0= %f", rec1.t);
      printf(", t1= %f\n", rec2.t);
    }

    if (rec1.t < t_min)
      rec1.t = t_min;
    if (rec2.t > t_max)
      rec2.t = t_max;

    if (rec1.t >= rec2.t)
      return false;

    if (rec1.t < 0)
      rec1.t = 0;

    const float ray_length = r.direction().length();
    const float distance_inside_boundary =
        (rec2.t - rec1.t) * ray_length;
    const float hit_distance =
        neg_inv_density * log(curand_uniform(rState));

    if (hit_distance > distance_inside_boundary)
      return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    if (debugging) {
      printf("hit_distance = %f\n", hit_distance);
      printf("rec.t = %f\n", rec.t);
      printf("rec.p = %f ", rec.p.x());
      printf("%f ", rec.p.y());
      printf("%f ", rec.p.z());
    }

    rec.normal = Vec3(1, 0, 0); // arbitrary
    rec.front_face = true;      // also arbitrary
    rec.mat_ptr = phase_function;

    return true;
  }

  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    Aabb temp;
    bool isBound =
        boundary[start_index]->bounding_box(t0, t1, temp);
    for (int i = start_index + 1; i <= end_index; i++) {
      Aabb temp2;
      if (boundary[i]->bounding_box(t0, t1, temp2))
        isBound = true;
      temp = surrounding_box(temp, temp2);
    }
    output_box = temp;
    return isBound;
  }
  __device__ bool bound_hit(const Ray &r, float t_min,
                            float t_max,
                            HitRecord &rec) const {
    HitRecord temp;
    bool hit_anything = false;
    float closest_far = t_max;
    for (int i = start_index; i <= end_index; i++) {
      Hittable *h = boundary[i];
      bool isHit = h->hit(r, t_min, closest_far, temp);
      if (isHit == true) {
        hit_anything = true;
        closest_far = temp.t;
        rec = temp;
      }
    }
    return hit_anything;
  }

public:
  Hittable **boundary;
  Material *phase_function;
  float neg_inv_density;
  curandState *rState;
  int start_index, end_index;
};

#endif
