#ifndef HITTABLES_CUH
#define HITTABLES_CUH

#include <nextweek/debug.hpp>
#include <nextweek/external.hpp>
#include <nextweek/hittable.cuh>

class Hittables : public Hittable {
public:
  Hittable **list;
  int list_size;

public:
  __device__ Hittables() { list_size = 0; }
  __device__ Hittables(Hittable **hlist, int size) {
    list_size = size;
    list = hlist;
  }
  __device__ ~Hittables() { delete list; }

  __device__ bool hit(const Ray &r, float d_min,
                      float d_max,
                      HitRecord &rec) const override {
    HitRecord temp;
    bool hit_anything = false;
    float closest_far = d_max;
    for (int i = 0; i < list_size; i++) {
      Hittable *h = list[i];
      bool isHit = h->hit(r, d_min, closest_far, temp);
      if (isHit == true) {
        hit_anything = true;
        closest_far = temp.t;
        rec = temp;
      }
    }
    return hit_anything;
  }
  __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    if (list_size == 0) {
      return false;
    }
    Aabb temp;
    bool first_box = true;
    for (int i = 0; i < list_size; i++) {
      Hittable *h = list[i];
      bool isBounding = h->bounding_box(t0, t1, temp);
      if (isBounding == false) {
        return false;
      }
      output_box = first_box
                       ? temp
                       : surrounding_box(output_box, temp);
      first_box = false;
      // center = output_box.center;
    }
    return true;
  }
};

#endif
