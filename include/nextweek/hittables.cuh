#ifndef HITTABLES_CUH
#define HITTABLES_CUH

#include <nextweek/debug.hpp>
#include <nextweek/external.hpp>
#include <nextweek/hittable.cuh>

/**
  @brief Scene object that gathers all of the object in the
  scene.

  Currently this is just a list of Hittable objects with a
  current number of
  objects.
 */
class Hittables : public Hittable {
public:
  Hittable **list;
  int list_size;

public:
  __device__ Hittables() { list_size = 0; }
  __device__ Hittables(Hittable **hlist, int size)
      : list(hlist), list_size(size) {}
  __device__ ~Hittables() { delete list; }

  /**
    @brief check if there is any object that is hit by
    incoming ray.

    The main method that checks if a ray hits something.
    It is in most cases the most time consuming part of the
    algorithm due to traversing of the entire scene. Most of
    the acceleration structures try to find ways of
    iterating the scene more efficiently. They thus try to
    come up with a way to circumvent the simple iteration
    that we are doing here. I have also implemented a couple
    of them in cpu. The general strategy seems to be
    implement and construct everything on cpu and traverse
    everything on gpu.

    See also the documentation of Hittable"::"hit()

    @param r incoming ray. Essentially this the ray that is
    either generated
    by the camera or that is scattered from a surface.

    @param d_min minimum distance.
    @param d_max maximum distance
    @param rec struct that would hold all the necessary
    information for evaluating a bsdf.
   */
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

  /**
   @brief compute bounding box of the entire scene

   See the documentation of Hittable"::"bounding_box()

   @param t0 time0
   @param t1 time1
   */
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
