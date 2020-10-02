#ifndef HITTABLES_CUH
#define HITTABLES_CUH

#include <nextweek/debug.hpp>
#include <nextweek/external.hpp>
#include <nextweek/hittable.cuh>

class HittableGroup : public Hittable {
public:
  Hittable **list;
  const int start_index;
  const int end_index;

public:
  __host__ __device__ HittableGroup()
      : start_index(0), end_index(0), list(nullptr) {}
  __host__ __device__ HittableGroup(Hittable **&hlist,
                                    int size)
      : list(hlist), start_index(0), end_index(size) {}
  __host__ __device__ HittableGroup(Hittable **&hlist,
                                    int si, int ei)
      : list(hlist), start_index(si), end_index(ei) {}

  __host__ __device__ ~HittableGroup() { delete list; }

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
  __device__ bool hit(const Ray &r, double d_min,
                      double d_max,
                      HitRecord &rec) const override {
    return hit_to_hittables(list, start_index, end_index, r,
                            d_min, d_max, rec);
  }

  /**
   @brief compute bounding box of the entire scene

   See the documentation of Hittable"::"bounding_box()

   @param t0 time0
   @param t1 time1
   */
  __host__ __device__ bool
  bounding_box(double t0, double t1,
               Aabb &output_box) const override {
    return bounding_box_to_hittables(
        list, start_index, end_index, t0, t1, output_box);
  }
};
class SafeHittableGroup : public HittableGroup {
public:
  __host__ __device__ SafeHittableGroup()
      : HittableGroup() {}
  __host__ __device__ SafeHittableGroup(Hittable **hlist,
                                        int size)
      : HittableGroup(hlist, size) {}
  __host__ __device__ SafeHittableGroup(Hittable **hlist,
                                        int si, int ei)
      : HittableGroup(hlist, si, ei) {}
};

/**
  @brief Scene object that gathers all of the object in the
  scene.

  Currently this is just a list of Hittable objects with a
  current number of
  objects.
 */
class Hittables : public SafeHittableGroup {

public:
  __host__ __device__ Hittables() {}
  __host__ __device__ Hittables(Hittable **hlist, int size)
      : SafeHittableGroup(hlist, size) {}
};

#endif
