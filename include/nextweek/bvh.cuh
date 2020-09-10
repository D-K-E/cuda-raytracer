#ifndef BVH_CUH
#define BVH_CUH

#include <nextweek/external.hpp>
#include <nextweek/hittable.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/vec3.cuh>

__host__ __device__ void swap(Hittable **hlist,
                              int index_h1, int index_h2) {
  Hittable *temp = hlist[index_h1];
  hlist[index_h1] = hlist[index_h2];
  hlist[index_h2] = temp;
}

class BvhNode : public Hittable {
public:
  __device__ BvhNode();
  __device__ ~BvhNode() {
    delete left;
    delete right;
  }

  __device__ BvhNode(Hittables &hlist, float time0,
                     float time1)
      : BvhNode(hlist.list, hlist.list_size, time0, time1) {
  }

  __device__ BvhNode(Hittable **list, int list_size,
                     float time0, float time1);

  __device__ bool hit(const Ray &r, float tmin, float tmax,
                      HitRecord &rec) const override {
    if (!box.hit(r, t_min, t_max))
      return false;

    bool hit_left = left->hit(r, t_min, t_max, rec);
    bool hit_right =
        right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

    return hit_left || hit_right;
  }
  __device__ void odd_even_sort(Hittable **hlist,
                                int list_size) {
    bool sorted = false;
    while (!sorted) {
      sorted = true;
      for (int i = 1; i < list_size - 1; i += 2) {
        float d1 =
            distance_between_boxes(hlist[i - 1], hlist[i]);
        float d2 =
            distance_between_boxes(hlist[i + 1], hlist[i]);
        if (d2 < d1) {
          swap(hlist, i - 1, i + 1);
          sorted = false;
        }
      }
      for (int i = 2; i < list_size - 1; i += 2) {
        float d1 =
            distance_between_boxes(hlist[i - 1], hlist[i]);
        float d2 =
            distance_between_boxes(hlist[i + 1], hlist[i]);
        if (d2 < d1) {
          swap(hlist, i - 1, i + 1);
          sorted = false;
        }
      }
    }
  }

  __host__ __device__ float
  distance_between_boxes(Hittable *h1, Hittable *h2) {
    float h1_center;
    Aabb b1;
    if (!h1->bounding_box(time0, time1, b1)) {
      h1_center = h1->center;
    } else {
      h1_center = b1.center;
    }
    Aabb b2;
    if (!h2->bounding_box(time0, time1, b2)) {
      h2_center = h2->center;
    } else {
      h2_center = b2.center;
    }
    return distance(h1_center, h2_center);
  }
  __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {

    output_box = box;
    return true;
  }

public:
  Hittable *left;
  Hittable *right;
  Aabb box;
  float time0, time1;
};

#endif
