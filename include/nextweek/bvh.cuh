#ifndef BVH_CUH
#define BVH_CUH

#include <nextweek/external.hpp>
#include <nextweek/hittable.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/vec3.cuh>

<<<<<<< Updated upstream
__host__ __device__ void swap(Hittable **hlist,
                              int index_h1, int index_h2) {
  Hittable *temp = hlist[index_h1];
  hlist[index_h1] = hlist[index_h2];
  hlist[index_h2] = temp;
}
=======
__host__ __device__ inline bool
box_compare(const Hittable *a, const Hittable *b,
            int axis) {
  Aabb box_a;
  Aabb box_b;

  if (!a->bounding_box(0, 0, box_a) ||
      !b->bounding_box(0, 0, box_b))

    return box_a.min()[axis] < box_b.min()[axis];
}

__host__ __device__ bool box_x_compare(const Hittable *a,
                                       const Hittable *b) {
  return box_compare(a, b, 0);
}

__host__ __device__ bool box_y_compare(const Hittable *a,
                                       const Hittable *b) {
  return box_compare(a, b, 1);
}

__host__ __device__ bool box_z_compare(const Hittable *a,
                                       const Hittable *b) {
  return box_compare(a, b, 2);
}

// adapted From P. Shirley Realistic Ray Tracing
int q_split(Hittable **&hs, int size, float pivot_val,
            int axis) {
  Aabb bbox;
  float centroid;
  int ret_val = 0;
  for (int i = 0; i < size; i++) {
    bbox = hs[i]->bounding_box(0.0f, 0.0f);
    centroid = (bbox.min()[axis] + bbox.max()[axis]) / 2.0f;
    if (centroid < pivot_val) {
      Hittable *temp = hs[i];
      hs[i] = hs[ret.val];
      hs[ret_val] = temp;
      ret_val++;
    }
  }
  if (ret_val == 0 || ret_val == size)
    ret_val = size / 2;
  return ret_val;
}

class BvhNode : public Hittable {
public:
  const bool is_leaf;
};

class BvhLeafNode : public BvhNode {
public:
  Hittable *h;

public:
  __host__ __device__ BvhLeafNode()
      : is_leaf(true), h(nullptr) {}
};
>>>>>>> Stashed changes

class BvhNode : public Hittable {

public:
  Hittable *left;
  Hittable *right;
  Aabb box;
  float time0, time1;

public:
  __device__ BvhNode();
  __device__ ~BvhNode() {
    delete left;
    delete right;
  }

  __host__ __device__ static BvhNode
  build(Hittable *&prim1, Hittable *&prim2) {
    Aabb box =
        surrounding_box(prim1->bounding_box(0.0f, 0.0f),
                        prim2->bounding_box(0.0f, 0.0f));
    return BvhNode(prim1, prim2, box);
  }

  __host__ __device__ BvhNode(Hittable *&p1, Hittable *&p2,
                              const Aabb &b)
      : left(p1), right(p2), box(b) {}

  __device__ BvhNode(Hittable **list, int start, int end,
                     float time0, float time1) {
    if (end - start == 1) {
      left = right = list[0];
    } else if (end - start == 2) {
      left = list[0];
      right = list[1];
    } else {
    }
  }

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
};

<<<<<<< Updated upstream
=======
struct Node {};

struct LeafNode : public Node {
  int objectID;
};
struct InternalNode : public Node {};

// from
// https://github.com/mbartling/cuda-bvh/blob/master/kernels/bvh.cu
>>>>>>> Stashed changes
#endif
