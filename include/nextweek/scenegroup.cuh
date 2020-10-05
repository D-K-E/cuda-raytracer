#ifndef SCENEGROUP_CUH
#define SCENEGROUP_CUH

#include <nextweek/aabb.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/sceneparam.cuh>
#include <nextweek/sceneprim.cuh>
#include <nextweek/scenetype.cuh>
#include <nextweek/vec3.cuh>

struct SceneGroup {
  int group_id; // for instances and different mediums
  GroupType group_type;
  int group_size;
  ScenePrimitive *prims;
  Aabb box;
  Point3 center;
  __host__ __device__ SceneGroup() {}
  __host__ __device__ SceneGroup(int gid, int gsize,
                                 GroupType gtype,
                                 ScenePrimitive *&ps)
      : group_id(gid), group_size(gsize), group_type(gtype),
        prims(ps) {
    order_scene(ps, gsize);
    Aabb b;
    prims[0].h->bounding_box(0.0f, 0.0f, b);
    for (int i = 1; i < group_size; i++) {
      Aabb temp;
      prims[i].h->bounding_box(0.0f, 0.0f, temp);
      b = surrounding_box(temp, b);
    }
    box = b;
    center = box.center();
  }
  __host__ __device__ HittableGroup *to_instance() {
    Hittable **hs = new Hittable *[group_size];
    for (int i = 0; i < group_size; i++) {
      ScenePrimitive p = prims[i];
      Hittable *h;
      p.to_obj(h);
      hs[i] = h;
    }
    HittableGroup *g = new HittableGroup(hs, group_size);
    return g;
  }
  __host__ __device__ void to_constant_medium() {}
  __host__ __device__ HittableGroup *to_hittable_group() {
    HittableGroup *group;
    switch (group_type) {
    case INSTANCE:
      group = to_instance();
      break;
    case CONSTANT_MEDIUM:
      break;
    }
    return group;
  }
};

__host__ __device__ int farthest_index(const SceneGroup &g,
                                       SceneGroup *&gs,
                                       int nb_group) {
  float max_dist = FLT_MIN;
  int max_dist_index = 0;
  Point3 g_center = g.center;
  for (int i = 0; i < nb_group; i++) {
    Point3 scene_center = gs[i].center;
    float dist = distance(g_center, scene_center);
    if (dist > max_dist) {
      max_dist = dist;
      max_dist_index = i;
    }
  }
  return max_dist_index;
}

// implementing list structure from
// Foley et al. 2013, p. 1081
__host__ __device__ void order_scene(SceneGroup *&gs,
                                     int nb_group) {
  for (int i = 0; i < nb_group - 1; i += 2) {
    SceneGroup g = gs[i];
    int fgi = farthest_index(g, gs, nb_group);
    swap(gs, i + 1, fgi);
  }
}

#endif
