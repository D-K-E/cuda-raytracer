// DBSCAN Acceleration Structure
#ifndef DBSCAN_CUH
#define DBSCAN_CUH

// k nearest neighbour acceleration structure
#include <nextweek/external.hpp>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

class DBSCAN {
public:
  curandState *rState;
  Hittable **groups;
  int *labels;
  const int group_size;
  const float threshold;
  const float min_neighbor_nb;
  const int mxX, mxY, mxZ;

public:
  __device__ Point3 bbox_center(int index) {
    Hittable *h = groups[index];
    Aabb temp;
    if (h->bounding_box(0, 1, temp)) {
      return temp.center();
    } else {
      return random_vec(rState, mxX, mxY, mxZ);
    }
  }

  __device__ int neighbour_number(int obj_index) {
    Point3 hcenter = bbox_center(obj_index);
    int counter = 0;
    for (int i = 0; i < obj_index; i++) {
      Point3 nc = bbox_center(i);
      if (distance(hcenter, nc) < threshold) {
        counter++;
      }
    }
    for (int i = obj_index + 1; i < group_size; i++) {
      Point3 nc = bbox_center(i);
      if (distance(hcenter, nc) < min_distance) {
        counter++;
      }
    }
    return counter;
  }
  __device__ void grow_set(int &index, int&label);
  __device__ void unite_same_label_hittables(int label);
};

#endif
