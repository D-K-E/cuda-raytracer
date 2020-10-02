#ifndef KNN_CUH
#define KNN_CUH
// k nearest neighbour acceleration structure
#include <nextweek/external.hpp>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

//
class KNN {
public:
  const unsigned int k;
  Point3 centers[k];
  Hittable **groups;
  const int group_size;
  curandState *rState;
  const double err_margin;
  const int mxX, mxY, mxZ;
  static const int max_iter_size = 5000;

public:
  __device__ KNN() : k(0) {}
  __device__ KNN(unsigned int knb, Hittable **&gs,
                 double err, curandState *randState,
                 double maxX, double maxY, double maxZ,
                 int gsize)
      : k(knb), groups(gs), err_margin(err),
        rState(randState), mxX(maxX), mxY(maxY), mxZ(maxZ),
        group_size(gsize) {
    Point3 *cnts = new Point3[k];
    for (int i = 0; i < k; i++) {
      cnts[i] = random_vec(rState, mxX, mxY, mxZ);
    }
    centers = cnts;
  }

  __device__ Point3 bbox_center(int index) {
    Hittable *h = groups[index];
    Aabb temp;
    if (h->bounding_box(0, 1, temp)) {
      return temp.center();
    } else {
      return random_vec(rState, mxX, mxY, mxZ);
    }
  }

  __device__ double distance_to_point(Point3 p1, Point3 p2) {
    return distance(p1, p2);
  }
  __device__ double hittable_to_center_distance(int index,
                                               Point3 c) {
    Point3 hcenter = bbox_center(index);
    return distance_to_point(hcenter, c);
  }
};

#endif
