#ifndef BOX_CUH
#define BOX_CUH
#include <nextweek/aarect.cuh>
#include <nextweek/hittables.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

class Box : public HittableGroup {
public:
  __host__ __device__ Box() : HittableGroup() {}
  __host__ __device__ Box(const Point3 &p0,
                          const Point3 &p1, Material *mptr,
                          Hittable **&ss, int &k)
      : HittableGroup(ss, k, k + hittable_size) {
    box_min = p0;
    box_max = p1;

    ss[k] = new XYRect(p0.x(), p1.x(), p0.y(), p1.y(),
                       p1.z(), mptr);
    k++;
    ss[k] = new XYRect(p0.x(), p1.x(), p0.y(), p1.y(),
                       p0.z(), mptr);
    k++;
    ss[k] = new XZRect(p0.x(), p1.x(), p0.z(), p1.z(),
                       p1.y(), mptr);
    k++;
    ss[k] = new XZRect(p0.x(), p1.x(), p0.z(), p1.z(),
                       p0.y(), mptr);

    k++;
    ss[k] = new YZRect(p0.y(), p1.y(), p0.z(), p1.z(),
                       p1.x(), mptr);
    k++;
    ss[k] = new YZRect(p0.y(), p1.y(), p0.z(), p1.z(),
                       p0.x(), mptr);
    list = ss;
  }

  __host__ __device__ void translate(Hittable **&ss,
                                     const Point3 &p) {
    for (int i = start_index; i < end_index; i++) {
      list[i] = new Translate(list[i], p);
      ss[i] = list[i];
    }
  }
  __host__ __device__ void rotate_y(Hittable **&ss,
                                    float angle) {
    for (int i = start_index; i < end_index; i++) {
      list[i] = new RotateY(list[i], angle);
      ss[i] = list[i];
    }
  }

public:
  Point3 box_min;
  Point3 box_max;
  static const int hittable_size = 6;
};

#endif
