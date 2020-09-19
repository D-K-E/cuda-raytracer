#ifndef BOX_CUH
#define BOX_CUH
#include <nextweek/aarect.cuh>
#include <nextweek/hittables.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

class Box : public Hittable {
public:
  __host__ __device__ Box() {}
  __host__ __device__ ~Box() { delete sides; }
  __host__ __device__ Box(const Point3 &p0,
                          const Point3 &p1,
                          Material *mptr) {
    box_min = p0;
    box_max = p1;

    // sides[0] =
    Hittable *hsides[6];

    hsides[0] = new XYRect(p0.x(), p1.x(), p0.y(), p1.y(),
                           p1.z(), mptr);
    hsides[1] = new XYRect(p0.x(), p1.x(), p0.y(), p1.y(),
                           p0.z(), mptr);

    hsides[2] = new XZRect(p0.x(), p1.x(), p0.z(), p1.z(),
                           p1.y(), mptr);
    hsides[3] = new XZRect(p0.x(), p1.x(), p0.z(), p1.z(),
                           p0.y(), mptr);

    hsides[4] = new YZRect(p0.y(), p1.y(), p0.z(), p1.z(),
                           p1.x(), mptr);
    hsides[5] = new YZRect(p0.y(), p1.y(), p0.z(), p1.z(),
                           p0.x(), mptr);

    sides =
        new Hittables((Hittable **)&hsides, hittable_size);
  }

  __host__ __device__ bool
  hit(const Ray &r, float t0, float t1,
      HitRecord &rec) const override {
    return sides->hit(r, t0, t1, rec);
  }

  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    output_box = Aabb(box_min, box_max);
    return true;
  }

public:
  Point3 box_min;
  Point3 box_max;
  const int hittable_size = 6;
  Hittables *sides;
};

#endif
