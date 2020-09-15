#ifndef BOX_CUH
#define BOX_CUH
#include <nextweek/aarect.cuh>
#include <nextweek/hittables.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

class Box : public Hittable {
public:
  __device__ Box() {}
  __device__ ~Box() { delete sides; }
  __device__ Box(const Point3 &p0, const Point3 &p1,
                 Material *mptr)
      : sides(new Hittables(hittable_size)) {
    box_min = p0;
    box_max = p1;

    // sides[0] =

    sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(),
                                   p1.y(), p1.z(), ptr));
    sides.add(make_shared<xy_rect>(p0.x(), p1.x(), p0.y(),
                                   p1.y(), p0.z(), ptr));

    sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(),
                                   p1.z(), p1.y(), ptr));
    sides.add(make_shared<xz_rect>(p0.x(), p1.x(), p0.z(),
                                   p1.z(), p0.y(), ptr));

    sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(),
                                   p1.z(), p1.x(), ptr));
    sides.add(make_shared<yz_rect>(p0.y(), p1.y(), p0.z(),
                                   p1.z(), p0.x(), ptr));
  }

  __device__ bool hit(const Ray &r, float t0, float t1,
                      HitRecord &rec) const override;

  __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    output_box = Aabb(box_min, box_max);
    return true;
  }

public:
  Point3 box_min;
  Point3 box_max;
  Hittables *sides;
  int hittable_size = 6;
};

bool box::hit(const ray &r, double t0, double t1,
              hit_record &rec) const {
  return sides.hit(r, t0, t1, rec);
}

#ifndef
