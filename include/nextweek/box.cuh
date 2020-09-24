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
                          const Point3 &p1, Material *mptr,
                          Hittable **&ss, int &i) {
    box_min = p0;
    box_max = p1;
    const int sindex = i;

    start_index = sindex;

    ss[i] = new XYRect(p0.x(), p1.x(), p0.y(), p1.y(),
                       p1.z(), mptr);
    i++;
    ss[i] = new XYRect(p0.x(), p1.x(), p0.y(), p1.y(),
                       p0.z(), mptr);
    i++;
    ss[i] = new XZRect(p0.x(), p1.x(), p0.z(), p1.z(),
                       p1.y(), mptr);
    i++;
    ss[i] = new XZRect(p0.x(), p1.x(), p0.z(), p1.z(),
                       p0.y(), mptr);

    i++;
    ss[i] = new YZRect(p0.y(), p1.y(), p0.z(), p1.z(),
                       p1.x(), mptr);
    i++;
    ss[i] = new YZRect(p0.y(), p1.y(), p0.z(), p1.z(),
                       p0.x(), mptr);

    const int eindex = i;
    end_index = eindex;
    sides = ss;
  }

  __host__ __device__ bool
  hit(const Ray &r, float t0, float t1,
      HitRecord &rec) const override {
    HitRecord temp;
    bool hit_anything = false;
    float closest_far = t1;
    for (int i = start_index; i <= end_index; i++) {
      Hittable *h = sides[i];
      bool isHit = h->hit(r, t0, closest_far, temp);
      if (isHit == true) {
        hit_anything = true;
        closest_far = temp.t;
        rec = temp;
      }
    }
    return hit_anything;
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
  int start_index;
  int end_index;
  Hittable **sides;
};

#endif
