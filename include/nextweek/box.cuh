#ifndef BOX_CUH
#define BOX_CUH
#include <nextweek/aarect.cuh>
#include <nextweek/hittables.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

enum State : int { SOLID, LIQUID, GAS };

class Box : public Hittable {
public:
  __host__ __device__ Box() {}
  __host__ __device__ ~Box() { delete sides; }
  __host__ __device__ Box(const Point3 &p0,
                          const Point3 &p1, Material *mptr,
                          Hittable **&ss, int &i, State s)
      : m_ptr(mptr) {
    box_min = p0;
    box_max = p1;
    state = s;
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
  __host__ __device__ Box(const Point3 &p0,
                          const Point3 &p1, Material *mptr,
                          Hittable **&ss, int &i)
      : Box(p0, p1, mptr, ss, i, SOLID) {}

  __device__ bool hit(const Ray &r, float t0, float t1,
                      HitRecord &rec) const override {
    if (state == SOLID) {
      return solid_hit(r, t0, t1, rec);
    } else if (state == GAS) {
      return gas_hit(r, t0, t1, rec);
    } else {
      return true;
    }
  }

  __device__ bool solid_hit(const Ray &r, float t0,
                            float t1,
                            HitRecord &rec) const {
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

  __device__ bool gas_hit(const Ray &r, float t_min,
                          float t_max,
                          HitRecord &rec) const {
    //
    // Print occasional samples when debugging. To enable,
    // set enableDebug true.
    const bool enableDebug = false;

    const bool debugging =
        enableDebug && curand_uniform(rState) < 0.00001;

    HitRecord rec1, rec2;

    if (!solid_hit(r, -FLT_MAX, FLT_MAX, rec1))
      return false;

    if (!solid_hit(r, rec1.t + 0.0001f, FLT_MAX, rec2))
      return false;

    if (debugging) {
      printf("\nt0= %f", rec1.t);
      printf(", t1= %f\n", rec2.t);
    }

    if (rec1.t < t_min)
      rec1.t = t_min;
    if (rec2.t > t_max)
      rec2.t = t_max;

    if (rec1.t >= rec2.t)
      return false;

    if (rec1.t < 0)
      rec1.t = 0;

    const float ray_length = r.direction().length();
    const float distance_inside_boundary =
        (rec2.t - rec1.t) * ray_length;
    const float hit_distance =
        neg_inv_density * log(curand_uniform(rState));

    if (hit_distance > distance_inside_boundary)
      return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    if (debugging) {
      printf("hit_distance = %f\n", hit_distance);
      printf("rec.t = %f\n", rec.t);
      printf("rec.p = %f ", rec.p.x());
      printf("%f ", rec.p.y());
      printf("%f ", rec.p.z());
    }

    rec.normal = Vec3(1, 0, 0); // arbitrary
    rec.front_face = true;      // also arbitrary
    rec.mat_ptr = m_ptr;

    return true;
  }

  __device__ void to_gas(float density, curandState *loc,
                         Texture *t, Hittable **&ss) {
    neg_inv_density = -1.0f / density;
    state = GAS;
    rState = loc;
    //
    for (int i = start_index; i <= end_index; i++) {
      ss[i]->mat_ptr = new Isotropic(t);
      sides[i] = ss[i];
    }
  }
  __device__ void to_gas(float density, curandState *loc,
                         Color c, Hittable **&ss) {
    to_gas(density, loc, new SolidColor(c), ss);
  }

  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    output_box = Aabb(box_min, box_max);
    return true;
  }

  __host__ __device__ void translate(Hittable **&ss,
                                     const Point3 &p) {
    for (int i = start_index; i <= end_index; i++) {
      sides[i] = new Translate(sides[i], p);
      ss[i] = sides[i];
    }
  }
  __host__ __device__ void rotate_y(Hittable **&ss,
                                    float angle) {
    for (int i = start_index; i <= end_index; i++) {
      sides[i] = new RotateY(sides[i], angle);
      ss[i] = sides[i];
    }
  }

public:
  Point3 box_min;
  Point3 box_max;
  int start_index;
  int end_index;
  Hittable **sides;
  float neg_inv_density = 0.01f;
  curandState *rState;
  State state;
  Material *m_ptr;
};

#endif
