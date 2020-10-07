#ifndef PDF_CUH
#define PDF_CUH

#include <rest/onb.cuh>
#include <rest/ray.cuh>
#include <rest/vec3.cuh>

class Pdf {
public:
  __host__ __device__ Pdf() {}
  __device__ virtual ~Pdf() {}

  __device__ virtual float
  value(const Vec3 &direction) const = 0;

  __device__ virtual Vec3
  generate(curandState *loc) const = 0;
};

class CosinePdf : public Pdf {
public:
  __host__ __device__ CosinePdf(const Vec3 &w) {
    uvw.build_from_w(w);
  }

  __device__ float
  value(const Vec3 &direction) const override {
    auto cosine = dot(to_unit(direction), uvw.w());
    return (cosine <= 0) ? 0 : cosine / M_PI;
  }

  __device__ Vec3
  generate(curandState *loc) const override {
    return uvw.local(random_cosine_direction(loc));
  }

public:
  Onb uvw;
};

class HittablePdf : public Pdf {
public:
  __host__ __device__ HittablePdf(Hittable *&p,
                                  const Point3 &origin)
      : ptr(p), o(origin) {}

  __device__ float
  value(const Vec3 &direction) const override {
    return ptr->pdf_value(o, direction);
  }

  __device__ Vec3
  generate(curandState *loc) const override {
    return ptr->random(o, loc);
  }

public:
  Point3 o;
  Hittable *ptr;
};

class MixturePdf : public Pdf {
public:
  __host__ __device__ MixturePdf(Pdf *&p0, Pdf *&p1) {
    p[0] = p0;
    p[1] = p1;
  }

  __device__ float
  value(const Vec3 &direction) const override {
    return 0.5 * p[0]->value(direction) +
           0.5 * p[1]->value(direction);
  }

  __device__ Vec3
  generate(curandState *loc) const override {
    if (curand_uniform(loc) < 0.5)
      return p[0]->generate(loc);
    else
      return p[1]->generate(loc);
  }

public:
  Pdf *p[2];
};

#endif
