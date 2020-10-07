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
  generate(curandState *&loc) const = 0;
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
  generate(curandState *&loc) const override {
    return uvw.local(random_cosine_direction(loc));
  }

public:
  Onb uvw;
};

template <class T = Hittable>
class HittablePdf : public Pdf {
public:
  __host__ __device__ HittablePdf(T &p,
                                  const Point3 &origin)
      : ptr(p), o(origin) {}

  __device__ float
  value(const Vec3 &direction) const override {
    return ptr.pdf_value(o, direction);
  }

  __device__ Vec3
  generate(curandState *&loc) const override {
    return ptr.random(o, loc);
  }

public:
  Point3 o;
  T ptr;
};

template <class T = Pdf, class U = Pdf>
class MixturePdf : public Pdf {
public:
  __host__ __device__ MixturePdf(T &_p0, U &_p1)
      : p1(_p0), p2(_p1) {}

  __device__ float
  value(const Vec3 &direction) const override {
    return 0.5 * p1.value(direction) +
           0.5 * p2.value(direction);
  }

  __device__ Vec3
  generate(curandState *&loc) const override {
    if (curand_uniform(loc) < 0.5f)
      return p1.generate(loc);
    else
      return p2.generate(loc);
  }

public:
  T p1;
  U p2;
};

#endif
