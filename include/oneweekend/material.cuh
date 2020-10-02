#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include <oneweekend/hittable.cuh>
#include <oneweekend/ray.cuh>

struct HitRecord;

__device__ double fresnelCT(double costheta, double ridx) {
  // cook torrence fresnel equation
  double etao = 1 + sqrt(ridx);
  double etau = 1 - sqrt(ridx);
  double eta = etao / etau;
  double g = sqrt(pow(eta, 2) + pow(costheta, 2) - 1);
  double g_c = g - costheta;
  double gplusc = g + costheta;
  double gplus_cc = (gplusc * costheta) - 1;
  double g_cc = (g_c * costheta) + 1;
  double oneplus_gcc = 1 + pow(gplus_cc / g_cc, 2);
  double half_plus_minus = 0.5 * pow(g_c / gplusc, 2);
  return half_plus_minus * oneplus_gcc;
}

__device__ bool refract(const Vec3 &v, const Vec3 &n,
                        double ni_over_nt, Vec3 &refracted) {
  Vec3 uv = to_unit(v);
  double dt = dot(uv, n);
  double discriminant =
      1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
  if (discriminant > 0) {
    refracted =
        ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
    return true;
  } else {
    return false;
  }
}

__device__ Vec3 reflect(const Vec3 &v, const Vec3 &n) {
  return v - 2.0f * dot(v, n) * n;
}
class Material {
public:
  __device__ virtual bool
  scatter(const Ray &r_in, const HitRecord &rec,
          Vec3 &attenuation, Ray &scattered,
          curandState *local_rand_state) const = 0;
};

class Lambertian : public Material {
public:
  __device__ Lambertian(const Vec3 &a) : albedo(a) {}
  __device__ bool
  scatter(const Ray &r_in, const HitRecord &rec,
          Vec3 &attenuation, Ray &scattered,
          curandState *local_rand_state) const override {
    Vec3 target = rec.p + rec.normal +
                  random_in_unit_sphere(local_rand_state);
    scattered = Ray(rec.p, target - rec.p, r_in.time());
    attenuation = albedo;
    return true;
  }

  Vec3 albedo;
};

class Metal : public Material {
public:
  __device__ Metal(const Vec3 &a, double f) : albedo(a) {
    if (f < 1)
      fuzz = f;
    else
      fuzz = 1;
  }
  __device__ bool
  scatter(const Ray &r_in, const HitRecord &rec,
          Vec3 &attenuation, Ray &scattered,
          curandState *local_rand_state) const override {
    Vec3 reflected =
        reflect(to_unit(r_in.direction()), rec.normal);
    scattered = Ray(rec.p, reflected +
                               fuzz * random_in_unit_sphere(
                                          local_rand_state),
                    r_in.time());
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
  }
  Vec3 albedo;
  double fuzz;
};

class Dielectric : public Material {
public:
  __device__ Dielectric(double ri) : ref_idx(ri) {}
  __device__ bool
  scatter(const Ray &r_in, const HitRecord &rec,
          Vec3 &attenuation, Ray &scattered,
          curandState *local_rand_state) const override {
    Vec3 outward_normal;
    Vec3 reflected = reflect(r_in.direction(), rec.normal);
    double ni_over_nt;
    attenuation = Vec3(1.0);
    Vec3 refracted;
    double reflect_prob;
    double cosine;
    if (dot(r_in.direction(), rec.normal) > 0.0f) {
      outward_normal = -rec.normal;
      ni_over_nt = ref_idx;
      cosine = dot(r_in.direction(), rec.normal) /
               r_in.direction().length();
      cosine = sqrt(
          1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
    } else {
      outward_normal = rec.normal;
      ni_over_nt = 1.0f / ref_idx;
      cosine = -dot(r_in.direction(), rec.normal) /
               r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal,
                ni_over_nt, refracted))
      reflect_prob = fresnelCT(cosine, ref_idx);
    else
      reflect_prob = 1.0f;
    if (curand_uniform(local_rand_state) < reflect_prob)
      scattered = Ray(rec.p, reflected, r_in.time());
    else
      scattered = Ray(rec.p, refracted, r_in.time());
    return true;
  }
  double ref_idx;
};

#endif
