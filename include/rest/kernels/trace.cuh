// trace kernel
#pragma once

#include <rest/camera.cuh>
#include <rest/external.hpp>
#include <rest/hittable.cuh>
#include <rest/hittables.cuh>
#include <rest/material.cuh>
#include <rest/pdf.cuh>
#include <rest/ray.cuh>
#include <rest/vec3.cuh>

/**
  @param Ray r is the incoming ray.
  @param Hittables** world pointer to list of hittables
 */
__device__ Color ray_color(const Ray &r, Hittables **world,
                           Hittable *light_shape,
                           Pdf **pdfs, curandState *loc,
                           int bounceNb) {
  Ray current_ray = r;
  Vec3 current_attenuation = Vec3(1.0f);
  Vec3 result = Vec3(0.0f);

  while (bounceNb > 0) {
    HitRecord rec;
    bool anyHit =
        world[0]->hit(current_ray, 0.001f, FLT_MAX, rec);
    if (anyHit) {
      Color emittedColor =
          rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
      Ray scattered;
      Vec3 attenuation;
      float pdf_val;
      bool isScattered = rec.mat_ptr->scatter(
          current_ray, rec, attenuation, scattered, pdf_val,
          loc);
      if (isScattered) {

        bounceNb--;
        //

        HittablePdf hpdf(light_shape, rec.p);

        CosinePdf cpdf(rec.normal);
        pdfs[0] = static_cast<Pdf *>(&cpdf);
        pdfs[1] = static_cast<Pdf *>(&hpdf);
        MixturePdf p(pdfs[0], pdfs[1]);
        // scattered =
        //    Ray(rec.p, p.generate(loc),
        //    current_ray.time());
        //
        // float pdf_val = p.value(scattered.direction());
        float s_pdf = rec.mat_ptr->scattering_pdf(
            current_ray, rec, scattered);
        //

        result += (current_attenuation * emittedColor);
        current_attenuation *=
            attenuation * s_pdf / pdf_val;
        current_ray = scattered;

        //
      } else {
        result += (current_attenuation * emittedColor);
        return result;
      }
    } else {
      return Color(0.0);
    }
  }
  return Vec3(0.0f); // background color
}
__global__ void render(Vec3 *fb, int maximum_x,
                       int maximum_y, int sample_nb,
                       int bounceNb, Camera dcam,
                       Hittables **world,
                       Hittable *light_shape, Pdf **pdfs,
                       curandState *randState) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= maximum_x) || (j >= maximum_y)) {
    return;
  }
  int pixel_index = j * maximum_x + i;
  curandState localS = randState[pixel_index];
  Vec3 rcolor(0.0f);
  Camera cam = dcam;
  for (int s = 0; s < sample_nb; s++) {
    float u = float(i + curand_uniform(&localS)) /
              float(maximum_x);
    float v = float(j + curand_uniform(&localS)) /
              float(maximum_y);
    Ray r = cam.get_ray(u, v, &localS);
    //
    rcolor += ray_color(r, world, light_shape, pdfs,
                        &localS, bounceNb);
  }
  // fix the bounce depth
  randState[pixel_index] = localS;
  rcolor /= float(sample_nb);
  rcolor.e[0] = sqrt(rcolor.x());
  rcolor.e[1] = sqrt(rcolor.y());
  rcolor.e[2] = sqrt(rcolor.z());
  fb[pixel_index] = rcolor;
}
