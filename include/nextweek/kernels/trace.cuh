// trace kernel
#pragma once

#include <nextweek/camera.cuh>
#include <nextweek/external.hpp>
#include <nextweek/hittable.cuh>
#include <nextweek/hittables.cuh>
#include <nextweek/material.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/vec3.cuh>

/**
  @param Ray r is the incoming ray.
  @param Hittables** world pointer to list of hittables
 */
__device__ Color ray_color(const Ray &r, Hittables **world,
                           curandState *local_rand_state,
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
      bool isScattered = rec.mat_ptr->scatter(
          current_ray, rec, attenuation, scattered,
          local_rand_state);
      if (isScattered) {
        bounceNb--;
        result += (current_attenuation * emittedColor);
        current_attenuation *= attenuation;
        current_ray = scattered;
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
    float u = float(i + curand_normal(&localS) / 2) /
              float(maximum_x);
    float v = float(j + curand_normal(&localS) / 2) /
              float(maximum_y);
    Ray r = cam.get_ray(u, v, &localS);
    rcolor += ray_color(r, world, randState, bounceNb);
  }
  // fix the bounce depth
  randState[pixel_index] = localS;
  rcolor /= float(sample_nb);
  rcolor.e[0] = sqrt(rcolor.x());
  rcolor.e[1] = sqrt(rcolor.y());
  rcolor.e[2] = sqrt(rcolor.z());
  fb[pixel_index] = rcolor;
}
