// make world kernel
#pragma once

#include <nextweek/hittable.cuh>
#include <nextweek/hittables.cuh>
#include <nextweek/material.cuh>
#include <nextweek/sphere.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>
#include <nextweek/ray.cuh>

/**
 Kernel that fiils the pointer of Hittables pointer.
 */
__global__ void make_world(Hittables **world, Hittable **ss,
                           int nx, int ny,
                           curandState *randState, int row,
                           unsigned char *imdata,
                           int *widths, int *heights,
                           int *bytes_per_pixels) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // declare objects
    CheckerTexture *check =
        new CheckerTexture(Vec3(0.2, 0.8, 0.1));
    Lambertian *lamb = new Lambertian(check);
    ss[0] = new Sphere(Vec3(0, -1000.0, -1), 1000, lamb);
    int i = 1;
    int halfRow = row / 2;
    for (int a = -halfRow; a < halfRow; a++) {
      for (int b = -halfRow; b < halfRow; b++) {
        float choose_mat = curand_uniform(randState);
        Vec3 center(a + curand_uniform(randState), 0.2,
                    b + curand_uniform(randState));
        if (choose_mat < 0.8f) {
          Point3 center2 =
              center +
              Vec3(0, random_float(randState, 0.0, 0.5), 0);
          Color albedo = random_vec(randState);
          albedo *= random_vec(randState);
          Material *lamb1 = new Lambertian(albedo);
          ss[i++] = new MovingSphere(center, center2, 0.0,
                                     1.0, 0.2, lamb1);
        } else if (choose_mat < 0.95f) {

          Material *met = new Metal(
              Vec3(0.7f), 0.5f * curand_uniform(randState));
          ss[i++] = new Sphere(center, 0.2, met);
        } else {
          Material *diel = new Dielectric(1.5);
          ss[i++] = new Sphere(center, 0.2, diel);
        }
      }
    }

    Material *diel = new Dielectric(1.5);
    ss[i++] = new Sphere(Vec3(0, 1, 0), 1.0, diel);

    ImageTexture *imtex1 = new ImageTexture(
        imdata, widths, heights, bytes_per_pixels, 1);

    Material *lamb2 = new Lambertian(imtex1);
    ss[i++] = new Sphere(Vec3(-4, 1, 0), 1.3, lamb2);

    // ImageTexture *imtex2 = new ImageTexture(
    //    imdata, widths, heights, bytes_per_pixels, 0);
    NoiseTexture *ntxt = new NoiseTexture(4.3, randState);
    Material *met2 = new Lambertian(ntxt);
    // Material *met2 = new Metal(Vec3(0.1, 0.2, 0.5), 0.3);

    ss[i++] = new Sphere(Vec3(4, 1, 0), 1.0, met2);

    world[0] = new Hittables(ss, 22 * 22 + 1 + 3);
  }
}

