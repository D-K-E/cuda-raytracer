// make world kernel
#pragma once

#include <nextweek/aarect.cuh>
#include <nextweek/box.cuh>
#include <nextweek/hittable.cuh>
#include <nextweek/hittables.cuh>
#include <nextweek/material.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/sphere.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

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
void free_world(
    thrust::device_ptr<Vec3> &fb,
    thrust::device_ptr<Hittables *> &world,
    thrust::device_ptr<Hittable *> &hs,
    thrust::device_ptr<unsigned char> imdata,
    thrust::device_ptr<int> imch,
    thrust::device_ptr<int> imhs,
    thrust::device_ptr<int>(imwidths),
    thrust::device_ptr<curandState> randState1,
    thrust::device_ptr<curandState> randState2) {
  thrust::device_free(fb);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(world);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(hs);
  CUDA_CONTROL(cudaGetLastError());
  // dcam.free();
  thrust::device_free(imdata);
  thrust::device_free(imch);
  thrust::device_free(imhs);
  thrust::device_free(imwidths);
  // free(ws_ptr);
  // free(nb_ptr);
  // free(hs_ptr);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState2);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState1);
  CUDA_CONTROL(cudaGetLastError());
}

__device__ void make_box(Hittable **s, int &i,
                         const Point3 &p0, const Point3 &p1,
                         Material *mptr) {
  // make a box with points
  i++;
  s[i] = new XYRect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(),
                    mptr);
  i++;
  s[i] = new XYRect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(),
                    mptr);

  i++;
  s[i] = new XZRect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(),
                    mptr);
  i++;
  s[i] = new XZRect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(),
                    mptr);

  i++;
  s[i] = new YZRect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(),
                    mptr);
  i++;
  s[i] = new YZRect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(),
                    mptr);
}

__global__ void make_empty_cornell_box(Hittables **world,
                                       Hittable **ss) {
  // declare objects
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    Material *red = new Lambertian(Color(.65, .05, .05));
    Material *blue = new Lambertian(Color(.05, .05, .65));
    Material *white = new Lambertian(Color(.73, .73, .73));
    Material *green = new Lambertian(Color(.12, .45, .15));
    Material *light = new DiffuseLight(Color(15, 15, 15));
    int i = 0;
    ss[i] = new YZRect(0, 555, 0, 555, 555, green);
    i++;
    ss[i] = new YZRect(0, 555, 0, 555, 0, red);
    i++;
    ss[i] = new XZRect(213, 343, 227, 332, 554, light);
    i++;
    ss[i] = new XZRect(0, 555, 0, 555, 0, white);
    i++;
    ss[i] = new XZRect(0, 555, 0, 555, 555, white);
    i++;
    ss[i] = new XYRect(0, 555, 0, 555, 555, blue);
    // -------------- Boxes -------------------------
    i++;
    Point3 bp1(0.0f);
    Point3 bp2(165, 330, 165);
    Box b1(bp1, bp2, white, ss, i);
    b1.rotate_y(ss, 15.0);
    b1.translate(ss, Vec3(265.0, 0.0, 295.0));
    i++;
    Point3 bp3(0.0f);
    Point3 bp4(165.0f);
    Box b2(bp3, bp4, white, ss, i);
    b2.rotate_y(ss, -18.0f);
    b2.translate(ss, Point3(130, 0, 165));
    i++;

    world[0] = new Hittables(ss, i);
  }
}

void free_empty_cornell(
    thrust::device_ptr<Vec3> &fb,
    thrust::device_ptr<Hittables *> &world,
    thrust::device_ptr<Hittable *> &hs,
    thrust::device_ptr<curandState> randState1,
    thrust::device_ptr<curandState> randState2) {

  thrust::device_free(fb);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(world);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(hs);
  CUDA_CONTROL(cudaGetLastError());

  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState2);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState1);
  CUDA_CONTROL(cudaGetLastError());
}
