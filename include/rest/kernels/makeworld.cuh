// make world kernel
#pragma once

#include <rest/aarect.cuh>
#include <rest/box.cuh>
#include <rest/hittable.cuh>
#include <rest/hittables.cuh>
#include <rest/material.cuh>
#include <rest/mediumc.cuh>
#include <rest/ray.cuh>
#include <rest/sphere.cuh>
#include <rest/utils.cuh>
#include <rest/vec3.cuh>

__device__ void make_side_boxes(Hittable **&ss, int &ocount,
                                int &start_index,
                                int &end_index,
                                int side_box_nb,
                                curandState *randState) {
  // ------- side boxes --------------
  const int sind = ocount;
  start_index = sind;

  // ------- side box materials ------
  Material *ground1 =
      new Lambertian(Color(0.48, 0.83, 0.53));
  CheckerTexture *check =
      new CheckerTexture(Vec3(1.0, 1.0, 1.0));
  Material *ground2 = new Lambertian(check);

  for (int i = 0; i < side_box_nb; i++) {
    for (int j = 0; j < side_box_nb; j++) {

      float w = 100.0f;
      float x0 = -1000.0f + i * w;
      float z0 = -1000.0f + j * w;
      float y0 = 0.0f;
      float x1 = x0 + w;
      float y1 = random_float(randState, 1, 101);
      float z1 = z0 + w;
      Material *ground = i * j % 2 == 0 ? ground1 : ground2;
      Box b(Point3(x0, y0, z0), Point3(x1, y1, z1), ground,
            ss, ocount);
      ocount = b.end_index;
    }
  }
  end_index = ocount;
}

__device__ void make_world_light(Hittable **&ss,
                                 int &ocount,
                                 int &start_index,
                                 int &end_index) {
  // ------- light -------------------
  const int si = ocount;
  start_index = si;
  Material *light_material =
      new DiffuseLight(Color(8.0f, 8.0f, 8.0f));
  Hittable *light =
      new XZRect(123, 423, 147, 412, 554, light_material);
  ss[ocount] = light;
  end_index = ocount + 1;
  ocount = end_index;
}

__device__ void make_moving_sphere(Hittable **&ss,
                                   int &ocount,
                                   int &start_index,
                                   int &end_index) {
  // ------- light -------------------
  const int si = ocount;
  start_index = si;

  // --------- moving sphere -----------
  Point3 cent1(200, 150, 200);
  Point3 cent2 = cent1 + Point3(30.0f, 0.0f, 0.0f);
  Material *moving_sphere_material =
      new Lambertian(Color(0.7f, 0.3f, 0.1f));
  Hittable *moving_sphere = new MovingSphere(
      cent1, cent2, 0, 1, 50, moving_sphere_material);
  ss[ocount] = moving_sphere;

  end_index = ocount + 1;
  ocount = end_index;
}

__device__ void make_two_sphere(Hittable **&ss, int &ocount,
                                int &start_index,
                                int &end_index) {
  const int si = ocount;
  start_index = si;
  Hittable *sp1 = new Sphere(Point3(560, 150, 45), 50,
                             new Dielectric(1.5f));
  ss[ocount] = sp1;
  end_index = ocount + 1;
  ocount = end_index;

  Hittable *sp2 =
      new Sphere(Point3(500, 150, 145), 50,
                 new Metal(Color(0.8f, 0.8f, 0.8f), 10.0f));
  ss[ocount] = sp2;
  end_index = ocount + 1;
  ocount = end_index;
}

__device__ void make_subsurface(Hittable **&ss, int &ocount,
                                int &start_index,
                                int &end_index) {
  const int si = ocount;
  start_index = si;
  Hittable *sp3 = new Sphere(Point3(620, 400, 145), 70,
                             new Dielectric(1.5));
  ss[ocount] = sp3;
  end_index = ocount + 1;
  ocount = end_index;
}

__device__ void make_volumetric(Hittable **&ss, int &ocount,
                                int &start_index,
                                int &end_index) {
  const int si = ocount;
  start_index = si;
  Hittable *sp4 =
      new Sphere(Point3(400, 400, 145), 70,
                 new Lambertian(Color(0.5, 0.1, 0.7)));

  ss[ocount] = sp4;
  end_index = ocount + 1;
  ocount = end_index;
}

__device__ void make_images(Hittable **&ss, int &ocount,
                            int &start_index,
                            int &end_index,
                            unsigned char *imdata,
                            int *widths, int *heights,
                            int *bytes_per_pixels) {
  //
  const int si = ocount;
  start_index = si;
  ImageTexture *imtex1 = new ImageTexture(
      imdata, widths, heights, bytes_per_pixels, 1);

  Material *lamb2 = new Lambertian(imtex1);
  Hittable *spImg =
      new Sphere(Point3(220, 400, 200), 100, lamb2);
  ss[ocount] = spImg;
  end_index = ocount + 1;
  ocount = end_index;

  ImageTexture *imtex2 = new ImageTexture(
      imdata, widths, heights, bytes_per_pixels, 0);
  Material *lamb3 = new Lambertian(imtex2);
  Hittable *spImg2 =
      new Sphere(Point3(0, 400, 200), 100, lamb3);

  ss[ocount] = spImg2;
  end_index = ocount + 1;
  ocount = end_index;
}

__device__ void make_noise(Hittable **&ss, int &ocount,
                           int &start_index, int &end_index,
                           curandState *randState) {
  //
  const int si = ocount;
  start_index = si;

  NoiseTexture *ntxt = new NoiseTexture(0.1, randState);
  Material *met2 = new Lambertian(ntxt);
  Hittable *noise_sp =
      new Sphere(Point3(-250, 400, 300), 80, met2);
  ss[ocount] = noise_sp;
  end_index = ocount + 1;
  ocount = end_index;
}

/**
 Kernel that fiils the pointer of Hittables pointer.
 */
__global__ void make_world(Hittables **world, Hittable **ss,
                           curandState *randState,
                           int side_box_nb,
                           unsigned char *imdata,
                           int *widths, int *heights,
                           int *bytes_per_pixels //
                           ) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // ------- declare counters ---------
    int ocount = 0; // object counter
    int start_index;
    int end_index;
    make_side_boxes(ss, ocount, start_index, end_index,
                    side_box_nb, randState);

    Hittable *g1 =
        new HittableGroup(ss, start_index, end_index);

    make_world_light(ss, ocount, start_index, end_index);

    Hittable *g2 =
        new HittableGroup(ss, start_index, end_index);

    make_moving_sphere(ss, ocount, start_index, end_index);

    Hittable *g3 =
        new HittableGroup(ss, start_index, end_index);

    make_two_sphere(ss, ocount, start_index, end_index);
    Hittable *g4 =
        new HittableGroup(ss, start_index, end_index);

    make_subsurface(ss, ocount, start_index, end_index);

    Hittable *g5 =
        new HittableGroup(ss, start_index, end_index);

    g5 = new ConstantMedium(g5, 0.2f, Color(0.2, 0.4, 0.9),
                            randState);

    make_volumetric(ss, ocount, start_index, end_index);
    Hittable *g6 =
        new HittableGroup(ss, start_index, end_index);

    g6 = new ConstantMedium(
        g6, 0.01f, Color(0.8f, 0.2, 0.4), randState);

    make_images(ss, ocount, start_index, end_index, imdata,
                widths, heights, bytes_per_pixels);

    Hittable *g7 =
        new HittableGroup(ss, start_index, end_index);

    make_noise(ss, ocount, start_index, end_index,
               randState);
    Hittable *g8 =
        new HittableGroup(ss, start_index, end_index);

    int group_size = 8;
    Hittable **groups = new Hittable *[group_size];

    groups[0] = g1; //
    groups[1] = g2; //
    groups[2] = g3; //
    groups[3] = g4; //
    groups[4] = g5; //
    groups[5] = g6; //
    groups[6] = g7; //
    groups[7] = g8; //

    //
    order_scene(groups, group_size);

    world[0] = new Hittables(groups, group_size);
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

__global__ void
make_empty_cornell_box(Hittables **world, Hittable **ss,
                       XZRect *light_shape,
                       curandState *randState) {
  // declare objects
  if (threadIdx.x == 0 && blockIdx.x == 0) {

    Material *red = new Lambertian(Color(.65, .05, .05));
    Material *blue = new Lambertian(Color(.05, .05, .65));
    Material *white = new Lambertian(Color(.73, .73, .73));
    Material *green = new Lambertian(Color(.12, .45, .15));
    Material *light = new DiffuseLight(Color(15, 15, 15));

    // ----------- Groups --------------------
    Hittable **groups = new Hittable *[3];

    int obj_count = 0;
    int group_count = 0;
    // --------------- cornell box group ----------------

    ss[obj_count] = new YZRect(0, 555, 0, 555, 555, green);
    obj_count++;
    ss[obj_count] = new YZRect(0, 555, 0, 555, 0, red);
    obj_count++;
    ss[obj_count] =
        new XZRect(213, 343, 227, 332, 554, light);
    obj_count++;
    ss[obj_count] = new XZRect(0, 555, 0, 555, 0, white);
    obj_count++;
    ss[obj_count] = new XZRect(0, 555, 0, 555, 555, white);
    obj_count++;
    ss[obj_count] = new XYRect(0, 555, 0, 555, 555, blue);

    Hittable *c_box =
        new HittableGroup(ss, 0, obj_count + 1);
    groups[group_count] = c_box;

    // -------------- Boxes -------------------------

    obj_count++;
    Point3 bp1(0.0f);
    Point3 bp2(165, 330, 165);
    Box b1(bp1, bp2, white, ss, obj_count);
    b1.rotate_y(ss, 15.0f);
    b1.translate(ss, Vec3(265, 0, 295));
    // b1.to_gas(0.01f, &randState, Color(1.0, 0.3, 0.7),
    // ss);
    Hittable *tall_box =
        new HittableGroup(ss, b1.start_index, b1.end_index);

    // Hittable *smoke_box1 = new ConstantMedium(
    //    tall_box, 0.01, Color(0.8f, 0.2, 0.4), randState);

    group_count++;

    groups[group_count] = tall_box;

    obj_count++;
    Point3 bp3(0.0f);
    Point3 bp4(165.0f);
    Box b2(bp3, bp4, white, ss, obj_count);
    b2.rotate_y(ss, -18.0f);
    b2.translate(ss, Point3(130, 0, 165));
    obj_count++;

    Hittable *short_box =
        new HittableGroup(ss, b2.start_index, b2.end_index);
    // Hittable *smoke_box2 = new ConstantMedium(
    //    short_box, 0.01, Color(0.8f, 0.3, 0.8),
    //    randState);

    group_count++;
    groups[group_count] = short_box;

    group_count++;
    order_scene(groups, group_count);

    world[0] = new Hittables(groups, group_count);
    Material *mt = nullptr;
    XZRect *light_shape =
        new XZRect(213, 343, 227, 332, 554, mt);
  }
}

void free_empty_cornell(
    thrust::device_ptr<Vec3> &fb,
    thrust::device_ptr<Hittables *> &world,
    thrust::device_ptr<Hittable *> &hs,
    thrust::device_ptr<XZRect> &lshape,
    thrust::device_ptr<curandState> randState1,
    thrust::device_ptr<curandState> randState2) {

  thrust::device_free(fb);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(world);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(hs);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(lshape);

  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState2);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState1);
  CUDA_CONTROL(cudaGetLastError());
}
