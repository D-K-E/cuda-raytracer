// make world kernel
#pragma once

#include <nextweek/aarect.cuh>
#include <nextweek/box.cuh>
#include <nextweek/hittable.cuh>
#include <nextweek/hittables.cuh>
#include <nextweek/material.cuh>
#include <nextweek/mediumc.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/sphere.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

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
  Point3 cent1(400, 400, 200);
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
  Hittable *sp1 = new Sphere(Point3(260, 150, 45), 50,
                             new Dielectric(1.5f));
  ss[ocount] = sp1;
  end_index = ocount + 1;
  ocount = end_index;

  Hittable *sp2 =
      new Sphere(Point3(0, 150, 145), 50,
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
  Hittable *sp3 = new Sphere(Point3(360, 150, 145), 70,
                             new Dielectric(1.5));
  ss[ocount] = sp3;
  end_index = ocount + 1;
  ocount = end_index;
}

/**
 Kernel that fiils the pointer of Hittables pointer.
 */
__global__ void make_world(Hittables **world, Hittable **ss,
                           curandState *randState,
                           int side_box_nb
                           // unsigned char *imdata,
                           // int *widths, int *heights,
                           // int *bytes_per_pixels
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

    // make_subsurface(ss, ocount, start_index, end_index);

    // Hittable *sp_sub =
    //    new HittableGroup(ss, start_index, end_index);

    // Hittable *g5 = new ConstantMedium(
    //    sp_sub, 0.2f, Color(0.2, 0.4, 0.9), randState);

    /*

        // ---------- add volumetric scattering box
       --------
        Hittable *sp4 =
            new Sphere(Point3(60, 250, 145), 70,
                       new Lambertian(Color(0.5, 0.1,
       0.7)));
        add_to_low_level(ss, ocount, sp4);
        const int spvol_start = ocount;
        Hittable *sp_vol_group =
            new HittableGroup(ss, spvol_start, ocount +
       1);

        Hittable *g6 = new ConstantMedium(sp_vol_group,
       0.01f,
                                          Color(0.8f,
       0.2,
       0.4),
                                          randState);

        // ---------- add image texture
       -------------------

        ImageTexture *imtex1 = new ImageTexture(
            imdata, widths, heights, bytes_per_pixels,
       1);

        Material *lamb2 = new Lambertian(imtex1);
        Hittable *spImg =
            new Sphere(Point3(400, 200, 400), 100,
       lamb2);
        add_to_low_level(ss, ocount, spImg);
        const int img_start = ocount;

        ImageTexture *imtex2 = new ImageTexture(
            imdata, widths, heights, bytes_per_pixels,
       0);
        Material *lamb3 = new Lambertian(imtex2);
        Hittable *spImg2 =
            new Sphere(Point3(200, 100, 400), 100,
       lamb3);
        add_to_low_level(ss, ocount, spImg2);

        Hittable *g7 =
            new HittableGroup(ss, img_start, ocount +
       1);

        // ----------- noise sphere ---------------

        NoiseTexture *ntxt = new NoiseTexture(0.1,
       randState);
        Material *met2 = new Lambertian(ntxt);
        Hittable *noise_sp =
            new Sphere(Point3(220, 280, 300), 80, met2);
        add_to_low_level(ss, ocount, noise_sp);
        // Material *met2 = new Metal(Vec3(0.1, 0.2,
       0.5),
       0.3);
        Hittable *g8 =
            new HittableGroup(ss, ocount, ocount + 1);
        */
    int group_size = 4;
    Hittable **groups = new Hittable *[group_size];

    groups[0] = g1; //
    groups[1] = g2; //
    groups[2] = g3; //
    groups[3] = g4; //
    // groups[4] = g5; //
    // g6, //
    // g7, //
    // g8

    world[0] = new Hittables(groups, group_size);
  }
}
void free_world(
    thrust::device_ptr<Vec3> &fb,
    thrust::device_ptr<Hittables *> &world,
    thrust::device_ptr<Hittable *> &hs,
    // thrust::device_ptr<unsigned char> imdata,
    // thrust::device_ptr<int> imch,
    // thrust::device_ptr<int> imhs,
    // thrust::device_ptr<int>(imwidths),
    thrust::device_ptr<curandState> randState1,
    thrust::device_ptr<curandState> randState2) {
  thrust::device_free(fb);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(world);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(hs);
  CUDA_CONTROL(cudaGetLastError());
  // dcam.free();
  // thrust::device_free(imdata);
  // thrust::device_free(imch);
  // thrust::device_free(imhs);
  // thrust::device_free(imwidths);
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

    curandState randState;
    curand_init(obj_count, 0, 0, &randState);
    Hittable *smoke_box1 = new ConstantMedium(
        tall_box, 0.01, Color(0.8f, 0.2, 0.4), &randState);
    group_count++;
    groups[group_count] = smoke_box1;

    obj_count++;
    Point3 bp3(0.0f);
    Point3 bp4(165.0f);
    Box b2(bp3, bp4, white, ss, obj_count);
    b2.rotate_y(ss, -18.0f);
    b2.translate(ss, Point3(130, 0, 165));
    obj_count++;

    Hittable *short_box =
        new HittableGroup(ss, b2.start_index, b2.end_index);
    Hittable *smoke_box2 = new ConstantMedium(
        short_box, 0.01, Color(0.8f, 0.3, 0.8), &randState);

    group_count++;
    groups[group_count] = smoke_box2;

    group_count++;

    world[0] = new Hittables(groups, group_count);
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
