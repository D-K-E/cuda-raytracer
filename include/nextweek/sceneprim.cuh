#ifndef SCENEPRIM_CUH
#define SCENEPRIM_CUH

#include <nextweek/aarect.cuh>
#include <nextweek/cbuffer.hpp>
#include <nextweek/hittable.cuh>
#include <nextweek/material.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/sphere.cuh>

#include <nextweek/sceneparam.cuh>
#include <nextweek/scenetype.cuh>

struct ScenePrimitive {
  int group_id; // for instances and different mediums
  GroupType gtype;

  // object related
  HittableType htype;
  float radius;
  float p1x, p1y, p1z;
  float p2x, p2y, p2z;
  float p3x, p3y, p3z;
  //
  // material related
  MaterialType mtype;
  float fuzz;

  // texture related
  TextureType ttype;

  float tp1x, tp1y, tp1z; // solid color, checker texture
  float tp2x, tp2y, tp2z; // checker texture
  unsigned char *data;    // image texture
  int data_size;          // image texture size
  int width, height;      // image texture
  int bytes_per_pixel;    // image texture
  int index;              // image texture
  float scale;            // noise texture

  curandState *rstate;

  Hittable *h;
  Texture *t;
  Material *m;

  __host__ __device__ ScenePrimitive() {}
  __host__ __device__ ScenePrimitive(HittableParams &hp,
                                     MatTextureParams &mp)
      : rstate(nullptr) {
    set_hittable_params(hp);
    set_material_props(mp);
    set_hittable();
  }

  __device__ ScenePrimitive(HittableParams &hp,
                            MatTextureParams &mp,
                            curandState *rs) {
    set_hittable_params(hp);
    set_material_props(mp, rs);
    set_hittable_device();
  }

  __host__ __device__ void
  set_hittable_params(HittableParams &params) {
    gtype = params.gtype;
    group_id = params.group_id;
    htype = params.htype;
    radius = params.radius;

    p1x = params.p1x;
    p1y = params.p1y;
    p1z = params.p1z;

    p2x = params.p2x;
    p2y = params.p2y;
    p2z = params.p2z;

    p3x = params.p3x;
    p3y = params.p3y;
    p3z = params.p3z;
  }

  __host__ __device__ void
  set_material_props(MatTextureParams &params) {
    mtype = params.mtype;
    fuzz = params.fuzz;
    ttype = params.ttype;
    tp1x = params.tp1x;
    tp1y = params.tp1y;
    tp1z = params.tp1z;
    tp2x = params.tp2x;
    tp2y = params.tp2y;
    tp2z = params.tp2z;
    data = params.data;
    data_size = params.data_size;
    width = params.width;
    height = params.height;
    index = params.index;
    scale = params.scale;
  }

  __device__ void
  set_material_props(MatTextureParams &params,
                     curandState *loc) {
    set_material_props(params);
    rstate = loc;
  }

  __host__ __device__ void set_hittable() {
    to_texture();
    to_material();
    to_hittable();
  }
  __device__ void set_hittable_device() {
    to_texture_device();
    to_material();
    to_hittable();
  }

  __host__ __device__ void
  mkSphere(Point3 cen, float r, MatTextureParams &params) {

    HittableParams hp;
    hp.mkSphere(cen, r);
    set_hittable_params(hp);
    set_material_props(params);
    set_hittable();
  }
  __device__ void mkSphere(Point3 cen, float r,
                           MatTextureParams &params,
                           curandState *rst) {
    HittableParams hp;
    hp.mkSphere(cen, r);
    set_hittable_params(hp);
    set_material_props(params, rst);
    set_hittable_device();
  }

  __host__ __device__ void mkRect(float a0,   //
                                  float a1,   //
                                  float b0,   //
                                  float b1,   //
                                  float _k,   //
                                  Vec3 anorm, //
                                  MatTextureParams params,
                                  GroupType gt, int gid) {
    HittableParams hp;
    hp.mkRect(a0, a1, b0, b1, _k, anorm, gid, gt);
    set_hittable_params(hp);
    set_material_props(params);
    set_hittable();
  }
  __device__ void mkRect(float a0, float a1, float b0,
                         float b1, float _k,
                         Vec3 anorm, //
                         MatTextureParams params,
                         curandState *rst, // noise texture
                         GroupType gt, int gid) {
    HittableParams hp;
    hp.mkRect(a0, a1, b0, b1, _k, anorm, gid, gt);
    set_material_props(params, rst);
    set_hittable_device();
  }

  __host__ __device__ void to_solid_color() {
    t = new SolidColor(tp1x, tp1y, tp1z);
  }
  __host__ __device__ void to_checker() {
    Color s1(tp1x, tp1y, tp1z);
    Color s2(tp2x, tp2y, tp2z);
    t = new CheckerTexture(s1, s2);
  }
  __host__ __device__ void to_image() {

    t = new ImageTexture(data, width, height,
                         width * bytes_per_pixel,
                         bytes_per_pixel, index);
  }
  __device__ void to_noise() {
    t = new NoiseTexture(scale, rstate);
  }
  __host__ __device__ void to_texture() {
    if (ttype == SOLID) {
      to_solid_color();
    } else if (ttype == CHECKER) {
      to_checker();
    } else {
      to_image();
    }
  }
  __device__ void to_texture_device() {
    if (ttype == SOLID) {
      to_solid_color();
    } else if (ttype == CHECKER) {
      to_checker();
    } else if (ttype == IMAGE) {
      to_image();
    } else {
      to_noise();
    }
  }
  __host__ __device__ void to_material() {

    switch (mtype) {
    case LAMBERT:
      m = new Lambertian(t);
      break;
    case METAL:
      m = new Metal(t, fuzz);
      break;
    case DIELECTRIC:
      m = new Dielectric(fuzz);
      break;
    case DIFFUSE_LIGHT:
      m = new DiffuseLight(t);
      break;
    case ISOTROPIC:
      m = new Isotropic(t);
      break;
    }
  }
  __host__ __device__ void to_sphere() {
    h = new Sphere(Point3(p1x, p1y, p1z), radius, m);
  }
  __host__ __device__ void to_moving_sphere() {
    h = new MovingSphere(Point3(p1x, p1y, p1z),
                         Point3(p2x, p2y, p2z), p3x, p3y,
                         radius, m);
  }
  __host__ __device__ void to_aarect() {
    h = new AaRect(p1x, p1y, p2x, p2y, p2z, m,
                   Vec3(p3x, p3y, p3z));
  }
  __host__ __device__ void to_triangle() {
    h = new Triangle(Point3(p1x, p1y, p1z),
                     Point3(p2x, p2y, p2z),
                     Point3(p3x, p3y, p3z), m);
  }
  __host__ __device__ void to_hittable() {
    //
    if (htype == TRIANGLE) {
      to_triangle();
    } else if (htype == SPHERE) {
      to_sphere();
    } else if (htype == MOVING_SPHERE) {
      to_moving_sphere();
    } else if (htype == RECTANGLE) {
      to_aarect();
    }
    // h->mat_ptr = m;
  }

  __host__ __device__ void to_obj(Hittable *&ht) { ht = h; }
  __device__ void to_obj_device(Hittable *&ht) { ht = h; }
};
__host__ __device__ int
farthest_index(const ScenePrimitive &g,
               const ScenePrimitive *&g, int nb_g) {
  //
  float max_dist = FLT_MIN;
  int max_dist_index = 0;
  Aabb tb;
  g.h->bounding_box(0.0f, 0.0f, tb);
  Point3 g_center = tb.center();
  for (int i = 0; i < nb_group; i++) {
    Aabb t;
    gs[i].h->bounding_box(0.0f, 0.0f, t);
    Point3 scene_center = t.center();
    float dist = distance(g_center, scene_center);
    if (dist > max_dist) {
      max_dist = dist;
      max_dist_index = i;
    }
  }
  return max_dist_index;
}
// implementing list structure from
// Foley et al. 2013, p. 1081
void order_scene(ScenePrimitive *&sp, int nb_prims) {
  for (int i = 0; i < nb_prims - 1; i += 2) {
    ScenePrimitive s = sp[i];
    int fgi = farthest_index(s, sp, nb_prims);
    swap(gs, i + 1, fgi);
  }
}

#endif
