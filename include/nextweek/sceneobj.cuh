#ifndef SCENEOBJ_CUH
#define SCENEOBJ_CUH
// an interface scene object between different primitives

#include <nextweek/aarect.cuh>
#include <nextweek/cbuffer.hpp>
#include <nextweek/hittable.cuh>
#include <nextweek/material.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/sphere.cuh>
#include <nextweek/triangle.cuh>
#include <nextweek/vec3.cuh>

enum TextureType : int {
  SOLID = 1,
  CHECKER = 2,
  NOISE = 3,
  IMAGE = 4
};

enum MaterialType : int {
  LAMBERT = 1,
  METAL = 2,
  DIELECTRIC = 3,
  DIFFUSE_LIGHT = 4,
  ISOTROPIC = 5
};

enum HittableType : int {
  LINE = 1,
  TRIANGLE = 2,
  SPHERE = 3,
  MOVING_SPHERE = 4,
  RECTANGLE = 5
};

enum GroupType : int { INSTANCE = 1, CONSTANT_MEDIUM = 2 };

struct ImageParams {
  int width;
  int height;
  int index;
  int bytes_per_pixel;
  __host__ __device__ ImageParams(int w, int h, int bpp,
                                  int idx)
      : width(w), height(h), bytes_per_pixel(bpp),
        index(idx) {}
};

struct MatTextureParams {
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

  //
  __host__ __device__ MatTextureParams(
      MaterialType mt, float f, TextureType tt, float _tp1x,
      float _tp1y, float _tp1z, float _tp2x, float _tp2y,
      float _tp2z, unsigned char *&dt, int dsize,
      ImageParams imp, float sc)
      : mtype(mt), fuzz(f), ttype(tt), tp1x(_tp1x),
        tp1y(_tp1y), tp1z(_tp1z), tp2x(_tp2x), tp2y(_tp2y),
        tp2z(_tp2z), width(imp.width), height(imp.height),
        bytes_per_pixel(imp.bytes_per_pixel), data(dt),
        data_size(dsize), index(imp.index), scale(sc) {}
  __host__ __device__
  MatTextureParams(MaterialType mt, float f, TextureType tt,
                   Color c1, Color c2, unsigned char *&dt,
                   int dsize, ImageParams imp, float sc)
      : mtype(mt), fuzz(f), ttype(tt), tp1x(c1.x()),
        tp1y(c1.y()), tp1z(c1.z()), tp2x(c2.x()),
        tp2y(c2.y()), tp2z(c2.z()), width(imp.width),
        height(imp.height),
        bytes_per_pixel(imp.bytes_per_pixel), data(dt),
        data_size(dsize), index(imp.index), scale(sc) {}
};

struct HittableParams {
  int group_id; // for instances and different mediums
  GroupType gtype;

  HittableType htype;
  float radius;
  float p1x, p1y, p1z;
  float p2x, p2y, p2z;
  float p3x, p3y, p3z;

  __host__ __device__ HittableParams() {}
  __host__ __device__ HittableParams(GroupType gt, int gid,
                                     HittableType ht,
                                     float r, float _p1x,
                                     float _p1y, float _p1z,
                                     float _p2x, float _p2y,
                                     float _p2z, float _p3x,
                                     float _p3y, float _p3z)
      : gtype(gt), group_id(gid), htype(ht), radius(r),
        p1x(_p1x), p1y(_p1y), p1z(_p1z), p2x(_p2x),
        p2y(_p2y), p2z(_p2z), p3x(_p3x), p3y(_p3y),
        p3z(_p3z) {}
  __host__ __device__ HittableParams(GroupType gt, int gid,
                                     HittableType ht,
                                     float r, Point3 p1,
                                     Point3 p2, Point3 p3)
      : gtype(gt), group_id(gid), htype(ht), radius(r),
        p1x(p1.x()), p1y(p1.y()), p1z(p1.z()), p2x(p2.x()),
        p2y(p2.y()), p2z(p2.z()), p3x(p3.x()), p3y(p3.y()),
        p3z(p3.z()) {}

  __host__ __device__ void set_to_zero() {
    group_id = 0;
    gtype = INSTANCE;
    p1x = 0.0f;
    p1y = 0.0f;
    p1z = 0.0f;

    p2x = 0.0f;
    p2y = 0.0f;
    p2z = 0.0f;

    p3x = 0.0f;
    p3y = 0.0f;
    p3z = 0.0f;
    radius = 0.0f;
  }

  __host__ __device__ void mkSphere(Point3 cent, float r) {
    set_to_zero();
    htype = SPHERE;
    radius = r;
    p1x = cent.x();
    p1y = cent.y();
    p1z = cent.z();
  }
  __host__ __device__ void mkSphere(Point3 cent, float r,
                                    int gid, GroupType gt) {
    mkSphere(cent, r);
    group_id = gid;
    gtype = gt;
  }
  __host__ __device__ void mkRect(float a0, //
                                  float a1, //
                                  float b0, //
                                  float b1, //
                                  float _k, //
                                  Vec3 anorm) {
    set_to_zero();
    htype = RECTANGLE;
    p1x = a0;
    p1y = a1;
    p2x = b0;
    p2y = b1;
    p2z = _k;
    p3x = anorm.x();
    p3y = anorm.y();
    p3z = anorm.z();
  }
  __host__ __device__ void mkRect(float a0, //
                                  float a1, //
                                  float b0, //
                                  float b1, //
                                  float _k, //
                                  Vec3 anorm, int gid,
                                  GroupType gt) {
    mkRect(a0, a1, b0, b1, _k, anorm);
    group_id = gid;
    gtype = gt;
  }
};

struct ScenePrimitive {
public:
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

struct SceneObjects {
  // objects
  int *gtypes;
  int *group_ids;
  int *htypes;
  float *rads;
  float *p1xs, *p1ys, *p1zs;
  float *p2xs, *p2ys, *p2zs;
  float *p3xs, *p3ys, *p3zs;

  // materials
  int *mtypes;
  float *fuzzs;

  // textures
  int *ttypes;
  float *tp1xs;
  float *tp1ys;
  float *tp1zs; // solid color, checker texture
  float *tp2xs;
  float *tp2ys;
  float *tp2zs; // solid color, checker texture

  unsigned char *tdata; // image texture
  int tdata_size;       // image texture
  int *ws, *hs;         // image texture
  int *bpps;            // image texture
  int *tindices;        // image texture
  float *scales;        // noise texture
  //
  const int hlength; // objects size

  __host__ __device__ SceneObjects() : hlength(0) {}
  __host__ __device__ SceneObjects(int obj_size)
      : hlength(obj_size) {
    init_arrays();
  }

  __host__ __device__ void free() {
    cudaFree(gtypes);
    cudaFree(group_ids);
    cudaFree(htypes);
    cudaFree(rads);
    cudaFree(p1xs);
    cudaFree(p1ys);
    cudaFree(p1zs);
    cudaFree(p2xs);
    cudaFree(p2ys);
    cudaFree(p2zs);
    cudaFree(p3xs);
    cudaFree(p3ys);
    cudaFree(p3zs);
    cudaFree(mtypes);
    cudaFree(fuzzs);
    cudaFree(ttypes);
    cudaFree(tp1xs);
    cudaFree(tp1ys);
    cudaFree(tp1zs);
    cudaFree(tp2xs);
    cudaFree(tp2ys);
    cudaFree(tp2zs);
    cudaFree(ws);
    cudaFree(hs);
    cudaFree(bpps);
    cudaFree(tindices);
    cudaFree(scales);
  }

  __host__ __device__ void init_arrays() {
    gtypes = new int[hlength];
    group_ids = new int[hlength];
    htypes = new int[hlength];
    rads = new float[hlength];
    p1xs = new float[hlength];
    p1ys = new float[hlength];
    p1zs = new float[hlength];
    p2xs = new float[hlength];
    p2ys = new float[hlength];
    p2zs = new float[hlength];
    p3xs = new float[hlength];
    p3ys = new float[hlength];
    p3zs = new float[hlength];
    mtypes = new int[hlength];
    fuzzs = new float[hlength];
    ttypes = new int[hlength];
    tp1xs = new float[hlength];
    tp1ys = new float[hlength];
    tp1zs = new float[hlength];
    tp2xs = new float[hlength];
    tp2ys = new float[hlength];
    tp2zs = new float[hlength];

    tdata = nullptr;
    tdata_size = 0;
    ws = new int[hlength];
    hs = new int[hlength];
    bpps = new int[hlength];
    tindices = new int[hlength];
    scales = new float[hlength];
  }

  __host__ __device__ SceneObjects(ScenePrimitive *&objs,
                                   int obj_length)
      : hlength(obj_length) {
    init_arrays();
    bool data_check = false;
    for (int i = 0; i < hlength; i++) {
      //
      ScenePrimitive sobj = objs[i];
      gtypes[i] = sobj.gtype;
      group_ids[i] = sobj.group_id;
      htypes[i] = sobj.htype;
      rads[i] = sobj.radius;

      p1xs[i] = sobj.p1x;
      p1ys[i] = sobj.p1y;
      p1zs[i] = sobj.p1z;

      p2xs[i] = sobj.p2x;
      p2ys[i] = sobj.p2y;
      p2zs[i] = sobj.p2z;

      p3xs[i] = sobj.p3x;
      p3ys[i] = sobj.p3y;
      p3zs[i] = sobj.p3z;

      //
      mtypes[i] = sobj.mtype;
      fuzzs[i] = sobj.fuzz;

      //
      ttypes[i] = sobj.ttype;

      tp1xs[i] = sobj.tp1x;
      tp1ys[i] = sobj.tp1y;
      tp1zs[i] = sobj.tp1z;

      tp2xs[i] = sobj.tp2x;
      tp2ys[i] = sobj.tp2y;
      tp2zs[i] = sobj.tp2z;

      //
      ws[i] = sobj.width;
      hs[i] = sobj.height;
      bpps[i] = sobj.bytes_per_pixel;
      tindices[i] = sobj.index;
      scales[i] = sobj.scale;

      if (data_check == false && sobj.data) {
        tdata = sobj.data;
        data_check = true;
        tdata_size = sobj.data_size;
      }
    }
  }

  __host__ SceneObjects to_device() {
    // device copy
    SceneObjects sobjs(hlength);

    int *gts = nullptr;
    CUDA_CONTROL(upload<int>(gts, gtypes, hlength));
    sobjs.gtypes = gts;
    int *gids = nullptr;
    CUDA_CONTROL(upload<int>(gids, group_ids, hlength));
    sobjs.group_ids = gids;

    int *hts = nullptr;
    CUDA_CONTROL(upload<int>(hts, htypes, hlength));
    sobjs.htypes = hts;

    //
    float *rds;
    CUDA_CONTROL(upload<float>(rds, rads, hlength));
    sobjs.rads = rds;

    //
    float *d_p1xs;
    CUDA_CONTROL(upload<float>(d_p1xs, p1xs, hlength));
    sobjs.p1xs = d_p1xs;
    //
    float *d_p1ys;
    CUDA_CONTROL(upload<float>(d_p1ys, p1ys, hlength));
    sobjs.p1ys = d_p1ys;
    //
    float *d_p1zs;
    CUDA_CONTROL(upload<float>(d_p1zs, p1zs, hlength));
    sobjs.p1zs = d_p1zs;
    //
    float *d_p2xs;
    CUDA_CONTROL(upload<float>(d_p2xs, p2xs, hlength));
    sobjs.p2xs = d_p2xs;
    //
    float *d_p2ys;
    CUDA_CONTROL(upload<float>(d_p2ys, p2ys, hlength));
    sobjs.p2ys = d_p2ys;
    //
    float *d_p2zs;
    CUDA_CONTROL(upload<float>(d_p2zs, p2zs, hlength));
    sobjs.p2zs = d_p2zs;
    //
    float *d_p3xs;
    CUDA_CONTROL(upload<float>(d_p3xs, p3xs, hlength));
    sobjs.p3xs = d_p3xs;
    //
    float *d_p3ys;
    CUDA_CONTROL(upload<float>(d_p3ys, p3ys, hlength));
    sobjs.p3ys = d_p3ys;
    //
    float *d_p3zs;
    CUDA_CONTROL(upload<float>(d_p3zs, p3zs, hlength));
    sobjs.p3zs = d_p3zs;
    //
    int *d_mtypes;
    CUDA_CONTROL(upload<int>(d_mtypes, mtypes, hlength));
    sobjs.mtypes = d_mtypes;
    //
    float *d_fuzzs;
    CUDA_CONTROL(upload<float>(d_fuzzs, fuzzs, hlength));
    sobjs.fuzzs = d_fuzzs;
    //
    int *d_ttypes;
    CUDA_CONTROL(upload<int>(d_ttypes, ttypes, hlength));
    sobjs.ttypes = d_ttypes;
    //
    float *d_tp1xs;
    CUDA_CONTROL(upload<float>(d_tp1xs, tp1xs, hlength));
    sobjs.tp1xs = d_tp1xs;
    //
    float *d_tp1ys;
    CUDA_CONTROL(upload<float>(d_tp1ys, tp1ys, hlength));
    sobjs.tp1ys = d_tp1ys;
    //
    float *d_tp1zs;
    CUDA_CONTROL(upload<float>(d_tp1zs, tp1zs, hlength));
    sobjs.tp1zs = d_tp1zs;
    //
    float *d_tp2xs;
    CUDA_CONTROL(upload<float>(d_tp2xs, tp2xs, hlength));
    sobjs.tp2xs = d_tp2xs;
    //
    float *d_tp2ys;
    CUDA_CONTROL(upload<float>(d_tp2ys, tp2ys, hlength));
    sobjs.tp2ys = d_tp2ys;
    //
    float *d_tp2zs;
    CUDA_CONTROL(upload<float>(d_tp2zs, tp2zs, hlength));
    sobjs.tp2zs = d_tp2zs;
    //
    unsigned char *d_tdata;
    CUDA_CONTROL(upload<unsigned char>(d_tdata, tdata,
                                       sobjs.tdata_size));
    sobjs.tdata = d_tdata;
    //
    sobjs.tdata_size = tdata_size;
    //
    int *d_ws;
    CUDA_CONTROL(upload<int>(d_ws, ws, hlength));
    sobjs.ws = d_ws;
    //
    int *d_hs;
    CUDA_CONTROL(upload<int>(d_hs, hs, hlength));
    sobjs.hs = d_hs;
    //
    int *d_bpps;
    CUDA_CONTROL(upload<int>(d_bpps, bpps, hlength));
    sobjs.bpps = d_bpps;
    //
    int *d_tindices;
    CUDA_CONTROL(
        upload<int>(d_tindices, tindices, hlength));
    sobjs.tindices = d_tindices;
    //
    float *d_scales;
    CUDA_CONTROL(upload<float>(d_scales, scales, hlength));
    sobjs.scales = d_scales;
    return sobjs;
  }
  __host__ __device__ HittableParams
  get_hittable_params(int obj_index) {

    GroupType gtype =
        static_cast<GroupType>(gtypes[obj_index]);

    HittableType htype =
        static_cast<HittableType>(htypes[obj_index]);
    HittableParams hp(
        gtype, group_ids[obj_index], htype, rads[obj_index],
        Point3(p1xs[obj_index], p1ys[obj_index],
               p1zs[obj_index]),
        Point3(p2xs[obj_index], p2ys[obj_index],
               p2zs[obj_index]),
        Point3(p3xs[obj_index], p3ys[obj_index],
               p3zs[obj_index]));
    return hp;
  }
  __host__ __device__ ImageParams
  get_img_params(int obj_index) {
    ImageParams imp(ws[obj_index], hs[obj_index],
                    bpps[obj_index], tindices[obj_index]);
    return imp;
  }

  __host__ __device__ MatTextureParams
  get_mat_params(int obj_index) {
    //
    MaterialType mtype =
        static_cast<MaterialType>(mtypes[obj_index]);
    TextureType ttype =
        static_cast<TextureType>(ttypes[obj_index]);

    unsigned char *imdata;
    int imsize = 0;
    if (ttype == IMAGE) {
      imdata = tdata;
      imsize = tdata_size;
    }

    ImageParams imp = get_img_params(obj_index);

    MatTextureParams mp(
        mtype, fuzzs[obj_index], ttype,
        Color(tp1xs[obj_index], tp1ys[obj_index],
              tp1zs[obj_index]),
        Color(tp2xs[obj_index], tp2ys[obj_index],
              tp2zs[obj_index]),
        imdata, imsize, imp, scales[obj_index]);
    return mp;
  }
  __host__ __device__ ScenePrimitive get(int obj_index) {
    HittableParams hp = get_hittable_params(obj_index);
    MatTextureParams mp = get_mat_params(obj_index);

    ScenePrimitive sobj(hp, mp);
    return sobj;
  }
  __device__ ScenePrimitive get(int obj_index,
                                curandState *loc) {

    HittableParams hp = get_hittable_params(obj_index);
    MatTextureParams mp = get_mat_params(obj_index);

    ScenePrimitive sobj(hp, mp, loc);

    return sobj;
  }
  __host__ __device__ void
  to_hittable_list(Hittable **&hs) {
    for (int i = 0; i < hlength; i++) {

      ScenePrimitive s = get(i);
      Hittable *h;
      s.to_obj(h);
      hs[i] = h;
    }
  }
  __device__ void to_hittable_list(Hittable **&hs,
                                   curandState *loc) {
    for (int i = 0; i < hlength; i++) {

      ScenePrimitive s = get(i, loc);
      Hittable *h;
      s.to_obj_device(h);
      hs[i] = h;
    }
  }
};

#endif
