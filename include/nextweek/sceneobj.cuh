#ifndef SCENEOBJ_CUH
#define SCENEOBJ_CUH
// an interface scene object between different primitives

#include <nextweek/aarect.cuh>
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

struct SceneObj {
public:
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
  int width, height;      // image texture
  int bytes_per_pixel;    // image texture
  int index;              // image texture
  float scale;            // noise texture

  curandState *rstate;

  __host__ __device__ SceneObj()
      : htype(TRIANGLE), mtype(LAMBERT), ttype(SOLID) {}
  __host__ __device__ SceneObj(
      HittableType ht, float rad, double c1x, double c1y,
      float c1z, double c2x, double c2y, double c2z,
      float c3x, double c3y, double c3z, MaterialType mt,
      float f, TextureType tp, double tx, float ty,
      float tz, float tx2, float ty2, float tz2,
      unsigned char *&td, int w, int h, int idx, float sc)
      : htype(ht), radius(rad), p1x(c1x), p1y(c1y),
        p1z(c1z), p2x(c2x), p2y(c2y), p2z(c2z), p3x(c3x),
        p3y(c3y), p3z(c3z), mtype(mt), fuzz(f), ttype(tp),
        tp1x(tx), tp1y(ty), tp1z(tz), tp2x(tx2), tp2y(ty2),
        tp2z(tz2), width(w), height(h), index(idx),
        scale(sc), rstate(nullptr) {}

  __host__ __device__ SceneObj(HittableType ht, float rad,
                               Point3 c1, Point3 c2,
                               Point3 c3, MaterialType mt,
                               float f, TextureType tp,
                               Color clr1, Color clr2,
                               unsigned char *&td, int w,
                               int h, int idx, float sc)
      : htype(ht), radius(rad), p1x(c1.x()), p1y(c1.y()),
        p1z(c1.x()), p2x(c2.x()), p2y(c2.y()), p2z(c2.z()),
        p3x(c3.x()), p3y(c3.y()), p3z(c3.z()), mtype(mt),
        fuzz(f), ttype(tp), tp1x(clr1.x()), tp1y(clr1.y()),
        tp1z(clr1.z()), tp2x(clr2.x()), tp2y(clr2.y()),
        tp2z(clr2.z()), width(w), height(h), index(idx),
        scale(sc), rstate(nullptr) {}
  __device__
  SceneObj(HittableType ht, float rad, double c1x,
           float c1y, float c1z, float c2x, double c2y,
           float c2z, float c3x, float c3y, double c3z,
           MaterialType mt, float f, TextureType tp,
           float tx, double ty, float tz, float tx2,
           float ty2, float tz2, unsigned char *&td, int w,
           int h, int idx, float sc, curandState *rs)
      : htype(ht), radius(rad), p1x(c1x), p1y(c1y),
        p1z(c1z), p2x(c2x), p2y(c2y), p2z(c2z), p3x(c3x),
        p3y(c3y), p3z(c3z), mtype(mt), fuzz(f), ttype(tp),
        tp1x(tx), tp1y(ty), tp1z(tz), tp2x(tx2), tp2y(ty2),
        tp2z(tz2), width(w), height(h), index(idx),
        scale(sc), rstate(rs) {}

  __device__ SceneObj(HittableType ht, float rad, Point3 c1,
                      Point3 c2, Point3 c3, MaterialType mt,
                      float f, TextureType tp, Color clr1,
                      Color clr2, unsigned char *&td, int w,
                      int h, int idx, float sc,
                      curandState *rs)
      : htype(ht), radius(rad), p1x(c1.x()), p1y(c1.y()),
        p1z(c1.x()), p2x(c2.x()), p2y(c2.y()), p2z(c2.z()),
        p3x(c3.x()), p3y(c3.y()), p3z(c3.z()), mtype(mt),
        fuzz(f), ttype(tp), tp1x(clr1.x()), tp1y(clr1.y()),
        tp1z(clr1.z()), tp2x(clr2.x()), tp2y(clr2.y()),
        tp2z(clr2.z()), width(w), height(h), index(idx),
        scale(sc), rstate(rs) {}

  __host__ __device__ void set_material_props(
      MaterialType mt, float f, TextureType tt, float _tp1x,
      float _tp1y,
      float _tp1z, // solid color, checker texture
      float _tp2x, float _tp2y,
      float _tp2z,             // checker texture
      unsigned char *&_data,   // image texture
      int _width, int _height, // image texture
      int _bpp,                // image texture
      int idx,                 // image texture
      float sc                 // noise texture
      ) {
    mtype = mt;
    fuzz = f;
    ttype = tt;
    tp1x = _tp1x;
    tp1y = _tp1y;
    tp1z = _tp1z;
    tp2x = _tp1x;
    tp1y = _tp1y;
    tp1z = _tp1z;
    data = _data;
    width = _width;
    height = _height;
    index = idx;
    scale = sc;
  }
  __device__ void set_material_props(
      MaterialType mt, float f, TextureType tt, float _tp1x,
      float _tp1y,
      float _tp1z, // solid color, checker texture
      float _tp2x, float _tp2y,
      float _tp2z,             // checker texture
      unsigned char *&_data,   // image texture
      int _width, int _height, // image texture
      int _bpp,                // image texture
      int idx,                 // image texture
      float sc, curandState *loc) {
    set_material_props(mt, f, tt, _tp1x, _tp1y, _tp1z,
                       _tp2x, _tp2y, _tp2z, _data, _width,
                       _height, _bpp, idx, sc);
    rstate = loc;
  }

  __host__ __device__ void
  mkSphere(Point3 cen, float r, //
           //
           MaterialType mt, float f, TextureType tt,
           float _tp1x, float _tp1y,
           float _tp1z, // solid color, checker texture
           float _tp2x, float _tp2y,
           float _tp2z,             // checker texture
           unsigned char *&_data,   // image texture
           int _width, int _height, // image texture
           int _bpp,                // image texture
           int idx,                 // image texture
           float sc                 // noise texture
           ) {
    radius = r;
    p1x = cen.x();
    p1y = cen.y();
    p1z = cen.z();
    set_material_props(mt, f, tt, _tp1x, _tp1y, _tp1z,
                       _tp2x, _tp2y, _tp2z, _data, _width,
                       _height, _bpp, idx, sc);
  }
  __device__ void
  mkSphere(Point3 cen, float r, MaterialType mt, float f,
           TextureType tt, float _tp1x, float _tp1y,
           float _tp1z, // solid color, checker texture
           float _tp2x, float _tp2y,
           float _tp2z,             // checker texture
           unsigned char *&_data,   // image texture
           int _width, int _height, // image texture
           int _bpp,                // image texture
           int idx,                 // image texture
           float sc,                // noise texture
           curandState *rst) {
    radius = r;
    p1x = cen.x();
    p1y = cen.y();
    p1z = cen.z();
    set_material_props(mt, f, tt, _tp1x, _tp1y, _tp1z,
                       _tp2x, _tp2y, _tp2z, _data, _width,
                       _height, _bpp, idx, sc, rst);
  }

  __host__ __device__ void
  mkRect(float a0,        //
         float a1,        //
         float b0,        //
         float b1,        //
         float _k,        //
         Vec3 anorm,      //
         MaterialType mt, //
         float f,         //
         TextureType tt,  //
         float _tp1x,     //
         float _tp1y,     //
         float _tp1z,     // solid color, checker texture
         float _tp2x,     //
         float _tp2y,     //
         float _tp2z,     // checker texture
         unsigned char *&_data, // image texture
         int _width,            //
         int _height,           // image texture
         int _bpp,              // image texture
         int idx,               // image texture
         float sc               // noise texture
         ) {
    htype = RECTANGLE;
    p1x = a0;
    p1y = a1;
    p2x = b0;
    p2y = b1;
    p2z = _k;
    p3x = anorm.x();
    p3y = anorm.y();
    p3z = anorm.z();
    set_material_props(mt, f, tt, _tp1x, _tp1y, _tp1z,
                       _tp2x, _tp2y, _tp2z, _data, _width,
                       _height, _bpp, idx, sc);
  }
  __device__ void
  mkRect(float a0, float a1, float b0, float b1, float _k,
         Vec3 anorm, //
         MaterialType mt, float f, TextureType tt,
         float _tp1x, float _tp1y,
         float _tp1z, // solid color, checker texture
         float _tp2x, float _tp2y,
         float _tp2z,               // checker texture
         unsigned char *&_data,     // image texture
         int _width, int _height,   // image texture
         int _bpp,                  // image texture
         int idx,                   // image texture
         float sc, curandState *rst // noise texture
         ) {
    htype = RECTANGLE;
    p1x = a0;
    p1y = a1;
    p2x = b0;
    p2y = b1;
    p2z = _k;
    p3x = anorm.x();
    p3y = anorm.y();
    p3z = anorm.z();
    set_material_props(mt, f, tt, _tp1x, _tp1y, _tp1z,
                       _tp2x, _tp2y, _tp2z, _data, _width,
                       _height, _bpp, idx, sc, rst);
  }

  __host__ __device__ void
  to_solid_color(Texture *&txt) const {
    txt = new SolidColor(tp1x, tp1y, tp1z);
  }
  __host__ __device__ void to_checker(Texture *&txt) const {
    Color s1(tp1x, tp1y, tp1z);
    Color s2(tp2x, tp2y, tp2z);
    txt = new CheckerTexture(s1, s2);
  }
  __host__ __device__ void to_image(Texture *&txt) const {

    txt = new ImageTexture(data, width, height,
                           width * bytes_per_pixel,
                           bytes_per_pixel, index);
  }
  __device__ void to_noise(Texture *&txt) const {
    txt = new NoiseTexture(scale, rstate);
  }
  __host__ __device__ void to_texture(Texture *&t) const {
    if (ttype == SOLID) {
      to_solid_color(t);
    } else if (ttype == CHECKER) {
      to_checker(t);
    } else if (ttype == IMAGE) {
      to_image(t);
    }
  }
  __host__ __device__ void
  to_lambertian(Material *&m) const {
    Texture *t;
    to_texture(t);
    m = new Lambertian(t);
  }
  __device__ void to_lambertian_noise(Material *&m) const {
    Texture *t;
    to_noise(t);
    m = new Lambertian(t);
  }
  __host__ __device__ void to_metal(Material *&m) const {
    Texture *t;
    to_texture(t);
    m = new Metal(t, fuzz);
  }

  __device__ void to_metal_noise(Material *&m) const {
    Texture *t;
    to_noise(t);
    m = new Metal(t, fuzz);
  }
  __host__ __device__ void
  to_dielectric(Material *&m) const {
    m = new Dielectric(fuzz);
  }
  __host__ __device__ void
  to_diffuse_light(Material *&m) const {
    Texture *t;
    to_texture(t);
    m = new DiffuseLight(t);
  }
  __device__ void
  to_diffuse_light_noise(Material *&m) const {
    Texture *t;
    to_noise(t);
    m = new DiffuseLight(t);
  }
  __host__ __device__ void
  to_isotropic(Material *&m) const {
    Texture *t;
    to_texture(t);
    m = new Isotropic(t);
  }
  __device__ void to_isotropic_noise(Material *&m) const {
    Texture *t;
    to_noise(t);
    m = new Isotropic(t);
  }
  __host__ __device__ void
  to_material(Material *&mat) const {
    switch (mtype) {
    case LAMBERT:
      to_lambertian(mat);
      break;
    case METAL:
      to_metal(mat);
      break;
    case DIELECTRIC:
      to_dielectric(mat);
      break;
    case DIFFUSE_LIGHT:
      to_diffuse_light(mat);
      break;
    case ISOTROPIC:
      to_isotropic(mat);
      break;
    }
  }
  __device__ void to_material_noise(Material *&mat) const {
    switch (mtype) {
    case LAMBERT:
      to_lambertian_noise(mat);
      break;
    case METAL:
      to_metal_noise(mat);
      break;
    case DIELECTRIC:
      to_dielectric(mat);
      break;
    case DIFFUSE_LIGHT:
      to_diffuse_light_noise(mat);
      break;
    case ISOTROPIC:
      to_isotropic_noise(mat);
      break;
    }
  }
  __host__ __device__ void to_sphere(Hittable *&h) {
    Material *m;
    to_material(m);
    h = new Sphere(Point3(p1x, p1y, p1z), radius, m);
  }
  __device__ void to_sphere_noise(Hittable *&h) {
    Material *m;
    to_material_noise(m);
    h = new Sphere(Point3(p1x, p1y, p1z), radius, m);
  }
  __host__ __device__ void to_moving_sphere(Hittable *&h) {
    Material *m;
    to_material(m);
    h = new MovingSphere(Point3(p1x, p1y, p1z),
                         Point3(p2x, p2y, p2z), p3x, p3y,
                         radius, m);
  }
  __device__ void to_moving_sphere_noise(Hittable *&h) {
    Material *m;
    to_material_noise(m);
    h = new MovingSphere(Point3(p1x, p1y, p1z),
                         Point3(p2x, p2y, p2z), p3x, p3y,
                         radius, m);
  }
  __host__ __device__ void to_aarect(Hittable *&h) {
    Material *m;
    to_material(m);
    h = new AaRect(p1x, p1y, p2x, p2y, p2z, m,
                   Vec3(p3x, p3y, p3z));
  }
  __device__ void to_aarect_noise(Hittable *&h) {
    Material *m;
    to_material_noise(m);
    h = new AaRect(p1x, p1y, p2x, p2y, p2z, m,
                   Vec3(p3x, p3y, p3z));
  }
  __host__ __device__ void to_triangle(Hittable *&h) {
    Material *m;
    to_material(m);
    h = new Triangle(Point3(p1x, p1y, p1z),
                     Point3(p2x, p2y, p2z),
                     Point3(p3x, p3y, p3z), m);
  }
  __device__ void to_triangle_noise(Hittable *&h) {
    Material *m;
    to_material_noise(m);
    h = new Triangle(Point3(p1x, p1y, p1z),
                     Point3(p2x, p2y, p2z),
                     Point3(p3x, p3y, p3z), m);
  }

  __host__ __device__ void to_obj(Hittable *&h) {
    if (htype == TRIANGLE) {
      to_triangle(h);
    } else if (htype == SPHERE) {
      to_sphere(h);
    } else if (htype == MOVING_SPHERE) {
      to_moving_sphere(h);
    } else if (htype == RECTANGLE) {
      to_aarect(h);
    }
  }
  __device__ void to_obj_device(Hittable *&h) {
    if (htype == TRIANGLE) {
      if (ttype == NOISE) {
        to_triangle_noise(h);
      } else {
        to_triangle(h);
      }
    } else if (htype == SPHERE) {
      if (ttype == NOISE) {
        to_sphere_noise(h);
      } else {
        to_sphere(h);
      }
    } else if (htype == MOVING_SPHERE) {
      if (ttype == NOISE) {
        to_moving_sphere_noise(h);
      } else {
        to_moving_sphere(h);
      }
    } else if (htype == RECTANGLE) {
      if (ttype == RECTANGLE) {
        to_aarect_noise(h);
      } else {
        to_aarect(h);
      }
    }
  }
};

struct SceneObjects {
  // objects
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
    ws = new int[hlength];
    hs = new int[hlength];
    bpps = new int[hlength];
    tindices = new int[hlength];
    scales = new float[hlength];
  }

  __host__ __device__ SceneObjects(SceneObj *&objs,
                                   int obj_length)
      : hlength(obj_length) {
    init_arrays();
    bool data_check = false;
    for (int i = 0; i < hlength; i++) {
      //
      SceneObj sobj = objs[i];
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
      }
    }
  }

  __host__ __device__ SceneObj get(int obj_index) {
    HittableType htype =
        static_cast<HittableType>(htypes[obj_index]);
    MaterialType mtype =
        static_cast<MaterialType>(mtypes[obj_index]);
    TextureType ttype =
        static_cast<TextureType>(ttypes[obj_index]);

    unsigned char *imdata;
    if (ttype == IMAGE) {
      imdata = tdata;
    }
    SceneObj sobj(htype, rads[obj_index],
                  Point3(p1xs[obj_index], p1ys[obj_index],
                         p1zs[obj_index]),
                  Point3(p2xs[obj_index], p2ys[obj_index],
                         p2zs[obj_index]),
                  Point3(p3xs[obj_index], p3ys[obj_index],
                         p3zs[obj_index]),
                  mtype, fuzzs[obj_index], ttype,
                  Color(tp1xs[obj_index], tp1ys[obj_index],
                        tp1zs[obj_index]),
                  Color(tp2xs[obj_index], tp2ys[obj_index],
                        tp2zs[obj_index]),
                  imdata, ws[obj_index], hs[obj_index],
                  tindices[obj_index], scales[obj_index]);
    return sobj;
  }
  __device__ SceneObj get(int obj_index, curandState *loc) {

    int htype = htypes[obj_index];
    unsigned char *imdata;
    if (htype == IMAGE) {
      imdata = tdata;
    }

    SceneObj sobj(
        static_cast<HittableType>(htypes[obj_index]),
        rads[obj_index], p1xs[obj_index], p1ys[obj_index],
        p1zs[obj_index], p2xs[obj_index], p2ys[obj_index],
        p2zs[obj_index], p3xs[obj_index], p3ys[obj_index],
        p3zs[obj_index],
        static_cast<MaterialType>(mtypes[obj_index]),
        fuzzs[obj_index],
        static_cast<TextureType>(ttypes[obj_index]),
        tp1xs[obj_index], tp1ys[obj_index],
        tp1zs[obj_index], tp2xs[obj_index],
        tp2ys[obj_index], tp2zs[obj_index], imdata,
        ws[obj_index], hs[obj_index], tindices[obj_index],
        scales[obj_index], loc);

    return sobj;
  }
  __host__ __device__ void
  to_hittable_list(Hittable **&hs) {
    for (int i = 0; i < hlength; i++) {
      Hittable *h;
      SceneObj s = get(i);
      s.to_obj(h);
      hs[i] = h;
    }
  }
  __device__ void to_hittable_list(Hittable **&hs,
                                   curandState *loc) {
    for (int i = 0; i < hlength; i++) {
      Hittable *h;
      SceneObj s = get(i, loc);
      s.to_obj_device(h);
      hs[i] = h;
    }
  }
};

#endif
