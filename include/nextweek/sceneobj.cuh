#ifndef SCENEOBJ_CUH
#define SCENEOBJ_CUH
// an interface scene object between different primitives

#include <nextweek/aarect.cuh>
#include <nextweek/hittable.cuh>
#include <nextweek/material.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/sphere.cuh>
#include <nextweek/vec3.cuh>

enum TextureType : int {
  SOLID = 1,
  CHECKER = 2,
  NOISE = 3,
  IMAGE = 4
};

struct SceneTexture {
  TextureType ttype;

  double p1x, p1y, p1z; // solid color, checker texture
  unsigned char *data;  // image texture
  int width, height;    // image texture
  int bytes_per_pixel;  // image texture
  int index;            // image texture
  double scale;         // noise texture

  __host__ __device__ SceneTexture() {}
  __host__ __device__ SceneTexture(TextureType ty,
                                   double c1, double c2,
                                   double c3,
                                   unsigned char *&ds,
                                   int w, int h, int bpp,
                                   int idx, double s)
      : ttype(ty), p1x(c1), p1y(c2), p1z(c3), data(ds),
        width(w), height(h), bytes_per_pixel(bpp),
        index(idx), scale(s) {}

  __host__ __device__ SceneTexture(TextureType ty, Color c,
                                   unsigned char *&ds,
                                   int w, int h, int bpp,
                                   int idx, double s)
      : SceneTexture(ty, c.x(), c.y(), c.z(), ds, w, h, bpp,
                     idx, s) {}
  __host__ __device__ SceneTexture(SolidColor &scolor)
      : ttype(SOLID), data(nullptr), width(0), height(0),
        bytes_per_pixel(0), index(0), scale(0.0f) {
    Color c = scolor.value(0.0f, 0.0f, Point3(0, 0, 0));
    p1x = c.x();
    p1y = c.y();
    p1z = c.z();
  }
  __host__ __device__ SceneTexture(CheckerTexture &ct)
      : ttype(CHECKER), width(0), height(0), data(nullptr),
        bytes_per_pixel(0), index(0), scale(0.0f) {
    Color c = ct.odd->value(0.0f, 0.0f, Point3(0, 0, 0));
    p1x = c.x();
    p1y = c.y();
    p1z = c.z();
  }

  __host__ __device__ SceneTexture(NoiseTexture &nt)
      : ttype(NOISE), p1x(0), p1y(0), p1z(0), data(nullptr),
        width(0), height(0), bytes_per_pixel(0), index(0),
        scale(nt.scale) {}

  __host__ __device__ SceneTexture(ImageTexture &img)
      : ttype(IMAGE), p1x(0), p1y(0), p1z(0),
        width(img.width), height(img.height),
        data(img.data),
        bytes_per_pixel(img.bytes_per_pixel),
        index(img.index), scale(0.0f) {}

  __host__ __device__ SolidColor to_solid() const {
    return SolidColor(Color(p1x, p1y, p1z));
  }
  __host__ __device__ CheckerTexture to_checker() const {
    return CheckerTexture(Color(p1x, p1y, p1z));
  }
  __device__ NoiseTexture to_noise(curandState *loc) const {
    return NoiseTexture(scale, loc);
  }
  __host__ __device__ ImageTexture to_image() const {
    return ImageTexture(data, width, height,
                        width * bytes_per_pixel,
                        bytes_per_pixel);
  }
  template <typename T>
  __host__ __device__ T to_texture() const {
    if (ttype == SOLID) {
      return to_solid();
    } else if (ttype == CHECKER) {
      return to_checker();
    } else if (ttype == IMAGE) {
      return to_image();
    }
  }
  __device__ NoiseTexture
  to_texture(curandState *loc) const {
    return to_noise(loc);
  }
};

enum MaterialType : int {
  LAMBERT = 1,
  METAL = 2,
  DIELECTRIC = 3,
  DIFFUSE_LIGHT = 4,
  ISOTROPIC = 5
};

struct SceneMaterial {
  MaterialType mtype;
  SceneTexture st;
  double fuzz;
  __host__ __device__ SceneMaterial(SceneTexture &s,
                                    MaterialType m)
      : st(s), mtype(m) {}
  __host__ __device__ SceneMaterial(Color &c,
                                    MaterialType m)
      : mtype(m) {
    SolidColor s = SolidColor(c);
    st = SceneTexture(s);
  }

  __host__ __device__ SceneMaterial(SceneTexture &s,
                                    double f,
                                    MaterialType m)
      : st(s), mtype(m), fuzz(f) {}

  __host__ __device__ SceneMaterial(Color &c, double f,
                                    MaterialType m)
      : mtype(m), fuzz(f) {
    SolidColor s = SolidColor(c);
    st = SceneTexture(s);
  }

  __host__ __device__ SceneMaterial(double ri,
                                    MaterialType m)
      : mtype(m), fuzz(ri) {}

  __host__ __device__ SceneMaterial(Lambertian &lamb)
      : SceneMaterial(SceneTexture((*lamb.albedo)),
                      LAMBERT) {}
  __host__ __device__ SceneMaterial(Metal &met)
      : mtype(METAL), fuzz(met.fuzz) {
    auto text = (*met.albedo);
    st = SceneTexture(text);
  }
  __host__ __device__ SceneMaterial(Dielectric &diel)
      : SceneMaterial(SceneTexture(), diel.ref_idx,
                      DIELECTRIC) {}
  __host__ __device__ SceneMaterial(DiffuseLight &dl)
      : SceneMaterial(SceneTexture(dl.emit),
                      DIFFUSE_LIGHT) {}
  __host__ __device__ SceneMaterial(Isotropic &iso)
      : SceneMaterial(SceneTexture(iso.albedo), ISOTROPIC) {
  }
};

enum HittableType: int {
    LINE = 1, TRIANGLE = 2, SPHERE = 3, MOVING_SPHERE = 4, RECTANGLE = 5
}

struct SceneObj {
  // object related
  HittableType htype;
  double radius;
  double p1x, p1y, p1z;
  double p2x, p2y, p2z;
  double p3x, p3y, p3z;
  //
  // material related
  MaterialType mtype;
  double fuzz;

  // texture related
  TextureType ttype;

  double tp1x, tp1y, tp1z; // solid color, checker texture
  double tp2x, tp2y, tp2z; // checker texture
  unsigned char *data;     // image texture
  int width, height;       // image texture
  int bytes_per_pixel;     // image texture
  int index;               // image texture
  double scale;            // noise texture

  curandState *rstate;

  __host__ __device__ SceneObj() {}
  __host__ __device__ SceneObj(
      HittableType ht, double rad, double c1x, double c1y,
      double c1z, double c2x, double c2y, double c2z,
      double c3x, double c3y, double c3z, MaterialType mt,
      double f, TextureType tp, double tx, double ty,
      double tz, double tx2, double ty2, double tz2,
      unsigned char *&td, int w, int h, int idx, double sc)
      : htype(ht), mat(sm), radius(rad), p1x(c1x), p1y(c1y),
        p1z(c1z), p2x(c2x), p2y(c2y), p2z(c2z), p3x(c3x),
        p3y(c3y), p3z(c3z), mtype(mt), fuzz(f), ttype(tp),
        tp1x(tx), tp1y(ty), tp1z(tz), tp2x(tx2), tp2y(ty2),
        tp2z(tz2), width(w), height(h), index(idx),
        scale(sc), rstate(nullptr) {}

  __host__ __device__ SceneObj(HittableType ht, double rad,
                               Point3 c1, Point3 c2,
                               Point3 c3, MaterialType mt,
                               double f, TextureType tp,
                               Color clr1, Color clr2,
                               unsigned char *&td, int w,
                               int h, int idx, double sc)
      : htype(ht), mat(sm), radius(rad), p1x(c1.x()),
        p1y(c1.y()), p1z(c1.x()), p2x(c2.x()), p2y(c2.y()),
        p2z(c2.z()), p3x(c3.x()), p3y(c3.y()), p3z(c3.z()),
        mtype(mt), fuzz(f), ttype(tp), tp1x(clr1.x()),
        tp1y(clr1.y()), tp1z(clr1.z()), tp2x(clr2.x()),
        tp2y(clr2.y()), tp2z(clr2.z()), width(w), height(h),
        index(idx), scale(sc) rstate(nullptr) {}
  __device__
  SceneObj(HittableType ht, double rad, double c1x,
           double c1y, double c1z, double c2x, double c2y,
           double c2z, double c3x, double c3y, double c3z,
           MaterialType mt, double f, TextureType tp,
           double tx, double ty, double tz, double tx2,
           double ty2, double tz2, unsigned char *&td,
           int w, int h, int idx, double sc,
           curandState *rs)
      : htype(ht), mat(sm), radius(rad), p1x(c1x), p1y(c1y),
        p1z(c1z), p2x(c2x), p2y(c2y), p2z(c2z), p3x(c3x),
        p3y(c3y), p3z(c3z), mtype(mt), fuzz(f), ttype(tp),
        tp1x(tx), tp1y(ty), tp1z(tz), tp2x(tx2), tp2y(ty2),
        tp2z(tz2), width(w), height(h), index(idx),
        scale(sc), rstate(rs) {}

  __host__ __device__ SceneObj(HittableType ht, double rad,
                               Point3 c1, Point3 c2,
                               Point3 c3, MaterialType mt,
                               double f, TextureType tp,
                               Color clr1, Color clr2,
                               unsigned char *&td, int w,
                               int h, int idx, double sc)
      : htype(ht), mat(sm), radius(rad), p1x(c1.x()),
        p1y(c1.y()), p1z(c1.x()), p2x(c2.x()), p2y(c2.y()),
        p2z(c2.z()), p3x(c3.x()), p3y(c3.y()), p3z(c3.z()),
        mtype(mt), fuzz(f), ttype(tp), tp1x(clr1.x()),
        tp1y(clr1.y()), tp1z(clr1.z()), tp2x(clr2.x()),
        tp2y(clr2.y()), tp2z(clr2.z()), width(w), height(h),
        index(idx), scale(sc) rstate(nullptr) {}

  __device__
  SceneObj(HittableType ht, double rad, Point3 c1,
           Point3 c2, Point3 c3, MaterialType mt, double f,
           TextureType tp, Color clr1, Color clr2,
           unsigned char *&td, int w, int h, int idx,
           double sc, curandState *rs)
      : htype(ht), mat(sm), radius(rad), p1x(c1.x()),
        p1y(c1.y()), p1z(c1.x()), p2x(c2.x()), p2y(c2.y()),
        p2z(c2.z()), p3x(c3.x()), p3y(c3.y()), p3z(c3.z()),
        mtype(mt), fuzz(f), ttype(tp), tp1x(clr1.x()),
        tp1y(clr1.y()), tp1z(clr1.z()), tp2x(clr2.x()),
        tp2y(clr2.y()), tp2z(clr2.z()), width(w), height(h),
        index(idx), scale(sc) rstate(rs) {}

  __host__ __device__ SolidColor to_solid_color() const {
    return SolidColor(tp1x, tp1y, tp1z);
  }
  __host__ __device__ CheckerTexture to_checker() const {
    Color s1(tp1x, tp1y, tp1z);
    Color s2(tp2x, tp2y, tp2z);
    return CheckerTexture(s1, s2);
  }
  __host__ __device__ ImageTexture to_image() const {

    return ImageTexture(data, width, height,
                        width * bytes_per_pixel,
                        bytes_per_pixel, index);
  }
  __device__ NoiseTexture to_noise() const {
    return NoiseTexture(scale, rstate);
  }
  template <typename T = Texture>
  __host__ T to_texture() const {
    if (ttype == SOLID) {
      return to_solid_color();
    } else if (ttype == CHECKER) {
      return to_checker();
    } else if (ttype == IMAGE) {
      return to_image();
    }
  }
  template <typename T = Texture>
  __device__ T to_texture() const {
    if (ttype == SOLID) {
      return to_solid_color();
    } else if (ttype == CHECKER) {
      return to_checker();
    } else if (ttype == IMAGE) {
      return to_image();
    } else {
      return to_noise();
    }
  }
  template <typename T = Texture>
  __host__ __device__ Lambertian to_lambertian() const {
    T text = to_texture<T>();
    Lambertian lamb(&text);
    return lamb;
  }
  template <typename T = Texture>
  __host__ __device__ Metal to_metal() const {
    T text = to_texture<T>();
    Metal met(&text, fuzz);
    return met;
  }
  __host__ __device__ Dielectric to_dielectric() const {
    Dielectric die(fuzz);
    return die;
  }
  template <typename T = Texture>
  __host__ __device__ DiffuseLight
  to_diffuse_light() const {
    T text = to_texture<T>();
    DiffuseLight met(&text);
    return met;
  }
  template <typename T = Texture>
  __host__ __device__ Isotropic to_isotropic() const {
    T text = to_texture<T>();
    Isotropic isot(&text);
    return isot;
  }
  template <class Mat = Material, class Tex = Texture>
  __host__ __device__ Mat to_material() const {
    if (mtype == LAMBERT) {
      return to_lambertian<Tex>();
    } else if (mtype == METAL) {
      return to_metal<Tex>();
    } else if (mtype == DIELECTRIC) {
      return to_dielectric();
    } else if (mtype == DIFFUSE_LIGHT) {
      return to_diffuse_light<Tex>();
    } else {
      return to_isotropic<Tex>();
    }
  }
  template <class Mat = Material, class Tex = Texture>
  __host__ __device__ Sphere to_sphere() const {
    Mat m = to_material<Mat, Tex>();
    Sphere s(Point3(p1x, p1y, p1z), radius, &m);
    return s;
  }
  template <class Mat = Material, class Tex = Texture>
  __host__ __device__ MovingSphere
  to_moving_sphere() const {
    Mat m = to_material<Mat, Tex>();
    MovingSphere s(Point3(p1x, p1y, p1z),
                   Point3(p2x, p2y, p2z), p3x, p3y, radius,
                   &m);
    return s;
  }
  template <class Mat = Material, class Tex = Texture>
  __host__ __device__ AaRect to_aarect() const {
    Mat m = to_material<Mat, Tex>();
    AaRect rect(p1x, p1y, p1z, p2x, p2y, p2z, &m,
                Vec3(p3x, p3y, p3z));
    return rect;
  }
  template <class Mat = Material, class Tex = Texture>
  __host__ __device__ Triangle to_triangle() const {
    Mat m = to_material<Mat, Tex>();
    Triangle tri(Point3(p1x, p1y, p1z),
                 Point3(p2x, p2y, p2z),
                 Point3(p3x, p3y, p3z), &m);
    return tri;
  }
  template <class Obj = Hittable, class Mat = Material,
            class Tex = Texture>
  __host__ __device__ Obj to_obj() const {
    if (htype == TRIANGLE) {
      return to_triangle<Mat, Tex>();
    } else if (htype == SPHERE) {
      return to_sphere<Mat, Tex>();
    } else if (htype == MOVING_SPHERE) {
      return to_moving_sphere<Mat, Tex>();
    } else if (htype == RECTANGLE) {
      return to_aarect<Mat, Tex>();
    }
  }
};

struct SceneObjects {
  // objects
  int *htypes;
  double *rads;
  double *p1xs, *p1ys, *p1zs;
  double *p2xs, *p2ys, *p2zs;
  double *p3xs, *p3ys, *p3zs;

  // materials
  int *mtypes;
  double *fuzzs;

  // textures
  int *ttypes;
  double *tp1xs;
  double *tp1ys;
  double *tp1zs; // solid color, checker texture
  double *tp2xs;
  double *tp2ys;
  double *tp2zs; // solid color, checker texture

  unsigned char *tdata; // image texture
  int *ws, *hs;         // image texture
  int *bpps;            // image texture
  int *tindices;        // image texture
  double *scales;       // noise texture
  //
  const int hlength; // objects size

  __host__ __device__ SceneObjects()
      : htypes(nullptr), rads(nullptr), p1x(nullptr),
        p1y(nullptr), p1z(nullptr), p2x(nullptr),
        p2y(nullptr), p2z(nullptr), p3x(nullptr),
        p3y(nullptr), p3z(nullptr), mtypes(nullptr),
        fuzzs(nullptr), ttypes(nullptr), tp1x(nullptr),
        tp1y(nullptr), tp1z(nullptr), tdata(nullptr),
        ws(nullptr), hs(nullptr), bpps(nullptr),
        tindices(nullptr), scales(nullptr), hlength(0) {}
  __host__ __device__ SceneObjects(int obj_size)
      : hlength(obj_size) {
    init_arrays();
  }

  __host__ __device__ void init_arrays() {
    htypes = new int[hlength];
    rads = new double[hlength];
    p1x = new double[hlength];
    p1y = new double[hlength];
    p1z = new double[hlength];
    p2x = new double[hlength];
    p2y = new double[hlength];
    p2z = new double[hlength];
    p3x = new double[hlength];
    p3y = new double[hlength];
    p3z = new double[hlength];
    mtypes = new int[hlength];
    fuzzs = new double[hlength];
    ttypes = new int[hlength];
    tp1x = new double[hlength];
    tp1y = new double[hlength];
    tp1z = new double[hlength];
    tdata = nullptr;
    ws = new int[hlength];
    hs = new int[hlength];
    bpps = new int[hlength];
    tindices = new int[hlength];
    scales = new double[hlength];
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
      mtypes[i] = sobj.mat.mtype;
      fuzzs[i] = sobj.mat.fuzz;

      //
      ttypes[i] = sobj.mat.st.ttype;
      tp1xs[i] = sobj.mat.st.p1x;
      tp1ys[i] = sobj.mat.st.p1y;
      tp1zs[i] = sobj.mat.st.p1z;
      ws[i] = sobj.mat.st.width;
      hs[i] = sobj.mat.st.height;
      bpps[i] = sobj.mat.st.bytes_per_pixel;
      tindices[i] = sobj.mat.st.index;
      scales[i] = sobj.mat.st.scale;
      //
      if (data_check == false && sobj.mat.st.data) {
        tdata = sobj.mat.st.data;
        data_check = true;
      }
    }
  }

  __host__ SceneObj get(int obj_index) {
    int htype = htypes[obj_index];
    SceneObj sobj(
        htypes[obj_index], rads[obj_index], p1xs[obj_index],
        p1ys[obj_index], p1zs[obj_index], p2xs[obj_index],
        p2ys[obj_index], p2zs[obj_index], p3xs[obj_index],
        p3ys[obj_index], p3zs[obj_index], mtypes[obj_index],
        fuzzs[obj_index], ttypes[obj_index],
        tp1xs[obj_index], tp1ys[obj_index],
        tp1zs[obj_index], tp2xs[obj_index],
        tp2ys[obj_index], tp2zs[obj_index],
        htype == IMAGE ? tdata[obj_index] : nullptr,
        ws[obj_index], hs[obj_index], tindices[obj_index],
        scales[obj_index]);
    return sobj;
  }
  __device__ SceneObj get(int obj_index, curandState *loc) {
    int htype = htypes[obj_index];
    SceneObj sobj(
        htypes[obj_index], rads[obj_index], p1xs[obj_index],
        p1ys[obj_index], p1zs[obj_index], p2xs[obj_index],
        p2ys[obj_index], p2zs[obj_index], p3xs[obj_index],
        p3ys[obj_index], p3zs[obj_index], mtypes[obj_index],
        fuzzs[obj_index], ttypes[obj_index],
        tp1xs[obj_index], tp1ys[obj_index],
        tp1zs[obj_index], tp2xs[obj_index],
        tp2ys[obj_index], tp2zs[obj_index],
        htype == IMAGE ? tdata[obj_index] : nullptr,
        ws[obj_index], hs[obj_index], tindices[obj_index],
        scales[obj_index], loc);
    return sobj;
  }
};

#endif
