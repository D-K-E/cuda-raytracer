#ifndef SCENEPARAM_CUH
#define SCENEPARAM_CUH

#include <nextweek/scenetype.cuh>
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

#endif
