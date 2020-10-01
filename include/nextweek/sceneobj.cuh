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

  Color color;         // solid color, checker texture
  unsigned char *data; // image texture
  int width, height;   // image texture
  int bytes_per_pixel; // image texture
  int index;           // image texture
  float scale;         // noise texture

  __host__ __device__ SceneTexture() {}
  __host__ __device__ SceneTexture(SolidColor &scolor)
      : data(nullptr), ttype(SOLID),
        color(scolor.value(0.0f, 0.0f, Point3(0, 0, 0))) {}
  __host__ __device__ SceneTexture(CheckerTexture &ct)
      : data(nullptr), ttype(CHECKER),
        color(ct.odd->value(0.0f, 0.0f, Point3(0, 0, 0))) {}
  __host__ __device__ SceneTexture(NoiseTexture &nt)
      : data(nullptr), scale(nt.scale), ttype(NOISE) {}
  __host__ __device__ SceneTexture(ImageTexture &img)
      : data(img.data), width(img.width),
        height(img.height), index(img.index),
        bytes_per_pixel(img.bytes_per_pixel) {}
  __host__ __device__ SolidColor to_solid() const {
    return SolidColor(color);
  }
  __host__ __device__ CheckerTexture to_checker() const {
    return CheckerTexture(color);
  }
  __device__ NoiseTexture to_noise(curandState *loc) const {
    return NoiseTexture(scale, loc);
  }
  __host__ __device__ ImageTexture to_image() const {
    //
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

enum MaterialType: int {
    LAMBERT = 1, METAL = 2, DIELECTRIC = 3, DIFFUSE_LIGHT = 4, ISOTROPIC = 5
}

struct SceneMaterial {
  MaterialType mtype;
  SceneTexture st;
  float fuzz;
  __host__ __device__ SceneMaterial(SceneTexture &s,
                                    MaterialType m)
      : st(s), mtype(m) {}
  __host__ __device__ SceneMaterial(Color &c,
                                    MaterialType m)
      : st(SolidColor(c)), mtype(m) {}

  __host__ __device__ SceneMaterial(SceneTexture &s,
                                    float f, MaterialType m)
      : st(s), mtype(m), fuzz(f) {}

  __host__ __device__ SceneMaterial(Color &c, float f,
                                    MaterialType m)
      : st(SolidColor(c)), mtype(m), fuzz(f) {}

  __host__ __device__ SceneMaterial(float ri,
                                    MaterialType m)
      : mtype(m), fuzz(ri) {}

  __host__ __device__ SceneMaterial(Lambertian &lamb)
      : SceneMaterial(SceneTexture((*lamb.albedo)),
                      LAMBERT) {}
  __host__ __device__ SceneMaterial(Metal &met)
      : SceneMaterial(SceneTexture((*met.albedo)), met.fuzz,
                      METAL) {}
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
  HittableType htype;
  SceneMaterial mat;
  float radius;
  float p1x, p1y, p1z;
  float p2x, p2y, p2z;
  float p3x, p3y, p3z;
  __host__ __device__ SceneObj() {}
  __host__ __device__ SceneObj(HittableType ht,
                               SceneMaterial sm, float rad,
                               Point3 c1, Point3 c2,
                               Point3 c3)
      : htype(ht), mat(sm), radius(rad), p1x(c1.x()),
        p1y(c1.y()), p1z(c1.z()), p2x(c2.x()), p2y(c2.y()),
        p2z(c2.z()), p3x(c3.x()), p3y(c3.y()), p3z(c3.z()) {
  }
  __host__ __device__ SceneObj(HittableType ht,
                               SceneMaterial sm, float rad,
                               Point3 c1, Point3 c2)
      : htype(ht), mat(sm), radius(rad), p1(c1), p2(c2) {}

  __host__ __device__ SceneObj(Sphere &sp)
      : htype(SPHERE), mat((*sp.mat_ptr)),
        radius(sp.radius), p1x(sp.center.x()),
        p1y(sp.center.y()), p1z(sp.center.z()) {}

  __host__ __device__ SceneObj(MovingSphere &sp)
      : htype(MOVING_SPHERE), mat((*sp.mat_ptr)),
        radius(sp.radius), p1x(sp.center1.x()),
        p1y(sp.center1.y()), p1z(sp.center1.z()),
        p2x(sp.center2.x()), p2y(sp.center2.y()),
        p2z(sp.center2.z()), p3x(sp.time0), p3y(sp.time1) {}

  __host__ __device__ SceneObj(AaRect &rect)
      : htype(RECTANGLE), mat((*rect.mat_ptr)),
        radius(rect.k), p1x(rect.a0), p1y(rect.a1),
        p1z(rect.b0), p2x(rect.b1),
        p2y(rect.axis_normal.x()),
        p2z(rect.axis_normal.y()),
        p3x(rect.axis_normal.z()) {}
  __host__ __device__ SceneObj(Triangle &tri)
      : htype(TRIANGLE), mat((*tri.mat_ptr)),
        p1x(tri.p1.x()), p1y(tri.p1.y()),
        p1z(tri.p1.z()) p2x(tri.p2.x()), p2y(tri.p2.y()),
        p2z(tri.p2.z()) p3x(tri.p3.x()), p3y(tri.p3.y()),
        p3z(tri.p3.z()) {}

  template <typename T> __host__ __device__ T to_object() {
    //
    if (htype == SPHERE) {
      Sphere sp(Point3(p1x, p1y, p1z), radius, &mat);
      return sp;
    } else if (htype == MOVING_SPHERE) {
      MovingSphere sp(Point3(p1x, p1y, p1z),
                      Point3(p2x, p2y, p2z), p3x, p3y,
                      radius, &mat);
      return sp;
    } else if (htype == RECTANGLE) {
      AaRect rect(p1x, p1y, p1z, p2x, radius, &mat,
                  Vec3(p2y, p2z, p3x));
      return rect;
    } else if (htype == TRIANGLE) {
      Triangle tri(Point3(p1x, p1y, p1z),
                   Point3(p2x, p2y, p2z),
                   Point3(p3x, p3y, p3z), &mat);
      return tri;
    }
  }
};
