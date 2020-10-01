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
    } else if (ttype == CHECKER)
      return to_checker();
    else if (ttype == IMAGE)
      return to_image();
  }
  __device__ NoiseTexture
  to_texture(curandState *loc) const {
    return to_noise(loc);
  }
};

enum MaterialType: int {
    LAMBERT = 1, METAL = 2, DIELECTRIC = 3
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
};

enum HittableType: int {
    LINE = 1, TRIANGLE = 2, SPHERE = 3
}

struct SceneObj {
  HittableType htype;
  SceneMaterial mat;
  float radius;
  Point3 p1, p2, p3;
};
