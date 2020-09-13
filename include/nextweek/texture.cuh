#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include <nextweek/external.hpp>
//
#include <nextweek/perlin.cuh>
#include <nextweek/ray.cuh>
#include <nextweek/utils.cuh>
#include <nextweek/vec3.cuh>

class Texture {
public:
  __device__ virtual Color value(float u, float v,
                                 const Point3 &p) const = 0;
};

class SolidColor : public Texture {
public:
  __device__ SolidColor() {}
  __device__ SolidColor(Color c) : color_value(c) {}

  __device__ SolidColor(float red, float green, float blue)
      : SolidColor(Color(red, green, blue)) {}

  __device__ Color value(float u, float v,
                         const Point3 &p) const override {
    return color_value;
  }

private:
  Color color_value;
};

class CheckerTexture : public Texture {
public:
  Texture *odd;
  Texture *even;

public:
  __device__ CheckerTexture() {}
  __device__ ~CheckerTexture() {
    delete odd;
    delete even;
  }
  __device__ CheckerTexture(Color c1, Color c2) {
    odd = new SolidColor(c1);
    even = new SolidColor(c2);
  }
  __device__ CheckerTexture(Color c1) {
    odd = new SolidColor(c1);
    even = new SolidColor(1.0 - c1);
  }

  __device__ CheckerTexture(Texture *c1, Texture *c2) {
    odd = c1;
    even = c2;
  }
  __device__ Color value(float u, float v,
                         const Point3 &p) const override {
    //
    float sines = sin(10 * p.x()) * sin(10.0f * p.y()) *
                  sin(10.0f * p.z());
    if (sines < 0) {
      return odd->value(u, v, p);
    } else {
      return even->value(u, v, p);
    }
  }
};

class NoiseTexture : public Texture {
public:
  __device__ NoiseTexture() {}
  __device__ NoiseTexture(float s, curandState *loc)
      : scale(s), noise(Perlin(loc)) {}
  __device__ Color value(float u, float v,
                         const Point3 &p) const override {
    float zscale = scale * p.z();
    float turbulance = 10.0f * noise.turb(p);
    Color white(1.0f);
    return white * 0.5f * (1.0f + sin(zscale + turbulance));
  }

public:
  float scale;
  Perlin noise;
};

class ImageTexture : public Texture {
public:
  unsigned char *data;
  int width, height;
  int bytes_per_line; // == bytes_per_pixel * width;
  int bytes_per_pixel;
  int index;

public:
  __host__ __device__ ImageTexture()
      : data(nullptr), width(0), height(0),
        bytes_per_line(0), bytes_per_pixel(0), index(0) {}
  __host__ ~ImageTexture() { cudaFree(data); }
  __host__ ImageTexture(const char *impath) {
    imread(impath);
  }
  __host__ void imread(const char *impath) {
    int w, h, comp;
    data = stbi_load(impath, &w, &h, &comp, 0);
    if (!data) {
      throw std::runtime_error("Image can not be loaded");
    }
    set_im_vals(w, h, comp);
  }
  __host__ void set_im_vals(int w, int h, int per_pixel) {
    width = w;
    height = h;
    bytes_per_pixel = per_pixel;
    bytes_per_line = width * bytes_per_pixel;
  }
  __device__ ImageTexture(unsigned char *d, int w, int h,
                          int bpl, int bpp, int ind)
      : data(d), width(w), height(h), bytes_per_line(bpl),
        bytes_per_pixel(bpp), index(ind) {}
  __device__ ImageTexture(unsigned char *d, int *ws,
                          int *hs, int *bpps, int ind)
      : data(d) {
    width = ws[ind];
    height = hs[ind];
    bytes_per_pixel = bpps[ind];
    bytes_per_line = bytes_per_pixel * width;
    int start_point = 0;
    for (int i = 0; i < ind; i++) {
      start_point += ws[i] * hs[i] * bpps[i];
    }
    index = start_point;
  }
  __device__ ImageTexture(unsigned char *d, int w, int h,
                          int bpl, int bpp)
      : data(d), width(w), height(h), bytes_per_line(bpl),
        bytes_per_pixel(bpp) {}

  __device__ Color value(float u, float v,
                         const Point3 &p) const override {
    if (data == nullptr) {
      return Color(1.0, 0.0, 0.0);
    }
    u = clamp(u, 0.0, 1.0);
    v = 1.0 - clamp(v, 0.0, 1.0); // flip v to im coords
    int xi = (int)(u * width);
    int yj = (int)(v * height);
    xi = xi >= width ? width - 1 : xi;
    yj = yj >= height ? height - 1 : yj;

    //
    int pixel =
        yj * bytes_per_line + xi * bytes_per_pixel + index;
    Color c(0.0f);
    for (int i = 0; i < bytes_per_pixel; i++) {
      c.e[i] = (float)data[pixel + i] / 255;
    }
    return c;
  }
};

#endif
